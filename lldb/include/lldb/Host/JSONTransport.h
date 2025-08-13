//===-- JSONTransport.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transport layer for encoding and decoding JSON protocol messages.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_JSONTRANSPORT_H
#define LLDB_HOST_JSONTRANSPORT_H

#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace lldb_private {

class TransportUnhandledContentsError
    : public llvm::ErrorInfo<TransportUnhandledContentsError> {
public:
  static char ID;

  explicit TransportUnhandledContentsError(std::string unhandled_contents);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getUnhandledContents() const {
    return m_unhandled_contents;
  }

private:
  std::string m_unhandled_contents;
};

/// A transport is responsible for maintaining the connection to a client
/// application, and reading/writing structured messages to it.
///
/// Transports have limited thread safety requirements:
///  - Messages will not be sent concurrently.
///  - Messages MAY be sent while Run() is reading, or its callback is active.
template <typename Req, typename Resp, typename Evt> class Transport {
public:
  using Message = std::variant<Req, Resp, Evt>;

  virtual ~Transport() = default;

  // Called by transport to send outgoing messages.
  virtual void Event(const Evt &) = 0;
  virtual void Request(const Req &) = 0;
  virtual void Response(const Resp &) = 0;

  /// Implemented to handle incoming messages. (See Run() below).
  class MessageHandler {
  public:
    virtual ~MessageHandler() = default;
    virtual void OnEvent(const Evt &) = 0;
    virtual void OnRequest(const Req &) = 0;
    virtual void OnResponse(const Resp &) = 0;
  };

  /// Called by server or client to receive messages from the connection.
  /// The transport should in turn invoke the handler to process messages.
  /// The MainLoop is used to handle reading from the incoming connection and
  /// will run until the loop is terminated.
  virtual llvm::Error Run(MainLoop &, MessageHandler &) = 0;

protected:
  template <typename... Ts> inline auto Logv(const char *Fmt, Ts &&...Vals) {
    Log(llvm::formatv(Fmt, std::forward<Ts>(Vals)...).str());
  }
  virtual void Log(llvm::StringRef message) = 0;
};

/// A JSONTransport will encode and decode messages using JSON.
template <typename Req, typename Resp, typename Evt>
class JSONTransport : public Transport<Req, Resp, Evt> {
public:
  using Transport<Req, Resp, Evt>::Transport;

  JSONTransport(lldb::IOObjectSP in, lldb::IOObjectSP out)
      : m_in(in), m_out(out) {}

  void Event(const Evt &evt) override { Write(evt); }
  void Request(const Req &req) override { Write(req); }
  void Response(const Resp &resp) override { Write(resp); }

  /// Run registers the transport with the given MainLoop and handles any
  /// incoming messages using the given MessageHandler.
  llvm::Error
  Run(MainLoop &loop,
      typename Transport<Req, Resp, Evt>::MessageHandler &handler) override {
    llvm::Error error = llvm::Error::success();
    Status status;
    auto read_handle = loop.RegisterReadObject(
        m_in,
        std::bind(&JSONTransport::OnRead, this, &error, std::placeholders::_1,
                  std::ref(handler)),
        status);
    if (status.Fail()) {
      // This error is only set if the read object handler is invoked, mark it
      // as consumed if registration of the handler failed.
      llvm::consumeError(std::move(error));
      return status.takeError();
    }

    status = loop.Run();
    if (status.Fail())
      return status.takeError();
    return error;
  }

  /// Public for testing purposes, otherwise this should be an implementation
  /// detail.
  static constexpr size_t kReadBufferSize = 1024;

protected:
  virtual llvm::Expected<std::vector<std::string>> Parse() = 0;
  virtual std::string Encode(const llvm::json::Value &message) = 0;
  void Write(const llvm::json::Value &message) {
    this->Logv("<-- {0}", message);
    std::string output = Encode(message);
    size_t bytes_written = output.size();
    Status status = m_out->Write(output.data(), bytes_written);
    if (status.Fail())
      this->Logv("writing failed: {0}", status.AsCString());
  }

  llvm::SmallString<kReadBufferSize> m_buffer;

private:
  void OnRead(llvm::Error *err, MainLoopBase &loop,
              typename Transport<Req, Resp, Evt>::MessageHandler &handler) {
    llvm::ErrorAsOutParameter ErrAsOutParam(err);
    char buf[kReadBufferSize];
    size_t num_bytes = sizeof(buf);
    if (Status status = m_in->Read(buf, num_bytes); status.Fail()) {
      *err = status.takeError();
      loop.RequestTermination();
      return;
    }

    if (num_bytes)
      m_buffer.append(llvm::StringRef(buf, num_bytes));

    // If the buffer has contents, try parsing any pending messages.
    if (!m_buffer.empty()) {
      llvm::Expected<std::vector<std::string>> raw_messages = Parse();
      if (llvm::Error error = raw_messages.takeError()) {
        *err = std::move(error);
        loop.RequestTermination();
        return;
      }

      for (const std::string &raw_message : *raw_messages) {
        llvm::Expected<typename Transport<Req, Resp, Evt>::Message> message =
            llvm::json::parse<typename Transport<Req, Resp, Evt>::Message>(
                raw_message);
        if (!message) {
          *err = message.takeError();
          loop.RequestTermination();
          return;
        }

        if (Evt *evt = std::get_if<Evt>(&*message)) {
          handler.OnEvent(*evt);
          continue;
        }

        if (Req *req = std::get_if<Req>(&*message)) {
          handler.OnRequest(*req);
          continue;
        }

        if (Resp *resp = std::get_if<Resp>(&*message)) {
          handler.OnResponse(*resp);
          continue;
        }

        llvm_unreachable("unknown message type");
      }
    }

    if (num_bytes == 0) {
      // If we're at EOF and we have unhandled contents in the buffer, return an
      // error for the partial message.
      if (m_buffer.empty())
        *err = llvm::Error::success();
      else
        *err = llvm::make_error<TransportUnhandledContentsError>(
            std::string(m_buffer));
      loop.RequestTermination();
    }
  }

  lldb::IOObjectSP m_in;
  lldb::IOObjectSP m_out;
};

/// A transport class for JSON with a HTTP header.
template <typename Req, typename Resp, typename Evt>
class HTTPDelimitedJSONTransport : public JSONTransport<Req, Resp, Evt> {
public:
  using JSONTransport<Req, Resp, Evt>::JSONTransport;

protected:
  /// Encodes messages based on
  /// https://microsoft.github.io/debug-adapter-protocol/overview#base-protocol
  std::string Encode(const llvm::json::Value &message) override {
    std::string output;
    std::string raw_message = llvm::formatv("{0}", message).str();
    llvm::raw_string_ostream OS(output);
    OS << kHeaderContentLength << kHeaderFieldSeparator << ' '
       << std::to_string(raw_message.size()) << kEndOfHeader << raw_message;
    return output;
  }

  /// Parses messages based on
  /// https://microsoft.github.io/debug-adapter-protocol/overview#base-protocol
  llvm::Expected<std::vector<std::string>> Parse() override {
    std::vector<std::string> messages;
    llvm::StringRef buffer = this->m_buffer;
    while (buffer.contains(kEndOfHeader)) {
      auto [headers, rest] = buffer.split(kEndOfHeader);
      size_t content_length = 0;
      // HTTP Headers are formatted like `<field-name> ':' [<field-value>]`.
      for (const llvm::StringRef &header :
           llvm::split(headers, kHeaderSeparator)) {
        auto [key, value] = header.split(kHeaderFieldSeparator);
        // 'Content-Length' is the only meaningful key at the moment. Others are
        // ignored.
        if (!key.equals_insensitive(kHeaderContentLength))
          continue;

        value = value.trim();
        if (!llvm::to_integer(value, content_length, 10))
          return llvm::createStringError(std::errc::invalid_argument,
                                         "invalid content length: %s",
                                         value.str().c_str());
      }

      // Check if we have enough data.
      if (content_length > rest.size())
        break;

      llvm::StringRef body = rest.take_front(content_length);
      buffer = rest.drop_front(content_length);
      messages.emplace_back(body.str());
      this->Logv("--> {0}", body);
    }

    // Store the remainder of the buffer for the next read callback.
    this->m_buffer = buffer.str();

    return std::move(messages);
  }

  static constexpr llvm::StringLiteral kHeaderContentLength = "Content-Length";
  static constexpr llvm::StringLiteral kHeaderFieldSeparator = ":";
  static constexpr llvm::StringLiteral kHeaderSeparator = "\r\n";
  static constexpr llvm::StringLiteral kEndOfHeader = "\r\n\r\n";
};

/// A transport class for JSON RPC.
template <typename Req, typename Resp, typename Evt>
class JSONRPCTransport : public JSONTransport<Req, Resp, Evt> {
public:
  using JSONTransport<Req, Resp, Evt>::JSONTransport;

protected:
  std::string Encode(const llvm::json::Value &message) override {
    return llvm::formatv("{0}{1}", message, kMessageSeparator).str();
  }

  llvm::Expected<std::vector<std::string>> Parse() override {
    std::vector<std::string> messages;
    llvm::StringRef buf = this->m_buffer;
    while (buf.contains(kMessageSeparator)) {
      auto [raw_json, rest] = buf.split(kMessageSeparator);
      buf = rest;
      messages.emplace_back(raw_json.str());
      this->Logv("--> {0}", raw_json);
    }

    // Store the remainder of the buffer for the next read callback.
    this->m_buffer = buf.str();

    return messages;
  }

  static constexpr llvm::StringLiteral kMessageSeparator = "\n";
};

} // namespace lldb_private

#endif
