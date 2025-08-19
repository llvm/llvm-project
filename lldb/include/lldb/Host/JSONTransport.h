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

  /// Sends an event, a message that does not require a response.
  virtual llvm::Error Send(const Evt &) = 0;
  /// Sends a request, a message that expects a response.
  virtual llvm::Error Send(const Req &) = 0;
  /// Sends a response to a specific request.
  virtual llvm::Error Send(const Resp &) = 0;

  /// Implemented to handle incoming messages. (See Run() below).
  class MessageHandler {
  public:
    virtual ~MessageHandler() = default;
    /// Called when an event is received.
    virtual void Received(const Evt &) = 0;
    /// Called when a request is received.
    virtual void Received(const Req &) = 0;
    /// Called when a response is received.
    virtual void Received(const Resp &) = 0;

    /// Called when an error occurs while reading from the transport.
    ///
    /// NOTE: This does *NOT* indicate that a specific request failed, but that
    /// there was an error in the underlying transport.
    virtual void OnError(llvm::Error) = 0;

    /// Called on EOF or client disconnect.
    virtual void OnClosed() = 0;
  };

  using MessageHandlerSP = std::shared_ptr<MessageHandler>;

  /// RegisterMessageHandler registers the Transport with the given MainLoop and
  /// handles any incoming messages using the given MessageHandler.
  ///
  /// If an unexpected error occurs, the MainLoop will be terminated and a log
  /// message will include additional information about the termination reason.
  virtual llvm::Expected<MainLoop::ReadHandleUP>
  RegisterMessageHandler(MainLoop &loop, MessageHandler &handler) = 0;

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
  using MessageHandler = typename Transport<Req, Resp, Evt>::MessageHandler;

  JSONTransport(lldb::IOObjectSP in, lldb::IOObjectSP out)
      : m_in(in), m_out(out) {}

  llvm::Error Send(const Evt &evt) override { return Write(evt); }
  llvm::Error Send(const Req &req) override { return Write(req); }
  llvm::Error Send(const Resp &resp) override { return Write(resp); }

  llvm::Expected<MainLoop::ReadHandleUP>
  RegisterMessageHandler(MainLoop &loop, MessageHandler &handler) override {
    Status status;
    MainLoop::ReadHandleUP read_handle = loop.RegisterReadObject(
        m_in,
        std::bind(&JSONTransport::OnRead, this, std::placeholders::_1,
                  std::ref(handler)),
        status);
    if (status.Fail()) {
      return status.takeError();
    }
    return read_handle;
  }

  /// Public for testing purposes, otherwise this should be an implementation
  /// detail.
  static constexpr size_t kReadBufferSize = 1024;

protected:
  virtual llvm::Expected<std::vector<std::string>> Parse() = 0;
  virtual std::string Encode(const llvm::json::Value &message) = 0;
  llvm::Error Write(const llvm::json::Value &message) {
    this->Logv("<-- {0}", message);
    std::string output = Encode(message);
    size_t bytes_written = output.size();
    return m_out->Write(output.data(), bytes_written).takeError();
  }

  llvm::SmallString<kReadBufferSize> m_buffer;

private:
  void OnRead(MainLoopBase &loop, MessageHandler &handler) {
    char buf[kReadBufferSize];
    size_t num_bytes = sizeof(buf);
    if (Status status = m_in->Read(buf, num_bytes); status.Fail()) {
      handler.OnError(status.takeError());
      return;
    }

    if (num_bytes)
      m_buffer.append(llvm::StringRef(buf, num_bytes));

    // If the buffer has contents, try parsing any pending messages.
    if (!m_buffer.empty()) {
      llvm::Expected<std::vector<std::string>> raw_messages = Parse();
      if (llvm::Error error = raw_messages.takeError()) {
        handler.OnError(std::move(error));
        return;
      }

      for (const std::string &raw_message : *raw_messages) {
        llvm::Expected<typename Transport<Req, Resp, Evt>::Message> message =
            llvm::json::parse<typename Transport<Req, Resp, Evt>::Message>(
                raw_message);
        if (!message) {
          handler.OnError(message.takeError());
          return;
        }

        std::visit([&handler](auto &&msg) { handler.Received(msg); }, *message);
      }
    }

    // Check if we reached EOF.
    if (num_bytes == 0) {
      // EOF reached, but there may still be unhandled contents in the buffer.
      if (!m_buffer.empty())
        handler.OnError(llvm::make_error<TransportUnhandledContentsError>(
            std::string(m_buffer.str())));
      handler.OnClosed();
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
        if (!llvm::to_integer(value, content_length, 10)) {
          // Clear the buffer to avoid re-parsing this malformed message.
          this->m_buffer.clear();
          return llvm::createStringError(std::errc::invalid_argument,
                                         "invalid content length: %s",
                                         value.str().c_str());
        }
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
