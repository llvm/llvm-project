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

#include "lldb/Host/MainLoopBase.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <system_error>
#include <vector>

namespace lldb_private {

class TransportEOFError : public llvm::ErrorInfo<TransportEOFError> {
public:
  static char ID;

  TransportEOFError() = default;

  void log(llvm::raw_ostream &OS) const override {
    OS << "transport end of file reached";
  }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

class TransportTimeoutError : public llvm::ErrorInfo<TransportTimeoutError> {
public:
  static char ID;

  TransportTimeoutError() = default;

  void log(llvm::raw_ostream &OS) const override {
    OS << "transport operation timed out";
  }
  std::error_code convertToErrorCode() const override {
    return std::make_error_code(std::errc::timed_out);
  }
};

class TransportInvalidError : public llvm::ErrorInfo<TransportInvalidError> {
public:
  static char ID;

  TransportInvalidError() = default;

  void log(llvm::raw_ostream &OS) const override {
    OS << "transport IO object invalid";
  }
  std::error_code convertToErrorCode() const override {
    return std::make_error_code(std::errc::not_connected);
  }
};

/// A transport class that uses JSON for communication.
class JSONTransport {
public:
  using ReadHandleUP = MainLoopBase::ReadHandleUP;
  template <typename T>
  using Callback = std::function<void(MainLoopBase &, llvm::Expected<T>)>;

  JSONTransport(lldb::IOObjectSP input, lldb::IOObjectSP output);
  virtual ~JSONTransport() = default;

  /// Transport is not copyable.
  /// @{
  JSONTransport(const JSONTransport &rhs) = delete;
  void operator=(const JSONTransport &rhs) = delete;
  /// @}

  /// Writes a message to the output stream.
  template <typename T> llvm::Error Write(const T &t) {
    const std::string message = llvm::formatv("{0}", toJSON(t)).str();
    return WriteImpl(message);
  }

  /// Registers the transport with the MainLoop.
  template <typename T>
  llvm::Expected<ReadHandleUP> RegisterReadObject(MainLoopBase &loop,
                                                  Callback<T> callback) {
    Status error;
    ReadHandleUP handle = loop.RegisterReadObject(
        m_input,
        [&](MainLoopBase &loop) {
          char buf[1024];
          size_t len = sizeof(buf);
          do {
            if (llvm::Error error = m_input->Read(buf, len).takeError()) {
              callback(loop, std::move(error));
              return;
            }

            if (len == 0) // EOF
              break;

            m_buffer.append(std::string(buf, len));
          } while (len == sizeof(buf));

          llvm::Expected<std::vector<std::string>> messages = Parse();
          if (llvm::Error error = messages.takeError()) {
            callback(loop, std::move(error));
            return;
          }

          for (const auto &message : *messages)
            if constexpr (std::is_same<T, std::string>::value)
              callback(loop, message);
            else
              callback(loop, llvm::json::parse<T>(message));

          // On EOF, notify the callback after the remaining messages were
          // handled.
          if (len == 0)
            callback(loop, llvm::make_error<TransportEOFError>());
        },
        error);
    if (error.Fail())
      return error.takeError();
    return handle;
  }

protected:
  virtual void Log(llvm::StringRef message);

  virtual llvm::Error WriteImpl(const std::string &message) = 0;
  virtual llvm::Expected<std::vector<std::string>> Parse() = 0;

  lldb::IOObjectSP m_input;
  lldb::IOObjectSP m_output;
  std::string m_buffer;
};

/// A transport class for JSON with a HTTP header.
class HTTPDelimitedJSONTransport : public JSONTransport {
public:
  HTTPDelimitedJSONTransport(lldb::IOObjectSP input, lldb::IOObjectSP output)
      : JSONTransport(input, output) {}
  virtual ~HTTPDelimitedJSONTransport() = default;

protected:
  llvm::Error WriteImpl(const std::string &message) override;
  llvm::Expected<std::vector<std::string>> Parse() override;

  static constexpr llvm::StringLiteral kHeaderContentLength = "Content-Length";
  static constexpr llvm::StringLiteral kHeaderFieldSeparator = ":";
  static constexpr llvm::StringLiteral kHeaderSeparator = "\r\n";
};

/// A transport class for JSON RPC.
class JSONRPCTransport : public JSONTransport {
public:
  JSONRPCTransport(lldb::IOObjectSP input, lldb::IOObjectSP output)
      : JSONTransport(input, output) {}
  virtual ~JSONRPCTransport() = default;

protected:
  llvm::Error WriteImpl(const std::string &message) override;
  llvm::Expected<std::vector<std::string>> Parse() override;

  static constexpr llvm::StringLiteral kMessageSeparator = "\n";
};

} // namespace lldb_private

#endif
