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
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
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
  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;
};

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

class TransportInvalidError : public llvm::ErrorInfo<TransportInvalidError> {
public:
  static char ID;

  TransportInvalidError() = default;

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;
};

/// A transport class that uses JSON for communication.
class JSONTransport {
public:
  using ReadHandleUP = MainLoopBase::ReadHandleUP;
  template <typename T>
  using Callback = std::function<void(MainLoopBase &, const llvm::Expected<T>)>;

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
                                                  Callback<T> read_cb) {
    Status error;
    ReadHandleUP handle = loop.RegisterReadObject(
        m_input,
        [read_cb, this](MainLoopBase &loop) {
          char buf[kReadBufferSize];
          size_t num_bytes = sizeof(buf);
          if (llvm::Error error = m_input->Read(buf, num_bytes).takeError()) {
            read_cb(loop, std::move(error));
            return;
          }
          if (num_bytes)
            m_buffer.append(std::string(buf, num_bytes));

          // If the buffer has contents, try parsing any pending messages.
          if (!m_buffer.empty()) {
            llvm::Expected<std::vector<std::string>> messages = Parse();
            if (llvm::Error error = messages.takeError()) {
              read_cb(loop, std::move(error));
              return;
            }

            for (const auto &message : *messages)
              if constexpr (std::is_same<T, std::string>::value)
                read_cb(loop, message);
              else
                read_cb(loop, llvm::json::parse<T>(message));
          }

          // On EOF, notify the callback after the remaining messages were
          // handled.
          if (num_bytes == 0) {
            if (m_buffer.empty())
              read_cb(loop, llvm::make_error<TransportEOFError>());
            else
              read_cb(loop, llvm::make_error<TransportUnhandledContentsError>(
                                std::string(m_buffer)));
          }
        },
        error);
    if (error.Fail())
      return error.takeError();
    return handle;
  }

protected:
  template <typename... Ts> inline auto Logv(const char *Fmt, Ts &&...Vals) {
    Log(llvm::formatv(Fmt, std::forward<Ts>(Vals)...).str());
  }
  virtual void Log(llvm::StringRef message);

  virtual llvm::Error WriteImpl(const std::string &message) = 0;
  virtual llvm::Expected<std::vector<std::string>> Parse() = 0;

  static constexpr size_t kReadBufferSize = 1024;

  lldb::IOObjectSP m_input;
  lldb::IOObjectSP m_output;
  llvm::SmallString<kReadBufferSize> m_buffer;
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
  static constexpr llvm::StringLiteral kEndOfHeader = "\r\n\r\n";
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
