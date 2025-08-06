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

#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <chrono>
#include <system_error>

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

  /// Reads the next message from the input stream.
  template <typename T>
  llvm::Expected<T> Read(const std::chrono::microseconds &timeout) {
    llvm::Expected<std::string> message = ReadImpl(timeout);
    if (!message)
      return message.takeError();
    return llvm::json::parse<T>(/*JSON=*/*message);
  }

protected:
  virtual void Log(llvm::StringRef message);

  virtual llvm::Error WriteImpl(const std::string &message) = 0;
  virtual llvm::Expected<std::string>
  ReadImpl(const std::chrono::microseconds &timeout) = 0;

  lldb::IOObjectSP m_input;
  lldb::IOObjectSP m_output;
};

/// A transport class for JSON with a HTTP header.
class HTTPDelimitedJSONTransport : public JSONTransport {
public:
  HTTPDelimitedJSONTransport(lldb::IOObjectSP input, lldb::IOObjectSP output)
      : JSONTransport(input, output) {}
  virtual ~HTTPDelimitedJSONTransport() = default;

protected:
  virtual llvm::Error WriteImpl(const std::string &message) override;
  virtual llvm::Expected<std::string>
  ReadImpl(const std::chrono::microseconds &timeout) override;

  // FIXME: Support any header.
  static constexpr llvm::StringLiteral kHeaderContentLength =
      "Content-Length: ";
  static constexpr llvm::StringLiteral kHeaderSeparator = "\r\n\r\n";
};

/// A transport class for JSON RPC.
class JSONRPCTransport : public JSONTransport {
public:
  JSONRPCTransport(lldb::IOObjectSP input, lldb::IOObjectSP output)
      : JSONTransport(input, output) {}
  virtual ~JSONRPCTransport() = default;

protected:
  virtual llvm::Error WriteImpl(const std::string &message) override;
  virtual llvm::Expected<std::string>
  ReadImpl(const std::chrono::microseconds &timeout) override;

  static constexpr llvm::StringLiteral kMessageSeparator = "\n";
};

} // namespace lldb_private

#endif
