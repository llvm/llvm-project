//===-- CommandReturnObject.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_COMMANDRETURNOBJECT_H
#define LLDB_INTERPRETER_COMMANDRETURNOBJECT_H

#include "lldb/Core/StreamFile.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StreamTee.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"

#include <memory>

namespace lldb_private {

class CommandReturnObject {
public:
  CommandReturnObject(bool colors);

  ~CommandReturnObject();

  llvm::StringRef GetOutputData() {
    lldb::StreamSP stream_sp(m_out_stream.GetStreamAtIndex(eStreamStringIndex));
    if (stream_sp)
      return std::static_pointer_cast<StreamString>(stream_sp)->GetString();
    return llvm::StringRef();
  }

  llvm::StringRef GetErrorData() {
    lldb::StreamSP stream_sp(m_err_stream.GetStreamAtIndex(eStreamStringIndex));
    if (stream_sp)
      return std::static_pointer_cast<StreamString>(stream_sp)->GetString();
    return llvm::StringRef();
  }

  Stream &GetOutputStream() {
    // Make sure we at least have our normal string stream output stream
    lldb::StreamSP stream_sp(m_out_stream.GetStreamAtIndex(eStreamStringIndex));
    if (!stream_sp) {
      stream_sp = std::make_shared<StreamString>();
      m_out_stream.SetStreamAtIndex(eStreamStringIndex, stream_sp);
    }
    return m_out_stream;
  }

  Stream &GetErrorStream() {
    // Make sure we at least have our normal string stream output stream
    lldb::StreamSP stream_sp(m_err_stream.GetStreamAtIndex(eStreamStringIndex));
    if (!stream_sp) {
      stream_sp = std::make_shared<StreamString>();
      m_err_stream.SetStreamAtIndex(eStreamStringIndex, stream_sp);
    }
    return m_err_stream;
  }

  void SetImmediateOutputFile(lldb::FileSP file_sp) {
    lldb::StreamSP stream_sp(new StreamFile(file_sp));
    m_out_stream.SetStreamAtIndex(eImmediateStreamIndex, stream_sp);
  }

  void SetImmediateErrorFile(lldb::FileSP file_sp) {
    lldb::StreamSP stream_sp(new StreamFile(file_sp));
    m_err_stream.SetStreamAtIndex(eImmediateStreamIndex, stream_sp);
  }

  void SetImmediateOutputStream(const lldb::StreamSP &stream_sp) {
    m_out_stream.SetStreamAtIndex(eImmediateStreamIndex, stream_sp);
  }

  void SetImmediateErrorStream(const lldb::StreamSP &stream_sp) {
    m_err_stream.SetStreamAtIndex(eImmediateStreamIndex, stream_sp);
  }

  lldb::StreamSP GetImmediateOutputStream() {
    return m_out_stream.GetStreamAtIndex(eImmediateStreamIndex);
  }

  lldb::StreamSP GetImmediateErrorStream() {
    return m_err_stream.GetStreamAtIndex(eImmediateStreamIndex);
  }

  void Clear();

  void AppendMessage(llvm::StringRef in_string);

  void AppendMessageWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  void AppendRawWarning(llvm::StringRef in_string);

  void AppendWarning(llvm::StringRef in_string);

  void AppendWarningWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  void AppendError(llvm::StringRef in_string);

  void AppendRawError(llvm::StringRef in_string);

  void AppendErrorWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  template <typename... Args>
  void AppendMessageWithFormatv(const char *format, Args &&... args) {
    AppendMessage(llvm::formatv(format, std::forward<Args>(args)...).str());
  }

  template <typename... Args>
  void AppendWarningWithFormatv(const char *format, Args &&... args) {
    AppendWarning(llvm::formatv(format, std::forward<Args>(args)...).str());
  }

  template <typename... Args>
  void AppendErrorWithFormatv(const char *format, Args &&... args) {
    AppendError(llvm::formatv(format, std::forward<Args>(args)...).str());
  }

  void SetError(const Status &error, const char *fallback_error_cstr = nullptr);

  void SetError(llvm::StringRef error_cstr);

  lldb::ReturnStatus GetStatus();

  void SetStatus(lldb::ReturnStatus status);

  bool Succeeded();

  bool HasResult();

  bool GetDidChangeProcessState();

  void SetDidChangeProcessState(bool b);

  bool GetInteractive() const;

  void SetInteractive(bool b);

private:
  enum { eStreamStringIndex = 0, eImmediateStreamIndex = 1 };

  StreamTee m_out_stream;
  StreamTee m_err_stream;

  lldb::ReturnStatus m_status;
  bool m_did_change_process_state;
  bool m_interactive; // If true, then the input handle from the debugger will
                      // be hooked up
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_COMMANDRETURNOBJECT_H
