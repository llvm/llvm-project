//===-- CommandReturnObject.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandReturnObject_h_
#define liblldb_CommandReturnObject_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/STLUtils.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StreamTee.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class CommandReturnObject {
public:
  CommandReturnObject();

  ~CommandReturnObject();

  const char *GetOutputData() {
    lldb::StreamSP stream_sp(m_out_stream.GetStreamAtIndex(eStreamStringIndex));
    if (stream_sp)
      return static_cast<StreamString *>(stream_sp.get())->GetData();
    return "";
  }

  const char *GetErrorData() {
    lldb::StreamSP stream_sp(m_err_stream.GetStreamAtIndex(eStreamStringIndex));
    if (stream_sp)
      return static_cast<StreamString *>(stream_sp.get())->GetData();
    else
      return "";
  }

  Stream &GetOutputStream() {
    // Make sure we at least have our normal string stream output stream
    lldb::StreamSP stream_sp(m_out_stream.GetStreamAtIndex(eStreamStringIndex));
    if (!stream_sp) {
      stream_sp.reset(new StreamString());
      m_out_stream.SetStreamAtIndex(eStreamStringIndex, stream_sp);
    }
    return m_out_stream;
  }

  Stream &GetErrorStream() {
    // Make sure we at least have our normal string stream output stream
    lldb::StreamSP stream_sp(m_err_stream.GetStreamAtIndex(eStreamStringIndex));
    if (!stream_sp) {
      stream_sp.reset(new StreamString());
      m_err_stream.SetStreamAtIndex(eStreamStringIndex, stream_sp);
    }
    return m_err_stream;
  }

  void SetImmediateOutputFile(FILE *fh, bool transfer_fh_ownership = false) {
    lldb::StreamSP stream_sp(new StreamFile(fh, transfer_fh_ownership));
    m_out_stream.SetStreamAtIndex(eImmediateStreamIndex, stream_sp);
  }

  void SetImmediateErrorFile(FILE *fh, bool transfer_fh_ownership = false) {
    lldb::StreamSP stream_sp(new StreamFile(fh, transfer_fh_ownership));
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

  void AppendMessage(const char *in_string);

  void AppendMessageWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  void AppendRawWarning(const char *in_string);

  void AppendWarning(const char *in_string);

  void AppendWarningWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  void AppendError(const char *in_string);

  void AppendRawError(const char *in_string);

  void AppendErrorWithFormat(const char *format, ...)
      __attribute__((format(printf, 2, 3)));

  void SetError(const Error &error, const char *fallback_error_cstr = nullptr);

  void SetError(const char *error_cstr);

  lldb::ReturnStatus GetStatus();

  void SetStatus(lldb::ReturnStatus status);

  bool Succeeded();

  bool HasResult();

  bool GetDidChangeProcessState();

  void SetDidChangeProcessState(bool b);

  bool GetInteractive() const;

  void SetInteractive(bool b);

  bool GetAbnormalStopWasExpected() const {
    return m_abnormal_stop_was_expected;
  }

  void SetAbnormalStopWasExpected(bool signal_was_expected) {
    m_abnormal_stop_was_expected = signal_was_expected;
  }

private:
  enum { eStreamStringIndex = 0, eImmediateStreamIndex = 1 };

  StreamTee m_out_stream;
  StreamTee m_err_stream;

  lldb::ReturnStatus m_status;
  bool m_did_change_process_state;
  bool m_interactive; // If true, then the input handle from the debugger will
                      // be hooked up
  bool m_abnormal_stop_was_expected; // This is to support
                                     // eHandleCommandFlagStopOnCrash vrs.
                                     // attach.
  // The attach command often ends up with the process stopped due to a signal.
  // Normally that would mean stop on crash should halt batch execution, but we
  // obviously don't want that for attach.  Using this flag, the attach command
  // (and anything else for which this is relevant) can say that the signal is
  // expected, and batch command execution can continue.
};

} // namespace lldb_private

#endif // liblldb_CommandReturnObject_h_
