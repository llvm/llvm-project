//===-- ThreadTrace.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_THREADTRACE_H
#define LLDB_TARGET_THREADTRACE_H

#include "lldb/Target/Thread.h"

namespace lldb_private {

/// \class ThreadTrace ThreadTrace.h
///
/// Thread implementation used for representing threads gotten from trace
/// session files, which are similar to threads from core files.
///
/// See \a TraceSessionFileParser for more information regarding trace session
/// files.
class ThreadTrace : public Thread {
public:
  /// \param[in] process
  ///     The process who owns this thread.
  ///
  /// \param[in] tid
  ///     The tid of this thread.
  ///
  /// \param[in] trace_file
  ///     The file that contains the list of instructions that were traced when
  ///     this thread was being executed.
  ThreadTrace(Process &process, lldb::tid_t tid, const FileSpec &trace_file)
      : Thread(process, tid), m_trace_file(trace_file) {}

  void RefreshStateAfterStop() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override;

  /// \return
  ///   The trace file of this thread.
  const FileSpec &GetTraceFile() const;

protected:
  bool CalculateStopInfo() override;

  lldb::RegisterContextSP m_thread_reg_ctx_sp;

private:
  FileSpec m_trace_file;
};

typedef std::shared_ptr<ThreadTrace> ThreadTraceSP;

} // namespace lldb_private

#endif // LLDB_TARGET_THREADTRACE_H
