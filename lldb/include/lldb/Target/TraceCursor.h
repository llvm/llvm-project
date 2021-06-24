//===-- TraceCursor.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACE_CURSOR_H
#define LLDB_TARGET_TRACE_CURSOR_H

#include "lldb/lldb-private.h"

namespace lldb_private {

/// Class used for iterating over the instructions of a thread's trace.
///
/// This class attempts to be a generic interface for accessing the instructions
/// of the trace so that each Trace plug-in can reconstruct, represent and store
/// the instruction data in an flexible way that is efficient for the given
/// technology.
///
/// Live processes:
///  In the case of a live process trace, an instance of a \a TraceCursor should
///  point to the trace at the moment it was collected. If the process is later
///  resumed and new trace data is collected, that should leave that old cursor
///  unaffected.
///
/// Errors in the trace:
///  As there could be errors when reconstructing the instructions of a trace,
///  these errors are represented as failed instructions, and the cursor can
///  point at them. The consumer should invoke \a TraceCursor::GetError() to
///  check if the cursor is pointing to either a valid instruction or an error.
///
/// Instructions:
///  A \a TraceCursor always points to a specific instruction or error in the
///  trace.
///
///  The Trace initially points to the last item in the trace.
///
/// Sample usage:
///
///  TraceCursorUP cursor = trace.GetTrace(thread);
///
///  auto granularity = eTraceInstructionControlFlowTypeCall |
///  eTraceInstructionControlFlowTypeReturn;
///
///  do {
///     if (llvm::Error error = cursor->GetError())
///       cout << "error found at: " << llvm::toString(error) << endl;
///     else if (cursor->GetInstructionControlFlowType() &
///     eTraceInstructionControlFlowTypeCall)
///       std::cout << "call found at " << cursor->GetLoadAddress() <<
///       std::endl;
///     else if (cursor->GetInstructionControlFlowType() &
///     eTraceInstructionControlFlowTypeReturn)
///       std::cout << "return found at " << cursor->GetLoadAddress() <<
///       std::endl;
///  } while(cursor->Prev(granularity));
class TraceCursor {
public:
  virtual ~TraceCursor() = default;

  /// Move the cursor to the next instruction more recent chronologically in the
  /// trace given the provided granularity. If such instruction is not found,
  /// the cursor doesn't move.
  ///
  /// \param[in] granularity
  ///     Bitmask granularity filter. The cursor stops at the next
  ///     instruction that matches the specified granularity.
  ///
  /// \param[in] ignore_errors
  ///     If \b false, the cursor stops as soon as it finds a failure in the
  ///     trace and points at it.
  ///
  /// \return
  ///     \b true if the cursor effectively moved and now points to a different
  ///     item in the trace, including errors when \b ignore_errors is \b false.
  ///     In other words, if \b false is returned, then the trace is pointing at
  ///     the same item in the trace as before.
  virtual bool Next(lldb::TraceInstructionControlFlowType granularity =
                        lldb::eTraceInstructionControlFlowTypeInstruction,
                    bool ignore_errors = false) = 0;

  /// Similar to \a TraceCursor::Next(), but moves backwards chronologically.
  virtual bool Prev(lldb::TraceInstructionControlFlowType granularity =
                        lldb::eTraceInstructionControlFlowTypeInstruction,
                    bool ignore_errors = false) = 0;

  /// Force the cursor to point to the end of the trace, i.e. the most recent
  /// item.
  virtual void SeekToEnd() = 0;

  /// Force the cursor to point to the beginning of the trace, i.e. the oldest
  /// item.
  virtual void SeekToBegin() = 0;

  /// \return
  ///   \b true if the trace corresponds to a live process who has resumed after
  ///   the trace cursor was created. Otherwise, including the case in which the
  ///   process is a post-mortem one, return \b false.
  bool IsStale();

  /// Instruction or error information
  /// \{

  /// Get the corresponding error message if the cursor points to an error in
  /// the trace.
  ///
  /// \return
  ///     \b llvm::Error::success if the cursor is not pointing to an error in
  ///     the trace. Otherwise return an \a llvm::Error describing the issue.
  virtual llvm::Error GetError() = 0;

  /// \return
  ///     The load address of the instruction the cursor is pointing at. If the
  ///     cursor points to an error in the trace, return \b
  ///     LLDB_INVALID_ADDRESS.
  virtual lldb::addr_t GetLoadAddress() = 0;

  /// \return
  ///     The \a lldb::TraceInstructionControlFlowType categories the
  ///     instruction the cursor is pointing at falls into. If the cursor points
  ///     to an error in the trace, return \b 0.
  virtual lldb::TraceInstructionControlFlowType
  GetInstructionControlFlowType() = 0;

  /// \}

private:
  /// The stop ID when the cursor was created.
  uint32_t m_stop_id = 0;
  /// The trace that owns this cursor.
  lldb::TraceSP m_trace_sp;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_CURSOR_H
