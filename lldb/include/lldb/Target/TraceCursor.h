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

#include "lldb/Target/ExecutionContext.h"

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
/// Defaults:
///   By default, the cursor points at the end item of the trace, moves
///   backwards, has a move granularity of \a
///   eTraceInstructionControlFlowTypeInstruction (i.e. visit every instruction)
///   and stops at every error (the "ignore errors" flag is \b false). See the
///   \a TraceCursor::Next() method for more documentation.
///
/// Sample usage:
///
///  TraceCursorUP cursor = trace.GetTrace(thread);
///
///  cursor->SetGranularity(eTraceInstructionControlFlowTypeCall |
///    eTraceInstructionControlFlowTypeReturn);
///
///  do {
///     if (llvm::Error error = cursor->GetError())
///       cout << "error found at: " << llvm::toString(error) << endl;
///     else if (cursor->GetInstructionControlFlowType() &
///         eTraceInstructionControlFlowTypeCall)
///       std::cout << "call found at " << cursor->GetLoadAddress() <<
///       std::endl;
///     else if (cursor->GetInstructionControlFlowType() &
///         eTraceInstructionControlFlowTypeReturn)
///       std::cout << "return found at " << cursor->GetLoadAddress() <<
///       std::endl;
///  } while(cursor->Next());
///
/// Low level traversal:
///   Unlike the \a TraceCursor::Next() API, which uses a given granularity and
///   direction to advance the cursor, the \a TraceCursor::Seek() method can be
///   used to reposition the cursor to an offset of the end, beginning, or
///   current position of the trace.
class TraceCursor {
public:
  /// Helper enum to indicate the reference point when invoking
  /// \a TraceCursor::Seek().
  enum class SeekType {
    /// The beginning of the trace, i.e the oldest item.
    Set = 0,
    /// The current position in the trace.
    Current,
    /// The end of the trace, i.e the most recent item.
    End
  };

  /// Create a cursor that initially points to the end of the trace, i.e. the
  /// most recent item.
  TraceCursor(lldb::ThreadSP thread_sp);

  virtual ~TraceCursor() = default;

  /// Set the granularity to use in the \a TraceCursor::Next() method.
  void SetGranularity(lldb::TraceInstructionControlFlowType granularity);

  /// Set the "ignore errors" flag to use in the \a TraceCursor::Next() method.
  void SetIgnoreErrors(bool ignore_errors);

  /// Set the direction to use in the \a TraceCursor::Next() method.
  ///
  /// \param[in] forwards
  ///     If \b true, then the traversal will be forwards, otherwise backwards.
  void SetForwards(bool forwards);

  /// Check if the direction to use in the \a TraceCursor::Next() method is
  /// forwards.
  ///
  /// \return
  ///     \b true if the current direction is forwards, \b false if backwards.
  bool IsForwards() const;

  /// Move the cursor to the next instruction that matches the current
  /// granularity.
  ///
  /// Direction:
  ///     The traversal is done following the current direction of the trace. If
  ///     it is forwards, the instructions are visited forwards
  ///     chronologically. Otherwise, the traversal is done in
  ///     the opposite direction. By default, a cursor moves backwards unless
  ///     changed with \a TraceCursor::SetForwards().
  ///
  /// Granularity:
  ///     The cursor will traverse the trace looking for the first instruction
  ///     that matches the current granularity. If there aren't any matching
  ///     instructions, the cursor won't move, to give the opportunity of
  ///     changing granularities.
  ///
  /// Ignore errors:
  ///     If the "ignore errors" flags is \b false, the traversal will stop as
  ///     soon as it finds an error in the trace and the cursor will point at
  ///     it.
  ///
  /// \return
  ///     \b true if the cursor effectively moved, \b false otherwise.
  virtual bool Next() = 0;

  /// Make the cursor point to an item in the trace based on an origin point and
  /// an offset. This API doesn't distinguishes instruction types nor errors in
  /// the trace, unlike the \a TraceCursor::Next() method.
  ///
  /// The resulting position of the trace is
  ///     origin + offset
  ///
  /// If this resulting position would be out of bounds, it will be adjusted to
  /// the last or first item in the trace correspondingly.
  ///
  /// \param[in] offset
  ///     How many items to move forwards (if positive) or backwards (if
  ///     negative) from the given origin point.
  ///
  /// \param[in] origin
  ///     The reference point to use when moving the cursor.
  ///
  /// \return
  ///     The number of trace items moved from the origin.
  virtual size_t Seek(ssize_t offset, SeekType origin) = 0;

  /// \return
  ///   The \a ExecutionContextRef of the backing thread from the creation time
  ///   of this cursor.
  ExecutionContextRef &GetExecutionContextRef();

  /// Instruction or error information
  /// \{

  /// \return
  ///     Whether the cursor points to an error or not.
  virtual bool IsError() = 0;

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

  /// Get the timestamp counter associated with the current instruction.
  /// Modern Intel, ARM and AMD processors support this counter. However, a
  /// trace plugin might decide to use a different time unit instead of an
  /// actual TSC.
  ///
  /// \return
  ///     The timestamp or \b llvm::None if not available.
  virtual llvm::Optional<uint64_t> GetTimestampCounter() = 0;

  /// \return
  ///     The \a lldb::TraceInstructionControlFlowType categories the
  ///     instruction the cursor is pointing at falls into. If the cursor points
  ///     to an error in the trace, return \b 0.
  virtual lldb::TraceInstructionControlFlowType
  GetInstructionControlFlowType() = 0;
  /// \}

protected:
  ExecutionContextRef m_exe_ctx_ref;

  lldb::TraceInstructionControlFlowType m_granularity =
      lldb::eTraceInstructionControlFlowTypeInstruction;
  bool m_ignore_errors = false;
  bool m_forwards = false;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_CURSOR_H
