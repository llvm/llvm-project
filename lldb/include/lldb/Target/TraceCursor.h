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
///  resumed and new trace data is collected, then it's up to each trace plug-in
///  to decide whether to leave the old cursor unaffected or not.
///
/// Errors in the trace:
///  As there could be errors when reconstructing the instructions of a trace,
///  these errors are represented as failed instructions, and the cursor can
///  point at them. The consumer should invoke \a TraceCursor::IsError() to
///  check if the cursor is pointing to either a valid instruction or an error,
///  and then \a TraceCursor::GetError() can return the actual error message.
///
/// Instructions:
///  A \a TraceCursor always points to a specific instruction or error in the
///  trace.
///
/// Defaults:
///   By default, the cursor points at the most recent item in the trace and is
///   set up to iterate backwards. See the \a TraceCursor::Next() method for
///   more documentation.
///
/// Sample usage:
///
///  TraceCursorUP cursor = trace.GetTrace(thread);
///
///  for (; cursor->HasValue(); cursor->Next()) {
///     if (cursor->IsError()) {
///       cout << "error found at: " << cursor->GetError() << endl;
///       continue;
///     }
///
///     switch (cursor->GetInstructionControlFlowType()) {
///       eTraceInstructionControlFlowTypeCall:
///         std::cout << "CALL found at " << cursor->GetLoadAddress() <<
///         std::endl; break;
///       eTraceInstructionControlFlowTypeReturn:
///         std::cout << "RETURN found at " << cursor->GetLoadAddress() <<
///         std::endl; break;
///     }
///  }
///
///  As the trace might be empty or the cursor might have reached the end of the
///  trace, you should always invoke \a HasValue() to make sure you don't access
///  invalid memory.
///
/// Random accesses:
///
///   The Trace Cursor offer random acesses in the trace via two APIs:
///
///     TraceCursor::Seek():
///       Unlike the \a TraceCursor::Next() API, which moves instruction by
///       instruction, the \a TraceCursor::Seek() method can be used to
///       reposition the cursor to an offset of the end, beginning, or current
///       position of the trace.
///
///     TraceCursor::GetId() / TraceCursor::SetId(id):
///       Each item (error or instruction) in the trace has a numeric identifier
///       which is defined by the trace plug-in. It's possible to access the id
///       of the current item using GetId(), and to reposition the cursor to a
///       given id using SetId(id).
///
///   You can read more in the documentation of these methods.
class TraceCursor {
public:
  /// Helper enum to indicate the reference point when invoking
  /// \a TraceCursor::Seek().
  /// The following values are inspired by \a std::istream::seekg.
  enum class SeekType {
    /// The beginning of the trace, i.e the oldest item.
    Beginning = 0,
    /// The current position in the trace.
    Current,
    /// The end of the trace, i.e the most recent item.
    End
  };

  /// Create a cursor that initially points to the end of the trace, i.e. the
  /// most recent item.
  TraceCursor(lldb::ThreadSP thread_sp);

  virtual ~TraceCursor() = default;

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

  /// Move the cursor to the next item (instruction or error).
  ///
  /// Direction:
  ///     The traversal is done following the current direction of the trace. If
  ///     it is forwards, the instructions are visited forwards
  ///     chronologically. Otherwise, the traversal is done in
  ///     the opposite direction. By default, a cursor moves backwards unless
  ///     changed with \a TraceCursor::SetForwards().
  virtual void Next() = 0;

  /// \return
  ///     \b true if the cursor is pointing to a valid item. \b false if the
  ///     cursor has reached the end of the trace.
  virtual bool HasValue() const = 0;

  /// Instruction identifiers:
  ///
  /// When building complex higher level tools, fast random accesses in the
  /// trace might be needed, for which each instruction requires a unique
  /// identifier within its thread trace. For example, a tool might want to
  /// repeatedly inspect random consecutive portions of a trace. This means that
  /// it will need to first move quickly to the beginning of each section and
  /// then start its iteration. Given that the number of instructions can be in
  /// the order of hundreds of millions, fast random access is necessary.
  ///
  /// An example of such a tool could be an inspector of the call graph of a
  /// trace, where each call is represented with its start and end instructions.
  /// Inspecting all the instructions of a call requires moving to its first
  /// instruction and then iterating until the last instruction, which following
  /// the pattern explained above.
  ///
  /// Instead of using 0-based indices as identifiers, each Trace plug-in can
  /// decide the nature of these identifiers and thus no assumptions can be made
  /// regarding their ordering and sequentiality. The reason is that an
  /// instruction might be encoded by the plug-in in a way that hides its actual
  /// 0-based index in the trace, but it's still possible to efficiently find
  /// it.
  ///
  /// Requirements:
  /// - For a given thread, no two instructions have the same id.
  /// - In terms of efficiency, moving the cursor to a given id should be as
  ///   fast as possible, but not necessarily O(1). That's why the recommended
  ///   way to traverse sequential instructions is to use the \a
  ///   TraceCursor::Next() method and only use \a TraceCursor::GoToId(id)
  ///   sparingly.

  /// Make the cursor point to the item whose identifier is \p id.
  ///
  /// \return
  ///     \b true if the given identifier exists and the cursor effectively
  ///     moved to it. Otherwise, \b false is returned and the cursor now points
  ///     to an invalid item, i.e. calling \a HasValue() will return \b false.
  virtual bool GoToId(lldb::user_id_t id) = 0;

  /// \return
  ///     \b true if and only if there's an instruction item with the given \p
  ///     id.
  virtual bool HasId(lldb::user_id_t id) const = 0;

  /// \return
  ///     A unique identifier for the instruction or error this cursor is
  ///     pointing to.
  virtual lldb::user_id_t GetId() const = 0;
  /// \}

  /// Make the cursor point to an item in the trace based on an origin point and
  /// an offset.
  ///
  /// The resulting position of the trace is
  ///     origin + offset
  ///
  /// If this resulting position would be out of bounds, the trace then points
  /// to an invalid item, i.e. calling \a HasValue() returns \b false.
  ///
  /// \param[in] offset
  ///     How many items to move forwards (if positive) or backwards (if
  ///     negative) from the given origin point. For example, if origin is \b
  ///     End, then a negative offset would move backward in the trace, but a
  ///     positive offset would move past the trace to an invalid item.
  ///
  /// \param[in] origin
  ///     The reference point to use when moving the cursor.
  ///
  /// \return
  ///     \b true if and only if the cursor ends up pointing to a valid item.
  virtual bool Seek(int64_t offset, SeekType origin) = 0;

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
  ///     \b nullptr if the cursor is not pointing to an error in
  ///     the trace. Otherwise return the actual error message.
  virtual const char *GetError() = 0;

  /// \return
  ///     The load address of the instruction the cursor is pointing at. If the
  ///     cursor points to an error in the trace, return \b
  ///     LLDB_INVALID_ADDRESS.
  virtual lldb::addr_t GetLoadAddress() = 0;

  /// Get the hardware counter of a given type associated with the current
  /// instruction. Each architecture might support different counters. It might
  /// happen that only some instructions of an entire trace have a given counter
  /// associated with them.
  ///
  /// \param[in] counter_type
  ///    The counter type.
  /// \return
  ///    The value of the counter or \b llvm::None if not available.
  virtual llvm::Optional<uint64_t> GetCounter(lldb::TraceCounter counter_type) = 0;

  /// Get a bitmask with a list of events that happened chronologically right
  /// before the current instruction or error, but after the previous
  /// instruction.
  ///
  /// \return
  ///   The bitmask of events.
  virtual lldb::TraceEvents GetEvents() = 0;

  /// \return
  ///     The \a lldb::TraceInstructionControlFlowType categories the
  ///     instruction the cursor is pointing at falls into. If the cursor points
  ///     to an error in the trace, return \b 0.
  virtual lldb::TraceInstructionControlFlowType
  GetInstructionControlFlowType() = 0;
  /// \}

protected:
  ExecutionContextRef m_exe_ctx_ref;
  bool m_forwards = false;
};

namespace trace_event_utils {
/// Convert an individual event to a display string.
///
/// \param[in] event
///     An individual event.
///
/// \return
///     A display string for that event, or nullptr if wrong data is passed
///     in.
const char *EventToDisplayString(lldb::TraceEvents event);

/// Invoke the given callback for each individual event of the given events
/// bitmask.
///
/// \param[in] events
///     A list of events to inspect.
///
/// \param[in] callback
///     The callback to invoke for each event.
void ForEachEvent(lldb::TraceEvents events,
                  std::function<void(lldb::TraceEvents event)> callback);
} // namespace trace_event_utils

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_CURSOR_H
