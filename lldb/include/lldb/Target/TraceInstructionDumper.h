//===-- TraceInstructionDumper.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TraceCursor.h"

#ifndef LLDB_TARGET_TRACE_INSTRUCTION_DUMPER_H
#define LLDB_TARGET_TRACE_INSTRUCTION_DUMPER_H

namespace lldb_private {

/// Class used to dump the instructions of a \a TraceCursor using its current
/// state and granularity.
class TraceInstructionDumper {
public:
  /// Create a instruction dumper for the cursor.
  ///
  /// \param[in] cursor
  ///     The cursor whose instructions will be dumped.
  ///
  /// \param[in] initial_index
  ///     Presentation index to use for referring to the current instruction
  ///     of the cursor. If the direction is forwards, the index will increase,
  ///     and if the direction is backwards, the index will decrease.
  ///
  /// \param[in] raw
  ///     Dump only instruction addresses without disassembly nor symbol
  ///     information.
  ///
  /// \param[in] show_tsc
  ///     For each instruction, print the corresponding timestamp counter if
  ///     available.
  TraceInstructionDumper(lldb::TraceCursorUP &&cursor_up, int initial_index = 0,
                         bool raw = false, bool show_tsc = false);

  /// Dump \a count instructions of the thread trace starting at the current
  /// cursor position.
  ///
  /// This effectively moves the cursor to the next unvisited position, so that
  /// a subsequent call to this method continues where it left off.
  ///
  /// \param[in] s
  ///     The stream object where the instructions are printed.
  ///
  /// \param[in] count
  ///     The number of instructions to print.
  void DumpInstructions(Stream &s, size_t count);

  /// Indicate the dumper that no more data is available in the trace.
  void SetNoMoreData();

  /// \return
  ///     \b true if there's still more data to traverse in the trace.
  bool HasMoreData();

private:
  /// Move the cursor one step.
  ///
  /// \return
  ///     \b true if the cursor moved.
  bool TryMoveOneStep();

  lldb::TraceCursorUP m_cursor_up;
  int m_index;
  bool m_raw;
  bool m_show_tsc;
  /// If \b true, all the instructions have been traversed.
  bool m_no_more_data = false;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_INSTRUCTION_DUMPER_H
