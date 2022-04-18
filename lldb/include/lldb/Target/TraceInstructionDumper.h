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

/// Class that holds the configuration used by \a TraceInstructionDumper for
/// traversing and dumping instructions.
struct TraceInstructionDumperOptions {
  /// If \b true, the cursor will be iterated forwards starting from the
  /// oldest instruction. Otherwise, the iteration starts from the most
  /// recent instruction.
  bool forwards = false;
  /// Dump only instruction addresses without disassembly nor symbol
  /// information.
  bool raw = false;
  /// For each instruction, print the corresponding timestamp counter if
  /// available.
  bool show_tsc = false;
  /// Dump the events that happened between instructions.
  bool show_events = false;
  /// Optional custom id to start traversing from.
  llvm::Optional<uint64_t> id = llvm::None;
  /// Optional number of instructions to skip from the starting position
  /// of the cursor.
  llvm::Optional<size_t> skip = llvm::None;
};

/// Class used to dump the instructions of a \a TraceCursor using its current
/// state and granularity.
class TraceInstructionDumper {
public:
  /// Create a instruction dumper for the cursor.
  ///
  /// \param[in] cursor
  ///     The cursor whose instructions will be dumped.
  ///
  /// \param[in] s
  ///     The stream where to dump the instructions to.
  ///
  /// \param[in] options
  ///     Additional options for configuring the dumping.
  TraceInstructionDumper(lldb::TraceCursorUP &&cursor_up, Stream &s,
                         const TraceInstructionDumperOptions &options);

  /// Dump \a count instructions of the thread trace starting at the current
  /// cursor position.
  ///
  /// This effectively moves the cursor to the next unvisited position, so that
  /// a subsequent call to this method continues where it left off.
  ///
  /// \param[in] count
  ///     The number of instructions to print.
  ///
  /// \return
  ///     The instruction id of the last traversed instruction, or \b llvm::None
  ///     if no instructions were visited.
  llvm::Optional<lldb::user_id_t> DumpInstructions(size_t count);

  /// \return
  ///     \b true if there's still more data to traverse in the trace.
  bool HasMoreData();

private:
  /// Indicate to the dumper that no more data is available in the trace.
  /// This will prevent further iterations.
  void SetNoMoreData();

  /// Move the cursor one step.
  ///
  /// \return
  ///     \b true if the cursor moved.
  bool TryMoveOneStep();

  void PrintEvents();

  lldb::TraceCursorUP m_cursor_up;
  TraceInstructionDumperOptions m_options;
  Stream &m_s;
  /// If \b true, all the instructions have been traversed.
  bool m_no_more_data = false;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_INSTRUCTION_DUMPER_H
