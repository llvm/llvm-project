//===-- Trace.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACE_H
#define LLDB_TARGET_TRACE_H

#include "llvm/Support/JSON.h"

#include "lldb/Core/PluginInterface.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// \class Trace Trace.h "lldb/Target/Trace.h"
/// A plug-in interface definition class for trace information.
///
/// Trace plug-ins allow processor trace information to be loaded into LLDB so
/// that the data can be dumped, used for reverse and forward stepping to allow
/// introspection into the reason your process crashed or found its way to its
/// current state.
///
/// Trace information can be loaded into a target without a process to allow
/// introspection of the trace information during post mortem analysis, such as
/// when loading core files.
///
/// Processor trace information can also be fetched through the process
/// interfaces during a live debug session if your process supports gathering
/// this information.
///
/// In order to support live tracing, the name of the plug-in should match the
/// name of the tracing type returned by the gdb-remote packet
/// \a jLLDBTraceSupportedType.
class Trace : public PluginInterface,
              public std::enable_shared_from_this<Trace> {
public:
  enum class TraceDirection {
    Forwards = 0,
    Backwards,
  };

  /// Dump the trace data that this plug-in has access to.
  ///
  /// This function will dump all of the trace data for all threads in a user
  /// readable format. Options for dumping can be added as this API is iterated
  /// on.
  ///
  /// \param[in] s
  ///     A stream object to dump the information to.
  virtual void Dump(Stream *s) const = 0;

  /// Find a trace plug-in using JSON data.
  ///
  /// When loading trace data from disk, the information for the trace data
  /// can be contained in multiple files and require plug-in specific
  /// information about the CPU. Using data like JSON provides an
  /// easy way to specify all of the settings and information that we will need
  /// to load trace data into LLDB. This structured data can include:
  ///   - The plug-in name (this allows a specific plug-in to be selected)
  ///   - Architecture or target triple
  ///   - one or more paths to the trace data file on disk
  ///     - core trace data
  ///     - thread events or related information
  ///   - shared library load information to use for this trace data that
  ///     allows a target to be created so the trace information can be
  ///     symbolicated so that the trace information can be displayed to the
  ///     user
  ///     - shared library path
  ///     - load address
  ///     - information on how to fetch the shared library
  ///       - path to locally cached file on disk
  ///       - URL to download the file
  ///   - Any information needed to load the trace file
  ///     - CPU information
  ///     - Custom plug-in information needed to decode the trace information
  ///       correctly.
  ///
  /// \param[in] debugger
  ///     The debugger instance where new Targets will be created as part of the
  ///     JSON data parsing.
  ///
  /// \param[in] trace_session_file
  ///     The contents of the trace session file describing the trace session.
  ///     See \a TraceSessionFileParser::BuildSchema for more information about
  ///     the schema of this JSON file.
  ///
  /// \param[in] session_file_dir
  ///     The path to the directory that contains the session file. It's used to
  ///     resolved relative paths in the session file.
  static llvm::Expected<lldb::TraceSP>
  FindPlugin(Debugger &debugger, const llvm::json::Value &trace_session_file,
             llvm::StringRef session_file_dir);

  /// Get the schema of a Trace plug-in given its name.
  ///
  /// \param[in] plugin_name
  ///     Name of the trace plugin.
  static llvm::Expected<llvm::StringRef>
  FindPluginSchema(llvm::StringRef plugin_name);

  /// \return
  ///     The JSON schema of this Trace plug-in.
  virtual llvm::StringRef GetSchema() = 0;

  /// Each decoded thread contains a cursor to the current position the user is
  /// stopped at. When reverse debugging, each operation like reverse-next or
  /// reverse-continue will move this cursor, which is then picked by any
  /// subsequent dump or reverse operation.
  ///
  /// The initial position for this cursor is the last element of the thread,
  /// which is the most recent chronologically.
  ///
  /// \return
  ///     The current position of the thread's trace or \b 0 if empty.
  virtual size_t GetCursorPosition(const Thread &thread) = 0;

  /// Dump \a count instructions of the given thread's trace ending at the
  /// given \a end_position position.
  ///
  /// The instructions are printed along with their indices or positions, which
  /// are increasing chronologically. This means that the \a index 0 represents
  /// the oldest instruction of the trace chronologically.
  ///
  /// \param[in] thread
  ///     The thread whose trace will be dumped.
  ///
  /// \param[in] s
  ///     The stream object where the instructions are printed.
  ///
  /// \param[in] count
  ///     The number of instructions to print.
  ///
  /// \param[in] end_position
  ///     The position of the last instruction to print.
  ///
  /// \param[in] raw
  ///     Dump only instruction addresses without disassembly nor symbol
  ///     information.
  void DumpTraceInstructions(Thread &thread, Stream &s, size_t count,
                             size_t end_position, bool raw);

  /// Run the provided callback on the instructions of the trace of the given
  /// thread.
  ///
  /// The instructions will be traversed starting at the given \a position
  /// sequentially until the callback returns \b false, in which case no more
  /// instructions are inspected.
  ///
  /// The purpose of this method is to allow inspecting traced instructions
  /// without exposing the internal representation of how they are stored on
  /// memory.
  ///
  /// \param[in] thread
  ///     The thread whose trace will be traversed.
  ///
  /// \param[in] position
  ///     The instruction position to start iterating on.
  ///
  /// \param[in] direction
  ///     If \b TraceDirection::Forwards, then then instructions will be
  ///     traversed forwards chronologically, i.e. with incrementing indices. If
  ///     \b TraceDirection::Backwards, the traversal is done backwards
  ///     chronologically, i.e. with decrementing indices.
  ///
  /// \param[in] callback
  ///     The callback to execute on each instruction. If it returns \b false,
  ///     the iteration stops.
  virtual void TraverseInstructions(
      const Thread &thread, size_t position, TraceDirection direction,
      std::function<bool(size_t index, llvm::Expected<lldb::addr_t> load_addr)>
          callback) = 0;

  /// Get the number of available instructions in the trace of the given thread.
  ///
  /// \param[in] thread
  ///     The thread whose trace will be inspected.
  ///
  /// \return
  ///     The total number of instructions in the trace.
  virtual size_t GetInstructionCount(const Thread &thread) = 0;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_H
