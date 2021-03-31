//===-- Trace.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACE_H
#define LLDB_TARGET_TRACE_H

#include <unordered_map>

#include "llvm/Support/JSON.h"

#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/TraceGDBRemotePackets.h"
#include "lldb/Utility/UnimplementedError.h"
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
/// \a jLLDBTraceSupported.
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
  FindPluginForPostMortemProcess(Debugger &debugger,
                                 const llvm::json::Value &trace_session_file,
                                 llvm::StringRef session_file_dir);

  /// Find a trace plug-in to trace a live process.
  ///
  /// \param[in] plugin_name
  ///     Plug-in name to search.
  ///
  /// \param[in] process
  ///     Live process to trace.
  ///
  /// \return
  ///     A \a TraceSP instance, or an \a llvm::Error if the plug-in name
  ///     doesn't match any registered plug-ins or tracing couldn't be
  ///     started.
  static llvm::Expected<lldb::TraceSP>
  FindPluginForLiveProcess(llvm::StringRef plugin_name, Process &process);

  /// Get the schema of a Trace plug-in given its name.
  ///
  /// \param[in] plugin_name
  ///     Name of the trace plugin.
  static llvm::Expected<llvm::StringRef>
  FindPluginSchema(llvm::StringRef plugin_name);

  /// Get the command handle for the "process trace start" command.
  virtual lldb::CommandObjectSP
  GetProcessTraceStartCommand(CommandInterpreter &interpreter) = 0;

  /// Get the command handle for the "thread trace start" command.
  virtual lldb::CommandObjectSP
  GetThreadTraceStartCommand(CommandInterpreter &interpreter) = 0;

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
  virtual size_t GetCursorPosition(Thread &thread) = 0;

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
      Thread &thread, size_t position, TraceDirection direction,
      std::function<bool(size_t index, llvm::Expected<lldb::addr_t> load_addr)>
          callback) = 0;

  /// Get the number of available instructions in the trace of the given thread.
  ///
  /// \param[in] thread
  ///     The thread whose trace will be inspected.
  ///
  /// \return
  ///     The total number of instructions in the trace.
  virtual size_t GetInstructionCount(Thread &thread) = 0;

  /// Check if a thread is currently traced by this object.
  ///
  /// \param[in] thread
  ///     The thread in question.
  ///
  /// \return
  ///     \b true if the thread is traced by this instance, \b false otherwise.
  virtual bool IsTraced(const Thread &thread) = 0;

  /// Stop tracing live threads.
  ///
  /// \param[in] tids
  ///     The threads to stop tracing on.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error StopThreads(const std::vector<lldb::tid_t> &tids);

  /// Stop tracing a live process.
  ///
  /// \param[in] request
  ///     The information determining which threads or process to stop tracing.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error StopProcess();

  /// Get the trace file of the given post mortem thread.
  llvm::Expected<const FileSpec &> GetPostMortemTraceFile(lldb::tid_t tid);

protected:
  /// Get binary data of a live thread given a data identifier.
  ///
  /// \param[in] tid
  ///     The thread whose data is requested.
  ///
  /// \param[in] kind
  ///     The kind of data requested.
  ///
  /// \return
  ///     A vector of bytes with the requested data, or an \a llvm::Error in
  ///     case of failures.
  llvm::Expected<std::vector<uint8_t>>
  GetLiveThreadBinaryData(lldb::tid_t tid, llvm::StringRef kind);

  /// Get binary data of the current process given a data identifier.
  ///
  /// \param[in] kind
  ///     The kind of data requested.
  ///
  /// \return
  ///     A vector of bytes with the requested data, or an \a llvm::Error in
  ///     case of failures.
  llvm::Expected<std::vector<uint8_t>>
  GetLiveProcessBinaryData(llvm::StringRef kind);

  /// Get the size of the data returned by \a GetLiveThreadBinaryData
  llvm::Optional<size_t> GetLiveThreadBinaryDataSize(lldb::tid_t tid,
                                                     llvm::StringRef kind);

  /// Get the size of the data returned by \a GetLiveProcessBinaryData
  llvm::Optional<size_t> GetLiveProcessBinaryDataSize(llvm::StringRef kind);
  /// Constructor for post mortem processes
  Trace() = default;

  /// Constructor for a live process
  Trace(Process &live_process) : m_live_process(&live_process) {}

  /// Start tracing a live process or its threads.
  ///
  /// \param[in] request
  ///     JSON object with the information necessary to start tracing. In the
  ///     case of gdb-remote processes, this JSON object should conform to the
  ///     jLLDBTraceStart packet.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error Start(const llvm::json::Value &request);

  /// Get the current tracing state of a live process and its threads.
  ///
  /// \return
  ///     A JSON object string with custom data depending on the trace
  ///     technology, or an \a llvm::Error in case of errors.
  llvm::Expected<std::string> GetLiveProcessState();

  /// Method to be overriden by the plug-in to refresh its own state.
  ///
  /// This is invoked by RefreshLiveProcessState when a new state is found.
  ///
  /// \param[in] state
  ///     The jLLDBTraceGetState response.
  virtual void
  DoRefreshLiveProcessState(llvm::Expected<TraceGetStateResponse> state) = 0;

  /// Method to be invoked by the plug-in to refresh the live process state.
  ///
  /// The result is cached through the same process stop.
  void RefreshLiveProcessState();

  /// Process traced by this object if doing live tracing. Otherwise it's null.
  int64_t m_stop_id = -1;
  Process *m_live_process = nullptr;
  /// tid -> data kind -> size
  std::map<lldb::tid_t, std::unordered_map<std::string, size_t>>
      m_live_thread_data;
  /// data kind -> size
  std::unordered_map<std::string, size_t> m_live_process_data;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_H
