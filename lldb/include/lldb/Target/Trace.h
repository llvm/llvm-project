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
#include "lldb/Target/TraceCursor.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/TraceGDBRemotePackets.h"
#include "lldb/Utility/UnimplementedError.h"
#include "lldb/lldb-private.h"
#include "lldb/lldb-types.h"

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
  /// Dump the trace data that this plug-in has access to.
  ///
  /// This function will dump all of the trace data for all threads in a user
  /// readable format. Options for dumping can be added as this API is iterated
  /// on.
  ///
  /// \param[in] s
  ///     A stream object to dump the information to.
  virtual void Dump(Stream *s) const = 0;

  /// Save the trace of a live process to the specified directory, which
  /// will be created if needed.
  /// This will also create a a file \a <directory>/trace.json with the main
  /// properties of the trace session, along with others files which contain
  /// the actual trace data. The trace.json file can be used later as input
  /// for the "trace load" command to load the trace in LLDB.
  /// The process being trace is not a live process, return an error.
  ///
  /// \param[in] directory
  ///   The directory where the trace files will be saved.
  ///
  /// \return
  ///   \a llvm::success if the operation was successful, or an \a llvm::Error
  ///   otherwise.
  virtual llvm::Error SaveLiveTraceToDisk(FileSpec directory) = 0;

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

  /// Get a \a TraceCursor for the given thread's trace.
  ///
  /// \return
  ///     A \a TraceCursorUP. If the thread is not traced or its trace
  ///     information failed to load, the corresponding error is embedded in the
  ///     trace.
  virtual lldb::TraceCursorUP GetCursor(Thread &thread) = 0;

  /// Dump general info about a given thread's trace. Each Trace plug-in
  /// decides which data to show.
  ///
  /// \param[in] thread
  ///     The thread that owns the trace in question.
  ///
  /// \param[in] s
  ///     The stream object where the info will be printed printed.
  ///
  /// \param[in] verbose
  ///     If \b true, print detailed info
  ///     If \b false, print compact info
  virtual void DumpTraceInfo(Thread &thread, Stream &s, bool verbose) = 0;

  /// Check if a thread is currently traced by this object.
  ///
  /// \param[in] tid
  ///     The id of the thread in question.
  ///
  /// \return
  ///     \b true if the thread is traced by this instance, \b false otherwise.
  virtual bool IsTraced(lldb::tid_t tid) = 0;

  /// \return
  ///     A description of the parameters to use for the \a Trace::Start method.
  virtual const char *GetStartConfigurationHelp() = 0;

  /// Start tracing a live process.
  ///
  /// \param[in] configuration
  ///     See \a SBTrace::Start(const lldb::SBStructuredData &) for more
  ///     information.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  virtual llvm::Error Start(
      StructuredData::ObjectSP configuration = StructuredData::ObjectSP()) = 0;

  /// Start tracing live threads.
  ///
  /// \param[in] tids
  ///     Threads to trace. This method tries to trace as many threads as
  ///     possible.
  ///
  /// \param[in] configuration
  ///     See \a SBTrace::Start(const lldb::SBThread &, const
  ///     lldb::SBStructuredData &) for more information.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  virtual llvm::Error Start(
      llvm::ArrayRef<lldb::tid_t> tids,
      StructuredData::ObjectSP configuration = StructuredData::ObjectSP()) = 0;

  /// Stop tracing live threads.
  ///
  /// \param[in] tids
  ///     The threads to stop tracing on.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error Stop(llvm::ArrayRef<lldb::tid_t> tids);

  /// Stop tracing all current and future threads of a live process.
  ///
  /// \param[in] request
  ///     The information determining which threads or process to stop tracing.
  ///
  /// \return
  ///     \a llvm::Error::success if the operation was successful, or
  ///     \a llvm::Error otherwise.
  llvm::Error Stop();

  /// \return
  ///     The stop ID of the live process being traced, or an invalid stop ID
  ///     if the trace is in an error or invalid state.
  uint32_t GetStopID();

  using OnBinaryDataReadCallback =
      std::function<llvm::Error(llvm::ArrayRef<uint8_t> data)>;
  /// Fetch binary data associated with a thread, either live or postmortem, and
  /// pass it to the given callback. The reason of having a callback is to free
  /// the caller from having to manage the life cycle of the data and to hide
  /// the different data fetching procedures that exist for live and post mortem
  /// threads.
  ///
  /// The fetched data is not persisted after the callback is invoked.
  ///
  /// \param[in] tid
  ///     The tid who owns the data.
  ///
  /// \param[in] kind
  ///     The kind of data to read.
  ///
  /// \param[in] callback
  ///     The callback to be invoked once the data was successfully read. Its
  ///     return value, which is an \a llvm::Error, is returned by this
  ///     function.
  ///
  /// \return
  ///     An \a llvm::Error if the data couldn't be fetched, or the return value
  ///     of the callback, otherwise.
  llvm::Error OnThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                     OnBinaryDataReadCallback callback);

  /// Get the current traced live process.
  ///
  /// \return
  ///     The current traced live process. If it's not a live process,
  ///     return \a nullptr.
  Process *GetLiveProcess();

protected:
  /// Implementation of \a OnThreadBinaryDataRead() for live threads.
  llvm::Error OnLiveThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                         OnBinaryDataReadCallback callback);

  /// Implementation of \a OnThreadBinaryDataRead() for post mortem threads.
  llvm::Error
  OnPostMortemThreadBinaryDataRead(lldb::tid_t tid, llvm::StringRef kind,
                                   OnBinaryDataReadCallback callback);

  /// Get the file path containing data of a postmortem thread given a data
  /// identifier.
  ///
  /// \param[in] tid
  ///     The thread whose data is requested.
  ///
  /// \param[in] kind
  ///     The kind of data requested.
  ///
  /// \return
  ///     The file spec containing the requested data, or an \a llvm::Error in
  ///     case of failures.
  llvm::Expected<FileSpec> GetPostMortemThreadDataFile(lldb::tid_t tid,
                                                       llvm::StringRef kind);

  /// Associate a given thread with a data file using a data identifier.
  ///
  /// \param[in] tid
  ///     The thread associated with the data file.
  ///
  /// \param[in] kind
  ///     The kind of data being registered.
  ///
  /// \param[in] file_spec
  ///     The path of the data file.
  void SetPostMortemThreadDataFile(lldb::tid_t tid, llvm::StringRef kind,
                                   FileSpec file_spec);

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
  ///
  /// \param[in] json_response
  ///     The original JSON response as a string. It might be useful to redecode
  ///     it if it contains custom data for a specific trace plug-in.
  ///
  /// \return
  ///     \b Error::success() if this operation succeedes, or an actual error
  ///     otherwise.
  virtual llvm::Error
  DoRefreshLiveProcessState(TraceGetStateResponse state,
                            llvm::StringRef json_response) = 0;

  /// Method to be invoked by the plug-in to refresh the live process state. It
  /// will invoked DoRefreshLiveProcessState at some point, which should be
  /// implemented by the plug-in for custom state handling.
  ///
  /// The result is cached through the same process stop. Even in the case of
  /// errors, it caches the error.
  ///
  /// \return
  ///   An error message if this operation failed, or \b nullptr otherwise.
  const char *RefreshLiveProcessState();

private:
  uint32_t m_stop_id = LLDB_INVALID_STOP_ID;
  /// Process traced by this object if doing live tracing. Otherwise it's null.
  Process *m_live_process = nullptr;

  /// These data kinds are returned by lldb-server when fetching the state of
  /// the tracing session. The size in bytes can be used later for fetching the
  /// data in batches.
  /// \{

  /// tid -> data kind -> size
  llvm::DenseMap<lldb::tid_t, std::unordered_map<std::string, size_t>>
      m_live_thread_data;

  /// data kind -> size
  std::unordered_map<std::string, size_t> m_live_process_data;
  /// \}

  /// Postmortem traces can specific additional data files, which are
  /// represented in this variable using a data kind identifier for each file.
  /// tid -> data kind -> file
  llvm::DenseMap<lldb::tid_t, std::unordered_map<std::string, FileSpec>>
      m_postmortem_thread_data;

  llvm::Optional<std::string> m_live_refresh_error;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_H
