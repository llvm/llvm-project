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
#include "lldb/Target/TraceSettingsParser.h"
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
class Trace : public PluginInterface {
public:
  ~Trace() override = default;

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
  ///     The debugger instance were new Target will be created as part of the
  ///     JSON data parsing.
  ///
  /// \param[in] settings
  ///     JSON object describing a trace.
  ///
  /// \param[in] settings_dir
  ///     Path to a directory used to resolve relative paths in the JSON data.
  ///     If the JSON data is defined in a file, this should be the
  ///     folder containing it.
  static llvm::Expected<lldb::TraceSP>
  FindPlugin(Debugger &debugger, const llvm::json::Value &settings,
             llvm::StringRef settings_dir);

  /// Create an instance of trace plug-in by name.
  ///
  /// \param[in] plugin_name
  ///     Name of the trace plugin.
  static llvm::Expected<lldb::TraceSP> FindPlugin(llvm::StringRef plugin_name);

  /// Parse the JSON settings and create the corresponding \a Target
  /// objects. In case of an error, no targets are created.
  ///
  /// \param[in] debugger
  ///   The debugger instance where the targets are created.
  ///
  /// \param[in] settings
  ///     JSON object describing a trace.
  ///
  /// \param[in] settings_dir
  ///     Path to a directory used to resolve relative paths in the JSON data.
  ///     If the JSON data is defined in a file, this should be the
  ///     folder containing it.
  ///
  /// \return
  ///   An error object containing the reason if there is a failure.
  llvm::Error ParseSettings(Debugger &debugger,
                            const llvm::json::Value &settings,
                            llvm::StringRef settings_dir);

  /// Get the JSON schema of the settings for the trace plug-in.
  llvm::StringRef GetSchema();

protected:
  Trace() {}

  /// The actual plug-in should define its own implementation of \a
  /// TraceSettingsParser for doing any custom parsing.
  virtual std::unique_ptr<lldb_private::TraceSettingsParser> CreateParser() = 0;

private:
  Trace(const Trace &) = delete;
  const Trace &operator=(const Trace &) = delete;

protected:
  friend class TraceSettingsParser;
  /// JSON object that holds all settings for this trace session.
  llvm::json::Object m_settings;
  /// The directory that contains the settings file.
  std::string m_settings_dir;

  std::map<lldb::pid_t, std::map<lldb::tid_t, lldb_private::FileSpec>>
      m_thread_to_trace_file_map;
  std::vector<lldb::TargetSP> m_targets;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACE_H
