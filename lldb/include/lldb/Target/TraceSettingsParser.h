//===-- TraceSettingsParser.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACE_SETTINGS_PARSER_H
#define LLDB_TARGET_TRACE_SETTINGS_PARSER_H

#include "llvm/ADT/Optional.h"

#include "lldb/Target/Trace.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// \class TraceSettingsParser TraceSettingsParser.h
/// A plug-in interface definition class for parsing \a Trace settings.
///
/// As \a Trace plug-ins support plug-in specific settings, this class should be
/// overriden and implement the plug-in specific parsing logic.
class TraceSettingsParser {
public:
  TraceSettingsParser(Trace &trace) : m_trace(trace) {}

  virtual ~TraceSettingsParser() = default;

  /// Get the JSON schema of the settings for the trace plug-in.
  llvm::StringRef GetSchema();

  /// Parse the structured data settings and create the corresponding \a Target
  /// objects. In case of and error, no targets are created.
  ///
  /// \param[in] debugger
  ///   The debugger instance where the targets are created.
  ///
  /// \param[in] settings
  ///   The settings to parse.
  ///
  /// \param[in] settings_dir
  ///   The directory that contains the settings file used to resolve relative
  ///   paths.
  ///
  /// \return
  ///   An error object containing the reason if there is a failure.
  llvm::Error ParseSettings(Debugger &debugger,
                            const llvm::json::Object &settings,
                            llvm::StringRef settings_dir);

protected:
  /// Method that should be overriden by implementations of this class to
  /// provide the specific plug-in schema inside the "trace" section of the
  /// global schema.
  virtual llvm::StringRef GetPluginSchema() = 0;

  /// Method that should be overriden to parse the plug-in specific settings.
  ///
  /// \return
  ///   An error object containing the reason if there is a failure.
  virtual llvm::Error ParsePluginSettings() = 0;

private:
  /// Resolve non-absolute paths relativejto the settings folder
  void NormalizePath(lldb_private::FileSpec &file_spec);
  llvm::Error ParseProcess(lldb_private::Debugger &debugger,
                           const llvm::json::Object &process);
  llvm::Error ParseProcesses(lldb_private::Debugger &debugger);
  llvm::Error ParseThread(lldb::ProcessSP &process_sp,
                          const llvm::json::Object &thread);
  llvm::Error ParseThreads(lldb::ProcessSP &process_sp,
                           const llvm::json::Object &process);
  llvm::Error ParseModule(lldb::TargetSP &target_sp,
                          const llvm::json::Object &module);
  llvm::Error ParseModules(lldb::TargetSP &target_sp,
                           const llvm::json::Object &process);
  llvm::Error ParseSettingsImpl(lldb_private::Debugger &debugger);

  Trace &m_trace;

protected:
  /// Objects created as product of the parsing
  /// \{
  /// JSON object that holds all settings for this trace session.
  llvm::json::Object m_settings;
  /// The directory that contains the settings file.
  std::string m_settings_dir;

  std::map<lldb::pid_t, std::map<lldb::tid_t, lldb_private::FileSpec>>
      m_thread_to_trace_file_map;
  std::vector<lldb::TargetSP> m_targets;
  /// \}
};

} // namespace lldb_private

namespace json_helpers {
/// JSON parsing helpers based on \a llvm::Expected.
/// \{
llvm::Error CreateWrongTypeError(const llvm::json::Value &value,
                                 llvm::StringRef type);

llvm::Expected<int64_t> ToIntegerOrError(const llvm::json::Value &value);

llvm::Expected<llvm::StringRef> ToStringOrError(const llvm::json::Value &value);

llvm::Expected<const llvm::json::Array &>
ToArrayOrError(const llvm::json::Value &value);

llvm::Expected<const llvm::json::Object &>
ToObjectOrError(const llvm::json::Value &value);

llvm::Error CreateMissingKeyError(llvm::json::Object obj, llvm::StringRef key);

llvm::Expected<const llvm::json::Value &>
GetValueOrError(const llvm::json::Object &obj, llvm::StringRef key);

llvm::Expected<int64_t> GetIntegerOrError(const llvm::json::Object &obj,
                                          llvm::StringRef key);

llvm::Expected<llvm::StringRef> GetStringOrError(const llvm::json::Object &obj,
                                                 llvm::StringRef key);

llvm::Expected<const llvm::json::Array &>
GetArrayOrError(const llvm::json::Object &obj, llvm::StringRef key);

llvm::Expected<const llvm::json::Object &>
GetObjectOrError(const llvm::json::Object &obj, llvm::StringRef key);

llvm::Expected<llvm::Optional<llvm::StringRef>>
GetOptionalStringOrError(const llvm::json::Object &obj, llvm::StringRef key);
/// \}
} // namespace json_helpers

#endif // LLDB_TARGET_TRACE_SETTINGS_PARSER_H
