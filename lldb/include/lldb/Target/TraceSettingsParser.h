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
  struct JSONAddress {
    lldb::addr_t value;
  };

  struct JSONModule {
    std::string system_path;
    llvm::Optional<std::string> file;
    JSONAddress load_address;
    llvm::Optional<std::string> uuid;
  };

  struct JSONThread {
    int64_t tid;
    std::string trace_file;
  };

  struct JSONProcess {
    int64_t pid;
    std::string triple;
    std::vector<JSONThread> threads;
    std::vector<JSONModule> modules;
  };

  struct JSONTracePluginSettings {
    std::string type;
  };

  struct JSONTraceSettings {
    std::vector<JSONProcess> processes;
    JSONTracePluginSettings trace;
  };

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
                            const llvm::json::Value &settings,
                            llvm::StringRef settings_dir);

protected:
  /// Method that should be overriden by implementations of this class to
  /// provide the specific plug-in schema inside the "trace" section of the
  /// global schema.
  virtual llvm::StringRef GetPluginSchema() = 0;

  /// Method that should be overriden to parse the plug-in specific settings.
  ///
  /// \param[in] plugin_settings
  ///   The settings to parse specific to the plugin.
  ///
  /// \return
  ///   An error object containing the reason if there is a failure.
  virtual llvm::Error
  ParsePluginSettings(const llvm::json::Value &plugin_settings) = 0;

  /// Create a user-friendly error message upon a JSON-parsing failure using the
  /// \a json::ObjectMapper functionality.
  ///
  /// \param[in] root
  ///   The \a llvm::json::Path::Root used to parse the JSON \a value.
  ///
  /// \param[in] value
  ///   The json value that failed to parse.
  ///
  /// \return
  ///   An \a llvm::Error containing the user-friendly error message.
  llvm::Error CreateJSONError(llvm::json::Path::Root &root,
                              const llvm::json::Value &value);

private:
  /// Resolve non-absolute paths relativeto the settings folder
  void NormalizePath(lldb_private::FileSpec &file_spec);

  llvm::Error ParseProcess(lldb_private::Debugger &debugger,
                           const JSONProcess &process);
  void ParseThread(lldb::ProcessSP &process_sp, const JSONThread &thread);
  llvm::Error ParseModule(lldb::TargetSP &target_sp, const JSONModule &module);
  llvm::Error ParseSettingsImpl(lldb_private::Debugger &debugger,
                                const llvm::json::Value &settings);

  Trace &m_trace;

protected:
  /// Objects created as product of the parsing
  /// \{
  /// The directory that contains the settings file.
  std::string m_settings_dir;

  std::map<lldb::pid_t, std::map<lldb::tid_t, lldb_private::FileSpec>>
      m_thread_to_trace_file_map;
  std::vector<lldb::TargetSP> m_targets;
  /// \}
};

} // namespace lldb_private

namespace llvm {
namespace json {

inline bool fromJSON(const llvm::json::Value &value,
                     lldb_private::TraceSettingsParser::JSONAddress &address,
                     llvm::json::Path path) {
  llvm::Optional<llvm::StringRef> s = value.getAsString();
  if (s.hasValue() && !s->getAsInteger(0, address.value))
    return true;

  path.report("expected numeric string");
  return false;
}

inline bool fromJSON(const llvm::json::Value &value,
                     lldb_private::TraceSettingsParser::JSONModule &module,
                     llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("systemPath", module.system_path) &&
         o.map("file", module.file) &&
         o.map("loadAddress", module.load_address) &&
         o.map("uuid", module.uuid);
}

inline bool fromJSON(const llvm::json::Value &value,
                     lldb_private::TraceSettingsParser::JSONThread &thread,
                     llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("tid", thread.tid) && o.map("traceFile", thread.trace_file);
}

inline bool fromJSON(const llvm::json::Value &value,
                     lldb_private::TraceSettingsParser::JSONProcess &process,
                     llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("pid", process.pid) && o.map("triple", process.triple) &&
         o.map("threads", process.threads) && o.map("modules", process.modules);
}

inline bool fromJSON(
    const llvm::json::Value &value,
    lldb_private::TraceSettingsParser::JSONTracePluginSettings &plugin_settings,
    llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("type", plugin_settings.type);
}

inline bool
fromJSON(const llvm::json::Value &value,
         lldb_private::TraceSettingsParser::JSONTraceSettings &settings,
         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("trace", settings.trace) &&
         o.map("processes", settings.processes);
}

} // namespace json
} // namespace llvm

#endif // LLDB_TARGET_TRACE_SETTINGS_PARSER_H
