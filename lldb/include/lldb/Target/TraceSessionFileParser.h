//===-- TraceSessionFileParser.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACESESSIONPARSER_H
#define LLDB_TARGET_TRACESESSIONPARSER_H

#include "llvm/Support/JSON.h"

#include "lldb/lldb-private.h"

namespace lldb_private {

/// \class TraceSessionFileParser TraceSessionFileParser.h
///
/// Base class for parsing the common information of JSON trace session files.
/// Contains the basic C++ structs that represent the JSON data, which include
/// \a JSONTraceSession as the root object.
///
/// See \a Trace::FindPlugin for more information regarding these JSON files.
class TraceSessionFileParser {
public:
  /// C++ structs representing the JSON trace session.
  /// \{
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

  /// The trace plug-in implementation should provide its own TPluginSettings,
  /// which corresponds to the "trace" section of the schema.
  template <class TPluginSettings> struct JSONTraceSession {
    std::vector<JSONProcess> processes;
    TPluginSettings trace;
  };
  /// \}

  TraceSessionFileParser(llvm::StringRef session_file_dir,
                         llvm::StringRef schema)
      : m_session_file_dir(session_file_dir), m_schema(schema) {}

  /// Build the full schema for a Trace plug-in.
  ///
  /// \param[in] plugin_schema
  ///   The subschema that corresponds to the "trace" section of the schema.
  ///
  /// \return
  ///   The full schema containing the common attributes and the plug-in
  ///   specific attributes.
  static std::string BuildSchema(llvm::StringRef plugin_schema);

protected:
  /// Resolve non-absolute paths relative to the session file folder. It
  /// modifies the given file_spec.
  void NormalizePath(lldb_private::FileSpec &file_spec);

  llvm::Error ParseModule(lldb::TargetSP &target_sp, const JSONModule &module);

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

  std::string m_session_file_dir;
  llvm::StringRef m_schema;
};
} // namespace lldb_private

namespace llvm {
namespace json {

bool fromJSON(const Value &value,
              lldb_private::TraceSessionFileParser::JSONAddress &address,
              Path path);

bool fromJSON(const Value &value,
              lldb_private::TraceSessionFileParser::JSONModule &module,
              Path path);

bool fromJSON(const Value &value,
              lldb_private::TraceSessionFileParser::JSONThread &thread,
              Path path);

bool fromJSON(const Value &value,
              lldb_private::TraceSessionFileParser::JSONProcess &process,
              Path path);

bool fromJSON(const Value &value,
              lldb_private::TraceSessionFileParser::JSONTracePluginSettings
                  &plugin_settings,
              Path path);

template <class TPluginSettings>
bool fromJSON(
    const Value &value,
    lldb_private::TraceSessionFileParser::JSONTraceSession<TPluginSettings>
        &session,
    Path path) {
  ObjectMapper o(value, path);
  return o && o.map("trace", session.trace) &&
         o.map("processes", session.processes);
}

} // namespace json
} // namespace llvm

#endif // LLDB_TARGET_TRACESESSIONPARSER_H
