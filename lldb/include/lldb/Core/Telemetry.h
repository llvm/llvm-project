#ifndef LLDB_CORE_TELEMETRY_H
#define LLDB_CORE_TELEMETRY_H

#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>

#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock>;

struct TelemetryEventStats {
  // REQUIRED: Start time of event
  SteadyTimePoint m_start;
  // OPTIONAL: End time of event - may be empty if not meaningful.
  std::optional<SteadyTimePoint> m_end;

  // TBD: could add some memory stats here too?

  TelemetryEventStats() = default;
  TelemetryEventStats(SteadyTimePoint start) : m_start(start) {}
  TelemetryEventStats(SteadyTimePoint start, SteadyTimePoint end)
      : m_start(start), m_end(end) {}

  std::optional<std::chrono::nanoseconds> Duration() const {
    if (m_end.has_value())
      return *m_end - m_start;
    else
      return std::nullopt;
  }
};

struct LoggerConfig {
  // If true, loggings will be enabled.
  bool enable_logging;

  // Additional destinations to send the logged entries.
  // Could be stdout, stderr, or some local paths.
  // Note: these are destinations are __in addition to__ whatever the default
  // destination(s) are, as implemented by vendors.
  std::vector<std::string> additional_destinations;
};

// The base class contains the basic set of data.
// Downstream implementations can add more fields as needed.
struct BaseTelemetryEntry {
  // A "session" corresponds to every time lldb starts.
  // All entries emitted for the same session will have
  // the same session_uuid
  std::string session_uuid;

  TelemetryEventStats stats;

  // Counting number of entries.
  // (For each set of entries with the same session_uuid, this value should
  // be unique for each entry)
  size_t counter;

  virtual ~BaseTelemetryEntry() = default;
  virtual std::string ToString() const;
};

struct ExitDescription {
  int exit_code;
  std::string description;
};

struct DebuggerInfoEntry : public BaseTelemetryEntry {
  std::string username;
  std::string lldb_git_sha;
  std::string lldb_path;
  std::string cwd;

  // The debugger exit info. Not populated if this entry was emitted for startup
  // event.
  std::optional<ExitDescription> lldb_exit;

  std::string ToString() const override;
};

struct TargetInfoEntry : public BaseTelemetryEntry {
  // All entries emitted for the same SBTarget will have the same
  // target_uuid.
  std::string target_uuid;
  std::string file_format;

  std::string binary_path;
  size_t binary_size;

  // The process(target) exit info. Not populated of this entry was emitted for
  // startup event.
  std::optional<ExitDescription> process_exit;

  std::string ToString() const override;
};

// Entry from client (eg., SB-API)
struct ClientTelemetryEntry : public BaseTelemetryEntry {
  std::string request_name;
  std::string error_msg;
  std::string ToString() const override;
};

struct CommandInfoEntry : public BaseTelemetryEntry {
  // If the command is/can be associated with a target entry,
  // this field contains that target's UUID.
  // <EMPTY> otherwise.
  std::string target_uuid;
  std::string command_uuid;

  // Eg., "breakpoint set"
  std::string command_name;

  // !!NOTE!!: The following fields may be omitted due to PII risk.
  // (Configurable via the LoggerConfig struct)
  std::string original_command;
  std::string args;

  ExitDescription exit_description;

  std::string ToString() const override;
};

// The "catch-all" entry to store a set of custom/non-standard
// data.
struct MiscInfoEntry : public BaseTelemetryEntry {
  // If the event is/can be associated with a target entry,
  // this field contains that target's UUID.
  // <EMPTY> otherwise.
  std::string target_uuid;

  // Set of key-value pairs for any optional (or impl-specific) data
  std::unordered_map<std::string, std::string> meta_data;

  std::string ToString() const override;
};

// Where/how to send the logged entries.
class TelemetryDestination {
public:
  virtual ~TelemetryDestination() = default;
  virtual Status EmitEntry(const BaseTelemetryEntry *entry) = 0;
  virtual std::string name() const = 0;
};

// The logger itself!
class TelemetryLogger {
public:
  static std::shared_ptr<TelemetryLogger> CreateInstance(Debugger *);

  virtual ~TelemetryLogger() = default;

  // Invoked upon lldb startup
  virtual void LogStartup(llvm::StringRef lldb_path,
                          TelemetryEventStats stats) = 0;

  // Invoked upon lldb exit.
  virtual void LogExit(llvm::StringRef lldb_path,
                       TelemetryEventStats stats) = 0;

  // Invoked upon process exit
  virtual void LogProcessExit(int status, llvm::StringRef exit_string,
                              TelemetryEventStats stats,
                              Target *target_ptr) = 0;

  // Invoked upon loading the main executable module
  // We log in a fire-n-forget fashion so that if the load
  // crashes, we don't lose the entry.
  virtual void LogMainExecutableLoadStart(lldb::ModuleSP exec_mod,
                                          TelemetryEventStats stats) = 0;
  virtual void LogMainExecutableLoadEnd(lldb::ModuleSP exec_mod,
                                        TelemetryEventStats stats) = 0;

  // Invoked for each command
  // We log in a fire-n-forget fashion so that if the command execution
  // crashes, we don't lose the entry.
  virtual void LogCommandStart(llvm::StringRef uuid,
                               llvm::StringRef original_command,
                               TelemetryEventStats stats,
                               Target *target_ptr) = 0;
  virtual void LogCommandEnd(llvm::StringRef uuid, llvm::StringRef command_name,
                             llvm::StringRef command_args,
                             TelemetryEventStats stats, Target *target_ptr,
                             CommandReturnObject *result) = 0;

  virtual std::string GetNextUUID() = 0;

  // For client (eg., SB API) to send telemetry entries.
  virtual void
  LogClientTelemetry(lldb_private::StructuredData::Object *entry) = 0;

  virtual void AddDestination(TelemetryDestination *destination) = 0;
};

/*

Logger configs: LLDB users can also supply their own configs via:
$XDG_CONFIG_HOME/.lldb_telemetry_config


We can propose simple syntax: <field_name><colon><value>
Eg.,
enable_logging:true
destination:stdout
destination:stderr
destination:/path/to/some/file

The allowed field_name values are:
 * enable_logging
       If the fields are specified more than once, the last line will take precedence
       If enable_logging is set to false, no logging will occur.
 * destination.
      This is allowed to be specified multiple times - it will add to the default
      (ie, specified by vendor) list of destinations.
      The value can be either of
         + one of the two magic values "stdout" or "stderr".
         + a path to a local file

!!NOTE!!: We decided to use a separate file instead of the existing settings
         file because that file is parsed too late in the process and by the
         there might have been lots of telemetry-entries that need to be
         sent already.
         This approach avoid losing log entries if LLDB crashes during init.

*/
LoggerConfig *GetLoggerConfig();

} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
