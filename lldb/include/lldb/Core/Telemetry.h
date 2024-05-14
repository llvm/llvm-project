#ifndef LLDB_CORE_TELEMETRY_H
#define LLDB_CORE_TELEMETRY_H

#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Telemetry/Telemetry.h"

using namespace llvm::telemetry;

namespace lldb_private {

struct DebuggerInfoEntry : public ::llvm::telemetry::BaseTelemetryEntry {
  std::string username;
  std::string lldb_git_sha;
  std::string lldb_path;
  std::string cwd;

  std::string ToString() const override;
};

struct TargetInfoEntry : public ::llvm::telemetry::BaseTelemetryEntry {
  // All entries emitted for the same SBTarget will have the same
  // target_uuid.
  std::string target_uuid;
  std::string file_format;

  std::string binary_path;
  size_t binary_size;

  std::string ToString() const override;
};

// Entry from client (eg., SB-API)
struct ClientTelemetryEntry : public ::llvm::telemetry::BaseTelemetryEntry {
  std::string request_name;
  std::string error_msg;
  std::string ToString() const override;
};

struct CommandInfoEntry : public ::llvm::telemetry::BaseTelemetryEntry {
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

  std::string ToString() const override;
};

// The "catch-all" entry to store a set of custom/non-standard
// data.
struct MiscInfoEntry : public ::llvm::telemetry::BaseTelemetryEntry {
  // If the event is/can be associated with a target entry,
  // this field contains that target's UUID.
  // <EMPTY> otherwise.
  std::string target_uuid;

  // Set of key-value pairs for any optional (or impl-specific) data
  std::unordered_map<std::string, std::string> meta_data;

  std::string ToString() const override;
};

class LldbTelemetryLogger : public llvm::telemetry::BaseTelemetryLogger {
public:
  static std::shared_ptr<LldbTelemetryLogger> CreateInstance(Debugger *);

  virtual ~LldbTelemetryLogger() = default;

  // void LogStartup(llvm::StringRef lldb_path,
  //                 BaseTelemetryEntry *entry) override;
  // void LogExit(llvm::StringRef lldb_path, BaseTelemetryEntry *entry)
  // override;

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
       If the fields are specified more than once, the last line will take
precedence If enable_logging is set to false, no logging will occur.
 * destination.
      This is allowed to be specified multiple times - it will add to the
default (ie, specified by vendor) list of destinations. The value can be either
of
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
