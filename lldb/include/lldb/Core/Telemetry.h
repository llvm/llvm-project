//===-- Telemetry.h ----------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_TELEMETRY_H
#define LLDB_CORE_TELEMETRY_H

#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Telemetry/Telemetry.h"

namespace lldb_private {

using llvm::telemetry::KindType;

struct LldbEntryKind : public ::llvm::telemetry::EntryKind {
  static const KindType BaseInfo = 0b11000;
  static const KindType DebuggerInfo = 0b11001;
  static const KindType TargetInfo = 0b11010;
  static const KindType ClientInfo = 0b11100;
  static const KindType CommandInfo = 0b11101;
  static const KindType MiscInfo = 0b11110;
};

struct LldbBaseTelemetryInfo : public ::llvm::telemetry::TelemetryInfo {
  // For dyn_cast, isa, etc operations.
  KindType getKind() const override { return LldbEntryKind::BaseInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    // Subclasses of this is also acceptable.
    return (T->getKind() & LldbEntryKind::BaseInfo) == LldbEntryKind::BaseInfo;
  }

  // Returns a human-readable string description of the struct.
  // This is for debugging purposes only.
  // It is NOT meant as a data-serialisation method.
  virtual std::string ToString() const;
};

struct DebuggerTelemetryInfo : public LldbBaseTelemetryInfo {
  std::string username;
  std::string lldb_git_sha;
  std::string lldb_path;
  std::string cwd;

  DebuggerTelemetryInfo() = default;

  // Provide a copy ctor because we may need to make a copy before
  // sanitizing the data.
  // (The sanitization might differ between different Destination classes).
  DebuggerTelemetryInfo(const DebuggerTelemetryInfo &other) {
    username = other.username;
    lldb_git_sha = other.lldb_git_sha;
    lldb_path = other.lldb_path;
    cwd = other.cwd;
  };

  KindType getKind() const override { return LldbEntryKind::DebuggerInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LldbEntryKind::DebuggerInfo;
  }

  llvm::json::Object serializeToJson() const override;

  std::string ToString() const override;
};

struct TargetTelemetryInfo : public LldbBaseTelemetryInfo {
  // The same as the executable-module's UUID.
  std::string target_uuid;
  std::string file_format;

  std::string binary_path;
  size_t binary_size;

  TargetTelemetryInfo() = default;

  TargetTelemetryInfo(const TargetTelemetryInfo &other) {
    target_uuid = other.target_uuid;
    file_format = other.file_format;
    binary_path = other.binary_path;
    binary_size = other.binary_size;
  }

  KindType getKind() const override { return LldbEntryKind::TargetInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LldbEntryKind::TargetInfo;
  }

  llvm::json::Object serializeToJson() const override;

  std::string ToString() const override;
};

// Entry from client (eg., SB-API)
struct ClientTelemetryInfo : public LldbBaseTelemetryInfo {
  std::string request_name;
  std::string error_msg;

  ClientTelemetryInfo() = default;

  ClientTelemetryInfo(const ClientTelemetryInfo &other) {
    request_name = other.request_name;
    error_msg = other.error_msg;
  }

  KindType getKind() const override { return LldbEntryKind::ClientInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LldbEntryKind::ClientInfo;
  }

  llvm::json::Object serializeToJson() const override;

  std::string ToString() const override;
};

struct CommandExitDescription : public ::llvm::telemetry::ExitDescription {
  lldb::ReturnStatus ret_status;
  CommandExitDescription(int ret_code, std::string ret_desc,
                         lldb::ReturnStatus status) {
    ExitCode = ret_code;
    Description = std::move(ret_desc);
    ret_status = status;
  }
};

struct CommandTelemetryInfo : public LldbBaseTelemetryInfo {
  // If the command is/can be associated with a target entry,
  // this field contains that target's UUID.
  // <EMPTY> otherwise.
  std::string target_uuid;
  std::string command_uuid;

  // Eg., "breakpoint set"
  std::string command_name;

  // !!NOTE!!: The following fields may be omitted due to PII risk.
  // (Configurable via the telemery::Config struct)
  std::string original_command;
  std::string args;

  lldb::ReturnStatus ret_status;

  CommandTelemetryInfo() = default;

  CommandTelemetryInfo(const CommandTelemetryInfo &other) {
    target_uuid = other.target_uuid;
    command_uuid = other.command_uuid;
    command_name = other.command_name;
    original_command = other.original_command;
    args = other.args;
  }

  KindType getKind() const override { return LldbEntryKind::CommandInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LldbEntryKind::CommandInfo;
  }

  llvm::json::Object serializeToJson() const override;

  std::string ToString() const override;
};

/// The "catch-all" entry to store a set of custom/non-standard
/// data.
struct MiscTelemetryInfo : public LldbBaseTelemetryInfo {
  /// If the event is/can be associated with a target entry,
  /// this field contains that target's UUID.
  /// <EMPTY> otherwise.
  std::string target_uuid;

  /// Set of key-value pairs for any optional (or impl-specific) data
  std::unordered_map<std::string, std::string> meta_data;

  MiscTelemetryInfo() = default;

  MiscTelemetryInfo(const MiscTelemetryInfo &other) {
    target_uuid = other.target_uuid;
    meta_data = other.meta_data;
  }

  KindType getKind() const override { return LldbEntryKind::MiscInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LldbEntryKind::MiscInfo;
  }

  llvm::json::Object serializeToJson() const override;

  std::string ToString() const override;
};

/// The base Telemeter instance in LLDB.
/// This class declares additional instrumentation points
/// applicable to LLDB.
class LldbTelemeter : public llvm::telemetry::Telemeter {
public:
  /// Creates an instance of LldbTelemeter.
  /// This uses the plugin registry to find an instance:
  ///  - If a vendor supplies a implementation, it will use it.
  ///  - If not, it will either return a no-op instance or a basic
  ///    implementation for testing.
  ///
  /// See also lldb_private::TelemetryVendor.
  static std::unique_ptr<LldbTelemeter> CreateInstance(Debugger *debugger);

  virtual ~LldbTelemeter() = default;

  /// Invoked upon process exit
  virtual void LogProcessExit(int status, llvm::StringRef exit_string,
                              llvm::telemetry::EventStats stats,
                              Target *target_ptr) = 0;

  /// Invoked upon loading the main executable module
  /// We log in a fire-n-forget fashion so that if the load
  /// crashes, we don't lose the entry.
  virtual void
  LogMainExecutableLoadStart(lldb::ModuleSP exec_mod,
                             llvm::telemetry::EventStats stats) = 0;
  virtual void LogMainExecutableLoadEnd(lldb::ModuleSP exec_mod,
                                        llvm::telemetry::EventStats stats) = 0;

  /// Invoked for each command
  /// We log in a fire-n-forget fashion so that if the command execution
  /// crashes, we don't lose the entry.
  virtual void LogCommandStart(llvm::StringRef uuid,
                               llvm::StringRef original_command,
                               llvm::telemetry::EventStats stats,
                               Target *target_ptr) = 0;
  virtual void LogCommandEnd(llvm::StringRef uuid, llvm::StringRef command_name,
                             llvm::StringRef command_args,
                             llvm::telemetry::EventStats stats,
                             Target *target_ptr,
                             CommandReturnObject *result) = 0;

  virtual std::string GetNextUUID() = 0;

  /// For client (eg., SB API) to send telemetry entries.
  virtual void
  LogClientTelemetry(const lldb_private::StructuredDataImpl &entry) = 0;
};

/// Logger configs: LLDB users can also supply their own configs via:
/// $HOME/.lldb_telemetry_config
///
/// We can propose simple syntax: <field_name><colon><value>
/// Eg.,
/// enable_telemetry:true
/// destination:stdout
/// destination:stderr
/// destination:/path/to/some/file
///
/// The allowed field_name values are:
///  * enable_telemetry
///       If the fields are specified more than once, the last line will take
///       precedence If enable_logging is set to false, no logging will occur.
///  * destination.
///       This is allowed to be specified multiple times - it will add to the
///       default (ie, specified by vendor) list of destinations.
///       The value can be either:
///          + one of the two magic values "stdout" or "stderr".
///          + a path to a local file
/// !!NOTE!!: We decided to use a separate file instead of the existing settings
///         file because that file is parsed too late in the process and by the
///         there might have been lots of telemetry-entries that need to be
///         sent already.
///         This approach avoid losing log entries if LLDB crashes during init.
llvm::telemetry::Config *GetTelemetryConfig();

} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
