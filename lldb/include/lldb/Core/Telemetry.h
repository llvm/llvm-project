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

using llvm::telemetry::Destination;
using llvm::telemetry::KindType;
using llvm::telemetry::Serializer;
using llvm::telemetry::TelemetryInfo;

struct LldbEntryKind : public ::llvm::telemetry::EntryKind {
  static const KindType BaseInfo = 0b11000;
  static const KindType DebuggerInfo = 0b11001;
  static const KindType TargetInfo = 0b11010;
  static const KindType ClientInfo = 0b11100;
  static const KindType CommandInfo = 0b11101;
  static const KindType MiscInfo = 0b11110;
};

/// Defines a convenient type for timestamp of various events.
/// This is used by the EventStats below.
using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock>;

/// Various time (and possibly memory) statistics of an event.
struct EventStats {
  // REQUIRED: Start time of an event
  SteadyTimePoint start;
  // OPTIONAL: End time of an event - may be empty if not meaningful.
  std::optional<SteadyTimePoint> end;
  // TBD: could add some memory stats here too?

  EventStats() = default;
  EventStats(SteadyTimePoint start) : start(start) {}
  EventStats(SteadyTimePoint start, SteadyTimePoint end)
      : start(start), end(end) {}
};

/// Describes the exit signal of an event.
struct ExitDescription {
  int exit_code;
  std::string description;
};

struct LldbBaseTelemetryInfo : public TelemetryInfo {
  EventStats stats;

  // For dyn_cast, isa, etc operations.
  KindType getKind() const override { return LldbEntryKind::BaseInfo; }

  static bool classof(const TelemetryInfo *t) {
    if (t == nullptr)
      return false;
    // Subclasses of this is also acceptable.
    return (t->getKind() & LldbEntryKind::BaseInfo) == LldbEntryKind::BaseInfo;
  }

  void serialize(Serializer &serializer) const override;
};

struct DebuggerTelemetryInfo : public LldbBaseTelemetryInfo {
  std::string username;
  std::string lldb_git_sha;
  std::string lldb_path;
  std::string cwd;

  std::optional<ExitDescription> exit_desc;
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

  void serialize(Serializer &serializer) const override;
};

struct TargetTelemetryInfo : public LldbBaseTelemetryInfo {
  lldb::ModuleSP exec_mod;
  Target *target_ptr;

  // The same as the executable-module's UUID.
  std::string target_uuid;
  std::string file_format;

  std::string binary_path;
  size_t binary_size;

  std::optional<ExitDescription> exit_desc;
  TargetTelemetryInfo() = default;

  TargetTelemetryInfo(const TargetTelemetryInfo &other) {
    exec_mod = other.exec_mod;
    target_uuid = other.target_uuid;
    file_format = other.file_format;
    binary_path = other.binary_path;
    binary_size = other.binary_size;
    exit_desc = other.exit_desc;
  }

  KindType getKind() const override { return LldbEntryKind::TargetInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LldbEntryKind::TargetInfo;
  }

  void serialize(Serializer &serializer) const override;
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

  void serialize(Serializer &serializer) const override;
};

struct CommandTelemetryInfo : public LldbBaseTelemetryInfo {
  Target *target_ptr;
  CommandReturnObject *result;

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

  std::optional<ExitDescription> exit_desc;
  lldb::ReturnStatus ret_status;

  CommandTelemetryInfo() = default;

  CommandTelemetryInfo(const CommandTelemetryInfo &other) {
    target_uuid = other.target_uuid;
    command_uuid = other.command_uuid;
    command_name = other.command_name;
    original_command = other.original_command;
    args = other.args;
    exit_desc = other.exit_desc;
    ret_status = other.ret_status;
  }

  KindType getKind() const override { return LldbEntryKind::CommandInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LldbEntryKind::CommandInfo;
  }

  void serialize(Serializer &serializer) const override;
};

/// The "catch-all" entry to store a set of custom/non-standard
/// data.
struct MiscTelemetryInfo : public LldbBaseTelemetryInfo {
  /// If the event is/can be associated with a target entry,
  /// this field contains that target's UUID.
  /// <EMPTY> otherwise.
  std::string target_uuid;

  /// Set of key-value pairs for any optional (or impl-specific) data
  std::map<std::string, std::string> meta_data;

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

  void serialize(Serializer &serializer) const override;
};

/// The base Telemetry manager instance in LLDB
/// This class declares additional instrumentation points
/// applicable to LLDB.
class LldbTelemeter : public llvm::telemetry::Manager {
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

  /// To be invoked upon LLDB startup.
  virtual void LogStartup(DebuggerTelemetryInfo *entry) = 0;

  /// To be invoked upon LLDB exit.
  virtual void LogExit(DebuggerTelemetryInfo *entry) = 0;

  /// To be invoked upon loading the main executable module.
  /// We log in a fire-n-forget fashion so that if the load
  /// crashes, we don't lose the entry.
  virtual void LogMainExecutableLoadStart(TargetTelemetryInfo *entry) = 0;
  virtual void LogMainExecutableLoadEnd(TargetTelemetryInfo *entry) = 0;

  /// To be invoked upon process exit.
  virtual void LogProcessExit(TargetTelemetryInfo *entry);

  /// Invoked for each command
  /// We log in a fire-n-forget fashion so that if the command execution
  /// crashes, we don't lose the entry.
  virtual void LogCommandStart(CommandTelemetryInfo *entry) = 0;
  virtual void LogCommandEnd(CommandTelemetryInfo *entry) = 0;

  virtual std::string GetNextUUID() = 0;

  /// For client (eg., SB API) to send telemetry entries.
  virtual void
  LogClientTelemetry(const lldb_private::StructuredDataImpl &entry) = 0;

private:
  const std::string SessionId;
  std::vector<std::unique_ptr<Destination>> destinations;
};

/// Logger configs. This should be overriden by vendor's specific config.
/// The default (upstream) config will have telemetry disabled.
llvm::telemetry::Config *GetTelemetryConfig();

} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
