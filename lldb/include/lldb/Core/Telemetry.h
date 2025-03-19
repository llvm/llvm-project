//===-- Telemetry.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_TELEMETRY_H
#define LLDB_CORE_TELEMETRY_H

#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/UUID.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Telemetry/Telemetry.h"
#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace lldb_private {
namespace telemetry {

struct LLDBConfig : public ::llvm::telemetry::Config {
  // If true, we will collect full details about a debug command (eg., args and
  // original command). Note: This may contain PII, hence can only be enabled by
  // the vendor while creating the Manager.
  const bool detailed_command_telemetry;

  explicit LLDBConfig(bool enable_telemetry, bool detailed_command_telemetry)
      : ::llvm::telemetry::Config(enable_telemetry),
        detailed_command_telemetry(detailed_command_telemetry) {}
};

// We expect each (direct) subclass of LLDBTelemetryInfo to
// have an LLDBEntryKind in the form 0b11xxxxxxxx
// Specifically:
//  - Length: 8 bits
//  - First two bits (MSB) must be 11 - the common prefix
//  - Last two bits (LSB) are reserved for grand-children of LLDBTelemetryInfo
// If any of the subclass has descendents, those descendents
// must have their LLDBEntryKind in the similar form (ie., share common prefix
// and differ by the last two bits)
struct LLDBEntryKind : public ::llvm::telemetry::EntryKind {
  // clang-format off
  static const llvm::telemetry::KindType BaseInfo        = 0b11000000;
  static const llvm::telemetry::KindType CommandInfo     = 0b11010000;
  static const llvm::telemetry::KindType DebuggerInfo    = 0b11001000;
  static const llvm::telemetry::KindType ExecModuleInfo  = 0b11000100;
  static const llvm::telemetry::KindType ProcessExitInfo = 0b11001100;
  // clang-format on
};

/// Defines a convenient type for timestamp of various events.
using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock,
                                                std::chrono::nanoseconds>;
struct LLDBBaseTelemetryInfo : public llvm::telemetry::TelemetryInfo {
  /// Start time of an event
  SteadyTimePoint start_time;
  /// End time of an event - may be empty if not meaningful.
  std::optional<SteadyTimePoint> end_time;
  // TBD: could add some memory stats here too?

  lldb::user_id_t debugger_id = LLDB_INVALID_UID;
  Debugger *debugger = nullptr;

  // For dyn_cast, isa, etc operations.
  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::BaseInfo;
  }

  static bool classof(const llvm::telemetry::TelemetryInfo *t) {
    // Subclasses of this is also acceptable.
    return (t->getKind() & LLDBEntryKind::BaseInfo) == LLDBEntryKind::BaseInfo;
  }

  void serialize(llvm::telemetry::Serializer &serializer) const override;
};

struct CommandInfo : public LLDBBaseTelemetryInfo {
  /// If the command is/can be associated with a target entry this field
  /// contains that target's UUID. <EMPTY> otherwise.
  UUID target_uuid;
  /// A unique ID for a command so the manager can match the start entry with
  /// its end entry. These values only need to be unique within the same
  /// session. Necessary because we'd send off an entry right before a command's
  /// execution and another right after. This is to avoid losing telemetry if
  /// the command does not execute successfully.
  uint64_t command_id = 0;
  /// The command name(eg., "breakpoint set")
  std::string command_name;
  /// These two fields are not collected by default due to PII risks.
  /// Vendor may allow them by setting the
  /// LLDBConfig::detailed_command_telemetry.
  /// @{
  std::optional<std::string> original_command;
  std::optional<std::string> args;
  /// @}
  /// Return status of a command and any error description in case of error.
  std::optional<lldb::ReturnStatus> ret_status;
  std::optional<std::string> error_data;

  CommandInfo() = default;

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::CommandInfo;
  }

  static bool classof(const llvm::telemetry::TelemetryInfo *T) {
    return (T->getKind() & LLDBEntryKind::CommandInfo) ==
           LLDBEntryKind::CommandInfo;
  }

  void serialize(llvm::telemetry::Serializer &serializer) const override;

  static uint64_t GetNextID();

private:
  // We assign each command (in the same session) a unique id so that their
  // "start" and "end" entries can be matched up.
  // These values don't need to be unique across runs (because they are
  // secondary-key), hence a simple counter is sufficent.
  static std::atomic<uint64_t> g_command_id_seed;
};

struct DebuggerInfo : public LLDBBaseTelemetryInfo {
  std::string lldb_version;

  bool is_exit_entry = false;

  DebuggerInfo() = default;

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::DebuggerInfo;
  }

  static bool classof(const llvm::telemetry::TelemetryInfo *T) {
    // Subclasses of this is also acceptable
    return (T->getKind() & LLDBEntryKind::DebuggerInfo) ==
           LLDBEntryKind::DebuggerInfo;
  }

  void serialize(llvm::telemetry::Serializer &serializer) const override;
};

struct ExecutableModuleInfo : public LLDBBaseTelemetryInfo {
  lldb::ModuleSP exec_mod;
  /// The same as the executable-module's UUID.
  UUID uuid;
  /// PID of the process owned by this target.
  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
  /// The triple of this executable module.
  std::string triple;

  /// If true, this entry was emitted at the beginning of an event (eg., before
  /// the executable is set). Otherwise, it was emitted at the end of an
  /// event (eg., after the module and any dependency were loaded.)
  bool is_start_entry = false;

  ExecutableModuleInfo() = default;

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::ExecModuleInfo;
  }

  static bool classof(const TelemetryInfo *T) {
    // Subclasses of this is also acceptable
    return (T->getKind() & LLDBEntryKind::ExecModuleInfo) ==
           LLDBEntryKind::ExecModuleInfo;
  }
  void serialize(llvm::telemetry::Serializer &serializer) const override;
};

/// Describes an exit status.
struct ExitDescription {
  int exit_code;
  std::string description;
};

struct ProcessExitInfo : public LLDBBaseTelemetryInfo {
  // The executable-module's UUID.
  UUID module_uuid;
  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
  bool is_start_entry = false;
  std::optional<ExitDescription> exit_desc;

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::ProcessExitInfo;
  }

  static bool classof(const TelemetryInfo *T) {
    // Subclasses of this is also acceptable
    return (T->getKind() & LLDBEntryKind::ProcessExitInfo) ==
           LLDBEntryKind::ProcessExitInfo;
  }
  void serialize(llvm::telemetry::Serializer &serializer) const override;
};

/// The base Telemetry manager instance in LLDB.
/// This class declares additional instrumentation points
/// applicable to LLDB.
class TelemetryManager : public llvm::telemetry::Manager {
public:
  llvm::Error preDispatch(llvm::telemetry::TelemetryInfo *entry) override;

  const LLDBConfig *GetConfig() { return m_config.get(); }

  virtual llvm::StringRef GetInstanceName() const = 0;

  static TelemetryManager *GetInstance();

protected:
  TelemetryManager(std::unique_ptr<LLDBConfig> config);

  static void SetInstance(std::unique_ptr<TelemetryManager> manger);

private:
  std::unique_ptr<LLDBConfig> m_config;
  // Each instance of a TelemetryManager is assigned a unique ID.
  const std::string m_id;
  static std::unique_ptr<TelemetryManager> g_instance;
};

/// Helper RAII class for collecting telemetry.
template <typename Info> struct ScopedDispatcher {
  // The debugger pointer is optional because we may not have a debugger yet.
  // In that case, caller must set the debugger later.
  ScopedDispatcher(Debugger *debugger = nullptr) {
    // Start the timer.
    m_start_time = std::chrono::steady_clock::now();
    this->debugger = debugger;
  }
  ScopedDispatcher(llvm::unique_function<void(Info *info)> final_callback,
                   Debugger *debugger = nullptr)
      : m_final_callback(std::move(final_callback)) {
    // Start the timer.
    m_start_time = std::chrono::steady_clock::now();
    this->debugger = debugger;
  }

  void SetDebugger(Debugger *debugger) { this->debugger = debugger; }

  void DispatchOnExit(llvm::unique_function<void(Info *info)> final_callback) {
    // We probably should not be overriding previously set cb.
    assert(!m_final_callback);
    m_final_callback = std::move(final_callback);
  }

  void DispatchNow(llvm::unique_function<void(Info *info)> populate_fields_cb) {
    TelemetryManager *manager = TelemetryManager::GetInstance();
    if (!manager->GetConfig()->EnableTelemetry)
      return;
    Info info;
    // Populate the common fields we know about.
    info.start_time = m_start_time;
    info.end_time = std::chrono::steady_clock::now();
    info.debugger = debugger;
    // The callback will set the rest.
    populate_fields_cb(&info);
    // And then we dispatch.
    if (llvm::Error er = manager->dispatch(&info)) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(er),
                     "Failed to dispatch entry of type: {0}", info.getKind());
    }
  }

  ~ScopedDispatcher() {
    if (m_final_callback)
      DispatchNow(std::move(m_final_callback));
  }

private:
  SteadyTimePoint m_start_time;
  llvm::unique_function<void(Info *info)> m_final_callback;
  Debugger *debugger;
};

} // namespace telemetry
} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
