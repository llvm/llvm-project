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
#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Telemetry/Telemetry.h"
#include <atomic>
#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace lldb_private {
namespace telemetry {

struct LLDBConfig : public ::llvm::telemetry::Config {
  const bool m_collect_original_command;

  explicit LLDBConfig(bool enable_telemetry, bool collect_original_command)
      : ::llvm::telemetry::Config(enable_telemetry), m_collect_original_command(collect_original_command) {}
};

struct LLDBEntryKind : public ::llvm::telemetry::EntryKind {
  static const llvm::telemetry::KindType BaseInfo = 0b11000000;
  static const llvm::telemetry::KindType CommandInfo = 0b11010000;
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
  Debugger *debugger;

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

  // If the command is/can be associated with a target entry this field contains
  // that target's UUID. <EMPTY> otherwise.
  std::string target_uuid;
  // A unique ID for a command so the manager can match the start entry with
  // its end entry. These values only need to be unique within the same session.
  // Necessary because we'd send off an entry right before a command's execution
  // and another right after. This is to avoid losing telemetry if the command
  // does not execute successfully.
  int command_id;

  // Eg., "breakpoint set"
  std::string command_name;

  // !!NOTE!! These two fields are not collected (upstream) due to PII risks.
  // (Downstream impl may add them if needed).
  // std::string original_command;
  // std::string args;

  lldb::ReturnStatus ret_status;
  std::string error_data;


  CommandInfo() = default;

  llvm::telemetry::KindType getKind() const override { return LLDBEntryKind::CommandInfo; }

  static bool classof(const llvm::telemetry::TelemetryInfo *T) {
    return (T->getKind() & LLDBEntryKind::CommandInfo) == LLDBEntryKind::CommandInfo;
  }

  void serialize(Serializer &serializer) const override;
};

/// The base Telemetry manager instance in LLDB.
/// This class declares additional instrumentation points
/// applicable to LLDB.
class TelemetryManager : public llvm::telemetry::Manager {
public:
  llvm::Error preDispatch(llvm::telemetry::TelemetryInfo *entry) override;

  int MakeNextCommandId();

  LLDBConfig* GetConfig() { return m_config.get(); }

  virtual llvm::StringRef GetInstanceName() const = 0;
  static TelemetryManager *getInstance();

protected:
  TelemetryManager(std::unique_ptr<LLDBConfig> config);

  static void setInstance(std::unique_ptr<TelemetryManager> manger);

private:
  std::unique_ptr<LLDBConfig> m_config;
  const std::string m_id;
  // We assign each command (in the same session) a unique id so that their
  // "start" and "end" entries can be matched up.
  // These values don't need to be unique across runs (because they are
  // secondary-key), hence a simple counter is sufficent.
  std::atomic<int> command_id_seed = 0;
  static std::unique_ptr<TelemetryManager> g_instance;
};

/// Helper RAII class for collecting telemetry.
template <typename Info> struct ScopedDispatcher {
  // The debugger pointer is optional because we may not have a debugger yet.
  // In that case, caller must set the debugger later.
  ScopedDispatcher(Debugger *debugger = nullptr) {
    // Start the timer.
    m_start_time = std::chrono::steady_clock::now();
    debugger = debugger;
  }
  ScopedDispatcher(llvm::unique_function<void(Info *info)> final_callback,
                   Debugger *debugger = nullptr)
      : m_final_callback(std::move(final_callback)) {
    // Start the timer.
    m_start_time = std::chrono::steady_clock::now();
    debugger = debugger;
  }

  void SetDebugger(Debugger *debugger) { debugger = debugger; }

  void SetFinalCallback(llvm::unique_function<void(Info *info)> final_callback) {
    m_final_callback(std::move(final_callback));
  }

  void DispatchIfEnable(llvm::unique_function<void(Info *info)> populate_fields_cb) {
    TelemetryManager *manager = TelemetryManager::GetInstanceIfEnabled();
    if (!manager)
      return;
    Info info;
    // Populate the common fields we know aboutl
    info.start_time = m_start_time;
    info.end_time = std::chrono::steady_clock::now();
    info.debugger = debugger;
    // The callback will set the rest.
    populate_fields_cb(&info);
    // And then we dispatch.
    if (llvm::Error er = manager->dispatch(&info)) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(er),
                     "Failed to dispatch entry of type: {0}", m_info.getKind());
    }

  }

  ~ScopedDispatcher() {
    // TODO: check if there's a cb to call?
    DispatchIfEnable(std::move(m_final_callback));
  }

private:
  SteadyTimePoint m_start_time;
  llvm::unique_function<void(Info *info)> m_final_callback;
  Debugger * debugger;
};

} // namespace telemetry
} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
