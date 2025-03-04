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
#include "lldb/lldb-forward.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Telemetry/Telemetry.h"
#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <string>

#include <functional>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace lldb_private {
namespace telemetry {

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
  static const llvm::telemetry::KindType BaseInfo = 0b11000000;
  static const llvm::telemetry::KindType DebuggerInfo = 0b11000100;
  static const llvm::telemetry::KindType TargetInfo = 0b11001000;
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

/// Describes an exit status.
struct ExitDescription {
  int exit_code;
  std::string description;
};

struct TargetInfo : public LLDBBaseTelemetryInfo {
  lldb::ModuleSP exec_mod;

  // The same as the executable-module's UUID.
  std::string target_uuid;
  std::string arch_name;

  // If true, this entry was emitted at the beginning of an event (eg., before
  // the executable laod). Otherwise, it was emitted at the end of an event
  // (eg., after the module and any dependency were loaded.)
  bool is_start_entry;

  // Describes the exit of the executable module.
  std::optional<ExitDescription> exit_desc;
  TargetInfo() = default;

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::TargetInfo;
  }

  static bool classof(const TelemetryInfo *T) {
    // Subclasses of this is also acceptable
    return (T->getKind() & LLDBEntryKind::TargetInfo) ==
           LLDBEntryKind::TargetInfo;
  }
  void serialize(llvm::telemetry::Serializer &serializer) const override;
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

/// The base Telemetry manager instance in LLDB.
/// This class declares additional instrumentation points
/// applicable to LLDB.
class TelemetryManager : public llvm::telemetry::Manager {
public:
  llvm::Error preDispatch(llvm::telemetry::TelemetryInfo *entry) override;

  const llvm::telemetry::Config *GetConfig();

  virtual llvm::StringRef GetInstanceName() const = 0;

  static TelemetryManager *GetInstance();

protected:
  TelemetryManager(std::unique_ptr<llvm::telemetry::Config> config);

  static void SetInstance(std::unique_ptr<TelemetryManager> manger);

private:
  std::unique_ptr<llvm::telemetry::Config> m_config;

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
    debugger = debugger;
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
    // Populate the common fields we know aboutl
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
