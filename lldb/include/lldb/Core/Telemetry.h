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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Telemetry/Telemetry.h"
#include <chrono>
#include <ctime>
#include <memory>
#include <optional>
#include <stack>
#include <string>

namespace lldb_private {
namespace telemetry {

struct LLDBEntryKind : public ::llvm::telemetry::EntryKind {
  static const llvm::telemetry::KindType BaseInfo = 0b11000;
  static const llvm::telemetry::KindType DebuggerInfo = 0b11001;
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

  uint64_t debugger_id = LLDB_INVALID_UID;
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

/// Describes the exit status of a debugger.
struct ExitDescription {
  int exit_code;
  std::string description;
};

struct DebuggerInfo : public LLDBBaseTelemetryInfo {
  std::string lldb_version;
  std::optional<ExitDescription> exit_desc;

  DebuggerInfo() = default;

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::DebuggerInfo;
  }

  static bool classof(const llvm::telemetry::TelemetryInfo *T) {
    return T->getKind() == LLDBEntryKind::DebuggerInfo;
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
template <typename I> struct ScopedDispatcher {
  // The debugger pointer is optional because we may not have a debugger yet.
  // In that case, caller must set the debugger later.
  ScopedDispatcher(std::function<void(I *info)> callback,
                   Debugger *debugger = nullptr)
      : m_callback(callback) {
    // Start the timer.
    m_start_time = std::chrono::steady_clock::now();
    m_info.debugger = debugger;
  }

  void SetDebugger(Debugger *debugger) { m_info.debugger = debugger; }

  ~ScopedDispatcher() {
    TelemetryManager *manager = TelemetryManager::GetInstance();
    // If Telemetry is disabled (either at buildtime or runtime),
    // then don't do anything.
    if (!manager || !manager->GetConfig()->EnableTelemetry)
      return;
    m_info.start_time = m_start_time;
    // Populate common fields that we can only set now.
    m_info.end_time = std::chrono::steady_clock::now();
    // The callback will set the remaining fields.
    m_callback(&m_info);
    // And then we dispatch.
    if (llvm::Error er = manager->dispatch(&m_info)) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(er),
                     "Failed to dispatch entry of type: {0}", m_info.getKind());
    }
  }

private:
  SteadyTimePoint m_start_time;
  std::function<void(I *info)> m_callback;
  I m_info;
};

} // namespace telemetry
} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
