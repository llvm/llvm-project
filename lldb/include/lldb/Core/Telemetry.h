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
#include <string>
#include <unordered_map>

namespace lldb_private {
namespace telemetry {

struct LLDBEntryKind : public ::llvm::telemetry::EntryKind {
  static const llvm::telemetry::KindType BaseInfo = 0b11000;
  static const llvm::telemetry::KindType DebuggerInfo = 0b11001;
  // There are other entries in between (added in separate PRs)
  static const llvm::telemetry::KindType MiscInfo = 0b11110;
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

  std::string lldb_path;
  std::string cwd;
  std::string username;

  DebuggerInfo() = default;

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::DebuggerInfo;
  }

  static bool classof(const llvm::telemetry::TelemetryInfo *T) {
    return T->getKind() == LLDBEntryKind::DebuggerInfo;
  }

  void serialize(llvm::telemetry::Serializer &serializer) const override;
};

/// The "catch-all" entry to store a set of non-standard data, such as
/// error-messages, etc.
struct MiscTelemetryInfo : public LLDBBaseTelemetryInfo {
  /// If the event is/can be associated with a target entry,
  /// this field contains that target's UUID.
  /// <EMPTY> otherwise.
  std::string target_uuid;

  /// Set of key-value pairs for any optional (or impl-specific) data
  llvm::StringMap<std::string> meta_data;

  MiscTelemetryInfo() = default;

  MiscTelemetryInfo(const MiscTelemetryInfo &other) {
    target_uuid = other.target_uuid;
    meta_data = other.meta_data;
  }

  llvm::telemetry::KindType getKind() const override {
    return LLDBEntryKind::MiscInfo;
  }

  static bool classof(const llvm::telemetry::TelemetryInfo *T) {
    return T->getKind() == LLDBEntryKind::MiscInfo;
  }

  void serialize(llvm::telemetry::Serializer &serializer) const override;
};

/// The base Telemetry manager instance in LLDB.
/// This class declares additional instrumentation points
/// applicable to LLDB.
class TelemetryManager : public llvm::telemetry::Manager {
public:
  llvm::Error preDispatch(llvm::telemetry::TelemetryInfo *entry) override;

  const llvm::telemetry::Config *getConfig();

  virtual void AtDebuggerStartup(DebuggerInfo *entry);
  virtual void AtDebuggerExit(DebuggerInfo *entry);

  virtual llvm::StringRef GetInstanceName() const = 0;
  static TelemetryManager *getInstance();

protected:
  TelemetryManager(std::unique_ptr<llvm::telemetry::Config> config);

  static void setInstance(std::unique_ptr<TelemetryManager> manger);

private:
  std::unique_ptr<llvm::telemetry::Config> m_config;
  // Each instance of a TelemetryManager is assigned a unique ID.
  const std::string m_id;

  // Map of debugger's ID to a unique session_id string.
  // All TelemetryInfo entries emitted for the same debugger instance
  // will get the same session_id.
  llvm::DenseMap<lldb::user_id_t, std::string> session_ids;
  static std::unique_ptr<TelemetryManager> g_instance;
};

} // namespace telemetry
} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
