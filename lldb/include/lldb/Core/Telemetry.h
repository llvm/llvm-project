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
#include <type_traits>
#include <utility>
#include <functional>
#include <stack>

namespace lldb_private {
namespace telemetry {

struct LLDBEntryKind : public ::llvm::telemetry::EntryKind {
  static const llvm::telemetry::KindType BaseInfo = 0b11000;
  static const llvm::telemetry::KindType TargetInfo = 0b11010;
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

/// Describes an exit status.
struct ExitDescription {
  int exit_code;
  std::string description;
};


struct TargetInfo : public LLDBBaseTelemetryInfo {
  lldb::ModuleSP exec_mod;
  Target *target_ptr;

  // The same as the executable-module's UUID.
  std::string target_uuid;
  std::string executable_path;
  size_t executable_size;
  std::string arch_name;

  std::optional<ExitDescription> exit_desc;
  TargetInfo() = default;

  llvm::telemetry::KindType getKind() const override { return LLDBEntryKind::TargetInfo; }

  static bool classof(const TelemetryInfo *T) {
    if (T == nullptr)
      return false;
    return T->getKind() == LLDBEntryKind::TargetInfo;
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

  /// The following methods are for reporting the load of an executable.
  /// One is invoked at the beginning of the process and the other at
  /// the end.
  /// This is done in two passes to avoid losing date in case of any error/crash
  /// during the action.
  ///
  /// Invoked at the begining of the load of the main-executable.
  virtual void AtMainExecutableLoadStart(TargetInfo * entry);
  /// Invoked at the end of the load.
  virtual void AtMainExecutableLoadEnd(TargetInfo *entry);

  virtual llvm::StringRef GetInstanceName() const = 0;

  static TelemetryManager *GetInstance();

protected:
  TelemetryManager(std::unique_ptr<llvm::telemetry::Config> config);

  static void SetInstance(std::unique_ptr<TelemetryManager> manger);

private:
  std::unique_ptr<llvm::telemetry::Config> m_config;

  static std::unique_ptr<TelemetryManager> g_instance;
};

class Helper {
 public:
  Helper () : m_start(std::chrono::steady_clock::now()) {}
  ~Helper() {
    while(! m_exit_funcs.empty()) {
      (m_exit_funcs.top())();
      m_exit_funcs.pop();
    }
  }

  bool TelemetryEnabled() {
    TelemetryManager* instance = TelemetryManager::GetInstance();
  return instance != nullptr && instance->GetConfig()->EnableTelemetry;
  }


  SteadyTimePoint GetStartTime() {return m_start;}
  SteadyTimePoint GetCurrentTime()  { return std::chrono::steady_clock::now(); }

 template <typename Fp>
 void RunAtScopeExit(Fp&& F){
   m_exit_funcs.push(std::forward<Fp>(F));
 }

 private:
  const SteadyTimePoint m_start;
  std::stack<std::function<void()>> m_exit_funcs;

};

} // namespace telemetry
} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
