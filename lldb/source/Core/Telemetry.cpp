//===-- Telemetry.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/llvm-config.h"

#ifdef LLVM_BUILD_TELEMETRY

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Telemetry.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/UUID.h"
#include "lldb/Version/Version.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Telemetry/Telemetry.h"
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace lldb_private {
namespace telemetry {

using namespace llvm::telemetry;

static uint64_t ToNanosec(const SteadyTimePoint Point) {
  return std::chrono::nanoseconds(Point.time_since_epoch()).count();
}

// Generate a unique string. This should be unique across different runs.
// We build such string by combining three parts:
// <16 random bytes>_<timestamp>_<hash of username>
// This reduces the chances of getting the same UUID, even when the same
// user runs the two copies of binary at the same time.
static std::string MakeUUID() {
  uint8_t random_bytes[16];
  std::string randomString = "_";
  if (auto ec = llvm::getRandomBytes(random_bytes, 16)) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Failed to generate random bytes for UUID: {0}", ec.message());
  } else {
    randomString = UUID(random_bytes).GetAsString();
  }

  std::string username = "_";
  UserIDResolver &resolver = lldb_private::HostInfo::GetUserIDResolver();
  std::optional<llvm::StringRef> opt_username =
      resolver.GetUserName(lldb_private::HostInfo::GetUserID());
  if (opt_username)
    username = *opt_username;

  return llvm::formatv(
      "{0}_{1}_{2}", randomString,
      std::chrono::steady_clock::now().time_since_epoch().count(),
      llvm::MD5Hash(username));
}

void LLDBBaseTelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("entry_kind", getKind());
  serializer.write("session_id", SessionId);
  serializer.write("start_time", ToNanosec(start_time));
  if (end_time.has_value())
    serializer.write("end_time", ToNanosec(end_time.value()));
}

void DebuggerInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);

  serializer.write("username", username);
  serializer.write("lldb_git_sha", lldb_git_sha);
  serializer.write("lldb_path", lldb_path);
  serializer.write("cwd", cwd);
  if (exit_desc.has_value()) {
    serializer.write("exit_code", exit_desc->exit_code);
    serializer.write("exit_desc", exit_desc->description);
  }
}

void MiscTelemetryInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);
  serializer.write("target_uuid", target_uuid);
  serializer.beginObject("meta_data");
  for (const auto &kv : meta_data)
    serializer.write(kv.first, kv.second);
  serializer.endObject();
}

TelemetryManager::TelemetryManager(std::unique_ptr<Config> config)
    : m_config(std::move(config)), m_id(MakeUUID) {}

llvm::Error TelemetryManager::preDispatch(TelemetryInfo *entry) {
  // Look up the session_id to assign to this entry or make one
  // if none had been computed for this debugger.
  LLDBBaseTelemetryInfo *lldb_entry =
      llvm::dyn_cast<LLDBBaseTelemetryInfo>(entry);
  std::string session_id = m_id;
  if (Debugger *debugger = lldb_entry->debugger) {
    auto session_id_pos = session_ids.find(debugger->getID());
    if (session_id_pos != session_ids.end())
      session_id = session_id_pos->second;
    else
      session_id_pos->second = session_id =
          llvm::formatv("{0}_{1}", m_id, debugger->getID());
  }
  lldb_entry->SessionId = session_id;

  return llvm::Error::success();
}

const Config *getConfig() { return m_config.get(); }

void TelemetryManager::AtDebuggerStartup(DebuggerInfo *entry) {
  if (auto er = dispatch(entry)) {
    LLDB_LOG(GetLog(LLDBLog::Object), "Failed to dispatch entry at startup");
  }
}

void TelemetryManager::AtDebuggerExit(DebuggerInfo *entry) {
  // There must be a reference to the debugger at this point.
  assert(entry->debugger != nullptr);

  if (auto *selected_target =
          entry->debugger->GetSelectedExecutionContext().GetTargetPtr()) {
    if (!selected_target->IsDummyTarget()) {
      const lldb::ProcessSP proc = selected_target->GetProcessSP();
      if (proc == nullptr) {
        // no process has been launched yet.
        entry->exit_desc = {-1, "no process launched."};
      } else {
        entry->exit_desc = {proc->GetExitStatus(), ""};
        if (const char *description = proc->GetExitDescription())
          entry->exit_desc->description = std::string(description);
      }
    }
  }
  dispatch(entry);
}

std::unique_ptr<TelemetryManager> TelemetryManager::g_instance = nullptr;
TelemetryManager *TelemetryManager::getInstance() { return g_instance.get(); }

void TelemetryManager::setInstance(std::unique_ptr<TelemetryManager> manager) {
  g_instance = std::move(manager);
}

} // namespace telemetry
} // namespace lldb_private

#endif // LLVM_BUILD_TELEMETRY
