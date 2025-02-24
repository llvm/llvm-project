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

static std::string MakeUUID(Debugger *debugger) {
  uint8_t random_bytes[16];
  if (auto ec = llvm::getRandomBytes(random_bytes, 16)) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Failed to generate random bytes for UUID: {0}", ec.message());
    // Fallback to using timestamp + debugger ID.
    return llvm::formatv(
        "{0}_{1}", std::chrono::steady_clock::now().time_since_epoch().count(),
        debugger->GetID());
  }
  return UUID(random_bytes).GetAsString();
}

void LLDBBaseTelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("entry_kind", getKind());
  serializer.write("session_id", SessionId);
  serializer.write("start_time", ToNanosec(start_time));
  if (end_time.has_value())
    serializer.write("end_time", ToNanosec(end_time.value()));
}

void TargetInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);

  serializer.write("target_uuid", target_uuid);
  serializer.write("executable_path", executable_path);
  serializer.write("executable_size", executable_size);
  serializer.write("arch_name", arch_name);
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
    : m_config(std::move(config)) {}

llvm::Error TelemetryManager::preDispatch(TelemetryInfo *entry) {
  // Do nothing for now.
  // In up-coming patch, this would be where the manager
  // attach the session_uuid to the entry.
  return llvm::Error::success();
}

const Config * TelemetryManager::GetConfig() { return m_config.get(); }


void TelemetryManager::AtMainExecutableLoadStart(TargetInfo *entry) {
  if (entry->exec_mod != nullptr) {
    entry->target_uuid = entry->exec_mod->GetUUID().GetAsString();
    entry->executalbe_path = entry->exec_mod->GetFileSpec().GetPathAsConstString().GetCString();
    if (auto err = llvm::sys::fs::file_size(
            entry->exec_mod->GetFileSpec().GetPath(), entry->binary_size)) {
      // If there was error obtaining it, just reset the size to 0.
      // Maybe log the error too?
      entry->binary_size = 0;
    }
    entry->arch_name = entry->exec_mod->GetArchitecture().GetArchitectureName();
  }

  if (llvm::Error er = dispatch(entry)) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(er),
                   "Failed to dispatch entry at main executable load start: {0}");
  }
}

void TelemetryManager::AtMainExecutableLoadEnd(TargetInfo *entry) {
    if (entry->exec_mod != nullptr) {
      entry->target_uuid = entry->exec_mod->GetUUID().GetAsString();
      // We don't need the rest of the data since they are already in the start entry.

        if (llvm::Error er = dispatch(entry)) {
          LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(er),
                         "Failed to dispatch entry at main executable load start: {0}");
        }
    }
}

std::unique_ptr<TelemetryManager> TelemetryManager::g_instance = nullptr;
TelemetryManager *TelemetryManager::GetInstance() { return g_instance.get(); }

void TelemetryManager::SetInstance(std::unique_ptr<TelemetryManager> manager) {
  g_instance = std::move(manager);
}

} // namespace telemetry
} // namespace lldb_private

#endif // LLVM_BUILD_TELEMETRY
