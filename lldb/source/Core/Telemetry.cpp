//===-- Telemetry.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/Telemetry.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Telemetry.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/UUID.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Telemetry/Telemetry.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
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
// <16 random bytes>_<timestamp>
// This reduces the chances of getting the same UUID, even when the same
// user runs the two copies of binary at the same time.
static std::string MakeUUID() {
  auto timestmap = std::chrono::steady_clock::now().time_since_epoch().count();
  UUID uuid = UUID::Generate();
  return llvm::formatv("{0}_{1}", uuid.GetAsString(), timestmap);
}

void LLDBBaseTelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("entry_kind", getKind());
  serializer.write("session_id", SessionId);
  serializer.write("start_time", ToNanosec(start_time));
  if (end_time.has_value())
    serializer.write("end_time", ToNanosec(end_time.value()));
}

void ClientInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);
  serializer.write("client_data", client_data);
  serializer.write("client_name", client_name);
  if (error_msg.has_value())
    serializer.write("error_msg", error_msg.value());
}

void CommandInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);

  serializer.write("target_uuid", target_uuid.GetAsString());
  serializer.write("command_id", command_id);
  serializer.write("command_name", command_name);
  if (original_command.has_value())
    serializer.write("original_command", original_command.value());
  if (args.has_value())
    serializer.write("args", args.value());
  if (ret_status.has_value())
    serializer.write("ret_status", ret_status.value());
  if (error_data.has_value())
    serializer.write("error_data", error_data.value());
}

std::atomic<uint64_t> CommandInfo::g_command_id_seed = 1;
uint64_t CommandInfo::GetNextID() { return g_command_id_seed.fetch_add(1); }

void DebuggerInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);

  serializer.write("lldb_version", lldb_version);
  serializer.write("is_exit_entry", is_exit_entry);
}

void ExecutableModuleInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);

  serializer.write("uuid", uuid.GetAsString());
  serializer.write("pid", pid);
  serializer.write("triple", triple);
  serializer.write("is_start_entry", is_start_entry);
}

void ProcessExitInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);

  serializer.write("module_uuid", module_uuid.GetAsString());
  serializer.write("pid", pid);
  serializer.write("is_start_entry", is_start_entry);
  if (exit_desc.has_value()) {
    serializer.write("exit_code", exit_desc->exit_code);
    serializer.write("exit_desc", exit_desc->description);
  }
}

TelemetryManager::TelemetryManager(std::unique_ptr<LLDBConfig> config)
    : m_config(std::move(config)), m_id(MakeUUID()) {}

llvm::Error TelemetryManager::preDispatch(TelemetryInfo *entry) {
  // Assign the manager_id, and debugger_id, if available, to this entry.
  LLDBBaseTelemetryInfo *lldb_entry = llvm::cast<LLDBBaseTelemetryInfo>(entry);
  lldb_entry->SessionId = m_id;
  if (Debugger *debugger = lldb_entry->debugger)
    lldb_entry->debugger_id = debugger->GetID();
  return llvm::Error::success();
}

void TelemetryManager::DispatchClientTelemetry(
    const lldb_private::StructuredDataImpl &entry, Debugger *debugger) {
  if (!m_config->enable_client_telemetry)
    return;

  ClientInfo client_info;
  client_info.debugger = debugger;
  if (entry.GetObjectSP()->GetType() != lldb::eStructuredDataTypeDictionary) {
    LLDB_LOG(GetLog(LLDBLog::Object), "Expected Dictionary type but got {0}.",
             entry.GetObjectSP()->GetType());
    return;
  }

  auto *dict = entry.GetObjectSP()->GetAsDictionary();

  llvm::StringRef client_name;
  if (dict->GetValueForKeyAsString("client_name", client_name))
    client_info.client_name = client_name.str();
  else
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Cannot determine client_name from client-telemetry entry");

  llvm::StringRef client_data;
  if (dict->GetValueForKeyAsString("client_data", client_data))
    client_info.client_data = client_data.str();
  else
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Cannot determine client_data from client-telemetry entry");

  int64_t start_time;
  if (dict->GetValueForKeyAsInteger("start_time", start_time)) {
    client_info.start_time +=
        std::chrono::nanoseconds(static_cast<size_t>(start_time));
  } else {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Cannot determine start-time from client-telemetry entry");
  }

  int64_t end_time;
  if (dict->GetValueForKeyAsInteger("end_time", end_time)) {
    SteadyTimePoint epoch;
    client_info.end_time =
        epoch + std::chrono::nanoseconds(static_cast<size_t>(end_time));
  } else {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Cannot determine end-time from client-telemetry entry");
  }

  llvm::StringRef error_msg;
  if (dict->GetValueForKeyAsString("error", error_msg))
    client_info.error_msg = error_msg.str();

  if (llvm::Error er = dispatch(&client_info))
    LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(er),
                   "Failed to dispatch client telemetry");
}

class NoOpTelemetryManager : public TelemetryManager {
public:
  llvm::Error preDispatch(llvm::telemetry::TelemetryInfo *entry) override {
    // Does nothing.
    return llvm::Error::success();
  }

  explicit NoOpTelemetryManager()
      : TelemetryManager(std::make_unique<LLDBConfig>(
            /*EnableTelemetry=*/false, /*DetailedCommand=*/false,
            /*ClientTelemery=*/false)) {}

  virtual llvm::StringRef GetInstanceName() const override {
    return "NoOpTelemetryManager";
  }

  void DispatchClientTelemetry(const lldb_private::StructuredDataImpl &entry,
                               Debugger *debugger) override {
    // Does nothing.
  }

  llvm::Error dispatch(llvm::telemetry::TelemetryInfo *entry) override {
    // Does nothing.
    return llvm::Error::success();
  }

  static NoOpTelemetryManager *GetInstance() {
    static std::unique_ptr<NoOpTelemetryManager> g_ins =
        std::make_unique<NoOpTelemetryManager>();
    return g_ins.get();
  }
};

std::unique_ptr<TelemetryManager> TelemetryManager::g_instance = nullptr;
TelemetryManager *TelemetryManager::GetInstance() {
  // If Telemetry is disabled or if there is no default instance, then use the
  // NoOp manager. We use a dummy instance to avoid having to do nullchecks in
  // various places.
  if (!Config::BuildTimeEnableTelemetry || !g_instance)
    return NoOpTelemetryManager::GetInstance();
  return g_instance.get();
}

void TelemetryManager::SetInstance(std::unique_ptr<TelemetryManager> manager) {
  if (Config::BuildTimeEnableTelemetry)
    g_instance = std::move(manager);
}

} // namespace telemetry
} // namespace lldb_private
