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
#include "lldb/Utility/UUID.h"
#include "lldb/Version/Version.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
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
// <16 random bytes>_<timestamp>
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

  return llvm::formatv(
      "{0}_{1}", randomString,
      std::chrono::steady_clock::now().time_since_epoch().count());
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
  serializer.write("request_name", request_name);
  if (error_msg.has_value())
    serializer.write("error_msg", error_msg.value());
}

void DebuggerInfo::serialize(Serializer &serializer) const {
  LLDBBaseTelemetryInfo::serialize(serializer);

  serializer.write("lldb_version", lldb_version);
  serializer.write("is_exit_entry", is_exit_entry);
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
  if (!m_config->m_enable_client_telemetry)
    return;

  ClientInfo client_info;
  client_info.debugger = debugger;

  std::optional<llvm::StringRef> request_name = entry.getString("request_name");
  if (!request_name.has_value())
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Cannot determine request name from client-telemetry entry");
  else
    client_info.request_name = request_name->str();

  std::optional<int64_t> start_time = entry.getInteger("start_time");
  std::optional<int64_t> end_time = entry.getInteger("end_time");
  SteadyTimePoint epoch;
  if (!start_time.has_value()) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Cannot determine start-time from client-telemetry entry");
    client_info.start_time = 0;
  } else {
    client_info.start_time =
        epoch + std::chrono::nanoseconds(static_cast<size_t>(*start_time));
  }

  if (!end_time.has_value()) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Cannot determine end-time from client-telemetry entry");
  } else {
    client_info.end_time =
        epoch + std::chrono::nanoseconds(static_cast<size_t>(*end_time));
  }

  std::optional<llvm::StringRef> error_msg = entry.getString("error");
  if (error_msg.has_value())
    client_info.error_msg = error_msg->str();

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
            /*EnableTelemetry*/ false, /*DetailedCommand*/ false)) {}

  llvm::StringRef GetInstanceName() const override {
    return "NoOpTelemetryManager";
  }

  llvm::Error dispatch(llvm::telemetry::TelemetryInfo *entry) override {
    // Does nothing.
    return llvm::Error::success();
  }

  void DispatchClientTelemetry(const lldb_private::StructuredDataImpl &entry,
                               Debugger *debugger) override {
    // Does nothing.
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
