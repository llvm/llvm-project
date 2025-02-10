//===-- Telemetry.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/Telemetry.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/UUID.h"
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

using ::llvm::Error;
using ::llvm::telemetry::Destination;
using ::llvm::telemetry::Serializer;
using ::llvm::telemetry::TelemetryInfo;

static uint64_t ToNanosec(const SteadyTimePoint Point) {
  return std::chrono::nanoseconds(Point.time_since_epoch()).count();
}

void LLDBBaseTelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("entry_kind", getKind());
  serializer.write("session_id", SessionId);
  serializer.write("start_time", ToNanosec(start_time));
  if (end_time.has_value())
    serializer.write("end_time", ToNanosec(end_time.value()));
}

[[maybe_unused]] static std::string MakeUUID(lldb_private::Debugger *debugger) {
  uint8_t random_bytes[16];
  if (auto ec = llvm::getRandomBytes(random_bytes, 16)) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Failed to generate random bytes for UUID: {0}", ec.message());
    // fallback to using timestamp + debugger ID.
    return llvm::formatv(
        "{0}_{1}", std::chrono::steady_clock::now().time_since_epoch().count(),
        debugger->GetID());
  }
  return lldb_private::UUID(random_bytes).GetAsString();
}

TelemetryManager::TelemetryManager(
    std::unique_ptr<llvm::telemetry::Config> config)
    : m_config(std::move(config)) {}

llvm::Error TelemetryManager::preDispatch(TelemetryInfo *entry) {
  // Do nothing for now.
  // In up-coming patch, this would be where the manager
  // attach the session_uuid to the entry.
  return Error::success();
}

} // namespace telemetry
} // namespace lldb_private
