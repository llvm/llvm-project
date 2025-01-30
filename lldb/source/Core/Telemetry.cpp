//===-- Telemetry.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/Telemetry.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/UUID.h"
#include "lldb/Version/Version.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Telemetry/Telemetry.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lldb_private {

using ::llvm::Error;
using ::llvm::telemetry::Destination;
using ::llvm::telemetry::TelemetryInfo;

static uint64_t ToNanosec(const SteadyTimePoint Point) {
  return nanoseconds(Point.value().time_since_epoch()).count();
}

void LldbBaseTelemetryInfo::serialize(Serializer &serializer) const {
  serializer.write("entry_kind", getKind());
  serializer.write("session_id", SessionId);
  serializer.write("start_time", ToNanosec(stats.start));
  if (stats.end.has_value())
    serializer.write("end_time", ToNanosec(stats.end.value()));
  if (exit_desc.has_value()) {
    serializer.write("exit_code", exit_desc->exit_code);
    serializer.write("exit_msg", exit_desc->description);
  }
}

static std::string MakeUUID(lldb_private::Debugger *debugger) {
  std::string ret;
  uint8_t random_bytes[16];
  if (auto ec = llvm::getRandomBytes(random_bytes, 16)) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Failed to generate random bytes for UUID: {0}", ec.message());
    // fallback to using timestamp + debugger ID.
    ret = std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          "_" + std::to_string(debugger->GetID());
  } else {
    ret = lldb_private::UUID(random_bytes).GetAsString();
  }

  return ret;
}

TelemetryManager::TelemetryManager(
    std::unique_ptr<llvm::telemetry::Config> config,
    lldb_private::Debugger *debugger)
    : m_config(std::move(config)), m_debugger(debugger),
      m_session_uuid(MakeUUID(debugger)) {}

llvm::Error TelemetryManager::dispatch(TelemetryInfo *entry) {
  entry->SessionId = m_session_uuid;

  llvm::Error defferedErrs = llvm::Error::success();
  for (auto &destination : m_destinations)
    deferredErrs = llvm::joinErrors(std::move(deferredErrs),
                                    destination->receiveEntry(entry));

  return std::move(deferredErrs);
}

void TelemetryManager::addDestination(
    std::unique_ptr<Destination> destination) {
  m_destinations.push_back(std::move(destination));
}

} // namespace lldb_private
