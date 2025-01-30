//===-- Telemetry.h ----------------------------------------------*- C++
//-*-===//
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

using llvm::telemetry::Destination;
using llvm::telemetry::KindType;
using llvm::telemetry::Serializer;
using llvm::telemetry::TelemetryInfo;

struct LldbEntryKind : public ::llvm::telemetry::EntryKind {
  static const KindType BaseInfo = 0b11000;
};

/// Defines a convenient type for timestamp of various events.
/// This is used by the EventStats below.
using SteadyTimePoint = std::chrono::time_point<std::chrono::steady_clock,
                                                std::chrono::nanoseconds>;

/// Various time (and possibly memory) statistics of an event.
struct EventStats {
  // REQUIRED: Start time of an event
  SteadyTimePoint start;
  // OPTIONAL: End time of an event - may be empty if not meaningful.
  std::optional<SteadyTimePoint> end;
  // TBD: could add some memory stats here too?

  EventStats() = default;
  EventStats(SteadyTimePoint start) : start(start) {}
  EventStats(SteadyTimePoint start, SteadyTimePoint end)
      : start(start), end(end) {}
};

/// Describes the exit signal of an event.
struct ExitDescription {
  int exit_code;
  std::string description;
};

struct LldbBaseTelemetryInfo : public TelemetryInfo {
  EventStats stats;

  std::optional<ExitDescription> exit_desc;

  Debugger *debugger;

  // For dyn_cast, isa, etc operations.
  KindType getKind() const override { return LldbEntryKind::BaseInfo; }

  static bool classof(const TelemetryInfo *t) {
    // Subclasses of this is also acceptable.
    return (t->getKind() & LldbEntryKind::BaseInfo) == LldbEntryKind::BaseInfo;
  }

  void serialize(Serializer &serializer) const override;
};

/// The base Telemetry manager instance in LLDB
/// This class declares additional instrumentation points
/// applicable to LLDB.
class TelemetryManager : public llvm::telemetry::Manager {
public:
  TelemetryManager(std::unique_ptr<llvm::telemetry::Config> config);

  llvm::Error dispatch(TelemetryInfo *entry) override;

  void addDestination(std::unique_ptr<Destination> destination) override;

private:
  std::unique_ptr<llvm::telemetry::Config> m_config;
  const std::string m_session_uuid;
  std::vector<std::unique_ptr<Destination>> m_destinations;
};

} // namespace lldb_private
#endif // LLDB_CORE_TELEMETRY_H
