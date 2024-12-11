
//===-- Telemetry.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/Telemetry.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBProcess.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/TelemetryVendor.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/UUID.h"
#include "lldb/Version/Version.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Telemetry/Telemetry.h"

namespace lldb_private {

using ::llvm::telemetry::Destination;
using ::llvm::telemetry::TelemetryInfo;

static std::string GetDuration(const EventStats &stats) {
  if (stats.end.has_value())
    return std::to_string((stats.end.value() - stats.start).count()) +
           "(nanosec)";
  return "<NONE>";
}

static size_t ToNanosecOrZero(const std::optional<SteadyTimePoint> &Point) {
  if (!Point.has_value())
    return 0;

  return Point.value().time_since_epoch().count();
}

void LldbBaseTelemetryInfo::serialize(Serializer &serializer) const {
  serializer.writeInt32("EntryKind", getKind());
  serializer.writeString("SessionId", SessionId);
}

void DebuggerTelemetryInfo::serialize(Serializer &serializer) const {
  LldbBaseTelemetryInfo::serialize(serializer);
  serializer.writeString("username", username);
  serializer.writeString("lldb_path", lldb_path);
  serializer.writeString("cwd", cwd);
  serializer.writeSizeT("start", stats.start.time_since_epoch().count());
  serializer.writeSizeT("end", ToNanosecOrZero(stats.end));
}

void ClientTelemetryInfo::serialize(Serializer &serializer) const {
  LldbBaseTelemetryInfo::serialize(serializer);
  serializer.writeString("request_name", request_name);
  serializer.writeString("error_msg", error_msg);
  serializer.writeSizeT("start", stats.start.time_since_epoch().count());
  serializer.writeSizeT("end", ToNanosecOrZero(stats.end));
}

void TargetTelemetryInfo::serialize(Serializer &serializer) const {
  LldbBaseTelemetryInfo::serialize(serializer);
  serializer.writeString("target_uuid", target_uuid);
  serializer.writeString("binary_path", binary_path);
  serializer.writeSizeT("binary_size", binary_size);
}

void CommandTelemetryInfo::serialize(Serializer &serializer) const {
  LldbBaseTelemetryInfo::serialize(serializer);
  serializer.writeString("target_uuid", target_uuid);
  serializer.writeString("command_uuid", command_uuid);
  serializer.writeString("args", args);
  serializer.writeString("original_command", original_command);
  serializer.writeSizeT("start", stats.start.time_since_epoch().count());
  serializer.writeSizeT("end", ToNanosecOrZero(stats.end));

  // If this entry was emitted at the end of the command-execution,
  // then calculate the runtime too.
  if (stats.end.has_value()) {
    serializer.writeSizeT("command_runtime",
                          (stats.end.value() - stats.start).count());
    if (exit_desc.has_value()) {
      serializer.writeInt32("exit_code", exit_desc->exit_code);
      serializer.writeString("exit_msg", exit_desc->description);
      serializer.writeInt32("return_status", static_cast<int>(ret_status));
    }
  }
}

void MiscTelemetryInfo::serialize(Serializer &serializer) const {
  LldbBaseTelemetryInfo::serialize(serializer);
  serializer.writeString("target_uuid", target_uuid);
  serializer.writeKeyValueMap("meta_data", meta_data);
}

std::unique_ptr<LldbTelemeter>
LldbTelemeter::CreateInstance(lldb_private::Debugger *debugger) {
  // TODO: do we cache the plugin?
  TelemetryVendor *vendor = TelemetryVendor::FindPlugin();
  if (vendor == nullptr) {
    LLDB_LOG(GetLog(LLDBLog::Object),
             "Failed to find a TelemetryVendor plugin instance");
    return nullptr;
  }

  return vendor->CreateTelemeter(debugger);
}

} // namespace lldb_private
