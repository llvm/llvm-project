
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
using ::llvm::telemetry::EventStats;
using ::llvm::telemetry::ExitDescription;
using ::llvm::telemetry::SteadyTimePoint;
using ::llvm::telemetry::TelemetryInfo;

static std::string
ExitDescToString(const llvm::telemetry::ExitDescription *desc) {
  return ("ExitCode:" + desc->ExitCode) +
         (" ExixitDescription: " + desc->Description);
}

static std::string GetDuration(const EventStats &stats) {
  if (stats.End.has_value())
    return std::to_string((stats.End.value() - stats.Start).count()) +
           "(nanosec)";
  return "<NONE>";
}

std::string LldbBaseTelemetryInfo::ToString() const {
  return ("[LldbBaseTelemetryInfo]\n") + (" SessionId: " + SessionId + "\n");
}

std::string DebuggerTelemetryInfo::ToString() const {
  std::string duration_desc =
      (ExitDesc.has_value() ? "  lldb session duration: "
                            : "  lldb startup duration: ") +
      std::to_string((Stats.End.value() - Stats.Start).count()) + "(nanosec)\n";

  return LldbBaseTelemetryInfo::ToString() + "\n" +
         ("[DebuggerTelemetryInfo]\n") + ("  username: " + username + "\n") +
         ("  lldb_git_sha: " + lldb_git_sha + "\n") +
         ("  lldb_path: " + lldb_path + "\n") + ("  cwd: " + cwd + "\n") +
         duration_desc + "\n";
}

static size_t ToNanosecOrZero(const std::optional<SteadyTimePoint> &Point) {
  if (!Point.has_value())
    return 0;

  return Point.value().time_since_epoch().count();
}

llvm::json::Object DebuggerTelemetryInfo::serializeToJson() const {
  return llvm::json::Object{
      {"DebuggerInfo",
       {
           {"SessionId", SessionId},
           {"username", username},
           {"lldb_git_sha", lldb_git_sha},
           {"lldb_path", lldb_path},
           {"cwd", cwd},
           {
               "EventStats",
               {
                   {"Start", Stats.Start.time_since_epoch().count()},
                   {"End", ToNanosecOrZero(Stats.End)},
               },
           },
           // TODO: fill in more?
       }}};
}

std::string ClientTelemetryInfo::ToString() const {
  return LldbBaseTelemetryInfo::ToString() + "\n" +
         ("[DapRequestInfoEntry]\n") +
         ("  request_name: " + request_name + "\n") +
         ("  request_duration: " + GetDuration(Stats) + "(nanosec)\n") +
         ("  error_msg: " + error_msg + "\n");
}

llvm::json::Object ClientTelemetryInfo::serializeToJson() const {
  return llvm::json::Object{
      {"ClientInfo",
       {
           {"SessionId", SessionId},
           {"request_name", request_name},
           {"error_msg", error_msg},
           {
               "EventStats",
               {
                   {"Start", Stats.Start.time_since_epoch().count()},
                   {"End", ToNanosecOrZero(Stats.End)},
               },
           },
       }}};
}

std::string TargetTelemetryInfo::ToString() const {
  std::string exit_or_load_desc;
  if (ExitDesc.has_value()) {
    // If this entry was emitted for an exit
    exit_or_load_desc = "  process_duration: " + GetDuration(Stats) +
                        ExitDescToString(&(ExitDesc.value())) + "\n";
  } else {
    // This was emitted for a load event.
    // See if it was the start-load or end-load entry
    if (Stats.End.has_value()) {
      exit_or_load_desc =
          "  startup_init_duration: " + GetDuration(Stats) + "\n";
    } else {
      exit_or_load_desc = " startup_init_start\n";
    }
  }
  return LldbBaseTelemetryInfo::ToString() + "\n" +
         ("[TargetTelemetryInfo]\n") +
         ("  target_uuid: " + target_uuid + "\n") +
         ("  file_format: " + file_format + "\n") +
         ("  binary_path: " + binary_path + "\n") +
         ("  binary_size: " + std::to_string(binary_size) + "\n") +
         exit_or_load_desc;
}

llvm::json::Object TargetTelemetryInfo::serializeToJson() const {
  return llvm::json::Object{{
      "TargetInfo",
      {
          {"SessionId", SessionId},
          {"target_uuid", target_uuid},
          {"binary_path", binary_path},
          {"binary_size", binary_size},
          // TODO: fill in more
      },
  }};
}

std::string CommandTelemetryInfo::ToString() const {
  // Whether this entry was emitted at the start or at the end of the
  // command-execution.
  if (Stats.End.has_value()) {
    return LldbBaseTelemetryInfo::ToString() + "\n" +
           ("[CommandTelemetryInfo] - END\n") +
           ("  target_uuid: " + target_uuid + "\n") +
           ("  command_uuid: " + command_uuid + "\n") +
           ("  command_name: " + command_name + "\n") +
           ("  args: " + args + "\n") +
           ("  command_runtime: " + GetDuration(Stats) + "\n") +
           (ExitDesc.has_value() ? ExitDescToString(&(ExitDesc.value()))
                                 : "no exit-description") +
           "\n";
  } else {
    return LldbBaseTelemetryInfo::ToString() + "\n" +
           ("[CommandTelemetryInfo] - START\n") +
           ("  target_uuid: " + target_uuid + "\n") +
           ("  command_uuid: " + command_uuid + "\n") +
           ("  original_command: " + original_command + "\n");
  }
}

llvm::json::Object CommandTelemetryInfo::serializeToJson() const {
  llvm::json::Object inner;

  inner.insert({"SessionId", SessionId});
  inner.insert({"target_uuid", target_uuid});
  inner.insert({"command_uuid", command_uuid});
  inner.insert({"args", args});
  inner.insert({"original_command", original_command});
  inner.insert({
      "EventStats",
      {
          {"Start", Stats.Start.time_since_epoch().count()},
          {"End", ToNanosecOrZero(Stats.End)},
      },
  });

  // If this entry was emitted at the end of the command-execution,
  // then calculate the runtime too.
  if (Stats.End.has_value()) {
    inner.insert(
        {"command_runtime", (Stats.End.value() - Stats.Start).count()});
    if (ExitDesc.has_value()) {
      inner.insert({"exit_code", ExitDesc->ExitCode});
      inner.insert({"exit_msg", ExitDesc->Description});
      inner.insert({"return_status", static_cast<int>(ret_status)});
    }
  }

  return llvm::json::Object{{"CommandInfo", std::move(inner)}};
}

std::string MiscTelemetryInfo::ToString() const {
  std::string ret;
  llvm::raw_string_ostream ret_strm(ret);
  ret_strm << LldbBaseTelemetryInfo::ToString() << "\n[MiscTelemetryInfo]\n"
           << "  target_uuid: " << target_uuid + "\n"
           << "  meta_data:\n";
  for (const auto &kv : meta_data) {
    ret_strm << "    " << kv.first << ": " << kv.second << "\n";
  }
  return ret;
}

llvm::json::Object MiscTelemetryInfo::serializeToJson() const {
  llvm::json::Object meta_data_obj;

  for (const auto &kv : meta_data)
    meta_data_obj.insert({kv.first, kv.second});

  return llvm::json::Object{{
      "MiscInfo",
      {
          {"SessionId", SessionId},
          {"target_uuid", target_uuid},
          {"meta_data", std::move(meta_data_obj)},
      },
  }};
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
