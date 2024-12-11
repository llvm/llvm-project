
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
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Telemetry/Telemetry.h"

namespace lldb_private {

using ::llvm::Error;
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

std::unique_ptr<TelemetryManager> TelemetryManager::CreateInstance(
    std::unique_ptr<llvm::telemetry::Config> config,
    lldb_private::Debugger *debugger) {

  TelemetryManager *ins = new TelemetryManager(std::move(config), debugger);

  return std::unique_ptr<TelemetryManager>(ins);
}

llvm::Error TelemetryManager::dispatch(TelemetryInfo *entry) {
  entry->SessionId = m_session_uuid;

  for (auto &destination : m_destinations) {
    llvm::Error err = destination->receiveEntry(entry);
    if (err) {
      return std::move(err);
    }
  }
  return Error::success();
}

void TelemetryManager::addDestination(
    std::unique_ptr<Destination> destination) {
  m_destinations.push_back(std::move(destination));
}

void TelemetryManager::LogStartup(DebuggerTelemetryInfo *entry) {
  UserIDResolver &resolver = lldb_private::HostInfo::GetUserIDResolver();
  std::optional<llvm::StringRef> opt_username =
      resolver.GetUserName(lldb_private::HostInfo::GetUserID());
  if (opt_username)
    entry->username = *opt_username;

  entry->lldb_git_sha =
      lldb_private::GetVersion(); // TODO: find the real git sha?

  llvm::SmallString<64> cwd;
  if (!llvm::sys::fs::current_path(cwd)) {
    entry->cwd = cwd.c_str();
  } else {
    MiscTelemetryInfo misc_info;
    misc_info.meta_data["internal_errors"] = "Cannot determine CWD";
    if (auto er = dispatch(&misc_info)) {
      LLDB_LOG(GetLog(LLDBLog::Object),
               "Failed to dispatch misc-info from startup");
    }
  }

  if (auto er = dispatch(entry)) {
    LLDB_LOG(GetLog(LLDBLog::Object), "Failed to dispatch entry from startup");
  }

  // Optional part
  CollectMiscBuildInfo();
}

void TelemetryManager::LogExit(DebuggerTelemetryInfo *entry) {
  if (auto *selected_target =
          m_debugger->GetSelectedExecutionContext().GetTargetPtr()) {
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

void TelemetryManager::LogProcessExit(TargetTelemetryInfo *entry) {
  entry->target_uuid =
      entry->target_ptr && !entry->target_ptr->IsDummyTarget()
          ? entry->target_ptr->GetExecutableModule()->GetUUID().GetAsString()
          : "";

  dispatch(entry);
}

void TelemetryManager::CollectMiscBuildInfo() {
  // collecting use-case specific data
}

void TelemetryManager::LogMainExecutableLoadStart(TargetTelemetryInfo *entry) {
  entry->binary_path =
      entry->exec_mod->GetFileSpec().GetPathAsConstString().GetCString();
  entry->file_format = entry->exec_mod->GetArchitecture().GetArchitectureName();
  entry->target_uuid = entry->exec_mod->GetUUID().GetAsString();
  if (auto err = llvm::sys::fs::file_size(
          entry->exec_mod->GetFileSpec().GetPath(), entry->binary_size)) {
    // If there was error obtaining it, just reset the size to 0.
    // Maybe log the error too?
    entry->binary_size = 0;
  }
  dispatch(entry);
}

void TelemetryManager::LogMainExecutableLoadEnd(TargetTelemetryInfo *entry) {
  lldb::ModuleSP exec_mod = entry->exec_mod;
  entry->binary_path =
      exec_mod->GetFileSpec().GetPathAsConstString().GetCString();
  entry->file_format = exec_mod->GetArchitecture().GetArchitectureName();
  entry->target_uuid = exec_mod->GetUUID().GetAsString();
  entry->binary_size = exec_mod->GetObjectFile()->GetByteSize();

  dispatch(entry);

  // Collect some more info, might be useful?
  MiscTelemetryInfo misc_info;
  misc_info.target_uuid = exec_mod->GetUUID().GetAsString();
  misc_info.meta_data["symtab_index_time"] =
      std::to_string(exec_mod->GetSymtabIndexTime().get().count());
  misc_info.meta_data["symtab_parse_time"] =
      std::to_string(exec_mod->GetSymtabParseTime().get().count());
  dispatch(&misc_info);
}

void TelemetryManager::LogClientTelemetry(
    const lldb_private::StructuredDataImpl &entry) {
  // TODO: pull the dictionary out of entry
  ClientTelemetryInfo client_info;
  /*
  std::optional<llvm::StringRef> request_name = entry.getString("request_name");
  if (!request_name.has_value()) {
    MiscTelemetryInfo misc_info = MakeBaseEntry<MiscTelemetryInfo>();
    misc_info.meta_data["internal_errors"] =
        "Cannot determine request name from client entry";
    // TODO: Dump the errornous entry to stderr too?
    EmitToDestinations(&misc_info);
    return;
  }
  client_info.request_name = request_name->str();

  std::optional<int64_t> start_time = entry.getInteger("start_time");
  std::optional<int64_t> end_time = entry.getInteger("end_time");

  if (!start_time.has_value() || !end_time.has_value()) {
    MiscTelemetryInfo misc_info = MakeBaseEntry<MiscTelemetryInfo>();
    misc_info.meta_data["internal_errors"] =
        "Cannot determine start/end time from client entry";
    EmitToDestinations(&misc_info);
    return;
  }

  SteadyTimePoint epoch;
  client_info.Stats.Start =
      epoch + std::chrono::nanoseconds(static_cast<size_t>(*start_time));
  client_info.Stats.End =
      epoch + std::chrono::nanoseconds(static_cast<size_t>(*end_time));

  std::optional<llvm::StringRef> error_msg = entry.getString("error");
  if (error_msg.has_value())
    client_info.error_msg = error_msg->str();
  */

  dispatch(&client_info);
}

void TelemetryManager::LogCommandStart(CommandTelemetryInfo *entry) {
  // If we have a target attached to this command, then get the UUID.
  if (entry->target_ptr &&
      entry->target_ptr->GetExecutableModule() != nullptr) {
    entry->target_uuid =
        entry->target_ptr->GetExecutableModule()->GetUUID().GetAsString();
  } else {
    entry->target_uuid = "";
  }

  dispatch(entry);
}

void TelemetryManager::LogCommandEnd(CommandTelemetryInfo *entry) {
  // If we have a target attached to this command, then get the UUID.
  if (entry->target_ptr &&
      entry->target_ptr->GetExecutableModule() != nullptr) {
    entry->target_uuid =
        entry->target_ptr->GetExecutableModule()->GetUUID().GetAsString();
  } else {
    entry->target_uuid = "";
  }

  entry->exit_desc = {entry->result->Succeeded() ? 0 : -1, ""};
  if (llvm::StringRef error_data = entry->result->GetErrorData();
      !error_data.empty()) {
    entry->exit_desc->description = error_data.str();
  }
  entry->ret_status = entry->result->GetStatus();
  dispatch(entry);
}

} // namespace lldb_private
