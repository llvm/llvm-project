
//===-- Telemetry.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/Telemetry.h"

#include <stdbool.h>
#include <sys/auxv.h>

#include <memory>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBProcess.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
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

#ifdef HAS_VENDOR_TELEMETRY_PLUGINS
// TODO: could make this path a build-variable rather than hard-coded header
// path
#include "lldb/Core/VendorTelemetryPlugin.h"

namespace vendor_specific {

// Set any additional configurations, as needed.
extern void ApplyVendorSpecificConfigs(
    llvm::telemetry::TelemetryConfig *config) /* __attribute__((weak))*/;

// Return a copy of the given entry, but with certain fields that are deemed
// PII risk removed.
extern std::shared_ptr<llvm::telemetry::TelemetryInfo> SanitizeSensitiveFields(
    const llvm::telemetry::TelemetryInfo *entry) /*__attribute__((weak))*/;

extern std::shared_ptr<lldb_private::LldbTelemeter>
CreateVendorSpecificTelemeter(
    llvm::telemetry::TelemetryConfig *config) /*__attribute__((weak))*/;
} // namespace vendor_specific
#endif

namespace lldb_private {

using ::llvm::telemetry::Destination;
using ::llvm::telemetry::EventStats;
using ::llvm::telemetry::ExitDescription;
using ::llvm::telemetry::SteadyTimePoint;
using ::llvm::telemetry::TelemetryInfo;

static std::string ExitDescToString(const ExitDescription *desc) {
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

class StreamTelemetryDestination : public Destination {
public:
  StreamTelemetryDestination(llvm::raw_ostream &os, std::string desc)
      : os(os), desc(desc) {}
  llvm::Error emitEntry(const llvm::telemetry::TelemetryInfo *entry) override {
    // Unless there exists a custom (vendor-defined) data-cleanup
    // for printing, upstream Telemetry should not leak anything other than the
    // basic.
#ifdef HAS_TELEMETRY_FIELDS_PRINTER
    os << SanitizeSensitiveFields(entry)->ToString() << "\n";
#else
    os << "session_uuid: " << entry->SessionId
       << "<the rest is omitted due to PII risk>\n";
#endif
    os.flush();
    return llvm::ErrorSuccess();
  }

  std::string name() const override { return desc; }

private:
  llvm::raw_ostream &os;
  const std::string desc;
};

// No-op logger to use when users disable telemetry
class NoOpTelemeter : public LldbTelemeter {
public:
  static std::unique_ptr<LldbTelemeter> CreateInstance(Debugger *debugger) {
    return std::unique_ptr<LldbTelemeter>(new NoOpTelemeter(debugger));
  }

  NoOpTelemeter(Debugger *debugger) {}
  void logStartup(llvm::StringRef tool_path, TelemetryInfo *entry) override {}
  void logExit(llvm::StringRef tool_path, TelemetryInfo *entry) override {}

  void LogProcessExit(int status, llvm::StringRef exit_string, EventStats stats,
                      Target *target_ptr) override {}
  void LogMainExecutableLoadStart(lldb::ModuleSP exec_mod,
                                  EventStats stats) override {}
  void LogMainExecutableLoadEnd(lldb::ModuleSP exec_mod,
                                EventStats stats) override {}

  void LogCommandStart(llvm::StringRef uuid, llvm::StringRef original_command,
                       EventStats stats, Target *target_ptr) override {}
  void LogCommandEnd(llvm::StringRef uuid, llvm::StringRef command_name,
                     llvm::StringRef command_args, EventStats stats,
                     Target *target_ptr, CommandReturnObject *result) override {
  }

  void
  LogClientTelemetry(const lldb_private::StructuredDataImpl &entry) override {}

  void addDestination(llvm::telemetry::Destination *destination) override {}
  std::string GetNextUUID() override { return ""; }
};

class BasicTelemeter : public LldbTelemeter {
public:
  static std::unique_ptr<BasicTelemeter> CreateInstance(Debugger *);

  virtual ~BasicTelemeter() = default;

  void logStartup(llvm::StringRef lldb_path, TelemetryInfo *entry) override;
  void logExit(llvm::StringRef lldb_path, TelemetryInfo *entry) override;

  void LogProcessExit(int status, llvm::StringRef exit_string, EventStats stats,
                      Target *target_ptr) override;
  void LogMainExecutableLoadStart(lldb::ModuleSP exec_mod,
                                  EventStats stats) override;
  void LogMainExecutableLoadEnd(lldb::ModuleSP exec_mod,
                                EventStats stats) override;

  void LogCommandStart(llvm::StringRef uuid, llvm::StringRef original_command,
                       EventStats stats, Target *target_ptr) override;
  void LogCommandEnd(llvm::StringRef uuid, llvm::StringRef command_name,
                     llvm::StringRef command_args, EventStats stats,
                     Target *target_ptr, CommandReturnObject *result) override;

  void
  LogClientTelemetry(const lldb_private::StructuredDataImpl &entry) override;

  void addDestination(Destination *destination) override {
    m_destinations.push_back(destination);
  }

  std::string GetNextUUID() override {
    return std::to_string(uuid_seed.fetch_add(1));
  }

protected:
  BasicTelemeter(Debugger *debugger);

  void CollectMiscBuildInfo();

private:
  template <typename EntrySubType> EntrySubType MakeBaseEntry() {
    EntrySubType entry;
    entry.SessionId = m_session_uuid;
    entry.Counter = counter.fetch_add(1);
    return entry;
  }

  void EmitToDestinations(const TelemetryInfo *entry);

  Debugger *m_debugger;
  const std::string m_session_uuid;
  std::string startup_lldb_path;

  // counting number of entries.
  std::atomic<size_t> counter = 0;

  std::vector<Destination *> m_destinations;

  std::atomic<size_t> uuid_seed = 0;
};

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

BasicTelemeter::BasicTelemeter(lldb_private::Debugger *debugger)
    : m_debugger(debugger), m_session_uuid(MakeUUID(debugger)) {}

std::unique_ptr<BasicTelemeter>
BasicTelemeter::CreateInstance(lldb_private::Debugger *debugger) {
  auto *config = GetTelemetryConfig();

  BasicTelemeter *ins = new BasicTelemeter(debugger);
  for (const std ::string &dest : config->AdditionalDestinations) {
    if (dest == "stdout") {
      ins->addDestination(
          new StreamTelemetryDestination(llvm::outs(), "stdout"));
    } else if (dest == "stderr") {
      ins->addDestination(
          new StreamTelemetryDestination(llvm::errs(), "stderr"));
    } else {
      // TODO: handle custom values as needed?
    }
  }

  return std::unique_ptr<BasicTelemeter>(ins);
}

void BasicTelemeter::EmitToDestinations(const TelemetryInfo *entry) {
  // TODO: can do this in a separate thread (need to own the ptrs!).
  for (Destination *destination : m_destinations) {
    llvm::Error err = destination->emitEntry(entry);
    if (err) {
      LLDB_LOG(GetLog(LLDBLog::Object),
               "Error emitting to destination(name = {0})",
               destination->name());
    }
  }
}

void BasicTelemeter::logStartup(llvm::StringRef lldb_path,
                                TelemetryInfo *entry) {
  startup_lldb_path = lldb_path.str();
  lldb_private::DebuggerTelemetryInfo startup_info =
      MakeBaseEntry<DebuggerTelemetryInfo>();

  UserIDResolver &resolver = lldb_private::HostInfo::GetUserIDResolver();
  std::optional<llvm::StringRef> opt_username =
      resolver.GetUserName(lldb_private::HostInfo::GetUserID());
  if (opt_username)
    startup_info.username = *opt_username;

  startup_info.lldb_git_sha =
      lldb_private::GetVersion(); // TODO: find the real git sha
  startup_info.lldb_path = startup_lldb_path;
  startup_info.Stats = entry->Stats;

  llvm::SmallString<64> cwd;
  if (!llvm::sys::fs::current_path(cwd)) {
    startup_info.cwd = cwd.c_str();
  } else {
    MiscTelemetryInfo misc_info = MakeBaseEntry<MiscTelemetryInfo>();
    misc_info.meta_data["internal_errors"] = "Cannot determine CWD";
    EmitToDestinations(&misc_info);
  }

  EmitToDestinations(&startup_info);

  // Optional part
  CollectMiscBuildInfo();
}

void BasicTelemeter::logExit(llvm::StringRef lldb_path, TelemetryInfo *entry) {
  // we should be shutting down the same instance that we started?!
  // llvm::Assert(startup_lldb_path == lldb_path.str());

  lldb_private::DebuggerTelemetryInfo exit_info =
      MakeBaseEntry<lldb_private::DebuggerTelemetryInfo>();
  exit_info.Stats = entry->Stats;
  exit_info.lldb_path = startup_lldb_path;
  if (auto *selected_target =
          m_debugger->GetSelectedExecutionContext().GetTargetPtr()) {
    if (!selected_target->IsDummyTarget()) {
      const lldb::ProcessSP proc = selected_target->GetProcessSP();
      if (proc == nullptr) {
        // no process has been launched yet.
        exit_info.ExitDesc = {-1, "no process launched."};
      } else {
        exit_info.ExitDesc = {proc->GetExitStatus(), ""};
        if (const char *description = proc->GetExitDescription())
          exit_info.ExitDesc->Description = std::string(description);
      }
    }
  }
  EmitToDestinations(&exit_info);
}

void BasicTelemeter::LogProcessExit(int status, llvm::StringRef exit_string,
                                    EventStats stats, Target *target_ptr) {
  lldb_private::TargetTelemetryInfo exit_info =
      MakeBaseEntry<TargetTelemetryInfo>();
  exit_info.Stats = std::move(stats);
  exit_info.target_uuid =
      target_ptr && !target_ptr->IsDummyTarget()
          ? target_ptr->GetExecutableModule()->GetUUID().GetAsString()
          : "";
  exit_info.ExitDesc = {status, exit_string.str()};

  EmitToDestinations(&exit_info);
}

void BasicTelemeter::CollectMiscBuildInfo() {
  // collecting use-case specific data
}

void BasicTelemeter::LogMainExecutableLoadStart(lldb::ModuleSP exec_mod,
                                                EventStats stats) {
  TargetTelemetryInfo target_info = MakeBaseEntry<TargetTelemetryInfo>();
  target_info.Stats = std::move(stats);
  target_info.binary_path =
      exec_mod->GetFileSpec().GetPathAsConstString().GetCString();
  target_info.file_format = exec_mod->GetArchitecture().GetArchitectureName();
  target_info.target_uuid = exec_mod->GetUUID().GetAsString();
  if (auto err = llvm::sys::fs::file_size(exec_mod->GetFileSpec().GetPath(),
                                          target_info.binary_size)) {
    // If there was error obtaining it, just reset the size to 0.
    // Maybe log the error too?
    target_info.binary_size = 0;
  }
  EmitToDestinations(&target_info);
}

void BasicTelemeter::LogMainExecutableLoadEnd(lldb::ModuleSP exec_mod,
                                              EventStats stats) {
  TargetTelemetryInfo target_info = MakeBaseEntry<TargetTelemetryInfo>();
  target_info.Stats = std::move(stats);
  target_info.binary_path =
      exec_mod->GetFileSpec().GetPathAsConstString().GetCString();
  target_info.file_format = exec_mod->GetArchitecture().GetArchitectureName();
  target_info.target_uuid = exec_mod->GetUUID().GetAsString();
  target_info.binary_size = exec_mod->GetObjectFile()->GetByteSize();

  EmitToDestinations(&target_info);

  // Collect some more info,  might be useful?
  MiscTelemetryInfo misc_info = MakeBaseEntry<MiscTelemetryInfo>();
  misc_info.target_uuid = exec_mod->GetUUID().GetAsString();
  misc_info.meta_data["symtab_index_time"] =
      std::to_string(exec_mod->GetSymtabIndexTime().get().count());
  misc_info.meta_data["symtab_parse_time"] =
      std::to_string(exec_mod->GetSymtabParseTime().get().count());
  EmitToDestinations(&misc_info);
}

void BasicTelemeter::LogClientTelemetry(
    const lldb_private::StructuredDataImpl &entry) {
  // TODO: pull the dictionary out of entry
  ClientTelemetryInfo client_info = MakeBaseEntry<ClientTelemetryInfo>();
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

  EmitToDestinations(&client_info);
}

void BasicTelemeter::LogCommandStart(llvm::StringRef uuid,
                                     llvm::StringRef original_command,
                                     EventStats stats, Target *target_ptr) {

  lldb_private::CommandTelemetryInfo command_info =
      MakeBaseEntry<CommandTelemetryInfo>();

  // If we have a target attached to this command, then get the UUID.
  command_info.target_uuid = "";
  if (target_ptr && target_ptr->GetExecutableModule() != nullptr) {
    command_info.target_uuid =
        target_ptr->GetExecutableModule()->GetUUID().GetAsString();
  }
  command_info.command_uuid = uuid.str();
  command_info.original_command = original_command.str();
  command_info.Stats = std::move(stats);

  EmitToDestinations(&command_info);
}

void BasicTelemeter::LogCommandEnd(llvm::StringRef uuid,
                                   llvm::StringRef command_name,
                                   llvm::StringRef command_args,
                                   EventStats stats, Target *target_ptr,
                                   CommandReturnObject *result) {

  lldb_private::CommandTelemetryInfo command_info =
      MakeBaseEntry<CommandTelemetryInfo>();

  // If we have a target attached to this command, then get the UUID.
  command_info.target_uuid = "";
  if (target_ptr && target_ptr->GetExecutableModule() != nullptr) {
    command_info.target_uuid =
        target_ptr->GetExecutableModule()->GetUUID().GetAsString();
  }
  command_info.command_uuid = uuid.str();
  command_info.command_name = command_name.str();
  command_info.args = command_args.str();
  command_info.Stats = std::move(stats);
  command_info.ExitDesc = {result->Succeeded() ? 0 : -1, ""};
  if (llvm::StringRef error_data = result->GetErrorData();
      !error_data.empty()) {
    command_info.ExitDesc->Description = error_data.str();
  }
  command_info.ret_status = result->GetStatus();
  EmitToDestinations(&command_info);
}

llvm::StringRef parse_value(llvm::StringRef str, llvm::StringRef label) {
  return str.substr(label.size()).trim();
}

bool parse_field(llvm::StringRef str, llvm::StringRef label) {
  if (parse_value(str, label) == "true")
    return true;
  return false;
}

llvm::telemetry::Config *MakeTelemetryConfig() {
  bool enable_telemetry = false;
  std::vector<std::string> additional_destinations;

  // Look in the $HOME/.lldb_telemetry_config file to populate the struct
  llvm::SmallString<64> init_file;
  FileSystem::Instance().GetHomeDirectory(init_file);
  llvm::sys::path::append(init_file, ".lldb_telemetry_config");
  FileSystem::Instance().Resolve(init_file);
  if (llvm::sys::fs::exists(init_file)) {
    auto contents = llvm::MemoryBuffer::getFile(init_file, /*IsText*/ true);
    if (contents) {
      llvm::line_iterator iter =
          llvm::line_iterator(contents->get()->getMemBufferRef());
      for (; !iter.is_at_eof(); ++iter) {
        if (iter->starts_with("enable_telemetry:")) {
          enable_telemetry = parse_field(*iter, "enable_telemetry:");
        } else if (iter->starts_with("destination:")) {
          llvm::StringRef dest = parse_value(*iter, "destination:");
          if (dest == "stdout") {
            additional_destinations.push_back("stdout");
          } else if (dest == "stderr") {
            additional_destinations.push_back("stderr");
          } else {
            additional_destinations.push_back(dest.str());
          }
        }
      }
    } else {
      LLDB_LOG(GetLog(LLDBLog::Object), "Error reading config file at {0}",
               init_file.c_str());
    }
  }

  auto *ret =
      new llvm::telemetry::Config{enable_telemetry, additional_destinations};
#ifdef HAS_VENDOR_TELEMETRY_PLUGINS
  vendor_specific::ApplyVendorSpecificConfigs(ret);
#endif
  return ret;
}

llvm::telemetry::Config *GetTelemetryConfig() {
  static llvm::telemetry::Config *config = MakeTelemetryConfig();
  return config;
}

std::unique_ptr<LldbTelemeter>
LldbTelemeter::CreateInstance(lldb_private::Debugger *debugger) {
  auto *config = GetTelemetryConfig();
  if (!config->EnableTelemetry) {
    return NoOpTelemeter::CreateInstance(debugger);
  }

#ifdef HAS_VENDOR_TELEMETRY_PLUGINS
  return vendor_specific::CreateVendorSpecificTelemeter(config);
#else
  return BasicTelemeter::CreateInstance(debugger);
#endif
}
} // namespace lldb_private
