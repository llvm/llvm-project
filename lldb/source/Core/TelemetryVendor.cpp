//===-- TelemetryVendor.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/TelemetryVendor.h"

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
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Telemetry.h"
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

using namespace lldb;
using namespace lldb_private;

namespace {

using ::llvm::Error;
using ::llvm::StringRef;
using ::llvm::telemetry::Destination;
using ::llvm::telemetry::TelemetryInfo;

// No-op logger to use when users disable telemetry
class NoOpTelemeter : public LldbTelemeter {
public:
  static std::unique_ptr<LldbTelemeter> CreateInstance(Debugger *debugger) {
    return std::unique_ptr<LldbTelemeter>(new NoOpTelemeter(debugger));
  }

  NoOpTelemeter(Debugger *debugger) {}

  void addDestination(std::unique_ptr<Destination> destination) override {}
  llvm::Error dispatch(TelemetryInfo *entry) override {
    return Error::success();
  }
  void LogStartup(DebuggerTelemetryInfo *entry) override {}
  void LogExit(DebuggerTelemetryInfo *entry) override {}
  void LogMainExecutableLoadStart(TargetTelemetryInfo *entry) override {}
  void LogMainExecutableLoadEnd(TargetTelemetryInfo *entry) override {}
  void LogProcessExit(TargetTelemetryInfo *entry) override {}
  void LogCommandStart(CommandTelemetryInfo *entry) override {}
  void LogCommandEnd(CommandTelemetryInfo *entry) override {}
  void
  LogClientTelemetry(const lldb_private::StructuredDataImpl &entry) override {}

  std::string GetNextUUID() override { return ""; }
};

class BasicTelemeter : public LldbTelemeter {
public:
  static std::unique_ptr<BasicTelemeter>
  CreateInstance(std::unique_ptr<llvm::telemetry::Config> config,
                 Debugger *debugger);

  virtual ~BasicTelemeter() = default;

  void LogStartup(DebuggerTelemetryInfo *entry) override;
  void LogExit(DebuggerTelemetryInfo *entry) override;
  void LogMainExecutableLoadStart(TargetTelemetryInfo *entry) override;
  void LogMainExecutableLoadEnd(TargetTelemetryInfo *entry) override;
  void LogProcessExit(TargetTelemetryInfo *entry) override;
  void LogCommandStart(CommandTelemetryInfo *entry) override;
  void LogCommandEnd(CommandTelemetryInfo *entry) override;
  void
  LogClientTelemetry(const lldb_private::StructuredDataImpl &entry) override;

  void addDestination(std::unique_ptr<Destination> destination) override {
    m_destinations.push_back(std::move(destination));
  }

  std::string GetNextUUID() override {
    return std::to_string(uuid_seed.fetch_add(1));
  }

  llvm::Error dispatch(TelemetryInfo *entry) override;

protected:
  BasicTelemeter(std::unique_ptr<llvm::telemetry::Config> config,
                 Debugger *debugger);

  void CollectMiscBuildInfo();

private:
  template <typename EntrySubType> EntrySubType MakeBaseEntry() {
    EntrySubType entry;
    entry.SessionId = m_session_uuid;
    return entry;
  }

  std::unique_ptr<llvm::telemetry::Config> m_config;
  Debugger *m_debugger;
  const std::string m_session_uuid;
  std::string startup_lldb_path;

  // counting number of entries.
  std::atomic<size_t> counter = 0;

  std::vector<std::unique_ptr<Destination>> m_destinations;

  std::atomic<size_t> uuid_seed = 0;
};

class BasicSerializer : public Serializer {
public:
  const std::string &getString() { return Buffer; }

  llvm::Error start() override {
    if (started)
      return llvm::createStringError("Serializer already in use");
    started = true;
    Buffer.clear();
    return Error::success();
  }

  void writeBool(StringRef KeyName, bool Value) override {
    writeHelper(KeyName, Value);
  }

  void writeInt32(StringRef KeyName, int Value) override {
    writeHelper(KeyName, Value);
  }

  void writeSizeT(StringRef KeyName, size_t Value) override {
    writeHelper(KeyName, Value);
  }
  void writeString(StringRef KeyName, StringRef Value) override {
    assert(started && "serializer not started");
  }

  void
  writeKeyValueMap(StringRef KeyName,
                   const std::map<std::string, std::string> &Value) override {
    std::string Inner;
    for (auto kv : Value) {
      writeHelper(StringRef(kv.first), StringRef(kv.second), &Inner);
    }
    writeHelper(KeyName, StringRef(Inner));
  }

  llvm::Error finish() override {
    if (!started)
      return llvm::createStringError("Serializer not currently in use");
    started = false;
    return Error::success();
  }

private:
  template <typename T>
  void writeHelper(StringRef Name, T Value, std::string *Buff) {
    assert(started && "serializer not started");
    Buff->append((Name + ":" + llvm::Twine(Value) + "\n").str());
  }

  template <typename T> void writeHelper(StringRef Name, T Value) {
    writeHelper(Name, Value, &Buffer);
  }

  bool started = false;
  std::string Buffer;
};

class StreamTelemetryDestination : public Destination {
public:
  StreamTelemetryDestination(llvm::raw_ostream &os) : os(os) {}
  llvm::Error
  receiveEntry(const llvm::telemetry::TelemetryInfo *entry) override {
    // Upstream Telemetry should not leak anything other than the
    // basic data, unless running in test mode.
#ifdef TEST_TELEMETRY
    if (Error err = serializer.start()) {
      return err;
    }
    entry->serialize(serializer);
    if (Error err = serializer.finish()) {
      return err;
    }
    os << serializer.getString() << "\n";
#else
    os << "session_uuid: " << entry->SessionId
       << "<the rest is omitted due to PII risk>\n";
#endif
    os.flush();
    return llvm::ErrorSuccess();
  }

  llvm::StringLiteral name() const override { return "StreamDestination"; }

private:
  llvm::raw_ostream &os;
  BasicSerializer serializer;
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

BasicTelemeter::BasicTelemeter(std::unique_ptr<llvm::telemetry::Config> config,
                               lldb_private::Debugger *debugger)
    : m_config(std::move(config)), m_debugger(debugger),
      m_session_uuid(MakeUUID(debugger)) {}

std::unique_ptr<BasicTelemeter>
BasicTelemeter::CreateInstance(std::unique_ptr<llvm::telemetry::Config> config,
                               lldb_private::Debugger *debugger) {

  BasicTelemeter *ins = new BasicTelemeter(std::move(config), debugger);

  return std::unique_ptr<BasicTelemeter>(ins);
}

llvm::Error BasicTelemeter::dispatch(TelemetryInfo *entry) {
  entry->SessionId = m_session_uuid;

  for (auto &destination : m_destinations) {
    llvm::Error err = destination->receiveEntry(entry);
    if (err) {
      return std::move(err);
    }
  }
  return Error::success();
}

void BasicTelemeter::LogStartup(DebuggerTelemetryInfo *entry) {
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
    MiscTelemetryInfo misc_info = MakeBaseEntry<MiscTelemetryInfo>();
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

void BasicTelemeter::LogExit(DebuggerTelemetryInfo *entry) {
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

void BasicTelemeter::LogProcessExit(TargetTelemetryInfo *entry) {
  entry->target_uuid =
      entry->target_ptr && !entry->target_ptr->IsDummyTarget()
          ? entry->target_ptr->GetExecutableModule()->GetUUID().GetAsString()
          : "";

  dispatch(entry);
}

void BasicTelemeter::CollectMiscBuildInfo() {
  // collecting use-case specific data
}

void BasicTelemeter::LogMainExecutableLoadStart(TargetTelemetryInfo *entry) {
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

void BasicTelemeter::LogMainExecutableLoadEnd(TargetTelemetryInfo *entry) {
  lldb::ModuleSP exec_mod = entry->exec_mod;
  entry->binary_path =
      exec_mod->GetFileSpec().GetPathAsConstString().GetCString();
  entry->file_format = exec_mod->GetArchitecture().GetArchitectureName();
  entry->target_uuid = exec_mod->GetUUID().GetAsString();
  entry->binary_size = exec_mod->GetObjectFile()->GetByteSize();

  dispatch(entry);

  // Collect some more info, might be useful?
  MiscTelemetryInfo misc_info = MakeBaseEntry<MiscTelemetryInfo>();
  misc_info.target_uuid = exec_mod->GetUUID().GetAsString();
  misc_info.meta_data["symtab_index_time"] =
      std::to_string(exec_mod->GetSymtabIndexTime().get().count());
  misc_info.meta_data["symtab_parse_time"] =
      std::to_string(exec_mod->GetSymtabParseTime().get().count());
  dispatch(&misc_info);
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

  dispatch(&client_info);
}

void BasicTelemeter::LogCommandStart(CommandTelemetryInfo *entry) {
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

void BasicTelemeter::LogCommandEnd(CommandTelemetryInfo *entry) {
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

} // namespace

TelemetryVendor *TelemetryVendor::FindPlugin() {
  // The default implementation (ie., upstream impl) returns
  // the basic instance.
  //
  // Vendors can provide their plugins as needed.

  std::unique_ptr<TelemetryVendor> instance_up;
  TelemetryVendorCreateInstance create_callback;

  for (size_t idx = 0;
       (create_callback =
            PluginManager::GetTelemetryVendorCreateCallbackAtIndex(idx)) !=
       nullptr;
       ++idx) {
    instance_up.reset(create_callback());

    if (instance_up) {
      return instance_up.release();
    }
  }

  return new TelemetryVendor();
}

llvm::StringRef TelemetryVendor::GetPluginName() {
  return "DefaultTelemetryVendor";
}

static llvm::StringRef ParseValue(llvm::StringRef str, llvm::StringRef label) {
  return str.substr(label.size()).trim();
}

static bool ParseBoolValue(llvm::StringRef str, llvm::StringRef label) {
  if (ParseValue(str, label) == "true")
    return true;
  return false;
}

std::unique_ptr<llvm::telemetry::Config> TelemetryVendor::GetTelemetryConfig() {
  // Telemetry is disabled by default.
  bool enable_telemetry = false;

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
          enable_telemetry = ParseBoolValue(*iter, "enable_telemetry:");
        }
      }
    } else {
      LLDB_LOG(GetLog(LLDBLog::Object), "Error reading config file at {0}",
               init_file.c_str());
    }
  }

// Enable Telemetry in upstream config only if we are running tests.
#ifdef TEST_TELEMETRY
  enable_telemetry = true;
#endif

  auto config = std::make_unique<llvm::telemetry::Config>(enable_telemetry);

  // Now apply any additional vendor config, if available.
  // TODO: cache the Config? (given it's not going to change after LLDB starts
  // up) However, it's possible we want to supporting restarting the Telemeter
  // with new config?
  return GetVendorSpecificConfig(std::move(config));
}

std::unique_ptr<llvm::telemetry::Config>
TelemetryVendor::GetVendorSpecificConfig(
    std::unique_ptr<llvm::telemetry::Config> default_config) {
  return std::move(default_config);
}

std::unique_ptr<LldbTelemeter>
TelemetryVendor::CreateTelemeter(Debugger *debugger) {
  auto config = GetTelemetryConfig();

  if (!config->EnableTelemetry) {
    return NoOpTelemeter::CreateInstance(debugger);
  }

  return BasicTelemeter::CreateInstance(std::move(config), debugger);
}
