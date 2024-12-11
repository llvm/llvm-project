//===-- TelemetryVendor.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/TelemetryVendor.h"

#include <atomic>
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
class NoOpTelemeter : public TelemetryManager {
public:
  static std::unique_ptr<TelemetryManager> CreateInstance(Debugger *debugger) {
    return std::unique_ptr<TelemetryManager>(new NoOpTelemeter(debugger));
  }

  NoOpTelemeter(Debugger *debugger) {}
  NoOpTelemeter() = default;
  ~NoOpTelemeter() = default;

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

std::unique_ptr<TelemetryManager>
TelemetryVendor::CreateTelemetryManager(Debugger *debugger) {
  auto config = GetTelemetryConfig();

  if (!config->EnableTelemetry) {
    return NoOpTelemeter::CreateInstance(debugger);
  }

  return TelemetryManager::CreateInstance(std::move(config), debugger);
}
