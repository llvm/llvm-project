//===-- ScriptedProcessInterface.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_SCRIPTEDPROCESSINTERFACE_H
#define LLDB_INTERPRETER_SCRIPTEDPROCESSINTERFACE_H

#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Interpreter/ScriptedInterface.h"
#include "lldb/Target/MemoryRegionInfo.h"

#include "lldb/lldb-private.h"

#include <optional>
#include <string>

namespace lldb_private {
class ScriptedProcessInterface : virtual public ScriptedInterface {
public:
  StructuredData::GenericSP
  CreatePluginObject(llvm::StringRef class_name, ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) override {
    return {};
  }

  virtual StructuredData::DictionarySP GetCapabilities() { return {}; }

  virtual Status Attach(const ProcessAttachInfo &attach_info) {
    return Status("ScriptedProcess did not attach");
  }

  virtual Status Launch() { return Status("ScriptedProcess did not launch"); }

  virtual Status Resume() { return Status("ScriptedProcess did not resume"); }

  virtual std::optional<MemoryRegionInfo>
  GetMemoryRegionContainingAddress(lldb::addr_t address, Status &error) {
    error.SetErrorString("ScriptedProcess have no memory region.");
    return {};
  }

  virtual StructuredData::DictionarySP GetThreadsInfo() { return {}; }

  virtual bool CreateBreakpoint(lldb::addr_t addr, Status &error) {
    error.SetErrorString("ScriptedProcess don't support creating breakpoints.");
    return {};
  }

  virtual lldb::DataExtractorSP
  ReadMemoryAtAddress(lldb::addr_t address, size_t size, Status &error) {
    return {};
  }

  virtual lldb::offset_t WriteMemoryAtAddress(lldb::addr_t addr,
                                              lldb::DataExtractorSP data_sp,
                                              Status &error) {
    return LLDB_INVALID_OFFSET;
  };

  virtual StructuredData::ArraySP GetLoadedImages() { return {}; }

  virtual lldb::pid_t GetProcessID() { return LLDB_INVALID_PROCESS_ID; }

  virtual bool IsAlive() { return true; }

  virtual std::optional<std::string> GetScriptedThreadPluginName() {
    return std::nullopt;
  }

  virtual StructuredData::DictionarySP GetMetadata() { return {}; }

protected:
  friend class ScriptedThread;
  virtual lldb::ScriptedThreadInterfaceSP CreateScriptedThreadInterface() {
    return {};
  }
};

class ScriptedThreadInterface : virtual public ScriptedInterface {
public:
  StructuredData::GenericSP
  CreatePluginObject(llvm::StringRef class_name, ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) override {
    return {};
  }

  virtual lldb::tid_t GetThreadID() { return LLDB_INVALID_THREAD_ID; }

  virtual std::optional<std::string> GetName() { return std::nullopt; }

  virtual lldb::StateType GetState() { return lldb::eStateInvalid; }

  virtual std::optional<std::string> GetQueue() { return std::nullopt; }

  virtual StructuredData::DictionarySP GetStopReason() { return {}; }

  virtual StructuredData::ArraySP GetStackFrames() { return {}; }

  virtual StructuredData::DictionarySP GetRegisterInfo() { return {}; }

  virtual std::optional<std::string> GetRegisterContext() {
    return std::nullopt;
  }

  virtual StructuredData::ArraySP GetExtendedInfo() { return {}; }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_SCRIPTEDPROCESSINTERFACE_H
