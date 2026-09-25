//===-- ScriptedInterface.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDINTERFACE_H

#include "ScriptedInterfaceUsages.h"
#include "lldb/Interpreter/ScriptedInstanceRegistry.h"

#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/Utility/UnimplementedError.h"
#include "lldb/lldb-private.h"

#include "llvm/Support/Compiler.h"

#include <optional>
#include <string>

namespace lldb_private {
class ScriptedInterface {
public:
  ScriptedInterface() = default;
  virtual ~ScriptedInterface() { UnregisterInstance(); }

  // Copies would share the registry entry, and the first one destroyed would
  // drop it while the other is still alive.
  ScriptedInterface(const ScriptedInterface &) = delete;
  ScriptedInterface &operator=(const ScriptedInterface &) = delete;

  StructuredData::GenericSP GetScriptObjectInstance() {
    return m_object_instance_sp;
  }

  const std::optional<ScriptedMetadata> &GetScriptedMetadata() const {
    return m_scripted_metadata;
  }

  virtual llvm::StringRef GetPluginName() = 0;

  /// Whether the user can invoke this extension directly, the way a scripted
  /// command can. Those never introduce the target's API mutex bypass, so at
  /// top level they serialize like any other command; nested inside an
  /// already-bypassed callback every extension inherits the ambient policy.
  virtual bool UserCanRunDirectly() const { return false; }

  struct AbstractMethodRequirement {
    llvm::StringLiteral name;
    size_t min_arg_count = 0;
  };

  virtual llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const = 0;

  /// Methods a script may legitimately leave out, for which LLDB has a
  /// documented answer.
  ///
  /// This is the counterpart of GetAbstractMethodRequirements(): a method is
  /// either required, and its absence rejects the class outright, or listed
  /// here, and its absence is an expected answer. Only methods named here may
  /// be dispatched with ScriptedPythonInterface::DispatchToOptional().
  virtual llvm::SmallVector<llvm::StringLiteral> GetOptionalMethods() const {
    return {};
  }

  virtual llvm::Expected<FileSpec> GetScriptedModulePath() {
    return llvm::make_error<UnimplementedError>();
  }

  llvm::SmallVector<llvm::StringLiteral> const GetAbstractMethods() const {
    llvm::SmallVector<llvm::StringLiteral> abstract_methods;
    llvm::transform(GetAbstractMethodRequirements(), abstract_methods.begin(),
                    [](const AbstractMethodRequirement &requirement) {
                      return requirement.name;
                    });
    return abstract_methods;
  }

  template <typename Ret>
  static Ret ErrorWithMessage(llvm::StringRef caller_name,
                              llvm::StringRef user_msg, Status &error,
                              LLDBLog log_category = LLDBLog::Process) {
    LLDB_LOGF(GetLog(log_category), "%s ERROR = %s", caller_name.data(),
              user_msg.data());

    // If `error` already has detailed content (e.g. a Python traceback),
    // prepend this call's friendlier message to it instead of discarding
    // either one.
    std::string existing_error = error.Fail() ? error.AsCString() : "";
    if (existing_error.empty())
      error = Status::FromErrorString(user_msg.data());
    else
      error = Status::FromErrorStringWithFormatv("{0}: {1}", user_msg,
                                                 existing_error);

    return {};
  }

  static bool CreateInstance(lldb::ScriptLanguage language,
                             ScriptedInterfaceUsages usages) {
    return false;
  }

protected:
  void RegisterInstance(const lldb::ScriptedInstanceRegistrySP &registry_sp,
                        llvm::StringRef class_name) {
    UnregisterInstance();
    if (!registry_sp)
      return;
    ScriptedInstanceInfo info;
    info.plugin_name = GetPluginName();
    info.class_name = class_name.str();
    if (m_object_instance_sp)
      info.object_address =
          reinterpret_cast<uintptr_t>(m_object_instance_sp->GetValue());
    if (m_scripted_metadata) {
      info.source_path = m_scripted_metadata->GetSourcePath();
      info.args_sp = m_scripted_metadata->GetArgsSP();
    }
    m_registry_id = registry_sp->Add(std::move(info));
    m_registry_wp = registry_sp;
  }

  StructuredData::GenericSP m_object_instance_sp;
  std::optional<ScriptedMetadata> m_scripted_metadata;

private:
  void UnregisterInstance() {
    // The interpreter owning the registry may already be gone.
    if (auto registry_sp = m_registry_wp.lock())
      registry_sp->Remove(m_registry_id);
    m_registry_wp.reset();
  }

  lldb::ScriptedInstanceRegistryWP m_registry_wp;
  uint64_t m_registry_id = 0;
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDINTERFACE_H
