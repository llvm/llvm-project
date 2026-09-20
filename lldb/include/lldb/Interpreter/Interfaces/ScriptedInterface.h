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
  virtual ~ScriptedInterface() = default;

  StructuredData::GenericSP GetScriptObjectInstance() {
    return m_object_instance_sp;
  }

  const std::optional<ScriptedMetadata> &GetScriptedMetadata() const {
    return m_scripted_metadata;
  }

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

  template <typename T = StructuredData::ObjectSP>
  static bool CheckStructuredDataObject(llvm::StringRef caller, T obj,
                                        Status &error) {
    if (!obj)
      return ErrorWithMessage<bool>(caller, "Null Structured Data object",
                                    error);

    if (!obj->IsValid()) {
      return ErrorWithMessage<bool>(caller, "Invalid StructuredData object",
                                    error);
    }

    if (error.Fail())
      return ErrorWithMessage<bool>(caller, error.AsCString(), error);

    return true;
  }

  static bool CreateInstance(lldb::ScriptLanguage language,
                             ScriptedInterfaceUsages usages) {
    return false;
  }

protected:
  StructuredData::GenericSP m_object_instance_sp;
  std::optional<ScriptedMetadata> m_scripted_metadata;
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDINTERFACE_H
