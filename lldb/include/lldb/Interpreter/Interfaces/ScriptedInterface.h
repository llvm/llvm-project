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

#include <functional>
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

  /// Set error callback to surface Python exceptions directly to users.
  ///
  /// This allows command handlers to receive Python exception details
  /// immediately rather than relying on diagnostic broadcasts.
  ///
  /// \param callback Function to call with Status containing exception details.
  virtual void SetErrorCallback(std::function<void(const Status &)> callback) {}

  /// Clear the error callback.
  virtual void ClearErrorCallback() {}

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
                              llvm::StringRef error_msg, Status &error,
                              LLDBLog log_category = LLDBLog::Process) {
    // Log the error for debugging (includes function signature for context).
    LLDB_LOGF(GetLog(log_category), "%s ERROR = %s", caller_name.data(),
              error_msg.data());

    // For user-facing messages, just pass through the Status if it already
    // has detailed information (like Python tracebacks); otherwise set it.
    llvm::StringRef existing_error = error.AsCString();
    if (!error.Fail() || existing_error.empty()) {
      // Status is empty, populate it with the simple error message.
      error = Status::FromErrorString(error_msg.data());
    }
    // If Status already has content, leave it as-is (it has the Python
    // traceback).

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
