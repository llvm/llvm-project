//===-- ScriptedInterface.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_SCRIPTEDINTERFACE_H
#define LLDB_INTERPRETER_SCRIPTEDINTERFACE_H

#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/lldb-private.h"

#include "llvm/Support/Compiler.h"

#include <string>

namespace lldb_private {
class ScriptedInterface {
public:
  ScriptedInterface() = default;
  virtual ~ScriptedInterface() = default;

  virtual StructuredData::GenericSP
  CreatePluginObject(llvm::StringRef class_name, ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) = 0;

  StructuredData::GenericSP GetScriptObjectInstance() {
    return m_object_instance_sp;
  }

  template <typename Ret>
  static Ret ErrorWithMessage(llvm::StringRef caller_name,
                              llvm::StringRef error_msg, Status &error,
                              LLDBLog log_caterogy = LLDBLog::Process) {
    LLDB_LOGF(GetLog(log_caterogy), "%s ERROR = %s", caller_name.data(),
              error_msg.data());
    error.SetErrorString(llvm::Twine(caller_name + llvm::Twine(" ERROR = ") +
                                     llvm::Twine(error_msg))
                             .str());
    return {};
  }

  template <typename T = StructuredData::ObjectSP>
  bool CheckStructuredDataObject(llvm::StringRef caller, T obj, Status &error) {
    if (!obj) {
      return ErrorWithMessage<bool>(caller,
                                    llvm::Twine("Null StructuredData object (" +
                                                llvm::Twine(error.AsCString()) +
                                                llvm::Twine(")."))
                                        .str(),
                                    error);
    }

    if (!obj->IsValid()) {
      return ErrorWithMessage<bool>(
          caller,
          llvm::Twine("Invalid StructuredData object (" +
                      llvm::Twine(error.AsCString()) + llvm::Twine(")."))
              .str(),
          error);
    }

    if (error.Fail())
      return ErrorWithMessage<bool>(caller, error.AsCString(), error);

    return true;
  }

protected:
  StructuredData::GenericSP m_object_instance_sp;
};
} // namespace lldb_private
#endif // LLDB_INTERPRETER_SCRIPTEDINTERFACE_H
