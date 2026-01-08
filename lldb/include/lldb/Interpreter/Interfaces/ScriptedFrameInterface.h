//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDFRAMEINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDFRAMEINTERFACE_H

#include "ScriptedInterface.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/lldb-private.h"
#include <optional>
#include <string>

namespace lldb_private {
class ScriptedFrameInterface : virtual public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) = 0;

  virtual lldb::user_id_t GetID() { return LLDB_INVALID_FRAME_ID; }

  virtual lldb::addr_t GetPC() { return LLDB_INVALID_ADDRESS; }

  virtual std::optional<SymbolContext> GetSymbolContext() {
    return std::nullopt;
  }

  virtual std::optional<std::string> GetFunctionName() { return std::nullopt; }

  virtual std::optional<std::string> GetDisplayFunctionName() {
    return std::nullopt;
  }

  virtual bool IsInlined() { return false; }

  virtual bool IsArtificial() { return false; }

  virtual bool IsHidden() { return false; }

  virtual StructuredData::DictionarySP GetRegisterInfo() { return {}; }

  virtual std::optional<std::string> GetRegisterContext() {
    return std::nullopt;
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDFRAMEINTERFACE_H
