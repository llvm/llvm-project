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
#include "lldb/lldb-private.h"

#include <string>

namespace lldb_private {
class ScriptedInterface {
public:
  ScriptedInterface() = default;
  virtual ~ScriptedInterface() = default;

  virtual StructuredData::GenericSP
  CreatePluginObject(llvm::StringRef class_name, ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp) = 0;

protected:
  StructuredData::GenericSP m_object_instance_sp;
};
} // namespace lldb_private
#endif // LLDB_INTERPRETER_SCRIPTEDINTERFACE_H
