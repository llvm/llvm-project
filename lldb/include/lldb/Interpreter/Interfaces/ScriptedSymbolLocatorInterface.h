//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDSYMBOLLOCATORINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDSYMBOLLOCATORINTERFACE_H

#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Interpreter/Interfaces/ScriptedInterface.h"
#include "lldb/Utility/Status.h"

#include "lldb/lldb-private.h"

#include <optional>
#include <string>

namespace lldb_private {
class ScriptedSymbolLocatorInterface : virtual public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) = 0;

  virtual std::optional<FileSpec>
  LocateSourceFile(const lldb::ModuleSP &module_sp,
                   const FileSpec &original_source_file, Status &error) {
    return {};
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDSYMBOLLOCATORINTERFACE_H
