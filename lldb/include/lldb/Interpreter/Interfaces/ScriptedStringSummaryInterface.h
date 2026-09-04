//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTRINGSUMMARYINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTRINGSUMMARYINTERFACE_H

#include "ScriptedInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {
class ScriptedStringSummaryInterface : virtual public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name) = 0;

  virtual llvm::Expected<std::string>
  GetSummary(ValueObject &valobj, const TypeSummaryOptions &options) {
    return llvm::createStringError("not implemented");
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTRINGSUMMARYINTERFACE_H
