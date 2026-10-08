//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDSYNTHETICCHILDRENINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDSYNTHETICCHILDRENINTERFACE_H

#include "ScriptedInterface.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/ErrorExtras.h"

namespace lldb_private {
class ScriptedSyntheticChildrenInterface : virtual public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, ValueObject &backend) = 0;

  virtual llvm::Expected<uint32_t> CalculateNumChildren(uint32_t max) {
    return 0;
  }

  virtual lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) {
    return lldb::ValueObjectSP();
  }

  virtual llvm::Expected<uint32_t> GetIndexOfChildWithName(ConstString name) {
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  }

  virtual lldb::ChildCacheState Update() { return lldb::eRefetch; }

  virtual bool MightHaveChildren() { return true; }

  virtual lldb::ValueObjectSP GetSyntheticValue() { return nullptr; }

  virtual ConstString GetSyntheticTypeName() { return ConstString(); }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDSYNTHETICCHILDRENINTERFACE_H
