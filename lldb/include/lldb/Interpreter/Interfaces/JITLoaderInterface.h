//===-- JITLoaderInterface.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_JITLOADERINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_JITLOADERINTERFACE_H

#include "ScriptedThreadInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {
class JITLoaderInterface : virtual public ScriptedInterface {
public:

  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, 
                     lldb_private::ExecutionContext &exe_ctx) = 0;

  virtual void DidAttach() {};
  virtual void DidLaunch() {};
  virtual void ModulesDidLoad(lldb_private::ModuleList &module_list) {};

};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_JITLOADERINTERFACE_H
