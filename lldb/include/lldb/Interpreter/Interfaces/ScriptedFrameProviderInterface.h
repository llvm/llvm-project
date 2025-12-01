//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDFRAMEPROVIDERINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDFRAMEPROVIDERINTERFACE_H

#include "lldb/lldb-private.h"

#include "ScriptedInterface.h"

namespace lldb_private {
class ScriptedFrameProviderInterface : public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name,
                     lldb::StackFrameListSP input_frames,
                     StructuredData::DictionarySP args_sp) = 0;

  virtual StructuredData::ObjectSP GetFrameAtIndex(uint32_t index) {
    return {};
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDFRAMEPROVIDERINTERFACE_H
