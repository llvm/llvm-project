//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTACKFRAMERECOGNIZERINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTACKFRAMERECOGNIZERINTERFACE_H

#include "ScriptedInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {
class ScriptedStackFrameRecognizerInterface : virtual public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(const ScriptedMetadata &scripted_metadata) = 0;

  virtual lldb::ValueObjectListSP
  GetRecognizedArguments(lldb::StackFrameSP frame_sp) {
    return lldb::ValueObjectListSP();
  }

  virtual bool ShouldHide(lldb::StackFrameSP frame_sp) { return false; }

  virtual lldb::StackFrameSP
  SelectMostRelevantFrame(lldb::StackFrameSP frame_sp) {
    return nullptr;
  }

  virtual lldb::ValueObjectSP GetException(lldb::StackFrameSP frame_sp) {
    return lldb::ValueObjectSP();
  }

  virtual std::string GetStopDescription(lldb::StackFrameSP frame_sp) {
    return "";
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTACKFRAMERECOGNIZERINTERFACE_H
