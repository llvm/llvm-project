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
  virtual bool AppliesToThread(llvm::StringRef class_name,
                               lldb::ThreadSP thread_sp) {
    return true;
  }

  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name,
                     lldb::StackFrameListSP input_frames,
                     StructuredData::DictionarySP args_sp) = 0;

  /// Get a description string for the frame provider.
  ///
  /// This is called by the descriptor to fetch a description from the
  /// scripted implementation. Implementations should call a static method
  /// on the scripting class to retrieve the description.
  ///
  /// \param class_name The name of the scripting class implementing the
  /// provider.
  ///
  /// \return A string describing what this frame provider does, or an
  ///         empty string if no description is available.
  virtual std::string GetDescription(llvm::StringRef class_name) { return {}; }

  virtual StructuredData::ObjectSP GetFrameAtIndex(uint32_t index) {
    return {};
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDFRAMEPROVIDERINTERFACE_H
