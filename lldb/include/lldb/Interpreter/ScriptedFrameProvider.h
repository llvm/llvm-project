//===-- ScriptedFrameProvider.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_SCRIPTEDFRAMEPROVIDER_H
#define LLDB_INTERPRETER_SCRIPTEDFRAMEPROVIDER_H

#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"

namespace lldb_private {

class ScriptedFrameProvider {
public:
  /// Constructor that initializes the scripted frame provider.
  ///
  /// \param[in] thread_sp
  ///     The thread for which to provide scripted frames.
  ///
  /// \param[in] scripted_metadata
  ///     The metadata containing the class name and arguments for the
  ///     scripted frame provider.
  ///
  /// \param[out] error
  ///     Status object to report any errors during initialization.
  ScriptedFrameProvider(lldb::ThreadSP thread_sp,
                        const ScriptedMetadata &scripted_metadata,
                        Status &error);
  ~ScriptedFrameProvider();

  /// Get the stack frames from the scripted frame provider.
  ///
  /// \return
  ///     An Expected containing the StackFrameListSP if successful,
  ///     otherwise an error describing what went wrong.
  llvm::Expected<lldb::StackFrameListSP>
  GetStackFrames(lldb::StackFrameListSP real_frames);

private:
  lldb::ThreadSP m_thread_sp;
  lldb::ScriptedFrameProviderInterfaceSP m_interface_sp;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_SCRIPTEDFRAMEPROVIDER_H
