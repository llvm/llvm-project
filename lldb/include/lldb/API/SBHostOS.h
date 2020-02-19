//===-- SBHostOS.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBHOSTOS_H
#define LLDB_API_SBHOSTOS_H

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBFileSpec.h"

namespace lldb {

class LLDB_API SBHostOS {
public:
  static lldb::SBFileSpec GetProgramFileSpec();

  static lldb::SBFileSpec GetLLDBPythonPath();

  static lldb::SBFileSpec GetLLDBPath(lldb::PathType path_type);

  static lldb::SBFileSpec GetUserHomeDirectory();

  static void ThreadCreated(const char *name);

  static lldb::thread_t ThreadCreate(const char *name,
                                     lldb::thread_func_t thread_function,
                                     void *thread_arg, lldb::SBError *err);

  static bool ThreadCancel(lldb::thread_t thread, lldb::SBError *err);

  static bool ThreadDetach(lldb::thread_t thread, lldb::SBError *err);
  static bool ThreadJoin(lldb::thread_t thread, lldb::thread_result_t *result,
                         lldb::SBError *err);

private:
};

} // namespace lldb

#endif // LLDB_API_SBHOSTOS_H
