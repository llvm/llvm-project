//===-- SBHostOS.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeThread.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Utility/FileSpec.h"

#include "Plugins/ExpressionParser/Clang/ClangHost.h"
#ifndef LLDB_DISABLE_PYTHON
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#endif

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

using namespace lldb;
using namespace lldb_private;

SBFileSpec SBHostOS::GetProgramFileSpec() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(lldb::SBFileSpec, SBHostOS,
                                    GetProgramFileSpec);

  SBFileSpec sb_filespec;
  sb_filespec.SetFileSpec(HostInfo::GetProgramFileSpec());
  return LLDB_RECORD_RESULT(sb_filespec);
}

SBFileSpec SBHostOS::GetLLDBPythonPath() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(lldb::SBFileSpec, SBHostOS,
                                    GetLLDBPythonPath);

  return LLDB_RECORD_RESULT(GetLLDBPath(ePathTypePythonDir));
}

SBFileSpec SBHostOS::GetLLDBPath(lldb::PathType path_type) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBFileSpec, SBHostOS, GetLLDBPath,
                            (lldb::PathType), path_type);

  FileSpec fspec;
  switch (path_type) {
  case ePathTypeLLDBShlibDir:
    fspec = HostInfo::GetShlibDir();
    break;
  case ePathTypeSupportExecutableDir:
    fspec = HostInfo::GetSupportExeDir();
    break;
  case ePathTypeHeaderDir:
    fspec = HostInfo::GetHeaderDir();
    break;
  case ePathTypePythonDir:
#ifndef LLDB_DISABLE_PYTHON
    fspec = ScriptInterpreterPython::GetPythonDir();
#endif
    break;
  case ePathTypeLLDBSystemPlugins:
    fspec = HostInfo::GetSystemPluginDir();
    break;
  case ePathTypeLLDBUserPlugins:
    fspec = HostInfo::GetUserPluginDir();
    break;
  case ePathTypeLLDBTempSystemDir:
    fspec = HostInfo::GetProcessTempDir();
    break;
  case ePathTypeGlobalLLDBTempSystemDir:
    fspec = HostInfo::GetGlobalTempDir();
    break;
  case ePathTypeClangDir:
    fspec = GetClangResourceDir();
    break;
  }

  SBFileSpec sb_fspec;
  sb_fspec.SetFileSpec(fspec);
  return LLDB_RECORD_RESULT(sb_fspec);
}

SBFileSpec SBHostOS::GetUserHomeDirectory() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(lldb::SBFileSpec, SBHostOS,
                                    GetUserHomeDirectory);

  SBFileSpec sb_fspec;

  llvm::SmallString<64> home_dir_path;
  llvm::sys::path::home_directory(home_dir_path);
  FileSpec homedir(home_dir_path.c_str());
  FileSystem::Instance().Resolve(homedir);

  sb_fspec.SetFileSpec(homedir);
  return LLDB_RECORD_RESULT(sb_fspec);
}

lldb::thread_t SBHostOS::ThreadCreate(const char *name,
                                      lldb::thread_func_t thread_function,
                                      void *thread_arg, SBError *error_ptr) {
  LLDB_RECORD_DUMMY(lldb::thread_t, SBHostOS, ThreadCreate,
                    (lldb::thread_func_t, void *, SBError *), name,
                    thread_function, thread_arg, error_ptr);
  HostThread thread(
      ThreadLauncher::LaunchThread(name, thread_function, thread_arg,
                                   error_ptr ? error_ptr->get() : nullptr));
  return thread.Release();
}

void SBHostOS::ThreadCreated(const char *name) {
  LLDB_RECORD_STATIC_METHOD(void, SBHostOS, ThreadCreated, (const char *),
                            name);
}

bool SBHostOS::ThreadCancel(lldb::thread_t thread, SBError *error_ptr) {
  LLDB_RECORD_DUMMY(bool, SBHostOS, ThreadCancel,
                            (lldb::thread_t, lldb::SBError *), thread,
                            error_ptr);

  Status error;
  HostThread host_thread(thread);
  error = host_thread.Cancel();
  if (error_ptr)
    error_ptr->SetError(error);
  host_thread.Release();
  return error.Success();
}

bool SBHostOS::ThreadDetach(lldb::thread_t thread, SBError *error_ptr) {
  LLDB_RECORD_DUMMY(bool, SBHostOS, ThreadDetach,
                            (lldb::thread_t, lldb::SBError *), thread,
                            error_ptr);

  Status error;
#if defined(_WIN32)
  if (error_ptr)
    error_ptr->SetErrorString("ThreadDetach is not supported on this platform");
#else
  HostThread host_thread(thread);
  error = host_thread.GetNativeThread().Detach();
  if (error_ptr)
    error_ptr->SetError(error);
  host_thread.Release();
#endif
  return error.Success();
}

bool SBHostOS::ThreadJoin(lldb::thread_t thread, lldb::thread_result_t *result,
                          SBError *error_ptr) {
  LLDB_RECORD_DUMMY(
      bool, SBHostOS, ThreadJoin,
      (lldb::thread_t, lldb::thread_result_t *, lldb::SBError *), thread,
      result, error_ptr);

  Status error;
  HostThread host_thread(thread);
  error = host_thread.Join(result);
  if (error_ptr)
    error_ptr->SetError(error);
  host_thread.Release();
  return error.Success();
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBHostOS>(Registry &R) {
  LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS, GetProgramFileSpec,
                              ());
  LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS, GetLLDBPythonPath,
                              ());
  LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS, GetLLDBPath,
                              (lldb::PathType));
  LLDB_REGISTER_STATIC_METHOD(lldb::SBFileSpec, SBHostOS,
                              GetUserHomeDirectory, ());
  LLDB_REGISTER_STATIC_METHOD(void, SBHostOS, ThreadCreated, (const char *));
}

}
}
