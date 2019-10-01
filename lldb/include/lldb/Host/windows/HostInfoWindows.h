//===-- HostInfoWindows.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_windows_HostInfoWindows_h_
#define lldb_Host_windows_HostInfoWindows_h_

#include "lldb/Host/HostInfoBase.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/Support/VersionTuple.h"

namespace lldb_private {
class UserIDResolver;

class HostInfoWindows : public HostInfoBase {
  friend class HostInfoBase;

private:
  // Static class, unconstructable.
  HostInfoWindows();
  ~HostInfoWindows();

public:
  static void Initialize();
  static void Terminate();

  static size_t GetPageSize();
  static UserIDResolver &GetUserIDResolver();

  static llvm::VersionTuple GetOSVersion();
  static bool GetOSBuildString(std::string &s);
  static bool GetOSKernelDescription(std::string &s);
  static bool GetHostname(std::string &s);
  static FileSpec GetProgramFileSpec();
  static FileSpec GetDefaultShell();

  static bool GetEnvironmentVar(const std::string &var_name, std::string &var);

private:
  static FileSpec m_program_filespec;
};
}

#endif
