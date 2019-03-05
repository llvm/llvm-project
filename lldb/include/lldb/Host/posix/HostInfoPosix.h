//===-- HostInfoPosix.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_posix_HostInfoPosix_h_
#define lldb_Host_posix_HostInfoPosix_h_

#include "lldb/Host/HostInfoBase.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private {

class UserIDResolver;

class HostInfoPosix : public HostInfoBase {
  friend class HostInfoBase;

public:
  static size_t GetPageSize();
  static bool GetHostname(std::string &s);

  static uint32_t GetUserID();
  static uint32_t GetGroupID();
  static uint32_t GetEffectiveUserID();
  static uint32_t GetEffectiveGroupID();

  static FileSpec GetDefaultShell();

  static bool GetEnvironmentVar(const std::string &var_name, std::string &var);

  static bool ComputePathRelativeToLibrary(FileSpec &file_spec,
                                           llvm::StringRef dir);

  static UserIDResolver &GetUserIDResolver();

protected:
  static bool ComputeSupportExeDirectory(FileSpec &file_spec);
  static bool ComputeHeaderDirectory(FileSpec &file_spec);
};
}

#endif
