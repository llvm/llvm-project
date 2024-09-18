//===-- HostInfoQNX.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_qnx_HostInfoQNX_h_
#define lldb_Host_qnx_HostInfoQNX_h_

#include "lldb/Host/posix/HostInfoPosix.h"

namespace lldb_private {

class HostInfoQNX : public HostInfoPosix {
public:
  static llvm::VersionTuple GetOSVersion();
  static std::optional<std::string> GetOSBuildString();
  static FileSpec GetProgramFileSpec();
};
} // namespace lldb_private

#endif
