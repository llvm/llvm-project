//===-- HostInfoAIX.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_AIX_HOSTINFOAIX_H_
#define LLDB_HOST_AIX_HOSTINFOAIX_H_

#include "lldb/Host/posix/HostInfoPosix.h"
#include "lldb/Utility/FileSpec.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VersionTuple.h"

namespace lldb_private {

class HostInfoAIX : public HostInfoPosix {
  friend class HostInfoBase;

public:
  static void Initialize(SharedLibraryDirectoryHelper *helper = nullptr);
  static void Terminate();

  static llvm::StringRef GetDistributionId();
  static FileSpec GetProgramFileSpec();

protected:
  static void ComputeHostArchitectureSupport(ArchSpec &arch_32,
                                             ArchSpec &arch_64);
};
}

#endif
