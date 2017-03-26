//===-- HostInfoOpenBSD.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_openbsd_HostInfoOpenBSD_h_
#define lldb_Host_openbsd_HostInfoOpenBSD_h_

#include "lldb/Host/posix/HostInfoPosix.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private {

class HostInfoOpenBSD : public HostInfoPosix {
public:
  static bool GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update);
  static bool GetOSBuildString(std::string &s);
  static bool GetOSKernelDescription(std::string &s);
  static FileSpec GetProgramFileSpec();
};
}

#endif
