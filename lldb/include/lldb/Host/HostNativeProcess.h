//===-- HostNativeProcess.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostNativeProcess_h_
#define lldb_Host_HostNativeProcess_h_

#if defined(_WIN32)
#include "lldb/Host/windows/HostProcessWindows.h"
namespace lldb_private {
typedef HostProcessWindows HostNativeProcess;
}
#else
#include "lldb/Host/posix/HostProcessPosix.h"
namespace lldb_private {
typedef HostProcessPosix HostNativeProcess;
}
#endif

#endif
