//===-- GetProcessList.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_POSIX_GETPROCESSLIST_H
#define LLDB_HOST_POSIX_GETPROCESSLIST_H

#include "llvm/Support/Error.h"

#include <sys/sysctl.h>
#include <vector>

namespace lldb_private {

/// Fetches the list of all processes into \p kinfos using the BSD/Darwin
/// sysctl(KERN_PROC_ALL) interface. Note that this is not technically a POSIX
/// interface but used in several UNIX-like systems.
llvm::Error GetProcessList(std::vector<struct kinfo_proc> &kinfos);

} // namespace lldb_private

#endif // LLDB_HOST_POSIX_GETPROCESSLIST_H
