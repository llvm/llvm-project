//===-- Host.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_LINUX_HOST_H
#define LLDB_HOST_LINUX_HOST_H

#include "lldb/lldb-types.h"
#include "llvm/ADT/Optional.h"

namespace lldb_private {

// Get PID (i.e. the primary thread ID) corresponding to the specified TID.
llvm::Optional<lldb::pid_t> getPIDForTID(lldb::pid_t tid);

} // namespace lldb_private

#endif // #ifndef LLDB_HOST_LINUX_HOST_H
