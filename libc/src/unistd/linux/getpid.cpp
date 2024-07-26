//===-- Linux implementation of getpid ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpid.h"
#include "src/__support/OSUtil/pid.h"
#include "src/__support/common.h"
namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(pid_t, getpid, (void)) { return ProcessIdentity::get(); }

} // namespace LIBC_NAMESPACE_DECL
