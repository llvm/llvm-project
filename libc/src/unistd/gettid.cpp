//===-- Implementation file for gettid --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/gettid.h"
#include "src/__support/common.h"
#include "src/__support/threads/identifier.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(pid_t, gettid, ()) { return internal::gettid(); }

} // namespace LIBC_NAMESPACE_DECL
