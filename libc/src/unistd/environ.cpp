//===-- Declaration of POSIX environ --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/environ.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

// This is initialized to the correct value by the statup code.
LLVM_LIBC_GLOBAL(char **, environ) = nullptr;

} // namespace LIBC_NAMESPACE
