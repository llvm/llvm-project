//===-- Declaration of POSIX environ --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// This is initialized to the correct value by the statup code.
extern "C" {
char **environ = nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
