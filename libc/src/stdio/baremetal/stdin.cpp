//===--- Definition of baremetal stdin ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/stdin.h"

#include "hdr/types/FILE.h"
#include "src/__support/OSUtil/baremetal/io.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" struct __llvm_libc_stdio_cookie __llvm_libc_stdin_cookie;

LLVM_LIBC_VARIABLE(FILE *,
                   stdin) = reinterpret_cast<FILE *>(&__llvm_libc_stdin_cookie);

} // namespace LIBC_NAMESPACE_DECL
