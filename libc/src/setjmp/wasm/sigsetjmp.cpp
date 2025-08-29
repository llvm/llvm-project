//===-- Implementation of sigsetjmp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/sigsetjmp.h"
#include "hdr/offsetof_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#if !defined(LIBC_TARGET_ARCH_IS_WASM)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE_DECL {
[[gnu::returns_twice]] int sigsetjmp(jmp_buf sigjmp_buf, int savesigs) {
    return setjmp(sigjmp_buf);
}
}
