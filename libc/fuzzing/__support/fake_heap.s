//===-- Test fake definition for heap symbols -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

.globl _end, __llvm_libc_heap_limit

.bss
_end:
.fill 1024
__llvm_libc_heap_limit:

