//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/architectures.h"

#include <setjmp.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, longjmp, (__jmp_buf * buf, int val)) {
#ifdef LIBC_TARGET_IS_X86_64
  register __UINT64_TYPE__ rbx __asm__("rbx");
  register __UINT64_TYPE__ rbp __asm__("rbp");
  register __UINT64_TYPE__ r12 __asm__("r12");
  register __UINT64_TYPE__ r13 __asm__("r13");
  register __UINT64_TYPE__ r14 __asm__("r14");
  register __UINT64_TYPE__ r15 __asm__("r15");
  register __UINT64_TYPE__ rsp __asm__("rsp");
  register __UINT64_TYPE__ rax __asm__("rax");

  // ABI requires that the return value should be stored in rax. So, we store
  // |val| in rax. Note that this has to happen before we restore the registers
  // from values in |buf|. Otherwise, once rsp and rbp are updated, we cannot
  // read |val|.
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(rax) : "m"(val) :);
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(rbx) : "m"(buf->rbx) :);
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(rbp) : "m"(buf->rbp) :);
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(r12) : "m"(buf->r12) :);
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(r13) : "m"(buf->r13) :);
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(r14) : "m"(buf->r14) :);
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(r15) : "m"(buf->r15) :);
  LIBC_INLINE_ASM("mov %1, %0\n\t" : "=r"(rsp) : "m"(buf->rsp) :);
  LIBC_INLINE_ASM("jmp *%0\n\t" : : "m"(buf->rip));
#else // LIBC_TARGET_IS_X86_64
#error "longjmp implementation not available for the target architecture."
#endif
}

} // namespace __llvm_libc
