//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/offsetof-macro.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid file include"
#endif

#include "src/setjmp/x86_64/common.h"

#if LIBC_COPT_SETJMP_FORTIFICATION
#include "src/setjmp/checksum.h"
#endif

namespace LIBC_NAMESPACE_DECL {
[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  asm volatile(
      // clang-format off
    LOAD_BASE()
    LOAD_CHKSUM_STATE_REGS()      
    STORE_ALL_REGS(STORE_REG_ACCUMULATE)
    STORE_STACK()
    ACCUMULATE_CHECKSUM()
    STORE_PC()
    ACCUMULATE_CHECKSUM()
    STORE_CHECKSUM()
    XOR(RET_REG, RET_REG)
    RETURN()
      // clang-format on
      :
#if LIBC_COPT_SETJMP_FORTIFICATION
      [value_mask] "=m"(jmpbuf::value_mask)
#endif
      : DECLARE_ALL_REGS(DECLARE_OFFSET)
#if LIBC_COPT_SETJMP_FORTIFICATION
        // clang-format off
      ,[rotation] "i"(jmpbuf::ROTATION)
      ,[__chksum] "i"(offsetof(__jmp_buf, __chksum))
      ,[checksum_cookie] "m"(jmpbuf::checksum_cookie)

#endif
      : STR(RET_REG), STR(BASE_REG), STR(MUL_REG));
}

} // namespace LIBC_NAMESPACE_DECL
