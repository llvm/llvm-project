//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid file include"
#endif

#include "src/setjmp/x86_64/common.h"

#if LIBC_COPT_SETJMP_FORTIFICATION
#include "src/setjmp/checksum.h"
#endif

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]] LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf, int)) {
  asm volatile(
      // clang-format off
      LOAD_BASE()
      LOAD_CHKSUM_STATE_REGS() 
      LOAD_ALL_REGS(RESTORE_REG)
      EXAMINE_CHECKSUM()
      CALCULATE_RETURN_VALUE()
      RESTORE_PC()
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
