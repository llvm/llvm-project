# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 --defsym direct=0 %s -o %t.o
# RUN: not ld.lld %t.o -o %t 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=aarch64 --defsym direct=1 %s -o %t.direct.o
# RUN: not ld.lld %t.direct.o -o %t.direct 2>&1 | FileCheck %s

# CHECK: error: AUTH GOT entry for non-preemptible ifunc 'ifunc' requested, but R_AARCH64_AUTH_IRELATIVE is not supported yet

.globl ifunc
.type ifunc, @gnu_indirect_function
ifunc:
  ret

.globl _start
.type _start, @function
_start:
  adrp x0, :got_auth:ifunc
  ldr x1, [x0, :got_auth_lo12:ifunc]
.if direct == 1
  adrp x2, ifunc
.endif
