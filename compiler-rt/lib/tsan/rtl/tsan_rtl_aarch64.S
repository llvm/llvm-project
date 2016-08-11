#include "sanitizer_common/sanitizer_asm.h"

.section .bss
.type	__tsan_pointer_chk_guard, %object
.size	__tsan_pointer_chk_guard, 8
__tsan_pointer_chk_guard:
.zero	8

.section .text

// GLIBC mangles the function pointers in jmp_buf (used in {set,long}*jmp
// functions) by XORing them with a random guard pointer.  For AArch64 it is a
// global variable rather than a TCB one (as for x86_64/powerpc) and althought
// its value is exported by the loader, it lies within a private GLIBC
// namespace (meaning it should be only used by GLIBC itself and the ABI is
// not stable). So InitializeGuardPtr obtains the pointer guard value by
// issuing a setjmp and checking the resulting pointers values against the
// original ones.
.hidden _Z18InitializeGuardPtrv
.global _Z18InitializeGuardPtrv
.type _Z18InitializeGuardPtrv, @function
_Z18InitializeGuardPtrv:
  CFI_STARTPROC
  // Allocates a jmp_buf for the setjmp call.
  stp	x29, x30, [sp, -336]!
  CFI_DEF_CFA_OFFSET (336)
  CFI_OFFSET (29, -336)
  CFI_OFFSET (30, -328)
  add	x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)
  add	x0, x29, 24

  // Call libc setjmp that mangle the stack pointer value
  adrp  x1, :got:_ZN14__interception12real__setjmpE
  ldr   x1, [x1, #:got_lo12:_ZN14__interception12real__setjmpE]
  ldr   x1, [x1]
  blr   x1

  // glibc setjmp mangles both the frame pointer (FP, pc+4 on blr) and the
  // stack pointer (SP). FP will be placed on ((uintptr*)jmp_buf)[11] and
  // SP at ((uintptr*)jmp_buf)[13].
  // The mangle operation is just 'value' xor 'pointer guard value' and
  // if we know the original value (SP) and the expected one, we can derive
  // the guard pointer value.
  mov	x0, sp

  // Loads the mangled SP pointer.
  ldr	x1, [x29, 128]
  eor	x0, x0, x1
  adrp	x2, __tsan_pointer_chk_guard
  str	x0, [x2, #:lo12:__tsan_pointer_chk_guard]
  ldp	x29, x30, [sp], 336
  CFI_RESTORE (30)
  CFI_RESTORE (19)
  CFI_DEF_CFA (31, 0)
  ret
  CFI_ENDPROC
.size _Z18InitializeGuardPtrv, .-_Z18InitializeGuardPtrv

.hidden __tsan_setjmp
.comm _ZN14__interception11real_setjmpE,8,8
.type setjmp, @function
setjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf
  str     x19, [sp, 16]
  CFI_OFFSET (19, -16)
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, __tsan_pointer_chk_guard
  ldr     x2, [x2, #:lo12:__tsan_pointer_chk_guard]
  add     x0, x29, 32
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  // restore env parameter
  mov     x0, x19
  ldr     x19, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (19)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc setjmp
  adrp    x1, :got:_ZN14__interception11real_setjmpE
  ldr     x1, [x1, #:got_lo12:_ZN14__interception11real_setjmpE]
  ldr     x1, [x1]
  br      x1

  CFI_ENDPROC
.size setjmp, .-setjmp

.comm _ZN14__interception12real__setjmpE,8,8
.globl _setjmp
.type _setjmp, @function
_setjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf
  str     x19, [sp, 16]
  CFI_OFFSET (19, -16)
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, __tsan_pointer_chk_guard
  ldr     x2, [x2, #:lo12:__tsan_pointer_chk_guard]
  add     x0, x29, 32
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  // Restore jmp_buf parameter
  mov     x0, x19
  ldr     x19, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (19)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc setjmp
  adrp    x1, :got:_ZN14__interception12real__setjmpE
  ldr     x1, [x1, #:got_lo12:_ZN14__interception12real__setjmpE]
  ldr     x1, [x1]
  br      x1

  CFI_ENDPROC
.size _setjmp, .-_setjmp

.comm _ZN14__interception14real_sigsetjmpE,8,8
.globl sigsetjmp
.type sigsetjmp, @function
sigsetjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf and savesigs
  stp     x19, x20, [sp, 16]
  CFI_OFFSET (19, -16)
  CFI_OFFSET (20, -8)
  mov     w20, w1
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, __tsan_pointer_chk_guard
  ldr     x2, [x2, #:lo12:__tsan_pointer_chk_guard]
  add     x0, x29, 32
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  // restore env parameter
  mov     w1, w20
  mov     x0, x19
  ldp     x19, x20, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (29)
  CFI_RESTORE (19)
  CFI_RESTORE (20)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc sigsetjmp
  adrp    x2, :got:_ZN14__interception14real_sigsetjmpE
  ldr     x2, [x2, #:got_lo12:_ZN14__interception14real_sigsetjmpE]
  ldr     x2, [x2]
  br      x2
  CFI_ENDPROC
.size sigsetjmp, .-sigsetjmp

.comm _ZN14__interception16real___sigsetjmpE,8,8
.globl __sigsetjmp
.type __sigsetjmp, @function
__sigsetjmp:
  CFI_STARTPROC

  // save env parameters for function call
  stp     x29, x30, [sp, -32]!
  CFI_DEF_CFA_OFFSET (32)
  CFI_OFFSET (29, -32)
  CFI_OFFSET (30, -24)

  // Adjust the SP for previous frame
  add     x29, sp, 0
  CFI_DEF_CFA_REGISTER (29)

  // Save jmp_buf and savesigs
  stp     x19, x20, [sp, 16]
  CFI_OFFSET (19, -16)
  CFI_OFFSET (20, -8)
  mov     w20, w1
  mov     x19, x0

  // SP pointer mangling (see glibc setjmp)
  adrp    x2, __tsan_pointer_chk_guard
  ldr     x2, [x2, #:lo12:__tsan_pointer_chk_guard]
  add     x0, x29, 32
  eor     x1, x2, x0

  // call tsan interceptor
  bl      __tsan_setjmp

  mov     w1, w20
  mov     x0, x19
  ldp     x19, x20, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (29)
  CFI_RESTORE (19)
  CFI_RESTORE (20)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc __sigsetjmp
  adrp    x2, :got:_ZN14__interception16real___sigsetjmpE
  ldr     x2, [x2, #:got_lo12:_ZN14__interception16real___sigsetjmpE]
  ldr     x2, [x2]
  br      x2
  CFI_ENDPROC
.size __sigsetjmp, .-__sigsetjmp

#if defined(__linux__)
/* We do not need executable stack.  */
.section        .note.GNU-stack,"",@progbits
#endif
