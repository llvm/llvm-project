// The content of this file is AArch64-only:
#if defined(__aarch64__)

#include "sanitizer_common/sanitizer_asm.h"

#if defined(__APPLE__)
.align  2

.section  __DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
.long _setjmp$non_lazy_ptr
_setjmp$non_lazy_ptr:
.indirect_symbol _setjmp
.long 0

.section  __DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
.long __setjmp$non_lazy_ptr
__setjmp$non_lazy_ptr:
.indirect_symbol __setjmp
.long 0

.section  __DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
.long _sigsetjmp$non_lazy_ptr
_sigsetjmp$non_lazy_ptr:
.indirect_symbol _sigsetjmp
.long 0
#endif

#if !defined(__APPLE__)
.section .text
#else
.section __TEXT,__text
.align 3
#endif

ASM_HIDDEN(__tsan_setjmp)
.comm _ZN14__interception11real_setjmpE,8,8
.globl ASM_SYMBOL_INTERCEPTOR(setjmp)
ASM_TYPE_FUNCTION(ASM_SYMBOL_INTERCEPTOR(setjmp))
ASM_SYMBOL_INTERCEPTOR(setjmp):
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

  // Obtain SP, first argument to `void __tsan_setjmp(uptr sp)`
  add     x0, x29, 32

  // call tsan interceptor
  bl      ASM_SYMBOL(__tsan_setjmp)

  // restore env parameter
  mov     x0, x19
  ldr     x19, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (19)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc setjmp
#if !defined(__APPLE__)
  adrp    x1, :got:_ZN14__interception11real_setjmpE
  ldr     x1, [x1, #:got_lo12:_ZN14__interception11real_setjmpE]
  ldr     x1, [x1]
#else
  adrp    x1, _setjmp$non_lazy_ptr@page
  add     x1, x1, _setjmp$non_lazy_ptr@pageoff
  ldr     x1, [x1]
#endif
  br      x1

  CFI_ENDPROC
ASM_SIZE(ASM_SYMBOL_INTERCEPTOR(setjmp))

.comm _ZN14__interception12real__setjmpE,8,8
.globl ASM_SYMBOL_INTERCEPTOR(_setjmp)
ASM_TYPE_FUNCTION(ASM_SYMBOL_INTERCEPTOR(_setjmp))
ASM_SYMBOL_INTERCEPTOR(_setjmp):
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

  // Obtain SP, first argument to `void __tsan_setjmp(uptr sp)`
  add     x0, x29, 32

  // call tsan interceptor
  bl      ASM_SYMBOL(__tsan_setjmp)

  // Restore jmp_buf parameter
  mov     x0, x19
  ldr     x19, [sp, 16]
  ldp     x29, x30, [sp], 32
  CFI_RESTORE (30)
  CFI_RESTORE (19)
  CFI_DEF_CFA (31, 0)

  // tail jump to libc setjmp
#if !defined(__APPLE__)
  adrp    x1, :got:_ZN14__interception12real__setjmpE
  ldr     x1, [x1, #:got_lo12:_ZN14__interception12real__setjmpE]
  ldr     x1, [x1]
#else
  adrp    x1, __setjmp$non_lazy_ptr@page
  add     x1, x1, __setjmp$non_lazy_ptr@pageoff
  ldr     x1, [x1]
#endif
  br      x1

  CFI_ENDPROC
ASM_SIZE(ASM_SYMBOL_INTERCEPTOR(_setjmp))

.comm _ZN14__interception14real_sigsetjmpE,8,8
.globl ASM_SYMBOL_INTERCEPTOR(sigsetjmp)
ASM_TYPE_FUNCTION(ASM_SYMBOL_INTERCEPTOR(sigsetjmp))
ASM_SYMBOL_INTERCEPTOR(sigsetjmp):
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

  // Obtain SP, first argument to `void __tsan_setjmp(uptr sp)`
  add     x0, x29, 32

  // call tsan interceptor
  bl      ASM_SYMBOL(__tsan_setjmp)

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
#if !defined(__APPLE__)
  adrp    x2, :got:_ZN14__interception14real_sigsetjmpE
  ldr     x2, [x2, #:got_lo12:_ZN14__interception14real_sigsetjmpE]
  ldr     x2, [x2]
#else
  adrp    x2, _sigsetjmp$non_lazy_ptr@page
  add     x2, x2, _sigsetjmp$non_lazy_ptr@pageoff
  ldr     x2, [x2]
#endif
  br      x2
  CFI_ENDPROC
ASM_SIZE(ASM_SYMBOL_INTERCEPTOR(sigsetjmp))

#if !defined(__APPLE__)
.comm _ZN14__interception16real___sigsetjmpE,8,8
.globl ASM_SYMBOL_INTERCEPTOR(__sigsetjmp)
ASM_TYPE_FUNCTION(ASM_SYMBOL_INTERCEPTOR(__sigsetjmp))
ASM_SYMBOL_INTERCEPTOR(__sigsetjmp):
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

  // Obtain SP, first argument to `void __tsan_setjmp(uptr sp)`
  add     x0, x29, 32

  // call tsan interceptor
  bl      ASM_SYMBOL(__tsan_setjmp)

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
#if !defined(__APPLE__)
  adrp    x2, :got:_ZN14__interception16real___sigsetjmpE
  ldr     x2, [x2, #:got_lo12:_ZN14__interception16real___sigsetjmpE]
  ldr     x2, [x2]
#else
  adrp    x2, ASM_SYMBOL(__sigsetjmp)@page
  add     x2, x2, ASM_SYMBOL(__sigsetjmp)@pageoff
#endif
  br      x2
  CFI_ENDPROC
ASM_SIZE(ASM_SYMBOL_INTERCEPTOR(__sigsetjmp))
#endif

NO_EXEC_STACK_DIRECTIVE

#endif
