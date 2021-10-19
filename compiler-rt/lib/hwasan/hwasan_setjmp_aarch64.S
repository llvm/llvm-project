//===-- hwasan_setjmp_aarch64.S -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// HWAddressSanitizer runtime.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_asm.h"
#include "builtins/assembly.h"

#if HWASAN_WITH_INTERCEPTORS && defined(__aarch64__)
#include "sanitizer_common/sanitizer_platform.h"

// We want to save the context of the calling function.
// That requires
// 1) No modification of the link register by this function.
// 2) No modification of the stack pointer by this function.
// 3) (no modification of any other saved register, but that's not really going
// to occur, and hence isn't as much of a worry).
//
// There's essentially no way to ensure that the compiler will not modify the
// stack pointer when compiling a C function.
// Hence we have to write this function in assembly.

.section .text
.file "hwasan_setjmp_aarch64.S"

.global __interceptor_setjmp
ASM_TYPE_FUNCTION(__interceptor_setjmp)
__interceptor_setjmp:
  CFI_STARTPROC
  BTI_C
  mov	x1, #0
  b	__interceptor_sigsetjmp
  CFI_ENDPROC
ASM_SIZE(__interceptor_setjmp)

#if SANITIZER_ANDROID
// Bionic also defines a function `setjmp` that calls `sigsetjmp` saving the
// current signal.
.global __interceptor_setjmp_bionic
ASM_TYPE_FUNCTION(__interceptor_setjmp_bionic)
__interceptor_setjmp_bionic:
  CFI_STARTPROC
  BTI_C
  mov	x1, #1
  b	__interceptor_sigsetjmp
  CFI_ENDPROC
ASM_SIZE(__interceptor_setjmp_bionic)
#endif

.global __interceptor_sigsetjmp
ASM_TYPE_FUNCTION(__interceptor_sigsetjmp)
__interceptor_sigsetjmp:
  CFI_STARTPROC
  BTI_C
  stp	x19, x20, [x0, #0<<3]
  stp	x21, x22, [x0, #2<<3]
  stp	x23, x24, [x0, #4<<3]
  stp	x25, x26, [x0, #6<<3]
  stp	x27, x28, [x0, #8<<3]
  stp	x29, x30, [x0, #10<<3]
  stp	 d8,  d9, [x0, #14<<3]
  stp	d10, d11, [x0, #16<<3]
  stp	d12, d13, [x0, #18<<3]
  stp	d14, d15, [x0, #20<<3]
  mov	x2,  sp
  str	x2,  [x0, #13<<3]
  // We always have the second argument to __sigjmp_save (savemask) set, since
  // the _setjmp function above has set it for us as `false`.
  // This function is defined in hwasan_interceptors.cc
  b	__sigjmp_save
  CFI_ENDPROC
ASM_SIZE(__interceptor_sigsetjmp)


.macro WEAK_ALIAS first second
  .weak \second
  .equ \second\(), \first
.endm

#if SANITIZER_ANDROID
WEAK_ALIAS __interceptor_sigsetjmp, sigsetjmp
WEAK_ALIAS __interceptor_setjmp_bionic, setjmp
#else
WEAK_ALIAS __interceptor_sigsetjmp, __sigsetjmp
#endif

WEAK_ALIAS __interceptor_setjmp, _setjmp
#endif

// We do not need executable stack.
NO_EXEC_STACK_DIRECTIVE

GNU_PROPERTY_BTI_PAC
