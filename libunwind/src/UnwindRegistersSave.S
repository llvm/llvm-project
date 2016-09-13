//===------------------------ UnwindRegistersSave.S -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "assembly.h"

    .text

#if defined(__i386__)

#
# extern int unw_getcontext(unw_context_t* thread_state)
#
# On entry:
#   +                       +
#   +-----------------------+
#   + thread_state pointer  +
#   +-----------------------+
#   + return address        +
#   +-----------------------+   <-- SP
#   +                       +
#
DEFINE_LIBUNWIND_FUNCTION(unw_getcontext)
  push  %eax
  movl  8(%esp), %eax
  movl  %ebx,  4(%eax)
  movl  %ecx,  8(%eax)
  movl  %edx, 12(%eax)
  movl  %edi, 16(%eax)
  movl  %esi, 20(%eax)
  movl  %ebp, 24(%eax)
  movl  %esp, %edx
  addl  $8, %edx
  movl  %edx, 28(%eax)  # store what sp was at call site as esp
  # skip ss
  # skip eflags
  movl  4(%esp), %edx
  movl  %edx, 40(%eax)  # store return address as eip
  # skip cs
  # skip ds
  # skip es
  # skip fs
  # skip gs
  movl  (%esp), %edx
  movl  %edx, (%eax)  # store original eax
  popl  %eax
  xorl  %eax, %eax    # return UNW_ESUCCESS
  ret

#elif defined(__x86_64__)

#
# extern int unw_getcontext(unw_context_t* thread_state)
#
# On entry:
#  thread_state pointer is in rdi
#
DEFINE_LIBUNWIND_FUNCTION(unw_getcontext)
  movq  %rax,   (%rdi)
  movq  %rbx,  8(%rdi)
  movq  %rcx, 16(%rdi)
  movq  %rdx, 24(%rdi)
  movq  %rdi, 32(%rdi)
  movq  %rsi, 40(%rdi)
  movq  %rbp, 48(%rdi)
  movq  %rsp, 56(%rdi)
  addq  $8,   56(%rdi)
  movq  %r8,  64(%rdi)
  movq  %r9,  72(%rdi)
  movq  %r10, 80(%rdi)
  movq  %r11, 88(%rdi)
  movq  %r12, 96(%rdi)
  movq  %r13,104(%rdi)
  movq  %r14,112(%rdi)
  movq  %r15,120(%rdi)
  movq  (%rsp),%rsi
  movq  %rsi,128(%rdi) # store return address as rip
  # skip rflags
  # skip cs
  # skip fs
  # skip gs
  xorl  %eax, %eax    # return UNW_ESUCCESS
  ret

# elif defined(__mips__)

#
# extern int unw_getcontext(unw_context_t* thread_state)
#
# Just trap for the time being.
DEFINE_LIBUNWIND_FUNCTION(unw_getcontext)
  teq $0, $0

#elif defined(__ppc__)

;
; extern int unw_getcontext(unw_context_t* thread_state)
;
; On entry:
;  thread_state pointer is in r3
;
DEFINE_LIBUNWIND_FUNCTION(unw_getcontext)
  stw    r0,  8(r3)
  mflr  r0
  stw    r0,  0(r3)  ; store lr as ssr0
  stw    r1, 12(r3)
  stw    r2, 16(r3)
  stw    r3, 20(r3)
  stw    r4, 24(r3)
  stw    r5, 28(r3)
  stw    r6, 32(r3)
  stw    r7, 36(r3)
  stw    r8, 40(r3)
  stw    r9, 44(r3)
  stw     r10, 48(r3)
  stw     r11, 52(r3)
  stw     r12, 56(r3)
  stw     r13, 60(r3)
  stw     r14, 64(r3)
  stw     r15, 68(r3)
  stw     r16, 72(r3)
  stw     r17, 76(r3)
  stw     r18, 80(r3)
  stw     r19, 84(r3)
  stw     r20, 88(r3)
  stw     r21, 92(r3)
  stw     r22, 96(r3)
  stw     r23,100(r3)
  stw     r24,104(r3)
  stw     r25,108(r3)
  stw     r26,112(r3)
  stw     r27,116(r3)
  stw     r28,120(r3)
  stw     r29,124(r3)
  stw     r30,128(r3)
  stw     r31,132(r3)

  ; save VRSave register
  mfspr  r0,256
  stw    r0,156(r3)
  ; save CR registers
  mfcr  r0
  stw    r0,136(r3)
  ; save CTR register
  mfctr  r0
  stw    r0,148(r3)

  ; save float registers
  stfd    f0, 160(r3)
  stfd    f1, 168(r3)
  stfd    f2, 176(r3)
  stfd    f3, 184(r3)
  stfd    f4, 192(r3)
  stfd    f5, 200(r3)
  stfd    f6, 208(r3)
  stfd    f7, 216(r3)
  stfd    f8, 224(r3)
  stfd    f9, 232(r3)
  stfd    f10,240(r3)
  stfd    f11,248(r3)
  stfd    f12,256(r3)
  stfd    f13,264(r3)
  stfd    f14,272(r3)
  stfd    f15,280(r3)
  stfd    f16,288(r3)
  stfd    f17,296(r3)
  stfd    f18,304(r3)
  stfd    f19,312(r3)
  stfd    f20,320(r3)
  stfd    f21,328(r3)
  stfd    f22,336(r3)
  stfd    f23,344(r3)
  stfd    f24,352(r3)
  stfd    f25,360(r3)
  stfd    f26,368(r3)
  stfd    f27,376(r3)
  stfd    f28,384(r3)
  stfd    f29,392(r3)
  stfd    f30,400(r3)
  stfd    f31,408(r3)


  ; save vector registers

  subi  r4,r1,16
  rlwinm  r4,r4,0,0,27  ; mask low 4-bits
  ; r4 is now a 16-byte aligned pointer into the red zone

#define SAVE_VECTOR_UNALIGNED(_vec, _offset) \
  stvx  _vec,0,r4           @\
  lwz    r5, 0(r4)          @\
  stw    r5, _offset(r3)    @\
  lwz    r5, 4(r4)          @\
  stw    r5, _offset+4(r3)  @\
  lwz    r5, 8(r4)          @\
  stw    r5, _offset+8(r3)  @\
  lwz    r5, 12(r4)         @\
  stw    r5, _offset+12(r3)

  SAVE_VECTOR_UNALIGNED( v0, 424+0x000)
  SAVE_VECTOR_UNALIGNED( v1, 424+0x010)
  SAVE_VECTOR_UNALIGNED( v2, 424+0x020)
  SAVE_VECTOR_UNALIGNED( v3, 424+0x030)
  SAVE_VECTOR_UNALIGNED( v4, 424+0x040)
  SAVE_VECTOR_UNALIGNED( v5, 424+0x050)
  SAVE_VECTOR_UNALIGNED( v6, 424+0x060)
  SAVE_VECTOR_UNALIGNED( v7, 424+0x070)
  SAVE_VECTOR_UNALIGNED( v8, 424+0x080)
  SAVE_VECTOR_UNALIGNED( v9, 424+0x090)
  SAVE_VECTOR_UNALIGNED(v10, 424+0x0A0)
  SAVE_VECTOR_UNALIGNED(v11, 424+0x0B0)
  SAVE_VECTOR_UNALIGNED(v12, 424+0x0C0)
  SAVE_VECTOR_UNALIGNED(v13, 424+0x0D0)
  SAVE_VECTOR_UNALIGNED(v14, 424+0x0E0)
  SAVE_VECTOR_UNALIGNED(v15, 424+0x0F0)
  SAVE_VECTOR_UNALIGNED(v16, 424+0x100)
  SAVE_VECTOR_UNALIGNED(v17, 424+0x110)
  SAVE_VECTOR_UNALIGNED(v18, 424+0x120)
  SAVE_VECTOR_UNALIGNED(v19, 424+0x130)
  SAVE_VECTOR_UNALIGNED(v20, 424+0x140)
  SAVE_VECTOR_UNALIGNED(v21, 424+0x150)
  SAVE_VECTOR_UNALIGNED(v22, 424+0x160)
  SAVE_VECTOR_UNALIGNED(v23, 424+0x170)
  SAVE_VECTOR_UNALIGNED(v24, 424+0x180)
  SAVE_VECTOR_UNALIGNED(v25, 424+0x190)
  SAVE_VECTOR_UNALIGNED(v26, 424+0x1A0)
  SAVE_VECTOR_UNALIGNED(v27, 424+0x1B0)
  SAVE_VECTOR_UNALIGNED(v28, 424+0x1C0)
  SAVE_VECTOR_UNALIGNED(v29, 424+0x1D0)
  SAVE_VECTOR_UNALIGNED(v30, 424+0x1E0)
  SAVE_VECTOR_UNALIGNED(v31, 424+0x1F0)

  li  r3, 0    ; return UNW_ESUCCESS
  blr


#elif defined(__arm64__) || defined(__aarch64__)

//
// extern int unw_getcontext(unw_context_t* thread_state)
//
// On entry:
//  thread_state pointer is in x0
//
  .p2align 2
DEFINE_LIBUNWIND_FUNCTION(unw_getcontext)
  stp    x0, x1,  [x0, #0x000]
  stp    x2, x3,  [x0, #0x010]
  stp    x4, x5,  [x0, #0x020]
  stp    x6, x7,  [x0, #0x030]
  stp    x8, x9,  [x0, #0x040]
  stp    x10,x11, [x0, #0x050]
  stp    x12,x13, [x0, #0x060]
  stp    x14,x15, [x0, #0x070]
  stp    x16,x17, [x0, #0x080]
  stp    x18,x19, [x0, #0x090]
  stp    x20,x21, [x0, #0x0A0]
  stp    x22,x23, [x0, #0x0B0]
  stp    x24,x25, [x0, #0x0C0]
  stp    x26,x27, [x0, #0x0D0]
  stp    x28,x29, [x0, #0x0E0]
  str    x30,     [x0, #0x0F0]
  mov    x1,sp
  str    x1,      [x0, #0x0F8]
  str    x30,     [x0, #0x100]    // store return address as pc
  // skip cpsr
  stp    d0, d1,  [x0, #0x110]
  stp    d2, d3,  [x0, #0x120]
  stp    d4, d5,  [x0, #0x130]
  stp    d6, d7,  [x0, #0x140]
  stp    d8, d9,  [x0, #0x150]
  stp    d10,d11, [x0, #0x160]
  stp    d12,d13, [x0, #0x170]
  stp    d14,d15, [x0, #0x180]
  stp    d16,d17, [x0, #0x190]
  stp    d18,d19, [x0, #0x1A0]
  stp    d20,d21, [x0, #0x1B0]
  stp    d22,d23, [x0, #0x1C0]
  stp    d24,d25, [x0, #0x1D0]
  stp    d26,d27, [x0, #0x1E0]
  stp    d28,d29, [x0, #0x1F0]
  str    d30,     [x0, #0x200]
  str    d31,     [x0, #0x208]
  mov    x0, #0                   // return UNW_ESUCCESS
  ret

#elif defined(__arm__) && !defined(__APPLE__)

#if !defined(__ARM_ARCH_ISA_ARM)
  .thumb
#endif

@
@ extern int unw_getcontext(unw_context_t* thread_state)
@
@ On entry:
@  thread_state pointer is in r0
@ 
@ Per EHABI #4.7 this only saves the core integer registers.
@ EHABI #7.4.5 notes that in general all VRS registers should be restored
@ however this is very hard to do for VFP registers because it is unknown
@ to the library how many registers are implemented by the architecture.
@ Instead, VFP registers are demand saved by logic external to unw_getcontext.
@
  .p2align 2
DEFINE_LIBUNWIND_FUNCTION(unw_getcontext)
#if !defined(__ARM_ARCH_ISA_ARM)
  stm r0, {r0-r7}
  mov r2, sp
  mov r3, lr
  str r2, [r0, #52]
  str r3, [r0, #56]
  str r3, [r0, #60]  @ store return address as pc
#else
  @ 32bit thumb-2 restrictions for stm:
  @ . the sp (r13) cannot be in the list
  @ . the pc (r15) cannot be in the list in an STM instruction
  stm r0, {r0-r12}
  str sp, [r0, #52]
  str lr, [r0, #56]
  str lr, [r0, #60]  @ store return address as pc
#endif
#if __ARM_ARCH_ISA_THUMB == 1
  @ T1 does not have a non-cpsr-clobbering register-zeroing instruction.
  @ It is safe to use here though because we are about to return, and cpsr is
  @ not expected to be preserved.
  movs r0, #0        @ return UNW_ESUCCESS
#else
  mov r0, #0         @ return UNW_ESUCCESS
#endif
  JMP(lr)

@
@ static void libunwind::Registers_arm::saveVFPWithFSTMD(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
  .fpu vfpv3-d16
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm16saveVFPWithFSTMDEPy)
  vstmia r0, {d0-d15}
  JMP(lr)

@
@ static void libunwind::Registers_arm::saveVFPWithFSTMX(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
  .fpu vfpv3-d16
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm16saveVFPWithFSTMXEPy)
  vstmia r0, {d0-d15} @ fstmiax is deprecated in ARMv7+ and now behaves like vstmia
  JMP(lr)

@
@ static void libunwind::Registers_arm::saveVFPv3(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
  .fpu vfpv3
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm9saveVFPv3EPy)
  @ VFP and iwMMX instructions are only available when compiling with the flags
  @ that enable them. We do not want to do that in the library (because we do not
  @ want the compiler to generate instructions that access those) but this is
  @ only accessed if the personality routine needs these registers. Use of
  @ these registers implies they are, actually, available on the target, so
  @ it's ok to execute.
  @ So, generate the instructions using the corresponding coprocessor mnemonic.
  vstmia r0, {d16-d31}
  JMP(lr)

#if defined(_LIBUNWIND_ARM_WMMX)

@
@ static void libunwind::Registers_arm::saveiWMMX(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
  .arch armv5te
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm9saveiWMMXEPy)
  stcl p1, cr0, [r0], #8  @ wstrd wR0, [r0], #8
  stcl p1, cr1, [r0], #8  @ wstrd wR1, [r0], #8
  stcl p1, cr2, [r0], #8  @ wstrd wR2, [r0], #8
  stcl p1, cr3, [r0], #8  @ wstrd wR3, [r0], #8
  stcl p1, cr4, [r0], #8  @ wstrd wR4, [r0], #8
  stcl p1, cr5, [r0], #8  @ wstrd wR5, [r0], #8
  stcl p1, cr6, [r0], #8  @ wstrd wR6, [r0], #8
  stcl p1, cr7, [r0], #8  @ wstrd wR7, [r0], #8
  stcl p1, cr8, [r0], #8  @ wstrd wR8, [r0], #8
  stcl p1, cr9, [r0], #8  @ wstrd wR9, [r0], #8
  stcl p1, cr10, [r0], #8  @ wstrd wR10, [r0], #8
  stcl p1, cr11, [r0], #8  @ wstrd wR11, [r0], #8
  stcl p1, cr12, [r0], #8  @ wstrd wR12, [r0], #8
  stcl p1, cr13, [r0], #8  @ wstrd wR13, [r0], #8
  stcl p1, cr14, [r0], #8  @ wstrd wR14, [r0], #8
  stcl p1, cr15, [r0], #8  @ wstrd wR15, [r0], #8
  JMP(lr)

@
@ static void libunwind::Registers_arm::saveiWMMXControl(unw_uint32_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
  .arch armv5te
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm16saveiWMMXControlEPj)
  stc2 p1, cr8, [r0], #4  @ wstrw wCGR0, [r0], #4
  stc2 p1, cr9, [r0], #4  @ wstrw wCGR1, [r0], #4
  stc2 p1, cr10, [r0], #4  @ wstrw wCGR2, [r0], #4
  stc2 p1, cr11, [r0], #4  @ wstrw wCGR3, [r0], #4
  JMP(lr)

#endif

#elif defined(__or1k__)

#
# extern int unw_getcontext(unw_context_t* thread_state)
#
# On entry:
#  thread_state pointer is in r3
#
DEFINE_LIBUNWIND_FUNCTION(unw_getcontext)
  l.sw       0(r3), r0
  l.sw       4(r3), r1
  l.sw       8(r3), r2
  l.sw      12(r3), r3
  l.sw      16(r3), r4
  l.sw      20(r3), r5
  l.sw      24(r3), r6
  l.sw      28(r3), r7
  l.sw      32(r3), r8
  l.sw      36(r3), r9
  l.sw      40(r3), r10
  l.sw      44(r3), r11
  l.sw      48(r3), r12
  l.sw      52(r3), r13
  l.sw      56(r3), r14
  l.sw      60(r3), r15
  l.sw      64(r3), r16
  l.sw      68(r3), r17
  l.sw      72(r3), r18
  l.sw      76(r3), r19
  l.sw      80(r3), r20
  l.sw      84(r3), r21
  l.sw      88(r3), r22
  l.sw      92(r3), r23
  l.sw      96(r3), r24
  l.sw     100(r3), r25
  l.sw     104(r3), r26
  l.sw     108(r3), r27
  l.sw     112(r3), r28
  l.sw     116(r3), r29
  l.sw     120(r3), r30
  l.sw     124(r3), r31
#endif

NO_EXEC_STACK_DIRECTIVE

