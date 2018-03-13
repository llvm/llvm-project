//===-------------------- UnwindRegistersRestore.S ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "assembly.h"

  .text

#if !defined(__USING_SJLJ_EXCEPTIONS__)

#if defined(__i386__)
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_x866jumptoEv)
#
# void libunwind::Registers_x86::jumpto()
#
#if defined(_WIN32)
# On windows, the 'this' pointer is passed in ecx instead of on the stack
  movl   %ecx, %eax
#else
# On entry:
#  +                       +
#  +-----------------------+
#  + thread_state pointer  +
#  +-----------------------+
#  + return address        +
#  +-----------------------+   <-- SP
#  +                       +
  movl   4(%esp), %eax
#endif
  # set up eax and ret on new stack location
  movl  28(%eax), %edx # edx holds new stack pointer
  subl  $8,%edx
  movl  %edx, 28(%eax)
  movl  0(%eax), %ebx
  movl  %ebx, 0(%edx)
  movl  40(%eax), %ebx
  movl  %ebx, 4(%edx)
  # we now have ret and eax pushed onto where new stack will be
  # restore all registers
  movl   4(%eax), %ebx
  movl   8(%eax), %ecx
  movl  12(%eax), %edx
  movl  16(%eax), %edi
  movl  20(%eax), %esi
  movl  24(%eax), %ebp
  movl  28(%eax), %esp
  # skip ss
  # skip eflags
  pop    %eax  # eax was already pushed on new stack
  ret        # eip was already pushed on new stack
  # skip cs
  # skip ds
  # skip es
  # skip fs
  # skip gs

#elif defined(__x86_64__)

DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind16Registers_x86_646jumptoEv)
#
# void libunwind::Registers_x86_64::jumpto()
#
#if defined(_WIN64)
# On entry, thread_state pointer is in rcx; move it into rdi
# to share restore code below. Since this routine restores and
# overwrites all registers, we can use the same registers for
# pointers and temporaries as on unix even though win64 normally
# mustn't clobber some of them.
  movq  %rcx, %rdi
#else
# On entry, thread_state pointer is in rdi
#endif

  movq  56(%rdi), %rax # rax holds new stack pointer
  subq  $16, %rax
  movq  %rax, 56(%rdi)
  movq  32(%rdi), %rbx  # store new rdi on new stack
  movq  %rbx, 0(%rax)
  movq  128(%rdi), %rbx # store new rip on new stack
  movq  %rbx, 8(%rax)
  # restore all registers
  movq    0(%rdi), %rax
  movq    8(%rdi), %rbx
  movq   16(%rdi), %rcx
  movq   24(%rdi), %rdx
  # restore rdi later
  movq   40(%rdi), %rsi
  movq   48(%rdi), %rbp
  # restore rsp later
  movq   64(%rdi), %r8
  movq   72(%rdi), %r9
  movq   80(%rdi), %r10
  movq   88(%rdi), %r11
  movq   96(%rdi), %r12
  movq  104(%rdi), %r13
  movq  112(%rdi), %r14
  movq  120(%rdi), %r15
  # skip rflags
  # skip cs
  # skip fs
  # skip gs

#if defined(_WIN64)
  movdqu 176(%rdi),%xmm0
  movdqu 192(%rdi),%xmm1
  movdqu 208(%rdi),%xmm2
  movdqu 224(%rdi),%xmm3
  movdqu 240(%rdi),%xmm4
  movdqu 256(%rdi),%xmm5
  movdqu 272(%rdi),%xmm6
  movdqu 288(%rdi),%xmm7
  movdqu 304(%rdi),%xmm8
  movdqu 320(%rdi),%xmm9
  movdqu 336(%rdi),%xmm10
  movdqu 352(%rdi),%xmm11
  movdqu 368(%rdi),%xmm12
  movdqu 384(%rdi),%xmm13
  movdqu 400(%rdi),%xmm14
  movdqu 416(%rdi),%xmm15
#endif
  movq  56(%rdi), %rsp  # cut back rsp to new location
  pop    %rdi      # rdi was saved here earlier
  ret            # rip was saved here


#elif defined(__powerpc64__)

DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind15Registers_ppc646jumptoEv)
//
// void libunwind::Registers_ppc64::jumpto()
//
// On entry:
//  thread_state pointer is in r3
//

  // restore integral registers
  // skip r0 for now
  // skip r1 for now
  ld    %r2,   32(%r3)
  // skip r3 for now
  // skip r4 for now
  // skip r5 for now
  ld    %r6,   64(%r3)
  ld    %r7,   72(%r3)
  ld    %r8,   80(%r3)
  ld    %r9,   88(%r3)
  ld    %r10,  96(%r3)
  ld    %r11, 104(%r3)
  ld    %r12, 112(%r3)
  ld    %r13, 120(%r3)
  ld    %r14, 128(%r3)
  ld    %r15, 136(%r3)
  ld    %r16, 144(%r3)
  ld    %r17, 152(%r3)
  ld    %r18, 160(%r3)
  ld    %r19, 168(%r3)
  ld    %r20, 176(%r3)
  ld    %r21, 184(%r3)
  ld    %r22, 192(%r3)
  ld    %r23, 200(%r3)
  ld    %r24, 208(%r3)
  ld    %r25, 216(%r3)
  ld    %r26, 224(%r3)
  ld    %r27, 232(%r3)
  ld    %r28, 240(%r3)
  ld    %r29, 248(%r3)
  ld    %r30, 256(%r3)
  ld    %r31, 264(%r3)

  //restore float registers
  lfd   %f0,  312(%r3)
  lfd   %f1,  320(%r3)
  lfd   %f2,  328(%r3)
  lfd   %f3,  336(%r3)
  lfd   %f4,  344(%r3)
  lfd   %f5,  352(%r3)
  lfd   %f6,  360(%r3)
  lfd   %f7,  368(%r3)
  lfd   %f8,  376(%r3)
  lfd   %f9,  384(%r3)
  lfd   %f10, 392(%r3)
  lfd   %f11, 400(%r3)
  lfd   %f12, 408(%r3)
  lfd   %f13, 416(%r3)
  lfd   %f14, 424(%r3)
  lfd   %f15, 432(%r3)
  lfd   %f16, 440(%r3)
  lfd   %f17, 448(%r3)
  lfd   %f18, 456(%r3)
  lfd   %f19, 464(%r3)
  lfd   %f20, 472(%r3)
  lfd   %f21, 480(%r3)
  lfd   %f22, 488(%r3)
  lfd   %f23, 496(%r3)
  lfd   %f24, 504(%r3)
  lfd   %f25, 512(%r3)
  lfd   %f26, 520(%r3)
  lfd   %f27, 528(%r3)
  lfd   %f28, 536(%r3)
  lfd   %f29, 544(%r3)
  lfd   %f30, 552(%r3)
  lfd   %f31, 560(%r3)

  //TODO: restore vector registers

  // Lnovec:
  ld    %r0, 272(%r3) // __cr
  mtcr  %r0
  ld    %r0, 296(%r3) // __ctr
  mtctr %r0
  ld    %r0,   0(%r3) // __ssr0
  mtctr %r0

  ld    %r0,   16(%r3)
  ld    %r5,   56(%r3)
  ld    %r4,   48(%r3)
  ld    %r1,   24(%r3)
  ld    %r3,   40(%r3)
  bctr

#elif defined(__ppc__)

DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_ppc6jumptoEv)
;
; void libunwind::Registers_ppc::jumpto()
;
; On entry:
;  thread_state pointer is in r3
;

  ; restore integral registerrs
  ; skip r0 for now
  ; skip r1 for now
  lwz     r2, 16(r3)
  ; skip r3 for now
  ; skip r4 for now
  ; skip r5 for now
  lwz     r6, 32(r3)
  lwz     r7, 36(r3)
  lwz     r8, 40(r3)
  lwz     r9, 44(r3)
  lwz    r10, 48(r3)
  lwz    r11, 52(r3)
  lwz    r12, 56(r3)
  lwz    r13, 60(r3)
  lwz    r14, 64(r3)
  lwz    r15, 68(r3)
  lwz    r16, 72(r3)
  lwz    r17, 76(r3)
  lwz    r18, 80(r3)
  lwz    r19, 84(r3)
  lwz    r20, 88(r3)
  lwz    r21, 92(r3)
  lwz    r22, 96(r3)
  lwz    r23,100(r3)
  lwz    r24,104(r3)
  lwz    r25,108(r3)
  lwz    r26,112(r3)
  lwz    r27,116(r3)
  lwz    r28,120(r3)
  lwz    r29,124(r3)
  lwz    r30,128(r3)
  lwz    r31,132(r3)

  ; restore float registers
  lfd    f0, 160(r3)
  lfd    f1, 168(r3)
  lfd    f2, 176(r3)
  lfd    f3, 184(r3)
  lfd    f4, 192(r3)
  lfd    f5, 200(r3)
  lfd    f6, 208(r3)
  lfd    f7, 216(r3)
  lfd    f8, 224(r3)
  lfd    f9, 232(r3)
  lfd    f10,240(r3)
  lfd    f11,248(r3)
  lfd    f12,256(r3)
  lfd    f13,264(r3)
  lfd    f14,272(r3)
  lfd    f15,280(r3)
  lfd    f16,288(r3)
  lfd    f17,296(r3)
  lfd    f18,304(r3)
  lfd    f19,312(r3)
  lfd    f20,320(r3)
  lfd    f21,328(r3)
  lfd    f22,336(r3)
  lfd    f23,344(r3)
  lfd    f24,352(r3)
  lfd    f25,360(r3)
  lfd    f26,368(r3)
  lfd    f27,376(r3)
  lfd    f28,384(r3)
  lfd    f29,392(r3)
  lfd    f30,400(r3)
  lfd    f31,408(r3)

  ; restore vector registers if any are in use
  lwz    r5,156(r3)  ; test VRsave
  cmpwi  r5,0
  beq    Lnovec

  subi  r4,r1,16
  rlwinm  r4,r4,0,0,27  ; mask low 4-bits
  ; r4 is now a 16-byte aligned pointer into the red zone
  ; the _vectorRegisters may not be 16-byte aligned so copy via red zone temp buffer


#define LOAD_VECTOR_UNALIGNEDl(_index) \
  andis.  r0,r5,(1<<(15-_index))  @\
  beq    Ldone  ## _index     @\
  lwz    r0, 424+_index*16(r3)  @\
  stw    r0, 0(r4)        @\
  lwz    r0, 424+_index*16+4(r3)  @\
  stw    r0, 4(r4)        @\
  lwz    r0, 424+_index*16+8(r3)  @\
  stw    r0, 8(r4)        @\
  lwz    r0, 424+_index*16+12(r3)@\
  stw    r0, 12(r4)        @\
  lvx    v ## _index,0,r4    @\
Ldone  ## _index:

#define LOAD_VECTOR_UNALIGNEDh(_index) \
  andi.  r0,r5,(1<<(31-_index))  @\
  beq    Ldone  ## _index    @\
  lwz    r0, 424+_index*16(r3)  @\
  stw    r0, 0(r4)        @\
  lwz    r0, 424+_index*16+4(r3)  @\
  stw    r0, 4(r4)        @\
  lwz    r0, 424+_index*16+8(r3)  @\
  stw    r0, 8(r4)        @\
  lwz    r0, 424+_index*16+12(r3)@\
  stw    r0, 12(r4)        @\
  lvx    v ## _index,0,r4    @\
  Ldone  ## _index:


  LOAD_VECTOR_UNALIGNEDl(0)
  LOAD_VECTOR_UNALIGNEDl(1)
  LOAD_VECTOR_UNALIGNEDl(2)
  LOAD_VECTOR_UNALIGNEDl(3)
  LOAD_VECTOR_UNALIGNEDl(4)
  LOAD_VECTOR_UNALIGNEDl(5)
  LOAD_VECTOR_UNALIGNEDl(6)
  LOAD_VECTOR_UNALIGNEDl(7)
  LOAD_VECTOR_UNALIGNEDl(8)
  LOAD_VECTOR_UNALIGNEDl(9)
  LOAD_VECTOR_UNALIGNEDl(10)
  LOAD_VECTOR_UNALIGNEDl(11)
  LOAD_VECTOR_UNALIGNEDl(12)
  LOAD_VECTOR_UNALIGNEDl(13)
  LOAD_VECTOR_UNALIGNEDl(14)
  LOAD_VECTOR_UNALIGNEDl(15)
  LOAD_VECTOR_UNALIGNEDh(16)
  LOAD_VECTOR_UNALIGNEDh(17)
  LOAD_VECTOR_UNALIGNEDh(18)
  LOAD_VECTOR_UNALIGNEDh(19)
  LOAD_VECTOR_UNALIGNEDh(20)
  LOAD_VECTOR_UNALIGNEDh(21)
  LOAD_VECTOR_UNALIGNEDh(22)
  LOAD_VECTOR_UNALIGNEDh(23)
  LOAD_VECTOR_UNALIGNEDh(24)
  LOAD_VECTOR_UNALIGNEDh(25)
  LOAD_VECTOR_UNALIGNEDh(26)
  LOAD_VECTOR_UNALIGNEDh(27)
  LOAD_VECTOR_UNALIGNEDh(28)
  LOAD_VECTOR_UNALIGNEDh(29)
  LOAD_VECTOR_UNALIGNEDh(30)
  LOAD_VECTOR_UNALIGNEDh(31)

Lnovec:
  lwz    r0, 136(r3) ; __cr
  mtocrf  255,r0
  lwz    r0, 148(r3) ; __ctr
  mtctr  r0
  lwz    r0, 0(r3)  ; __ssr0
  mtctr  r0
  lwz    r0, 8(r3)  ; do r0 now
  lwz    r5,28(r3)  ; do r5 now
  lwz    r4,24(r3)  ; do r4 now
  lwz    r1,12(r3)  ; do sp now
  lwz    r3,20(r3)  ; do r3 last
  bctr

#elif defined(__arm64__) || defined(__aarch64__)

//
// void libunwind::Registers_arm64::jumpto()
//
// On entry:
//  thread_state pointer is in x0
//
  .p2align 2
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind15Registers_arm646jumptoEv)
  // skip restore of x0,x1 for now
  ldp    x2, x3,  [x0, #0x010]
  ldp    x4, x5,  [x0, #0x020]
  ldp    x6, x7,  [x0, #0x030]
  ldp    x8, x9,  [x0, #0x040]
  ldp    x10,x11, [x0, #0x050]
  ldp    x12,x13, [x0, #0x060]
  ldp    x14,x15, [x0, #0x070]
  ldp    x16,x17, [x0, #0x080]
  ldp    x18,x19, [x0, #0x090]
  ldp    x20,x21, [x0, #0x0A0]
  ldp    x22,x23, [x0, #0x0B0]
  ldp    x24,x25, [x0, #0x0C0]
  ldp    x26,x27, [x0, #0x0D0]
  ldp    x28,x29, [x0, #0x0E0]
  ldr    x30,     [x0, #0x100]  // restore pc into lr
  ldr    x1,      [x0, #0x0F8]
  mov    sp,x1                  // restore sp

  ldp    d0, d1,  [x0, #0x110]
  ldp    d2, d3,  [x0, #0x120]
  ldp    d4, d5,  [x0, #0x130]
  ldp    d6, d7,  [x0, #0x140]
  ldp    d8, d9,  [x0, #0x150]
  ldp    d10,d11, [x0, #0x160]
  ldp    d12,d13, [x0, #0x170]
  ldp    d14,d15, [x0, #0x180]
  ldp    d16,d17, [x0, #0x190]
  ldp    d18,d19, [x0, #0x1A0]
  ldp    d20,d21, [x0, #0x1B0]
  ldp    d22,d23, [x0, #0x1C0]
  ldp    d24,d25, [x0, #0x1D0]
  ldp    d26,d27, [x0, #0x1E0]
  ldp    d28,d29, [x0, #0x1F0]
  ldr    d30,     [x0, #0x200]
  ldr    d31,     [x0, #0x208]

  ldp    x0, x1,  [x0, #0x000]  // restore x0,x1
  ret    x30                    // jump to pc

#elif defined(__arm__) && !defined(__APPLE__)

#if !defined(__ARM_ARCH_ISA_ARM)
  .thumb
#endif

@
@ void libunwind::Registers_arm::restoreCoreAndJumpTo()
@
@ On entry:
@  thread_state pointer is in r0
@
  .p2align 2
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm20restoreCoreAndJumpToEv)
#if !defined(__ARM_ARCH_ISA_ARM) && __ARM_ARCH_ISA_THUMB == 1
  @ r8-r11: ldm into r1-r4, then mov to r8-r11
  adds r0, #0x20
  ldm r0!, {r1-r4}
  subs r0, #0x30
  mov r8, r1
  mov r9, r2
  mov r10, r3
  mov r11, r4
  @ r12 does not need loading, it it the intra-procedure-call scratch register
  ldr r2, [r0, #0x34]
  ldr r3, [r0, #0x3c]
  mov sp, r2
  mov lr, r3         @ restore pc into lr
  ldm r0, {r0-r7}
#else
  @ Use lr as base so that r0 can be restored.
  mov lr, r0
  @ 32bit thumb-2 restrictions for ldm:
  @ . the sp (r13) cannot be in the list
  @ . the pc (r15) and lr (r14) cannot both be in the list in an LDM instruction
  ldm lr, {r0-r12}
  ldr sp, [lr, #52]
  ldr lr, [lr, #60]  @ restore pc into lr
#endif
  JMP(lr)

@
@ static void libunwind::Registers_arm::restoreVFPWithFLDMD(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
#if defined(__ELF__)
  .fpu vfpv3-d16
#endif
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm19restoreVFPWithFLDMDEPy)
  @ VFP and iwMMX instructions are only available when compiling with the flags
  @ that enable them. We do not want to do that in the library (because we do not
  @ want the compiler to generate instructions that access those) but this is
  @ only accessed if the personality routine needs these registers. Use of
  @ these registers implies they are, actually, available on the target, so
  @ it's ok to execute.
  @ So, generate the instruction using the corresponding coprocessor mnemonic.
  vldmia r0, {d0-d15}
  JMP(lr)

@
@ static void libunwind::Registers_arm::restoreVFPWithFLDMX(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
#if defined(__ELF__)
  .fpu vfpv3-d16
#endif
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm19restoreVFPWithFLDMXEPy)
  vldmia r0, {d0-d15} @ fldmiax is deprecated in ARMv7+ and now behaves like vldmia
  JMP(lr)

@
@ static void libunwind::Registers_arm::restoreVFPv3(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
#if defined(__ELF__)
  .fpu vfpv3
#endif
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm12restoreVFPv3EPy)
  vldmia r0, {d16-d31}
  JMP(lr)

#if defined(__ARM_WMMX)

@
@ static void libunwind::Registers_arm::restoreiWMMX(unw_fpreg_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
#if defined(__ELF__)
  .arch armv5te
#endif
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm12restoreiWMMXEPy)
  ldcl p1, cr0, [r0], #8  @ wldrd wR0, [r0], #8
  ldcl p1, cr1, [r0], #8  @ wldrd wR1, [r0], #8
  ldcl p1, cr2, [r0], #8  @ wldrd wR2, [r0], #8
  ldcl p1, cr3, [r0], #8  @ wldrd wR3, [r0], #8
  ldcl p1, cr4, [r0], #8  @ wldrd wR4, [r0], #8
  ldcl p1, cr5, [r0], #8  @ wldrd wR5, [r0], #8
  ldcl p1, cr6, [r0], #8  @ wldrd wR6, [r0], #8
  ldcl p1, cr7, [r0], #8  @ wldrd wR7, [r0], #8
  ldcl p1, cr8, [r0], #8  @ wldrd wR8, [r0], #8
  ldcl p1, cr9, [r0], #8  @ wldrd wR9, [r0], #8
  ldcl p1, cr10, [r0], #8  @ wldrd wR10, [r0], #8
  ldcl p1, cr11, [r0], #8  @ wldrd wR11, [r0], #8
  ldcl p1, cr12, [r0], #8  @ wldrd wR12, [r0], #8
  ldcl p1, cr13, [r0], #8  @ wldrd wR13, [r0], #8
  ldcl p1, cr14, [r0], #8  @ wldrd wR14, [r0], #8
  ldcl p1, cr15, [r0], #8  @ wldrd wR15, [r0], #8
  JMP(lr)

@
@ static void libunwind::Registers_arm::restoreiWMMXControl(unw_uint32_t* values)
@
@ On entry:
@  values pointer is in r0
@
  .p2align 2
#if defined(__ELF__)
  .arch armv5te
#endif
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind13Registers_arm19restoreiWMMXControlEPj)
  ldc2 p1, cr8, [r0], #4  @ wldrw wCGR0, [r0], #4
  ldc2 p1, cr9, [r0], #4  @ wldrw wCGR1, [r0], #4
  ldc2 p1, cr10, [r0], #4  @ wldrw wCGR2, [r0], #4
  ldc2 p1, cr11, [r0], #4  @ wldrw wCGR3, [r0], #4
  JMP(lr)

#endif

#elif defined(__or1k__)

DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind14Registers_or1k6jumptoEv)
#
# void libunwind::Registers_or1k::jumpto()
#
# On entry:
#  thread_state pointer is in r3
#

  # restore integral registerrs
  l.lwz     r0,  0(r3)
  l.lwz     r1,  4(r3)
  l.lwz     r2,  8(r3)
  # skip r3 for now
  l.lwz     r4, 16(r3)
  l.lwz     r5, 20(r3)
  l.lwz     r6, 24(r3)
  l.lwz     r7, 28(r3)
  l.lwz     r8, 32(r3)
  l.lwz     r9, 36(r3)
  l.lwz    r10, 40(r3)
  l.lwz    r11, 44(r3)
  l.lwz    r12, 48(r3)
  l.lwz    r13, 52(r3)
  l.lwz    r14, 56(r3)
  l.lwz    r15, 60(r3)
  l.lwz    r16, 64(r3)
  l.lwz    r17, 68(r3)
  l.lwz    r18, 72(r3)
  l.lwz    r19, 76(r3)
  l.lwz    r20, 80(r3)
  l.lwz    r21, 84(r3)
  l.lwz    r22, 88(r3)
  l.lwz    r23, 92(r3)
  l.lwz    r24, 96(r3)
  l.lwz    r25,100(r3)
  l.lwz    r26,104(r3)
  l.lwz    r27,108(r3)
  l.lwz    r28,112(r3)
  l.lwz    r29,116(r3)
  l.lwz    r30,120(r3)
  l.lwz    r31,124(r3)

  # at last, restore r3
  l.lwz    r3,  12(r3)

  # jump to pc
  l.jr     r9
   l.nop

#elif defined(__mips__) && defined(_ABIO32) && defined(__mips_soft_float)

//
// void libunwind::Registers_mips_o32::jumpto()
//
// On entry:
//  thread state pointer is in a0 ($4)
//
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind18Registers_mips_o326jumptoEv)
  .set push
  .set noat
  .set noreorder
  .set nomacro
  // restore hi and lo
  lw    $8, (4 * 33)($4)
  mthi  $8
  lw    $8, (4 * 34)($4)
  mtlo  $8
  // r0 is zero
  lw    $1, (4 * 1)($4)
  lw    $2, (4 * 2)($4)
  lw    $3, (4 * 3)($4)
  // skip a0 for now
  lw    $5, (4 * 5)($4)
  lw    $6, (4 * 6)($4)
  lw    $7, (4 * 7)($4)
  lw    $8, (4 * 8)($4)
  lw    $9, (4 * 9)($4)
  lw    $10, (4 * 10)($4)
  lw    $11, (4 * 11)($4)
  lw    $12, (4 * 12)($4)
  lw    $13, (4 * 13)($4)
  lw    $14, (4 * 14)($4)
  lw    $15, (4 * 15)($4)
  lw    $16, (4 * 16)($4)
  lw    $17, (4 * 17)($4)
  lw    $18, (4 * 18)($4)
  lw    $19, (4 * 19)($4)
  lw    $20, (4 * 20)($4)
  lw    $21, (4 * 21)($4)
  lw    $22, (4 * 22)($4)
  lw    $23, (4 * 23)($4)
  lw    $24, (4 * 24)($4)
  lw    $25, (4 * 25)($4)
  lw    $26, (4 * 26)($4)
  lw    $27, (4 * 27)($4)
  lw    $28, (4 * 28)($4)
  lw    $29, (4 * 29)($4)
  lw    $30, (4 * 30)($4)
  // load new pc into ra
  lw    $31, (4 * 32)($4)
  // jump to ra, load a0 in the delay slot
  jr    $31
  lw    $4, (4 * 4)($4)
  .set pop

#elif defined(__mips__) && defined(_ABI64) && defined(__mips_soft_float)

//
// void libunwind::Registers_mips_n64::jumpto()
//
// On entry:
//  thread state pointer is in a0 ($4)
//
DEFINE_LIBUNWIND_PRIVATE_FUNCTION(_ZN9libunwind18Registers_mips_n646jumptoEv)
  .set push
  .set noat
  .set noreorder
  .set nomacro
  // restore hi and lo
  ld    $8, (8 * 33)($4)
  mthi  $8
  ld    $8, (8 * 34)($4)
  mtlo  $8
  // r0 is zero
  ld    $1, (8 * 1)($4)
  ld    $2, (8 * 2)($4)
  ld    $3, (8 * 3)($4)
  // skip a0 for now
  ld    $5, (8 * 5)($4)
  ld    $6, (8 * 6)($4)
  ld    $7, (8 * 7)($4)
  ld    $8, (8 * 8)($4)
  ld    $9, (8 * 9)($4)
  ld    $10, (8 * 10)($4)
  ld    $11, (8 * 11)($4)
  ld    $12, (8 * 12)($4)
  ld    $13, (8 * 13)($4)
  ld    $14, (8 * 14)($4)
  ld    $15, (8 * 15)($4)
  ld    $16, (8 * 16)($4)
  ld    $17, (8 * 17)($4)
  ld    $18, (8 * 18)($4)
  ld    $19, (8 * 19)($4)
  ld    $20, (8 * 20)($4)
  ld    $21, (8 * 21)($4)
  ld    $22, (8 * 22)($4)
  ld    $23, (8 * 23)($4)
  ld    $24, (8 * 24)($4)
  ld    $25, (8 * 25)($4)
  ld    $26, (8 * 26)($4)
  ld    $27, (8 * 27)($4)
  ld    $28, (8 * 28)($4)
  ld    $29, (8 * 29)($4)
  ld    $30, (8 * 30)($4)
  // load new pc into ra
  ld    $31, (8 * 32)($4)
  // jump to ra, load a0 in the delay slot
  jr    $31
  ld    $4, (8 * 4)($4)
  .set pop

#endif

#endif /* !defined(__USING_SJLJ_EXCEPTIONS__) */

NO_EXEC_STACK_DIRECTIVE

