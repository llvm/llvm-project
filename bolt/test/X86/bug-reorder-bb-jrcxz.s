# Test performs a BB reordering with unsupported
# instruction jrcxz. Reordering works correctly with the
# follow options: None, Normal or Reverse. Other strategies
# are completed with Assertion `isIntN(Size * 8 + 1, Value).
# The cause is the distance between BB where one contains
# jrcxz instruction.
# Example: OpenSSL
# https://github.com/openssl/openssl/blob/master/crypto/bn/asm/x86_64-mont5.pl#L3319

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %t.o -falign-labels -march=native -o %t.exe -Wl,-q

# RUN:  llvm-bolt %t.exe -o %t.bolted -data %t.fdata \
# RUN:    -reorder-blocks=cache+ -reorder-functions=hfsort \
# RUN:    -split-functions=2 -split-all-cold -split-eh -dyno-stats \
# RUN:    -print-finalized 2>&1 | FileCheck %s

# CHECK-NOT: value of -2105 is too large for field of 1 byte.

  .text
  .section .text.startup,"ax",@progbits
  .p2align 5,,31
  .globl main
  .type main, @function
main:
  jmp bn_sqrx8x_internal

.globl bn_sqrx8x_internal
.hidden bn_sqrx8x_internal
.type bn_sqrx8x_internal,@function
.align 32
bn_sqrx8x_internal:
__bn_sqrx8x_internal:
# FDATA: 1 bn_from_mont8x 160 1 bn_sqrx8x_internal 0 0 56
# FDATA: 1 bn_sqrx8x_internal 13 1  bn_sqrx8x_internal 40 0 60972
# FDATA: 1 bn_sqrx8x_internal 5f 1  bn_sqrx8x_internal 2c 0 60972
# FDATA: 1 bn_sqrx8x_internal 2f1 1 bn_sqrx8x_internal 500 0 60972
# FDATA: 1 bn_sqrx8x_internal 34a 1 bn_sqrx8x_internal 360 0 60972
# FDATA: 1 bn_sqrx8x_internal 411 1 bn_sqrx8x_internal 360 0 447888
# FDATA: 1 bn_sqrx8x_internal 411 1 bn_sqrx8x_internal 417 0 63984
# FDATA: 1 bn_sqrx8x_internal 427 1 bn_sqrx8x_internal 480 0 60972
# FDATA: 1 bn_sqrx8x_internal 427 1 bn_sqrx8x_internal 429 0 3012
# FDATA: 1 bn_sqrx8x_internal 467 1 bn_sqrx8x_internal 360 0 3012
# FDATA: 1 bn_sqrx8x_internal 4ba 1 bn_sqrx8x_internal 80 0 58964
# FDATA: 1 bn_sqrx8x_internal 4ba 1 bn_sqrx8x_internal 4c0 0 2008
# FDATA: 1 bn_sqrx8x_internal 4fb 1 bn_sqrx8x_internal 80 0 2008
# FDATA: 1 bn_sqrx8x_internal 5f0 1 bn_sqrx8x_internal 5f2 0 180908
# FDATA: 1 bn_sqrx8x_internal 61b 1 bn_sqrx8x_internal 540 0 180908
# FDATA: 1 bn_sqrx8x_internal 632 1 bn_sqrx8x_internal 637 0 59020
# FDATA: 1 bn_sqrx8x_internal 657 1 bn_sqrx8x_internal 660 0 59020
# FDATA: 1 bn_sqrx8x_internal 696 1 bn_sqrx8x_internal 6a0 0 120048
# FDATA: 1 bn_sqrx8x_internal 75a 1 bn_sqrx8x_internal 6a0 0 840336
# FDATA: 1 bn_sqrx8x_internal 75a 1 bn_sqrx8x_internal 760 0 120048
# FDATA: 1 bn_sqrx8x_internal 768 1 bn_sqrx8x_internal 76e 0 120048
# FDATA: 1 bn_sqrx8x_internal 7b2 1 bn_sqrx8x_internal 7c0 0 120048
# FDATA: 1 bn_sqrx8x_internal 86e 1 bn_sqrx8x_internal 7c0 0 896560
# FDATA: 1 bn_sqrx8x_internal 86e 1 bn_sqrx8x_internal 874 0 128080
# FDATA: 1 bn_sqrx8x_internal 879 1 bn_sqrx8x_internal 8c0 0 120048
# FDATA: 1 bn_sqrx8x_internal 879 1 bn_sqrx8x_internal 87b 0 8032
# FDATA: 1 bn_sqrx8x_internal 8bb 1 bn_sqrx8x_internal 7c0 0 8032
# FDATA: 1 bn_sqrx8x_internal 8e8 1 bn_sqrx8x_internal 8ed 0 120048
# FDATA: 1 bn_sqrx8x_internal 955 1 bn_sqrx8x_internal 660 0 61028
# FDATA: 1 bn_sqrx8x_internal 955 1 bn_sqrx8x_internal 95b 0 59020
# FDATA: 0 [unknown] 0 1 bn_sqrx8x_internal 5f0 0 59020
.cfi_startproc
  leaq 48+8(%rsp),%rdi
  leaq (%rsi,%r9,1),%rbp
  movq %r9,0+8(%rsp)
  movq %rbp,8+8(%rsp)
  jmp .Lsqr8x_zero_start

.align 32
.byte 0x66,0x66,0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00
.Lsqrx8x_zero:
.byte 0x3e
  movdqa %xmm0,0(%rdi)
  movdqa %xmm0,16(%rdi)
  movdqa %xmm0,32(%rdi)
  movdqa %xmm0,48(%rdi)
.Lsqr8x_zero_start:
  movdqa %xmm0,64(%rdi)
  movdqa %xmm0,80(%rdi)
  movdqa %xmm0,96(%rdi)
  movdqa %xmm0,112(%rdi)
  leaq 128(%rdi),%rdi
  subq $64,%r9
  jnz .Lsqrx8x_zero

  movq 0(%rsi),%rdx

  xorq %r10,%r10
  xorq %r11,%r11
  xorq %r12,%r12
  xorq %r13,%r13
  xorq %r14,%r14
  xorq %r15,%r15
  leaq 48+8(%rsp),%rdi
  xorq %rbp,%rbp
  jmp .Lsqrx8x_outer_loop

.align 32
.Lsqrx8x_outer_loop:
  mulxq 8(%rsi),%r8,%rax
  adcxq %r9,%r8
  adoxq %rax,%r10
  mulxq 16(%rsi),%r9,%rax
  adcxq %r10,%r9
  adoxq %rax,%r11
.byte 0xc4,0xe2,0xab,0xf6,0x86,0x18,0x00,0x00,0x00
  adcxq %r11,%r10
  adoxq %rax,%r12
.byte 0xc4,0xe2,0xa3,0xf6,0x86,0x20,0x00,0x00,0x00
  adcxq %r12,%r11
  adoxq %rax,%r13
  mulxq 40(%rsi),%r12,%rax
  adcxq %r13,%r12
  adoxq %rax,%r14
  mulxq 48(%rsi),%r13,%rax
  adcxq %r14,%r13
  adoxq %r15,%rax
  mulxq 56(%rsi),%r14,%r15
  movq 8(%rsi),%rdx
  adcxq %rax,%r14
  adoxq %rbp,%r15
  adcq 64(%rdi),%r15
  movq %r8,8(%rdi)
  movq %r9,16(%rdi)
  sbbq %rcx,%rcx
  xorq %rbp,%rbp

  mulxq 16(%rsi),%r8,%rbx
  mulxq 24(%rsi),%r9,%rax
  adcxq %r10,%r8
  adoxq %rbx,%r9
  mulxq 32(%rsi),%r10,%rbx
  adcxq %r11,%r9
  adoxq %rax,%r10
.byte 0xc4,0xe2,0xa3,0xf6,0x86,0x28,0x00,0x00,0x00
  adcxq %r12,%r10
  adoxq %rbx,%r11
.byte 0xc4,0xe2,0x9b,0xf6,0x9e,0x30,0x00,0x00,0x00
  adcxq %r13,%r11
  adoxq %r14,%r12
.byte 0xc4,0x62,0x93,0xf6,0xb6,0x38,0x00,0x00,0x00
  movq 16(%rsi),%rdx
  adcxq %rax,%r12
  adoxq %rbx,%r13
  adcxq %r15,%r13
  adoxq %rbp,%r14
  adcxq %rbp,%r14

  movq %r8,24(%rdi)
  movq %r9,32(%rdi)

  mulxq 24(%rsi),%r8,%rbx
  mulxq 32(%rsi),%r9,%rax
  adcxq %r10,%r8
  adoxq %rbx,%r9
  mulxq 40(%rsi),%r10,%rbx
  adcxq %r11,%r9
  adoxq %rax,%r10
.byte 0xc4,0xe2,0xa3,0xf6,0x86,0x30,0x00,0x00,0x00
  adcxq %r12,%r10
  adoxq %r13,%r11
.byte 0xc4,0x62,0x9b,0xf6,0xae,0x38,0x00,0x00,0x00
.byte 0x3e
  movq 24(%rsi),%rdx
  adcxq %rbx,%r11
  adoxq %rax,%r12
  adcxq %r14,%r12
  movq %r8,40(%rdi)
  movq %r9,48(%rdi)
  mulxq 32(%rsi),%r8,%rax
  adoxq %rbp,%r13
  adcxq %rbp,%r13

  mulxq 40(%rsi),%r9,%rbx
  adcxq %r10,%r8
  adoxq %rax,%r9
  mulxq 48(%rsi),%r10,%rax
  adcxq %r11,%r9
  adoxq %r12,%r10
  mulxq 56(%rsi),%r11,%r12
  movq 32(%rsi),%rdx
  movq 40(%rsi),%r14
  adcxq %rbx,%r10
  adoxq %rax,%r11
  movq 48(%rsi),%r15
  adcxq %r13,%r11
  adoxq %rbp,%r12
  adcxq %rbp,%r12

  movq %r8,56(%rdi)
  movq %r9,64(%rdi)

  mulxq %r14,%r9,%rax
  movq 56(%rsi),%r8
  adcxq %r10,%r9
  mulxq %r15,%r10,%rbx
  adoxq %rax,%r10
  adcxq %r11,%r10
  mulxq %r8,%r11,%rax
  movq %r14,%rdx
  adoxq %rbx,%r11
  adcxq %r12,%r11

  adcxq %rbp,%rax

  mulxq %r15,%r14,%rbx
  mulxq %r8,%r12,%r13
  movq %r15,%rdx
  leaq 64(%rsi),%rsi
  adcxq %r14,%r11
  adoxq %rbx,%r12
  adcxq %rax,%r12
  adoxq %rbp,%r13

.byte 0x67,0x67
  mulxq %r8,%r8,%r14
  adcxq %r8,%r13
  adcxq %rbp,%r14

  cmpq 8+8(%rsp),%rsi
  je .Lsqrx8x_outer_break

  negq %rcx
  movq $-8,%rcx
  movq %rbp,%r15
  movq 64(%rdi),%r8
  adcxq 72(%rdi),%r9
  adcxq 80(%rdi),%r10
  adcxq 88(%rdi),%r11
  adcq 96(%rdi),%r12
  adcq 104(%rdi),%r13
  adcq 112(%rdi),%r14
  adcq 120(%rdi),%r15
  leaq (%rsi),%rbp
  leaq 128(%rdi),%rdi
  sbbq %rax,%rax

  movq -64(%rsi),%rdx
  movq %rax,16+8(%rsp)
  movq %rdi,24+8(%rsp)


  xorl %eax,%eax
  jmp .Lsqrx8x_loop

.align 32
.Lsqrx8x_loop:
  movq %r8,%rbx
  mulxq 0(%rbp),%rax,%r8
  adcxq %rax,%rbx
  adoxq %r9,%r8

  mulxq 8(%rbp),%rax,%r9
  adcxq %rax,%r8
  adoxq %r10,%r9

  mulxq 16(%rbp),%rax,%r10
  adcxq %rax,%r9
  adoxq %r11,%r10

  mulxq 24(%rbp),%rax,%r11
  adcxq %rax,%r10
  adoxq %r12,%r11

.byte 0xc4,0x62,0xfb,0xf6,0xa5,0x20,0x00,0x00,0x00
  adcxq %rax,%r11
  adoxq %r13,%r12

  mulxq 40(%rbp),%rax,%r13
  adcxq %rax,%r12
  adoxq %r14,%r13

  mulxq 48(%rbp),%rax,%r14
  movq %rbx,(%rdi,%rcx,8)
  movl $0,%ebx
  adcxq %rax,%r13
  adoxq %r15,%r14

.byte 0xc4,0x62,0xfb,0xf6,0xbd,0x38,0x00,0x00,0x00
  movq 8(%rsi,%rcx,8),%rdx
  adcxq %rax,%r14
  adoxq %rbx,%r15
  adcxq %rbx,%r15

.byte 0x67
  incq %rcx
  jnz .Lsqrx8x_loop

  leaq 64(%rbp),%rbp
  movq $-8,%rcx
  cmpq 8+8(%rsp),%rbp
  je .Lsqrx8x_break

  subq 16+8(%rsp),%rbx
.byte 0x66
  movq -64(%rsi),%rdx
  adcxq 0(%rdi),%r8
  adcxq 8(%rdi),%r9
  adcq 16(%rdi),%r10
  adcq 24(%rdi),%r11
  adcq 32(%rdi),%r12
  adcq 40(%rdi),%r13
  adcq 48(%rdi),%r14
  adcq 56(%rdi),%r15
  leaq 64(%rdi),%rdi
.byte 0x67
  sbbq %rax,%rax
  xorl %ebx,%ebx
  movq %rax,16+8(%rsp)
  jmp .Lsqrx8x_loop

.align 32
.Lsqrx8x_break:
  xorq %rbp,%rbp
  subq 16+8(%rsp),%rbx
  adcxq %rbp,%r8
  movq 24+8(%rsp),%rcx
  adcxq %rbp,%r9
  movq 0(%rsi),%rdx
  adcq $0,%r10
  movq %r8,0(%rdi)
  adcq $0,%r11
  adcq $0,%r12
  adcq $0,%r13
  adcq $0,%r14
  adcq $0,%r15
  cmpq %rcx,%rdi
  je .Lsqrx8x_outer_loop

  movq %r9,8(%rdi)
  movq 8(%rcx),%r9
  movq %r10,16(%rdi)
  movq 16(%rcx),%r10
  movq %r11,24(%rdi)
  movq 24(%rcx),%r11
  movq %r12,32(%rdi)
  movq 32(%rcx),%r12
  movq %r13,40(%rdi)
  movq 40(%rcx),%r13
  movq %r14,48(%rdi)
  movq 48(%rcx),%r14
  movq %r15,56(%rdi)
  movq 56(%rcx),%r15
  movq %rcx,%rdi
  jmp .Lsqrx8x_outer_loop

.align 32
.Lsqrx8x_outer_break:
  movq %r9,72(%rdi)
.byte 102,72,15,126,217
  movq %r10,80(%rdi)
  movq %r11,88(%rdi)
  movq %r12,96(%rdi)
  movq %r13,104(%rdi)
  movq %r14,112(%rdi)
  leaq 48+8(%rsp),%rdi
  movq (%rsi,%rcx,1),%rdx

  movq 8(%rdi),%r11
  xorq %r10,%r10
  movq 0+8(%rsp),%r9
  adoxq %r11,%r11
  movq 16(%rdi),%r12
  movq 24(%rdi),%r13

.align 32
.Lsqrx4x_shift_n_add:
  mulxq %rdx,%rax,%rbx
  adoxq %r12,%r12
  adcxq %r10,%rax
.byte 0x48,0x8b,0x94,0x0e,0x08,0x00,0x00,0x00
.byte 0x4c,0x8b,0x97,0x20,0x00,0x00,0x00
  adoxq %r13,%r13
  adcxq %r11,%rbx
  movq 40(%rdi),%r11
  movq %rax,0(%rdi)
  movq %rbx,8(%rdi)

  mulxq %rdx,%rax,%rbx
  adoxq %r10,%r10
  adcxq %r12,%rax
  movq 16(%rsi,%rcx,1),%rdx
  movq 48(%rdi),%r12
  adoxq %r11,%r11
  adcxq %r13,%rbx
  movq 56(%rdi),%r13
  movq %rax,16(%rdi)
  movq %rbx,24(%rdi)

  mulxq %rdx,%rax,%rbx
  adoxq %r12,%r12
  adcxq %r10,%rax
  movq 24(%rsi,%rcx,1),%rdx
  leaq 32(%rcx),%rcx
  movq 64(%rdi),%r10
  adoxq %r13,%r13
  adcxq %r11,%rbx
  movq 72(%rdi),%r11
  movq %rax,32(%rdi)
  movq %rbx,40(%rdi)

  mulxq %rdx,%rax,%rbx
  adoxq %r10,%r10
  adcxq %r12,%rax
  jrcxz .Lsqrx4x_shift_n_add_break
.byte 0x48,0x8b,0x94,0x0e,0x00,0x00,0x00,0x00
  adoxq %r11,%r11
  adcxq %r13,%rbx
  movq 80(%rdi),%r12
  movq 88(%rdi),%r13
  movq %rax,48(%rdi)
  movq %rbx,56(%rdi)
  leaq 64(%rdi),%rdi
  nop
  jmp .Lsqrx4x_shift_n_add

.align 32
.Lsqrx4x_shift_n_add_break:
  adcxq %r13,%rbx
  movq %rax,48(%rdi)
  movq %rbx,56(%rdi)
  leaq 64(%rdi),%rdi
.byte 102,72,15,126,213
__bn_sqrx8x_reduction:
  xorl %eax,%eax
  movq 32+8(%rsp),%rbx
  movq 48+8(%rsp),%rdx
  leaq -64(%rbp,%r9,1),%rcx

  movq %rcx,0+8(%rsp)
  movq %rdi,8+8(%rsp)

  leaq 48+8(%rsp),%rdi
  jmp .Lsqrx8x_reduction_loop

.align 32
.Lsqrx8x_reduction_loop:
  movq 8(%rdi),%r9
  movq 16(%rdi),%r10
  movq 24(%rdi),%r11
  movq 32(%rdi),%r12
  movq %rdx,%r8
  imulq %rbx,%rdx
  movq 40(%rdi),%r13
  movq 48(%rdi),%r14
  movq 56(%rdi),%r15
  movq %rax,24+8(%rsp)

  leaq 64(%rdi),%rdi
  xorq %rsi,%rsi
  movq $-8,%rcx
  jmp .Lsqrx8x_reduce

.align 32
.Lsqrx8x_reduce:
  movq %r8,%rbx
  mulxq 0(%rbp),%rax,%r8
  adcxq %rbx,%rax
  adoxq %r9,%r8

  mulxq 8(%rbp),%rbx,%r9
  adcxq %rbx,%r8
  adoxq %r10,%r9

  mulxq 16(%rbp),%rbx,%r10
  adcxq %rbx,%r9
  adoxq %r11,%r10

  mulxq 24(%rbp),%rbx,%r11
  adcxq %rbx,%r10
  adoxq %r12,%r11

.byte 0xc4,0x62,0xe3,0xf6,0xa5,0x20,0x00,0x00,0x00
  movq %rdx,%rax
  movq %r8,%rdx
  adcxq %rbx,%r11
  adoxq %r13,%r12

  mulxq 32+8(%rsp),%rbx,%rdx
  movq %rax,%rdx
  movq %rax,64+48+8(%rsp,%rcx,8)

  mulxq 40(%rbp),%rax,%r13
  adcxq %rax,%r12
  adoxq %r14,%r13

  mulxq 48(%rbp),%rax,%r14
  adcxq %rax,%r13
  adoxq %r15,%r14

  mulxq 56(%rbp),%rax,%r15
  movq %rbx,%rdx
  adcxq %rax,%r14
  adoxq %rsi,%r15
  adcxq %rsi,%r15

.byte 0x67,0x67,0x67
  incq %rcx
  jnz .Lsqrx8x_reduce

  movq %rsi,%rax
  cmpq 0+8(%rsp),%rbp
  jae .Lsqrx8x_no_tail

  movq 48+8(%rsp),%rdx
  addq 0(%rdi),%r8
  leaq 64(%rbp),%rbp
  movq $-8,%rcx
  adcxq 8(%rdi),%r9
  adcxq 16(%rdi),%r10
  adcq 24(%rdi),%r11
  adcq 32(%rdi),%r12
  adcq 40(%rdi),%r13
  adcq 48(%rdi),%r14
  adcq 56(%rdi),%r15
  leaq 64(%rdi),%rdi
  sbbq %rax,%rax

  xorq %rsi,%rsi
  movq %rax,16+8(%rsp)
  jmp .Lsqrx8x_tail

.align 32
.Lsqrx8x_tail:
  movq %r8,%rbx
  mulxq 0(%rbp),%rax,%r8
  adcxq %rax,%rbx
  adoxq %r9,%r8

  mulxq 8(%rbp),%rax,%r9
  adcxq %rax,%r8
  adoxq %r10,%r9

  mulxq 16(%rbp),%rax,%r10
  adcxq %rax,%r9
  adoxq %r11,%r10

  mulxq 24(%rbp),%rax,%r11
  adcxq %rax,%r10
  adoxq %r12,%r11

.byte 0xc4,0x62,0xfb,0xf6,0xa5,0x20,0x00,0x00,0x00
  adcxq %rax,%r11
  adoxq %r13,%r12

  mulxq 40(%rbp),%rax,%r13
  adcxq %rax,%r12
  adoxq %r14,%r13

  mulxq 48(%rbp),%rax,%r14
  adcxq %rax,%r13
  adoxq %r15,%r14

  mulxq 56(%rbp),%rax,%r15
  movq 72+48+8(%rsp,%rcx,8),%rdx
  adcxq %rax,%r14
  adoxq %rsi,%r15
  movq %rbx,(%rdi,%rcx,8)
  movq %r8,%rbx
  adcxq %rsi,%r15

  incq %rcx
  jnz .Lsqrx8x_tail

  cmpq 0+8(%rsp),%rbp
  jae .Lsqrx8x_tail_done

  subq 16+8(%rsp),%rsi
  movq 48+8(%rsp),%rdx
  leaq 64(%rbp),%rbp
  adcq 0(%rdi),%r8
  adcq 8(%rdi),%r9
  adcq 16(%rdi),%r10
  adcq 24(%rdi),%r11
  adcq 32(%rdi),%r12
  adcq 40(%rdi),%r13
  adcq 48(%rdi),%r14
  adcq 56(%rdi),%r15
  leaq 64(%rdi),%rdi
  sbbq %rax,%rax
  subq $8,%rcx

  xorq %rsi,%rsi
  movq %rax,16+8(%rsp)
  jmp .Lsqrx8x_tail

.align 32
.Lsqrx8x_tail_done:
  xorq %rax,%rax
  addq 24+8(%rsp),%r8
  adcq $0,%r9
  adcq $0,%r10
  adcq $0,%r11
  adcq $0,%r12
  adcq $0,%r13
  adcq $0,%r14
  adcq $0,%r15
  adcq $0,%rax

  subq 16+8(%rsp),%rsi
.Lsqrx8x_no_tail:
  adcq 0(%rdi),%r8
.byte 102,72,15,126,217
  adcq 8(%rdi),%r9
  movq 56(%rbp),%rsi
.byte 102,72,15,126,213
  adcq 16(%rdi),%r10
  adcq 24(%rdi),%r11
  adcq 32(%rdi),%r12
  adcq 40(%rdi),%r13
  adcq 48(%rdi),%r14
  adcq 56(%rdi),%r15
  adcq $0,%rax

  movq 32+8(%rsp),%rbx
  movq 64(%rdi,%rcx,1),%rdx

  movq %r8,0(%rdi)
  leaq 64(%rdi),%r8
  movq %r9,8(%rdi)
  movq %r10,16(%rdi)
  movq %r11,24(%rdi)
  movq %r12,32(%rdi)
  movq %r13,40(%rdi)
  movq %r14,48(%rdi)
  movq %r15,56(%rdi)

  leaq 64(%rdi,%rcx,1),%rdi
  cmpq 8+8(%rsp),%r8
  jb .Lsqrx8x_reduction_loop
  .byte 0xf3,0xc3
.cfi_endproc
.size  bn_sqrx8x_internal,.-bn_sqrx8x_internal
