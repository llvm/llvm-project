# REQUIRES: system-linux

## Check that BOLT correctly processes jump table that contains function start
## as one of its entries.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -no-pie -Wl,-q

# RUN: llvm-bolt %t.exe --print-normalized --print-only=foo -o %t.out \
# RUN:   |& FileCheck %s



  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  call foo
  ret
  .cfi_endproc
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
	.cfi_startproc
.LBB00:
          movq	0x8(%rdi), %rdi
          movzbl	0x1(%rdi), %eax
.LBB00_br:
	        jmpq	*"JUMP_TABLE/foo.0"(,%rax,8)
# CHECK:  jmpq {{.*}} # JUMPTABLE
# CHECK-NEXT: Successors: {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}

.Ltmp87085:
	xorl	%eax, %eax
	retq

.Ltmp87086:
	cmpb	$0x0, 0x8(%rdi)
	setne	%al
	retq

.Ltmp87088:
	movb	$0x1, %al
	retq

.Ltmp87087:
	movzbl	0x14(%rdi), %eax
	andb	$0x2, %al
	shrb	%al
	retq

	.cfi_endproc
.size foo, .-foo

# Jump tables
.section .rodata
"JUMP_TABLE/foo.0":
	.quad	.Ltmp87085
	.quad	.Ltmp87086
	.quad	.Ltmp87087
	.quad	.LBB00
	.quad	.Ltmp87088

# CHECK: Jump table {{.*}} for function foo
# CHECK-NEXT: 0x{{.*}} :
# CHECK-NEXT: 0x{{.*}} :
# CHECK-NEXT: 0x{{.*}} :
# CHECK-NEXT: 0x{{.*}} :
# CHECK-NEXT: 0x{{.*}} :
