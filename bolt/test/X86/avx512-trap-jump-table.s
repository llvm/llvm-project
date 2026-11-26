## Check that a function with a jump table that traps on entry due to AVX-512
## isn't ignored in strict or use-old-text mode.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -no-pie -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --trap-avx512=1 --strict 2>&1 | FileCheck %s
# RUN: llvm-bolt %t.exe -o %t.out --trap-avx512=1 --use-old-text 2>&1 \
# RUN:   | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.out | FileCheck %s --check-prefix=CHECK-TRAP

# CHECK-NOT: BOLT-ERROR
# CHECK-NOT: unable to analyze jump table
# CHECK: BOLT-WARNING: 1 function will trap on entry

# CHECK-TRAP: <func>:
# CHECK-TRAP-NEXT: ud2

	.text
	.globl	func
	.type	func, @function
func:
	.cfi_startproc
	cmpl	$2, %edi
	ja	.Ldef
	movl	%edi, %eax
	jmpq	*.LJT(,%rax,8)
.L0:
	vmovdqu64	%zmm0, (%rsi)
	retq
.L1:
	movl	$1, %eax
	retq
.L2:
	movl	$2, %eax
	retq
.Ldef:
	xorl	%eax, %eax
	retq
	.cfi_endproc
	.size	func, .-func

	.globl	main
	.type	main, @function
main:
	.cfi_startproc
	movl	$1, %edi
	callq	func
	xorl	%eax, %eax
	retq
	.cfi_endproc
	.size	main, .-main

	.section	.rodata,"a",@progbits
	.p2align	3
.LJT:
	.quad	.L0
	.quad	.L1
	.quad	.L2
