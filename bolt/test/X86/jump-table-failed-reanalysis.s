## A jump table entry whose target is not an instruction boundary is accepted
## while the function is being disassembled but rejected when the table is
## re-analyzed afterwards. Check that BOLT skips the function instead of aborting
## with "jump table heuristic failure".

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -no-pie -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out 2>&1 | FileCheck %s

# CHECK-NOT: jump table heuristic failure
# CHECK: BOLT-WARNING: unable to analyze jump table in function {{.*}}; ignoring the function

	.text
	.globl	func
	.type	func, @function
func:
	.cfi_startproc
	jmp	*JT(,%rdi,8)
.Lc0:
	movl	$10, %eax          # 5-byte movl (b8 0a 00 00 00); .Lc0+1 is mid-instruction
	ret
.Lc1:
	movl	$11, %eax
	ret
.Lc2:
	movl	$12, %eax
	ret
	.cfi_endproc
.Lfunc_end:
	.size	func, .Lfunc_end - func

	.globl	main
	.type	main, @function
main:
	.cfi_startproc
	xorl	%edi, %edi
	call	func
	xorl	%eax, %eax
	ret
	.cfi_endproc
	.size	main, .-main

	.section	.rodata
	.align	8
JT:
	.quad	.Lc0               # entry 0: real instruction boundary
	.quad	.Lc0 + 1           # entry 1: MID-INSTRUCTION - no instruction at this offset
	.quad	.Lc1
	.quad	.Lc2
