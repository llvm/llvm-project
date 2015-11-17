# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld2 %t -o %t2 --gc-sections
# RUN: llvm-readobj -t %t2 | FileCheck %s

# CHECK-NOT: foo

	.section	.text,"ax",@progbits,unique,0
	.globl	foo
foo:
	.cfi_startproc
	.cfi_endproc

	.section	.text,"ax",@progbits,unique,1
	.globl	_start
_start:
	.cfi_startproc
	.cfi_endproc
