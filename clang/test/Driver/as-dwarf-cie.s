# REQUIRES: x86-registered-target
# Test that there is a sane default CIE version.
# RUN: %clang -cc1as -triple i386-apple-darwin -filetype obj %s -o %t
# RUN: llvm-objdump -dwarf=frames %t | FileCheck %s
# CHECK: .debug_frame contents:
# CHECK: CIE
# CHECK: Version:               1
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_f
	.p2align	4, 0x90
_f:                                     ## @f
Lfunc_begin0:
	.file	1 "test.c"
	.loc	1 1 0                   ## test.c:1:0
	.cfi_startproc
## BB#0:                                ## %entry
	pushl	%ebp
Ltmp0:
	.cfi_def_cfa_offset 8
Ltmp1:
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
Ltmp2:
	.cfi_def_cfa_register %ebp
Ltmp3:
	.loc	1 1 11 prologue_end     ## test.c:1:11
	popl	%ebp
	retl
Ltmp4:
Lfunc_end0:
	.cfi_endproc
	.cfi_sections .debug_frame

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
