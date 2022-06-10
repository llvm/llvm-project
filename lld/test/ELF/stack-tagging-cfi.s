# RUN: llvm-mc -filetype=obj -triple aarch64 -mattr=+mte %s -o %t.o
# RUN: ld.lld --eh-frame-hdr %t.o -o %t
# RUN: llvm-objdump --dwarf=frames %t | FileCheck %s

# CHECK: Augmentation:          "zRG"

	.text
	.globl	WithUnwind
	.p2align	2
	.type	WithUnwind,@function
WithUnwind:
	.cfi_startproc
	.cfi_mte_tagged_frame
	ret
.Lfunc_end0:
	.size	WithUnwind, .Lfunc_end0-WithUnwind
	.cfi_endproc
