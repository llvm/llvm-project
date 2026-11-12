# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-cloudabi %s -o %t1.o
# RUN: ld.lld -pie %t1.o -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOCS %s
# RUN: llvm-objdump -d --no-show-raw-insn --no-print-imm-hex --no-leading-addr %t1.o | FileCheck --check-prefix=DIS %s

# Bug 27174: R_X86_64_TPOFF32 and R_X86_64_GOTTPOFF relocations should
# be eliminated when building a PIE executable, as the static TLS layout
# is fixed.
#
# RELOCS:      Relocations [
# RELOCS-NEXT: ]
#
# DIS: <_start>:
# DIS-NEXT:                movq    %fs:0, %rax
# DIS-NEXT:                movl    $3, (%rax)
# DIS-NEXT:                movq    %fs:0, %rdx
# DIS-NEXT:                movq    (%rip), %rcx
# DIS-NEXT:                movl    $3, (%rdx,%rcx)
# DIS-NEXT:                movabsq 0, %rax

	.globl	_start
_start:
	movq	%fs:0, %rax
	movl	$3, i@TPOFF(%rax)

	movq	%fs:0, %rdx
	movq	i@GOTTPOFF(%rip), %rcx
	movl	$3, (%rdx,%rcx)

	# This additionally tests support for R_X86_64_TPOFF64 relocations.
	movabs  i@TPOFF, %rax

	.section	.tbss.i,"awT",@nobits
	.globl	i
i:
	.long	0
	.size	i, 4
