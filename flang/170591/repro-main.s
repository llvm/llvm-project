	.file	"FIRModule"
	.text
	.globl	kohb_exit_
	.p2align	4
	.type	kohb_exit_,@function
kohb_exit_:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movslq	(%rdi), %rbx
	testq	%rbx, %rbx
	jle	.LBB0_3
	movl	%ebx, %edi
	callq	_FortranAExit@PLT
	incq	%rbx
	movl	$1, %ebp
	movq	_QQclX28412C493029@GOTPCREL(%rip), %r14
	movq	_QQclX9022f1477d515e3e3f2d41fd0f2d14e2@GOTPCREL(%rip), %r15
	movq	_QQclX4B4F48622023@GOTPCREL(%rip), %r12
	.p2align	4
.LBB0_2:
	movl	$6, %esi
	movq	%r14, %rdi
	xorl	%edx, %edx
	movl	$6, %ecx
	movq	%r15, %r8
	movl	$6, %r9d
	callq	_FortranAioBeginExternalFormattedOutput@PLT
	movq	%rax, %r13
	movl	$6, %edx
	movq	%rax, %rdi
	movq	%r12, %rsi
	callq	_FortranAioOutputAscii@PLT
	movq	%r13, %rdi
	movl	%ebp, %esi
	callq	_FortranAioOutputInteger32@PLT
	movq	%r13, %rdi
	callq	_FortranAioEndIoStatement@PLT
	incl	%ebp
	decq	%rbx
	cmpq	$1, %rbx
	ja	.LBB0_2
.LBB0_3:
	xorl	%eax, %eax
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	kohb_exit_, .Lfunc_end0-kohb_exit_
	.cfi_endproc

	.type	_QQclX28412C493029,@object
	.section	.rodata._QQclX28412C493029,"aG",@progbits,_QQclX28412C493029,comdat
	.weak	_QQclX28412C493029
_QQclX28412C493029:
	.ascii	"(A,I0)"
	.size	_QQclX28412C493029, 6

	.type	_QQclX9022f1477d515e3e3f2d41fd0f2d14e2,@object
	.section	.rodata._QQclX9022f1477d515e3e3f2d41fd0f2d14e2,"aG",@progbits,_QQclX9022f1477d515e3e3f2d41fd0f2d14e2,comdat
	.weak	_QQclX9022f1477d515e3e3f2d41fd0f2d14e2
	.p2align	4, 0x0
_QQclX9022f1477d515e3e3f2d41fd0f2d14e2:
	.asciz	"/home/eepshteyn/eugene-tasks/gh-issue-dump/170591/repro-main.f90"
	.size	_QQclX9022f1477d515e3e3f2d41fd0f2d14e2, 65

	.type	_QQclX4B4F48622023,@object
	.section	.rodata._QQclX4B4F48622023,"aG",@progbits,_QQclX4B4F48622023,comdat
	.weak	_QQclX4B4F48622023
_QQclX4B4F48622023:
	.ascii	"KOHb #"
	.size	_QQclX4B4F48622023, 6

	.ident	"flang version 22.0.0 (https://github.com/eugeneepshteyn/llvm-project.git e74b425ddcac22ccc4d0bd5d65f95ffc2682b62f)"
	.section	".note.GNU-stack","",@progbits
