## Check that llvm-bolt is able to read a file with DWARF Exception CFI
## information and annotate this into a disassembled function.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.null --print-cfg 2>&1 | FileCheck %s
#
# CHECK:  Binary Function "_Z7catchitv" after building cfg {
# CHECK:    CFI Instrs  : 6
# CHECK:  }
# CHECK:  DWARF CFI Instructions:
# CHECK:      0:   OpDefCfaOffset
# CHECK:      1:   OpOffset
# CHECK:      2:   OpDefCfaRegister
# CHECK:      3:   OpOffset
# CHECK:      4:   OpOffset
# CHECK:      5:   OpDefCfa
# CHECK:  End of Function "_Z7catchitv"

	.text
  .globl main
  .type main, %function
main:
# FDATA: 0 [unknown] 0 1 main 0 0 0
	.cfi_startproc
.LBB000:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0x400c4c, %esi
	movl	$0x6012e0, %edi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	movl	$0x400908, %esi
	movq	%rax, %rdi
	callq	_ZNSolsEPFRSoS_E@PLT
	callq	_Z7catchitv
	movl	$0x0, %eax
	leave
	.cfi_def_cfa %rsp, 8
	retq

	.cfi_endproc
.size main, .-main

  .globl _Z7catchitv
  .type _Z7catchitv, %function
_Z7catchitv:
# FDATA: 0 [unknown] 0 1 _Z7catchitv 0 0 0
	.cfi_startproc
.LBB00:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r12
	pushq	%rbx
	subq	$0x10, %rsp
	.cfi_offset %rbx, -32
	.cfi_offset %r12, -24
	callq	_Z5raisev
.LBB00_br: 	jmp	.Ltmp0
# FDATA: 1 _Z7catchitv #.LBB00_br# 1 _Z7catchitv #.Ltmp0# 0 0

.LLP0:
	cmpq	$0x1, %rdx
.LLP0_br: 	je	.Ltmp1
# FDATA: 1 _Z7catchitv #.LLP0_br# 1 _Z7catchitv #.Ltmp1# 0 0
# FDATA: 1 _Z7catchitv #.LLP0_br# 1 _Z7catchitv #.LFT0# 0 0

.LFT0:
	movq	%rax, %rdi
.LFT0_br: 	callq	_Unwind_Resume@PLT
# FDATA: 1 _Z7catchitv #.LFT0_br# 1 _Z7catchitv #.Ltmp1# 0 0

.Ltmp1:
	movq	%rax, %rdi
	callq	__cxa_begin_catch@PLT
	movq	%rax, -0x18(%rbp)
	movl	$0x400c40, %esi
	movl	$0x6012e0, %edi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
.Ltmp1_br: 	jmp	.Ltmp2
# FDATA: 1 _Z7catchitv #.Ltmp1_br# 1 _Z7catchitv #.Ltmp2# 0 0

.LLP1:
	movl	%edx, %ebx
	movq	%rax, %r12
	callq	__cxa_end_catch@PLT
	movq	%r12, %rax
	movslq	%ebx, %rdx
	movq	%rax, %rdi
.LLP1_br: 	callq	_Unwind_Resume@PLT
# FDATA: 1 _Z7catchitv #.LLP1_br# 1 _Z7catchitv #.Ltmp2# 0 0

.Ltmp2:
.Ltmp2_br: 	callq	__cxa_end_catch@PLT
# FDATA: 1 _Z7catchitv #.Ltmp2_br# 1 _Z7catchitv #.Ltmp0# 0 0

.Ltmp0:
	addq	$0x10, %rsp
	popq	%rbx
	popq	%r12
	leave
	.cfi_def_cfa %rsp, 8
	retq

	.cfi_endproc
.size _Z7catchitv, .-_Z7catchitv
