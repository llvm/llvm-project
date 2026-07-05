// Generated from g++ exception4.cpp -fomit-frame-pointer -S -O3
// Manually modified to trigger shrink-wrapping in function main
	.text
	.p2align 4,,15
	.globl	_Z3fooi
	.type	_Z3fooi, @function
_Z3fooi:
.LFB12:
	.cfi_startproc
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	cmpl	$1, %edi
	movl	$1, %edi
	jle	.L2
	call	__cxa_allocate_exception
	xorl	%edx, %edx
	movl	$_ZTI4ExcG, %esi
	movq	%rax, %rdi
	call	__cxa_throw
.L2:
	call	__cxa_allocate_exception
	xorl	%edx, %edx
	movl	$_ZTI4ExcC, %esi
	movq	%rax, %rdi
	call	__cxa_throw
	.cfi_endproc
.LFE12:
	.size	_Z3fooi, .-_Z3fooi
	.p2align 4,,15
	.globl	_Z11filter_onlyi
	.type	_Z11filter_onlyi, @function
_Z11filter_onlyi:
.LFB13:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA13
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
.LEHB0:
	call	_Z3fooi
.LEHE0:
.L8:
	addq	$1, %rdx
	movq	%rax, %rdi
	je	.L7
.LEHB1:
	call	_Unwind_Resume
.L7:
	call	__cxa_call_unexpected
.LEHE1:
	.cfi_endproc
.LFE13:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
	.align 4
.LLSDA13:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT13-.LLSDATTD13
.LLSDATTD13:
	.byte	0x1
	.uleb128 .LLSDACSE13-.LLSDACSB13
.LLSDACSB13:
	.uleb128 .LEHB0-.LFB13
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L8-.LFB13
	.uleb128 0x1
	.uleb128 .LEHB1-.LFB13
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE13:
	.byte	0x7f
	.byte	0
	.align 4
	.long	_ZTI4ExcA
	.long	_ZTI4ExcB
	.long	_ZTI4ExcC
	.long	_ZTI4ExcD
	.long	_ZTI4ExcE
	.long	_ZTI4ExcF
.LLSDATT13:
	.byte	0x1
	.byte	0x2
	.byte	0x3
	.byte	0x4
	.byte	0x5
	.byte	0x6
	.byte	0
	.text
	.size	_Z11filter_onlyi, .-_Z11filter_onlyi
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"this statement is cold and should be outlined"
	.text
	.p2align 4,,15
	.globl	_Z12never_throwsv
	.type	_Z12never_throwsv, @function
_Z12never_throwsv:
.LFB14:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA14
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	movl	$.LC0, %edi
.LEHB2:
	call	puts
.LEHE2:
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L13:
	.cfi_restore_state
	addq	$1, %rdx
	movq	%rax, %rdi
	je	.L12
.LEHB3:
	call	_Unwind_Resume
.L12:
	call	__cxa_call_unexpected
.LEHE3:
	.cfi_endproc
.LFE14:
	.section	.gcc_except_table,"a",@progbits
	.align 4
.LLSDA14:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT14-.LLSDATTD14
.LLSDATTD14:
	.byte	0x1
	.uleb128 .LLSDACSE14-.LLSDACSB14
.LLSDACSB14:
	.uleb128 .LEHB2-.LFB14
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L13-.LFB14
	.uleb128 0x1
	.uleb128 .LEHB3-.LFB14
	.uleb128 .LEHE3-.LEHB3
	.uleb128 0
	.uleb128 0
.LLSDACSE14:
	.byte	0x7f
	.byte	0
	.align 4
.LLSDATT14:
	.byte	0
	.text
	.size	_Z12never_throwsv, .-_Z12never_throwsv
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"catch 2"
.LC2:
	.string	"catch 1"
.LC3:
	.string	"caught ExcC"
.LC4:
	.string	"caught ExcG"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB15:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA15
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	movl	$10, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movl	%edi, %ebx
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
.Lcheck:
  cmpl $2, %ebx
  je .Lmyexit
.Lloop_pre_header:
	movl	$10, %ebp
  jmp .L18
.L18:
	cmpl	$2, %ebx
	je	.Lmyexit
.L17:
	movl	$1, %edi
	call	__cxa_allocate_exception
	xorl	%edx, %edx
	movl	$_ZTI4ExcA, %esi
	movq	%rax, %rdi
.LEHB4:
	call	__cxa_throw
.LEHE4:
.L47:
	call	_Z12never_throwsv
	call	_Z12never_throwsv
	.p2align 4,,5
	jmp	.L17
.L37:
	cmpq	$2, %rdx
	movq	%rax, %rdi
	jne	.L22
	call	__cxa_begin_catch
	movl	$.LC1, %edi
.LEHB5:
	call	puts
.LEHE5:
	movl	$8, %edi
	call	__cxa_allocate_exception
	movl	$4, %edi
	movq	%rax, %r12
.LEHB6:
	call	_Znwm
.LEHE6:
	movl	$0, (%rax)
	xorl	%edx, %edx
	movq	%rax, (%r12)
	movl	$_ZTIPi, %esi
	movq	%r12, %rdi
.LEHB7:
	call	__cxa_throw
.LEHE7:
.L39:
	movq	%rax, %r13
	movq	%r12, %rdi
	movq	%r13, %r12
	call	__cxa_free_exception
.L24:
	call	__cxa_end_catch
	movq	%r12, %rdi
.L22:
	call	__cxa_begin_catch
	movl	$.LC2, %edi
.LEHB8:
	call	puts
.LEHE8:
.LEHB9:
	call	__cxa_end_catch
.LEHE9:
	movl	%ebx, %edi
.LEHB10:
	call	_Z11filter_onlyi
.LEHE10:
.L38:
	movq	%rax, %r12
	jmp	.L24
.L41:
	cmpq	$4, %rdx
	movq	%rax, %rdi
	movq	%rdx, %rax
	jne	.L29
	call	__cxa_begin_catch
	movl	$.LC3, %edi
.LEHB11:
	call	puts
.LEHE11:
.L35:
	call	__cxa_end_catch
	subl	$1, %ebp
	jne	.L18
.Lmyexit:
	popq	%rdx
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	xorl	%eax, %eax
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
.L40:
	.cfi_restore_state
	movq	%rax, %rbx
	call	__cxa_end_catch
	movq	%rbx, %rdi
.LEHB12:
	call	_Unwind_Resume
.LEHE12:
.L42:
	movq	%rax, %r13
	movq	%rdx, %r12
	call	__cxa_end_catch
	movq	%r13, %rdi
	movq	%r12, %rax
.L29:
	cmpq	$3, %rax
	jne	.L46
	call	__cxa_begin_catch
	movl	$.LC4, %edi
.LEHB13:
	call	puts
.LEHE13:
	jmp	.L35
.L43:
	movq	%rax, %rbx
	call	__cxa_end_catch
	movq	%rbx, %rdi
.L46:
.LEHB14:
	call	_Unwind_Resume
  jmp .L43
.LEHE14:
	.cfi_endproc
.LFE15:
	.section	.gcc_except_table,"a",@progbits
	.align 4
.LLSDA15:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT15-.LLSDATTD15
.LLSDATTD15:
	.byte	0x1
	.uleb128 .LLSDACSE15-.LLSDACSB15
.LLSDACSB15:
	.uleb128 .LEHB4-.LFB15
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L37-.LFB15
	.uleb128 0x3
	.uleb128 .LEHB5-.LFB15
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L38-.LFB15
	.uleb128 0x5
	.uleb128 .LEHB6-.LFB15
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L39-.LFB15
	.uleb128 0x5
	.uleb128 .LEHB7-.LFB15
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L38-.LFB15
	.uleb128 0x5
	.uleb128 .LEHB8-.LFB15
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L40-.LFB15
	.uleb128 0
	.uleb128 .LEHB9-.LFB15
	.uleb128 .LEHE9-.LEHB9
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB10-.LFB15
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L41-.LFB15
	.uleb128 0x9
	.uleb128 .LEHB11-.LFB15
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L42-.LFB15
	.uleb128 0xb
	.uleb128 .LEHB12-.LFB15
	.uleb128 .LEHE12-.LEHB12
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB13-.LFB15
	.uleb128 .LEHE13-.LEHB13
	.uleb128 .L43-.LFB15
	.uleb128 0
	.uleb128 .LEHB14-.LFB15
	.uleb128 .LEHE14-.LEHB14
	.uleb128 0
	.uleb128 0
.LLSDACSE15:
	.byte	0x1
	.byte	0
	.byte	0x2
	.byte	0x7d
	.byte	0
	.byte	0x7b
	.byte	0x3
	.byte	0
	.byte	0x4
	.byte	0x7d
	.byte	0
	.byte	0x7b
	.align 4
	.long	_ZTI4ExcC
	.long	_ZTI4ExcG
	.long	_ZTI4ExcA
	.long	0

.LLSDATT15:
	.section	.text.startup
	.size	main, .-main
	.weak	_ZTI4ExcG
	.section	.rodata._ZTI4ExcG,"aG",@progbits,_ZTI4ExcG,comdat
	.align 16
	.type	_ZTI4ExcG, @object
	.size	_ZTI4ExcG, 16
_ZTI4ExcG:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4ExcG
	.weak	_ZTS4ExcG
	.section	.rodata._ZTS4ExcG,"aG",@progbits,_ZTS4ExcG,comdat
	.type	_ZTS4ExcG, @object
	.size	_ZTS4ExcG, 6
_ZTS4ExcG:
	.string	"4ExcG"
	.weak	_ZTI4ExcC
	.section	.rodata._ZTI4ExcC,"aG",@progbits,_ZTI4ExcC,comdat
	.align 16
	.type	_ZTI4ExcC, @object
	.size	_ZTI4ExcC, 16
_ZTI4ExcC:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4ExcC
	.weak	_ZTS4ExcC
	.section	.rodata._ZTS4ExcC,"aG",@progbits,_ZTS4ExcC,comdat
	.type	_ZTS4ExcC, @object
	.size	_ZTS4ExcC, 6
_ZTS4ExcC:
	.string	"4ExcC"
	.weak	_ZTI4ExcA
	.section	.rodata._ZTI4ExcA,"aG",@progbits,_ZTI4ExcA,comdat
	.align 16
	.type	_ZTI4ExcA, @object
	.size	_ZTI4ExcA, 16
_ZTI4ExcA:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4ExcA
	.weak	_ZTS4ExcA
	.section	.rodata._ZTS4ExcA,"aG",@progbits,_ZTS4ExcA,comdat
	.type	_ZTS4ExcA, @object
	.size	_ZTS4ExcA, 6
_ZTS4ExcA:
	.string	"4ExcA"
	.weak	_ZTI4ExcB
	.section	.rodata._ZTI4ExcB,"aG",@progbits,_ZTI4ExcB,comdat
	.align 16
	.type	_ZTI4ExcB, @object
	.size	_ZTI4ExcB, 16
_ZTI4ExcB:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4ExcB
	.weak	_ZTS4ExcB
	.section	.rodata._ZTS4ExcB,"aG",@progbits,_ZTS4ExcB,comdat
	.type	_ZTS4ExcB, @object
	.size	_ZTS4ExcB, 6
_ZTS4ExcB:
	.string	"4ExcB"
	.weak	_ZTI4ExcD
	.section	.rodata._ZTI4ExcD,"aG",@progbits,_ZTI4ExcD,comdat
	.align 16
	.type	_ZTI4ExcD, @object
	.size	_ZTI4ExcD, 16
_ZTI4ExcD:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4ExcD
	.weak	_ZTS4ExcD
	.section	.rodata._ZTS4ExcD,"aG",@progbits,_ZTS4ExcD,comdat
	.type	_ZTS4ExcD, @object
	.size	_ZTS4ExcD, 6
_ZTS4ExcD:
	.string	"4ExcD"
	.weak	_ZTI4ExcE
	.section	.rodata._ZTI4ExcE,"aG",@progbits,_ZTI4ExcE,comdat
	.align 16
	.type	_ZTI4ExcE, @object
	.size	_ZTI4ExcE, 16
_ZTI4ExcE:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4ExcE
	.weak	_ZTS4ExcE
	.section	.rodata._ZTS4ExcE,"aG",@progbits,_ZTS4ExcE,comdat
	.type	_ZTS4ExcE, @object
	.size	_ZTS4ExcE, 6
_ZTS4ExcE:
	.string	"4ExcE"
	.weak	_ZTI4ExcF
	.section	.rodata._ZTI4ExcF,"aG",@progbits,_ZTI4ExcF,comdat
	.align 16
	.type	_ZTI4ExcF, @object
	.size	_ZTI4ExcF, 16
_ZTI4ExcF:
	.quad	_ZTVN10__cxxabiv117__class_type_infoE+16
	.quad	_ZTS4ExcF
	.weak	_ZTS4ExcF
	.section	.rodata._ZTS4ExcF,"aG",@progbits,_ZTS4ExcF,comdat
	.type	_ZTS4ExcF, @object
	.size	_ZTS4ExcF, 6
_ZTS4ExcF:
	.string	"4ExcF"
