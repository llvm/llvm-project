	.file	"exc_args.cpp"
	.text
	.globl	_Z3fooiiiiiiii
	.type	_Z3fooiiiiiiii, @function
_Z3fooiiiiiiii:
.LFB15:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	%edx, -12(%rbp)
	movl	%ecx, -16(%rbp)
	movl	%r8d, -20(%rbp)
	movl	%r9d, -24(%rbp)
	cmpl	$1, -4(%rbp)
	jle	.L2
	movl	$1, %edi
	call	__cxa_allocate_exception
	movl	$0, %edx
	movl	$_ZTI4ExcG, %esi
	movq	%rax, %rdi
	call	__cxa_throw
.L2:
	movl	$1, %edi
	call	__cxa_allocate_exception
	movl	$0, %edx
	movl	$_ZTI4ExcC, %esi
	movq	%rax, %rdi
	call	__cxa_throw
	.cfi_endproc
.LFE15:
	.size	_Z3fooiiiiiiii, .-_Z3fooiiiiiiii
	.globl	_Z11filter_onlyi
	.type	_Z11filter_onlyi, @function
_Z11filter_onlyi:
.LFB16:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA16
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movl	-4(%rbp), %eax
	pushq	$7
	pushq	$6
	movl	$5, %r9d
	movl	$4, %r8d
	movl	$3, %ecx
	movl	$2, %edx
	movl	$1, %esi
	movl	%eax, %edi
.LEHB0:
	.cfi_escape 0x2e,0x10
	call	_Z3fooiiiiiiii
.LEHE0:
	addq	$16, %rsp
	jmp	.L7
.L6:
	cmpq	$-1, %rdx
	je	.L5
	movq	%rax, %rdi
.LEHB1:
	call	_Unwind_Resume
.L5:
	movq	%rax, %rdi
	call	__cxa_call_unexpected
.LEHE1:
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
	.align 4
.LLSDA16:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT16-.LLSDATTD16
.LLSDATTD16:
	.byte	0x1
	.uleb128 .LLSDACSE16-.LLSDACSB16
.LLSDACSB16:
	.uleb128 .LEHB0-.LFB16
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L6-.LFB16
	.uleb128 0x1
	.uleb128 .LEHB1-.LFB16
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE16:
	.byte	0x7f
	.byte	0
	.align 4
	.long	_ZTI4ExcA
	.long	_ZTI4ExcB
	.long	_ZTI4ExcC
	.long	_ZTI4ExcD
	.long	_ZTI4ExcE
	.long	_ZTI4ExcF
.LLSDATT16:
	.byte	0x1
	.byte	0x2
	.byte	0x3
	.byte	0x4
	.byte	0x5
	.byte	0x6
	.byte	0
	.text
	.size	_Z11filter_onlyi, .-_Z11filter_onlyi
	.section	.rodata
	.align 8
.LC0:
	.string	"this statement is cold and should be outlined"
	.text
	.globl	_Z12never_throwsv
	.type	_Z12never_throwsv, @function
_Z12never_throwsv:
.LFB17:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA17
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$.LC0, %edi
.LEHB2:
	call	puts
.LEHE2:
	jmp	.L12
.L11:
	cmpq	$-1, %rdx
	je	.L10
	movq	%rax, %rdi
.LEHB3:
	call	_Unwind_Resume
.L10:
	movq	%rax, %rdi
	call	__cxa_call_unexpected
.LEHE3:
.L12:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.section	.gcc_except_table
	.align 4
.LLSDA17:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT17-.LLSDATTD17
.LLSDATTD17:
	.byte	0x1
	.uleb128 .LLSDACSE17-.LLSDACSB17
.LLSDACSB17:
	.uleb128 .LEHB2-.LFB17
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L11-.LFB17
	.uleb128 0x1
	.uleb128 .LEHB3-.LFB17
	.uleb128 .LEHE3-.LEHB3
	.uleb128 0
	.uleb128 0
.LLSDACSE17:
	.byte	0x7f
	.byte	0
	.align 4
.LLSDATT17:
	.byte	0
	.text
	.size	_Z12never_throwsv, .-_Z12never_throwsv
	.section	.rodata
.LC1:
	.string	"caught exception"
.LC2:
	.string	"caught ExcC"
	.text
	.globl	main
	.type	main, @function
main:
.LFB18:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA18
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movl	$1768710518, -26(%rbp)
	movw	$100, -22(%rbp)
	movl	$0, -20(%rbp)
.L17:
	cmpl	$999999, -20(%rbp)
	ja	.L14
	cmpl	$2, -36(%rbp)
	jne	.L15
	call	_Z12never_throwsv
.L15:
	cmpl	$2, -36(%rbp)
	jne	.L16
	movl	-36(%rbp), %eax
	movl	%eax, %edi
.LEHB4:
	call	_Z11filter_onlyi
.LEHE4:
.L16:
	movl	-36(%rbp), %eax
	pushq	$7
	pushq	$6
	movl	$5, %r9d
	movl	$4, %r8d
	movl	$3, %ecx
	movl	$2, %edx
	movl	$1, %esi
	movl	%eax, %edi
.LEHB5:
	.cfi_escape 0x2e,0x10
	call	_Z3fooiiiiiiii
.LEHE5:
	addq	$16, %rsp
.L25:
	addl	$1, -20(%rbp)
	jmp	.L17
.L14:
	movl	$0, %eax
	jmp	.L31
.L27:
	movq	%rax, %rdi
	call	__cxa_begin_catch
	movl	$.LC1, %edi
	movl	$0, %eax
.LEHB6:
	.cfi_escape 0x2e,0
	call	printf
.LEHE6:
.LEHB7:
	call	__cxa_end_catch
.LEHE7:
	jmp	.L16
.L28:
	movq	%rax, %rbx
	call	__cxa_end_catch
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB8:
	call	_Unwind_Resume
.L29:
	cmpq	$2, %rdx
	je	.L22
	movq	%rax, %rdi
	call	_Unwind_Resume
.LEHE8:
.L22:
	movq	%rax, %rdi
	call	__cxa_begin_catch
	movzbl	-26(%rbp), %eax
	cmpb	$118, %al
	je	.L23
	call	abort
.L23:
	movzbl	-25(%rbp), %eax
	cmpb	$97, %al
	je	.L24
	call	abort
.L24:
	movl	$.LC2, %edi
.LEHB9:
	call	puts
.LEHE9:
	call	__cxa_end_catch
	jmp	.L25
.L30:
	movq	%rax, %rbx
	call	__cxa_end_catch
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB10:
	call	_Unwind_Resume
.LEHE10:
.L31:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.section	.gcc_except_table
	.align 4
.LLSDA18:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT18-.LLSDATTD18
.LLSDATTD18:
	.byte	0x1
	.uleb128 .LLSDACSE18-.LLSDACSB18
.LLSDACSB18:
	.uleb128 .LEHB4-.LFB18
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L27-.LFB18
	.uleb128 0x1
	.uleb128 .LEHB5-.LFB18
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L29-.LFB18
	.uleb128 0x3
	.uleb128 .LEHB6-.LFB18
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L28-.LFB18
	.uleb128 0
	.uleb128 .LEHB7-.LFB18
	.uleb128 .LEHE7-.LEHB7
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB8-.LFB18
	.uleb128 .LEHE8-.LEHB8
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB9-.LFB18
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L30-.LFB18
	.uleb128 0
	.uleb128 .LEHB10-.LFB18
	.uleb128 .LEHE10-.LEHB10
	.uleb128 0
	.uleb128 0
.LLSDACSE18:
	.byte	0x1
	.byte	0
	.byte	0x2
	.byte	0
	.align 4
	.long	_ZTI4ExcC
	.long	0

.LLSDATT18:
	.text
	.size	main, .-main
	.weak	_ZTI4ExcF
	.section	.rodata._ZTI4ExcF,"aG",@progbits,_ZTI4ExcF,comdat
	.align 8
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
	.weak	_ZTI4ExcE
	.section	.rodata._ZTI4ExcE,"aG",@progbits,_ZTI4ExcE,comdat
	.align 8
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
	.weak	_ZTI4ExcD
	.section	.rodata._ZTI4ExcD,"aG",@progbits,_ZTI4ExcD,comdat
	.align 8
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
	.weak	_ZTI4ExcB
	.section	.rodata._ZTI4ExcB,"aG",@progbits,_ZTI4ExcB,comdat
	.align 8
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
	.weak	_ZTI4ExcA
	.section	.rodata._ZTI4ExcA,"aG",@progbits,_ZTI4ExcA,comdat
	.align 8
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
	.weak	_ZTI4ExcC
	.section	.rodata._ZTI4ExcC,"aG",@progbits,_ZTI4ExcC,comdat
	.align 8
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
	.weak	_ZTI4ExcG
	.section	.rodata._ZTI4ExcG,"aG",@progbits,_ZTI4ExcG,comdat
	.align 8
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
	.ident	"GCC: (GNU) 8.5.0 20210514 (Red Hat 8.5.0-3)"
	.section	.note.GNU-stack,"",@progbits
