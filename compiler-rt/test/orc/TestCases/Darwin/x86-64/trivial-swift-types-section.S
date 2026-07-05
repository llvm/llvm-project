# RUN: %clang -c -o %t %s
# RUN: %llvm_jitlink -preload /usr/lib/swift/libswiftCore.dylib %t

# REQUIRES: jit-compatible-osx-swift-runtime

# Check that _typeByName is able to find a type registered by metadata in
# the __swift5_types section.

        .section        __TEXT,__text,regular,pure_instructions
        .build_version macos, 14, 0     sdk_version 14, 0
        .globl  _main
        .p2align        4, 0x90
_main:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        # Constant String for "4Test3FooV"
        movabsq $8018152761824990260, %rdi
        movabsq $-1585267068834392465, %rsi
        callq   _$ss11_typeByNameyypXpSgSSF
        xorl    %edi, %edi
        testq   %rax, %rax
        sete    %dil
        callq   _exit
        .cfi_endproc

	.private_extern	_$s4Test3FooVACycfC
	.globl	_$s4Test3FooVACycfC
	.p2align	4, 0x90
_$s4Test3FooVACycfC:
	pushq	%rbp
	movq	%rsp, %rbp
	popq	%rbp
	retq

	.private_extern	_$s4Test3FooVMa
	.globl	_$s4Test3FooVMa
	.p2align	4, 0x90
_$s4Test3FooVMa:
	leaq	_$s4Test3FooVMf+8(%rip), %rax
	xorl	%edx, %edx
	retq

	.section	__TEXT,__swift5_entry,regular,no_dead_strip
	.p2align	2
l_entry_point:
	.long	_main-l_entry_point
	.long	0

	.section	__TEXT,__const
l_.str.4.Test:
	.asciz	"Test"

	.private_extern	_$s4TestMXM
	.globl	_$s4TestMXM
	.weak_definition	_$s4TestMXM
	.p2align	2
_$s4TestMXM:
	.long	0
	.long	0
	.long	(l_.str.4.Test-_$s4TestMXM)-8

l_.str.3.Foo:
	.asciz	"Foo"

	.private_extern	_$s4Test3FooVMn
	.globl	_$s4Test3FooVMn
	.p2align	2
_$s4Test3FooVMn:
	.long	81
	.long	(_$s4TestMXM-_$s4Test3FooVMn)-4
	.long	(l_.str.3.Foo-_$s4Test3FooVMn)-8
	.long	(_$s4Test3FooVMa-_$s4Test3FooVMn)-12
	.long	(_$s4Test3FooVMF-_$s4Test3FooVMn)-16
	.long	0
	.long	2

	.section	__DATA,__const
	.p2align	3
_$s4Test3FooVMf:
	.quad	_$sytWV
	.quad	512
	.quad	_$s4Test3FooVMn

	.private_extern	"_symbolic _____ 4Test3FooV"
	.section	__TEXT,__swift5_typeref
	.globl	"_symbolic _____ 4Test3FooV"
	.weak_definition	"_symbolic _____ 4Test3FooV"
	.p2align	1
"_symbolic _____ 4Test3FooV":
	.byte	1
	.long	(_$s4Test3FooVMn-"_symbolic _____ 4Test3FooV")-1
	.byte	0

	.section	__TEXT,__swift5_fieldmd
	.p2align	2
_$s4Test3FooVMF:
	.long	"_symbolic _____ 4Test3FooV"-_$s4Test3FooVMF
	.long	0
	.short	0
	.short	12
	.long	0

	.section	__TEXT,__swift5_types
	.p2align	2
l_$s4Test3FooVHn:
	.long	_$s4Test3FooVMn-l_$s4Test3FooVHn

	.private_extern	___swift_reflection_version
	.section	__TEXT,__const
	.globl	___swift_reflection_version
	.weak_definition	___swift_reflection_version
	.p2align	1
___swift_reflection_version:
	.short	3

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	84477760

	.globl	_$s4Test3FooVN
	.private_extern	_$s4Test3FooVN
	.alt_entry	_$s4Test3FooVN
.set _$s4Test3FooVN, _$s4Test3FooVMf+8
.subsections_via_symbols
