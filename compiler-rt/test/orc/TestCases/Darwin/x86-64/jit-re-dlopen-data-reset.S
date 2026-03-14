// Test that __orc_rt_macho_jit_dlopen resets the state of the data sections.
//
// RUN: %clang -c -o %t.main.o %p/Inputs/dlopen-dlclose-x2.S
// RUN: %clang -c -o %t.inits.o %s
// RUN: %llvm_jitlink \
// RUN:   -alias Platform:_dlopen=___orc_rt_macho_jit_dlopen \
// RUN:   -alias Platform:_dlclose=___orc_rt_macho_jit_dlclose \
// RUN:   %t.main.o -jd inits %t.inits.o -lmain | FileCheck %s

// CHECK: entering main
// CHECK-NEXT: X = 1, Y = 2
// CHECK-NEXT: X = 1, Y = 2
// CHECK-NEXT: leaving main

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 0
	.section	__TEXT,__StaticInit,regular,pure_instructions
	.p2align	4, 0x90
_initializer:
	movq	_X(%rip), %rsi
	addq	$1, %rsi
	movq	%rsi, _X(%rip)
	movq	_Y(%rip), %rdx
	addq	$1, %rdx
	movq	%rdx, _Y(%rip)
	leaq	L_.str(%rip), %rdi
	xorl	%eax, %eax
	jmp	_printf

	.section	__TEXT,__cstring,cstring_literals
L_.str:
	.asciz	"X = %zu, Y = %zu\n"

	.globl	_X                              ## @X
.zerofill __DATA,__common,_X,8,3
	.section	__DATA,__data
	.globl	_Y                              ## @Y
	.p2align	3
_Y:
	.quad	1                               ## 0x1

	.section	__DATA,__mod_init_func,mod_init_funcs
	.p2align	3
	.quad	_initializer

.subsections_via_symbols
