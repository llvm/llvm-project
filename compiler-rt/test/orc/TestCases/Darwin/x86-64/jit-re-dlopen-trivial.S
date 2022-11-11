// Test that __orc_rt_macho_jit_dlopen and __orc_rt_macho_jit_dlclose work as
// expected when dlopen and dlclose are run twice in sequence (i.e. dlopen,
// dlclose, dlopen, dlclose) on a testcase containing simple stateless
// constructors and destructors (in this case they just print "constructor" and
// "destructor" respectively).
//
// RUN: %clang -c -o %t.main.o %p/Inputs/dlopen-dlclose-x2.S
// RUN: %clang -c -o %t.inits.o %s
// RUN: %llvm_jitlink \
// RUN:   -alias _dlopen=___orc_rt_macho_jit_dlopen \
// RUN:   -alias _dlclose=___orc_rt_macho_jit_dlclose \
// RUN:   %t.main.o -jd inits %t.inits.o -lmain | FileCheck %s

// CHECK: entering main
// CHECK-NEXT: constructor
// CHECK-NEXT: destructor
// CHECK-NEXT: constructor
// CHECK-NEXT: destructor
// CHECK-NEXT: leaving main

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0	sdk_version 12, 0, 1
	.globl	_deinitializer
	.p2align	4, 0x90
_deinitializer:
	pushq	%rbp
	movq	%rsp, %rbp
	leaq	L_str.2(%rip), %rdi
	popq	%rbp
	jmp	_puts

	.section	__TEXT,__StaticInit,regular,pure_instructions
	.p2align	4, 0x90
_initializer:
	pushq	%rbp
	movq	%rsp, %rbp
	leaq	L_str(%rip), %rdi
	callq	_puts
	movq	_deinitializer@GOTPCREL(%rip), %rdi
	leaq	_I(%rip), %rsi
	leaq	___dso_handle(%rip), %rdx
	popq	%rbp
	jmp	___cxa_atexit

	.globl	_I
.zerofill __DATA,__common,_I,1,0
	.section	__DATA,__mod_init_func,mod_init_funcs
	.p2align	3
	.quad	_initializer
	.section	__TEXT,__cstring,cstring_literals
L_str:
	.asciz	"constructor"

L_str.2:
	.asciz	"destructor"

.subsections_via_symbols
