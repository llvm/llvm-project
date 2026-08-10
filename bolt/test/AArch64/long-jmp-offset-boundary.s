# This test checks long call negative offset boundary(0x8000000) for aarch64.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostartfiles -fuse-ld=lld -Wl,-q \
# RUN:   -Wl,--script=%p/Inputs/long-jmp-offset-boundary.ld
# RUN: llvm-bolt %t.exe -o %t.bolt.exe -skip-funcs="foo.*"
# RUN: llvm-objdump -d -j .text --print-imm-hex %t.bolt.exe | FileCheck %s

# The default alignment of the new program header table and the new text is
# HugePageSize(2MB).
# CHECK: [[#%x,ADDR:]]: [[#]]     	bl
# CHECK-SAME: 	0x[[#ADDR-0x8000000]] <foo>

	.text
	.section	foo_section,"ax",@progbits
	.globl	foo
	.type	foo,@function
foo:
	ret
	.size	foo, .-foo

	.section	main_section,"ax",@progbits
	.globl	_start
	.type	_start,@function
_start:
	bl	foo
	ret
	.size	_start, .-_start
