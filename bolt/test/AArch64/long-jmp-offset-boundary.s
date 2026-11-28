# This test checks long call negative offset boundary(0x8000000) for aarch64.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostartfiles -fuse-ld=lld -Wl,-q \
# RUN:   -Wl,--script=%p/Inputs/long-jmp-offset-boundary.ld

# RUN: llvm-bolt %t.exe -o %t.bolt.exe -skip-funcs="foo.*" \
# RUN:   --relax-with-clusters=false
# RUN: llvm-objdump -d --print-imm-hex %t.bolt.exe \
# RUN:   | FileCheck %s --check-prefixes=CHECK,LEGACY

# RUN: llvm-bolt %t.exe -o %t.bolt.exe -skip-funcs="foo.*"
# RUN: llvm-objdump -d --print-imm-hex %t.bolt.exe \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CLUSTER

# The default alignment of the new program header table and the new text is
# HugePageSize(2MB).

# CHECK: Disassembly of section .bolt.org.text:
#
# CHECK:      {{.*}} <foo>:
# CHECK-NEXT: {{.*}}   ret
#
# CHECK: Disassembly of section .text:
#
# CHECK: [[#%x,ADDR:]] <_start>:
#
# LEGACY-NEXT:  [[#ADDR]]: {{.*}} bl 0x[[#ADDR-0x8000000]] <foo>
#
# CLUSTER-NEXT: [[#ADDR]]: {{.*}} bl {{.*}} <[[THUNK:__AArch64_forward_ADRPThunk_foo_[0-9]+]]>
#
# CLUSTER:      {{.*}} <[[THUNK]]>:
# CLUSTER-NEXT: {{.*}}   adrp x16, 0x[[#ADDR-0x8000000]] <foo>
# CLUSTER-NEXT: {{.*}}   add x16, x16, #0x0
# CLUSTER-NEXT: {{.*}}   br x16

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
