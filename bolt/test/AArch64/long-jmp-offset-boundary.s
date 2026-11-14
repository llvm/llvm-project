# This test checks long call negative offset boundary(0x8000000) for aarch64.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostartfiles -fuse-ld=lld -Wl,-q \
# RUN:   -Wl,--script=%p/Inputs/long-jmp-offset-boundary.ld
# RUN: llvm-bolt %t.exe -o %t.legacy.bolt.exe -skip-funcs="foo.*" \
# RUN:   --relax-with-clusters=false
# RUN: llvm-objdump -d -j .text --print-imm-hex %t.legacy.bolt.exe \
# RUN:   | FileCheck %s --check-prefix=LEGACY
# RUN: llvm-bolt %t.exe -o %t.cluster.bolt.exe -skip-funcs="foo.*" \
# RUN:   --relax-with-clusters
# RUN: llvm-objdump -d -j .text --print-imm-hex %t.cluster.bolt.exe \
# RUN:   | FileCheck %s --check-prefix=CLUSTER

# The default alignment of the new program header table and the new text is
# HugePageSize(2MB).
# LEGACY: [[#%x,ADDR:]]: [[#]]     	bl
# LEGACY-SAME: 	0x[[#ADDR-0x8000000]] <foo>

# Clustered relaxation uses a long thunk for the skipped foo, which is outside
# the clustered layout.
# CLUSTER: [[#%x,ADDR:]]: [[#]]     	bl
# CLUSTER-SAME: 	0x[[#ADDR+8]] <[[THUNK:__AArch64_forward_ADRPThunk_foo_[0-9]+]]>
# CLUSTER: <[[THUNK]]>:
# CLUSTER-NEXT: {{[0-9a-f]+}}: {{[0-9a-f]+}} 	adrp x16, 0x[[#ADDR-0x8000000]] <foo>
# CLUSTER-NEXT: {{[0-9a-f]+}}: {{[0-9a-f]+}} 	add x16, x16, #0x0
# CLUSTER-NEXT: {{[0-9a-f]+}}: {{[0-9a-f]+}} 	br x16

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
