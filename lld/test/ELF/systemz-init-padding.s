# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %p/Inputs/systemz-init.s -o systemz-init.o
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld -dynamic-linker /lib/ld64.so.1 %t.o systemz-init.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn -j .init %t | FileCheck %s

# glibc < 2.39 used to align .init and .fini code at a 4-byte boundary.
# When that happens, the linker must not pad the code with invalid
# instructions, e.g. null bytes.
	.section        .init,"ax",@progbits
	brasl %r14, startup

# CHECK:      <.init>:
# CHECK-NEXT: brasl %r14,
# CHECK-NEXT: nopr   %r7
# CHECK-NEXT: lg %r4, 272(%r15)

	.text
	.globl startup
	.p2align 4
startup:
	br %r14

	.globl main
	.p2align 4
main:
	br %r14
