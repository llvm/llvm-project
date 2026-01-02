# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %tout
# RUN: llvm-objdump -d %tout | FileCheck %s

	.global foo
	.local bar
	.global __llvm_prefetch_target_bar

	.section .text.foo,"ax",%progbits
foo:
# CHECK:      prefetchit1 0x7(%rip)
	prefetchit1 __llvm_prefetch_target_bar(%rip)
# CHECK-NEXT: prefetchit1 (%rip)
	prefetchit1 __llvm_prefetch_target_baz(%rip)

	.section .text.bar,"ax",%progbits
bar:
__llvm_prefetch_target_bar:
        nop

