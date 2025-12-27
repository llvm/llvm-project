# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %tout
# RUN: llvm-nm -D %tout

	.global foo
	.local bar
	.global __llvm_prefetch_target_bar

	.section .text.foo,"ax",%progbits
foo:
	prefetchit1 __llvm_prefetch_target_bar(%rip)
	prefetchit1 __llvm_prefetch_target_baz(%rip)

	.section .text.bar,"ax",%progbits
bar:
__llvm_prefetch_target_bar:
        nop

