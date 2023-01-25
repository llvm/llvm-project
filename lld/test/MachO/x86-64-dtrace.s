# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/x86_64-dtrace.s -o %t/x86_64-dtrace.o
# RUN: %lld -arch x86_64 -o %t/x86_64-dtrace %t/x86_64-dtrace.o

## If references of dtrace symbols are handled by lld, their relocation should be replaced with the following instructions
# RUN: llvm-objdump --macho -D %t/x86_64-dtrace | FileCheck %s --check-prefix=CHECK

# CHECK:      33 c0                   xorl    %eax, %eax
# CHECK-NEXT: 90                      nop
# CHECK-NEXT: 90                      nop
# CHECK-NEXT: 90                      nop

# CHECK:      90                      nop
# CHECK-NEXT: 0f 1f 40 00             nopl    (%rax)

#--- x86_64-dtrace.s
	.globl	_main
_main:
	callq	___dtrace_isenabled$Foo$added$v1
	.reference	___dtrace_typedefs$Foo$v2
	callq	___dtrace_probe$Foo$added$v1$696e74
	.reference	___dtrace_stability$Foo$v1$1_1_0_1_1_0_1_1_0_1_1_0_1_1_0
	retq

.subsections_via_symbols
