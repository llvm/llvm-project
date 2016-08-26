# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -t %t | FileCheck %s
# CHECK: 0000000000010040 *ABS* 00000000 .hidden __ehdr_start

.text
.global _start, __ehdr_start
_start:
	.quad __ehdr_start
