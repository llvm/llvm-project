# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { }" > %t.script
# RUN: not ld.lld %t.o -script %t.script -o %t 2>&1 | FileCheck %s
# CHECK: error: undefined symbol: __ehdr_start
# CHECK: >>> referenced by {{.*}}:(.text+0x0)

.text
.global _start, __ehdr_start
_start:
	.quad __ehdr_start
