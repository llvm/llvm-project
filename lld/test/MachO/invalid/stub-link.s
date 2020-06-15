# REQUIRES: x86

# RUN: mkdir -p %t
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-ios %s -o %t/test.o
# RUN: not lld -flavor darwinnew -arch x86_64 -o %t/test -Z -L%S/../Inputs/iPhoneSimulator.sdk/usr/lib -lSystem %t/test.o 2>&1 | FileCheck %s

# CHECK: error: undefined symbol __cache_handle_memory_pressure_event

.section __TEXT,__text
.global _main

_main:
  movq __cache_handle_memory_pressure_event@GOTPCREL(%rip), %rax
  ret
