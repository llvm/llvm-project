// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld %t.o -o %t.so -shared 2>&1 | FileCheck %s

foo:
.quad foo

// CHECK: {{.*}}.o:(.text+0x0): can't create dynamic relocation R_X86_64_64 against local symbol in readonly segment defined in {{.*}}.o
