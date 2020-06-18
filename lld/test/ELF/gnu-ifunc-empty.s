// REQUIRES: x86

// Verifies that .rela_iplt_{start,end} point to a dummy section
// if .rela.iplt does not exist.

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld -static %t.o -o %t.exe
// RUN: llvm-readelf -S -s %t.exe | FileCheck %s

// CHECK: Name  Type     Address          Off
// CHECK: .text PROGBITS 0000000000201120 000120
// CHECK:    Value         Size Type   Bind   Vis    Ndx Name
// CHECK: 0000000000200000    0 NOTYPE LOCAL  HIDDEN   1 __rela_iplt_end
// CHECK: 0000000000200000    0 NOTYPE LOCAL  HIDDEN   1 __rela_iplt_start

.globl _start
_start:
 movl $__rela_iplt_start, %edx
 movl $__rela_iplt_end, %edx
