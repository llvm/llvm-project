// REQUIRES: x86
// RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t.obj %s
// RUN: echo -e ".globl myabsolute\nmyabsolute = 1" > %t.dupl.s
// RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t.dupl.obj %t.dupl.s
// RUN: not lld-link /out:%t.exe %t.obj %t.dupl.obj 2>&1 | FileCheck %s

// CHECK: error: duplicate symbol: myabsolute

.globl myabsolute
myabsolute = 0

.globl entry
entry:
    ret
