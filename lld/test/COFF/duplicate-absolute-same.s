// REQUIRES: x86
// RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t.obj %s
// RUN: echo -e ".globl myabsolute\nmyabsolute = 0" > %t.dupl.s
// RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t.dupl.obj %t.dupl.s
// RUN: lld-link /out:%t.exe %t.obj %t.dupl.obj -subsystem:console -entry:entry 2>&1 | count 0

// This shouldn't produce any duplicate symbol error.

.globl myabsolute
myabsolute = 0

.globl entry
entry:
    ret
