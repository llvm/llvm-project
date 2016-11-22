// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-mc %p/Inputs/copy-rel-pie.s -o %t2.o -filetype=obj -triple=x86_64-pc-linux
// RUN: ld.lld %t2.o -o %t2.so -shared
// RUN: not ld.lld %t.o %t2.so -o %t.exe -pie 2>&1 | FileCheck %s

// CHECK: {{.*}}.o:(.text+0x0): can't create dynamic relocation R_X86_64_64 against symbol 'bar' defined in {{.*}}.so
// CHECK: {{.*}}.o:(.text+0x8): can't create dynamic relocation R_X86_64_64 against symbol 'foo' defined in {{.*}}.so

.global _start
_start:
        .quad bar
        .quad foo
