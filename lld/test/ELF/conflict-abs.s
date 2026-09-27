// REQUIRES: x86
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
// RUN: ld.lld a.o a.o -o a.so -shared
// RUN: llvm-readobj --dyn-symbols a.so | FileCheck %s

// CHECK:      Name: foo
// CHECK-NEXT: Value: 0x123

.global foo
foo = 0x123

// RUN: echo ".global foo; foo = 0x124" | llvm-mc -filetype=obj -triple=x86_64 - -o b.o
// RUN: not ld.lld a.o b.o -shared 2>&1 | FileCheck --check-prefix=DUP %s --implicit-check-not=error:

// DUP:      error: duplicate symbol: foo
// DUP-NEXT: >>> defined in {{.*}}
// DUP-NEXT: >>> defined in {{.*}}
