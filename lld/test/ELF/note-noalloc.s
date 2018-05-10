// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-freebsd %s -o %t.o
// RUN: ld.lld %t.o -o %t -shared
// RUN: llvm-readobj -program-headers -sections %t | FileCheck %s

// PR37361: A note without SHF_ALLOC should not create a PT_NOTE program
// header (but should have a SHT_NOTE section).

// CHECK: Name: .note.tag
// CHECK: Type: SHT_NOTE
// CHECK: Name: .debug.ghc-link-info
// CHECK: Type: SHT_NOTE
// CHECK-NOT: Type: SHT_NOTE

// CHECK: Type: PT_NOTE
// CHECK-NOT: Type: PT_NOTE

        .section        .note.tag,"a",@note
        .quad 1234

        .section        .debug.ghc-link-info,"",@note
        .quad 5678
