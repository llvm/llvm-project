// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld %t.o %t.o -o %t -shared
// RUN: llvm-readobj -r %t | FileCheck %s

        .section        bar,"axG",@progbits,foo,comdat
        .cfi_startproc
        .cfi_personality 0x8c, foo
        .cfi_endproc

// CHECK:      Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:   0x1DA R_X86_64_64 foo 0x0
// CHECK-NEXT: }
