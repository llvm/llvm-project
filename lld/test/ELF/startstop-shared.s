// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld2 %t.o -o %t.so -shared
// RUN: llvm-readobj -r -t %t.so | FileCheck  %s

        .quad __start_foo
        .section foo,"a"
// By default the symbol is visible and we need a dynamic reloc.
// CHECK:  R_X86_64_64 __start_foo 0x0

        .hidden __start_bar
        .quad __start_bar
        .section bar,"a"
// Test that we are able to hide the symbol.
// CHECK:      R_X86_64_RELATIVE - 0x[[ADDR:.*]]

// CHECK:      Name: __start_bar
// CHECK-NEXT: Value: 0x[[ADDR]]
