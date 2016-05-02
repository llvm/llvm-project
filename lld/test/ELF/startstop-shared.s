// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readobj -r -t %t.so | FileCheck  %s

        .quad __start_foo
        .section foo,"a"
// By default the symbol is hidden.
// CHECK:      R_X86_64_RELATIVE - 0x[[ADDR1:.*]]

        .hidden __start_bar
        .quad __start_bar
        .section bar,"a"
// References do not affect the visibility.
// CHECK:      R_X86_64_RELATIVE - 0x[[ADDR2:.*]]

// CHECK:      Name: __start_bar
// CHECK-NEXT: Value: 0x[[ADDR2]]
// CHECK-NEXT: Size:
// CHECK-NEXT: Binding: Local

// CHECK:      Name: __start_foo
// CHECK-NEXT: Value: 0x[[ADDR1]]
// CHECK-NEXT: Size:
// CHECK-NEXT: Binding: Local
