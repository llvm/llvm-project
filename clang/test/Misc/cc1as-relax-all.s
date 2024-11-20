// REQUIRES: x86-registered-target
// RUN: %clang -cc1as -triple x86_64 -filetype obj -mrelax-all %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s

// CHECK:      <.text>:
// CHECK-NEXT:   0: e9 06 00 00 00                jmp     0xb <foo>
// CHECK-NEXT:   5: 0f 84 00 00 00 00             je      0xb <foo>
// CHECK-EMPTY:

jmp foo
je foo

foo: ret
