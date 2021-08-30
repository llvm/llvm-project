// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-unknown-unknown -S -ffreestanding -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=i386-unknown-unknown -S -ffreestanding -o - | FileCheck %s

#include <immintrin.h>

// The ebx may be use for base pointer, we need to restore it in time.
void ssc_mark() {
// CHECK-LABEL: ssc_mark
// CHECK: #APP
// CHECK: movl    %ebx, %eax
// CHECK: movl    $0, %ebx
// CHECK: .byte   100
// CHECK: .byte   103
// CHECK: .byte   144
// CHECK: movl    %eax, %ebx
// CHECK: #NO_APP

  __SSC_MARK(0x0);
}
