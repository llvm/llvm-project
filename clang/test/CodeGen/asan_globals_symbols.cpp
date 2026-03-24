// RUN: %clang_cc1 -S -x c++ -std=c++11 -triple x86_64-linux \
// RUN:   -fsanitize=address -o %t.out %s
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-A

// Verify that alignment comes before both symbols
// CHECK-A:      .globl myGlobal
// CHECK-A-NEXT: .p2align
// CHECK-A-NEXT: myGlobal:
// CHECK-A-NEXT: .size   myGlobal, 4
// CHECK-A-NEXT: myGlobal__sanitized_padded_global:
// CHECK-A:      .size   myGlobal__sanitized_padded_global, 32

int myGlobal;

int main() {
    myGlobal = 0;
    return 0;
}
