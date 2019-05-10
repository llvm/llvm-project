// RUN: %clang_cc1 -E %s | FileCheck %s --match-full-lines --strict-whitespace --check-prefix CHECK-1
// CHECK-1:a: x
// RUN: %clang_cc1 -E %s | FileCheck %s --match-full-lines --strict-whitespace --check-prefix CHECK-2
// CHECK-2:b: x y, z,h
// RUN: %clang_cc1 -E %s | FileCheck %s --match-full-lines --strict-whitespace --check-prefix CHECK-3
// CHECK-3:c: foo(x)

#define A(b, c...) b c
a: A(x)
b: A(x, y, z,h)

#define B(b, c...) foo(b, ## c)
c: B(x)
