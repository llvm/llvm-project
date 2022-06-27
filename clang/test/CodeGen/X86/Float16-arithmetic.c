// RUN: %clang_cc1 -triple  x86_64-unknown-unknown \
// RUN: -emit-llvm -o - %s  | FileCheck %s --check-prefixes=CHECK

// CHECK-NOT: fpext
// CHECK-NOT: fptrunc

_Float16 add1(_Float16 a, _Float16 b) {
  return a + b;
}

_Float16 add2(_Float16 a, _Float16 b, _Float16 c) {
  return a + b + c;
}

_Float16 div(_Float16 a, _Float16 b) {
  return a / b;
}

_Float16 mul(_Float16 a, _Float16 b) {
  return a * b;
}

_Float16 add_and_mul1(_Float16 a, _Float16 b, _Float16 c, _Float16 d) {
  return a * b + c * d;
}

_Float16 add_and_mul2(_Float16 a, _Float16 b, _Float16 c, _Float16 d) {
  return (a - 6 * b) + c;
}
