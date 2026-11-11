// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -ast-print %s | FileCheck %s

union U {
  int a;
  float b;
  U &operator=(const U &) = default;
  U &operator=(U &&) = default;
};

void odr_use(U &x, const U &y, U &&z) {
  x = y;
  x = static_cast<U &&>(z);
}

// The synthesized memcpy body must not leak into -ast-print.

// CHECK: union U {
// CHECK: U &operator=(const U &) noexcept = default;
// CHECK: U &operator=(U &&) noexcept = default;
// CHECK-NOT: __builtin_memcpy
// CHECK-NOT: (void *)
