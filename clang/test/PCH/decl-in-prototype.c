// Test that we serialize the enum decl in the function prototype somehow.
// These decls aren't serialized quite the same way as parameters.

// Test this without pch.
// RUN: %clang_cc1 -include %s -emit-llvm -o - %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define {{.*}}i32 @main()
// CHECK:   ret i32 1

#ifndef HEADER
#define HEADER

static inline __attribute__((always_inline)) int f(enum { x, y } p) {
  return y;
}

#else

int main(void) {
  return f(0);
}

#endif
