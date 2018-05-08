// RUN: clang-tidy -enable-check-profile -checks='-*,readability-function-size' %s 2>&1 | FileCheck --match-full-lines -implicit-check-not='{{warning:|error:}}' %s

// CHECK: ===-------------------------------------------------------------------------===
// CHECK-NEXT: {{.*}}  --- Name ---
// CHECK-NEXT: {{.*}}  readability-function-size
// CHECK-NEXT: {{.*}}  Total
// CHECK-NEXT: ===-------------------------------------------------------------------------===

// CHECK-NOT: ===-------------------------------------------------------------------------===
// CHECK-NOT: {{.*}}  --- Name ---
// CHECK-NOT: {{.*}}  readability-function-size
// CHECK-NOT: {{.*}}  Total
// CHECK-NOT: ===-------------------------------------------------------------------------===

class A {
  A() {}
  ~A() {}
};
