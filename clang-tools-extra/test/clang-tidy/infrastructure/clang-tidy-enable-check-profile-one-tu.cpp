// RUN: clang-tidy -enable-check-profile -checks='-*,readability-function-size' %s -- 2>&1 | FileCheck --match-full-lines -implicit-check-not='{{warning:|error:}}' %s

// CHECK: ===-------------------------------------------------------------------------===
// CHECK-NEXT:                          clang-tidy checks profiling
// CHECK-NEXT: ===-------------------------------------------------------------------------===
// CHECK-NEXT: Total Execution Time: {{.*}} seconds ({{.*}} wall clock)

// CHECK: {{.*}}  --- Name ---
// CHECK-NEXT: {{.*}}  readability-function-size
// CHECK-NEXT: {{.*}}  Total

// CHECK-NOT: ===-------------------------------------------------------------------------===
// CHECK-NOT:                          clang-tidy checks profiling
// CHECK-NOT: ===-------------------------------------------------------------------------===
// CHECK-NOT: Total Execution Time: {{.*}} seconds ({{.*}} wall clock)

// CHECK-NOT: {{.*}}  --- Name ---
// CHECK-NOT: {{.*}}  readability-function-size
// CHECK-NOT: {{.*}}  Total

class A {
  A() {}
  ~A() {}
};
