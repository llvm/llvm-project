// RUN: %clang -S -fsanitize=address -emit-llvm -o - -fsanitize=address %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK-NO-OUTLINE
// RUN: %clang -S -fsanitize=address -emit-llvm -o - -fsanitize=address %s \
// RUN:     -fsanitize-address-outline-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-OUTLINE

// CHECK-NO-OUTLINE-NOT: call{{.*}}@__asan_load4
// CHECK-OUTLINE: call{{.*}}@__asan_load4

int deref(int *p) {
  return *p;
}
