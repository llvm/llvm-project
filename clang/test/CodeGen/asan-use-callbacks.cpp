// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -fsanitize=address \
// RUN:     -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK-NO-OUTLINE
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - \
// RUN:     -fsanitize=address %s -fsanitize-address-outline-instrumentation \
// RUN:     | FileCheck %s --check-prefixes=CHECK-OUTLINE

// CHECK-NO-OUTLINE-NOT: call{{.*}}@__asan_load4
// CHECK-OUTLINE: call{{.*}}@__asan_load4

int deref(int *p) {
  return *p;
}
