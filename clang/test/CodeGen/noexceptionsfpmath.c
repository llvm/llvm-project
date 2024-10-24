// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: main
// CHECK: attributes #0 = {{.*}}"no-trapping-math"="true"{{.*}}

int main(void) {
  return 0;
}
