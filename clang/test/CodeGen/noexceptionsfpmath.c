// RUN: %clang_cc1 -S %s -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: main
// CHECK: attributes #0 = {{.*}}"no-trapping-math"="true"{{.*}}

int main() {
  return 0;
}
