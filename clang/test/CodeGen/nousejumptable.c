// RUN: %clang_cc1 -S -fno-jump-tables %s -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: main
// CHECK: attributes #0 = {{.*}}"no-jump-tables"="true"{{.*}}

int main(void) {
  return 0;
}
