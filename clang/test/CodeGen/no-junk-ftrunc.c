// RUN: %clang_cc1 -S -ffp-cast-overflow-workaround %s -emit-llvm -o - | FileCheck %s
// CHECK-LABEL: main
// CHECK: attributes #0 = {{.*}}"fp-cast-overflow-workaround"="true"{{.*}}

// The workaround attribute is not applied by default.

// RUN: %clang_cc1 -S %s -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT
// DEFAULT-LABEL: main
// DEFAULT-NOT: fp-cast-overflow-workaround

int main() {
  return 0;
}

