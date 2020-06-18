// RUN: %clang_cc1 -S -fno-strict-float-cast-overflow %s -emit-llvm -o - | FileCheck %s --check-prefix=NOSTRICT
// NOSTRICT-LABEL: main
// NOSTRICT: attributes #0 = {{.*}}"strict-float-cast-overflow"="false"{{.*}}

// The workaround attribute is not applied by default.

// RUN: %clang_cc1 -S %s -emit-llvm -o - | FileCheck %s --check-prefix=STRICT
// STRICT-LABEL: main
// STRICT-NOT: strict-float-cast-overflow

int main() {
  return 0;
}

