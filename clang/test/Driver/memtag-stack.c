// RUN: %clang     -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag-stack -mllvm -stack-safety-print=1 %s -S -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SAFETY
// RUN: %clang -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag-stack -mllvm -stack-safety-print=1 %s -S -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SAFETY
// RUN: %clang -O2 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag-stack -mllvm -stack-safety-print=1 %s -S -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SAFETY
// RUN: %clang -O3 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag-stack -mllvm -stack-safety-print=1 %s -S -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SAFETY

// REQUIRES: aarch64-registered-target

int z;
__attribute__((noinline)) void use(int *p) { *p = z; }
int foo() {
  int x;
  use(&x);
  return x;
}

// CHECK-NO-SAFETY-NOT: allocas uses

// CHECK-SAFETY-LABEL: @foo
// CHECK-SAFETY-LABEL: allocas uses:
// CHECK-SAFETY-NEXT: [4]: [0,4)
