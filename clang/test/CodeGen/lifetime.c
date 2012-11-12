// RUN: %clang -S -emit-llvm -o - -O0 %s | FileCheck %s -check-prefix=O0
// RUN: %clang -S -emit-llvm -o - -O1 %s | FileCheck %s -check-prefix=O1
// RUN: %clang -S -emit-llvm -o - -O2 %s | FileCheck %s -check-prefix=O2
// RUN: %clang -S -emit-llvm -o - -O3 %s | FileCheck %s -check-prefix=O3

extern void use(char *a);

__attribute__((always_inline)) void helper_no_markers() {
  char a;
  use(&a);
}

void lifetime_test() {
// O0: lifetime_test
// O1: lifetime_test
// O2: lifetime_test
// O3: lifetime_test
// O0-NOT: @llvm.lifetime.start
// O1: @llvm.lifetime.start
// O2: @llvm.lifetime.start
// O3: @llvm.lifetime.start
  helper_no_markers();
}
