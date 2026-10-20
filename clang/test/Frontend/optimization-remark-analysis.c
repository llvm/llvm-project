// RUN: %clang -O1 -fvectorize -target x86_64-unknown-unknown -emit-llvm -Rpass-analysis -S %s -o - 2>&1 | FileCheck %s --check-prefix=RPASS
// RUN: %clang -O1 -fvectorize -target x86_64-unknown-unknown -emit-llvm -S %s -o - 2>&1 | FileCheck %s

// RPASS: {{.*}}:12:5: remark: loop not vectorized: call instruction cannot be vectorized
// CHECK-NOT: remark: loop not vectorized

void bar(void);

void foo(int N) {
  #pragma clang loop vectorize(enable)
  for (int i = 0; i < N; i++) {
    bar();
  }
}
