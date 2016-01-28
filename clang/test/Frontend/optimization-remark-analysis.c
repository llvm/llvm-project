// RUN: %clang -O1 -fvectorize -target x86_64-unknown-unknown -emit-llvm -Rpass-analysis -S %s -o - 2>&1 | FileCheck %s --check-prefix=RPASS
// RUN: %clang -O1 -fvectorize -target x86_64-unknown-unknown -emit-llvm -S %s -o - 2>&1 | FileCheck %s

// RPASS: {{.*}}:21:1: remark: loop not vectorized: loop contains a switch statement
// CHECK-NOT: {{.*}}:21:1: remark: loop not vectorized: loop contains a switch statement

double foo(int N, int *Array) {
  double v = 0.0;

  #pragma clang loop vectorize(enable)
  for (int i = 0; i < N; i++) {
    switch(Array[i]) {
    case 0: v += 1.0f; break;
    case 1: v -= 0.5f; break;
    case 2: v *= 2.0f; break;
    default: v = 0.0f;
    }
  }

  return v;
}
