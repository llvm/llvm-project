// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -o %t.o -O2 -flto=thin -fexperimental-new-pass-manager -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o

// Test to ensure the slp vectorize codegen option is passed down to the
// ThinLTO backend. -vectorize-slp is a cc1 option and will be added
// automatically when O2/O3/Os/Oz is available for clang. Also check that
// "-mllvm -vectorize-slp=false" will disable slp vectorization, overriding
// the cc1 option.
//
// Check both the new and old PMs.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -vectorize-slp -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=SLP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -vectorize-slp -mllvm -vectorize-slp=false -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=NOSLP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -vectorize-slp -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=SLP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -vectorize-slp -mllvm -vectorize-slp=false -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s --check-prefix=NOSLP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -vectorize-slp -o - -x ir %t.o -fthinlto-index=%t.thinlto.bc -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=NOSLP
// SLP: extractelement
// NOSLP-NOT: extractelement

int foo(double *A, int n, int m) {
  double sum = 0, v1 = 2, v0 = 3;
  for (int i=0; i < n; ++i)
    sum += 7*A[i*2] + 7*A[i*2+1];
  return sum;
}

