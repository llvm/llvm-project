// RUN: %clang_cc1 -triple powerpc64le -emit-llvm-bc -fopenmp %s \
// RUN:   -fopenmp-targets=powerpc64le,x86_64 -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -triple x86_64 -aux-triple powerpc64le -fopenmp \
// RUN:   -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc %s \
// RUN:   -fsyntax-only

void foo(__ibm128 x); // expected-note {{'foo' defined here}}

void loop(int n, __ibm128 *arr) {
#pragma omp target parallel
  for (int i = 0; i < n; ++i) {
    // expected-error@+1 {{'foo' requires 128 bit size '__ibm128' type support, but target 'x86_64' does not support it}}
    foo(arr[i]);
  }
}
