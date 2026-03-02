// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECKC
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -x c++ -o - %s | FileCheck %s -check-prefix=CHECKCXX
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -verify %s

void __attribute__((aarch64_sve_pcs)) f(int *); // expected-warning {{'aarch64_sve_pcs' calling convention is not supported for this target}}

// CHECKC: define{{.*}} void @g(
// CHECKCXX: define{{.*}} void @_Z1gPi(
void g(int *a) {

  // CHECKC: call aarch64_sve_vector_pcs void @f(
  // CHECKCXX: call aarch64_sve_vector_pcs void @_Z1fPi
  f(a);
}

// CHECKC: declare aarch64_sve_vector_pcs void @f(
// CHECKCXX: declare aarch64_sve_vector_pcs void @_Z1fPi

void __attribute__((aarch64_sve_pcs)) h(int *a) { // expected-warning {{'aarch64_sve_pcs' calling convention is not supported for this target}}
                                                  // CHECKC: define{{.*}} aarch64_sve_vector_pcs void @h(
                                                  // CHECKCXX: define{{.*}} aarch64_sve_vector_pcs void @_Z1hPi(
  f(a);
}
