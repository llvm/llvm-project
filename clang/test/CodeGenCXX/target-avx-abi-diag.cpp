// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm -disable-llvm-passes -verify=noavx,noavx512 -o - %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -target-feature +avx -emit-llvm -disable-llvm-passes -verify=noavx512 -o - %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -target-feature +avx512f -emit-llvm -disable-llvm-passes -verify=both -o - %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-psabi -emit-llvm -disable-llvm-passes -verify=suppressed -o - %s
// REQUIRES: x86-registered-target

// both-no-diagnostics
// suppressed-no-diagnostics

typedef float v8f __attribute__((vector_size(32)));
typedef float v16f __attribute__((vector_size(64)));

// Prototype-only declarations do not warn.
v8f proto256(v8f);
v16f proto512(v16f);

// noavx-warning@+2 {{AVX vector return of type 'v8f' (vector of 8 'float' values) without 'avx' enabled changes the ABI}}
// noavx-warning@+1 {{AVX vector argument of type 'v8f' (vector of 8 'float' values) without 'avx' enabled changes the ABI}}
v8f def256(v8f x) {
  return x;
}

// noavx512-warning@+2 {{AVX vector return of type 'v16f' (vector of 16 'float' values) without 'avx512f' enabled changes the ABI}}
// noavx512-warning@+1 {{AVX vector argument of type 'v16f' (vector of 16 'float' values) without 'avx512f' enabled changes the ABI}}
v16f def512(v16f x) {
  return x;
}

// Internal definitions do not warn
static v8f internal_def256(v8f x) {
  return x;
}

__attribute__((target("avx")))
v8f def256_avx(v8f x) {
  return x;
}

__attribute__((target("avx512f")))
v16f def512_avx512(v16f x) {
  return x;
}

// Check to make sure deduced return lambdas still have valid source location which allows these pragmas to work
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpsabi"
template <class T>
T template_lambda_return() {
  return []() { return T{}; }();
}

v8f use_template_lambda_return() {
  return template_lambda_return<v8f>();
}
#pragma clang diagnostic pop
