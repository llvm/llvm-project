// RUN: %clang_cc1 -fopenmp -x c++ -std=c++20 -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++20 -triple nvptx64-unknown-unknown -aux-triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -fsyntax-only

// Unsupported types are diagnosed in runtime code, not in
// constant-evaluated code.

constexpr long double half = 0.5L; // expected-note {{'half' defined here}}
template <int N> struct S { static constexpr int value = N; };
constexpr float bar(long double f) { return f; }

void foo() {
#pragma omp target
  {
    constexpr float x = (float)half;
    constexpr auto z = bar(1.1);
    int a = S<(int)(half * 2)>::value;
    float y = (float)half; // expected-error {{'half' requires 128 bit size 'const long double' type support, but target 'nvptx64-unknown-unknown' does not support it}}
  }
}
