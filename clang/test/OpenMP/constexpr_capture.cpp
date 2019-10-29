// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-linux -S -emit-llvm %s -o - -std=c++11 2>&1 | FileCheck %s
// expected-no-diagnostics

template <int __v> struct integral_constant {
  static constexpr int value = __v;
};

template <typename _Tp, int v = 0, bool _IsArray = integral_constant<v>::value>
struct decay {
  typedef int type;
};
struct V {
  template <typename TArg0 = int, typename = typename decay<TArg0>::type> V();
};
int main() {
#pragma omp target
  V v;
  return 0;
}

// CHECK: call void @__omp_offloading_{{.+}}_main_l16()
