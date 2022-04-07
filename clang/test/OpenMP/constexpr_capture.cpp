// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-linux -S -emit-llvm %s -o - -std=c++11 2>&1 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++  -fopenmp-targets=x86_64-pc-linux-gnu -triple powerpc64le-unknown-linux -S -emit-llvm %s -o - -std=c++11 2>&1 | FileCheck %s
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

constexpr double h_chebyshev_coefs[] = {
    1.0000020784639703, 0.0021491446496202074};

void test(double *d_value)
{
#pragma omp target map(tofrom                          \
                       : d_value [0:1]) map(always, to \
                                            : h_chebyshev_coefs [0:2])
  *d_value = h_chebyshev_coefs[1];  return;
}

// CHECK: void @__omp_offloading_{{.+}}test{{.+}}(double* %0)

int main() {
#pragma omp target
  V v;
  return 0;
}

// CHECK: call void @__omp_offloading_{{.+}}_main_{{.+}}()
