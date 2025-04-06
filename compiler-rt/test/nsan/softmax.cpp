// RUN: %clangxx_nsan -O0 -g -DSOFTMAX=softmax %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0,log2_max_relative_error=19 %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_nsan -O3 -g -DSOFTMAX=softmax %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=0,log2_max_relative_error=19 %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_nsan -O0 -g -DSOFTMAX=stable_softmax %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=1,log2_max_relative_error=19 %run %t 

// RUN: %clangxx_nsan -O3 -g -DSOFTMAX=stable_softmax %s -o %t
// RUN: env NSAN_OPTIONS=check_nan=true,halt_on_error=1,log2_max_relative_error=19 %run %t

#include<iostream>
#include<vector>
#include<algorithm>
#include<cmath>

// unstable softmax
template <typename T>
__attribute__((noinline)) void softmax(std::vector<T> &values) {
    T sum_exp = 0.0;
    for (auto &i: values) {
      i = std::exp(i);
      sum_exp += i;
    }
    for (auto &i: values) {
      i /= sum_exp;
    }
}

// use max value to avoid overflow
// \sigma_i exp(x_i) / \sum_j exp(x_j) = \sigma_i exp(x_i - max(x)) / \sum_j exp(x_j - max(x))
template <typename T>
__attribute__((noinline)) void stable_softmax(std::vector<T> &values) {
  T sum_exp = 0.0;
  T max_values = *std::max_element(values.begin(), values.end());
  for (auto &i: values) {
    i = std::exp(i - max_values);
    sum_exp += i;
  }
  for (auto &i:values) {
    i /= sum_exp;
  }
}

int main() {
  std::vector<double> data = {1000, 1001, 1002};
  SOFTMAX(data);
  for (auto i: data) {
    printf("%f", i);
    // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
  }
  return 0;
}
