// RUN: %clangxx_nsan -O0 -g %s -o %t
// RUN: NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=19 %run %t 2>&1 | FileCheck %s
#include<iostream>
#include<vector>
#include<algorithm>
#include<cmath>

__attribute__((noinline)) void softmax(std::vector<double> &values) {
    double sum_exp = 0.0;
    for (auto &i: values) {
      i = std::exp(i);
      sum_exp += i;
    }
    for (auto &i: values) {
      i /= sum_exp;
    }
}

__attribute__((noinline)) void stable_softmax(std::vector<double> values) {
  double sum_exp = 0.0;
  double max_values = *std::max_element(values.begin(), values.end());
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
  stable_softmax(data);
  for (auto i:data) {
    printf("%f ", i);
  }
  printf("\n");
  data = {1000, 1001, 1002};
  softmax(data);
  for (auto i:data) {
    printf("%f", i);
    // CHECK: WARNING: NumericalStabilitySanitizer: NaN detected
  }
  printf("\n");
}