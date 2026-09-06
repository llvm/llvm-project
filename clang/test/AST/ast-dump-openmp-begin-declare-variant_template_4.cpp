// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++ | FileCheck %s
// expected-no-diagnostics

template <typename A, typename B>
int template_number_mismatch_1() {
  return 0;
}

template <typename A, typename B>
int template_number_mismatch_2() {
  return 1;
}

#pragma omp begin declare variant match(implementation = {extension(allow_templates)})
template <typename Q>
int template_number_mismatch_1() {
  return 2;
}
template <typename Q>
int template_number_mismatch_2() {
  return 0;
}
#pragma omp end declare variant

int test() {
  // Should return 0.
  return template_number_mismatch_1<int, float>() + template_number_mismatch_2<double>();
}

// CHECK: call {{.*}} @_Z26template_number_mismatch_1IifEiv
// CHECK: call {{.*}} @"_Z61template_number_mismatch_2$ompvariant$S4$s12$Pallow_templatesIdEiv"
