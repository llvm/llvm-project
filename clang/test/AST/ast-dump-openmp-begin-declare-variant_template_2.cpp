// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++ | FileCheck %s
// expected-no-diagnostics

template <typename T>
int also_before(T) {
  return 1;
}
template <int V>
int also_before_mismatch(void) {
  return 0;
}
int also_before_non_template(void) {
  return 0;
}

#pragma omp begin declare variant match(implementation = {extension(allow_templates)})
template <typename T>
int also_before(T) {
  return 0;
}
template <typename T>
int also_after(T) {
  return 0;
}
template <typename T, typename Q>
int also_after_mismatch(T, Q) {
  return 2;
}
template <typename T>
int also_before_mismatch(T) {
  return 3;
}
template <typename T>
int also_before_non_template(T) {
  return 4;
}
template <int V>
int only_def(void) {
  return 0;
}
#pragma omp end declare variant

template <typename T>
int also_after(T) {
  return 6;
}
template <typename T>
int also_after_mismatch(T) {
  return 0;
}

int test() {
  // Should return 0.
  return also_before(0.) + also_before_mismatch<0>() + also_before_non_template() + also_after<char>(0) + also_after_mismatch(0) + only_def<0>();
}

// CHECK: call {{.*}} @"_Z46also_before$ompvariant$S4$s12$Pallow_templatesIdEiT_"
// CHECK: call {{.*}} @_Z20also_before_mismatchILi0EEiv
// CHECK: call {{.*}} @_Z24also_before_non_templatev
// CHECK: call {{.*}} @"_Z45also_after$ompvariant$S4$s12$Pallow_templatesIcEiT_"
// CHECK: call {{.*}} @_Z19also_after_mismatchIiEiT_
// CHECK: call {{.*}} @"_Z43only_def$ompvariant$S4$s12$Pallow_templatesILi0EEiv"
