// Test without serialization:
// RUN: %clang_cc1 %s -ast-dump | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

// Veryify the ordering of the address_space attribute still comes before the
// type whereas other attributes are still printed after.

template <int I>
void func() {
  // CHECK: VarDecl {{.*}} x '__attribute__((address_space(1))) int *'
  __attribute__((address_space(1))) int *x;

  // CHECK: VarDecl {{.*}} a 'int * __attribute__((noderef))'
  int __attribute__((noderef)) * a;

  // CHECK: VarDecl {{.*}} y '__attribute__((address_space(2))) int *'
  __attribute__((address_space(I))) int *y;

  // CHECK: VarDecl {{.*}} z '__attribute__((address_space(3))) int *'
  [[clang::address_space(3)]] int *z;
}

void func2() {
  func<2>();
}
