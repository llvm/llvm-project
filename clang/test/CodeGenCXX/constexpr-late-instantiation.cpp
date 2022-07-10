// Make sure foo is instantiated and we don't get a link error
// RUN: %clang_cc1 -S -emit-llvm -triple %itanium_abi_triple %s -o- | FileCheck %s

template <typename T>
constexpr T foo(T a);

// CHECK-LABEL: define {{.*}} @main
int main() {
  // CHECK: call {{.*}} @_Z3fooIiET_S0_
  int k = foo<int>(5);
}
// CHECK: }

template <typename T>
constexpr T foo(T a) {
  return a;
}
