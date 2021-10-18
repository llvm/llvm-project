// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

template<typename ...Types>
int get_num_types(Types...) {
  return sizeof...(Types);
}

// CHECK-LABEL: define weak_odr i32 @_Z13get_num_typesIJifdEEiDpT_
// CHECK: ret i32 3
template int get_num_types(int, float, double);

// PR10260 - argument packs that expand to nothing
namespace test1 {
  template <class... T> void foo() {
    int values[sizeof...(T)+1] = { T::value... };
    // CHECK-LABEL: define linkonce_odr void @_ZN5test13fooIJEEEvv()
    // CHECK: alloca [1 x i32], align 4
  }

  void test() {
    foo<>();
  }
}
