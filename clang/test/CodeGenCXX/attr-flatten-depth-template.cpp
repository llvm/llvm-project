// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

template<typename T, unsigned I> struct S {

  __attribute__((flatten_depth(I)))
  void func1(){/*...*/}

  __attribute__((flatten_depth(T::value)))
  void func2(){/*...*/}

};

struct HasVal { static constexpr int value = 3; };

void foo() {
  S<HasVal, 2> s;
  s.func1();
  s.func2();
}

// Verify the attribute values are correct
// CHECK-DAG: flatten_depth=2
// CHECK-DAG: flatten_depth=3
