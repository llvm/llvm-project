// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

template <typename T> static int Foo(T t);
template <typename T>
int Foo(T t) {
  return t;
}
template<> int Foo<int>(int i) {
  return i;
}

// CHECK-NOT: define
