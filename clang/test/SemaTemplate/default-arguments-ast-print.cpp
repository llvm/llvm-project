// RUN: %clang_cc1 -ast-print %s | FileCheck %s

template <typename T, typename U = double> class Foo;

template <> class Foo<int, double> { int method1(); };

using int_type = int;

int Foo<int_type, double>::method1() {
  // CHECK: int Foo<int_type, double>::method1()
  return 10;
}

int test_typedef() {
  typedef Foo<int, double> TypedefArg;
  // CHECK: typedef Foo<int, double> TypedefArg;
  return 10;
}

int test_typedef2() {
  typedef Foo<int> TypedefArg;
  // CHECK: typedef Foo<int> TypedefArg;
  return 10;
}
