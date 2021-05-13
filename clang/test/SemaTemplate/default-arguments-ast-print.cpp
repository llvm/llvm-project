// RUN: %clang_cc1 -fsyntax-only -ast-print %s | FileCheck %s

template <typename T, typename U = double> class Foo;

template <> class Foo<int, double> { int method1(); };

using int_type = int;

int Foo<int_type, double>::method1() {
  // CHECK: int Foo<int_type, double>::method1()
  return 10;
}
