// RUN: %clang_cc1 -ast-dump %s > %t
// RUN: FileCheck < %t %s -check-prefix=CHECK1
// RUN: FileCheck < %t %s -check-prefix=CHECK2

template <int X, typename Y, int Z = 5>
struct foo {
  int constant;
  foo() {}
  Y getSum() { return Y(X + Z); }
};

template <int A, typename B>
B bar() {
  return B(A);
}

void baz() {
  int x = bar<5, int>();
  int y = foo<5, int>().getSum();
  double z = foo<2, double, 3>().getSum();
}

// Template instantiation - foo
// Since the order of instantiation may vary during runs, run FileCheck twice
// to make sure each instantiation is in the correct spot.
// CHECK1: template <int X = 5, typename Y = int, int Z = 5> struct foo {
// CHECK2: template <int X = 2, typename Y = double, int Z = 3> struct foo {

// Template definition - foo
// CHECK1: template <int X, typename Y, int Z = (IntegerLiteral {{.*}} 'int' 5)
// CHECK2: template <int X, typename Y, int Z = (IntegerLiteral {{.*}} 'int' 5)

// Template instantiation - bar
// CHECK1: template <int A = 5, typename B = int> int bar()
// CHECK2: template <int A = 5, typename B = int> int bar()

// Template definition - bar
// CHECK1: template <int A, typename B> B bar()
// CHECK2: template <int A, typename B> B bar()
