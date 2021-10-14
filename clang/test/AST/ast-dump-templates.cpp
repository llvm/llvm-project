// RUN: %clang_cc1 -std=c++1z -ast-print %s > %t
// RUN: FileCheck < %t %s -check-prefix=CHECK1
// RUN: FileCheck < %t %s -check-prefix=CHECK2
// RUN: %clang_cc1 -std=c++1z -ast-dump %s | FileCheck --check-prefix=DUMP %s

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

// Template definition - foo
// CHECK1: template <int X, typename Y, int Z = 5> struct foo {
// CHECK2: template <int X, typename Y, int Z = 5> struct foo {

// Template instantiation - foo
// Since the order of instantiation may vary during runs, run FileCheck twice
// to make sure each instantiation is in the correct spot.
// CHECK1: template<> struct foo<5, int, 5> {
// CHECK2: template<> struct foo<2, double, 3> {

// Template definition - bar
// CHECK1: template <int A, typename B> B bar()
// CHECK2: template <int A, typename B> B bar()

// Template instantiation - bar
// CHECK1: template<> int bar<5, int>()
// CHECK2: template<> int bar<5, int>()

// CHECK1-LABEL: template <typename ...T> struct A {
// CHECK1-NEXT:    template <T ...x[3]> struct B {
template <typename ...T> struct A {
  template <T ...x[3]> struct B {};
};

// CHECK1-LABEL: template <typename ...T> void f(T ...[3]) {
// CHECK1-NEXT:    A<T[3]...> a;
template <typename ...T> void f(T ...[3]) {
  A<T[3]...> a;
}

namespace test2 {
void func(int);
void func(float);
template<typename T>
void tmpl() {
  func(T());
}

// DUMP: UnresolvedLookupExpr {{.*}} <col:3> '<overloaded function type>' lvalue (ADL) = 'func'
}

namespace test3 {
  template<typename T> struct A {};
  template<typename T> A(T) -> A<int>;
  // CHECK1: template <typename T> A(T) -> A<int>;
}
