// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify -std=c++11 %s

// Make sure we handle contexts correctly with sizeof
template<typename T> void f(T n) {
  int buffer[n];
  [] { int x = sizeof(sizeof(buffer)); }();
}
int main() {
  f<int>(1);
}

// Make sure we handle references to non-static data members in unevaluated
// contexts in class template methods correctly. Previously we assumed these
// would be valid MemberRefExprs, but they have no 'this' so we need to form a
// DeclRefExpr to the FieldDecl instead.
// PR26893
template <class T>
struct M {
  M() {}; // expected-note {{in instantiation of default member initializer 'M<S>::m' requested here}}
  int m = *T::x; // expected-error {{invalid use of non-static data member 'x'}}
  void f() {
    // These are valid.
    static_assert(sizeof(T::x) == 8, "ptr");
    static_assert(sizeof(*T::x) == 4, "int");
  }
};
struct S { int *x; };
template struct M<S>; // expected-note {{in instantiation of member function 'M<S>::M' requested here}}

// Similar test case for PR26893.
template <typename T=void>
struct bar {
  struct foo { int array[10]; };
  int baz() { return sizeof(foo::array); }
};
template struct bar<>;
