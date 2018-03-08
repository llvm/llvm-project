// RUN: %clang_cc1 -std=c++11 %s -verify -fno-spell-checking

// These test cases are constructed to make clang call ActOnStartOfFunctionDef
// with nullptr.

struct ImplicitDefaultCtor1 {};
struct Foo {
  typedef int NameInClass;
  void f();
};
namespace bar {
// FIXME: Improved our recovery to make this a redeclaration of Foo::f,
// even though this is in the wrong namespace. That will allow name lookup to
// find NameInClass below. Users are likely to hit this when they forget to
// close namespaces.
// expected-error@+1 {{cannot define or redeclare 'f' here}}
void Foo::f() {
  switch (0) { case 0: ImplicitDefaultCtor1 o; }
  // expected-error@+1 {{unknown type name 'NameInClass'}}
  NameInClass var;
}
} // namespace bar

struct ImplicitDefaultCtor2 {};
template <typename T> class TFoo { void f(); };
// expected-error@+1 {{nested name specifier 'decltype(TFoo<T>())::'}}
template <typename T> void decltype(TFoo<T>())::f() {
  switch (0) { case 0: ImplicitDefaultCtor1 o; }
}

namespace tpl2 {
struct ImplicitDefaultCtor3 {};
template <class T1> class A {
  template <class T2> class B {
    void mf2();
  };
};
template <class Y>
template <>
// expected-error@+1 {{nested name specifier 'A<Y>::B<double>::'}}
void A<Y>::B<double>::mf2() {
  switch (0) { case 0: ImplicitDefaultCtor3 o; }
}
}
