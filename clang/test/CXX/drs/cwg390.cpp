// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify

namespace cwg390 { // cwg390: 3.3
  template<typename T>
  struct A {
    A() { f(); }
    // expected-warning@-1 {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the constructor of 'cwg390::A<int>'}}
    //   expected-note@#cwg390-A-int {{in instantiation of member function 'cwg390::A<int>::A' requested here}}
    //   expected-note@#cwg390-f {{'f' declared here}}
    virtual void f() = 0; // #cwg390-f
    virtual ~A() = 0;
  };
  template<typename T> A<T>::~A() { T::error; }
  // expected-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
  //   expected-note@#cwg390-A-int {{in instantiation of member function 'cwg390::A<int>::~A' requested here}}
  template<typename T> void A<T>::f() { T::error; } // ok, not odr-used
  struct B : A<int> { // #cwg390-A-int
    void f() {}
  } b;
} // namespace cwg390
