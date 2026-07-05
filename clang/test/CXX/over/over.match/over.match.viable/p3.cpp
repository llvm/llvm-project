// RUN:  %clang_cc1 -std=c++2a -verify %s

struct S2 {};
// expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'S1<int>' to 'const S2' for 1st argument}}
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'S1<int>' to 'S2' for 1st argument}}
// expected-note@-3 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

template<typename T>
struct S1 {
  void foo() const requires true {}
  void foo() const requires false {}
  void bar() const requires false {}
  // expected-note@-1 {{because 'false' evaluated to false}}
  operator bool() const requires true { return true; }
  explicit operator bool() const requires false;
  explicit operator S2() const requires false;
  // expected-note@-1 {{candidate function not viable: constraints not satisfied}}
  // expected-note@-2 {{because 'false' evaluated to false}}
};

void foo() {
  S1<int>().foo();
  S1<int>().bar();
  // expected-error@-1 {{invalid reference to function 'bar': constraints not satisfied}}
  (void) static_cast<bool>(S1<int>());
  (void) static_cast<S2>(S1<int>());
  // expected-error@-1 {{no matching conversion for static_cast from 'S1<int>' to 'S2'}}
}

// Test that constraints are checked before implicit conversions are formed.

template<typename T>
struct invalid_template { using X = typename T::non_existant; };
struct A {
  template<typename T, bool=invalid_template<T>::aadasas>
  operator T() {}
};

template<typename>
struct WrapsStatics {
  static void foo(int) requires false;
  static void foo(A) requires true;
};

template<typename T>
struct S {
  void foo(int) requires false;
  void foo(A) requires true;
  S(A) requires false;
  S(double) requires true;
  ~S() requires false;
  ~S() requires true;
  operator int() requires true;
  operator int() requires false;
};

void bar() {
  WrapsStatics<int>::foo(A{});
  S<int>{1.}.foo(A{});

  S<int> s = 1;
  int a = s;
}
