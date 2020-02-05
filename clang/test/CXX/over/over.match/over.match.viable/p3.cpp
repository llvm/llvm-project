// RUN:  %clang_cc1 -std=c++2a -verify %s

struct S2 {};
// expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'S1' to 'const S2' for 1st argument}}
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'S1' to 'S2' for 1st argument}}
// expected-note@-3 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

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
  S1().foo();
  S1().bar();
  // expected-error@-1 {{invalid reference to function 'bar': constraints not satisfied}}
  (void) static_cast<bool>(S1());
  (void) static_cast<S2>(S1());
  // expected-error@-1 {{no matching conversion for static_cast from 'S1' to 'S2'}}
}

// Test that constraints are checked before implicit conversions are formed.

template<typename T>
struct invalid_template { using X = typename T::non_existant; };
struct A {
  template<typename T, bool=invalid_template<T>::aadasas>
  operator T() {}
};

void foo(int) requires false;
void foo(A) requires true;

struct S {
  void foo(int) requires false;
  void foo(A) requires true;
  S(A) requires false;
  S(double) requires true;
  ~S() requires false;
  // expected-note@-1 2{{because 'false' evaluated to false}}
  ~S() requires true;
  operator int() requires true;
  operator int() requires false;
};

void bar() {
  foo(A{});
  S{1.}.foo(A{});
  // expected-error@-1{{invalid reference to function '~S': constraints not satisfied}}
  // Note - this behavior w.r.t. constrained dtors is a consequence of current
  // wording, which does not invoke overload resolution when a dtor is called.
  // P0848 is set to address this issue.
  S s = 1;
  // expected-error@-1{{invalid reference to function '~S': constraints not satisfied}}
  int a = s;
}