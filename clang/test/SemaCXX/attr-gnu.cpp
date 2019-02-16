// RUN: %clang_cc1 -std=gnu++17 -fsyntax-only -fms-compatibility -verify %s

void f() {
  // GNU-style attributes are prohibited in this position.
  auto P = new int * __attribute__((vector_size(8))); // expected-error {{an attribute list cannot appear here}} \
                                                      // expected-error {{invalid vector element type 'int *'}}

  // Ensure that MS type attribute keywords are still supported in this
  // position.
  auto P2 = new int * __sptr; // Ok
}

void g(int a[static [[]] 5]); // expected-error {{static array size is a C99 feature, not permitted in C++}}

namespace {
class B {
public:
  virtual void test() {}
  virtual void test2() {}
  virtual void test3() {}
};

class D : public B {
public:
  void test() __attribute__((deprecated)) final {} // expected-warning {{GCC does not allow an attribute in this position on a function declaration}}
  void test2() [[]] override {} // Ok
  void test3() __attribute__((cf_unknown_transfer)) override {} // Ok, not known to GCC.
};
}

template<typename T>
union Tu { T b; } __attribute__((transparent_union)); // expected-warning {{'transparent_union' attribute ignored}}

template<typename T>
union Tu2 { int x; T b; } __attribute__((transparent_union)); // expected-warning {{'transparent_union' attribute ignored}}

union Tu3 { int x; } __attribute((transparent_union)); // expected-warning {{'transparent_union' attribute ignored}}

void tuTest1(Tu<int> u); // expected-note {{candidate function not viable: no known conversion from 'int' to 'Tu<int>' for 1st argument}}
void tuTest2(Tu3 u); // expected-note {{candidate function not viable: no known conversion from 'int' to 'Tu3' for 1st argument}}
void tu() {
  int x = 2;
  tuTest1(x); // expected-error {{no matching function for call to 'tuTest1'}}
  tuTest2(x); // expected-error {{no matching function for call to 'tuTest2'}}
}

[[gnu::__const__]] int f2() { return 12; }
[[__gnu__::__const__]] int f3() { return 12; }
[[using __gnu__ : __const__]] int f4() { return 12; }

static_assert(__has_cpp_attribute(gnu::__const__));
static_assert(__has_cpp_attribute(__gnu__::__const__));
