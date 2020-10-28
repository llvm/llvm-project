// RUN: %clang_cc1 -std=c++11 -verify -fcxx-exceptions %s

[[noreturn]] void a() {
  return; // expected-warning {{function 'a' declared 'noreturn' should not return}}
}
void a2 [[noreturn]] () {
  return; // expected-warning {{function 'a2' declared 'noreturn' should not return}}
}

template <typename T> void a3 [[noreturn]] () {}
template <> void a3<int>() { return; } // expected-warning {{function 'a3<int>' declared 'noreturn' should not return}}

template <typename T> void a4 [[noreturn]] () { return; } // expected-warning {{function 'a4' declared 'noreturn' should not return}}
                                                          // expected-warning@-1 {{function 'a4<long>' declared 'noreturn' should not return}}
void a4_test() { a4<long>(); } // expected-note {{in instantiation of function template specialization 'a4<long>' requested here}}

[[noreturn, noreturn]] void b() { throw 0; } // expected-error {{attribute 'noreturn' cannot appear multiple times in an attribute specifier}}
[[noreturn]] [[noreturn]] void b2() { throw 0; } // ok

[[noreturn()]] void c(); // expected-error {{attribute 'noreturn' cannot have an argument list}}

void d() [[noreturn]]; // expected-error {{'noreturn' attribute cannot be applied to types}}
int d2 [[noreturn]]; // expected-error {{'noreturn' attribute only applies to functions}}

[[noreturn]] int e() { b2(); } // ok

int f(); // expected-note {{declaration missing '[[noreturn]]' attribute is here}}
[[noreturn]] int f(); // expected-error {{function declared '[[noreturn]]' after its first declaration}}
int f();

[[noreturn]] int g();
int g() { while (true) b(); } // ok
[[noreturn]] int g();

[[gnu::noreturn]] int h();

template<typename T> void test_type(T) { T::error; } // expected-error {{has no members}}
template<> void test_type(int (*)()) {}

void check() {
  // We do not consider [[noreturn]] to be part of the function's type.
  // However, we do treat [[gnu::noreturn]] as being part of the type.
  //
  // This isn't quite GCC-compatible; it treats [[gnu::noreturn]] as
  // being part of a function *pointer* type, but not being part of
  // a function type.
  test_type(e);
  test_type(f);
  test_type(g);
  test_type(h); // expected-note {{instantiation}}
}
