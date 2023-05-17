// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify=expected,enabled -emit-codegen-only %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -emit-codegen-only -Wno-attribute-warning %s

__attribute__((error("oh no foo"))) void foo(void);

__attribute__((error("oh no bar"))) void bar(void);

int x(void) {
  return 8 % 2 == 1;
}

__attribute__((warning("oh no quux"))) void quux(void);

__attribute__((error("demangle me"))) void __compiletime_assert_455(void);

__attribute__((error("one"), error("two"))) // expected-warning {{attribute 'error' is already applied with different arguments}}
void                                        // expected-note@-1 {{previous attribute is here}}
duplicate_errors(void);

__attribute__((warning("one"), warning("two"))) // expected-warning {{attribute 'warning' is already applied with different arguments}}
void                                            // expected-note@-1 {{previous attribute is here}}
duplicate_warnings(void);

void baz(void) {
  foo(); // expected-error {{call to 'foo()' declared with 'error' attribute: oh no foo}}
  if (x())
    bar(); // expected-error {{call to 'bar()' declared with 'error' attribute: oh no bar}}

  quux();                     // enabled-warning {{call to 'quux()' declared with 'warning' attribute: oh no quux}}
  __compiletime_assert_455(); // expected-error {{call to '__compiletime_assert_455()' declared with 'error' attribute: demangle me}}
  duplicate_errors();         // expected-error {{call to 'duplicate_errors()' declared with 'error' attribute: two}}
  duplicate_warnings();       // enabled-warning {{call to 'duplicate_warnings()' declared with 'warning' attribute: two}}
}

#ifdef __cplusplus
template <typename T>
__attribute__((error("demangle me, too")))
T
nocall(T t);

struct Widget {
  __attribute__((warning("don't call me!")))
  operator int() { return 42; }
};

void baz_cpp(void) {
  foo(); // expected-error {{call to 'foo()' declared with 'error' attribute: oh no foo}}
  if (x())
    bar(); // expected-error {{call to 'bar()' declared with 'error' attribute: oh no bar}}
  quux();  // enabled-warning {{call to 'quux()' declared with 'warning' attribute: oh no quux}}

  // Test that we demangle correctly in the diagnostic for C++.
  __compiletime_assert_455(); // expected-error {{call to '__compiletime_assert_455()' declared with 'error' attribute: demangle me}}
  nocall<int>(42);            // expected-error {{call to 'int nocall<int>(int)' declared with 'error' attribute: demangle me, too}}

  Widget W;
  int w = W; // enabled-warning {{call to 'Widget::operator int()' declared with 'warning' attribute: don't call me!}}
}
#endif
