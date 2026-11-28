// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

constexpr int erroneous_scalar() {
  int x;    // expected-note {{declared here}}
  return x; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int v1 = erroneous_scalar(); // expected-error {{constexpr variable 'v1' must be initialized by a constant expression}} \
                                       // expected-note {{in call to 'erroneous_scalar()'}}

struct S {
  int m;
  constexpr S() {}
};
constexpr int erroneous_member() {
  S s;        // expected-note {{declared here}}
  return s.m; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int v2 = erroneous_member(); // expected-error {{constexpr variable 'v2' must be initialized by a constant expression}} \
                                       // expected-note {{in call to 'erroneous_member()'}}

constexpr int erroneous_element() {
  int a[3];    // expected-note {{declared here}}
  return a[0]; // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int v3 = erroneous_element(); // expected-error {{constexpr variable 'v3' must be initialized by a constant expression}} \
                                        // expected-note {{in call to 'erroneous_element()'}}

constexpr int indeterminate_scalar() {
  [[indeterminate]] int x; // expected-note {{declared here}}
  return x;                // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int v4 = indeterminate_scalar(); // expected-error {{constexpr variable 'v4' must be initialized by a constant expression}} \
                                           // expected-note {{in call to 'indeterminate_scalar()'}}

constexpr int initialized() {
  int x = 42;
  return x;
}
static_assert(initialized() == 42);

constexpr int partially_initialized() {
  int a[3] = {1};
  return a[1];
}
static_assert(partially_initialized() == 0);
