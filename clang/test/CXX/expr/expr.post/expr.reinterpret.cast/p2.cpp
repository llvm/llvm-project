// RUN: %clang_cc1 %s -verify

// The reinterpret_cast operator shall not cast away constness.
struct X {};
struct Y {};
void f(const int * X::* Y::* *p) {
  // This applies for similar types...
  (void)reinterpret_cast<int * X::* Y::* *>(p); // expected-error {{casts away qualifiers}}
  // ... and for cases where the base type is different ...
  (void)reinterpret_cast<float * X::* Y::* *>(p); // expected-error {{casts away qualifiers}}
  // ... and for cases where pointers to members point to members of different classes ...
  (void)reinterpret_cast<int * Y::* X::* *>(p); // expected-error {{casts away qualifiers}}
  // ... and even for cases where the path is wholly different!
  // (Though we accept such cases as an extension.)
  (void)reinterpret_cast<double Y::* X::* * *>(p); // expected-warning {{casts away qualifiers}}

  // If qualifiers are added, we need a 'const' at every level above.
  (void)reinterpret_cast<const volatile double Y::* X::* * *>(p); // expected-warning {{casts away qualifiers}}
  (void)reinterpret_cast<const volatile double Y::*const X::*const **>(p); // expected-warning {{casts away qualifiers}}
  (void)reinterpret_cast<const volatile double Y::*const X::**const *>(p); // expected-warning {{casts away qualifiers}}
  (void)reinterpret_cast<const volatile double Y::*X::*const *const *>(p); // expected-warning {{casts away qualifiers}}
  (void)reinterpret_cast<const volatile double Y::*const X::*const *const *>(p); // ok

  (void)reinterpret_cast<const double Y::*volatile X::**const *>(p); // expected-warning {{casts away qualifiers}}
  (void)reinterpret_cast<const double Y::*volatile X::*const *const *>(p); // ok
}
