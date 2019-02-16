// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr2229 { // dr2229: 7
struct AnonBitfieldQualifiers {
  const unsigned : 1; // expected-error {{anonymous bit-field cannot have qualifiers}}
  volatile unsigned : 1; // expected-error {{anonymous bit-field cannot have qualifiers}}
  const volatile unsigned : 1; // expected-error {{anonymous bit-field cannot have qualifiers}}

  unsigned : 1;
  const unsigned i1 : 1;
  volatile unsigned i2 : 1;
  const volatile unsigned i3 : 1;
};
}

#if __cplusplus >= 201103L
namespace dr2211 { // dr2211: 8
void f() {
  int a;
  auto f = [a](int a) { (void)a; }; // expected-error {{a lambda parameter cannot shadow an explicitly captured entity}}
  // expected-note@-1{{variable 'a' is explicitly captured here}}
  auto g = [=](int a) { (void)a; };
}
}
#endif
