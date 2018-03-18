// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr2229 { // dr2229: yes
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
