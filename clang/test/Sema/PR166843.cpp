// RUN: %clang_cc1 -fsyntax-only %s -verify
namespace a {
template <class b>
void c() {
  ((::c::)); // expected-error {{expected unqualified-id}}
}
}
