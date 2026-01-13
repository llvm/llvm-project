// RUN: %clang_cc1 -fsyntax-only %s -verify
namespace a {
template <class b>
void c() {
  ((::c::x)); // expected-error {{'c' is not a class, namespace, or enumeration}}
}
}
