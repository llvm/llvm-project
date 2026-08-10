// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

// expected-no-diagnostics
@import cxx_many_overloads;

void g() {
  f(N::X<0>());
}
