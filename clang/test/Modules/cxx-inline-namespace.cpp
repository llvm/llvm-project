// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

@import cxx_inline_namespace;
@import cxx_inline_namespace_b;

T x; // expected-error {{unknown type name 'T'}}

X::Elaborated *p;
