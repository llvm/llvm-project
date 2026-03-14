// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

// expected-no-diagnostics

@import recursive_visibility_c;

template<typename T> struct Z { typedef T type; };
template void g<Z>();
