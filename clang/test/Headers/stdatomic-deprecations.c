// RUN: %clang_cc1 -fsyntax-only -std=c11 %s -verify=okay
// RUN: %clang_cc1 -fsyntax-only -std=c17 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -x c++ %s -verify=okay
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -x c++ %s -verify=cxx,expected
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -x c++ %s -verify=okay -D_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS
// RUN: %clang_cc1 -fsyntax-only -std=c17 %s -verify=okay -D_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS
//
// okay-no-diagnostics

#include <stdatomic.h>

void func(void) {
  _Atomic int i = ATOMIC_VAR_INIT(12); // expected-warning {{macro 'ATOMIC_VAR_INIT' has been marked as deprecated}} \
                                       // expected-note@stdatomic.h:* {{macro marked 'deprecated' here}}
  #if defined(ATOMIC_FLAG_INIT) // Ok, deprecated in C++20, undeprecated in C++23 via LWG4659.
  #endif
}
