// RUN: %clang_cc1 %s -fsyntax-only -std=c++11 -verify -fms-extensions

#pragma warning(push, 4_D) // expected-warning {{requires a level between 0 and 4}}

extern "C" {
#pragma function(memset) // no-warning
}
