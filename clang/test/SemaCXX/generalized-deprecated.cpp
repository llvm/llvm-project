// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only -fms-extensions -Wno-deprecated -Wc++14-extensions %s

// NOTE: use -Wno-deprecated to avoid cluttering the output with deprecated
// warnings

[[deprecated("1")]] int function_1();
// expected-warning@-1 {{use of the 'deprecated' attribute is a C++14 extension}}

[[gnu::deprecated("3")]] int function_3();

int __attribute__ (( deprecated("2") )) function_2();

__declspec(deprecated("4")) int function_4();

