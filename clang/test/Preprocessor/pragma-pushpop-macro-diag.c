// RUN: %clang_cc1 -fms-extensions %s -fsyntax-only -verify

#pragma push_macro("") // expected-warning {{'#pragma push_macro' expected a non-empty string}}
#pragma pop_macro("") // expected-warning {{'#pragma pop_macro' expected a non-empty string}}
