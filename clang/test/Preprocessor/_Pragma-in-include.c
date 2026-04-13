// RUN: %clang_cc1 -E %s -verify

// Don't crash, verify that diagnostics are preserved
#include _Pragma( // expected-error {{_Pragma takes a parenthesized string literal}} expected-error {{expected "FILENAME"}} 
