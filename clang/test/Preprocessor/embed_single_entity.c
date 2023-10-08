// RUN: %clang_cc1 %s -fsyntax-only -embed-dir=%S/Inputs -verify

const char data =
#embed "single_byte.txt"
;
_Static_assert('a' == data[0]);
// expected-no-diagnostics
