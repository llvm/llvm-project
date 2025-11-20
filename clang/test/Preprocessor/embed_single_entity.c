// RUN: %clang_cc1 %s -fsyntax-only -std=c23 --embed-dir=%S/Inputs -verify

const char data =
#embed <single_byte.txt>
;
_Static_assert('b' == data);
// expected-no-diagnostics
