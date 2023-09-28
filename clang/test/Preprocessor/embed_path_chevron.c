// RUN: %clang_cc1 %s -fsyntax-only -embed-dir=%S/Inputs -CC -verify

const char data[] = {
#embed <single_byte.txt>
};
_Static_assert(sizeof(data) == 1, "");
_Static_assert('b' == data[0], "");
// expected-no-diagnostics
