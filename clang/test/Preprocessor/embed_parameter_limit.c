// RUN: %clang_cc1 %s -embed-dir=%S/Inputs -fsyntax-only -verify

const char data[] = {
#embed <jk.txt>
};
const char offset_data[] = {
#embed <jk.txt> limit(1)
};
_Static_assert(sizeof(data) == 2, "");
_Static_assert('j' == data[0], "");
_Static_assert('k' == data[1], "");
_Static_assert(sizeof(offset_data) == 1, "");
_Static_assert('j' == offset_data[0], "");
_Static_assert(offset_data[0] == data[0], "");
// expected-no-diagnostics
