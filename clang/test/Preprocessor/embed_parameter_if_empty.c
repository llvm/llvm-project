// RUN: %clang_cc1 %s -embed-dir=%S/Inputs -fsyntax-only -verify

const char data[] = {
#embed <media/empty> if_empty(123, 124, 125)
};
const char non_empty_data[] = {
#embed <jk.txt> if_empty(123, 124, 125)
};
_Static_assert(sizeof(data) == 3, "");
_Static_assert(123 == data[0], "");
_Static_assert(124 == data[1], "");
_Static_assert(125 == data[2], "");
_Static_assert(sizeof(non_empty_data) == 2, "");
_Static_assert('j' == non_empty_data[0], "");
_Static_assert('k' == non_empty_data[1], "");
// expected-no-diagnostics
