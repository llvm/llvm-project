// RUN: %clang_cc1 %s -embed-dir=%S/Inputs -fsyntax-only -verify

const char data[] = {
#embed <single_byte.txt> suffix(, '\xA')
};
const char empty_data[] = {
#embed <media/empty> suffix(, '\xA')
1
};
_Static_assert(sizeof(data) == 2, "");
_Static_assert('b' == data[0], "");
_Static_assert('\xA' == data[1], "");
_Static_assert(sizeof(empty_data) == 1, "");
_Static_assert(1 == empty_data[0], "");
// expected-no-diagnostics
