// RUN: %clang_cc1 %s -embed-dir=%S/Inputs -fsyntax-only -verify
// expected-no-diagnostics

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

struct S {
  int x, y, z;
};

const struct S s = {
#embed <single_byte.txt> suffix( , .y = 100, .z = 10 )
};

_Static_assert(s.x == 'b', "");
_Static_assert(s.y == 100, "");
_Static_assert(s.z == 10, "");
