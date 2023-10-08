// RUN: %clang_cc1 %s -fsyntax-only -embed-dir=%S/Inputs -verify
// RUN: %clang_cc1 -x c %s -fsyntax-only -embed-dir=%S/Inputs -verify

const char data[] = {
#embed <media/art.txt>
};
const char data2[] = {
#embed <media/art.txt>
, 0
};
const char data3[] = {
#embed <media/art.txt> suffix(, 0)
};
const char data4[] = {
#embed <media/art.txt> suffix(,)
0
};
_Static_assert(sizeof(data) == 274, "");
_Static_assert(' ' == data[0], "");
_Static_assert('_' == data[11], "");
_Static_assert('\n' == data[273], "");
_Static_assert(sizeof(data2) == 275, "");
_Static_assert(' ' == data2[0], "");
_Static_assert('_' == data2[11], "");
_Static_assert('\n' == data2[273], "");
_Static_assert('\0' == data2[274], "");
_Static_assert(sizeof(data3) == 275, "");
_Static_assert(' ' == data3[0], "");
_Static_assert('_' == data3[11], "");
_Static_assert('\n' == data3[273], "");
_Static_assert('\0' == data3[274], "");
_Static_assert(sizeof(data4) == 275, "");
_Static_assert(' ' == data4[0], "");
_Static_assert('_' == data4[11], "");
_Static_assert('\n' == data4[273], "");
_Static_assert('\0' == data4[274], "");

const signed char data5[] = {
#embed <media/art.txt>
};
const signed char data6[] = {
#embed <media/art.txt>
, 0
};
const signed char data7[] = {
#embed <media/art.txt> suffix(, 0)
};
const signed char data8[] = {
#embed <media/art.txt> suffix(,)
0
};
_Static_assert(sizeof(data5) == 274, "");
_Static_assert(' ' == data5[0], "");
_Static_assert('_' == data5[11], "");
_Static_assert('\n' == data5[273], "");
_Static_assert(sizeof(data6) == 275, "");
_Static_assert(' ' == data6[0], "");
_Static_assert('_' == data6[11], "");
_Static_assert('\n' == data6[273], "");
_Static_assert('\0' == data6[274], "");
_Static_assert(sizeof(data7) == 275, "");
_Static_assert(' ' == data7[0], "");
_Static_assert('_' == data7[11], "");
_Static_assert('\n' == data7[273], "");
_Static_assert('\0' == data7[274], "");
_Static_assert(sizeof(data8) == 275, "");
_Static_assert(' ' == data8[0], "");
_Static_assert('_' == data8[11], "");
_Static_assert('\n' == data8[273], "");
_Static_assert('\0' == data8[274], "");

const unsigned char data9[] = {
#embed <media/art.txt>
};
const unsigned char data10[] = {
0,
#embed <media/art.txt>
};
const unsigned char data11[] = {
#embed <media/art.txt> prefix(0,)
};
const unsigned char data12[] = {
0
#embed <media/art.txt> prefix(,)
};
_Static_assert(sizeof(data9) == 274, "");
_Static_assert(' ' == data9[0], "");
_Static_assert('_' == data9[11], "");
_Static_assert('\n' == data9[273], "");
_Static_assert(sizeof(data10) == 275, "");
_Static_assert(' ' == data10[1], "");
_Static_assert('_' == data10[12], "");
_Static_assert('\n' == data10[274], "");
_Static_assert('\0' == data10[0], "");
_Static_assert(sizeof(data11) == 275, "");
_Static_assert(' ' == data11[1], "");
_Static_assert('_' == data11[12], "");
_Static_assert('\n' == data11[274], "");
_Static_assert('\0' == data11[0], "");
_Static_assert(sizeof(data12) == 275, "");
_Static_assert(' ' == data12[1], "");
_Static_assert('_' == data12[12], "");
_Static_assert('\n' == data12[274], "");
_Static_assert('\0' == data12[0], "");

// expected-no-diagnostics
