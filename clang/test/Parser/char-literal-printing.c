// RUN: %clang_cc1 -ast-print %s
// RUN: %clang_cc1 -x c++ -ast-print %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ast-print %s

#include <stddef.h>

char    test1(void) { return '\\'; }
wchar_t test2(void) { return L'\\'; }
char    test3(void) { return '\''; }
wchar_t test4(void) { return L'\''; }
char    test5(void) { return '\a'; }
wchar_t test6(void) { return L'\a'; }
char    test7(void) { return '\b'; }
wchar_t test8(void) { return L'\b'; }
char    test9(void) { return '\e'; }
wchar_t test10(void) { return L'\e'; }
char    test11(void) { return '\f'; }
wchar_t test12(void) { return L'\f'; }
char    test13(void) { return '\n'; }
wchar_t test14(void) { return L'\n'; }
char    test15(void) { return '\r'; }
wchar_t test16(void) { return L'\r'; }
char    test17(void) { return '\t'; }
wchar_t test18(void) { return L'\t'; }
char    test19(void) { return '\v'; }
wchar_t test20(void) { return L'\v'; }

char    test21(void) { return 'c'; }
wchar_t test22(void) { return L'c'; }
char    test23(void) { return '\x3'; }
wchar_t test24(void) { return L'\x3'; }

wchar_t test25(void) { return L'\x333'; }

#if __cplusplus >= 201103L
char16_t test26(void) { return u'\\'; }
char16_t test27(void) { return u'\''; }
char16_t test28(void) { return u'\a'; }
char16_t test29(void) { return u'\b'; }
char16_t test30(void) { return u'\e'; }
char16_t test31(void) { return u'\f'; }
char16_t test32(void) { return u'\n'; }
char16_t test33(void) { return u'\r'; }
char16_t test34(void) { return u'\t'; }
char16_t test35(void) { return u'\v'; }

char16_t test36(void) { return u'c'; }
char16_t test37(void) { return u'\x3'; }

char16_t test38(void) { return u'\x333'; }

char32_t test39(void) { return U'\\'; }
char32_t test40(void) { return U'\''; }
char32_t test41(void) { return U'\a'; }
char32_t test42(void) { return U'\b'; }
char32_t test43(void) { return U'\e'; }
char32_t test44(void) { return U'\f'; }
char32_t test45(void) { return U'\n'; }
char32_t test46(void) { return U'\r'; }
char32_t test47(void) { return U'\t'; }
char32_t test48(void) { return U'\v'; }

char32_t test49(void) { return U'c'; }
char32_t test50(void) { return U'\x3'; }

char32_t test51(void) { return U'\x333'; }
#endif
