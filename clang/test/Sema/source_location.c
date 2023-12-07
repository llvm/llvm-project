// RUN: %clang_cc1 -std=c90 -fconst-strings -DCONST_STRINGS -verify %s
// RUN: %clang_cc1 -std=c90 -verify %s
// RUN: %clang_cc1 -std=c90 -fms-extensions -DMS -fconst-strings -DCONST_STRINGS -verify %s
// RUN: %clang_cc1 -std=c90 -fms-extensions -DMS -verify %s

// expected-no-diagnostics

#define IsEqual(L, R) (__builtin_strcmp(L, R) == 0)

const char *const FILE = __builtin_FILE();
const char *const FUNC = __builtin_FUNCTION();
const unsigned LINE = __builtin_LINE();
const unsigned COL = __builtin_COLUMN();

#ifndef CONST_STRINGS
char *const NCFILE = __builtin_FILE();
char *const NCFUNC = __builtin_FUNCTION();
#ifdef MS
char *const NCFNSG = __builtin_FUNCSIG();
#endif
#endif

#ifdef CONST_STRINGS
_Static_assert(IsEqual(__builtin_FILE(), __FILE__), "");
_Static_assert(IsEqual(__builtin_FILE_NAME(), __FILE_NAME__), "");
_Static_assert(__builtin_LINE() == __LINE__, "");
_Static_assert(IsEqual("", __builtin_FUNCTION()), "");
#ifdef MS
_Static_assert(IsEqual("", __builtin_FUNCSIG()), "");
#endif

#line 42 "my_file.c"
_Static_assert(__builtin_LINE() == 42, "");
_Static_assert(IsEqual(__builtin_FILE(), "my_file.c"), "");
_Static_assert(IsEqual(__builtin_FILE_NAME(), "my_file.c"), "");

_Static_assert(__builtin_COLUMN() == __builtin_strlen("_Static_assert(_"), "");

void foo(void) {
  _Static_assert(IsEqual(__builtin_FUNCTION(), "foo"), "");
#ifdef MS
  _Static_assert(IsEqual(__builtin_FUNCSIG(), "void __cdecl foo(void)"), "");
#endif
}
#endif // CONST_STRINGS
