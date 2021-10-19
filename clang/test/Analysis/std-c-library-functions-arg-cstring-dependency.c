// This test case crashes if strncasecmp is modeled in StdCLibraryFunctions.
// Either we fix CStringChecker to handle the call prerequisites in
// checkPreCall, or we must not evaluate any pure functions in
// StdCLibraryFunctions that are also handled in CStringChecker.

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=unix.cstring.NullArg \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify

typedef __typeof(sizeof(int)) size_t;
int strncasecmp(const char *s1, const char *s2, size_t n);

int strncasecmp_null_argument(char *a, size_t n) {
  char *b = 0;
  return strncasecmp(a, b, n); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}
