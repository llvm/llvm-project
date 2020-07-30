// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=core

typedef __typeof(sizeof(int)) size_t;

struct FILE;
typedef struct FILE FILE;

size_t fread(void *restrict, size_t, size_t, FILE *restrict) __attribute__((nonnull(1)));

void f(FILE *F) {
  int *p = 0;
  fread(p, sizeof(int), 5, F); // expected-warning{{Null pointer passed to 1st parameter expecting 'nonnull' [core.NonNullParamChecker]}}
}
