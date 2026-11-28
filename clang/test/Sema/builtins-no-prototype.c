// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify -Wno-deprecated-non-prototype %s
// expected-no-diagnostics

// A builtin redeclared without a prototype, where the earlier declaration is
// hidden, gets its arguments unconverted.

typedef __SIZE_TYPE__ size_t;
struct S;
size_t strlen(const char *);
char *strcpy(char *, const char *);

// A void pointer still gives the length.
void test_void(void) {
  int strlen = 0;
  {
    extern size_t strlen();
    _Static_assert(strlen((const void *)"abc") == 3, "");
  }
}

// A pointer to an incomplete struct must not crash.
void test_incomplete(void) {
  int strcpy = 0;
  {
    extern char *strcpy();
    char buf[2];
    strcpy(buf, (const struct S *)"abcdef");
  }
}
