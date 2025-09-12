// RUN: %clang_cc1 -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1                                         %s -verify=ref

// expected-no-diagnostics
// ref-no-diagnostics

extern __SIZE_TYPE__ strlen(const char *);

struct str_t {
  char s1[sizeof("a")];
};
static const struct str_t str1 = {"a"};
#define str ((const char *)&str1)
int structStrlen(void) {
  if (strlen(str) == 1)
    return 0;
  return 1;
}

