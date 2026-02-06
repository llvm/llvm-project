// RUN: %clang_cc1 -fexperimental-new-constant-interpreter %s -verify=expected,both
// RUN: %clang_cc1                                         %s -verify=ref,both

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

void f() { __builtin_memcpy(f, f, 1); }
void f2()  { __builtin_memchr(f2, 0, 1); }


_Static_assert(__atomic_is_lock_free(4, (void*)2), ""); // both-error {{not an integral constant expression}}
