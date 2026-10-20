// RUN: %clang_cc1 -fsyntax-only -verify -Wnonnull %s

#ifdef __cplusplus
# define EXTERN_C extern "C"
#else
# define EXTERN_C extern
#endif

typedef struct _FILE FILE;
typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;

EXTERN_C int printf(const char *, ...);
EXTERN_C int fprintf(FILE *, const char *restrict, ...);
EXTERN_C int sprintf(char* restrict, char const* res, ...);
EXTERN_C int vfprintf(FILE* restrict, char const* res, __builtin_va_list);

EXTERN_C int scanf(char const *restrict, ...);
EXTERN_C int fscanf(FILE* restrict, char const* res, ...);
EXTERN_C int sscanf(char const* restrict, char const* res, ...);

void test(FILE *fp, va_list ap) {
  char buf[256];
  int num;

  __builtin_printf(__null, "x");
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}

  printf(__null, "xxd");
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}

  fprintf(fp, __null, 42);
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}

  sprintf(buf, __null);
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}

  scanf(__null);
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}

  fscanf(fp, __null);
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}

  vfprintf(__null, "xxd", ap);
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}

  sscanf(__null, "%d", &num);
  // expected-warning@-1 {{null passed to a callee that requires a non-null argument}}
}
