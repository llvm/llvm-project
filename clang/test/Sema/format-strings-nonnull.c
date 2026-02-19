// RUN: %clang_cc1 -fsyntax-only --std=c23 -verify -Wnonnull -Wno-format-security %s
// RUN: %clang_cc1 -fsyntax-only --std=c23 -verify -Wnonnull -Wno-format-security -fexperimental-new-constant-interpreter %s

#define NULL  (void*)0

typedef struct _FILE FILE;
typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;
int printf(char const* restrict, ...);
int __builtin_printf(char const* restrict, ...);
int fprintf(FILE* restrict, char const* restrict, ...);
int snprintf(char* restrict, size_t, char const* restrict, ...);
int sprintf(char* restrict, char const* restrict, ...);
int vprintf(char const* restrict, __builtin_va_list);
int vfprintf(FILE* restrict, char const* restrict, __builtin_va_list);
int vsnprintf(char* restrict, size_t, char const* restrict, __builtin_va_list);
int vsprintf(char* restrict, char const* restrict, __builtin_va_list);

int scanf(char const* restrict, ...);
int fscanf(FILE* restrict, char const* restrict, ...);
int sscanf(char const* restrict, char const* restrict, ...);
int vscanf(char const* restrict, __builtin_va_list);
int vfscanf(FILE* restrict, char const* restrict, __builtin_va_list);
int vsscanf(char const* restrict, char const* restrict, __builtin_va_list);


void check_format_string(FILE *fp, va_list ap) {
    char buf[256];
    int num;
    char* const fmt = NULL;

    printf(fmt);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    __builtin_printf(NULL, "xxd");
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    fprintf(fp, NULL, 25);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    sprintf(NULL, NULL, 42);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}
    // expected-warning@-2{{null passed to a callee that requires a non-null argument}}

    snprintf(buf, 10, 0, 42);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    vprintf(fmt, ap);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    vfprintf(fp, 0, ap);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    vsprintf(buf, nullptr, ap);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    vsnprintf(buf, 10, fmt, ap);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    scanf(NULL);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    fscanf(nullptr, nullptr);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}
    // expected-warning@-2{{null passed to a callee that requires a non-null argument}}

    sscanf(NULL, "%d %s", &num, buf);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}
    sscanf(buf, fmt);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    vscanf(NULL, ap);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    vfscanf(fp, fmt, ap);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}

    vsscanf(buf, NULL, ap);
    // expected-warning@-1{{null passed to a callee that requires a non-null argument}}
}
