// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify -std=c++23 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#ifndef __cplusplus
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
typedef __WCHAR_TYPE__ wchar_t;
#endif

typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;

__attribute__((__format__(__printf__, 1, 2)))
int printf(const char *, ...); // #printf

__attribute__((__format__(__scanf__, 1, 2)))
int scanf(const char *, ...); // #scanf

__attribute__((__format__(__printf__, 1, 0)))
int vprintf(const char *, va_list); // #vprintf

__attribute__((__format__(__scanf__, 1, 0)))
int vscanf(const char *, va_list); // #vscanf

__attribute__((__format__(__printf__, 2, 0)))
int vsprintf(char *, const char *, va_list); // #vsprintf

__attribute__((__format__(__printf__, 3, 0)))
int vsnprintf(char *, size_t, const char *, va_list); // #vsnprintf

#ifndef __cplusplus
int vwscanf(const wchar_t *, va_list); // #vwscanf
#endif

__attribute__((__format__(__scanf__, 1, 4)))
void f1(char *out, const size_t len, const char *format, ... /* args */) // #f1
{
    va_list args;
    vsnprintf(out, len, format, args);
}

__attribute__((__format__(__printf__, 1, 4)))
void f2(char *out, const size_t len, const char *format, ... /* args */) // #f2
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-warning@#f2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f2'}}
                                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 3, 4)))"
                                       // expected-note@-2 {{'printf' format function}}
}

void f3(char *out, va_list args) // #f3
{
    vprintf(out, args); // expected-warning@#f3 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f3'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(printf, 1, 0)))"
                        // expected-note@-2 {{'printf' format function}}
}

void f4(char* out, ... /* args */) // #f4
{
    va_list args;
    vprintf("test", args);

    const char *ch;
    vprintf(ch, args);
}

void f5(va_list args) // #f5
{
    char *ch;
    vscanf(ch, args);
}

void f6(char *out, va_list args) // #f6
{
    char *ch;
    vprintf(ch, args);
    vprintf("test", args);
    vprintf(out, args); // expected-warning@#f6 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f6'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
                        // expected-note@-2 {{'printf' format function}}
}

void f7(const char *out, ... /* args */) // #f7
{
    va_list args;

    vscanf(out, args); // expected-warning@#f7 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f7'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:6-[[@LINE-5]]:6}:"__attribute__((format(scanf, 1, 2)))"
                       // expected-note@-2 {{'scanf' format function}}
}

void f8(const char *out, ... /* args */) // #f8
{
    va_list args;

    vscanf(out, args);
    vprintf(out, args);
}

void f9(const char out[], ... /* args */) // #f9
{
    va_list args;
    char *ch;
    vprintf(ch, args);
    vsprintf(ch, out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f9'}}
                             // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 2)))"
                             // expected-note@-2 {{'printf' format function}}
}

#ifndef __cplusplus
void f10(const wchar_t *out, ... /* args */) // #f10
{
    va_list args;
    vwscanf(out, args);
}
#endif

void f11(const char *out) // #f11
{
    va_list args;
    vscanf(out, args); // expected-warning@#f11 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f11'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 0)))"
                       // expected-note@-2 {{'scanf' format function}}
}

void f12(char* out) // #f12
{
    va_list args;
    const char* ch;
    vsprintf(out, ch, args);
    vprintf(out, args); // expected-warning@#f12 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f12'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
                        // expected-note@-2 {{'printf' format function}}
}

void f13(const char *out, ... /* args */) // #f13
{
    va_list args;
    printf(out, args);
}

void f14(char *out, ... /* args */) // #f14
{
    va_list args;
    vscanf(out, args);
    vprintf(out, args);
}

void f15(char *out, ... /* args */) // #f15
{
    va_list args;
    vscanf(out, args);
    {
        vprintf(out, args);
    }
}

void f16(char *out, va_list args) // #f16
{
    {
        vscanf(out, args);
        vprintf(out, args);
    }
}

// expected-warning@#f17 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f17'}}
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:6-[[@LINE+1]]:6}:"__attribute__((format(scanf, 1, 2)))"
void f17(char *out, ... /* args */) // #f17
{
    va_list args;
    vscanf(out, args); // expected-note {{'scanf' format function}}
    {
        vscanf(out, args);
    }
}

void f18(char *out, int n, ... /* args */) // #f18
{
    va_list args;
    if (n > 0) {
        vprintf(out, args); // expected-warning@#f18 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f18'}}
                            // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:6-[[@LINE-5]]:6}:"__attribute__((format(printf, 1, 3)))"
                            // expected-note@-2 {{'printf' format function}}
    }
}

void f19(char *out, int n, ... /* args */) // #f19
{
    va_list args;
    if (n > 0) {}
    else {
        vprintf(out, args); // expected-warning@#f19 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f19'}}
                            // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 3)))"
                            // expected-note@-2 {{'printf' format function}}
    }
}

void f20(char *out, int n, ... /* args */) // #f20
{
    va_list args;
    if (n > 0) {
        vprintf(out, args);
    } else {
        vscanf(out, args);
    }
}

void f21(char *ch, const char *out, ... /* args */) // #f21
{
    va_list args;
    vprintf(ch, args); // expected-warning@#f21 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f21'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 3)))"
                       // expected-note@-2 {{'printf' format function}}
    vprintf(out, args); // expected-warning@#f21 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f21'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:6-[[@LINE-7]]:6}:"__attribute__((format(printf, 2, 3)))"
                        // expected-note@-2 {{'printf' format function}}
}

typedef va_list tdVaList;
typedef int tdInt;

void f22(const char *out, ... /* args */) // #f22
{
    tdVaList args;
    vprintf(out, args); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 2)))"
                        // expected-note@-2 {{'printf' format function}}
}

void f23(const char *out, tdVaList args) // #f23
{
    vscanf(out, args); // expected-warning@#f23 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f23'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(scanf, 1, 0)))"
                       // expected-note@-2 {{'scanf' format function}}
}

void f24(const char *out, tdVaList args) // #f24
{
    vscanf(out, args);
    vprintf(out, args);
}

void f25(char *out, ... /* args */) // #f25
{
    va_list args;
    char *ch;
    vscanf(ch, args);
    vprintf(out, args);
}

void f26(char *out, ... /* args */) // #f26
{
    va_list args;
    vscanf("%s", args);
    vprintf(out, args);
}
