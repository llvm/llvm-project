// RUN: %clang_cc1 -fsyntax-only -verify=expected,c_diagnostics -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefixes=CHECK,C-CHECK
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -std=c++2b -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -std=c++23 -Wmissing-format-attribute %s
// RUN: not %clang_cc1 -fsyntax-only -x c++ -Wmissing-format-attribute -fdiagnostics-parseable-fixits -triple x86_64-linux %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-LIN64
// RUN: not %clang_cc1 -fsyntax-only -x c++ -Wmissing-format-attribute -fdiagnostics-parseable-fixits -triple x86_64-windows %s 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: not %clang_cc1 -fsyntax-only -x c++ -Wmissing-format-attribute -fdiagnostics-parseable-fixits -triple i386-windows %s 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: not %clang_cc1 -fsyntax-only -x c++ -Wmissing-format-attribute -fdiagnostics-parseable-fixits -triple i386-windows %s 2>&1 | FileCheck %s --check-prefixes=CHECK

#ifndef __cplusplus
typedef unsigned short char16_t;
typedef unsigned int char32_t;
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
int vsnprintf(char *ch, size_t, const char *, va_list); // #vsnprintf

__attribute__((__format__(__scanf__, 1, 4)))
void f1(char *out, const size_t len, const char *format, ... /* args */) // #f1
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-no-warning@#f1
}

__attribute__((__format__(__printf__, 1, 4)))
void f2(char *out, const size_t len, const char *format, ... /* args */) // #f2
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-warning@#f2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f2'}}
                                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 3, 4)))"
}

void f3(char *out, va_list args) // #f3
{
    vprintf(out, args); // expected-warning@#f3 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f3'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f4(char* out, ... /* args */) // #f4
{
    va_list args;
    vprintf("test", args); // expected-no-warning@#f4

    const char *ch;
    vprintf(ch, args); // expected-no-warning@#f4
}

void f5(va_list args) // #f5
{
    char *ch;
    vscanf(ch, args); // expected-no-warning@#f5
}

void f6(char *out, va_list args) // #f6
{
    char *ch;
    vprintf(ch, args); // expected-no-warning@#f6
    vprintf("test", args); // expected-no-warning@#f6
    vprintf(out, args); // expected-warning@#f6 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f6'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f7(const char *out, ... /* args */) // #f7
{
    va_list args;

    vscanf(out, &args[0]); // expected-warning@#f7 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f7'}}
                           // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:6-[[@LINE-5]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f8(const char *out, ... /* args */) // #f8
{
    va_list args;

    vscanf(out, &args[0]); // expected-no-warning@#f8
    vprintf(out, &args[0]); // expected-no-warning@#f8
}

void f9(const char out[], ... /* args */) // #f9
{
    va_list args;
    char *ch;
    vprintf(ch, args); // expected-no-warning
    vsprintf(ch, out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f9'}}
                             // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 2)))"
}

void f10(const wchar_t *out, ... /* args */) // #f10
{
    va_list args;
    vscanf(out, args);
#if __SIZEOF_WCHAR_T__ == 4
                        // c_diagnostics-warning@-2 {{incompatible pointer types passing 'const wchar_t *' (aka 'const int *') to parameter of type 'const char *'}}
#else
                        // c_diagnostics-warning@-4 {{incompatible pointer types passing 'const wchar_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
#endif
                        // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                        // c_diagnostics-warning@#f10 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f10'}}
                        // cpp_diagnostics-error@-8 {{no matching function for call to 'vscanf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'const wchar_t *' to 'const char *' for 1st argument}}
                        // C-CHECK: fix-it:"{{.*}}":{[[@LINE-13]]:6-[[@LINE-13]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

void f11(const wchar_t *out, ... /* args */) // #f11
{
    va_list args;
    vscanf((const char *) out, args); // expected-warning@#f11 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f11'}}
                                      // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

void f12(const wchar_t *out, ... /* args */) // #f12
{
    va_list args;
    vscanf((char *) out, args); // expected-warning@#f12 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f12'}}
                                // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

void f13(const wchar_t *out, ... /* args */) // #f13
{
    va_list args;
    vscanf(out, args);
#if __SIZEOF_WCHAR_T__ == 4
                        // c_diagnostics-warning@-2 {{incompatible pointer types passing 'const wchar_t *' (aka 'const int *') to parameter of type 'const char *'}}
#else
                        // c_diagnostics-warning@-4 {{incompatible pointer types passing 'const wchar_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
#endif
                        // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                        // cpp_diagnostics-error@-7 {{no matching function for call to 'vscanf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'const wchar_t *' to 'const char *' for 1st argument}}
                        // expected-warning@#f13 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f13'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-13]]:6-[[@LINE-13]]:6}:"__attribute__((format(scanf, 1, 2)))"
    vscanf((const char *) out, args);
    vscanf((char *) out, args);
}

void f14(const char *out) // #f14
{
    va_list args;
    vscanf(out, args); // expected-warning@#f14 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f14'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f15(const char16_t *out, ... /* args */) // #f15
{
    va_list args;
    vscanf(out, args); // c_diagnostics-warning {{incompatible pointer types passing 'const char16_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                       // c_diagnostics-warning@#f15 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f15'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
                       // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                       // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'const char16_t *' to 'const char *' for 1st argument}}
}

void f16(const char32_t *out, ... /* args */) // #f16
{
    va_list args;
    vscanf(out, args); // c_diagnostics-warning {{incompatible pointer types passing 'const char32_t *' (aka 'const unsigned int *') to parameter of type 'const char *'}}
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                       // c_diagnostics-warning@#f16 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f16'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
                       // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                       // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'const char32_t *' to 'const char *' for 1st argument}}
}

void f17(const unsigned char *out, ... /* args */) // #f17
{
    va_list args;
    vscanf(out, args); // c_diagnostics-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                       // c_diagnostics-warning@#f17 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f17'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
                       // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                       // cpp_diagnostics-note@#vprintf {{candidate function not viable: no known conversion from 'const unsigned char *' to 'const char *' for 1st argument}}
}

void f18(const unsigned char *out, ... /* args */) // #f18
{
    va_list args;
    vscanf((const char *) out, args); // expected-warning@#f18 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f18'}}
                                      // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

void f19(const unsigned char *out, ... /* args */) // #f19
{
    va_list args;
    vscanf((char *) out, args); // expected-warning@#f19 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f19'}}
                                // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

__attribute__((format(printf, 1, 2)))
void f20(const unsigned char *out, ... /* args */) // #f20
{
    va_list args;
    vprintf(out, args); // c_diagnostics-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // c_diagnostics-note@#vprintf {{passing argument to parameter here}}
                        // cpp_diagnostics-error@-2 {{no matching function for call to 'vprintf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'const unsigned char *' to 'const char *' for 1st argument}}
    vscanf((const char *) out, args); // expected-no-warning
    vprintf((const char *) out, args); // expected-no-warning
    vscanf((char *) out, args); // expected-no-warning
    vprintf((char *) out, args); // expected-no-warning
}

void f21(signed char *out, ... /* args */) // #f21
{
    va_list args;
    vscanf(out, args); // c_diagnostics-warning {{passing 'signed char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}} \
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                       // c_diagnostics-warning@#f21 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f21'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
                       // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                       // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'signed char *' to 'const char *' for 1st argument}}
}

void f22(signed char *out, ... /* args */) // #f22
{
    va_list args;
    vscanf((const char *) out, args); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f22'}}
                                      // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

void f23(signed char *out, ... /* args */) // #f23
{
    va_list args;
    vprintf((char *) out, args); // expected-warning@#f23 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f23'}}
                                 // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 2)))"
}

__attribute__((format(scanf, 1, 2)))
void f24(signed char *out, ... /* args */) // #f24
{
    va_list args;
    vprintf((const char *) out, args); // expected-no-warning@#f24
    vprintf((char *) out, args); // expected-no-warning@#f24
}

__attribute__((format(printf, 1, 2)))
void f25(unsigned char out[], ... /* args */) // #f25
{
    va_list args;
    vscanf((const char *) out, args); // expected-no-warning@#f25
    vscanf((char *) out, args); // expected-no-warning@#f25
}

void f26(char* out) // #f26
{
    va_list args;
    const char* ch;
    vsprintf(out, ch, args); // expected-no-warning@#f26
    vprintf(out, args); // expected-warning@#f26 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f26'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f27(const char *out, ... /* args */) // #f27
{
    int a;
    printf(out, a); // expected-warning@#f27 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f27'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f28(const char *out, ... /* args */) // #f28
{
    printf(out, 1); // expected-warning@#f28 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f28'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(printf, 1, 0)))"
}

__attribute__((format(printf, 1, 2)))
void f29(const char *out, ... /* args */) // #f29
{
    int a;
    printf(out, a); // expected-no-warning@#f29
}

__attribute__((format(printf, 1, 2)))
void f30(const char *out, ... /* args */) // #f30
{
    printf(out, 1); // expected-no-warning@#f30
}

__attribute__((format(printf, 1, 2)))
void f31(const char *out, ... /* args */) // #f31
{
    int a;
    printf(out, a); // expected-no-warning@#f31
    printf(out, 1); // expected-no-warning@#f31
}

void f32(char *out, ... /* args */) // #f32
{
    va_list args;
    scanf(out, args); // expected-no-warning@#f32
    {
        printf(out, args); // expected-no-warning@#f32
    }
}

void f33(char *out, va_list args) // #f33
{
    {
        scanf(out, args); // expected-no-warning@#f33
        printf(out, args); // expected-no-warning@#f33
    }
}

// expected-warning@#f34 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f34'}}
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:6-[[@LINE+1]]:6}:"__attribute__((format(scanf, 1, 2)))"
void f34(char *out, ... /* args */) // #f34
{
    va_list args;
    scanf(out, args); // expected-no-warning@#f34
    {
        scanf(out, args); // expected-no-warning@#f34
    }
}

void f35(char* ch, const char *out, ... /* args */) // #f35
{
    va_list args;
    printf(ch, args); // expected-warning@#f35 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f35}}
                      // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 3)))"
    int a;
    printf(out, a); // expected-warning@#f35 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f35'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:6-[[@LINE-7]]:6}:"__attribute__((format(printf, 2, 0)))"
    printf(out, 1); // no warning because first command above emitted same warning with same fix-it text
    printf(out, args); // expected-warning@#f35 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f35'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:6-[[@LINE-10]]:6}:"__attribute__((format(printf, 2, 3)))"
}

typedef va_list tdVaList;
typedef int tdInt;

void f36(const char *out, ... /* args */) // #f36
{
    tdVaList args;
    printf(out, args); // expected-warning@#f36 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f36'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 2)))"
}

void f37(const char *out, ... /* args */) // #f37
{
    tdInt a;
    scanf(out, a); // expected-warning@#f37 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f37'}}
                   // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f38(const char *out, tdVaList args) // #f38
{
    scanf(out, args); // expected-warning@#f38 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f38'}}
                      // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f39(const char *out, tdVaList args) // #f39
{
    tdInt a;
    printf(out, a); // expected-warning@#f39 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f39'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f40(char *out, ... /* args */) // #f40
{
    va_list args;
    char *ch;
    vscanf(ch, args); // expected-no-warning@#f40
    vprintf(out, args); // expected-no-warning@#f40
}

void f41(char *out, ... /* args */) // #f41
{
    va_list args;
    char *ch;
    vscanf("%s", ch);
#if defined(__x86_64__) && defined(__linux__)
                        // c_diagnostics-warning@-2 {{incompatible pointer types passing 'char *' to parameter of type 'struct __va_list_tag *'}}
                        // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                        // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'char *' to '__va_list_tag *' for 2nd argument}}
#endif
    vprintf(out, args);
#if defined(__x86_64__) && defined(__linux__)
                        // cpp_diagnostics-warning@#f41 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f41'}}
                        // CHECK-LIN64: fix-it:"{{.*}}":{[[@LINE-14]]:6-[[@LINE-14]]:6}:"__attribute__((format(printf, 1, 2)))"
#endif
}
