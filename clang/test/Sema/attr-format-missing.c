// RUN: %clang_cc1 -fsyntax-only -verify=expected,c_diagnostics -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefixes=CHECK,C-CHECK
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -std=c++2b -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -std=c++23 -Wmissing-format-attribute %s
// RUN: not %clang_cc1 -fsyntax-only -x c++ -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefixes=CHECK

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
    vsnprintf(out, len, format, args); // expected-warning@#f1 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f1'}}
                                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 3, 4)))"
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
    vscanf(out, args); // expected-warning@#f3 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f3'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:6-[[@LINE-5]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f4(char* out, ... /* args */) // #f4
{
    va_list args;
    vprintf("test", args); // expected-no-warning

    const char *ch;
    vscanf(ch, args); // expected-no-warning
}

void f5(va_list args) // #f5
{
    char *ch;
    vscanf(ch, args); // expected-no-warning
}

void f6(char *out, va_list args) // #f6
{
    char *ch;
    vscanf(ch, args); // expected-no-warning
    vprintf("test", args); // expected-no-warning
    vprintf(out, args); // expected-warning@#f6 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f6'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f7(const char *out, ... /* args */) // #f7
{
    va_list args;

    vscanf(out, &args[0]); // expected-warning@#f7 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f7'}}
                           // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:6-[[@LINE-5]]:6}:"__attribute__((format(scanf, 1, 0)))"
    vprintf(out, &args[0]); // expected-warning@#f7 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f7'}}
                            // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:6-[[@LINE-7]]:6}:"__attribute__((format(printf, 1, 0)))"
}

__attribute__((format(scanf, 1, 0)))
__attribute__((format(printf, 1, 2)))
void f8(const char *out, ... /* args */) // #f8
{
    va_list args;

    vscanf(out, &args[0]); // expected-no-warning
    vprintf(out, &args[0]); // expected-no-warning
}

void f9(const char out[], ... /* args */) // #f9
{
    va_list args;
    char *ch;
    vscanf(ch, args); // expected-no-warning
    vscanf(out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f9'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
    vsprintf(ch, out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f9'}}
                             // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:6-[[@LINE-8]]:6}:"__attribute__((format(printf, 1, 2)))"
}

void f10(const wchar_t *out, ... /* args */) // #f10
{
    va_list args;
    vprintf(out, args);
#if __SIZEOF_WCHAR_T__ == 4
                        // c_diagnostics-warning@-2 {{incompatible pointer types passing 'const wchar_t *' (aka 'const int *') to parameter of type 'const char *'}}
#else
                        // c_diagnostics-warning@-4 {{incompatible pointer types passing 'const wchar_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
#endif
                        // c_diagnostics-note@#vprintf {{passing argument to parameter here}}
                        // c_diagnostics-warning@#f10 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f10'}}
                        // cpp_diagnostics-error@-8 {{no matching function for call to 'vprintf'}}
                        // cpp_diagnostics-note@#vprintf {{candidate function not viable: no known conversion from 'const wchar_t *' to 'const char *' for 1st argument}}
                        // C-CHECK: fix-it:"{{.*}}":{[[@LINE-13]]:6-[[@LINE-13]]:6}:"__attribute__((format(printf, 1, 2)))"
    vscanf((const char *) out, args); // expected-warning@#f10 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f10'}}
                                      // CHECK: fix-it:"{{.*}}":{[[@LINE-15]]:6-[[@LINE-15]]:6}:"__attribute__((format(scanf, 1, 2)))"
    vscanf((char *) out, args); // expected-warning@#f10 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f10'}}
                                // CHECK: fix-it:"{{.*}}":{[[@LINE-17]]:6-[[@LINE-17]]:6}:"__attribute__((format(scanf, 1, 2)))"

}

void f11(const char *out) // #f11
{
    va_list args;
    vscanf(out, args); // expected-warning@#f11 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f11'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f12(const char16_t *out, ... /* args */) // #f12
{
    va_list args;
    vscanf(out, args); // c_diagnostics-warning {{incompatible pointer types passing 'const char16_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                       // c_diagnostics-warning@#f12 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f12'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
                        // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'const char16_t *' to 'const char *' for 1st argument}}
}

void f13(const char32_t *out, ... /* args */) // #f13
{
    va_list args;
    vscanf(out, args); // c_diagnostics-warning {{incompatible pointer types passing 'const char32_t *' (aka 'const unsigned int *') to parameter of type 'const char *'}}
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                       // c_diagnostics-warning@#f13 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f13'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
                        // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'const char32_t *' to 'const char *' for 1st argument}}
}

void f14(const unsigned char *out, ... /* args */) // #f14
{
    va_list args;
    vprintf(out, args); // c_diagnostics-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // c_diagnostics-note@#vprintf {{passing argument to parameter here}}
                        // c_diagnostics-warning@#f14 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f14'}}
                        // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 2)))"
                        // cpp_diagnostics-error@-4 {{no matching function for call to 'vprintf'}}
                        // cpp_diagnostics-note@#vprintf {{candidate function not viable: no known conversion from 'const unsigned char *' to 'const char *' for 1st argument}}
    vscanf((const char *) out, args); // expected-warning@#f14 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f14'}}
                                      // CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:6-[[@LINE-10]]:6}:"__attribute__((format(scanf, 1, 2)))"
    vscanf((char *) out, args); // expected-warning@#f14 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f14'}}
                                // CHECK: fix-it:"{{.*}}":{[[@LINE-12]]:6-[[@LINE-12]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

__attribute__((format(printf, 1, 2)))
void f15(const unsigned char *out, ... /* args */) // #f15
{
    va_list args;
    vprintf(out, args); // c_diagnostics-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // c_diagnostics-note@#vprintf {{passing argument to parameter here}}
                        // cpp_diagnostics-error@-2 {{no matching function for call to 'vprintf'}}
                        // cpp_diagnostics-note@#vprintf {{candidate function not viable: no known conversion from 'const unsigned char *' to 'const char *' for 1st argument}}
    vscanf((const char *) out, args); // expected-warning@#f15 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f15'}}
                                      // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:6-[[@LINE-8]]:6}:"__attribute__((format(scanf, 1, 2)))"
    vprintf((const char *) out, args); // expected-no-warning
    vscanf((char *) out, args); // expected-warning@#f15 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f15'}}
                                // CHECK: fix-it:"{{.*}}":{[[@LINE-11]]:6-[[@LINE-11]]:6}:"__attribute__((format(scanf, 1, 2)))"
    vprintf((char *) out, args); // expected-no-warning
}

void f16(signed char *out, ... /* args */) // #f16
{
    va_list args;
    vscanf(out, args); // c_diagnostics-warning {{passing 'signed char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}} \
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}} \
                       // c_diagnostics-warning@#f16 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f16'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 2)))"
                        // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'signed char *' to 'const char *' for 1st argument}}
    vscanf((const char *) out, args); // expected-warning@#f16 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f16'}}
                                      // CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:6-[[@LINE-10]]:6}:"__attribute__((format(scanf, 1, 2)))"
    vprintf((char *) out, args); // expected-warning@#f16 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f16'}}
                                 // CHECK: fix-it:"{{.*}}":{[[@LINE-12]]:6-[[@LINE-12]]:6}:"__attribute__((format(printf, 1, 2)))"
}

__attribute__((format(scanf, 1, 2)))
void f17(signed char *out, ... /* args */) // #f17
{
    va_list args;
    vprintf(out, args); // c_diagnostics-warning {{passing 'signed char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // c_diagnostics-note@#vprintf {{passing argument to parameter here}}
                        // c_diagnostics-warning@#f17 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f17'}}
                        // C-CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 2)))"
                        // cpp_diagnostics-error@-4 {{no matching function for call to 'vprintf'}}
                        // cpp_diagnostics-note@#vprintf {{candidate function not viable: no known conversion from 'signed char *' to 'const char *' for 1st argument}}
    vscanf((const char *) out, args); // expected-no-warning
    vprintf((const char *) out, args); // expected-warning@#f17 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f17'}}
                                       // CHECK: fix-it:"{{.*}}":{[[@LINE-11]]:6-[[@LINE-11]]:6}:"__attribute__((format(printf, 1, 2)))"
    vscanf((char *) out, args); // expected-no-warning
    vprintf((char *) out, args); // expected-warning@#f17 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f17'}}
                                 // CHECK: fix-it:"{{.*}}":{[[@LINE-14]]:6-[[@LINE-14]]:6}:"__attribute__((format(printf, 1, 2)))"
}

__attribute__((format(printf, 1, 2)))
void f18(unsigned char out[], ... /* args */) // #f18
{
    va_list args;
    vprintf(out, args); // c_diagnostics-warning {{passing 'unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // c_diagnostics-note@#vprintf {{passing argument to parameter here}}
                        // cpp_diagnostics-error@-2 {{no matching function for call to 'vprintf'}}
                        // cpp_diagnostics-note@#vprintf {{candidate function not viable: no known conversion from 'unsigned char *' to 'const char *' for 1st argument}}
    vscanf(out, args); // c_diagnostics-warning {{passing 'unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                       // c_diagnostics-note@#vscanf {{passing argument to parameter here}}
                       // c_diagnostics-warning@#f18 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f18'}}
                       // C-CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:6-[[@LINE-10]]:6}:"__attribute__((format(scanf, 1, 2)))"
                        // cpp_diagnostics-error@-4 {{no matching function for call to 'vscanf'}}
                        // cpp_diagnostics-note@#vscanf {{candidate function not viable: no known conversion from 'unsigned char *' to 'const char *' for 1st argument}}
}

void f19(char* out) // #f19
{
    va_list args;
    const char* ch;
    vsprintf(out, ch, args); // expected-no-warning
    vscanf(out, args); // expected-warning@#f19 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f19'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f20(const char *out, ... /* args */) // #f20
{
    int a;
    printf(out, a); // expected-warning@#f20 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f20'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 0)))"
    printf(out, 1); // expected-warning@#f20 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f20'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
}

__attribute__((format(printf, 1, 2)))
void f21(const char *out, ... /* args */) // #f21
{
    int a;
    printf(out, a); // expected-no-warning
    printf(out, 1); // expected-no-warning
}

void f22(char* ch, const char *out, ... /* args */) // #f22
{
    va_list args;
    printf(ch, args); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22}}
                      // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 3)))"
    int a;
    printf(out, a); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:6-[[@LINE-7]]:6}:"__attribute__((format(printf, 2, 0)))"
    printf(out, 1); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:6-[[@LINE-9]]:6}:"__attribute__((format(printf, 2, 0)))"
    printf(out, args); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-11]]:6-[[@LINE-11]]:6}:"__attribute__((format(printf, 2, 3)))"
}

typedef va_list tdVaList;
typedef int tdInt;

void f23(const char *out, ... /* args */) // #f23
{
    tdVaList args;
    printf(out, args); // expected-warning@#f23 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f23'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 2)))"
    tdInt a;
    scanf(out, a); // expected-warning@#f23 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f23'}}
                   // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:6-[[@LINE-7]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f24(const char *out, tdVaList args) // #f24
{
    scanf(out, args); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f24'}}
                      // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(scanf, 1, 0)))"
    tdInt a;
    printf(out, a); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
}
