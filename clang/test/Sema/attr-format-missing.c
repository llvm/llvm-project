// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-format-attribute %s

#include <stdarg.h>
#include <stdio.h>
#include <uchar.h>
#include <wchar.h>

__attribute__((__format__ (__scanf__, 1, 4)))
void f1(char *out, const size_t len, const char *format, ... /* args */)
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f1'}}
                                       // CHECK-FIXES: __attribute__((format(printf, 3, 4)))
}

__attribute__((__format__ (__printf__, 1, 4)))
void f2(char *out, const size_t len, const char *format, ... /* args */)
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f2'}}
                                       // CHECK-FIXES: __attribute__((format(printf, 3, 4)))
}

void f3(char *out, va_list args)
{
    vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f3'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
    vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f3'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
}

void f4(char* out, ... /* args */)
{
    va_list args;
    vprintf("test", args); // no warning

    const char *ch;
    vscanf(ch, args); // no warning
}

void f5(va_list args)
{
    char *ch;
    vscanf(ch, args); // no warning
}

void f6(char *out, va_list args)
{
    char *ch;
    vscanf(ch, args); // no warning
    vprintf("test", args); // no warning
}

void f7(const char *out, ... /* args */)
{
    va_list args;

    vscanf(out, &args[0]); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f7'}}
                           // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
    vprintf(out, &args[0]); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f7'}}
                            // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
}

__attribute__((format(scanf, 1, 0)))
__attribute__((format(printf, 1, 2)))
void f8(const char *out, ... /* args */)
{
    va_list args;

    vscanf(out, &args[0]); // no warning
    vprintf(out, &args[0]); // no warning
}

void f9(const char out[], ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f9'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    char *ch;
    vprintf(ch, args); // no warning
}

void f10(const wchar_t *out, ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-warning {{incompatible pointer types passing 'const wchar_t *' (aka 'const int *') to parameter of type 'const char *'}}
                        // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f10'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((const char *) out, args); // no warning
                                      // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f10'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((char *) out, args); // no warning
                                // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f10'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f11(const wchar_t *out, ... /* args */);

void f12(const char16_t *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-warning {{incompatible pointer types passing 'const char16_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
                       // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f12'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f13(const char16_t *out, ... /* args */);

void f14(const char32_t *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-warning {{incompatible pointer types passing 'const char32_t *' (aka 'const unsigned int *') to parameter of type 'const char *'}}
                       // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f14'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(scanf, 1, 2))) // expected-error {{format argument not a string type}}
void f15(const char32_t *out, ... /* args */);

void f16(const unsigned char *out, ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f16'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((const char *) out, args); // no warning
                                      // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f16'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((char *) out, args); // no warning
                                // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f16'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2)))
void f17(const unsigned char *out, ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
    vscanf((const char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f17'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vprintf((const char *) out, args); // no warning
    vscanf((char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f17'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vprintf((char *) out, args); // no warning
}

void f18(signed char *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-warning {{passing 'signed char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                       // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f18'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((const char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f18'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vprintf((char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f18'}}
                                 // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
}

__attribute__((format(scanf, 1, 2)))
void f19(signed char *out, ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'signed char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f19'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((const char *) out, args); // no warning
    vprintf((const char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f19'}}
                                       // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((char *) out, args); // no warning
    vprintf((char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f19'}}
                                 // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
}

__attribute__((format(printf, 1, 2)))
void f20(unsigned char out[], ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
    vscanf(out, args); // expected-warning {{passing 'unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                       // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f20'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

void f21(char* out) {
    va_list args;
    const char* ch;
    vsprintf(out, ch, args); // no warning
    vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f21'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
}

void f22(const char *out, ... /* args */)
{
    int a;
    printf(out, a); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                    // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
    printf(out, 1); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                    // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
}

__attribute__((format(printf, 1, 2)))
void f23(const char *out, ... /* args */)
{
    int a;
    printf(out, a); // no warning
    printf(out, 1); // no warning
}

void f24(char* ch, const char *out, ... /* args */)
{
    va_list args;
    printf(ch, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24}}
                      // CHECK-FIXES: __attribute__((format(printf, 1, 3)))
    int a;
    printf(out, a); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                    // CHECK-FIXES: __attribute__((format(printf, 2, 0)))
    printf(out, 1); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                    // CHECK-FIXES: __attribute__((format(printf, 2, 0)))
    printf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                       // CHECK-FIXES: __attribute__((format(printf, 2, 3)))
}
