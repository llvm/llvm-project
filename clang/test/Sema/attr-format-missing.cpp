// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-format-attribute %s

#include <iostream>
#include <cstdarg>

void f1(const std::string &str, ... /* args */)
{
    va_list args;
    vscanf(str.c_str(), args); // no warning
    vprintf(str.c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error: {{format argument not a string type}}
void f2(const std::string &str, ... /* args */);

void f3(std::string_view str, ... /* args */)
{
    va_list args;
    vscanf(std::string(str).c_str(), args); // no warning
    vprintf(std::string(str).c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f4(std::string_view str, ... /* args */);

void f5(const std::wstring &str, ... /* args */)
{
    va_list args;
    vscanf((const char *)str.c_str(), args); // no warning
    vprintf((const char *)str.c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f6(const std::wstring &str, ... /* args */);

void f7(std::wstring_view str, ... /* args */)
{
    va_list args;
    vscanf((const char *) std::wstring(str).c_str(), args); // no warning
    vprintf((const char *) std::wstring(str).c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f8(std::wstring_view str, ... /* args */);

void f9(const wchar_t *out, ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-error {{no matching function for call to 'vprintf'}}
    vscanf((const char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f9'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((char *) out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f9'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f10(const wchar_t *out, ... /* args */);

void f11(const char16_t *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f12(const char16_t *out, ... /* args */);

void f13(const char32_t *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
}

__attribute__((format(scanf, 1, 2))) // expected-error {{format argument not a string type}}
void f14(const char32_t *out, ... /* args */);

void f15(const char *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f15'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(scanf, 1, 2)))
void f16(const char *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // no warning
}

void f17(const unsigned char *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
}

__attribute__((format(scanf, 1, 2)))
void f18(const unsigned char *out, ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-error {{no matching function for call to 'vprintf'}}
}

void f19(const signed char *out, ... /* args */)
{
    va_list args;
    vprintf(out, args); // expected-error {{no matching function for call to 'vprintf'}}
}

__attribute__((format(scanf, 1, 2)))
void f20(const signed char *out, ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
}

void f21(const char out[], ... /* args */)
{
    va_list args;
    vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f21'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(scanf, 1, 0)))
void f22(const char out[], ... /* args */)
{
    va_list args;
    vscanf(out, args); // no warning
}

void f23(const char *out)
{
    va_list args;
    vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f23'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
}

void f24(const char *out, va_list args)
{
    vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
}

struct S1
{
    void fn1(const char *out, ... /* args */)
    {
        va_list args;
        vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn1'}}
                           // CHECK-FIXES: __attribute__((format(scanf, 2, 3)))
    }

    __attribute__((format(scanf, 2, 0)))
    void fn2(const char *out, va_list args);

    void fn3(const char *out, ... /* args */);
};

void S1::fn3(const char *out, ... /* args */)
{
    va_list args;
    fn2(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn3'}}
                    // CHECK-FIXES: __attribute__((format(scanf, 2, 3)))
}

union U1
{
    __attribute__((format(printf, 2, 0)))
    void fn1(const char *out, va_list args);

    void fn2(const char *out, ... /* args */)
    {
        va_list args;
        fn1(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn2'}}
                        // CHECK-FIXES: __attribute__((format(printf, 2, 3)))
    }
};

class C1
{
    __attribute__((format(printf, 3, 0)))
    void fn1(const int n, const char *out, va_list args);

    void fn2(const char *out, const int n, ... /* args */)
    {
        va_list args;
        fn1(n, out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn2'}}
                           // CHECK-FIXES: __attribute__((format(printf, 2, 4)))
    }

    C1(const int n, const char *out)
    {
        va_list args;
        fn1(n, out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'C1'}}
                           // CHECK-FIXES: __attribute__((format(printf, 3, 0)))
    }

    ~C1()
    {
        const char *out;
        va_list args;
        vprintf(out, args); // no warning
    }
};

template <int N>
void func(char (&str)[N], ... /* args */)
{
    va_list args;
    vprintf(str, args); // no warning
}
