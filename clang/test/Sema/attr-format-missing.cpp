// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 -Wmissing-format-attribute %s
// RUN: not %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits -std=c++23 %s 2>&1
// FileCheck %s --check-prefixes=CHECK,CHECK-EXPLICIT-THIS-PARAMETER

typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;

namespace std
{
    template<class Elem> struct basic_string_view {};
    template<class Elem> struct basic_string {
        const Elem *c_str() const noexcept;
        basic_string(const basic_string_view<Elem> SW);
    };

    using string = basic_string<char>;
    using wstring = basic_string<wchar_t>;
    using string_view = basic_string_view<char>;
    using wstring_view = basic_string_view<wchar_t>;
}

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

int vwprintf(const wchar_t *, va_list); // #vwprintf

void f1(const std::string &str, ... /* args */) // #f1
{
    va_list args;
    vscanf(str.c_str(), args);
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f2(const std::string &str, ... /* args */); // #f2

void f3(std::string_view str, ... /* args */) // #f3
{
    va_list args;
    vscanf(std::string(str).c_str(), args);
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f4(std::string_view str, ... /* args */); // #f4

void f5(const std::wstring &str, ... /* args */) // #f5
{
    va_list args;
    vwprintf(str.c_str(), args);
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f6(const std::wstring &str, ... /* args */); // #f6

void f7(std::wstring_view str, ... /* args */) // #f7
{
    va_list args;
    vwprintf(std::wstring(str).c_str(), args);
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f8(std::wstring_view str, ... /* args */); // #f8

struct S1
{
    void fn1(const char *out, ... /* args */) // #S1_fn1
    {
        va_list args;
        vscanf(out, args); // expected-warning@#S1_fn1 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn1'}}
                           // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:10-[[@LINE-4]]:10}:"__attribute__((format(scanf, 2, 3)))"
                           // expected-note@-2 {{'scanf' format function}}
    }

    __attribute__((format(scanf, 2, 0)))
    void fn2(const char *out, va_list args); // #S1_fn2

    void fn3(const char *out, ... /* args */); // #S1_fn3

#if __has_extension(cxx_explicit_this_parameter)
    void fn4(this S1& explicitThis, const char *out, va_list args) // #S1_fn4
    {
        explicitThis.fn2(out, args); // expected-warning@#S1_fn4 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn4'}}
                                     // CHECK-EXPLICIT-THIS-PARAMETER: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"__attribute__((format(scanf, 2, 0)))"
                                     // expected-note@-2 {{'scanf' format function}}
    }
#endif
};

void S1::fn3(const char *out, ... /* args */)
{
    va_list args;
    fn2(out, args); // expected-warning@#S1_fn3 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn3'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-16]]:10-[[@LINE-16]]:10}:"__attribute__((format(scanf, 2, 3)))"
                    // expected-note@-2 {{'scanf' format function}}
}

union U1
{
    __attribute__((format(printf, 2, 0)))
    void fn1(const char *out, va_list args); // #U1_fn1

    void fn2(const char *out, ... /* args */) // #U1_fn2
    {
        va_list args;
        fn1(out, args); // expected-warning@#U1_fn2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn2'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:10-[[@LINE-4]]:10}:"__attribute__((format(printf, 2, 3)))"
                        // expected-note@-2 {{'printf' format function}}
    }

#if __has_extension(cxx_explicit_this_parameter)
    void fn3(this U1&, const char *out) // #U1_fn3
    {
        va_list args;
        vprintf(out, args); // expected-warning@#U1_fn3 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn3'}}
                            // CHECK-EXPLICIT-THIS-PARAMETER: fix-it:"{{.*}}":{[[@LINE-4]]:10-[[@LINE-4]]:10}:"__attribute__((format(printf, 2, 0)))"
                            // expected-note@-2 {{'printf' format function}}
    }
#endif
};

class C1
{
    __attribute__((format(printf, 3, 0)))
    void fn1(const int n, const char *out, va_list args); // #C1_fn1

    void fn2(const char *out, const int n, ... /* args */) // #C1_fn2
    {
        va_list args;
        fn1(n, out, args); // expected-warning@#C1_fn2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn2'}}
                           // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:10-[[@LINE-4]]:10}:"__attribute__((format(printf, 2, 4)))"
                           // expected-note@-2 {{'printf' format function}}
    }

#if __has_extension(cxx_explicit_this_parameter)
    void fn3(this const C1&, const char *out, va_list args) // #C1_fn3
    {
        vscanf(out, args); // expected-warning@#C1_fn3 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn3'}}
                           // CHECK-EXPLICIT-THIS-PARAMETER: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"__attribute__((format(scanf, 2, 0)))"
                           // expected-note@-2 {{'scanf' format function}}
    }
#endif

    C1(const int n, const char *out) //#C1_C1a
    {
        va_list args;
        fn1(n, out, args); // expected-warning@#C1_C1a {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'C1'}}
                           // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:5-[[@LINE-4]]:5}:"__attribute__((format(printf, 3, 0)))"
                           // expected-note@-2 {{'printf' format function}}
    }

    C1(const char *out, ... /* args */) // #C1_C1b
    {
        va_list args;
        vprintf(out, args); // expected-warning@#C1_C1b {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'C1'}}
                            // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:5-[[@LINE-4]]:5}:"__attribute__((format(printf, 2, 3)))"
                            // expected-note@-2 {{'printf' format function}}
    }

    ~C1() // #d_C1
    {
        const char *out;
        va_list args;
        vprintf(out, args);
    }
};

// TODO: implement for templates
template <int N>
void func(char (&str)[N], ... /* args */) // #func
{
    va_list args;
    vprintf(str, args);
}
