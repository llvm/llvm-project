// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify=c -x c %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify=cxx -x c++ %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify=c -x c -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify=cxx -x c++ -fexperimental-new-constant-interpreter %s

#ifdef __cplusplus
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

CONSTEXPR float f1 = __builtin_nanf(L"");
// c-error@-1 {{incompatible pointer types passing 'int[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const wchar_t[1]'}}

CONSTEXPR double d1 = __builtin_nan(L"");
// c-error@-1 {{incompatible pointer types passing 'int[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const wchar_t[1]'}}

CONSTEXPR long double ld1 = __builtin_nanl(L"");
// c-error@-1 {{incompatible pointer types passing 'int[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const wchar_t[1]'}}

CONSTEXPR float f2 = __builtin_nanf(u"");
// c-error@-1 {{incompatible pointer types passing 'unsigned short[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char16_t[1]'}}

CONSTEXPR double d2 = __builtin_nan(u"");
// c-error@-1 {{incompatible pointer types passing 'unsigned short[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char16_t[1]'}}

CONSTEXPR long double ld2 = __builtin_nanl(u"");
// c-error@-1 {{incompatible pointer types passing 'unsigned short[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char16_t[1]'}}

CONSTEXPR float f3 = __builtin_nanf(U"");
// c-error@-1 {{incompatible pointer types passing 'unsigned int[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char32_t[1]'}}

CONSTEXPR double d3 = __builtin_nan(U"");
// c-error@-1 {{incompatible pointer types passing 'unsigned int[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char32_t[1]'}}

CONSTEXPR long double ld3 = __builtin_nanl(U"");
// c-error@-1 {{incompatible pointer types passing 'unsigned int[1]' to parameter of type 'const char *'}}
// c-error@-2 {{initializer element is not a compile-time constant}}
// cxx-error@-3 {{cannot initialize a parameter of type 'const char *' with an lvalue of type 'const char32_t[1]'}}
