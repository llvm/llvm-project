// RUN: %clang_cc1 -std=c++98 -fsyntax-only -pedantic %s -verify=precxx11,precxx17,precxx26
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -pedantic %s -verify=since-cxx11,precxx17,precxx26 -Wc++98-compat
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -pedantic %s -verify=since-cxx11,since-cxx17,precxx26 -Wc++98-compat -Wpre-c++17-compat
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -pedantic %s -verify=since-cxx11,since-cxx17,since-cxx26 -Wc++98-compat -Wpre-c++17-compat -Wpre-c++26-compat

static_assert(false, "a");
// precxx11-error@-1 {{a type specifier is required for all declarations}}
// since-cxx11-warning@-2 {{'static_assert' declarations are incompatible with C++98}}
// since-cxx11-error@-3 {{static assertion failed: a}}

#if __cplusplus >= 201103L
static_assert(false);
// since-cxx11-warning@-1 {{'static_assert' declarations are incompatible with C++98}}
// precxx17-warning@-2 {{'static_assert' with no message is a C++17 extension}}
// since-cxx17-warning@-3 {{'static_assert' with no message is incompatible with C++ standards before C++17}}
// since-cxx11-error@-4 {{static assertion failed}}

struct X {
    static constexpr int size() { return 1; } // since-cxx11-warning {{'constexpr'}}
    static constexpr const char* data() { return "b"; } // since-cxx11-warning {{'constexpr'}}
};

static_assert(false, X());
// since-cxx11-warning@-1 {{'static_assert' declarations are incompatible with C++98}}
// precxx26-warning@-2 {{'static_assert' with a user-generated message is a C++26 extension}}
// since-cxx26-warning@-3 {{'static_assert' with a user-generated message is incompatible with C++ standards before C++26}}
// since-cxx11-error@-4 {{static assertion failed: b}}
#endif
