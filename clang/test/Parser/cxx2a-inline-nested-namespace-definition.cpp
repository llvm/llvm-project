// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++17
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++20 -Wc++17-compat

namespace inline foo1::foo2::foo3 { // expected-error {{expected identifier or '{'}} expected-error {{use of undeclared identifier 'foo1'}}
}

inline namespace foo4::foo5::foo6 { // expected-error {{nested namespace definition cannot be 'inline'}}}
}

#if __cplusplus <= 201402L
// expected-warning@+7 {{nested namespace definition is a C++17 extension; define each namespace separately}}
// expected-warning@+6 {{inline nested namespace definition is a C++20 extension}}
#elif __cplusplus <= 201703L
// expected-warning@+4 {{inline nested namespace definition is a C++20 extension}}
#else
// expected-warning@+2 {{inline nested namespace definition is incompatible with C++ standards before C++20}}
#endif
namespace valid1::valid2::inline valid3::inline valid4::valid5 {}
// expected-note@-1 2 {{previous definition is here}}

#if __cplusplus <= 201402L
// expected-warning@+3 {{nested namespace definition is a C++17 extension; define each namespace separately}}
#endif
//expected-warning@+1 2 {{inline namespace reopened as a non-inline namespace}}
namespace valid1::valid2::valid3::valid4::valid5 {}

#if __cplusplus <= 201402L
// expected-warning@+7 {{nested namespace definition is a C++17 extension; define each namespace separately}}
// expected-warning@+6 {{inline nested namespace definition is a C++20 extension}}
#elif __cplusplus <= 201703L
// expected-warning@+4 {{inline nested namespace definition is a C++20 extension}}
#else
// expected-warning@+2 {{inline nested namespace definition is incompatible with C++ standards before C++20}}
#endif
namespace valid1::valid2::inline valid3::inline valid4::valid5 {}
// expected-note@-1 2 {{previous definition is here}}

namespace valid1 {
namespace valid2 {
//expected-warning@+1 {{inline namespace reopened as a non-inline namespace}}
namespace valid3 {
//expected-warning@+1 {{inline namespace reopened as a non-inline namespace}}
namespace valid4 {
namespace valid5 {
}
} // namespace valid4
} // namespace valid3
} // namespace valid2
} // namespace valid1

