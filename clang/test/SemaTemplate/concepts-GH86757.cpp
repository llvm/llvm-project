// RUN: %clang_cc1 -std=c++20 -Wfatal-errors -verify %s

template <typename> int a;
template <typename... b> concept c = a<b...>;
template <typename> concept e = c<>;

// must be a fatal error to trigger the crash
undefined; // expected-error {{a type specifier is required for all declarations}}

template <typename d> concept g = e<d>;
template <g> struct h
template <g d>
struct h<d>;
