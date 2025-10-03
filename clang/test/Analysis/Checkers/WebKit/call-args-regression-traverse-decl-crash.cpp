// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

template <class Class> struct T;
template <template <class> class Class, class Type>
struct T<Class<Type>>
{ };
