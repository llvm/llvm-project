// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: use of undeclared identifier 'foo'

template <template <decltype(foo())> typename> struct S {};
template <int &p> struct P;
S<P> s;
