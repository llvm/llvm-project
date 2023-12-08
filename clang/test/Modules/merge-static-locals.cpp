// RUN: %clang_cc1 -std=c++17 -fmodules -verify %s
// expected-no-diagnostics

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
template<int*> struct X {};
auto get() { static int n; return X<&n>(); }
using A = decltype(get());
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
template<int*> struct X {};
auto get() { static int n; return X<&n>(); }
using B = decltype(get());
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import A
#pragma clang module import B
using T = A;
using T = B;
