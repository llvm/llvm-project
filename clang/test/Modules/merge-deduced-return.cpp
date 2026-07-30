// RUN: %clang_cc1 -fmodules -std=c++17 -verify %s
// RUN: %clang_cc1 -fmodules -std=c++17 -verify %s -DLOCAL
// expected-no-diagnostics

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
inline auto f() { struct X {}; return X(); }
inline auto a = f();
auto g(int);
template<typename T> auto h(T t) { return g(t); }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
inline auto f() { struct X {}; return X(); }
inline auto b = f();
auto g(int) { return 0; }
#pragma clang module end
#pragma clang module endbuild

#ifdef LOCAL
inline auto f() { struct X {}; return X(); }
inline auto b = f();
auto g(int) { return 0; }
#else
#pragma clang module import B
#endif

#pragma clang module import A

using T = decltype(a);
using T = decltype(b);

int test_g = h(0);
