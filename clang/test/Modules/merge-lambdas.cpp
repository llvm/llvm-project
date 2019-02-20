// RUN: %clang_cc1 -std=c++14 -fmodules -verify %s -emit-llvm-only
// expected-no-diagnostics

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
template<typename T> auto f() { return []{}; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
#pragma clang module import A
inline auto x1() { return f<int>(); }
inline auto z() { return []{}; }
inline auto x2() { return z(); }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build C
module C {}
#pragma clang module contents
#pragma clang module begin C
#pragma clang module import A
inline auto y1() { return f<int>(); }
inline auto z() { return []{}; }
inline auto y2() { return z(); }
inline auto q() { return []{}; }
inline auto y3() { return q(); }
#pragma clang module end
#pragma clang module endbuild

inline auto q() { return []{}; }
inline auto x3() { return q(); }

#pragma clang module import B
#pragma clang module import C
using T = decltype(x1);
using T = decltype(y1);

using U = decltype(x2);
using U = decltype(y2);

using V = decltype(x3);
using V = decltype(y3);

#pragma clang module import A
void (*p)() = f<int>();
