// RUN: %clang_cc1 -fmodules -emit-llvm-only %s -verify

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
template<typename T> void f(const T&) { T::error; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
template<typename T> void f(const T&) { T::error; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build C
module C {}
#pragma clang module contents
#pragma clang module begin C
#pragma clang module load B
template<typename T> void f(const T&) { T::error; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module load A
inline void f() {}
void x() { f(); }

#pragma clang module import C
// expected-error@* {{cannot be used prior to}}
void y(int n) { f(n); } // expected-note {{instantiation of}}
