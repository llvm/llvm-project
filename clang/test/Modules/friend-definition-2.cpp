// RUN: %clang_cc1 -std=c++14 -fmodules %s -verify
// RUN: %clang_cc1 -std=c++14 -fmodules %s -verify -triple i686-windows
// expected-no-diagnostics
#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
template<typename T> struct ct { friend auto operator-(ct, ct) { struct X {}; return X(); } void x(); };
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
template<typename T> struct ct { friend auto operator-(ct, ct) { struct X{}; return X(); } void x(); };
inline auto f() { return ct<float>() - ct<float>(); }
#pragma clang module end
#pragma clang module endbuild

// Force the definition of ct in module A to be the primary definition.
#pragma clang module import A
template<typename T> void ct<T>::x() {}

// Attempt to cause the definition of operator- in the ct primary template in
// module B to be the primary definition of that function. If that happens,
// we'll be left with a class template ct that appears to not contain a
// definition of the inline friend function.
#pragma clang module import B
auto v = f();

ct<int> make();
void h() {
  make() - make();
}
