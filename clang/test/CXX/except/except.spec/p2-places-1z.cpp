// RUN: %clang_cc1 -std=c++1z -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// In C++1z, we can put an exception-specification on any function declarator; the
// corresponding paragraph from C++14 and before was deleted.
// expected-no-diagnostics

void f() noexcept;
void (*fp)() noexcept;
void (**fpp)() noexcept;
void g(void (**pfa)() noexcept);
void (**h())() noexcept;

template<typename T> struct A {};
template<void() noexcept> struct B {};
A<void() noexcept> a;
B<f> b;
auto *p = new decltype(f)**;
