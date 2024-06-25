// RUN: %clang_cc1 -std=c++1z -fexceptions -verify %s

using size_t = decltype(sizeof(0));

namespace std { enum class align_val_t : size_t {}; }

// p2 says "A template instance is never a usual deallocation function,
// regardless of its signature." We (and every other implementation) assume
// this means "A function template specialization [...]"
template<typename...Ts> struct A {
  void *operator new(size_t);
  void operator delete(void*, Ts...) = delete; // expected-note 4{{deleted}}
};

auto *a1 = new A<>; // expected-error {{deleted}}
auto *a2 = new A<size_t>; // expected-error {{deleted}}
auto *a3 = new A<std::align_val_t>; // expected-error {{deleted}}
auto *a4 = new A<size_t, std::align_val_t>; // expected-error {{deleted}}
auto *a5 = new A<std::align_val_t, size_t>; // ok, not usual
