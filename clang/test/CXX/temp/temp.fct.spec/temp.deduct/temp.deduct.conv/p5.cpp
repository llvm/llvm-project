// RUN: %clang_cc1 -std=c++1z -verify %s

template<typename T, bool B> using Fn = T () noexcept(B);

// - If the original A is a function pointer type, A can be "pointer to
//   function" even if the deduced A is "pointer to noexcept function".
struct A {
  template<typename T> operator Fn<T, false>*(); // expected-note {{candidate}}
};
struct B {
  template<typename T> operator Fn<T, true>*();
};
void (*p1)() = A();
void (*p2)() = B();
void (*p3)() noexcept = A(); // expected-error {{no viable conversion}}
void (*p4)() noexcept = B();

// - If the original A is a pointer to member function type, A can be "pointer
//   to member of type function" even if the deduced A is "pointer to member of
//   type noexcept function".
struct C {
  template<typename T> operator Fn<T, false> A::*(); // expected-note {{candidate}}
};
struct D {
  template<typename T> operator Fn<T, true> A::*();
};
void (A::*q1)() = C();
void (A::*q2)() = D();
void (A::*q3)() noexcept = C(); // expected-error {{no viable conversion}}
void (A::*q4)() noexcept = D();

// There is no corresponding rule for references.
// FIXME: This seems like a defect.
// FIXME: We don't actually implement the final check for equal types at all!
// Instead, we handle the matching via [over.ics.user]p3:
//   "If the user-defined conversion is specified by a specialization of a
//   conversion function template, the second standard conversion sequence
//   shall have exact match rank."
// Note that this *does* allow discarding noexcept, since that conversion has
// Exact Match rank.
struct E {
  template<typename T> operator Fn<T, false>&(); // expected-note {{candidate}}
};
struct F {
  template<typename T> operator Fn<T, true>&();
};
void (&r1)() = E();
void (&r2)() = F();
void (&r3)() noexcept = E(); // expected-error {{no viable conversion}}
void (&r4)() noexcept = F();

// FIXME: We reject this for entirely the wrong reason. We incorrectly succeed
// in deducing T = void, U = G::B, and only fail due to [over.ics.user]p3.
struct G {
  template<typename, typename> struct A {};
  template<typename U> struct A<U, int> : A<U, void> {};
  struct B { typedef int type; };

  template<typename T, typename U = B> operator A<T, typename U::type> *(); // expected-note {{candidate function [with T = void, U = G::B]}}
};
G::A<void, void> *g = G(); // expected-error {{no viable conversion}}
