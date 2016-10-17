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
struct E {
  template<typename T> operator Fn<T, false>&(); // expected-note {{candidate}}
};
struct F {
  template<typename T> operator Fn<T, true>&(); // expected-note {{candidate}}
};
void (&r1)() = E();
void (&r2)() = F(); // expected-error {{no viable conversion}}
void (&r3)() noexcept = E(); // expected-error {{no viable conversion}}
void (&r4)() noexcept = F();
