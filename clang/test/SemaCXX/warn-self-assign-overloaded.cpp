// RUN: %clang_cc1 -fsyntax-only -Wself-assign -DDUMMY -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign -DV0 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign -DV1 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign -DV2 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign -DV3 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign -DV4 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-self-assign -Wself-assign-overloaded -DDUMMY -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-self-assign -Wself-assign-overloaded -DV0 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-self-assign -Wself-assign-overloaded -DV1 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-self-assign -Wself-assign-overloaded -DV2 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-self-assign -Wself-assign-overloaded -DV3 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-self-assign -Wself-assign-overloaded -DV4 -verify %s

#ifdef DUMMY
struct S {};
#else
struct S {
#if defined(V0)
  S() = default;
#elif defined(V1)
  S &operator=(const S &) = default;
#elif defined(V2)
  S &operator=(S &) = default;
#elif defined(V3)
  S &operator=(const S &);
#elif defined(V4)
  S &operator=(S &);
#else
#error Define something!
#endif
  S &operator*=(const S &);
  S &operator/=(const S &);
  S &operator%=(const S &);
  S &operator+=(const S &);
  S &operator-=(const S &);
  S &operator<<=(const S &);
  S &operator>>=(const S &);
  S &operator&=(const S &);
  S &operator|=(const S &);
  S &operator^=(const S &);
  S &operator=(const volatile S &) volatile;
};
#endif

void f() {
  S a, b;
  a = a; // expected-warning{{explicitly assigning}}
  b = b; // expected-warning{{explicitly assigning}}
  a = b;
  b = a = b;
  a = a = a; // expected-warning{{explicitly assigning}}
  a = b = b = a;

#ifndef DUMMY
  a *= a;
  a /= a; // expected-warning {{explicitly assigning}}
  a %= a; // expected-warning {{explicitly assigning}}
  a += a;
  a -= a; // expected-warning {{explicitly assigning}}
  a <<= a;
  a >>= a;
  a &= a; // expected-warning {{explicitly assigning}}
  a |= a; // expected-warning {{explicitly assigning}}
  a ^= a; // expected-warning {{explicitly assigning}}
#endif
}

void false_positives() {
#define OP =
#define LHS a
#define RHS a
  S a;
  // These shouldn't warn due to the use of the preprocessor.
  a OP a;
  LHS = a;
  a = RHS;
  LHS OP RHS;
#undef OP
#undef LHS
#undef RHS

  // Ways to silence the warning.
  a = *&a;
  a = (S &)a;
  a = static_cast<decltype(a) &>(a);

#ifndef DUMMY
  // Volatile stores aren't side-effect free.
  volatile S vol_a;
  vol_a = vol_a;
  volatile S &vol_a_ref = vol_a;
  vol_a_ref = vol_a_ref;
#endif
}

// Do not diagnose self-assigment in an unevaluated context
struct SNoExcept {
  SNoExcept() = default;
  SNoExcept &operator=(const SNoExcept &) noexcept;
};
void false_positives_unevaluated_ctx(SNoExcept a) noexcept(noexcept(a = a)) {
  decltype(a = a) b = a;
  static_assert(noexcept(a = a), "");
  static_assert(sizeof(a = a), "");
}

template <typename T>
void g() {
  T a;
  a = a; // expected-warning{{explicitly assigning}}
}
void instantiate() {
  g<int>();
  g<S>();
}
