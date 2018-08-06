// RUN: %clang_cc1 -fsyntax-only -Wself-assign-field -DDUMMY -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign-field -DV0 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign-field -DV1 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign-field -DV2 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign-field -DV3 -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wself-assign-field -DV4 -verify %s

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
struct C {
  S a;
  S b;

  void f() {
    a = a; // expected-warning {{assigning field to itself}}
    b = b; // expected-warning {{assigning field to itself}}
    a = b;

    this->a = a;       // expected-warning {{assigning field to itself}}
    this->b = b;       // expected-warning {{assigning field to itself}}
    a = this->a;       // expected-warning {{assigning field to itself}}
    b = this->b;       // expected-warning {{assigning field to itself}}
    this->a = this->a; // expected-warning {{assigning field to itself}}
    this->b = this->b; // expected-warning {{assigning field to itself}}

    a = b;
    a = this->b;
    this->a = b;
    this->a = this->b;

#ifndef DUMMY
    a *= a;
    a /= a; // expected-warning {{assigning field to itself}}
    a %= a; // expected-warning {{assigning field to itself}}
    a += a;
    a -= a; // expected-warning {{assigning field to itself}}
    a <<= a;
    a >>= a;
    a &= a; // expected-warning {{assigning field to itself}}
    a |= a; // expected-warning {{assigning field to itself}}
    a ^= a; // expected-warning {{assigning field to itself}}
#endif
  }

  void false_positives() {
#define OP =
#define LHS a
#define RHS a
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
  }

#ifndef DUMMY
  volatile S vol_a;
  void vol_test() {
    // Volatile stores aren't side-effect free.
    vol_a = vol_a;
    volatile S &vol_a_ref = vol_a;
    vol_a_ref = vol_a_ref;
  }
#endif
};

// Do not diagnose self-assigment in an unevaluated context
struct SNoExcept {
  SNoExcept() = default;
  SNoExcept &operator=(const SNoExcept &) noexcept;
};
struct false_positives_unevaluated_ctx_class {
  SNoExcept a;

  void false_positives_unevaluated_ctx(SNoExcept a) noexcept(noexcept(a = a)) {
    decltype(a = a) b = a;
    static_assert(noexcept(a = a), "");
    static_assert(sizeof(a = a), "");
  }
};

template <typename T>
struct TemplateClass {
  T var;
  void f() {
    var = var; // expected-warning {{assigning field to itself}}
  }
};
void instantiate() {
  {
    TemplateClass<int> c;
    c.f();
  }
  {
    TemplateClass<S> c;
    c.f();
  }
}

// It may make sense not to warn on the rest of the tests.
// It may be a valid use-case to self-assign to tell the compiler that
// it is ok to vectorize the store.

void f0(C *s, C *t) {
  s->a = s->a;
  t->a = s->a;
}

void f1(C &s, C &t) {
  s.a = s.a;
  t.a = s.a;
}

struct T {
  C *s;
};

void f2(T *t, T *t2) {
  t->s->a = t->s->a;
  t2->s->a = t->s->a;
}

void f3(T &t, T &t2) {
  t.s->a = t.s->a;
  t2.s->a = t.s->a;
}
