// RUN: %clang_cc1 -fsyntax-only -Wself-assign-field -verify %s

struct C {
  int a;
  int b;

  C(int a, int b) : a(a), b(b) {}

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

    a *= a;
    a /= a;
    a %= a;
    a += a;
    a -= a;
    a <<= a;
    a >>= a;
    a &= a;
    a |= a;
    a ^= a;
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

    // A way to silence the warning.
    a = (int &)a;
  }

  // Do not diagnose self-assigment in an unevaluated context
  void false_positives_unevaluated_ctx() noexcept(noexcept(a = a)) // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  {
    decltype(a = a) b = a;              // expected-warning {{expression with side effects has no effect in an unevaluated context}}
    static_assert(noexcept(a = a), ""); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
    static_assert(sizeof(a = a), "");   // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  }

  volatile int vol_a;
  void vol_test() {
    // Volatile stores aren't side-effect free.
    vol_a = vol_a;
    volatile int &vol_a_ref = vol_a;
    vol_a_ref = vol_a_ref;
  }
};

// Dummy type.
struct Dummy {};

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
    TemplateClass<Dummy> c;
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
