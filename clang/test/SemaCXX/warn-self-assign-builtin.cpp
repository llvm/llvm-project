// RUN: %clang_cc1 -fsyntax-only -Wself-assign -verify %s

void f() {
  int a = 42, b = 42;
  a = a; // expected-warning{{explicitly assigning}}
  b = b; // expected-warning{{explicitly assigning}}
  a = b;
  b = a = b;
  a = a = a; // expected-warning{{explicitly assigning}}
  a = b = b = a;

  a *= a;
  a /= a;
  a %= a;
  a += a;
  a -= a;
  a <<= a;
  a >>= a;
  a &= a; // expected-warning {{explicitly assigning}}
  a |= a; // expected-warning {{explicitly assigning}}
  a ^= a;
}

// Dummy type.
struct S {};

void false_positives() {
#define OP =
#define LHS a
#define RHS a
  int a = 42;
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

  // Volatile stores aren't side-effect free.
  volatile int vol_a;
  vol_a = vol_a;
  volatile int &vol_a_ref = vol_a;
  vol_a_ref = vol_a_ref;
}

// Do not diagnose self-assigment in an unevaluated context
void false_positives_unevaluated_ctx(int a) noexcept(noexcept(a = a)) // expected-warning {{expression with side effects has no effect in an unevaluated context}}
{
  decltype(a = a) b = a;              // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  static_assert(noexcept(a = a), ""); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  static_assert(sizeof(a = a), "");   // expected-warning {{expression with side effects has no effect in an unevaluated context}}
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

struct Foo {
  int A;

  Foo(int A) : A(A) {}

  void setA(int A) {
    A = A; // expected-warning{{explicitly assigning value of variable of type 'int' to itself; did you mean to assign to member 'A'?}}
  }

  void setThroughLambda() {
    [](int A) {
      // To fix here we would need to insert an explicit capture 'this'
      A = A; // expected-warning{{explicitly assigning}}
    }(5);

    [this](int A) {
      this->A = 0;
      // This fix would be possible by just adding this-> as above, but currently unsupported.
      A = A; // expected-warning{{explicitly assigning}}
    }(5);
  }
};
