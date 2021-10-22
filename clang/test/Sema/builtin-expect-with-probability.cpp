// RUN: %clang_cc1 -fsyntax-only -verify %s

__attribute__((noreturn)) extern void bar();

int test_no_warn(int x) {
  if (x) {
    if (__builtin_expect_with_probability(1, 1, 1))
      bar();
  } else {
    return 0;
  }
} // should not emit warn "control may reach end of non-void function" here since expr is constantly true, so the "if(__bui..)" should be constantly true condition and be ignored

template <int b> void tempf() {
  static_assert(b == 1, "should be evaluated as 1"); // should not have error here
}

constexpr int constf() {
  return __builtin_expect_with_probability(1, 1, 1);
}

void foo() {
  tempf<__builtin_expect_with_probability(1, 1, 1)>();
  constexpr int f = constf();
  static_assert(f == 1, "should be evaluated as 1"); // should not have error here
}

extern int global;

struct S {
  static constexpr float prob = 0.7;
};

template<typename T>
void expect_taken(int x) {
  if (__builtin_expect_with_probability(x > 0, 1, T::prob)) {
    global++;
  }
}

void test(int x, double p) { // expected-note {{declared here}}
  bool dummy;
  dummy = __builtin_expect_with_probability(x > 0, 1, 0.9);
  dummy = __builtin_expect_with_probability(x > 0, 1, 1.1); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, -1); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, p); // expected-error {{probability argument to __builtin_expect_with_probability must be constant floating-point expression}} expected-note {{function parameter 'p'}}
  dummy = __builtin_expect_with_probability(x > 0, 1, "aa"); // expected-error {{cannot initialize a parameter of type 'double' with an lvalue of type 'const char[3]'}}
  dummy = __builtin_expect_with_probability(x > 0, 1, __builtin_nan("")); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, __builtin_inf()); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, -0.0);
  dummy = __builtin_expect_with_probability(x > 0, 1, 1.0 + __DBL_EPSILON__); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  dummy = __builtin_expect_with_probability(x > 0, 1, -__DBL_DENORM_MIN__); // expected-error {{probability argument to __builtin_expect_with_probability is outside the range [0.0, 1.0]}}
  constexpr double pd = 0.7;
  dummy = __builtin_expect_with_probability(x > 0, 1, pd);
  constexpr int pi = 1;
  dummy = __builtin_expect_with_probability(x > 0, 1, pi);
  expect_taken<S>(x);
}
