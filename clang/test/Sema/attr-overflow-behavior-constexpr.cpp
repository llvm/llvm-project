// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -foverflow-behavior-types -verify -fsyntax-only -std=c++14

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __nowrap __attribute__((overflow_behavior(no_wrap)))

typedef int __wrap wrap_int;
typedef int __nowrap nowrap_int;

constexpr wrap_int add(wrap_int a, wrap_int b) {
  return a + b;
}

constexpr nowrap_int sub(nowrap_int a, nowrap_int b) {
  return a - b; // expected-note {{-2147483649 is outside the range of representable values}}
}

void constexpr_test() {
  constexpr wrap_int max = 2147483647;
  constexpr wrap_int one = 1;
  static_assert(add(max, one) == -2147483648, "constexpr wrapping failed");

  constexpr nowrap_int min = -2147483648;
  constexpr nowrap_int one_nw = 1;
  // This should fail to compile because of overflow.
  constexpr nowrap_int res = sub(min, one_nw); // expected-error {{constexpr variable 'res' must be initialized by a constant expression}} expected-note {{in call to 'sub(-2147483648, 1)'}}
}

template <typename T>
void check_deduction_wrap(T) {
  static_assert(__is_same(T, wrap_int), "T should be deduced as wrap_int");
}

template <typename T>
void check_deduction_nowrap(T) {
  static_assert(__is_same(T, nowrap_int), "T should be deduced as nowrap_int");
}

void template_deduction_test() {
  wrap_int w = 0;
  check_deduction_wrap(w);

  nowrap_int nw = 0;
  check_deduction_nowrap(nw);
}
