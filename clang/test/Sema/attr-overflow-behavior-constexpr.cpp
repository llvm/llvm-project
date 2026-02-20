// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -fexperimental-overflow-behavior-types -verify -fsyntax-only -std=c++14

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_trap __attribute__((overflow_behavior(trap)))

typedef int __ob_wrap wrap_int;
typedef int __ob_trap no_trap_int;

constexpr wrap_int add(wrap_int a, wrap_int b) {
  return a + b;
}

constexpr no_trap_int sub(no_trap_int a, no_trap_int b) {
  return a - b; // expected-note {{-2147483649 is outside the range of representable values}}
}

void constexpr_test() {
  constexpr wrap_int max = 2147483647;
  constexpr wrap_int one = 1;
  static_assert(add(max, one) == -2147483648, "constexpr wrapping failed");

  constexpr no_trap_int min = -2147483648;
  constexpr no_trap_int one_nw = 1;
  constexpr no_trap_int res = sub(min, one_nw); // expected-error {{constexpr variable 'res' must be initialized by a constant expression}} expected-note {{in call to 'sub(-2147483648, 1)'}}
}

template <typename T>
void check_deduction_wrap(T) {
  static_assert(__is_same(T, wrap_int), "T should be deduced as wrap_int");
}

template <typename T>
void check_deduction_no_trap(T) {
  static_assert(__is_same(T, no_trap_int), "T should be deduced as no_trap_int");
}

void template_deduction_test() {
  wrap_int w = 0;
  check_deduction_wrap(w);

  no_trap_int nw = 0;
  check_deduction_no_trap(nw);
}
