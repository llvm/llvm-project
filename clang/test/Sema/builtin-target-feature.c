// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify %s

void test_good_cpu_condition(void) {
  if (__builtin_target_is_cpu("haswell"))
    ;
}

void test_good_is_invocable(void) {
  if (__builtin_is_invocable(__builtin_popcount))
    ;
}

const char *str = "avx2";

_Bool test_not_literal_cpu(void) {
  return __builtin_target_is_cpu(str); // expected-error {{expression is not a string literal}}
}

void not_a_builtin(void);

_Bool test_is_invocable_string_literal(void) {
  return __builtin_is_invocable("avx2"); // expected-error {{expression must be a valid builtin function for the target}}
}

_Bool test_is_invocable_non_builtin(void) {
  return __builtin_is_invocable(not_a_builtin); // expected-error {{expression must be a valid builtin function for the target}}
}

_Bool test_is_invocable_variable(void) {
  return __builtin_is_invocable(str); // expected-error {{expression must be a valid builtin function for the target}}
}
