// RUN: %clang_cc1 %s -fsyntax-only -verify -fsanitize=safe-stack

void *test_bottom(void) {
  return __builtin___get_unsafe_stack_bottom(); // expected-warning {{builtin '__builtin___get_unsafe_stack_bottom' is deprecated; use __safestack_get_unsafe_stack_bottom instead}}
}

void *test_top(void) {
  return __builtin___get_unsafe_stack_top(); // expected-warning {{builtin '__builtin___get_unsafe_stack_top' is deprecated; use __safestack_get_unsafe_stack_top instead}}
}

void *test_ptr(void) {
  return __builtin___get_unsafe_stack_ptr(); // expected-warning {{builtin '__builtin___get_unsafe_stack_ptr' is deprecated; use __safestack_get_unsafe_stack_ptr instead}}
}

void *test_start(void) {
  return __builtin___get_unsafe_stack_start(); // expected-warning {{builtin '__builtin___get_unsafe_stack_start' is deprecated; use __safestack_get_unsafe_stack_bottom instead}}
}
