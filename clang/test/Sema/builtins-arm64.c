// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-apple-ios -DTEST1 -fsyntax-only -verify %s

#ifdef TEST1
void __clear_cache(void *start, void *end);
#endif

void test_clear_cache_chars(char *start, char *end) {
  __clear_cache(start, end);
}

void test_clear_cache_voids(void *start, void *end) {
  __clear_cache(start, end);
}

void test_clear_cache_no_args(void) {
  __clear_cache(); // expected-error {{too few arguments to function call}}
}

void test_memory_barriers(void) {
  __builtin_arm_dmb(16); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_dsb(17); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_isb(18); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_prefetch(void) {
  __builtin_arm_prefetch(0, 2, 0, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_prefetch(0, 0, 4, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_prefetch(0, 0, 0, 2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_prefetch(0, 0, 0, 0, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_trap(short s, unsigned short us) {
  __builtin_arm_trap(42);
  __builtin_arm_trap(65535);
  __builtin_arm_trap(-1);
  __builtin_arm_trap(65536); // expected-warning {{implicit conversion from 'int' to 'unsigned short' changes value from 65536 to 0}}
  __builtin_arm_trap(s); // expected-error {{argument to '__builtin_arm_trap' must be a constant integer}}
  __builtin_arm_trap(us); // expected-error {{argument to '__builtin_arm_trap' must be a constant integer}}
}