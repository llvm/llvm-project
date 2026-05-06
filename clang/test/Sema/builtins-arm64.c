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

void test_range_prefetch(void) {
  __builtin_arm_range_prefetch(0, 2, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_range_prefetch(0, 0, 2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  __builtin_arm_range_prefetch_x(0, 2, 0, 0, 0, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_range_prefetch_x(0, 0, 2, 0, 0, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_range_prefetch_x(0, 0, 0, -2097153, 0, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_range_prefetch_x(0, 0, 0, 2097152, 0, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_range_prefetch_x(0, 0, 0, 0, 65537, 0, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_range_prefetch_x(0, 0, 0, 0, 0, -2097153, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_arm_range_prefetch_x(0, 0, 0, 0, 0, 2097152, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_trap(short s, unsigned short us) {
  __builtin_arm_trap(42);
  __builtin_arm_trap(65535);
  __builtin_arm_trap(-1);
  __builtin_arm_trap(65536); // expected-warning {{implicit conversion from 'int' to 'unsigned short' changes value from 65536 to 0}}
  __builtin_arm_trap(s); // expected-error {{argument to '__builtin_arm_trap' must be a constant integer}}
  __builtin_arm_trap(us); // expected-error {{argument to '__builtin_arm_trap' must be a constant integer}}
}

void test_atomic_store_hint(char *c_ptr, __int128 *inv_ptr, float *f_ptr,
                            char c_data, __int128 inv_data, float f_data,
                            int inv_int) {
  __builtin_arm_atomic_store_with_hint(c_ptr, c_data, 0); // expected-error {{too few arguments to function call, expected 4, have 3}}
  __builtin_arm_atomic_store_with_hint(c_ptr, c_data, 0, 0, 0); // expected-error {{too many arguments to function call, expected 4, have 5}}

  __builtin_arm_atomic_store_with_hint(0, c_data, 0, 0); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  __builtin_arm_atomic_store_with_hint(c_ptr, f_data, 0, 0); // expected-error {{arguments are of different types ('char' vs 'float')}}
  __builtin_arm_atomic_store_with_hint(inv_ptr, inv_data, 0, 0); // expected-error {{address argument to atomic store with hint must be of size 8, 16, 32 or 64 bits}}

  __builtin_arm_atomic_store_with_hint(c_ptr, c_data, inv_int, 0); // expected-error {{invalid memory order argument to atomic hint operation ('int' invalid)}}
  __builtin_arm_atomic_store_with_hint(c_ptr, c_data, 2, 0); // expected-error {{invalid memory order argument to atomic hint operation (2 invalid)}}

  __builtin_arm_atomic_store_with_hint(c_ptr, c_data, 0, inv_int); // expected-error {{invalid hint type argument to atomic hint operation ('int' invalid)}}
  __builtin_arm_atomic_store_with_hint(c_ptr, c_data, 0, 3); // expected-error {{invalid hint type argument to atomic hint operation (3 invalid)}}
}
