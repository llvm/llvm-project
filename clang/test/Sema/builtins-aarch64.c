// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -DTEST1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -DTEST1 -fsyntax-only -verify %s

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
  // AArch32 version of this is variadic (at least syntactically).
  // However, on AArch64 GCC does not permit this call and the
  // implementation I've seen would go disastrously wrong.
  __clear_cache(); // expected-error {{too few arguments to function call}}
}
