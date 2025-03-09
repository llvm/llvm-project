// RUN: %clang_cc1 -std=c++23 -verify=expected,cxx20_plus %s

#ifdef __ASSERT_FUNCTION
#undef __ASSERT_FUNCTION
#endif
extern "C" void __assert_fail(const char*, const char*, unsigned, const char*);

#define assert(cond) \
  ((cond) ? (void)0 : __assert_fail(#cond, __FILE__, __LINE__, __func__))

consteval int square(int x) {
  int result = x * x;
  assert(result == 42); // expected-note {{assertion failed in consteval context: 'result == 42'}}
  return result;
}

void test() {
  auto val = square(2); // expected-note {{in call to 'square(2)'}} \
  // expected-error {{call to consteval function 'square' is not a constant expression}}
}
