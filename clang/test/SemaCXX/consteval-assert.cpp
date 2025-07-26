// RUN: %clang_cc1 -std=c++23 -verify -DTEST_LINUX %s
// RUN: %clang_cc1 -std=c++23 -verify -DTEST_WINDOWS %s
// RUN: %clang_cc1 -std=c++23 -verify -DTEST_DARWIN %s

// RUN: %clang_cc1 -std=c++23 -verify -DTEST_LINUX %s   -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++23 -verify -DTEST_WINDOWS %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++23 -verify -DTEST_DARWIN %s  -fexperimental-new-constant-interpreter

#ifdef __ASSERT_FUNCTION
#undef __ASSERT_FUNCTION
#endif

#if defined(TEST_LINUX)
  extern "C" void __assert_fail(const char*, const char*, unsigned, const char*);
  #define assert(cond) \
    ((cond) ? (void)0 : __assert_fail(#cond, __FILE__, __LINE__, __func__))
#elif defined(TEST_DARWIN)
  void __assert_rtn(const char *, const char *, int, const char *);
  #define assert(cond) \
  (__builtin_expect(!(cond), 0) ? __assert_rtn(__func__, __FILE__, __LINE__, #cond) : (void)0)
#elif defined(TEST_WINDOWS)
  void /*__cdecl*/ _wassert(const wchar_t*, const wchar_t*, unsigned);
  #define _CRT_WIDE_(s) L ## s
  #define _CRT_WIDE(s) _CRT_WIDE_(s)
  #define assert(cond) \
    (void)((!!(cond)) || (_wassert(_CRT_WIDE(#cond), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0))
#endif

consteval int square(int x) {
  int result = x * x;
  assert(result == 42); // expected-note {{assertion failed during evaluation of constant expression}}
  return result;
}

void test() {
  auto val = square(2); // expected-note {{in call to 'square(2)'}} \
  // expected-error {{call to consteval function 'square' is not a constant expression}}
}
