// RUN: %check_clang_tidy -std=c11-or-later %s misc-static-assert %t
// RUN: clang-tidy %s -checks=-*,misc-static-assert -- -std=c99 | count 0

void abort(void) {}
#ifdef NDEBUG
#define assert(x) 1
#else
#define assert(x)                                                              \
  if (!(x))                                                                    \
  abort()
#endif

void f(void) {
  int x = 1;
  assert(x == 0);
  // CHECK-FIXES: assert(x == 0);

  #define static_assert(x, msg) _Static_assert(x, msg)
  assert(11 == 5 + 6);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: static_assert(11 == 5 + 6, "");
  #undef static_assert

  assert(10 == 5 + 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: static_assert(10 == 5 + 5, "");
}
