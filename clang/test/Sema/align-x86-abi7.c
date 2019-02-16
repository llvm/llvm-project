// RUN: %clang_cc1 -std=c11 -triple i386-apple-darwin9 -fsyntax-only -verify -fclang-abi-compat=7 %s
// expected-no-diagnostics

#define STATIC_ASSERT(cond) _Static_assert(cond, #cond)

// PR3433
#define CHECK_ALIGNMENT(type, name, pref) \
  type name; \
  STATIC_ASSERT(__alignof__(name) == pref); \
  STATIC_ASSERT(__alignof__(type) == pref); \
  STATIC_ASSERT(_Alignof(type) == pref)

CHECK_ALIGNMENT(double, g_double, 8);
CHECK_ALIGNMENT(long long, g_longlong, 8);
CHECK_ALIGNMENT(unsigned long long, g_ulonglong, 8);

typedef double arr3double[3];
CHECK_ALIGNMENT(arr3double, g_arr3double, 8);

enum big_enum { x = 18446744073709551615ULL };
CHECK_ALIGNMENT(enum big_enum, g_bigenum, 8);