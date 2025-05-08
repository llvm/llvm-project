// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-unknown -verify -std=c++20 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection

template <typename T> void clang_analyzer_dump(T);
using size_t = decltype(sizeof(int));

__attribute__((always_inline)) static inline constexpr unsigned int _castf32_u32(float __A) {
  return __builtin_bit_cast(unsigned int, __A); // no-warning
}

void test(int i) {
  _castf32_u32(42);

  float f = 42;

  // Loading from a floating point value results in unknown,
  // which later materializes as a conjured value.
  auto g = __builtin_bit_cast(unsigned int, f);
  clang_analyzer_dump(g);
  // expected-warning-re@-1 {{{{^conj_\$[0-9]+{unsigned int,}}}}

  auto g2 = __builtin_bit_cast(unsigned int, 42.0f);
  clang_analyzer_dump(g2);
  // expected-warning-re@-1 {{{{^conj_\$[0-9]+{unsigned int,}}}}

  auto g3 = __builtin_bit_cast(unsigned int, i);
  clang_analyzer_dump(g3);
  // expected-warning-re@-1 {{{{^reg_\$[0-9]+<int i>}}}}

  auto g4 = __builtin_bit_cast(unsigned long, &i);
  clang_analyzer_dump(g4);
  // expected-warning@-1 {{&i [as 64 bit integer]}}
}

struct A {
  int n;
  void set(int x) {
    n = x;
  }
};
void gh_69922(size_t p) {
  // expected-warning@+1 {{Unknown}}
  clang_analyzer_dump(__builtin_bit_cast(A*, p & 1));

  __builtin_bit_cast(A*, p & 1)->set(2); // no-crash
  // However, since the `this` pointer is expected to be a Loc, but we have
  // NonLoc there, we simply give up and resolve it as `Unknown`.
  // Then, inside the `set()` member function call we can't evaluate the
  // store to the member variable `n`.

  clang_analyzer_dump(__builtin_bit_cast(A*, p & 1)->n); // Ideally, this should print "2".
  // expected-warning@-1 {{Unknown}}
}

static void issue_71174() {
  auto res = __builtin_bit_cast(unsigned long long, &issue_71174) | 1; // no-crash
}

#if __cplusplus >= 202002L
#include "Inputs/system-header-simulator-cxx.h"
using intptr_t = decltype(sizeof(int*));

namespace std {
template< class To, class From >
constexpr To bit_cast( const From& from ) noexcept {
  #if __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(To, from);
#else
  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
#endif
}
}

bool issue_137417(std::string* x) {
  return x == std::bit_cast<std::string*>(static_cast<intptr_t>(-1)); // no-crash
}
#endif