// RUN: %clang_analyze_cc1 -analyzer-checker=unix.MallocSizeof -std=c++11 -verify %s

// Verify no false positives for layout-compatible types: a record type that
// wraps a scalar with identical size and alignment (e.g. std::atomic<int>
// wrapping int, or struct { int c; } vs int).

// expected-no-diagnostics

#include <stddef.h>

void *malloc(size_t size);
void free(void *ptr);

namespace std {
// Minimal atomic stub with the same size/alignment as T.
template <typename T>
struct atomic {
  T _value;
};
} // namespace std

typedef std::atomic<int> u_atomic_int_t;

struct s_int {
  int c;
};

template <typename T>
void work() {
  u_atomic_int_t *p1 = (u_atomic_int_t *)malloc(sizeof(int));
  free(p1);

  T *p2 = (T *)malloc(sizeof(int));
  free(p2);
}

int main() {
  work<u_atomic_int_t>();
  work<int>();
  work<s_int>();
  return 0;
}
