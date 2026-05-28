// RUN: %clang_analyze_cc1 -analyzer-checker=unix.MallocSizeof -std=c++11 -verify %s

// Verify no false positives for layout-compatible types: a record type that
// wraps a scalar with identical size and alignment (e.g. std::atomic<int>
// wrapping int, or struct { int c; } vs int).

// expected-no-diagnostics

typedef int int32_t;
using size_t = unsigned long long;
void *malloc(size_t size);
void free(void *ptr);

namespace std {
// Minimal atomic stub with the same size/alignment as T.
template <typename T>
struct atomic {
  T _value;
};
} // namespace std

typedef std::atomic<int32_t> u_atomic_int32_t;

struct s_int {
  int32_t c;
};

template <typename T>
void work() {
  u_atomic_int32_t *p1 = (u_atomic_int32_t *)malloc(sizeof(int32_t));
  free(p1);

  T *p2 = (T *)malloc(sizeof(int32_t));
  free(p2);
}

int main() {
  work<u_atomic_int32_t>();
  work<int>();
  work<s_int>();
  return 0;
}
