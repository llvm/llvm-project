// RUN: %clang_analyze_cc1 -analyzer-checker=unix.MallocSizeof -std=c++11 -verify %s

// Verify no false positives for layout-compatible types: a record type that
// wraps a scalar with identical size and alignment (e.g. std::atomic<int>
// wrapping int, or struct { int c; } vs int).

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

void test_no_false_negatives() {
  // Unrelated struct with the same size
  struct Color { float r; };
  int *p = (int *)malloc(sizeof(Color));
  // expected-warning@-1{{Result of 'malloc' is converted to a pointer of type 'int', which is incompatible with sizeof operand type 'Color'}}

  // Multi-field struct with the same size and alignment as the scalar
  struct Pair { int a; int b; };  // 8 bytes, align 4
  Pair *pr = (Pair *)malloc(sizeof(long));
  // expected-warning@-1{{Result of 'malloc' is converted to a pointer of type 'Pair', which is incompatible with sizeof operand type 'long'}}

  struct Status { int code; };
  float *f = (float *)malloc(sizeof(Status));
  // expected-warning@-1{{Result of 'malloc' is converted to a pointer of type 'float', which is incompatible with sizeof operand type 'Status'}}

  // Pointer-sized struct vs. pointer-sized scalar
  struct Handle { long opaque; };
  double *d = (double *)malloc(sizeof(Handle));
  // expected-warning@-1{{Result of 'malloc' is converted to a pointer of type 'double', which is incompatible with sizeof operand type 'Handle'}}
}
