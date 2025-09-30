// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wno-all -Wunsafe-buffer-usage -fexperimental-bounds-safety-attributes -verify %s

#include <ptrcheck.h>
#include <stddef.h>

namespace std {
template <typename T> struct span {
  T *data() const noexcept;
  size_t size() const noexcept;
  size_t size_bytes() const noexcept;
  span<T> first(size_t count) const noexcept;
  span<T> last(size_t count) const noexcept;
  span<T> subspan(size_t offset, size_t count) const noexcept;
  const T &operator[](const size_t idx) const;
};
} // namespace std

struct cb {
  int *__counted_by(count) p;
  size_t count;
};

// Simple pointer and count

void good_null(int *__counted_by(count) p, int count) {
  p = nullptr;
  count = 0;
}

void good_null_loop(int *__counted_by(count) p, int count) {
  for (int i = 0; i < 2; i++) {
    p = nullptr;
    count = 0;
  }
}

void good_null_ifelse(int *__counted_by(count) p, int count) {
  if (count > 10) {
    p = nullptr;
    count = 0;
  } else {
    count = 0;
    p = nullptr;
  }
}

void good_null_loop_if(int *__counted_by(count) p, int count) {
  for (int i = 0; i < 2; ++i) {
    if (i == 0) {
      p = nullptr;
      count = 0;
    }
    count = 0;
    p = nullptr;
  }
}

// Simple pointer and count in struct

void good_struct_self(cb *c) {
  c->p = c->p;
  c->count = c->count;
}

void good_struct_self_loop(cb *c) {
  for (int i = 0; i < 2; i++) {
    c->p = c->p;
    c->count = c->count;
  }
}

// Assigns to bounds-attributed that we consider too complex to analyze.

void too_complex_assign_to_ptr(int *__counted_by(count) p, int count, int *q) {
  q = p;
  q = p = q;     // expected-warning{{assignment to count-attributed pointer 'p' must be a simple statement 'p = ...'}}
  q = q = p = q; // expected-warning{{assignment to count-attributed pointer 'p' must be a simple statement 'p = ...'}}

  p++; // expected-warning{{unsafe pointer arithmetic}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

  if (count > 2) {
    q = p;
    q = p = q;     // expected-warning{{assignment to count-attributed pointer 'p' must be a simple statement 'p = ...'}}
    q = q = p = q; // expected-warning{{assignment to count-attributed pointer 'p' must be a simple statement 'p = ...'}}

    p++; // expected-warning{{unsafe pointer arithmetic}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  }

  if (count > 2) {
    if (count > 4)
      for (int i = 0; i < 2; i++)
        q = p = q; // expected-warning{{assignment to count-attributed pointer 'p' must be a simple statement 'p = ...'}}
  }

  for (; ; p++); // expected-warning{{unsafe pointer arithmetic}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
}

void too_complex_assign_to_count(int *__counted_by(count) p, int count, int n) {
  n = count;
  n = count = n;     // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
  n = n = count = n; // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}

  count++;         // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
  --count;         // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
  count += 42;     // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
  count -= 42;     // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
  n = n + count++; // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}

  if (n > 42) {
    n = count;
    n = count = n;     // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
    n = n = count = n; // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
  } else {
    count++;         // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
    --count;         // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
    count += 42;     // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
    count -= 42;     // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
    n = n + count++; // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
  }

  if (n > 42)
    if (count > 0)
      for (int i = 0; i < 2; i++)
        count++; // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}

  for (; ; count += 42); // expected-warning{{assignment to dependent count 'count' must be a simple statement 'count = ...'}}
}
