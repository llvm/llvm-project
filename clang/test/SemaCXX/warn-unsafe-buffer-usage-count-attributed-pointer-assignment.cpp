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

// Inout pointer and count

void good_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
  *p = sp.data();
  *count = sp.size();
}

void bad_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
  *p = sp.data(); // TODO-expected-warning{{unsafe assignment to count-attributed pointer}}
  *count = 42;
}

void good_inout_subspan_const(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
  *p = sp.first(42).data();
  *count = 42;
}

void good_inout_subspan_var(int *__counted_by(*count) *p, size_t *count, std::span<int> sp, size_t new_count) {
  *p = sp.first(new_count).data();
  *count = new_count;
}

void good_inout_subspan_complex(int *__counted_by(*count) *p, size_t *count, std::span<int> sp, size_t i, size_t j) {
  *p = sp.first(i + j * 2).data();
  *count = i + j * 2;
}

void good_inout_span_if(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
  if (p && count) {
    *p = sp.data();
    *count = sp.size();
  }
}

void bad_inout_span_if(int *__counted_by(*count) *p, size_t *count, std::span<int> sp, size_t size) {
  if (p && count) {
    *p = sp.data(); // TODO-expected-warning{{unsafe assignment to count-attributed pointer}}
    *count = size;
  }
}

class inout_class {
  void good_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
    *p = sp.data();
    *count = sp.size();
  }

  void bad_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
    *p = sp.data(); // TODO-expected-warning{{unsafe assignment to count-attributed pointer}}
    *count = 42;
  }

  void good_inout_subspan_const(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
    *p = sp.first(42).data();
    *count = 42;
  }
};

// Inout pointer

void bad_inout_ptr_span(int *__counted_by(count) *p, int count, std::span<int> sp) {
  *p = sp.data(); // TODO-expected-warning{{unsafe assignment to count-attributed pointer}}
}

void good_inout_ptr_subspan(int *__counted_by(count) *p, size_t count, std::span<int> sp) {
  *p = sp.first(count).data();
}

void good_inout_ptr_const_subspan(int *__counted_by(42) *p, std::span<int> sp) {
  *p = sp.first(42).data();
}

void good_inout_ptr_multi_subspan(int *__counted_by(a + b) *p, size_t a, size_t b, std::span<int> sp) {
  *p = sp.first(a + b).data();
}

class inout_ptr_class {
  void bad_inout_ptr_span(int *__counted_by(count) *p, int count, std::span<int> sp) {
    *p = sp.data(); // TODO-expected-warning{{unsafe assignment to count-attributed pointer}}
  }

  void good_inout_ptr_subspan(int *__counted_by(count) *p, size_t count, std::span<int> sp) {
    *p = sp.first(count).data();
  }
};

// Immutable pointers/dependent values

void immutable_ptr_to_ptr(int *__counted_by(*count) *p, int *count) {
  p = nullptr; // expected-warning{{cannot assign to parameter 'p' because it points to a count-attributed pointer}}
  *count = 0;
}

void immutable_ptr_to_value(int *__counted_by(*count) *p, int *count) {
  *p = nullptr;
  count = nullptr; // expected-warning{{cannot assign to parameter 'count' because it points to a dependent count}}
}

void immutable_ptr_with_inout_value(int *__counted_by(*count) p, int *count) {
  p = nullptr; // expected-warning{{cannot assign to parameter 'p' because its type depends on an inout dependent count}}
  *count = 0;
}

void immutable_ptr_with_inout_value2(int *__counted_by(*count) p, int *__counted_by(*count) *q, int *count) {
  p = nullptr;  // expected-warning{{cannot assign to parameter 'p' because its type depends on an inout dependent count}}
  *q = nullptr;
  *count = 0;
}

void immutable_value_with_inout_ptr(int *__counted_by(count) *p, int count) {
  *p = nullptr;
  count = 0; // expected-warning{{cannot assign to parameter 'count' because it's used as dependent count in an inout count-attributed pointer}}
}

void immutable_value_with_inout_ptr2(int *__counted_by(count) p, int *__counted_by(count) *q, int count) {
  p = nullptr;
  *q = nullptr;
  count = 0; // expected-warning{{cannot assign to parameter 'count' because it's used as dependent count in an inout count-attributed pointer}}
}

class immutable_class {
  void immutable_ptr_to_ptr(int *__counted_by(*count) *p, int *count) {
    p = nullptr; // expected-warning{{cannot assign to parameter 'p' because it points to a count-attributed pointer}}
    *count = 0;
  }

  void immutable_ptr_to_value(int *__counted_by(*count) *p, int *count) {
    *p = nullptr;
    count = nullptr; // expected-warning{{cannot assign to parameter 'count' because it points to a dependent count}}
  }

  void immutable_ptr_with_inout_value(int *__counted_by(*count) p, int *count) {
    p = nullptr; // expected-warning{{cannot assign to parameter 'p' because its type depends on an inout dependent count}}
    *count = 0;
  }

  void immutable_value_with_inout_ptr(int *__counted_by(count) *p, int count) {
    *p = nullptr;
    count = 0; // expected-warning{{cannot assign to parameter 'count' because it's used as dependent count in an inout count-attributed pointer}}
  }
};

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
