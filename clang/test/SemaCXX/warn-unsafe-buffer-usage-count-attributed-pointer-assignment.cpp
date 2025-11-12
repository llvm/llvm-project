// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wno-all -Wunsafe-buffer-usage -fexperimental-bounds-safety-attributes -verify %s

#include <ptrcheck.h>
#include <stddef.h>
#include <stdint.h>

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

struct cb_multi {
  int *__counted_by(m * n) p;
  size_t m;
  size_t n;
};

struct cb_nested {
  cb nested;
};

// Simple pointer and count

void good_null(int *__counted_by(count) p, int count) {
  p = nullptr;
  count = 0;
}

void good_simple_self(int *__counted_by(count) p, int count) {
  p = p;
  count = count;
}

void good_simple_other(int *__counted_by(count) p, int count, int *__counted_by(len) q, int len) {
  p = q;
  count = len;
}

void good_simple_span(int *__counted_by(count) p, size_t count, std::span<int> sp) {
  p = sp.data();
  count = sp.size();
}

void bad_simple_span(int *__counted_by(count) p, size_t count, std::span<int> sp) {
  p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  count = 42;
}

void good_simple_subspan_const(int *__counted_by(count) p, int count, std::span<int> sp) {
  p = sp.first(42).data();
  count = 42;
}

void bad_simple_subspan_const(int *__counted_by(count) p, int count, std::span<int> sp) {
  p = sp.first(42).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  count = 43;
}

void good_simple_subspan_var(int *__counted_by(count) p, int count, std::span<int> sp, int new_count) {
  p = sp.first(new_count).data();
  count = new_count;
}

void bad_simple_subspan_var(int *__counted_by(count) p, int count, std::span<int> sp, int new_count, int new_count2) {
  p = sp.first(new_count).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  count = new_count2;
}

void good_null_loop(int *__counted_by(count) p, int count) {
  for (int i = 0; i < 2; i++) {
    p = nullptr;
    count = 0;
  }
}

void good_simple_loop(int *__counted_by(count) p, int count, std::span<int> sp) {
  for (int i = 0; i < 2; i++) {
    p = sp.data();
    count = sp.size();
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

void good_simple_ifelse(int *__counted_by(count) p, size_t count, std::span<int> a, std::span<int> b) {
  if (count % 2 == 0) {
    p = a.data();
    count = a.size();
  } else {
    count = b.size();
    p = b.data();
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

void good_size_bytes_char(char *__counted_by(count) p, size_t count, std::span<char> sp) {
  p = sp.data();
  count = sp.size_bytes();
}

void good_size_bytes_void(void *__sized_by(size) p, size_t size, std::span<uint8_t> sp) {
  p = sp.data();
  size = sp.size_bytes();
}

// Simple pointer and count in struct

void good_struct_self(cb *c) {
  c->p = c->p;
  c->count = c->count;
}

void good_struct_other_struct(cb *c, cb *d) {
  c->p = d->p;
  c->count = d->count;
}

void good_struct_other_param(cb *c, int *__counted_by(count) p, int count) {
  c->p = p;
  c->count = count;
}

void good_struct_span(cb *c, std::span<int> sp) {
  c->p = sp.data();
  c->count = sp.size();
}

void bad_struct_span(cb *c, std::span<int> sp) {
  c->p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  c->count = 42;
}

void good_struct_subspan(cb *c, std::span<int> sp) {
  c->p = sp.first(42).data();
  c->count = 42;
}

void bad_struct_subspan(cb *c, std::span<int> sp) {
  c->p = sp.first(42).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  c->count = 43;
}

void bad_struct_unrelated_structs(cb *c, cb *d, cb *e) {
  c->p = d->p; // expected-warning{{unsafe assignment to count-attributed pointer}}
  c->count = e->count;
}

void good_struct_self_loop(cb *c) {
  for (int i = 0; i < 2; i++) {
    c->p = c->p;
    c->count = c->count;
  }
}

void good_struct_nested_span(cb_nested *n, std::span<int> sp) {
  n->nested.p = sp.data();
  n->nested.count = sp.size();
}

void bad_struct_nested_span(cb_nested *n, std::span<int> sp, size_t unrelated_size) {
  n->nested.p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  n->nested.count = unrelated_size;
}

class struct_test {
  int *__counted_by(count_) data_;
  size_t count_;
  cb cb_;
  cb_nested nested_;
  cb *p_cb_;

  void set_data(std::span<int> sp) {
    data_ = sp.data();
    count_ = sp.size();
  }

  void bad_set_data(std::span<int> sp) {
    data_ = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
    count_ = 42;
  }

  void set_cb(std::span<int> sp) {
    cb_.p = sp.data();
    cb_.count = sp.size();
  }

  void bad_set_cb(std::span<int> sp) {
    cb_.p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
    cb_.count = 42;
  }

  void set_nested(std::span<int> sp) {
    nested_.nested.p = sp.data();
    nested_.nested.count = sp.size();
  }

  void set_p_cb(std::span<int> sp) {
    p_cb_->p = sp.data();
    p_cb_->count = sp.size();
  }
};

// Pointer with multiple counts

void bad_multicounts_span(int *__counted_by(a + b) p, size_t a, size_t b, std::span<int> sp) {
  p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  a = sp.size();
  b = sp.size();
}

void good_multicounts_subspan_const(int *__counted_by(a + b) p, int a, int b, std::span<int> sp) {
  p = sp.first(42 + 100).data();
  a = 42;
  b = 100;
}

void bad_multicounts_subspan_const(int *__counted_by(a + b) p, int a, int b, std::span<int> sp) {
  p = sp.first(42 + 100).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  a = 42;
  b = 99;
}

// TODO: Currently, we only do pattern matching against count expr. With some
// const-eval, we could support the following pattern.
void todo_multicounts_subspan_const(int *__counted_by(a + b) p, int a, int b, std::span<int> sp) {
  p = sp.first(142).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  a = 42;
  b = 100;
}

void good_multicounts_subspan_var(int *__counted_by(a + b) p, int a, int b, std::span<int> sp, int n, int m) {
  p = sp.first(n + m).data();
  a = n;
  b = m;
}

void bad_multicounts_subspan_var(int *__counted_by(a + b) p, int a, int b, std::span<int> sp, int n, int m) {
  p = sp.first(n + m).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  a = n;
  b = n;
}

// Pointer with multiple counts in struct

void bad_multicount_struct_span(cb_multi *cbm, std::span<int> sp) {
  cbm->p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  cbm->m = sp.size();
  cbm->n = sp.size();
}

void good_multicount_struct_subspan_const(cb_multi *cbm, std::span<int> sp) {
  cbm->p = sp.first(42 * 100).data();
  cbm->m = 42;
  cbm->n = 100;
}

void bad_multicount_struct_subspan_const(cb_multi *cbm, std::span<int> sp) {
  cbm->p = sp.first(42 * 100).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  cbm->m = 43;
  cbm->n = 100;
}

void good_multicount_struct_subspan_var(cb_multi *cbm, std::span<int> sp, size_t a, size_t b) {
  cbm->p = sp.first(a * b).data();
  cbm->m = a;
  cbm->n = b;
}

void bad_multicount_struct_subspan_var(cb_multi *cbm, std::span<int> sp, size_t a, size_t b) {
  cbm->p = sp.first(a * b).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  cbm->m = a;
  cbm->n = a;
}

bool good_multicount_struct_realistic(cb_multi *cbm, std::span<int> sp, size_t stride) {
  if (sp.size() % stride != 0)
    return false;
  size_t length = sp.size() / stride;
  cbm->p = sp.first(length * stride).data();
  cbm->m = length;
  cbm->n = stride;
  return true;
}

struct multicount_struct_test {
  cb_multi multi_;

  void set_multi(std::span<int> sp, size_t a, size_t b) {
    multi_.p = sp.first(a * b).data();
    multi_.m = a;
    multi_.n = b;
  }
};

// Multiple pointers

void good_multiptr_span(int *__counted_by(count) p, int *__counted_by(count) q, size_t count, std::span<int> sp) {
  p = sp.data();
  q = sp.data();
  count = sp.size();
}

void bad_multiptr_span(int *__counted_by(count) p, int *__counted_by(count) q, size_t count, std::span<int> sp) {
  p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  q = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  count = 42;
}

void bad_multiptr_span_subspan(int *__counted_by(count) p, int *__counted_by(count) q, size_t count, std::span<int> sp) {
  p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  q = sp.first(42).data();
  count = 42;
}

void bad_multiptr_span_subspan2(int *__counted_by(count) p, int *__counted_by(count) q, size_t count, std::span<int> sp) {
  p = sp.data();
  q = sp.first(42).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  count = sp.size();
}

void good_multiptr_subspan(int *__counted_by(count) p, int *__counted_by(count) q, size_t count, std::span<int> sp) {
  p = sp.first(42).data();
  q = sp.last(42).data();
  count = 42;
}

// Multiple pointers with multiple counts

void good_multimix_subspan_complex(int *__counted_by(a * b) p, int *__counted_by((a + b) * c) q, size_t a, size_t b, size_t c,
                                   std::span<int> sp, size_t i, size_t j, size_t k) {
  p = sp.first(i * j).data();
  q = sp.last((i + j) * k).data();
  a = i;
  b = j;
  c = k;
}

void bad_multimix_subspan_complex(int *__counted_by(a * b) p, int *__counted_by((a + b) * c) q, size_t a, size_t b, size_t c,
                                  std::span<int> sp, size_t i, size_t j, size_t k) {
  p = sp.first(i * j).data();      // expected-warning{{unsafe assignment to count-attributed pointer}}
  q = sp.last((i + j) * k).data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
  a = k;
  b = j;
  c = i;
}

void good_multimix_subspan_complex2(int *__counted_by(a * b) p, int *__counted_by((a + b) * c) q, size_t a, size_t b, size_t c,
                                    std::span<int> sp, size_t i, size_t j, size_t k) {
  a = i * 2;
  b = j;
  c = j + k;
  p = sp.first((i * 2) * j).data();
  q = sp.last(((i * 2) + j) * (j + k)).data();
}

void good_multimix_subspan_complex_multispan(int *__counted_by(a * b) p, int *__counted_by((a + b) * c) q, size_t a, size_t b, size_t c,
                                   std::span<int> sp, std::span<int> sp2, size_t i, size_t j, size_t k) {
  a = i;
  b = j;
  c = k;
  p = sp.first(i * j).data();
  q = sp2.first((i + j) * k).data();
}

// Inout pointer and count

void good_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
  *p = sp.data();
  *count = sp.size();
}

void bad_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
  *p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
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
    *p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
    *count = size;
  }
}

class inout_class {
  void good_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
    *p = sp.data();
    *count = sp.size();
  }

  void bad_inout_span(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
    *p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
    *count = 42;
  }

  void good_inout_subspan_const(int *__counted_by(*count) *p, size_t *count, std::span<int> sp) {
    *p = sp.first(42).data();
    *count = 42;
  }
};

// Inout pointer

void bad_inout_ptr_span(int *__counted_by(count) *p, int count, std::span<int> sp) {
  *p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
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
    *p = sp.data(); // expected-warning{{unsafe assignment to count-attributed pointer}}
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

// Missing assignments

void missing_ptr(int *__counted_by(count) p, int count) {
  count = 0; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
}

void missing_count(int *__counted_by(count) p, int count) {
  p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
}

void missing_structure(int *__counted_by(count) p, int count) {
  {
    p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
  }
  {
    count = 0;   // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
  }
}

void missing_structure2(int *__counted_by(count) p, int count) {
  p = nullptr;   // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
  {
    count = 0;   // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
  }
}

void missing_structure3(int *__counted_by(count) p, int count) {
  p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
  if (count > 0) {
    count = 0; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
  }
}

void missing_unrelated(int *__counted_by(count) p, int count, int *__counted_by(len) q, int len) {
  p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
  len = 0;     // expected-warning{{bounds-attributed group requires assigning 'len, q', assignments to 'q' missing}}
}

void missing_complex_count1(int *__counted_by(a + b) p, int a, int b) {
  p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'a, b, p', assignments to 'a, b' missing}}
}

void missing_complex_count2(int *__counted_by(a + b) p, int a, int b) {
  p = nullptr;
  a = 0; // expected-warning{{bounds-attributed group requires assigning 'a, b, p', assignments to 'b' missing}}
}

void missing_complex_count3(int *__counted_by(a + b) p, int a, int b) {
  b = 0;
  p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'a, b, p', assignments to 'a' missing}}
}

void missing_complex_count4(int *__counted_by(a + b) p, int a, int b) {
  a = 0;
  b = 0; // expected-warning{{bounds-attributed group requires assigning 'a, b, p', assignments to 'p' missing}}
}

void missing_complex_ptr1(int *__counted_by(count) p, int *__counted_by(count) q, int count) {
  p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p, q', assignments to 'count, q' missing}}
}

void missing_complex_ptr2(int *__counted_by(count) p, int *__counted_by(count) q, int count) {
  p = nullptr;
  q = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p, q', assignments to 'count' missing}}
}

void missing_complex_ptr3(int *__counted_by(count) p, int *__counted_by(count) q, int count) {
  count = 0;
  p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p, q', assignments to 'q' missing}}
}

void missing_complex_ptr4(int *__counted_by(count) p, int *__counted_by(count) q, int count) {
  q = nullptr;
  count = 0; // expected-warning{{bounds-attributed group requires assigning 'count, p, q', assignments to 'p' missing}}
}

// Missing assignments in struct

void missing_struct_ptr(cb *c) {
  c->count = 0; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
}

void missing_struct_count(cb *c) {
  c->p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
}

void missing_struct_unrelated(cb *c, cb *d) {
  c->p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
  d->count = 0;   // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
}

void missing_struct_nested_ptr(cb_nested *c) {
  c->nested.count = 0; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
}

void missing_struct_nested_unrelated(cb_nested *c, cb_nested *d) {
  c->nested.p = nullptr; // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
  d->nested.count = 0;   // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
}

struct missing_struct_test {
  cb cb_;
  cb_nested nested_;

  void set_cb_missing_ptr(std::span<int> sp) {
    cb_.count = sp.size(); // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
  }

  void set_cb_missing_count(std::span<int> sp) {
    cb_.p = sp.data(); // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
  }

  void set_nested_missing_ptr(std::span<int> sp) {
    nested_.nested.count = sp.size(); // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
  }

  void set_missing_unrelated(std::span<int> sp) {
    cb_.p = sp.data();                // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'count' missing}}
    nested_.nested.count = sp.size(); // expected-warning{{bounds-attributed group requires assigning 'count, p', assignments to 'p' missing}}
  }
};

// Duplicated assignments

void duplicated_ptr(int *__counted_by(count) p, int count) {
  p = nullptr; // expected-note{{previously assigned here}}
  p = nullptr; // expected-warning{{duplicated assignment to parameter 'p' in bounds-attributed group}}
  count = 0;
}

void duplicated_ptr2(int *__counted_by(count) p, int count) {
  p = nullptr; // expected-note{{previously assigned here}}
  count = 0;
  p = nullptr; // expected-warning{{duplicated assignment to parameter 'p' in bounds-attributed group}}
}

void duplicated_count(int *__counted_by(count) p, int count) {
  p = nullptr;
  count = 0; // expected-note{{previously assigned here}}
  count = 0; // expected-warning{{duplicated assignment to parameter 'count' in bounds-attributed group}}
}

void duplicated_count2(int *__counted_by(count) p, int count) {
  count = 0; // expected-note{{previously assigned here}}
  p = nullptr;
  count = 0; // expected-warning{{duplicated assignment to parameter 'count' in bounds-attributed group}}
}

void duplicated_complex(int *__counted_by(a + b) p,
                        int *__counted_by(a + b + c) q,
                        int a, int b, int c) {
  p = nullptr;
  q = nullptr; // expected-note{{previously assigned here}}
  a = 0;
  b = 0;
  c = 0;
  q = nullptr; // expected-warning{{duplicated assignment to parameter 'q' in bounds-attributed group}}
}

// Assigned and used

void good_assigned_and_used(int *__counted_by(count) p, int count, std::span<int> sp) {
  p = sp.first(count).data();
  count = count;
}

void bad_assigned_and_used(int *__counted_by(count) p, int count, std::span<int> sp, int new_count) {
  p = sp.first(count).data(); // expected-note{{used here}}
  count = new_count;          // expected-warning{{parameter 'count' is assigned and used in the same bounds-attributed group}}
}

void bad_assigned_and_used2(int *__counted_by(a + b) p, int a, int b, std::span<int> sp) {
  p = sp.first(b + 42).data(); // expected-note{{used here}}
  b = 42;                      // expected-warning{{parameter 'b' is assigned and used in the same bounds-attributed group}}
  a = b;
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
