// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wno-all -Wunsafe-buffer-usage -fexperimental-bounds-safety-attributes -verify %s

#include <ptrcheck.h>
#include <stddef.h>

namespace std {

template <typename T, size_t N>
struct array {
  T &operator[](size_t n) noexcept;
};

template <typename CharT>
struct basic_string {
  CharT &operator[](size_t n) noexcept;
};

typedef basic_string<char> string;

template <typename CharT>
struct basic_string_view {
  const CharT &operator[](size_t n) const noexcept;
};

typedef basic_string_view<char> string_view;

template <typename T>
struct span {
  T *data() const noexcept;
  span<T> first(size_t count) const noexcept;
  span<T> last(size_t count) const noexcept;
  span<T> subspan(size_t offset, size_t count) const noexcept;
  T &operator[](size_t n) noexcept;
};

template <typename T>
struct vector {
  T &operator[](size_t n) noexcept;
};

}  // namespace std

template <typename T>
struct my_vec {
  T &operator[](size_t n) noexcept;
};

extern "C" {

void single_char(char *__single s);
void single_cchar(const char *__single s);
void single_int(int *__single p);
void single_void(void *__single p);

void single_int_int(int *__single p, int *__single q);

}  // extern "C"

// Check passing to `void *__single`.

void pass_to_single_void(void *pv, std::span<int> sp, my_vec<int> &mv) {
  char array[42] = {};

  single_void(pv);

  single_void(sp.data());
  single_void(sp.first(1).data());
  single_void(sp.first(42).data());
  single_void(&sp[42]);

  single_void(&mv[0]);

  single_void(array);
}

// Check passing `nullptr`.

void null() {
  single_char(nullptr);
  single_cchar(nullptr);
  single_int(nullptr);
  single_void(nullptr);
}

// Check `&var` pattern.

void addr_of_var() {
  char c = 0;
  single_char(&c);
  single_cchar(&c);
  single_void(&c);

  int i = 0;
  single_int(&i);
  single_void(&i);
}

// Check allowed classes in `&C[index]` pattern.

void allowed_class(std::array<int, 42> &a, std::string &s, std::string_view sv,
                   std::span<int> sp, std::vector<int> &v) {
  single_int(&a[0]);
  single_void(&a[0]);

  single_char(&s[0]);
  single_cchar(&s[0]);
  single_void(&s[0]);

  single_cchar(&sv[0]);

  single_int(&sp[0]);
  single_void(&sp[0]);

  single_int(&v[0]);
  single_void(&v[0]);
}

void not_allowed_class(my_vec<int> &mv) {
  single_int(&mv[0]); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
}

// Check if index doesn't matter in `&C[index]` pattern.

void index_does_not_matter(std::span<int> sp, size_t index) {
  single_int(&sp[0]);
  single_int(&sp[1]);
  single_int(&sp[index]);
  single_int(&sp[42 - index]);
}

// Check span's subview pattern.

void span_subview(std::span<int> sp, int n) {
  single_int(sp.first(1).data());
  single_int(sp.first(0).data()); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
  single_int(sp.first(n).data()); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}

  single_int(sp.last(1).data());
  single_int(sp.last(0).data()); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
  single_int(sp.last(n).data()); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}

  single_int(sp.subspan(0, 1).data());
  single_int(sp.subspan(42, 1).data());
  single_int(sp.subspan(n, 1).data());
  single_int(sp.subspan(0, 0).data()); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
  single_int(sp.subspan(0, n).data()); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
}

// Check multiple args.

void multiple_args(int i, int *p) {
  single_int_int(&i, &i);
  single_int_int(&i, p); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
  single_int_int(p, &i); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}

  single_int_int(
      p, // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
      p  // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
  );
}

// Check common unsafe patterns.

void unsafe(std::span<int> sp, int *p) {
  single_int(sp.data()); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}

  single_int(p); // expected-warning{{unsafe assignment to function parameter of __single pointer type}}
}
