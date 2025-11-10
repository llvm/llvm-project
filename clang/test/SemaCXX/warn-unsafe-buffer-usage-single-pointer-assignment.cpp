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

// Check assignment to `void *__single`.

void single_void(void *__single p_void, void *pv, std::span<int> sp, my_vec<int> &mv) {
  char array[42] = {};

  p_void = pv;

  p_void = sp.data();
  p_void = sp.first(1).data();
  p_void = sp.first(42).data();
  p_void = &sp[42];

  p_void = &mv[0];

  p_void = array;
}

// Check `nullptr`.

void null(char *__single p_char,
          const char *__single p_cchar,
          int *__single p_int,
          void *__single p_void) {
  p_char = nullptr;
  p_cchar = nullptr;
  p_int = nullptr;
  p_void = nullptr;
}

// Check `&var` pattern.

void addr_of_var(char *__single p_char,
                 const char *__single p_cchar,
                 int *__single p_int,
                 void *__single p_void) {
  char c = 0;
  p_char = &c;
  p_cchar = &c;
  p_void = &c;

  int i = 0;
  p_int = &i;
  p_void = &i;
}

// Check allowed classes in `&C[index]` pattern.

void allowed_class(char *__single p_char,
                   const char *__single p_cchar,
                   int *__single p_int,
                   void *__single p_void,
                   std::array<int, 42> &a,
                   std::string &s,
                   std::string_view sv,
                   std::span<int> sp,
                   std::vector<int> &v) {
  p_int = &a[0];
  p_void = &a[0];

  p_char = &s[0];
  p_cchar = &s[0];
  p_void = &s[0];

  p_cchar = &sv[0];

  p_int = &sp[0];
  p_void = &sp[0];

  p_int = &v[0];
  p_void = &v[0];
}

void not_allowed_class(int *__single p_int, my_vec<int> &mv) {
  p_int = &mv[0]; // expected-warning{{unsafe assignment to __single pointer type}}
}

// Check if index doesn't matter in `&C[index]` pattern.

void index_does_not_matter(int *__single p_int, std::span<int> sp, size_t index) {
  p_int = &sp[0];
  p_int = &sp[1];
  p_int = &sp[index];
  p_int = &sp[42 - index];
}

// Check span's subview pattern.

void span_subview(int *__single p_int, std::span<int> sp, int n) {
  p_int = sp.first(1).data();
  p_int = sp.first(0).data(); // expected-warning{{unsafe assignment to __single pointer type}}
  p_int = sp.first(n).data(); // expected-warning{{unsafe assignment to __single pointer type}}

  p_int = sp.last(1).data();
  p_int = sp.last(0).data(); // expected-warning{{unsafe assignment to __single pointer type}}
  p_int = sp.last(n).data(); // expected-warning{{unsafe assignment to __single pointer type}}

  p_int = sp.subspan(0, 1).data();
  p_int = sp.subspan(42, 1).data();
  p_int = sp.subspan(n, 1).data();
  p_int = sp.subspan(0, 0).data(); // expected-warning{{unsafe assignment to __single pointer type}}
  p_int = sp.subspan(0, n).data(); // expected-warning{{unsafe assignment to __single pointer type}}
}

// Check common unsafe patterns.

void unsafe(int *__single p_int, std::span<int> sp, int *p) {
  p_int = sp.data(); // expected-warning{{unsafe assignment to __single pointer type}}

  p_int = p; // expected-warning{{unsafe assignment to __single pointer type}}
}
