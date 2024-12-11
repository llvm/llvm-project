// REQUIRES: apple-disclosure-ios
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage-in-container -fexperimental-bounds-safety-attributes -verify %s

#include <ptrcheck.h>
#include <stddef.h>

namespace std {

template<typename T>
struct span {
  constexpr span(T *, size_t) {}
};

}  // namespace std

struct cb {
  size_t len;
  int *__counted_by(len) p;
};

void span_from_cb(cb *c, cb *d, size_t len) {
  std::span<int>{c->p, c->len};
  std::span<int>{c->p, 42};         // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, len};        // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, d->len};     // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, c->len + 1}; // expected-warning{{the two-parameter std::span construction is unsafe}}
}

struct cb_const {
  int *__counted_by(42) p;
};

void span_from_cb_const(cb_const *c, size_t len) {
  std::span<int>{c->p, 42};
  std::span<int>{c->p, 43};       // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, len};      // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, 42 + len}; // expected-warning{{the two-parameter std::span construction is unsafe}}
}

struct cb_multi {
  size_t a, b, c, d;
  int *__counted_by((a + b) * (c - d)) p;
};

void span_from_cb_multi(cb_multi *c, cb_multi *d, size_t len) {
  std::span<int>{c->p, (c->a + c->b) * (c->c - c->d)};
  std::span<int>{c->p, (c->a + c->b) * (c->c + c->d)}; // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, (c->a + c->b) + (c->c - c->d)}; // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, c->a + c->b};                   // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, 42};                            // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, (c->a + c->b) * (42 - c->d)};   // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, (d->a + d->b) * (d->c - d->d)}; // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{c->p, (c->a + d->b) * (c->c - c->d)}; // expected-warning{{the two-parameter std::span construction is unsafe}}
}

int *__counted_by(len) fn_cb(size_t len);

void span_from_fn_cb(cb *c, size_t len, size_t len2) {
  std::span<int>{fn_cb(len), len};
  std::span<int>{fn_cb(len), 42};      // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb(len), len2};    // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb(len), c->len};  // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb(len), len + 1}; // expected-warning{{the two-parameter std::span construction is unsafe}}
}

int *__counted_by(42) fn_cb_const();

void span_from_fn_cb_const(cb *c, size_t len) {
  std::span<int>{fn_cb_const(), 42};
  std::span<int>{fn_cb_const(), 43};     // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb_const(), len};    // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb_const(), c->len}; // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb_const(), 42 + 1}; // expected-warning{{the two-parameter std::span construction is unsafe}}
}

int *__counted_by((a + b) * (c - d)) fn_cb_multi(size_t a, size_t b, size_t c, size_t d);

void span_from_fn_cb_multi(cb *c, size_t w, size_t x, size_t y, size_t z) {
  std::span<int>{fn_cb_multi(4, 3, 2, 1), (4 + 3) * (2 - 1)};
  std::span<int>{fn_cb_multi(w, x, y, z), (w + x) * (y - z)};
  std::span<int>{fn_cb_multi(1, x, y, 2), (1 + x) * (y - 2)};
  std::span<int>{fn_cb_multi(x, x, x, x), (x + x) * (x - x)};
  std::span<int>{fn_cb_multi((1+2), (1+w), (x+2), (y+z)), ((1+2) + (1+w)) * ((x+2) - (y+z))};
  std::span<int>{fn_cb_multi(4, 3, 2, 1), (4 + 3) * (42 - 1)};      // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb_multi(4, 3, 2, 1), (4 + 3) * (x - 1)};       // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb_multi(4, 3, x, 1), (4 + 3) * (42 - 1)};      // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb_multi(w, x, y, z), (w + x) * (y - (z + z))}; // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<int>{fn_cb_multi(w, x, y, z), (w + x) * (y + z)};       // expected-warning{{the two-parameter std::span construction is unsafe}}
}

struct sb_int {
  size_t size;
  int *__sized_by(size) p;
};

struct sb_char {
  size_t size;
  char *__sized_by(size) p;
};

void span_from_sb_int(sb_int *i, sb_char *c) {
  std::span<int>{i->p, i->size}; // expected-warning{{the two-parameter std::span construction is unsafe}}
  std::span<char>{c->p, c->size};
}
