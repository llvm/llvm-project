// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-static-sized-array \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fstrict-flex-arrays=0 -verify=expected,level0,level01,level012 %s
// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-static-sized-array \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fstrict-flex-arrays=1 -verify=expected,level01,level012 %s
// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-static-sized-array \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fstrict-flex-arrays=2 -verify=expected,level012 %s
// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-static-sized-array \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fstrict-flex-arrays=3 -verify=expected %s

// -Wno-unsafe-buffer-usage-in-static-sized-array exists for code built with
// -fsanitize=array-bounds, which bounds-checks subscripts on arrays of known
// size.  The sanitizer does not trust the declared size of a trailing array
// member that -fstrict-flex-arrays treats as a flexible array member, but the
// opt-out silences accesses to those too.

struct Zero {
  int len;
  int buf[0];
};

struct One {
  int len;
  int buf[1];
};

struct Many {
  int len;
  int buf[16];
};

struct Incomplete {
  int len;
  int buf[];
};

struct NotTrailing {
  int buf[16];
  int len;
};

union U {
  int x;
  int buf[1];
};

void zero(Zero *z, unsigned idx) {
  z->buf[idx] = 0;
}

void one(One *o, unsigned idx) {
  o->buf[idx] = 0;
  // The struct hack: a constant index past the declared size.
  o->buf[1] = 0;
}

void many(Many *m, unsigned idx) {
  m->buf[idx] = 0;
  m->buf[3] = 0;   // a constant index within the declared size is always safe
  m->buf[20] = 0;
}

void incomplete(Incomplete *i, unsigned idx) {
  i->buf[idx] = 0; // expected-warning{{unsafe buffer access}}
}

void not_trailing(NotTrailing *n, unsigned idx) {
  n->buf[idx] = 0;
}

void union_member(U *u, unsigned idx) {
  u->buf[idx] = 0;
}

struct Method {
  int len;
  int buf[16];

  void set(unsigned idx) {
    buf[idx] = 0;
  }
};
