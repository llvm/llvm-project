// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -Wno-suspicious-memcmp -verify=quiet %s
// quiet-no-diagnostics

typedef __SIZE_TYPE__ size_t;
int memcmp(const void *s1, const void *s2, size_t n);
int bcmp(const void *s1, const void *s2, size_t n);

struct Padded { char tag; int x; };     // 3 padding bytes after 'tag'
struct Dense { int a; int b; };         // no padding
struct WithFloat { float x; float y; }; // no padding, but float encodings
struct WithAtomic { _Atomic(int) x; };  // layout-identical to int
union Slack { char c; int i; };         // 'c' leaves 3 bytes of slack
struct NeverDefined;

void test_padded(struct Padded *a, struct Padded *b, size_t n) {
  memcmp(a, b, sizeof(struct Padded)); // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'struct Padded' which does not have a unique object representation; consider comparing the members of the object manually}} \
                                       // expected-note{{explicitly cast the pointer to silence this warning}}
  memcmp(a, b, 8);                     // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'struct Padded'}} \
                                       // expected-note{{explicitly cast the pointer to silence this warning}}
  memcmp(a, b, sizeof(int)); // prefix compare of leading members: no warning
  memcmp(a, b, n);           // non-constant size: no warning
  memcmp((const void *)a, (const void *)b, sizeof(struct Padded)); // silenced
}

void test_spellings(struct Padded *a, struct Padded *b) {
  bcmp(a, b, sizeof(struct Padded)); // expected-warning{{first operand of this 'bcmp' call is a pointer to type 'struct Padded'}} \
                                     // expected-note{{explicitly cast the pointer to silence this warning}}
  __builtin_memcmp(a, b, sizeof(struct Padded)); // expected-warning{{first operand of this '__builtin_memcmp' call is a pointer to type 'struct Padded'}} \
                                                 // expected-note{{explicitly cast the pointer to silence this warning}}
}

void test_scalars(float *x, float *y) {
  memcmp(x, y, sizeof(float)); // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'float' which does not have a unique object representation; consider comparing the values manually}} \
                               // expected-note{{explicitly cast the pointer to silence this warning}}
}

void test_float_struct(struct WithFloat *a, struct WithFloat *b) {
  memcmp(a, b, sizeof(struct WithFloat)); // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'struct WithFloat' which does not have a unique object representation; consider comparing the members of the object manually}} \
                                          // expected-note{{explicitly cast the pointer to silence this warning}}
}

void test_union(union Slack *a, union Slack *b) {
  memcmp(a, b, sizeof(union Slack)); // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'union Slack'}} \
                                     // expected-note{{explicitly cast the pointer to silence this warning}}
}

void test_arrays(void) {
  struct Padded a[3], b[3];
  memcmp(a, b, sizeof(a)); // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'struct Padded}} \
                           // expected-note{{explicitly cast the pointer to silence this warning}}
}

// No warnings below this point.

void test_dense(struct Dense *a, struct Dense *b) {
  memcmp(a, b, sizeof(struct Dense));
}

void test_atomic(struct WithAtomic *a, struct WithAtomic *b) {
  memcmp(a, b, sizeof(struct WithAtomic));
}

void test_incomplete(struct NeverDefined *a, struct NeverDefined *b) {
  memcmp(a, b, 16);
}
