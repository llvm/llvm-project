// RUN: %clang_cc1 -fsyntax-only -Wdynamic-class-memaccess -verify %s

extern "C" void *memset(void *, int, unsigned);
extern "C" void *memmove(void *s1, const void *s2, unsigned n);
extern "C" void *memcpy(void *s1, const void *s2, unsigned n);
extern "C" void *memcmp(void *s1, const void *s2, unsigned n);

// Several types that should not warn.
struct S1 {} s1;
struct S2 { int x; } s2;
struct S3 { float x, y; S1 s[4]; void (*f)(S1**); } s3;

class C1 {
  int x, y, z;
public:
  void foo() {}
} c1;

struct X1 { virtual void f(); } x1;
struct X2 : virtual S1 {} x2;

void test_warn() {
  memset(&x1, 0, sizeof x1); // \
      // expected-warning {{destination for this 'memset' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x2, 0, sizeof x2); // \
      // expected-warning {{destination for this 'memset' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}

  memmove(&x1, 0, sizeof x1); // \
      // expected-warning{{destination for this 'memmove' call is a pointer to dynamic class 'struct X1'; vtable pointer will be overwritten}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memmove(0, &x1, sizeof x1); // \
      // expected-warning{{source of this 'memmove' call is a pointer to dynamic class 'struct X1'; vtable pointer will be moved}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memcpy(&x1, 0, sizeof x1); // \
      // expected-warning{{destination for this 'memcpy' call is a pointer to dynamic class 'struct X1'; vtable pointer will be overwritten}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memcpy(0, &x1, sizeof x1); // \
      // expected-warning{{source of this 'memcpy' call is a pointer to dynamic class 'struct X1'; vtable pointer will be copied}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memcmp(&x1, 0, sizeof x1); // \
      // expected-warning{{first operand of this 'memcmp' call is a pointer to dynamic class 'struct X1'; vtable pointer will be compared}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memcmp(0, &x1, sizeof x1); // \
      // expected-warning{{second operand of this 'memcmp' call is a pointer to dynamic class 'struct X1'; vtable pointer will be compared}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}

  __builtin_memset(&x1, 0, sizeof x1); // \
      // expected-warning {{destination for this '__builtin_memset' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin_memset(&x2, 0, sizeof x2); // \
      // expected-warning {{destination for this '__builtin_memset' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}

  __builtin_memmove(&x1, 0, sizeof x1); // \
      // expected-warning{{destination for this '__builtin_memmove' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin_memmove(0, &x1, sizeof x1); // \
      // expected-warning{{source of this '__builtin_memmove' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin_memcpy(&x1, 0, sizeof x1); // \
      // expected-warning{{destination for this '__builtin_memcpy' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin_memcpy(0, &x1, sizeof x1); // \
      // expected-warning{{source of this '__builtin_memcpy' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}

  __builtin___memset_chk(&x1, 0, sizeof x1, sizeof x1); //                    \
      // expected-warning {{destination for this '__builtin___memset_chk' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin___memset_chk(&x2, 0, sizeof x2, sizeof x2); //                    \
      // expected-warning {{destination for this '__builtin___memset_chk' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}

  __builtin___memmove_chk(&x1, 0, sizeof x1, sizeof x1); //                   \
      // expected-warning{{destination for this '__builtin___memmove_chk' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin___memmove_chk(0, &x1, sizeof x1, sizeof x1); //                   \
      // expected-warning{{source of this '__builtin___memmove_chk' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin___memcpy_chk(&x1, 0, sizeof x1, sizeof x1); //                    \
      // expected-warning{{destination for this '__builtin___memcpy_chk' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  __builtin___memcpy_chk(0, &x1, sizeof x1, sizeof x1); //                    \
      // expected-warning{{source of this '__builtin___memcpy_chk' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
}

void test_nowarn(void *void_ptr) {
  int i, *iptr;
  float y;
  char c;

  memset(&i, 0, sizeof i);
  memset(&iptr, 0, sizeof iptr);
  memset(&y, 0, sizeof y);
  memset(&c, 0, sizeof c);
  memset(void_ptr, 0, 42);
  memset(&s1, 0, sizeof s1);
  memset(&s2, 0, sizeof s2);
  memset(&s3, 0, sizeof s3);
  memset(&c1, 0, sizeof c1);

  // Unevaluated code shouldn't warn.
  (void)sizeof memset(&x1, 0, sizeof x1);

  // Dead code shouldn't warn.
  if (false) memset(&x1, 0, sizeof x1);
}

namespace N {
  void *memset(void *, int, unsigned);
  void test_nowarn() {
    N::memset(&x1, 0, sizeof x1);
  }
}
