// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -xc++ -std=c++11 -fsyntax-only -verify %s

_Static_assert("foo", "string is nonzero");
#ifndef __cplusplus
// expected-error@-2 {{static_assert expression is not an integral constant expression}}
#endif

_Static_assert(1, "1 is nonzero");
_Static_assert(0, "0 is nonzero"); // expected-error {{static_assert failed "0 is nonzero"}}

void foo(void) {
  _Static_assert(1, "1 is nonzero");
  _Static_assert(0, "0 is nonzero"); // expected-error {{static_assert failed "0 is nonzero"}}
}

_Static_assert(1, invalid); // expected-error {{expected string literal for diagnostic message in static_assert}}

struct A {
  int a;
  _Static_assert(1, "1 is nonzero");
  _Static_assert(0, "0 is nonzero"); // expected-error {{static_assert failed "0 is nonzero"}}
};

#ifdef __cplusplus
#define ASSERT_IS_TYPE(T) __is_same(T, T)
#else
#define ASSERT_IS_TYPE(T) __builtin_types_compatible_p(T, T)
#endif

#define UNION(T1, T2) union { \
    __typeof__(T1) one; \
    __typeof__(T2) two; \
    _Static_assert(ASSERT_IS_TYPE(T1), "T1 is not a type"); \
    _Static_assert(ASSERT_IS_TYPE(T2), "T2 is not a type"); \
    _Static_assert(sizeof(T1) == sizeof(T2), "type size mismatch"); \
  }

typedef UNION(unsigned, struct A) U1;
UNION(char[2], short) u2 = { .one = { 'a', 'b' } };
typedef UNION(char, short) U3; // expected-error {{static_assert failed due to requirement 'sizeof(char) == sizeof(short)' "type size mismatch"}}
typedef UNION(float, 0.5f) U4; // expected-error {{expected a type}}
