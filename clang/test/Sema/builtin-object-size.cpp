// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -fstrict-flex-arrays=0 -DSTRICT0 -std=c++23 -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -fstrict-flex-arrays=1 -DSTRICT1 -std=c++23 -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -fstrict-flex-arrays=2 -DSTRICT2 -std=c++23 -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -fstrict-flex-arrays=3 -DSTRICT3 -std=c++23 -verify %s

struct EmptyS {
  int i;
  char a[];
};

template <unsigned N>
struct S {
  int i;
  char a[N];
};

extern S<2> &s2;
static_assert(__builtin_object_size(s2.a, 0)); // expected-error {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_object_size(s2.a, 1) == 2);
#if defined(STRICT0)
// expected-error@-2 {{static assertion expression is not an integral constant expression}}
#endif
static_assert(__builtin_object_size(s2.a, 2) == 4);
static_assert(__builtin_object_size(s2.a, 3) == 2);

extern S<1> &s1;
static_assert(__builtin_object_size(s1.a, 0)); // expected-error {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_object_size(s1.a, 1) == 1);
#if defined(STRICT0) || defined(STRICT1)
// expected-error@-2 {{static assertion expression is not an integral constant expression}}
#endif
static_assert(__builtin_object_size(s1.a, 2) == 4);
static_assert(__builtin_object_size(s1.a, 3) == 1);

extern S<0> &s0;
static_assert(__builtin_object_size(s0.a, 0)); // expected-error {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_object_size(s0.a, 1) == 0);
#if defined(STRICT0) || defined(STRICT1) || defined(STRICT2)
// expected-error@-2 {{static assertion expression is not an integral constant expression}}
#endif
static_assert(__builtin_object_size(s0.a, 2) == 0);
static_assert(__builtin_object_size(s0.a, 3) == 0);

extern EmptyS &empty;
static_assert(__builtin_object_size(empty.a, 0)); // expected-error {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_object_size(empty.a, 1)); // expected-error {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_object_size(empty.a, 2) == 0);
static_assert(__builtin_object_size(empty.a, 3)); // expected-error {{static assertion expression is not an integral constant expression}}
