// RUN: %clang_cc1 %s -E -CC -verify
// RUN: %clang_cc1 %s -E -CC -DPEDANTIC -pedantic -verify

#define FOO && 1
#if defined FOO FOO
#endif

#define A
#define B defined(A)
#if B // expected-warning{{macro expansion producing 'defined' has undefined behavior}}
#endif

#define m_foo
#define TEST(a) (defined(m_##a) && a)

#if defined(PEDANTIC)
// expected-warning@+4{{macro expansion producing 'defined' has undefined behavior}}
#endif

// This shouldn't warn by default, only with pedantic:
#if TEST(foo)
#endif


// Only one diagnostic for this case:
#define INVALID defined(
#if INVALID // expected-error{{macro name missing}}
#endif
