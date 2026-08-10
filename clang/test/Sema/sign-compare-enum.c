// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -Wsign-compare %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -verify -Wsign-compare %s
// RUN: %clang_cc1 -x c++ -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -Wsign-compare %s
// RUN: %clang_cc1 -x c++ -triple=x86_64-pc-win32 -fsyntax-only -verify -Wsign-compare %s

// Check that -Wsign-compare is off by default.
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -DSILENCE %s

#ifdef SILENCE
// expected-no-diagnostics
#endif

enum PosEnum {
  A_a = 0,
  A_b = 1,
  A_c = 10
};

static const int message[] = {0, 1};

int test_pos(enum PosEnum a) {
  if (a < 2)
    return 0;

  // No warning, except in Windows C mode, where PosEnum is 'int' and it can
  // take on any value according to the C standard.
#if !defined(SILENCE) && defined(_WIN32) && !defined(__cplusplus)
  // expected-warning@+2 {{comparison of integers of different signs}}
#endif
  if (a < 2U)
    return 0;

  unsigned uv = 2;
#if !defined(SILENCE) && defined(_WIN32) && !defined(__cplusplus)
  // expected-warning@+2 {{comparison of integers of different signs}}
#endif
  if (a < uv)
    return 1;

#if !defined(SILENCE) && defined(_WIN32) && !defined(__cplusplus)
  // expected-warning@+2 {{comparison of integers of different signs}}
#endif
  if (a < sizeof(message)/sizeof(message[0]))
    return 0;
  return 4;
}

enum NegEnum {
  NE_a = -1,
  NE_b = 0,
  NE_c = 10
};

int test_neg(enum NegEnum a) {
  if (a < 2)
    return 0;

#ifndef SILENCE
  // expected-warning@+2 {{comparison of integers of different signs}}
#endif
  if (a < 2U)
    return 0;

  unsigned uv = 2;
#ifndef SILENCE
  // expected-warning@+2 {{comparison of integers of different signs}}
#endif
  if (a < uv)
    return 1;

#ifndef SILENCE
  // expected-warning@+2 {{comparison of integers of different signs}}
#endif
  if (a < sizeof(message)/sizeof(message[0]))
    return 0;
  return 4;
}
