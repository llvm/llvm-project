// RUN: %clang_cc1 -std=c++1z -verify %s

int foo(int n);

int Cilk_for_tests(int n) {
  /* int n = 10; */
  /* _Cilk_for(int i = 0; i < n; i += 2); */
  /* _Cilk_for(int j = 0, __begin = 0, __end = n/2; __begin < __end; j += 2, __begin++); */
  _Cilk_for (int i = 0; i < n; ++i); // expected-warning {{Cilk for loop has empty body}}
  _Cilk_for (int i = 0, __end = n; i < __end; ++i); // expected-warning {{Cilk for loop has empty body}}
  unsigned long long m = 10;
  _Cilk_for (int i = 0; i < m; ++i); // expected-warning {{Cilk for loop has empty body}}
  _Cilk_for (int i = 0, __end = m; i < __end; ++i); // expected-warning {{Cilk for loop has empty body}}
  _Cilk_for (int i = 0; i < n; ++i) return 7; // expected-error{{cannot return}}
  _Cilk_for (int i = 0; i < n; ++i) // expected-error{{cannot break}}
    if (i % 2)
      break;
  return 0;
}

int pragma_tests(int n) {
#pragma clang loop unroll_count(4)
  _Cilk_for (int i = 0; i < n; ++i)
    foo(i);

#pragma cilk grainsize(4)
  _Cilk_for (int i = 0; i < n; ++i)
    foo(i);

#pragma cilk grainsize 4
  _Cilk_for (int i = 0; i < n; ++i)
    foo(i);

#pragma cilk grainsize = 4 \
// expected-error{{expected expression}}                              \
   expected-warning{{extra tokens at end of '#pragma cilk grainsize' - ignored}}
  _Cilk_for (int i = 0; i < n; ++i)
    foo(i);

  return 0;
}

int scope_tests(int n) {
  int A[5];
  _Cilk_for(int i = 0; i < n; ++i) {
    int A[5];
    A[i%5] = i;
  }
  for(int i = 0; i < n; ++i) {
    A[i%5] = i%5;
  }
  return 0;
}
