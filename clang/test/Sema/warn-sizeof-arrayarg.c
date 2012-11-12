// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int Arr[10];

typedef int trungl_int;

void f(int a[10], Arr arr) {  // \
// expected-note {{declared here}} \
// expected-note {{declared here}} \
// expected-note {{declared here}} \
// expected-note {{declared here}}

  /* Should warn. */
  (void)sizeof(a);  // \
      // expected-warning{{sizeof on array function parameter will return size of 'int *' instead of 'int [10]'}}
  (void)sizeof((((a))));  // \
      // expected-warning{{sizeof on array function parameter will return size of 'int *' instead of 'int [10]'}}
  (void)sizeof a;  // \
      // expected-warning{{sizeof on array function parameter will return size of 'int *' instead of 'int [10]'}}
  (void)sizeof arr;  // \
      // expected-warning{{sizeof on array function parameter will return size of 'int *' instead of 'Arr' (aka 'int [10]')}}

  /* Shouldn't warn. */
  int b[10];
  (void)sizeof b;
  Arr brr;
  (void)sizeof brr;
  (void)sizeof(Arr);
  (void)sizeof(int);
}
