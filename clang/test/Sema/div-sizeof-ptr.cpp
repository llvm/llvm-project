// RUN: %clang_cc1 %s -verify -Wsizeof-pointer-div -fsyntax-only

template <typename Ty, int N>
int f(Ty (&Array)[N]) {
  return sizeof(Array) / sizeof(Ty); // Should not warn
}

typedef int int32;

void test(int *p, int **q) {          // expected-note 6 {{pointer 'p' declared here}}
  const int *r;                       // expected-note {{pointer 'r' declared here}}
  int a1 = sizeof(p) / sizeof(*p);    // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}
  int a2 = sizeof p / sizeof *p;      // expected-warning {{'sizeof p' will return the size of the pointer, not the array itself}}
  int a3 = sizeof(p) / sizeof(int);   // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}
  int a4 = sizeof(p) / sizeof(p[0]);  // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}
  int a5 = sizeof(p) / sizeof(int32); // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}
  int a6 = sizeof(r) / sizeof(int);   // expected-warning {{'sizeof (r)' will return the size of the pointer, not the array itself}}

  int32 *d;                           // expected-note 2 {{pointer 'd' declared here}}
  int a7 = sizeof(d) / sizeof(int32); // expected-warning {{'sizeof (d)' will return the size of the pointer, not the array itself}}
  int a8 = sizeof(d) / sizeof(int);  // expected-warning {{'sizeof (d)' will return the size of the pointer, not the array itself}}

  int a9 = sizeof(*q) / sizeof(**q); // expected-warning {{'sizeof (*q)' will return the size of the pointer, not the array itself}}
  int a10 = sizeof(p) / sizeof(decltype(*p)); // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}

  // Should not warn
  int b1 = sizeof(int *) / sizeof(int);
  int b2 = sizeof(p) / sizeof(p);
  int b3 = sizeof(*q) / sizeof(q);
  int b4 = sizeof(p) / sizeof(char);

  int arr[10];
  int c1 = sizeof(arr) / sizeof(*arr);
  int c2 = sizeof(arr) / sizeof(arr[0]);
  int c3 = sizeof(arr) / sizeof(int);

  int arr2[10][12];
  int d1 = sizeof(arr2) / sizeof(*arr2);
}
