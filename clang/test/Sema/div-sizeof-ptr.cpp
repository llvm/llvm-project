// RUN: %clang_cc1 %s -verify -Wsizeof-pointer-div -fsyntax-only

template <typename Ty, int N>
int f(Ty (&Array)[N]) {
  return sizeof(Array) / sizeof(Ty); // Should not warn
}

void test(int *p, int **q) {
  int a1 = sizeof(p) / sizeof(*p);   // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}
  int a2 = sizeof p / sizeof *p;     // expected-warning {{'sizeof p' will return the size of the pointer, not the array itself}}
  int a3 = sizeof(*q) / sizeof(**q); // expected-warning {{'sizeof (*q)' will return the size of the pointer, not the array itself}}
  int a4 = sizeof(p) / sizeof(int);  // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}
  int a5 = sizeof(p) / sizeof(p[0]); // expected-warning {{'sizeof (p)' will return the size of the pointer, not the array itself}}

  // Should not warn
  int b1 = sizeof(int *) / sizeof(int);
  int b2 = sizeof(p) / sizeof(p);
  int b3 = sizeof(*q) / sizeof(q);
  int b4 = sizeof(p) / sizeof(char);

  int arr[10];
  int b5 = sizeof(arr) / sizeof(*arr);
  int b6 = sizeof(arr) / sizeof(arr[0]);
  int b7 = sizeof(arr) / sizeof(int);

  int arr2[10][12];
  int b8 = sizeof(arr2) / sizeof(*arr2);
}
