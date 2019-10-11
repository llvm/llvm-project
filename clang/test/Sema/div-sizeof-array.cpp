// RUN: %clang_cc1 %s -verify -Wsizeof-array-div -fsyntax-only -triple=x86_64-linux-gnu
// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=x86_64-linux-gnu

template <typename Ty, int N>
int f(Ty (&Array)[N]) {
  return sizeof(Array) / sizeof(Ty); // Should not warn
}

typedef int int32;

void test(void) {
  int arr[12]; // expected-note 2 {{array 'arr' declared here}}
  unsigned long long arr2[4];
  int *p = &arr[0];
  int a1 = sizeof(arr) / sizeof(*arr);
  int a2 = sizeof arr / sizeof p; // expected-warning {{expression does not compute the number of elements in this array; element type is 'int', not 'int *'}}
  // expected-note@-1 {{place parentheses around the 'sizeof p' expression to silence this warning}}
  int a4 = sizeof arr2 / sizeof p;
  int a5 = sizeof(arr) / sizeof(short); // expected-warning {{expression does not compute the number of elements in this array; element type is 'int', not 'short'}}
  // expected-note@-1 {{place parentheses around the 'sizeof(short)' expression to silence this warning}}
  int a6 = sizeof(arr) / sizeof(int32);
  int a7 = sizeof(arr) / sizeof(int);
  int a9 = sizeof(arr) / sizeof(unsigned int);
  const char arr3[2] = "A";
  int a10 = sizeof(arr3) / sizeof(char);
  int a11 = sizeof(arr2) / (sizeof(unsigned));
  int a12 = sizeof(arr) / (sizeof(short));
  int a13 = sizeof(arr3) / sizeof(p);
  int a14 = sizeof(arr3) / sizeof(int);

  int arr4[10][12];
  int b1 = sizeof(arr4) / sizeof(arr2[12]);
  int b2 = sizeof(arr4) / sizeof(int *);
  int b3 = sizeof(arr4) / sizeof(short *);
  int arr5[][5] = {
      {1, 2, 3, 4, 5},
      {6, 7, 8, 9, 0},
  };
  int c1 = sizeof(arr5) / sizeof(*arr5);
  int c2 = sizeof(arr5) / sizeof(**arr5);
  int c3 = sizeof(arr5) / sizeof(int);

  float m[4][4];
  int d1 = sizeof(m) / sizeof(**m);
}
