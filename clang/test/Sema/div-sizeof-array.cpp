// RUN: %clang_cc1 %s -verify -Wsizeof-array-div -fsyntax-only

template <typename Ty, int N>
int f(Ty (&Array)[N]) {
  return sizeof(Array) / sizeof(Ty); // Should not warn
}

typedef int int32;

void test(void) {
  int arr[12]; // expected-note 2 {{array 'arr' declared here}}
  unsigned long long arr2[4]; // expected-note {{array 'arr2' declared here}}
  int *p = &arr[0];
  int a1 = sizeof(arr) / sizeof(*arr);
  int a2 = sizeof arr / sizeof p;       // expected-warning {{expresion will return the incorrect number of elements in the array; the array element type is 'int', not 'int *'}}
  int a4 = sizeof arr2 / sizeof p;      // expected-warning {{expresion will return the incorrect number of elements in the array; the array element type is 'unsigned long long', not 'int *'}}
  int a5 = sizeof(arr) / sizeof(short); // expected-warning {{expresion will return the incorrect number of elements in the array; the array element type is 'int', not 'short'}}
  int a6 = sizeof(arr) / sizeof(int32);
  const char arr3[2] = "A";
  int a7 = sizeof(arr3) / sizeof(char);

  int arr4[10][12];
  int b1 = sizeof(arr4) / sizeof(arr2[12]);
  int b2 = sizeof(arr4) / sizeof(int *);
  int b3 = sizeof(arr4) / sizeof(short *);
}
