// RUN: %clang_cc1 %s -verify -Wsizeof-array-div -fsyntax-only

template <typename Ty, int N>
int f(Ty (&Array)[N]) {
  return sizeof(Array) / sizeof(Ty); // Should not warn
}

typedef int int32;

void test(void) {
  int arr[12];                // expected-note 2 {{array 'arr' declared here}}
  unsigned long long arr2[4];
  int *p = &arr[0];
  int a1 = sizeof(arr) / sizeof(*arr);
  int a2 = sizeof arr / sizeof p; // expected-warning {{expression does not compute the number of elements in this array; element type is 'int', not 'int *'}}
  int a4 = sizeof arr2 / sizeof p;
  int a5 = sizeof(arr) / sizeof(short); // expected-warning {{expression does not compute the number of elements in this array; element type is 'int', not 'short'}}
  int a6 = sizeof(arr) / sizeof(int32);
  int a7 = sizeof(arr) / sizeof(int);
  int a9 = sizeof(arr) / sizeof(unsigned int);
  const char arr3[2] = "A";
  int a10 = sizeof(arr3) / sizeof(char);

  int arr4[10][12];                         // expected-note 3 {{array 'arr4' declared here}}
  int b1 = sizeof(arr4) / sizeof(arr2[12]); // expected-warning {{expression does not compute the number of elements in this array; element type is 'int [12]', not 'unsigned long long'}}
  int b2 = sizeof(arr4) / sizeof(int *);    // expected-warning {{expression does not compute the number of elements in this array; element type is 'int [12]', not 'int *'}}
  int b3 = sizeof(arr4) / sizeof(short *);  // expected-warning {{expression does not compute the number of elements in this array; element type is 'int [12]', not 'short *'}}
}
