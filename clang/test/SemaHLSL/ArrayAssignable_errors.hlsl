// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

void test_wrong_size1() {
  int Arr[2] = {0, 1};
  int Arr2[3] = {1, 2, 0};
  Arr = Arr2;
  // expected-error@-1 {{assigning to 'int[2]' from incompatible type 'int[3]'}}
}

void test_wrong_size2() {
  int Arr[2] = {0, 1};
  int Arr2[3] = {1, 2, 0};
  Arr2 = Arr;
  // expected-error@-1 {{assigning to 'int[3]' from incompatible type 'int[2]'}}
}

void test_wrong_size3() {
  int Arr[2][2] = {{0, 1}, {2, 3}};
  int Arr2[2] = {4, 5};
  Arr = Arr2;
  // expected-error@-1 {{assigning to 'int[2][2]' from incompatible type 'int[2]'}}
}

void test_wrong_size4() {
  int Arr[2][2] = {{0, 1}, {2, 3}};
  int Arr2[2] = {4, 5};
  Arr2 = Arr;
  // expected-error@-1 {{assigning to 'int[2]' from incompatible type 'int[2][2]'}}
}
