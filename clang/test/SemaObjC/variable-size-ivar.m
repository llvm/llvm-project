// RUN: %clang_cc1 -fsyntax-only %s -verify

const int ksize = 42;
int size = 42;

@interface X
{
  int arr1[ksize]; // expected-warning{{variable length array folded to constant array}}
  int arr2[size]; // expected-error{{instance variables must have a constant size}}
  int arr3[ksize-43]; // expected-error{{array size is negative}}
}
@end
