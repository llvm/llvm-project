void f1(int *ptr); // expected-warning{{pointer is missing a nullability type specifier}}

void f2(int * _Nonnull);

#include "nullability-consistency-2.h"

void f3(int *ptr) { // expected-warning{{pointer is missing a nullability type specifier}}
  int *other = ptr; // shouldn't warn
}

class X {
  void mf(int *ptr); // expected-warning{{pointer is missing a nullability type specifier}}
  int X:: *memptr; // expected-warning{{member pointer is missing a nullability type specifier}}
};



