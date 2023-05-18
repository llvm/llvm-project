void f1(int *ptr); // expected-warning{{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}

void f2(int * _Nonnull);

#include "nullability-consistency-2.h"

void f3(int *ptr) { // expected-warning{{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
  int *other = ptr; // shouldn't warn
}

class X {
  void mf(int *ptr); // expected-warning{{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
  int X:: *memptr; // expected-warning{{member pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the member pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the member pointer should never be null}}
};

template <typename T>
struct Typedefs {
  typedef T *Base; // no-warning
  typedef Base *type; // expected-warning{{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
};

Typedefs<int> xx;
Typedefs<void *> yy;


