#include "shared.h"

struct WrapperA {
  OuterY::Inner<unsigned int> y;
};

int main() {
  // WrapperA refers to the Inner and Outer class DIEs from this CU.
  WrapperA a;
  // WrapperB refers to the Inner and Outer DIEs from the other.cpp CU.
  // It is important that WrapperB is only forward-declared in shared.h.
  WrapperB* b = foo();

  // Evaluating 'b' here will parse other.cpp's DIEs for all
  // the Inner and Outer classes from shared.h.
  //
  // Evaluating 'a' here will find and reuse the already-parsed
  // versions of the Inner and Outer classes. In the associated test
  // we make sure that we can still resolve all the types properly
  // by evaluating 'a.y.oY_inner.oX_inner'.
  return 0;  // break here
}
