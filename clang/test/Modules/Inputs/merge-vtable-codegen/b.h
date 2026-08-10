#ifndef B_H
#define B_H

#include "a.h"

class B : virtual public A {
  virtual void x() {}
};

void b(A* p) {
  p->x();
  // Instantiating a class that virtually inherits 'A'
  // triggers calculation of the vtable offsets in 'A'.
  B b;
}

#endif
