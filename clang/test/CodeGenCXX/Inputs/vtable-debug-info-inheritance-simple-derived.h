#include "vtable-debug-info-inheritance-simple-base.h"

#ifndef DERIVED_H
#define DERIVED_H

struct CDerived : NSP::CBase {
  unsigned D = 2;
  void zero() override;
  int two() override;
  int three() override;
};

extern void fooDerived();
#endif
