#include "vtable-debug-info-inheritance-simple-base.h"

void NSP::CBase::zero() {}
int NSP::CBase::one() { return 1; }
int NSP::CBase::two() { return 2; };
int NSP::CBase::three() { return 3; }

#ifdef SYMBOL_AT_FILE_SCOPE
static NSP::CBase Base;
#else
void fooBase() {
  NSP::CBase Base;
}
#endif
