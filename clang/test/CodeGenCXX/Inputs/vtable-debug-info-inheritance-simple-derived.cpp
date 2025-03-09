#include "vtable-debug-info-inheritance-simple-derived.h"

void CDerived::zero() {}
int CDerived::two() { return 22; };
int CDerived::three() { return 33; }

#ifdef SYMBOL_AT_FILE_SCOPE
static CDerived Derived;
#else
void fooDerived() {
  CDerived Derived;
}
#endif
