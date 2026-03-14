int m = n;

#include "other.h"
#include "c.h"

#if defined(A) && !defined(ALLOW_NAME_LEAKAGE)
#warning A is defined
#endif

#define B

template<typename T> void b_template() {
  N::C::f(0);
}
