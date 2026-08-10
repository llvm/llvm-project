#include "instrprof-icall-promo.h"

A a;

A* ap = &a;

int ref(A* ap) { return ap->A::foo(); }
