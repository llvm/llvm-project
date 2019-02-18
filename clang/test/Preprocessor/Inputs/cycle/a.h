// Presence of 2 inclusion cycles
//    b.h -> a.h -> b.h -> ...
//    c.h -> a.h -> c.h -> ...
// makes it unfeasible to reach max inclusion depth in all possible ways. Need
// to stop earlier.

#include "b.h"
#include "c.h"
