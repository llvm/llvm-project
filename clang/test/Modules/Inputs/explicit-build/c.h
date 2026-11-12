#include "b.h"

#if !__building_module(c)
#error "should only get here when building module c"
#endif

const int c = 3;
