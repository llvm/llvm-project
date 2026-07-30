#include "a.h"

#if !__building_module(b)
#error "should only get here when building module b"
#endif

const int b = 2;
