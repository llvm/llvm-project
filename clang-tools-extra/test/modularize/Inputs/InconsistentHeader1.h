// Define symbol such that a declaration exists when this header
// is included, but not when InconsistentHeader2.h is included.
#define SYMBOL1 1
#include "InconsistentSubHeader.h"
