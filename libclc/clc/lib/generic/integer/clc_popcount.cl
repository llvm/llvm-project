#include <clc/internal/clc.h>

#define FUNCTION __clc_popcount
#define __CLC_FUNCTION(x) __builtin_elementwise_popcount
#define __CLC_BODY <clc/shared/unary_def.inc>

#include <clc/integer/gentype.inc>
