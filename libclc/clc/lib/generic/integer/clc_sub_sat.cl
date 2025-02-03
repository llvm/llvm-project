#include <clc/internal/clc.h>

#define FUNCTION __clc_sub_sat
#define __CLC_FUNCTION(x) __builtin_elementwise_sub_sat
#define __CLC_BODY <clc/shared/binary_def.inc>

#include <clc/integer/gentype.inc>
