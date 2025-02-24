#include <clc/clc.h>
#include <clc/math/clc_frexp.h>

#define FUNCTION frexp
#define __CLC_BODY <clc/math/unary_def_with_int_ptr.inc>
#include <clc/math/gentype.inc>
