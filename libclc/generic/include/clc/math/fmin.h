#undef fmin
#define fmin __clc_fmin

#define __CLC_BODY <clc/math/binary_decl.inc>
#define __CLC_FUNCTION __clc_fmin

#include <clc/math/gentype.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION

