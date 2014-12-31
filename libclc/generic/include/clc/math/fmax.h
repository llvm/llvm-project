#undef fmax
#define fmax __clc_fmax

#define __CLC_BODY <clc/math/binary_decl.inc>
#define __CLC_FUNCTION __clc_fmax

#include <clc/math/gentype.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION

