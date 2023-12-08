#undef erfc

#define __CLC_BODY <clc/math/unary_decl.inc>
#define __CLC_FUNCTION erf

#include <clc/math/gentype.inc>

#undef __CLC_BODY
#undef __CLC_FUNCTION
