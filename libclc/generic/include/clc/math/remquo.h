#define __CLC_FUNCTION remquo

#define __CLC_BODY <clc/math/remquo.inc>
#define __CLC_ADDRESS_SPACE global
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <clc/math/remquo.inc>
#define __CLC_ADDRESS_SPACE local
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <clc/math/remquo.inc>
#define __CLC_ADDRESS_SPACE private
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#undef __CLC_FUNCTION
