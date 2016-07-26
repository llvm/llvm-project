
#include "llvm.h"
#include "ockl.h"

#define CATTR __attribute__((always_inline, const))

CATTR uint
OCKL_MANGLE_U32(lerp)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_lerp(a, b, c);
}

