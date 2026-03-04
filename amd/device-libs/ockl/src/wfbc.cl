
#include "ockl.h"


uint
OCKL_MANGLE_U32(wfbcast)(uint a, uint i)
{
    uint j = __builtin_amdgcn_readfirstlane(i);
    return __builtin_amdgcn_readlane(a, j);
}

ulong
OCKL_MANGLE_U64(wfbcast)(ulong a, uint i)
{
    uint j = __builtin_amdgcn_readfirstlane(i);
    uint2 aa = __builtin_astype(a, uint2);
    aa.x = __builtin_amdgcn_readlane(aa.x, j);
    aa.y = __builtin_amdgcn_readlane(aa.y, j);
    return __builtin_astype(aa, ulong);
}

