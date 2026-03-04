
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

__attribute__((overloadable, const)) uint
amd_sad4(uint4 x, uint4 y, uint z)
{
    uint a = OCKL_MANGLE_U32(sad)(x.s0,y.s0,z);
    a =      OCKL_MANGLE_U32(sad)(x.s1,y.s1,a);
    a =      OCKL_MANGLE_U32(sad)(x.s2,y.s2,a);
    return   OCKL_MANGLE_U32(sad)(x.s3,y.s3,a);
}

