/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

uint
OCKL_MANGLE_U32(activelane)(void)
{
    if (__oclc_wavefrontsize64) {
        return __builtin_amdgcn_mbcnt_hi(__builtin_amdgcn_read_exec_hi(),
               __builtin_amdgcn_mbcnt_lo(__builtin_amdgcn_read_exec_lo(), 0u));
    } else {
        return __builtin_amdgcn_mbcnt_lo(__builtin_amdgcn_read_exec_lo(), 0u);
    }
}

