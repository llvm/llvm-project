/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

__attribute__((always_inline)) uint
OCKL_MANGLE_U32(activelane)(void)
{
    // TODO - check that this compiles to the desired 2 ISA instructions
    return __llvm_amdgcn_mbcnt_hi(__llvm_amdgcn_read_exec_hi(),
            __llvm_amdgcn_mbcnt_lo(__llvm_amdgcn_read_exec_lo(), 0u));
}

