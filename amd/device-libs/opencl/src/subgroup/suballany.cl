/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define ATTR __attribute__((overloadable, always_inline, const))

ATTR int
sub_group_all(int e)
{
    return OCKL_MANGLE_I32(wfall)(e);
}

ATTR int
sub_group_any(int e)
{
    return OCKL_MANGLE_I32(wfany)(e);
}

