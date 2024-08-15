/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

ulong
OCKL_MANGLE_U64(cyclectr)(void)
{
    return __builtin_readcyclecounter();
}

ulong
OCKL_MANGLE_U64(steadyctr)(void)
{
  return __builtin_readsteadycounter();
}

