/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define ATTR __attribute__((overloadable, const))

ATTR uint amd_pack(float4 v) { return OCKL_MANGLE_U32(pack)(v); }

