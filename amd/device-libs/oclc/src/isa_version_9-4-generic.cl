/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

// gfx9-4-generic matches gfx942 from the device-lib perspective.
// NOTE: gfx942 has fp8 instructions, fp8 conversion instructions, and support
// for xf32 format, while the gfx9-4-generic doesn't.
const __constant int __oclc_ISA_version = 9402;
