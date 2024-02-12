/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

// gfx11-generic is identical to gfx1103 from the device-lib perspective.
// NOTE: gfx1103 does not have the HW workarounds that gfx11-generic has.
const __constant int __oclc_ISA_version = 11003;
