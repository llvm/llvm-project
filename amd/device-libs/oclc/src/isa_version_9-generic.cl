/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

// gfx9-generic matches gfx900 from the device-lib perspective.
// NOTE: gfx900 has mad-mix while gfx9-generic does NOT.
const __constant int __oclc_ISA_version = 9000;
