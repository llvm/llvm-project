/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

// File should be compiled with -freciprocal-math and accuracy flags
// sufficient to select v_rcp_f16.
CONSTATTR half
MATH_MANGLE(native_rcp)(half x)
{
    return 1.0h / x;
}
