/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(tanpi)(half x)
{
    return (half)MATH_UPMANGLE(tanpi)((float)x);
}

