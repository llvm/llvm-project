/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#define ATTR __attribute__((const))

//-------- T __nv_llrint
ATTR long __nv_llrint(double x) { return (long)__ocml_rint_f64(x); }

//-------- T __nv_llrintf
ATTR long __nv_llrintf(float x) { return (long)__ocml_rint_f32(x); }

//-------- T __nv_llround
ATTR long __nv_llround(double x) { return (long)__ocml_round_f64(x); }

//-------- T __nv_llroundf
ATTR long __nv_llroundf(float x) { return (long)__ocml_round_f32(x); }

