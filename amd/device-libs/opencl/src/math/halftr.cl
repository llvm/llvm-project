/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"

__attribute__((always_inline, const)) float
__half_tr(float x, int regn)
{
    float r = x * x;

    float a = mad(r, -0.0172032480471481694693109f, 0.385296071263995406715129f);

    float b = mad(r,
                  mad(r, 0.01844239256901656082986661f, -0.51396505478854532132342f),
                  1.15588821434688393452299f);

    float t = mad(x*r, a * __llvm_amdgcn_rcp_f32(b), x);
    float tr = -__llvm_amdgcn_rcp_f32(t);

    return regn & 1 ? tr : t;
}

