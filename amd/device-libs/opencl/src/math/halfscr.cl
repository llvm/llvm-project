/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

float2
__half_scr(float x)
{
    float y = x * 0x1.45f306p-3f;
    float s = __builtin_amdgcn_sinf(y);
    float result =  fabs(x) < 0x1.0p-20f ? x : s;

    return (float2)(result, __builtin_amdgcn_cosf(y) );
}

