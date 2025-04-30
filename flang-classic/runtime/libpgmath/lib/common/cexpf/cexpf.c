
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <math.h>
#include <complex.h>


#include "names.h"
#include "math_common.h"
#include "sleef_common.h"

#include "exp_common.h"
#include "cis_common.h"
#include "ldexp_common.h"

F_VISIBILITY_VEC
vfloat cexpf_vec(vfloat x)
{
    // Algorithm description for cexp(x)
    // We follow the mathematical definition of the cexp
    //     cexp(re + I*im) = exp(re)*(cos(im) + I*sin(im))
    // and also handle C99 special cases separately
    //
    // sin and cos will be computed in parallel and returned
    // in the same SIMD register interleaved and placed properly
    // for real and imaginary. We will use single precision for sin/cos.
    //
    // We will compute exp() as a pair of (poly, scale), where integer
    // scale will give us an extended range for exp = 2^(scale) * poly
    // exp result will be delivered in a pair of SIMD registers, real and
    // imaginary positions will be duplicates.
    //
    // We multiply poly by sin/cos - this incurs one roundoff in
    // every component and then we do ldexp to carefully multiply by
    // 2^(scale).

    vfloat rx = vmoveldup_vf_vf(x);                                             PRINT(rx);
    vfloat ix = vmovehdup_vf_vf(x);                                             PRINT(ix);
    // sign of resulting poly is 0, except maybe if NaN
    vfloat poly, scale; __vexp_kernel(rx, &poly, &scale);                       PRINT(poly); PRINT(scale);
    // cis(Inf & NaN) --> NaN
    vfloat rcis = __vcis_kernel(ix);                                            PRINT(rcis);
    // store sign of cis result
    vint2  signcis = vand_vi2_vi2_vi2(vF2I(rcis), vSETi(FL_SIGN_BIT));          PRINT(signcis);
    // sign of NaN may be lost here in favor of sign of NaN coming from poly
    vfloat polycis = vmul_vf_vf_vf(rcis, poly);                                 PRINT(polycis);
           // NaN sign fixup, perhaps not worth the effort
           polycis = vI2F(vandnot_vi2_vi2_vi2(vSETi(FL_SIGN_BIT), vF2I(polycis))); PRINT(polycis);
           polycis = vI2F(vor_vi2_vi2_vi2(vF2I(polycis), signcis));             PRINT(polycis);

    // if creal(x) == +Inf, then creal(result) is +Inf
    // if cimag(x) == 0.0 then fixup product to
    // the same zero, even if poly were NaN
    // NOTE: cimag(x) may be denormal under DAZ flag
    // subsequent computation in ldexp will flush
    // it to zero if done under the same DAZ condition
    vopmask reset  = veq_vo_vf_vf(x, vI2F(vSETLLi(0x0, FL_PINF)));              PRINT(reset);
           polycis = vsel_vf_vo_vf_vf(reset, x, polycis);                       PRINT(polycis);

    // if creal(x) == -Inf, then result is +0 * sign_of_cis()
    // NOTE: this fixup is only needed in case cimag(x)=Inf/NaN.
    // Finite cases would deliver proper zero thanks to ldexp.
    vopmask zeromask = veq_vo_vf_vf(rx, vI2F(vSETi(FL_NINF)));                  PRINT(zeromask);
           polycis = vsel_vf_vo_vf_vf(zeromask, vI2F(signcis), polycis);        PRINT(polycis);

    // careful polycis * 2^(scale)
    vfloat vcexp = __vldexpf_kernel(polycis, scale);                            PRINT(vcexp);
    return vcexp;
}

#if ((_VL) == (1))

F_VISIBILITY_SCALAR
float _Complex cexpf_scalar(float _Complex a)
{
#if defined _MEASURE_BASELINE_
return cexpf(a);
#endif
#if defined DO_PRINT
    feclearexcept(FE_ALL_EXCEPT);
#endif
    float ra = crealf(a);                                   PRINT(ra);
    float ia = cimagf(a);                                   PRINT(ia);

#if !(defined __USE_PORTABLE_CODE)
    // optimize for performance by calling 2-simd function
    // fill in unused simd slots with safe values

    unsigned long long int ua = F2I(ra) | (((unsigned long long int)F2I(ia)) << 32);
    // load zeroes out the upper slots, zero is a safe value for cexp
    vfloat va = _mm_castpd_ps(_mm_load_sd((double const*)(&ua)));               PRINT(va);
    vfloat vr = cexpf_vec(va);                                                  PRINT(vr);
    float _Complex res = *(float _Complex *)(&vr);                              PRINT(res);
    return res;
#else

    float poly;
    int scale;
    // This exp clamps input and doesn't over/underflow
    __exp_scalar_kernel(ra, &poly, &scale);                 PRINT( poly ); PRINT( scale );
    assert(((poly > 0x1.p63) && (poly < 0x1.p65)) || isnanf(poly));

    float _Complex cmplx_cis = __cis_scalar_kernel(ia);
    float rsin = cimagf(cmplx_cis);                         PRINT( rsin );
    float rcos = crealf(cmplx_cis);                         PRINT( rcos );
    // cis(Inf/NaN) results in NaN
    assert((isinff(ia) || isnanf(ia)) ^ !(isnanf(rsin) && isnanf(rcos)));

    // sign and payload of NaN from cis may be lost here
    // in favor of sign/payload of NaN coming from poly
    float polycos = poly * rcos;                            PRINT( polycos );
    float polysin = poly * rsin;                            PRINT( polysin );
    // restore sign of NaN coming from cis only to pass
    // symmetry test, perhaps not worth the effort
    polycos = copysignf(polycos, rcos);
    polysin = copysignf(polysin, rsin);

    if ( ia == 0.0f )
    {
        // if cimag(x) == 0.0 then fixup product to
        // the same zero, even if poly were NaN
        polysin = ia;
        // NOTE: ia may be denormal under DAZ flag
        // subsequent computation in ldexp will flush
        // it to zero if done under the same DAZ condition
    }
    if ( ra == I2F(FL_PINF) )
    {
        // if creal(x) == +Inf, then creal(result) is +Inf
        polycos = ra;
    }
    if ( ra == I2F(FL_NINF) )
    {
        // if creal(x) == -Inf, then result is +0 * sign_of_cis()
        // NOTE: this fixup is only needed in case cimag(x)=Inf/NaN.
        // Finite cases would deliver proper zero thanks to ldexp.
        polycos = copysignf(0.0f, rcos);
        polysin = copysignf(0.0f, rsin);
    }

    // This scaling shall not produce new NaNs
    float cexp_real = __ldexpf_scalar_kernel(polycos, scale);     PRINT( cexp_real );
    float cexp_imag = __ldexpf_scalar_kernel(polysin, scale);     PRINT( cexp_imag );

    return set_cmplx(cexp_real, cexp_imag);
#endif
}
#endif
