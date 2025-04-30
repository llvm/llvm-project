
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include "names.h"
#include "math_common.h"
#include "sleef_common.h"
#include "ldexp_d_common.h"
#include "exp_d_common.h"
#include "cis_d_common.h"

F_VISIBILITY_VEC
vdouble cexp_vec(vdouble x)
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

    vdouble rx = vmoveldup_vd_vd(x);                                            PRINT(rx);
    vdouble ix = vmovehdup_vd_vd(x);                                            PRINT(ix);
    // sign of resulting poly is 0, except maybe if Na
    vdouble poly, scale; __vexp_d_kernel(rx, &poly, &scale);                    PRINT(poly); PRINT(scale);
    // cis(Inf & NaN) --> NaN
    vdouble rcis = __vcis_d_kernel(ix);                                         PRINT(rcis);
    // store sign of cis result
    vint2  signcis = vand_vi2_vi2_vi2(vD2L(rcis), vSETll(DB_SIGN_BIT));         PRINT(signcis);
    // sign of NaN may be lost here in favor of sign of NaN coming from poly
    vdouble polycis = vmul_vd_vd_vd(rcis, poly);                                PRINT(polycis);
           // NaN sign fixup, perhaps not worth the effort
           polycis = vL2D(vandnot_vi2_vi2_vi2(vSETll(DB_SIGN_BIT), vD2L(polycis))); PRINT(polycis);
           polycis = vL2D(vor_vi2_vi2_vi2(vD2L(polycis), signcis));             PRINT(polycis);
    // if creal(x) == +Inf, then creal(result) is +Inf
    // if cimag(x) == 0.0 then fixup product to
    // the same zero, even if poly were NaN
    // NOTE: cimag(x) may be denormal under DAZ flag
    // subsequent computation in ldexp will flush
    // it to zero if done under the same DAZ condition
    vopmask reset  = veq_vo_vd_vd(x, vL2D(vSETLLL(0x0, DB_PINF)));              PRINT(reset);
           polycis = vsel_vd_vo_vd_vd(reset, x, polycis);                       PRINT(polycis);

    // if creal(x) == -Inf, then result is +0 * sign_of_cis()
    // NOTE: this fixup is only needed in case cimag(x)=Inf/NaN.
    // Finite cases would deliver proper zero thanks to ldexp.
    vopmask zeromask = veq_vo_vd_vd(rx, vL2D(vSETll(DB_NINF)));                 PRINT(zeromask);
           polycis = vsel_vd_vo_vd_vd(zeromask, vL2D(signcis), polycis);        PRINT(polycis);

    // careful polycis * 2^(scale)
    vdouble vcexp = __vldexp_kernel(polycis, scale);                            PRINT(vcexp);
    return vcexp;
}

#if ((_VL) == (1))

F_VISIBILITY_SCALAR
double _Complex cexp_scalar_default_abi(double _Complex a)
{
#if defined DO_PRINT
    feclearexcept(FE_ALL_EXCEPT);
#endif
#if !(defined __USE_PORTABLE_CODE)
    vdouble va = _mm_loadu_pd((double const*)(&a));                             PRINT(va);
    vdouble vr = cexp_vec(va);                                                  PRINT(vr);
    double _Complex res = *(double _Complex *)(&vr);                            PRINT(res);
    return res;
#else

    double ra = creal(a);                                                       PRINT(ra);
    double ia = cimag(a);                                                       PRINT(ia);

    double poly;
    long long int scale;
    // This exp clamps input and doesn't over/underflow
    __exp_d_scalar_kernel(ra, &poly, &scale);                                   PRINT( poly ); PRINT( scale );
    assert(((poly > 0x1.p511) && (poly < 0x1.p513)) || isnan(poly));

    double _Complex cmplx_cis = __cis_d_scalar(ia);
    double rsin = cimag(cmplx_cis);                                             PRINT( rsin );
    double rcos = creal(cmplx_cis);                                             PRINT( rcos );
    // cis(Inf/NaN) results in NaN
    assert((isinf(ia) || isnan(ia)) ^ !(isnan(rsin) && isnan(rcos)));

    // sign and payload of NaN from cis may be lost here
    // in favor of sign/payload of NaN coming from poly
    double polycos = poly * rcos;                                               PRINT( polycos );
    double polysin = poly * rsin;                                               PRINT( polysin );
    // restore sign of NaN coming from cis only to pass
    // symmetry test, perhaps not worth the effort
    polycos = copysign(polycos, rcos);
    polysin = copysign(polysin, rsin);

    if ( ia == 0.0 )
    {
        // if cimag(x) == 0.0 then fixup product to
        // the same zero, even if poly were NaN
        polysin = ia;
        // NOTE: ia may be denormal under DAZ flag
        // subsequent computation in ldexp will flush
        // it to zero if done under the same DAZ condition
    }
    if ( ra == L2D(DB_PINF) )
    {
        // if creal(x) == +Inf, then creal(result) is +Inf
        polycos = ra;
    }

    if ( ra == L2D(DB_NINF) )
    {
        // if creal(x) == -Inf, then result is +0 * sign_of_cis()
        // NOTE: this fixup is only needed in case cimag(x)=Inf/NaN.
        // Finite cases would deliver proper zero thanks to ldexp.
        polycos = copysign(0.0, rcos);
        polysin = copysign(0.0, rsin);
    }

    // This scaling shall not produce new NaNs
    double cexp_real = __ldexp_d_scalar_kernel(polycos, scale);                 PRINT( cexp_real );
    double cexp_imag = __ldexp_d_scalar_kernel(polysin, scale);                 PRINT( cexp_imag );

    return set_cmplxd(cexp_real, cexp_imag);
#endif //if !(defined __USE_PORTABLE_CODE)
}
#if (defined _SCALAR_WITH_VECTOR_ABI_)
// scalar complex real/imag double precision values
// are passed in different registers by default
// here we define a function with single SIMD register
// calling convention
F_VISIBILITY_SCALAR_VECTOR
vdouble cexp_scalar_vector_abi(vdouble vx)
{
    double _Complex x = *(double _Complex *)&vx;
    vdouble vres;
    *(double _Complex *)&vres = cexp_scalar_default_abi(x);
    return vres;
}
#endif //if (defined _SCALAR_WITH_VECTOR_ABI_)

#endif //if ((_VL) == (1))
