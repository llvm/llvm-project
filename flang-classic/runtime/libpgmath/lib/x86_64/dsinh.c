/* ============================================================
Copyright (c) 2002-2015 Advanced Micro Devices, Inc.

All rights reserved.

Redistribution and  use in source and binary  forms, with or
without  modification,  are   permitted  provided  that  the
following conditions are met:

+ Redistributions  of source  code  must  retain  the  above
  copyright  notice,   this  list  of   conditions  and  the
  following disclaimer.

+ Redistributions  in binary  form must reproduce  the above
  copyright  notice,   this  list  of   conditions  and  the
  following  disclaimer in  the  documentation and/or  other
  materials provided with the distribution.

+ Neither the  name of Advanced Micro Devices,  Inc. nor the
  names  of  its contributors  may  be  used  to endorse  or
  promote  products  derived   from  this  software  without
  specific prior written permission.

THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
CONTRIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
INC.  OR CONTRIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
(INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
LIABILITY,  WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
(INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
POSSIBILITY OF SUCH DAMAGE.

It is  licensee's responsibility  to comply with  any export
regulations applicable in licensee's jurisdiction.
============================================================ */

#include "libm_amd.h"
#include "libm_util_amd.h"

#define USE_SPLITEXP
#define USE_SCALEDOUBLE_1
#define USE_SCALEDOUBLE_2
#define USE_INFINITY_WITH_FLAGS
#define USE_VAL_WITH_FLAGS
#include "libm_inlines_amd.h"
#undef USE_SPLITEXP
#undef USE_SCALEDOUBLE_1
#undef USE_SCALEDOUBLE_2
#undef USE_INFINITY_WITH_FLAGS
#undef USE_VAL_WITH_FLAGS

#include "libm_errno_amd.h"

/* Deal with errno for out-of-range result */
static inline double
retval_errno_erange(double x __attribute__((unused)), int xneg)
{
    if (xneg)
      return -infinity_with_flags(AMD_F_OVERFLOW);
    else
      return infinity_with_flags(AMD_F_OVERFLOW);
}

double
FN_PROTOTYPE(mth_i_dsinh)(double x)
{
  /*
    After dealing with special cases the computation is split into
    regions as follows:

    abs(x) >= max_sinh_arg:
    sinh(x) = sign(x)*Inf

    abs(x) >= small_threshold:
    sinh(x) = sign(x)*exp(abs(x))/2 computed using the
    splitexp and scaleDouble functions as for exp_amd().

    abs(x) < small_threshold:
    compute p = exp(y) - 1 and then z = 0.5*(p+(p/(p+1.0)))
    sinh(x) is then sign(x)*z.                             */

  static const double max_sinh_arg =
                          7.10475860073943977113e+02, /* 0x408633ce8fb9f87e */
      thirtytwo_by_log2 = 4.61662413084468283841e+01, /* 0x40471547652b82fe */
      log2_by_32_lead = 2.16608493356034159660e-02,   /* 0x3f962e42fe000000 */
      log2_by_32_tail = 5.68948749532545630390e-11,   /* 0x3dcf473de6af278e */
      small_threshold = 8 * BASEDIGITS_DP64 * 0.30102999566398119521373889;
  /* (8*BASEDIGITS_DP64*log10of2) ' exp(-x) insignificant c.f. exp(x) */

  /* Lead and tail tabulated values of sinh(i) and cosh(i)
     for i = 0,...,36. The lead part has 26 leading bits. */

  static const double sinh_lead[37] = {
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      1.17520117759704589844e+00,  /* 0x3ff2cd9fc0000000 */
      3.62686038017272949219e+00,  /* 0x400d03cf60000000 */
      1.00178747177124023438e+01,  /* 0x40240926e0000000 */
      2.72899169921875000000e+01,  /* 0x403b4a3800000000 */
      7.42032089233398437500e+01,  /* 0x40528d0160000000 */
      2.01713153839111328125e+02,  /* 0x406936d228000000 */
      5.48316116333007812500e+02,  /* 0x4081228768000000 */
      1.49047882080078125000e+03,  /* 0x409749ea50000000 */
      4.05154187011718750000e+03,  /* 0x40afa71570000000 */
      1.10132326660156250000e+04,  /* 0x40c5829dc8000000 */
      2.99370708007812500000e+04,  /* 0x40dd3c4488000000 */
      8.13773945312500000000e+04,  /* 0x40f3de1650000000 */
      2.21206695312500000000e+05,  /* 0x410b00b590000000 */
      6.01302140625000000000e+05,  /* 0x412259ac48000000 */
      1.63450865625000000000e+06,  /* 0x4138f0cca8000000 */
      4.44305525000000000000e+06,  /* 0x4150f2ebd0000000 */
      1.20774762500000000000e+07,  /* 0x4167093488000000 */
      3.28299845000000000000e+07,  /* 0x417f4f2208000000 */
      8.92411500000000000000e+07,  /* 0x419546d8f8000000 */
      2.42582596000000000000e+08,  /* 0x41aceb0888000000 */
      6.59407856000000000000e+08,  /* 0x41c3a6e1f8000000 */
      1.79245641600000000000e+09,  /* 0x41dab5adb8000000 */
      4.87240166400000000000e+09,  /* 0x41f226af30000000 */
      1.32445608960000000000e+10,  /* 0x4208ab7fb0000000 */
      3.60024494080000000000e+10,  /* 0x4220c3d390000000 */
      9.78648043520000000000e+10,  /* 0x4236c93268000000 */
      2.66024116224000000000e+11,  /* 0x424ef822f0000000 */
      7.23128516608000000000e+11,  /* 0x42650bba30000000 */
      1.96566712320000000000e+12,  /* 0x427c9aae40000000 */
      5.34323724288000000000e+12,  /* 0x4293704708000000 */
      1.45244246507520000000e+13,  /* 0x42aa6b7658000000 */
      3.94814795284480000000e+13,  /* 0x42c1f43fc8000000 */
      1.07321789251584000000e+14,  /* 0x42d866f348000000 */
      2.91730863685632000000e+14,  /* 0x42f0953e28000000 */
      7.93006722514944000000e+14,  /* 0x430689e220000000 */
      2.15561576592179200000e+15}; /* 0x431ea215a0000000 */

  static const double sinh_tail[37] = {
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      1.60467555584448807892e-08,  /* 0x3e513ae6096a0092 */
      2.76742892754807136947e-08,  /* 0x3e5db70cfb79a640 */
      2.09697499555224576530e-07,  /* 0x3e8c2526b66dc067 */
      2.04940252448908240062e-07,  /* 0x3e8b81b18647f380 */
      1.65444891522700935932e-06,  /* 0x3ebbc1cdd1e1eb08 */
      3.53116789999998198721e-06,  /* 0x3ecd9f201534fb09 */
      6.94023870987375490695e-06,  /* 0x3edd1c064a4e9954 */
      4.98876893611587449271e-06,  /* 0x3ed4eca65d06ea74 */
      3.19656024605152215752e-05,  /* 0x3f00c259bcc0ecc5 */
      2.08687768377236501204e-04,  /* 0x3f2b5a6647cf9016 */
      4.84668088325403796299e-05,  /* 0x3f09691adefb0870 */
      1.17517985422733832468e-03,  /* 0x3f53410fc29cde38 */
      6.90830086959560562415e-04,  /* 0x3f46a31a50b6fb3c */
      1.45697262451506548420e-03,  /* 0x3f57defc71805c40 */
      2.99859023684906737806e-02,  /* 0x3f9eb49fd80e0bab */
      1.02538800507941396667e-02,  /* 0x3f84fffc7bcd5920 */
      1.26787628407699110022e-01,  /* 0x3fc03a93b6c63435 */
      6.86652479544033744752e-02,  /* 0x3fb1940bb255fd1c */
      4.81593627621056619148e-01,  /* 0x3fded26e14260b50 */
      1.70489513795397629181e+00,  /* 0x3ffb47401fc9f2a2 */
      1.12416073482258713767e+01,  /* 0x40267bb3f55634f1 */
      7.06579578070110514432e+00,  /* 0x401c435ff8194ddc */
      5.91244512999659974639e+01,  /* 0x404d8fee052ba63a */
      1.68921736147050694399e+02,  /* 0x40651d7edccde3f6 */
      2.60692936262073658327e+02,  /* 0x40704b1644557d1a */
      3.62419382134885609048e+02,  /* 0x4076a6b5ca0a9dc4 */
      4.07689930834187271103e+03,  /* 0x40afd9cc72249aba */
      1.55377375868385224749e+04,  /* 0x40ce58de693edab5 */
      2.53720210371943067003e+04,  /* 0x40d8c70158ac6363 */
      4.78822310734952334315e+04,  /* 0x40e7614764f43e20 */
      1.81871712615542812273e+05,  /* 0x4106337db36fc718 */
      5.62892347580489004031e+05,  /* 0x41212d98b1f611e2 */
      6.41374032312148716301e+05,  /* 0x412392bc108b37cc */
      7.57809544070145115256e+06,  /* 0x415ce87bdc3473dc */
      3.64177136406482197344e+06,  /* 0x414bc8d5ae99ad14 */
      7.63580561355670914054e+06}; /* 0x415d20d76744835c */

  static const double cosh_lead[37] = {
      1.00000000000000000000e+00,  /* 0x3ff0000000000000 */
      1.54308062791824340820e+00,  /* 0x3ff8b07550000000 */
      3.76219564676284790039e+00,  /* 0x400e18fa08000000 */
      1.00676617622375488281e+01,  /* 0x402422a490000000 */
      2.73082327842712402344e+01,  /* 0x403b4ee858000000 */
      7.42099475860595703125e+01,  /* 0x40528d6fc8000000 */
      2.01715633392333984375e+02,  /* 0x406936e678000000 */
      5.48317031860351562500e+02,  /* 0x4081228948000000 */
      1.49047915649414062500e+03,  /* 0x409749eaa8000000 */
      4.05154199218750000000e+03,  /* 0x40afa71580000000 */
      1.10132329101562500000e+04,  /* 0x40c5829dd0000000 */
      2.99370708007812500000e+04,  /* 0x40dd3c4488000000 */
      8.13773945312500000000e+04,  /* 0x40f3de1650000000 */
      2.21206695312500000000e+05,  /* 0x410b00b590000000 */
      6.01302140625000000000e+05,  /* 0x412259ac48000000 */
      1.63450865625000000000e+06,  /* 0x4138f0cca8000000 */
      4.44305525000000000000e+06,  /* 0x4150f2ebd0000000 */
      1.20774762500000000000e+07,  /* 0x4167093488000000 */
      3.28299845000000000000e+07,  /* 0x417f4f2208000000 */
      8.92411500000000000000e+07,  /* 0x419546d8f8000000 */
      2.42582596000000000000e+08,  /* 0x41aceb0888000000 */
      6.59407856000000000000e+08,  /* 0x41c3a6e1f8000000 */
      1.79245641600000000000e+09,  /* 0x41dab5adb8000000 */
      4.87240166400000000000e+09,  /* 0x41f226af30000000 */
      1.32445608960000000000e+10,  /* 0x4208ab7fb0000000 */
      3.60024494080000000000e+10,  /* 0x4220c3d390000000 */
      9.78648043520000000000e+10,  /* 0x4236c93268000000 */
      2.66024116224000000000e+11,  /* 0x424ef822f0000000 */
      7.23128516608000000000e+11,  /* 0x42650bba30000000 */
      1.96566712320000000000e+12,  /* 0x427c9aae40000000 */
      5.34323724288000000000e+12,  /* 0x4293704708000000 */
      1.45244246507520000000e+13,  /* 0x42aa6b7658000000 */
      3.94814795284480000000e+13,  /* 0x42c1f43fc8000000 */
      1.07321789251584000000e+14,  /* 0x42d866f348000000 */
      2.91730863685632000000e+14,  /* 0x42f0953e28000000 */
      7.93006722514944000000e+14,  /* 0x430689e220000000 */
      2.15561576592179200000e+15}; /* 0x431ea215a0000000 */

  static const double cosh_tail[37] = {
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      6.89700037027478056904e-09,  /* 0x3e3d9f5504c2bd28 */
      4.43207835591715833630e-08,  /* 0x3e67cb66f0a4c9fd */
      2.33540217013828929694e-07,  /* 0x3e8f58617928e588 */
      5.17452463948269748331e-08,  /* 0x3e6bc7d000c38d48 */
      9.38728274131605919153e-07,  /* 0x3eaf7f9d4e329998 */
      2.73012191010840495544e-06,  /* 0x3ec6e6e464885269 */
      3.29486051438996307950e-06,  /* 0x3ecba3a8b946c154 */
      4.75803746362771416375e-06,  /* 0x3ed3f4e76110d5a4 */
      3.33050940471947692369e-05,  /* 0x3f017622515a3e2b */
      9.94707313972136215365e-06,  /* 0x3ee4dc4b528af3d0 */
      6.51685096227860253398e-05,  /* 0x3f11156278615e10 */
      1.18132406658066663359e-03,  /* 0x3f535ad50ed821f5 */
      6.93090416366541877541e-04,  /* 0x3f46b61055f2935c */
      1.45780415323416845386e-03,  /* 0x3f57e2794a601240 */
      2.99862082708111758744e-02,  /* 0x3f9eb4b45f6aadd3 */
      1.02539925859688602072e-02,  /* 0x3f85000b967b3698 */
      1.26787669807076286421e-01,  /* 0x3fc03a940fadc092 */
      6.86652631843830962843e-02,  /* 0x3fb1940bf3bf874c */
      4.81593633223853068159e-01,  /* 0x3fded26e1a2a2110 */
      1.70489514001513020602e+00,  /* 0x3ffb4740205796d6 */
      1.12416073489841270572e+01,  /* 0x40267bb3f55cb85d */
      7.06579578098005001152e+00,  /* 0x401c435ff81e18ac */
      5.91244513000686140458e+01,  /* 0x404d8fee052bdea4 */
      1.68921736147088438429e+02,  /* 0x40651d7edccde926 */
      2.60692936262087528121e+02,  /* 0x40704b1644557e0e */
      3.62419382134890611269e+02,  /* 0x4076a6b5ca0a9e1c */
      4.07689930834187453002e+03,  /* 0x40afd9cc72249abe */
      1.55377375868385224749e+04,  /* 0x40ce58de693edab5 */
      2.53720210371943103382e+04,  /* 0x40d8c70158ac6364 */
      4.78822310734952334315e+04,  /* 0x40e7614764f43e20 */
      1.81871712615542812273e+05,  /* 0x4106337db36fc718 */
      5.62892347580489004031e+05,  /* 0x41212d98b1f611e2 */
      6.41374032312148716301e+05,  /* 0x412392bc108b37cc */
      7.57809544070145115256e+06,  /* 0x415ce87bdc3473dc */
      3.64177136406482197344e+06,  /* 0x414bc8d5ae99ad14 */
      7.63580561355670914054e+06}; /* 0x415d20d76744835c */

  __UINT8_T ux, aux, xneg;
  double y, z, z1, z2;
  int m;

  /* Special cases */

  GET_BITS_DP64(x, ux);
  aux = ux & ~SIGNBIT_DP64;
  if (aux < 0x3e30000000000000) /* |x| small enough that sinh(x) = x */
  {
    if (aux == 0)
      /* with no inexact */
      return x;
    else
      return val_with_flags(x, AMD_F_INEXACT);
  } else if (aux >= 0x7ff0000000000000) /* |x| is NaN or Inf */
  {
    return x + x;
  }

  xneg = (aux != ux);

  y = x;
  if (xneg)
    y = -x;

  if (y >= max_sinh_arg) {
    /* Return +/-infinity with overflow flag */
    return retval_errno_erange(x, xneg);
  } else if (y >= small_threshold) {
    /* In this range y is large enough so that
       the negative exponential is negligible,
       so sinh(y) is approximated by sign(x)*exp(y)/2. The
       code below is an inlined version of that from
       exp() with two changes (it operates on
       y instead of x, and the division by 2 is
       done by reducing m by 1). */

    splitexp(y, 1.0, thirtytwo_by_log2, log2_by_32_lead, log2_by_32_tail, &m,
             &z1, &z2);
    m -= 1;

    if (m >= EMIN_DP64 && m <= EMAX_DP64)
      z = scaleDouble_1((z1 + z2), m);
    else
      z = scaleDouble_2((z1 + z2), m);
  } else {
    /* In this range we find the integer part y0 of y
       and the increment dy = y - y0. We then compute

       z = sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy)

       where sinh(y0) and cosh(y0) are tabulated above. */

    int ind;
    double dy, dy2, sdy, cdy, sdy1, sdy2;

    ind = (int)y;
    dy = y - ind;

    dy2 = dy * dy;
    sdy = dy * dy2 * (0.166666666666666667013899e0 +
                      (0.833333333333329931873097e-2 +
                       (0.198412698413242405162014e-3 +
                        (0.275573191913636406057211e-5 +
                         (0.250521176994133472333666e-7 +
                          (0.160576793121939886190847e-9 +
                           0.7746188980094184251527126e-12 * dy2) *
                              dy2) *
                             dy2) *
                            dy2) *
                           dy2) *
                          dy2);

    cdy = dy2 * (0.500000000000000005911074e0 +
                 (0.416666666666660876512776e-1 +
                  (0.138888888889814854814536e-2 +
                   (0.248015872460622433115785e-4 +
                    (0.275573350756016588011357e-6 +
                     (0.208744349831471353536305e-8 +
                      0.1163921388172173692062032e-10 * dy2) *
                         dy2) *
                        dy2) *
                       dy2) *
                      dy2) *
                     dy2);

    /* At this point sinh(dy) is approximated by dy + sdy.
       Shift some significant bits from dy to sdy. */

    GET_BITS_DP64(dy, ux);
    ux &= 0xfffffffff8000000;
    PUT_BITS_DP64(ux, sdy1);
    sdy2 = sdy + (dy - sdy1);

    z = ((((((cosh_tail[ind] * sdy2 + sinh_tail[ind] * cdy) +
             cosh_tail[ind] * sdy1) +
            sinh_tail[ind]) +
           cosh_lead[ind] * sdy2) +
          sinh_lead[ind] * cdy) +
         cosh_lead[ind] * sdy1) +
        sinh_lead[ind];
  }

  if (xneg)
    z = -z;
  return z;
}
