/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* inhibit floating point copy propagation */
#pragma global - Mx, 6, 0x100

#include <float.h>
#include "mthdecls.h"

#define xneg -106.5637380121098417363881585073946045229689l
#define sqrpi 5.6418958354775628694807945156077263153528602528974e-1l
#define xchg 12.0l
#define iteration 200

float128_t
__mth_i_qerfc_scaled(float128_t arg)
{
  if (arg >= xchg) {
    int i;
    float128_t presum;
    float128_t nowsum = 0.0;
    float128_t ps = 1.0;
    float128_t n1_2xx = 1.0 / (2.0 * arg * arg);

    for (i = 1; i < iteration; i++) {
      ps *= (1 - (2 * i)) * n1_2xx;
      presum = nowsum + ps;
      if (presum == nowsum)
        break;
      nowsum = presum;
    }
    return (1.0 + nowsum) / arg * sqrpi;
  } else {
    if (arg >= xneg) {
      return erfcl(arg) * expl(arg * arg);
    } else {
      return erfcl(xneg) * expl(xneg * xneg);
    }
  }
}
