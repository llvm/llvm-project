/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "exp_fvec.h"
#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "regutil.h"
#include "machreg.h"
#include "ilm.h"
#include "ilmtp.h"
#include "ili.h"
#include "expand.h"
#include "machar.h"

void
init_fvec(void)
{
}

void
fin_fvec(void)
{
}

void eval_fvec(int ilmx)
{
  int opc;

  opc = ILM_OPC((ILM *)(ilmb.ilm_base + ilmx));
  if (IM_VEC(opc)) {
    interr("eval_fvec: vector not impl", ilmx, ERR_Severe);
  } else {
    eval_ilm(ilmx);
  }
}
