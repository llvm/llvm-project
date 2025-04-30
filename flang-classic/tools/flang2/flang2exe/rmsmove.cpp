/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Fix up and optimize general IL_SMOVEJ operations.
 *
 * Smove may be added by the expander, or by other transformations, such as the
 * accelerator compiler or IPA, when adding struct assignments.
 */

#include "rmsmove.h"
#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "ili.h"

static struct {
  int msz, msize;
  ILI_OP ld, st;
} info[4] = {
    {MSZ_I8, 8, IL_LDKR, IL_STKR},
    {MSZ_WORD, 4, IL_LD, IL_ST},
    {MSZ_UHWORD, 2, IL_LD, IL_ST},
    {MSZ_UBYTE, 1, IL_LD, IL_ST},
};
#define SMOVE_CHUNK 8 /* using movsq */
#define SMOVE_MIN 64
#define INFO1 0

static int
fixup_nme(int nmex, int msize, int offset, int iter)
{
  SPTR new_sym;
  int new_nme;
  DTYPE new_dtype;
  static char buf[100];
  int buf_len = 100;
  int sym;
  char *name = NULL;
  int name_len;
  bool is_malloced = false;

  if (nmex <= 0 || NME_TYPE(nmex) != NT_VAR || NME_SYM(nmex) <= 0)
    return nmex;

  switch (msize) {
  case 8:
    new_dtype = DT_UINT8;
    break;
  case 4:
    new_dtype = DT_INT;
    break;
  case 2:
    new_dtype = DT_USINT;
    break;
  case 1:
    new_dtype = DT_BINT;
    break;
  }

  sym = NME_SYM(nmex);
  name_len = strlen(SYMNAME(sym)) + 15;
  if (name_len <= buf_len) {
    name = &buf[0];
  } else {
    name = (char *)malloc(name_len);
    assert(name != NULL, "Fail to malloc a buffer", nmex, ERR_Fatal);
    is_malloced = true;
  }

  sprintf(name, "..__smove__%s__%d", SYMNAME(sym), iter);
  new_sym = getsymbol(name);
  DTYPEP(new_sym, new_dtype);
  STYPEP(new_sym, ST_MEMBER);
  CCSYMP(new_sym, 1);
  ADDRESSP(new_sym, offset);
  new_nme = addnme(NT_MEM, SPTR_NULL, nmex, 0);
  NME_SYM(new_nme) = new_sym;

  if (is_malloced)
    free(name);
  return new_nme;
} /* fixup_nme */

void exp_remove_gsmove(void);
void
rm_smove(void)
{
  int bihx, iltx, ilix, new_acon;
  /*
   * First implementation of GSMOVE will be under XBIT(2,0x800000). When this
   * is the only method, presumably, the code in exp_remove_gsmove() will be
   * moved to the ensuing loop.
   */
  if (USE_GSMOVE)
    exp_remove_gsmove();
  for (bihx = gbl.entbih; bihx; bihx = BIH_NEXT(bihx)) {
    bool have_smove = false;
    rdilts(bihx);
    for (iltx = BIH_ILTFIRST(bihx); iltx; iltx = ILT_NEXT(iltx)) {
      ilix = ILT_ILIP(iltx);
      if (ILI_OPC(ilix) == IL_SMOVEJ) {
        int srcx, src_nme, destx, dest_nme, len;
        int i, n, offset = 0, any = 0;
        /* target-dependent optimizations */
        srcx = ILI_OPND(ilix, 1);
        src_nme = ILI_OPND(ilix, 3);
        destx = ILI_OPND(ilix, 2);
        dest_nme = ILI_OPND(ilix, 4);
        len = ILI_OPND(ilix, 5);
        offset = 0;
        if (len > SMOVE_MIN) {
          /* turn the SMOVEI into SMOVE, change the len to IL_ACON */
          ILI_OPCP(ilix, IL_SMOVE);
          ILI_OPND(ilix, 1) = srcx;

/* For LLVM we use memcpy and do not need to chunk up the
 * copies.
 * XXX: Fortran has a special case in make_stmt (cgmain.c)
 * for the 'case STMT_SMOVE'.  Fortran will multiply the
 * length by 8 or 4 depending on the architecture.
 * C just takes the length as given, and we do not want to chunk the
 * length since we will be using memcpy and let llvm's memcpy handle
 * how it performs the copy.
 */
          n = len / SMOVE_CHUNK;
          offset = n * SMOVE_CHUNK;
          new_acon = ad_aconi(n);
          ILI_OPND(ilix, 3) = new_acon;
          len -= offset;
          ++any;
          have_smove = true;
        }
        if (XBIT(2, 0x4000)) {
          src_nme = NME_UNK;
          dest_nme = NME_UNK;
        }
        for (i = INFO1; i < 4; ++i) {
          int msz = info[i].msz;
          int msize = info[i].msize;
          while (len >= msize) {
            int ilioffset, ilix2;
            int ndest_nme = dest_nme, nsrc_nme = src_nme;
            /* add the load, store */
            if (any == 1) {
              srcx = ad1ili(IL_CSEAR, srcx);
              destx = ad1ili(IL_CSEAR, destx);
            }
            if (!XBIT(2, 0x4000)) {
              nsrc_nme = fixup_nme(src_nme, msize, offset, i);
              ndest_nme = fixup_nme(dest_nme, msize, offset, i);
            }
            ilioffset = ad_aconi(offset);
            ilix = ad3ili(IL_AADD, srcx, ilioffset, 0);
            ilix = ad3ili(info[i].ld, ilix, nsrc_nme, msz);
            ilix2 = ad3ili(IL_AADD, destx, ilioffset, 0);
            ilix = ad4ili(info[i].st, ilix, ilix2, ndest_nme, msz);
            if (!any) {
              /* reuse this ILT */
              ILT_ILIP(iltx) = ilix;
              /* flag this as a store operation */
              ILT_ST(iltx) = 1;
            } else {
              iltx = addilt(iltx, ilix);
              /* ILT_ST gets set by addilt here */
            }
            ++any;
            offset += msize;
            len -= msize;
          }
        }
      }
    }
    wrilts(bihx);
    if (have_smove)
      BIH_SMOVE(bihx) = 1;
  }
} /* rm_smove */
