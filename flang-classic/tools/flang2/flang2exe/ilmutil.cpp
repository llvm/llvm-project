/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* \file
 * ilmutil.c  -  SCC/SCFTN ILM utilities used by Semantic Analyzer. */

/* Contents:
 *
 *     addlabel(sptr)          - add label ilm
 *     ad1ilm(opc)             - append 1 ILM word to ILM buffer.
 *     ad2ilm(opc,op1)         - append 2 ILM words.
 *     ad3ilm(opc,op1,op2)     - append 3 ILM words.
 *     ad4ilm(opc,op1,op2,op3) - append 4 ILM words.
 *     ad5ilm(opc,op1,op2,op3,
 *            op4)             - append 5 ILM words.
 *     wrilms(linenum)         - write block of ILM's to ILM file.
 *     save_ilms(area)         - copy ilms into working storage area and
 *				 return pointer to copy.
 *     add_ilms(p)             - copy ILM's into ILM buffer.
 *     mkbranch(ilmptr,truelb) - convert logical expr into branches.
 *     dmpilms()               - dump block of ILM's to debug listing file.
 *     int rdilms()            - read in an ILM block
 */

#include "ilmutil.h"
#include "error.h"
#include "ilmtp.h"
#include "ilm.h"
#include "fih.h"
#include "semant.h"
#include "pragma.h"
#include "outliner.h"
#include "symfun.h"
#include "mp.h"

ILMB ilmb;

GILMB gilmb = {0, 0, 0, 0, 0, 0, 0, 0};
GILMB next_gilmb = {0, 0, 0, 0, 0, 0, 0, 0};

/* reserve a few words before each ILM block in gilmb
 * to store global information */
#define GILMSAVE 2
/* are we in global mode? */
int ilmpos = 0;
extern ILM_T *ilm_base; /* base ptr for ILMs read in (from inliner.c) */
static int gilmb_mode = 0;

#define TY(n) ((int)(n & 0x03))

/*******************************************************************/

void
addlabel(int sptr)
/*  add label ILM */ {
  (void)ad2ilm(IM_LABEL, sptr);
}

/*******************************************************************/

#if DEBUG
#define ILMNAME(opc) ((opc) > 0 && (opc) < N_ILM ? ilms[opc].name : "???")
#endif

/*
 * Add 1 ILM word to current ILM buffer.
 */
int
ad1ilm(int opc)
{
#if DEBUG
  if (DBGBIT(4, 0x4))
    fprintf(gbl.dbgfil, "%5d %s\n", ilmb.ilmavl, ILMNAME(opc));
#endif
  NEED(ilmb.ilmavl + 1, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + 1000);
  ilmb.ilm_base[ilmb.ilmavl] = opc;
  return ilmb.ilmavl++;
}

/******************************************************************/

/*
 * Add 2 ILM words to current ILM buffer.
 */
int
ad2ilm(int opc, int opr1)
{
  ILM_T *p;

#if DEBUG
  if (DBGBIT(4, 0x4))
    fprintf(gbl.dbgfil, "%5d %s %d\n", ilmb.ilmavl, ILMNAME(opc), opr1);
#endif
  NEED(ilmb.ilmavl + 2, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + 1000);
  p = ilmb.ilm_base + ilmb.ilmavl;
  ilmb.ilmavl += 2;
  *p++ = opc;
  *p = opr1;
  return ilmb.ilmavl - 2;
}

/******************************************************************/

/*
 * Add 3 ILM words to current ILM buffer.
 */
int
ad3ilm(int opc, int opr1, int opr2)
{
  ILM_T *p;

#if DEBUG
  if (DBGBIT(4, 0x4))
    fprintf(gbl.dbgfil, "%5d %s %d,%d\n", ilmb.ilmavl, ILMNAME(opc), opr1,
            opr2);
#endif
  NEED(ilmb.ilmavl + 3, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + 1000);
  p = ilmb.ilm_base + ilmb.ilmavl;
  ilmb.ilmavl += 3;
  *p++ = opc;
  *p++ = opr1;
  *p = opr2;
  return ilmb.ilmavl - 3;
}

/******************************************************************/

/*
 * add 4 ILM words to current ILM buffer.
 */
int
ad4ilm(int opc, int opr1, int opr2, int opr3)
{
  ILM_T *p;

#if DEBUG
  if (DBGBIT(4, 0x4))
    fprintf(gbl.dbgfil, "%5d %s %d,%d,%d\n", ilmb.ilmavl, ILMNAME(opc), opr1,
            opr2, opr3);
#endif
  NEED(ilmb.ilmavl + 4, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + 1000);
  p = ilmb.ilm_base + ilmb.ilmavl;
  ilmb.ilmavl += 4;
  *p++ = opc;
  *p++ = opr1;
  *p++ = opr2;
  *p = opr3;
  return ilmb.ilmavl - 4;
}

/******************************************************************/

/*
 * add 5 ILM words to current ILM buffer.
 */
int
ad5ilm(int opc, int opr1, int opr2, int opr3, int opr4)
{
  ILM_T *p;

#if DEBUG
  if (DBGBIT(4, 0x4))
    fprintf(gbl.dbgfil, "%5d %s %d,%d,%d,%d\n", ilmb.ilmavl, ILMNAME(opc), opr1,
            opr2, opr3, opr4);
#endif
  NEED(ilmb.ilmavl + 5, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + 1000);
  p = ilmb.ilm_base + ilmb.ilmavl;
  ilmb.ilmavl += 5;
  *p++ = opc;
  *p++ = opr1;
  *p++ = opr2;
  *p++ = opr3;
  *p = opr4;
  return ilmb.ilmavl - 5;
}

/******************************************************************/

/**
 * Add 'n' ILM words to current ILM buffer, including the opc.
 */
int
adNilm(int n, int opc, ...)
{
  ILM_T *p;
  int i, opr;
  va_list vargs;

  assert(n > 5, "adNilm should only be used for ILMs with >5 arguments", opc,
         ERR_Fatal);

#if DEBUG
  if (DBGBIT(4, 0x4))
    fprintf(gbl.dbgfil, "%5d %s %d operands\n", ilmb.ilmavl, ILMNAME(opc), n);
#endif
  NEED(ilmb.ilmavl + n, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + 1000);
  p = ilmb.ilm_base + ilmb.ilmavl;
  ilmb.ilmavl += n;

  va_start(vargs, opc);

  *p++ = opc;
  for (i = 1; i < n; ++i, ++p) {
    opr = va_arg(vargs, int);
    *p = opr;
  }
  --p;
  va_end(vargs);
  return ilmb.ilmavl - n;
}

int
ilm_callee_index(ILM_OP opc)
{
  assert(IM_TYPE(opc) == IMTY_PROC, "ilm_callee_index: opc must have proc type",
         opc, ERR_Fatal);
  switch (opc) {
  case IM_FAPPLY:
  case IM_VAPPLY:
    return 3;
  case IM_FINVOKE:
  case IM_VINVOKE:
    return 4;
  default:
    return 2;
  }
}

SYMTYPE
ilm_symtype_of_return_slot(DTYPE ret_type)
{
  if (DTY(ret_type) == TY_STRUCT || DTY(ret_type) == TY_UNION) {
    /* function returns struct or union */
    return DTY(ret_type) == TY_STRUCT ? ST_STRUCT : ST_UNION;
  }
  if (DT_ISCMPLX(ret_type)) {
    /* function returns complex */
    return ST_VAR;
  }
  return ST_UNKNOWN;
}

int
ilm_return_slot_index(ILM_T *ilmp)
{
  DEBUG_ASSERT(0 < ilmp[0] && ilmp[0] < N_ILM,
               "ilm_return_slot_index: bad ILM");

  /* Each case either returns the return slot index, or sets callee_index
     so that logic after the switch can find the return type and return slot. */
  switch (ilmp[0]) {
  case IM_VAPPLY:
  case IM_FAPPLY:
  case IM_VINVOKE:
  case IM_FINVOKE:
    break;
  case IM_SFUNC:
  case IM_CFUNC:
  case IM_CDFUNC:
#ifdef LONG_DOUBLE_FLOAT128
  case IM_CFLOAT128FUNC:
#endif
    return 3;
  default:
    return 0;
  }
  interr("ilm_return_slot_index: ILM not implemented yet", ilmp[0], ERR_Severe);
  return 0;
}

/******************************************************************/
static const char *nullname = "";

/*
 * allocate the ILMs, free the ILMs
 */
void
init_ilm(int ilmsize)
{
  ilmb.ilmavl = BOS_SIZE;
  ilmb.ilm_size = ilmsize;
  NEW(ilmb.ilm_base, ILM_T, ilmb.ilm_size);
  fihb.stg_size = 10;
  NEW(fihb.stg_base, FIH, fihb.stg_size);
  fihb.stg_avail = 1;
  BZERO(fihb.stg_base + 0, FIH, 2);
  FIH_DIRNAME(0) = NULL;
  FIH_FILENAME(0) = nullname;
  FIH_FULLNAME(0) = nullname;
  FIH_DIRNAME(1) = NULL;
  FIH_FILENAME(1) = nullname;
  FIH_FULLNAME(1) = nullname;
} /* init_ilm */

void
init_global_ilm_mode()
{
  gilmb.ilmavl = 0;
  gilmb.ilmpos = 0;
  gilmb.globalilmtotal = 0;
  gilmb.globalilmfirst = 0;
  gilmb.ilm_size = ilmb.ilm_size * 5;
  if (gilmb.ilm_size == 0)
    gilmb.ilm_size = 1000;
  NEW(gilmb.ilm_base, ILM_T, gilmb.ilm_size);
} /* init_global_ilm_mode */

void
reset_global_ilm_position()
{
  gilmb.ilmpos = GILMSAVE;
  ilmb.globalilmstart = gilmb.globalilmstart;
  ilmb.globalilmcount = gilmb.globalilmcount;
} /* reset_global_ilm_position */

void
init_global_ilm_position()
{
  gilmb.globalilmstart = ilmb.globalilmstart;
  gilmb.globalilmcount = ilmb.globalilmcount;
} /* init_global_ilm_position */

/*
 * while inlining, we read from gilm, write to next_gilmb,
 * one block at a time
 */
void
init_next_gilm()
{
  next_gilmb.ilmavl = GILMSAVE;
  next_gilmb.ilmpos = GILMSAVE;
  next_gilmb.ilm_size = gilmb.ilm_size;
  NEW(next_gilmb.ilm_base, ILM_T, next_gilmb.ilm_size);
  gilmb.globalilmcount = ilmb.globalilmcount;
  gilmb.globalilmstart = ilmb.globalilmstart;
  next_gilmb.globalilmcount = ilmb.globalilmcount;
  next_gilmb.globalilmstart = ilmb.globalilmstart;
} /* init_next_gilm */

/*
 * after inlining one level, swap the next_gilmb space with the gilmb space
 * prepare for the next level of inlining
 */
void
swap_next_gilm()
{
  GILMB temp;
  next_gilmb.globalilmfirst = gilmb.globalilmfirst; /* preserve */
  next_gilmb.globalilmtotal = gilmb.globalilmtotal; /* preserve */
  memcpy(&temp, &gilmb, sizeof(gilmb));
  memcpy(&gilmb, &next_gilmb, sizeof(gilmb));
  memcpy(&next_gilmb, &temp, sizeof(gilmb));
  next_gilmb.ilmavl = GILMSAVE;
  next_gilmb.ilmpos = GILMSAVE;
  ilmb.globalilmcount = gilmb.globalilmcount;
  ilmb.globalilmstart = gilmb.globalilmstart;
} /* swap_next_gilm */

/*
 * write the current ILM block to next_gilmb
 */
void
gwrilms(int nilms)
{
  ilmb.ilm_base[3] = nilms;
  NEED(next_gilmb.ilmavl + nilms + GILMSAVE * 2, next_gilmb.ilm_base, ILM_T,
       next_gilmb.ilm_size, next_gilmb.ilmavl + nilms + 1000);
  BCOPY(next_gilmb.ilm_base + next_gilmb.ilmavl, ilmb.ilm_base, ILM_T, nilms);
  next_gilmb.ilm_base[next_gilmb.ilmavl - 1] =
      ilmb.globalilmcount - ilmb.globalilmstart;
  next_gilmb.ilmavl += nilms + GILMSAVE;
  /* reinitialize with empty ILM block */
  ilmb.ilmavl = BOS_SIZE;
  ilmb.ilm_base[0] = IM_BOS;
  ilmb.ilm_base[1] = 0;
  ilmb.ilm_base[2] = 1;
  ilmb.ilm_base[3] = BOS_SIZE;
} /* gwrilms */

/*
 * done with inlining
 */
void
fini_next_gilm()
{
  FREE(next_gilmb.ilm_base);
  next_gilmb.ilm_base = NULL;
  next_gilmb.ilm_size = 0;
  next_gilmb.ilmavl = 0;
  next_gilmb.ilmpos = 0;
} /* fini_next_gilm */

/*
 * free ILMs when we're done
 */
void
fini_ilm()
{
  FREE(ilmb.ilm_base);
  ilmb.ilm_base = NULL;
  if (gilmb_mode && gilmb.ilm_base) {
    FREE(gilmb.ilm_base);
    gilmb.ilm_base = NULL;
    gilmb.ilm_size = 0;
    gilmb.ilmavl = 0;
    gilmb.ilmpos = 0;
    gilmb.globalilmtotal = 0;
    gilmb.globalilmfirst = 0;
  }
} /* fini_ilm */
  /******************************************************************/

/*
 * write one block of ILM's to ILM file.
 */
void
wrilms(int linenum)
{
  void dmpilms();
  ILM_T *p;
  int nw;

  /* if nocode, then just return */
  if (ilmb.ilmavl == BOS_SIZE)
    return;

  p = ilmb.ilm_base;
  *p++ = IM_BOS;
  if (linenum == -1 || linenum == 0)
    *p++ = gbl.lineno;
  else
    *p++ = linenum;

  *p++ = gbl.findex;

  *p = ilmb.ilmavl;

  if (sem.wrilms) {
    nw = fwrite((char *)ilmb.ilm_base, sizeof(ILM_T), ilmb.ilmavl, gbl.ilmfil);
    if (nw != ilmb.ilmavl)
      error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0, "(IL file)", CNULL);
  }

  if (DBGBIT(4, 1))
    dmpilms();

  ilmb.ilmavl = BOS_SIZE;
  gilmb_mode = 0; /* next rdilms comes from file */
}

/*****************************************************************/

/*
 * if the ILM area is not empty, allocate a block in the indicated working
 * storage area and copy the current ILM's into it.  The first word of the
 * block will contain the number of ILM words.  The ILM area is reset to the
 * empty state.
 */
ILM_T *
save_ilms(int area)
{
  ILM_T *p;
  int count;
  ILM_T *q;

  if (ilmb.ilmavl == BOS_SIZE)
    return NULL;
  count = ilmb.ilmavl - BOS_SIZE;
  q = p = (ILM_T *)getitem(area, sizeof(ILM_T) * (count + 1));
  *p++ = count;
  BCOPY(p, ilmb.ilm_base + BOS_SIZE, ILM_T, count);
#if DEBUG
  if (DBGBIT(4, 8)) {
    ILMA(0) = IM_BOS;
    ILMA(1) = gbl.lineno;
    ILMA(2) = gbl.findex;
    ILMA(BOS_SIZE - 1) = ilmb.ilmavl;
    dmpilms();
  }
#endif
  ilmb.ilmavl = BOS_SIZE;
  return q;
}

/*
 * Similar to above, except save into an already-allocated area.
 */
ILM_T *
save_ilms0(void *area)
{
  ILM_T *p;
  int count;
  ILM_T *q;

  if (ilmb.ilmavl == BOS_SIZE)
    return NULL;
  count = ilmb.ilmavl - BOS_SIZE;
  q = p = (ILM_T*)area;
  *p++ = count;
  BCOPY(p, ilmb.ilm_base + BOS_SIZE, ILM_T, count);
#if DEBUG
  if (DBGBIT(4, 8)) {
    ILMA(0) = IM_BOS;
    ILMA(1) = gbl.lineno;
    ILMA(2) = gbl.findex;
    ILMA(BOS_SIZE - 1) = ilmb.ilmavl;
    dmpilms();
  }
#endif
  ilmb.ilmavl = BOS_SIZE;
  return q;
}

/************************************************************************/

/*
 * Copy block of ILM's, previously saved by save_ilms, directly into the ILM
 * buffer:
 */
void
add_ilms(ILM_T *p)
{
  int count;
  int need;

  if (p == NULL)
    return;
  count = *p++;
  /* lfm bug fix 12/16/91 */
  assert(count > 0, "add_ilms: non-positive count", 0, ERR_Fatal);
  need = count;
  if (need < 1000)
    need = 1000;
  NEED(ilmb.ilmavl + count, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + need);
  BCOPY(ilmb.ilm_base + ilmb.ilmavl, p, ILM_T, count);
  ilmb.ilmavl += count;
}

/************************************************************************/

/*
 * Copy block of ILM's, previously saved by save_ilms, directly into the ILM
 * buffer and relocate links.
 */
void
reloc_ilms(ILM_T *p)
{
  int count;
  int need;
  int ilmptr;
  int rlc;

  if (p == NULL)
    return;
  ilmptr = ilmb.ilmavl;
  rlc = ilmb.ilmavl - BOS_SIZE;
  count = *p++;
  assert(count > 0, "reloc_ilms: non-positive count", 0, ERR_Fatal);
  need = count;
  if (need < 1000)
    need = 1000;
  NEED(ilmb.ilmavl + count, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
       ilmb.ilmavl + need);
  BCOPY(ilmb.ilm_base + ilmb.ilmavl, p, ILM_T, count);
  ilmb.ilmavl += count;

  /* scan current ILM buffer, and relocate links */
  do { /*  loop once for each opcode  */
    int opc, len, noprs, varpart, opnd;

    opc = ILMA(ilmptr);
    len = noprs = ilms[opc].oprs; /* number of "fixed" operands */
    if (IM_VAR(opc)) {
      varpart = ILMA(ilmptr + 1);
      len += varpart;
      /*
       * for an ILM with a variable number operands, we only want to
       * examine the operands only if they are links.  In any case,
       * we want to begin at the second operand.
       */
      if (IM_OPRFLAG(opc, noprs + 1) != OPR_LNK)
        varpart = 0;
      noprs--;
      opnd = 2;
    } else {
      /*
       * the ILM does not have any variable ILM links -- the
       * analysis begins with the first operand.
       */
      varpart = 0;
      opnd = 1;
    }
    for (;; opnd++) {
      if (noprs == 0) {
        if ((varpart--) == 0)
          break;
      } else {
        noprs--;
        if (IM_OPRFLAG(opc, opnd) != OPR_LNK)
          continue;
      }
#if DEBUG
      assert(ILMA(ilmptr + opnd) >= BOS_SIZE && ILMA(ilmptr + opnd) < ilmptr,
             "reloc_ilms: bad lnk", ilmptr, ERR_Severe);
#endif
      ILMA(ilmptr + opnd) += rlc;
    }
    ilmptr += (len + 1);
  } while (ilmptr < ilmb.ilmavl);
}

/************************************************************************/

/*
 * Convert ILM 'tree' pointed to by ilmptr into a series of one or more
 * conditional branches which have the net effect of branching to truelb
 * if the condition has the truth value of flag:
 */
void
mkbranch(int ilmptr, int truelb, int flag)
{
  int opc, falselb;

  opc = ILMA(ilmptr);
  if (opc == IM_LAND) {
    if (!flag) {
      /* if (!ilm1 || !ilm2) goto truelb */
      mkbranch(ILMA(ilmptr + 1), truelb, false);
      mkbranch(ILMA(ilmptr + 2), truelb, false);
      /* erase the AND ilm: */
      if (ilmptr + 3 == ilmb.ilmavl)
        ilmb.ilmavl = ilmptr;
      else {
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr) = IM_NOP;
      }
    } else {
      /*		if (ilm1 && ilm2) goto truelb -->
       *
       *		if (!ilm1) goto falselb
       *		if (ilm2) goto truelb
       * falselb:
       */
      falselb = getlab();
      mkbranch(ILMA(ilmptr + 1), falselb, false);
      mkbranch(ILMA(ilmptr + 2), truelb, true);
      ILMA(ilmptr++) = IM_LABEL;
      ILMA(ilmptr++) = falselb;
      ILMA(ilmptr) = IM_NOP;
    }
  } else if (opc == IM_LAND8) {
    /*
     * tpr3035, the relational expressions are always logical*4.  In the
     * presence of a logical*8 expression, the relational is converted
     * with a IM_ITOI8.  Unfortunately, this means that the ILM order of
     * the operands to LAND8 to LOR8 may change, and the assumptions to
     * effect short-circuiting are no longer valid; e.g.,
     *  (1) a
     *  (2) b
     *  (3) LOR  (1) (2)  [operand 1 precedes operand 2]
     * becomes
     *  (1) a
     *  (2) b
     *  (3) ITOI8 a
     *  (4) LOR8 (3) (2)  [operand 1 no longer precedes operand 2]
     *
     * Need to detect this situation and to correct the order.
     */
    if (ILMA(ilmptr + 1) > ILMA(ilmptr + 2)) {
      int i1, i2;
      i1 = ILMA(ilmptr + 1);
      i2 = ILMA(ilmptr + 2);
      ILMA(ilmptr + 1) = i2;
      ILMA(ilmptr + 2) = i1;
      mkbranch(ilmptr, truelb, flag);
      return;
    }
    if (!flag) {
      /* if (!ilm1 || !ilm2) goto truelb */
      mkbranch(ILMA(ilmptr + 1), truelb, false);
      mkbranch(ILMA(ilmptr + 2), truelb, false);
      /* erase the AND ilm: */
      if (ilmptr + 3 == ilmb.ilmavl)
        ilmb.ilmavl = ilmptr;
      else {
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr) = IM_NOP;
      }
    } else {
      /*		if (ilm1 && ilm2) goto truelb -->
       *
       *		if (!ilm1) goto falselb
       *		if (ilm2) goto truelb
       * falselb:
       */
      falselb = getlab();
      mkbranch(ILMA(ilmptr + 1), falselb, false);
      mkbranch(ILMA(ilmptr + 2), truelb, true);
      ILMA(ilmptr++) = IM_LABEL;
      ILMA(ilmptr++) = falselb;
      ILMA(ilmptr) = IM_NOP;
    }
  } else if (opc == IM_LOR) {
    if (flag) {
      /* if (ilm1 || ilm2) goto truelb */
      mkbranch(ILMA(ilmptr + 1), truelb, true);
      mkbranch(ILMA(ilmptr + 2), truelb, true);

      /* erase the OR ilm: */
      if (ilmptr + 3 == ilmb.ilmavl)
        ilmb.ilmavl = ilmptr;
      else {
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr) = IM_NOP;
      }
    } else {
      /*		if (!ilm1 && !ilm2) goto truelb -->
       *
       *		if (ilm1) goto falselb
       *		if (!ilm2) goto truelb
       * falselb:
       */
      falselb = getlab();
      mkbranch(ILMA(ilmptr + 1), falselb, true);
      mkbranch(ILMA(ilmptr + 2), truelb, false);
      ILMA(ilmptr++) = IM_LABEL;
      ILMA(ilmptr++) = falselb;
      ILMA(ilmptr) = IM_NOP;
    }
  } else if (opc == IM_LOR8) {
    /*
     * tpr3035, the relational expressions are always logical*4.  In the
     * presence of a logical*8 expression, the relational is converted
     * with a IM_ITOI8.  Unfortunately, this means that the ILM order of
     * the operands to LAND8 to LOR8 may change, and the assumptions to
     * effect short-circuiting are no longer valid; e.g.,
     *  (1) a
     *  (2) b
     *  (3) LOR  (1) (2)  [operand 1 precedes operand 2]
     * becomes
     *  (1) a
     *  (2) b
     *  (3) ITOI8 a
     *  (4) LOR8 (3) (2)  [operand 1 no longer precedes operand 2]
     *
     * Need to detect this situation and to correct the order.
     */
    if (ILMA(ilmptr + 1) > ILMA(ilmptr + 2)) {
      int i1, i2;
      i1 = ILMA(ilmptr + 1);
      i2 = ILMA(ilmptr + 2);
      ILMA(ilmptr + 1) = i2;
      ILMA(ilmptr + 2) = i1;
      mkbranch(ilmptr, truelb, flag);
      return;
    }
    if (flag) {
      /* if (ilm1 || ilm2) goto truelb */
      mkbranch(ILMA(ilmptr + 1), truelb, true);
      mkbranch(ILMA(ilmptr + 2), truelb, true);

      /* erase the OR ilm: */
      if (ilmptr + 3 == ilmb.ilmavl)
        ilmb.ilmavl = ilmptr;
      else {
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr++) = IM_NOP;
        ILMA(ilmptr) = IM_NOP;
      }
    } else {
      /*		if (!ilm1 && !ilm2) goto truelb -->
       *
       *		if (ilm1) goto falselb
       *		if (!ilm2) goto truelb
       * falselb:
       */
      falselb = getlab();
      mkbranch(ILMA(ilmptr + 1), falselb, true);
      mkbranch(ILMA(ilmptr + 2), truelb, false);
      ILMA(ilmptr++) = IM_LABEL;
      ILMA(ilmptr++) = falselb;
      ILMA(ilmptr) = IM_NOP;
    }
  } else {
    if (IS_COMPARE(opc)) {
      /* follow opcode with BRT or BRF*/
      ILMA(ilmptr + 2) = flag ? IM_BRT : IM_BRF;
      ILMA(ilmptr + 3) = ilmptr;
      ILMA(ilmptr + 4) = truelb;
    } else if (opc == IM_LNOP || opc == IM_LNOP8) {
      ILMA(ilmptr) = flag ? IM_BRT : IM_BRF;
      ILMA(ilmptr + 2) = truelb;
    } else if (opc == IM_LNOT || opc == IM_LNOT8) {
      ILMA(ilmptr) = flag ? IM_BRF : IM_BRT;
      ILMA(ilmptr + 2) = truelb;
    } else
      (void)ad3ilm(flag ? IM_BRT : IM_BRF, ilmptr, truelb);

    RFCNTI(truelb); /* increment label reference count */
  }
}

/*****************************************************************/
static int globfile = 0, globindex = 0;

/*
 * dump one ilm
 */
int
_dumponeilm(ILM_T *ilm_base, int i, int check)
{
  int opc, opcp, varpart, val, ty, bsize, sym, pr;
  int j, k;
  INT oprflg; /* bit map defining operand types */
  opc = ilm_base[i];
  opcp = i;
  bsize = ilm_base[BOS_SIZE - 1]; /* number of words in this ILM block */
#define SPECIALOPC 65000
  /* mark opcode, make sure links point to one of these */
  if (check)
    ilm_base[i] = SPECIALOPC;
  if (opc <= 0 || opc >= N_ILM) {
    fprintf(gbl.dbgfil, "%4d ? %6d ?", i, opc);
    k = 0;
    varpart = 0;
  } else {
    k = ilms[opc].oprs;
    oprflg = ilms[opc].oprflag;
    varpart = ((TY(oprflg) == OPR_N) ? ilm_base[i + 1] : 0);
    if (i + k + varpart >= bsize) {
      fprintf(gbl.dbgfil, " (BAD ARG COUNT=%d)", k + varpart);
      varpart = 0;
    }
    j = i + 1 + k + varpart;
    if (j < bsize && ilm_base[j] == IM_FILE) {
      globfile = ilm_base[j + 2];
      globindex = ilm_base[j + 3];
    }
    if (DBGBIT(4, 0x8000)) {
      if (opc == IM_FILE) {
        fprintf(gbl.dbgfil, "%4s %5s  ", "    ", "     ");
      } else {
        fprintf(gbl.dbgfil, "%4d/%5d  ", globfile, globindex);
      }
    }
    if (opc == IM_FILE) {
      /* do nothing */
    } else {
      globindex += k + varpart + 1;
    }
    fprintf(gbl.dbgfil, "%4d %-10.20s", i, ilms[opc].name);
  }

  j = 0;
  sym = 0;
  pr = 0;
  do {
    i++;
    j++;
    if (j <= k) {
      ty = TY(oprflg);
      oprflg >>= 2;
    } else if (j <= k + varpart) {
      if (j == k + 1)
        ty = TY(oprflg);
    } else
      break;

    val = (int)ilm_base[i];
    switch (ty) {
    case OPR_LNK:
      fprintf(gbl.dbgfil, " %4d^", val);
      if (val >= opcp || val < BOS_SIZE ||
          (check && ilm_base[val] != SPECIALOPC)) {
        fprintf(gbl.dbgfil, "<-BAD LINK");
      }
      break;

    case OPR_SYM:
      if (sym == 0)
        sym = val;
      fprintf(gbl.dbgfil, " %5d", val);
      break;

    case OPR_STC:
      if (pr == 0)
        pr = val;
      fprintf(gbl.dbgfil, " %5d", val);
      break;

    case OPR_N:
      fprintf(gbl.dbgfil, " %5d", val);
      if (j != 1 || val < 0) {
        fprintf(gbl.dbgfil, "<-BAD ARG COUNT");
      }
    }
  } while (true);
  if (pr) {
    const char *s;
    switch (opc) {
    case IM_MP_MAP:
    case IM_PRAGMA:
    case IM_PRAGMASYM:
    case IM_PRAGMASLIST:
    case IM_PRAGMAEXPR:
    case IM_PRAGMASYMEXPR:
    case IM_PRAGMASELIST:
    case IM_PRAGMADPSELIST:
    case IM_PRAGMAGEN:
      switch (pr) {
      case PR_NONE:
        s = "NONE";
        break;
      case PR_INLININGON:
        s = "INLININGON";
        break;
      case PR_INLININGOFF:
        s = "INLININGOFF";
        break;
      case PR_ALWAYSINLINE:
        s = "ALWAYSINLINE";
        break;
      case PR_MAYINLINE:
        s = "MAYINLINE";
        break;
      case PR_NEVERINLINE:
        s = "NEVERINLINE";
        break;
      case PR_ACCEL:
        s = "ACCEL";
        break;
      case PR_ENDACCEL:
        s = "ENDACCEL";
        break;
      case PR_INLINEONLY:
        s = "INLINEONLY";
        break;
      case PR_INLINETYPE:
        s = "INLINETYPE";
        break;
      case PR_INLINEAS:
        s = "INLINEAS";
        break;
      case PR_INLINEALIGN:
        s = "INLINEALIGN";
        break;
      case PR_ACCCOPYIN:
        s = "ACCCOPYIN";
        break;
      case PR_ACCCOPYOUT:
        s = "ACCCOPYOUT";
        break;
      case PR_ACCLOCAL:
        s = "ACCLOCAL";
        break;
      case PR_ACCDELETE:
        s = "ACCDELETE";
        break;
      case PR_ACCELLP:
        s = "ACCELLP";
        break;
      case PR_ACCVECTOR:
        s = "ACCVECTOR";
        break;
      case PR_ACCPARALLEL:
        s = "ACCPARALLEL";
        break;
      case PR_ACCSEQ:
        s = "ACCSEQ";
        break;
      case PR_ACCHOST:
        s = "ACCHOST";
        break;
      case PR_ACCPRIVATE:
        s = "ACCPRIVATE";
        break;
      case PR_ACCCACHE:
        s = "ACCCACHE";
        break;
      case PR_ACCSHORTLOOP:
        s = "ACCSHORTLOOP";
        break;
      case PR_ACCBEGINDIR:
        s = "ACCBEGINDIR";
        break;
      case PR_ACCIF:
        s = "ACCIF";
        break;
      case PR_ACCUNROLL:
        s = "ACCUNROLL";
        break;
      case PR_ACCKERNEL:
        s = "ACCKERNEL";
        break;
      case PR_ACCCOPY:
        s = "ACCCOPY";
        break;
      case PR_ACCDATAREG:
        s = "ACCDATAREG";
        break;
      case PR_ACCENTERDATA:
        s = "ACCENTERDATA";
        break;
      case PR_ACCEXITDATA:
        s = "ACCEXITDATA";
        break;
      case PR_ACCENDDATAREG:
        s = "ACCENDDATAREG";
        break;
      case PR_ACCUPDATEHOST:
        s = "ACCUPDATEHOST";
        break;
      case PR_ACCUPDATESELF:
        s = "ACCUPDATESELF";
        break;
      case PR_ACCUPDATEDEVICE:
        s = "ACCUPDATEDEVICE";
        break;
      case PR_ACCUPDATE:
        s = "ACCUPDATE";
        break;
      case PR_ACCINDEPENDENT:
        s = "ACCINDEPENDENT";
        break;
      case PR_ACCWAIT:
        s = "ACCWAIT";
        break;
      case PR_ACCNOWAIT:
        s = "ACCNOWAIT";
        break;
      case PR_ACCIMPDATAREG:
        s = "ACCIMPDATAREG";
        break;
      case PR_ACCENDIMPDATAREG:
        s = "ACCENDIMPDATAREG";
        break;
      case PR_ACCMIRROR:
        s = "ACCMIRROR";
        break;
      case PR_ACCREFLECT:
        s = "ACCREFLECT";
        break;
      case PR_KERNELBEGIN:
        s = "KERNELBEGIN";
        break;
      case PR_KERNEL:
        s = "KERNEL";
        break;
      case PR_ENDKERNEL:
        s = "ENDKERNEL";
        break;
      case PR_KERNELTILE:
        s = "KERNELTILE";
        break;
      case PR_ACCDEVSYM:
        s = "ACCDEVSYM";
        break;
      case PR_ACCIMPDATAREGX:
        s = "ACCIMPDATAREGX";
        break;
      case PR_KERNEL_NEST:
        s = "KERNEL_NEST";
        break;
      case PR_KERNEL_GRID:
        s = "KERNEL_GRID";
        break;
      case PR_KERNEL_BLOCK:
        s = "KERNEL_BLOCK";
        break;
      case PR_ACCDEVICEPTR:
        s = "ACCDEVICEPTR";
        break;
      case PR_ACCPARUNROLL:
        s = "ACCPARUNROLL";
        break;
      case PR_ACCVECUNROLL:
        s = "ACCVECUNROLL";
        break;
      case PR_ACCSEQUNROLL:
        s = "ACCSEQUNROLL";
        break;
      case PR_ACCCUDACALL:
        s = "ACCCUDACALL";
        break;
      case PR_ACCSCALARREG:
        s = "ACCSCALARREG";
        break;
      case PR_ACCENDSCALARREG:
        s = "ACCENDSCALARREG";
        break;
      case PR_ACCSERIAL:
        s = "ACCSERIAL";
        break;
      case PR_ACCENDSERIAL:
        s = "ACCENDSERIAL";
        break;
      case PR_ACCPARCONSTRUCT:
        s = "ACCPARCONSTRUCT";
        break;
      case PR_ACCENDPARCONSTRUCT:
        s = "ACCENDPARCONSTRUCT";
        break;
      case PR_ACCKERNELS:
        s = "ACCKERNELS";
        break;
      case PR_ACCENDKERNELS:
        s = "ACCENDKERNELS";
        break;
      case PR_ACCCREATE:
        s = "ACCCREATE";
        break;
      case PR_ACCPRESENT:
        s = "ACCPRESENT";
        break;
      case PR_ACCPCOPY:
        s = "ACCPCOPY";
        break;
      case PR_ACCPCOPYIN:
        s = "ACCPCOPYIN";
        break;
      case PR_ACCPCOPYOUT:
        s = "ACCPCOPYOUT";
        break;
      case PR_ACCPCREATE:
        s = "ACCPCREATE";
        break;
      case PR_ACCPNOT:
        s = "ACCPNOT";
        break;
      case PR_ACCNO_CREATE:
        s = "ACCNO_CREATE";
        break;
      case PR_ACCPDELETE:
        s = "ACCPDELETE";
        break;
      case PR_ACCASYNC:
        s = "ACCASYNC";
        break;
      case PR_KERNEL_STREAM:
        s = "KERNEL_STREAM";
        break;
      case PR_KERNEL_DEVICE:
        s = "KERNEL_DEVICE";
        break;
      case PR_ACCWAITDIR:
        s = "ACCWAITDIR";
        break;
      case PR_ACCSLOOP:
        s = "ACCSLOOP";
        break;
      case PR_ACCTSLOOP:
        s = "ACCTSLOOP";
        break;
      case PR_ACCKLOOP:
        s = "ACCKLOOP";
        break;
      case PR_ACCTKLOOP:
        s = "ACCTKLOOP";
        break;
      case PR_ACCPLOOP:
        s = "ACCPLOOP";
        break;
      case PR_ACCTPLOOP:
        s = "ACCTPLOOP";
        break;
      case PR_ACCGANG:
        s = "ACCGANG";
        break;
      case PR_ACCWORKER:
        s = "ACCWORKER";
        break;
      case PR_ACCFIRSTPRIVATE:
        s = "ACCFIRSTPRIVATE";
        break;
      case PR_ACCNUMGANGS:
        s = "ACCNUMGANGS";
        break;
      case PR_ACCNUMGANGS2:
        s = "ACCNUMGANGS2";
        break;
      case PR_ACCNUMGANGS3:
        s = "ACCNUMGANGS3";
        break;
      case PR_ACCGANGDIM:
        s = "ACCGANGDIM";
        break;
      case PR_ACCNUMWORKERS:
        s = "ACCNUMWORKERS";
        break;
      case PR_ACCVLENGTH:
        s = "ACCVLENGTH";
        break;
      case PR_ACCWAITARG:
        s = "ACCWAITARG";
        break;
      case PR_ACCREDUCTION:
        s = "ACCREDUCTION";
        break;
      case PR_ACCREDUCTOP:
        s = "ACCREDUCTOP";
        break;
      case PR_ACCCACHEDIR:
        s = "ACCCACHEDIR";
        break;
      case PR_ACCCACHEARG:
        s = "ACCCACHEARG";
        break;
      case PR_ACCHOSTDATA:
        s = "ACCHOSTDATA";
        break;
      case PR_ACCENDHOSTDATA:
        s = "ACCENDHOSTDATA";
        break;
      case PR_ACCUSEDEVICE:
        s = "ACCUSEDEVICE";
        break;
      case PR_ACCUSEDEVICEIFP:
        s = "ACCUSEDEVICEIFP";
        break;
      case PR_ACCCOLLAPSE:
        s = "ACCCOLLAPSE";
        break;
      case PR_ACCFORCECOLLAPSE:
        s = "ACCFORCECOLLAPSE";
        break;
      case PR_ACCDEVICERES:
        s = "ACCDEVICERES";
        break;
      case PR_ACCLINK:
        s = "ACCLINK";
        break;
      case PR_ACCDEVICEID:
        s = "ACCDEVICEID";
        break;
      case PR_ACCLOOPPRIVATE:
        s = "ACCLOOPPRIVATE";
        break;
      case PR_CUFLOOPPRIVATE:
        s = "CUFLOOPPRIVATE";
        break;
      case PR_ACCTILE:
        s = "ACCTILE";
        break;
      case PR_ACCAUTO:
        s = "ACCAUTO";
        break;
      case PR_ACCGANGCHUNK:
        s = "ACCGANGCHUNK";
        break;
      case PR_ACCDEFNONE:
        s = "ACCDEFAULTNONE";
        break;
      case PR_ACCDEFPRESENT:
        s = "ACCDEFAULTPRESENT";
        break;
      case PR_ACCCACHEREADONLY:
        s = "ACCCACHEREADONLY";
        break;
      case PR_ACCFINALEXITDATA:
        s = "ACCFINALEXITDATA";
        break;
      case PR_ACCUPDATEHOSTIFP:
        s = "ACCUPDATEHOSTIFP";
        break;
      case PR_ACCUPDATEDEVICEIFP:
        s = "ACCUPDATEDEVICEIFP";
        break;
      case PR_ACCUPDATESELFIFP:
        s = "ACCUPDATESELFIFP";
        break;
      case PR_ACCATTACH:
        s = "ACCATTACH";
        break;
      case PR_ACCDETACH:
        s = "ACCDETACH";
        break;
      case PR_ACCCOMPARE:
        s = "ACCCOMPARE";
        break;
      case PR_PGICOMPARE:
        s = "PGICOMPARE";
        break;
      case PR_PCASTCOMPARE:
        s = "PCASTCOMPARE";
        break;
      case PR_MAPALLOC:
        s = "MAPALLOC";
        break;
      case PR_MAPDELETE:
        s = "MAPDELETE";
        break;
      case PR_MAPFROM:
        s = "MAPFROM";
        break;
      case PR_MAPRELEASE:
        s = "MAPRELEASE";
        break;
      case PR_MAPTO:
        s = "MAPTO";
        break;
      case PR_MAPTOFROM:
        s = "MAPTOFROM";
        break;
      default:
        s = "?";
        break;
      }
      fprintf(gbl.dbgfil, "		;%s", s);
      break;
#ifdef IM_BTARGET
    case IM_BTARGET:
      fprintf(gbl.dbgfil, "		;");
      if (pr & MP_TGT_NOWAIT)
        fprintf(gbl.dbgfil, " NOWAIT");
      if (pr & MP_TGT_IFTARGET)
        fprintf(gbl.dbgfil, " IFTARGET");
      if (pr & MP_TGT_IFPAR)
        fprintf(gbl.dbgfil, " IFPAR");
      if (pr & MP_TGT_DEPEND_IN)
        fprintf(gbl.dbgfil, " DEPEND_IN");
      if (pr & MP_TGT_DEPEND_OUT)
        fprintf(gbl.dbgfil, " DEPEND_OUT");
      if (pr & MP_TGT_DEPEND_IN)
        fprintf(gbl.dbgfil, " DEPEND_INOUT");
      if (pr & MP_CMB_TEAMS)
        fprintf(gbl.dbgfil, " TEAMS");
      if (pr & MP_CMB_DISTRIBUTE)
        fprintf(gbl.dbgfil, " DISTRIBUTE");
      if (pr & MP_CMB_PARALLEL)
        fprintf(gbl.dbgfil, " PARALLEL");
      if (pr & MP_CMB_FOR)
        fprintf(gbl.dbgfil, " FOR");
      if (pr & MP_CMB_SIMD)
        fprintf(gbl.dbgfil, " SIMD");
      break;
#endif /* BTARGET */
    }
  }
  if (sym) {
    switch (opc) {
    case IM_EHREG_ST:
    case IM_EHRESUME:
      fprintf(gbl.dbgfil, "\t;__catch_clause_number,__caught_object_address");
      break;
    default:
      fprintf(gbl.dbgfil, "		;%s", getprint(sym));
      break;
    }
  }
  return i;
} /* _dumponeilm */


/*
 * dump block of ILM's to debug listing file.
 */
void
_dumpilms(ILM_T *ilm_base, int check)
{
  int i, bsize;
  globfile = 0;
  globindex = 0;

  if (gbl.dbgfil == NULL)
    gbl.dbgfil = stderr;

  if (ilm_base[0] != IM_BOS) {
    fprintf(gbl.dbgfil, "dmpilms: no IM_BOS (ilm_base[0]==%d)\n", ilm_base[0]);
  }

  fprintf(gbl.dbgfil, "\n----- lineno: %d"
#if DEBUG
                      " ----- global ILM index %d:%d"
#endif
                      "\n",
          ilm_base[1]
#if DEBUG
          ,
          ilmb.globalilmstart, ilmb.globalilmcount
#endif
          );
  bsize = ilm_base[BOS_SIZE - 1]; /* number of words in this ILM block */

  i = 0;
  globfile = 1;
  globindex = ilmb.globalilmstart;
  do { /* loop once for each ILM opcode: */
    i = _dumponeilm(ilm_base, i, check);
    fprintf(gbl.dbgfil, "\n");
    if (i > bsize) {
      fprintf(gbl.dbgfil, "BAD BLOCK LENGTH: %d\n", bsize);
    }
  } while (i < bsize);
  globfile = 0;
  globindex = 0;
}

void
dumpilms()
{
  ILMA(BOS_SIZE - 1) = ilmb.ilmavl;
  if (gbl.dbgfil == NULL)
    gbl.dbgfil = stderr;
  _dumpilms(ilmb.ilm_base, 0);
} /* dumpilms */

void
dmpilms()
{
  _dumpilms(ilmb.ilm_base, 1);
} /* dmpilms */

#if DEBUG
static int xsize, xavl;
static int *x;
static FILE *xfile;

static void
putsym(int sptr)
{
  /* we want to print the name; if the name is ..inline, we use
   * an index into the list of inlined names */
  if (strncmp(SYMNAME(sptr), "..inline", 8) != 0) {
    fprintf(xfile, " %s", SYMNAME(sptr));
  } else {
    int xx;
    for (xx = 0; xx < xavl; ++xx) {
      if (x[xx] == sptr)
        break;
    }
    if (xx >= xavl) {
      xx = xavl;
      ++xavl;
      NEED(xavl, x, int, xsize, xsize + 100);
      x[xx] = sptr;
    }
    fprintf(xfile, " ..inline.%d", xx);
  }
} /* putsym */

/** \brief Write a DTYPE to xfile.
    This routine is spelled with an underscore to
    distinguish it from routine putdtype in mwd.c */
static void
put_dtype(DTYPE dtype)
{
  int dty;
  ADSC *ad;
  int numdim;
  dty = DTY(dtype);
  switch (dty) {
  case TY_CMPLX:
  case TY_DBLE:
  case TY_DCMPLX:
  case TY_FLOAT:
  case TY_INT:
  case TY_INT8:
  case TY_LOG:
  case TY_LOG8:
  case TY_NONE:
  case TY_QUAD:
  case TY_SINT:
  case TY_UINT:
  case TY_UINT8:
  case TY_USINT:
  case TY_WORD:
    fprintf(xfile, "%s", stb.tynames[dty]);
    break;
  case TY_CHAR:
    fprintf(xfile, "%s*%" ISZ_PF "d", stb.tynames[dty], DTyCharLength(dtype));
    break;
  case TY_ARRAY:
    fprintf(xfile, "%s", stb.tynames[dty]);
    ad = AD_DPTR(dtype);
    numdim = AD_NUMDIM(ad);
    fprintf(xfile, "(");
    if (numdim >= 1 && numdim <= 7) {
      int i;
      for (i = 0; i < numdim; ++i) {
        if (i)
          fprintf(xfile, ",");
        putsym(AD_LWBD(ad, i));
        fprintf(xfile, ":");
        putsym(AD_UPBD(ad, i));
      }
    }
    fprintf(xfile, ")");
    break;
  case TY_PTR:
    fprintf(xfile, "*(");
    put_dtype(DTySeqTyElement(dtype));
    fprintf(xfile, ")");
    break;

  case TY_PARAM:
    break;
  case TY_STRUCT:
  case TY_UNION:
    if (dty == TY_STRUCT)
      fprintf(xfile, "struct");
    if (dty == TY_UNION)
      fprintf(xfile, "union");
    DTySet(dtype, -dty);
    if (DTyAlgTyTag(dtype)) {
      fprintf(xfile, " ");
      putsym(DTyAlgTyTag(dtype));
    }
    fprintf(xfile, "{");
    if (DTyAlgTyMember(dtype)) {
      int member;
      for (member = DTyAlgTyMember(dtype); member > NOSYM && member < stb.stg_avail;) {
        put_dtype(DTYPEG(member));
        fprintf(xfile, " ");
        putsym(member);
        member = SYMLKG(member);
        fprintf(xfile, ";");
      }
    }
    fprintf(xfile, "}");
    DTySet(dtype, dty);
    break;
  case -TY_STRUCT:
  case -TY_UNION:
    if (dty == -TY_STRUCT)
      fprintf(xfile, "struct");
    if (dty == -TY_UNION)
      fprintf(xfile, "union");
    if (DTyAlgTyTagNeg(dtype)) {
      fprintf(xfile, " ");
      putsym(DTyAlgTyTagNeg(dtype));
    } else {
      fprintf(xfile, " %d", dtype);
    }
    break;
  default:
    break;
  }

} /* put_dtype */

void
dumpsingleilm(ILM_T *ilm_base, int i)
{
  int opc, args, varargs, oprflg, j, sym;
  opc = ilm_base[0];
  args = ilms[opc].oprs;
  oprflg = ilms[opc].oprflag;
  varargs = ((TY(oprflg) == OPR_N) ? ilm_base[1] : 0);
  sym = 0;
  if (ilm_base[0] == IM_BOS) {
  fprintf(gbl.dbgfil, "\n----- lineno: %d"
                      " ----- global ILM index %d:%d"
                      "\n",
          ilm_base[1] , ilm_base[2], ilm_base[3]
          );
  }
  fprintf(gbl.dbgfil, "%4d %s",i, ilms[opc].name);
  for (j = 1; j <= args + varargs; ++j) {
    int ty, val;
    ty = TY(oprflg);
    if (j <= args) {
      oprflg >>= 2;
    }
    val = ilm_base[j];
    switch (ty) {
    case OPR_LNK:
      fprintf(gbl.dbgfil, " op%d", j);
      break;

    case OPR_STC:
      fprintf(gbl.dbgfil, " %5d", val);
      break;
      break;

    case OPR_N:
      fprintf(gbl.dbgfil, " n%d", val);
      break;

    case OPR_SYM:
      if (sym == 0)
        sym = val;
      fprintf(gbl.dbgfil, " %5d", val);
      break;
    }
  }
  if (sym) {
    switch (opc) {
    default:
      fprintf(gbl.dbgfil, "		;%s", getprint(sym));
      break;
    }
  }
  fprintf(gbl.dbgfil, "\n");
} /* dumpsingleilm */

/* dump a single ILM tree */
static void
_dumpilmtree(int i, int indent)
{
  int opc, args, varargs, oprflg, j, sym;
  opc = ILMA(i);
  args = ilms[opc].oprs;
  oprflg = ilms[opc].oprflag;
  varargs = ((TY(oprflg) == OPR_N) ? ILMA(i + 1) : 0);
  for (j = 0; j < indent; ++j)
    fprintf(xfile, "  ");
  fprintf(xfile, "%s", ilms[opc].name);
  sym = 0;
  for (j = 1; j <= args + varargs; ++j) {
    int ty;
    int val;
    ty = TY(oprflg);
    if (j <= args) {
      oprflg >>= 2;
    }
    val = ILMA(i + j);
    switch (ty) {
    case OPR_LNK:
      fprintf(xfile, " op%d", j);
      break;

    case OPR_STC:
      if (opc != IM_FARG && opc != IM_ELEMENT) {
        fprintf(xfile, " %d", val);
      } else {
        /* this is a datatype */
        fprintf(xfile, " ");
        put_dtype((DTYPE)val); // ???
      }
      break;

    case OPR_N:
      fprintf(xfile, " n%d", val);
      break;

    case OPR_SYM:
      if (sym == 0)
        sym = val;
      fprintf(xfile, " %5d", val);
      break;
    }
  }
  if (sym) {
    switch (opc) {
    default:
      fprintf(xfile, "		;%s", getprint(sym));
      break;
    }
  }
  fprintf(xfile, "\n");
  /* now print the subtrees */
  oprflg = ilms[opc].oprflag;
  for (j = 1; j <= args + varargs; ++j) {
    int ty, val;
    ty = TY(oprflg);
    if (j <= args) {
      oprflg >>= 2;
    }
    val = ILMA(i + j);
    switch (ty) {
    case OPR_LNK:
      _dumpilmtree(val, indent + 1);
      break;
    }
  }
} /* _dumpilmtree */

/* dump ILM trees, for comparisons between two different compiles */
void
dumpilmtree(int ilmptr)
{
  int xx;
  xfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  xsize = 100;
  xavl = 0;
  NEW(x, int, xsize);
  fprintf(xfile, "\n----- lineno: %d\n", ILMA(1));
  _dumpilmtree(ilmptr, 0);
  for (xx = 0; xx < xavl; ++xx) {
    fprintf(xfile, "..inline.%d = sptr:%d\n", xx, x[xx]);
  }
  FREE(x);
  xsize = 0;
  xavl = 0;
} /* dumpilmtree */

/* dump ILM trees, for comparisons between two different compiles */
void
dumpilmtrees()
{
  int i, args, xx;
  xfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  xsize = 100;
  xavl = 0;
  NEW(x, int, xsize);
  fprintf(xfile, "\n----- lineno: %d\n", ILMA(1));
  for (i = 0; i < ilmb.ilmavl; i += args + 1) {
    int opc, oprflg;
    opc = ILMA(i);
    if (opc <= 0 || opc >= N_ILM) {
      fprintf(xfile, " OPC=%6d\n", opc);
      args = 0;
    } else if (IM_TRM(opc)) {
      _dumpilmtree(i, 0);
    }
    args = ilms[opc].oprs;
    oprflg = ilms[opc].oprflag;
    args += ((TY(oprflg) == OPR_N) ? ILMA(i + 1) : 0);
  }
  for (xx = 0; xx < xavl; ++xx) {
    fprintf(xfile, "..inline.%d = sptr:%d\n", xx, x[xx]);
  }
  FREE(x);
  xsize = 0;
  xavl = 0;
} /* dumpilmtrees */
#endif

/****************************************************************/
extern void rewindilms();
//#if defined(PGC)
/*
 * ILM file position before starting the current function
 */
static long gilmpos = 0;

long
get_ilmpos()
{
  return gilmpos;
} /* get_ilmpos */

long
get_ilmstart()
{
  ilmb.globalilmstart = gilmb.globalilmtotal;
  ilmb.globalilmcount = gilmb.globalilmtotal;
  return gilmb.globalilmfirst;
} /* get_ilmstart */

void
set_ilmpos(long pos)
{
  int r;
  r = fseek(gbl.ilmfil, pos, SEEK_SET);
  if (r != 0) {
    interr("seek on ILM file failed", 0, ERR_Fatal);
  }
} /* set_ilmpos */

void
set_ilmstart(int start)
{
  ilmb.globalilmstart = start;
  ilmb.globalilmcount = start;
} /* set_ilmstart */

int
get_entry()
{
  if (gilmb.ilm_base[GILMSAVE + BOS_SIZE] == IM_ENTRY) {
    return gilmb.ilm_base[GILMSAVE + BOS_SIZE + 1];
  }
  return 0;
} /* get_entry */
  //#endif

/*
 * read in a function's worth of ILMs into gilmb.ilm_base
 *   gilmb_mode = 1: Read in from gbl.ilmfil into gilmb.ilm_base
 *   gilmb_mode = 2: Read in from ilm_base into gilmb.ilm_base
 */
int
rdgilms(int mode)
{
  int i, nilms, nw, pos, sumilms = 0;
  int ilmx, opc, len;
  gilmb.ilmavl = GILMSAVE;
  gilmb.ilmpos = GILMSAVE;
  gilmb.ilm_base[0] = 0;
  gilmb.ilm_base[1] = 0;
  gilmb.ilm_base[2] = 1;
  if (flg.smp && llvm_ilms_rewrite_mode()) {
    gilmb_mode = 0;
  } else
  {
    gilmb_mode = 0;
    rewindilms();
  }

  gilmb_mode = mode;
  gilmb.globalilmfirst = gilmb.globalilmtotal;
  if (mode != 2) {
    gilmpos = ftell(gbl.ilmfil);
  }
#if DEBUG
  if (DBGBIT(4, 0x80)) {
    fprintf(gbl.dbgfil, "------rdgilms-----\n");
  }
#endif
  do {
    /* we've already determine that we have enough space
     * to read in the BOS block */
    if (gilmb_mode == 1) {
#if DEBUG
      if (DBGBIT(4, 0x80)) {
        fprintf(gbl.dbgfil, "Reading at %ld\n", ftell(gbl.ilmfil));
      }
#endif
      i = fread((void *)(gilmb.ilm_base + gilmb.ilmavl), sizeof(ILM_T),
                BOS_SIZE, gbl.ilmfil);
      if (i == 0) {
        if (gilmb.ilmavl == GILMSAVE)
          return 0;
        return gilmb.ilmavl;
      }
      assert(i == BOS_SIZE, "rdgilms: BOS error", i, ERR_Severe);
    }

    /*
     * determine the number of words remaining in the ILM block
     */
    nilms = gilmb.ilm_base[gilmb.ilmavl + 3];
    gilmb.globalilmtotal += nilms;
    nw = nilms - BOS_SIZE;
    /* read in the remaining part of the ILM block  */
    /* make sure we have enough for this ILM block and the
     * BOS of the next ILM block */
    NEED(gilmb.ilmavl + nilms + 2 * BOS_SIZE + GILMSAVE, gilmb.ilm_base, ILM_T,
         gilmb.ilm_size, gilmb.ilm_size + nilms + 1000);

    if (gilmb_mode == 1) {
      i = fread((void *)(gilmb.ilm_base + gilmb.ilmavl + BOS_SIZE),
                sizeof(ILM_T), nw, gbl.ilmfil);
      assert(i == nw, "grdilms: BLOCK error", nilms, ERR_Severe);
    }

    sumilms += nilms;
    pos = gilmb.ilmavl;
    gilmb.ilm_base[pos - 1] = nilms;
    gilmb.ilmavl += nilms + GILMSAVE;
    /* find the last IM, look for IM_END */
    for (ilmx = BOS_SIZE; ilmx < nilms; ilmx += len) {
      opc = gilmb.ilm_base[pos + ilmx];
#if DEBUG
      if (DBGBIT(4, 0x80)) {
        if (opc < 0 || opc >= N_ILM) {
          fprintf(gbl.dbgfil, "opc:%d\n", opc);
        } else {
          fprintf(gbl.dbgfil, "opc:%s\n", ilms[opc].name);
        }
      }
#endif
      len = ilms[opc].oprs + 1; /* length is number of words */
      if (IM_VAR(opc)) {
        len += gilmb.ilm_base[pos + ilmx + 1]; /* include the variable opnds */
      }
    }
    gilmb.ilm_base[pos + nilms] = 0;
    gilmb.ilm_base[pos + nilms + 1] = 1;
  }
  while (opc != IM_END && opc != IM_ENDF);

  return gilmb.ilmavl;
} /* rdgilms */

/*
 * for Unified binary, save (and below, restore) the gilmb structure
 */
void
SaveGilms(FILE *fil)
{
  int nw;
  /* output the size of the gilmb structure */
  nw = fwrite((void *)&gilmb.ilmavl, sizeof(gilmb.ilmavl), 1, fil);
  if (nw != 1) {
    error(S_0155_OP1_OP2, ERR_Fatal, 0, "Error writing temp file:", "gilmavl");
    exit(1);
  }
  nw = fwrite((void *)gilmb.ilm_base, sizeof(ILM_T), gilmb.ilmavl, fil);
  if (nw != gilmb.ilmavl) {
    error(S_0155_OP1_OP2, ERR_Fatal, 0, "Error writing temp file:", "gilms");
    exit(1);
  }
} /* SaveGilms */

void
RestoreGilms(FILE *fil)
{
  int nw;
  /* output the size of the gilmb structure */
  nw = fread((void *)&gilmb.ilmavl, sizeof(gilmb.ilmavl), 1, fil);
  if (nw != 1) {
    error(S_0155_OP1_OP2, ERR_Fatal, 0, "Error reading temp file:", "gilmavl");
    exit(1);
  }
  NEED(gilmb.ilmavl + BOS_SIZE + GILMSAVE, gilmb.ilm_base, ILM_T,
       gilmb.ilm_size, gilmb.ilmavl + 1000);
  nw = fread((void *)gilmb.ilm_base, sizeof(ILM_T), gilmb.ilmavl, fil);
  if (nw != gilmb.ilmavl) {
    error(S_0155_OP1_OP2, ERR_Fatal, 0, "Error reading temp file:", "gilms");
    exit(1);
  }
  gilmb.ilmpos = GILMSAVE;
  gilmb_mode = 1;
} /* RestoreGilms */

/****************************************************************/

/* Set the value of gilmb_mode */
void
set_gilmb_mode(int mode)
{
  gilmb_mode = mode;
}

/*
 * read in a block of ILMs
 */
int
rdilms()
{
  int i, nw, nilms;

  /* read in the BOS ILM  */

  ilmb.globalilmstart = ilmb.globalilmcount;

  /* gilmb_mode = 0 : Read ILMs from gbl.ilmfil to ilmb.ilm_base
   * gilmb_mode !=0 : Read ILMs from gilmb.ilm_base to ilmb.ilm_base
   */
  if (!gilmb_mode) {
    i = fread((char *)ilmb.ilm_base, sizeof(ILM_T), BOS_SIZE, gbl.ilmfil);
    if (i == 0)
      return 0;
    assert(i == BOS_SIZE, "rdilms: BOS error", i, ERR_Severe);
    fihb.nextfindex = ilmb.ilm_base[2];
    nilms = ilmb.ilm_base[3];
    ilmb.globalilmcount += nilms;
  } else {
    if (gilmb.ilmpos >= gilmb.ilmavl)
      return 0;
    BCOPY(ilmb.ilm_base, gilmb.ilm_base + gilmb.ilmpos, ILM_T, BOS_SIZE);
    fihb.nextfindex = ilmb.ilm_base[2];
    nilms = ilmb.ilm_base[3];
    ilmb.globalilmcount += gilmb.ilm_base[gilmb.ilmpos - 1];
    gilmb.ilmpos += BOS_SIZE;
  }

  /*
   * determine the number of words remaining in the ILM block
   */
  nw = nilms - BOS_SIZE;
  ilmb.ilmavl = nilms;

  if (!gilmb_mode) {
    /* read in the remaining part of the ILM block  */
    i = fread((char *)(ilmb.ilm_base + BOS_SIZE), sizeof(ILM_T), nw,
              gbl.ilmfil);
    assert(i == nw, "rdilms: BLOCK error", nilms, ERR_Severe);
  } else {
    assert(gilmb.ilmpos + nw <= gilmb.ilmavl, "rdilms: BLOCK error", nilms, ERR_Severe);
    NEED(nilms, ilmb.ilm_base, ILM_T, ilmb.ilm_size,
         ilmb.ilm_size + nilms + 1000);
    BCOPY(ilmb.ilm_base + BOS_SIZE, gilmb.ilm_base + gilmb.ilmpos, ILM_T, nw);
    gilmb.ilmpos += nw + GILMSAVE;
  }

#if DEBUG
  if (DBGBIT(4, 0x20)) {
    dumpilms();
  }
#endif

  return nilms;
}

static int saveilmstart = 0, saveilmcount = 0;

/*
 * rewind ilm file, reset counters
 */
void
rewindilms()
{
  int i;
  if (gilmb_mode) {
    reset_global_ilm_position();
  } else {
    i = fseek(gbl.ilmfil, 0L, 0);
    assert(i == 0, "ilmfil seek error", i, ERR_Severe);
  }
  if (fihb.stg_base == NULL) {
    fihb.stg_size = 10;
    NEW(fihb.stg_base, FIH, fihb.stg_size);
    fihb.stg_avail = 1;
    BZERO(fihb.stg_base + 0, FIH, 1);
    FIH_DIRNAME(0) = NULL;
    FIH_FILENAME(0) = nullname;
    FIH_FULLNAME(0) = nullname;
  }
  fihb.nextfindex = 1;
  fihb.nextftag = 0;
  fihb.currfindex = 1;
  fihb.currftag = 0;
  ilmb.globalilmstart = saveilmstart;
  ilmb.globalilmcount = saveilmcount;
} /* rewindilms */

/*
 * rewind ilm file in preparation for starting a new Fortran subprogram
 */
void
restartilms(void)
{
  int i;
  i = fseek(gbl.ilmfil, 0L, 0);
  assert(i == 0, "ilmfil seek error", i, ERR_Severe);
  /* save ilmstart/ilmcount values so when we rewind to start the next
   * subprogram, we're starting at the same point */
  saveilmstart = ilmb.globalilmstart;
  saveilmcount = ilmb.globalilmcount;
} /* restartilms */

/*
 * Count the number of ILMs only contribute to the code generation
 */
int
count_ilms()
{
  int ilmx, len, newnumilms, nilms, begin_ilm;
  ILM_OP opc;

  nilms = ilmb.ilm_base[BOS_SIZE - 1];
  newnumilms = nilms;
  begin_ilm = BOS_SIZE;

  for (ilmx = begin_ilm; ilmx < nilms; ilmx += len) {
    opc = (ILM_OP)ilmb.ilm_base[ilmx]; // ???
#if DEBUG
    assert(opc > IM_null && opc < N_ILM, "count_ilms: bad ilm", opc, ERR_Severe);
#endif
    len = ilms[opc].oprs + 1;
    if (IM_VAR(opc))
      len += *(ilmb.ilm_base + ilmx + 1);
#if DEBUG
    assert(len > 0, "count_ilms: bad len", opc, ERR_Severe);
#endif
    if (IM_NOINLC(opc)) {
      newnumilms -= len;
    }
  }
#if DEBUG
  assert(nilms >= newnumilms, "count_ilms: bad newnumilms", opc, ERR_Severe);
#endif
  return newnumilms;
}

