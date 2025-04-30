/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Register allocation module used by the expander and the optimizer
 *
 * Contains:
 * - void reg_init(int)  - initializes the register information for a function
 * - void addrcand(int)  - adds an ILI to the appropriate  candidate list
 * - void static dump_cand(int) - dumps a candidate list for debug purposes
 * - void dmprcand()     - dumps the register candidate lists for debug
 *   purposes
 * - void dmprat(int)    - dumps the register assign table for a block
 * - int getrcand(int)   - gets a candidate for register assignment from a
 *   candidate list
 * - void endrcand()     - clean up the register candidate lists
 * - void storedums(int, int)  - store back dummies assigned registers (SCFTN
 *   only)
 * - void mkrtemp_init() - initialize for register temporaries created for a
 *   function
 * - int mkrtemp(int)    - create a register temporary (based on ili)
 * - int mkrtemp_sc(int, int) - same as mkrtemp(), but storage class is passed
 * - int mkrtemp_cpx(int)- create a complex register temporary (base on dtype)
 * - int mkrtemp_cpx_sc(int, int)- same as mkrtemp_cpx(), but storage class is
 *   passed
 * - void mkrtemp_end()  - end the register temporaries for the current
 *   function
 * - void mkrtemp_update(int[]) - record maximum values of register temporaries
 * - void mkrtemp_reinit(int[]) - initialize for register temporaries from the
 *   recorded set of values
 * - void mkrtemp_copy(int[])   - copy maximum values of register temporaries
 * - int assn_rtemp(int) - assign a register temporary for an ili and make a
 *   candidate.
 * - int assn_rtemp_sc(int, int) - same as assn_rtemp, but storage class is
 *   passed
 */
#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "ili.h"
#include "expand.h"
#include "regutil.h"
#include "machreg.h"
#include "machar.h"

#define RATA_ALL RATA_KR

static const char *atype_names[] = {"nme ", "cons", "ili ", "temp",
                                    "ind ", "arr ", "rpl "};
static const char *get_msize(int);

/* NOTE: init. is dependent on defines in regutil.h */
int il_rtype_df[RATA_RTYPES_TOTAL] = {
    IL_IRDF, IL_SPDF, /* RATA_IR   , RATA_SP */
    IL_DPDF, IL_ARDF, /* RATA_DP   , RATA_AR */
    IL_KRDF, 0,       /* RATA_KR   , RATA_VECT */
    0,       0,       /* RATA_QP   , RATA_CSP */
    0,       0,       /* RATA_CDP  , RATA_CQP */
    0,       0,       /* RATA_X87  , RATA_CX87*/
};
int il_mv_rtype[RATA_RTYPES_TOTAL] = {
    IL_MVIR, IL_MVSP, /* RATA_IR   , RATA_SP */
    IL_MVDP, IL_MVAR, /* RATA_DP   , RATA_AR */
    IL_MVKR, 0,       /* RATA_KR   , RATA_VECT */
    0,       0,       /* RATA_QP   , RATA_CSP */
    0,       0,       /* RATA_CDP  , RATA_CQP */
    0,       0,       /* RATA_X87  , RATA_CX87*/
};

void
reg_init(int entr)
{
  mr_init();
  rcandb.static_cnt = rcandb.const_cnt = 0;
  aux.curr_entry->arasgn = NULL;
}

/**
  \brief Add a potential candidate, an ILI, to the appropriate candidate list.
 */
void
addrcand(int ilix)
{
  int atype, /* type of the candidate    */
      rtype, /* register type of the ili */
      msize, /* memory size of the ili   */
      nme,   /* names entry of the ili   */
      rcand; /* RCAND for the ili        */

  switch (ILI_OPC(ilix)) { /* case according to the ILI opcode */

  case IL_ACEXT:
    /* these are never seen by optimizer in c , for now, just return */
    assert(STYPEG(CONVAL1G(ILI_OPND(ilix, 1))) == ST_LABEL,
           "unexpected acext in addrcand", ilix, ERR_Severe);
    return;
  case IL_LD: /* load data register */
    rtype = RATA_IR;
    msize = ILI_OPND(ilix, 3);
    goto ac_load;

  case IL_LDKR:
    rtype = RATA_KR;
    msize = MSZ_I8;
    goto ac_load;

  case IL_VCON:
  case IL_VLD:
  case IL_VLDU:
  case IL_VST:
  case IL_VSTU:
    /* ignore these */
    return;

  case IL_LDA: /* load address register */
    rtype = RATA_AR;
    msize = MSZ_PTR;
    goto ac_load;
  case IL_LDQ: /* load m128*/
    rtype = RATA_DP;
    msize = MSZ_F16;
    goto ac_load;
  case IL_LD256: /* load m256*/
    rtype = RATA_DP;
    msize = MSZ_F32;
    goto ac_load;
  case IL_LD256A: /* load m256*/
    rtype = RATA_DP;
    msize = MSZ_F32;
    goto ac_load;
  case IL_LDSCMPLX:
    rtype = RATA_CSP;
    msize = MSZ_F8;
    goto ac_load;
  case IL_LDDCMPLX:
    rtype = RATA_CDP;
    msize = MSZ_F16;
    goto ac_load;

  case IL_LDSP: /* load single precision */
    rtype = RATA_SP;
    msize = MSZ_F4;
    goto ac_load;

  case IL_LDDP: /* load double precision */
    rtype = RATA_DP;
    msize = MSZ_F8;

  ac_load: /* common entry for the loads */

    /*
     * check if the ili is already a candidate; if so, just increment its
     * (the candidate entry for the ili) count
     */
    if ((rcand = ILI_RAT(ilix)) != 0) {
      RCAND_COUNT(rcand) += rcandb.weight;
      if (RCAND_ATYPE(rcand) == RATA_ILI && RCAND_VAL(rcand) != ilix) {
        RCAND_OLOAD(rcand) = ilix;
      }
    } else if ((rcand = NME_RAT(nme = ILI_OPND(ilix, 2))) != 0) {
      if (RCAND_MSIZE(rcand) != msize) {
        /*
         * the nme was entered by a store and then another type of
         * load ILI followed, or just another type of load ILI
         * occurred.  This occurs when type casting the & of a
         * variable and then using the lvalue in a context which is
         * inconsistent with its data type.
         */
        RCAND_CONFL(rcand) = 1;
      } else {
        /*
         * the store for this load has already been processed. enter
         * the load ili into the candidate list and back fill the ili
         */
        ILI_RAT(ilix) = rcand;
        if (RCAND_ATYPE(rcand) == RATA_ILI && RCAND_VAL(rcand) != ilix) {
          RCAND_OLOAD(rcand) = ilix;
        } else {
          RCAND_VAL(rcand) = ilix;
          RCAND_ATYPE(rcand) = RATA_ILI;
        }
        RCAND_COUNT(rcand) += rcandb.weight;
      }
    } else {

      /*
       * the ili is a new entry; enter it into the appropriate
       * candidate list and back fill the ili and its names entry
       */
      atype = RATA_ILI;
      goto add_entry;
    }
    break;

  add_entry: /* add an entry to the candidate table; there
              * are three variables -- rtype, atype, and
              * ilix
              */
    GET_RCAND(rcand);
    RCAND_NEXT(rcand) = reg[rtype].rcand;
    reg[rtype].rcand = rcand;
    RCAND_RTYPE(rcand) = rtype;
    RCAND_ATYPE(rcand) = atype;
    RCAND_MSIZE(rcand) = msize;
    RCAND_OLOAD(rcand) = 0;

    if (atype != RATA_NME) { /* for loads, back fill the ili */
      ILI_RAT(ilix) = rcand;
      RCAND_STORE(rcand) = 0;
    } else /* for RATA_NME (stores) don't back fill the
            * store ILI */
      RCAND_STORE(rcand) = 1;

    RCAND_VAL(rcand) = ilix;
    NME_RAT(nme) = rcand;
    RCAND_COUNT(rcand) = rcandb.weight;
    break;

  case IL_ST: /* store data register */
    rtype = RATA_IR;
    msize = ILI_OPND(ilix, 4);
    goto ac_store;

  case IL_STKR:
    rtype = RATA_KR;
    msize = MSZ_I8;
    goto ac_store;

  case IL_STA: /* store address register */
    rtype = RATA_AR;
    msize = MSZ_PTR;
    goto ac_store;
  case IL_STQ: /* store m128 */
    rtype = RATA_DP;
    msize = MSZ_F16;
    goto ac_store;
  case IL_ST256: /* store m256 */
    rtype = RATA_DP;
    msize = MSZ_F32;
    goto ac_store;

  case IL_STSP: /* store single precision */
  case IL_SSTS_SCALAR:
    rtype = RATA_SP;
    msize = MSZ_F4;
    goto ac_store;
  case IL_STSCMPLX:
    rtype = RATA_CSP;
    msize = MSZ_F8;
    goto ac_store;
  case IL_STDCMPLX:
    rtype = RATA_CDP;
    msize = MSZ_F16;
    goto ac_store;
  case IL_STDP: /* store double precision */
  case IL_DSTS_SCALAR:
    rtype = RATA_DP;
    msize = MSZ_F8;
  ac_store: /* common entry for the stores	 */
    if ((rcand = NME_RAT(nme = ILI_OPND(ilix, 3))) != 0)
      if (RCAND_MSIZE(rcand) != msize) {
        /* this store conflicts with the existing candidate  */
        RCAND_CONFL(rcand) = 1;
      } else {
        /*
         * the store has already been added to the candidate list
         * (denoted by the rat field of its names entry)
         */
        RCAND_COUNT(rcand) += rcandb.weight;
        RCAND_STORE(rcand) = 1;
      }
    else {

      /*
       * the store is a new entry; enter the names entry into the
       * candidate list (the candidate type is names) and back fill the
       * name entry
       */
      atype = RATA_NME;
      goto add_entry;
    }
    break;
  case IL_ACON:
    rtype = RATA_AR;
    msize = MSZ_PTR;
    goto add_constant;
  case IL_ICON:
    rtype = RATA_IR;
    msize = MSZ_WORD;
    goto add_constant;
  case IL_KCON:
    rtype = RATA_KR;
    msize = MSZ_I8;
    goto add_constant;
  case IL_FCON:
    rtype = RATA_SP;
    msize = MSZ_F4;
    goto add_constant;
  case IL_SCMPLXCON:
    rtype = RATA_CSP;
    msize = MSZ_F8;
    goto add_constant;
  case IL_DCMPLXCON:
    rtype = RATA_CDP;
    msize = MSZ_F16;
    goto add_constant;
  case IL_DCON:
    rtype = RATA_DP;
    msize = MSZ_F8;
  add_constant:
    rcand = ILI_RAT(ilix);
    if (rcand) {
      RCAND_COUNT(rcand) += rcandb.weight;
    } else {
      GET_RCAND(rcand); /* RCAND_OK = 0; prove it's ok to assn reg */
      RCAND_RTYPE(rcand) = rtype;
      RCAND_ATYPE(rcand) = RATA_CONST;
      RCAND_MSIZE(rcand) = msize;
      RCAND_VAL(rcand) = ilix;
      RCAND_COUNT(rcand) = rcandb.weight;
      ILI_RAT(ilix) = rcand;
      RCAND_NEXT(rcand) = reg[rtype].rcand;
      reg[rtype].rcand = rcand;
    }
    break;

#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CON:
  case IL_FLOAT128LD:
  case IL_FLOAT128ST:
    /* float128 values are not register candidates. */
    return;
#endif /* LONG_DOUBLE_FLOAT128 */

  default:
    if (ILI_RAT(ilix) == 0) {
      assert(ILI_RAT(ilix) != 0, "addrcand: no cand for ili", ilix, ERR_Severe);
      return;
    }
    RCAND_COUNT(ILI_RAT(ilix)) += rcandb.weight;
    break;
  }
}

static void
dump_cand(int candl)
{
  int temp, val, sym, i;
  fprintf(gbl.dbgfil, "Type     Value     Count\n");
  do {
    val = RCAND_VAL(candl);
    fprintf(gbl.dbgfil, "%4s     ", atype_names[temp = RCAND_ATYPE(candl)]);
    fprintf(gbl.dbgfil, "%-5u     %4d", val, (int)RCAND_COUNT(candl));
    sym = 0;
    switch (temp) {
    case RATA_NME:
      fprintf(gbl.dbgfil, " - %-9s", ilis[temp = ILI_OPC(val)].name);
      if (IL_TYPE(temp) == ILTY_STORE) {
        fprintf(gbl.dbgfil, "\"");
        sym = print_nme((int)ILI_OPND(val, 3));
        fprintf(gbl.dbgfil, "\"");
      }
      break;
    case RATA_IND:
      fprintf(gbl.dbgfil, " - \"");
      sym = print_nme((int)ILI_OPND(val, 2));
      fprintf(gbl.dbgfil, "\"");
      for (i = RCAND_TEMP(candl); i; i = RCAND_NEXT(i))
        fprintf(gbl.dbgfil, ", %u^", RCAND_VAL(i));
      break;
    case RATA_ARR:
      fprintf(gbl.dbgfil, " - \"");
      sym = print_nme((int)ILI_OPND(val, 2));
      fprintf(gbl.dbgfil, "\"");
      i = RCAND_TEMP(candl);
      fprintf(gbl.dbgfil, ", %u^ %u^", RCAND_VAL(i), RCAND_TEMP(i));
      break;
    case RATA_ILI:
      fprintf(gbl.dbgfil, " - %-9s", ilis[temp = ILI_OPC(val)].name);
      if (IL_TYPE(temp) == ILTY_LOAD) {
        fprintf(gbl.dbgfil, "\"");
        sym = print_nme((int)ILI_OPND(val, 2));
        fprintf(gbl.dbgfil, "\"");
      }
      break;
    case RATA_TEMP:
      fprintf(gbl.dbgfil, " - %-9s", ilis[ILI_OPC(val)].name);
      fprintf(gbl.dbgfil, " \"%s\"", getprint((int)RCAND_TEMP(candl)));
      break;
    case RATA_RPL:
      fprintf(gbl.dbgfil, " - %-9s", ilis[ILI_OPC(val)].name);
      fprintf(gbl.dbgfil, " %u^", RCAND_TEMP(candl));
      break;
    case RATA_CONST:
      fprintf(gbl.dbgfil, " - %-9s", ilis[ILI_OPC(val)].name);
      temp = ILI_OPND(val, 1);
      if (ILI_OPC(val) != IL_ACON)
        fprintf(gbl.dbgfil, " \"%s\"", getprint(temp));
      else {
        if (CONVAL1G(temp))
          fprintf(gbl.dbgfil, " \"%s", getprint((int)CONVAL1G(temp)));
        else
          fprintf(gbl.dbgfil, " \"%d", CONVAL1G(temp));
        fprintf(gbl.dbgfil, ",%" ISZ_PF "d\"", ACONOFFG(temp));
      }
      break;
    }
    if (RCAND_RTYPE(candl) != RATA_VECT)
      fprintf(gbl.dbgfil, " <%s>", get_msize(RCAND_MSIZE(candl)));
    else
      fprintf(gbl.dbgfil, " <DT %d>", RCAND_MSIZE(candl));
    if (RCAND_CONFL(candl))
      fprintf(gbl.dbgfil, "<confl>");
    if (sym) {
      if (ADDRTKNG(sym))
        fprintf(gbl.dbgfil, "<&>");
      if (RCAND_STORE(candl))
        fprintf(gbl.dbgfil, "<stored>");
    }
    if (RCAND_OK(candl))
      fprintf(gbl.dbgfil, "<cnst ok>");
    if (RCAND_NOREG(candl))
      fprintf(gbl.dbgfil, "<noreg>");
    if (RCAND_CSE(candl))
      fprintf(gbl.dbgfil, "<cse>");
    if (RCAND_IGNORE(candl))
      fprintf(gbl.dbgfil, "<ignore>");
    if (RCAND_INV(candl))
      fprintf(gbl.dbgfil, "<inv>");
    fprintf(gbl.dbgfil, "\n");
    candl = RCAND_NEXT(candl);
  } while (candl != 0);
  fflush(gbl.dbgfil);
}

void
dmprcand(void)
{
  int candl;

  candl = reg[RATA_IR].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  IR Candidates  *****\n");
    dump_cand(candl);
  }
  candl = reg[RATA_AR].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  AR Candidates  *****\n");
    dump_cand(candl);
  }
  candl = reg[RATA_SP].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  SP Candidates  *****\n");
    dump_cand(candl);
  }
  candl = reg[RATA_DP].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  DP Candidates  *****\n");
    dump_cand(candl);
  }
  candl = reg[RATA_KR].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  KR Candidates  *****\n");
    dump_cand(candl);
  }
  candl = reg[RATA_VECT].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  VECT Candidates  *****\n");
    dump_cand(candl);
  }
  candl = reg[RATA_CSP].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  CSP Candidates  *****\n");
    dump_cand(candl);
  }
  candl = reg[RATA_CDP].rcand;
  if (candl != 0) {
    fprintf(gbl.dbgfil, "\n*****  CDP Candidates  *****\n");
    dump_cand(candl);
  }
}

void
dmp_rat(int rat)
{
  int val;

  switch (RAT_ATYPE(rat)) {
  case RATA_TEMP:
  case RATA_RPL:
    fprintf(gbl.dbgfil, "%-7d    ", RAT_REG(rat));
    break;
  default:
    switch (RAT_RTYPE(rat)) {
    case RATA_IR:
      fprintf(gbl.dbgfil, "ir(%2d)     ", (int)ILI_OPND(RAT_REG(rat), 1));
      break;
    case RATA_AR:
      fprintf(gbl.dbgfil, "ar(%2d)     ", (int)ILI_OPND(RAT_REG(rat), 1));
      break;
    case RATA_SP:
      fprintf(gbl.dbgfil, "sp(%2d)     ", (int)ILI_OPND(RAT_REG(rat), 1));
      break;
    case RATA_DP:
      fprintf(gbl.dbgfil, "dp(%3d)    ", (int)ILI_OPND(RAT_REG(rat), 1));
      break;
#ifdef RATA_CSP
    case RATA_CSP:
      fprintf(gbl.dbgfil, "csp(%3d)    ", (int)ILI_OPND(RAT_REG(rat), 1));
      break;
    case RATA_CDP:
      fprintf(gbl.dbgfil, "cdp(%3d)    ", (int)ILI_OPND(RAT_REG(rat), 1));
      break;
    case RATA_CQP:
      fprintf(gbl.dbgfil, "cqp(%3d)    ", (int)ILI_OPND(RAT_REG(rat), 1));
      break;
#endif
    case RATA_KR:
      fprintf(gbl.dbgfil, "kr(%2d,%2d)  ", KR_MSH(ILI_OPND(RAT_REG(rat), 1)),
              KR_LSH(ILI_OPND(RAT_REG(rat), 1)));
      break;
    }
  }
  fprintf(gbl.dbgfil, "     %4s", atype_names[RAT_ATYPE(rat)]);
  val = RAT_VAL(rat);
  fprintf(gbl.dbgfil, "     %-5u", val);
  switch (RAT_ATYPE(rat)) {
  case RATA_NME:
    fprintf(gbl.dbgfil, "     \"");
    (void)print_nme(val);
    fprintf(gbl.dbgfil, "\"");
    break;
  case RATA_TEMP:
  case RATA_ILI:
  case RATA_RPL:
    fprintf(gbl.dbgfil, " - %-9s", ilis[ILI_OPC(val)].name);
    break;
  }
  if (RAT_RTYPE(rat) != RATA_VECT)
    fprintf(gbl.dbgfil, " <%s>", get_msize(RAT_MSIZE(rat)));
  else
    fprintf(gbl.dbgfil, " <DT %d>", RAT_MSIZE(rat));

  if (RAT_CONFL(rat))
    fprintf(gbl.dbgfil, " <confl>");
  if (RAT_ATYPE(rat) == RATA_NME && RAT_STORE(rat))
    fprintf(gbl.dbgfil, " <stored>");
  fprintf(gbl.dbgfil, " addr: %u", RAT_ADDR(rat));
  fprintf(gbl.dbgfil, "\n");
}

static const char *
get_msize(int msz)
{
  const char *p;

  switch (msz) {
  case MSZ_SBYTE:
    p = "sb";
    break;
  case MSZ_SHWORD:
    p = "sh";
    break;
  case MSZ_SWORD:
    p = "wd";
    break;
  case MSZ_SLWORD:
    p = "wd";
    break;
  case MSZ_UBYTE:
    p = "ub";
    break;
  case MSZ_UHWORD:
    p = "uh";
    break;
  case MSZ_PTR:
    p = "pt";
    break;
  case MSZ_ULWORD:
    p = "uw";
    break;
  case MSZ_F4:
    p = "fl";
    break;
  case MSZ_F8:
    p = "db";
    break; /* this can get confused with csp */
  case MSZ_F16:
    p = "cdp";
    break;
  case MSZ_I8:
    p = "i8";
    break;
  default:
    interr("get_msize: unknown msize", msz, ERR_Warning);
    p = "??";
  }
  return p;
}

void
dmprat(int rat)
{
  int nregs;

  if (rat != 0) {
    fprintf(gbl.dbgfil, "\n*****  Register Assigned Table (%d)  ", rat);
    nregs = RAT_VAL(rat);
    fprintf(gbl.dbgfil, "nregs: %d  *****\n", nregs);
    fprintf(gbl.dbgfil, "Register        Type     Value\n");
    while (nregs > 0) {
      rat++;
      dmp_rat(rat);
      nregs--;
    }
  }
}

/**
   \brief Get a candidate from the candidate list, candl.
 */
int
getrcand(int candl)
{
  int cand, count;

  cand = 0;             /* value of getrcand if a candidate is not
                         * found
                         */
  count = GR_THRESHOLD; /* a candidate must have a count greater than
                         * this value (defined in machreg.h)
                         */

  /*
   * scan through the candidate list to find a the candidate with the
   * maximum count
   */
  for (; candl; candl = RCAND_NEXT(candl))
    if (RCAND_COUNT(candl) > count) {

      /*
       * if the candidate variable was used in contexts which conflicts
       * with its true register type, get the next candidate
       */
      if (RCAND_CONFL(candl)) {
        RCAND_COUNT(candl) = 0;
        continue;
      }
      cand = candl;              /* potential candidate */
      count = RCAND_COUNT(cand); /* new maximum count */
    }
  rcandb.count = RCAND_COUNT(cand);
  RCAND_COUNT(cand) = 0; /* clear the candidate's count. this
                          * "removes" the candidate from the list
                          */
  return (cand);
}

/**
   \brief Go through all of the candidate lists to reinitialize the back
   pointers of the names entries and ILI.

   The regs' rcand fields are set to null, and the register candidate list is
   reset.
 */
void
endrcand(void)
{
  int rtype, cand, val, i;

  for (rtype = 0; rtype <= RATA_ALL; rtype++) {
    for (cand = reg[rtype].rcand; cand; cand = RCAND_NEXT(cand)) {
      val = RCAND_VAL(cand);
      switch (RCAND_ATYPE(cand)) {
      case RATA_NME:
        NME_RAT(ILI_OPND(val, 3)) = 0;
        break;

#ifdef RATA_UPLV
      case RATA_UPLV:
#endif
      case RATA_ILI:
        NME_RAT(ILI_OPND(val, 2)) = 0;
        if (RCAND_OLOAD(cand))
          ILI_RAT(RCAND_OLOAD(cand)) = 0;
        FLANG_FALLTHROUGH;
      case RATA_TEMP:
      case RATA_CONST:
      case RATA_RPL:
        ILI_RAT(val) = 0;
        break;

      case RATA_IND:
        ILI_RAT(val) = 0;
        NME_RAT(ILI_OPND(val, 2)) = 0;
        for (i = RCAND_TEMP(cand); i; i = RCAND_NEXT(i))
          ILI_RAT(RCAND_VAL(i)) = 0;
        break;

      case RATA_ARR:
        ILI_RAT(val) = 0;
        NME_RAT(ILI_OPND(val, 2)) = 0;
        break;
      }
    }
    reg[rtype].rcand = 0;
  }

  rtype = RATA_VECT;
  {
    for (cand = reg[rtype].rcand; cand; cand = RCAND_NEXT(cand)) {
      val = RCAND_VAL(cand);
      switch (RCAND_ATYPE(cand)) {
      case RATA_TEMP:
        ILI_RAT(val) = 0;
        break;
      default:
        interr("endrcand: unexpected atype for RATA_VECT", RCAND_ATYPE(cand),
               ERR_Severe);
      }
    }
    reg[rtype].rcand = 0;
  }
  for (rtype = RATA_CSP; rtype <= RATA_CDP; rtype++) {
    for (cand = reg[rtype].rcand; cand; cand = RCAND_NEXT(cand)) {
      val = RCAND_VAL(cand);
      switch (RCAND_ATYPE(cand)) {
      case RATA_NME:
        NME_RAT(ILI_OPND(val, 3)) = 0;
        break;

      case RATA_ILI:
        NME_RAT(ILI_OPND(val, 2)) = 0;
        if (RCAND_OLOAD(cand))
          ILI_RAT(RCAND_OLOAD(cand)) = 0;
        FLANG_FALLTHROUGH;
      case RATA_TEMP:
      case RATA_CONST:
      case RATA_RPL:
        ILI_RAT(val) = 0;
        break;

      case RATA_IND:
        ILI_RAT(val) = 0;
        NME_RAT(ILI_OPND(val, 2)) = 0;
        for (i = RCAND_TEMP(cand); i; i = RCAND_NEXT(i))
          ILI_RAT(RCAND_VAL(i)) = 0;
        break;

      case RATA_ARR:
        ILI_RAT(val) = 0;
        NME_RAT(ILI_OPND(val, 2)) = 0;
        break;
      }
    }
    reg[rtype].rcand = 0;
  }

  rcandb.stg_avail = 1; /* reset the register candidate area */
}

/**
   \brief For ftn, all dummies which are assigned to registers and are stored
   into must have their values stored back into memory.  These stores are added
   to the beginning of the exit block.
 */
void
storedums(int exitbih, int first_rat)
{
  int rat, i, nme, cnt;
  int addr;

  rdilts(exitbih); /* read in the exit block */
  rat = first_rat; /* first rat entry */
  for (cnt = RAT_VAL(rat++); cnt--; rat++) {
    i = basesym_of(nme = RAT_VAL(rat));
    if (!IS_DUM(i))
      continue;
    if (!RAT_STORE(rat))
      continue;
    /*
     * For a language like PASCAL
     *   if (OUT(i)) continue;
     */
    addr = RAT_ADDR(rat);
    i = RAT_REG(rat);
    switch (RAT_RTYPE(rat)) {
    case RATA_AR:
      (void)addilt(0, ad3ili(IL_STA, i, addr, nme));
      break;
    case RATA_IR:
        (void)addilt(0, ad4ili(IL_ST, i, addr, nme, RAT_MSIZE(rat)));
      break;
    case RATA_KR:
      (void)addilt(0, ad4ili(IL_STKR, i, addr, nme, MSZ_I8));
      break;
    case RATA_SP:
      (void)addilt(0, ad4ili(IL_STSP, i, addr, nme, MSZ_F4));
      break;
    case RATA_DP:
      (void)addilt(0, ad4ili(IL_STDP, i, addr, nme, MSZ_F8));
      break;
    case RATA_CSP:
      (void)addilt(0, ad4ili(IL_STSCMPLX, i, addr, nme, MSZ_F8));
      break;
    case RATA_CDP:
      (void)addilt(0, ad4ili(IL_STDCMPLX, i, addr, nme, MSZ_F16));
      break;
    }
  }
  BIH_SMOVE(exitbih) = 1; /* (temp) mark block so sched limits scratch set */
  wrilts(exitbih);        /* write out the exit block */
}

/*  Register Temporary Section -
    These routines provide a mechanism to create temporaries during
    the expansion of ILM blocks which can be re-used for different
    ILM blocks and are unique across functions.  The scenario (at
    opt 0 or 1) is:
    1.  At the beginning of an ILM block, initialize (mkrtemp_init).
    2.  Create temporaries during expansion by calling mkrtemp.
    3.  At the end of a function, ensure that the temporaries created
        for the function are unique from those created in the next
        function (mkrtemp_end).

    For opt 2, the scenario for the loops in a function is:
    0.  At the beginning of a function, mkrtemp_copy
    1.  For innermost loops, mkrtemp_init.  Otherwise, use
        mkrtemp_reinit (this gets the values created from calls to
        mkrtemp_update of all contained loops).
    2.  During optimizations, use mkrtemp.
    3.  At the end of optimizing a loop, mkrtemp_update for the
        parent of the loop. This will keep track of the rtemps used
        in contained loops.
    4.  At the end of a function, mkrtemp_end();

    NOTE: any change in the number of elements in the rtemp array
    must also be applied to the macro RTEMPS which is defined in
    regutil.h.
*/

static struct {        /* Register temporary information */
  const char prefix;   /* beginning char of name */
  const char *arg1pfx; /* ARG1PTR Q&D - SEE f13720 */
  DTYPE dt;            /* data type chosen for the temp */
  int current;         /* current index to be used in name */
  int start;           /* start value of index for a function */
  int max;             /* maximum index used for the file */
} rtemps[] = {
    {'D', "Da", DT_INT, 0, 0, -1},    /* 0: data register temps */
    {'G', "Ga", DT_CPTR, 0, 0, -1},   /* 1: address register temps */
    {'H', "Ha", DT_FLOAT, 0, 0, -1},  /* 2: single register temps */
    {'K', "Ka", DT_DBLE, 0, 0, -1},   /* 3: double register temps */
    {'g', "ga", DT_INT8, 0, 0, -1},   /* 4: integer*8 temps */
    {'h', "ha", DT_CMPLX, 0, 0, -1},  /* 5: complex temps */
    {'k', "ka", DT_DCMPLX, 0, 0, -1}, /* 6: double complex temps */
    {'h', "ha", DT_NONE, 0, 0, -1},   /* 7: filler */
    {'v', "va", DT_NONE, 0, 0, -1},   /* 8: vector temps */
#if defined(LONG_DOUBLE_FLOAT128)
    {'X', "Xa", DT_FLOAT128, 0, 0, -1}, /* 9: float128 temps */
    {'x', "xa", DT_CMPLX128, 0, 0, -1}, /*10: float128 complex temps */
#elif defined(TARGET_SUPPORTS_QUADFP)
    {'X', "Xa", DT_QUAD, 0, 0, -1},   /* 9: quad precision temps  */
    {'x', "xa", DT_QCMPLX, 0, 0, -1}, /* 10: quad complex temps */
#else
    {'X', "Xa", DT_NONE, 0, 0, -1}, /* 9 and 10: filler */
    {'x', "xa", DT_NONE, 0, 0, -1}, /* 9 and 10: filler */
#endif
};

static int select_rtemp(int);

/**
   \brief This routine initializes for temporaries created during an ILM block.
    This guarantees that the temps used will be unique for a given block.
*/
void
mkrtemp_init(void)
{
  int i;
  /* Both DOUBLE_DOUBLE && LONG_DOUBLE_FLOAT128 may be defined. In that
   * case the mapping of "long doulbe" is determined by xbit.
   */

  assert(sizeof rtemps == RTEMPS * sizeof *rtemps,
         "mkrtemp_init: rtemps[] size inconsistent with RTEMPS value", RTEMPS,
         ERR_Severe);
  for (i = 0; i < RTEMPS; i++) {
    rtemps[i].current = rtemps[i].start;
  }
}

/**
   \brief This routine makes a temporary and returns its symbol table index. The
    rtemp entry which is used is determined by the type of the ILI which needs a
    temporary to store its value
*/
SPTR
mkrtemp(int ilix)
{
  return mkrtemp_sc(ilix, SC_AUTO);
}

/**
   \brief This routine makes a temporary and returns its symbol table index. The
    rtemp entry which is used is determined by the type of the ILI which needs a
    temporary to store its value
*/
SPTR
mkrtemp_sc(int ilix, SC_KIND sc)
{
  int index, type;
  SPTR sym;

  type = select_rtemp(ilix);
  if ((index = rtemps[type].current++) > rtemps[type].max)
    rtemps[type].max = index;

  sym = getccsym_sc(rtemps[type].prefix, index, ST_VAR, sc);
  DTYPEP(sym, rtemps[type].dt);
#ifdef NOCONFLICTP
  NOCONFLICTP(sym, 1);
#endif
#ifdef PTRSAFEP
  PTRSAFEP(sym, 1);
#endif

  return sym;
}

/**
   \brief This routine makes a complex temporary and returns its symbol table
    index. The rtemp entry which is used is determined by the dtype.

    Also used for INTEGER*8 support to allocate 64-bit temporaries for 64-bit
    int argument expressions.
 */
SPTR
mkrtemp_cpx(DTYPE dtype)
{
  SPTR sym;
  sym = mkrtemp_cpx_sc(dtype, SC_AUTO);
#ifdef NOCONFLICTP
  NOCONFLICTP(sym, 1);
#endif
#ifdef PTRSAFEP
  PTRSAFEP(sym, 1);
#endif
  return sym;
}

SPTR
mkrtemp_cpx_sc(DTYPE dtype, SC_KIND sc)
{
  int index, type;
  SPTR sym;

  switch (dtype) {
  case DT_CMPLX:
    type = 5;
    break;
  case DT_DCMPLX:
    type = 6;
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case DT_CMPLX128:
    type = 10;
    break;
#endif
  case DT_INT8:
    type = 4;
    break;
  default:
    interr("mkrtemp_cpx: illegal dtype", dtype, ERR_Severe);
    type = 6;
  }

  if ((index = rtemps[type].current++) > rtemps[type].max)
    rtemps[type].max = index;

  sym = getccsym_sc(rtemps[type].prefix, index, ST_VAR, sc);
  DTYPEP(sym, rtemps[type].dt);
#ifdef NOCONFLICTP
  NOCONFLICTP(sym, 1);
#endif
#ifdef PTRSAFEP
  PTRSAFEP(sym, 1);
#endif

  return sym;
}

SPTR
mkrtemp_arg1_sc(DTYPE dtype, SC_KIND sc)
{
#ifndef ARG1PTRP
  return mkrtemp_cpx_sc(dtype, sc);
#else
  int index, type;
  SPTR sym;

  if (dtype == DT_CMPLX)
    type = 5;
  else if (dtype == DT_DCMPLX)
    type = 6;
#ifdef TARGET_SUPPORTS_QUADFP
  else if (dtype == DT_QCMPLX)
    type = 10;
#endif
#ifdef LONG_DOUBLE_FLOAT128
  else if (dtype == DT_CMPLX128)
    type = 6;
#endif
  else if (dtype == DT_INT8)
    type = 4;
  else {
    interr("mkrtemp_cpx: illegal dtype", dtype, ERR_Severe);
    type = 6;
  }

  if ((index = rtemps[type].current++) > rtemps[type].max)
    rtemps[type].max = index;

  sym = getccssym_sc(rtemps[type].arg1pfx, index, ST_VAR, sc);
  DTYPEP(sym, rtemps[type].dt);
#ifdef NOCONFLICTP
  NOCONFLICTP(sym, 1);
#endif
#ifdef PTRSAFEP
  PTRSAFEP(sym, 1);
#endif
  ARG1PTRP(sym, 1);

  return sym;
#endif
}

/**
   \brief This routine ends the temp processing for a function. Values are
   updated so that temps created for the next function are unique from the
   temps created during the current function
 */
void
mkrtemp_end(void)
{
  int i;

  for (i = 0; i < RTEMPS; i++) {
    rtemps[i].start = rtemps[i].max + 1;
  }
}

/**
   \brief This routine copies the maximum values of the temporaries currently
   allocated -- this is typically used at the start of optimizing a function.
*/
void
mkrtemp_copy(int *rt)
{
  int i;

  for (i = 0; i < RTEMPS; i++) {
    rt[i] = rtemps[i].max;
  }
}

/**
   \brief This routine keeps track of the maximum values of rtemps in rt
 */
void
mkrtemp_update(int *rt)
{
  int i;

  for (i = 0; i < RTEMPS; i++) {
    if (rt[i] < rtemps[i].max)
      rt[i] = rtemps[i].max;
  }
}

/**
   \brief This routine resets the rtemps values from rt.
 */
void
mkrtemp_reinit(int *rt)
{
  int i;

  for (i = 0; i < RTEMPS; i++) {
    rtemps[i].max = rt[i];
    rtemps[i].current = rtemps[i].start = rtemps[i].max + 1;
  }
}

int
assn_rtemp(int ili)
{
  int temp;

  temp = assn_rtemp_sc(ili, SC_AUTO);
  return (temp);
}

static int
_assn_rtemp(int ili, int temp)
{
  int rcand, rtype = 0;
  ILI_OP opc = ILI_OPC(ili);
  GET_RCAND(rcand);

  switch (opc) {
  default:
    break;
  case IL_STA:
    opc = IL_LDA;
    break;
  case IL_ST:
    opc = IL_LD;
    break;
  case IL_STKR:
    opc = IL_LDKR;
    break;
  case IL_STSP:
    opc = IL_LDSP;
    break;
  case IL_STDP:
    opc = IL_LDDP;
    break;
  case IL_VST:
    opc = IL_VLD;
    break;
  case IL_VSTU:
    opc = IL_VLDU;
    break;
  case IL_STSCMPLX:
    opc = IL_LDSCMPLX;
    break;
  case IL_STDCMPLX:
    opc = IL_LDDCMPLX;
    break;
  }

  switch (IL_RES(opc)) {
  case ILIA_IR:
    rtype = RCAND_RTYPE(rcand) = RATA_IR;
    RCAND_MSIZE(rcand) = MSZ_WORD;
    break;
  case ILIA_AR:
    rtype = RCAND_RTYPE(rcand) = RATA_AR;
    RCAND_MSIZE(rcand) = MSZ_PTR;
    break;
  case ILIA_SP:
    rtype = RCAND_RTYPE(rcand) = RATA_SP;
    RCAND_MSIZE(rcand) = MSZ_F4;
    break;
  case ILIA_DP:
    rtype = RCAND_RTYPE(rcand) = RATA_DP;
    RCAND_MSIZE(rcand) = MSZ_F8;
    break;
  case ILIA_KR:
    rtype = RCAND_RTYPE(rcand) = RATA_KR;
    RCAND_MSIZE(rcand) = MSZ_I8;
    break;
#ifdef ILIA_CS
  case ILIA_CS:
    rtype = RCAND_RTYPE(rcand) = RATA_CSP;
    RCAND_MSIZE(rcand) = MSZ_F8;
    break;
#endif
#ifdef ILIA_CD
  case ILIA_CD:
    rtype = RCAND_RTYPE(rcand) = RATA_CDP;
    RCAND_MSIZE(rcand) = MSZ_F16;
    break;
#endif
  case ILIA_LNK:
    if (IL_VECT(ILI_OPC(ili))) {
      RCAND_MSIZE(rcand) = ili_get_vect_dtype(ili);
      rtype = RCAND_RTYPE(rcand) = RATA_VECT;
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    interr("_assn_rtemp: illegal ili for temp assn", ili, ERR_Severe);
  }
  RCAND_NEXT(rcand) = reg[rtype].rcand;
  reg[rtype].rcand = rcand;
  RCAND_TEMP(rcand) = temp;
  RCAND_ATYPE(rcand) = RATA_TEMP;
  RCAND_COUNT(rcand) = 0;
  RCAND_CONFL(rcand) = 0;
  RCAND_VAL(rcand) = ili;
  ILI_RAT(ili) = rcand;

  return temp;
}

void
assn_input_rtemp(int ili, int temp)
{
  _assn_rtemp(ili, temp);
}

int
assn_rtemp_sc(int ili, SC_KIND sc)
{
  int temp;
  temp = mkrtemp_sc(ili, sc);
  return _assn_rtemp(ili, temp);
}

int
assn_sclrtemp(int ili, SC_KIND sc)
{
  int temp;
  int type;
  DTYPE dtype;
  int retry;
  char name[16];

  type = select_rtemp(ili);
  dtype = rtemps[type].dt;

  if (sc != SC_PRIVATE)
    snprintf(name, sizeof(name), ".r%04d", ili); /* at least 4, could be more */
  else
    snprintf(name, sizeof(name), ".r%04dp",
             ili); /* at least 4, could be more */
  retry = 0;
again:
  temp = getsymbol(name);
  if (STYPEG(temp) == ST_VAR) {
    if (DTYPEG(temp) != dtype) {
      if (sc != SC_PRIVATE)
        snprintf(name, sizeof(name), ".r%d%04d", retry, ili);
      else
        snprintf(name, sizeof(name), ".r%d%04dp", retry, ili);
      retry++;
      goto again;
    }
  } else {
    STYPEP(temp, ST_VAR);
    CCSYMP(temp, 1);
    LSCOPEP(temp, 1);
#ifdef PTRSAFEP
    PTRSAFEP(temp, 1);
#endif
    SCP(temp, sc);
    DTYPEP(temp, dtype);
  }

  return _assn_rtemp(ili, temp);
}

static int
select_rtemp(int ili)
{
  int type;

  switch (IL_RES(ILI_OPC(ili))) {
  case ILIA_IR:
    type = 0;
    break;
  case ILIA_AR:
    type = 1;
    break;
  case ILIA_SP:
    type = 2;
    break;
  case ILIA_DP:
    type = 3;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case ILIA_QP:
    type = 9;
    break;
#endif
  case ILIA_KR:
    type = 4;
    break;
  case ILIA_LNK:
    if (IL_VECT(ILI_OPC(ili))) {
      DTYPE dt = ili_get_vect_dtype(ili);
      if (dt) {
        type = 8;
        rtemps[type].dt = dt;
        break;
      }
    }
    FLANG_FALLTHROUGH;
#ifdef ILIA_CS
  case ILIA_CS:
    type = 5;
    break;
#endif
#ifdef ILIA_CD
  case ILIA_CD:
    type = 6;
    break;
#endif
#ifdef TARGET_SUPPORTS_QUADFP
  case ILIA_CQ:
    type = 10;
    break;
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case ILIA_FLOAT128:
    type = 9;
    break;
#endif
  default:
    interr("select_rtemp: bad ili", ili, ERR_Severe);
    type = 0;
  }
  return type;
}
