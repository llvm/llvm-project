/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Register allocation performed by the expander at opt level 1
 */

#include "expreg.h"
#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "ili.h"
#include "expand.h"
#include "regutil.h"
#include "machreg.h"
#include "machar.h"

static int ilt;

static LST *list; /* linked list of ili added to entry blk */

static void assign1(int rtype);

/** \brief Add register candidate
 *
 * routine to add candidates to the register candidate lists of
 * the register module.  This routine is valid for opt 1 only.
 * This routine is called only for load and store ILI and will
 * eliminate a candidate which depend on their types, storage
 * class, etc.
 */
void
exp_rcand(int ilix, int nmex)
{
  register int sym;

  if (NME_TYPE(nmex) == NT_VAR                         /* var names only */
      && TY_ISSCALAR(DTY(DTYPEG(sym = NME_SYM(nmex)))) /* scalars only */
      && IS_LCL_OR_DUM(sym)                            /* autos, args or regs */
      && !VOLG(sym)                                    /* not volatile */
      && SOCPTRG(sym) == 0 /* not equivalenced */
      ) {
    addrcand(ilix);
    /* If a variable is loaded or stored in a critical section or a
     * parallel section, set the candidate's IGNORE flag.  This prevents
     * a register from being assigned.
     * Tprs 2077 & 2127 - can't assign registers to variables in a
     * parallel region (could restrict this to shared variables).
     */
    if (NME_RAT(nmex) &&
        (bihb.csfg || bihb.parsectfg || bihb.parfg || bihb.taskfg))
      RCAND_IGNORE(NME_RAT(nmex)) = 1;
  }
}

/***********************************************************************/

/** \brief Control the assignment of the registers at opt level 1
 */
void
reg_assign1(void)
{
  int first_rat, /* first entry of the block's RAT  */
      entr;      /* entry symbol of the function */
  int null_rat;
  int funcbih;
  int iltt; /* local tmp for ilt */
  int sym;
  LST *save;
  int tbih;

  funcbih = BIHNUMG(gbl.currsub);
  rdilts(funcbih);           /* fetch the entry block */
  entr = BIH_LABEL(funcbih); /* get the entry symbol */
  list = 0;
  ilt = ILT_PREV(0); /* locate the last ilt in the block */

#if DEBUG
  if (EXPDBG(8, 12))
    fprintf(gbl.dbgfil,
            "\n***** Register Information for Function \"%s\" *****\n",
            getprint(entr));
  if (EXPDBG(8, 4))
    dmprcand();
#endif

  GET_RAT(first_rat); /* fetch a RAT item which will be
                       * the header of the table
                       */
  /*
   * assign the data registers from the allowed register set.
   */
  assign1(RATA_IR);
  /*
   * assign the address registers from the allowed set.
   */
  assign1(RATA_AR);
  /*
   * assign the single precision registers from the allowed set.
   */
  assign1(RATA_SP);
  /*
   * assign the double precision registers from the allowed set.
   */
  assign1(RATA_DP);
  /*
   * assign the double integer registers from the allowed set.
   */
  assign1(RATA_KR);
  /*
   * communicate to the scheduler the first global register assigned
   * for each register class -- note that this will be the physical register
   * number; it reflects the number of registers assigned from the physical
   * set mapped from the generic register set. Because two or more generic
   * register sets can map to a single register set, this information
   * can only be computed after all of the assignments are done.
   *
   */
  mr_end();
  /*
   * fill in the VAL field of the header entry for this RAT with
   * the number of assignments.
   */
  RAT_VAL(first_rat) = ratb.stg_avail - first_rat - 1;

#if DEBUG
  if (EXPDBG(8, 8)) {
    dmprat(first_rat);
    fprintf(gbl.dbgfil,
            "   first_dr:%3d, first_ar:%3d, first_sp:%3d, first_dp:%3d\n",
            aux.curr_entry->first_dr, aux.curr_entry->first_ar,
            aux.curr_entry->first_sp, aux.curr_entry->first_dp);
  }
#endif
/*
 * link the BIH of the entry block to the RAT.  this block is written
 * out.
 */
  BIH_ASSN(BIH_NEXT(funcbih)) = first_rat;
  wrilts(funcbih);
  /*
   * add ilis that were added to the main entry to any other entries
   */
  if (list) {
    save = list;
    for (sym = SYMLKG(gbl.entries); sym != NOSYM; sym = SYMLKG(sym)) {
      rdilts(tbih = BIHNUMG(sym));
      GET_RAT(null_rat);
      RAT_VAL(null_rat) = 0;
      BIH_ASSN(tbih) = null_rat;
      /*
       * if we copied params, we do this mvxx stuff in block after the
       * entry block, so scheduler loads up dummies after copy to temp
       * param list has been done
       */
      if (COPYPRMSG(sym)) {
        wrilts(tbih);
        tbih = exp_addbih(tbih);
        rdilts(tbih);
        BIH_ASSN(tbih) = null_rat;
      }
      iltt = ILT_PREV(0);
      list = save;
      while (list) {
        iltt = addilt(iltt, list->item);
        list = list->next;
      }
      BIH_ASSN(BIH_NEXT(tbih)) = first_rat;
      wrilts(tbih);
    }
    freearea(LST_AREA);
  }
  /*
   * dummies that have stores into  and that are assigned registers , must
   * be stored back at exit
   */
  storedums(expb.curbih, first_rat);

  /* for the routine's exit block, assign a null rat table  */
  GET_RAT(null_rat);
  RAT_VAL(null_rat) = 0;
  BIH_ASSN(expb.curbih) = null_rat;

  /*
   * go through the candidate lists to clean up the back pointers.
   * also, reset the register candidate store area.
   */
  endrcand();

}

/** \brief Assign registers for opt level 1.
 *
 * The possible set of registers to use satisfies the relation first_global <=
 * reg <= last_global, where the bounds are found in the mach_reg structure of
 * the reg structure for the given register type.  The registers are assigned
 * beginning at last_global and continues down to first_global.  The value
 * returned by the function is the number of registers assigned.
 *
 * \param rtype type of registers
 */
static void
assign1(int rtype)
{
  int cand, /* candidate for a register */
      rat,  /* RAT for each assigned reg */
      ilix, /* ili for the candidate */
      areg; /* assigned register */

  int candl,  /* candidate list */
      nme,    /* NME for the candidate */
      sym,    /* ST for the candidate  */
      addr,   /* address ili for the candidate */
      ilitmp; /* ili temporary */

  /*
   * loop while there are candidates and registers available
   */
  candl = reg[rtype].rcand;
  for (;;) {
    /*
     * get a candidate; if one does not exist, exit the loop
     */
    if ((cand = getrcand(candl)) == 0)
      break;
    if (RCAND_IGNORE(cand))
      continue;
    if (!RCAND_ISILI(cand)) {
/*
 * if the candidate is not type ILI (implies that just stores
 * occurred, get the next candidate if it's not a regarg.
 * NOTE: C only.
 */
      continue;
    } else {
      /*
       * for the candidate, locate the load ili (ilix), the names entry
       * (nme), and the symbol (sym)
       */
      ilix = RCAND_VAL(cand);
      sym = NME_SYM(nme = ILI_OPND(ilix, 2));
      addr = ILI_OPND(ilix, 1);
    }
    /*
     * if the symbol has had its address taken, it cannot be assigned a
     * register
     */
    if (ADDRTKNG(sym))
      continue;
    if (gbl.internal > 1 && sym == aux.curr_entry->display) {
      /* printf("display temp %s cannot be assigned register\n",
             SYMNAME(sym));
       */
      continue;
    }
/*
 * if symbol is declared in an 'outer' scope, it cannot be assigned a
 * register - happens in a language with nested procedures (c++)
 */
    if (GSCOPEG(sym))
      continue;
    if (UPLEVELG(sym))
      continue;

    /*
     *for now, dissallow dummies if there are multiple entries
     */
    if (IS_DUM(sym)) {
      if (SYMLKG(gbl.entries) != NOSYM)
        continue;
    } else if (SCG(sym) == SC_LOCAL && DINITG(sym) &&
               (ADDRTKNG(sym) || RCAND_STORE(cand)) && gbl.rutype != RU_PROG)
      /*
       * if a local variable is data initialized, disallow it if it
       * has been stored; the thinking is that it falls into the
       * same category of a saved variable -- someday, may want
       * to override this if XBIT(124,0x80) is set (also optutil.c)
       */
      continue;
    /*
     * for C regargs, use arg register saved in address field;
     * for ftn use arg register saved in address field if it's the
     * symbol which represents the arg's address.  Otherwise, get a
     * register from the "global" set.
     */
    if (IS_REGARG(sym))
      areg = ADDRESSG(sym);
    else {
      /*
       * get a register from the global set
       */
      if ((areg = mr_getreg(rtype)) == NO_REG)
        break;
    }

    GET_RAT(rat) /* locate a rat entry */

    RAT_REG(rat) = ad1ili(RTYPE_DF(rtype), areg);

    RAT_RTYPE(rat) = rtype;             /* don't forget its rtype */
    RAT_ATYPE(rat) = RATA_NME;          /* opt 1 assigns nme's only */
    RAT_ADDR(rat) = addr;               /* locate addr ili of ref   */
    RAT_CONFL(rat) = 0;                 /* no conflict for opt 1    */
    RAT_STORE(rat) = RCAND_STORE(cand); /* copy up store flag       */
    RAT_VAL(rat) = nme;                 /* record nme assigned      */
    RAT_MSIZE(rat) = RCAND_MSIZE(cand); /* copy up mem size         */
                                        /*
                                         * if the symbol is an argument to the function, add a move ili of
                                         * the argument to its assigned register
                                         */
    /* for fortran, a REGARG sym has its address in the register, not
     * the value of the argument.  REDUC flag indicates that no extra
     * indirection is needed.
     */
    if (IS_DUM(sym)) {
      if (REGARGG(sym)) {
        if (!REDUCG(sym)) {
          /* we're preloading the dummy argument; we need to use
           * the arg register that's assigned to its address as the
           * address expression in the the load.
           */
          addr = ad1ili(IL_ARDF, ADDRESSG(sym));
          if (flg.endian &&
              (DTYPEG(sym) == DT_INT8 || DTYPEG(sym) == DT_LOG8)) {
            if (!XBIT(124, 0x400))
              /* 32bits of significance in 64 bits */
              addr = ad3ili(IL_AADD, addr, ad_aconi(4), 0);
          }
          RAT_ADDR(rat) = addr; /* switch addr expr */
          ilix = ad3ili(ILI_OPC(ilix), addr, ILI_OPND(ilix, 2),
                        ILI_OPND(ilix, 3));
          ilt = addilt(ilt, ilitmp = ad2ili(MV_RTYPE(rtype), ilix, areg));
          ADDNODE(list, ilitmp);
        }
      } else {
        ilt = addilt(ilt, ilitmp = ad2ili(MV_RTYPE(rtype), ilix, areg));
        ADDNODE(list, ilitmp);
      }
    } else {
      /*
       * FORTRAN only - if the symbol is local to the function and has
       * been data initialized, add a move ili of the argument to its
       * assigned register
       */
      if (DINITG(sym)) {
        ilt = addilt(ilt, ilitmp = ad2ili(MV_RTYPE(rtype), ilix, areg));
        ADDNODE(list, ilitmp);
      }
    }
  }
}
