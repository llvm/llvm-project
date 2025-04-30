/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Common expander routines
 */

#define EXPANDER_DECLARE_INTERNAL
#include "expand.h"
#include "exputil.h"
#include "exp_ftn.h"
#include "expatomics.h"
#include "expreg.h"
#include "expsmp.h"
#include "error.h"
#include "regutil.h"
#include "machreg.h"
#include "fih.h"
#include "ilmtp.h"
#include "ilm.h"
#include "ili.h"
#include "machar.h"
#include "scope.h"
#include "llassem.h"
#include "outliner.h"
#include "verify.h"
#include "ccffinfo.h"
#include "ilidir.h"
#include "exp_rte.h"
#include "dtypeutl.h"
#include "symfun.h"
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
#include "ompaccel.h"
#endif
#ifdef OMP_OFFLOAD_LLVM
#include "tgtutil.h"
#include "kmpcutil.h"
#endif
extern int in_extract_inline; /* Bottom-up auto-inlining */

static int efunc(const char *);
static int create_ref(SPTR sym, int *pnmex, int basenm, int baseilix,
                      int *pclen, int *pmxlen, int *prestype);
static int jsr2qjsr(int);

#define DO_PFO ((XBIT(148, 0x1000) && !XBIT(148, 0x4000)) || XBIT(148, 1))

/***************************************************************/

/*
 * Initialize global data structures
 */
void
ds_init(void)
{
  int i;
  ili_init();
  ilt_init();
  bih_init();
  nme_init();

  /*
   * allocate the register areas for use by the expander or the optimizer
   */
  i = 128;
  EXP_ALLOC(rcandb, RCAND, i);
  BZERO(&rcandb.stg_base[0], RCAND, i); /* purify umr when cand = 0 */
  rcandb.stg_avail = 1;
  rcandb.weight = 1;
  rcandb.kr = 0;
  EXP_ALLOC(ratb, RAT, i);
  ratb.stg_avail = 1;
  EXP_ALLOC(rgsetb, RGSET, i);
  BZERO(&rgsetb.stg_base[0], RGSET, i);
  rgsetb.stg_avail = 1;

} /* ds_init */

void
exp_init(void)
{
  /*
   * Allocate the space necessary to hold the auxiliary information for ILM
   * evaluation required by the expander.  If necessary the space could
   * depend on sem.ilm_size, but this is probably too much.  The ilm index
   * i is associated with the ith entry in this area (there will items that
   * are not used).  The following size is probably sufficient but a check
   * will be done each time rdilms is called.
   */
  EXP_ALLOC(expb.ilmb, ILM_AUX, 610);
  expb.flags.wd = 0;

  expb.gentmps = 0; /* PGC: counter increments across functions */
  expb.str_avail = 0;
  if (expb.str_size == 0) {
    expb.str_size = 32;
    NEW(expb.str_base, STRDESC, expb.str_size);
  }
  expb.logcjmp = XBIT(125, 0x8) ? IL_ICJMPZ : IL_LCJMPZ;
  aux.curr_entry->display = SPTR_NULL;

  ds_init();
  expb.curilt = 0;
  expb.curbih = 0;
  expb.isguarded = 0;
  expb.flags.bits.noblock = 1;
  expb.flags.bits.noheader = 1;
  if (CHARLEN_64BIT)
    expb.charlen_dtype = DT_INT8;
  else
    expb.charlen_dtype = DT_INT;

  if (flg.xon != 0 || flg.xoff ^ 0xFFFFFFFF)
    expb.flags.bits.excstat = 1;

  /* For C, only rewind the ilm file once (performed by main()) */
  rewindilms();

  /* set threshold of # of ilm words, if exceeded, to break ili blocks */

  if (flg.x[100])
    expb.ilm_thresh = 1 << (flg.x[100] & 0x1f);
  else {
#ifdef TM_ILM_THRESH
    expb.ilm_thresh = TM_ILM_THRESH;
    if (flg.opt >= 3 || flg.vect & 16)
      expb.ilm_thresh += TM_ILM_THRESH >> 1; /* allow for 50% more */
#else
    expb.ilm_thresh = 1 << 30; /* BIG */
#endif
  }
  expb.sc = SC_AUTO; /* default storage class for expander-created temps */
  exp_smp_init();
  expb.clobber_ir = expb.clobber_pr = 0;
}

/*
 * clean up allocated space when the program isn't compiled
 */
void
exp_cleanup(void)
{
  if (rgsetb.stg_base)
    EXP_FREE(rgsetb);
  rgsetb.stg_base = NULL;
  if (rcandb.stg_base) {
    EXP_FREE(rcandb);
  }
  rcandb.stg_base = NULL;
  if (ratb.stg_base)
    EXP_FREE(ratb);
  ratb.stg_base = NULL;
} /* exp_cleanup */

/*
 * Parse an IM_FILE ilm.
 *
 * - ilmp is an IM_FILE ilm.
 * - lineno_out becomes the line number, but only if the IM_FILE has a non-zero
 *   lineno operand. Otherwise, lineno_out is not touched.
 * - findex_out becomes a valid index into the FIH table.
 * - ftag_out becomes the ftag.
 */
static void
parse_im_file(const ILM *ilmp, int *lineno_out, int *findex_out, int *ftag_out)
{
  /* IM_FILE lineno findex ftag */
  int lineno = ILM_OPND(ilmp, 1);
  int findex = ILM_OPND(ilmp, 2);
  int ftag = ILM_OPND(ilmp, 3);

  assert(ILM_OPC(ilmp) == IM_FILE, "parse_im_file: Expected IM_FILE",
         ILM_OPC(ilmp), ERR_Fatal);

  /* The bottom-up inliner will generate some IM_FILE ilms with findex
   * operands that reference the IFIH table. These references are encoded as
   * negative numbers. Translate them back to FIH references here. */
  if (findex < 0) {
    int ifindex = -findex - 1;
    assert(ifindex < ifihb.stg_avail,
           "parse_im_file: Invalid IFIH reference on IM_FILE", ifindex,
           ERR_Warning);
    findex = IFIH_FINDEX(ifindex);
  }

  assert(findex < fihb.stg_avail,
         "parse_im_file: Invalid FIH reference on IM_FILE", findex,
         ERR_Warning);

  if (lineno_out && lineno)
    *lineno_out = lineno;
  if (findex_out)
    *findex_out = findex;
  if (ftag_out)
    *ftag_out = ftag;
}

/***************************************************************/

/** \brief Expand ILMs to ILIs */
int
expand(void)
{
  int ilmx;       /* index of the ILM		 */
  int len;        /* length of the ILM		 */
  ILM *ilmp;      /* absolute pointer to the ILM */
  ILM_OP opc;     /* opcode of the ILM		 */
  int last_label_bih = 0;
  int last_ftag = 0;
  int nextftag = 0, nextfindex = 0;
  int last_cpp_branch = 0;

  /*
   * NOTE, for an ILM: ilmx is needed to access the ILM_AUX area, ilmp is
   * needed to access the ILM area
   */
  exp_init();
  /* During expand, we want to generate unique proc ili each time a
   * proc ILM is processed.  The assumption is that the scheduler will
   * cse a proc ili if it appears multiple times in a block. E.g.,
   *    COMPLEX  c(10)
   *    x = f() + f()     ! two ili for calling f
   *    c(ifunc()) = ...  ! 1 call to ifunc (although two uses)
   * After expand, we share proc ili; the optimizer may create expressions
   * which contain calls where the intent is to cse a call if it already
   * exists in the block.
   */
  share_proc_ili = false;

  if (!XBIT(120, 0x4000000)) {
    set_allfiles(0);
  } else {
    gbl.findex = 1;
  }

  /*
   * process all blocks for a function. For Fortran, the terminating
   * condition is when the "end" ILM is seen (there may be multiple
   * subprograms per compilation -- the ilm file is reused). For C,
   * the ilm file contains the blocks for all function.  The loop
   * terminates when the "end" ILM is seen and a non-zero value is
   * returned; if the ilm file is at end-of-file, 0 is returned.
   */
  do {
    expb.nilms = rdilms();
    nextftag = fihb.nextftag;
    nextfindex = fihb.nextfindex;
#if DEBUG
    if (DBGBIT(4, 0x800))
      dumpilms();
#endif
    DEBUG_ASSERT(expb.nilms, "expand:ilm end of file");
    /*
     * the following check could be deleted if the max ilm block size is
     * known or if space doesn't have to be conserved during this phase
     */
    if (expb.nilms > expb.ilmb.stg_size) {
      EXP_MORE(expb.ilmb, ILM_AUX, expb.nilms + 100);
    }

      /* scan through all the ilms in the current ILM block */

    for (ilmx = 0; ilmx < expb.nilms; ilmx += len) {
      int saved_curbih = expb.curbih;
      int saved_findex = fihb.nextfindex;
      bool followed_by_file = false;
      bool ilmx_is_block_label = false;
      int findex, ftag;

      /* the first time an ilm is seen, it has no result  */

      ILM_RESULT(ilmx) = 0;
      ILM_EXPANDED_FOR(ilmx) = 0;

      ILM_RESTYPE(ilmx) = 0; /* zero out result types */
      ILM_NME(ilmx) = 0;     /* zero out name entry (?) */
      findex = 0;
      ftag = 0;

      ilmp = (ILM *)(ilmb.ilm_base + ilmx);
      opc = ILM_OPC(ilmp);

      if (opc == IM_BR) {
        last_cpp_branch = ILM_OPND(ilmp, 1);
      } else if (opc == IM_LABEL) {
        /* Scope labels don't cause block breaks. */
        ilmx_is_block_label = !is_scope_label(ILM_OPND(ilmp, 1));
        if (!ilmx_is_block_label) {
          new_callee_scope = ENCLFUNCG(ILM_OPND(ilmp, 1));
        }
      }

      DEBUG_ASSERT(opc > 0 && opc < N_ILM, "expand: bad ilm");
      len = ilms[opc].oprs + 1; /* length is number of words for the
                                 * fixed operands and the opcode */
      if (IM_VAR(opc))
        len += ILM_OPND(ilmp, 1); /* include the number of
                                   * variable operands */
      if (IM_TRM(opc)) {
        eval_ilm(ilmx);
      }
      else if (flg.smp && len) {
        ll_rewrite_ilms(-1, ilmx, len);
      }

      if (opc != IM_FILE) {
        ++nextftag;
        fihb.nextftag = nextftag;
      } else if ((XBIT(148, 0x1) || XBIT(148, 0x1000)) && !followed_by_file) {
        int ftag;
        int findex;
        parse_im_file((ILM *)&ilmb.ilm_base[ilmx], NULL, &findex, &ftag);
        if (ftag) {
          nextfindex = findex;
          nextftag = ftag;
          fihb.nextfindex = nextfindex;
          fihb.nextftag = nextftag;
        }
      }

      /* If a new bih is created, detect certain scenarios */

      if (expb.curbih > saved_curbih) {

        /* Pay special attention to the transition from inlinee to inliner.
         * If last bih (in the inlinee) is created by an IM_LABEL followed
         * by an IM_FILE, we need to honor the ftag info in the IM_FILE.
         */

        if ((saved_curbih != 0) && (saved_curbih == last_label_bih) &&
            (saved_findex > fihb.nextfindex))
          BIH_FTAG(last_label_bih) = last_ftag;

        /* Flag the scenario that the new bih is created by an IM_LABEL that is
         * followed by an IM_FILE.
         */

        if (ilmx_is_block_label && followed_by_file) {
          last_label_bih = expb.curbih;
          last_ftag = ftag;
        }
      }
    } /* end of loop through ILM block  */

    new_callee_scope = 0;
  }
  while (opc != IM_END && opc != IM_ENDF);

  if (DBGBIT(10, 2) && (bihb.stg_avail != 1)) {
    int bih;
    for (bih = 1; bih != 0; bih = BIH_NEXT(bih)) {
      if (BIH_EN(bih))
        dump_blocks(gbl.dbgfil, bih, "***** BIHs for Function \"%s\" *****", 1);
    }
    dmpili();
  }
#if DEBUG
  verify_function_ili(VERIFY_ILI_DEEP);
  if (DBGBIT(10, 16)) {
    dmpnme();
    {
      int i, j;
      for (i = nmeb.stg_avail - 1; i >= 2; i--) {
        for (j = nmeb.stg_avail - 1; j >= 2; j--) {
          if (i != j)
            (void)conflict(i, j);
        }
      }
    }
  }
  if (DBGBIT(8, 64)) {
    fprintf(gbl.dbgfil, "  ILM(%d)", expb.ilmb.stg_size);
    fprintf(gbl.dbgfil, "  ILI(%d)", ilib.stg_avail);
    fprintf(gbl.dbgfil, "  ILT(%d)", iltb.stg_size);
    fprintf(gbl.dbgfil, "  BIH(%d)", bihb.stg_size);
    fprintf(gbl.dbgfil, "  NME(%d)\n", nmeb.stg_avail);
  }
#endif

  ili_lpprg_init();
  /* for C, we don't free the ilm area until we reach end-of-file */
  FREE(ilmb.ilm_base);
  ilmb.ilm_base = NULL;
  EXP_FREE(expb.ilmb);
  freearea(STR_AREA);
  if (flg.opt < 2) {
    if (rcandb.stg_base) {
      EXP_FREE(rcandb);
      rcandb.stg_base = NULL;
    }
  }
  share_proc_ili = true;
  exp_smp_fini();
  fihb.nextftag = fihb.currftag = 0;

  if (!XBIT(120, 0x4000000)) {
    /* Restored file indexes to where they were before expand in case
       they got changed somewhere.
     */
    set_allfiles(1);
  } else {
    fihb.nextfindex = fihb.currfindex = 1;
  }
  return expb.nilms;
}

/***************************************************************/

/*
 * Check that operand opr of ILM ilmx has been expanded.
 * If this will be the first use of this ILM, then set ILM_EXPANDED_FOR
 * to ilmx.
 */
static void
eval_ilm_argument1(int opr, ILM *ilmpx, int ilmx)
{
  int op1, ilix;
  if ((ilix = ILI_OF(op1 = ILM_OPND(ilmpx, opr))) == 0) {
    /* hasn't been evaluated yet */
    eval_ilm(op1);
    /* mark this as expanded for this ILM */
    ILM_EXPANDED_FOR(op1) = -ilmx;
  } else if (ILM_EXPANDED_FOR(op1) < 0 && !is_cseili_opcode(ILI_OPC(ilix))) {
    /* This was originally added for a parent ILM, so it hasn't
     * been used as an operand ILI yet.  Take ownership of it here.
     * When it is reused later for a parent ILM,
     * it will get then get turned into a CSE ILI */
    ILM_EXPANDED_FOR(op1) = -ilmx;
  }
} /* eval_ilm_argument1 */

void
eval_ilm(int ilmx)
{

  ILM *ilmpx;
  int noprs,   /* number of operands in the ILM	 */
      ilix,    /* ili index				 */
      tmp,     /* temporary				 */
      op1;     /* operand 1				 */
  ILM_OP opcx; /**< ILM opcode of the ILM */

  int first_op = 0;

  opcx = ILM_OPC(ilmpx = (ILM *)(ilmb.ilm_base + ilmx));

  if (flg.smp) {
    if (IM_TYPE(opcx) != IMTY_SMP && ll_rewrite_ilms(-1, ilmx, 0)) {
      if (ilmx == 0 && opcx == IM_BOS) {
        /* Set line no for EPARx */
        gbl.lineno = ILM_OPND(ilmpx, 1);
      }
      return;
    }
  }

  if (EXPDBG(8, 2))
    fprintf(gbl.dbgfil, "---------- eval ilm  %d\n", ilmx);

  if (!ll_ilm_is_rewriting())
  {
#ifdef OMP_OFFLOAD_LLVM
    if (flg.omptarget && gbl.ompaccel_intarget) {
      if (opcx == IM_MP_BREDUCTION) {
        ompaccel_notify_reduction(true);
        exp_ompaccel_reduction(ilmpx, ilmx);
      } else if (opcx == IM_MP_EREDUCTION) {
        ompaccel_notify_reduction(false);
        return;
      }

      if (ompaccel_is_reduction_region())
        return;
    }
#endif
    /*-
     * evaluate unevaluated "fixed" arguments:
     * For each operand which is a link to another ilm, recurse (evaluate it)
     * if not already evaluated
     */
    if (opcx == IM_DCMPLX || opcx == IM_CMPLX
#ifdef TARGET_SUPPORTS_QUADFP
        || opcx == IM_QCMPLX
#endif
       ) {
      for (tmp = 1, noprs = 1; noprs <= ilms[opcx].oprs; ++tmp, ++noprs) {
        if (IM_OPRFLAG(opcx, noprs) == OPR_LNK) {
          eval_ilm_argument1(noprs, ilmpx, ilmx);
        }
      }
    } else {
      for (tmp = 1, noprs = ilms[opcx].oprs; noprs > first_op; ++tmp, --noprs) {
        if (IM_OPRFLAG(opcx, noprs) == OPR_LNK) {
          eval_ilm_argument1(noprs, ilmpx, ilmx);
        }
      }
    }

    /* evaluate unevaluated "variable" arguments  */

    if (IM_VAR(opcx) && IM_OPRFLAG(opcx, ilms[opcx].oprs + 1) == OPR_LNK) {
      for (noprs = ILM_OPND(ilmpx, 1); noprs > 0; --noprs, ++tmp) {
        eval_ilm_argument1(tmp, ilmpx, ilmx);
      }
    }

    /*-
     * check the "fixed" arguments for any duplicated values
     */
    for (tmp = 1, noprs = ilms[opcx].oprs; noprs > first_op; ++tmp, --noprs) {
      if (IM_OPRFLAG(opcx, noprs) == OPR_LNK) {
        /* all arguments will have been evaluated by now */
        ilix = ILI_OF(op1 = ILM_OPND(ilmpx, noprs));
        if (ILM_EXPANDED_FOR(op1) == -ilmx) {
          ILM_EXPANDED_FOR(op1) = ilmx;
        } else if (ilix && ILM_EXPANDED_FOR(op1) != ilmx) {
          if (ILM_RESTYPE(op1) != ILM_ISCMPLX &&
              ILM_RESTYPE(op1) != ILM_ISDCMPLX
#ifdef LONG_DOUBLE_FLOAT128
              && ILM_RESTYPE(op1) != ILM_ISFLOAT128CMPLX
#endif
          )
            /* not complex */
            ILM_RESULT(op1) = check_ilm(op1, ilix);
          else {
            /* complex */
            ILM_RRESULT(op1) = check_ilm(op1, (int)ILM_RRESULT(op1));
            ILM_IRESULT(op1) = check_ilm(op1, (int)ILM_IRESULT(op1));
          }
        }
      }
    }

    /* check the "variable" arguments for any duplicated values  */

    if (IM_VAR(opcx) && IM_OPRFLAG(opcx, ilms[opcx].oprs + 1) == OPR_LNK) {
      for (noprs = ILM_OPND(ilmpx, 1); noprs > 0; --noprs, ++tmp) {
        ilix = ILI_OF(op1 = ILM_OPND(ilmpx, tmp));
        if (ILM_EXPANDED_FOR(op1) == -ilmx) {
          ILM_EXPANDED_FOR(op1) = ilmx;
        } else if (ilix && ILM_EXPANDED_FOR(op1) != ilmx) {
          if (ILM_RESTYPE(op1) != ILM_ISCMPLX &&
              ILM_RESTYPE(op1) != ILM_ISDCMPLX
#ifdef LONG_DOUBLE_FLOAT128
              && ILM_RESTYPE(op1) != ILM_ISFLOAT128CMPLX
#endif
          ) {
            /* not complex */
            ILM_RESULT(op1) = check_ilm(op1, ilix);
          } else {
            /* complex */
            ILM_RRESULT(op1) = check_ilm(op1, (int)ILM_RRESULT(op1));
            ILM_IRESULT(op1) = check_ilm(op1, (int)ILM_IRESULT(op1));
          }
        }
      }
    }
  }
  /*
   * ready to evaluate the ilm.  opcx is opcode of current ilm, ilmpx is
   * pointer to current ilm, and ilmx is index to the current ilm.
   */
  if (EXPDBG(8, 2))
    fprintf(gbl.dbgfil, "ilm %s, index %d, lineno %d\n", ilms[opcx].name, ilmx,
            gbl.lineno);

  if (!IM_SPEC(opcx))
  {
    /* expand the macro definition */
    tmp = exp_mac(opcx, ilmpx, ilmx);
    if (IM_I8(opcx))
      ILM_RESTYPE(ilmx) = ILM_ISI8;

    return;
  }
  switch (IM_TYPE(opcx)) { /* special-cased ILM		 */

  case IMTY_REF: /* reference  */
    exp_ref(opcx, ilmpx, ilmx);
    break;

  case IMTY_LOAD: /* load  */
    exp_load(opcx, ilmpx, ilmx);
    break;

  case IMTY_STORE: /* store  */
    exp_store(opcx, ilmpx, ilmx);
    break;

  case IMTY_BRANCH: /* branch  */
    exp_bran(opcx, ilmpx, ilmx);
    break;

  case IMTY_PROC: /* procedure  */
    exp_call(opcx, ilmpx, ilmx);
    break;

  case IMTY_INTR: /* intrinsic */
  case IMTY_ARTH: /* arithmetic  */
  case IMTY_CONS: /* constant  */
    exp_ac(opcx, ilmpx, ilmx);
    break;

  case IMTY_MISC: /* miscellaneous  */
    exp_misc(opcx, ilmpx, ilmx);
    break;

  case IMTY_FSTR: /* fortran string */
    exp_fstring(opcx, ilmpx, ilmx);
    break;

  case IMTY_SMP: /* smp ILMs  */
    exp_smp(opcx, ilmpx, ilmx);
    break;

  default: /* error */
    interr("eval_ilm: bad op type", IM_TYPE(opcx), ERR_Severe);
    break;
  } /* end of switch on ILM opc  */

#ifdef OMP_OFFLOAD_LLVM

  if (flg.omptarget && opcx == IM_ENLAB) {
    /* Enables creation of libomptarget related structs in the main function,
     * but it is not recommended option. Default behaviour is to initialize and
     * create them in the global constructor. */
    if (XBIT(232, 0x10)) {
      if (!ompaccel_is_tgt_registered() && !OMPACCRTG(gbl.currsub) &&
          !gbl.outlined) {
        ilix = ll_make_tgt_register_lib2();
        iltb.callfg = 1;
        chk_block(ilix);
        ompaccel_register_tgt();
      }
    }
    /* We do not initialize spmd kernel library since we do not use spmd data
     * sharing model. It does extra work and allocates device on-chip memory.
     * */
    if (XBIT(232, 0x40) && gbl.ompaccel_intarget) {
      ilix = ompaccel_nvvm_get(threadIdX);
      ilix = ll_make_kmpc_spmd_kernel_init(ilix);
      iltb.callfg = 1;
      chk_block(ilix);
    }
  }
#endif
  if (IM_I8(opcx))
    ILM_RESTYPE(ilmx) = ILM_ISI8;
}

/***************************************************************/
/*
 * An ESTMT ILM (or an ILI whose value is to be discarded) is processed by
 * walking the ILI tree (located by ilix) and creating ILTs for any function
 * calls that exist in the tree. This routine is similar to reduce_ilt
 * (iltutil.c) except that chk_block is used to add an ILT.  This is done so
 * that the "end of block" checks are performed.
 */
void
exp_estmt(int ilix)
{
  int noprs, i;

  ILI_OP opc = ILI_OPC(ilix);
  if (IL_TYPE(opc) == ILTY_PROC && opc >= IL_JSR) {
    iltb.callfg = 1; /* create an ILT for the function */
    chk_block(ilix);
  } else if (opc == IL_DFRDP && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    chk_block(ad1ili(IL_FREEDP, ilix));
  } else if (opc == IL_DFRSP && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    chk_block(ad1ili(IL_FREESP, ilix));
  } else if (opc == IL_DFRCS && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    chk_block(ad1ili(IL_FREECS, ilix));
  }
#ifdef LONG_DOUBLE_FLOAT128
  else if (opc == IL_FLOAT128RESULT && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    chk_block(ad1ili(IL_FLOAT128FREE, ilix));
  }
#endif
  else if (opc == IL_VA_ARG) {
    iltb.callfg = 1;
    chk_block(ilix);
  } else if (IL_HAS_FENCE(opc)) {
    chk_block(ad_free(ilix));
  } else {
    /* otherwise, walk all of the link operands of the ILI  */
    noprs = ilis[opc].oprs;
    for (i = 1; i <= noprs; ++i)
      if (IL_ISLINK(opc, i))
        exp_estmt((int)ILI_OPND(ilix, i));
  }
}

/***************************************************************/

/* Expand a scope label that should be inserted as an in-stream IL_LABEL ilt
 * instead of splitting the current block.
 *
 * These scope labels are generated by enter_lexical_block() and
 * exit_lexical_block(). They are verified by scope_verify().
 */
static void
exp_scope_label(int lbl)
{
  int ilt, ilix;

  /* Each scope label can only appear in one block. The ILIBLK field for the
   * label must point to the unique BIH containing the IL_LABEL ilt.
   */
  assert(ILIBLKG(lbl) == 0 || ISTASKDUPG(GBL_CURRFUNC),
         "Duplicate appearance of scope label", lbl, ERR_Severe);

  /* This IM_LABEL may have been created for a lexical scope that turned out
   * to not contain any variables. Such a label should simply be ignored. See
   * cancel_lexical_block(). */
  if (!ENCLFUNCG(lbl))
    return;

  ilix = ad1ili(IL_LABEL, lbl);

  /* Insert the label at the top of the current block instead of appending
   * it. Labels are not supposed to affect code generation, but they
   * interfere with the trailing branches in a block. We also have code which
   * expects the last three ilts in a block to follow a certain pattern for
   * indiction variable updates.
   *
   * Skip any existing labels at the beginning of the block so that multiple
   * labels appear in source order.
   *
   * The first and last ilts in the current block are stored in ILT_NEXT(0)
   * and ILT_PREV(0) respectively; BIH_ILTFIRST isn't up-to-date. See
   * wrilts().
   */
  ilt = ILT_NEXT(0);
  while (ilt && ILI_OPC(ILT_ILIP(ilt)) == IL_LABEL)
    ilt = ILT_NEXT(ilt);

  if (!ilt) {
    /* This block is all labels. Append the new label. */
    expb.curilt = addilt(expb.curilt, ilix);
  } else {
    /* Now, ilt is the first non-label ilt in the block.
     * Insert new label before ilt.
     * This also does the right thing when ILT_PREV(ilt) == 0.
     */
    addilt(ILT_PREV(ilt), ilix);
  }

  ILIBLKP(lbl, expb.curbih);
}

void
exp_label(SPTR lbl)
{
  int ilix; /* ili of an ilt	 */

  /* Handle in-stream labels by creating an IL_LABEL ilt. */
  if (is_scope_label(lbl)) {
    exp_scope_label(lbl);
    /* In-stream labels newer cause a new block to be created, so we're
     * done. */
    return;
  }

  if (expb.flags.bits.waitlbl) {
    /*
     * the current ilt points to a conditional branch. saveili locates an
     * unconditional branch. If the conditional label is lbl, then the
     * conditional is complemented whose label is changed to locate the
     * one specified in the unconditional. The unconditional ili is not
     * added.
     */
    expb.flags.bits.waitlbl = 0;
    ilix = ILT_ILIP(expb.curilt); /* conditional branch ili */

    if (expb.curilt && (ILI_OPND(ilix, ilis[ILI_OPC(ilix)].oprs)) == lbl) {
      ILT_ILIP(expb.curilt) = compl_br(ilix, (int)(ILI_OPND(expb.saveili, 1)));
      RFCNTD(lbl);
    } else {
      if (flg.opt != 1) {
        wr_block();
        cr_block();
      }
      expb.curilt = addilt(expb.curilt, expb.saveili);
    }
  }
  /*
   * check to see if the current ilt locates an ili which is a branch to
   * lbl  --  this only happens for opt levels other than 0.
   */
  if (flg.opt != 0 && ILT_BR(expb.curilt)) {
    ilix = ILT_ILIP(expb.curilt);
    if (ILI_OPND(ilix, ilis[ILI_OPC(ilix)].oprs) == lbl &&
        ILI_OPC(ilix) != IL_JMPA && ILI_OPC(ilix) != IL_JMPMK &&
        ILI_OPC(ilix) != IL_JMPM) {
      int curilt = expb.curilt;

      /*
       * delete the branch ilt  --  this may create ilts which locate
       * functions
       */
      if (EXPDBG(8, 32))
        fprintf(gbl.dbgfil,
                "---exp_label: deleting branch ili %d from block %d\n", ilix,
                expb.curbih);

      expb.curilt = ILT_PREV(curilt);
      ILT_NEXT(expb.curilt) = 0;
      ILT_PREV(0) = expb.curilt;
      STG_ADD_FREELIST(iltb, curilt);
      expb.curilt = reduce_ilt(expb.curilt, ilix);
      RFCNTD(lbl);
    }
  }
  /*-
   * finish off by checking lbl --
   * 1. If opt 0 is requested, the label will always begin a block
   *    if it is a user label.  NOTE that this covers the case when
   *    just -debug is specified (no -opt); if debug is requested along
   *    with a higher opt, we do not allow unreferenced labels to
   *    appear in the blocks since this can drastically affect code.
   *    WARNING:  coffasm needs to be follow these conventions --- see
   *    the Is_user_label macro in all versions of coffasm.c.
   *    KLUDGE:  for C blocks, labels are created -- their RFCNT's must
   *    be nonzero (set by semant).
   * 2. If the reference count is still non-zero, a new block is
   *    created labeled by lbl.
   */
  if (flg.opt == 0 && CCSYMG(lbl) == 0) {
    if (BIH_LABEL(expb.curbih) != 0 ||
        (expb.curilt != 0 && !ILT_DBGLINE(expb.curilt))) {
      wr_block();
      cr_block();
    }
    BIH_LABEL(expb.curbih) = lbl;
    ILIBLKP(lbl, expb.curbih);
    fihb.currftag = fihb.nextftag;
    fihb.currfindex = fihb.nextfindex;
  } else if (RFCNTG(lbl) != 0) {
    if (BIH_LABEL(expb.curbih) != 0 ||
        (expb.curilt != 0 && !ILT_DBGLINE(expb.curilt))) {
      wr_block();
      cr_block();
    } else if ((XBIT(148, 0x1) || XBIT(148, 0x1000)) && (expb.curilt == 0) &&
               (fihb.currfindex != fihb.nextfindex)) {
      fihb.currfindex = fihb.nextfindex;
      fihb.currftag = fihb.nextftag;
    }

    BIH_LABEL(expb.curbih) = lbl;
    ILIBLKP(lbl, expb.curbih);
    fihb.currftag = fihb.nextftag;
    fihb.currfindex = fihb.nextfindex;
  }

  else if (CCSYMG(lbl) == 0 && DBGBIT(8, 4096))
    /* defd but not refd  */
    errlabel((error_code_t)120, ERR_Informational, gbl.lineno, SYMNAME(lbl),
             CNULL);
}

/***************************************************************/

/*
 * the following macro is used by the load and store code to determine if the
 * load or store operation conflicts with the data type of the item being
 * fetched or stored.  This is done for those names entries which are
 * constant array or indirection references.
 * Conflicts could occur when:
 * 1. if the operation is for a double data item and the data type is not
 *    double.
 * 2. if the operation is for a float data item and the data type is not
 *    float.
 * 3. if the operation is for an integral type and its size is inconsistent
 *    with the size of the data type.
 * A conflict is resolved by creating an array (or indirection) reference
 * which has a non-constant offset. The macro argument, "cond", specifies the
 * whether or not there is a conflict.
 */
#define CHECK_NME(nme, cond)                                         \
  {                                                                  \
    NT_KIND i = NME_TYPE(nme);                                       \
    if (NME_SYM(nme) == 0 && (i == NT_ARR || i == NT_IND) && (cond)) \
      nme = add_arrnme(i, NME_NULL, NME_NM(nme), 0, NME_SUB(nme),    \
                       NME_INLARR(nme));                             \
  }

static int
SCALAR_SIZE(DTYPE dtype, int n)
{
  if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR)
    /*  assume that this a pointer to an adjustable length character */
    return n;
  if (dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR)
    return n;
  return size_of(dtype);
}

/***************************************************************/
             /*
              * when inlining a function with an optional argument, where the
              * optional argument is missing in the call, the compiler passes
              * a placeholder, pghpf_03, which it then can test for in PRESENT() calls.
              */
int
optional_missing(int nme)
{
  int sptr, cmblk;
  sptr = NME_SYM(nme);
  if (CCSYMG(sptr) && SCG(sptr) == SC_CMBLK && ADDRESSG(sptr) == 8) {
    cmblk = MIDNUMG(sptr);
    if (strcmp(SYMNAME(cmblk), "pghpf_0") == 0) {
      return 1;
    }
  }
  return 0;
} /* optional_missing */

/*
 * same as above, given an ILM pointer
 */
int
optional_missing_ilm(ILM *ilmpin)
{
  int sptr, cmblk;
  ILM *ilmp;
  ilmp = ilmpin;
  while (1) {
    switch (ILM_OPC(ilmp)) {
    case IM_BASE:
      sptr = ILM_OPND(ilmp, 1);
      if (CCSYMG(sptr) && SCG(sptr) == SC_CMBLK && ADDRESSG(sptr) == 8) {
        cmblk = MIDNUMG(sptr);
        if (strcmp(SYMNAME(cmblk), "pghpf_0") == 0) {
          return 1;
        }
      }
      return 0;
    case IM_PLD:
    case IM_MEMBER:
      ilmp = (ILM *)(ilmb.ilm_base + ILM_OPND(ilmp, 1));
      break;
    case IM_ELEMENT:
    case IM_INLELEM:
      ilmp = (ILM *)(ilmb.ilm_base + ILM_OPND(ilmp, 2));
      break;
    default:
      return 0;
    }
  }
} /* optional_missing_ilm */

/*
 * here, we have a load of the missing optional, replace by a zero
 */
void
replace_by_zero(ILM_OP opc, ILM *ilmp, int curilm)
{
  INT num[4];
  int zero = 0;
  ILM_OP newopc;
  int i1 = ILM_OPND(ilmp, 1);
  switch (opc) {
  /* handle complex */
  case IM_CLD:
    num[0] = 0;
    num[1] = 0;
    zero = getcon(num, DT_CMPLX);
    newopc = IM_CDCON;
    break;
  case IM_CDLD:
    num[0] = stb.dbl0;
    num[1] = stb.dbl0;
    zero = getcon(num, DT_DCMPLX);
    newopc = IM_CCON;
    break;
  case IM_ILD:
  case IM_LLD:
  case IM_LFUNC: /* LFUNC, for PRESENT calls replaced by zero */
    zero = stb.i0;
    newopc = IM_ICON;
    break;

  case IM_KLD:
  case IM_KLLD:
    zero = stb.k0;
    newopc = IM_KCON;
    break;

  case IM_SLLD:
  case IM_SILD:
  case IM_CHLD:
    zero = stb.i0;
    newopc = IM_ICON;
    break;

  case IM_RLD:
    zero = stb.flt0;
    newopc = IM_RCON;
    break;

  case IM_DLD:
    zero = stb.dbl0;
    newopc = IM_DCON;
    break;

  case IM_PLD:
    zero = stb.i0;
    newopc = IM_ICON;
    break;

  default:
    interr("replace_by_zero opc not cased", opc, ERR_Severe);
    newopc = IM_ICON;
    break;
  }
  /* CHANGE the ILM in place */
  SetILM_OPC(ilmp, newopc);
  ILM_OPND(ilmp, 1) = zero;
  /* process as a constant */
  eval_ilm(curilm);
  SetILM_OPC(ilmp, opc);
  ILM_OPND(ilmp, 1) = i1;
} /* replace_by_zero */

/*
 * when inlining a function with an optional argument, where the
 * optional argument is present in the call, the compiler passes
 * the argument, which we can detect as present since it's
 * not a DUMMY
 */
int
optional_present(int nme)
{
  int sptr, cmblk, ptr;
  sptr = NME_SYM(nme);
  if (SCG(sptr) == SC_LOCAL) {
    return 1;
  } else if (SCG(sptr) == SC_BASED) {
    ptr = MIDNUMG(sptr);
    if (SCG(ptr) == SC_LOCAL || SCG(ptr) == SC_CMBLK) {
      return 1;
    }
  } else if (SCG(sptr) == SC_CMBLK) {
    cmblk = MIDNUMG(sptr);
    if (strcmp(SYMNAME(cmblk), "pghpf_0") != 0) {
      return 1;
    }
  }
  return 0;
} /* optional_present */

/*
 * replace this by one
 * use this to inline a function call that we know is TRUE
 */
void
replace_by_one(ILM_OP opc, ILM *ilmp, int curilm)
{
  int one = 0;
  ILM_OP newopc;
  int i1;
  i1 = ILM_OPND(ilmp, 1);
  switch (opc) {
  case IM_LFUNC: /* LFUNC, for PRESENT calls replaced by one */
    one = stb.i1;
    newopc = IM_ICON;
    break;

  default:
    interr("replace_by_one opc not cased", opc, ERR_Severe);
    return;
  }
  /* CHANGE the ILM in place */
  SetILM_OPC(ilmp, newopc);
  ILM_OPND(ilmp, 1) = one;
  /* process as a constant */
  eval_ilm(curilm);
  SetILM_OPC(ilmp, opc);
  ILM_OPND(ilmp, 1) = i1;
} /* replace_by_one */
/***************************************************************/
void
exp_load(ILM_OP opc, ILM *ilmp, int curilm)
{
  int op1;
  int imag;     /* address of the imag. part if complex */

  int nme;      /* names entry */
  int addr,     /* address of the load */
      load = 0; /* load ili generated */
  SPTR tmp;
  int siz;      /* MSZ value for load */
  DTYPE dt;
  bool confl;
  ILM *tmpp;

  op1 = ILM_OPND(ilmp, 1);
  addr = op1;
  nme = NME_OF(addr);
  if (optional_missing_ilm(ilmp)) {
    replace_by_zero(opc, ilmp, curilm);
    return;
  }

  /*
   * if the names entry is for a variable which is an array, then a new
   * names entry is created which will denote the first element (offset 0)
   * of the array -- this catches the cases of '*(a)', where a is an array
   * name
   */
  if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
    nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));

  addr = ILI_OF(addr);
  switch (opc) {
  /* handle complex */
  case IM_CLD:
    if (XBIT(70, 0x40000000)) {
      CHECK_NME(nme, dt_nme(nme) != DT_CMPLX);
      load = ad3ili(IL_LDSCMPLX, addr, nme, MSZ_F8);
      goto cand_load;
    } else {
      imag = ad3ili(IL_AADD, addr, ad_aconi(size_of(DT_FLOAT)), 0);
      tmp = addnme(NT_MEM, SPTR_NULL, nme, 0);
      ILM_RRESULT(curilm) = ad3ili(IL_LDSP, addr, tmp, MSZ_F4);
      tmp = addnme(NT_MEM, NOSYM, nme, 4);
      ILM_IRESULT(curilm) = ad3ili(IL_LDSP, imag, tmp, MSZ_F4);
      ILM_RESTYPE(curilm) = ILM_ISCMPLX;
      return;
    }
  case IM_CDLD:
    if (XBIT(70, 0x40000000)) {
      CHECK_NME(nme, dt_nme(nme) != DT_DCMPLX);
      load = ad3ili(IL_LDDCMPLX, addr, nme, MSZ_F16);
      goto cand_load;
    } else {
      imag = ad3ili(IL_AADD, addr, ad_aconi(size_of(DT_DBLE)), 0);
      tmp = addnme(NT_MEM, SPTR_NULL, nme, 0);
      ILM_RRESULT(curilm) = ad3ili(IL_LDDP, addr, tmp, MSZ_F8);
      tmp = addnme(NT_MEM, NOSYM, nme, 8);
      ILM_IRESULT(curilm) = ad3ili(IL_LDDP, imag, tmp, MSZ_F8);
      ILM_RESTYPE(curilm) = ILM_ISDCMPLX;
      return;
    }
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQLD:
    if (XBIT(70, 0x40000000)) {
      CHECK_NME(nme, dt_nme(nme) != DT_QCMPLX);
      load = ad3ili(IL_LDQCMPLX, addr, nme, MSZ_F32);
      goto cand_load;
    } else {
      imag = ad3ili(IL_AADD, addr, ad_aconi(size_of(DT_QUAD)), 0);
      tmp = addnme(NT_MEM, SPTR_NULL, nme, 0);
      ILM_RRESULT(curilm) = ad3ili(IL_LDQP, addr, tmp, MSZ_F16);
      tmp = addnme(NT_MEM, NOSYM, nme, 16);
      ILM_IRESULT(curilm) = ad3ili(IL_LDQP, imag, tmp, MSZ_F16);
      ILM_RESTYPE(curilm) = ILM_ISQCMPLX;
      return;
    }
#endif
  case IM_ILD:
  case IM_LLD:
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && SCALAR_SIZE(dt, 4) != 4)
      confl = true;
    CHECK_NME(nme, confl);
    load = ad3ili(IL_LD, addr, nme, MSZ_WORD);
  cand_load:
    ADDRCAND(load, nme);
    break;

  case IM_KLD:
  case IM_KLLD:
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && SCALAR_SIZE(dt, 8) != 8)
      confl = true;
    CHECK_NME(nme, confl);
    if (XBIT(124, 0x400)) {
      load = ad3ili(IL_LDKR, addr, nme, MSZ_I8);
      rcandb.kr = 1;
    } else {
      if (flg.endian)
        addr = ad3ili(IL_AADD, addr, ad_aconi((INT)size_of(DT_INT)), 0);
      load = ad3ili(IL_LD, addr, nme, MSZ_WORD);
    }
    ADDRCAND(load, nme);
    break;

  case IM_SLLD:
  case IM_SILD:
    siz = MSZ_SHWORD;
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && size_of(dt) != 2)
      confl = true;
    CHECK_NME(nme, confl);
    load = ad3ili(IL_LD, addr, nme, siz);
    goto cand_load;

  case IM_CHLD:
    siz = MSZ_SBYTE;
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && size_of(dt) != 1)
      confl = true;
    CHECK_NME(nme, confl);
    load = ad3ili(IL_LD, addr, nme, siz);
    goto cand_load;

  case IM_RLD:
    CHECK_NME(nme, dt_nme(nme) != DT_FLOAT);
    load = ad3ili(IL_LDSP, addr, nme, MSZ_F4);
    goto cand_load;

  case IM_DLD:
    CHECK_NME(nme, dt_nme(nme) != DT_DBLE);
    load = ad3ili(IL_LDDP, addr, nme, MSZ_F8);
    goto cand_load;
  case IM_QLD: /*m128*/
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_128);
    load = ad3ili(IL_LDQ, addr, nme, MSZ_F16);
    goto cand_load;
#ifdef TARGET_SUPPORTS_QUADFP
  /* transform the QFLD ilm to LDQP ili */
  case IM_QFLD: /* fp128 */
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_QUAD);
    load = ad3ili(IL_LDQP, addr, nme, MSZ_F16);
    goto cand_load;
#endif
  case IM_M256LD: /* m256 */
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_256);
    load = ad3ili(IL_LD256, addr, nme, MSZ_F32);
    goto cand_load;
#ifdef LONG_DOUBLE_FLOAT128
  case IM_FLOAT128LD:
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_FLOAT128);
    load = ad3ili(IL_FLOAT128LD, addr, nme, MSZ_F16);
    goto cand_load;
#endif /* LONG_DOUBLE_FLOAT128 */

  case IM_PLD:
/* fortran: pointer variables are really integer variables;
 * later phases 'depend' on seeing references via pointers
 * via the 'LDA' ili.
 */
    /* if using integer*8 variables and not 64-bit precision,
       adjust the address of pointer */
    /* ???
 if (flg.endian && !XBIT(124,0x400))
 */
    if (flg.endian) {
      tmp = ILM_SymOPND(ilmp, 2);
      tmpp = (ILM *)(ilmb.ilm_base + ILM_OPND(ilmp, 1));
      if (((tmp == SPTR_NULL) && DTYPEG(ILM_OPND(tmpp, 1)) == DT_INT8) ||
          (SCG(tmp) == SC_BASED && DTYPEG(MIDNUMG(tmp)) == DT_INT8))
        addr = ad3ili(IL_AADD, addr, ad_aconi(size_of(DT_INT)), 0);
    }
    load = ad2ili(IL_LDA, addr, nme);
    ADDRCAND(load, nme);
    /*
     * if the 2nd operand is non-zero, then the 2nd operand is the
     * symbol table entry of some sort of based object.  The symbol
     * table entry is the object in a POINTER statement
     *
     * For POINTER, a names entry of NT_IND through the pointer variable
     * is sufficent.
     *
     * When the PLD is to load the pointer to a character object, the
     * additional character information needs to be created (examine
     * the data type of the symbol which is the second operand.
     */
    tmp = ILM_SymOPND(ilmp, 2);
    if (tmp) {
      DTYPE dtype;
#if DEBUG
      if (!(tmp && DEVICECOPYG(tmp) && DEVCOPYG(tmp))) {
        assert(STYPEG(tmp) == ST_MEMBER || SCG(tmp) == SC_BASED ||
               SCG(tmp) == SC_EXTERN,
               "exp_load:PLD op#2 not based sym, member, or procedure pointer",
               tmp, ERR_Severe); 
      }
#endif
      dtype = DDTG(DTYPEG(tmp));
      if (DTY(dtype) == TY_PTR)
        dtype = DTySeqTyElement(dtype);
      if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
        int mxlen, clen;
        mxlen = 0;
        if ((dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR) && SDSCG(tmp)) {
          if (STYPEG(tmp) == ST_MEMBER) {
            int member, base;
            member = ILM_OPND(ilmp, 1);
            base = ilmb.ilm_base[member + 1];
            clen = exp_get_sdsc_len(tmp, ILI_OF(base), NME_OF(base));
          } else {
            clen = exp_get_sdsc_len(tmp, 0, 0);
          }
        } else
            if (
                STYPEG(tmp) != ST_MEMBER &&
                CLENG(tmp) > 0) {
          if (CHARLEN_64BIT) {
            int clensym, ili;
            clensym = CLENG(tmp);
            if (size_of(DTYPEG(clensym)) == 8) {
              ili = mk_address(CLENG(tmp));
              clen = ad3ili(IL_LDKR, ili, addnme(NT_VAR, CLENG(tmp), 0, 0),
                            MSZ_I8);
            } else {
              /*
               * -Mlarge_arrays (large character lengths WORK-AROUND)
               * there are several cases where the front-end IS NOT creating
               * 64-bit length temps, e.g., the length temp for the adjustl
               * intrinisc.  When we're ready to correct the support of
               * large character, this section of code ought to turn into
               * an assert.
               */
              ili = mk_address(CLENG(tmp));
              clen = ad3ili(IL_LD, ili, addnme(NT_VAR, CLENG(tmp), 0, 0),
                            MSZ_WORD);
              clen = ad1ili(IL_IKMV, clen);
            }
          } else {
            int ili = mk_address(CLENG(tmp));
            clen =
                ad3ili(IL_LD, ili, addnme(NT_VAR, CLENG(tmp), 0, 0), MSZ_WORD);
          }
        }
        else if (DTyCharLength(dtype) == 0 && SDSCG(tmp)) {
          clen = exp_get_sdsc_len(tmp, 0, 0);
        }
        else if (CHARLEN_64BIT)
          clen = mxlen = ad_kconi(DTyCharLength(dtype));
        else
          clen = mxlen = ad_icon(DTyCharLength(dtype));
        ILM_CLEN(curilm) = clen;
        ILM_MXLEN(curilm) = mxlen;
        ILM_RESTYPE(curilm) = ILM_ISCHAR;
      } else if (STYPEG(tmp) == ST_MEMBER) {
        ILM_NME(curilm) = addnme(NT_IND, SPTR_NULL, nme, 0);
#ifdef DEVICEG
      } else if (DEVICEG(tmp) && DT_ISBASIC(DTYPEG(tmp))) {
        ILM_NME(curilm) = addnme(NT_VAR, tmp, 0, 0);
#ifdef TEXTUREG
      } else if (DEVICEG(tmp) && TEXTUREG(tmp)) {
        ILM_NME(curilm) = addnme(NT_VAR, tmp, 0, 0);
#endif
#endif
      } else if (NOCONFLICTG(tmp)) {
        /* the frontend has determined that this pointer-based object
         * cannot conflict with other references via pointers; for
         * example, allocatable arrays and automatic arrays.
         */
        ILM_NME(curilm) = addnme(NT_VAR, tmp, 0, 0);
      } else if (XBIT(125, 0x40)) {
        /* Cray's pointer semantics */
        ILM_NME(curilm) = addnme(NT_VAR, tmp, 0, 0);
      } else {
        ILM_NME(curilm) = addnme(NT_IND, SPTR_NULL, nme, 0);
      }
    } else {
      ILM_NME(curilm) = addnme(NT_IND, SPTR_NULL, nme, 0);
    }
    break;

#ifdef LONG_DOUBLE_FLOAT128
  case IM_CFLOAT128LD:
    ILM_RRESULT(curilm) =
        ad3ili(IL_FLOAT128LD, addr, addnme(NT_MEM, 0, nme, 0), MSZ_F16);
    ILM_IRESULT(curilm) =
        ad3ili(IL_FLOAT128LD, ad3ili(IL_AADD, addr, ad_aconi(16), 0),
               addnme(NT_MEM, 1, nme, 16), MSZ_F16);
    ILM_RESTYPE(curilm) = ILM_ISFLOAT128CMPLX;
    return;
#endif /* LONG_DOUBLE_FLOAT128 */

  default:
    interr("exp_load opc not cased", opc, ERR_Severe);
    break;
  }

  ILM_RESULT(curilm) = load;
}

/***************************************************************/
  /***************************************************************/

/*****  try to use ASSN for all user variables, all compilers *****/
void
set_assn(int nme)
{
  int s = basesym_of(nme);
  if (s)
    ASSNP(s, 1);
}
#define SET_ASSN(n) set_assn(n)

void
exp_store(ILM_OP opc, ILM *ilmp, int curilm)
{
  int nme;       /* names entry                          */
  int op1,       /* operand 1 of the ILM                 */
      op2;       /* operand 2 of the ILM                 */
  int store = 0, /* store ili generated                  */
      addr,      /* address ili where value stored       */
      expr,      /* ili of value being stored            */
      siz,       /* size of the field in the field store */
      ilix,      /* ili indexi                           */
      ilix1;     /* ili index                            */
  int tmp;
  DTYPE dt;
  bool confl;

  int imag; /* address of the imag. part if complex */

  op1 = ILM_OPND(ilmp, 1);

  op2 = ILM_OPND(ilmp, 2);
  nme = NME_OF(op1);
  if (opc != IM_PSEUDOST) {
    if (optional_missing_ilm(ilmp)) {
      /* this is a store to a missing optional argument.
       * it must be on a path that is branched around, or it is illegal.
       * simply drop the expression */
      return;
    }
  }

  switch (opc) {
  case IM_LST:
  case IM_IST:
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && SCALAR_SIZE(dt, 4) != 4)
      confl = true;
    CHECK_NME(nme, confl);
    ilix = ILI_OF(op2);
    if (IL_RES(ILI_OPC(ilix)) == ILIA_AR)
      ilix = ad1ili(IL_AIMV, ilix);
    store = ad4ili(IL_ST, ilix, ILI_OF(op1), nme, MSZ_WORD);
  cand_store:
    if (NME_TYPE(nme) == NT_VAR)
      ASSNP(NME_SYM(nme), 1);
    ADDRCAND(store, nme);
    SET_ASSN(nme);
    break;

  case IM_KLST:
  case IM_KST:
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && SCALAR_SIZE(dt, 8) != 8)
      confl = true;
    if (XBIT(124, 0x400)) {
      /* problem arose with the pointer statement and the value
       * returned by the call to ftn_allocate being an IR
       * AND (as of 12/09/2010) with the result being an AR
       */
      ilix = ILI_OF(op2);
      if (IL_RES(ILI_OPC(ilix)) == ILIA_AR)
        ilix = ad1ili(IL_AKMV, ilix);
      else {
        if (IL_RES(ILI_OPC(ilix)) != ILIA_KR)
          ilix = ad1ili(IL_IKMV, ilix);
      }
      store = ad4ili(IL_STKR, ilix, ILI_OF(op1), nme, MSZ_I8);
      rcandb.kr = 1;
    } else {
      addr = ILI_OF(op1);
      if (flg.endian)
        addr = ad3ili(IL_AADD, (int)ILI_OF(op1), ad_aconi((INT)size_of(DT_INT)),
                      0);
      ilix = ILI_OF(op2);
      if (IL_RES(ILI_OPC(ilix)) == ILIA_AR)
        ilix = ad1ili(IL_AIMV, ilix);
      else if (IL_RES(ILI_OPC(ilix)) == ILIA_KR)
        ilix = ad1ili(IL_KIMV, ilix);
      store = ad4ili(IL_ST, ilix, addr, nme, MSZ_WORD);
    }
    CHECK_NME(nme, confl);
    if (NME_TYPE(nme) == NT_VAR)
      ASSNP(NME_SYM(nme), 1);
    ADDRCAND(store, nme);
    SET_ASSN(nme);
    break;

  case IM_SLST:
  case IM_SIST:
    siz = MSZ_SHWORD;
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && size_of(dt) != 2)
      confl = true;
    CHECK_NME(nme, confl);
    expr = ILI_OF(op2);
    store = ad4ili(IL_ST, expr, (int)ILI_OF(op1), nme, siz);
    goto cand_store;

  case IM_CHST:
    siz = MSZ_SBYTE;
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && size_of(dt) != 1)
      confl = true;
    CHECK_NME(nme, confl);
    expr = ILI_OF(op2);
    store = ad4ili(IL_ST, expr, (int)ILI_OF(op1), nme, siz);
    goto cand_store;

  case IM_AST:
    expr = ILI_OF(op2);
    if (IL_RES(ILI_OPC(expr)) == ILIA_AR)
      expr = ad1ili(IL_AIMV, expr);
    store = ad4ili(IL_ST, expr, (int)ILI_OF(op1), nme, MSZ_WORD);
    SET_ASSN(nme);
    break;

  case IM_KAST:
    addr = ILI_OF(op1);
    expr = ILI_OF(op2);
    if (IL_RES(ILI_OPC(expr)) == ILIA_AR)
      expr = ad1ili(IL_AKMV, expr);
    store = ad4ili(IL_STKR, expr, addr, nme, MSZ_I8);
    SET_ASSN(nme);
    break;

  case IM_PSTRG1:
    store = ad2ili(IL_STRG1, (int)ILI_OF(op1), op2);
    break;

  case IM_PST:
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    confl = false;
    dt = dt_nme(nme);
    if (dt && DT_ISSCALAR(dt) && SCALAR_SIZE(dt, 8) != 8)
      confl = true;
    CHECK_NME(nme, confl);
    expr = ILI_OF(op2);
    switch (ILI_OPC(expr)) {
    case IL_AIMV:
    case IL_AKMV:
      expr = ILI_OPND(expr, 1);
      break;
    default:
      break;
    }
    if (IL_RES(ILI_OPC(expr)) != ILIA_AR) {
      expr = ad1ili(IL_KAMV, expr);
    }
    store = ad3ili(IL_STA, expr, (int)ILI_OF(op1), nme);

    /*
     * check if &var is being stored.  If so, the base symbol's "address
     * taken" flag is set.
     */
    loc_of((int)NME_OF(op2));

    /*
     * store the names result of the store -- this is just an indirection
     * based on the names entry of the STA
     */
    ILM_NME(curilm) = addnme(NT_IND, SPTR_NULL, nme, (INT)0);
    goto cand_store;

  case IM_RST:
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    CHECK_NME(nme, dt_nme(nme) != DT_FLOAT);
    store = ad4ili(IL_STSP, (int)ILI_OF(op2), (int)ILI_OF(op1), nme, MSZ_F4);
    goto cand_store;

  case IM_DST:
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    CHECK_NME(nme, dt_nme(nme) != DT_DBLE);
    store = ad4ili(IL_STDP, (int)ILI_OF(op2), (int)ILI_OF(op1), nme, MSZ_F8);
    goto cand_store;
  case IM_QST: /*m128*/
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_128);
    store = ad4ili(IL_STQ, (int)ILI_OF(op2), (int)ILI_OF(op1), nme, MSZ_F16);
	goto cand_store;
#ifdef TARGET_SUPPORTS_QUADFP
  /* transform the QFST ilm TO STQP ili */
  case IM_QFST:
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_QUAD);
    store = ad4ili(IL_STQP, (int)ILI_OF(op2), (int)ILI_OF(op1), nme, MSZ_F16);
    goto cand_store;
#endif
  case IM_M256ST: /*m256*/
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_256);
    store = ad4ili(IL_ST256, ILI_OF(op2), ILI_OF(op1), nme, MSZ_F32);
    goto cand_store;

#ifdef LONG_DOUBLE_FLOAT128
  case IM_FLOAT128ST:
    if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
      nme = add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
    CHECK_NME(nme, DTY(dt_nme(nme)) != TY_FLOAT128);
    store = ad4ili(IL_FLOAT128ST, ILI_OF(op2), ILI_OF(op1), nme, MSZ_F16);
    goto cand_store;
#endif /* LONG_DOUBLE_FLOAT128 */

  case IM_SMOVE: /* make sure this works for both languages */
    SET_ASSN(NME_OF(op1));
    {
      ILM *ilmpx = (ILM *)(ilmb.ilm_base + op2);
      int rsi = ilm_return_slot_index((ILM_T *)ilmpx);
      if (rsi) {
        ilmpx = (ILM *)(ilmb.ilm_base + ILM_OPND(ilmpx, rsi));
        if (ILM_OPC(ilmpx) == IM_LOC && ILM_OPND(ilmpx, 1) == op1) {
          /* avoid useless struct copy for functions returning structs */
          chk_block(ILI_OF(op2));
          ILM_NME(curilm) = NME_OF(op2);
          ILM_RESULT(curilm) = ILI_OF(op2);
          return;
        }
        if (XBIT(121, 0x800) &&
            ILM_OPC((ILM *)(ilmb.ilm_base + op2)) == IM_SFUNC &&
            ILM_OPC(ilmpx) == IM_FARG) {
          /*
           * Have SMOVE representing LHS = SFUNC().
           * SFUNC expands to a JSR with the result as the first hidden
           * argument; make the LHS the result.
           */
          ilix = ILI_OF(op2);               /* IL_JSR */
          ilix1 = ILI_OPND(ilix, 2);        /* IL_ARGAR of the result */
          ILI_OPND(ilix1, 1) = ILI_OF(op1); /* replace result with LHS */
          ilix1 = ILI_ALT(ilix);            /* IL_JSR's IL_GJSR */
          ilix1 = ILI_OPND(ilix1, 2);       /* IL_GARGRET */
          ILI_OPND(ilix1, 1) = ILI_OF(op1);
          ILI_OPND(ilix1, 4) = NME_OF(op1);
          chk_block(ilix);
          ILM_NME(curilm) = NME_OF(op1);
          ILM_RESULT(curilm) = ilix;
          return;
        }
      }
    }
    expand_smove(op1, op2, ILM_DTyOPND(ilmp, 3));
    ILM_RESULT(curilm) = ILI_OF(op2);
    ILM_NME(curilm) = NME_OF(op2);
    return;

  case IM_SZERO: /* make sure this works for both languages */
    SET_ASSN(NME_OF(op1));
    exp_szero(ilmp, curilm, op1, op2, (int)ILM_OPND(ilmp, 3));
    ILM_RESULT(curilm) = 0;
    ILM_NME(curilm) = NME_UNK;
    return;

  case IM_PSEUDOST:
    expr = ILI_OF(op2);
    switch (IL_RES(ILI_OPC(expr))) {
    case ILIA_IR:
      store = ad1ili(IL_FREEIR, expr);
      break;

    case ILIA_SP:
      /*
       * For complex, store the imaginary part and then the real part.
       * Then fall thru to set the ilm's real result and block number
       * and update the block.
       */
      if (ILM_RESTYPE(op2) == ILM_ISCMPLX) {
        store = ad1ili(IL_FREESP, (int)ILM_IRESULT(op2));
        chk_block(store);
        ILM_IRESULT(curilm) = store;
        ILM_RESTYPE(curilm) = ILM_ISCMPLX;
        if (EXPDBG(8, 16))
          fprintf(gbl.dbgfil, "store imag: ilm %d, block %d, ili %d\n", curilm,
                  expb.curbih, store);
      }
      store = ad1ili(IL_FREESP, expr);
      break;

    case ILIA_DP:
      if (ILM_RESTYPE(op2) == ILM_ISDCMPLX) {
        store = ad1ili(IL_FREEDP, (int)ILM_IRESULT(op2));
        chk_block(store);
        ILM_IRESULT(curilm) = store;
        ILM_RESTYPE(curilm) = ILM_ISDCMPLX;
        if (EXPDBG(8, 16))
          fprintf(gbl.dbgfil, "store imag: ilm %d, block %d, ili %d\n", curilm,
                  expb.curbih, store);
      }
      store = ad1ili(IL_FREEDP, expr);
      break;
#ifdef ILIA_CS
    case ILIA_CS:
      store = ad1ili(IL_FREECS, expr);
      break;
    case ILIA_CD:
      store = ad1ili(IL_FREECD, expr);
      break;
#endif
    case ILIA_AR:
      store = ad1ili(IL_FREEAR, expr);
      ILM_NME(curilm) = NME_OF(op2);
      break;

    case ILIA_KR:
      store = ad1ili(IL_FREEKR, expr);
      break;

#ifdef LONG_DOUBLE_FLOAT128
    case ILIA_FLOAT128:
      if (ILM_RESTYPE(op2) == ILM_ISFLOAT128CMPLX) {
        store = ad1ili(IL_FLOAT128FREE, (int)ILM_IRESULT(op2));
        chk_block(store);
        ILM_IRESULT(curilm) = store;
        ILM_RESTYPE(curilm) = ILM_ISFLOAT128CMPLX;
        if (EXPDBG(8, 16))
          fprintf(gbl.dbgfil, "store imag: ilm %d, block %d, ili %d\n", curilm,
                  expb.curbih, store);
      }
      store = ad1ili(IL_FLOAT128FREE, expr);
      break;
#endif /* LONG_DOUBLE_FLOAT128 */

    case ILIA_LNK:
      dt = ili_get_vect_dtype(expr);
      if (dt) {
        store = ad2ili(IL_FREE, expr, dt);
        break;
      }
      FLANG_FALLTHROUGH;

    default:
      interr("PSEUDOST: bad link", curilm, ERR_Severe);
    }
    break;
  /* complex stuff */
  case IM_CST:
    if (XBIT(70, 0x40000000)) {
      if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
        nme =
            add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
      CHECK_NME(nme, dt_nme(nme) != DT_CMPLX);
      store = ad4ili(IL_STSCMPLX, ILI_OF(op2), ILI_OF(op1), nme, MSZ_F8);
      goto cand_store;
    } else {
      /*
       * For complex, store the imaginary part and then the real part.
       * Then fall thru to set the ilm's real result and block number
       * and update the block.

       * If this is a store of return value of float complex,
       * need to make nme to NME_UNK otherwise cg will not do correct store.
       */
      tmp = expb.curilt;
      store = ad1ili(IL_FREESP, (int)ILM_RRESULT(op2));
      chk_block(store);
      if (tmp != expb.curilt)
        ILT_CPLX(expb.curilt) = 1;

      nme = addnme(NT_MEM, NOSYM, (int)NME_OF(op1), (INT)4);
      imag = ad3ili(IL_AADD, (int)ILI_OF(op1), ad_aconi((INT)size_of(DT_FLOAT)),
                    0);
      store = ad4ili(IL_STSP, (int)ILM_IRESULT(op2), imag, nme, MSZ_F4);
      tmp = expb.curilt;
      chk_block(store);
      ILM_IRESULT(curilm) = store;
      if (tmp != expb.curilt)
        ILT_CPLX(expb.curilt) = 1;
      if (EXPDBG(8, 16))
        fprintf(gbl.dbgfil, "store imag: ilm %d, block %d, ili %d\n", curilm,
                expb.curbih, store);
      nme = addnme(NT_MEM, SPTR_NULL, (int)NME_OF(op1), (INT)0);
      store = ad4ili(IL_STSP, ad1ili(IL_CSESP, (int)ILM_RRESULT(op2)),
                     (int)ILI_OF(op1), nme, MSZ_F4);
      ILM_RESTYPE(curilm) = ILM_ISCMPLX;
      SET_ASSN(nme);
    }
    goto cmplx_shared;

  case IM_CDST:
    if (XBIT(70, 0x40000000)) {
      if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
        nme =
            add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
      CHECK_NME(nme, dt_nme(nme) != DT_DCMPLX);
      store = ad4ili(IL_STDCMPLX, ILI_OF(op2), ILI_OF(op1), nme, MSZ_F16);
      goto cand_store;
    } else {
      tmp = expb.curilt;
      store = ad1ili(IL_FREEDP, (int)ILM_RRESULT(op2));
      chk_block(store);
      if (tmp != expb.curilt)
        ILT_CPLX(expb.curilt) = 1;

      nme = addnme(NT_MEM, NOSYM, NME_OF(op1), 8);
      imag = ad3ili(IL_AADD, ILI_OF(op1), ad_aconi(size_of(DT_DBLE)), 0);
      store = ad4ili(IL_STDP, ILM_IRESULT(op2), imag, nme, MSZ_F8);
      tmp = expb.curilt;
      chk_block(store);
      if (tmp != expb.curilt)
        ILT_CPLX(expb.curilt) = 1;
      ILM_IRESULT(curilm) = store;
      if (EXPDBG(8, 16))
        fprintf(gbl.dbgfil, "store imag: ilm %d, block %d, ili %d\n", curilm,
                expb.curbih, store);

      nme = addnme(NT_MEM, SPTR_NULL, NME_OF(op1), 0);
      store = ad4ili(IL_STDP, ad1ili(IL_CSEDP, ILM_RRESULT(op2)), ILI_OF(op1),
                     nme, MSZ_F8);
      ILM_RESTYPE(curilm) = ILM_ISDCMPLX;
    }
#ifdef TARGET_SUPPORTS_QUADFP
    goto cmplx_shared;

  case IM_CQST:
    if (XBIT(70, 0x40000000)) {
      if (NME_TYPE(nme) == NT_VAR && DTY(DTYPEG(NME_SYM(nme))) == TY_ARRAY)
        nme =
            add_arrnme(NT_ARR, SPTR_NULL, nme, 0, ad_icon(0), NME_INLARR(nme));
      CHECK_NME(nme, dt_nme(nme) != DT_QCMPLX);
      store = ad4ili(IL_STQCMPLX, ILI_OF(op2), ILI_OF(op1), nme, MSZ_F32);
      goto cand_store;
    } else {
      tmp = expb.curilt;
      store = ad1ili(IL_FREEQP, (int)ILM_RRESULT(op2));
      chk_block(store);
      if (tmp != expb.curilt)
        ILT_CPLX(expb.curilt) = 1;

      nme = addnme(NT_MEM, NOSYM, NME_OF(op1), 16);
      imag = ad3ili(IL_AADD, ILI_OF(op1), ad_aconi(size_of(DT_QUAD)), 0);
      store = ad4ili(IL_STQP, ILM_IRESULT(op2), imag, nme, MSZ_F16);
      tmp = expb.curilt;
      chk_block(store);
      if (tmp != expb.curilt)
        ILT_CPLX(expb.curilt) = 1;
      ILM_IRESULT(curilm) = store;
      if (EXPDBG(8, 16))
        fprintf(gbl.dbgfil, "store imag: ilm %d, block %d, ili %d\n", curilm,
                expb.curbih, store);

      nme = addnme(NT_MEM, SPTR_NULL, NME_OF(op1), 0);
      store = ad4ili(IL_STQP, ad1ili(IL_CSEQP, ILM_RRESULT(op2)), ILI_OF(op1),
                     nme, MSZ_F16);
      ILM_RESTYPE(curilm) = ILM_ISQCMPLX;
    }
#endif
  cmplx_shared:
    SET_ASSN(NME_OF(op1));
    tmp = expb.curilt;
    chk_block(store);
    if (tmp != expb.curilt && !XBIT(70, 0x40000000))
      ILT_CPLX(expb.curilt) = 1;
    ILM_RESULT(curilm) = store;
    ILM_BLOCK(curilm) = expb.curbih;

    if (XBIT(70, 0x40000000)) {
      if (EXPDBG(8, 16))
        fprintf(gbl.dbgfil, "store complex: ilm %d, block %d, ili %d\n", curilm,
                expb.curbih, store);
    } else {
      if (EXPDBG(8, 16))
        fprintf(gbl.dbgfil, "store real: ilm %d, block %d, ili %d\n", curilm,
                expb.curbih, store);
    }
    return;
  case IM_CSTR:
    /* ONLY store the real part of a complex */
    nme = NME_OF(op1);
    nme = addnme(NT_MEM, SPTR_NULL, nme, 0);
    addr = ILI_OF(op1);
    store = ad4ili(IL_STSP, ILI_OF(op2), addr, nme, MSZ_F4);
    ILM_RESULT(curilm) = store;
    if (EXPDBG(8, 16))
      fprintf(gbl.dbgfil, "ONLY store real: ilm %d, block %d, ili %d\n", curilm,
              expb.curbih, store);
    SET_ASSN(nme);
    break;
  case IM_CSTI:
    /* ONLY store the imaginary part of a complex */
    nme = NME_OF(op1);
    nme = addnme(NT_MEM, NOSYM, nme, 4);
    addr = ILI_OF(op1);
    addr = ad3ili(IL_AADD, addr, ad_aconi((INT)size_of(DT_FLOAT)), 0);
    store = ad4ili(IL_STSP, ILI_OF(op2), addr, nme, MSZ_F4);
    ILM_RESULT(curilm) = store;
    if (EXPDBG(8, 16))
      fprintf(gbl.dbgfil, "ONLY store imag: ilm %d, block %d, ili %d\n", curilm,
              expb.curbih, store);
    SET_ASSN(nme);
    break;
  case IM_CDSTR:
    /* ONLY store the real part of a complex */
    nme = NME_OF(op1);
    nme = addnme(NT_MEM, SPTR_NULL, nme, 0);
    addr = ILI_OF(op1);
    store = ad4ili(IL_STDP, ILI_OF(op2), addr, nme, MSZ_F8);
    ILM_RESULT(curilm) = store;
    if (EXPDBG(8, 16))
      fprintf(gbl.dbgfil, "ONLY store real: ilm %d, block %d, ili %d\n", curilm,
              expb.curbih, store);
    SET_ASSN(nme);
    break;
  case IM_CDSTI:
    /* ONLY store the imaginary part of a complex */
    nme = NME_OF(op1);
    nme = addnme(NT_MEM, NOSYM, nme, 8);
    addr = ILI_OF(op1);
    addr = ad3ili(IL_AADD, addr, ad_aconi(size_of(DT_DBLE)), 0);
    store = ad4ili(IL_STDP, ILI_OF(op2), addr, nme, MSZ_F8);
    ILM_RESULT(curilm) = store;
    if (EXPDBG(8, 16))
      fprintf(gbl.dbgfil, "ONLY store imag: ilm %d, block %d, ili %d\n", curilm,
              expb.curbih, store);
    SET_ASSN(nme);
    break;

#ifdef LONG_DOUBLE_FLOAT128
  case IM_CFLOAT128ST: {
    int real = ILM_RRESULT(op2);
    store = ad1ili(IL_FLOAT128FREE, real);
    tmp = expb.curilt;
    chk_block(store);
    if (tmp != expb.curilt)
      ILT_CPLX(expb.curilt) = 1;
    nme = addnme(NT_MEM, 1, NME_OF(op1), 16);
    tmp = ad3ili(IL_AADD, ILI_OF(op1), ad_aconi(16), 0);
    store = ad4ili(IL_FLOAT128ST, ILM_IRESULT(op2), tmp, nme, MSZ_F16);
    ILM_IRESULT(curilm) = store;
    tmp = expb.curilt;
    chk_block(store);
    if (tmp != expb.curilt)
      ILT_CPLX(expb.curilt) = 1;
    nme = addnme(NT_MEM, 0, NME_OF(op1), 0);
    real = ad_cse(real);
    store = ad4ili(IL_FLOAT128ST, real, ILI_OF(op1), nme, MSZ_F16);
    tmp = expb.curilt;
    chk_block(store);
    if (tmp != expb.curilt)
      ILT_CPLX(expb.curilt) = 1;
    ILM_RRESULT(curilm) = store;
    ILM_BLOCK(curilm) = expb.curbih;
    ILM_RESTYPE(curilm) = ILM_ISFLOAT128CMPLX;
    SET_ASSN(NME_OF(op1));
    return;
  }

  case IM_CFLOAT128STR:
    /* ONLY store the real part of a complex */
    nme = NME_OF(op1);
    nme = addnme(NT_MEM, 0, nme, (INT)0);
    addr = ILI_OF(op1);
    store = ad4ili(IL_FLOAT128ST, ILI_OF(op2), addr, nme, MSZ_F16);
    ILM_RESULT(curilm) = store;
    SET_ASSN(nme);
    break;

  case IM_CFLOAT128STI:
    /* ONLY store the imaginary part of a complex */
    nme = NME_OF(op1);
    nme = addnme(NT_MEM, 1, nme, (INT)16);
    addr = ILI_OF(op1);
    addr = ad3ili(IL_AADD, addr, ad_aconi(16), 0);
    store = ad4ili(IL_FLOAT128ST, ILI_OF(op2), addr, nme, MSZ_F16);
    ILM_RESULT(curilm) = store;
    SET_ASSN(nme);
    break;
#endif /* LONG_DOUBLE_FLOAT128 */

  default:
    interr("exp_store: ilm not cased", curilm, ERR_Severe);
    break;
  } /*****  end of switch(opc)  *****/

  if (!exp_end_atomic(store, curilm)) {
    chk_block(store);
    ILM_RESULT(curilm) = store;
    ILM_BLOCK(curilm) = expb.curbih;
  }

  if (EXPDBG(8, 16))
    fprintf(gbl.dbgfil, "store result: ilm %d, block %d, ili %d\n", curilm,
            expb.curbih, store);
}

/***************************************************************/
/*
 * this routine expands the ilm which are defined as macros. The macro
 * expansion of an ilm is relatively straight forward and is defined by the
 * information in the template definitions as processed by the ilmtp utility.
 */
int
exp_mac(ILM_OP opc, ILM *ilmp, int curilm)
{

  int ilicnt, noprs, i;
  unsigned int pattern, index;
  DTYPE dtype;
  union {
    INT numi[2];
    DBLE numd;
  } num;
  ILI newili;
  ILMOPND *ilmopr;
  ILMMAC *ilmtpl;
  const char *nmptr;

  /*
   * locate the following for the ilm - the number of ili the ilm expands
   * to (ilicnt), its length, and the index into the template area of the
   * first ili (pattern)
   */
  index = 0;
  ilicnt = ilms[opc].ilict;
  pattern = ilms[opc].pattern;

  /* Loop for each ili template in this ILM expansion */
  while (ilicnt-- > 0) {
    ilmtpl = (ILMMAC *)&ilmtp[pattern];

    newili.opc = (ILI_OP)ilmtpl->opc; /* get ili opcode */ // ???

    /* Loop for each operand in this ili template */
    for (i = 0, noprs = ilis[newili.opc].oprs; noprs > 0; ++i, --noprs) {

      ilmopr = (ILMOPND *)&ilmopnd[ilmtpl->opnd[i]];
      switch (ilmopr->type) {

      case ILMO_P:
        newili.opnd[i] = ILM_RESULT(ILM_OPND(ilmp, ilmopr->aux));
        break;

      case ILMO_RP:
        newili.opnd[i] = ILM_RRESULT(ILM_OPND(ilmp, ilmopr->aux));
        break;

      case ILMO_IP:
        newili.opnd[i] = ILM_IRESULT(ILM_OPND(ilmp, ilmopr->aux));
        break;

      case ILMO_T:
        newili.opnd[i] = ILM_TEMP(ilmopr->aux);
        break;

      case ILMO_V:
        newili.opnd[i] = ILM_OPND(ilmp, ilmopr->aux);
        break;

      case ILMO_IV:
        newili.opnd[i] = ilmopr->aux;
        break;
      case ILMO_DR:
        newili.opnd[i] = IR(ilmopr->aux);
        break;
      case ILMO_AR:
        newili.opnd[i] = AR(ilmopr->aux);
        break;
      case ILMO_SP:
        newili.opnd[i] = SP(ilmopr->aux);
        break;
      case ILMO_DP:
        newili.opnd[i] = DP(ilmopr->aux);
        break;
      case ILMO_ISP:
        newili.opnd[i] = ISP(ilmopr->aux);
        break;
      case ILMO_IDP:
        newili.opnd[i] = IDP(ilmopr->aux);
        break;

      case ILMO_SZ:
        dtype = DT_INT;
        num.numi[0] = 0;
        num.numi[1] = size_of((DTYPE)ILM_OPND(ilmp, ilmopr->aux));
        if (num.numi[1] == 0)
          num.numi[1] = 1;
        goto get_con;

      case ILMO_SCZ: /* size with the scale factored out */
        dtype = DT_INT;
        num.numi[0] = 0;
        scale_of((DTYPE)ILM_OPND(ilmp, ilmopr->aux), &num.numi[1]);
        goto get_con;

      case ILMO_RSYM:
        nmptr = ilmaux[ilmopr->aux];
        dtype = DT_FLOAT;
        num.numi[0] = 0;
        if (atoxf(nmptr, &num.numi[1], strlen(nmptr)) != 0)
          interr("exp_mac: RSYM error", curilm, ERR_Severe);
        goto get_con;

      case ILMO_DSYM:
        nmptr = ilmaux[ilmopr->aux];
        dtype = DT_DBLE;
        if (atoxd(nmptr, num.numd, strlen(nmptr)) != 0)
          interr("exp_mac: DSYM error", curilm, ERR_Severe);
        goto get_con;

      case ILMO_XRSYM:
        nmptr = ilmaux[ilmopr->aux];
        dtype = DT_FLOAT;
        num.numi[0] = 0;
        if (atoxi(nmptr, &num.numi[1], strlen(nmptr), 16) != 0)
          interr("exp_mac: XRSYM error", curilm, ERR_Severe);
        goto get_con;

      case ILMO_XDSYM:
        nmptr = ilmaux[ilmopr->aux];
        dtype = DT_DBLE;
        {
          int len;
          const char *p;
          for (len = 0, p = nmptr; *p != ','; p++) {
            if (*p)
              len++;
            else {
              interr("exp_mac: XDSYM error1", curilm, ERR_Severe);
              goto get_con;
            }
          }
          if (atoxi(nmptr, &num.numi[0], len, 16) != 0) {
            interr("exp_mac: XDSYM error2", curilm, ERR_Severe);
            goto get_con;
          }
          p++;
          if (atoxi(p, &num.numi[1], strlen(p), 16) != 0) {
            interr("exp_mac: XDSYM error3", curilm, ERR_Severe);
          }
        }
        goto get_con;
      case ILMO_LLSYM:
        nmptr = ilmaux[ilmopr->aux];
        dtype = DT_INT8;
        num.numi[0] = 0;
        if (atoxi64(nmptr, &num.numi[0], strlen(nmptr), 10) != 0)
          interr("exp_mac: LSYM error", curilm, ERR_Severe);
        goto get_con;
      case ILMO_ISYM:
        nmptr = ilmaux[ilmopr->aux];
        dtype = DT_INT;
        num.numi[0] = 0;
        if (atoxi(nmptr, &num.numi[1], strlen(nmptr), 10) != 0)
          interr("exp_mac: ISYM error", curilm, ERR_Severe);

      get_con:
        newili.opnd[i] = getcon(num.numi, dtype);
        break;

      case ILMO__ESYM:
        /*
         * need to generate the name of an external function taking into
         * consideration the number of '_'s beginning the name.  the name passed
         * from ilmtp.n is exactly how the name should appear in the generated
         * code.  This processing is necessary since an additional '_' may be
         * prependend by getsname() (assem.c).
         */
        /* otherwise, fall thru */
      case ILMO_ESYM:
        newili.opnd[i] = efunc(ilmaux[ilmopr->aux]);
        break;

      case ILMO_SCF: /* scale factor of size - an immediate val */
        newili.opnd[i] = scale_of(ILM_DTyOPND(ilmp, ilmopr->aux), &num.numi[1]);
        break;

      case ILMO_DRRET:
#if defined(IR_RETVAL)
        newili.opnd[i] = IR_RETVAL;
#else
        interr("exp_mac: need IR_RETVAL", ilmopr->type, ERR_Severe);
#endif
        break;
      case ILMO_ARRET:
#if defined(AR_RETVAL)
        newili.opnd[i] = AR_RETVAL;
#else
        interr("exp_mac: need AR_RETVAL", (int)ilmopr->type, ERR_Severe);
#endif
        break;
      case ILMO_SPRET:
#if defined(SP_RETVAL)
        newili.opnd[i] = SP_RETVAL;
#else
        interr("exp_mac: need SP_RETVAL", (int)ilmopr->type, ERR_Severe);
#endif
        break;
      case ILMO_DPRET:
#if defined(DP_RETVAL)
        newili.opnd[i] = DP_RETVAL;
#else
        interr("exp_mac: need DP_RETVAL", (int)ilmopr->type, ERR_Severe);
#endif
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case ILMO_QPRET:
#if defined(QP_RETVAL)
        newili.opnd[i] = QP_RETVAL;
#else
        interr("exp_mac: need QP_RETVAL", (int)ilmopr->type, ERR_Severe);
#endif
        break;
#endif
      case ILMO_KRRET:
#if defined(KR_RETVAL)
        newili.opnd[i] = KR_RETVAL;
#else
        interr("exp_mac: need KR_RETVAL", (int)ilmopr->type, ERR_Severe);
#endif
        break;
#if defined(ILMO_DRPOS)
      case ILMO_DRPOS:
#if defined(TARGET_WIN)
        newili.opnd[i] = IR((ilmopr->aux >> 8) & 0xff);
#else
        newili.opnd[i] = IR((ilmopr->aux) & 0xff);
#endif
        break;
      case ILMO_ARPOS:
#if defined(TARGET_WIN)
        newili.opnd[i] = AR((ilmopr->aux >> 8) & 0xff);
#else
        newili.opnd[i] = AR((ilmopr->aux) & 0xff);
#endif
        break;
      case ILMO_SPPOS:
#if defined(TARGET_WIN)
        newili.opnd[i] = SP((ilmopr->aux >> 8) & 0xff);
#else
        newili.opnd[i] = SP((ilmopr->aux) & 0xff);
#endif
        break;
      case ILMO_DPPOS:
#if defined(TARGET_WIN)
        newili.opnd[i] = DP((ilmopr->aux >> 8) & 0xff);
#else
        newili.opnd[i] = DP((ilmopr->aux) & 0xff);
#endif
        break;
#endif

      default:
        interr("exp_mac: opnd not handled", opc /*(int)ilmopr->type*/,
               ERR_Severe);

      } /***  end of switch on operand type  ***/
    }   /*** end of noprs loop ***/

    /*
     * add the ili just formed
     */

    /*
     printf("%s, %u, %u\n",
         ilis[newili.opc].name, newili.opnd[0], newili.opnd[1]);
     */

    index = addili((ILI *)&newili);
    /*
     * store away the location (index) of the ili just created
     */
    ilmopr = (ILMOPND *)&ilmopnd[ilmtpl->result];
    switch (ilmopr->type) {

    case ILMO_R:
      ILM_RESULT(curilm) = index;
      break;

    case ILMO_KR:
      ILM_RESULT(curilm) = index;
      break;

    case ILMO_T:
      ILM_TEMP(ilmopr->aux) = index;
      break;

    case ILMO_NULL:
      break;

    case ILMO_RR:
      ILM_RRESULT(curilm) = index;
      ILM_RESTYPE(curilm) = IM_DCPLX(opc) ? ILM_ISDCMPLX : ILM_ISCMPLX;
      break;

    case ILMO_IR:
      ILM_IRESULT(curilm) = index;
      ILM_RESTYPE(curilm) = IM_DCPLX(opc) ? ILM_ISDCMPLX : ILM_ISCMPLX;
      break;

    default:
      interr("exp_mac: bad ilmopr->type", newili.opc /*(int)ilmopr->type*/,
             ERR_Severe);
    }
    /*
     * skip to the next ili template -- the length of the template is the
     * number of operands + 2 (1 for the opcode and 1 for the result
     * specifier
     */
    pattern += ilis[newili.opc].oprs + 2;

  } /*** end of ilicnt loop ***/

  return index; /* return the last ili created */
}

static int
efunc(const char *nm)
{
  const char *p;
  int func;
  DTYPE resdt;

  resdt = (DTYPE)-1;
  p = nm;
  if (*p == '%') {
    switch (*++p) {
    case 's':
      resdt = DT_FLOAT;
      break;
    case 'd':
      resdt = DT_DBLE;
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case 'q':
      resdt = DT_QUAD;
      break;
#endif
    case 'i':
      resdt = DT_INT;
      break;
    case 'l':
      resdt = DT_INT8;
      break;
    case 'u':
      p++;
      if (*p == 'i')
        resdt = DT_UINT;
      else if (*p == 'l')
        resdt = DT_UINT8;
      else {
        interr("efunc: unexpected u type", *p, ERR_Severe);
      }
      break;
    case 'v':
      resdt = DT_NONE;
      break;
    default:
      interr("efunc: unexpected result type", *p, ERR_Severe);
      break;
    }
    while (*++p != '%') {
      if (*p == 0) {
        interr("efunc: malformed result type", 0, ERR_Severe);
        p = nm;
        break;
      }
      p++;
    }
    p++;
  }
  func = mkfunc(p);
  if (((int)resdt) >= 0) {
    DTYPEP(func, resdt);
  }
  return func;
}

  /***************************************************************/

#define EXP_ISFUNC(s) (STYPEG(s) == ST_PROC)
#define EXP_ISINDIR(s) (SCG(s) == SC_DUMMY)

/***************************************************************/

void
exp_ref(ILM_OP opc, ILM *ilmp, int curilm)
{
  SPTR sym;   /* symbol table entry		 */
  int ili1;   /* ili pointer			 */
  int base;   /* base ili of reference	 */
  int basenm; /* names entry of base ili	 */

  switch (opc) {
  default:
    return;

  case IM_BASE:
    /* get the base symbol entry  */
    sym = ILM_SymOPND(ilmp, 1);
    ili1 = create_ref(sym, &basenm, 0, 0, &ILM_CLEN(curilm), &ILM_MXLEN(curilm),
                      &ILM_RESTYPE(curilm));
    break;

  case IM_MEMBER:
    base = ILM_OPND(ilmp, 1);
    sym = ILM_SymOPND(ilmp, 2);
    ili1 =
        create_ref(sym, &basenm, NME_OF(base), ILI_OF(base), &ILM_CLEN(curilm),
                   &ILM_MXLEN(curilm), &ILM_RESTYPE(curilm));
    break;

  case IM_INLELEM: /* when inlining ftn and dummys/actuals don't match */
  case IM_ELEMENT:
    exp_array(opc, ilmp, curilm);
    return;
  }

  ILM_RESULT(curilm) = ili1;
  ILM_NME(curilm) = basenm;
}

/* Updates the nme to be an IND (indirection) if the sptr
 * is local in the caller of the outlined function.
 */
static int
update_local_nme(int nme, int sptr)
{
  const SC_KIND sc = SCG(sptr);

  if (((gbl.outlined || ISTASKDUPG(GBL_CURRFUNC)) && PARREFG(sptr)) ||
      TASKG(sptr)) {

    /* Only consider updating the nme if there is one given and its not ind */
    if (!nme || NME_TYPE(nme) == NT_IND)
      return nme;

    if (sc == SC_EXTERN || sc == SC_STATIC)
      return nme;

    if (sc == SC_CMBLK)
      return nme;

    /* We only want to generate indirect if the private is not local to this
     * region.
     */
    if (sc == SC_PRIVATE && is_llvm_local_private(sptr))
      return nme;
    return addnme(NT_IND, SPTR_NULL, nme, 0);
  }
  return nme;
}

static int
create_ref(SPTR sym, int *pnmex, int basenm, int baseilix, int *pclen,
           int *pmxlen, int *prestype)
{
  ISZ_T val[2]; /* constant value array		 */
  int ilix;     /* result */
  int ili1;     /* ili pointer			 */
  int ili2;     /* another ili pointer		 */
  int nmex = 0;
  DTYPE dtype;
  int clen = 0, mxlen = 0, restype = 0;

  if (STYPEG(sym) == ST_MEMBER) {
    val[1] = ADDRESSG(sym); /* get offset of the ref */
    ili2 = ad_aconi(val[1]);

    /* (1)  AADD  base  ili2   */

    if (baseilix)
      ilix = ad3ili(IL_AADD, baseilix, ili2, 0);
    else {
      /* the second argument of a PARG could be a BASE ilm whose
       * symbol is a ST_MEMBER; in this case, baseilix is 0.  Need
       * to continue since we're not going to use the ili of the
       * BASE; all that's need is its length if character (see
       * exp_rte.c and its handling of IM_PARG).
       */
      ;
      ilix = ili2;
    }

    /*
     * enter a names entry for the MEMBER ILM - always use the psmem
     * field of the member ST item (sym).  In most cases, the field is
     * sym.  The exceptions possibly occur when the member is a field.
     */
    if (baseilix) {
      nmex = addnme(NT_MEM, PSMEMG(sym), basenm, 0);
    } else
      nmex = NME_UNK;
    dtype = DTYPEG(sym);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTySeqTyElement(dtype);
    if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
      restype = ILM_ISCHAR;
      clen = mxlen = ad_icon(DTyCharLength(dtype));
    }
  } else {
    if (IS_STATIC(sym) ||
        (IS_LCL(sym) && (!flg.recursive || DINITG(sym) || SAVEG(sym))))
      rcandb.static_cnt++;
    dtype = DTYPEG(sym);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTySeqTyElement(dtype);
    if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
      restype = ILM_ISCHAR;
      nmex = addnme(NT_VAR, sym, 0, 0);
      if (SCG(sym) == SC_DUMMY) {

        if (dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR) {
          if (SDSCG(sym))
            clen = exp_get_sdsc_len(sym, 0, 0);
          else {
            clen = charlen(sym);
#if DEBUG
            assert(SDSCG(sym) != 0, "create_ref:Missing descriptor", sym,
                   ERR_Severe);
#endif /* DEBUG */
          }
          mxlen = 0;
          ADDRCAND(clen, ILI_OPND(clen, 2));
        } else
            if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR) {
          clen = charlen(sym);
          mxlen = 0;
          ADDRCAND(clen, ILI_OPND(clen, 2));
        } else {
          clen = mxlen = ad_icon(DTyCharLength(dtype));
        }
        ilix = charaddr(sym);
        ADDRCAND(ilix, ILI_OPND(ilix, 2));
      } else {
        if (dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR) {
          if (SDSCG(sym)) {
            clen = exp_get_sdsc_len(sym, 0, 0);
          } else {
            clen = charlen(sym);
          }
          mxlen = 0;
          ADDRCAND(clen, ILI_OPND(clen, 2));
        } else if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR) {
          /* nondummy adjustable length character */
          if (CLENG(sym)) {
            clen = charlen(sym);
            mxlen = 0;
            ADDRCAND(clen, ILI_OPND(clen, 2));
          } else {
            clen = mxlen = ad_icon(DTyCharLength(dtype));
          }
        } else
          clen = mxlen = ad_icon(DTyCharLength(dtype));
        if (SCG(sym) == SC_CMBLK && ALLOCG(sym)) {
          /*
           * BASE is of a member which is in an allocatable common.
           * generate an indirection using the first member's address
           * and then add the offset of this member.
           */
          SPTR s;
          /*
           * REVISION: the base of the allocatable common is retrieved
           * from a compiler-created temporary.  This temporary
           * represents the word created by assem for the allocatable
           * common.  Generate an indirection through this temp.
           */
          s = getccsym('z', (int)MIDNUMG(sym), ST_VAR);
          SCP(s, SC_CMBLK);
          ADDRESSP(s, 0);
          MIDNUMP(s, MIDNUMG(sym));
          DTYPEP(s, __POINT_T);
          nmex = addnme(NT_VAR, s, 0, (INT)0);
          ili1 = ad_acon(s, (INT)0);
          ili1 = ad2ili(IL_LDA, ili1, nmex);
          ili2 = ad_aconi(ADDRESSG(sym));
          ilix = ad3ili(IL_AADD, ili1, ili2, 0);
        }

        else if (flg.smp && SCG(sym) == SC_CMBLK && IS_THREAD_TP(sym)) {
          /*
           * BASE is of a member which is in a threadprivate common.
           * generate an indirection using the threadprivate common's
           * vector and then add the offset of this member. The
           * indirection will be of the form:
           *    vector[_mp_lcpu3()]
           */
          int nm;
          int adr;
          ref_threadprivate(sym, &adr, &nm);
          ilix = adr;
        } else if (IS_THREAD_TP(sym)) {
          /*
           * BASE is a threadprivate variable; generate an
           * indirection using the threadprivate's vector.  The
           * indirection will be of the form:
           *    vector[_mp_lcpu3()]
           */
          int nm;
          int adr;
          ref_threadprivate_var(sym, &adr, &nm, 1);
          ilix = adr;
        } else {
          ilix = mk_address(sym);
        }
      }
      if (pclen)
        *pclen = clen;
      if (pmxlen)
        *pmxlen = mxlen;
      if (prestype)
        *prestype = restype;
      if (pnmex)
        *pnmex = nmex;
      return ilix;
    }
/* create the ACON ILI representing the base symbol  */
      ilix = mk_address(sym);
    if (flg.smp || XBIT(34, 0x200)) {
      if (SCG(sym) == SC_STATIC)
        sym_is_refd(sym);
    }
    /* for cuda fortran, if we use an initialized static or local,
     * call sym_is_refd */
    if (XBIT(137, 1) &&
        ((SCG(sym) == SC_STATIC || SCG(sym) == SC_LOCAL) && DINITG(sym)))
      sym_is_refd(sym);

    /*
     * create the names entry for the BASE -- don't care if the symbol is
     * a function
     */
    if (EXP_ISFUNC(sym))
      nmex = NME_UNK;
    else
        /* ST_MEMBERs may be BASE ilm for PARG 2nd argument */
        if (STYPEG(sym) != ST_MEMBER)
      nmex = addnme(NT_VAR, sym, 0, (INT)0);

      /*
       * if sym is a dummy (of type (double for 32-bit), struct, or union
       * for C) then this is really an indirection.  create a symbol which
       * represents the address of the dummy and use it to create a new
       * names entry.
       */

    if ((gbl.outlined || ISTASKDUPG(GBL_CURRFUNC)) && PARREFG(sym)) {
      if (EXP_ISINDIR(sym)) {
        int asym;
        asym = mk_argasym(sym);
      }
    }
    else if (gbl.internal > 1 && INTERNREFG(sym)) {
      if (EXP_ISINDIR(sym)) {
        int asym;
        asym = mk_argasym(sym);
      }
    }
    else
        if (EXP_ISINDIR(sym)) {
      SPTR asym = mk_argasym(sym);
      int anme = addnme(NT_VAR, asym, 0, (INT)0);
      ilix = ad2ili(IL_LDA, ilix, anme);
      ADDRCAND(ilix, anme);
    }

    if (VOLG(sym))
      nmex = NME_VOL;
    else if (SCG(sym) == SC_CMBLK && ALLOCG(sym)) {
      /*
       * BASE is of a member which is in an allocatable common.
       * generate an indirection using the first member's address
       * and then add the offset of this member.
       */
      SPTR s;
      /*
       * REVISION: the base of the allocatable common is retrieved
       * from a compiler-created temporary.  This temporary
       * represents the word created by assem for the allocatable
       * common.  Generate an indirection through this temp.
       */
      s = getccsym('z', (int)MIDNUMG(sym), ST_VAR);
      SCP(s, SC_CMBLK);
      ADDRESSP(s, 0);
      MIDNUMP(s, MIDNUMG(sym));
      DTYPEP(s, __POINT_T);
      nmex = addnme(NT_VAR, s, 0, (INT)0);
      ili1 = ad_acon(s, (INT)0);
      ili1 = ad2ili(IL_LDA, ili1, nmex);
      ili2 = ad_aconi(ADDRESSG(sym));
      ilix = ad3ili(IL_AADD, ili1, ili2, 0);
      /*
       * -x 125 32: if set, indicates that the allocatable common is
       * allocated once per execution, in which case, 'precise' nmes
       * are generated.  Otherwise, create 'via ptr' (indirection) nmes.
       */
      if (XBIT(125, 0x20))
        nmex = addnme(NT_VAR, sym, 0, (INT)0);
      else
        nmex = addnme(NT_IND, SPTR_NULL, nmex, (INT)0);
    }

    else if (SCG(sym) == SC_CMBLK && IS_THREAD_TP(sym)) {
      /*
       * BASE is of a member which is in a threadprivate common.
       * generate an indirection using the threadprivate common's
       * vector and then add the offset of this member. The
       * indirection will be of the form:
       *    vector[_mp_lcpu3()]
       */
      int nm;
      int adr;
      ref_threadprivate(sym, &adr, &nm);
      ilix = adr;
      /*nmex = addnme(NT_IND, 0, nmex, (INT) 0);*/
      /* should be safe to just use the nme of the original common
       * symbol.
       */
      nmex = addnme(NT_VAR, sym, 0, (INT)0);
    } else if (IS_THREAD_TP(sym)) {
      /*
       * BASE is a threadprivate variable; generate an indirection using
       * the threadprivate's vector.  The indirection will be of the form:
       *    vector[_mp_lcpu3()]
       */
      int nm;
      int adr;
      ref_threadprivate_var(sym, &adr, &nm, 1);
      ilix = adr;
      /*nmex = addnme( NT_IND, 0, nmex, (INT)0 );*/
      /* should be safe to just use the nme of the original common
       * symbol.
       */
      nmex = addnme(NT_VAR, sym, 0, (INT)0);
    }
  }
  if (pclen)
    *pclen = clen;
  if (pmxlen)
    *pmxlen = mxlen;
  if (prestype)
    *prestype = restype;

  if (XBIT(183, 0x80000))
    nmex = update_local_nme(nmex, sym);
  if (pnmex)
    *pnmex = nmex;
  return ilix;
} /* create_ref */

void
ll_set_new_threadprivate(int oldsptr)
{

  int newsptr = THPRVTOPTG(oldsptr);
  if (!newsptr) {
    newsptr = getnewccsym('T', stb.stg_avail, ST_VAR);
    DTYPEP(newsptr, DT_CPTR);
    THPRVTOPTP(oldsptr, newsptr);
  }

  /* This is cheating because we want to reuse the same field so we need to
   * reset
   * SCP and enclfunction to current function
   */
  if (gbl.outlined || ISTASKDUPG(GBL_CURRFUNC))
    SCP(newsptr, SC_PRIVATE);
  else
    SCP(newsptr, SC_AUTO);
  ENCLFUNCP(newsptr, GBL_CURRFUNC);
}

int
llGetThreadprivateAddr(int sptr)
{
  int addr;
  SPTR cm;
  int basenm;

  ll_set_new_threadprivate(sptr);
  cm = THPRVTOPTG(sptr);
  addr = ad_acon(cm, 0);
  basenm = addnme(NT_VAR, cm, 0, (INT)0);
  addr = ad2ili(IL_LDA, addr, basenm);

  return addr;
}

int
getThreadPrivateTp(int sptr)
{
  int tpv = sptr;

  tpv = MIDNUMG(sptr);

  if (SCG(sptr) == SC_BASED && POINTERG(sptr)) {
    int pv = MIDNUMG(sptr);
    if (SCG(pv) == SC_CMBLK) {
      tpv = MIDNUMG(MIDNUMG(pv));
    } else {
      tpv = MIDNUMG(pv);
    }
  } else if (SCG(sptr) == SC_CMBLK) {
    sptr = MIDNUMG(sptr);
    tpv = MIDNUMG(sptr);
  }

  return tpv;
}

/** \brief Have a reference to a member of a threadprivate common.
 *
 * Generate an indirection using the threadprivate common's vector and
 * then add the offset of this member.  The actual address computation is:
 *    vector[_mp_lcpu3()] + offset(member)
 */
void
ref_threadprivate(int cmsym, int *addr, int *nm)
{
  SPTR vector;
  int basenm;
  int ili1;
  int ili2;

  /* compute the base address of vector */
  vector = MIDNUMG(cmsym);
  /* at this point, vector locates the common block */
  vector = MIDNUMG(vector);
  basenm = addnme(NT_VAR, vector, 0, (INT)0);
  ili1 = ad_acon(vector, (INT)0);

  if (XBIT(69, 0x80)) {
    /* compute the base address of vector */
    vector = MIDNUMG(cmsym);
    /* at this point, vector locates the common block */
    vector = MIDNUMG(vector);
    basenm = addnme(NT_VAR, vector, 0, (INT)0);
    ili1 = ad_acon(vector, (INT)0);
    ili1 = ad2ili(IL_LDA, ili1, basenm);
  } else {
    ili1 = llGetThreadprivateAddr(vector);
  }

  /* add in the common member's offset */
  ili2 = ad_aconi(ADDRESSG(cmsym));
  ili1 = ad3ili(IL_AADD, ili1, ili2, 0);

  *addr = ili1;
  *nm = basenm;
}

/** \brief Have a reference to a Fortran or C threadprivate variable.
 *
 * Generate an indirection using the threadprivate's vector.  The actual
 * address computations is:
 *    vector[_mp_lcpu3()]
 * mark : 1 - mark TPLNKP , and add it go gbl.threadprivate : this is normal
 * processing. When calling this function later on, during exception fixup,
 * call with mark = 0
 */
void
ref_threadprivate_var(int cmsym, int *addr, int *nm, int mark)
{
  SPTR vector;
  int basenm;
  int ili1;
  int ili2;

  /* compute the base address of vector */
  vector = MIDNUMG(cmsym);
  basenm = addnme(NT_VAR, vector, 0, (INT)0);
  ili1 = ad_acon(vector, (INT)0);

  if (XBIT(69, 0x80)) {
    vector = MIDNUMG(cmsym);
    basenm = addnme(NT_VAR, vector, 0, 0);
    ili1 = ad_acon(vector, (INT)0);
    ili1 = ad2ili(IL_LDA, ili1, basenm);
  } else {
    ili1 = llGetThreadprivateAddr(vector);
  }

  if (DESCARRAYG(cmsym)) {
    /*
     * for a f90 pointer, subscripting of the TP vector gives the address
     * of the thread's copy of the internal pointer variable; the
     * descriptor is 2 pointer units away from the pointer variable
     */
    ili2 = ad_acon(SPTR_NULL, 2 * size_of(DT_ADDR));
    ili1 = ad3ili(IL_AADD, ili1, ili2, 0);
  }

  *addr = ili1;
  *nm = basenm;

}

void
exp_pure(SPTR extsym, int nargs, ILM *ilmp, int curilm)
{
#define MAX_PUREARGS 2
  int args[MAX_PUREARGS];
  int cili;
  int ilix;
  int n, i;
  int ilmx;
  ILM *ilmpx;
  int first_arg_index;

  if (nargs > MAX_PUREARGS)
    return;

  first_arg_index = 1 + ilm_callee_index(ILM_OPC(ilmp));

  n = nargs;
  i = first_arg_index;
  while (n--) {
    ilmx = ILM_OPND(ilmp, i); /* locates ARG ilm */
    ilmpx = (ILM *)(ilmb.ilm_base + ilmx);
    ilmx = ILM_OPND(ilmpx, 2);
    args[i - first_arg_index] = ILI_OF(ilmx);
    i++;
  }
  cili = ILI_OF(curilm);
  switch (ILI_OPC(cili)) {
  case IL_DFRAR:
    if (nargs == 0) {
      cili = jsr2qjsr(cili);
      ilix = ad_acon(extsym, 0);
      ilix = ad2ili(IL_APURE, ilix, cili);
      ILM_RESULT(curilm) = ilix;
      break;
    } else if (nargs == 1) {
      switch (IL_RES(ILI_OPC(args[0]))) {
      case ILIA_AR:
        cili = jsr2qjsr(cili);
        ilix = ad_acon(extsym, 0);
        ilix = ad3ili(IL_APUREA, ilix, args[0], cili);
        ILM_RESULT(curilm) = ilix;
        break;
      case ILIA_IR:
        cili = jsr2qjsr(cili);
        ilix = ad_acon(extsym, 0);
        ilix = ad3ili(IL_APUREI, ilix, args[0], cili);
        ILM_RESULT(curilm) = ilix;
        break;
      default:
        break;
      }
    }
    break;

  case IL_DFRIR:
    if (nargs == 0) {
      cili = jsr2qjsr(cili);
      ilix = ad_acon(extsym, 0);
      ilix = ad2ili(IL_IPURE, ilix, cili);
      ILM_RESULT(curilm) = ilix;
    } else if (nargs == 1) {
      switch (IL_RES(ILI_OPC(args[0]))) {
      case ILIA_AR:
        cili = jsr2qjsr(cili);
        ilix = ad_acon(extsym, 0);
        ilix = ad3ili(IL_IPUREA, ilix, args[0], cili);
        ILM_RESULT(curilm) = ilix;
        break;
      case ILIA_IR:
        cili = jsr2qjsr(cili);
        ilix = ad_acon(extsym, 0);
        ilix = ad3ili(IL_IPUREI, ilix, args[0], cili);
        ILM_RESULT(curilm) = ilix;
        break;
      default:
        break;
      }
    }
    break;

  default:
    break;
  }
}

static int
jsr2qjsr(int dfili)
{
  int New;
  int cl;
#if DEBUG
  assert(ILI_OPC(dfili) == IL_DFRIR || ILI_OPC(dfili) == IL_DFRAR,
         "jsr2qjsr:dfr ili expected", dfili, ERR_unused);

#endif
  New = dfili;
  cl = ILI_OPND(dfili, 1);
  if (ILI_OPC(cl) == IL_JSR) {
    New = ad2ili(IL_QJSR, ILI_OPND(cl, 1), ILI_OPND(cl, 2));
    New = ad2ili(ILI_OPC(dfili), New, ILI_OPND(dfili, 2));
  }
  return New;
}

  /***************************************************************/

