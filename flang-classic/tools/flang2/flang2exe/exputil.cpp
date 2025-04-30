/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Expander utility routines
 */

#include "exputil.h"
#include "expreg.h"
#include "dinit.h"
#include "dinitutl.h"
#include "dtypeutl.h"
#include "llassem.h"
#include "ilm.h"
#include "ilmtp.h"
#include "fih.h"
#include "ili.h"
#include "iliutil.h"
#define EXPANDER_DECLARE_INTERNAL
#include "expand.h"
#include "machar.h"
#include "regutil.h"
#include "machreg.h"
#include "symfun.h"

static void propagate_bihflags(void);
static void flsh_saveili(void);

#define DO_PFO (XBIT(148, 0x1000) && !XBIT(148, 0x4000))

#ifdef __cplusplus
inline SPTR getSptr_bnd_con(ISZ_T i) {
  return static_cast<SPTR>(get_bnd_con(i));
}
#else
#define getSptr_bnd_con get_bnd_con
#endif

/** \brief Make an argument list
 *
 *  Create a compiler generated array temporary for an argument list.
 *  its size is cnt and its dtype is dt.  NOTE that the caller may modify
 *  the size of this array (a target may require alignment other than int).
 *  mkarglist guarantees reuse as long as expb.arglcnt.next is reset
 *  to expb.arglcnt.start.
 */
void
mkarglist(int cnt, DTYPE dt)
{

  DTYPE dtype;
  ADSC *ad;
  static INT ival[2];

  ival[1] = cnt;
  expb.arglist = getccsym('a', expb.arglcnt.next++, ST_ARRAY);
  dtype = DTYPEG(expb.arglist);
  if (dtype) {
    ad = AD_DPTR(dtype);
    if (ival[1] > ad_val_of(AD_NUMELM(ad)))
      AD_NUMELM(ad) = AD_UPBD(ad, 0) = getSptr_bnd_con(ival[1]);
  } else {
    SCP(expb.arglist, SC_AUTO);
    STYPEP(expb.arglist, ST_ARRAY);
    dtype = get_array_dtype(1, dt);
    DTYPEP(expb.arglist, dtype);

    ad = AD_DPTR(dtype);
    AD_MLPYR(ad, 0) = stb.i1;
    AD_LWBD(ad, 0) = stb.i1;
    AD_UPBD(ad, 0) = getSptr_bnd_con(ival[1]);
    AD_NUMDIM(ad) = 1;
    AD_SCHECK(ad) = 0;
    AD_ZBASE(ad) = stb.i1;
    AD_NUMELM(ad) = AD_UPBD(ad, 0); /* numelm must be set ater numdim */
  }

  if (expb.arglcnt.max < expb.arglcnt.next)
    expb.arglcnt.max = expb.arglcnt.next;
}

/** \brief Create a block during expand
 */
void
cr_block(void)
{
  expb.curbih = exp_addbih(expb.curbih);
  BIH_LINENO(expb.curbih) = expb.curlin ? expb.curlin : gbl.lineno;
  BIH_GUARDEE(expb.curbih) = expb.isguarded >= 1 ? 1 : 0;
  expb.flags.bits.noblock = 0;
  ILT_NEXT(0) = 0;
  ILT_PREV(0) = 0;
  if (expb.curlin && expb.flags.bits.dbgline) {
    expb.curilt =
        addilt(0, ad2ili(IL_QJSR, mkfunc("dbg_i_line"), ad1ili(IL_NULL, 0)));
    ILT_DBGLINE(expb.curilt) = 1;
  } else
    expb.curilt = 0;
  expb.curlin = 0;
  if (EXPDBG(8, 32))
    fprintf(gbl.dbgfil, "---cr_block: bih %d, ILM line %d\n", expb.curbih,
            gbl.lineno);
}

/** \brief Write a block during expand
 */
void
wr_block(void)
{
  int iltx, opc = 0;

  if (EXPDBG(8, 64))
    fprintf(gbl.dbgfil, "%6d ilm words in block %6d\n", expb.ilm_words,
            expb.curbih);
  expb.ilm_words = 0;
  iltx = ILT_PREV(0);
  if (iltx != 0)
    opc = ILI_OPC(ILT_ILIP(iltx));
  if (iltx == 0 || !ILT_BR(iltx) || (opc != IL_JMP && opc != IL_QSWITCH &&
                                     opc != IL_JMPMK &&
                                     opc != IL_JMPM))
    BIH_FT(expb.curbih) = 1;

  propagate_bihflags();

  expb.flags.bits.callfg |= bihb.callfg;
  wrilts(expb.curbih); /* wrilts zeros bihb.callfg & qjsrfg */
#ifdef BIH_ASM
  bihb.gasm = 0;
#endif
  if (EXPDBG(8, 32))
    fprintf(gbl.dbgfil, "---wr_block: bih %d, ILM line %d\n", expb.curbih,
            gbl.lineno);
  expb.flags.bits.noblock = 1;
}

static void
propagate_bihflags(void)
{
  /* Set of flags for which it's necessary to propagate from iltb
   * to bihb to BIH because the write of an ilt may be deferred.
   */
  BIH_EX(expb.curbih) = bihb.callfg;
  BIH_LDVOL(expb.curbih) = bihb.ldvol;
  BIH_STVOL(expb.curbih) = bihb.stvol;
  BIH_QJSR(expb.curbih) = bihb.qjsrfg;
#ifdef BIH_ASM
  BIH_ASM(expb.curbih) = bihb.gasm;
#endif

  /*
   * Set of flags where it's known that a block has been created
   * because of the context implied by the flags.
   */
  BIH_PAR(expb.curbih) |= bihb.parfg;
  BIH_CS(expb.curbih) |= bihb.csfg;
  BIH_PARSECT(expb.curbih) |= bihb.parsectfg;
  BIH_TASK(expb.curbih) |= bihb.taskfg;
}

/** \brief Close current block and write out.
 */
void
flsh_block(void)
{
  flsh_saveili();
  wr_block();
}

/** \brief Put out expb.saveili if necesssary.
 *
 * May be an unadded ili if expb.flags.bits.waitlbl is set
 */
static void
flsh_saveili(void)
{
  int savefg;  /* save for the ilt call flag	 */
  char ldvol;  /* save for the ilt ldvol flag	 */
  char stvol;  /* save for the ilt stvol flag	 */
  bool qjsrfg; /* save for the ilt qjsrfg flag	 */

  if (expb.flags.bits.waitlbl) {

    /*
     * waiting for a label ilm; curilt will become the end of the current
     * block if the opt level is not 1 -- i.e., the current block is
     * written out and a new block is created.  And, the ilt for saveili
     * becomes the first ilt of this block.  For opt level 1, an ilt for
     * saveili is added to the current block
     *
     * NOTE:  this does not happen at opt=0 -- this flag is not set (see the
     * end of this routine)
     */

    expb.flags.bits.waitlbl = 0;
    if (flg.opt != 1) {
      if (EXPDBG(8, 32))
        fprintf(gbl.dbgfil, "---flsh_saveili: wait end, curilt %d\n",
                expb.curilt);

      /* write out the block with curilt as its last ilt */

      wr_block();

      /*
       * create a new block - saveili will be the first ilt of the new
       * block
       */
      cr_block();
    }
    /* append the ilt to the current block  */

    if (EXPDBG(8, 32))
      fprintf(gbl.dbgfil, "---flsh_saveili: wait add, curilt %d, saveili %d\n",
              expb.curilt, expb.saveili);
    savefg = iltb.callfg;
    ldvol = iltb.ldvol;
    stvol = iltb.stvol;
    qjsrfg = iltb.qjsrfg;
    iltb.callfg = 0;
    iltb.ldvol = 0;
    iltb.stvol = 0;
    iltb.qjsrfg = false;
    expb.curilt = addilt(expb.curilt, expb.saveili);
    iltb.callfg = savefg;
    iltb.ldvol = ldvol;
    iltb.stvol = stvol;
    iltb.qjsrfg = qjsrfg;
  }
}

/** \brief Check for end of ILT block
 *
 * Check if the current block is at its end. newili locates an
 * ili tree which represents an ili statement.
 */
void
chk_block(int newili)
{
  int ili;     /* ili of the current ilt	 */
  ILI_OP opc;  /* ili opcode of the current ilt */
  int c_noprs; /* # of operands for conditional branch */
  int c_lb;    /* label of conditional branch  */
  int u_lb;    /* label of unconditional branch */
  int old_lastilt;

  if (newili == 0)
    /* seems odd that newili is 0, but this can happen if one is adding
     * a branch ili -- iliutil could change the branch into a nop.
     * The caller could check the ili first -- there are many cases
     * where we do -- but it's something that can be easily overlooked.
     */
    return;

  flsh_saveili(); /* write out saveili  if waitlbl is set. (see
                   * below) */

  if (!ILT_BR(expb.curilt) && (flg.opt < 2 || !ILT_CAN_THROW(expb.curilt))) {

    /* The current end of the block is not a branch and [at -O2] cannot throw.
     * Just create a new ilt and add it to the block.  Note that if the block
     * is null (expb.curilt is zero), the flags of ilt 0 are zero.
     */
    if (EXPDBG(8, 32))
      fprintf(gbl.dbgfil, "---chk_block: curilt %d not br, add newili %d\n",
              expb.curilt, newili);
    old_lastilt = expb.curilt;  
    expb.curilt = addilt(expb.curilt, newili);
    /* addilt does not update BIH_ILTLAST().  We need to do it here: */
    if (expb.curbih && BIH_ILTLAST(expb.curbih) == old_lastilt) {
      BIH_ILTLAST(expb.curbih) = expb.curilt;
    }
    return;
  }
  /* the current end of the block is some sort of branch,
     call that can throw, or store of result of a call
     that can throw. */

  ili = ILT_ILIP(expb.curilt);
  opc = ILI_OPC(ili);
  if (opc == IL_JMP) {

    /*
     * the current end of the block is an unconditional branch. newili is
     * unreachable, don't add it
     */
    opc = ILI_OPC(newili);
    if (IL_TYPE(opc) == ILTY_BRANCH &&
        IL_OPRFLAG(opc, ilis[opc].oprs) == ILIO_SYM)
      RFCNTD(ILI_OPND(newili, ilis[opc].oprs));
    iltb.callfg = 0;
    if (EXPDBG(8, 32))
      fprintf(gbl.dbgfil, "---chk_block: newili %d not reached\n", newili);
    return;
  }
  /* the current end of the block is a conditional branch,
     or something that can throw. */

  if (ILI_OPC(newili) != IL_JMP || ILT_CAN_THROW(expb.curilt)) {

    /* the new ili is not an unconditional branch, or the new ili is
       an IL_JMP just past a potential throw point. */

    if (flg.opt == 1 && !XBIT(137, 1) && !XBIT(163, 1))
      /* create an extended basic block -- add a new ilt  */

      expb.curilt = addilt(expb.curilt, newili);
    else {

      /*
       * the current block is at its end; write it out, create a new
       * block with its first ilt locating newili
       */
      wr_block();
      cr_block();
      expb.curilt = addilt(expb.curilt, newili);
    }
    if (EXPDBG(8, 32))
      fprintf(gbl.dbgfil, "---chk_block: add newili %d\n", newili);
    return;
  }
  /*
   * the current end of the block is a conditional branch and the new ili
   * is an unconditional branch
   */
  c_noprs = ilis[opc].oprs;
  c_lb = ILI_OPND(ili, c_noprs);
  u_lb = ILI_OPND(newili, 1);
  if (opc != IL_JMPM &&
      opc != IL_JMPA &&
      opc != IL_JMPMK &&
      u_lb == c_lb) {

    /*
     * the labels are the same; search the current ilt and create new
     * ilts for any procedures found.  These ilts are added before the
     * current ilt
     */
    (void)reduce_ilt((int)ILT_PREV(expb.curilt), ili);

    /*
     * the current ilt is changed to be the unconditional branch; the
     * label's reference count is also decremented
     */
    if (EXPDBG(8, 32))
      fprintf(gbl.dbgfil,
              "---chk_block: uncond/cond, newili %d, to same label\n", newili);
    ILT_ILIP(expb.curilt) = newili;
    RFCNTD(ILI_OPND(newili, 1));
    return;
  }
  /*
   * the current end of the block is a conditional branch and the new ili
   * is an unconditional branch whose labels are different. the checking of
   * this block is delayed just in case a label ilm is processed before any
   * other ilt producing ilms.  This is done to catch cases of the form:
   *     if <cond> goto x;
   *     goto y;
   *  x: --- NOTE:  this does not happen at opt 0;
   * the current block is written out (with previlt as the last ilt) and
   * previlt becomes the first ilt of the new block.
   */
  if (flg.opt != 0) {
    if (EXPDBG(8, 32))
      fprintf(gbl.dbgfil,
              "---chk_block: uncond/cond, newili %d, to diff label\n", newili);
    if (opc != IL_JMPM &&
        opc != IL_JMPA &&
        opc != IL_JMPMK &&
        ILIBLKG(c_lb) == 0 && ILIBLKG(u_lb)) {
      /* conditional branch is a forward branch; unconditional branch
       * is a backward branch
       */
      ILT_ILIP(expb.curilt) = compl_br(ili, u_lb);
      newili = ad1ili(IL_JMP, c_lb);
      if (EXPDBG(8, 32))
        fprintf(gbl.dbgfil, "---chk_block: swap lbls %d %d, c_br %d, u_br %d\n",
                c_lb, u_lb, (int)ILT_ILIP(expb.curilt), newili);
    }
    expb.flags.bits.waitlbl = 1;
    expb.saveili = newili;

  } else {
    wr_block(); /* the current block		 */
    cr_block();
    expb.curilt = addilt(expb.curilt, newili);
  }
}

/** \brief Like chk_block, but suppress CAN_THROW flag.
 *
 * When a call can throw and defines two result registers, we have an ad-hoc
 * rule that only the second store is marked as "can throw".  This utility
 * is useful for ensuring that the first store is not marked "can throw".
 */
void
chk_block_suppress_throw(int newili)
{
  chk_block(newili);
  ILT_SET_CAN_THROW(expb.curilt, 0);
}

/** \brief Check an ILM which has been evaluated
 *
 * This routine checks an ILM which has already been evaluated
 * (i.e., the ILM is referenced again).  Depending on the type
 * of the ili which defines this ILM, certain actions may occur
 * to "redefine" the ILM (i.e., create a new ILI for the ILM).
 *
 * \param ilmx -- ILM index of the ILM evaluated
 * \param ilix -- ILI index of the ILI for the ILM
 */
int
check_ilm(int ilmx, int ilix)
{
  int cse;    /* cse ILI                                */
  SPTR sym;   /* symbol table index                     */
  int base,   /* address ILI                            */
      nme,    /* names entry                            */
      blk,    /* bih index                              */
      iltx;   /* ilt index                              */
  ILI_OP opc; /* opcode of ilix                         */

  int saveilix = ilix;
  switch (IL_TYPE(opc = ILI_OPC(ilix))) {

  case ILTY_CONS:
    /* a constant ILI is okay to re-use  */
    if (EXPDBG(8, 2))
      fprintf(gbl.dbgfil, "check_ilm, ILM const: ilm %d, result ili %d\n", ilmx,
              ilix);
    return ilix;

  case ILTY_ARTH:
  case ILTY_LOAD:
  case ILTY_DEFINE:
  case ILTY_MOVE:
/*
 * these results represent an expression which may contain
 * side-effect operations.  The new result is a CSE ILI of ilix
 */
    /* assertion: for fortran the only side-effects which may occur are
     * those due to function calls;  however, there are cases where
     * it's necessary that several ili statements "belong" to the
     * statement.  To represent this, the sequence will begin with
     * a pseudo store, followed by statements which have cse uses.
     * For this case, the assertion (fortran-only) is that this
     * will only occur for pseudo stores and stores.
     */
    cse = ilix;
    if (EXPDBG(8, 2))
      fprintf(gbl.dbgfil,
              "check_ilm, ILM expr: ilm %d, old ili %d, cse ili %d\n", ilmx,
              ilix, cse);
    /*
     * no longer have:
     *   !BIH_QJSR(expb.curbih) &&
     * leader block may not yet be created
     */
    if (!iltb.qjsrfg && qjsr_in(ilix)) {
      iltb.qjsrfg = true;
      if (EXPDBG(8, 2))
        fprintf(gbl.dbgfil, "check_ilm - qjsr_in(%d)\n", ilix);
    }
    return cse;

  case ILTY_OTHER:
#ifdef ILTY_PSTORE
  case ILTY_PSTORE:
#endif
#ifdef ILTY_PLOAD
  case ILTY_PLOAD:
#endif
    /* handle FREEIR... ili with ILTY_STORE */
    if (!is_freeili_opcode(opc)) {
      /* not a FREE ili */
      return ilix;
    }
    goto like_store;

  case ILTY_STORE:
/*
 * assertion: for fortran the only side-effects which may occur are
 * those due to function calls;  however, there are cases where
 * it's necessary that several ili statements "belong" to the
 * statement.  To represent this, the sequence will begin with
 * a pseudo store, followed by statements which have cse uses.
 * For this case, the assertion (fortran-only) is that this
 * will only occur for pseudo stores and stores.
 */

  like_store:
    /*** __OLDCSE ***/

    cse = ad_cse((int)ILI_OPND(ilix, 1));

    if ((blk = ILM_BLOCK(ilmx)) == expb.curbih) {
      if (ILM_OPC((ILM *)(ilmb.ilm_base + ilmx)) == IM_PSEUDOST) {
        /* We generate a pseudo store for postfix expressions.
         * If the value prior to incrementing/decrementing is needed
         * later by an arithmetic op, then we cannot safely CSE it.
         * For example (from PH lang test): i++ != 2 ? 0 : i; In
         * this case, a comparison followed by a branch needs the
         * old value.
         */
        int len, opc2, i, found_use, found_arth;
        int use = ILM_OPND((ILM *)(ilmb.ilm_base + ilmx), 2);
        int ilmx2 = ilmx + ilms[IM_PSEUDOST].oprs + 1;
        for (found_use = 0, found_arth = 0; ilmx2 < expb.nilms; ilmx2 += len) {
          opc2 = ILM_OPC((ILM *)(ilmb.ilm_base + ilmx2));
          len = ilms[opc2].oprs + 1;
          if (IM_VAR(opc2))
            len += ILM_OPND((ILM *)(ilmb.ilm_base + ilmx2), 1);
          for (i = 1; i < len; ++i) {
            if (IM_OPRFLAG(opc2, i) == OPR_LNK &&
                ILM_OPND((ILM *)(ilmb.ilm_base + ilmx2), i) == use) {
              found_use = 1;
            } else if (found_use && IM_OPRFLAG(opc2, i) == OPR_LNK &&
                       ILM_OPND((ILM *)(ilmb.ilm_base + ilmx2), i) == ilmx) {
              /* Convert the pseudo store to a real
               * store on comparisons and other artimetics only.
               */
              if (IM_TYPE(opc2) != IMTY_ARTH) {
                found_arth = 0;
                goto break_out;
              }
              found_arth = 1;
            }
          }
        }
      break_out:
        if (found_arth)
          goto conv_pseudo_st;

      } /* end if (ILM_OPC(...) == IM_PSEUDOST) */

      /* reference is in the current block  */

      if (EXPDBG(8, 2))
        fprintf(gbl.dbgfil,
                "check_ilm, store re-used: ilm %d, old ili %d, cse ili %d\n",
                ilmx, ilix, cse);
      return cse;
    } else {
      /* The reference is in the current block if we jumped here
       * from above (see the lines after "break_out:"), otherwise
       * it's across block boundaries.
       */
      int save_iltb_callfg, save_iltb_ldvol, save_iltb_stvol;
      bool save_iltb_qjsrfg;

    conv_pseudo_st:
      /* JHM (8 Dec 2011) bug-fix:
       * This is quite complicated so I'll explain the logic in
       * full detail.  We're about to read in the ILTs of a
       * block 'blk' (which may be 'expb.curbih' itself or a
       * different block), then possibly add a new store ILT to
       * it, then read in the ILTs of 'expb.curbih' again.
       * Before doing this we must do the following:
       *
       * (1) Save the values of 'iltb.{x}', where {x} = {callfg,
       * ldvol, stvol, qjsrfg}, to be restored later.
       *
       * (2) Set bihb.{x} |= iltb.{x}
       *
       * (3) Call 'propagate_bihflags()' to copy each bihb.{x}
       * value to the corresponding BIH_{X}( expb.curbih ) field.
       *
       * Then at the end, after reading in 'expb.curbih's ILTs again:
       *
       * (4) Copy the saved values back to 'iltb.{x}'.
       *
       * Normally (2) would be performed by the next call to
       * 'addilt()' and (3) by the next call to 'wr_block()'.
       * However, because we call 'rdilts( blk )' and possibly
       * add an ILT to 'blk', the above fields may be over-written
       * before these actions can take place, as follows:
       *
       * -- 'rdilts( blk )' sets:
       *    bihb.{x} = BIH_{X}( blk )
       * thus (potentially) over-writing the current values of bihb.{x}.
       *
       * -- Then 'addilt()' (if it's called) sets:
       *    bihb.{x} |= iltb.{x}
       *    iltb.{x} = 0
       * thus over-writing the current values of iltb.{x}.
       *
       * -- Finally 'rdilts( expb.curbih )' sets:
       *    bihb.{x} = BIH_{X}( expb.curbih )
       * which is only correct if BIH_{X}( expb.curbih ) contains
       * the correct values, i.e. if actions (2) and (3) above
       * have already been performed.
       *
       * By performing actions (2) and (3) we ensure that
       * bihb.{x} and BIH_{X}( expb.curbih ) are correct at the
       * end of this code, and (1) and (4) ensure that the
       * current values are restored to 'iltb.{x}'.
       */
      bihb.callfg |= (save_iltb_callfg = iltb.callfg);
      bihb.ldvol |= (save_iltb_ldvol = iltb.ldvol);
      bihb.stvol |= (save_iltb_stvol = iltb.stvol);
      bihb.qjsrfg |= (save_iltb_qjsrfg = iltb.qjsrfg);

      iltb.callfg = 0; /* ...it's used by 'addilt()' to set ILT_EX() */
      propagate_bihflags();
      fihb.currfindex = BIH_FINDEX(expb.curbih);
      fihb.currftag = BIH_FTAG(expb.curbih);
      wrilts(expb.curbih); /* write out the current block	and	 */
      rdilts(blk);         /* read in the block of the store	 */

      sym = mkrtemp_sc(cse, expb.sc);
      base = ad_acon(sym, (INT)0);
      nme = addnme(NT_VAR, sym, 0, (INT)0);

      for (iltx = ILT_PREV(0); iltx != 0; iltx = ILT_PREV(iltx)) {
        if (ILT_ILIP(iltx) == ilix) {
          switch (opc) {
          case IL_ST:
            ilix = ad4ili(IL_ST, cse, base, nme, ILI_OPND(ilix, 4));
            break;
          case IL_FREEIR:
            ilix = ad4ili(IL_ST, cse, base, nme, MSZ_WORD);
            break;
          case IL_STKR:
            ilix = ad4ili(IL_STKR, cse, base, nme, ILI_OPND(ilix, 4));
            break;
          case IL_FREEKR:
            ilix = ad4ili(IL_STKR, cse, base, nme, MSZ_I8);
            break;
          case IL_STA:
          case IL_FREEAR:
            ilix = ad3ili(IL_STA, cse, base, nme);
            ILM_NME(ilmx) = addnme(NT_IND, SPTR_NULL, nme, (INT)0);
            break;
          case IL_STSP:
          case IL_FREESP:
            ilix = ad4ili(IL_STSP, cse, base, nme, MSZ_F4);
            break;
          case IL_STDP:
          case IL_FREEDP:
            ilix = ad4ili(IL_STDP, cse, base, nme, MSZ_F8);
            break;
          case IL_STSCMPLX:
          case IL_FREECS:
            ilix = ad4ili(IL_STSCMPLX, cse, base, nme, MSZ_F8);
            break;
          case IL_STDCMPLX:
          case IL_FREECD:
            ilix = ad4ili(IL_STDCMPLX, cse, base, nme, MSZ_F16);
            break;
#ifdef LONG_DOUBLE_FLOAT128
          case IL_FLOAT128ST:
          case IL_FLOAT128FREE:
            ilix = ad4ili(IL_FLOAT128ST, cse, base, nme, MSZ_F16);
            break;
#endif /* LONG_DOUBLE_FLOAT128 */
          default:
            interr("check_ilm: illegal store", ilix, ERR_Severe);
            goto wr_out;
          }
          ADDRCAND(ilix, nme);
          iltx = addilt(iltx, ilix);
          ilix = ad_load(ilix);
          if (ilix) {
            if (EXPDBG(8, 2))
              fprintf(gbl.dbgfil,
                      "check_ilm: store across block, ilm %d, ilt %d, ili %d\n",
                      ilmx, iltx, ilix);
            ADDRCAND(ilix, nme);
          } else {
            ilix = ILT_ILIP(iltx);
            interr("check_ilm: illegal store1", ilix, ERR_Severe);
          }
          goto wr_out;
        }
      }

      /* no store found; just use cse as the result  */

      ilix = cse;
      if (EXPDBG(8, 2))
        fprintf(gbl.dbgfil,
                "check_ilm: store not found, ilm %d, ilt %d, ili %d\n", ilmx,
                iltx, ilix);

    wr_out:
      wrilts(blk);         /* write out the modified block */
      rdilts(expb.curbih); /* read back in the current block */

      iltb.callfg = save_iltb_callfg;
      iltb.ldvol = save_iltb_ldvol;
      iltb.stvol = save_iltb_stvol;
      iltb.qjsrfg = save_iltb_qjsrfg;
    }
    return ilix;

  default:
    if (EXPDBG(8, 2))
      fprintf(gbl.dbgfil,
              "check_ilm: bad reference, ilm %d(%s), ili %d, iliopc %d\n", ilmx,
              ilms[ILM_OPC((ILM *)(ilmb.ilm_base + ilmx))].name, ilix, opc);
    interr("check_ilm: bad reference", ilmx, ERR_Severe);
  }

  return saveilix;
}

/***************************************************************/

#if defined(DINIT_FUNCCOUNT)
void
put_funccount(void)
{
  dinit_put(DINIT_FUNCCOUNT, gbl.func_count);
} /* put_funccount */
#endif

/** \brief Make a switch list for the intrinsic
 *
 * The switch list consists of the number of cases which occurred, the
 * default label, and followed by the pairs (in sorted order based on
 * case values) of case values and their respective case labels.
 */
int
mk_swlist(INT n, SWEL *swhdr, int doinit)
{
  SPTR sym;
  int i;
  SWEL *swel;
  DTYPE dtype;

  sym = getccsym('J', expb.swtcnt++, ST_ARRAY); /* get switch array */
  SCP(sym, SC_STATIC);
  i = dtype = get_type(3, TY_ARRAY, DT_INT);
  DTYPEP(sym, dtype);
  DTySetArrayDesc(dtype, get_bnd_con(2 * (n + 1)));

  if (doinit) {
    /* initialized this array with the switch list  */
    DINITP(sym, 1);

#if defined(DINIT_FUNCCOUNT)
      put_funccount();
#endif
    dinit_put(DINIT_LOC, sym);

    dinit_put(DT_INT, n); /* number of cases */
    dinit_put(DINIT_LABEL, (INT)swhdr->clabel); /* default label   */
    i = swhdr->next;
    do {
      swel = switch_base + i;
      dinit_put(DT_INT, swel->val);              /* case value */
      dinit_put(DINIT_LABEL, (INT)swel->clabel); /* case label */
      i = swel->next;
    } while (i != 0);
  }

  return ad_acon(sym, 0);
}

int
access_swtab_base_label(int base_label, int sptr, int flag)
{

  /* Store a list of jump table sptr's and their branch label. Used
   * on the ST100 for the enhanced method 3 scheme.
   * if flag == -2 then we are just access the routine's mode (GP16/GP32).
   * if flag == -1 then we are just accessing the routine sptr
   * If flag ==  1 then we are just accessing the value.
   * If flag ==  0 then we are adding a new value.
   * If flag ==  2 then we are also removing the record from the list.
   * Returns 0 if not found, else branch_label_sptr.
   */

  typedef struct swtab_branch_label {
    int branch_label;
    int sptr_swtab;
    int sptr_routine;
    int mode;
    struct swtab_branch_label *next;
  } swtab_branch_label;

  int branch_label_sptr = 0, mode = 0, routine_sptr = 0;

  static swtab_branch_label swtab_info = {0, 0, 0, 0, 0};

  swtab_branch_label *curr, *prev;

  if (flag != 0) {

    prev = &swtab_info;
    for (curr = swtab_info.next; curr; curr = curr->next) {

      if (curr->sptr_swtab == sptr) {

        branch_label_sptr = curr->branch_label;
        mode = curr->mode;
        routine_sptr = curr->sptr_routine;
        if (flag == 2) {
          /* Remove record from list */
          prev->next = curr->next;
          FREE(curr);
        }

        if (flag == -1) {
          return routine_sptr;
        } else if (flag == -2) {
          return mode;
        } else {
          return branch_label_sptr;
        }
      }
      prev = curr;
    }

  } else {
    NEW(curr, swtab_branch_label, sizeof(swtab_branch_label));
    curr->branch_label = base_label;
    curr->sptr_swtab = sptr;
    curr->sptr_routine = gbl.currsub;
    curr->mode = 0;
    curr->next = swtab_info.next;
    swtab_info.next = curr;
    branch_label_sptr = base_label;
  }

  return branch_label_sptr;
}

int
access_swtab_case_label(int case_label, int *case_val, int sptr, int flag)
{
  /* Store a list of jump table sptr's and their case labels. Used
   * on the ST100 for the constant time method (an inline jump table).
   * If flag == 1 then we are just accessing the value.
   * If flag == 0 then we are adding a new value.
   * If flag == 2 then we are also removing the record from the list.
   * Returns 0 if not found, else case_label_sptr.
   */

  typedef struct swtab_case_label {
    int case_label;
    int case_val;
    int sptr_swtab;
    struct swtab_case_label *next;
  } swtab_case_label;

  int case_label_sptr = 0;

  static swtab_case_label swtab_info = {0, 0, 0, 0};

  swtab_case_label *curr, *prev;

  if (flag) {

    prev = &swtab_info;
    for (curr = swtab_info.next; curr; curr = curr->next) {

      if (curr->sptr_swtab == sptr) {

        case_label_sptr = curr->case_label;
        *case_val = curr->case_val;
        if (flag == 2) {
          /* Remove record from list */
          prev->next = curr->next;
          FREE(curr);
        }
        return case_label_sptr;
      }
      prev = curr;
    }

  } else {
    NEW(curr, swtab_case_label, sizeof(swtab_case_label));
    curr->case_label = case_label;
    curr->sptr_swtab = sptr;
    curr->next = swtab_info.next;
    curr->case_val = *case_val;
    swtab_info.next = curr;
    case_label_sptr = case_label;
  }

  return case_label_sptr;
}


#define DINITSWTAB_put(c, s)
#define DINITSWTAB_put2(c, s, t)

/** \brief Make a switch address table
 */
int
mk_swtab(INT n, SWEL *swhdr, int deflab, int doinit)
{
  int sym;
  int i;
  INT case_val;
  SWEL *swel;
  DTYPE tabdtype = DT_CPTR;

  sym = getccsym('J', expb.swtcnt++, ST_PLIST); /* get switch array */
  DTYPEP(sym, tabdtype);
  SCP(sym, SC_STATIC);
  PLLENP(sym, n);

  /* initialize this array with the switch table  */

  SWELP(sym, swhdr - switch_base);
  DEFLABP(sym, deflab);
  if (doinit) {
/*
 * generate the entry for the default label.
 */
      DINITSWTAB_put(DINIT_LOC, sym);

    if (deflab) {
      /*
       * If the default label is passed, then it will not appear in the
       * switch list.  Just make sure that swel locates the first case.
       */
      swel = swhdr;
    } else {
      /*
       * Since the default label is not passed, then the switch list
       * contains the default label.  Extract it and make sure that it
       * gets added to the initialization of the switch array.
       */
      deflab = swhdr->clabel;
      DINITSWTAB_put(DINIT_LABEL, (INT)deflab);
      swel = switch_base + swhdr->next;
    }

    case_val = swel->val; /* start with first case value */
    do {
      /*
       * generate the remainder of the label table -- if there are
       * holes, the default label is generated.  case_val denotes
       * the expected case value.
       */
      for (case_val = swel->val - case_val; case_val; case_val--) {
        DINITSWTAB_put(DINIT_LABEL, deflab); /* default */
        RFCNTI((int)swhdr->clabel);
#if DEBUG
        interr("mk_swtab: CGOTO has holes, swidx", swhdr - switch_base, ERR_Warning);
#endif
      }
      DINITSWTAB_put(DINIT_LABEL, (INT)swel->clabel); /* case label */
      case_val = swel->val + 1;
      i = swel->next;
      swel = switch_base + i;
    } while (i != 0);
  }

  return sym;
}

int
mk_swtab_ll(INT n, SWEL *swhdr, int deflab, int doinit)
{
  int sym;
  int i;
  INT case_val[2];
  SWEL *swel;
  static INT one[2] = {0, 1};
  static INT zero[2] = {0, 0};
  INT vv[2];

  sym = getccsym('J', expb.swtcnt++, ST_PLIST); /* get switch array */
  DTYPEP(sym, DT_INT8);
  SCP(sym, SC_STATIC);
  PLLENP(sym, n);

  /* initialize this array with the switch table  */

  SWELP(sym, swhdr - switch_base);
  DEFLABP(sym, deflab);
  if (doinit) {
    DINITP(sym, 1);
/*
 * generate the entry for the default label.
 */
      dinit_put(DINIT_LOC, sym);
    if (deflab)
      /*
       * If the default label is passed, then it will not appear in the
       * switch list.  Just make sure that swel locates the first case.
       */
      swel = swhdr;
    else {
      /*
       * Since the default label is not passed, then the switch list
       * contains the default label.  Extract it and make sure that it
       * gets added to the initialization of the switch array.
       */
      deflab = swhdr->clabel;
      dinit_put(DINIT_LABEL, (INT)deflab);
      swel = switch_base + swhdr->next;
    }
    case_val[0] = CONVAL1G(swel->val); /* start with first case value */
    case_val[1] = CONVAL2G(swel->val);
    do {
      /*
       * generate the remainder of the label table -- if there are
       * holes, the default label is generated.  case_val denotes
       * the expected case value.
       */
      vv[0] = CONVAL1G(swel->val);
      vv[1] = CONVAL2G(swel->val);
      sub64(vv, case_val, case_val);
      /*for (case_val = swel->val - case_val; case_val; case_val--)*/
      while (true) {
        if (cmp64(case_val, zero) == 0)
          break;
        dinit_put(DINIT_LABEL, deflab); /* default */
        RFCNTI((int)swhdr->clabel);
#if DEBUG
        interr("mk_swtab: CGOTO has holes, swidx", swhdr - switch_base, ERR_Warning);
#endif
        sub64(case_val, one, case_val);
      }
      dinit_put(DINIT_LABEL, (INT)swel->clabel); /* case label */
      /*case_val = swel->val + 1;*/
      vv[0] = CONVAL1G(swel->val);
      vv[1] = CONVAL2G(swel->val);
      add64(vv, one, case_val);
      i = swel->next;
      swel = switch_base + i;
    } while (i != 0);
  }

  return sym;
}

/** \brief Make a sym for an arg's address
 */
SPTR
mk_argasym(int sptr)
{
  SPTR asym;
  asym = getccsym('c', sptr, ST_VAR);
  IS_PROC_DESCRP(asym, IS_PROC_DESCRG(sptr));
  DESCARRAYP(asym, DESCARRAYG(sptr));
  CLASSP(asym, CLASSG(sptr));
  SDSCP(asym, SDSCG(sptr));
  if (gbl.internal == 1 && CLASSG(asym) && DESCARRAYG(asym)) {
    /* Do not set lscope on class arguments within host subroutines */
    LSCOPEP(asym, 0);
  }
  SCP(asym, SCG(sptr));
  DTYPEP(asym, DT_CPTR);
  REDUCP(asym, 1);     /* mark sym --> no further indirection */
  MIDNUMP(asym, sptr); /* link indirection temp to formal */
  QALNP(asym, QALNG(sptr));
  NOCONFLICTP(asym, 1);
  GSCOPEP(asym, GSCOPEG(sptr));
  if (INTERNREFG(sptr)) {
    INTERNREFP(asym, 1);
    ADDRESSP(asym, ADDRESSG(sptr));
    MEMARGP(asym, MEMARGG(sptr));
  }
  if (UPLEVELG(sptr)) {
    /* Currently in an internal procedure and the argument is from
     * the host; need to propagate a few flags to the indirection
     * temp.
     */
    UPLEVELP(asym, 1);
    ADDRESSP(asym, ADDRESSG(sptr));
    MEMARGP(asym, MEMARGG(sptr));
  }
  return asym;
}

/*
 * is there an indirection symbol for this dummy?
 */
SPTR
find_argasym(int sptr)
{
  char name[16];
  SPTR asym;
  sprintf(name, ".%c%04d", 'c', sptr);
  asym = lookupsym(name, strlen(name));
  if (asym && DTYPEG(asym) == DT_CPTR && MIDNUMG(asym) == sptr)
    return asym;
  return SPTR_NULL;
} /* find_argasym */

/***************************************************************/

int
mk_impsym(SPTR sptr)
{
  char bf[3 * MAXIDLEN + 10]; /* accommodate "__imp_" and possibly mod name
                               * as prefixes
                               */
  int impsym;

  switch (STYPEG(sptr)) {
  case ST_ENTRY:
  case ST_PROC:
    if (INMODULEG(sptr)) {
      sprintf(bf, "__imp_%s_%s", SYMNAME(INMODULEG(sptr)), SYMNAME(sptr));
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    sprintf(bf, "__imp_%s", getsname(sptr));
  }

  impsym = getsymbol(bf);

  if (SCG(impsym) == SC_NONE) {
    STYPEP(impsym, ST_VAR);
    SCP(impsym, SC_EXTERN);
    DTYPEP(impsym, __POINT_T);
  }
  return impsym;
}

/***************************************************************/

SPTR
mkfunc_cncall(const char *nmptr)
{
  SPTR sptr;
  sptr = mkfunc(nmptr);
  CNCALLP(sptr, 1);
  return sptr;
}

/***************************************************************/

static const char *
skipws(const char *q)
{
  while (*q <= ' ' && *q != '\0')
    ++q;
  return q;
}

SPTR
mkfunc_sflags(const char *nmptr, const char *flags)
{
  SPTR sptr;
  const char *p;
  sptr = mkfunc(nmptr);
  p = flags;
  while (true) {
    p = skipws(p);
    if (*p == '\0')
      break;
    if (strncmp(p, "cncall", 6) == 0) {
      CNCALLP(sptr, 1);
      p += 6;
    } else if (strncmp(p, "xmmsafe", 7) == 0) {
      if (!XBIT(7, 0x4000))
        XMMSAFEP(sptr, 1);
      p += 7;
    }
#if DEBUG
    else {
      interr("mkfunc_sflags(): urecognized flag", sptr, ERR_Severe);
      break;
    }
#endif
  }
  return sptr;
}

void
exp_add_copy(SPTR lhssptr, SPTR rhssptr)
{
  int rhsacon, lhsacon, rhsnme, lhsnme, rhsld, lhsst, sz;
  ILI_OP rhsopc = IL_LD, lhsopc = IL_ST;
  MSZ msz;
  if (lhssptr == rhssptr)
    return;
  rhsacon = ad_acon(rhssptr, 0);
  sz = size_of(DTYPEG(rhssptr));
  if (sz == 8) {
    rhsopc = IL_LDKR;
    lhsopc = IL_STKR;
    msz = MSZ_I8;
  } else if (sz == 4) {
    msz = MSZ_WORD;
  } else if (sz == 2) {
    msz = MSZ_SHWORD;
  } else if (sz == 1) {
    msz = MSZ_BYTE;
  } else {
#if DEBUG
    interr("exp_add_copy: illegal type size", sz, ERR_Severe);
#endif
    msz = MSZ_BYTE;
  }
  rhsnme = addnme(NT_VAR, rhssptr, 0, 0);
  rhsld = ad3ili(rhsopc, rhsacon, rhsnme, msz);
  lhsacon = ad_acon(lhssptr, 0);
  lhsnme = addnme(NT_VAR, lhssptr, 0, 0);
  lhsst = ad4ili(lhsopc, rhsld, lhsacon, lhsnme, msz);
  chk_block(lhsst);
}

SPTR
get_byval_local(int argsptr)
{
  char *new_name;
  SPTR newsptr;
  int new_length;

  newsptr = MIDNUMG(argsptr);
  if (newsptr > SPTR_NULL)
    return newsptr;
  new_name = SYMNAME(argsptr);
  new_name += 3; /* move past appended _V_ */
  new_length = strlen(new_name);
  newsptr = getsymbol(new_name);
  for (; newsptr; newsptr = HASHLKG(newsptr)) {
    if (strncmp(new_name, SYMNAME(newsptr), new_length) != 0 ||
        *(SYMNAME(newsptr) + new_length) != '\0')
      continue;
    if (STYPEG(newsptr) == STYPEG(argsptr)) {
      if (SCG(newsptr) == SC_LOCAL)
        return newsptr;
    }
  }
  newsptr = getsymbol(new_name); /* OH OH -- ICE */
  return newsptr;
}

/** \brief Add a register argument
 *
 * Add argument expression argili to existing argument list, arglist,
 * using registers. If arglist = 0, begin a new list.
 */
int
add_reg_arg_ili(int arglist, int argili, int nmex, DTYPE dtype)
{
  int rg, ilix;
  ILI_OP opc;
  static int avail_ireg; /* next available integer register for jsr */
  static int avail_freg; /* next available floating point register for jsr */

  if (arglist == 0) {
    arglist = ad1ili(IL_NULL, 0);
    avail_ireg = 0;
    avail_freg = 0;
  }
  if (DTY(dtype) == TY_PTR) {
    rg = IR(avail_ireg++);
    opc = IL_DAAR;
  } else if (DT_ISINT(dtype)) {
    rg = IR(avail_ireg++);
    opc = IL_RES(ILI_OPC(argili)) != ILIA_KR ? IL_DAIR : IL_DAKR;
  } else {
    if (DTY(dtype) == TY_DBLE && (avail_freg & 1))
      avail_freg++;
    rg = SP(avail_freg);
    avail_freg++;
    if (DTY(dtype) == TY_DBLE) {
      opc = IL_DADP;
      avail_freg++;
    } else {
      opc = IL_DASP;
    }
  }

  ilix = ad3ili(opc, argili, rg, arglist);
  return ilix;
} /* add_reg_arg_ili */

#if DEBUG
void
expdumpilms()
{
  int i, bsize;
  ilmb.ilm_base[BOS_SIZE - 1] = ilmb.ilmavl;
  if (gbl.dbgfil == NULL)
    gbl.dbgfil = stderr;

  if (ilmb.ilm_base[0] != IM_BOS) {
    fprintf(gbl.dbgfil, "expdumpilms: no IM_BOS (ilm_base[0]==%d)\n", ilmb.ilm_base[0]);
  }

  fprintf(gbl.dbgfil, "\n----- lineno: %d"
                      " ----- global ILM index %d:%d"
                      "\n",
          ilmb.ilm_base[1] , ilmb.globalilmstart, ilmb.globalilmcount
          );
  bsize = ilmb.ilm_base[BOS_SIZE - 1]; /* number of words in this ILM block */

  i = 0;
  do { /* loop once for each ILM opcode: */
    int _dumponeilm(ILM_T *, int, int check);
    int j = i;
    i = _dumponeilm(ilmb.ilm_base, i, 0);
    if (ILM_RESULT(j))
      fprintf(gbl.dbgfil, "  result:%d", ILM_RESULT(j));
    if (ILM_IRESULT(j))
      fprintf(gbl.dbgfil, "  iresult/clen:%d", ILM_IRESULT(j));
    if (ILM_RESTYPE(j))
      fprintf(gbl.dbgfil, "  restype:%d", ILM_RESTYPE(j));
    if (ILM_NME(j))
      fprintf(gbl.dbgfil, "  nme:%d", ILM_NME(j));
    if (ILM_BLOCK(j))
      fprintf(gbl.dbgfil, "  block:%d", ILM_BLOCK(j));
    if (ILM_SCALE(j))
      fprintf(gbl.dbgfil, "  scale:%d", ILM_SCALE(j));
    if (ILM_MXLEN(j))
      fprintf(gbl.dbgfil, "  mxlen/clen:%d", ILM_MXLEN(j));
    if (ILM_EXPANDED_FOR(j))
      fprintf(gbl.dbgfil, "  expanded_for:%d", ILM_EXPANDED_FOR(j));

    fprintf(gbl.dbgfil, "\n");
    if (i > bsize) {
      fprintf(gbl.dbgfil, "BAD BLOCK LENGTH: %d\n", bsize);
    }
  } while (i < bsize);
} /* expdumpilms */
#endif
