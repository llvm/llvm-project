/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief ILT utility module
 */

#include "iltutil.h"
#include "error.h"
#include "global.h"
#include "symtab.h" /* prerequisite for expand.h and ili.h */
#include "ilm.h"
#include "fih.h"
#include "ili.h"
#include "expand.h"
#include "ccffinfo.h"

#include <stdarg.h>
#define MAXILT 67108864

static int iltcur;

/** \brief Initialize the ILT area
 */
void
ilt_init(void)
{
  STG_ALLOC(iltb, 128);
  STG_SET_FREELINK(iltb, ILT, next);
}

void
ilt_cleanup(void)
{
  STG_DELETE(iltb);
} /* ilt_cleanup */

/********************************************************************/

/** \brief Add an ilt
 *
 * Add an ilt inserting it after the ilt "after"; ilix locates
 * the ili which represents the root of the ili tree
 */
int
addilt(int after, int ilix)
{
  int i;
  ILT *p;
  ILTY_KIND type;
  ILI_OP opc;

  i = STG_NEXT_FREELIST(iltb);
  p = iltb.stg_base + i;
  p->flags.all = 0;
  p->prev = after;
  p->next = ILT_NEXT(after);
  p->lineno = gbl.lineno;
  p->order = -1;
  ILT_NEXT(after) = i;
  ILT_PREV(p->next) = i;
  p->ilip = ilix;
  opc = ILI_OPC(ilix);
  type = IL_TYPE(opc);
  if (type == ILTY_STORE)
    p->flags.bits.st = 1;
  else if (type == ILTY_BRANCH)
    p->flags.bits.br = 1;
  p->flags.bits.ex = iltb.callfg;
  bihb.callfg |= iltb.callfg;
  bihb.ldvol |= iltb.ldvol;
  bihb.stvol |= iltb.stvol;
  bihb.qjsrfg |= iltb.qjsrfg;
  if (after == 0 && p->next == 0) {
    /* this is the first ILT for this block */
    fihb.currfindex = fihb.nextfindex;
    fihb.currftag = fihb.nextftag;
  }
  p->findex = fihb.currfindex;
  iltb.callfg = 0;
  iltb.ldvol = 0;
  iltb.stvol = 0;
  iltb.qjsrfg = false;
  return (i);
}

/** \brief Delete an ilt from a block which is "read" and reuse it
 */
void
delilt(int iltx)
{
  int prev, next, ignore, ilip;

  next = ILT_NEXT(iltx);
  prev = ILT_PREV(iltx);
  /* preserve the ILT_ILIP field, ILT_IGNORE flag, set ILT_FREE flag */
  ignore = ILT_IGNORE(iltx);
  ilip = ILT_ILIP(iltx);
  ILT_PREV(next) = prev;
  ILT_NEXT(prev) = next;
  STG_ADD_FREELIST(iltb, iltx);
  ILT_IGNORE(iltx) = ignore;
  ILT_ILIP(iltx) = ilip;
  ILT_FREE(iltx) = 1;
}

/** \brief delete an ilt where the block may not be "read", and possibly reuse
 * it
 *
 * \param iltx    ilt to be deleted
 * \param bihx    bih of block from which ilt is deleted (0 => read)
 * \param reuse   true if ilt is to be reused
 */
void
unlnkilt(int iltx, int bihx, bool reuse)
{
  int i, j;

  if (bihx) {
    i = ILT_PREV(iltx);
    j = ILT_NEXT(iltx);
    if (j)
      ILT_PREV(j) = i;
    else
      BIH_ILTLAST(bihx) = i;
    if (i)
      ILT_NEXT(i) = j;
    else
      BIH_ILTFIRST(bihx) = j;
  } else {
    j = ILT_NEXT(iltx);
    i = ILT_PREV(j) = ILT_PREV(iltx);
    ILT_NEXT(i) = j;
  }
  if (reuse) {
    /* preserve the ILT_ILIP field, ILT_IGNORE flag */
    int ignore = ILT_IGNORE(iltx);
    int ilip = ILT_ILIP(iltx);
    STG_ADD_FREELIST(iltb, iltx);
    ILT_IGNORE(iltx) = ignore;
    ILT_ILIP(iltx) = ilip;
  }
  ILT_FREE(iltx) = 1;
  /* else:  hopefully, scans will still work if we start with an ilt which
   * was removed but not reused
   */

}

/*
 * move an ilt to this BIH after removing it with unlnkilt
 * iltx = ilt to be added
 * bihx = bih of block to which ilt is to be added
 */
void
relnkilt(int iltx, int bihx)
{
  int j;
  j = BIH_ILTLAST(bihx);
  ILT_PREV(iltx) = j;
  if (j)
    ILT_NEXT(j) = iltx;
  else
    BIH_ILTFIRST(bihx) = iltx;
  BIH_ILTLAST(bihx) = iltx;
  ILT_FREE(iltx) = 0;
} /* relnkilt */

/********************************************************************/

/** \brief Move an ilt before another ilt
 */
void
moveilt(int iltx, int before)
{
  int i, j;

  /**  remove iltx from list  **/
  i = ILT_PREV(iltx);
  ILT_NEXT(i) = j = ILT_NEXT(iltx);
  ILT_PREV(j) = i;
  /**  insert iltx before 'before' **/
  ILT_PREV(iltx) = i = ILT_PREV(before);
  ILT_NEXT(i) = iltx;
  ILT_PREV(before) = iltx;
  ILT_NEXT(iltx) = before;
}

/********************************************************************/

/**
  search the ili subtree located by ilix for functions and creating an ilt for
  each one found.  The static variable iltcur (local to this module indicates
  where an ilt is added.  iltcur is updated to locate the new ilt
 */
static void
srcfunc(int ilix)
{
  int noprs; /* number of lnk operands in ilix	 */
  int i;              /* index variable			 */
  ILI_OP opc;            /* ili opcode of ilix			 */

  if (IL_TYPE(opc = ILI_OPC(ilix)) == ILTY_PROC && opc >= IL_JSR) {
    iltb.callfg = 1;
    iltcur = addilt(iltcur, ilix); /* create a function ilt */
  } else if (opc == IL_DFRDP && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    iltcur = addilt(iltcur, ad1ili(IL_FREEDP, ilix));
  } else if (opc == IL_DFRSP && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    iltcur = addilt(iltcur, ad1ili(IL_FREESP, ilix));
  }
  else if (opc == IL_DFRCS && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    iltcur = addilt(iltcur, ad1ili(IL_FREECS, ilix));
  }
#ifdef LONG_DOUBLE_FLOAT128
  else if (opc == IL_FLOAT128RESULT && ILI_OPC(ILI_OPND(ilix, 1)) != IL_QJSR) {
    iltb.callfg = 1;
    iltcur = addilt(iltcur, ad1ili(IL_FLOAT128FREE, ilix));
  }
#endif /* LONG_DOUBLE_FLOAT128 */
  else {
    noprs = ilis[opc].oprs;
    for (i = 1; i <= noprs; i++) {
      if (IL_ISLINK(opc, i))
        srcfunc((int)(ILI_OPND(ilix, i)));
    }
  }
}

/********************************************************************/

/** \brief Reduce an ilt to a sequence of function ilts
 *
 * Reduces the ili tree located by ilix producing ilts which locate function
 * ilis occuring in ilix.  This routine sets iltucr with iltx which indicates
 * where ilts are to be added and calls srcfunc.
 */
int
reduce_ilt(int iltx, int ilix)
{
  iltcur = iltx; /* avoid passing iltx recursively	 */
  srcfunc(ilix);
  return (iltcur);
}

/********************************************************************/

/** \brief Dump ILT to a file
 *
 * \param ff - file pointer
 * \param bihx - BIH number
 */
void
dump_ilt(FILE *ff, int bihx)
{
  int p, q, throw_count;

  if (ff == NULL)
    ff = stderr;
  iltb.privtmp = 0;
  if (BIH_PAR(bihx) || BIH_TASK(bihx))
    iltb.privtmp = 2;
  else
    iltb.privtmp = 1;
  fprintf(ff, "\nBlock %5d, line:%6d, label:%6d, assn:%6d, fih:%3d", bihx,
          BIH_LINENO(bihx), BIH_LABEL(bihx), BIH_ASSN(bihx), BIH_FINDEX(bihx));
  fprintf(ff, ", flags:");
  if (BIH_PAR(bihx))
    fprintf(ff, " PAR");
  if (BIH_RD(bihx))
    fprintf(ff, " RD");
  if (BIH_FT(bihx))
    fprintf(ff, " FT");
  if (BIH_EN(bihx))
    fprintf(ff, " EN");
  if (BIH_EX(bihx))
    fprintf(ff, " EX");
  if (BIH_XT(bihx))
    fprintf(ff, " XT");
  if (BIH_LAST(bihx))
    fprintf(ff, " LAST");
  if (BIH_PL(bihx))
    fprintf(ff, " PL");
  if (BIH_ZTRP(bihx))
    fprintf(ff, " ZT");
  if (BIH_NOBLA(bihx))
    fprintf(ff, " NOBLA");
  if (BIH_NOMERGE(bihx))
    fprintf(ff, " NOMERGE");
#ifdef BIH_ASM
  if (BIH_ASM(bihx))
    fprintf(ff, " ASM");
#endif
  if (BIH_QJSR(bihx))
    fprintf(ff, " QJSR");
  if (BIH_HEAD(bihx))
    fprintf(ff, " HEAD");
  if (BIH_TAIL(bihx))
    fprintf(ff, " TAIL");
  if (BIH_INNERMOST(bihx))
    fprintf(ff, " INNERMOST");
#ifdef BIH_GUARDEE
  if (BIH_GUARDEE(bihx))
    fprintf(ff, " GUARDEE");
#endif
#ifdef BIH_GUARDER
  if (BIH_GUARDER(bihx))
    fprintf(ff, " GUARDER");
#endif
  if (BIH_MEXITS(bihx))
    fprintf(ff, " MEXITS");
  if (BIH_SMOVE(bihx))
    fprintf(ff, " SMOVE");
  if (BIH_CS(bihx))
    fprintf(ff, " CS");
  if (BIH_PARSECT(bihx))
    fprintf(ff, " PARSECT");
  if (BIH_ENLAB(bihx))
    fprintf(ff, " ENLAB");
  if (BIH_PARLOOP(bihx))
    fprintf(ff, " PARLOOP");
  if (BIH_UJRES(bihx))
    fprintf(ff, " UJRES");
  if (BIH_SIMD(bihx))
    fprintf(ff, " SIMD");
  if (BIH_NOSIMD(bihx))
    fprintf(ff, " NOSIMD");
  if (BIH_UNROLL(bihx))
    fprintf(ff, " UNROLL");
  if (BIH_UNROLL_COUNT(bihx))
    fprintf(ff, " UNROLL_COUNT");
  if (BIH_NOUNROLL(bihx))
    fprintf(ff, " NOUNROLL");
  if (BIH_LDVOL(bihx))
    fprintf(ff, " LDVOL");
  if (BIH_STVOL(bihx))
    fprintf(ff, " STVOL");
#ifdef BIH_STREG
  if (BIH_STREG(bihx))
    fprintf(ff, " STREG");
#endif
  if (BIH_VPAR(bihx))
    fprintf(ff, " VPAR");
  if (BIH_PARALN(bihx))
    fprintf(ff, " PARALN");
  if (BIH_COMBST(bihx))
    fprintf(ff, " COMBST");
  if (BIH_TASK(bihx))
    fprintf(ff, " TASK");
  if (BIH_RESID(bihx))
    fprintf(ff, " RESID");
  if (BIH_VCAND(bihx))
    fprintf(ff, " VCAND");
  if (BIH_MIDIOM(bihx))
    fprintf(ff, " MIDIOM");
  if (BIH_DOCONC(bihx))
    fprintf(ff, " DOCONC");
#ifdef BIH_LPCNTFROM
  if (BIH_LPCNTFROM(bihx))
    fprintf(ff, " lpcntfrom: %d:", BIH_LPCNTFROM(bihx));
#endif
  fprintf(ff, "\n");

  q = 0;
  throw_count = 0;
  for (p = BIH_ILTFIRST(bihx); p != 0; p = ILT_NEXT(p)) {
    q = p;
    if (DBGBIT(10, 128) && DBGBIT(10, 512)) {
      if (ILT_DELETE(p))
        fprintf(ff, "[%4d]@\t", p);
      else if (ILT_IGNORE(p))
        fprintf(ff, "[%4d]#\t", p);
      else
        fprintf(ff, "[%4d]\t", p);
#if DEBUG
      if (ff != stderr)
        dmpilitree((int)ILT_ILIP(p));
      else
        ddilitree((int)ILT_ILIP(p), 1);
#endif
    } else {
      if (DBGBIT(10, 128)) {
        fprintf(ff, "[%4d]", p);
      } else {
        fprintf(ff, " %5d  %5d^  flags:", p, ILT_ILIP(p));
      }

      if (ILT_EX(p))
        fprintf(ff, " EX");
      if (ILT_BR(p))
        fprintf(ff, " BR");
      if (ILT_CAN_THROW(p)) {
        int lab;
        fprintf(ff, " CAN_THROW");
        ++throw_count;
        assert(throw_count == 1, "block should have at most one CAN_THROW",
               bihx, ERR_Severe);
        lab = ili_throw_label(ILT_ILIP(p));
        assert(lab, "ILT marked as CAN_THROW but does not", bihx, ERR_Severe);
      }
      if (ILT_ST(p))
        fprintf(ff, " ST");
      if (ILT_DELETE(p))
        fprintf(ff, " DELETE");
      if (ILT_IGNORE(p))
        fprintf(ff, " IGNORE");
      if (ILT_DBGLINE(p))
        fprintf(ff, " DBGL");
      if (ILT_SPLIT(p))
        fprintf(ff, " SPLIT");
      if (ILT_CPLX(p))
        fprintf(ff, " CPLX");
      if (ILT_MCACHE(p))
        fprintf(ff, " MCACHE");
      if (ILT_DELEBB(p))
        fprintf(ff, " DELEBB");
      if (ILT_PREDC(p))
        fprintf(ff, " PREDC");
#if defined(ILT_GUARD)
      if (ILT_GUARD(p) != -1) {
        fprintf(ff, "\t iff [%d]", ILT_GUARD(p));
      }
#endif
      if (ILT_INV(p))
        fprintf(ff, " INV");
      fprintf(ff, "\n");
#if DEBUG
      if (DBGBIT(10, 128)) {
        dmpilitree((int)ILT_ILIP(p));
      }
#endif
    }
  }
  assert(q == BIH_ILTLAST(bihx), "dmpilt: bad end of block", bihx, ERR_Severe);
  iltb.privtmp = 0;
}

/** \brief Dump ILT to global debug file
 *
 * Synonym to dump_ilt() with gbl.dbgfil as the file argument
 *
 * \param bihx - BIH number
 */
void
dmpilt(int bihx)
{
  dump_ilt(gbl.dbgfil, bihx);
}

/** \brief Write out an ilt/ili block given its bih
 *
 * Write out ilts for the block denoted by bih.  Various flags have already
 * been set in bih. This routine buffers up the ilt block in the bih area.
 */
void
wrilts(int bihx)
{
  BIH_ILTFIRST(bihx) = ILT_NEXT(0);
  BIH_ILTLAST(bihx) = ILT_PREV(0);
  bihb.callfg = 0;
  bihb.ldvol = 0;
  bihb.stvol = 0;
  bihb.qjsrfg = 0;
  if (bihx != gbl.entbih)
    BIH_FINDEX(bihx) = fihb.currfindex;
#ifdef BIH_FTAG
  if (bihx != gbl.entbih) {
    BIH_FINDEX(bihx) = fihb.currfindex;
    BIH_FTAG(bihx) = fihb.currftag;
    ++fihb.currftag;
    if ((fihb.currftag > fihb.nextftag) && (fihb.currfindex == fihb.nextfindex))
      fihb.nextftag = fihb.currftag;
  }
#endif
#if DEBUG
  if (flg.dbg[8] & 1)
    dmpilt(bihx);
#endif
}

/** \brief Read in an ilt/ili block
 *
 * "Read" in the block specified by bihx.  the 0th entry in the ILT area (next
 * and prev) are set to the first and last ilts respectively
 */
void
rdilts(int bihx)
{
  if (BIH_RD(bihx) != 0) {
    ILT_NEXT(0) = BIH_ILTFIRST(bihx);
    ILT_PREV(0) = BIH_ILTLAST(bihx);
  }
  bihb.callfg = BIH_EX(bihx);
  bihb.ldvol = BIH_LDVOL(bihx);
  bihb.stvol = BIH_STVOL(bihx);
  bihb.qjsrfg = BIH_QJSR(bihx);
#ifdef BIH_FTAG
  fihb.nextfindex = fihb.currfindex = BIH_FINDEX(bihx);
  fihb.nextftag = fihb.currftag = BIH_FTAG(bihx);
#endif
}

/*
 * save one message
 *  call ccff_info
 */
void *
ccff_ilt_info(int msgtype, const char *msgid, int iltx, int bihx, const char *message,
              ...)
{
  va_list argptr;
  int fihx, lineno;
  va_start(argptr, message);

  fihx = -1;
  lineno = -1;
  if (iltx > 0) {
    fihx = ILT_FINDEX(iltx);
    lineno = ILT_LINENO(iltx);
  }
  if (fihx <= 0 || lineno <= 0) {
    fihx = BIH_FINDEX(bihx);
    lineno = BIH_LINENO(bihx);
  }
  if (fihx == 0)
    return NULL; /* no info */
  return _ccff_info(msgtype, msgid, fihx, lineno, NULL, NULL, NULL, message,
                    argptr);
} /* ccff_ilt_info */

/*
 * save one message
 *  call subccff_info
 */
void *
subccff_ilt_info(void *xparent, int msgtype, const char *msgid, int iltx, int bihx,
                 const char *message, ...)
{
  va_list argptr;
  int fihx, lineno;
  va_start(argptr, message);

  fihx = -1;
  lineno = -1;
  if (iltx > 0) {
    fihx = ILT_FINDEX(iltx);
    lineno = ILT_LINENO(iltx);
  }
  if (fihx <= 0 || lineno <= 0) {
    fihx = BIH_FINDEX(bihx);
    lineno = BIH_LINENO(bihx);
  }
  if (fihx == 0)
    return NULL; /* no info */
  return _ccff_info(msgtype, msgid, fihx, lineno, NULL, NULL, xparent, message,
                    argptr);
} /* subccff_ilt_info */
