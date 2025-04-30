/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief C and Fortran BIH utility module
 */

#include "bihutil.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "ilm.h"
#include "fih.h"
#include "ili.h"
#include "expand.h"
#include "regutil.h"
#include "machreg.h"

extern int getcon();
extern int mkfunc();

extern void rdilts();
extern void wrilts();

#define MAXBIH 67108864

/** \brief Initialize BIH area
 */
void
bih_init(void)
{
  STG_ALLOC(bihb, 128);
  STG_SET_FREELINK(bihb, BIH, next);
  BIH_LPCNTFROM(0) = 0;
  BIH_NEXT(bihb.stg_size - 1) = 0;
  BIH_FLAGS(0) = 0;
  BIH_NEXT(0) = 0;
  BIH_PREV(0) = 0;
  BIH_AVLPCNT(0) = 0;
  BIH_GUARDEE(0) = 0;
  BIH_GUARDER(0) = 0;
#ifdef BIH_FTAG
  BIH_FTAG(0) = 0;
  BIH_FINDEX(0) = 0;
  BIH_LINENO(0) = 0;
  BIH_ASM(0) = 0;
#endif
  bihb.stg_max = 0;
#if DEBUG
  assert(((char *)&BIH_BLKCNT(0) - (char *)&bihb.stg_base[0]) % 8 == 0,
         "offset of BIH_BLKCNT must be a multiple of 8", 0, ERR_Fatal);
  assert(sizeof(BIH) % 8 == 0, "size of BIH must be a multiple of 8", 0, ERR_Fatal);
#endif
}

void
bih_cleanup()
{
  STG_DELETE(bihb);
} /* bih_cleanup */

/********************************************************************/

/** \brief Add a new block during the expand phase.
 *
 * Create a new bih entry and add it after the one specified by the formal
 * argument set the FINDEX/FTAG from fihb structure; used during expand
 *
 * \param after BIH entry to add after
 * \return new BIH entry
 */
int
exp_addbih(int after)
{

  int i;
  BIH *p;

  i = STG_NEXT_FREELIST(bihb);
  p = bihb.stg_base + i;
  p->prev = after;
  p->next = BIH_NEXT(after);
  BIH_NEXT(after) = i;
  BIH_PREV(p->next) = i;
  p->label = SPTR_NULL;
  p->lineno = 0;
  p->flags.all = 0;
  p->flags2.all = 0;
  p->flags.bits.rd = 1;
  p->first = 0;
  p->last = 0;
  p->assn = 0;
  p->lpcntFrom = 0;
  p->rgset = 0;
#ifdef BIH_FTAG
  p->findex = fihb.nextfindex;
  p->ftag = fihb.nextftag;
#endif
  p->blkCnt = UNKNOWN_EXEC_CNT;
  p->aveLpCnt = 0;
  if (i > bihb.stg_max)
    bihb.stg_max = i;

  return (i);
} /* exp_addbih */

/** \brief Universal function to create new BIH entry
 *
 * \param after BIH anery to the new one after
 * \param flags BIH entry to take flags from
 * \param fih BIH entry to take FIH information from
 *
 * \return new BIH entry
 */
int
addnewbih(int after, int flags, int fih)
{
  int i, next;
  BIH *p;

  i = STG_NEXT_FREELIST(bihb);
  p = bihb.stg_base + i;
  p->prev = after;
  next = BIH_NEXT(after);
  BIH_NEXT(after) = i;
  if (next >= 0) {
    p->next = next;
    BIH_PREV(next) = i;
  }
  p->label = SPTR_NULL;
  p->lineno = 0;
  p->flags.all = 0;
  p->flags2.all = 0;
  p->flags.bits.rd = 1;
  p->first = 0;
  p->last = 0;
  p->assn = 0;
  p->lpcntFrom = 0;
  p->rgset = 0;
#ifdef BIH_FTAG
  p->findex = 0;
  p->ftag = 0;
#endif
  p->blkCnt = 0;
#ifdef BIH_FTAG
  p->ftag = BIH_FTAG(fih);
#endif
  p->lineno = BIH_LINENO(fih);
  p->findex = BIH_FINDEX(fih);
  if (flags) {
    p->flags.bits.par = BIH_PAR(flags);
    p->flags.bits.parsect = BIH_PARSECT(flags);
    p->flags2.bits.task = BIH_TASK(flags);
    p->blkCnt = BIH_BLKCNT(flags);
    p->aveLpCnt = BIH_AVLPCNT(flags);
  }
  if (i > bihb.stg_max)
    bihb.stg_max = i;
  return i;
} /* addnewbih */

/** \brief Create a new BIH entry and add it after the one specified by the
 * formal argument
 *
 * \param after BIH entry to add after
 * \return new BIH entry
 */
int
addbih(int after)
{
  return addnewbih(after, 0, after);
} /* addbih */

/********************************************************************/

/** \brief Delete a bih entry from the BIH list
 *
 * \param bihx BIH entry to delete
 */
void
delbih(int bihx)
{
  int prev, next;
  BIH bb;
  next = BIH_NEXT(bihx);
  prev = BIH_PREV(bihx);
  BIH_PREV(next) = prev;
  BIH_NEXT(prev) = next;
  /* STG_ADD_FREELIST clears the fields;
   * instead we want to preserve the fields,
   * except the freelist link in BIH_NEXT */
  bb = bihb.stg_base[bihx];
  STG_ADD_FREELIST(bihb, bihx);
  bb.next = BIH_NEXT(bihx);
  bihb.stg_base[bihx] = bb;
}

/********************************************************************/

void
merge_bih_flags(int to_bih, int fm_bih)
{
  /* after merging two blocks, merge their BIH flags */

  BIH_FT(to_bih) = BIH_FT(fm_bih);
  /* BIH_EX(to_bih) |= BIH_EX(fm_bih); ***causes SUN cc to error */
  BIH_EX(to_bih) = BIH_EX(to_bih) | BIH_EX(fm_bih);
  BIH_ZTRP(to_bih) = BIH_ZTRP(to_bih) | BIH_ZTRP(fm_bih);
  BIH_SMOVE(to_bih) = BIH_SMOVE(to_bih) | BIH_SMOVE(fm_bih);
  BIH_NOBLA(to_bih) = BIH_NOBLA(to_bih) | BIH_NOBLA(fm_bih);
  BIH_QJSR(to_bih) = BIH_QJSR(to_bih) | BIH_QJSR(fm_bih);
  BIH_INVIF(to_bih) = BIH_INVIF(to_bih) | BIH_INVIF(fm_bih);
  BIH_NOINVIF(to_bih) = BIH_NOINVIF(to_bih) | BIH_NOINVIF(fm_bih);
  BIH_SIMD(to_bih) = BIH_SIMD(to_bih) | BIH_SIMD(fm_bih);
  BIH_NOSIMD(to_bih) = BIH_NOSIMD(to_bih) | BIH_NOSIMD(fm_bih);
  BIH_RESID(to_bih) = BIH_RESID(to_bih) | BIH_RESID(fm_bih);
  BIH_VCAND(to_bih) = BIH_VCAND(to_bih) | BIH_VCAND(fm_bih);
  BIH_MIDIOM(to_bih) = BIH_MIDIOM(to_bih) | BIH_MIDIOM(fm_bih);
  BIH_COMBST(to_bih) = BIH_COMBST(to_bih) | BIH_COMBST(fm_bih);
  BIH_ASM(to_bih) = BIH_ASM(to_bih) | BIH_ASM(fm_bih);
  BIH_LDVOL(to_bih) = BIH_LDVOL(to_bih) | BIH_LDVOL(fm_bih);
  BIH_STVOL(to_bih) = BIH_STVOL(to_bih) | BIH_STVOL(fm_bih);
  BIH_NODEPCHK(to_bih) = BIH_NODEPCHK(to_bih) | BIH_NODEPCHK(fm_bih);
  BIH_UNROLL(to_bih) = BIH_UNROLL(to_bih) | BIH_UNROLL(fm_bih);
  BIH_UNROLL_COUNT(to_bih) = BIH_UNROLL_COUNT(to_bih) | BIH_UNROLL_COUNT(fm_bih);
  BIH_NOUNROLL(to_bih) = BIH_NOUNROLL(to_bih) | BIH_NOUNROLL(fm_bih);

  if (BIH_TAIL(fm_bih))
    BIH_TAIL(to_bih) = 1;
  if (BIH_LAST(fm_bih))
    BIH_LAST(to_bih) = 1;
}

/** \brief Merge a block with its successor
 *
 * \param curbih BIH block to merge with its successor
 * \return BIH of the merged block if merging occurred; otherwise, return 0.
 */
int
merge_bih(int curbih)
{
  int nextbih, label;
  int firstilt, lastilt, iltx;

  if (XBIT(8, 0x80000000))
    return 0;

  nextbih = BIH_NEXT(curbih);

  if (BIH_EN(curbih) || BIH_EN(nextbih) || BIH_XT(curbih) || BIH_XT(nextbih) ||
      BIH_NOMERGE(curbih) || BIH_NOMERGE(nextbih) || BIH_ENLAB(curbih) ||
      BIH_ENLAB(nextbih)
#ifdef BIH_GASM
      || BIH_GASM(curbih) || BIH_GASM(nextbih)
#endif
          ) {
    return 0;
  }
  if (BIH_EN(nextbih))
    return 0;

  if (BIH_ASSN(curbih) != BIH_ASSN(nextbih) ||
      BIH_PAR(curbih) != BIH_PAR(nextbih) ||
      BIH_PARSECT(curbih) != BIH_PARSECT(nextbih) ||
      BIH_PL(curbih) != BIH_PL(nextbih) || BIH_CS(curbih) != BIH_CS(nextbih) ||
      BIH_TASK(curbih) != BIH_TASK(nextbih)) {
    return 0;
  }

  if (BIH_COMBST(curbih) && !BIH_COMBST(nextbih)) {
    return 0;
  }

  label = BIH_LABEL(nextbih);
  if (label) {
    if (RFCNTG(label) || VOLG(label)) {
      return 0;
    }
    ILIBLKP(label, 0);
    BIH_LABEL(nextbih) = SPTR_NULL;
  }

  firstilt = BIH_ILTFIRST(nextbih);
  if (ILT_DBGLINE(firstilt)) /* watch out for debugger call */
    firstilt = ILT_NEXT(firstilt);
  if (firstilt == 0) {
    if (BIH_LINENO(nextbih) && !BIH_LINENO(curbih))
      BIH_LINENO(curbih) = BIH_LINENO(nextbih);
    delbih(nextbih); /* remove empty block */
    return curbih;
  }

  lastilt = BIH_ILTLAST(curbih);
  if (lastilt && IL_TYPE(ILI_OPC(ILT_ILIP(lastilt))) == ILTY_BRANCH)
    return 0;

  /*
   * Add the "first" ilt of the block to the end of the current
   * block.
   */
  rdilts(curbih);
  ILT_NEXT(lastilt) = firstilt;
  ILT_PREV(firstilt) = lastilt;
  ILT_PREV(0) = BIH_ILTLAST(nextbih);
#if DEBUG
  if (EXPDBG(9, 32))
    fprintf(gbl.dbgfil, "                  block %d merged, label %d\n",
            nextbih, label);
#endif

  iltx = lastilt;
  while (iltx) {
    if (ILT_DELEBB(iltx)) {
      ILT_DELETE(iltx) = 1;
      ILT_DELEBB(iltx) = 0;
#if DEBUG
      if (EXPDBG(9, 32))
        fprintf(gbl.dbgfil,
                "                  ilt %d: ILT_DELEBB->ILT_DELETE\n", iltx);
#endif
    }
    iltx = ILT_PREV(iltx);
  }

  if (BIH_LINENO(nextbih) && !BIH_LINENO(curbih))
    BIH_LINENO(curbih) = BIH_LINENO(nextbih);
  /* update the BIH flags of the current block  */
  merge_bih_flags(curbih, nextbih);

#if DEBUG
  assert((BIH_PARSECT(curbih) ^ BIH_PARSECT(nextbih)) == 0,
         "merge_bih:parsect,nonparsect", curbih, ERR_Severe);
#endif

  wrilts(curbih);

  merge_rgset(curbih, nextbih, false);

  /* remove the block from the BIH list  */

  delbih(nextbih);

  return curbih;
}

/*
 * \brief If a routine contains any PAR blocks, return true
 */
bool
contains_par_blocks(void)
{
  int bihx;

  for (bihx = gbl.entbih; bihx; bihx = BIH_NEXT(bihx))
    if (BIH_PAR(bihx))
      return true;

  return false;
}

/** \brief Merge block b2 into b1
 */
void
merge_blks(int b1, int b2)
{
  int i;
  int j;

  rdilts(b1);
  i = ILT_PREV(0);      /* last ilt in first block */
  j = BIH_ILTFIRST(b2); /* first ilt in second block */
  if (i == 0) {
    /* first block is empty: just copy contents of second block */
    ILT_NEXT(0) = j;
    ILT_PREV(0) = BIH_ILTLAST(b2);
  } else if (j) {
    /* first & second blocks are not empty */
    ILT_NEXT(i) = j; /* link last ilt to first */
    ILT_PREV(j) = i;
    ILT_PREV(0) = BIH_ILTLAST(b2);
  }
  /* else first block nonempty & second block empty -- nothing to do */

  wrilts(b1);

  /* update necessary BIH flags */

  BIH_FT(b1) = BIH_FT(b2);
  BIH_EX(b1) = BIH_EX(b1) | BIH_EX(b2);
  BIH_QJSR(b1) = BIH_QJSR(b1) | BIH_QJSR(b2);
}

/* BIH_RGSET(tobih) U= BIH_RGSET(frombih) */
void
merge_rgset(int tobih, int frombih, bool reuse_to)
{
  if (BIH_RGSET(tobih) != BIH_RGSET(frombih)) {
    if (!BIH_RGSET(tobih))
      BIH_RGSET(tobih) = mr_get_rgset();
    if (BIH_RGSET(frombih))
      RGSET_XR(BIH_RGSET(tobih)) |= RGSET_XR(BIH_RGSET(frombih));
  }
}

#if DEBUG
int *badpointer1 = (int *)0;
long *badpointer2 = (long *)1;
long badnumerator = 99;
long baddenominator = 0;
#endif

/*
 * get rid of useless splits between blocks;
 * if we have two blocks b1 and its lexical successor b2
 *   b1 is fall-through, does not end in a jump
 *   neither has BIH_XT BIH_PL BIH_NOMERGE BIH_ENLAB
 *               BIH_GASM BIH_LAST BIH_PAR BIH_CS BIH_PARSECT
 *  merge the two blocks
 */
void
unsplit(void)
{
  int bihx;
#if DEBUG
  /* convenient place for a segfault */
  if (XBIT(4, 0x2000)) {
    if (!XBIT(4, 0x1000) || gbl.func_count > 2) {
      /* store to null pointer */
      *badpointer1 = 99;
    }
  }
  if (XBIT(4, 0x4000)) {
    if (!XBIT(4, 0x1000) || gbl.func_count > 2) {
      /* divide by zero */
      badnumerator = badnumerator / baddenominator;
    }
  }
  if (XBIT(4, 0x8000)) {
    if (!XBIT(4, 0x1000) || gbl.func_count > 2) {
      /* infinite loop */
      while (badnumerator) {
        badnumerator = (badnumerator < 1) | 3;
      }
    }
  }
#endif
  for (bihx = gbl.entbih; bihx; bihx = BIH_NEXT(bihx)) {
    int b;
    if (BIH_LAST(bihx))
      break;
    for (b = bihx; b;) {
      /* if 'bihx' ends in a branch or jump or call that can throw, stop here */
      bihx = b;
      if (!BIH_FT(bihx))
        break;
      if (BIH_ILTLAST(bihx) && ILT_BR_OR_CAN_THROW(BIH_ILTLAST(bihx)))
        break;
      b = merge_bih(bihx);
    }
  }
} /* unsplit */

/*
 * Split blocks after an internal jump; that is, eliminate extended basic blocks
 */
void
split_extended()
{
  int bihx, iltx, iltnext;
  for (bihx = gbl.entbih; bihx; bihx = BIH_NEXT(bihx)) {
    if (BIH_LAST(bihx))
      break;
    for (iltx = BIH_ILTFIRST(bihx); iltx; iltx = iltnext) {
      int ilix, opc;
      iltnext = ILT_NEXT(iltx);
      ilix = ILT_ILIP(iltx);
      opc = ILI_OPC(ilix);
      if (iltnext && IL_TYPE(opc) == ILTY_BRANCH) {
        /* split this block just after the branch.
         * move ILTs after this one to the new block. */
        int newbihx;
        newbihx = addbih(bihx);
        BIH_ILTFIRST(newbihx) = iltnext;
        BIH_ILTLAST(newbihx) = BIH_ILTLAST(bihx);
        BIH_ILTLAST(bihx) = iltx;
        ILT_NEXT(iltx) = 0;
        ILT_PREV(iltnext) = 0;
        BIH_FT(newbihx) = BIH_FT(bihx);
        BIH_FT(bihx) = 1;
        BIH_EX(newbihx) = BIH_EX(bihx);
        BIH_LDVOL(newbihx) = BIH_LDVOL(bihx);
        BIH_STVOL(newbihx) = BIH_STVOL(bihx);
        BIH_QJSR(newbihx) = BIH_QJSR(bihx);
        BIH_SMOVE(newbihx) = BIH_SMOVE(bihx);
        BIH_NOMERGE(newbihx) = BIH_NOMERGE(bihx);
        BIH_PAR(newbihx) = BIH_PAR(bihx);
        iltnext = 0;
      }
    }
  }
} /* split_extended */

void
dump_blocks(FILE *ff, int bih, const char *fmt, int fihflag)
{
  dump_one_block(ff, bih, fmt);
  if (ff == NULL)
    ff = stderr;
  for (;;) {
    if (BIH_LAST(bih))
      break;
    bih = BIH_NEXT(bih);
    dump_ilt(ff, bih);
  }
  if (fihflag && fihb.stg_base && fihb.stg_avail) {
    int fihx;
    fprintf(ff, "\n*****   FIH Table   *****\n");
    fprintf(ff, "   FIH   FULLNAME\n");
    for (fihx = 1; fihx < fihb.stg_avail; fihx++) {
      fprintf(ff, " %5d.  %s\n", fihx, FIH_FULLNAME(fihx));
    }
  }
}

void
dump_one_block(FILE *ff, int bih, const char *fmt)
{
  if (ff == NULL)
    ff = stderr;
  if (fmt) {
    fprintf(ff, "\n");
    if (bih)
      fprintf(ff, fmt, getprint((int)BIH_LABEL(bih)));
    else
      fprintf(ff, "%s", fmt);
    fprintf(ff, "\n");
  }
  dump_ilt(ff, bih);
}
