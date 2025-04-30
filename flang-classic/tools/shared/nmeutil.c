/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *  \brief names table utility module
 */

#include "nmeutil.h"
#include "error.h"
#include "expand.h"
#if defined(PGF90) && !defined(FE90)
/* Fortran backend only */
#include "upper.h"
#include "mwd.h"
#include "ili.h"
#endif
#include "dtypeutl.h"

#ifndef FE90
#include "ili.h"
#include "soc.h"
#endif
#include "symfun.h"

#ifdef FLANG_NMEUTIL_UNUSED
static bool found_rpct(int rpct_nme1, int rpct_nme2);
#endif

#if DEBUG
#define asrt(c) \
  if (c)        \
    ;           \
  else          \
  fprintf(stderr, "asrt failed. line %d, file %s\n", __LINE__, __FILE__)
#else
#define asrt(c)
#endif

#define NMEHSZ 1217
#define MAXNME 67108864
#define SPTR_ONE  ((SPTR) 1)

static int nmehsh[NMEHSZ];
static int ptehsh[NMEHSZ];
static int rpcthsh[NMEHSZ];

/** \brief Query whether the given nme is a PRE temp. */
bool
is_presym(int nme)
{
  if (nme && (NME_TYPE(nme) == NT_VAR)) {
    SPTR sptr = NME_SYM(nme);

    if ((sptr > 0) && CCSYMG(sptr) && SYMNAME(sptr) &&
        (strncmp(SYMNAME(sptr), ".pre", 4) == 0))
      return true;
  }

  return false;
}

/** \brief Initialize names module
 */
void
nme_init(void)
{
  int i;
  static int firstcall = 1;

  STG_ALLOC(nmeb, 128);
  nmeb.stg_avail = 2; /* 0, NME_UNK; 1, NME_VOL */
  STG_CLEAR_ALL(nmeb);

  NME_TYPE(NME_UNK) = NT_UNK;

  NME_TYPE(NME_VOL) = NT_UNK;
  NME_SYM(NME_VOL) = SPTR_ONE;

  if (firstcall)
    firstcall = 0;
  else {
    for (i = 0; i < NMEHSZ; i++) {
      nmehsh[i] = 0;
      ptehsh[i] = 0;
      rpcthsh[i] = 0;
    }
  }

  STG_ALLOC(nmeb.pte, 128);
  PTE_NEXT(PTE_UNK) = PTE_END;
  PTE_TYPE(PTE_UNK) = PT_UNK;
  PTE_VAL(PTE_UNK) = 0;

  STG_ALLOC(nmeb.rpct, 128);

} /* nme_init */

void
nme_end(void)
{
  STG_DELETE(nmeb);

  STG_DELETE(nmeb.pte);

  STG_DELETE(nmeb.rpct);
} /* nme_end */

/*
 * for F90:
 *   type dt
 *     integer :: i, j
 *   end type
 *   type, extends(dt)::et
 *     integer :: k
 *   end type
 * type(et) :: m
 * a reference to m%i may appear in the ILMs as
 * IM_MEMBER(IM_BASE(m),IM_MEMBER(i))
 * but in the datatype table as
 *   dt-> TY_STRUCT ( i(integer), j(integer) )
 *   et-> TY_STRUCT ( dt(dt), k(integer) )
 * so the datatype table has an extra member, named dt, for the datatype dt
 * we want to insert the reference to %dt%i here
 */
static int
find_parent_member(int nmex, DTYPE dt, int sym)
{
  SPTR mem;
  int newnmex;
  SPTR sptr;
  mem = DTyAlgTyMember(dt);
  if (mem > NOSYM && PARENTG(mem)) {
    DTYPE dtmem = DTYPEG(mem);
    for (sptr = DTyAlgTyMember(dtmem); sptr > NOSYM; sptr = SYMLKG(sptr)) {
      if (sym == sptr)
        break;
    }
    newnmex = add_arrnme(NT_MEM, mem, nmex, 0, 0, 0);
    if (mem > NOSYM)
      return newnmex;
    return find_parent_member(newnmex, dtmem, sym);
  }
  return 0;
} /* find_parent_member */

int
add_arrnme(NT_KIND type, SPTR insym, int nm, ISZ_T cnst, int sub, bool inlarr)
{
  int val, i;
  DTYPE nmdt;
  SPTR sym;

  if (EXPDBG(10, 256))
    fprintf(gbl.dbgfil,
            "type = %d sym = %d nm = %d cnst = %" ISZ_PF "d sub=%d inlarr=%d\n",
            type, insym, nm, cnst, sub, inlarr);
#if DEBUG
  //assert(true, "add_arrnme: bad inlarr", inlarr, ERR_Severe);
#endif

  /* evaluate nme and fold any values  */

  if (insym < SPTR_NULL) {
    DEBUG_ASSERT(insym == NME_NULL, "add_arrnme: bad negative insym");
  } else {
    DEBUG_ASSERT(insym <= SPTR_MAX, "add_arrnme: out of bounds insym");
  }
  sym = insym;

  switch (type) {

  case NT_SAFE:
  case NT_VAR:
  case NT_IND:
    break;

  case NT_ARR:
#ifdef NT_INDARR
  case NT_INDARR:
#endif
#ifndef FE90
    if (sub)
      sub = ili_subscript(sub);
#endif
    if (nm == NME_VOL)
      return NME_VOL;
    { /* tpr 564:
       * for a subscripted reference, ensure the 'base' is not
       * a union or structure.
       */
      DTYPE dt = dt_nme(nm);
      if (DTY(dt) == TY_UNION)
        return nm;
    }
    break;

  case NT_MEM:
    /*
     * don't create union NMEs for fortran; fortran's conflict() does not
     * have the left-to-right walk of the nmes on which checking of unions
     * depends.
     */
    nmdt = dt_nme(nm);
    if (DTY(nmdt) == TY_STRUCT
#ifdef TY_DERIVED
        || DTY(nmdt) == TY_DERIVED
#endif
        ) {
      if (sym > 0) {
#if defined(INLNG)
        if (!INLNG(sym))
#else
        if (1)
#endif
        {
          for (i = DTyAlgTyMember(nmdt); i > NOSYM; i = SYMLKG(i)) {
            if (sym == i)
              goto is_member;
          }
          {
            int nnm;
            /* look if this is an extended member */
            nnm = find_parent_member(nm, nmdt, sym);
            if (nnm) {
              nm = nnm;
              goto is_member;
            }
          }
        } else {
          SPTR i;
          /* the member names are inlined separately;
           * the datatype gets duplicated.
           * allow for this */
          for (i = DTyAlgTyMember(nmdt); i > NOSYM; i = SYMLKG(i)) {
            if (sym == i || (strcmp(SYMNAME(i), SYMNAME(sym)) == 0 &&
                             ADDRESSG(i) == ADDRESSG(sym))) {
              sym = i;
              goto is_member;
            }
          }
        }
      }
    }
    if (nm == NME_VOL || sym > 1) {
      return nm;
    }
    break; /* => real/imag (sym = 0/1) parts of complex */
  is_member:
    break;

  case NT_ADD:
    switch (NME_TYPE(nm)) {
    case NT_INDARR:
    case NT_ADD:
    case NT_SAFE:
      break;

    case NT_VAR:
    case NT_MEM:
    case NT_ARR:
      /*
       * attempt to create an array names entry for these cases. This
       * is only done when nm is for an item which is an array.
       */
      if (DTY(dt_nme(nm)) != TY_ARRAY) {

        /*
         * since nm is not for an array, it is determine if the
         * amount being added is zero.
         */
        if (sym == 0 && cnst == 0)
          return nm; /* go ahead and use nm */

        /*
         * this occurs when adding a nonzero or variable value to
         * an & expression; i.e., &x + 4, &x + i -- it is not
         * known what is actually being referenced.  The addrtkn
         * flag is set for the variable in nm and the unknown
         * names entry is returned.
         */
        loc_of(nm);
        return NME_UNK;
      }
      type = NT_ARR; /* create an array names entry  */
      break;

    case NT_IND:
      if (sym == 0) {
        if (NME_SYM(nm) == 0) {
          cnst += NME_CNST(nm);
          nm = NME_NM(nm);
        } else
          return nm;
      } else {
        nm = NME_NM(nm);
        cnst = 0;
      }
      type = NT_IND;
      sub = 0;
      inlarr = false;
      break;

    case NT_UNK:
      return nm;
    }
    break;
  case NT_UNK:
    return NME_UNK;
  }

  /* compute the hash index for this NME  */
  val = (int)((type ^ sym ^ nm ^ sub) & 0x7fff) % NMEHSZ;

  /* search the hash links for this NME  */
  for (i = nmehsh[val]; i != 0; i = NME_HSHLNK(i))
    if (NME_TYPE(i) == type && NME_INLARR(i) == inlarr && NME_SYM(i) == sym &&
        NME_NM(i) == nm && NME_CNST(i) == cnst && NME_SUB(i) == sub &&
#ifdef NME_PTE
        NME_PTE(i) == 0 &&
#endif
        NME_RPCT_LOOP(i) == 0)
      return i; /* F O U N D  */

  /*
   * N O T   F O U N D -- if no more storage is available, try to get more
   * storage
   */
  i = STG_NEXT(nmeb);
  if (i > MAXNME)
    error((error_code_t)7, ERR_Fatal, 0, CNULL, CNULL);
  /*
   * NEW ENTRY - add the nme to the nme area and to its hash chain
   */
  if (EXPDBG(10, 256))
    fprintf(gbl.dbgfil,
            "adding nme %d:type = %d sym = %d nm = %d cnst = %" ISZ_PF
            "d sub = %d, inlarr=%d\n",
            i, type, sym, nm, cnst, sub, inlarr);
  NME_TYPE(i) = type;
  NME_INLARR(i) = inlarr;
  NME_SYM(i) = sym;
  NME_NM(i) = nm;
  NME_CNST(i) = cnst;
  NME_HSHLNK(i) = nmehsh[val];
  NME_SUB(i) = sub;
  nmehsh[val] = i;
  return i;
}

int
lookupnme(NT_KIND type, int insym, int nm, ISZ_T cnst)
{
  int i, val, sub = 0, sym = insym;
  bool inlarr = false;
  if (insym < 0)
    sym = NME_NULL;

  val = (int)((type ^ sym ^ nm ^ sub) & 0x7fff) % NMEHSZ;
  for (i = nmehsh[val]; i != 0; i = NME_HSHLNK(i)) {
    if (NME_TYPE(i) == type && NME_INLARR(i) == inlarr && NME_SYM(i) == sym &&
        NME_NM(i) == nm && NME_CNST(i) == cnst && NME_SUB(i) == sub &&
#ifdef NME_PTE
        NME_PTE(i) == 0 &&
#endif
        NME_RPCT_LOOP(i) == 0)
      return i; /* F O U N D  */
  }
  return 0;
} /* lookupnme */

/** \brief Add a new nme based on this nme with a new PTE list; otherwise all
           fields the same.
 */
int
add_nme_with_pte(int nm, int ptex)
{
#ifndef NME_PTE
  return nm;
#else
  int val, i;

  val =
      (int)((NME_TYPE(nm) ^ NME_SYM(nm) ^ NME_NM(nm) ^ NME_SYM(nm)) & 0x7fff) %
      NMEHSZ;
  for (i = nmehsh[val]; i > 0; i = NME_HSHLNK(i)) {
    if (NME_TYPE(i) == NME_TYPE(nm) && NME_INLARR(i) == NME_INLARR(nm) &&
        NME_SYM(i) == NME_SYM(nm) && NME_NM(i) == NME_NM(nm) &&
        NME_CNST(i) == NME_CNST(nm) && NME_SUB(i) == NME_SUB(nm) &&
        NME_RPCT_LOOP(i) == NME_RPCT_LOOP(nm) && NME_PTE(i) == ptex)
      return i; /* F O U N D  */
  }
  i = STG_NEXT(nmeb);
  if (i > MAXNME)
    error((error_code_t)7, ERR_Fatal, 0, CNULL, CNULL);
  if (EXPDBG(10, 256))
    fprintf(gbl.dbgfil, "adding based nme %d, based on %d with pte %d\n", i, nm,
            ptex);
  BCOPY(nmeb.stg_base + i, nmeb.stg_base + nm, NME, 1);
  NME_BASE(i) = nm;
  NME_PTE(i) = ptex;
  NME_HSHLNK(i) = nmehsh[val];
  nmehsh[val] = i;
  return i;
#endif
} /* add_nme_with_pte */

/**
 * This creates and returns a new NME, 'rpct_nme', whose fields are
 * the same as those of 'orig_nme' except that it has
 * NME_RPCT_LOOP( rpct_nme ) == rpct_loop (> 0), whereas
 * NME_RPCT_LOOP( orig_nme ) == 0.  'rpct_nme' will replace all
 * occurrences of 'orig_nme' within the loop denoted by 'rpct_loop',
 * whereas 'orig_nme' will continue to be used elsewhere.  This loop
 * is guarded by runtime pointer conflict tests ('RPCT's) which prove
 * at runtime that the 'rpct_nme' references don't conflict with
 * various other references in the loop that they could potentially
 * conflict with, as indicated in the 'RPCT table', and 'conflict()'
 * will use this information to return NOCONFLICT for pairs of such
 * references.
 */
int
add_rpct_nme(int orig_nme, int rpct_loop)
{
  int hashval, rpct_nme;

  asrt(NME_RPCT_LOOP(orig_nme) == 0 && rpct_loop > 0);

  /* Compute the hash index for 'orig_nme' and 'rpct_nme' (they both
   * have the same hash index).
   */
  hashval = (int)((NME_TYPE(orig_nme) ^ NME_SYM(orig_nme) ^ NME_NM(orig_nme) ^
                   NME_SUB(orig_nme)) &
                  0x7fff) %
            NMEHSZ;

#if DEBUG
  /* Search the nme hash links for this NME.  If it already exists
   * we return it and generate an 'asrt()' error message, since this
   * shouldn't happen.
   */
  for (rpct_nme = nmehsh[hashval]; rpct_nme; rpct_nme = NME_HSHLNK(rpct_nme)) {
    if (NME_TYPE(rpct_nme) == NME_TYPE(orig_nme) &&
        NME_INLARR(rpct_nme) == NME_INLARR(orig_nme) &&
        NME_SYM(rpct_nme) == NME_SYM(orig_nme) &&
        NME_NM(rpct_nme) == NME_NM(orig_nme) &&
        NME_CNST(rpct_nme) == NME_CNST(orig_nme) &&
        NME_SUB(rpct_nme) == NME_SUB(orig_nme) &&
#ifdef NME_PTE
        NME_PTE(rpct_nme) == NME_PTE(orig_nme) &&
#endif
        NME_RPCT_LOOP(rpct_nme) == rpct_loop) {
      /* Found.
       */
      asrt(false); /* we don't expect it to be found! */
      return rpct_nme;
    }
  }
#endif

  /* Not found, so create and initialise a new nme, and link it info
   * its hash chain.  If necessary get more storage.
   */
  rpct_nme = STG_NEXT(nmeb);

  if (rpct_nme > MAXNME)
    error((error_code_t)7, ERR_Fatal, 0, CNULL, CNULL);

  if (EXPDBG(10, 256))
    fprintf(gbl.dbgfil,
            "adding rpct nme %d, based on nme %d, in rpct loop %d\n", rpct_nme,
            orig_nme, rpct_loop);

  BCOPY(nmeb.stg_base + rpct_nme, nmeb.stg_base + orig_nme, NME, 1);

  NME_RPCT_LOOP(rpct_nme) = rpct_loop;
  NME_HSHLNK(rpct_nme) = nmehsh[hashval];
  nmehsh[hashval] = rpct_nme;

  return rpct_nme;

} /* end add_rpct_nme( int orig_nme, int rpct_loop ) */

SPTR
addnme(NT_KIND type, SPTR insym, int nm, ISZ_T cnst)
{
  return (SPTR) add_arrnme(type, insym, nm, cnst, 0, false);
}

/** \brief Build an nme entry using a sym and an offset relative
           to base address of this symbol */
int
build_sym_nme(SPTR sym, int offset, bool ptr_mem_op)
{
  int nme;
  int sub;
  bool inlarr;
  DTYPE dt;
  int i;

  if (!sym)
    return 0;

  if (ptr_mem_op)
    return 0;

  nme = addnme(NT_VAR, sym, 0, 0);
  dt = DTYPEG(sym);
  if (DTY(dt) == TY_PTR) {
    sub = 0;
    inlarr = false;
    for (i = nme; i < nmeb.stg_avail; i++) {
      /* Check if subscript doesn't already exist */
      if (NME_TYPE(i) == NT_IND && NME_NM(i) == nme && NME_SYM(i) == 0 &&
          NME_CNST(i) == 0) {
        sub = NME_SUB(i);
#ifndef FE90
        if (!ili_isdeleted(sub)) {
#endif
          inlarr = NME_INLARR(i);
          break;
#ifndef FE90
        } else
          sub = 0;
#endif
      }
    }
    nme = add_arrnme(NT_IND, SPTR_NULL, nme, 0, sub, inlarr);
    dt = DTySeqTyElement(dt);
  }
  return 0;
}

/** \brief Get data type of a names entry
 */
DTYPE
dt_nme(int nm)
{
  int i;
  switch (NME_TYPE(nm)) {
  case NT_INDARR:
  case NT_ADD:
  case NT_UNK:
    break;
  case NT_VAR:
    return DTYPEG(NME_SYM(nm));
  case NT_MEM:
    if (NME_SYM(nm) == 0 || NME_SYM(nm) == 1) {
      if (dt_nme((int)NME_NM(nm)) == DT_DCMPLX) {
        return DT_DBLE;
      } else {
        return DT_REAL;
      }
    }
    if ((i = NME_CNST(nm)) == 0)
      return DTYPEG(NME_SYM(nm));
    else
      return DTYPEG(i);
  case NT_ARR: {
    DTYPE i = dt_nme((int)NME_NM(nm));
    if (DTY(i) == TY_ARRAY)
      return DTySeqTyElement(i);
    /*
     * for fortran, the TY_ARRAY dtype only occurs once; the element
     * type is returned after the array dtype is returned when we hit
     * the NT_VAR or NT_MEM case.  All we need to do is to just pass
     * up the dtype.  NOTE that this will work if try to fake arrays
     * using PLISTs.
     */
    return i;
  }
  case NT_IND: {
    DTYPE i = dt_nme(NME_NM(nm));
    if (DTY(i) == TY_PTR)
      return DTySeqTyElement(i);
    return i;
  }
  case NT_SAFE:
    return dt_nme(NME_NM(nm));
  }
  return DT_NONE;
}

/** \brief Location of a names entry
 *
 * LOC (explicit or implicit) has been performed.  The names entry for the
 * lvalue is scanned to determine if a symbol's ADDRTKN flag must be set
 */
void
loc_of(int nme)
{
  int type;

  while (true) {
    if ((type = NME_TYPE(nme)) == NT_VAR) {
      if (!is_presym(nme))
        ADDRTKNP(NME_SYM(nme), 1);
      break;
    }
    if (type == NT_ARR || type == NT_MEM)
      nme = NME_NM(nme);
    else
      break;
  }
}

/** \brief Location of a name entry when a reference is volatile
 *
 * LOC (explicit or implicit) has been performed.  The names entry for the
 * lvalue is scanned to determine if a symbol's ADDRTKN flag must be set.  This
 * function is called when the reference is volatile; consequently, pointer
 * loads must be traversed.
 */
void
loc_of_vol(int nme)
{
  int type;

  while (true) {
    if ((type = NME_TYPE(nme)) == NT_VAR) {
      ADDRTKNP(NME_SYM(nme), 1);
      break;
    }
    if (type == NT_ARR || type == NT_MEM || type == NT_IND)
      nme = NME_NM(nme);
    else
      break;
  }
}

#ifdef PTRSTOREP
/**
   \brief LOC (explicit or implicit) has been performed.  The names entry for
   the lvalue is scanned to determine if a symbol's PTRSTOREP flag must be set
*/
void
ptrstore_of(int nme)
{
  int type;

  while (true) {
    if ((type = NME_TYPE(nme)) == NT_VAR) {
      PTRSTOREP(NME_SYM(nme), 1);
      break;
    }
    if (type == NT_ARR || type == NT_MEM)
      nme = NME_NM(nme);
    else
      break;
  }
}
#endif

/** Walk through the given nme and its chains of base nmes, query whether all
 * of them are static, i.e., a struct, array, var other than an object
 * dereferenced by a pointer.
 */
bool
basenme_is_static(int nme)
{
  while (true) {
    switch (NME_TYPE(nme)) {
    case NT_IND:
      return false;
    case NT_MEM:
    case NT_ARR:
    case NT_SAFE:
      nme = NME_NM(nme);
      break;
    default:
      return true;
    }
  }
}

/** \brief Return the base st index of the names entry
 *
 * Returns the base symbol of a reference given its names entry -- returns 0 if
 * unknown.
 */
SPTR
basesym_of(int nme)
{
  while (true) {
    switch (NME_TYPE(nme)) {
    case NT_MEM:
    case NT_IND:
    case NT_ARR:
    case NT_SAFE:
      nme = NME_NM(nme);
      break;
    case NT_VAR:
      return NME_SYM(nme);
    default:
      goto not_found;
    }
  }
not_found:
  return SPTR_NULL;
}

/** \brief Return the base nme of a reference given its names entry;
           return 0 if unknown.
 */
int
basenme_of(int nme)
{
  while (true) {
    switch (NME_TYPE(nme)) {
    case NT_MEM:
    case NT_IND:
    case NT_ARR:
    case NT_SAFE:
      nme = NME_NM(nme);
      break;
    default:
      goto found;
    }
  }
found:
  return nme;
}

/** \brief Return the base nme of a reference given its names entry,
           return 0 if unknown.

    WARNING - zbasenme_of() cannot traverse NT_IND

    SOMEDAY, replace with basenme_of() when flow's def/use can distinguish
    between a 'base' which is static ( a struct, array, var) or an object
    located by a pointer.
 */
int
zbasenme_of(int nme)
{
  while (true) {
    switch (NME_TYPE(nme)) {
    case NT_MEM:
    case NT_ARR:
    case NT_SAFE:
      nme = NME_NM(nme);
      break;
    default:
      goto found;
    }
  }
found:
  return nme;
}

/** \brief Add pointer target entry
 */
int
addpte(int type, SPTR sptr, int val, int next)
{
  int hshval, p;
  hshval = (int)((type ^ sptr ^ val ^ next) & 0x7fff) % NMEHSZ;
  for (p = ptehsh[hshval]; p > 0; p = PTE_HSHLNK(p)) {
    if (PTE_TYPE(p) == type && PTE_SPTR(p) == sptr && PTE_VAL(p) == val &&
        PTE_NEXT(p) == next) {
      return p;
    }
  }
  p = STG_NEXT(nmeb.pte);
  PTE_TYPE(p) = type;
  PTE_SPTR(p) = sptr;
  PTE_VAL(p) = val;
  PTE_NEXT(p) = next;
  PTE_HSHLNK(p) = ptehsh[hshval];
  ptehsh[hshval] = p;
  return p;
} /* addpte */

/**
 * \brief Add a "runtime pointer conflict test" record
 *
 * This creates and returns a new RPCT (runtime pointer conflict test)
 * record containing the pair of NMEs (rpct_nme1, rpct_nme2).  These
 * NMEs are 'RPCT NMEs' that were created by 'add_rpct_nme()'.  The
 * existence of this RPCT record indicates that a pointer conflict
 * test has been generated which proves at runtime that this pair of
 * references do not conflict, so 'conflict( rpct_nme1, rpct_nme2 )'
 * will return NOCONFLICT.
 */
void
add_rpct(int rpct_nme1, int rpct_nme2)
{
  int hashval, rpct;

  asrt(rpct_nme1 != rpct_nme2 && NME_RPCT_LOOP(rpct_nme1) &&
       NME_RPCT_LOOP(rpct_nme2) == NME_RPCT_LOOP(rpct_nme1));

  /* If necessary swap 'rpct_nme1' & 'rpct_nme2' so that
   * (rpct_nme1 < rpct_nme2), since that makes it quicker to search
   * the RPCT table for a match.
   */
  if (rpct_nme1 > rpct_nme2) {
    int tmp;

    tmp = rpct_nme1;
    rpct_nme1 = rpct_nme2;
    rpct_nme2 = tmp;
  }

  /* Compute the hash index for this RPCT record.
   */
  hashval = (int)((rpct_nme1 ^ rpct_nme2) & 0x7fff) % NMEHSZ;

#if DEBUG
  /* Search the RPCT hash links for this RPCT.  If it already exists
   * we generate an 'asrt()' error message, since this shouldn't happen.
   */
  for (rpct = rpcthsh[hashval]; rpct; rpct = RPCT_HSHLNK(rpct)) {
    if (RPCT_NME1(rpct) == rpct_nme1 && RPCT_NME2(rpct) == rpct_nme2) {
      /* Found.
       */
      asrt(false); /* we don't expect it to be found! */
      return;
    }
  }
#endif

  /* Create and initialise a new RPCT record and link it into its hash chain.
   */
  rpct = STG_NEXT(nmeb.rpct);
  RPCT_NME1(rpct) = rpct_nme1;
  RPCT_NME2(rpct) = rpct_nme2;
  RPCT_HSHLNK(rpct) = rpcthsh[hashval];
  rpcthsh[hashval] = rpct;

} /* end add_rpct( int rpct_nme1, int rpct_nme2 ) */

#ifdef FLANG_NMEUTIL_UNUSED
/**
 * This function returns true if there is an RPCT (runtime pointer
 * conflict test) record containing the pair of NMEs (rpct_nme1,
 * rpct_nme2), and false otherwise.  In the former case
 * 'conflict( rpct_nme1, rpct_nme2 )' will return NOCONFLICT, as
 * explained in the comment for function 'add_rpct()'.
 */
static bool
found_rpct(int rpct_nme1, int rpct_nme2)
{
  int hashval, rpct;

  asrt(rpct_nme1 != rpct_nme2 && NME_RPCT_LOOP(rpct_nme1) &&
       NME_RPCT_LOOP(rpct_nme2) == NME_RPCT_LOOP(rpct_nme1));

  /* If necessary swap 'rpct_nme1' & 'rpct_nme2' so that
   * (rpct_nme1 < rpct_nme2).
   */
  if (rpct_nme1 > rpct_nme2) {
    int tmp;

    tmp = rpct_nme1;
    rpct_nme1 = rpct_nme2;
    rpct_nme2 = tmp;
  }

  /* Compute the hash index for this RPCT record.
   */
  hashval = (int)((rpct_nme1 ^ rpct_nme2) & 0x7fff) % NMEHSZ;

  /* Search the RPCT hash links for this RPCT.
   */
  for (rpct = rpcthsh[hashval]; rpct; rpct = RPCT_HSHLNK(rpct)) {
    if (RPCT_NME1(rpct) == rpct_nme1 && RPCT_NME2(rpct) == rpct_nme2) {
      return true; /* found */
    }
  }

  return false; /* not found */

} /* end found_rpct( int rpct_nme1, int rpct_nme2 ) */
#endif

#ifndef FE90
/* #if defined(I386) || defined(X86_64) || defined(X86_32) || defined(LX) ||
 * defined(SPARC) || defined(ST100) */

static int hlcf = 0;

#if DEBUG
extern int _conflict(int, int);
static int conflict_count[5] = {0, 0, 0, 0, 0};

/** \brief Determine conflicts between the loads/stores represented by the two
 * names table pointers
 */
int
conflict(int nm1, int nm2)
{
  int r;
  r = _conflict(nm1, nm2);
  if (DBGBIT(10, 0x10000) || (r != NOCONFLICT && DBGBIT(10, 0x40000))) {
    int s = r;
    static const char *cname[] = {"same", "noconflict", "conflict",
                                  "unconflict", "?"};
    if (s < -1 || s > 3)
      s = 4;
    fprintf(gbl.dbgfil, "conflict(%d=", nm1);
    printnme(nm1);
    fprintf(gbl.dbgfil, ",%d=", nm2);
    printnme(nm2);
    fprintf(gbl.dbgfil, ")=%s\n", cname[s + 1]);
    if (DBGBIT(10, 0x20000)) {
      dumpnme(nm1);
      if (nm1 != nm2) {
        dumpnme(nm2);
      }
    }
  }
  if (r >= -1 && r <= 2) {
    ++conflict_count[r + 1];
  } else {
    ++conflict_count[4];
  }
  return r;
}

#define conflict _conflict

void
count_conflict(void)
{
  FILE *d;
  d = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(d, "conflicts: %5d=same, %5d=conflict, %5d=noconflict, %5d=unknown",
          conflict_count[0], conflict_count[2], conflict_count[1],
          conflict_count[3]);
  if (conflict_count[4])
    fprintf(d, " %5d=???", conflict_count[4]);
  if (GBL_CURRFUNC)
    fprintf(d, "    %s", SYMNAME(GBL_CURRFUNC));
  fprintf(d, "\n");
  conflict_count[0] = 0;
  conflict_count[1] = 0;
  conflict_count[2] = 0;
  conflict_count[3] = 0;
  conflict_count[4] = 0;
}
#endif

/**\brief Determine if the two names table entries, nm1 and nm2, conflict.
 *
 * Return SAME, CONFLICT, NOCONFLICT or UNKCONFLICT
 */
int
conflict(int nm1, int nm2)
{
  int t1, t2, n1, n2, sptr2, sptr1, c;

  /*  if necessary, switch nm1 and nm2 so that t2 >= t1
      (reduces number of cases later):  */

  if (nm1 == NME_VOL || nm2 == NME_VOL)
    return CONFLICT;

  if ((t2 = NME_TYPE(nm2)) < (t1 = NME_TYPE(nm1))) {
    c = nm1;
    nm1 = nm2;
    nm2 = c;
    t1 = t2;
    t2 = NME_TYPE(nm2);
  }

  if (gbl.internal > 1 && nm1 != nm2) {
    if (NME_TYPE(nm1) == NT_VAR && NME_SYM(nm1) == aux.curr_entry->display)
      return NOCONFLICT;
    if (NME_TYPE(nm2) == NT_VAR && NME_SYM(nm2) == aux.curr_entry->display)
      return NOCONFLICT;
  }

  if (t2 == NT_MEM || t2 == NT_ARR) {
    /*  scan back past member and array records:  */

    for (n1 = nm1; NME_TYPE(n1) >= NT_MEM; n1 = NME_NM(n1))
      ;
    for (n2 = nm2; NME_TYPE(n2) >= NT_MEM; n2 = NME_NM(n2))
      ;

    c = conflict(n1, n2);
    /* check if one if .real and one is .imag */
    if (t1 == NT_MEM && t2 == NT_MEM &&
        ((int)(NME_SYM(nm1) | NME_SYM(nm2)) <= 1) &&
        NME_SYM(nm1) != NME_SYM(nm2)) {
      return NOCONFLICT;
    }
    if (!XBIT(104, 0x10) && (c == CONFLICT || c == UNKCONFLICT)) {
      int m1 = 0, m2 = 0;
      int n1m, n2m;
      /* if either nm1 or nm2 has a member or array of a member
       * that is marked as noconflict, there can be no conflict
       * this is particularly to handle section descriptor members */
      n1m = nm1;
      while (NME_TYPE(n1m) == NT_ARR)
        n1m = NME_NM(n1m);
      /* problem with block moves  */
      if (NME_TYPE(n1m) == NT_IND && NME_SYM(n1m) == NME_NULL) {
        return UNKCONFLICT;
      }
      if (NME_TYPE(n2) == NT_IND && NME_SYM(n2) == NME_NULL) {
        return UNKCONFLICT;
      }
      if (NME_TYPE(n1m) == NT_MEM) {
        m1 = NME_SYM(n1m);
      }
      n2m = nm2;
      while (NME_TYPE(n2m) == NT_ARR)
        n2m = NME_NM(n2m);
      if (NME_TYPE(n2m) == NT_MEM) {
        m2 = NME_SYM(n2m);
      }
      if (m1 > NOSYM && m1 == m2 && NOCONFLICTG(m1)) {
        /* same noconflict member */
        if (NME_TYPE(nm1) == NT_ARR && NME_TYPE(nm2) == NT_ARR) {
          /* constant dimension? */
          if (NME_SYM(nm1) == 0 && NME_SYM(nm2) == 0) {
            if (NME_CNST(nm1) != NME_CNST(nm2))
              return NOCONFLICT;
            return SAME;
          }
        }
        /* same member */
        return UNKCONFLICT;
      } else {
/* if either is a descriptor symbol, no conflict */
#ifdef DESCARRAYG
        if (m1 > NOSYM && DESCARRAYG(m1))
          return NOCONFLICT;
        if (m2 > NOSYM && DESCARRAYG(m2))
          return NOCONFLICT;
#endif
        /* if either comes from inlining, there might be a conflict
         * even if NOCONFLICT is set on the member */
        {
          int nn1, nn2, sptr1, sptr2;
          for (nn1 = n1; NME_NM(nn1); nn1 = NME_NM(nn1))
            ;
          for (nn2 = n2; NME_NM(nn2); nn2 = NME_NM(nn2))
            ;
          sptr1 = NME_SYM(nn1);
          sptr2 = NME_SYM(nn2);
          if (sptr1 > NOSYM && INLNG(sptr1) && UNSAFEG(sptr1))
            return c;
          if (sptr2 > NOSYM && INLNG(sptr2) && UNSAFEG(sptr2))
            return c;
        }

        /* if either member is a NOCONFLICT symbol, no conflict */
        if (m1 > NOSYM && NOCONFLICTG(m1))
          return NOCONFLICT;
        if (m2 > NOSYM && NOCONFLICTG(m2))
          return NOCONFLICT;
      }
    }
    if (c != SAME)
      return c;

    /*  check if these are complex elements:  */

    if (t1 == NT_MEM && t2 == NT_MEM) {

      /* If complex, real NME_SYM==0, imag NME_SYM==1 */

      /* ****************** */
      /* Test for 2 Complex */
      /* ****************** */

      if ((int)(NME_SYM(nm1) | NME_SYM(nm2)) <= 1) {

        /* If both complex but one is real part *
         *    and one is complex part, then OK  */

        if (NME_SYM(nm1) != NME_SYM(nm2))
          return NOCONFLICT;

        /* Set up for traversing back into the array processing */
        nm1 = NME_NM(nm1);
        nm2 = NME_NM(nm2);

      }
      /* ****************************** */
      /* Test for 1 Complex w/ 1 Member */
      /* ****************************** */
      else if ((NME_SYM(nm1) <= 1 || NME_SYM(nm2) <= 1)) {
        if (!XBIT(70, 0x40000000))
          return NOCONFLICT;
        else if (NME_SYM(nm1) <= 1) {
          if (NME_SYM(nm2) == NME_NM(nm1))
            return CONFLICT;
          else
            return NOCONFLICT;
        } else {
          if (NME_SYM(nm1) == NME_NM(nm2))
            return CONFLICT;
          else
            return NOCONFLICT;
        }
      }
      /* ****************** */
      /* Test for 2 Members */
      /* ****************** */
      else {
        int sptr1, sptr2;

        if (nm1 == nm2)
          return conflict(NME_NM(nm1), NME_NM(nm2));

        sptr1 = NME_SYM(nm1);
        sptr2 = NME_SYM(nm2);
        for (;;) {
          sptr1 = SYMLKG(sptr1);
          if (sptr1 <= NOSYM)
            break;
          if (sptr1 == sptr2)
            return NOCONFLICT;
        }
        sptr1 = NME_SYM(nm1);
        for (;;) {
          sptr2 = SYMLKG(sptr2);
          if (sptr2 <= NOSYM)
            break;
          if (sptr2 == sptr1)
            return NOCONFLICT;
        }
        return CONFLICT;
      }
    } else if (t1 == NT_MEM) {
      /* Then t2 == NT_ARR and they are SAME */
      int sptr1, sptr2;

      if (nm1 == nm2)
        return conflict(NME_NM(nm1), nm2);

      sptr1 = NME_SYM(nm1);

      for (n2 = nm2; NME_TYPE(n2) == NT_ARR; n2 = NME_NM(n2))
        ;

      sptr2 = NME_SYM(n2);
      if (sptr2 <= NOSYM) {
        assert(0, "conflict: Unexpected member/array conflict. nm2: ", nm2,
               ERR_Informational);
        return CONFLICT;
      }

      for (;;) {
        sptr1 = SYMLKG(sptr1);
        if (sptr1 <= NOSYM)
          break;
        if (sptr1 == sptr2)
          return NOCONFLICT;
      }
      sptr1 = NME_SYM(nm1);
      for (;;) {
        sptr2 = SYMLKG(sptr2);
        if (sptr2 <= NOSYM)
          break;
        if (sptr2 == sptr1)
          return NOCONFLICT;
      }
      return CONFLICT;
    } else if (t2 == NT_MEM) {
/* same base nme, but one is a MEM and one a VAR */
#if DEBUG
      assert(t1 == NT_VAR, "conflict: t1 not NT_VAR", t1, ERR_Severe);
#endif
      return CONFLICT;
    }

    /* They are possibly both arrays.  Check now that they
     * might be arrays that are members
     */
    for (n1 = nm1, n2 = nm2; NME_TYPE(n1) == NT_ARR && NME_TYPE(n2) == NT_ARR;
         n1 = NME_NM(n1), n2 = NME_NM(n2))
      ;
    if (NME_TYPE(n1) == NT_MEM || NME_TYPE(n2) == NT_MEM) {
      if (NME_TYPE(n1) == NT_MEM && NME_TYPE(n2) == NT_MEM && n1 != n2) {
        /* If members are not identical then noconflict (right?) */
        if (!XBIT(104, 1))
          return NOCONFLICT;
      }
      return CONFLICT;
    }

    /*  scan back thru array dimensions to find non-equal const dim: */

    c = SAME;
    for (n1 = nm1, n2 = nm2; NME_TYPE(n1) == NT_ARR;
         n1 = NME_NM(n1), n2 = NME_NM(n2)) {
      if (NME_TYPE(n2) != NT_ARR) {
        if (flg.depchk)
          return CONFLICT;
        return UNKCONFLICT;
      }
      if (NME_SYM(n1) == 0 && NME_SYM(n2) == 0) {
        /*  both dimensions are constant.  */
        if (NME_CNST(n1) != NME_CNST(n2))
          return NOCONFLICT;
      } else
        c = UNKCONFLICT;
    }

    return c;
  }

  switch (t1) {
  case NT_UNK:
    if (t2 == NT_UNK)
      return CONFLICT;
    else if (t2 != NT_VAR)
      break;
    sptr2 = NME_SYM(nm2);
    if (STYPEG(sptr2) == ST_CONST) {
      return NOCONFLICT;
    }
    if (SOCPTRG(sptr2))
      break;
    return NOCONFLICT;

  case NT_VAR:
    if (t2 == NT_VAR) {
      if (nm1 == nm2)
        return SAME;
      sptr1 = NME_SYM(nm1);
      sptr2 = NME_SYM(nm2);
      if (STYPEG(sptr1) == ST_CONST) {
        return NOCONFLICT;
      }
      if (STYPEG(sptr2) == ST_CONST) {
        return NOCONFLICT;
      }
      /*  check for overlapping symbols:  */
      if (SOCPTRG(sptr1) && flg.depchk) {
        if (SOCPTRG(sptr2))
          for (t1 = SOCPTRG(sptr1); t1; t1 = SOC_NEXT(t1))
            if (sptr2 == SOC_SPTR(t1))
              return CONFLICT;
      }

      /* If sptr1 and sptr2 are overlapping common block variables, do not
       * optimize their loads/stores, like how variables in an equivalence
       * statement are handled.
       */
      if (is_overlap_cmblk_var(sptr1, sptr2))
        return CONFLICT;
      if (!hlcf || XBIT(104, 0x8)) {
        if (SCG(sptr1) == SC_BASED && INLNG(sptr1) && UNSAFEG(sptr1) &&
            SCG(sptr2) == SC_BASED && INLNG(sptr2) && UNSAFEG(sptr2))
          /* HACK-in dependence for multiple instances of inlined
           * functions and inline-created Cray pointees.
           */
          return UNKCONFLICT;

        if (SCG(sptr1) == SC_BASED && INLNG(sptr1) && UNSAFEG(sptr1)) {
          int dty;
          dty = DTY(DTYPEG(sptr2));
          if ((dty == TY_ARRAY || dty == TY_STRUCT || dty == TY_UNION) &&
              ADDRTKNG(sptr2)
#ifdef PTRSAFEG
              && !PTRSAFEG(sptr2)
#endif
                  ) {
            return UNKCONFLICT;
          }
        } else if (SCG(sptr2) == SC_BASED && INLNG(sptr2) && UNSAFEG(sptr2)) {
          int dty;
          dty = DTY(DTYPEG(sptr1));
          if ((dty == TY_ARRAY || dty == TY_STRUCT || dty == TY_UNION) &&
              ADDRTKNG(sptr1)
#ifdef PTRSAFEG
              && !PTRSAFEG(sptr1)
#endif
                  ) {
            return UNKCONFLICT;
          }
        }
      }
      return NOCONFLICT;
    }
    break;
  case NT_IND:
#ifndef FE90
    /* F90 back end */
    if (!F90_nme_conflict(nm1, nm2)) {
      return NOCONFLICT;
    }
#endif
    if (t2 == NT_VAR) {
      sptr2 = NME_SYM(nm2);
      if (STYPEG(sptr2) == ST_CONST) {
        return NOCONFLICT;
      }
      if (SOCPTRG(sptr2) == 0 &&
          (NOCONFLICTG(sptr2) || (CCSYMG(sptr2) && XBIT(104, 0x20)))) {
        /* nme1 is a pointer, nme2 is a symbol
         * not declared as TARGET, so no conflict */
        return NOCONFLICT;
      }
      if (PTRSAFEG(sptr2)) {
        return NOCONFLICT;
      }
    } else if (t2 == NT_ARR) {
      int nm22, t22;
      for (nm22 = nm2; NME_TYPE(nm22) == NT_ARR; nm22 = NME_NM(nm22))
        ;
      t22 = NME_TYPE(nm22);
      if (t22 == NT_VAR) {
        sptr2 = NME_SYM(nm2);
        if (STYPEG(sptr2) == ST_CONST) {
          return NOCONFLICT;
        }
        if (SOCPTRG(sptr2) == 0 && NOCONFLICTG(sptr2)) {
          /* nme1 is a pointer, nme2 is an array symbol
           * not declared as TARGET, so no conflict */
          return NOCONFLICT;
        }
      }
    }
    break;
  }

  /* for fortran, may reach here because of inlining or use of ptr types... */
  return CONFLICT;
} /* endroutine conflict */

/* Query whether the given nme represents loads and stores for remaining parts
 * of
 * a structure copy.
 */

bool
is_smove_member(int nme)
{
  if ((NME_TYPE(nme) == NT_MEM) && (NME_SYM(nme) > 1)) {
    int sym = NME_SYM(nme);
    if (strncmp(SYMNAME(sym), "..__smove__", 11) == 0)
      return true;
  }

  return false;
}

#ifdef conflict
#undef conflict
#endif

/** \brief Determine if the two names table entries, nm1 and nm2, conflict.
 *
 * Return SAME, CONFLICT, NOCONFLICT or UNKCONFLICT
 */
int
hlconflict(int nm1, int nm2)
{
  int cf;

  hlcf = 1;
  cf = conflict(nm1, nm2);
  hlcf = 0;
  return cf;
} /* endroutine hlconflict */

/* #endif */
#endif /* end #ifndef FE90 */

/** \brief Return the base symbol of a reference given its names entry;
           return 0 if unknown or compiler-created.
*/
int
usersym_of(int nme)
{
  int sym;

  sym = basesym_of(nme);
  if (sym == 0 || CCSYMG(sym))
    return 0;
  return sym;
}

static void
prsym(int sym, FILE *ff)
{
  const char *p;
  p = getprint((int)sym);
  fprintf(ff, "%s", p);
  if (strncmp("..inline", p, 8) == 0)
    fprintf(ff, "%d", sym);
}

/* FIXME: this functionality is duplicated and extended in mwd.c.
   Merge these functions with their duplicates in mwd.c */
/**
 * Prints the symbol reference represented by a names entry and
 * returns the base symbol of a reference given its names entry --
 * this is for scalar and structure references only
 */
int
__print_nme(FILE *ff, int nme)
{
  int i;

  if (ff == NULL)
    ff = stderr;
  switch (NME_TYPE(nme)) {
  case NT_VAR:
    i = NME_SYM(nme);
    prsym(i, ff);
    break;
  case NT_MEM:
    i = print_nme((int)NME_NM(nme));
    if (NME_SYM(nme) == 0) {
      fprintf(ff, ".real");
      break;
    }
    if (NME_SYM(nme) == 1) {
      fprintf(ff, ".imag");
      break;
    }
    fprintf(ff, ".%s", getprint((int)NME_SYM(nme)));
    break;
  default:
    interr("print_nme:ill.sym", nme, ERR_Severe);
    i = 0;
    break;
  }

  return i;
}

int
print_nme(int nme)
{
  FILE *ff = gbl.dbgfil;
  return __print_nme(ff, nme);
}

#if DEBUG
void
__dmpnme(FILE *f, int i, int flag)
{
  FILE *ff;
  int j;

  ff = f;
  if (f == NULL)
    ff = stderr;
  if (!flag)
    fprintf(ff, "%5u   ", i);
  else
    fprintf(ff, "%5u   rfptr %d sub %d hshlk %d f6 %d inlarr %d\n\t", i,
            NME_RFPTR(i), NME_SUB(i), NME_HSHLNK(i), NME_DEF(i), NME_INLARR(i));
  switch (NME_TYPE(i)) {
  case NT_VAR:
    j = NME_SYM(i);
    fprintf(ff, "variable            sym:%5u  \"", j);
    prsym(j, ff);
    fprintf(ff, "\"\n");
    break;
  case NT_ARR:
    if (NME_SYM(i) == NME_NULL)
      fprintf(ff, "variable array       nm:%5u", NME_NM(i));
    else
      fprintf(ff, "constant array       nm:%5u    cnst:%5" ISZ_PF "d",
              NME_NM(i), NME_CNST(i));
    fprintf(ff, "    sub: %5u %s", NME_SUB(i), NME_INLARR(i) ? "<inl>" : " ");
    if (NME_OVS(i)) {
      fprintf(ff, "<ovs>");
    }
    fprintf(ff, "\n");
    break;
  case NT_MEM:
    j = NME_SYM(i);
    if (j == 0) {
      fprintf(ff, "member              sym:%5u      nm:%5u  \"real\"\n", j,
              NME_NM(i));
      break;
    }
    if (j == 1) {
      fprintf(ff, "member              sym:%5u      nm:%5u  \"imag\"\n", j,
              NME_NM(i));
      break;
    }
    j = NME_SYM(i);
    fprintf(ff, "member              sym:%5u      nm:%5u  \"%s\"\n", j,
            NME_NM(i), getprint((int)j));
    break;
  case NT_IND:
    if (NME_SYM(i) == NME_NULL)
      fprintf(ff, "variable indirection nm:%5u\n", NME_NM(i));
    else
      fprintf(ff, "constant indirection nm:%5u    cnst:%5" ISZ_PF "d\n",
              NME_NM(i), NME_CNST(i));
    break;
  case NT_SAFE:
    fprintf(ff, "safe nme             nm:%5u\n", NME_NM(i));
    break;
  case NT_UNK:
    if (NME_SYM(i))
      fprintf(ff, "unknown/volatile\n");
    else
      fprintf(ff, "unknown\n");
    break;
  default:
    interr("__dmpnme: illegal nme", NME_TYPE(i), ERR_Severe);
    fprintf(ff, "\n");
  }
}

/// \brief Dump names table (all fields)
static void
dmpnmeall(int flag)
{
  int i, j;
  int tmp;

  fprintf(gbl.dbgfil, "\n\n***** NME Area Dump *****\n\n");
  for (i = 0; i < nmeb.stg_avail; i++) {
    __dmpnme(gbl.dbgfil, i, flag);
  }

  if ((flg.dbg[10] & 8) != 0) {
    fprintf(gbl.dbgfil, "\n\n***** NME Hash Table *****\n");
    for (i = 0; i < NMEHSZ; i++)
      if ((j = nmehsh[i]) != 0) {
        tmp = 0;
        fprintf(gbl.dbgfil, "%3d.", i);
        for (; j != 0; j = NME_HSHLNK(j)) {
          fprintf(gbl.dbgfil, " %5u^", j);
          if ((++tmp) == 6) {
            tmp = 0;
            fprintf(gbl.dbgfil, "\n    ");
          }
        }
        if (tmp != 0)
          fprintf(gbl.dbgfil, "    \n");
      }
  }
}

void
dmpnme(void)
{
  dmpnmeall(0);
}
#endif

static void
DumpnameHelper(FILE *f, int opn)
{
  static int level = 0;
  FILE *ff;

#if DEBUG
  ff = f;
  if (f == NULL)
    ff = stderr;

  if (opn < 0 || opn >= nmeb.stg_size) {
    interr("DumpnameHelper:bad names ptr", opn, ERR_Severe);
    fprintf(ff, " %5u <BAD>", opn);
    return;
  }

  if (level == 0)
    fprintf(ff, " %5u~ <", opn);

  level++;

  switch (NME_TYPE(opn)) {
  case NT_INDARR:
  case NT_ADD:
    break;
  case NT_VAR:
    prsym(NME_SYM(opn), ff);
    break;
  case NT_MEM:
    DumpnameHelper(ff, NME_NM(opn));
    if (NME_SYM(opn) == 0) {
      fprintf(ff, "->real");
      break;
    }
    if (NME_SYM(opn) == 1) {
      fprintf(ff, "->imag");
      break;
    }
    fprintf(ff, "->%s", getprint((int)NME_SYM(opn)));
    break;
  case NT_IND:
    fprintf(ff, "*(");
    DumpnameHelper(ff, NME_NM(opn));
    if (NME_SYM(opn) == 0)
      if (NME_CNST(opn))
        fprintf(ff, "%+" ISZ_PF "d)", NME_CNST(opn));
      else
        fprintf(ff, ")");
    else
      fprintf(ff, "+i)");
    break;
  case NT_ARR:
    DumpnameHelper(ff, NME_NM(opn));
    fprintf(ff, "[");
    if (NME_SYM(opn) == 0)
      fprintf(ff, "%" ISZ_PF "d]", NME_CNST(opn));
    else
      fprintf(ff, "i]");
    break;
  case NT_SAFE:
    fprintf(ff, "safe(");
    DumpnameHelper(ff, NME_NM(opn));
    fprintf(ff, ")");
    break;
  case NT_UNK:
    if (NME_SYM(opn))
      fprintf(ff, "?vol");
    else
      fprintf(ff, "?");
    break;
  }

  --level;

  if (level == 0)
    fprintf(ff, ">");
#endif
}

void
__dumpname(FILE *f, int opn)
{
  DumpnameHelper(f, opn);
}

void
dumpname(int opn)
{
  DumpnameHelper(gbl.dbgfil, opn);
}

#if DEBUG
void
DumpnmeHelper(FILE *f, int opn)
{
  FILE *ff;

  ff = f;
  if (f == NULL)
    ff = stderr;

  if (opn < 0 || opn >= nmeb.stg_size) {
    interr("DumpnmeHelper:bad names ptr", opn, ERR_Severe);
    fprintf(ff, " %5u <BAD>", opn);
    return;
  }

  __dmpnme(ff, opn, 0);
  switch (NME_TYPE(opn)) {
  case NT_INDARR:
  case NT_ADD:
  case NT_VAR:
    break;
  case NT_MEM:
    DumpnmeHelper(ff, NME_NM(opn));
    break;
  case NT_IND:
    DumpnmeHelper(ff, NME_NM(opn));
    break;
  case NT_ARR:
    DumpnmeHelper(ff, NME_NM(opn));
    break;
  case NT_SAFE:
    DumpnmeHelper(ff, NME_NM(opn));
    break;
  case NT_UNK:
    break;
  }
}

#define TOP 10
void
PrintTopNMEHash(void)
{
  int h, s, t, nmex;
  int topten[TOP], toptensize[TOP];
  if (nmeb.stg_base == NULL)
    return;
  for (s = 0; s < TOP; ++s) {
    topten[s] = 0;
    toptensize[s] = 0;
  }
  for (h = 0; h < NMEHSZ; ++h) {
    s = 0;
    for (nmex = nmehsh[h]; nmex > 0; nmex = NME_HSHLNK(nmex))
      ++s;
    if (s) {
      for (t = 0; t < TOP; ++t) {
        if (s == toptensize[t]) {
          ++topten[t];
          break;
        } else if (s > toptensize[t]) {
          /* move the others */
          int tt;
          for (tt = TOP - 1; tt > t; --tt) {
            toptensize[tt] = toptensize[tt - 1];
            topten[tt] = topten[tt - 1];
          }
          toptensize[t] = s;
          topten[t] = 1;
          break;
        }
      }
    }
  }
  fprintf(gbl.dbgfil, "Function %d = %s\nTop %d NME Hash Table Entries\n %d "
                      "NME entries, Hash Size %d, Average Length %d\n",
          gbl.func_count, GBL_CURRFUNC ? SYMNAME(GBL_CURRFUNC) : "", TOP,
          nmeb.stg_avail - 1, NMEHSZ, (nmeb.stg_avail - 1 + NMEHSZ) / NMEHSZ);
  for (s = 0; s < TOP; ++s) {
    fprintf(gbl.dbgfil, " [%2d] %d * %d\n", s + 1, toptensize[s], topten[s]);
  }
} /* PrintTopNMEHash */

void
PrintTopHash(void)
{
  int h, s, t, sptr;
  int topten[TOP], toptensize[TOP];
  for (s = 0; s < TOP; ++s) {
    topten[s] = 0;
    toptensize[s] = 0;
  }
  for (h = 0; h < HASHSIZE + 1; ++h) {
    s = 0;
    for (sptr = stb.hashtb[h]; sptr > NOSYM; sptr = HASHLKG(sptr))
      ++s;
    if (s) {
      for (t = 0; t < TOP; ++t) {
        if (s == toptensize[t]) {
          ++topten[t];
          break;
        } else if (s > toptensize[t]) {
          /* move the others */
          int tt;
          for (tt = TOP - 1; tt > t; --tt) {
            toptensize[tt] = toptensize[tt - 1];
            topten[tt] = topten[tt - 1];
          }
          toptensize[t] = s;
          topten[t] = 1;
          break;
        }
      }
    }
  }
  fprintf(gbl.dbgfil, "Function %d = %s\nTop %d Symbol Hash Table Entries\n %d "
                      "symbols, Hash Size %d, Average length %d:\n",
          gbl.func_count, GBL_CURRFUNC ? SYMNAME(GBL_CURRFUNC) : "", TOP,
          stb.stg_avail - 1, HASHSIZE + 1, (stb.stg_avail - 1 + HASHSIZE) / HASHSIZE);
  for (s = 0; s < TOP; ++s) {
    fprintf(gbl.dbgfil, " [%2d] %d * %d\n", s + 1, toptensize[s], topten[s]);
  }
  PrintTopNMEHash();
} /* PrintTopHash */
#endif
