/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief PGFTN/N10 "assembler" module
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "machar.h"
#include "version.h"
#include "fih.h"

static ISZ_T bss_addr; /* local STATIC area (uninit'd) */

#define DT_QUADALIGN 15

void
assemble(void)
{
}

void
assemble_init(void)
{
}

void
assemble_end(void)
{
}

void
sym_is_refd(int sptr)
{
  int a, dtype, stype;
  ISZ_T size;
  ISZ_T addr;
  dtype = DTYPEG(sptr);
  switch ((stype = STYPEG(sptr))) {
  case ST_PLIST:
  case ST_VAR:
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_STRUCT:
  case ST_UNION:
    if (REFG(sptr))
      break;
    if (gbl.internal > 1 && !INTERNALG(sptr))
      break;
    switch (SCG(sptr)) {
    case SC_LOCAL:
/*
 * assign address to automatic variable: auto offsets are
 * negative relative to the frame pointer. the current size of
 * of the stack frame is saved as a positive value; the last
 * offset assigned is the negative of the current frame size.
 * The negative of the current frame size is aligned so that the
 * variable ends on this boundary.  The offset assigned is this
 * value minus its size in bytes. The new size of the stack frame
 * is the negative of the offset.
 * ASSUMPTIONS:
 *     1.  the value frame pointer is an address whose alignment
 *         matches that of the scalar item having the most strict
 *         requirement.
 *     2.  there are not gaps between the address located by the
 *         frame pointer and the auto area (first offset is -1)
 */
#ifdef RECURBSS
      if (!flg.recursive || DINITG(sptr) || SAVEG(sptr))
#else
      if (DINITG(sptr) || SAVEG(sptr))
#endif
      {
        SCP(sptr, SC_STATIC);
        goto static_shared;
      }
      if (stype == ST_PLIST)
        size = PLLENG(sptr) * size_of(dtype);
      else
        size = size_of(dtype);
      if (stype == ST_DESCRIPTOR && size > 4) {
        a = alignment(DT_PTR);
        size = ALIGN(size, a);
      } else if ((flg.quad && size >= 16)) {
        a = DT_QUADALIGN;
        /* round-up size since sym's offset is 'aligned next' - size */
        size = ALIGN(size, DT_QUADALIGN);
      } else
        a = alignment(dtype);
#if DEBUG
      assert(size == ALIGN(size, a), "sym_is_refd: sym unaligned", sptr, 2);
#endif
      addr = -gbl.locaddr;
      addr = ALIGN_AUTO(addr, a) - size;
      ADDRESSP(sptr, addr);
      gbl.locaddr = -addr;
      SYMLKP(sptr, gbl.locals);
      gbl.locals = sptr;
#if DEBUG
      if (DBGBIT(5, 32)) {
        fprintf(gbl.dbgfil,
                "addr: %6" ISZ_PF "d size: %6" ISZ_PF "d  %-32s   (%s)\n", addr,
                size, getprint(sptr), getprint((int)gbl.currsub));
      }
#endif
      break;
    case SC_STATIC:
    static_shared:
      if (DINITG(sptr))
        addr = gbl.saddr;
      else
        addr = bss_addr;
      if (stype == ST_PLIST)
        size = PLLENG(sptr) * size_of(dtype);
      else
        size = size_of(dtype);
      if (stype == ST_DESCRIPTOR) {
        a = alignment(DT_PTR);
        size = ALIGN(size, a);
      } else if ((flg.quad && size >= 16)) {
        a = DT_QUADALIGN;
      } else
        a = alignment(dtype);
      addr = ALIGN(addr, a);
      ADDRESSP(sptr, addr);
      if (DINITG(sptr)) {
        gbl.saddr = addr + size;
        SYMLKP(sptr, gbl.statics);
        gbl.statics = sptr;
#if DEBUG
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "saddr: %6" ISZ_PF "d size: %6" ISZ_PF "d  %-32s   (%s)\n",
                  addr, size, getprint(sptr), getprint((int)gbl.currsub));
        }
#endif
      } else {
        bss_addr = addr + size;
        if (!bss_addr && no_data_components(dtype)) {
          /* FS#17746 - need to assign a default size to bss_addr
           * for zero size derived types to make sure we declare
           * the BSS symbol in the assembly file.
           */
          bss_addr = size_of(DT_INT4);
        }
#if DEBUG
        if (DBGBIT(5, 32)) {
          fprintf(gbl.dbgfil,
                  "baddr: %6" ISZ_PF "d size: %6" ISZ_PF "d  %-32s   (%s)\n",
                  addr, size, getprint(sptr), getprint((int)gbl.currsub));
        }
#endif
      }
      break;
    case SC_CMBLK:
    case SC_DUMMY:
      break;
    case SC_EXTERN:
      break;
    case SC_PRIVATE:
      /* don't assign address to privates & don't set the REF bit */
      return;
    case SC_NONE:
    default:
      interr("sym_is_refd: bad sc\n", SCG(sptr), 3);
    }
    REFP(sptr, 1);
    break;

  case ST_PROC:
    if (REFG(sptr) == 0 && SCG(sptr) == SC_EXTERN) {
      SYMLKP(sptr, gbl.externs);
      gbl.externs = sptr;
      REFP(sptr, 1);
    }
    break;
  case ST_CONST:
    if (SYMLKG(sptr) == 0) {
      SYMLKP(sptr, gbl.consts);
      gbl.consts = sptr;
      if (DTYPEG(sptr) == DT_ADDR && CONVAL1G(sptr))
        sym_is_refd((int)CONVAL1G(sptr));
    }
    break;

  case ST_ENTRY: /* (found on entry ili only) */
  case ST_LABEL:
    break;

  default:
    interr("sym_is_refd:bad sty", sptr, 2);
  }

}

/** \brief The equivalence processor assigns positive offsets to the local
   variables
    which appear in equivalence statements.  Target addresses must be
    assigned using the offsets provided by the equivalence processor.
    \param loc_list  list of local symbols linked by SYMLK
    \param loc_addr  total size of the equivalenced locals
 */
void
fix_equiv_locals(int loc_list, ISZ_T loc_addr)
{
  int sym;
  int maxa;

  if (loc_list != NOSYM) {
    /*
     * align beginning of this group just in case.  Implementation note:
     * could loop through the syms in the group to determine the maximum
     * alignment required; this would involve looking at the size if -quad
     * or just the alignment of the symbol.
     */
    if (flg.quad)
      maxa = DT_QUADALIGN;
    else
      maxa = alignment(DT_DBLE);
    gbl.locaddr = ALIGN(gbl.locaddr + loc_addr, maxa);
    do {
      /* NOTE:  REF flag of sym set during equivalence processing */
      sym = loc_list;
      loc_list = SYMLKG(loc_list);
      ADDRESSP(sym, -gbl.locaddr + ADDRESSG(sym));
      SCP(sym, SC_LOCAL);
      SYMLKP(sym, gbl.locals);
      gbl.locals = sym;
    } while (loc_list != NOSYM);
  }

}

/** \brief Similiar to fix_equiv_locals except that these local variables were
    saved and/or dinit'd.  for these variables, switch the storage class to
    SC_STATIC.
    \param loc_list  list of local symbols linked by SYMLK
    \param loc_addr  total size of the equivalenced locals
    \param dinitflg  variables were dinit'd

    The equivalence processor assigns positive offsets to the local variables
    which appear in equivalence statements.  Target addresses must be
    assigned using the offsets provided by the equivalence processor.
 */
void
fix_equiv_statics(int loc_list, ISZ_T loc_addr, LOGICAL dinitflg)
{
  int sym;
  int maxa;
  ISZ_T addr;

#if DEBUG
  assert(loc_list != NOSYM, "fix_equiv_statics: bad loc_list", 0, 3);
#endif
  /*
   * align beginning of this group just in case.  Implementation note:
   * could loop through the syms in the group to determine the maximum
   * alignment required; this would involve looking at the size if -quad
   * or just the alignment of the symbol.
   */
  if (flg.quad)
    maxa = DT_QUADALIGN;
  else
    maxa = alignment(DT_DBLE);
  if (dinitflg) {
    addr = gbl.saddr;
    addr = ALIGN(addr, maxa);
    do {
      /* NOTE:  REF flag of sym set during equivalence processing */
      sym = loc_list;
      loc_list = SYMLKG(loc_list);
      ADDRESSP(sym, addr + ADDRESSG(sym));
      SCP(sym, SC_STATIC);
      SYMLKP(sym, gbl.statics);
      gbl.statics = sym;
      DINITP(sym, 1); /* ensure getsname thinks it's in STATIC */
    } while (loc_list != NOSYM);
    gbl.saddr = addr += loc_addr;
  } else {
    addr = bss_addr;
    addr = ALIGN(addr, maxa);
    do {
      /* NOTE:  REF flag of sym set during equivalence processing */
      sym = loc_list;
      loc_list = SYMLKG(loc_list);
      ADDRESSP(sym, addr + ADDRESSG(sym));
      SCP(sym, SC_STATIC);
    } while (loc_list != NOSYM);
    bss_addr = addr += loc_addr;
  }

}

ISZ_T
get_bss_addr(void)
{
  return bss_addr;
} /* get_bss_addr */

ISZ_T
set_bss_addr(ISZ_T addr)
{
  bss_addr = addr;
  return bss_addr;
} /* get_bss_addr */

#define ALN_MINSZ 128000
#define ALN_UNIT 64
#define ALN_MAXADJ 4096
#define ALN_THRESH (ALN_MAXADJ / ALN_UNIT)

ISZ_T
pad_cmn_mem(int mem, ISZ_T msz, int *p_aln_n)
{
  int aln_n;

  aln_n = *p_aln_n;
#ifdef PDALN_IS_DEFAULT
  if (!XBIT(57, 0x1000000) && !PDALN_IS_DEFAULT(mem) && !PDALN_IS_DEFAULT(CMBLKG(mem)) &&
      msz > ALN_MINSZ) {
    if (aln_n == 0)
      aln_n = 1;
    msz += ALN_UNIT * aln_n;
    if (aln_n <= ALN_THRESH)
      aln_n++;
    else
      aln_n = 1;
    *p_aln_n = aln_n;
  }
#endif
  return msz;
}
