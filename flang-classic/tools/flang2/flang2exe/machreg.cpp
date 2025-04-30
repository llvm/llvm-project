/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  machreg.c - Machine register definitions for the i386/387 */

#include "machreg.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "regutil.h"
#include "machreg.h"
#include "ili.h"

/* local functions for mr_getreg() & mr_getnext(); these routines called
 * twice to fetch two IR registers for a KR register.
 */
static int _mr_getreg(int rtype);
static int _mr_getnext(int rtype);

static int getnext_reg; /* current register for retry (mr_getnext) */

static bool mr_restore;          /* need to backout for KR registers? */
static char mr_restore_next_global; /* saving the mr.next_global field */
static char mr_restore_nused;       /* saving the mr.nused field */

static MACH_REG mach_reg[MR_UNIQ] = {
    {1, 8, 8 /*TBD*/, MR_L1, MR_U1, MR_U1, MR_U1, 0, 0, 'i'},       /*  %r's  */
    {1, 8, 8 /*TBD*/, MR_L2, MR_U2, MR_U2, MR_U2, 0, MR_MAX1, 'f'}, /*  %f's  */
    {1, 8, 8 /*TBD*/, MR_L3, MR_U3, MR_U3, MR_U3, 0, (MR_MAX1 + MR_MAX2),
     'x'} /*  %f's  xmm */
};

REG reg[RATA_RTYPES_TOTAL] = {
    {6, 0, 0, 0, &mach_reg[0], RCF_NONE}, /*  IR  */
    {3, 0, 0, 0, &mach_reg[1], RCF_NONE}, /*  SP  */
    {3, 0, 0, 0, &mach_reg[1], RCF_NONE}, /*  DP  */
    {6, 0, 0, 0, &mach_reg[0], RCF_NONE}, /*  AR  */
    {3, 0, 0, 0, &mach_reg[0], RCF_NONE}, /*  KR  */
    {0, 0, 0, 0, 0, 0},                   /*  VECT  */
    {0, 0, 0, 0, 0, 0},                   /*  QP    */
    {3, 0, 0, 0, 0, RCF_NONE},            /*  CSP   */
    {3, 0, 0, 0, 0, RCF_NONE},            /*  CDP   */
    {0, 0, 0, 0, 0, 0},                   /*  CQP   */
    {0, 0, 0, 0, 0, 0},                   /*  X87   */
    {0, 0, 0, 0, 0, 0},                   /*  CX87  */
    /* the following will be mapped over SP and DP above */
    {3, 0, 0, 0, &mach_reg[2], RCF_NONE}, /*  SPXM  */
    {3, 0, 0, 0, &mach_reg[2], RCF_NONE}, /*  DPXM  */
};

RGSETB rgsetb;

const int scratch_regs[3] = {IR_RAX, IR_RCX, IR_RDX};

#if defined(TARGET_LLVM_ARM) || defined(TARGET_LLVM_POWER)

/* arguments passed in registers */
int mr_arg_ir[MR_MAX_IREG_ARGS + 1];
/*  xmm0 --> xmm7 */
int mr_arg_xr[MR_MAX_XREG_ARGS + 1] = {XR_XMM0, XR_XMM1, XR_XMM2, XR_XMM3,
                                       XR_XMM4, XR_XMM5, XR_XMM6, XR_XMM7};

/* return result registers */
/* rax, rdx */
int mr_res_ir[MR_MAX_IREG_RES + 1] = {IR_RAX, IR_RDX};
/* xmm0, xmm1 */
int mr_res_xr[MR_MAX_XREG_RES + 1] = {XR_XMM0, XR_XMM1};

#elif defined(TARGET_WIN_X8664)

/* arguments passed in registers */
/*  rcx,rdx,r8,r9 */
int mr_arg_ir[MR_MAX_IREG_ARGS] = {IR_RCX, IR_RDX, IR_R8, IR_R9};
/*  xmm0 --> xmm3 */
int mr_arg_xr[MR_MAX_XREG_ARGS] = {XR_XMM0, XR_XMM1, XR_XMM2, XR_XMM3};

/* return result registers */
/* rax */
int mr_res_ir[MR_MAX_IREG_RES] = {IR_RAX};
/* xmm0 */
int mr_res_xr[MR_MAX_XREG_RES] = {XR_XMM0};

#else

/* arguments passed in registers */
/*  rdi,rsi,rdx,rcx,r8,r9 */
int mr_arg_ir[MR_MAX_IREG_ARGS] = {IR_RDI, IR_RSI, IR_RDX,
                                   IR_RCX, IR_R8,  IR_R9};
/*  xmm0 --> xmm7 */
int mr_arg_xr[MR_MAX_XREG_ARGS] = {XR_XMM0, XR_XMM1, XR_XMM2, XR_XMM3,
                                   XR_XMM4, XR_XMM5, XR_XMM6, XR_XMM7};

/* return result registers */
/* rax, rdx */
int mr_res_ir[MR_MAX_IREG_RES] = {IR_RAX, IR_RDX};
/* xmm0, xmm1 */
int mr_res_xr[MR_MAX_XREG_RES] = {XR_XMM0, XR_XMM1};

#endif

/** \brief Initialize mach_reg structs and reg array. This is done for each
 *  function (subprogram)
 */
void
mr_init()
{
  int i;

  aux.curr_entry->first_dr = reg[RATA_IR].mach_reg->first_global;
  aux.curr_entry->first_sp = reg[RATA_SP].mach_reg->first_global;
  aux.curr_entry->first_dp = reg[RATA_DP].mach_reg->first_global;
  aux.curr_entry->first_ar = reg[RATA_AR].mach_reg->first_global;

  for (i = 0; i < MR_UNIQ; i++) {
    mach_reg[i].next_global = mach_reg[i].first_global;
    mach_reg[i].nused = 0;
  }

  for (i = 0; i <= RATA_RTYPES_ACTIVE; i++) {
    reg[i].nused = 0;
    reg[i].rcand = 0;
  }

  /* for pic code, we need to reserve %ebx -- treat it like it
   * has already been assigned. Since it is register #1, this
   * is not too difficult.
   */
  if (XBIT(62, 8)) {
    mach_reg[0].next_global++;
    mach_reg[0].nused = 1;
    reg[RATA_IR].nused = 1;
    reg[RATA_AR].nused = 1;
  }

}

#ifdef FLANG_MACHREG_UNUSED
static int
mr_isxmm(int rtype)
{
#if DEBUG
  assert((rtype == RATA_SP || rtype == RATA_DP || rtype == RATA_CSP ||
          rtype == RATA_CDP),
         "mr_isxmm bad rtype", rtype, ERR_Severe);
#endif
  return (reg[rtype].mach_reg->Class == 'x');
}
#endif

void
mr_reset_numglobals(int reduce_by)
{
  mach_reg[0].last_global = mach_reg[0].end_global - reduce_by;
}

void
mr_reset_frglobals()
{
  /* effectively turn off fp global regs. */
  mach_reg[1].last_global = mach_reg[1].first_global - 1;
  mach_reg[2].last_global = mach_reg[2].first_global - 1;
}

/** \brief get a global register for a given register type (RATA_IR, etc.).
 *  NOTE that the global registers are allocated in increasing order.
 *  next_global locates the next available global register.  The range
 *  of global register values is from first_global to last_global,
 *  inclusive.
 */
int
mr_getreg(int rtype)
{
  int rg;

  rg = _mr_getreg(rtype);

  return rg;
}

static int
_mr_getreg(int rtype)
{
  register MACH_REG *mr;

  if (reg[rtype].nused >= reg[rtype].max)
    return NO_REG;

  mr = reg[rtype].mach_reg;
  if (mr->next_global > mr->last_global)
    return NO_REG;

  if (BIH_SMOVE(gbl.entbih) && mr->next_global > 1)
    return NO_REG;

  /* currently, only allow more than one floating point
   * global register if an xbit is set.
   */
  if ((rtype == RATA_SP || rtype == RATA_DP || rtype == RATA_CSP ||
       rtype == RATA_CDP) &&
      (!XBIT(4, 0x4) || ratb.mexits))
    if (mr->next_global > mr->first_global)
      return NO_REG;

  /* floating point globals need to always start from fp2 (fp1 is
   * by convention where the return value of fp functions is placed)
   * and then increment for each inner loop being processed. Thus,
   * the nused field records the largest number of fp registers
   * assigned to any loop. This is done differently for the I386
   * fp as opposed to the I386 integers or any other register set
   * (due primarily to the fact that the fp registers on x86 are
   * actually a stack).
   */
  if ((rtype != RATA_SP && rtype != RATA_DP && rtype != RATA_CSP &&
       rtype != RATA_CDP) ||
      (mr->next_global - mr->first_global + 1 > mr->nused)) {
    reg[rtype].nused++;
    mr->nused++;
  }
  return (mr->next_global++);
}

/** \brief map a register type and global register number to an index value in
 * the range 0 .. MR_NUMGLB-1, taking into consideration that certain
 * register types map to the same machine register set.
 * 
 * This is used by * the optimizer to index into its register history table.
 */
int
mr_gindex(int rtype, int regno)
{
  MACH_REG *mr = reg[rtype].mach_reg;
  return ((regno - mr->first_global) + mr->mapbase);
}

/** \brief communicate to the scheduler the first global register not assigned
 * for each register class 
 *
 * Note that this will be the physical register
 * number; it reflects the number of registers assigned from the physical
 * set mapped from the generic register set. Because two or more generic
 * register sets can map to a single register set, this information
 * can only be computed after all of the assignments are done.
 *
 */
void
mr_end()
{
  aux.curr_entry->first_dr += reg[RATA_IR].mach_reg->nused;
  aux.curr_entry->first_ar += reg[RATA_AR].mach_reg->nused;
  aux.curr_entry->first_sp += reg[RATA_SP].mach_reg->nused;
  aux.curr_entry->first_dp += reg[RATA_DP].mach_reg->nused;

}

#ifdef FLANG_MACHREG_UNUSED
void
static mr_reset_fpregs()
{
  mach_reg[1].next_global = mach_reg[1].first_global;
  mach_reg[2].next_global = mach_reg[2].first_global;
}
#endif

/** \brief Initialize for scanning the entire machine register set used for
 *  rtype.
 *
 *  This mechanism for retrieving registers is done when we can no longer
 *  retrieve registers from mr_getreg (we're out of rtype registers).
 *  Ensuing calls to mr_getnext will attempt to retrieve a register
 *  from the set.  The assumption is that the caller (optimizer)
 *  will first call mr_reset, and then call mr_getnext one or more
 *  times.
 */
void
mr_reset(int rtype)
{
  getnext_reg = reg[rtype].mach_reg->first_global;

  /* if we are generating pic code, we must exclude %ebx as
   * a potential register.
   */
  if ((rtype == RATA_IR || rtype == RATA_AR || rtype == RATA_KR) && XBIT(62, 8))
    getnext_reg++;

}

/** \func Attempt to retrieve the next available register from the set used
 * for rtype.
 *
 * If one is found, it may be necessary to update the
 * mach_reg info since we're scanning the entire set. mr_getreg uses a
 * portion of the set (as defined by the reg structure); things could
 * get out of sync when registers of different rtypes share the same
 * register set.
 */

int 
mr_getnext(int rtype)
{
  int rg;

  rg = _mr_getnext(rtype);
  return rg;
}

static
int _mr_getnext(int rtype)
{
  int mreg;
  MACH_REG *mr;

  mr = reg[rtype].mach_reg;
  if (getnext_reg > mr->last_global)
    return NO_REG;
  if (BIH_SMOVE(gbl.entbih) && mr->next_global > 1)
    return NO_REG;

  if ((rtype == RATA_SP || rtype == RATA_DP || rtype == RATA_CSP ||
       rtype == RATA_CDP) &&
      (!XBIT(4, 0x4) || ratb.mexits))
    if (getnext_reg > mr->first_global)
      return NO_REG;

  mreg = getnext_reg;
  getnext_reg++;
  if (mreg >= mr->next_global) {
    mr_restore = true;
    mr_restore_nused = mr->nused;
    mr_restore_next_global = mr->next_global;
    /* same comment as in _mr_getreg */
    if ((rtype != RATA_SP && rtype != RATA_DP && rtype != RATA_CSP &&
         rtype != RATA_CDP) ||
        ((mr->next_global - mr->first_global + 1) > mr->nused))
      mr->nused++;
    mr->next_global = getnext_reg;
  }
  return mreg;
}

#ifdef FLANG_MACHREG_UNUSED
/*  RGSET functions   */
static void
mr_init_rgset()
{
  RGSET tmp;
  int bihx;

/* just verify that regs all fit in RGSET fields.  (+1 below is because
 * current RGSET macro's assume regs start at 1, position 0 in bitfields
 * is  wasted.  TST_ and SET_ macros could be changed along with these
 * asserts to save the bit.
 */
  assert(sizeof(tmp.xr) * 8 >= mach_reg[2].max + 1, "RGSET xr ops invalid", 0,
         ERR_Severe);

  rgsetb.stg_avail = 1;

  /* make sure BIH_RGSET fields are fresh. */
  bihx = gbl.entbih;
  for (;;) {
    BIH_RGSET(bihx) = 0;
    if (BIH_LAST(bihx))
      break;
    bihx = BIH_NEXT(bihx);
  }
}
#endif

/** \brief allocate and initialize a RGSET entry.  */
int
mr_get_rgset()
{
  int rgset;

  rgset = rgsetb.stg_avail++;
  if (rgsetb.stg_avail > MAXRAT)
    error((error_code_t)7, ERR_Fatal, 0, CNULL, CNULL);
  NEED(rgsetb.stg_avail, rgsetb.stg_base, RGSET, rgsetb.stg_size,
       rgsetb.stg_size + 100);
  if (rgsetb.stg_base == NULL)
    error((error_code_t)7, ERR_Fatal, 0, CNULL, CNULL);

  RGSET_XR(rgset) = 0;

  return rgset;
}

#ifdef FLANG_MACHREG_UNUSED
static void
mr_dmp_rgset(int rgseti)
{
  int i;
  int cnt = 0;

  fprintf(gbl.dbgfil, "rgset %d:", rgseti);
  if (rgseti == 0) {
    fprintf(gbl.dbgfil, " null");
    assert(RGSET_XR(0) == 0, "mr_dmp_rgset says someone was writing 0", 0, ERR_Severe);
  }
  for (i = XR_FIRST; i <= XR_LAST; i++) {
    if (TST_RGSET_XR(rgseti, i)) {
      fprintf(gbl.dbgfil, " xmm%d", i);
      cnt++;
    }
  }
  fprintf(gbl.dbgfil, " total %d\n", cnt);
}

/* called from flow.c to tell globalreg, and scheduler which
   xmm regs are used by the vectorizer.
 */
static void
mr_bset_xmm_rgset(int ili, int bih)
{
  int j, opn;
  ILI_OP opc;
  int noprs;

  if (BIH_RGSET(bih) == 0) {
    BIH_RGSET(bih) = mr_get_rgset();
  }

  opc = ILI_OPC(ili);
  noprs = ilis[opc].oprs;
  for (j = 1; j <= noprs; j++) {
    opn = ILI_OPND(ili, j);
    switch (IL_OPRFLAG(opc, j)) {
    case ILIO_XMM:
      assert(opn >= XR_FIRST && opn <= XR_LAST,
             "mr_bset_xmm_rgset: bad xmm register value", ili, ERR_Warning);
      SET_RGSET_XR(BIH_RGSET(bih), opn);
      break;
    default:
      break;
    }
  }
}
#endif
