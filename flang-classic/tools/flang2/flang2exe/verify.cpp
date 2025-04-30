/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "verify.h"
#include "ili.h"
#include "iliutil.h"

#if DEBUG

#define VERIFY(cond, message) ((cond)?(void)0 : verify_failure(__LINE__, #cond, message))

/* Out-of-line helper for macro VERIFY. */
static void
verify_failure(int line, const char *cond, const char *message) {
  if (XBIT(160, 2)) {
    fprintf(stderr, "%s:%d: VERIFY(%s): %s\n", __FILE__, line, cond, message);
    interr(message, 0, error_max_severity());
  }
}

typedef unsigned char epoch_t;

/* A visit_info holds information about an ILI node */
typedef struct {
  /* This record is considered absent unless epoch==current_epoch. */
  epoch_t epoch;
} visit_info;

static visit_info *visited_ili;
static int visited_size; /* Number of elements pointed to by visited_ili. */
static epoch_t current_epoch;
static int walk_depth;

/* If doing deep ILI verification, prepare for remembering which ILI have
   already been walked.  Must be paired with end_walk(level). 
   No-op if level does not require any remembering. */
static void
begin_walk(VERIFY_LEVEL level)
{
  if (level >= VERIFY_ILI_DEEP) {
    if (++walk_depth == 1) {
      ++current_epoch;
      if (current_epoch == 0) {
        int i;
        /* epoch wrapped - wipe the slate clean */
        for (i = 0; i < visited_size; ++i)
          visited_ili[i].epoch = 0;
        current_epoch = 1;
      }
    }
  }
}

/* If at outermost walk, forget which ILI have already been walked.
   The level *must* match the level in the corresponding call to begin_walk().
   */
static void
end_walk(VERIFY_LEVEL level)
{
  if (level >= VERIFY_ILI_DEEP) {
    VERIFY(walk_depth >= 1, "walk not started?");
    --walk_depth;
  }
}

static void
ili_mark_first_visit(int ilix)
{
  int old_size;
  visit_info *v;
  DEBUG_ASSERT(walk_depth > 0, "internal error in verifier: not in a walk");
  DEBUG_ASSERT(0 < ilix, "internal error in verifier itself");
  old_size = visited_size;
  if (ilix >= old_size) {
    int i;
    int new_size = 1;
    while (ilix >= new_size)
      new_size *= 2;
    NEED(ilix + 1, visited_ili, visit_info, visited_size, new_size);
    for (i = old_size; i < visited_size; ++i)
      visited_ili[i].epoch = 0;
  }
  DEBUG_ASSERT(ilix < visited_size, "internal error in verifier itself");
  v = &visited_ili[ilix];
  VERIFY(v->epoch != current_epoch, "ILI already visited!");
  v->epoch = current_epoch;
}

/** NULL if ilix has not been visited yet. */
static visit_info *
ili_visit_info(int ilix)
{
  DEBUG_ASSERT(walk_depth > 0, "internal error in verifier: not in a walk");
  if (0 < ilix && ilix < visited_size &&
      visited_ili[ilix].epoch == current_epoch)
    return &visited_ili[ilix];
  else
    return NULL;
}

/** Return true if jth operand of opc is allowed to have operation j_opc
   contrary to expectations of IL_OPRFLAG(opc, j).  This routine
   is called infrequently so consider readability over speed when
   adding another case. */
static bool
is_known_bug(ILI_OP opc, int j, ILI_OP j_opc)
{
  /* r is the kind of operand supplied. */
  ILIA_RESULT r = IL_RES(j_opc);
  /* o is the kind of operand expected. */
  ILIO_KIND o = IL_OPRFLAG(opc, j);
  if (opc == IL_KNEG && o == ILIO_KRLNK && r == ILIA_AR)
    return true;
  if (opc == IL_KADD && o == ILIO_KRLNK && r == ILIA_IR)
    return true;
#ifdef TARGET_X86
  if ((opc == IL_DASPSP || opc == IL_MVSPSP) && o == ILIO_DPLNK && r == ILIA_CS)
    return true;
#endif
  if (opc == IL_STKR && o == ILIO_KRLNK && (r == ILIA_IR || r == ILIA_AR) &&
      j == 1)
    return true;
  if ((opc == IL_FREEIR || opc == IL_CSEIR) && o == ILIO_IRLNK && r == ILIA_KR)
    return true;
  if (opc == IL_UKADD && o == ILIO_KRLNK && r == ILIA_IR)
    return true;
  if (opc == IL_ST && o == ILIO_IRLNK && r == ILIA_KR && j == 1)
    return true;
  if (opc == IL_IMUL && o == ILIO_IRLNK && r == ILIA_KR && j == 2)
    return true;
  if (opc == IL_IDIV && o == ILIO_IRLNK && r == ILIA_KR && j == 2)
    return true;
  if ((opc == IL_UKCMP || opc == IL_UKCJMP) && o == ILIO_KRLNK && r == ILIA_AR)
    return true;
  if (opc == IL_STKR && o == ILIO_KRLNK && r == ILIA_IR && j == 1)
    return true;
  if (opc == IL_AADD && o == ILIO_ARLNK && r == ILIA_IR && j == 1)
    return true;
  if (opc == IL_ST && o == ILIO_IRLNK && r == ILIA_AR && j == 1)
    return true;
  if (opc == IL_UKNEG && o == ILIO_KRLNK && r == ILIA_IR && j == 1)
    return true; 
  if ((opc == IL_KMUL || opc == IL_IADD || opc == IL_IKMV) && j_opc == IL_ACCLDSYM)
    return true;
  if ((opc == IL_ACMPZ || opc == IL_ACJMPZ || opc == IL_LDA) && o == ILIO_ARLNK && r == ILIA_KR && j == 1)
    return true;
  if (opc == IL_IMUL && o == ILIO_IRLNK && r == ILIA_KR && j == 1)
    return true;
  if (opc == IL_DAIR && o == ILIO_IRLNK && j_opc == IL_KCON)
    return true;
  if ((opc == IL_IADD || opc == IL_IAMV) && o == ILIO_IRLNK && j_opc==IL_DFRKR)
    return true;
  if (opc == IL_KXOR && o == ILIO_KRLNK && j_opc == IL_ICMPZ)
    return true;
  if ((opc == IL_KCMP || opc == IL_KCJMP) && o == ILIO_KRLNK && j_opc == IL_LDDP)
    return true;
  if (opc == IL_PI8MV_LOW && o == ILIO_KRLNK && r == ILIA_AR)
    return true;
#ifdef IL_PI8BROADCAST
  if (opc == IL_PI8BROADCAST && o == ILIO_KRLNK && r == ILIA_AR)
    return true;
#endif
  if (opc == IL_IMUL && o == ILIO_IRLNK && j_opc == IL_KCON)
    return true;
  if (opc == IL_IKMV && o == ILIO_IRLNK && j_opc == IL_KCON)
    return true;
  if (opc == IL_IADD && o == ILIO_IRLNK && j_opc == IL_LDKR)
    return true;
  if (opc == IL_KIMV && o == ILIO_KRLNK && r == ILIA_IR)
    return true;
  return false;
}

/** Check that operation opc can have jth operand that has operation operand_opc. */
static void
verify_compatible(ILI_OP opc, int j, ILI_OP j_opc)
{
  ILIA_RESULT r = IL_RES(j_opc);
  ILIO_KIND o = IL_OPRFLAG(opc, j);
  ILIA_RESULT expected = (ILIA_RESULT)(-1);
  if (j_opc == IL_ACCLDSYM)
    return;  /* satisfies any kind of link. used in device code only */
  switch (o) {
  case ILIO_LNK:
    /* Any kind of link allowed. */
    return;
  case ILIO_IRLNK:
    expected = ILIA_IR;
    break;
  case ILIO_SPLNK:
    expected = ILIA_SP;
    break;
  case ILIO_DPLNK:
    expected = ILIA_DP;
    break;
  case ILIO_KRLNK:
    expected = ILIA_KR;
    break;
  case ILIO_ARLNK:
    expected = ILIA_AR;
    break;
  case ILIO_QPLNK:
    expected = ILIA_QP;
    break;
  case ILIO_CSLNK:
    expected = ILIA_CS;
    break;
  case ILIO_CDLNK:
    expected = ILIA_CD;
    break;
  case ILIO_CQLNK:
    expected = ILIA_CQ;
    break;
  case ILIO_128LNK:
    expected = ILIA_128;
    break;
  case ILIO_256LNK:
    expected = ILIA_256;
    break;
  case ILIO_512LNK:
    expected = ILIA_512;
    break;
  case ILIO_FLOAT128LNK:
    expected = ILIA_FLOAT128;
    break;
  case ILIO_X87LNK:
    expected = ILIA_X87;
    break;
  case ILIO_DOUBLEDOUBLELNK:
    expected = ILIA_DOUBLEDOUBLE;
    break;
  default:
    VERIFY(false, "unexpected ILIO_KIND");
    break;
  }
  VERIFY(r == expected || is_known_bug(opc, j, j_opc),
         "IL_RES of ILI is incompatible with context");
}

/** \brief Ad-hoc operand checks.

    This routine does ad-hoc checking of the operands that needs
    to be specific to an operation.  This routine is called from
    verify_ili_aux after generic checks have been done. */
static void
verify_ili_ad_hoc(int ilix)
{
  ILI_OP opc = ILI_OPC(ilix);
  switch (opc) {
  default:
    break;
  case IL_STSP:
    VERIFY(ILI_OPND(ilix, 4) == MSZ_F4, "4th operand to STSP must be MSZ_F4");
    break;
  case IL_STDP:
    VERIFY(ILI_OPND(ilix, 4) == MSZ_F8, "4th operand to STDP must be MSZ_F8");
    break;
#ifdef IL_STSPSP
  case IL_STSPSP: {
    ILI_OP opc1 = ILI_OPC(ILI_OPND(ilix, 1));
    VERIFY(opc1 == IL_DFRDP || opc1 == IL_DPDF, "1st operand to STSPSP must be DFRDP or DPDF");
    VERIFY(ILI_OPND(ilix, 4) == MSZ_F8, "4th operand to STSPSP must be MSZ_F8");
    break;
  }
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ST:
    VERIFY(ILI_OPND(ilix, 4) == MSZ_F16, "4th operand to FLOAT128ST must be MSZ_16");
    break;
  case IL_FLOAT128LD:
    VERIFY(ILI_OPND(ilix, 3) == MSZ_F16, "3rd operand to FLOAT128LD must be MSZ_F16");
    break;
#endif /* LONG_DOUBLE_FLAOT128 */
  }
}

/** Recursive helper routine for verifying ILI.  Because it is recursive,
    please try to keep it relatively uncluttered so that the control flow
    remains clear. */
static void
verify_ili_aux(int ilix, ILIO_KIND context, VERIFY_LEVEL level)
{
  ILI_OP opc;
  int noprs, j;

  VERIFY(1 <= ilix, "unexpected zero or negative ili index");
  VERIFY(ilix < ilib.stg_avail, "out of bounds ili index");
  if (level < VERIFY_ILI_SHALLOW)
    return;

  if (level >= VERIFY_ILI_DEEP && ili_visit_info(ilix))
    /* Already checked this node. */
    return;

  /* General operand checks */
  opc = ILI_OPC(ilix);
  noprs = ilis[opc].oprs;
  for (j = 1; j <= noprs; ++j) {
    if (ILIO_ISLINK(IL_OPRFLAG(opc, j))) {
      int operand = ILI_OPND(ilix, j);
      verify_compatible(opc, j, ILI_OPC(operand));
      if (level >= VERIFY_ILI_DEEP)
        verify_ili_aux(operand, IL_OPRFLAG(opc, j), level);
    }
  }

  verify_ili_ad_hoc(ilix);

  if (level >= VERIFY_ILI_DEEP)
    ili_mark_first_visit(ilix);
}

void
verify_ili(int ilix, VERIFY_LEVEL level)
{
  begin_walk(level);
  verify_ili_aux(ilix, ILIO_LNK, level);
  end_walk(level);
}

#ifdef FLANG2_VERIFY_UNUSED
/** Check if iltx is store of first result of a pair of results returned by a
   JSR.  This is a helper for verify_ilt, and assumes that iltx is index of
   ILT of type ILTY_STORE and ili_throw_label(iltx)!=0. */
static bool
is_first_store_of_pair(int iltx1)
{
  int iltx2 = ILT_NEXT(iltx1);
  if (iltx2) {
    int ilix1 = ILT_ILIP(iltx1);
    int ilix2 = ILT_ILIP(iltx2);
    if (IL_TYPE(ILI_OPC(ilix2)) == ILTY_STORE) {
      if (ili_throw_label(ilix2)) {
        /* At this point, we know that ilix and ilix2 are both stores of results
           from JSR/JSRA operations that can throw. Extract the JSR/JSRA
           "pointers" and check that they are equal. */
        int throw_jsr1 = ILI_OPND(ILI_OPND(ilix1, 1), 1);
        int throw_jsr2 = ILI_OPND(ILI_OPND(ilix2, 1), 1);
        /* there are cases where the following is not necessarily true; change to
           only check the equality of throw labels for now */
        /* return throw_jsr1 == throw_jsr2; */
        return ili_throw_label(ilix1) == ili_throw_label(ilix2);
      }
    }
  }
  return false;
}
#endif

void
verify_ilt(int iltx, VERIFY_LEVEL level)
{
  int ilix, throw_label;
  VERIFY(0 < iltx && iltx < iltb.stg_size, "invalid ILT index");

  if (level < VERIFY_ILT)
    return;

  begin_walk(level);

  ilix = ILT_ILIP(iltx);
  verify_ili_aux(ilix, ILIO_LNK, level);

  /* Check that ILT_CAN_THROW is set correctly. */
  switch (IL_TYPE(ILI_OPC(ilix))) {
  case ILTY_STORE:
    throw_label = ili_throw_label(ilix);
    if (!throw_label) {
      VERIFY(!ILT_CAN_THROW(iltx), "ILT_CAN_THROW should be false for "
                                   "ILTY_STORE that does not store result of "
                                   "JSR that can throw");
    } /* is_first_store_of_pair() does not always reliably return the correct
         answer.  We need to sort that out before invoking the following
      else if (is_first_store_of_pair(iltx)) {
      VERIFY(!ILT_CAN_THROW(iltx), "ILT_CAN_THROW should be false for "
                                   "ILTY_STORE that stores first of a pair of "
                                   "results from a JSR that can throw");

    } else {
      VERIFY(ILT_CAN_THROW(iltx), "ILT_CAN_THROW should be true for ILTY_STORE "
                                  "that stores sole or second result of a JSR "
                                  "that can throw");
    } */
    break;
  case ILTY_PROC:
    throw_label = ili_throw_label(ilix);
    if (throw_label)
      VERIFY(ILT_CAN_THROW(iltx),
             "ILT_CAN_THROW should be true for ILTY_PROC that can throw");
    else
      VERIFY(!ILT_CAN_THROW(iltx),
             "ILT_CAN_THROW should be false for ILTY_PROC that cannot throw");
    break;
  default:
    VERIFY(!ILT_CAN_THROW(iltx),
           "ILT_CAN_THROW can be true only for store or call");
    break;
  }

  /* With non-extended basic blocks, an ILT that can throw terminates the
     block. */
  if (flg.opt >= 2 && ILT_NEXT(iltx)) {
    VERIFY(!ILT_CAN_THROW(iltx), "ILT_CAN_THROW should be false at -O2 "
                                 "if ILT is not last in block");
  }

  end_walk(level);
}

void
verify_block(int bihx, VERIFY_LEVEL level)
{
  int iltx;

  VERIFY(0 < bihx && bihx < bihb.stg_size, "invalid BIH index");
  begin_walk(level);
  for (iltx = BIH_ILTFIRST(bihx); iltx; iltx = ILT_NEXT(iltx)) {
    verify_ilt(iltx, level);
  }
  end_walk(level);
}

void
verify_function_ili(VERIFY_LEVEL level)
{
  int bih;
  begin_walk(level);
  for (bih = gbl.entbih; bih; bih = BIH_NEXT(bih))
    verify_block(bih, level);
  end_walk(level);
}

#endif /* DEBUG */
