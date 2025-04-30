/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief optimization/peephole/inst simplification routines for LLVM Code
   Generator
 */

#include "llopt.h"
#include "dtypeutl.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "llutil.h"
#include "cgllvm.h"
#include "ili.h"
#include "mwd.h"
#include <stdlib.h>

#define DEC_UCOUNT(i) ((i)->tmps->use_count--)

#ifdef TARGET_LLVM_ARM
static void
replace_by_call_to_llvm_instrinsic(INSTR_LIST *instr, const char *fname,
                                   OPERAND *params)
{
  OPERAND *call_op;
  char *intrinsic_name;
  static char buf[MAXIDLEN];
  LL_Type *return_ll_type = NULL;

  DBGXTRACEIN1(DBGBIT(12, 0x20), 1, "ilix %d", instr->ilix);

  intrinsic_name = (char *)getitem(LLVM_LONGTERM_AREA, strlen(fname) + 1);
  strcpy(intrinsic_name, fname);
  return_ll_type = instr->ll_type;
  instr->i_name = I_PICALL;
  instr->flags = CALL_INTRINSIC_FLAG;
  call_op = make_operand();
  call_op->ot_type = OT_CALL;
  call_op->string = intrinsic_name;
  call_op->ll_type = return_ll_type;
  instr->operands = call_op;
  call_op->next = params;
  /* add global define of llvm.xxx to external function list, if needed */
  sprintf(buf, "declare %s %s(", return_ll_type->str, intrinsic_name);
  if (params) {
    sprintf(buf, "%s%s", buf, params->ll_type->str);
    params = params->next;
  }
  while (params) {
    sprintf(buf, "%s, %s", buf, params->ll_type->str);
    params = params->next;
  }
  strcat(buf, ")");
  update_external_function_declarations(intrinsic_name, buf, EXF_INTRINSIC);

  DBGXTRACEOUT1(DBGBIT(12, 0x20), 1, " %s", buf)
}

static void
update_param_use_count(OPERAND *params)
{
  while (params) {
    if (params->ot_type == OT_TMP) {
      params->tmps->use_count++;
    }
    params = params->next;
  }
}

static void
replace_by_fma_intrinsic(INSTR_LIST *instr, OPERAND *op, INSTR_LIST *mul_instr)
{
  OPERAND *params;
  const char *intrinsic_name = NULL;

  switch (instr->ll_type->data_type) {
  case LL_FLOAT:
    if (instr->i_name == I_FADD)
      intrinsic_name = "@llvm.pgi.arm.vmla.f32";
    else if (instr->i_name == I_FSUB)
      intrinsic_name = "@llvm.pgi.arm.vmls.f32";
    break;
  case LL_DOUBLE:
    if (instr->i_name == I_FADD)
      intrinsic_name = "@llvm.pgi.arm.vmla.f64";
    else if (instr->i_name == I_FSUB)
      intrinsic_name = "@llvm.pgi.arm.vmls.f64";
    break;
  default:
    break;
  }
  if (intrinsic_name) {
    params = gen_copy_op(op);
    params->next = gen_copy_list_op(mul_instr->operands);
    update_param_use_count(params);
    replace_by_call_to_llvm_instrinsic(instr, intrinsic_name, params);
    DEC_UCOUNT(mul_instr);
  }
}

static INSTR_LIST *
is_from_instr(int i_name, OPERAND *op)
{
  if (op->ot_type == OT_TMP) {
    INSTR_LIST *idef;
    idef = op->tmps->info.idef;
    if (idef && (idef->i_name == i_name))
      return idef;
  }
  return NULL;
}

static void
optimize_instruction(INSTR_LIST *instr)
{
  INSTR_LIST *op_instr;

  if (instr->tmps && instr->tmps->use_count == 0)
    return;
  switch (instr->i_name) {
  default:
    break;
  case I_FADD:
    if ((op_instr = is_from_instr(I_FMUL, instr->operands)))
      replace_by_fma_intrinsic(instr, instr->operands, op_instr);
    else if ((op_instr = is_from_instr(I_FMUL, instr->operands->next)))
      replace_by_fma_intrinsic(instr, instr->operands->next, op_instr);
    break;
  case I_FSUB:
    if ((op_instr = is_from_instr(I_FMUL, instr->operands->next)))
      replace_by_fma_intrinsic(instr, instr->operands->next, op_instr);
    break;
  }
}
#endif

void
optimize_block(INSTR_LIST *last_block_instr)
{
#ifdef TARGET_LLVM_ARM
  INSTR_LIST *instr, *last_instr;

  last_instr = NULL;
  for (instr = last_block_instr; instr; instr = instr->prev) {
    instr->flags |= INST_VISITED;

    if (last_instr == NULL && instr->i_name == I_NONE)
      last_instr = instr;
    if (instr->flags & STARTEBB) {
      if (last_instr != NULL)
        break;
    }
  }

  for (instr = last_block_instr; instr; instr = instr->prev) {
    optimize_instruction(instr);

    if (last_instr == NULL && instr->i_name == I_NONE)
      last_instr = instr;
    if (instr->flags & STARTEBB) {
      if (last_instr != NULL)
        break;
    }
  }

  for (instr = last_block_instr; instr; instr = instr->prev) {
    instr->flags &= ~INST_VISITED;

    if (last_instr == NULL && instr->i_name == I_NONE)
      last_instr = instr;
    if (instr->flags & STARTEBB) {
      if (last_instr != NULL)
        break;
    }
  }
#endif
}

/**
   \brief Determine if \p cand has the form <tt>1.0 / y</tt>
 */
static bool
is_recip(OPERAND *cand)
{
  if (cand && cand->tmps) {
    INSTR_LIST *il = cand->tmps->info.idef;
    const int divIli = il ? il->ilix : 0;
    OPERAND *ilOp = divIli ? il->operands : NULL;
    if (ilOp && (cand->tmps->use_count == 1) && (il->i_name == I_FDIV) &&
        (ilOp->ot_type == OT_CONSTSPTR)) {
      const int sptr = ilOp->val.sptr;
      switch (ILI_OPC(ILI_OPND(divIli, 1))) {
      case IL_FCON:
        return sptr == stb.flt1;
      case IL_DCON:
        return sptr == stb.dbl1;
#ifdef LONG_DOUBLE_FLOAT128
      case IL_FLOAT128CON:
        return sptr == stb.float128_1;
#endif
      default:
        break;
      }
    }
  }
  return false;
}

/**
   \brief Helper function
   \param x  This is the \c x operand in a <tt>(/ x)</tt> insn [precondition]
   \param recip  The <tt>(/ 1.0 y)</tt> term for splicing

   Peephole rewrite of the bridge IR. The C compiler will DCE the unused div
   operation. The C++ compiler will not, but instead leans on LLVM to DCE the
   <tt>(/ 1.0 undef)</tt> operation.
 */
static void
fixup_recip_div(OPERAND *x, OPERAND *recip)
{
  INSTR_LIST *il = recip->tmps->info.idef; // il <- (/ 1.0 y)
  OPERAND *undef = make_undef_op(il->operands->next->ll_type);
  x->next = il->operands->next; // (/ x) ==> (/ x y)
  il->operands->next = undef;   // (/ 1.0 y) ==> (/ 1.0 undef)
  recip->tmps->use_count--;
}

/**
   \brief Translate a fp mul to a fp div ILI opcode
   \param opc  The opcode to translate
   \return The DIV form if \c opc is a FP MUL, otherwise \c opc itself

   NB: Used to overwrite the opcode in the ILI. Any subsequent passes (FMA) that
   examine the ILI must not conclude that this is still a multiply operation.
 */
static ILI_OP
convert_mul_to_div(ILI_OP opc)
{
  switch (opc) {
  case IL_FMUL:
    return IL_FDIV;
  case IL_DMUL:
    return IL_DDIV;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128MUL:
    return IL_FLOAT128DIV;
#endif
  default:
    break;
  }
  return opc;
}

/**
   \brief Translate <tt>x * 1.0 / y</tt> to <tt>x / y</tt>.
   \param mul  A FP multiply instruction

   Preconditions: \p mul is a well-formed I_FMUL, has a positive use count
 */
void
maybe_undo_recip_div(INSTR_LIST *mul)
{
  OPERAND *lop = mul->operands;
  OPERAND *rop = lop->next;

  if (is_recip(lop)) {
    // case: (1.0 / y) * x
    mul->i_name = I_FDIV;
    ILI_OPCP(mul->ilix, convert_mul_to_div(ILI_OPC(mul->ilix)));
    mul->operands = rop; // x
    fixup_recip_div(rop, lop);
  } else if (is_recip(rop)) {
    // case: x * (1.0 / y)
    mul->i_name = I_FDIV;
    ILI_OPCP(mul->ilix, convert_mul_to_div(ILI_OPC(mul->ilix)));
    fixup_recip_div(lop, rop);
  } else {
    // mul not recognized as a mult-by-recip form
    // ok, do nothing
  }
}

/* ---------------------------------------------------------------------- */
//
// Widening transform:
//
// Rewrite the ILIs such that address arithmetic is done in the i64 domain
// rather than mostly in the i32 domain and then extended late to i64. It is
// understood that the behavior at two's complement i32's boundaries is not
// semantically identical, and we don't make wraparound guarantees here.

/**
   \brief Return a wide integer dtype, either signed or unsigned
 */
INLINE static DTYPE
getWideDType(bool isUnsigned)
{
  return isUnsigned ? DT_UINT8 : DT_INT8;
}

/**
   \brief Create a new temp that is a wide integer type
   \param dty  The wide integer dtype
 */
static SPTR
getNewWideSym(DTYPE dty)
{
  static int bump;
  SPTR wideSym = getccsym('w', ++bump, ST_VAR);

  SCP(wideSym, SC_AUTO);
  SCOPEP(wideSym, 3);
  DTYPEP(wideSym, dty);
  return wideSym;
}

/**
   \brief Push a cast of \p opc down the tree \p ilix
 */
static int
widenPushdown(ILI_OP opc, int ilix)
{
  int l, r, r3, r4;
  ILI_OP newOp;

  switch (ILI_OPC(ilix)) {
  case IL_KIMV:
    return ILI_OPND(ilix, 1);
  case IL_ICJMP:
    l = widenPushdown(opc, ILI_OPND(ilix, 1));
    r = widenPushdown(opc, ILI_OPND(ilix, 2));
    r3 = ILI_OPND(ilix, 3);
    r4 = ILI_OPND(ilix, 4);
    return ad4ili(IL_KCJMP, l, r, r3, r4);
  case IL_UICJMP:
    l = widenPushdown(opc, ILI_OPND(ilix, 1));
    r = widenPushdown(opc, ILI_OPND(ilix, 2));
    r3 = ILI_OPND(ilix, 3);
    r4 = ILI_OPND(ilix, 4);
    return ad4ili(IL_UKCJMP, l, r, r3, r4);
  case IL_IKMV:
  case IL_UIKMV:
    return widenPushdown(opc, ILI_OPND(ilix, 1));
  case IL_LDKR:
  case IL_KMUL:
  case IL_KADD:
  case IL_KSUB:
  case IL_KCON:
    return ilix;
  default:
    return ad1ili(opc, ilix);
  case IL_IADD:
  case IL_UIADD:
    newOp = IL_KADD;
    break;
  case IL_ISUB:
  case IL_UISUB:
    newOp = IL_KSUB;
    break;
  case IL_IMUL:
  case IL_UIMUL:
    newOp = IL_KMUL;
    break;
  case IL_IMAX:
    newOp = IL_KMAX;
    break;
  case IL_IMIN:
    newOp = IL_KMIN;
    break;
  }
  l = widenPushdown(opc, ILI_OPND(ilix, 1));
  r = widenPushdown(opc, ILI_OPND(ilix, 2));
  return ad2ili(newOp, l, r);
}

/**
   \brief Perform widening on address arithmetic
   \param ilix  The root of the tree to be widened

   The root of the tree will already be wide because of large arrays.  However,
   we want to force any sign- or zero-extension operations down towards the
   leaves of the tree and promote the arithmetic operations to 64 bits.
 */
static bool
widenAddressArithmetic(int ilix)
{
  int i;
  bool rv = false;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;

  for (i = 1; i <= noprs; ++i)
    if (IL_ISLINK(opc, i)) {
      const int opnd = ILI_OPND(ilix, i);
      ILI_OP opx = ILI_OPC(opnd);
      if ((opx == IL_IKMV) || (opx == IL_UIKMV)) {
        int x = widenPushdown(opx, ILI_OPND(opnd, 1));
        ILI_OPND(ilix, i) = x;
        rv = true;
      } else {
        rv |= widenAddressArithmetic(ILI_OPND(ilix, i));
      }
    }
  return rv;
}

/**
   \brief Find address arithmetic and widen it
   \param ilix  The root of the ILI tree to be examined

   We traverse the tree and look for any address arithmetic. If any is found,
   call widenAddressArithmetic().
 */
static bool
widenAnyAddressing(int ilix)
{
  bool rv = false;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;

  if (opc == IL_KAMV) {
    rv = widenAddressArithmetic(ILI_OPND(ilix, 1));
  } else {
    int i;
    for (i = 1; i <= noprs; ++i)
      if (IL_ISLINK(opc, i))
        rv |= widenAnyAddressing(ILI_OPND(ilix, i));
  }
  return rv;
}

/**
   \brief Add widen targets in \p ilix to the var map.
 */
static void
widenAddNarrowVars(int ilix, hashset_t widenVar_set)
{
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;

  for (i = 1; i <= noprs; ++i)
    if (IL_ISLINK(opc, i)) {
      const int opnd = ILI_OPND(ilix, i);
      if (((opc == IL_IKMV) || (opc == IL_UIKMV))) {
        if (IL_TYPE(ILI_OPC(opnd)) == ILTY_LOAD) 
          hashset_replace(widenVar_set, INT2HKEY(opnd));
      } else {
        widenAddNarrowVars(opnd, widenVar_set);
      }
    }
}

/**
   \brief Test if this load is from a private variable
 */
static bool
widenAconIsPrivate(int ilix)
{
  SPTR sym;

  assert(ILI_OPC(ilix) == IL_ACON, "ilix must be ACON", ilix, ERR_Fatal);
  sym = ILI_SymOPND(ilix, 1);
  if (DTY(DTYPEG(sym)) == TY_PTR)
    sym = SymConval1(sym);

  if (DT_ISINT(DTYPEG(sym)))
    return (SCG(sym) == SC_PRIVATE);
  return false;
}

/**
   \brief Add a new wider store

   Generate a new tree and insert it after \p ilt.
 */
INLINE static int
widenInsertWideStore(int ilt, int lhsIli, int rhsIli, int nme)
{
  const int newIli = ad4ili(IL_STKR, rhsIli, lhsIli, nme, MSZ_I8);
  addilt(ilt, newIli);
  return newIli;
}

INLINE static void
widenProcessDirectLoad(int ldIli, hashmap_t map)
{
  const DTYPE dty = getWideDType(false); // FIXME
  const SPTR wideVar = getNewWideSym(dty);
  const int nme = ILI_OPND(ldIli, 2);
  const DTYPE wty = get_type(2, TY_PTR, dty);
  const int wideAddr = ad1ili(IL_ACON, get_acon3(wideVar, 0, wty));
  const int wideLoad = ad1ili(IL_KIMV, ad3ili(IL_LDKR, wideAddr, nme, MSZ_I8));
  hash_data_t data = INT2HKEY(wideLoad);
  hashmap_replace(map, INT2HKEY(ldIli), &data);
}

INLINE static bool
hasExactlyOneStore(int aconIli, int *cilt)
{
  int bih, ilt;
  unsigned count = 0;

  for (bih = BIH_NEXT(0); bih; bih = BIH_NEXT(bih)) {
    // make sure aconSym is written to once
    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
      const int ilix = ILT_ILIP(ilt);
      if (ILT_DELETE(ilt))
        continue;
      if ((IL_TYPE(ILI_OPC(ilix)) == ILTY_STORE) &&
          (ILI_OPND(ilix, 2) == aconIli)) {
        ++count;
        *cilt = ilt;
      }
    }
  }
  return (count == 1);
}

INLINE static void
widenProcessIndirectLoad(int ldIli, int aconIli, hashmap_t map)
{
  int cilt;
  if (hasExactlyOneStore(aconIli, &cilt)) {
    int st;
    const DTYPE dty = getWideDType(false); // FIXME
    const SPTR wideVar = getNewWideSym(dty);
    const int nme = ILI_OPND(ldIli, 2);
    const DTYPE tyw = get_type(2, TY_PTR, dty);
    const int wideAddr = ad1ili(IL_ACON, get_acon3(wideVar, 0, tyw));
    int wideLoad = ad1ili(IL_KIMV, ad3ili(IL_LDKR, wideAddr, nme, MSZ_I8));
    hash_data_t data = INT2HKEY(wideLoad);
    hashmap_replace(map, INT2HKEY(ldIli), &data);
    wideLoad = ad1ili(IL_IKMV, ldIli);
    st = widenInsertWideStore(cilt, wideAddr, wideLoad, nme);
    data = 0;
    hashmap_replace(map, INT2HKEY(st), &data);
  }
}

/**
   \brief Create a wide load from the given narrow load
   \param key  The narrow load (as a key)
   \param map  A hashmap to add the (narrow -> wide) mapping
 */
static void
widenCreateWideLocal(hash_key_t key, void *map)
{
  const int loadIli = HKEY2INT(key);
  hashmap_t newMap = (hashmap_t)map;
  const ILI_OP ldOpc = ILI_OPC(loadIli);

  assert(ldOpc == IL_LD, "load map does not contain load", ldOpc, ERR_Fatal);

  // Does loadIli match indirect form?
  if ((ILI_OPC(ILI_OPND(loadIli, 1)) == IL_LDA) &&
      (ILI_OPC(ILI_OPND(ILI_OPND(loadIli, 1), 1)) == IL_ACON)) {
    const int aconIli = ILI_OPND(ILI_OPND(loadIli, 1), 1);
    if (widenAconIsPrivate(aconIli))
      widenProcessIndirectLoad(loadIli, aconIli, newMap);
  } else if (ILI_OPC(ILI_OPND(loadIli, 1)) == IL_ACON) {
    const int aconIli = ILI_OPND(loadIli, 1);
    if (widenAconIsPrivate(aconIli))
      widenProcessDirectLoad(loadIli, newMap);
  }
}

INLINE static void
widenApplyFree(int ilix, hashmap_t map)
{
  const ILI_OP opc = ILI_OPC(ilix);
  if (opc == IL_FREEIR) {
    const int argIli = ILI_OPND(ilix, 1);
    hash_data_t data;
    if (hashmap_lookup(map, INT2HKEY(argIli), &data)) {
      const int newIli = HKEY2INT(data);
      ILI_OPCP(ilix, IL_FREEKR);
      ILI_OPND(ilix, 1) = ILI_OPND(newIli, 1);
    }
  }
}

static void
widenApplyVarMap(int ilix, hashmap_t map)
{
  hash_data_t data;
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;

  if (hashmap_lookup(map, INT2HKEY(ilix), &data))
    if (!data)
      return;

  for (i = 1; i <= noprs; ++i)
    if (IL_ISLINK(opc, i)) {
      const int opnd = ILI_OPND(ilix, i);
      const ILI_OP opx = ILI_OPC(opnd);
      if ((opx == IL_IKMV) || (opx == IL_UIKMV)) {
        const int opnd2 = ILI_OPND(opnd, 1);
        if (hashmap_lookup(map, INT2HKEY(opnd2), &data)) {
          const int newIli = HKEY2INT(data);
          ILI_OPND(ilix, i) = ILI_OPND(newIli, 1);
        } else {
          widenApplyVarMap(opnd2, map);
        }
      } else if (opx == IL_CSEIR) {
        // ignore
      } else if (hashmap_lookup(map, INT2HKEY(opnd), &data)) {
        const int newIli = HKEY2INT(data);
        ILI_OPND(ilix, i) = newIli;
      } else {
        widenApplyVarMap(opnd, map);
      }
    }
  if (ILI_ALT(ilix))
    widenApplyVarMap(ILI_ALT(ilix), map);
}

static void
widenApplyCse(int ilix, hashmap_t map)
{
  hash_data_t data;
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;

  if (hashmap_lookup(map, INT2HKEY(ilix), &data))
    if (!data)
      return;

  for (i = 1; i <= noprs; ++i)
    if (IL_ISLINK(opc, i)) {
      const int opnd = ILI_OPND(ilix, i);
      const ILI_OP opx = ILI_OPC(opnd);
      if ((opx == IL_IKMV) || (opx == IL_UIKMV)) {
        const int opnd2 = ILI_OPND(opnd, 1);
        if ((ILI_OPC(opnd2) == IL_CSEIR) &&
            hashmap_lookup(map, INT2HKEY(ILI_OPND(opnd2, 1)), &data)) {
          const int replIlix = ILI_OPND(HKEY2INT(data), 1);
          ILI_OPND(ilix, i) = ad1ili(IL_CSEKR, replIlix);
        }
      } else {
        widenApplyCse(opnd, map);
      }
    }
}

#if DEBUG
static void
dumpWidenVars(hash_key_t key, hash_data_t data, void *_)
{
  FILE *f = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(f, "{\n\tkey:\n");
  dilitre(HKEY2INT(key));
  fprintf(f, "\tdata:\n");
  if (data)
    dilitre(HKEY2INT(data));
  fprintf(f, "}\n");
}
#endif

/**
   \brief If <tt>{key -> data}</tt> maps to an ACON, place ACON in \p map_
   \param key  A hashmap key
   \param data A hashmap value
   \param map_ A hashmap to be populated
   Called by hashmap_iterate().
 */
static void
gatherLocals(hash_key_t key, hash_data_t data, void *map_)
{
  const int ldIli = HKEY2INT(key);
  hashmap_t map = (hashmap_t)map_;

  if (!data)
    return;
  if (ILI_OPC(ILI_OPND(ldIli, 1)) == IL_ACON)
    hashmap_replace(map, INT2HKEY(ILI_OPND(ldIli, 1)), &data);
}

INLINE static void
widenApplyStore(int ilix, hashmap_t map)
{
  if (IL_TYPE(ILI_OPC(ilix)) == ILTY_STORE) {
    const int stAcon = ILI_OPND(ilix, 2);
    hash_data_t data;
    if (hashmap_lookup(map, INT2HKEY(stAcon), &data)) {
      ILI_OPND(ilix, 4) = MSZ_I8;
      ILI_OPND(ilix, 2) = ILI_OPND(ILI_OPND(HKEY2INT(data), 1), 1);
      ILI_OPND(ilix, 1) = ad1ili(IL_IKMV, ILI_OPND(ilix, 1));
    }
  }
}

/**
   \brief Does the current procedure have blocks marked no dep check?
 */
bool
funcHasNoDepChk(void)
{
  int bih = BIH_NEXT(0);
  for (; bih; bih = BIH_NEXT(bih))
    if (BIH_NODEPCHK(bih))
      return true;
  return false;
}

static void
widenElimAddrTaken(int ilix, hashset_t zt)
{
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;

  if ((IL_TYPE(opc) == ILTY_STORE) &&
      (ILI_OPC(ILI_OPND(ilix, 2)) == IL_ACON)) {
    widenElimAddrTaken(ILI_OPND(ilix, 1), zt);
    return;
  }
  if ((opc == IL_ACON) && hashset_lookup(zt, INT2HKEY(ilix))) {
    hashset_erase(zt, INT2HKEY(ilix));
    return;
  }
  for (i = 1; i <= noprs; ++i)
    if (IL_ISLINK(opc, i)) {
      const int opnd = ILI_OPND(ilix, i);
      const ILI_OP opx = ILI_OPC(opnd);
      if ((IL_TYPE(opx) == ILTY_LOAD) &&
          (ILI_OPC(ILI_OPND(opnd, 1)) == IL_ACON))
        continue;
      widenElimAddrTaken(ILI_OPND(ilix, i), zt);
    }
  if (ILI_ALT(ilix))
    widenElimAddrTaken(ILI_ALT(ilix), zt);
}

static void
widenGatherAcon(hash_key_t key, void *zet_)
{
  hashset_t zet = (hashset_t)zet_;
  const int ilix = HKEY2INT(key);
  const int acon = ILI_OPND(ilix, 1);
  assert(IL_TYPE(ILI_OPC(ilix)) == ILTY_LOAD, "expected load", 0, ERR_Fatal);
  hashset_replace(zet, INT2HKEY(acon));
}

INLINE static void
hashset_swap(hashset_t *s1, hashset_t *s2)
{
  hashset_t swap = *s1;
  *s1 = *s2;
  *s2 = swap;
}

typedef struct { hashset_t inSet; hashset_t aconSet; } AconSets;

/**
   \brief Populate \c inSet with \p key if acon is in \c aconSet
 */
static void
pruneAcon(hash_key_t key, void *p)
{
  AconSets *sp = (AconSets*) p;
  const int acon = ILI_OPND(HKEY2INT(key), 1);
  if (hashset_lookup(sp->aconSet, INT2HKEY(acon)))
    hashset_replace(sp->inSet, key);
}

/**
   \brief Prune \p loadSt to only those loads that reference acon in \p aconSt
   \param loadSt   The set of loads to be pruned
   \param aconSt   The set of legal ACON nodes
 */
INLINE static void
widenPruneAcon(hashset_t *loadSt, hashset_t aconSt)
{
  hashset_t newSet = hashset_alloc(hash_functions_direct);
  AconSets aconSets = {newSet, aconSt};
  hashset_iterate(*loadSt, pruneAcon, (void*)&aconSets);
  hashset_swap(loadSt, &newSet);
  hashset_free(newSet);
}

/**
   \brief Widen address arithmetic
 */
void
widenAddressArith(void)
{
  int bih, ilt;
  hashset_t widenVar_set;

  // paranoia check: only process functions with nodepchk flag
  if (!funcHasNoDepChk())
    return;

#if DEBUG
  if (DBGBIT(12, 0x40))
    dumpblocks("before widen");
#endif

  widenVar_set = hashset_alloc(hash_functions_direct);
  for (bih = BIH_NEXT(0); bih; bih = BIH_NEXT(bih))
    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
      const int ilix = ILT_ILIP(ilt);
      if (ILT_DELETE(ilt))
        continue;
      if (ILI_OPC(ilix) == IL_ICJMP) {
        const int x = widenPushdown(IL_IKMV, ilix);
        ILT_ILIP(ilt) = x;
        widenAddNarrowVars(x, widenVar_set);
      } else if (ILI_OPC(ilix) == IL_UICJMP) {
        const int x = widenPushdown(IL_UIKMV, ilix);
        ILT_ILIP(ilt) = x;
        widenAddNarrowVars(x, widenVar_set);
      } else {
        const bool found = widenAnyAddressing(ilix);
        if (found)
          widenAddNarrowVars(ilix, widenVar_set);
      }
    }

  if (hashset_size(widenVar_set)) {
    hashset_t zet = hashset_alloc(hash_functions_direct);
    hashset_iterate(widenVar_set, widenGatherAcon, (void*)zet);
    for (bih = BIH_NEXT(0); bih; bih = BIH_NEXT(bih))
      for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt))
        widenElimAddrTaken(ILT_ILIP(ilt), zet);
    widenPruneAcon(&widenVar_set, zet);
    hashset_free(zet);
  }

  if (hashset_size(widenVar_set)) {
    hashmap_t newMap = hashmap_alloc(hash_functions_direct);
    hashmap_t aconMap = hashmap_alloc(hash_functions_direct);
    hashset_iterate(widenVar_set, widenCreateWideLocal, newMap);
    hashmap_iterate(newMap, gatherLocals, aconMap);
#if DEBUG
    if (DBGBIT(12, 0x40)) {
      hashmap_iterate(newMap, dumpWidenVars, NULL);
    }
#endif
    for (bih = BIH_NEXT(0); bih; bih = BIH_NEXT(bih))
      for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
        const int ilix = ILT_ILIP(ilt);
        widenApplyFree(ilix, newMap);
        widenApplyVarMap(ilix, newMap);
        widenApplyStore(ilix, aconMap);
        widenApplyCse(ilix, newMap);
      }
    hashmap_free(aconMap);
    hashmap_free(newMap);
  }

#if DEBUG
  if (DBGBIT(12, 0x40))
    dumpblocks("after widen");
#endif

  hashset_free(widenVar_set);
}

/* ---------------------------------------------------------------------- */
//
// Replace load-load sequences in "omp simd" loops transform:
//
// We want to cache the result of the first load (a pointer) in a temp and
// have the second load use the temp in the body of the loop.  This is to
// workaround the issue that the outliner creates a bag of <tt>void*</tt>
// for all the enclosing routine's variables and introduces false aliasing.

/**
   \brief Does block \p bih branch to block \p target?
 */
bool
block_branches_to(int bih, int target)
{
  if (bih != 0) {
    const int ilt = BIH_ILTLAST(bih);
    const int ili = ILT_ILIP(ilt);
    const ILI_OP op = ILI_OPC(ili);
    if (IL_TYPE(op) == ILTY_BRANCH) {
      const int lab = ILI_OPND(ili, ilis[op].oprs);
      const int targ = ILIBLKG(lab);
      return (targ == target);
    }
  }
  return false;
}

INLINE static void
rlleMakeSet(hash_key_t key, hash_data_t data, void *set_)
{
  hashset_t zet = (hashset_t)set_;
  const int isGood = HKEY2INT(data);
  if (isGood)
    hashset_insert(zet, key);
}

/**
   \brief Search the omp simd block for load-load patterns
   \param ilix  the root of the ILI tree
   \param zt    set of candidates (<tt>ACON</tt>)
   \param mp    <tt>{ACON -> 0}</tt> (output map)
 */
static void
rlleFindLL(int ilix, hashset_t zt, hashmap_t mp)
{
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;
  hash_data_t data = INT2HKEY(0);

  for (i = 1; i <= noprs; ++i)
    if (IL_ISLINK(opc, i)) {
      const int opnd = ILI_OPND(ilix, i);
      const ILI_OP opx = ILI_OPC(opnd);
      if (IL_TYPE(opx) == ILTY_LOAD) {
        const int opnd2 = ILI_OPND(opnd, 1);
        if ((IL_TYPE(ILI_OPC(opnd2)) == ILTY_LOAD) &&
            hashset_lookup(zt, INT2HKEY(ILI_OPND(opnd2, 1)))) {
          const int opnd3 = ILI_OPND(opnd2, 1);
          hashmap_replace(mp, INT2HKEY(opnd3), &data);
        } else {
          rlleFindLL(opnd, zt, mp);
        }
      } else {
        rlleFindLL(opnd, zt, mp);
      }
    }
}

/**
   \brief Was an initializer for this load pair created?
   If a candidate's initializer cannot be found, the map will be 0.
 */
INLINE static bool
rlleInitializerFound(hash_data_t data)
{
  return data != INT2HKEY(0);
}

/**
   \brief Rewrite the ILIs with substitutions for load-load patterns
 */
static void
rlleReplace(int ilix, hashmap_t map)
{
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;

  for (i = 1; i <= noprs; ++i)
    if (IL_ISLINK(opc, i)) {
      const int opnd = ILI_OPND(ilix, i);
      const ILI_OP opx = ILI_OPC(opnd);
      // look for (LDA (LDA (ACON _))) where ACON is in map
      if (IL_TYPE(opx) == ILTY_LOAD) {
        const int opnd2 = ILI_OPND(opnd, 1);
        hash_data_t data;
        if ((IL_TYPE(ILI_OPC(opnd2)) == ILTY_LOAD) &&
            hashmap_lookup(map, INT2HKEY(ILI_OPND(opnd2, 1)), &data) &&
            rlleInitializerFound(data)) {
          // rewrite to (LDA (ACON' _))
          ILI_OPND(opnd, 1) = HKEY2INT(data);
        } else {
          rlleReplace(opnd2, map);
        }
      } else {
        rlleReplace(opnd, map);
      }
    }
}

/**
   \brief Eliminate load-load operations from the uplevel structure
   \pre The current procedure is an outlined function with no dep check blocks
 */
void
redundantLdLdElim(void)
{
  int bih, ilt;
  hashmap_t candMap = hashmap_alloc(hash_functions_direct);

  // scan forward to find candidates
  //  construct map of private variables written exactly once
  for (bih = BIH_NEXT(0); bih; bih = BIH_NEXT(bih)) {
    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
      const int ilix = ILT_ILIP(ilt);
      if (ILT_DELETE(ilt))
        continue;
      if ((IL_TYPE(ILI_OPC(ilix)) == ILTY_STORE) &&
          (ILI_OPC(ILI_OPND(ilix, 2)) == IL_ACON) &&
          widenAconIsPrivate(ILI_OPND(ilix, 2))) {
        const int acon = ILI_OPND(ilix, 2);
        hash_data_t data;
        if (hashmap_lookup(candMap, INT2HKEY(acon), &data)) {
          data = INT2HKEY(0);
          hashmap_replace(candMap, INT2HKEY(acon), &data);
        } else {
          data = INT2HKEY(1);
          hashmap_insert(candMap, INT2HKEY(acon), data);
        }
      }
    }
  }
  if (hashmap_size(candMap)) {
    // find candidates that appear in omp simd block, create new map
    bool inNoDep = false;
    hashset_t zet = hashset_alloc(hash_functions_direct);
    hashmap_iterate(candMap, rlleMakeSet, (void*)zet);
    hashmap_clear(candMap);
    for (bih = BIH_NEXT(0); bih; bih = BIH_NEXT(bih)) {
      if (!inNoDep)
        inNoDep = BIH_NODEPCHK(bih);
      if (inNoDep)
        for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
          const int ilix = ILT_ILIP(ilt);
          rlleFindLL(ilix, zet, candMap);
        }
      if (!(inNoDep && block_branches_to(BIH_NEXT(bih), bih)))
        inNoDep = false;
    }
    hashset_free(zet);
  }

  // if we have no candidates, we're done
  if (hashmap_size(candMap) == 0) {
    hashmap_free(candMap);
    return;
  }

  // scan the header blocks and create temps, updating map
  for (bih = BIH_NEXT(0); bih; bih = BIH_NEXT(bih)) {
    if (BIH_NODEPCHK(bih))
      break;
    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
      const int ilix = ILT_ILIP(ilt);
      hash_data_t data;
      if (IL_TYPE(ILI_OPC(ilix)) == ILTY_STORE) {
        if (hashmap_lookup(candMap, INT2HKEY(ILI_OPND(ilix, 2)), &data)) {
          const int acon = ILI_OPND(ilix, 2);
          const int nme = ILI_OPND(ilix, 3);
          const int lda = ad3ili(IL_LDA, acon, nme, MSZ_PTR);
          const int loada = ad3ili(IL_LDA, lda, nme, MSZ_PTR);
          const DTYPE dty = DT_CPTR;
          const SPTR wideVar = getNewWideSym(dty);
          const int wAddr = ad1ili(IL_ACON, get_acon3(wideVar, 0, dty));
          const int stv = ad4ili(IL_STA, loada, wAddr, nme, MSZ_PTR);
          assert(!data, "data should be null", HKEY2INT(data), ERR_Fatal);
          data = INT2HKEY(wAddr);
          hashmap_replace(candMap, INT2HKEY(acon), &data);
          addilt(ilt, stv);
          ilt = ILT_NEXT(ilt);
        }
      }
    }
  }

  // scan the function and replace load-load sequences with temps, if available
  for (; bih; bih = BIH_NEXT(bih))
    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt))
      rlleReplace(ILT_ILIP(ilt), candMap);

  hashmap_free(candMap);
}
