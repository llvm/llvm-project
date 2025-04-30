/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief ILI utility module
 */

#include "pgifeat.h" // defines TARGET_64BIT et al.
#include "iliutil.h"
#include "exputil.h"
#include "expreg.h"
#include "regutil.h"
#include "machreg.h"
#define EXPANDER_DECLARE_INTERNAL
#include "expand.h"
#include "machar.h"
#include "outliner.h"
#include "assem.h"
#include "mach.h"
#include "dtypeutl.h"
#include "llutil.h"
#include "llassem.h"
#include "llmputil.h"
#include "expsmp.h"
#include "verify.h"
#if DEBUG
#include "nme.h"
#endif
#include <stdarg.h>
#include "scutil.h"
#include "symfun.h"

#if defined(OMP_OFFLOAD_PGI) || defined(OMP_OFFLOAD_LLVM)
#include "ompaccel.h"
#endif

/*
 * MTH, FMTH, ... names
 */
#include "mth.h"

/** Union used to encode ATOMIC_INFO as an int or decode it.  The union
    contains a compacted version of ATOMIC_INFO using bitfields.  The bitfields
    are unsigned rather than the proper enum type so that the values aren't
    sign extended when extracted. */
union ATOMIC_ENCODER {
  struct {
    unsigned msz : 8;
    unsigned op : 8;
    unsigned origin : 2;
    unsigned scope : 1;
  } info;
  int encoding;
};

#define IL_spfunc IL_DFRSP
#define IL_dpfunc IL_DFRDP
#define IL_qpfunc IL_DFRQP

bool share_proc_ili = false;
bool share_qjsr_ili = false;
extern bool ishft;

static int addarth(ILI *);
static int red_iadd(int, INT);
static int red_kadd(int, INT[2]);
static int red_aadd(int, SPTR, ISZ_T, int);
static int red_damv(int, int, int);
static int red_minmax(ILI_OP, int, int);
static int red_negate(int, ILI_OP, int, int);
static int addbran(ILI *);
static int addother(ILI *);
static INT icmp(INT, INT);
static INT cmp_to_log(INT, int);
static bool isub_ovf(INT, INT, INT *);
static bool iadd_ovf(INT, INT, INT *);
static int get_ili(ILI *);
static int new_ili(ILI *);
static int ad1altili(ILI_OP, int, int);
static int ad2altili(ILI_OP, int, int, int);
static int ad3altili(ILI_OP, int, int, int, int);
static int ad2func_int(ILI_OP, const char *, int, int);
static int gen_sincos(ILI_OP, int, ILI_OP, ILI_OP, MTH_FN, DTYPE, ILI_OP);
#if defined(TARGET_X8664) || defined(TARGET_POWER)
static int _newton_fdiv(int, int);
#endif
static bool do_newton_sqrt(void);
static int _pwr2(INT, int);
static int _kpwr2(INT, INT, int);
static int _ipowi(int, int);
static int _xpowi(int, int, ILI_OP);
#if defined(TARGET_X8664) || defined(TARGET_POWER) || !defined(TARGET_LLVM_ARM)
static int _frsqrt(int);
#endif
static int _mkfunc(const char *);
static int DblIsSingle(SPTR dd);
static int _lshift_one(int);
static int cmpz_of_cmp(int, CC_RELATION);
static bool is_zero_one(int);
static bool _is_nanf(int);
static INT value_of_irlnk_operand(int ilix, int default_value);

#if defined(TARGET_WIN_X8664)
static void insert_argrsrv(ILI *);
#endif

// FIXME: mk_prototype_llvm should return an SPTR
#define mk_prototype (SPTR) mk_prototype_llvm

#define ILTABSZ 5
#define ILHSHSZ 173
#define MAXILIS 67108864

static int ilhsh[ILTABSZ][ILHSHSZ];
static bool safe_qjsr = false;

#define GARB_UNREACHABLE 0
#define GARB_VISITED 1

#ifdef __cplusplus
inline CC_RELATION CCRelationILIOpnd(ILI *p, int n) {
  return static_cast<CC_RELATION>(p->opnd[n]);
}
inline DTYPE DTypeILIOpnd(ILI *p, int n) {
  return static_cast<DTYPE>(p->opnd[n]);
}
inline MSZ ConvertMSZ(int n) {
  return static_cast<MSZ>(n);
}
inline ATOMIC_RMW_OP ConvertATOMIC_RMW_OP(int n) {
  return static_cast<ATOMIC_RMW_OP>(n);
}
inline SPTR sptrGetILI(ILI *p) {
  return static_cast<SPTR>(get_ili(p));
}
#else
#define CCRelationILIOpnd(p,n) (p)->opnd[n]
#define DTypeILIOpnd(p,n)      (p)->opnd[n]
#define ConvertMSZ(n)           (n)
#define ConvertATOMIC_RMW_OP(n) (n)
#define sptrGetILI get_ili
#endif

/**
   \brief initialize ili area
 */
void
ili_init(void)
{
  int *p, cnt;
  static int firstcall = 1;

  STG_ALLOC(ilib, 2048);
  STG_SET_FREELINK(ilib, ILI, hshlnk);
  cnt = ILHSHSZ * ILTABSZ;

  if (firstcall) {
    firstcall = 0;
  } else {
    p = (int *)ilhsh;
    do {
      *p++ = 0;
    } while (--cnt > 0);
  }
  /* reserve ili index 1 to be the NULL ili.  done so that a traversal
   * which uses the ILI_VISIT field as a thread can use an ili (#1) to
   * terminate the threaded list
   */
  ad1ili(IL_NULL, 0);
}

void
ili_cleanup(void)
{
  STG_DELETE(ilib);
}

/**
   \brief principle add ili routine
 */
int
addili(ILI *ilip)
{
  ILI_OP opc;   /* opcode of ili  */
  int ilix = 0; /* ili area index where ili was added  */
  int tmp;      /* temporary  */
  int cons1;
  INT numi[2];

  opc = ilip->opc;
  switch (IL_TYPE(opc)) {

  case ILTY_ARTH:
    ilix = addarth(ilip);
    break;

  case ILTY_CONS:
  case ILTY_LOAD:
  case ILTY_STORE:
  case ILTY_DEFINE:
    ilix = get_ili(ilip);
    break;
  case ILTY_PROC:
    if (opc == IL_QJSR && share_qjsr_ili) {
      /*
       * normally (while expanding), we want qjsr's to be shared.
       * in other cases (vectorizer), the qjsr's created may have
       * side-effects (i.e., streamin) so we don't to share these
       * (expand brackets its use with true, false).
       */
      ilix = get_ili(ilip);
      if (!safe_qjsr)
        expb.qjsr_flag = true;
      break;
    }
    if (share_proc_ili)
      ilix = get_ili(ilip); /* share proc ili if option requested */
    else
      ilix = new_ili(ilip); /* o.w., ensure unique ili for proc */
    if (opc == IL_QJSR) {
      if (!safe_qjsr)
        expb.qjsr_flag = true;
      iltb.qjsrfg = true;
      BIH_QJSR(expb.curbih) = 1;
    }
    break;

  case ILTY_MOVE:
    switch (opc) {
      int op1, op2;
    case IL_KIMV:
      if (ILI_OPC(op1 = ilip->opnd[0]) == IL_KCON) {
        ilix = ad_icon(CONVAL2G(ILI_OPND(op1, 1)));
      } else if (ILI_OPC(ilip->opnd[0]) == IL_IKMV)
        ilix = ILI_OPND(op1, 1);
      else
        ilix = get_ili(ilip);
      break;
    case IL_IKMV:
      switch (ILI_OPC(op1 = ilip->opnd[0])) {
      case IL_ICON:
        ilix = ad_kconi(CONVAL2G(ILI_OPND(op1, 1)));
        break;
      case IL_IADD:
        op2 = ILI_OPND(op1, 2);
        if (ILI_OPC(op2) == IL_ICON) {
          op2 = ad_kconi(CONVAL2G(ILI_OPND(op2, 1)));
          op1 = ad1ili(IL_IKMV, ILI_OPND(op1, 1));
          ilix = ad2ili(IL_KADD, op1, op2);
        } else
          ilix = get_ili(ilip);
        break;
      case IL_ISUB:
        op2 = ILI_OPND(op1, 2);
        op1 = ILI_OPND(op1, 1);
        if (ILI_OPC(op2) == IL_ICON) {
          op2 = ad_kconi(CONVAL2G(ILI_OPND(op2, 1)));
          op1 = ad1ili(IL_IKMV, op1);
          ilix = ad2ili(IL_KSUB, op1, op2);
        } else if (ILI_OPC(op1) == IL_ICON) {
          op1 = ad_kconi(CONVAL2G(ILI_OPND(op1, 1)));
          op2 = ad1ili(IL_IKMV, op2);
          ilix = ad2ili(IL_KSUB, op1, op2);
        } else
          ilix = get_ili(ilip);
        break;
      default:
        ilix = get_ili(ilip);
      }
      break;
    case IL_UIKMV:
      if (ILI_OPC(op1 = ilip->opnd[0]) == IL_ICON) {
        numi[0] = CONVAL1G(op1 = ILI_OPND(op1, 1));
        numi[1] = CONVAL2G(op1);
        ilix = ad1ili(IL_KCON, getcon(numi, DT_INT8));
      } else
        ilix = get_ili(ilip);
      break;
    case IL_MVKR:
    case IL_MVIR:
    case IL_MVSP:
#ifdef IL_MVSPSP
    case IL_MVSPSP:
#endif
    case IL_MVQ:   /*m128*/
    case IL_MV256: /*m256*/
    case IL_MVDP:
    case IL_MVAR:
#ifdef LONG_DOUBLE_FLOAT128
    case IL_FLOAT128RETURN:
#endif
      if (ilip->opnd[1] == -1)
        ilix = ilip->opnd[0];
      else
        ilix = get_ili(ilip);
      break;

#ifdef IL_MVSPX87
    case IL_MVSPX87:
    case IL_MVDPX87:
      ilix = get_ili(ilip);
      break;
#endif

    case IL_IAMV:
      if (ILI_OPC(tmp = ilip->opnd[0]) == IL_ICON)
        ilix = ad_aconi(CONVAL2G(ILI_OPND(tmp, 1)));
      else if (ILI_OPC(tmp) == IL_AIMV)
        ilix = ILI_OPND(tmp, 1);
      else
        ilix = get_ili(ilip);
      break;

    case IL_KAMV:
      if (ILI_OPC(tmp = ilip->opnd[0]) == IL_KCON) {
        cons1 = ILI_OPND(tmp, 1);
        ilix = ad_aconk(CONVAL1G(cons1), ACONOFFG(cons1));
        return ilix;
      }
      tmp = ilip->opnd[0];
      if (ILI_OPC(tmp) == IL_AKMV)
        ilix = ILI_OPND(tmp, 1);
      else
        ilix = get_ili(ilip);
      break;

    case IL_AIMV:
      if ((ILI_OPC(tmp = ilip->opnd[0]) == IL_ACON) &&
          (CONVAL1G((tmp = ILI_OPND(tmp, 1))) == 0)) {
        ilix = ad_icon(CONVAL2G(tmp));
      } else if (ILI_OPC(tmp = ilip->opnd[0]) == IL_IAMV)
        ilix = ILI_OPND(tmp, 1);
      else
        ilix = get_ili(ilip);
      break;

    case IL_AKMV:
      if ((ILI_OPC(tmp = ilip->opnd[0]) == IL_ACON) &&
          (CONVAL1G((tmp = ILI_OPND(tmp, 1))) == 0)) {
        ISZ_2_INT64(ACONOFFG(tmp), numi);
        ilix = ad1ili(IL_KCON, getcon(numi, DT_INT8));
      } else if (ILI_OPC(tmp = ilip->opnd[0]) == IL_KAMV)
        ilix = ILI_OPND(tmp, 1);
      else
        ilix = get_ili(ilip);
      break;

    case IL_RETURN:
      ilix = get_ili(ilip);
      break;
    default:
      assert(false, "addili: unrec move opcode:", opc, ERR_Severe);
    }
    break;

  case ILTY_BRANCH:
    ilix = addbran(ilip);
    break;

  case ILTY_OTHER:
#ifdef ILTY_PSTORE
  case ILTY_PSTORE:
#endif
#ifdef ILTY_PLOAD
  case ILTY_PLOAD:
#endif
    ilix = addother(ilip);
    break;
#if DEBUG
  default:
    interr("addili: illegal IL_TYPE(opc)", IL_TYPE(opc), ERR_Fatal);
    break;
#endif
  }

#if DEBUG
  if (DBGBIT(10, 32)) {
    if (ilix)
      dump_ili(gbl.dbgfil, ilix);
  }
  if (ilix)
    verify_ili(ilix, VERIFY_ILI_SHALLOW);
#endif
  return ilix;
}

/**
   \brief add ili with one operand
 */
int
ad1ili(ILI_OP opc, int opn1)
{
  ILI newili;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = 0; /* cause some FREE ili have two opnds (target dep)*/
  newili.opnd[2] = 0;
  newili.opnd[3] = 0;
  newili.opnd[4] = 0;
  return addili(&newili);
}

/** \brief add an ili with one operand which has an alternate
 *
 * The routine is static -- only addili should create ALT ili.
 */
static int
ad1altili(ILI_OP opc, int opn1, int alt)
{
  ILI newili;
  int ilix;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = 0; /* cause some FREE ili have two opnds (target dep)*/
  newili.opnd[2] = 0;
  newili.opnd[3] = 0;
  newili.opnd[4] = 0;
  ilix = get_ili(&newili);
  if (ILI_ALT(ilix) == 0)
    ILI_ALT(ilix) = alt;
  return ilix;
}

/**
   \brief add ili with two operands
 */
int
ad2ili(ILI_OP opc, int opn1, int opn2)
{
  ILI newili;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = opn2;
  newili.opnd[2] = 0;
  newili.opnd[3] = 0;
  newili.opnd[4] = 0;
  return addili(&newili);
}

/** \brief Add an ili with two operands which has an alternate
 *
 * The routine isstatic -- only addili should create ALT ili.
 */
static int
ad2altili(ILI_OP opc, int opn1, int opn2, int alt)
{
  ILI newili;
  int ilix;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = opn2;
  newili.opnd[2] = 0;
  newili.opnd[3] = 0;
  newili.opnd[4] = 0;
  ilix = get_ili(&newili);
  if (ILI_ALT(ilix) == 0)
    ILI_ALT(ilix) = alt;
  return ilix;
}

/** \brief Add func call with 2 integer arguments returning integer value
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
static int
ad2func_int(ILI_OP opc, const char *name, int opn1, int opn2)
{
  int tmp, tmp1, tmp2;
  tmp1 = ad1ili(IL_NULL, 0);
  tmp1 = ad2ili(IL_ARGIR, opn2, tmp1);
  tmp2 = ad2ili(IL_ARGIR, opn1, tmp1);
  tmp = ad2ili(opc, _mkfunc(name), tmp2);
  return ad2ili(IL_DFRIR, tmp, IR_RETVAL);
}

/** \brief Add func call with 1 complex argument returning complex value
 *
 * Note that a double complex value will be presented as a packed argument,
 * i.e., a double complex vector of length 1
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
static int
ad1func_cmplx(ILI_OP opc, char *name, int opn1)
{
  int tmp, tmp1, tmp2;

  tmp1 = ad1ili(IL_NULL, 0);
  if (IL_RES(ILI_OPC(opn1)) == ILIA_CS) {
    tmp2 = ad3ili(IL_DACS, opn1, DP(0), tmp1);
    tmp = ad2ili(opc, _mkfunc(name), tmp2);
    return ad2ili(IL_DFRCS, tmp, CS_RETVAL);
  }
  tmp2 = ad3ili(IL_DACD, opn1, DP(0), tmp1);
  tmp = ad2ili(opc, _mkfunc(name), tmp2);
  return ad2ili(IL_DFRCD, tmp, CD_RETVAL);
}

/** \brief Add func call with 2 arguments returning complex value
 *
 * The 2 arguments could be double complex or the 1st argument is double
 * complex and the 2nd argument is integer.
 *
 * A double complex value will be presented as a packed argument, i.e.,
 * a double complex vector of length 1
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
static int
ad2func_cmplx(ILI_OP opc, char *name, int opn1, int opn2)
{
  int tmp, tmp1, tmp2;
  int ireg; /* for integer pow argument just in case */

#if !defined(TARGET_WIN)
  ireg = IR(0);
#else
  ireg = IR(1); /* positional on windows */
#endif
  tmp1 = ad1ili(IL_NULL, 0);
  switch (IL_RES(ILI_OPC(opn2))) {
  case ILIA_CS:
    tmp1 = ad3ili(IL_DACS, opn2, DP(1), tmp1);
    break;
  case ILIA_CD:
    tmp1 = ad3ili(IL_DACD, opn2, DP(1), tmp1);
    break;
  case ILIA_IR:
#if defined(TARGET_X8664)
    tmp1 = ad3ili(IL_DAIR, opn2, ireg, tmp1);
#else
    tmp1 = ad3ili(IL_ARGIR, opn2, tmp1, 0);
#endif
    break;
  case ILIA_KR:
#if defined(TARGET_X8664)
    tmp1 = ad3ili(IL_DAKR, opn2, ireg, tmp1);
#else
    tmp1 = ad3ili(IL_ARGKR, opn2, tmp1, 0);
#endif
    break;
  default:
    interr("ad2func_cmplx: illegal ILIA arg2", opn2, ERR_unused);
    tmp1 = ad1ili(IL_NULL, 0);
  }
  if (IL_RES(ILI_OPC(opn1)) == ILIA_CS) {
    tmp2 = ad3ili(IL_DACS, opn1, DP(0), tmp1);
    tmp = ad2ili(opc, _mkfunc(name), tmp2);
    return ad2ili(IL_DFRCS, tmp, CS_RETVAL);
  }
  tmp2 = ad3ili(IL_DACD, opn1, DP(0), tmp1);
  tmp = ad2ili(opc, _mkfunc(name), tmp2);
  return ad2ili(IL_DFRCD, tmp, CD_RETVAL);
}

/** \brief Add func call with 1 complex argument returning complex value
 *
 * The C ABI is used here.  A complex will be treated as if it's a struct
 * 2 parts. On x64, a complex double will be passed as 2 arguments; the
 * vector abi will pass the complex double packed in a single register or
 * memory unit.
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
static int
ad1func_cmplx_abi(ILI_OP opc, char *name, int opn1)
{
  int tmp, tmp1, tmp2;

  tmp1 = ad1ili(IL_NULL, 0);
  if (IL_RES(ILI_OPC(opn1)) == ILIA_CS) {
    tmp2 = ad3ili(IL_DACS, opn1, DP(0), tmp1);
    tmp = ad2ili(opc, _mkfunc(name), tmp2);
    return ad2ili(IL_DFRCS, tmp, CS_RETVAL);
  }
  tmp2 = ad3ili(IL_DACD, opn1, DP(0), tmp1);
  tmp = ad2ili(opc, _mkfunc(name), tmp2);
  return ad2ili(IL_DFRCD, tmp, CD_RETVAL);
}

/** \brief Add func call with 2 arguments returning complex value
 *
 * The 2 arguments could be double complex or the 1st argument is double
 * complex and the 2nd argument is integer.
 * The C ABI is used here.  A complex will be treated as if it's a struct
 * 2 parts. On x64, a complex double will be passed as 2 arguments; the
 * vector abi will pass the complex double packed in a single register or
 * memory unit.
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
static int
ad2func_cmplx_abi(ILI_OP opc, char *name, int opn1, int opn2)
{
  int tmp, tmp1, tmp2;
  int ireg; /* for integer pow argument just in case */

#if !defined(TARGET_WIN)
  ireg = IR(0);
#else
  ireg = IR(1); /* positional on windows */
#endif
  tmp1 = ad1ili(IL_NULL, 0);
  switch (IL_RES(ILI_OPC(opn2))) {
  case ILIA_CS:
    tmp1 = ad3ili(IL_DACS, opn2, DP(1), tmp1);
    break;
  case ILIA_CD:
    tmp1 = ad3ili(IL_DACD, opn2, DP(1), tmp1);
    break;
  case ILIA_IR:
#if defined(TARGET_X8664)
    tmp1 = ad3ili(IL_DAIR, opn2, ireg, tmp1);
#else
    tmp1 = ad3ili(IL_ARGIR, opn2, tmp1, 0);
#endif
    break;
  case ILIA_KR:
#if defined(TARGET_X8664)
    tmp1 = ad3ili(IL_DAKR, opn2, ireg, tmp1);
#else
    tmp1 = ad3ili(IL_ARGKR, opn2, tmp1, 0);
#endif
    break;
  default:
    interr("ad2func_cmplx: illegal ILIA arg2", opn2, ERR_unused);
    tmp1 = ad1ili(IL_NULL, 0);
  }
  if (IL_RES(ILI_OPC(opn1)) == ILIA_CS) {
    tmp2 = ad3ili(IL_DACS, opn1, DP(0), tmp1);
    tmp = ad2ili(opc, _mkfunc(name), tmp2);
    return ad2ili(IL_DFRCS, tmp, CS_RETVAL);
  }
  tmp2 = ad3ili(IL_DACD, opn1, DP(0), tmp1);
  tmp = ad2ili(opc, _mkfunc(name), tmp2);
  return ad2ili(IL_DFRCD, tmp, CD_RETVAL);
}

/** \brief Add func call with 1 complex argument returning complex value
 *
 * Assumes the new (as defined by make_math) naming scheme.  For passing
 * complex and returning types, we can either follow the C ABI which says
 * complex is the same as a struct of two parts; or, we can follow the vector
 * ABI which views a complex scalar as a vector complex vector of length 1,
 * i.e., the complex is 'packed'.  For now, we use the C abi'.
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
static int
ad1mathfunc_cmplx(MTH_FN fn, ILI_OP opc, int op1, DTYPE res_dt, DTYPE arg1_dt)
{
  char *fname;
  int ilix;

  fname = make_math(fn, NULL, 1, false, res_dt, 1, arg1_dt);
  if (!XBIT_VECTORABI_FOR_SCALAR)
    ilix = ad1func_cmplx_abi(IL_QJSR, fname, op1);
  else
    ilix = ad1func_cmplx(IL_QJSR, fname, op1);
  ilix = ad1altili(opc, op1, ilix);
  return ilix;
}

/** \brief Add func call with 2 complex arguments returning complex value
 *
 * Assumes the new (as defined by make_math) naming scheme.  For passing
 * complex and returning types, we can either follow the C ABI which says
 * complex is the same as a struct of two parts; or, we can follow the vector
 * ABI which views a complex scalar as a vector complex vector of length 1,
 * i.e., the complex is 'packed'.  For now, we use the 'vector abi'.
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
static int
ad2mathfunc_cmplx(MTH_FN fn, ILI_OP opc, int op1, int op2, DTYPE res_dt,
                  DTYPE arg1_dt, DTYPE arg2_dt)
{
  char *fname;
  int ilix;

  fname = make_math(fn, NULL, 1, false, res_dt, 2, arg1_dt, arg2_dt);
  if (!XBIT_VECTORABI_FOR_SCALAR)
    ilix = ad2func_cmplx_abi(IL_QJSR, fname, op1, op2);
  else
    ilix = ad2func_cmplx(IL_QJSR, fname, op1, op2);
  ilix = ad2altili(opc, op1, op2, ilix);
  return ilix;
}

/*
 * WARNING - the arguments to ad_func are in lexical order
 */
static int
ad_func(ILI_OP result_opc, ILI_OP call_opc, const char *func_name, int nargs, ...)
{
  va_list vargs;
  int rg;
  int irg;
  int frg;
  int func;
  int argl;
  int ilix;
  int i;
  struct {
    ILI_OP opc;
    int arg;
    int reg;
    int is_argili;
  } args[6];

  va_start(vargs, nargs);
#if DEBUG
  assert((size_t)nargs <= sizeof(args) / sizeof(args[0]),
         "iliutil.c:ad_func, increase the size of args[]",
         sizeof(args) / sizeof(args[0]), ERR_unused);
#endif
  rg = 0;
  irg = 0;
  frg = 0;
  argl = ad1ili(IL_NULL, 0);

  BZERO(args, char, sizeof(args));
  for (i = 0; i < nargs; i++) {
    args[i].arg = va_arg(vargs, int);
    if (IL_VECT(ILI_OPC(args[i].arg))) {
      args[i].opc = IL_GARG;
      args[i].is_argili = 1;
    } else
      switch (IL_RES(ILI_OPC(args[i].arg))) {
      case ILIA_AR:
#if defined(TARGET_X8664)
        args[i].opc = IL_DAAR;
        args[i].reg = ARG_IR(irg);
#else
        args[i].opc = IL_ARGAR;
        args[i].is_argili = 1;
#endif
        rg++;
        irg++;
        break;
      case ILIA_IR:
#if defined(TARGET_X8664)
        args[i].opc = IL_DAIR;
        args[i].reg = IR(irg);
#else
        args[i].opc = IL_ARGIR;
        args[i].is_argili = 1;
#endif
        rg++;
        irg++;
        break;
      case ILIA_SP:
        args[i].opc = IL_DASP;
        args[i].reg = SP(frg);
        rg++;
        frg++;
        break;
      case ILIA_DP:
        args[i].opc = IL_DADP;
        args[i].reg = DP(frg);
        rg++;
        frg++;
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case ILIA_QP:
        args[i].opc = IL_DAQP;
        args[i].reg = QP(frg);
        rg++;
        frg++;
        break;
#endif
      case ILIA_CS:
        args[i].opc = IL_DACS;
        args[i].reg = DP(frg);
        rg++;
        frg++;
        break;
      case ILIA_CD:
        args[i].opc = IL_DACD; /* assumed to be packed when passed */
        args[i].reg = DP(frg);
        rg++;
        frg++;
        break;
      case ILIA_KR:
#if defined(TARGET_X8664)
        args[i].opc = IL_DAKR;
        args[i].reg = IR(irg);
        rg++;
        irg++;
#else
        args[i].opc = IL_ARGKR;
        args[i].is_argili = 1;
        rg += 2;
#endif
        break;
#ifdef LONG_DOUBLE_FLOAT128
      case ILIA_FLOAT128:
        args[i].opc = IL_FLOAT128ARG;
        args[i].is_argili = 1;
        rg++;
        break;
#endif
      default:
        interr("ad_func: illegal arg", args[i].arg, ERR_Severe);
        args[i].opc = IL_ARGIR;
        args[i].is_argili = 1;
        break;
      }
#if defined(TARGET_WIN_X8664)
    irg = rg; /* on win64, register # is positional */
    frg = rg; /* on win64, register # is positional */
#endif
  }

  for (i = nargs - 1; i >= 0; i--) {
    if (!args[i].is_argili) {
      argl = ad3ili(args[i].opc, args[i].arg, args[i].reg, argl);
    } else {
      if (IL_VECT(ILI_OPC(args[i].arg))) {
        int arg_dtype = 0, dtype_slot, arg_nme = 0;
        switch (IL_TYPE(ILI_OPC(args[i].arg))) {
        case ILTY_CONS:
          arg_nme = 0;
          arg_dtype = DTYPEG(ILI_OPND(args[i].arg, 1));
          break;
        case ILTY_LOAD:
          arg_nme = ILI_OPND(args[i].arg, 2);
          arg_dtype = ILI_OPND(args[i].arg, 3);
          break;
        case ILTY_ARTH:
        case ILTY_OTHER:
          arg_nme = 0;
          dtype_slot = IL_OPRS(ILI_OPC(args[i].arg));
          arg_dtype = ILI_OPND(args[i].arg, dtype_slot);

          break;
        default:
          assert(false, "ad_func(): unhandled vect ILI type",
                 IL_TYPE(ILI_OPC(args[i].arg)), ERR_Fatal);
        }
        argl = ad4ili(args[i].opc, args[i].arg, argl, arg_dtype, arg_nme);
      } else
        /* the 3rd argument is for dttype for an IL_ARGAR */
        argl = ad3ili(args[i].opc, args[i].arg, argl, 0);
    }
  }

  func = _mkfunc(func_name);
  ilix = call_opc == IL_GJSR ? ad3ili(IL_GJSR, func, argl, 0)
                             : ad2ili(call_opc, func, argl);
  switch (result_opc) {
  case IL_NONE:
    break; /* no return value */
  case IL_DFRAR:
    ilix = ad2ili(result_opc, ilix, AR_RETVAL);
    break;
  case IL_DFRIR:
    ilix = ad2ili(result_opc, ilix, IR_RETVAL);
    break;
  case IL_DFRKR:
    ilix = ad2ili(result_opc, ilix, KR_RETVAL);
    break;
  case IL_DFRSP:
    ilix = ad2ili(result_opc, ilix, SP_RETVAL);
    break;
  case IL_DFRDP:
    ilix = ad2ili(result_opc, ilix, DP_RETVAL);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_DFRQP:
    ilix = ad2ili(result_opc, ilix, QP_RETVAL);
    break;
#endif
  case IL_DFRCS:
    ilix = ad2ili(result_opc, ilix, CS_RETVAL);
    break;
#ifdef IL_DFRSPX87
  case IL_DFRSPX87:
    ilix = ad1ili(result_opc, ilix);
    break;
#endif
#ifdef IL_DFRDPX87
  case IL_DFRDPX87:
    ilix = ad1ili(result_opc, ilix);
    break;
#endif
  default:
    interr("ad_func: illegal result_opc", result_opc, ERR_Severe);
  }
  va_end(vargs);
  return ilix;
}

#if defined(TARGET_X8664)
static char *
fmth_name(const char *root)
{
  static char bf[64];
  const char *suf;
  suf = "";
  if (TEST_MACH(MACH_AMD_GH)) {
    suf = "_gh";
  }
  sprintf(bf, "%s%s", root, suf);
  return bf;
}
#endif

/*
 * fast math naming convention:
 *    __g[sv][sdcz]_BASE[L]
 *   [sv]   - scalar vector
 *   [sdcz] - single/double/singlecomplex/doublecomplex
 *   [L]    - vector length, i.e., 2, 4, 8, 16
 */
char *
gnr_math(const char *root, int widthc, int typec, const char *oldname, int masked)
{
  static char bf[32];
/*
 * widthc  - 's' (scalar), 'v' (vector)
 *            N (vector length` N) (2, 4, 8, 16, ...)
 * typec   - 's' (single), 'd' (double),
 *           'c'( single complex), 'z' (double complex)
 * oldname -  old 'mth' name (only if scalar)
 */
#if defined(TARGET_OSX) || defined(TARGET_WIN)
  sprintf(bf, "%s", oldname);
#else
  if (widthc == 's' || widthc == 'v')
    sprintf(bf, "__g%c%c_%s", widthc, typec, root);
  else {
    const char *mk;
    mk = !masked ? "" : "_mask";
    sprintf(bf, "__g%c%c_%s%d%s", 'v', typec, root, widthc, mk);
  }
#endif
  return bf;
}

static char *
vect_math(MTH_FN fn, const char *root, int nargs, DTYPE vdt, int vopc, int vdt1,
          int vdt2, bool mask)
{
  int typec = 0;
  int num_elem;
  DTYPE vdt_mask = DT_NONE;
  int func;
  char *func_name;
  char oldname[32];

  /*
   * determine type/precision (single,double, ...) and
   * construct the names for the lowest common denominator architecture
   * on x86 - MACH_AMD_K8 or MACH_INTEL without SSE3
   */
  if (DTY(vdt) != TY_VECT) {
    interr("vect_math: dtype is not vector", vdt, ERR_Severe);
    vdt = get_vector_dtype(DT_DBLE, 2);
  }
  if (XBIT_NEW_MATH_NAMES && fn != MTH_mod) {
    /*
     * DTY(vdt+1) -- res_dt
     * DTY(vdt+2) -- vect_len
     */
    func_name =
        make_math_name(fn, DTyVecLength(vdt), mask, DTySeqTyElement(vdt));
  } else {
    switch (DTY(DTySeqTyElement(vdt))) {
    case TY_FLOAT:
      typec = 's';
      sprintf(oldname, "__fvs_%s", root);
      break;
    case TY_DBLE:
      typec = 'd';
      sprintf(oldname, "__fvd_%s", root);
      break;
    default:
      interr("vect_math: unexpected element dtype", DTySeqTyElement(vdt),
             ERR_Severe);
      typec = 'd';
      break;
    }
#if defined(TARGET_LINUX_X8664)
    if (XBIT_NEW_RELAXEDMATH) {
      switch (vopc) {
      case IL_VEXP:
        func_name = relaxed_math(root, 'v', typec, oldname);
        goto llvm_hk;
      case IL_VTAN:
      case IL_VPOW:
        /*
         * a vector double relaxed math version of pow & tan does not exist
         * so continue to selection of a fast or generic version.
         */
        if (typec != 'd') {
          func_name = relaxed_math(root, 'v', typec, oldname);
          goto llvm_hk;
        }
        break;
      }
    }
    switch (vopc) {
    case IL_VPOW:
    case IL_VEXP:
    case IL_VLOG:
    case IL_VATAN:
      func_name = gnr_math(root, DTyVecLength(vdt), typec, oldname, 0);
      break;
    default:
      func_name = fast_math(root, DTyVecLength(vdt), typec, oldname);
      break;
    }
  llvm_hk:;
#else
    func_name = gnr_math(root, DTyVecLength(vdt), typec, oldname, 0);
#endif
  }

  if (XBIT_NEW_MATH_NAMES && mask) {
    /* dtype of mask is int or int8 */
    num_elem = DTyVecLength(vdt);

    switch (DTySeqTyElement(vdt)) {
    case DT_FLOAT:
    case DT_INT:
      vdt_mask = DT_INT;
      break;
    case DT_DBLE:
    case DT_INT8:
      vdt_mask = DT_INT8;
      break;
    default:
      assert(0, "vect_math, unexpected dtype", DTySeqTyElement(vdt), ERR_Fatal);
    }

    vdt_mask = get_vector_dtype(vdt_mask, num_elem);
  }

  switch (nargs) {
  case 1:
    func = mk_prototype(func_name, "f pure", vdt, 1, vdt);
    break;
  case 2:
    if (vdt1 && vdt2)
      func = mk_prototype(func_name, "f pure", vdt, 2, vdt1, vdt2);
    else if (vdt_mask)
      func = mk_prototype(func_name, "f pure", vdt, 2, vdt, vdt_mask);
    else
      func = mk_prototype(func_name, "f pure", vdt, 2, vdt, vdt);
    break;
  case 3:
    if (vdt1 && vdt2)
      func = mk_prototype(func_name, "f pure", vdt, 3, vdt1, vdt2, vdt);
    else if (vdt_mask)
      func = mk_prototype(func_name, "f pure", vdt, 3, vdt, vdt, vdt_mask);
    else
      func = mk_prototype(func_name, "f pure", vdt, 3, vdt, vdt, vdt);
    break;
  default:
    interr("vect_math: unexpected number of args", nargs, ERR_Severe);
    func = mk_prototype(func_name, "f pure", vdt, 1, vdt);
    break;
  }
#ifdef AVXP
  // FIXME: looks like a bug: DTY(aVectorLength)?
  if ((typec == 's' && DTY((DTYPE)DTyVecLength(vdt)) >= 8) ||
      (typec == 'd' && DTY((DTYPE)DTyVecLength(vdt)) >= 4)) {
    AVXP(func, 1); /* inhibit veroupper */
  }
#endif
  return func_name;
}

#if !defined(TARGET_POWER)
/*
 * fast math naming convention:
 *    __f[sv][sd]_BASE[_suf]
 *
 * NOTE that the naming convention will become
 *    __g[sv][sdcz]_BASE
 *   [sv]   - scalar vector
 *   [sdcz] - single/double/singlecomplex/doublecomplex
 *   [L]    - vector length, i.e., 2, 4, 8, 16
 */
char *
fast_math(const char *root, int widthc, int typec, const char *oldname)
{
  /*
   * widthc  - width indicator: 's' (scalar), 'v' (vector),
   *           or a vector length (2, 4, 8, ..); if length
   *           is passed, 'v' is used.
   * typec   - 's' (single), 'd' (double)
   * oldname - old 'fastmath' name
   */
  static char bf[32];
  const char *suf;
  int avxp;
  suf = "";
  avxp = 0;
  if (XBIT(15, 0x4000000)) {
    /*
     * Use the old fastmath names.
     */
    if (TEST_MACH(MACH_AMD_GH)) {
      suf = "_gh";
    }
    sprintf(bf, "%s%s", oldname, suf);
  } else {
    const char *simdw;
    simdw = "";
    if ((typec == 's' && widthc == 8) || (typec == 'd' && widthc == 4))
      simdw = "_256";
    if (TEST_MACH(MACH_INTEL_SANDYBRIDGE)) {
      suf = "_vex";
      avxp = 1;
    } else if (TEST_MACH(MACH_AMD_BULLDOZER)) {
      if (TEST_FEATURE(FEATURE_FMA4))
        suf = "_fma4";
      else
        suf = "_vex"; /* -nofma */
      avxp = 1;
    } else if (TEST_FEATURE(FEATURE_SSE3)) {
      ;
    } else {
      /***  MACH_AMD_K8 or MACH_INTEL without SSE3  ***/
      sprintf(bf, "%s%s", oldname, suf);
      return bf;
    }
    if (widthc != 's' && widthc != 'v')
      widthc = 'v'; /* vector length is passed */
    sprintf(bf, "__f%c%c_%s%s%s", widthc, typec, root, suf, simdw);
  }
  return bf;
}
#else
/*
 * fast math naming convention:
 *    __g[sv][sdcz]_BASEL
 *   [sv]   - scalar vector
 *   [sdcz] - single/double/singlecomplex/doublecomplex
 *   [L]    - vector length, i.e., 2, 4, 8, 16
 */
char *
fast_math(const char *root, int widthc, int typec, const char *oldname)
{
  static char bf[32];
  /*
   * widthc  - 's' (scalar), 'v' (vector)
   *           N (vector length` N) (2, 4, 8, 16, ...)
   * typec   - 's' (single), 'd' (double),
   *           'c'( single complex), 'z' (double complex)
   * oldname - old 'mth' name (at this time,  not used)
   */

  return gnr_math(root, widthc, typec, oldname, 0);
}
#endif

/** \brief Called if and only if there exists the possiblity of a
 * -Mfprelaxed=intrinsic version of the routine.
 */
char *
relaxed_math(const char *root, int widthc, int typec, const char *oldname)
{
  static char bf[32];
  /*
   * widthc - width indicator: 's' (scalar), 'v' (vector)
   */
  const char *suf;
  int avxp;

  avxp = 0;
  if (!XBIT_NEW_RELAXEDMATH)
    return fast_math(root, widthc, typec, oldname);
  suf = "";
  if (TEST_MACH(MACH_INTEL_SANDYBRIDGE)) {
    suf = "_vex";
    if (widthc == 'v' && !TEST_FEATURE(FEATURE_SIMD128)) {
      suf = "_vex_256";
    }
    avxp = 1;
  } else if (TEST_MACH(MACH_AMD_BULLDOZER)) {
    if (TEST_FEATURE(FEATURE_FMA4)) {
      suf = "_fma4";
      if (widthc == 'v' && !TEST_FEATURE(FEATURE_SIMD128)) {
        suf = "_fma4_256";
      }
    } else {
      suf = "_vex"; /* -nofma */
      if (widthc == 'v' && !TEST_FEATURE(FEATURE_SIMD128)) {
        suf = "_vex_256";
      }
    }
    avxp = 1;
  } else
    return fast_math(root, widthc, typec, oldname);
  sprintf(bf, "__r%c%c_%s%s", widthc, typec, root, suf);
  return bf;
}

/** \brief Add func call with 2 integer*8 arguments returning integer*8 value
 *
 * \param opc must be a function call ili opcode: QJSR, JSR
 */
int
ad2func_kint(ILI_OP opc, const char *name, int opn1, int opn2)
{
  int tmp, tmp1, tmp2;
  tmp1 = ad1ili(IL_NULL, 0);
  tmp1 = ad2ili(IL_ARGKR, opn2, tmp1);
  tmp2 = ad2ili(IL_ARGKR, opn1, tmp1);
  tmp = ad2ili(opc, _mkfunc(name), tmp2);
  return ad2ili(IL_DFRKR, tmp, KR_RETVAL);
}

int
ad3ili(ILI_OP opc, int opn1, int opn2, int opn3)
{
  ILI newili;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = opn2;
  newili.opnd[2] = opn3;
  newili.opnd[3] = 0;
  newili.opnd[4] = 0;
  return addili(&newili);
}

/** \brief Add an ili with three operands which has an alternate
 *
 * The routine isstatic -- only addili should create ALT ili.
 */
static int
ad3altili(ILI_OP opc, int opn1, int opn2, int opn3, int alt)
{
  ILI newili;
  int ilix;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = opn2;
  newili.opnd[2] = opn3;
  newili.opnd[3] = 0;
  newili.opnd[4] = 0;
  ilix = get_ili(&newili);
  if (ILI_ALT(ilix) == 0)
    ILI_ALT(ilix) = alt;
  return ilix;
}

int
ad4ili(ILI_OP opc, int opn1, int opn2, int opn3, int opn4)
{
  ILI newili;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = opn2;
  newili.opnd[2] = opn3;
  newili.opnd[3] = opn4;
  newili.opnd[4] = 0;
  return addili(&newili);
}

static int
ad4altili(ILI_OP opc, int opn1, int opn2, int opn3, int opn4, int alt)
{
  ILI newili;
  int ilix;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = opn2;
  newili.opnd[2] = opn3;
  newili.opnd[3] = opn4;
  newili.opnd[4] = 0;
  ilix = get_ili(&newili);
  if (ILI_ALT(ilix) == 0)
    ILI_ALT(ilix) = alt;
  return ilix;
}

int
ad5ili(ILI_OP opc, int opn1, int opn2, int opn3, int opn4, int opn5)
{
  ILI newili;

  newili.opc = opc;
  newili.opnd[0] = opn1;
  newili.opnd[1] = opn2;
  newili.opnd[2] = opn3;
  newili.opnd[3] = opn4;
  newili.opnd[4] = opn5;
  return addili(&newili);
}

int
ad_icon(INT val)
{
  static INT ival[] = {0, 0};
  ival[1] = val;
  return ad1ili(IL_ICON, getcon(ival, DT_INT));
}

int
ad_kcon(INT m32, INT l32)
{
  INT ival[2];

  ival[0] = m32;
  ival[1] = l32;
  return ad1ili(IL_KCON, getcon(ival, DT_INT8));
}

int
ad_kconi(ISZ_T v)
{
  INT num[2];
  int s;

  ISZ_2_INT64(v, num);
  s = getcon(num, DT_INT8);
  return ad1ili(IL_KCON, s);
}

int
ad_aconi(ISZ_T val)
{
  return ad1ili(IL_ACON, get_acon(SPTR_NULL, val));
}

int
ad_aconk(INT m32, INT l32)
{
  INT ival[2];
  ISZ_T v;

  ival[0] = m32;
  ival[1] = l32;
  INT64_2_ISZ(ival, v);
  return ad1ili(IL_ACON, get_acon(SPTR_NULL, v));
}

int
ad_acon(SPTR sym, ISZ_T val)
{
  return ad1ili(IL_ACON, get_acon(sym, val));
}

int
ad_cse(int ilix)
{
  switch (IL_RES(ILI_OPC(ilix))) {
  case ILIA_IR:
    ilix = ad1ili(IL_CSEIR, ilix);
    break;
  case ILIA_KR:
    ilix = ad1ili(IL_CSEKR, ilix);
    break;
  case ILIA_AR:
    ilix = ad1ili(IL_CSEAR, ilix);
    break;
  case ILIA_SP:
    ilix = ad1ili(IL_CSESP, ilix);
    break;
  case ILIA_DP:
    ilix = ad1ili(IL_CSEDP, ilix);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case ILIA_QP:
    ilix = ad1ili(IL_CSEQP, ilix);
    break;
#endif
#ifdef ILIA_CS
  case ILIA_CS:
    ilix = ad1ili(IL_CSECS, ilix);
    break;
  case ILIA_CD:
    ilix = ad1ili(IL_CSECD, ilix);
    break;
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case ILIA_FLOAT128:
    ilix = ad1ili(IL_FLOAT128CSE, ilix);
    break;
#endif
  case ILIA_LNK:
    if (ili_get_vect_dtype(ilix)) {
      ilix = ad1ili(IL_CSE, ilix);
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    interr("ad_cse: bad IL_RES", ilix, ERR_Severe);
    break;
  }
  return ilix;
}

int
ad_load(int stx)
{
  int nme;
  int base;
  int load;

  nme = ILI_OPND(stx, 3);
  base = ILI_OPND(stx, 2);
  load = 0;
  switch (ILI_OPC(stx)) {
  case IL_ST:
    load = ad3ili(IL_LD, base, nme, ILI_OPND(stx, 4));
    break;
  case IL_STKR:
    load = ad3ili(IL_LDKR, base, nme, MSZ_I8);
    break;
  case IL_STA:
    load = ad2ili(IL_LDA, base, nme);
    break;
  case IL_STSP:
    load = ad3ili(IL_LDSP, base, nme, MSZ_F4);
    break;
  case IL_STDP:
    load = ad3ili(IL_LDDP, base, nme, MSZ_F8);
    break;
  case IL_STSCMPLX:
    load = ad3ili(IL_LDSCMPLX, base, nme, MSZ_F8);
    break;
  case IL_STDCMPLX:
    load = ad3ili(IL_LDDCMPLX, base, nme, MSZ_F16);
    break;
  case IL_STQ:
    load = ad3ili(IL_LDQ, base, nme, MSZ_F16);
    break;
  case IL_ST256:
    load = ad3ili(IL_LD256, base, nme, MSZ_F32);
    break;
  case IL_VST:
    load = ad3ili(IL_VLD, base, nme, ILI_OPND(stx, 4));
    break;
  case IL_VSTU:
    load = ad3ili(IL_VLDU, base, nme, ILI_OPND(stx, 4));
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ST:
    load = ad3ili(IL_FLOAT128LD, base, nme, MSZ_F16);
    break;
#endif /* LONG_DOUBLE_FLOAT128 */
  default:
    break;
  }
  return load;
}

int
ad_free(int ilix)
{
  ILI_OP opc;
  ILIA_RESULT r = IL_RES(ILI_OPC(ilix));
  switch (r) {
  default:
    interr("ad_free: not yet implemented for this result type", r, ERR_Fatal);
    return 0;
  case ILIA_IR:
    opc = IL_FREEIR;
    break;
  case ILIA_AR:
    opc = IL_FREEAR;
    break;
  case ILIA_SP:
    opc = IL_FREESP;
    break;
  case ILIA_DP:
    opc = IL_FREEDP;
    break;
#ifdef ILIA_CS
  case ILIA_CS:
    opc = IL_FREECS;
    break;
  case ILIA_CD:
    opc = IL_FREECD;
    break;
#endif
  case ILIA_KR:
    opc = IL_FREEKR;
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case ILIA_FLOAT128:
    opc = IL_FLOAT128FREE;
    break;
#endif /* LONG_DOUBLE_FLOAT128 */
  }
  return ad1ili(opc, ilix);
}

int
ili_opnd(int ilix, int n)
{
  ilix = ILI_OPND(ilix, n);

  if (ILI_OPC(ilix) == IL_CSEAR || ILI_OPC(ilix) == IL_CSEIR ||
      ILI_OPC(ilix) == IL_CSEKR)
    return ILI_OPND(ilix, 1);
  return ilix;
} /* ili_opnd */

/** \brief Given a store opc, return a corresponding load opc.
 */
ILI_OP
ldopc_from_stopc(ILI_OP stopc)
{
  ILI_OP ldopc = IL_NONE;
  switch (stopc) {
  case IL_ST:
    ldopc = IL_LD;
    break;
  case IL_STKR:
    ldopc = IL_LDKR;
    break;
  case IL_STSP:
    ldopc = IL_LDSP;
    break;
  case IL_STDP:
    ldopc = IL_LDDP;
    break;
  case IL_STSCMPLX:
    ldopc = IL_LDSCMPLX;
    break;
  case IL_STDCMPLX:
    ldopc = IL_LDDCMPLX;
    break;
  case IL_STQ:
    ldopc = IL_LDQ;
    break;
  case IL_ST256:
    ldopc = IL_LD256;
    break;
  case IL_VST:
    ldopc = IL_VLD;
    break;
  case IL_VSTU:
    ldopc = IL_VLDU;
    break;
  case IL_STA:
    ldopc = IL_LDA;
    break;
#ifdef IL_DOUBLEDOUBLEST
  case IL_DOUBLEDOUBLEST:
    ldopc = IL_DOUBLEDOUBLELD;
    break;
#endif
  default:
    break;
  }
  return ldopc;
}

void
ldst_msz(DTYPE dtype, ILI_OP *ld, ILI_OP *st, MSZ *siz)
{
  *ld = IL_LD;
  *st = IL_ST;

  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_CHAR:
    *siz = MSZ_SBYTE;
    break;
  case TY_SINT:
  case TY_SLOG:
  case TY_NCHAR:
    *siz = MSZ_SHWORD;
    break;
  case TY_FLOAT:
    *siz = MSZ_F4;
    *ld = IL_LDSP;
    *st = IL_STSP;
    break;
  case TY_CMPLX:
    *siz = MSZ_F8;
    *ld = IL_LDSCMPLX;
    *st = IL_STSCMPLX;
    return;
  case TY_INT8:
  case TY_LOG8:
    *siz = MSZ_I8;
    *ld = IL_LDKR;
    *st = IL_STKR;
    return;
  case TY_QUAD:
  case TY_DBLE:
    *siz = MSZ_F8;
    *ld = IL_LDDP;
    *st = IL_STDP;
    break;
  case TY_DCMPLX:
    *siz = MSZ_F16;
    *ld = IL_LDDCMPLX;
    *st = IL_STDCMPLX;
    break;
  case TY_PTR:
    *siz = MSZ_PTR;
    *ld = IL_LDA;
    *st = IL_STA;
    break;
  case TY_STRUCT:
    switch (DTyAlgTySize(dtype)) {
    case 1:
      *siz = MSZ_BYTE;
      break;
    case 2:
      *siz = MSZ_SHWORD;
      break;
    case 8:
      *siz = MSZ_F8;
      break;
    case 16:
      *siz = MSZ_F16;
      break;
    case 4:
    default:
      *siz = MSZ_WORD;
      break;
    }
    break;
  case TY_INT:
  case TY_LOG:
  default:
    *siz = MSZ_WORD;
    break;
  }
  switch (*siz) {
  case MSZ_FWORD:
    *ld = IL_LDSP;
    *st = IL_STSP;
    break;
  case MSZ_DFLWORD:
    *ld = IL_LDDP;
    *st = IL_STDP;
    break;
  default:
    break;
  }
}

#if defined(TARGET_WIN_X8664)
/** \brief Insert an ARGRSRV ili between the last register arg and the first
 * memory arg or null.
 *
 * \param ilip represents a procedure
 */
static void
insert_argrsrv(ILI *ilip)
{
  int arg_ilix;
  int prev_ilix;
  ILI_OP opc;

  opc = ilip->opc;
  if (opc == IL_GJSR)
    return;
  if (opc == IL_GJSRA)
    return;
#if DEBUG
  if (opc != IL_QJSR && opc != IL_JSR && opc != IL_JSRA)
    interr("insert_argrsrv: unexpected proc", opc, ERR_Severe);
#endif

  /* do not insert IL_ARGRSRV for routines that depend on the
   * organization of the stack (i.e., _mp routines)
   */
  if (opc != IL_JSRA && NOPADG(ilip->opnd[0]))
    return;

  prev_ilix = 0;
  arg_ilix = ilip->opnd[1];

  /* no args */
  if (ILI_OPC(arg_ilix) == IL_NULL) {
    ilip->opnd[1] = ad2ili(IL_ARGRSRV, MR_MAX_ARGRSRV, arg_ilix);
    return;
  }

  /* one or more args */
  while (ILI_OPC(arg_ilix) != IL_NULL && ILI_OPC(arg_ilix) != IL_ARGRSRV) {
    switch (ILI_OPC(arg_ilix)) {
    case IL_DAIR:
    case IL_DASP:
    case IL_DASPSP:
    case IL_DADP:
#ifdef IL_DA128
    case IL_DA128:
#endif
#ifdef IL_DA256
    case IL_DA256:
#endif
    case IL_DACD:
    case IL_DACS:
    case IL_DAAR:
    case IL_DAKR:
    case IL_PSARG:
    case IL_PDARG:
      prev_ilix = arg_ilix;
      arg_ilix = ILI_OPND(arg_ilix, 3);
      if (ILI_OPC(arg_ilix) == IL_NULL) {
        int argrsrv = ad2ili(IL_ARGRSRV, MR_MAX_ARGRSRV, arg_ilix);
        ILI_OPND(prev_ilix, 3) = argrsrv;
      }
      break;
    case IL_ARGIR:
    case IL_ARGSP:
    case IL_ARGDP:
#ifdef TARGET_SUPPORTS_QUADFP
    case IL_ARGQP:
#endif
    case IL_ARGAR:
    case IL_ARGKR:
#ifdef LONG_DOUBLE_FLOAT128
    case IL_FLOAT128ARG:
#endif
      /* this path taken only with the first mem arg, after reg args */
      arg_ilix = ad2ili(IL_ARGRSRV, MR_MAX_ARGRSRV, arg_ilix);
#if DEBUG
      assert(prev_ilix, "insert_argrsrv: no reg args", ilip->opnd[0],
             ERR_Severe);
#endif
      ILI_OPND(prev_ilix, 3) = arg_ilix;
      break;
    default:
      interr("insert_argrsrv: unexpected arg ili", ILI_OPC(arg_ilix),
             ERR_Severe);
    }
  }
}
#endif

/** \brief Return i such that 2**i > |n| >= 2**(i+1)
 *
 * Use reciprocal division to compute n/d where d is a compile-time constant.
 * See PLDI '94, "Division by invariant integers using multiplication"
 * by Granlund and Montgomery.
 */
static int
lg(unsigned d)
{
  int i = 0;
  unsigned int twoi = 1, dd;
  dd = d;
  while (dd > twoi) {
    ++i;
    twoi = twoi << 1;
  }
  return i;
} /* lg */

static int
lg64(DBLUINT64 d)
{
  int i = 0;
  DBLUINT64 twoi = {0x0, 0x1};
  DBLUINT64 dd;

  dd[0] = d[0];
  dd[1] = d[1];
  while (ucmp64(dd, twoi) > 0) {
    i++;
    ushf64(twoi, 1, twoi);
  }
  return i;
}

/*
 * find reciprocal approximation, shift distance,
 * set 'alt' for which code alternative to generate.
 * We compute 2**(N+l)/d as ((2**N * (2**(l)-d))/d) + 2**N
 * This since l can be as big as N, so we prevent overflow in the numerator
 * (don't want 2**(2*N))
 */
static int shpost;
static DBLINT64 mrecip;
static DBLINT64 twon, twonm1;

void
choose_multiplier(int N, unsigned dd, int prec)
{
  DBLINT64 mlow, mhigh, d, mlow2, mhigh2;
  int l;

  l = lg(dd); /* ceiling(lg(d)) */
  shpost = l;

  d[0] = 0;
  d[1] = dd; /* DBLINT64 copy of d */

  twon[0] = 0; /* twon is 2**N, used for comparisons */
  twon[1] = 1;
  shf64(twon, N, twon);
  twonm1[0] = 0; /* twonm1 is 2**(N-1), used for comparisons */
  twonm1[1] = 1;
  shf64(twonm1, N - 1, twonm1);

  mlow[0] = 0; /* mlow = 1 */
  mlow[1] = 1;
  mhigh[0] = 0; /* mhigh = 1 */
  mhigh[1] = 1;
  shf64(mlow, l, mlow);              /* mlow = 2**l */
  shf64(mhigh, l + N - prec, mhigh); /* mhigh = 2**(N+l-prec) */
  sub64(mlow, d, mlow);              /* mlow = 2**l - d */
  mul64(mlow, twon, mlow);           /* mlow = 2**N * (2**l - d) */
  add64(mhigh, mlow, mhigh);         /* 2**N * (2**l - d + 2**(l-prec)) */
  div64(mlow, d, mlow);              /* mlow = (2**N * (2**l - d))/d */
  div64(mhigh, d, mhigh);            /* (2**N * (2**l - d + 2**(l-prec))) */
  add64(mlow, twon, mlow);           /* mlow = (2**N * 2**l)/d */
  add64(mhigh, twon, mhigh);         /* (2**N * (2**l + 2**(l-prec)))/d */

  /* here, mlow = floor(2**(N+l)/d); mhigh = floor(2**(N+l-prec)/d) */

  while (shpost > 0) {
    shf64(mlow, -1, mlow2);
    shf64(mhigh, -1, mhigh2);
    if (cmp64(mlow2, mhigh2) >= 0)
      break;
    mlow[0] = mlow2[0];
    mlow[1] = mlow2[1];
    mhigh[0] = mhigh2[0];
    mhigh[1] = mhigh2[1];
    --shpost;
  }
  mrecip[0] = mhigh[0];
  mrecip[1] = mhigh[1];
} /* choose_multiplier */

static INT128 mrecip_128;
static INT128 twon_128, twonm1_128;

void
choose_multiplier_64(int N, DBLUINT64 dd, int prec)
{
  INT128 mlow, mhigh, d, mlow2, mhigh2;
  int l;

  l = lg64(dd); /* ceiling(lg(d)) */
  shpost = l;

  d[0] = 0;
  d[1] = 0;
  d[2] = dd[0];
  d[3] = dd[1]; /* DBLINT64 copy of d */

  /* twon is 2**N, used for comparisons */
  twon_128[0] = 0;
  twon_128[1] = 0;
  twon_128[2] = 0;
  twon_128[3] = 1;

  shf128(twon_128, N, twon_128);

  twonm1_128[0] = 0; /* twonm1 is 2**(N-1), used for comparisons */
  twonm1_128[1] = 0;
  twonm1_128[2] = 0;
  twonm1_128[3] = 1;
  shf128(twonm1_128, N - 1, twonm1_128);

  mlow[0] = 0; /* mlow = 1 */
  mlow[1] = 0;
  mlow[2] = 0;
  mlow[3] = 1;

  mhigh[0] = 0; /* mhigh = 1 */
  mhigh[1] = 0;
  mhigh[2] = 0;
  mhigh[3] = 1;

  shf128(mlow, l, mlow);              /* mlow = 2**l */
  shf128(mhigh, l + N - prec, mhigh); /* mhigh = 2**(N+l-prec) */
  sub128(mlow, d, mlow);              /* mlow = 2**l - d */
  mul128l(mlow, twon_128, mlow);      /* mlow = 2**N * (2**l - d) */
  add128(mhigh, mlow, mhigh);         /* 2**N * (2**l - d + 2**(l-prec)) */
  div128(mlow, d, mlow);              /* mlow = (2**N * (2**l - d))/d */
  div128(mhigh, d, mhigh);            /* (2**N * (2**l - d + 2**(l-prec))) */
  add128(mlow, twon_128, mlow);       /* mlow = (2**N * 2**l)/d */
  add128(mhigh, twon_128, mhigh);     /* (2**N * (2**l + 2**(l-prec)))/d */

  /* here, mlow = floor(2**(N+l)/d); mhigh = floor(2**(N+l-prec)/d) */

  while (shpost > 0) {
    shf128(mlow, -1, mlow2);
    shf128(mhigh, -1, mhigh2);

    if (cmp128(mlow2, mhigh2) >= 0)
      break;

    mlow[0] = mlow2[0];
    mlow[1] = mlow2[1];
    mlow[2] = mlow2[2];
    mlow[3] = mlow2[3];

    mhigh[0] = mhigh2[0];
    mhigh[1] = mhigh2[1];
    mhigh[2] = mhigh2[2];
    mhigh[3] = mhigh2[3];
    --shpost;
  }

  mrecip_128[0] = mhigh[0];
  mrecip_128[1] = mhigh[1];
  mrecip_128[2] = mhigh[2];
  mrecip_128[3] = mhigh[3];
} /* choose_multiplier_64 */

static int
ICON(INT v)
{
  INT r[2], ilix, recipsym;
  r[0] = 0;
  r[1] = v;
  recipsym = getcon(r, DT_INT);
  ilix = ad1ili(IL_ICON, recipsym);
  return ilix;
} /* ICON */

#ifdef FLANG2_ILIUTIL_UNUSED
static int
KCON(INT v)
{
  INT r[2], ilix, recipsym;
  r[0] = 0;
  r[1] = v;
  recipsym = getcon(r, DT_INT8);
  ilix = ad1ili(IL_KCON, recipsym);
  return ilix;
} /* KCON */
#endif

static int
MULSH(int ilix, int iliy)
{
  /* copied from the intrinsic function __muln */
  int t1, t2, t3;

  t1 = ad2ili(IL_KMUL, ilix, iliy);
  t2 = ICON(32);
  t3 = ad2ili(IL_KURSHIFT, t1, t2);
  return t3;
} /* MULSH */

static int
KMULSH(int ilix, int iliy)
{
  /* copied from the intrinsic function __muln */
  int t1;

  t1 = ad2ili(IL_KMULH, ilix, iliy);
  return t1;
} /* MULSH */

static int
KMULUH(int ilix, int iliy)
{
  int t1;

  t1 = ad2ili(IL_UKMULH, ilix, iliy);
  return t1;
} /* KMULUH */

static int
MULUH(int ilix, int iliy)
{
  /* copied from the intrinsic function __muln */
  int t1, t2, t3;

  t1 = ad2ili(IL_UKMUL, ilix, iliy);
  t2 = ICON(32);
  t3 = ad2ili(IL_KURSHIFT, t1, t2);

  return t3;
} /* MULUH */

#ifdef FLANG2_ILIUTIL_UNUSED
static int
MULU(int ilix, int iliy)
{
  /* copied from the intrinsic function __muln */
  int t1, t2, t3, t4, t5, t6, t7, t8, t9;

  t1 = ad2ili(IL_UIMUL, ilix, iliy);

  return t1;
} /* MULU */
#endif

static int
MUL(int ilix, int iliy)
{
  /* copied from the intrinsic function __muln */
  int t1;

  t1 = ad2ili(IL_IMUL, ilix, iliy);

  return t1;
} /* MUL */

static int
KMUL(int ilix, int iliy)
{
  /* copied from the intrinsic function __muln */
  int t1;

  t1 = ad2ili(IL_KMUL, ilix, iliy);

  return t1;
} /* MUL */

static int
XSIGN(int ilix, int N)
{
  int t, t1;
  t1 = ICON(N - 1);
  t = ad2ili(IL_ARSHIFT, ilix, t1); /* XSIGN(n) in the paper */
  return t;
} /* XSIGN */

static int
KXSIGN(int ilix, int N)
{
  int t, t1;
  t1 = ICON(N - 1);
  t = ad2ili(IL_KARSHIFT, ilix, t1); /* XSIGN(n) in the paper */
  return t;
} /* XSIGN */

#ifdef FLANG2_ILIUTIL_UNUSED
static void
mod_decompose(int d, int *d0, int *k)
{
  int i;

  i = 1;

  do {
    if ((d >> i) & 0x1) {
      *d0 = d >> i;
      *k = i;
      return;
    }
    i++;
  } while (i <= 32);
}

/* From the book "Hacker's Delight", Henry S. Warren */
static int
test_mod_zero(int n, int d, int sgnd, int cc)
{
  int d0, k;
  int M, N, fl;
  int q0, t0, t1, t2, t3;

  N = 32;

  if (!sgnd) {
    if (d & 0x1) { /* divisor is odd */
      return 0;
    } else {
      mod_decompose(d, &d0, &k);
      choose_multiplier(n, d0, N);
      fl = 0xfffffffe / d;
      t1 = ICON(k);
      t0 = KCON(mrecip[1]);
      t2 = ad1ili(IL_IKMV, n);
      q0 = MULU(t2, t0);
      q0 = ad1ili(IL_KIMV, q0);
      q0 = ad2ili(IL_ROTR, q0, t1);
      t1 = ICON(fl);
      t1 = ad3ili(IL_ICMP, q0, t1, (cc == CC_EQ) ? CC_LE : CC_GT);
      return t1;
    }

  } else {
    return 0;
  }
}
#endif

static int
reciprocal_division(int n, INT divisor, int sgnd)
{
  int N;
  int t1, t2, q0, q3, q;
  unsigned udiv;

  /* edge case, doesn't work */
  if (divisor == 0)
    return 0;
  if (divisor == (int)(0x80000000))
    return 0;
  if (!sgnd && divisor < 0)
    return 0;
  /* cases where divisor == 1 and divisor == 2**N are already handled */

  N = 32; /* hopefully, we can determine when 16 bits are enough */

  if (sgnd && divisor < 0) {
    udiv = -divisor;
  } else {
    udiv = divisor;
  }

  if (sgnd) {
    choose_multiplier(N, udiv, N - 1);
  } else {
    choose_multiplier(N, udiv, N);
  }

  if (sgnd) {
    if (cmp64(mrecip, twonm1) < 0) {
      /* m < 2**(N-1) */
      t1 = ad1ili(IL_IKMV, n);
      q0 = MULSH(ad_kcon(mrecip[0], mrecip[1]), t1);
      q0 = ad1ili(IL_KIMV, q0);
      if (shpost > 0) {
        t1 = ICON(shpost);               /* SRA */
        q0 = ad2ili(IL_ARSHIFT, q0, t1); /* SRA */
      }
      q3 = XSIGN(n, N);
      q = ad2ili(IL_ISUB, q0, q3);
    } else {
      sub64(mrecip, twon, mrecip);
      t1 = ad1ili(IL_IKMV, n);
      q0 = MULSH(ad_kcon(mrecip[0], mrecip[1]), t1);
      q0 = ad1ili(IL_KIMV, q0);
      q0 = ad2ili(IL_IADD, n, q0);
      if (shpost > 0) {
        t1 = ICON(shpost);
        q0 = ad2ili(IL_ARSHIFT, q0, t1); /* SRA */
      }
      q3 = XSIGN(n, N);
      q = ad2ili(IL_ISUB, q0, q3);
    }
    if (divisor < 0) {
      q = ad1ili(IL_INEG, q);
    }
  } else {
    DBLINT64 twol;
    int shpre = 0;
    DBLINT64 one_64 = {0x0, 0x1};
    DBLINT64 divisor64 = {0x0, (int)udiv};
    const int l = lg(udiv);

    shf64(one_64, l, twol);
    if (cmp64(divisor64, twol) == 0) {
      /* d == 2**l, where: 'd' is the divisor, 'l' is log2(d) */
      return ad2ili(IL_URSHIFT, n, ICON(l)); /* Return n>>l */
    }

    if (cmp64(mrecip, twon) >= 0) {
      /* loop until udiv is odd */
      shpre = 0;
      while ((udiv & 0x01) == 0) {
        ++shpre;
        udiv = udiv >> 1;
      }
      if (shpre) {
        choose_multiplier(N, udiv, N - shpre);
      }
    }
    if (cmp64(mrecip, twon) >= 0) {
      /* shpre must be zero */
      sub64(mrecip, twon, mrecip);
      t1 = ad1ili(IL_UIKMV, n);
      t1 = MULUH(ad_kcon(mrecip[0], mrecip[1]), t1);
      t1 = ad1ili(IL_KIMV, t1);
      q0 = ad2ili(IL_ISUB, n, t1);
      t2 = ICON(1);
      q0 = ad2ili(IL_URSHIFT, q0, t2);
      q = ad2ili(IL_UIADD, t1, q0);
      if (shpost > 1) {
        t1 = ICON(shpost - 1);
        q = ad2ili(IL_URSHIFT, q, t1);
      }
    } else {
      q = n;
      if (shpre > 0) {
        t1 = ICON(shpre);
        q = ad2ili(IL_URSHIFT, n, t1);
      }
      t1 = ad1ili(IL_UIKMV, q);
      q = MULUH(ad_kcon(mrecip[0], mrecip[1]), t1);
      q = ad1ili(IL_KIMV, q);
      if (shpost > 0) {
        t1 = ICON(shpost);
        q = ad2ili(IL_URSHIFT, q, t1);
      }
    }
  }
  return q;
}

static int
reciprocal_division_64(int n, DBLINT64 divisor, int sgnd)
{

  /* TBD: 64-bit divides */

  int N;
  int t1, t2, q0, q3, q;
  /*unsigned udiv;*/
  DBLUINT64 udiv;
  DBLINT64 tmp_64;
  static DBLINT64 zero_64 = {0, 0}, one_64 = {0x0, 0x1};
  static DBLINT64 min_neg_64 = {(INT)0x80000000, 0x0};

  /* edge case, doesn't work */
  if (cmp64(divisor, zero_64) == 0)
    return 0;
  if (cmp64(divisor, min_neg_64) == 0)
    return 0;
  if (!sgnd && (cmp64(divisor, zero_64) < 0))
    return 0;

  /* cases where divisor == 1 and divisor == 2**N are already handled */

  N = 64; /* hopefully, we can determine when 16 bits are enough */

  if (sgnd && (cmp64(divisor, zero_64) < 0)) {
    /*udiv = -divisor; */
    neg64(divisor, (INT *)udiv);
  } else {
    /*udiv = divisor; */
    udiv[0] = divisor[0];
    udiv[1] = divisor[1];
  }
  if (sgnd) {
    choose_multiplier_64(N, udiv, N - 1);
    /*choose_multiplier( N, udiv[1], N-1 ); */
  } else {
    choose_multiplier_64(N, udiv, N);
    /*choose_multiplier( N, udiv[1], N );*/
  }

  if (sgnd) {
    if (cmp128(mrecip_128, twonm1_128) < 0) {
      /* m < 2**(N-1) */
      q0 = KMULSH(ad_kcon(mrecip_128[2], mrecip_128[3]), n);
      if (shpost > 0) {
        t1 = ICON(shpost);                /* SRA */
        q0 = ad2ili(IL_KARSHIFT, q0, t1); /* SRA */
      }
      q3 = KXSIGN(n, N);
      q = ad2ili(IL_KSUB, q0, q3);
    } else {
      sub128(mrecip_128, twon_128, mrecip_128);
      q0 = KMULSH(ad_kcon(mrecip_128[2], mrecip_128[3]), n);
      q0 = ad2ili(IL_KADD, n, q0);
      if (shpost > 0) {
        t1 = ICON(shpost);
        q0 = ad2ili(IL_KARSHIFT, q0, t1);
      }
      q3 = KXSIGN(n, N);
      q = ad2ili(IL_KSUB, q0, q3);
    }
    if (cmp64(divisor, zero_64) < 0) {
      q = ad1ili(IL_KNEG, q);
    }
  } else {
    DBLINT64 twol;
    int shpre = 0;
    const int l = lg64(udiv);

    shf64(one_64, l, twol);
    if (cmp64(divisor, twol) == 0) {
      /* d == 2**l, where: 'd' is the divisor, 'l' is log2(d) */
      return ad2ili(IL_KURSHIFT, n, ICON(l)); /* Return n>>l */
    }

    if (cmp128(mrecip_128, twon_128) >= 0) {
      /* loop until udiv is odd */
      shpre = 0;
      while (and64((INT *)udiv, one_64, tmp_64),
             (cmp64(tmp_64, zero_64) == 0)) {
        ++shpre;
        ushf64(udiv, -1, udiv);
      }
      if (shpre) {
        choose_multiplier_64(N, udiv, N - shpre);
      }
    }

    if (cmp128(mrecip_128, twon_128) >= 0) {
      /* shpre must be zero */
      sub128(mrecip_128, twon_128, mrecip_128);
      t1 = KMULUH(ad_kcon(mrecip_128[2], mrecip_128[3]), n);
      q0 = ad2ili(IL_KSUB, n, t1);
      t2 = ICON(1);
      q0 = ad2ili(IL_KURSHIFT, q0, t2);
      q = ad2ili(IL_UKADD, t1, q0);
      if (shpost > 1) {
        t1 = ICON(shpost - 1);
        q = ad2ili(IL_KURSHIFT, q, t1);
      }
    } else {
      q = n;
      if (shpre > 0) {
        t1 = ICON(shpre);
        q = ad2ili(IL_KURSHIFT, n, t1);
      }
      q = KMULUH(ad_kcon(mrecip_128[2], mrecip_128[3]), q);
      if (shpost > 0) {
        t1 = ICON(shpost);
        q = ad2ili(IL_KURSHIFT, q, t1);
      }
    }
  }
  return q;
}

static int
reciprocal_mod(int n, int d, int sgnd)
{
  int div;
  int mul, sub;

  div = reciprocal_division(n, d, sgnd);
  if (div == 0)
    return 0;

  mul = MUL(ICON(d), div);
  if (sgnd) {
    sub = ad2ili(IL_ISUB, n, mul);
  } else {
    sub = ad2ili(IL_UISUB, n, mul);
  }

  return sub;
}

static int
reciprocal_mod_64(int n, DBLINT64 d, int sgnd)
{
  int div, kcon;
  int mul, sub;

  /*
   * d[0] MSW
   * d[1] LSW
   */
  div = reciprocal_division_64(n, d, sgnd);
  if (div == 0)
    return 0;

  kcon = ad_kcon(d[0], d[1]);
  mul = KMUL(kcon, div);
  if (sgnd) {
    sub = ad2ili(IL_KSUB, n, mul);
  } else {
    sub = ad2ili(IL_UKSUB, n, mul);
  }

  return sub;
}

#ifdef __cplusplus
inline bool IS_FLT0(int x)
{
  return is_flt0(static_cast<SPTR>(x));
}

inline bool IS_DBL0(int x)
{
  return is_dbl0(static_cast<SPTR>(x));
}

inline bool IS_QUAD0(int x)
{
  return is_quad0(static_cast<SPTR>(x));
}
#else
#define IS_FLT0 is_flt0
#define IS_DBL0 is_dbl0
#define IS_QUAD0 is_quad0
#endif

/**
 * \brief adds arithmetic ili
 */
static int
addarth(ILI *ilip)
{
  ILI newili; /* space for a new ili		 */

  ILI_OP opc,     /* opcode of ilip 		 */
      opc1,       /* opcode operand one		 */
      opc2;       /* opcode operand two		 */
  int op1,        /* operand one of ilip	 */
      op2;        /* operand two of ilip	 */
  SPTR cons1;     /* constant ST one		 */
  SPTR cons2;     /* constant ST two		 */
  int con1v1,     /* constant ST one conval1g	 */
      con1v2,     /* constant ST one conval2g	 */
      con2v1,     /* constant ST two conval1g	 */
      con2v2;     /* constant ST two conval2g	 */
  int ncons,      /* number of constants	 */
                  /* 0 => no constants		 */
                  /* 1 => first operand is	 */
                  /* 2 => second operand is	 */
                  /* 3 => both are constants	 */
      ilix,       /* ili result			 */
      mask_ili,   /* for potential mask with vector intrinsics */
      vdt1, vdt2; /* data types of args 1 & 2 for VPOW[I,K] intrinsics */

  ISZ_T aconoff1v, /* constant ST one aconoff	 */
      aconoff2v;   /* constant ST two aconoff	 */

  int is_int, pw;

  int i, tmp, tmp1; /* temporary                 */

  union { /* constant value structure */
    INT numi[2];
    UINT numu[2];
    DBLE numd;
  } res, num1, num2;
#ifdef TARGET_SUPPORTS_QUADFP
  union { /* quad constant value structure */
    INT numi[NUMI_SIZE];
    UINT numu[NUMU_SIZE];
    QUAD numq;
  } qres, qnum1, qnum2;
#endif
  CC_RELATION cond;
  const char *root;
  char *fname;
  SPTR funcsptr;
  MTH_FN mth_fn;

#define GETVAL64(a, b)       \
  do {                       \
    a.numd[0] = CONVAL1G(b); \
    a.numd[1] = CONVAL2G(b); \
  } while (0)
#define GETVALI64(a, b)      \
  do {                       \
    a.numi[0] = CONVAL1G(b); \
    a.numi[1] = CONVAL2G(b); \
  } while (0)

#define GETVAL128(a, b)      \
  do {                       \
    a.numq[0] = CONVAL1G(b); \
    a.numq[1] = CONVAL2G(b); \
    a.numq[2] = CONVAL3G(b); \
    a.numq[3] = CONVAL4G(b); \
  } while (0)
#define GETVALI128(a, b)     \
  do {                       \
    a.numi[0] = CONVAL1G(b); \
    a.numi[1] = CONVAL2G(b); \
    a.numi[2] = CONVAL3G(b); \
    a.numi[3] = CONVAL4G(b); \
  } while (0)

  ncons = 0;
  opc = ilip->opc;
  op1 = ilip->opnd[0];
  opc1 = ILI_OPC(op1);
  if (IL_TYPE(opc1) == ILTY_CONS) {
    aconoff1v = 0;
    if (IL_OPRFLAG(opc1, 1) == ILIO_STC) {
      ncons = 1;
      con1v1 = 0;
      con1v2 = ILI_OPND(op1, 1);
      cons1 = (SPTR)-2147483647; /* get an error if used */
    } else if (IL_OPRFLAG(opc1, 1) == ILIO_SYM) {
      ncons = 1;
      cons1 = ILI_SymOPND(op1, 1);
      con1v1 = CONVAL1G(cons1);
      if (opc1 == IL_ACON) {
        aconoff1v = ACONOFFG(cons1);
        con1v2 = aconoff1v;
      } else
        con1v2 = CONVAL2G(cons1);
    }
  }
  op2 = 0; /* prevent UMR if op2 is refd and #operands is 1 */
  if (ilis[opc].oprs > 1) {
    op2 = ilip->opnd[1];
    if (IL_ISLINK(opc, 2)) {
      opc2 = ILI_OPC(op2);
      if (IL_TYPE(opc2) == ILTY_CONS) {
        aconoff2v = 0;
        if (IL_OPRFLAG(opc2, 1) == ILIO_STC) {
          ncons |= 2;
          con2v1 = 0;
          con2v2 = ILI_OPND(op2, 1);
          cons2 = (SPTR)-2147483647; /* get an error if used */
        } else if (IL_OPRFLAG(opc2, 1) == ILIO_SYM) {
          ncons |= 2;
          cons2 = ILI_SymOPND(op2, 1);
          con2v1 = CONVAL1G(cons2);
          if (opc2 == IL_ACON) {
            aconoff2v = ACONOFFG(cons2);
            con2v2 = aconoff2v;
          } else
            con2v2 = CONVAL2G(cons2);
        }
      }
      if (IL_COMM(opc) != 0)
      {
        if (ncons == 1) {
          ncons = 2;
          cons2 = cons1;
          con2v1 = con1v1;
          con2v2 = con1v2;
          aconoff2v = aconoff1v;
          tmp = op1;
          op1 = op2;
          op2 = tmp;
          opc1 = ILI_OPC(op1);
          opc2 = ILI_OPC(op2);
        } else if (ncons == 0 && op1 > op2) {
          tmp = op1;
          op1 = op2;
          op2 = tmp;
          opc1 = ILI_OPC(op1);
          opc2 = ILI_OPC(op2);
        }
      }
    }
  }
  switch (opc) {
  case IL_UITOI:
  case IL_ITOUI:
    break;
  case IL_IR2SP:
    break;
  case IL_KR2SP:
    if (ncons == 1) {
      res.numi[1] = con1v2;
      goto add_rcon;
    }
    break;
  case IL_KR2DP:
    break;
  case IL_KR2CS:
    break;
  case IL_SP2IR:
    if (ncons == 1) {
      res.numi[1] = con1v2;
      goto add_icon;
    }
    break;
  case IL_SP2KR:
    if (ncons == 1) {
      res.numi[1] = con1v2;
      goto add_kcon;
    }
    break;
  case IL_DP2KR:
    break;
  case IL_CS2KR:
    break;

  case IL_ISIGN:
    if (ncons & 2) {
      /* second operand is constant */
      ilix = ad1ili(IL_IABS, op1);
      if (con2v2 < 0)
        ilix = ad1ili(IL_INEG, ilix);
      return ilix;
    }
    break;
  case IL_NINT:
    /* add constant folding later */
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_NINT, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#ifdef IL_KNINT
  case IL_KNINT:
    /* add constant folding later */
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KNINT, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
  case IL_IDNINT:
    /* add constant folding later */
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_IDNINT, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_IQNINT:
    /* add constant folding later */
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_IQNINT, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#ifdef IL_KIDNINT
  case IL_KIDNINT:
    /* add constant folding later */
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KIDNINT, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
  case IL_IDIM:
  case IL_FDIM:
  case IL_DDIM:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QDIM:
#endif
    /* add constant folding later */
    break;

  case IL_FSQRT:
    if (ncons == 1) {
      xfsqrt(con1v2, &res.numi[1]);
      goto add_rcon;
    }
    if (!flg.ieee) {
      if (XBIT(15, 0x20000000)) {
        int zero;
        int x0;
        /*
         * Just use the approximating reciprocal sqrt instruction
         * to compute sqrt:
         *   x0 = rsqrtss(x) & (cmpness(x, 0.0))
         *   x1 = x*x0;
         */
        zero = ad1ili(IL_FCON, stb.flt0);
        tmp = ad2ili(IL_CMPNEQSS, op1, zero);
        x0 = ad1ili(IL_RSQRTSS, op1); /* x0 */
        x0 = ad2ili(IL_FAND, x0, tmp);
        ilix = ad2ili(IL_FMUL, op1, x0);
        return ad2altili(opc, op1, op2, ilix); /* keep sqrt visible */
      }
      if (do_newton_sqrt()) {
        int three;
        int mhalf;
        int zero;
        int x0;
        /*
         * Newton's appx for sqrt:
         *   x1 = (3.0*x0 - x*x0**3)/2.0
         * or
         *   x1 = (3.0 - x*x0*x0)*x0*.5
         * or
         *   x1 = (x*x0*x0 - 3.0)*x0*(-.5)
         *
         * For just sqrt:
         *   x0 = rsqrtss(x) & (cmpness(x, 0.0))
         *   A = x*x0
         *   x1 = (A*x0 - 3.0)*A*(-.5)
         */
        num1.numi[0] = 0;
        num1.numi[1] = 0x40400000;
        three = ad1ili(IL_FCON, getcon(num1.numi, DT_FLOAT));
        num1.numi[1] = 0xbf000000;
        mhalf = ad1ili(IL_FCON, getcon(num1.numi, DT_FLOAT));
        zero = ad1ili(IL_FCON, stb.flt0);
        tmp = ad2ili(IL_CMPNEQSS, op1, zero);
        x0 = ad1ili(IL_RSQRTSS, op1); /* x0 */
        x0 = ad2ili(IL_FAND, x0, tmp);
        tmp = ad2ili(IL_FMUL, x0, op1);
        tmp1 = ad2ili(IL_FMUL, x0, tmp);
        tmp1 = ad2ili(IL_FSUB, tmp1, three);
        tmp1 = ad2ili(IL_FMUL, tmp1, tmp);
        ilix = ad2ili(IL_FMUL, tmp1, mhalf);
        return ad2altili(opc, op1, op2, ilix); /* keep sqrt visible */
      }
    }
    break;
  case IL_DSQRT:
    if (ncons == 1) {
      GETVAL64(num1, cons1);
      xdsqrt(num1.numd, res.numd);
      goto add_dcon;
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QSQRT:
    if (ncons == 1) {
      GETVAL128(qnum1, cons1);
      xqsqrt(qnum1.numq, qres.numq);
      goto add_qcon;
    }
    break;
#endif
#ifdef IL_FRSQRT
  case IL_FRSQRT:
#if !defined(TARGET_LLVM_ARM)
    if (XBIT(183, 0x10000)) {
      if (ncons == 1) {
        xfsqrt(con1v2, &res.numi[1]);
        xfdiv(CONVAL2G(stb.flt1), res.numi[1], &res.numi[1]);
        goto add_rcon;
      }
      return _frsqrt(op1);
    }
    break;
#else
    break;
#endif
#endif

  case IL_RCPSS:
    break;
  case IL_RSQRTSS:
    break;
  case IL_FAND:
    break;
  case IL_CMPNEQSS:
    break;

  case IL_INEG:
  case IL_UINEG:
    if (ncons == 1) {
      res.numi[1] = -con1v2;
      goto add_icon;
    }
    if (ILI_OPC(op1) == opc)
      return ILI_OPND(op1, 1);
    if (ILI_OPC(op1) == IL_ISUB)
      return ad2ili(IL_ISUB, ILI_OPND(op1, 2), ILI_OPND(op1, 1));
    if (opc == IL_INEG && ILI_OPC(op1) == IL_IMUL) {
      ilix = red_negate(op1, IL_INEG, IL_IMUL, IL_IDIV);
      if (ilix != op1)
        return ilix;
    }
    break;

  case IL_IABS:
    if (ncons == 1) {
      res.numi[1] = con1v2 > 0 ? con1v2 : -con1v2;
      goto add_icon;
    }
    /* expand iabs as follows:
     *     t0 = (int)op1 >> 31;
     *     t1 = op1 ^ t0;
     *    res = (unsigned)t1 - (unsigned) t0
     * NOTE:  need to buffer op1 and t0 with CSE (assume postorder traversal
     *        by scheduler).
     */
    /* assertion: no need for cse since sched treats multiple uses of
     * the same function ili as one call.
     */
    tmp = ad2ili(IL_ARSHIFT, op1, ad_icon((INT)31));
    tmp1 = ad2ili(IL_XOR, op1, tmp);
    ilix = ad2ili(IL_UISUB, tmp1, tmp);
    return ilix;

  case IL_KABS:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, stb.k0);
      if (cmp64(num1.numi, num2.numi) == -1) {
        neg64(num1.numi, res.numi);
        goto add_kcon;
      }
      return op1;
    }
    tmp = ad2ili(IL_KARSHIFT, op1, ad_icon((INT)63));
    tmp1 = ad2ili(IL_KXOR, op1, tmp);
    ilix = ad2ili(IL_UKSUB, tmp1, tmp);
    return ilix;

  case IL_FABS:
    if (ncons == 1) {
      xfabsv(con1v2, &res.numi[1]);
      goto add_rcon;
    }
    break;

  case IL_DABS:
    if (ncons == 1) {
      GETVAL64(num1, cons1);
      xdabsv(num1.numd, res.numd);
      goto add_dcon;
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QABS:
    if (ncons == 1) {
      GETVAL128(qnum1, cons1);
      xqabsv(qnum1.numq, qres.numq);
      goto add_qcon;
    }
    break;
#endif
  case IL_KNEG:
  case IL_UKNEG:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, stb.k0);
      sub64(num2.numi, num1.numi, res.numi);
      goto add_kcon;
    }
    if (ILI_OPC(op1) == opc)
      return ILI_OPND(op1, 1);
    if (ILI_OPC(op1) == IL_KSUB)
      return ad2ili(IL_KSUB, ILI_OPND(op1, 2), ILI_OPND(op1, 1));
    if (opc == IL_KNEG && ILI_OPC(op1) == IL_KMUL) {
      ilix = red_negate(op1, IL_KNEG, IL_KMUL, IL_KDIV);
      if (ilix != op1)
        return ilix;
    }
    break;

  case IL_FNEG:
    if (ncons == 1) {
      xfneg(con1v2, &res.numi[1]);
      goto add_rcon;
    }
    if (!flg.ieee && ILI_OPC(op1) == IL_FSUB) {
      /* -(a - b) --> b - a */
      op2 = ILI_OPND(op1, 1);
      op1 = ILI_OPND(op1, 2);
      return ad2ili(IL_FSUB, op1, op2);
    }
    if (ILI_OPC(op1) == IL_FMUL) {
      ilix = red_negate(op1, IL_FNEG, IL_FMUL, IL_FDIV);
      if (ilix != op1)
        return ilix;
    }
    break;

  case IL_SCMPLXCMP:
    break;
  case IL_DCMPLXCMP:
    break;
  case IL_SCMPLXCONJG:
    if (ncons == 1) {
      res.numi[0] = con1v1;
      xfneg(con1v2, &res.numi[1]);
      return ad1ili(IL_SCMPLXCON, getcon(res.numi, DT_CMPLX));
    }
    break;
  case IL_DCMPLXCONJG:
    if (ncons == 1) {
      GETVAL64(num2, con1v2);
      xdneg(num2.numd, res.numd);
      cons2 = getcon(res.numd, DT_DBLE);
      res.numi[0] = con1v1;
      res.numi[1] = cons2;
      return ad1ili(IL_DCMPLXCON, getcon(res.numi, DT_DCMPLX));
    }
    break;

  case IL_SCMPLXNEG:
    if (ncons == 1) {
      xfneg(con1v1, &res.numi[0]);
      xfneg(con1v2, &res.numi[1]);
      return ad1ili(IL_SCMPLXCON, getcon(res.numi, DT_CMPLX));
    }
    if (!flg.ieee && ILI_OPC(op1) == IL_SCMPLXSUB) {
      /* -(a - b) --> b - a */
      op2 = ILI_OPND(op1, 1);
      op1 = ILI_OPND(op1, 2);
      return ad2ili(IL_SCMPLXSUB, op1, op2);
    }
    break;

  case IL_DNEG:
    if (ncons == 1) {
      GETVAL64(num2, cons1);
      xdneg(num2.numd, res.numd);
      goto add_dcon;
    }
    if (!flg.ieee && ILI_OPC(op1) == IL_DSUB) {
      /* -(a - b) --> b - a */
      op2 = ILI_OPND(op1, 1);
      op1 = ILI_OPND(op1, 2);
      return ad2ili(IL_DSUB, op1, op2);
    }
    if (ILI_OPC(op1) == IL_DMUL) {
      ilix = red_negate(op1, IL_DNEG, IL_DMUL, IL_DDIV);
      if (ilix != op1)
        return ilix;
    }
    break;

  case IL_DCMPLXNEG:
    if (ncons == 1) {
      GETVAL64(num1, con1v1);
      GETVAL64(num2, con1v2);
      xdneg(num1.numd, res.numd);
      cons1 = getcon(res.numd, DT_DBLE);
      xdneg(num2.numd, res.numd);
      cons2 = getcon(res.numd, DT_DBLE);
      res.numi[0] = cons1;
      res.numi[1] = cons2;
      return ad1ili(IL_DCMPLXCON, getcon(res.numi, DT_DCMPLX));
    }
    if (!flg.ieee && ILI_OPC(op1) == IL_DCMPLXSUB) {
      /* -(a - b) --> b - a */
      op2 = ILI_OPND(op1, 1);
      op1 = ILI_OPND(op1, 2);
      return ad2ili(IL_DCMPLXSUB, op1, op2);
    }
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QNEG:
    if (ncons == 1) {
      GETVAL128(qnum2, cons1);
      xqneg(qnum2.numq, qres.numq);
      goto add_qcon;
    }
    if (!flg.ieee && ILI_OPC(op1) == IL_QSUB) {
      /* -(a - b) --> b - a */
      op2 = ILI_OPND(op1, 1);
      op1 = ILI_OPND(op1, 2);
      return ad2ili(IL_QSUB, op1, op2);
    }
    if (ILI_OPC(op1) == IL_QMUL) {
      ilix = red_negate(op1, IL_QNEG, IL_QMUL, IL_QDIV);
      if (ilix != op1)
        return ilix;
    }
    break;
#endif
  case IL_FIX:
    if (ncons == 1) {
      xfix(con1v2, &res.numi[1]);
      goto add_icon;
    }
    break;

  case IL_UFIX:
    if (ncons == 1) {
      xfixu(con1v2, &res.numu[1]);
      goto add_icon;
    }
    break;

  case IL_FIXK:
  case IL_FIXUK:
    if (ncons == 1) {
      xfix64(con1v2, res.numi);
      goto add_kcon;
    }
    break;

  case IL_DFIX:
    if (ncons == 1) {
      GETVAL64(num1, cons1);
      xdfix(num1.numd, &res.numi[1]);
      goto add_icon;
    }
    break;

  case IL_DFIXU:
    if (ncons == 1) {
      GETVAL64(num1, cons1);
      xdfixu(num1.numd, &res.numu[1]);
      goto add_icon;
    }
    break;

  case IL_DFIXK:
  case IL_DFIXUK:
    if (ncons == 1) {
      GETVAL64(num1, cons1);
      xdfix64(num1.numi, res.numi);
      goto add_kcon;
    }
    break;

  case IL_FLOAT:
    if (ncons == 1) {
      xffloat(con1v2, &res.numi[1]);
      goto add_rcon;
    }
    break;

  case IL_FLOATU:
    if (ncons == 1) {
      xffloatu(con1v2, &res.numi[1]);
      goto add_rcon;
    }
    break;

  case IL_FLOATK:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      xflt64(num1.numi, &res.numi[1]);
      goto add_rcon;
    }
    break;

  case IL_FLOATUK:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      xfltu64(num1.numu, &res.numi[1]);
      goto add_rcon;
    }
    break;

  case IL_DFLOATK:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      xdflt64(num1.numi, res.numi);
      goto add_dcon;
    }
    break;

  case IL_DFLOATUK:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      xdfltu64(num1.numu, res.numi);
      goto add_dcon;
    }
    break;

  case IL_DFLOAT:
    if (ncons == 1) {
      xdfloat(con1v2, res.numd);
      goto add_dcon;
    }
    break;

  case IL_DFLOATU:
    if (ncons == 1) {
      xdfloatu(con1v2, res.numd);
      goto add_dcon;
    }
    break;

  case IL_SNGL:
    if (ncons == 1) {
      GETVAL64(res, cons1);
      xsngl(res.numd, &res.numi[1]);
      goto add_rcon;
    }
    break;

  case IL_DBLE:
    if (ncons == 1) {
      xdble(con1v2, res.numd);
      goto add_dcon;
    }
    break;

  case IL_UNOT:
  case IL_NOT:
    if (ncons == 1) {
      res.numi[1] = ~con1v2;
      goto add_icon;
    }
    goto involution;
    break;

  case IL_UKNOT:
  case IL_KNOT:
    if (ncons == 1) {
      res.numi[0] = ~con1v1;
      res.numi[1] = ~con1v2;
      goto add_kcon;
    }
  involution:
    if ((opc == opc1) && (opc1 == IL_NOT || opc1 == IL_UNOT ||
                          opc1 == IL_KNOT || opc1 == IL_UKNOT)) {
      /* Involution expression: ~~A = A */
      return ILI_OPND(op1, 1);
    }
    break;

  case IL_ICMPZ:
    if (ncons == 1) {
      res.numi[1] = cmp_to_log(icmp(con1v2, (INT)0), op2);
      goto add_icon;
    }
    if (ILI_OPC(op1) == IL_ISUB)
      return ad3ili(IL_ICMP, (int)ILI_OPND(op1, 1), (int)ILI_OPND(op1, 2), op2);
    break;

  case IL_UICMPZ:
    if (ncons == 1) {
      res.numi[1] = cmp_to_log(xucmp(con1v2, (INT)0), op2);
      goto add_icon;
    }
    switch (op2) {
    default:
      break;
    case CC_LT: /* <  0 becomes false */
      if (func_in(op1))
        break;
      res.numi[1] = 0;
      goto add_icon;
    case CC_GE: /* >= 0 becomes true */
      if (func_in(op1))
        break;
      res.numi[1] = 1;
      goto add_icon;
    case CC_LE: /* <= 0 becomes eq */
      op2 = CC_EQ;
      opc = IL_ICMPZ;
      break;
    case CC_GT: /* >  0 becomes ne */
      op2 = CC_NE;
      opc = IL_ICMPZ;
      break;
    }
    break;

  case IL_KCMP:
    newili.opnd[2] = ilip->opnd[2];
    if (ncons == 1 && con1v1 == 0 && con1v2 == 0)
      return ad3ili(IL_KCMP, op2, op1, commute_cc(CCRelationILIOpnd(ilip, 2)));
    if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      res.numi[1] = cmp_to_log(cmp64(num1.numi, num2.numi), ilip->opnd[2]);
      goto add_icon;
    } else if (op1 == op2 && !func_in(op1)) {
      res.numi[1] = cmp_to_log(0, ilip->opnd[2]);
      goto add_icon;
    }
    break;

  case IL_UKCMP:
    newili.opnd[2] = ilip->opnd[2];
    if (ncons == 1 && con1v1 == 0 && con1v2 == 0)
      return ad3ili(IL_UKCMP, op2, op1,
                    commute_cc(CCRelationILIOpnd(ilip, 2)));
    if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      res.numi[1] = cmp_to_log(ucmp64(num1.numu, num2.numu), ilip->opnd[2]);
      goto add_icon;
    } else if (op1 == op2 && !func_in(op1)) {
      res.numi[1] = cmp_to_log(0, ilip->opnd[2]);
      goto add_icon;
    }
    break;

  case IL_KCMPZ:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      num2.numi[0] = num2.numi[1] = 0;
      res.numi[1] = cmp_to_log(cmp64(num1.numi, num2.numi), op2);
      goto add_icon;
    }
    if (ILI_OPC(op1) == IL_KSUB)
      return ad3ili(IL_KCMP, (int)ILI_OPND(op1, 1), (int)ILI_OPND(op1, 2), op2);
    return ad3ili(IL_KCMP, op1, ad1ili(IL_KCON, stb.k0), op2);

  case IL_UKCMPZ:
    if (ncons == 1) {
      GETVALI64(num1, cons1);
      num2.numi[0] = num2.numi[1] = 0;
      res.numi[1] = cmp_to_log(ucmp64(num1.numu, num2.numu), op2);
      goto add_icon;
    }
    switch (op2) {
    default:
      return ad3ili(IL_UKCMP, op1, ad1ili(IL_KCON, stb.k0), op2);
    case CC_LT: /* <  0 becomes false */
      if (func_in(op1))
        break;
      res.numi[1] = 0;
      goto add_icon;
    case CC_GE: /* >= 0 becomes true */
      if (func_in(op1))
        break;
      res.numi[1] = 1;
      goto add_icon;
    case CC_LE: /* <= 0 becomes eq */
      op2 = CC_EQ;
      opc = IL_KCMPZ;
      break;
    case CC_GT: /* >  0 becomes ne */
      op2 = CC_NE;
      opc = IL_KCMPZ;
      break;
    }
    break;

  case IL_FCMPZ:
    if (ncons == 1) {
      res.numi[1] = cmp_to_log(xfcmp(con1v2, CONVAL2G(stb.flt0)), op2);
      goto add_icon;
    }
    if (!IEEE_CMP && ILI_OPC(op1) == IL_FSUB)
      return ad3ili(IL_FCMP, (int)ILI_OPND(op1, 1), (int)ILI_OPND(op1, 2), op2);
#ifndef TM_FCMPZ
    tmp = ad1ili(IL_FCON, stb.flt0);
    return ad3ili(IL_FCMP, op1, tmp, op2);
#endif
    break;

  case IL_DCMPZ:
    if (ncons == 1) {
      GETVAL64(num1, cons1);
      GETVAL64(num2, stb.dbl0);
      res.numi[1] = cmp_to_log(xdcmp(num1.numd, num2.numd), op2);
      goto add_icon;
    }
    if (!IEEE_CMP && ILI_OPC(op1) == IL_DSUB)
      return ad3ili(IL_DCMP, (int)ILI_OPND(op1, 1), (int)ILI_OPND(op1, 2), op2);
    if (ILI_OPC(op1) == IL_DBLE && !XBIT(15, 0x80))
      return ad2ili(IL_FCMPZ, ILI_OPND(op1, 1), op2);
#ifndef TM_DCMPZ
    tmp = ad1ili(IL_DCON, stb.dbl0);
    return ad3ili(IL_DCMP, op1, tmp, op2);
#endif
    break;

  case IL_ACMPZ:
    if (ncons == 1) {
      int sym;
      sym = con1v1;
      if (sym == 0) {
        res.numi[1] = cmp_to_log(icmp(con1v2, (INT)0), op2);
        goto add_icon;
      }
      /* comparing an address with NULL */
      switch (op2) {
      case CC_LT:
        res.numi[1] = 0;
        goto add_icon;
      case CC_EQ:
      case CC_LE:
        if (IS_LCL_OR_DUM(sym) || CCSYMG(sym)) {
          res.numi[1] = 0;
          goto add_icon;
        }
        break;
      case CC_GE:
        res.numi[1] = 1;
        goto add_icon;
      default:
        if (IS_LCL_OR_DUM(sym) || CCSYMG(sym)) {
          res.numi[1] = 1;
          goto add_icon;
        }
        break;
      }
    }
    switch (op2) {
    default:
      break;
    case CC_LT: /* <  0 becomes false */
      res.numi[1] = 0;
      goto add_icon;
    case CC_GE: /* >= 0 becomes true */
      res.numi[1] = 1;
      goto add_icon;
    case CC_LE: /* <= 0 becomes eq */
      op2 = CC_EQ;
      break;
    case CC_GT: /* >  0 becomes ne */
      op2 = CC_NE;
      break;
    }
    break;

  case IL_IMAX:
    if (ncons == 3) {
      if (con1v2 > con2v2)
        return op1;
      return op2;
    }
    return red_minmax(opc, op1, op2);

  case IL_IMIN:
    if (ncons == 3) {
      if (con1v2 < con2v2)
        return op1;
      return op2;
    }
    return red_minmax(opc, op1, op2);
  case IL_UIMAX:
    if (ncons == 3) {
      if (xucmp((UINT)con1v2, (UINT)con2v2) > 0)
        return op1;
      return op2;
    }
    if (ncons == 1 && cons1 == stb.i0) {
      return op2;
    }
    if (ncons == 2 && cons2 == stb.i0) {
      return op1;
    }
    return red_minmax(opc, op1, op2);
  case IL_UIMIN:
    if (ncons == 3) {
      if (xucmp((UINT)con1v2, (UINT)con2v2) < 0)
        return op1;
      return op2;
    }
    if (ncons == 1 && cons1 == stb.i0 && !func_in(op2)) {
      return op1;
    }
    if (ncons == 2 && cons2 == stb.i0 && !func_in(op1)) {
      return op2;
    }
    return red_minmax(opc, op1, op2);
  case IL_KMAX:
    if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      if (cmp64(num1.numi, num2.numi) > 0)
        return op1;
      return op2;
    }
    ilix = red_minmax(opc, op1, op2);
    return ilix;
  case IL_KMIN:
    if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      if (cmp64(num1.numi, num2.numi) < 0)
        return op1;
      return op2;
    }
    ilix = red_minmax(opc, op1, op2);
    return ilix;
  case IL_UKMAX:
    if (ncons == 1 && con1v1 == 0 && con1v2 == 0)
      return op2;
    else if (ncons == 2 && con2v1 == 0 && con2v2 == 0)
      return op1;
    else if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      if (ucmp64(num1.numu, num2.numu) > 0)
        return op1;
      return op2;
    }
    return red_minmax(opc, op1, op2);
  case IL_UKMIN:
    if (ncons == 1 && con1v1 == 0 && con1v2 == 0 && !func_in(op2))
      return op1;
    else if (ncons == 2 && con2v1 == 0 && con2v2 == 0 && !func_in(op1))
      return op2;
    else if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      if (ucmp64(num1.numu, num2.numu) < 0)
        return op1;
      return op2;
    }
    return red_minmax(opc, op1, op2);
  case IL_IADD:
    if (ILI_OPC(op2) == IL_INEG)
      return ad2ili(IL_ISUB, op1, (int)ILI_OPND(op2, 1));
    if (ILI_OPC(op1) == IL_INEG)
      return ad2ili(IL_ISUB, op2, (int)ILI_OPND(op1, 1));
    goto add_shared;
  case IL_UIADD:
    if (ILI_OPC(op2) == IL_UINEG)
      return ad2ili(IL_UISUB, op1, (int)ILI_OPND(op2, 1));
    if (ILI_OPC(op1) == IL_UINEG)
      return ad2ili(IL_UISUB, op2, (int)ILI_OPND(op1, 1));
  add_shared:
    if (ncons == 0)
      break;
    if ((res.numi[1] = con2v2) == 0)
      return op1;
    tmp = red_iadd(op1, res.numi[1]);
    if (tmp)
      return tmp;
    if (opc == IL_IADD && (res.numi[1] < 0 && res.numi[1] != (INT)0x80000000))
      return ad2ili(IL_ISUB, op1, ad_icon(-res.numi[1]));
    break;
  case IL_KADD:
    if (ILI_OPC(op2) == IL_KNEG)
      return ad2ili(IL_KSUB, op1, (int)ILI_OPND(op2, 1));
    if (ILI_OPC(op1) == IL_KNEG)
      return ad2ili(IL_KSUB, op2, (int)ILI_OPND(op1, 1));
    goto kadd_shared;
  case IL_UKADD:
    if (ILI_OPC(op2) == IL_UKNEG)
      return ad2ili(IL_UKSUB, op1, (int)ILI_OPND(op2, 1));
    if (ILI_OPC(op1) == IL_UKNEG)
      return ad2ili(IL_UKSUB, op2, (int)ILI_OPND(op1, 1));
  kadd_shared:
    if (ncons == 0) {
      break;
    }
    GETVALI64(res, cons2);
    if (res.numi[0] == 0 && res.numi[1] == 0)
      return op1;
    if (ncons == 3) {
      GETVALI64(num1, cons1);
      add64(num1.numi, res.numi, res.numi);
      goto add_kcon;
    }
    tmp = red_kadd(op1, res.numi);
    if (tmp)
      return tmp;
    if (opc == IL_KADD && res.numi[0] < 0 &&
        !(res.numi[0] == (INT)0x80000000 && res.numi[1] == 0)) {
      neg64(res.numi, res.numi);
      op2 = ad1ili(IL_KCON, getcon(res.numi, DT_INT8));
      return ad2ili(IL_KSUB, op1, op2);
    }
    break;

  case IL_FADD:
    if (ncons == 2 && is_flt0(cons2))
      return op1;
#ifdef FPSUB2ADD
  like_fadd:
#endif
    if (!flg.ieee && ncons == 3) {
      xfadd(con1v2, con2v2, &res.numi[1]);
      goto add_rcon;
    }
    if (ILI_OPC(op1) == IL_FNEG) {
      /* -a + b --> b - a */
      opc = IL_FSUB;
      tmp = op2;
      op2 = ILI_OPND(op1, 1);
      op1 = tmp;
    } else if (ILI_OPC(op2) == IL_FNEG) {
      /* a + -b --> a - b */
      opc = IL_FSUB;
      op2 = ILI_OPND(op2, 1);
    }
    break;

  case IL_DADD:
    if (ncons == 2 && is_dbl0(cons2))
      return op1;
#ifdef FPSUB2ADD
  like_dadd:
#endif
    if (!flg.ieee && ncons == 3) {
      GETVAL64(num1, cons1);
      GETVAL64(num2, cons2);
      xdadd(num1.numd, num2.numd, res.numd);
      goto add_dcon;
    }
    if (ILI_OPC(op1) == IL_DNEG) {
      /* -a + b --> b - a */
      opc = IL_DSUB;
      tmp = op2;
      op2 = ILI_OPND(op1, 1);
      op1 = tmp;
    } else if (ILI_OPC(op2) == IL_DNEG) {
      /* a + -b --> a - b */
      opc = IL_DSUB;
      op2 = ILI_OPND(op2, 1);
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QADD:
    if (ncons == 2 && is_quad0(cons2))
      return op1;
#ifdef FPSUB2ADD
  like_qadd:
#endif
    if (!flg.ieee && ncons == 3) {
      GETVAL128(qnum1, cons1);
      GETVAL128(qnum2, cons2);
      xqadd(qnum1.numq, qnum2.numq, qres.numq);
      goto add_qcon;
    }
    if (ILI_OPC(op1) == IL_QNEG) {
      /* -a + b --> b - a */
      opc = IL_QSUB;
      tmp = op2;
      op2 = ILI_OPND(op1, 1);
      op1 = tmp;
    } else if (ILI_OPC(op2) == IL_QNEG) {
      /* a + -b --> a - b */
      opc = IL_QSUB;
      op2 = ILI_OPND(op2, 1);
    }
    break;
#endif
  case IL_SCMPLXADD:
    if (ncons == 2 && IS_FLT0(con2v1) && IS_FLT0(con2v2))
      return op1;
#ifdef FPSUB2ADD
  like_scmplxadd:
#endif
    if (!flg.ieee && ncons == 3) {
      xfadd(con1v1, con2v1, &res.numi[0]);
      xfadd(con1v2, con2v2, &res.numi[1]);
      return ad1ili(IL_SCMPLXCON, getcon(res.numi, DT_CMPLX));
    }
    if (ILI_OPC(op1) == IL_SCMPLXNEG) {
      opc = IL_SCMPLXSUB;
      tmp = op2;
      op2 = ILI_OPND(op1, 1);
      op1 = tmp;
    } else if (ILI_OPC(op2) == IL_SCMPLXNEG) {
      opc = IL_SCMPLXSUB;
      op2 = ILI_OPND(op2, 1);
    }
    break;
  case IL_DCMPLXADD:
    if (ncons == 2 && IS_DBL0(con2v1) && IS_DBL0(con2v2))
      return op1;
#ifdef FPSUB2ADD
  like_dcmplxadd:
#endif
    if (!flg.ieee && ncons == 3) {
      GETVAL64(num1, con1v1);
      GETVAL64(num2, con2v1);
      xdadd(num1.numd, num2.numd, res.numd);
      cons1 = getcon(res.numd, DT_DBLE);
      GETVAL64(num1, con1v2);
      GETVAL64(num2, con2v2);
      xdadd(num1.numd, num2.numd, res.numd);
      cons2 = getcon(res.numd, DT_DBLE);
      res.numi[0] = cons1;
      res.numi[1] = cons2;
      return ad1ili(IL_DCMPLXCON, getcon(res.numi, DT_DCMPLX));
    }
    if (ILI_OPC(op1) == IL_DCMPLXNEG) {
      opc = IL_DCMPLXSUB;
      tmp = op2;
      op2 = ILI_OPND(op1, 1);
      op1 = tmp;
    } else if (ILI_OPC(op2) == IL_DCMPLXNEG) {
      opc = IL_DCMPLXSUB;
      op2 = ILI_OPND(op2, 1);
    }
    break;

  case IL_AADD:
    newili.opnd[2] = ilip->opnd[2]; /* save away scale factor */
#define RED_DAMV (!XBIT(15, 0x100))
    if (ncons == 0) {
      if (RED_DAMV) {
        if (ILI_OPC(op2) == IL_IAMV) {
          tmp = red_damv(op1, op2, (int)ilip->opnd[2]);
          if (tmp)
            return tmp;
        } else if (ILI_OPC(op1) == IL_IAMV) {
          tmp = red_damv(op2, op1, (int)ilip->opnd[2]);
          if (tmp)
            return tmp;
        } else if (ILI_OPC(op2) == IL_KAMV) {
          tmp = red_damv(op1, op2, (int)ilip->opnd[2]);
          if (tmp)
            return tmp;
        } else if (ILI_OPC(op1) == IL_KAMV) {
          tmp = red_damv(op2, op1, (int)ilip->opnd[2]);
          if (tmp)
            return tmp;
        }
      }
      break;
    }
    if (ncons == 1) { /* only the left operand is a constant */
      if (ilip->opnd[2] == 0 && con1v1 == 0 && con1v2 == 0)
        return op2;
      break;
    }
  like_aadd: {
    SPTR sptr_con2v1;
    if (con2v1 == 0 && aconoff2v == 0)
      return op1;
    sptr_con2v1 = (SPTR)con2v1;
    tmp = red_aadd(op1, sptr_con2v1, aconoff2v, ilip->opnd[2]);
    if (tmp)
      return tmp;
    } break;

  case IL_ISUB:
    if (ILI_OPC(op2) == IL_INEG)
      return ad2ili(IL_IADD, op1, ILI_OPND(op2, 1));
    goto sub_shared;
  case IL_UISUB:
    if (ILI_OPC(op2) == IL_UINEG)
      return ad2ili(IL_UIADD, op1, ILI_OPND(op2, 1));
  sub_shared:
    if (op1 == op2) {
      if (func_in(op1))
        break;
      res.numi[1] = 0;
      goto add_icon;
    }
    if (ncons == 0)
      break;
    if (ncons == 1) {
      if (cons1 == stb.i0)
        return ad1ili(IL_INEG, op2);
      break;
    }
    if ((res.numi[1] = con2v2) == 0)
      return op1;
    if (res.numi[1] == (INT)0x80000000)
      break;
    tmp = red_iadd(op1, -res.numi[1]);
    if (tmp)
      return tmp;
    if ((opc == IL_ISUB) && (res.numi[1] < 0))
      return ad2ili(IL_IADD, op1, ad_icon(-res.numi[1]));
    break;

  case IL_KSUB:
    if (ILI_OPC(op2) == IL_KNEG)
      return ad2ili(IL_KADD, op1, (int)ILI_OPND(op2, 1));
    goto ksub_shared;
  case IL_UKSUB:
    if (ILI_OPC(op2) == IL_UKNEG)
      return ad2ili(IL_UKADD, op1, (int)ILI_OPND(op2, 1));
  ksub_shared:
    if (op1 == op2) {
      if (func_in(op1))
        break;
      GETVALI64(res, stb.k0);
      goto add_kcon;
    }
    if (ncons == 0) {
      break;
    }
    if (ncons == 1) {
      if (cons1 == stb.k0)
        return ad1ili(IL_KNEG, op2);
      break;
    }
    if (cons2 == stb.k0)
      return op1;
    GETVALI64(res, cons2);
    if (res.numi[0] == (INT)0x80000000 && res.numi[1] == 0) {
      break;
    }
    neg64(res.numi, num2.numi);
    tmp = red_kadd(op1, num2.numi);
    if (tmp)
      return tmp;
    if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      sub64(num1.numi, num2.numi, res.numi);
      goto add_kcon;
    }
    if (res.numi[0] < 0) {
      neg64(res.numi, res.numi);
      op2 = ad1ili(IL_KCON, getcon(res.numi, DT_INT8));
      return ad2ili(IL_KADD, op1, op2);
    }
    break;

  case IL_FSUB:
#ifdef FPSUB2ADD
    if (!flg.ieee && ncons >= 2) {
      xfsub((INT)0, con2v2, &res.numi[1]);
      res.numi[0] = 0;
      cons2 = getcon(res.numi, DT_FLOAT);
      op2 = ad1ili(IL_FCON, cons2);
      opc = IL_FADD;
      goto like_fadd;
    }
#else
    if (!flg.ieee && ncons == 3) {
      xfsub(con1v2, con2v2, &res.numi[1]);
      goto add_rcon;
    }
    if (ncons == 2 && is_flt0(cons2))
      return op1;
#endif
    if (ncons == 1 && is_flt0(cons1))
      return ad1ili(IL_FNEG, op2);
    if (ILI_OPC(op2) == IL_FNEG) {
      /* a - -b --> a + b */
      opc = IL_FADD;
      op2 = ILI_OPND(op2, 1);
    }
    break;

  case IL_DSUB:
#ifdef FPSUB2ADD
    if (!flg.ieee && ncons >= 2) {
      GETVAL64(num1, stb.dbl0);
      GETVAL64(num2, cons2);
      xdsub(num1.numd, num2.numd, res.numd);
      cons2 = getcon(res.numi, DT_DBLE);
      op2 = ad1ili(IL_DCON, cons2);
      opc = IL_DADD;
      goto like_dadd;
    }
#else
    if (!flg.ieee && ncons == 3) {
      GETVAL64(num1, cons1);
      GETVAL64(num2, cons2);
      xdsub(num1.numd, num2.numd, res.numd);
      goto add_dcon;
    }
    if (ncons == 2 && is_dbl0(cons2))
      return op1;
#endif
    if (ncons == 1 && is_dbl0(cons1))
      return ad1ili(IL_DNEG, op2);
    if (ILI_OPC(op2) == IL_DNEG) {
      /* a - -b --> a + b */
      opc = IL_DADD;
      op2 = ILI_OPND(op2, 1);
    }
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QSUB:
#ifdef FPSUB2ADD
    if (!flg.ieee && ncons >= 2) {
      GETVAL128(qnum1, stb.quad0);
      GETVAL128(qnum2, cons2);
      xqsub(qnum1.numq, qnum2.numq, qres.numq);
      cons2 = getcon(qres.numi, DT_QUAD);
      op2 = ad1ili(IL_QCON, cons2);
      opc = IL_QADD;
      goto like_qadd;
    }
#else
    if (!flg.ieee && ncons == 3) {
      GETVAL128(qnum1, cons1);
      GETVAL128(qnum2, cons2);
      xqsub(qnum1.numq, qnum2.numq, qres.numq);
      goto add_qcon;
    }
    if (ncons == 2 && is_quad0(cons2))
      return op1;
#endif
    if (ncons == 1 && is_quad0(cons1))
      return ad1ili(IL_QNEG, op2);
    if (ILI_OPC(op2) == IL_QNEG) {
      /* a - -b --> a + b */
      opc = IL_QADD;
      op2 = ILI_OPND(op2, 1);
    }
    break;
#endif

  case IL_SCMPLXSUB:
#ifdef FPSUB2ADD
    if (!flg.ieee && ncons >= 2) {
      xfsub(0, con2v1, &res.numi[0]);
      xfsub(0, con2v2, &res.numi[1]);
      op2 = ad1ili(IL_SCMPLXCON, getcon(res.numi, DT_CMPLX));
      opc = IL_SCMPLXADD;
      goto like_scmplxadd;
    }
#else
    if (!flg.ieee && ncons == 3) {
      xfsub(con1v1, con2v1, &res.numi[0]);
      xfsub(con1v2, con2v2, &res.numi[1]);
      op2 = ad1ili(IL_SCMPLXCON, getcon(res.numi, DT_CMPLX));
      return op2;
    }
    if (ncons == 2 && IS_FLT0(con2v1) && IS_FLT0(con2v2))
      return op1;
#endif
    if (ncons == 1 && IS_FLT0(con1v1) && IS_FLT0(con1v2))
      return ad1ili(IL_SCMPLXNEG, op2);
    if (ILI_OPC(op2) == IL_SCMPLXNEG) {
      opc = IL_SCMPLXADD;
      op2 = ILI_OPND(op2, 1);
    }
    break;
  case IL_DCMPLXSUB:
#ifdef FPSUB2ADD
    if (!flg.ieee && ncons >= 2) {
      GETVAL64(num1, stb.dbl0);
      GETVAL64(num2, con2v1);
      xdsub(num1.numd, num2.numd, res.numd);
      cons1 = getcon(res.numi, DT_DBLE);
      GETVAL64(num1, stb.dbl0);
      GETVAL64(num2, con2v2);
      xdsub(num1.numd, num2.numd, res.numd);
      cons2 = getcon(res.numi, DT_DBLE);
      res.numi[0] = cons1;
      res.numi[1] = cons2;
      op2 = ad1ili(IL_DCMPLXCON, getcon(res.numi, DT_DCMPLX));
      opc = IL_DCMPLXADD;
      goto like_dcmplxadd;
    }
#else
    if (!flg.ieee && ncons == 3) {
      GETVAL64(num1, con1v1);
      GETVAL64(num2, con2v1);
      xdsub(num1.numd, num2.numd, res.numd);
      cons1 = getcon(res.numd, DT_DBLE);
      GETVAL64(num1, con1v2);
      GETVAL64(num2, con2v2);
      xdsub(num1.numd, num2.numd, res.numd);
      cons2 = getcon(res.numd, DT_DBLE);
      res.numi[0] = cons1;
      res.numi[1] = cons2;
      op2 = ad1ili(IL_DCMPLXCON, getcon(res.numi, DT_DCMPLX));
      return op2;
    }
    if (ncons == 2 && IS_DBL0(con2v1) && IS_DBL0(con2v2))
      return op1;
#endif
    if (ncons == 1 && IS_DBL0(con1v1) && IS_DBL0(con1v2))
      return ad1ili(IL_DCMPLXNEG, op2);
    if (ILI_OPC(op2) == IL_DCMPLXNEG) {
      opc = IL_DCMPLXADD;
      op2 = ILI_OPND(op2, 1);
    }
    break;

  case IL_ASUB:
    /* (p + <x>) - p -> <x> */
    if (ilip->opnd[2] == 0 && ILI_OPC(op1) == IL_AADD &&
        ILI_OPND(op1, 3) == 0) {
      if (op2 == ILI_OPND(op1, 1) && !func_in(op2))
        return ILI_OPND(op1, 2);
    }
    newili.opnd[2] = ilip->opnd[2];
    if (ncons >= 2 && con2v1 == 0) {
      con2v1 = SPTR_NULL;
      aconoff2v = -aconoff2v;
      cons2 = get_acon(SPTR_NULL, aconoff2v);
      op2 = ad1ili(IL_ACON, cons2);
      opc = IL_AADD;
      goto like_aadd;
    }
    break;

  case IL_IMUL:
  case IL_UIMUL:
    if (ncons == 2) {
      if (cons2 == stb.i0 && !func_in(op1))
        return op2;
      if (cons2 == stb.i1)
        return op1;
      if (ILI_OPC(op1) == opc && ILI_OPC(ilix = ILI_OPND(op1, 2)) == IL_ICON) {
        /*  (i * c1) * c2  --->  i * (c1*c2)  */
        res.numi[0] = con2v2 * CONVAL2G(ILI_OPND(ilix, 1));
        op2 = ad_icon(res.numi[0]);
        ilix = ad2ili(opc, (int)ILI_OPND(op1, 1), op2);
        return ilix;
      }
      if (ILI_OPC(op1) == IL_INEG || ILI_OPC(op1) == IL_UINEG) {
        /*  (-i) * c  --->  i * (-c)  */
        res.numi[0] = -con2v2;
        op2 = ad_icon(res.numi[0]);
        ilix = ad2ili(opc, (int)ILI_OPND(op1, 1), op2);
        return ilix;
      }
      if (opc == IL_IMUL && con2v2 == -1)
        return ad1ili(IL_INEG, op1);
    } else if (ncons == 3) {
      res.numi[1] = con1v2 * con2v2;
      goto add_icon;
    }
    break;

  case IL_KMUL:
  case IL_UKMUL:
    if (ncons == 2) {
      if (cons2 == stb.k0 && !func_in(op1))
        return op2;
      if (cons2 == stb.k1)
        return op1;
      if (con2v1 == 0 && con2v2 == 2)
        /* assertion: no need for cse since sched treats multiple uses of
         * the same function ili as one call.
         */
        return ad2ili(IL_KADD, op1, op1);
      if (ILI_OPC(op1) == opc && ILI_OPC(ilix = ILI_OPND(op1, 2)) == IL_KCON) {
        /*  (i * c1) * c2  --->  i * (c1*c2)  */
        GETVALI64(num1, ILI_OPND(ilix, 1));
        GETVALI64(num2, cons2);
        if (opc == IL_KMUL)
          mul64(num1.numi, num2.numi, res.numi);
        else
          umul64(num1.numu, num2.numu, res.numu);
        op2 = ad1ili(IL_KCON, getcon(res.numi, DT_INT8));
        ilix = ad2ili(opc, (int)ILI_OPND(op1, 1), op2);
        return ilix;
      }
      if (ILI_OPC(op1) == IL_KNEG || ILI_OPC(op1) == IL_UKNEG) {
        /*  (-i) * c  --->  i * (-c)  */
        GETVALI64(num2, cons2);
        neg64(num2.numi, res.numi);
        op2 = ad1ili(IL_KCON, getcon(res.numi, DT_INT8));
        ilix = ad2ili(opc, (int)ILI_OPND(op1, 1), op2);
        return ilix;
      }
      if (opc == IL_KMUL && con2v1 == -1 && con2v2 == (int)0xffffffff)
        return ad1ili(IL_KNEG, op1);
    } else if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      if (opc == IL_KMUL)
        mul64(num1.numi, num2.numi, res.numi);
      else
        umul64(num1.numu, num2.numu, res.numu);
      goto add_kcon;
    }
    /* KMUL KNEG x, y --> KNEG KMUL x,y */
    if (ILI_OPC(op1) == IL_KNEG && ILI_OPC(op2) == IL_KNEG) {
      op1 = ILI_OPND(op1, 1);
      op2 = ILI_OPND(op2, 1);
      break;
    }
    if (ILI_OPC(op1) == IL_KNEG) {
      return ad1ili(IL_KNEG, ad2ili(IL_KMUL, ILI_OPND(op1, 1), op2));
    }
    if (ILI_OPC(op2) == IL_KNEG) {
      return ad1ili(IL_KNEG, ad2ili(IL_KMUL, op1, ILI_OPND(op2, 1)));
    }
    break;

  case IL_FMUL:
    if (!flg.ieee) {
      if (ncons == 2) {
        if (is_flt0(cons2) && !func_in(op1))
          return op2;
        if (cons2 == stb.flt1)
          return op1;
        if (cons2 == stb.flt2) {
          /* assertion: no need for cse since sched treats multiple uses
           * of the same function ili as one call.
           */
          return ad2ili(IL_FADD, op1, op1);
        }
      } else if (ncons == 3) {
        xfmul(con1v2, con2v2, &res.numi[1]);
        /* don't constant fold if error occurred */
        if (gbl.fperror_status)
          break;
        goto add_rcon;
      }
    }
    /* FMUL FNEG x, FNEG y --> FMUL x,y */
    if (ILI_OPC(op1) == IL_FNEG && ILI_OPC(op2) == IL_FNEG) {
      op1 = ILI_OPND(op1, 1);
      op2 = ILI_OPND(op2, 1);
      break;
    }
    /* FMUL FNEG x, y --> FNEG FMUL x,y */
    if (ILI_OPC(op1) == IL_FNEG) {
      return ad1ili(IL_FNEG, ad2ili(IL_FMUL, ILI_OPND(op1, 1), op2));
    }
    /* FMUL x, FNEG y --> FNEG FMUL x,y */
    if (ILI_OPC(op2) == IL_FNEG) {
      return ad1ili(IL_FNEG, ad2ili(IL_FMUL, op1, ILI_OPND(op2, 1)));
    }
    break;

  case IL_DMUL:
    if (!flg.ieee) {
      if (ncons == 2) {
        if (is_dbl0(cons2) && !func_in(op1))
          return op2;
        if (cons2 == stb.dbl1)
          return op1;
        if (cons2 == stb.dbl2) {
          /* assertion: no need for cse since sched treats multiple uses
           * of the same function ili as one call.
           */
          return ad2ili(IL_DADD, op1, op1);
        }
      } else if (ncons == 3) {
        GETVAL64(num1, cons1);
        GETVAL64(num2, cons2);
        xdmul(num1.numd, num2.numd, res.numd);
        /* don't constant fold if error occurred */
        if (gbl.fperror_status)
          break;
        goto add_dcon;
      }
    }
    /* DMUL DNEG x, y --> DNEG DMUL x,y */
    if (ILI_OPC(op1) == IL_DNEG && ILI_OPC(op2) == IL_DNEG) {
      op1 = ILI_OPND(op1, 1);
      op2 = ILI_OPND(op2, 1);
      break;
    }
    if (ILI_OPC(op1) == IL_DNEG) {
      return ad1ili(IL_DNEG, ad2ili(IL_DMUL, ILI_OPND(op1, 1), op2));
    }
    if (ILI_OPC(op2) == IL_DNEG) {
      return ad1ili(IL_DNEG, ad2ili(IL_DMUL, op1, ILI_OPND(op2, 1)));
    }
    break;
  case IL_SCMPLXMUL:
    if (ncons == 1 && IS_FLT0(con1v1) && IS_FLT0(con1v2) && !func_in(op2))
      return op1;
    if (ncons == 2 && IS_FLT0(con2v1) && IS_FLT0(con2v2) && !func_in(op1))
      return op2;
    else if (ncons == 3) { /* should be done by front end already */
      if (IS_FLT0(con1v1) && IS_FLT0(con1v2))
        return op1;
      if (IS_FLT0(con2v1) && IS_FLT0(con2v2))
        return op2;
    } else {
      op1 = ilip->opnd[0];
      op2 = ilip->opnd[1];
      if (ILI_OPC(op1) == IL_SPSP2SCMPLXI0 &&
          ILI_OPC(op2) == IL_SPSP2SCMPLXI0) {
        int ilir;
        ilir = ad2ili(IL_FMUL, ILI_OPND(op1, 1), ILI_OPND(op2, 1));
        return ad1ili(IL_SPSP2SCMPLXI0, ilir);
      }
    }
    break;
  case IL_DCMPLXMUL:
    /* check if any is of complex is 0  then 0*/
    if (ncons == 1 && IS_DBL0(con1v1) && IS_DBL0(con1v2) && !func_in(op2))
      return op1;
    else if (ncons == 2 && IS_DBL0(con2v1) && IS_DBL0(con2v2) && !func_in(op1))
      return op2;
    else if (ncons == 3) { /* should be done by front end already */
      if (IS_DBL0(con1v1) && IS_DBL0(con1v2))
        return op1;
      if (IS_DBL0(con2v1) && IS_DBL0(con2v2))
        return op2;
    } else {
      op1 = ilip->opnd[0];
      op2 = ilip->opnd[1];
      if (ILI_OPC(op1) == IL_DPDP2DCMPLXI0 &&
          ILI_OPC(op2) == IL_DPDP2DCMPLXI0) {
        int ilir;
        ilir = ad2ili(IL_DMUL, ILI_OPND(op1, 1), ILI_OPND(op2, 1));
        return ad1ili(IL_DPDP2DCMPLXI0, ilir);
      }
    }

    break;

  case IL_FSINCOS:
  case IL_DSINCOS:
  case IL_FNSIN:
  case IL_FNCOS:
  case IL_DNSIN:
  case IL_DNCOS:
    break;
  case IL_QUOREM:
  case IL_NIDIV:
  case IL_NUIDIV:
  case IL_NMOD:
  case IL_NUIMOD:
    break;
  case IL_KQUOREM:
  case IL_NKDIV:
  case IL_NUKDIV:
  case IL_NKMOD:
  case IL_NUKMOD:
    break;
  case IL_IDIV:
    if (ncons == 3 && con2v2 != 0) {
      if (con1v2 == (int)0x80000000 && con2v2 == -1)
        res.numi[1] = 0x80000000;
      else
        res.numi[1] = con1v2 / con2v2;
      goto add_icon;
    }
    if (ncons == 2) {
      if (cons2 == stb.i1)
        return op1;
      if (ILI_OPC(op1) == IL_IMUL && ILI_OPND(op1, 2) == op2)
        return ILI_OPND(op1, 1);
      if ((res.numi[0] = con2v2) == -1)
        return ad1ili(IL_INEG, op1);
      if (res.numi[0] > 0) {
        /*
         * dividing by a constant which may be a power of 2.
         * if it is, call a different routine which is faster than
         * the ordinary divide. The possible powers are 2 thru 30.
         */
        i = _pwr2(res.numi[0], 30);
        if (i) {
#ifndef TM_IDIV2
          /* expand idiv2 as follows:
           *    t0 = (int)op1 >> (n - 1)
           *    t0 = (unsigned)t0 >> (32 - n)
           *    t0 = t0 + op1
           *   res = (int)t0 >> n
           * NOTE: need to buffer op1 with cse -- assumes
           *       postorder traversal by scheduler. Consequently,
           *       if op1 is a INEG the scheduler will see a
           *       cse of op1 before it sees op1 ('t0 + -i' is
           *       transformed into 't0 - i').  Although this is
           *       not harmful under the current sched scheme,
           *       handle INEG so that that an internal error
           *       message is not issued.
           */
          if (ILI_OPC(op1) == IL_INEG) {
            tmp = ad2ili(IL_ARSHIFT, op1, ad_icon((INT)(i - 1)));
            tmp = ad2ili(IL_URSHIFT, tmp, ad_icon((INT)(32 - i)));
            tmp = ad2ili(IL_ISUB, tmp, (int)ILI_OPND(op1, 1));
          } else {
            /* assertion: no need for cse since sched treats
             * multiple uses of the same function ili as one call.
             */
            tmp = ad2ili(IL_ARSHIFT, op1, ad_icon((INT)(i - 1)));
            tmp = ad2ili(IL_URSHIFT, tmp, ad_icon((INT)(32 - i)));
            tmp = ad2ili(IL_IADD, op1, tmp);
          }
          ilix = ad2ili(IL_ARSHIFT, tmp, ad_icon((INT)i));
          return ad2altili(opc, op1, op2, ilix);

#else  /* defined(TM_IDIV2) */
          ilix = ad2ili(IL_IDIV2, op1, i);
          return ad2altili(opc, op1, op2, ilix);
#endif /* #ifndef TM_IDIV2 */
        }
      }
      if (!XBIT(87, 0x1) && !XBIT(6, 0x400)) {
        ilix = reciprocal_division(op1, res.numi[0], 1);
        if (ilix)
          return ad2altili(opc, op1, op2, ilix);
      }
    }
#ifndef TM_IDIV
/* divide-by-constant is handled by reciprocal division */
#ifdef ALT_I_IDIV
    if (XBIT(122, 0x80))
      ilix = ad2func_int(IL_QJSR, ALT_I_IDIV, op1, op2);
    else
#endif
      ilix = ad2func_int(IL_QJSR, MTH_I_IDIV, op1, op2);
    return ilix;
#endif

#ifdef TM_IDIV
    break;
#endif

  case IL_IDIVZR:
    if (ncons == 2) {
      if (cons2 == stb.i1)
        return op1;
      if (ILI_OPC(op1) == IL_IMUL && ILI_OPND(op1, 2) == op2)
        return ILI_OPND(op1, 1);
      if (con2v2 > 0) {
        /*
         * dividing by a constant which may be a power of 2.
         * if it is, call a different routine which is faster than
         * the ordinary divide. The possible powers are 2 thru 30.
         */
        i = _pwr2(con2v2, 30);
        if (i) {
          /* expand idiv2 as
           *   res = (int)t0 >> n
           */
          ilix = ad2ili(IL_ARSHIFT, op1, ad_icon((INT)i));
          return ilix;
        }
      }
    }
    ilix = ad2ili(IL_IDIV, op1, op2);
    return ilix;

#ifdef TM_IDIV2
  case IL_IDIV2:
    break;
#endif

  case IL_UIDIV:
    if (ncons == 3 && con2v2 != 0) {
      tmp = xudiv(con1v2, con2v2, &res.numu[0]);
      return ad_icon((INT)res.numu[0]);
    }
    if (ncons == 2) {
      if (cons2 == stb.i1)
        return op1;
      res.numi[0] = con2v2;
      /*
       * dividing by a constant which may be a power of 2. if
       * it is, call a different routine which is faster than the
       * ordinary divide.  The possible powers are 2 through 31.
       */
      i = _pwr2(res.numi[0], 31);
      if (i) {
        ilix = ad2ili(IL_URSHIFT, op1, ad_icon((INT)i));
        return ilix;
      }
      if (!XBIT(87, 0x1) && !XBIT(6, 0x400)) {
        ilix = reciprocal_division(op1, res.numi[0], 0);
        if (ilix)
          return ad2altili(opc, op1, op2, ilix);
      }
    }
    /*
     * if the divisor is shifting 1 left, then replace the divide
     * with an unsigned right shift.
     */
    ilix = _lshift_one(op2);
    if (ilix)
      return ad2ili(IL_URSHIFT, op1, ilix);

#ifdef TM_UIDIV
    break;
#else /* #ifndef TM_UIDIV */
#ifdef ALT_I_UIDIV
    if (XBIT(122, 0x80))
      ilix = ad2func_int(IL_QJSR, ALT_I_UIDIV, op1, op2);
    else
#endif
      ilix = ad2func_int(IL_QJSR, MTH_I_UIDIV, op1, op2);
    return ilix;
#endif

  case IL_KDIV:
    if (ncons == 3 && cons2 != stb.k0) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      div64(num1.numi, num2.numi, res.numi);
      goto add_kcon;
    }
    if (ncons == 2) {
      if (cons2 == stb.k1 || (con2v1 == 0 && con2v2 == 1))
        return op1;
      if (ILI_OPC(op1) == IL_KMUL && ILI_OPND(op1, 2) == op2)
        return ILI_OPND(op1, 1);
      if (con2v1 == -1 && con2v2 == (int)0xffffffff)
        return ad1ili(IL_KNEG, op1);
      if (con2v1 >= 0) {
        /*
         * dividing by a constant which may be a power of 2.
         * if it is, call a different routine which is faster than
         * the ordinary divide. The possible powers are 2 thru 62.
         */
        i = _kpwr2(con2v1, con2v2, 62);
        if (i) {
          /* expand kdiv2 as follows:
           *    t0 = op1 >> (n - 1)
           *    t0 = (unsigned)t0 >> (64 - n)
           *    t0 = t0 + op1
           *   res = t0 >> n
           */
          if (ILI_OPC(op1) == IL_KNEG) {
            tmp = ad2ili(IL_KARSHIFT, op1, ad_icon((INT)(i - 1)));
            tmp = ad2ili(IL_KURSHIFT, tmp, ad_icon((INT)(64 - i)));
            tmp = ad2ili(IL_KSUB, tmp, (int)ILI_OPND(op1, 1));
          } else {
            /* assertion: no need for cse since sched treats
             * multiple uses of the same function ili as one call.
             */
            tmp = ad2ili(IL_KARSHIFT, op1, ad_icon((INT)(i - 1)));
            tmp = ad2ili(IL_KURSHIFT, tmp, ad_icon((INT)(64 - i)));
            tmp = ad2ili(IL_KADD, op1, tmp);
          }
          ilix = ad2ili(IL_KARSHIFT, tmp, ad_icon((INT)i));
          return ad2altili(opc, op1, op2, ilix);
        }
      }

      if (!XBIT(87, 0x1) && !XBIT(6, 0x400)) {
        res.numi[0] = con2v1;
        res.numi[1] = con2v2;
        ilix = reciprocal_division_64(op1, res.numi, 1);
        if (ilix)
          return ad2altili(opc, op1, op2, ilix);
      }
    }
    break;

  case IL_KDIVZR:
    /*
     * 64-bit signed integer divide when it's known that the remainder
     * is zero, such as the divide performed for a pointer subtract.
     */
    if (ncons == 2) {
      if (cons2 == stb.k1)
        return op1;
      if (ILI_OPC(op1) == IL_KMUL && ILI_OPND(op1, 2) == op2)
        return ILI_OPND(op1, 1);
      if (con2v1 >= 0) {
        /*
         * dividing by a constant which may be a power of 2.
         * if it is just shift the dividend.  The possible powers
         * are 2 thru 62.
         */
        i = _kpwr2(con2v1, con2v2, 62);
        if (i) {
          /* expand as
           *    res = op1 >> n
           */
          ilix = ad2ili(IL_KARSHIFT, op1, ad_icon((INT)i));
          return ilix;
        }
      }
    }
    ilix = ad2ili(IL_KDIV, op1, op2);
    return ilix;

  case IL_UKDIV:
    if (ncons == 3 && cons2 != stb.k0) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      udiv64(num1.numu, num2.numu, res.numu);
      goto add_kcon;
    }
    if (ncons == 2) {
      /* if (cons2 == stb.k1)*/
      if (cons2 == stb.k1 || (con2v1 == 0 && con2v2 == 1))
        return op1;
      if (ILI_OPC(op1) == IL_UKMUL && ILI_OPND(op1, 2) == op2)
        return ILI_OPND(op1, 1);
#if defined(TARGET_X8664)
      /*
       * dividing by a constant which may be a power of 2. if
       * it is, call a different routine which is faster than the
       * ordinary divide.  The possible powers are 2 through 63.
       */
      i = _kpwr2(con2v1, con2v2, 63);
      if (i) {
        ilix = ad2ili(IL_KURSHIFT, op1, ad_icon((INT)i));
        return ilix;
      }
      if (!XBIT(87, 0x1) && !XBIT(6, 0x400)) {
        res.numi[0] = con2v1;
        res.numi[1] = con2v2;
        ilix = reciprocal_division_64(op1, res.numi, 0);
        if (ilix)
          return ad2altili(opc, op1, op2, ilix);
      }
#endif
    } /* ncons == 2 */
    /*
     * if the divisor is shifting 1 left, then replace the divide
     * with an unsigned right shift.
     */
    ilix = _lshift_one(op2);
    if (ilix)
      return ad2ili(IL_KURSHIFT, op1, ilix);
    break;

  case IL_FDIV:
#ifdef TM_FDIV /*{ hardware divide */
    /* hardware divide present */
    if (ncons == 3 && !is_flt0(cons2)) {
      xfdiv(con1v2, con2v2, &res.numi[1]);
      goto add_rcon;
    }
    if (!flg.ieee) {
      if (XBIT(15, 0x1) && (ncons & 2) && !is_flt0(cons2)) {
        /*  x / y --> x * (1 / y)  */
        res.numi[0] = 0;
        xfdiv(CONVAL2G(stb.flt1), con2v2, &res.numi[1]);
        ilix = ad1ili(IL_FCON, getcon(res.numi, DT_FLOAT));
        return ad2ili(IL_FMUL, ilix, op1);
      }
      if (XBIT(15, 0x4) && (!((ncons & 1) && (cons1 == stb.flt1)))) {
        /*  x / y --> x * (1 / y)  */
        ilix = ad2ili(IL_FDIV, ad1ili(IL_FCON, stb.flt1), op2);
        return ad2ili(IL_FMUL, ilix, op1);
      }
#if defined(TARGET_X8664) || defined(TARGET_POWER)
      if (XBIT(183, 0x10000)) {
        if (XBIT(15, 0x40000000) && ILI_OPC(op2) == IL_FSQRT) {
          /*
           * Just use the approximating reciprocal sqrt instruction
           * for computing rsqrt:
           *    y/x -> y * rqsrt(x)
           */
          tmp1 = ILI_OPND(op2, 1); /* x  */
          tmp1 = ad1ili(IL_RSQRTSS, tmp1);
          ilix = ad2ili(IL_FMUL, op1, tmp1);
          return ilix;
        }
        if (XBIT(15, 0x10000000)) {
          /*
           * Just multiply the first operand by the approximating
           * reciprocal instruction
           */
          tmp1 = ad1ili(IL_RCPSS, op2);
          ilix = ad2ili(IL_FMUL, op1, tmp1);
          return ilix;
        }
        if (XBIT(15, 0x10)) {
          if (!XBIT(15, 0x20000) && ILI_OPC(op2) == IL_FSQRT) {
            /*
             * Newton's appx for recip sqrt:
             *   x1 = (3.0*x0 - x*x0**3)/2.0
             * or
             *   x1 = (3.0 - x*x0*x0)*x0*.5
             * or
             *   x1 = (x*x0*x0 - 3.0)*x0*(-.5)
             */
            tmp1 = _frsqrt(ILI_OPND(op2, 1));
            ilix = ad2ili(IL_FMUL, op1, tmp1);
            return ilix;
          }
          if (!XBIT(15, 0x40000)) {
            ilix = _newton_fdiv(op1, op2);
            return ilix;
          }
        }
      }
#endif /* defined(TARGET_X8664) || defined(TARGET_POWER) */
    }
    break;
#endif /*} hardware divide */
#ifndef TM_FDIV /*{ no hardware divide */
#ifdef TM_FRCP  /*{ mult - recip */
    /* perform divide by reciprocal approximation */
    if (flg.ieee) {
      if (ncons == 3 && !is_flt0(cons2)) {
        fdiv(con1v2, con2v2, res.numi[1]);
        goto add_rcon;
      }
      op1 = ad3ili(IL_DASP, op1, SP(0), ad1ili(IL_NULL, 0));
      op2 = ad3ili(IL_DASP, op2, SP(1), op1);
      ilix = ad2ili(IL_QJSR, _mkfunc(MTH_I_RDIV), op2);
      ilix = ad2ili(IL_DFRSP, ilix, SP(0));
      return ilix;
    }
    /*  WARNING: since op2 must be cse'd, the ili must be generated
     *  so that the scheduler sees op2 before its cse use.  Currently,
     *  the scheduler performs a postorder traversal.
     */
    tmp1 = ad1ili(IL_FCON, stb.flt2);
    /* assertion: no need for cse since sched treats multiple uses of
     * the same function ili as one call.
     */
    ilix = ad1ili(IL_FRCP, op2);

    tmp = ad2ili(IL_FMUL, op2, ilix);
    tmp = ad2ili(IL_FSUB, tmp1, tmp);
    ilix = ad2ili(IL_FMUL, ilix, tmp);

    tmp = ad2ili(IL_FMUL, op2, ilix);
    tmp = ad2ili(IL_FSUB, tmp1, tmp);
    ilix = ad3ili(IL_FNEWT, (int)ilip->opnd[1], tmp, ilix);
    ilix = ad2ili(IL_FMUL, op1, ilix);
    return ilix;

  case IL_FNEWT:
    /* since constant folding of the frcp instruction is not performed,
     * the 2nd & 3rd operands are never constants
     */
    if (ncons == 1) {
      frcp(con1v2, res.numi[1]);
      goto add_rcon;
    }
    newili.opnd[2] = ilip->opnd[2]; /* get 3rd operand */
    break;
#else /*} end: mult - recip */

    op1 = ad3ili(IL_DASP, op1, SP(0), ad1ili(IL_NULL, 0));
    op2 = ad3ili(IL_DASP, op2, SP(1), op1);
    ilix = ad2ili(IL_QJSR, _mkfunc(MTH_I_RDIV), op2);
    ilix = ad2ili(IL_DFRSP, ilix, SP(0));
    return ilix;
#endif
#endif /*} end: no hardware divide */

  case IL_DDIV:
#ifdef TM_DDIV /*{ hardware divide present */
    if (ncons == 3 && !is_dbl0(cons2)) {
      GETVAL64(num1, cons1);
      GETVAL64(num2, cons2);
      xddiv(num1.numd, num2.numd, res.numd);
      goto add_dcon;
    }
    if (!flg.ieee) {
      if (XBIT(15, 0x1) && (ncons & 2) && !is_dbl0(cons2)) {
        /*  x / y --> x * (1 / y)  */
        GETVAL64(num1, stb.dbl1);
        GETVAL64(num2, cons2);
        xddiv(num1.numd, num2.numd, res.numd);
        ilix = ad1ili(IL_DCON, getcon(res.numi, DT_DBLE));
        return ad2ili(IL_DMUL, ilix, op1);
      } else if (XBIT(15, 0x4) && (!((ncons & 1) && (cons1 == stb.dbl1)))) {
        /*  x / y --> x * (1 / y)  */
        ilix = ad2ili(IL_DDIV, ad1ili(IL_DCON, stb.dbl1), op2);
        return ad2ili(IL_DMUL, ilix, op1);
      }
    }
    break;
#endif /*} hardware divide */

#ifndef TM_DDIV /*{ no hardware divide */
#ifdef TM_DRCP  /*{ mult - recip */
    /* perform divide by reciprocal approximation */
    if (flg.ieee) {
      if (ncons == 3 && !is_dbl0(cons2)) {
        GETVAL64(num1, cons1);
        GETVAL64(num2, cons2);
        ddiv(num1.numd, num2.numd, res.numd);
        goto add_dcon;
      }
      op1 = ad3ili(IL_DADP, op1, DP(0), ad1ili(IL_NULL, 0));
      op2 = ad3ili(IL_DADP, op2, DP(1), op1);
      ilix = ad2ili(IL_QJSR, _mkfunc(MTH_I_DDIV), op2);
      ilix = ad2ili(IL_DFRDP, ilix, DP(0));
      return ilix;
    }
    tmp1 = ad1ili(IL_DCON, stb.dbl2);
    /* assertion: no need for cse since sched treats multiple uses of
     * the same function ili as one call.
     */
    ilix = ad1ili(IL_DRCP, op2);
    for (i = 2; i > 0; i--) {
      tmp = ad2ili(IL_DMUL, op2, ilix);
      tmp = ad2ili(IL_DSUB, tmp1, tmp);
      ilix = ad2ili(IL_DMUL, ilix, tmp);
    }
    tmp = ad2ili(IL_DMUL, op2, ilix);
    tmp = ad2ili(IL_DSUB, tmp1, tmp);
    ilix = ad3ili(IL_DNEWT, (int)ilip->opnd[1], tmp, ilix);
    ilix = ad2ili(IL_DMUL, op1, ilix);
    return ilix;

  case IL_DNEWT:
    /* since constant folding of the drcp instruction is not performed,
     * the 2nd & 3rd operands are never constants
     */
    if (ncons == 1) {
      /* TBD - need to constant fold by  recip(cons1) */
      GETVAL64(num2, cons1);
      drcp(num2.numd, res.numd);
      goto add_dcon;
    }
    newili.opnd[2] = ilip->opnd[2]; /* get 3rd operand */
    break;
#else /*} end: mult - recip */

    op1 = ad3ili(IL_DADP, op1, DP(0), ad1ili(IL_NULL, 0));
    op2 = ad3ili(IL_DADP, op2, DP(1), op1);
    ilix = ad2ili(IL_QJSR, _mkfunc(MTH_I_DDIV), op2);
    ilix = ad2ili(IL_DFRDP, ilix, DP(0));
    return ilix;
#endif
#endif /*} no hardware divide */

#if defined(TARGET_X8664)
  case IL_SCMPLXDIV:
    ilix = ad2func_cmplx(IL_QJSR, fast_math("div", 's', 'c', FMTH_I_CSDIV), op1,
                         op2);
    return ad2altili(opc, op1, op2, ilix);
  case IL_DCMPLXDIV:
    ilix = ad2func_cmplx(IL_QJSR, fast_math("div", 's', 'z', FMTH_I_CDDIV), op1,
                         op2);
    return ad2altili(opc, op1, op2, ilix);
#endif
  case IL_MOD:
    if (ncons == 3 && con2v2 != 0) {
      if (con1v2 == (int)0x80000000 && con2v2 == -1)
        res.numi[1] = 0;
      else
        res.numi[1] = con1v2 % con2v2;
      goto add_icon;
    }
    if (ncons == 2) {
      if (cons2 == stb.i1)
        return ad1ili(IL_ICON, stb.i0);

      /* a % con = a - (a / cons) * cons; catch power of two
       * optimization.
       */
      res.numi[0] = con2v2;
      if (res.numi[0] > 0 && _pwr2(res.numi[0], 30)) {
        tmp = ad2ili(IL_IDIV, op1, op2);
        tmp = ad2ili(IL_IMUL, tmp, op2);
        ilix = ad2ili(IL_ISUB, op1, tmp);
        return ilix;
      }
    }
#ifndef TM_IMOD
    ilix = ad2func_int(IL_QJSR, MTH_I_IMOD, op1, op2);
    return ilix;
#else
    if (ncons == 2 && !XBIT(6, 0x400)) {
      ilix = reciprocal_mod(op1, res.numi[0], 1);
      if (ilix)
        return ad2altili(opc, op1, op2, ilix);
    }
    break;
#endif

  case IL_KMOD:
    tmp = ad2ili(IL_KDIV, op1, op2);
    tmp = ad2ili(IL_KMUL, tmp, op2);
    ilix = ad2ili(IL_KSUB, op1, tmp);
    return ilix;

  case IL_UIMOD:
    if (ncons == 3) {
      tmp = xumod(con1v2, con2v2, &res.numu[0]);
      return ad_icon((INT)res.numu[0]);
    }
    if (ncons == 2) {
      if (cons2 == stb.i1)
        return ad1ili(IL_ICON, stb.i0);

      if ((res.numi[0] = con2v2) > 0) {
        /*
         * mod by a constant which may be a power of 2.
         * if it is, generate an and operation. The possible powers
         * are 2 through 30.
         */
        i = _pwr2(res.numi[0], 30);
        if (i) {
          /* expand uimod as follows:
           *   res = op1 & (con - 1)
           */
          ilix = ad2ili(IL_AND, op1, ad_icon(res.numi[0] - 1));
          return ilix;
        }
      }
    }
    /*
     * if the divisor is shifting 1 left, then replace the mod
     * with an and.
     */
    if (_lshift_one(op2)) {
      op2 = ad2ili(IL_ISUB, op2, ILI_OPND(op2, 1));
      return ad2ili(IL_AND, op1, op2);
    }
#ifndef TM_UIMOD
#error TM_UIMOD undefined
    ilix = ad2func_int(IL_QJSR, MTH_I_UIMOD, op1, op2);
    return ilix;
#else
    if (ncons == 2 && !XBIT(6, 0x400)) {
      ilix = reciprocal_mod(op1, res.numi[0], 0);
      if (ilix)
        return ad2altili(opc, op1, op2, ilix);
    }
    break;
#endif
  case IL_KUMOD:
    if (ncons == 2) {
      if (cons2 == stb.k1)
        return ad1ili(IL_KCON, stb.k0);

      if (((res.numu[0] = con2v2) > 0) && (res.numu[1] = con2v1) == 0) {
        /*
         * mod by a constant which may be a power of 2.
         * if it is, generate an and operation. The possible powers
         * are 2 through 31.
         */
        i = _kpwr2(con2v1, con2v2, 31);
        if (i) {
          /* expand uimod as follows:
           *   res = op1 & (con - 1)
           */
          ilix = ad2ili(IL_KAND, op1, ad_kcon(res.numu[1], res.numu[0] - 1));
          return ilix;
        }
      }
      if (!XBIT(6, 0x400)) {
        res.numi[0] = con2v1;
        res.numi[1] = con2v2;
        ilix = reciprocal_mod_64(op1, (res.numi), 0);
        if (ilix)
          return ad2altili(opc, op1, op2, ilix);
      }
    }
    /*
     * if the divisor is shifting 1 left, then replace the mod
     * with an and.
     */
    if (_lshift_one(op2)) {
      op2 = ad2ili(IL_KSUB, op2, ILI_OPND(op2, 1));
      return ad2ili(IL_KAND, op1, op2);
    }
    tmp = ad2ili(IL_UKDIV, op1, op2);
    tmp = ad2ili(IL_UKMUL, tmp, op2);
    ilix = ad2ili(IL_UKSUB, op1, tmp);
    return ilix;

#ifdef TM_FRCP
  case IL_FRCP:
    /* Can't constant fold until there is a utility routine */
    break;
#endif
#ifdef TM_DRCP
  case IL_DRCP:
    /* Can't constant fold until there is a utility routine */
    break;
#endif

  case IL_FMOD:
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(fast_math("mod", 's', 's', FMTH_I_AMOD), "f pure",
                         DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("mod", 's', 's', FMTH_I_AMOD),
                     2, op1, op2);
    } else {
      (void)mk_prototype(MTH_I_AMOD, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_AMOD, 2, op1, op2);
    }
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
#endif
    (void)mk_prototype(MTH_I_AMOD, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_AMOD, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
    break;
  case IL_DMOD:
    if (!flg.ieee) {
#ifdef TARGET_X8664
      (void)mk_prototype(fast_math("mod", 's', 'd', FMTH_I_DMOD), "f pure",
                         DT_DBLE, 2, DT_DBLE, DT_DBLE);
#else
      (void)mk_prototype(MTH_I_DMOD, "f pure", DT_DBLE, 2, DT_DBLE, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DMOD, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
#endif
      ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("mod", 's', 'd', FMTH_I_DMOD),
                     2, op1, op2);
    } else {
      (void)mk_prototype(MTH_I_DMOD, "f pure", DT_DBLE, 2, DT_DBLE, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DMOD, 2, op1, op2);
    }
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QMOD:
    if (!flg.ieee) {
#ifdef TARGET_X8664
      (void)mk_prototype(fast_math("mod", 's', 'd', FMTH_I_DMOD), "f pure",
                         DT_QUAD, 2, DT_QUAD, DT_QUAD);
#else
      (void)mk_prototype(MTH_I_QMOD, "f pure", DT_QUAD, ARGS_NUMBER, DT_QUAD,
                         DT_QUAD);
      ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QMOD, ARGS_NUMBER, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
#endif
      ilix = ad_func(IL_DFRQP, IL_QJSR, fast_math("mod", 's', 'd', FMTH_I_DMOD),
                     ARGS_NUMBER, op1, op2);
    } else {
      (void)mk_prototype(MTH_I_QMOD, "f pure", DT_QUAD, ARGS_NUMBER, DT_QUAD,
                         DT_QUAD);
      ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QMOD, ARGS_NUMBER, op1, op2);
    }
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
#endif

  case IL_FSINH:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_sinh, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    if (!flg.ieee) {
#ifdef TARGET_X8664
      (void)mk_prototype(fast_math("sinh", 's', 's', FMTH_I_SINH), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
#else
      (void)mk_prototype(MTH_I_SINH, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_SINH, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;

#endif
      ilix = ad_func(IL_DFRSP, IL_QJSR,
                     fast_math("sinh", 's', 's', FMTH_I_SINH), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_SINH, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_SINH, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_DSINH:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_sinh, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    if (!flg.ieee) {
#ifdef TARGET_X8664
      (void)mk_prototype(fast_math("sinh", 's', 'd', FMTH_I_DSINH), "f pure",
                         DT_DBLE, 1, DT_DBLE);
#else
      (void)mk_prototype(MTH_I_DSINH, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DSINH, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
#endif
      ilix = ad_func(IL_DFRDP, IL_QJSR,
                     fast_math("sinh", 's', 'd', FMTH_I_DSINH), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_DSINH, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DSINH, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_FCOSH:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_cosh, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    if (!flg.ieee) {
#ifdef TARGET_X8664
      (void)mk_prototype(fast_math("cosh", 's', 's', FMTH_I_COSH), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
#else
      (void)mk_prototype(MTH_I_COSH, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_COSH, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
#endif
      ilix = ad_func(IL_DFRSP, IL_QJSR,
                     fast_math("cosh", 's', 's', FMTH_I_COSH), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_COSH, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_COSH, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_DCOSH:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_cosh, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    if (!flg.ieee) {
#ifdef TARGET_X8664
      (void)mk_prototype(fast_math("cosh", 's', 'd', FMTH_I_DCOSH), "f pure",
                         DT_DBLE, 1, DT_DBLE);
#else
      (void)mk_prototype(MTH_I_DCOSH, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DCOSH, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
#endif
      ilix = ad_func(IL_DFRDP, IL_QJSR,
                     fast_math("cosh", 's', 'd', FMTH_I_DCOSH), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_DCOSH, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DCOSH, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_FTANH:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_tanh, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_TANH, "pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_TANH, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_DTANH:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_tanh, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_DTANH, "pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DTANH, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_ICMP:
    newili.opnd[2] = ilip->opnd[2];
    cond = CCRelationILIOpnd(ilip, 2);
    if (ncons == 1) {
      if (ILI_OPC(op2) == IL_IADD &&
          ILI_OPC(tmp = ILI_OPND(op2, 2)) == IL_ICON) {

        /* c1 :: i + c2  -->  c1 - c2 :: i  */

        if (isub_ovf(con1v2, CONVAL2G(ILI_OPND(tmp, 1)), &res.numi[1]))
          break;
        return ad3ili(IL_ICMP, ad_icon(res.numi[1]), ILI_OPND(op2, 1), cond);
      }
      if (ILI_OPC(op2) == IL_ISUB &&
          ILI_OPC(tmp = ILI_OPND(op2, 2)) == IL_ICON) {

        /* c1 :: i - c2  -->  c1 + c2 :: i  */

        if (iadd_ovf(con1v2, CONVAL2G(ILI_OPND(tmp, 1)), &res.numi[1]))
          break;
        return ad3ili(IL_ICMP, ad_icon(res.numi[1]), ILI_OPND(op2, 1), cond);
      }
      if (cons1 == stb.i0)
        /* 0 :: i  -->  i rev(::) 0 */
        return ad2ili(IL_ICMPZ, op2, commute_cc(cond));
      if (cons1 == stb.i1) {
        if (cond == CC_LE)
          /* 1 LE x  -->  x GT z */
          return ad2ili(IL_ICMPZ, op2, CC_GT);
        if (cond == CC_GT)
          /* 1 GT x  -->  x LE z */
          return ad2ili(IL_ICMPZ, op2, CC_LE);
      }
      if (con1v2 >= 1 && is_zero_one(op2)) {
        /* low-quality range analysis */
        switch (cond) {
        default:
          break;
        case CC_EQ:
        case CC_LE:
          if (con1v2 > 1) {
            /* 2 <= x is false */
            if (func_in(op2))
              break;
            return ad1ili(IL_ICON, stb.i0);
          }
          return op2; /* 1 <= x becomes x */
        case CC_NE:
        case CC_GT:
          if (con1v2 > 1) {
            /* 2 > x is true */
            if (func_in(op2))
              break;
            return ad1ili(IL_ICON, stb.i1);
          }
          /* 1 > x becomes x == 0 */
          return ad2ili(IL_ICMPZ, op2, CC_EQ);
        case CC_LT: /* 1 < x always false */
          if (func_in(op2))
            break;
          return ad1ili(IL_ICON, stb.i0);
        case CC_GE: /* 1 >= x always true */
          if (func_in(op2))
            break;
          return ad1ili(IL_ICON, stb.i1);
        }
      }
    } else if (ncons == 2) {
      if (ILI_OPC(op1) == IL_IADD &&
          ILI_OPC(tmp = ILI_OPND(op1, 2)) == IL_ICON) {

        /* i + c1 :: c2  -->  i :: c2 - c1  */

        if (isub_ovf(con2v2, CONVAL2G(ILI_OPND(tmp, 1)), &res.numi[1]))
          break;
        return ad3ili(IL_ICMP, ILI_OPND(op1, 1), ad_icon(res.numi[1]), cond);
      }
      if (ILI_OPC(op1) == IL_ISUB &&
          ILI_OPC(tmp = ILI_OPND(op1, 2)) == IL_ICON) {

        /* i - c1 :: c2  -->  i :: c2 + c1  */

        if (iadd_ovf(con2v2, CONVAL2G(ILI_OPND(tmp, 1)), &res.numi[1]))
          break;
        return ad3ili(IL_ICMP, ILI_OPND(op1, 1), ad_icon(res.numi[1]), cond);
      }
      if (cons2 == stb.i0)
        return ad2ili(IL_ICMPZ, op1, cond);
      if (cons2 == stb.i1) {
        if (cond == CC_GE)
          /* x GE 1  -->  x GT z */
          return ad2ili(IL_ICMPZ, op1, CC_GT);
        if (cond == CC_LT)
          /* x LT 1  -->  x LE z */
          return ad2ili(IL_ICMPZ, op1, CC_LE);
      }
      if (con2v2 >= 1 && is_zero_one(op1)) {
        /* low-quality range analysis */
        switch (cond) {
        default:
          break;
        case CC_EQ:
        case CC_GE:
          if (con2v2 > 1) {
            /* x >= 2 is false */
            if (func_in(op1))
              break;
            return ad1ili(IL_ICON, stb.i0);
          }
          return op1; /* x >= 1 becomes x */
        case CC_NE:
        case CC_LT:
          if (con2v2 > 1) {
            /* x < 2 is true */
            if (func_in(op1))
              break;
            return ad1ili(IL_ICON, stb.i1);
          }
          /* x < 1 becomes x == 0 */
          return ad2ili(IL_ICMPZ, op1, CC_EQ);
        case CC_GT: /* x > 1 always false */
          if (func_in(op1))
            break;
          return ad1ili(IL_ICON, stb.i0);
        case CC_LE: /* x <= 1 always true */
          if (func_in(op1))
            break;
          return ad1ili(IL_ICON, stb.i1);
        }
      }
    } else if (ncons == 3) {
      res.numi[1] = cmp_to_log(icmp(con1v2, con2v2), cond);
      goto add_icon;
    } else if (op1 == op2 && !func_in(op1)) {
      res.numi[1] = cmp_to_log((INT)0, cond);
      goto add_icon;
    }
    break;

  case IL_FMAX:
    if (ncons == 3) {
      if (xfcmp(con1v2, con2v2) > 0)
        return op1;
      return op2;
    }
    if (!flg.ieee)
      return red_minmax(opc, op1, op2);
    break;
  case IL_FMIN:
    if (ncons == 3) {
      if (xfcmp(con1v2, con2v2) < 0)
        return op1;
      return op2;
    }
    if (!flg.ieee)
      return red_minmax(opc, op1, op2);
    break;

  case IL_FCMP:
    newili.opnd[2] = ilip->opnd[2];
#ifdef TM_FCMPZ
    if (ncons == 2 && is_flt0(cons2))
      return ad2ili(IL_FCMPZ, op1, (int)ilip->opnd[2]);
    if (ncons == 1 && is_flt0(cons1))
      return ad2ili(IL_FCMPZ, op2, commute_cc(ilip->opnd[2]));
#else
    if (ncons == 1 && is_flt0(cons1))
      return ad3ili(IL_FCMP, op2, op1,
                    commute_cc(CCRelationILIOpnd(ilip, 2)));
#endif
    if (!flg.ieee) {
      if (ncons == 3) {
        res.numi[1] = cmp_to_log(xfcmp(con1v2, con2v2), ilip->opnd[2]);
        goto add_icon;
      }
      if (op1 == op2 && !func_in(op1)) {
        res.numi[1] = cmp_to_log(0, ilip->opnd[2]);
        goto add_icon;
      }
    }
    break;

  case IL_DMAX:
    if (ncons == 3) {
      GETVAL64(num1, cons1);
      GETVAL64(num2, cons2);
      if (xdcmp(num1.numd, num2.numd) > 0)
        return op1;
      return op2;
    }
    if (!flg.ieee)
      return red_minmax(opc, op1, op2);
    break;
  case IL_DMIN:
    if (ncons == 3) {
      GETVAL64(num1, cons1);
      GETVAL64(num2, cons2);
      if (xdcmp(num1.numd, num2.numd) < 0)
        return op1;
      return op2;
    }
    if (!flg.ieee)
      return red_minmax(opc, op1, op2);
    break;

  case IL_DCMP:
    newili.opnd[2] = ilip->opnd[2];
#ifdef TM_DCMPZ
    if (ncons == 2 && is_dbl0(cons2))
      return ad2ili(IL_DCMPZ, op1, (int)ilip->opnd[2]);
    if (ncons == 1 && is_dbl0(cons1))
      return ad2ili(IL_DCMPZ, op2, commute_cc(ilip->opnd[2]));
#else
    if (ncons == 1 && is_dbl0(cons1))
      return ad3ili(IL_DCMP, op2, op1,
                    commute_cc(CCRelationILIOpnd(ilip, 2)));
#endif
    if (ncons == 2 && ILI_OPC(op1) == IL_DBLE) {
      ilix = DblIsSingle(cons2);
      if (ilix) {
        return ad3ili(IL_FCMP, ILI_OPND(op1, 1), ilix, ilip->opnd[2]);
      }
    }
    if (!flg.ieee) {
      if (ncons == 3) {
        GETVAL64(num1, cons1);
        GETVAL64(num2, cons2);
        res.numi[1] =
            cmp_to_log(xdcmp(num1.numd, num2.numd), (int)ilip->opnd[2]);
        goto add_icon;
      }
      if (op1 == op2 && !func_in(op1)) {
        res.numi[1] = cmp_to_log((INT)0, (int)ilip->opnd[2]);
        goto add_icon;
      }
    }
    break;

  case IL_ACMP:
    newili.opnd[2] = ilip->opnd[2];
    if (ncons == 2 && con2v2 == 0 && con2v1 == 0) {
      return ad2ili(IL_ACMPZ, op1, ilip->opnd[2]);
    } else if (op1 == op2 && !func_in(op1)) {
      res.numi[1] = cmp_to_log(0, ilip->opnd[2]);
      goto add_icon;
    }
    break;

  case IL_UICMP:
    if (ncons == 3) {
      res.numi[1] =
          cmp_to_log(xucmp((UINT)con1v2, (UINT)con2v2), (int)ilip->opnd[2]);
      goto add_icon;
    }

#ifndef TM_UICMP
    /*
     * An out-of-line call is generated for those machines which
     * do not have an unsigned compare instruction.  The result
     * of this routine is:
     *    -1, if op1 < op2,
     *     0, if op1 = op2, or
     *     1, if op1 > op2.
     * This result is compared against zero to complete the code
     * sequence.
     */
    expb.uicmp = _mkfunc(MTH_I_UICMP);
    ilix = ad2func_int(IL_QJSR, MTH_I_UICMP, op1, op2);
    ilix = ad2ili(IL_ICMPZ, ilix, (int)ilip->opnd[2]);
    return ilix;
#endif
#ifdef TM_UICMP
    newili.opnd[2] = ilip->opnd[2];
    /*
     * The simplifications performed for signed int conditionals cannot
     * be performed for unsigned conditionals; e.g.,
     *     i + c1 :: c2
     * isn't always equivalent to
     *     i :: c2 - c1
     * Overflow may occur for 'i + c1', but the result is defined to be
     * congruent mod 2^32.
     * Example:
     *     unsigned i = 0xffffff80;
     *     if (i + 0x80 < 0x100) ...
     * The 'if' evaluates true since the result of the add is defined and
     * is 0.  However,
     *     if (i < 0x80) ...
     * evaluates false.
     */
    if (ncons == 1) {
      if (cons1 == stb.i0)
        /* 0 :: i  -->  i rev(::) 0 */
        return ad2ili(IL_UICMPZ, op2,
                      commute_cc(CCRelationILIOpnd(ilip, 2)));
    } else if (ncons == 2) {
      if (cons2 == stb.i0)
        return ad2ili(IL_UICMPZ, op1, CCRelationILIOpnd(ilip, 2));
    } else if (op1 == op2 && !func_in(op1)) {
      res.numi[1] = cmp_to_log(0, ilip->opnd[2]);
      goto add_icon;
    }
    break;
#endif

  case IL_AND:
    if (ncons == 2) {
      if (cons2 == stb.i0 && !func_in(op1))
        return op2;
      if (con2v2 == (INT)(-1))
        return op1;
      if (opc1 == IL_AND && ILI_OPC(ILI_OPND(op1, 2)) == IL_ICON) {
        op2 = ad2ili(IL_AND, ILI_OPND(op1, 2), op2);
        op1 = ILI_OPND(op1, 1);
        return ad2ili(IL_AND, op1, op2);
      }
    }
    if (ncons == 3) {
      res.numi[1] = con1v2 & con2v2;
      goto add_icon;
    }
    if ((opc1 == IL_NOT || opc1 == IL_UNOT) && ILI_OPND(op1, 1) == op2) {
      /* Complement expression: A & ~A = 0 */
      res.numi[0] = res.numi[1] = 0;
      goto add_icon;
    }
    if ((opc2 == IL_NOT || opc2 == IL_UNOT) && ILI_OPND(op2, 1) == op1) {
      /* Complement expression: A & ~A = 0 */
      res.numi[0] = res.numi[1] = 0;
      goto add_icon;
    }
    goto idempotent_and_or;

  case IL_KAND:
    if (ncons == 2) {
      if (con2v1 == 0 && con2v2 == 0 && !func_in(op1))
        return op2;
      if ((con2v1 == (INT)(-1)) && con2v2 == (INT)(-1))
        return op1;
      if (opc1 == IL_KAND && ILI_OPC(ILI_OPND(op1, 2)) == IL_KCON) {
        op2 = ad2ili(IL_KAND, ILI_OPND(op1, 2), op2);
        op1 = ILI_OPND(op1, 1);
        return ad2ili(IL_KAND, op1, op2);
      }
    }
    if (ncons == 3) {
      GETVALI64(num1, cons1);
      GETVALI64(num2, cons2);
      and64(num1.numi, num2.numi, res.numi);
      goto add_kcon;
    }
    if ((opc1 == IL_KNOT || opc1 == IL_UKNOT) && ILI_OPND(op1, 1) == op2) {
      /* Complement expression: A & ~A = 0 */
      res.numi[0] = res.numi[1] = 0;
      goto add_kcon;
    }
    if ((opc2 == IL_KNOT || opc2 == IL_UKNOT) && ILI_OPND(op2, 1) == op1) {
      /* Complement expression: A & ~A = 0 */
      res.numi[0] = res.numi[1] = 0;
      goto add_kcon;
    }
    goto idempotent_and_or;

  case IL_ROTL:
    break;

  case IL_OR:
    if (ncons == 2) {
      if (cons2 == stb.i0)
        return op1;
      if (con2v2 == (INT)(-1) && !func_in(op1))
        return op2;
    } else if (ncons == 3) {
      res.numi[1] = con1v2 | con2v2;
      goto add_icon;
    }
    if ((opc1 == IL_NOT || opc1 == IL_UNOT) && ILI_OPND(op1, 1) == op2) {
      /* Complement expression: A | ~A = ~0 */
      res.numi[0] = 0;
      res.numi[1] = ~0;
      goto add_icon;
    }
    if ((opc2 == IL_NOT || opc2 == IL_UNOT) && ILI_OPND(op2, 1) == op1) {
      /* Complement expression: A | ~A = ~0 */
      res.numi[0] = 0;
      res.numi[1] = ~0;
      goto add_icon;
    }
    goto idempotent_and_or;

  case IL_KOR:
    if (ncons == 2) {
      if (con2v1 == 0 && con2v2 == 0)
        return op1;
      if ((con2v1 == (INT)(-1)) && con2v2 == (INT)(-1) && !func_in(op1))
        return op2;
    }
    if ((opc1 == IL_KNOT || opc1 == IL_UKNOT) && ILI_OPND(op1, 1) == op2) {
      /* Complement expression: A | ~A = ~0 */
      res.numi[0] = res.numi[1] = ~0;
      goto add_kcon;
    }
    if ((opc2 == IL_KNOT || opc2 == IL_UKNOT) && ILI_OPND(op2, 1) == op1) {
      /* Complement expression: A | ~A = ~0 */
      res.numi[0] = res.numi[1] = ~0;
      goto add_kcon;
    }
    /* FALL THRU to idempotent_and_or */
  idempotent_and_or:
    if (op1 == op2) {
      /* Idempotent expression: A & A = A
       *                        A | A = A
       */
      return op1;
    }
    /* Check to see if we can apply DeMorgan's Law:
     * ~A | ~B = ~(A & B)
     * ~A & ~B = ~(A | B)
     */
    if (opc1 == opc2 && (opc1 == IL_NOT || opc1 == IL_UNOT || opc1 == IL_KNOT ||
                         opc1 == IL_UKNOT)) {

      /* Can apply DeMorgan's Law */

      int tmp, demorgans_opc = 0;

      switch (opc) {
      case IL_AND:
        demorgans_opc = IL_OR;
        break;
      case IL_KAND:
        demorgans_opc = IL_KOR;
        break;
      case IL_OR:
        demorgans_opc = IL_AND;
        break;
      case IL_KOR:
        demorgans_opc = IL_KAND;
        break;
      default:
        assert(0, "addarth: unexpected opcode DeMorgans ", opc, ERR_Fatal);
        break;
      }

      tmp = ad2ili((ILI_OP)demorgans_opc, //???
                   ILI_OPND(op1, 1), ILI_OPND(op2, 1));
      return ad1ili(opc1, tmp);
    }
    goto distributive;

  case IL_XOR:
    if (ncons == 3) {
      res.numi[1] = con1v2 ^ con2v2;
      goto add_icon;
    }
    if (ncons == 2) {
      /* Idempotent expression: A ^ 0 = A */
      if (con2v1 == 0 && con2v2 == 0)
        return op1;
    }
    if (op1 == op2) {
      /* Idempotent expression: A ^ A = 0 */
      res.numi[0] = res.numi[1] = 0;
      goto add_icon;
    }
    if ((opc1 == IL_NOT || opc1 == IL_UNOT) && ILI_OPND(op1, 1) == op2) {
      /* Complement expression: A ^ ~A = ~0 */
      res.numi[0] = 0;
      res.numi[1] = ~0;
      goto add_icon;
    }
    if ((opc2 == IL_NOT || opc2 == IL_UNOT) && ILI_OPND(op2, 1) == op1) {
      /* Complement expression: A ^ ~A = ~0 */
      res.numi[0] = 0;
      res.numi[1] = ~0;
      goto add_icon;
    }
    if (opc1 == IL_AND)
      goto distributive;
    break;

  case IL_KXOR:
    if (ncons == 3) {
      res.numi[0] = con1v1 ^ con2v1;
      res.numi[1] = con1v2 ^ con2v2;
      goto add_kcon;
    }
    if (ncons == 2) {
      /* Idempotent expression: A ^ 0 = A */
      if (con2v1 == 0 && con2v2 == 0)
        return op1;
    }
    if (op1 == op2) {
      /* Idempotent expression: A ^ A = 0 */
      res.numi[0] = res.numi[1] = 0;
      goto add_kcon;
    }
    if ((opc1 == IL_KNOT || opc1 == IL_UKNOT) && ILI_OPND(op1, 1) == op2) {
      /* Complement expression: A ^ ~A = ~0 */
      res.numi[0] = res.numi[1] = ~0;
      goto add_kcon;
    }
    if ((opc2 == IL_KNOT || opc2 == IL_UKNOT) && ILI_OPND(op2, 1) == op1) {
      /* Complement expression: A ^ ~A = ~0 */
      res.numi[0] = res.numi[1] = ~0;
      goto add_kcon;
    }
    if (opc1 != IL_KAND)
      break;
    /* else FALL THRU to distributive */
  distributive:
    /* Check to see if we can apply distributive law:
     * (A | B) & (A | C) = A | (B & C)
     * (A & B) | (A & C) = A & (B | C)
     * (A & B) ^ (A & C) = A & (B ^ C)
     */
    switch (opc1) {
    case IL_AND:
    case IL_OR:
    case IL_KAND:
    case IL_KOR:
      if (opc2 == opc1) {
        /* Look for a common factor */

        int tmp, factor;
        int nonfactor1, nonfactor2;

        if (ILI_OPND(op1, 1) == ILI_OPND(op2, 1)) {
          factor = 1;
          nonfactor1 = nonfactor2 = 2;
        } else if (ILI_OPND(op1, 2) == ILI_OPND(op2, 2)) {
          factor = 2;
          nonfactor1 = nonfactor2 = 1;
        } else if (ILI_OPND(op1, 1) == ILI_OPND(op2, 2)) {
          nonfactor2 = factor = 1;
          nonfactor1 = 2;
        } else if (ILI_OPND(op1, 2) == ILI_OPND(op2, 1)) {
          nonfactor2 = factor = 2;
          nonfactor1 = 1;
        } else {
          goto no_distribute; /* No common factor */
        }
        tmp = ad2ili(opc, ILI_OPND(op1, nonfactor1), ILI_OPND(op2, nonfactor2));
        return ad2ili(opc1, ILI_OPND(op1, factor), tmp);
      }
      break;
    default:
      break;
    }
  no_distribute:
    break;

  case IL_JISHFT:
    if (ncons >= 2) {
      if ((tmp = con2v2) >= 0) {
        if (tmp >= 32) {
          res.numi[1] = 0;
          goto add_icon;
        }
        return ad2ili(IL_ULSHIFT, op1, op2);
      }
      if (tmp <= -32) {
        res.numi[1] = 0;
        goto add_icon;
      }
      op2 = ad_icon(-tmp);
      return ad2ili(IL_URSHIFT, op1, op2);
    }
    ilix = ad2func_int(IL_JSR, "ftn_i_jishft", op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
    break;

#ifdef IL_KISHFT
  case IL_KISHFT:
    op2 = ad1ili(IL_KIMV, op2);
    if (ILI_OPC(op2) == IL_ICON) {
      con2v2 = SymConval2(ILI_SymOPND(op2, 1));
      if (con2v2 >= 64 || con2v2 <= -64) {
        res.numi[0] = res.numi[1] = ~0;
        goto add_kcon;
      }
      if (con2v2 >= 0)
        return ad2ili(IL_KLSHIFT, op1, op2);
      op2 = ad_icon((INT)-con2v2);
      return ad2ili(IL_KURSHIFT, op1, op2);
    }
    tmp1 = ad1ili(IL_NULL, 0);
    tmp1 = ad3ili(IL_DAIR, op2, IR(1), tmp1);
    tmp1 = ad3ili(IL_DAKR, op1, IR(0), tmp1);
    tmp = ad2ili(
        IL_JSR,
        mk_prototype("ftn_i_kishft", "pure", DT_INT8, 2, DT_INT8, DT_INT),
        tmp1);
    ilix = ad2ili(IL_DFRKR, tmp, KR_RETVAL);
    return ilix;
#endif /* #ifdef IL_KISHFT */

  case IL_USHIFT:
    break;

  case IL_LSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 31;
      if (con2v2 == 0)
        return op1;
    }
    if (ncons == 2) {
      if (cons2 == stb.i0)
        return op1;
    } else if (ncons == 3) {
      tmp = con2v2;
      if (SHIFTOK(tmp)) {
        res.numi[1] = LSHIFT(con1v2, tmp);
        goto add_icon;
      }
    }
#ifdef TM_SHIFTAR
    opc = IL_LSHIFTA;
    op2 = ad1ili(IL_IAMV, op2);
#endif
    break;

  case IL_RSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 31;
      if (con2v2 == 0)
        return op1;
    }
    if (ncons == 2) {
      if (cons2 == stb.i0)
        return op1;
    } else if (ncons == 3) {
      tmp = con2v2;
      if (SHIFTOK(tmp)) {
        res.numi[1] = RSHIFT(con1v2, tmp);
        goto add_icon;
      }
    }
#ifdef TM_SHIFTAR
    opc = IL_RSHIFTA;
    op2 = ad1ili(IL_IAMV, op2);
#endif
    break;
#ifdef TM_SHIFTAR
  case IL_SHIFTA:
    break;
#endif

  case IL_ULSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 31;
      if (con2v2 == 0)
        return op1;
    }
    if (ncons == 2) {
      if (cons2 == stb.i0)
        return op1;
    } else if (ncons == 3) {
      tmp = con2v2;
      if (SHIFTOK(tmp)) {
        res.numi[1] = ULSHIFT(con1v2, tmp);
        goto add_icon;
      }
    }
#ifdef TM_SHIFTAR
    opc = IL_ULSHIFTA;
    op2 = ad1ili(IL_IAMV, op2);
#endif
    break;

  case IL_URSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 31;
      if (con2v2 == 0)
        return op1;
    }
    if (ncons == 2) {
      if (cons2 == stb.i0)
        return op1;
    } else if (ncons == 3) {
      tmp = con2v2;
      if (SHIFTOK(tmp)) {
        if (ishft) {
          res.numi[1] = ARSHIFT(con1v2, tmp);
        } else {
          res.numi[1] = URSHIFT(con1v2, tmp);
        }
        goto add_icon;
      }
    }
#ifdef TM_SHIFTAR
    opc = IL_URSHIFTA;
    op2 = ad1ili(IL_IAMV, op2);
#endif
    break;
#ifdef TM_SHIFTAR
  case IL_USHIFTA:
    break;
#endif
  case IL_ARSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 31;
      if (con2v2 == 0)
        return op1;
    }
    if (ncons == 2) {
      if (cons2 == stb.i0)
        return op1;
    } else if (ncons == 3) {
      tmp = con2v2;
      if (SHIFTOK(tmp)) {
        res.numi[1] = ARSHIFT(con1v2, tmp);
        goto add_icon;
      }
    }
#ifdef TM_SHIFTAR
    opc = IL_ARSHIFTA;
    op2 = ad1ili(IL_IAMV, op2);
#endif
    break;

  case IL_KLSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 63;
      if (con2v2 == 0)
        return op1;
    }
    if ((ncons & 2) && cons2 == stb.i0)
      return op1;
    if (ncons == 3) {
      tmp = con2v2;
      GETVALI64(num1, cons1);
      ushf64(num1.numu, tmp, res.numu);
      goto add_kcon;
    }
    break;

  case IL_KARSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 63;
      if (con2v2 == 0)
        return op1;
    }
    if (ncons == 2 && cons2 == stb.i0)
      return op1;
    if (ncons == 3) {
      tmp = con2v2;
      GETVALI64(num1, cons1);
      shf64(num1.numi, -tmp, res.numi);
      goto add_kcon;
    }
    break;

  case IL_KURSHIFT:
    if (ncons & 2) {
      con2v2 = con2v2 & 63;
      if (con2v2 == 0)
        return op1;
    }
    if (ncons == 2 && cons2 == stb.i0)
      return op1;
    if (ncons == 3) {
      tmp = con2v2;
      GETVALI64(num1, cons1);
      ushf64(num1.numu, -tmp, res.numu);
      goto add_kcon;
    }
    break;

  case IL_CSE: {
    static int csecnt = 0;

    if (op2 == 0)
      return ad2ili(IL_CSE, op1, ++csecnt);
    if (ncons == 1 || ILI_OPC(op1) == opc)
      return op1;
    break;
  }

  case IL_CSEKR:
  case IL_CSEIR:
  case IL_CSESP:
  case IL_CSEDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_CSEQP:
#endif
  case IL_CSEAR:
  case IL_CSECS:
  case IL_CSECD:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CSE:
#endif
    if (ncons == 1 || ILI_OPC(op1) == opc)
      return op1;
    break;

#ifdef TM_FIELD_INST
  case IL_MERGE:
    newili.opnd[3] = ilip->opnd[3];
  case IL_EXTRACT:
    newili.opnd[2] = ilip->opnd[2];
    break;
#endif

  case IL_FSIN:
    if (ncons == 1 && is_flt0(cons1)) {
      return op1;
    }
    if (XBIT_NEW_MATH_NAMES) {
      ilix = gen_sincos(opc, op1, IL_FSINCOS, IL_FNSIN, MTH_sin, DT_FLOAT,
                        IL_spfunc);
      return ilix;
    }
#if defined(PGOCL) || defined(TARGET_LLVM_ARM)
    break;
#else
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_SIN, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_SIN, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#if defined(TARGET_LLVM_X8664)
    mk_prototype(fast_math("sin", 's', 's', FMTH_I_SIN), "f pure", DT_FLOAT, 1,
                 DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("sin", 's', 's', FMTH_I_SIN), 1,
                   op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#elif defined(TARGET_POWER)
    mk_prototype(fast_math("sin", 's', 's', MTH_I_SIN), "f pure", DT_FLOAT, 1,
                 DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("sin", 's', 's', MTH_I_SIN), 1,
                   op1);
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_SIN, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_SIN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
#endif
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QSIN:
    if (ncons == 1 && is_quad0(cons1)) {
      return op1;
    }
    if (XBIT_NEW_MATH_NAMES) {
      ilix =
          gen_sincos(opc, op1, IL_NONE, IL_NONE, MTH_sin, DT_QUAD, IL_qpfunc);
      return ilix;
    }
#if defined(PGOCL) || defined(TARGET_LLVM_ARM)
    break;
#else
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_DSIN, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DSIN, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#if defined(TARGET_LLVM_X8664)
    mk_prototype(fast_math("sin", 's', 'd', FMTH_I_DSIN), "f pure", DT_DBLE, 1,
                 DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("sin", 's', 'd', FMTH_I_DSIN),
                   1, op1);
    return ad1altili(opc, op1, ilix);
#elif defined(TARGET_POWER)
    mk_prototype(fast_math("sin", 's', 'd', MTH_I_DSIN), "f pure", DT_DBLE, 1,
                 DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("sin", 's', 'd', MTH_I_DSIN), 1,
                   op1);
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_DSIN, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_DSIN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
#endif
    break;
#endif

  case IL_DSIN:
    if (ncons == 1 && is_dbl0(cons1)) {
      return op1;
    }
    if (XBIT_NEW_MATH_NAMES) {
      ilix = gen_sincos(opc, op1, IL_DSINCOS, IL_DNSIN, MTH_sin, DT_DBLE,
                        IL_dpfunc);
      return ilix;
    }
#if defined(PGOCL) || defined(TARGET_LLVM_ARM)
    break;
#else
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_DSIN, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DSIN, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#if defined(TARGET_LLVM_X8664)
    mk_prototype(fast_math("sin", 's', 'd', FMTH_I_DSIN), "f pure", DT_DBLE, 1,
                 DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("sin", 's', 'd', FMTH_I_DSIN),
                   1, op1);
    return ad1altili(opc, op1, ilix);
#elif defined(TARGET_POWER)
    mk_prototype(fast_math("sin", 's', 'd', MTH_I_DSIN), "f pure", DT_DBLE, 1,
                 DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("sin", 's', 'd', MTH_I_DSIN), 1,
                   op1);
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_DSIN, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DSIN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
#endif
    break;

  case IL_FCOS:
    if (ncons == 1 && is_flt0(cons1)) {
      return ad1ili(IL_FCON, stb.flt1);
    }
    if (XBIT_NEW_MATH_NAMES) {
      ilix = gen_sincos(opc, op1, IL_FSINCOS, IL_FNCOS, MTH_cos, DT_FLOAT,
                        IL_spfunc);
      return ilix;
    }
#if defined(PGOCL) || defined(TARGET_LLVM_ARM)
    break;
#else
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_COS, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_COS, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#if defined(TARGET_LLVM_X8664)
    mk_prototype(fast_math("cos", 's', 's', FMTH_I_COS), "f pure", DT_FLOAT, 1,
                 DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("cos", 's', 's', FMTH_I_COS), 1,
                   op1);
    return ad1altili(opc, op1, ilix);
    return ilix;
#elif defined(TARGET_POWER)
    mk_prototype(fast_math("cos", 's', 's', MTH_I_COS), "f pure", DT_FLOAT, 1,
                 DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("cos", 's', 's', MTH_I_COS), 1,
                   op1);
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_COS, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_COS, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
#endif
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCOS:
    if (ncons == 1 && is_quad0(cons1)) {
      return ad1ili(IL_QCON, stb.quad1);
    }
    if (XBIT_NEW_MATH_NAMES) {
      ilix =
          gen_sincos(opc, op1, IL_NONE, IL_NONE, MTH_cos, DT_QUAD, IL_qpfunc);
      return ilix;
    }
#if defined(PGOCL) || defined(TARGET_LLVM_ARM)
    break;
#else
#endif
    break;
#endif

  case IL_DCOS:
    if (ncons == 1 && is_dbl0(cons1)) {
      return ad1ili(IL_DCON, stb.dbl1);
    }
    if (XBIT_NEW_MATH_NAMES) {
      ilix = gen_sincos(opc, op1, IL_DSINCOS, IL_DNCOS, MTH_cos, DT_DBLE,
                        IL_dpfunc);
      return ilix;
    }
#if defined(PGOCL) || defined(TARGET_LLVM_ARM)
    break;
#else
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_DCOS, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DCOS, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#if   defined(TARGET_POWER)
    mk_prototype(fast_math("cos", 's', 'd', MTH_I_DCOS), "f pure", DT_DBLE, 1,
                 DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("cos", 's', 'd', MTH_I_DCOS), 1,
                   op1);
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_DCOS, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DCOS, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
#endif
    break;

  case IL_FTAN:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_tan, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee && TEST_FEATURE(FEATURE_AVX)) {
      (void)mk_prototype(relaxed_math("tan", 's', 's', MTH_I_TAN), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR,
                     relaxed_math("tan", 's', 's', MTH_I_TAN), 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#endif
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_TAN, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_TAN, 1, op1);
    } else {
      (void)mk_prototype(fast_math("tan", 's', 's', MTH_I_TAN), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("tan", 's', 's', MTH_I_TAN),
                     1, op1);
    }
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_TAN, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_TAN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif

  case IL_DTAN:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_tan, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee && TEST_FEATURE(FEATURE_AVX)) {
      if (!XBIT(36, 0x04)) {
        (void)mk_prototype(fast_math("tan", 's', 'd', MTH_I_DTAN), "f pure",
                           DT_DBLE, 1, DT_DBLE);
        ilix = ad_func(IL_DFRDP, IL_QJSR,
                       fast_math("tan", 's', 'd', MTH_I_DTAN), 1, op1);
      } else {
        (void)mk_prototype(relaxed_math("tan", 's', 'd', MTH_I_DTAN), "f pure",
                           DT_DBLE, 1, DT_DBLE);
        ilix = ad_func(IL_DFRDP, IL_QJSR,
                       relaxed_math("tan", 's', 'd', MTH_I_DTAN), 1, op1);
      }
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#endif
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_DTAN, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DTAN, 1, op1);
    } else {
      (void)mk_prototype(fast_math("tan", 's', 'd', MTH_I_DTAN), "f pure",
                         DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("tan", 's', 'd', MTH_I_DTAN),
                     1, op1);
    }
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_DTAN, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DTAN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QTAN:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_tan, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    break;
#endif
#if defined(TARGET_POWER)
    break;
#else
    (void)mk_prototype(MTH_I_QTAN, "f pure", DT_QUAD, 1, DT_QUAD);
    ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_QTAN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
#endif

  case IL_FATAN:
    if (ncons == 1) {
      xfatan(con1v2, &res.numi[1]);
      goto add_rcon;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_atan, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(gnr_math("atan", 's', 's', MTH_I_ATAN, 0), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR,
                     gnr_math("atan", 's', 's', MTH_I_ATAN, 0), 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#endif
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_ATAN, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_ATAN, 1, op1);
    } else {
      (void)mk_prototype(fast_math("atan", 's', 's', MTH_I_ATAN), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("atan", 's', 's', MTH_I_ATAN),
                     1, op1);
    }
    return ad1altili(opc, op1, ilix);
#endif
    (void)mk_prototype(MTH_I_ATAN, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_ATAN, 1, op1);
    return ad1altili(opc, op1, ilix);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QATAN:
    if (ncons == 1) {
      GETVAL128(qnum1, cons1);
      xqatan(qnum1.numq, qres.numq);
      goto add_qcon;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_atan, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    break;
#endif
#if defined(TARGET_POWER)
    break;
#else
    (void)mk_prototype(MTH_I_QATAN, "f pure", DT_QUAD, 1, DT_QUAD);
    ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QATAN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
    break;
#endif

  case IL_DATAN:
    if (ncons == 1) {
      GETVAL64(num1, cons1);
      xdatan(num1.numd, res.numd);
      goto add_dcon;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_atan, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(gnr_math("atan", 's', 'd', MTH_I_DATAN, 0), "f pure",
                         DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR,
                     gnr_math("atan", 's', 'd', MTH_I_DATAN, 0), 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#endif
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_DATAN, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DATAN, 1, op1);
    } else {
      (void)mk_prototype(fast_math("atan", 's', 'd', MTH_I_DATAN), "f pure",
                         DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR,
                     fast_math("atan", 's', 'd', MTH_I_DATAN), 1, op1);
    }
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_DATAN, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DATAN, 1, op1);
    return ad1altili(opc, op1, ilix);
#endif
    break;

  case IL_FACOS:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_acos, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_ACOS, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_ACOS, 1, op1);
    return ad1altili(opc, op1, ilix);
    break;

  case IL_FASIN:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_asin, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_ASIN, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_ASIN, 1, op1);
    return ad1altili(opc, op1, ilix);
    break;

  case IL_FATAN2:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_atan2, &funcsptr, 1, false, DT_FLOAT, 2, DT_FLOAT,
                        DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_ATAN2, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
    ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_ATAN2, 2, op1, op2);
    return ad2altili(opc, op1, op2, ilix);
    break;

  case IL_DACOS:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_acos, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_DACOS, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DACOS, 1, op1);
    return ad1altili(opc, op1, ilix);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QACOS:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_acos, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_QACOS, "f pure", DT_QUAD, 1, DT_QUAD);
    ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QACOS, 1, op1);
    return ad1altili(opc, op1, ilix);
    break;
#endif

  case IL_DASIN:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_asin, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_DASIN, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DASIN, 1, op1);
    return ad1altili(opc, op1, ilix);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QASIN:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_asin, &funcsptr, VECTLEN1, false, DT_QUAD,
                        ARGS_NUMBER, DT_QUAD, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_QASIN, "f pure", DT_QUAD, 1, DT_QUAD);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_QASIN, 1, op1);
    return ad1altili(opc, op1, ilix);
    break;
#endif

  case IL_DATAN2:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_atan2, &funcsptr, 1, false, DT_DBLE, 2, DT_DBLE,
                        DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_DATAN2, "f pure", DT_DBLE, 2, DT_DBLE, DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DATAN2, 2, op1, op2);
    return ad2altili(opc, op1, op2, ilix);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QATAN2:
    if (ncons == 3) {
      GETVAL128(qnum1, cons1);
      GETVAL128(qnum2, cons2);
      xqatan2(qnum1.numq, qnum2.numq, qres.numq);
      goto add_qcon;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_atan2, &funcsptr, 1, false, DT_QUAD, 2, DT_QUAD,
                        DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_QATAN2, "f pure", DT_QUAD, 2, DT_QUAD, DT_QUAD);
    ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QATAN2, 2, op1, op2);
    return ad2altili(opc, op1, op2, ilix);
    break;
#endif

  case IL_FLOG:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_log, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(gnr_math("log", 's', 's', FMTH_I_ALOG, 0), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR,
                     gnr_math("log", 's', 's', FMTH_I_ALOG, 0), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_ALOG, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_ALOG, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#if !defined(PGOCL) && !defined(TARGET_LLVM_ARM)
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_LOG, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_LOG, 1, op1);
    } else {
      (void)mk_prototype(fast_math("log", 's', 's', MTH_I_LOG), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("log", 's', 's', MTH_I_LOG),
                     1, op1);
    }
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_ALOG, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_ALOG, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#endif /*if !defined(PGOCL) && !defined(TARGET_LLVM_ARM) */
    break;

  case IL_DLOG:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_log, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(gnr_math("log", 's', 'd', FMTH_I_DLOG, 0), "f pure",
                         DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR,
                     gnr_math("log", 's', 'd', FMTH_I_DLOG, 0), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_DLOG, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DLOG, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#if !defined(PGOCL) && !defined(TARGET_LLVM_ARM)
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_DLOG, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DLOG, 1, op1);
    } else {
      (void)mk_prototype(fast_math("log", 's', 'd', MTH_I_DLOG), "f pure",
                         DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("log", 's', 'd', MTH_I_DLOG),
                     1, op1);
    }
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_DLOG, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DLOG, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#endif /*if !defined(PGOCL) && !defined(TARGET_LLVM_ARM) */
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QLOG:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_log, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(gnr_math("log", 's', 'q', FMTH_I_QLOG, 0), "f pure",
                         DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_DFRQP, IL_QJSR,
                     gnr_math("log", 's', 'q', FMTH_I_QLOG, 0), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_QLOG, "f pure", DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QLOG, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
    break;
#endif

  case IL_FLOG10:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_log10, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(fast_math("log10", 's', 's', FMTH_I_ALOG10), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR,
                     fast_math("log10", 's', 's', FMTH_I_ALOG10), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_ALOG10, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_ALOG10, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#if defined(TARGET_LLVM) && !defined(PGOCL) && !defined(TARGET_LLVM_ARM)
    (void)mk_prototype(MTH_I_ALOG10, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_ALOG10, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif /*if TARGET_LLVM && !PGOCL && !TARGET_LLVM_ARM */
    break;
  case IL_DLOG10:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_log10, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(fast_math("log10", 's', 'd', FMTH_I_DLOG10), "f pure",
                         DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR,
                     fast_math("log10", 's', 'd', FMTH_I_DLOG10), 1, op1);
    } else {
      (void)mk_prototype(MTH_I_DLOG10, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DLOG10, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#if defined(TARGET_LLVM) && !defined(PGOCL) && !defined(TARGET_LLVM_ARM)
    (void)mk_prototype(MTH_I_DLOG10, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DLOG10, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif /*if TARGET_LLVM && !PGOCL && !TARGET_LLVM_ARM */
    break;

  case IL_FEXP:
    if (ncons == 1 && is_flt0(cons1)) {
      res.numi[1] = CONVAL2G(stb.flt1);
      goto add_rcon;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_exp, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      if (XBIT_NEW_RELAXEDMATH) {
        (void)mk_prototype(relaxed_math("exp", 's', 's', FMTH_I_EXP), "f pure",
                           DT_FLOAT, 1, DT_FLOAT);
        ilix = ad_func(IL_DFRSP, IL_QJSR,
                       relaxed_math("exp", 's', 's', FMTH_I_EXP), 1, op1);
      } else {
        (void)mk_prototype(gnr_math("exp", 's', 's', FMTH_I_EXP, 0), "f pure",
                           DT_FLOAT, 1, DT_FLOAT);
        ilix = ad_func(IL_DFRSP, IL_QJSR,
                       gnr_math("exp", 's', 's', FMTH_I_EXP, 0), 1, op1);
      }
    } else {
      (void)mk_prototype(MTH_I_EXP, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_EXP, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#if !defined(PGOCL) && !defined(TARGET_LLVM_ARM)
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_EXP, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_EXP, 1, op1);
    } else {
      (void)mk_prototype(fast_math("exp", 's', 's', MTH_I_EXP), "f pure",
                         DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_DFRSP, IL_QJSR, fast_math("exp", 's', 's', MTH_I_EXP),
                     1, op1);
    }
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_EXP, "f pure", DT_FLOAT, 1, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_EXP, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#endif /*if !defined(PGOCL) && !defined(TARGET_LLVM_ARM) */
    break;

  case IL_DEXP:
    if (ncons == 1 && is_dbl0(cons1)) {
      GETVAL64(res, stb.dbl1);
      goto add_dcon;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_exp, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      if (XBIT_NEW_RELAXEDMATH) {
        (void)mk_prototype(relaxed_math("exp", 's', 'd', FMTH_I_DEXP), "f pure",
                           DT_DBLE, 1, DT_DBLE);
        ilix = ad_func(IL_DFRDP, IL_QJSR,
                       relaxed_math("exp", 's', 'd', FMTH_I_DEXP), 1, op1);
      } else {
        /* Try the new naming convention -- only for exp. */
        (void)mk_prototype(gnr_math("exp", 's', 'd', FMTH_I_DEXP, 0), "f pure",
                           DT_DBLE, 1, DT_DBLE);
        ilix = ad_func(IL_DFRDP, IL_QJSR,
                       gnr_math("exp", 's', 'd', FMTH_I_DEXP, 0), 1, op1);
      }
    } else {
      (void)mk_prototype(MTH_I_DEXP, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DEXP, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#if !defined(PGOCL) && !defined(TARGET_LLVM_ARM)
#if defined(TARGET_POWER)
    if (flg.ieee) {
      (void)mk_prototype(MTH_I_DEXP, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DEXP, 1, op1);
    } else {
      (void)mk_prototype(fast_math("exp", 's', 'd', MTH_I_DEXP), "f pure",
                         DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR, fast_math("exp", 's', 'd', MTH_I_DEXP),
                     1, op1);
    }
    return ad1altili(opc, op1, ilix);
#else
    (void)mk_prototype(MTH_I_DEXP, "f pure", DT_DBLE, 1, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DEXP, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
#endif /*if !defined(PGOCL) && !defined(TARGET_LLVM_ARM) */
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QEXP:
    if (ncons == 1 && is_quad0(cons1)) {
      GETVAL128(qres, stb.quad1);
      goto add_qcon;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_exp, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      if (XBIT_NEW_RELAXEDMATH) {
        (void)mk_prototype(relaxed_math("exp", 's', 'q', FMTH_I_QEXP), "f pure",
                           DT_QUAD, 1, DT_QUAD);
        ilix = ad_func(IL_DFRQP, IL_QJSR,
                       relaxed_math("exp", 's', 'q', FMTH_I_QEXP), 1, op1);
      } else {
        /* Try the new naming convention -- only for exp. */
        (void)mk_prototype(gnr_math("exp", 's', 'q', FMTH_I_QEXP, 0), "f pure",
                           DT_QUAD, 1, DT_QUAD);
        ilix = ad_func(IL_DFRQP, IL_QJSR,
                       gnr_math("exp", 's', 'q', FMTH_I_QEXP, 0), 1, op1);
      }
    } else {
      (void)mk_prototype(MTH_I_QEXP, "f pure", DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QEXP, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
#endif
    break;
#endif

  /*
   * getting here for the ensuing cmplex intrinsics means XBIT_NEW_MATH_NAMES
   * is set
   */
  case IL_SCMPLXEXP:
    ilix = ad1mathfunc_cmplx(MTH_exp, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXCOS:
    ilix = ad1mathfunc_cmplx(MTH_cos, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXSIN:
    ilix = ad1mathfunc_cmplx(MTH_sin, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXTAN:
    ilix = ad1mathfunc_cmplx(MTH_tan, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXACOS:
    ilix = ad1mathfunc_cmplx(MTH_acos, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXASIN:
    ilix = ad1mathfunc_cmplx(MTH_asin, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXATAN:
    ilix = ad1mathfunc_cmplx(MTH_atan, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXCOSH:
    ilix = ad1mathfunc_cmplx(MTH_cosh, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXSINH:
    ilix = ad1mathfunc_cmplx(MTH_sinh, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXTANH:
    ilix = ad1mathfunc_cmplx(MTH_tanh, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXLOG:
    ilix = ad1mathfunc_cmplx(MTH_log, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXSQRT:
    ilix = ad1mathfunc_cmplx(MTH_sqrt, opc, op1, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXPOW:
    ilix =
        ad2mathfunc_cmplx(MTH_pow, opc, op1, op2, DT_CMPLX, DT_CMPLX, DT_CMPLX);
    return ilix;

  case IL_SCMPLXPOWI:
    /**** ad2mathfunc_cmplx needs WORK for the integer argument ****/
    ilix =
        ad2mathfunc_cmplx(MTH_powi, opc, op1, op2, DT_CMPLX, DT_CMPLX, DT_INT);
    return ilix;

  case IL_SCMPLXPOWK:
    /**** ad2mathfunc_cmplx needs WORK for the integer argument ****/
    ilix =
        ad2mathfunc_cmplx(MTH_powk, opc, op1, op2, DT_CMPLX, DT_CMPLX, DT_INT8);
    return ilix;

  case IL_DCMPLXEXP:
    ilix = ad1mathfunc_cmplx(MTH_exp, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXCOS:
    ilix = ad1mathfunc_cmplx(MTH_cos, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXSIN:
    ilix = ad1mathfunc_cmplx(MTH_sin, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXTAN:
    ilix = ad1mathfunc_cmplx(MTH_tan, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXACOS:
    ilix = ad1mathfunc_cmplx(MTH_acos, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXASIN:
    ilix = ad1mathfunc_cmplx(MTH_asin, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXATAN:
    ilix = ad1mathfunc_cmplx(MTH_atan, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXCOSH:
    ilix = ad1mathfunc_cmplx(MTH_cosh, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXSINH:
    ilix = ad1mathfunc_cmplx(MTH_sinh, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXTANH:
    ilix = ad1mathfunc_cmplx(MTH_tanh, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXLOG:
    ilix = ad1mathfunc_cmplx(MTH_log, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXSQRT:
    ilix = ad1mathfunc_cmplx(MTH_sqrt, opc, op1, DT_DCMPLX, DT_DCMPLX);
    return ilix;

  case IL_DCMPLXPOW:
    ilix = ad2mathfunc_cmplx(MTH_pow, opc, op1, op2, DT_DCMPLX, DT_DCMPLX,
                             DT_DCMPLX);
    return ilix;

  case IL_DCMPLXPOWI:
    /**** ad2mathfunc_cmplx needs WORK for the integer argument ****/
    ilix = ad2mathfunc_cmplx(MTH_powi, opc, op1, op2, DT_DCMPLX, DT_DCMPLX,
                             DT_INT);
    return ilix;

  case IL_DCMPLXPOWK:
    /**** ad2mathfunc_cmplx needs WORK for the integer argument ****/
    ilix = ad2mathfunc_cmplx(MTH_powk, opc, op1, op2, DT_DCMPLX, DT_DCMPLX,
                             DT_INT8);
    return ilix;

  case IL_JN:
    (void)mk_prototype(MTH_I_JN, "f pure", DT_FLOAT, 2, DT_INT, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_JN, 2, op1, op2);
    return ilix;
    break;

  case IL_DJN:
    (void)mk_prototype(MTH_I_DJN, "f pure", DT_DBLE, 2, DT_INT, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DJN, 2, op1, op2);
    return ilix;
    break;

  case IL_YN:
    (void)mk_prototype(MTH_I_YN, "f pure", DT_FLOAT, 2, DT_INT, DT_FLOAT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_YN, 2, op1, op2);
    return ilix;
    break;

  case IL_DYN:
    (void)mk_prototype(MTH_I_DYN, "f pure", DT_DBLE, 2, DT_INT, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DYN, 2, op1, op2);
    return ilix;
    break;

  case IL_IPOWI:
    if (ncons == 3) {
      res.numi[1] = _ipowi(con1v2, con2v2);
      goto add_icon;
    }
    if (ncons == 1 && con1v2 == 2) {
      tmp1 = ad_icon(1);
      tmp1 = ad2ili(IL_LSHIFT, tmp1, op2);
      /*  generate ili which computes (((1)<<i) & ~((i)>>31)) */
      tmp = ad_icon(31);
      tmp = ad2ili(IL_ARSHIFT, op2, tmp);
      tmp = ad1ili(IL_NOT, tmp);
      ilix = ad2ili(IL_AND, tmp1, tmp);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_IPOWI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
    break;

#ifdef IL_KPOWI
  case IL_KPOWI:
    if (ncons == 1 && con1v1 == 0 && con1v2 == 2) {
      tmp1 = ad_kcon(0, 1);
      tmp1 = ad2ili(IL_KLSHIFT, tmp1, op2);
#if defined(TARGET_X8664)
      /*  use select ili to compute  ( (i)>=0 ? 1<<(i) : 0 )  */
      ilix = ad2ili(IL_ICMPZ, op2, CC_LT);
      tmp = ad_kcon(0, 0);
      ilix = ad3ili(IL_KSELECT, ilix, tmp1, tmp);
#else
      /*  generate ili which computes (((1)<<i) & ~((i)>>63)) */
      tmp = ad_icon(63);
      tmp = ad2ili(IL_KARSHIFT, ad1ili(IL_IKMV, op2), tmp);
      tmp = ad1ili(IL_KNOT, tmp);
      ilix = ad2ili(IL_KAND, tmp1, tmp);
      ilix = ad2altili(opc, op1, op2, ilix);
#endif
      return ilix;
    }
    (void)mk_prototype(MTH_I_KPOWI, "pure", DT_INT8, 2, DT_INT8, DT_INT);
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KPOWI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
#endif

#ifdef IL_KPOWK
  case IL_KPOWK:
    if (ncons == 1 && con1v1 == 0 && con1v2 == 2) {
      tmp1 = ad_kcon(0, 1);
      tmp1 = ad2ili(IL_KLSHIFT, tmp1, ad1ili(IL_KIMV, op2));
#if defined(TARGET_X8664)
      /*  use select ili to compute  ( (i)>=0 ? 1<<(i) : 0 )  */
      ilix = ad2ili(IL_KCMPZ, op2, CC_LT);
      tmp = ad_kcon(0, 0);
      ilix = ad3ili(IL_KSELECT, ilix, tmp1, tmp);
#else
      /*  generate ili which computes (((1)<<i) & ~((i)>>63)) */
      tmp = ad_icon(63);
      tmp = ad2ili(IL_KARSHIFT, op2, tmp);
      tmp = ad1ili(IL_KNOT, tmp);
      ilix = ad2ili(IL_KAND, tmp1, tmp);
      ilix = ad2altili(opc, op1, op2, ilix);
#endif
      return ilix;
    }
    (void)mk_prototype(MTH_I_KPOWK, "pure", DT_INT8, 2, DT_INT8, DT_INT8);
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KPOWK, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
#endif

  case IL_FPOWI:
#define __MAXPOW 10
    /* Even with -Kieee OK to make 1 multiply and get exact answer,
     * instead of making an intrinsic call to pow(). Big performance gain.
     * That is why we check specifically for the power of 2 below.
     */
    if ((!flg.ieee || con2v2 == 1 || con2v2 == 2) && 
         ncons >= 2 && !XBIT(124, 0x200)) {
      if (con2v2 == 1)
        return op1;
      if (con2v2 > 1 && con2v2 <= __MAXPOW) {
        ilix = _xpowi(op1, con2v2, IL_FMUL);
        return ilix;
      }
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_powi, &funcsptr, 1, false, DT_FLOAT, 2, DT_FLOAT,
                        DT_INT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_RPOWI, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_INT);
    ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_RPOWI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_FPOWK:
    if ((!flg.ieee || con2v2 == 1 || con2v2 == 2) && 
         ncons >= 2 && !XBIT(124, 0x200) && con2v1 == 0) {
      if (con2v2 == 1)
        return op1;
      if (con2v2 > 1 && con2v2 <= __MAXPOW) {
        ilix = _xpowi(op1, con2v2, IL_FMUL);
        return ilix;
      }
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_powk, &funcsptr, 1, false, DT_FLOAT, 2, DT_FLOAT,
                        DT_INT8);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_RPOWK, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_INT8);
    ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_RPOWK, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_FPOWF:
    if (!flg.ieee && ncons >= 2) {
      if (con2v2 == 0x3e800000) {
        /* x ** .25 -> sqrt(sqrt(x)) */
        ilix = ad1ili(IL_FSQRT, op1);
        ilix = ad1ili(IL_FSQRT, ilix);
        return ilix;
      }
      if (con2v2 == 0x3f000000) {
        /* x ** 0.5 -> sqrt(x) */
        ilix = ad1ili(IL_FSQRT, op1);
        return ilix;
      }
      if (!do_newton_sqrt() && con2v2 == 0x3f400000) {
        /* x ** .75 -> sqrt(x) * sqrt(sqrt(x)) */
        ilix = ad1ili(IL_FSQRT, op1);
        op2 = ad1ili(IL_FSQRT, ilix);
        ilix = ad2ili(IL_FMUL, ilix, op2);
        return ilix;
      }
      if (con2v2 == 0x40200000) {
        /* x ** 2.5 -> x * x * sqrt(x)) */
        ilix = ad2ili(IL_FMUL, op1, op1);
        op2 = ad1ili(IL_FSQRT, op1);
        ilix = ad2ili(IL_FMUL, ilix, op2);
        return ilix;
      }
      if (con2v2 == 0x3fc00000) {
        /* x ** 1.5 -> sqrt(x)*x */
        ilix = ad1ili(IL_FSQRT, op1);
        ilix = ad2ili(IL_FMUL, op1, ilix);
        return ilix;
      }
#if defined(TARGET_X8664)
      if (con2v2 == 0x3eaaaaab && XBIT(15, 0x80000000)) {
        /* x ** (1./3.) */
        ilix = ad_func(IL_DFRSP, IL_QJSR, fmth_name(FMTH_I_CBRT), 1, op1);
        ilix = ad2altili(opc, op1, op2, ilix);
        return ilix;
      }
#endif
    }
    is_int = xfisint(con2v2, &pw);
    if ((!flg.ieee || pw == 1 || pw == 2) && 
         ncons >= 2 && is_int && !XBIT(124, 0x40000)) {
      ilix = ad2ili(IL_FPOWI, op1, ad_icon(pw));
      return ilix;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_pow, &funcsptr, 1, false, DT_FLOAT, 2, DT_FLOAT,
                        DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      if (XBIT_NEW_RELAXEDMATH) {
        (void)mk_prototype(relaxed_math("pow", 's', 's', FMTH_I_RPOWF),
                           "f pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
        ilix =
            ad_func(IL_DFRSP, IL_QJSR,
                    relaxed_math("pow", 's', 's', FMTH_I_RPOWF), 2, op1, op2);
      } else {
        (void)mk_prototype(gnr_math("pow", 's', 's', FMTH_I_RPOWF, 0), "f_pure",
                           DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
        ilix = ad_func(IL_DFRSP, IL_QJSR,
                       gnr_math("pow", 's', 's', FMTH_I_RPOWF, 0), 2, op1, op2);
      }
    } else
#endif
    {
#if defined(TARGET_POWER)
      if (flg.ieee) {
        mk_prototype(MTH_I_RPOWF, "f_pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
        ilix = ad_func(IL_DFRSP, IL_QJSR, MTH_I_RPOWF, 2, op1, op2);
      } else {
        (void)mk_prototype(fast_math("pow", 's', 's', MTH_I_RPOWF), "f pure",
                           DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
        ilix = ad_func(IL_DFRSP, IL_QJSR,
                       fast_math("pow", 's', 's', MTH_I_RPOWF), 2, op1, op2);
      }
#else
      mk_prototype(MTH_I_RPOWF, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_RPOWF, 2, op1, op2);
#endif
    }
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;

  case IL_DPOWI:
    if ((!flg.ieee || con2v2 == 1 || con2v2 == 2) 
         && ncons >= 2 && !XBIT(124, 0x200)) {
      if (con2v2 == 1)
        return op1;
      if (con2v2 > 1 && con2v2 <= __MAXPOW) {
        ilix = _xpowi(op1, con2v2, IL_DMUL);
        return ilix;
      }
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname =
          make_math(MTH_powi, &funcsptr, 1, false, DT_DBLE, 2, DT_DBLE, DT_INT);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_DPOWI, "f pure", DT_DBLE, 2, DT_DBLE, DT_INT);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DPOWI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_DPOWK:
    if ((!flg.ieee || con2v2 == 1 || con2v2 == 2) 
         && ncons >= 2 && !XBIT(124, 0x200) && con2v1 == 0) {
      if (con2v2 == 1)
        return op1;
      if (con2v2 > 1 && con2v2 <= __MAXPOW) {
        ilix = _xpowi(op1, con2v2, IL_DMUL);
        return ilix;
      }
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_powk, &funcsptr, 1, false, DT_DBLE, 2, DT_DBLE,
                        DT_INT8);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_DPOWK, "f pure", DT_DBLE, 2, DT_DBLE, DT_INT8);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DPOWK, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;

  case IL_DPOWD:
    if (!flg.ieee && ncons >= 2 && con2v2 == 0) {
      if (con2v1 == 0x3fd00000) {
        /* x ** .25 -> sqrt(sqrt(x)) */
        ilix = ad1ili(IL_DSQRT, op1);
        ilix = ad1ili(IL_DSQRT, ilix);
        return ilix;
      }
      if (con2v1 == 0x3fe00000) {
        /* x ** 0.5 -> sqrt(x) */
        ilix = ad1ili(IL_DSQRT, op1);
        return ilix;
      }
      if (con2v1 == 0x3fe80000) {
        /* && !do_newton_sqrt() if newton's is possible for DSQRT */
        /* x ** .75 -> sqrt(x) * sqrt(sqrt(x)) */
        ilix = ad1ili(IL_DSQRT, op1);
        op2 = ad1ili(IL_DSQRT, ilix);
        ilix = ad2ili(IL_DMUL, ilix, op2);
        return ilix;
      }
      if (con2v1 == 0x40040000) {
        /* x ** 2.5 -> x * x * sqrt(x)) */
        ilix = ad2ili(IL_DMUL, op1, op1);
        op2 = ad1ili(IL_DSQRT, op1);
        ilix = ad2ili(IL_DMUL, ilix, op2);
        return ilix;
      }
      if (con2v1 == 0x3ff80000) {
        /* x ** 1.5 -> sqrt(x)*x */
        ilix = ad1ili(IL_DSQRT, op1);
        ilix = ad2ili(IL_DMUL, op1, ilix);
        return ilix;
      }
    }
    is_int = 0;
    pw = 0;
    if( ncons >= 2 && !XBIT(124, 0x40000) )
    {
      GETVAL64(num2, cons2);
      is_int = xdisint(num2.numd, &pw);
    }
    if ((!flg.ieee || pw == 1 || pw == 2) && 
         ncons >= 2 && is_int && !XBIT(124, 0x40000)) {
      ilix = ad2ili(IL_DPOWI, op1, ad_icon(pw));
      return ilix;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname =
          make_math(MTH_pow, &funcsptr, 1, false, DT_DBLE, 2, DT_DBLE, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 2, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(gnr_math("pow", 's', 'd', FMTH_I_DPOWD, 0), "f pure",
                         DT_DBLE, 2, DT_DBLE, DT_DBLE);
      ilix = ad_func(IL_DFRDP, IL_QJSR,
                     gnr_math("pow", 's', 'd', FMTH_I_DPOWD, 0), 2, op1, op2);
    } else
#endif
    {
#if defined(TARGET_POWER)
      if (flg.ieee) {
        (void)mk_prototype(MTH_I_DPOWD, "f pure", DT_DBLE, 2, DT_DBLE, DT_DBLE);
        ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DPOWD, 2, op1, op2);
      } else {
        (void)mk_prototype(fast_math("pow", 's', 'd', MTH_I_DPOWD), "f pure",
                           DT_DBLE, 2, DT_DBLE, DT_DBLE);
        ilix = ad_func(IL_DFRDP, IL_QJSR,
                       fast_math("pow", 's', 'd', MTH_I_DPOWD), 2, op1, op2);
      }
#else
      (void)mk_prototype(MTH_I_DPOWD, "f pure", DT_DBLE, 2, DT_DBLE, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DPOWD, 2, op1, op2);
#endif
    }
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QPOWI:
    if ((!flg.ieee || con2v2 == POW1 || con2v2 == POW2) && ncons >= 2 &&
        !XBIT(124, 0x200)) {
      if (con2v2 == 1)
        return op1;
      if (con2v2 > 1 && con2v2 <= __MAXPOW) {
        ilix = _xpowi(op1, con2v2, IL_QMUL);
        return ilix;
      }
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname =
          make_math(MTH_powi, &funcsptr, 1, false, DT_QUAD, 2, DT_QUAD, DT_INT);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, ARGS_NUMBER, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_QPOWI, "f pure", DT_QUAD, ARGS_NUMBER, DT_QUAD,
                       DT_INT);
    ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_QPOWI, ARGS_NUMBER, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_QPOWK:
    if ((!flg.ieee || con2v2 == 1 || con2v2 == 2) && ncons >= 2 &&
        !XBIT(124, 0x200) && con2v1 == 0) {
      if (con2v2 == 1)
        return op1;
      if (con2v2 > 1 && con2v2 <= __MAXPOW) {
        ilix = _xpowi(op1, con2v2, IL_QMUL);
        return ilix;
      }
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_powk, &funcsptr, 1, false, DT_QUAD, ARGS_NUMBER,
                        DT_QUAD, DT_INT8);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, ARGS_NUMBER, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
    (void)mk_prototype(MTH_I_QPOWK, "f pure", DT_QUAD, ARGS_NUMBER, DT_QUAD,
                       DT_INT8);
    ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_QPOWK, ARGS_NUMBER, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_QPOWQ:
    if (!flg.ieee && ncons >= 2 && con2v2 == 0) {
      if (con2v1 == 0x3ffd0000) {
        /* x ** .25 -> sqrt(sqrt(x)) */
        ilix = ad1ili(IL_QSQRT, op1);
        ilix = ad1ili(IL_QSQRT, ilix);
        return ilix;
      }
      if (con2v1 == 0x3ffe0000) {
        /* x ** 0.5 -> sqrt(x) */
        ilix = ad1ili(IL_QSQRT, op1);
        return ilix;
      }
      if (con2v1 == 0x3ffe8000) {
        /* && !do_newton_sqrt() if newton's is possible for QSQRT */
        /* x ** .75 -> sqrt(x) * sqrt(sqrt(x)) */
        ilix = ad1ili(IL_QSQRT, op1);
        op2 = ad1ili(IL_QSQRT, ilix);
        ilix = ad2ili(IL_QMUL, ilix, op2);
        return ilix;
      }
      if (con2v1 == 0x40004000) {
        /* x ** 2.5 -> x * x * sqrt(x)) */
        ilix = ad2ili(IL_QMUL, op1, op1);
        op2 = ad1ili(IL_QSQRT, op1);
        ilix = ad2ili(IL_QMUL, ilix, op2);
        return ilix;
      }
      if (con2v1 == 0x3fff8000) {
        /* x ** 1.5 -> sqrt(x)*x */
        ilix = ad1ili(IL_QSQRT, op1);
        ilix = ad2ili(IL_QMUL, op1, ilix);
        return ilix;
      }
    }
    is_int = 0;
    pw = 0;
    if (ncons >= 2 && !XBIT(124, 0x40000)) {
      GETVAL128(qnum2, cons2);
      is_int = xqisint(qnum2.numq, &pw);
    }
    if ((!flg.ieee || pw == 1 || pw == 2) && ncons >= 2 && is_int &&
        !XBIT(124, 0x40000)) {
      ilix = ad2ili(IL_QPOWI, op1, ad_icon(pw));
      return ilix;
    }
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_pow, &funcsptr, 1, false, DT_QUAD, ARGS_NUMBER,
                        DT_QUAD, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, ARGS_NUMBER, op1, op2);
      ilix = ad2altili(opc, op1, op2, ilix);
      return ilix;
    }
#if defined(TARGET_X8664)
    if (!flg.ieee) {
      (void)mk_prototype(gnr_math("pow", 's', 'q', FMTH_I_QPOWQ, 0), "f pure",
                         DT_QUAD, ARGS_NUMBER, DT_QUAD, DT_QUAD);
      ilix =
          ad_func(IL_DFRQP, IL_QJSR, gnr_math("pow", 's', 'q', FMTH_I_QPOWQ, 0),
                  ARGS_NUMBER, op1, op2);
    } else
#endif
    {
#if defined(TARGET_POWER)
      if (flg.ieee) {
        (void)mk_prototype(MTH_I_QPOWQ, "f pure", DT_QUAD, ARGS_NUMBER, DT_QUAD,
                           DT_QUAD);
        ilix = ad_func(IL_DFRQP, IL_QJSR, MTH_I_QPOWQ, ARGS_NUMBER, op1, op2);
      } else {
        (void)mk_prototype(fast_math("pow", 's', 'q', MTH_I_QPOWQ), "f pure",
                           DT_QUAD, ARGS_NUMBER, DT_QUAD, DT_QUAD);
        ilix =
            ad_func(IL_DFRQP, IL_QJSR, fast_math("pow", 's', 'q', MTH_I_QPOWQ),
                    ARGS_NUMBER, op1, op2);
      }
#else
      (void)mk_prototype(MTH_I_QPOWQ, "f pure", DT_QUAD, ARGS_NUMBER, DT_QUAD,
                         DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_QPOWQ, ARGS_NUMBER, op1, op2);
#endif
    }
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
#endif

  case IL_SIGN:
    if (1) {
      /*
       * There is a performance degradation when calling a routine.
       * For the non-vector version of zeusmp, there's roughly a 50%
       * penalty on the test dataset.  The degradation when compiling
       * -Mvect is huge given that IL_SIGN & IL_DSIGN are now blockers
       * to vectorization.
       * For now, just use the original ILI sequence, but define it here
       * rather than messing with the ilmtp.n files.
       */
      ilix = ad2ili(IL_FCMPZ, op2, CC_LT);
      tmp = ad1ili(IL_FABS, op1);
      tmp1 = ad1ili(IL_FNEG, tmp);
      ilix = ad3ili(IL_FSELECT, ilix, tmp, tmp1);
      return ilix;
    }
#if defined(TARGET_POWER)
    (void)mk_prototype(MTH_I_FSIGN, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_FSIGN, 2, op1, op2);
#else
    (void)mk_prototype(MTH_I_FSIGN, "f pure", DT_FLOAT, 2, DT_FLOAT, DT_FLOAT);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_FSIGN, 2, op1, op2);
#endif
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;

  case IL_DSIGN:
    if (1) {
      /*
       * There is a performance degradation when calling a routine.
       * For the non-vector version of zeusmp, there's roughly a 50%
       * penalty on the test dataset.  The degradation when compiling
       * -Mvect is huge given that IL_SIGN & IL_DSIGN are now blockers
       * to vectorization.
       * For now, just use the original ILI sequence, but define it here
       * rather than messing with the ilmtp.n files.
       */
      ilix = ad2ili(IL_DCMPZ, op2, CC_LT);
      tmp = ad1ili(IL_DABS, op1);
      tmp1 = ad1ili(IL_DNEG, tmp);
      ilix = ad3ili(IL_DSELECT, ilix, tmp, tmp1);
      return ilix;
    }
#if defined(TARGET_POWER)
    (void)mk_prototype(MTH_I_DSIGN, "f pure", DT_DBLE, 2, DT_DBLE, DT_DBLE);
    ilix = ad_func(IL_DFRDP, IL_QJSR, MTH_I_DSIGN, 2, op1, op2);
#else
    (void)mk_prototype(MTH_I_DSIGN, "f pure", DT_DBLE, 2, DT_DBLE, DT_DBLE);
    ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DSIGN, 2, op1, op2);
#endif
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;

  case IL_ILEADZI:
    op2 = ad_icon(_ipowi(2, op2));
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_ILEADZI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_ILEADZ:
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_ILEADZ, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
  case IL_KLEADZ:
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KLEADZ, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
  case IL_ITRAILZI:
    op2 = ad_icon(_ipowi(2, op2));
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_ITRAILZI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_ITRAILZ:
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_ITRAILZ, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
  case IL_KTRAILZ:
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KTRAILZ, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
  case IL_IPOPCNTI:
    op2 = ad_icon(_ipowi(2, op2));
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_IPOPCNTI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_IPOPCNT:
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_IPOPCNT, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
  case IL_KPOPCNT:
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KPOPCNT, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_IPOPPARI:
    op2 = ad_icon(_ipowi(2, op2));
    ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_IPOPPARI, 2, op1, op2);
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_IPOPPAR:
    if (TEST_FEATURE2(FEATURE_ABM, FEATURE_SSE42)) {
      ilix = ad1ili(IL_IPOPCNT, op1);
      ilix = ad2ili(IL_AND, ilix, ad_icon((INT)1));
    } else {
      ilix = ad_func(IL_DFRIR, IL_QJSR, MTH_I_IPOPPAR, 1, op1);
    }
    ilix = ad1altili(opc, op1, ilix);
    return ilix;
  case IL_KPOPPAR:
    ilix = ad_func(IL_DFRKR, IL_QJSR, MTH_I_KPOPPAR, 1, op1);
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_LEQV:
    tmp = ad2ili(IL_XOR, op1, op2);
    tmp = ad1ili(IL_NOT, tmp);
    return ad2altili(opc, op1, op2, tmp);

  case IL_ALLOC:
    ilix = new_ili(ilip); /* Can't allow sharing; use new_ili() */
#if defined(TARGET_X8664)
    if (IL_RES(ILI_OPC(op1)) != ILIA_KR) {
      op1 = ad1ili(IL_IKMV, op1);
      ILI_OPND(ilix, 1) = op1;
    }
#endif
    return ilix;

  case IL_VA_ARG:
    ilix = new_ili(ilip); /* Can't allow sharing; use new_ili() */
    return ilix;

  case IL_APURE: /* call a pure function, no arguments, returning AR */
  case IL_IPURE: /* call a pure function, no arguments, returning IR */
    ilix = ilip->opnd[1];
    ilix = ad1altili(opc, op1, ilix);
    return ilix;

  case IL_APUREA: /* call a pure function, one AR argument, returning AR */
  case IL_APUREI: /* call a pure function, one IR argument, returning AR */
  case IL_IPUREA: /* call a pure function, one AR argument, returning IR */
  case IL_IPUREI: /* call a pure function, one IR argument, returning IR */
    ilix = ilip->opnd[2];
    ilix = ad2altili(opc, op1, op2, ilix);
    return ilix;
  case IL_ALLOCA:
    break;
  case IL_VDIV:
    if (ILI_OPC(ilip->opnd[2]) != IL_NULL) /* in a conditonal branch */
    {
      root = "div";
      mth_fn = MTH_div;
      goto do_vect2;
    }
    break;
  case IL_VSQRT:
    if (ILI_OPC(ilip->opnd[1]) != IL_NULL) /* in a conditonal branch */
    {
      root = "sqrt";
      mth_fn = MTH_sqrt;
      goto do_vect1;
    }
    break;
  case IL_VRSQRT:
  case IL_VRCP:
  case IL_VNEG:
  case IL_VADD:
  case IL_VSUB:
  case IL_VMUL:
  case IL_VDIVZ:
    break;
  case IL_VMOD:
    if (ILI_OPC(ilip->opnd[2]) != IL_NULL) /* in a conditonal branch */
    {
      root = "mod";
      mth_fn = MTH_mod;
      goto do_vect2;
    }
    break;
  case IL_VMODZ:
  case IL_VCVTV:
    break;
  case IL_VCVTS:
    if (ncons == 1) {
      switch (ILI_OPC(op1)) {
      case IL_ICON:
      case IL_FCON:
        ilix = ad1ili(IL_VCON,
                      get_vcon_scalar(CONVAL2G(ILI_OPND(op1, 1)), (DTYPE)op2));
        return ilix;
      case IL_KCON:
      case IL_DCON:
        ilix = ad1ili(IL_VCON, get_vcon_scalar(ILI_OPND(op1, 1), (DTYPE)op2));
        return ilix;
      default:
        break;
      }
    }
    break;
  /***** { do not forget to update ili_get_vect_dtype() { *****/
  case IL_VNOT:
  case IL_VAND:
  case IL_VOR:
  case IL_VXOR:
  case IL_VLSHIFTV:
  case IL_VRSHIFTV:
  case IL_VLSHIFTS:
  case IL_VRSHIFTS:
  case IL_VURSHIFTS:
  case IL_VMIN:
  case IL_VMAX:
  case IL_VABS:
  case IL_VCMP:
  case IL_VCMPNEQ:
  case IL_VFMA1:
  case IL_VFMA2:
  case IL_VFMA3:
  case IL_VFMA4:
    break;
  case IL_VCOS:
    root = "cos";
    mth_fn = MTH_cos;
    goto do_vect1;
  case IL_VSIN:
    root = "sin";
    mth_fn = MTH_sin;
    goto do_vect1;
  case IL_VACOS:
    root = "acos";
    mth_fn = MTH_acos;
    goto do_vect1;
  case IL_VASIN:
    root = "asin";
    mth_fn = MTH_asin;
    goto do_vect1;
  case IL_VTAN:
    root = "tan";
    mth_fn = MTH_tan;
    goto do_vect1;
  case IL_VSINCOS:
    break;
  case IL_VSINH:
    root = "sinh";
    mth_fn = MTH_sinh;
    goto do_vect1;
  case IL_VCOSH:
    root = "cosh";
    mth_fn = MTH_cosh;
    goto do_vect1;
  case IL_VTANH:
    root = "tanh";
    mth_fn = MTH_tanh;
    goto do_vect1;
  case IL_VATAN:
    root = "atan";
    mth_fn = MTH_atan;
    goto do_vect1;
  case IL_VEXP:
    root = "exp";
    mth_fn = MTH_exp;
    goto do_vect1;
  case IL_VLOG:
    root = "log";
    mth_fn = MTH_log;
    goto do_vect1;
  case IL_VLOG10:
    root = "log10";
    mth_fn = MTH_log10;
    goto do_vect1;
  case IL_VFLOOR:
    root = "floor";
    mth_fn = MTH_floor;
    goto do_vect1;
  case IL_VCEIL:
    root = "ceil";
    mth_fn = MTH_ceil;
    goto do_vect1;
  case IL_VAINT:
    root = "aint";
    mth_fn = MTH_aint;
    goto do_vect1;
  do_vect1:
    mask_ili = ilip->opnd[1];
    if (ILI_OPC(mask_ili) == IL_NULL) /* no mask */
    {
      ilix = ad_func(IL_NONE, IL_GJSR,
                     vect_math(mth_fn, root, 1, DTypeILIOpnd(ilip, 2), opc, 0, 0, false),
                     1, op1);
    } else /* need to generate call to mask version */
    {
      ilix = ad_func(IL_NONE, IL_GJSR,
                     vect_math(mth_fn, root, 2, DTypeILIOpnd(ilip, 2), opc, 0, 0, true),
                     2, op1, mask_ili);
    }
    return ad3altili(opc, op1, mask_ili, ilip->opnd[2], ilix);
  case IL_VATAN2:
    root = "atan2";
    mth_fn = MTH_atan2;
    goto do_vect2;
  case IL_VPOW:
    root = "pow";
    mth_fn = MTH_pow;
    goto do_vect2;
  do_vect2:
    mask_ili = ilip->opnd[2];
    if (ILI_OPC(mask_ili) == IL_NULL)
    {
      /* no mask */
      ilix = ad_func(IL_NONE, IL_GJSR,
                     vect_math(mth_fn, root, 2, DTypeILIOpnd(ilip, 3), opc, 0, 0, false),
                     2, op1, op2);
    } else
    {
      /* need to generate call to mask version */
      ilix = ad_func(IL_NONE, IL_GJSR,
                     vect_math(mth_fn, root, 3, DTypeILIOpnd(ilip, 3), opc, 0, 0, true),
                     3, op1, op2, mask_ili);
    }
    return ad4altili(opc, op1, op2, mask_ili, ilip->opnd[3], ilix);
  case IL_VPOWI:
  case IL_VPOWK:
  case IL_VPOWIS:
  case IL_VPOWKS:
  case IL_VFPOWK:
  case IL_VDPOWI:
  case IL_VFPOWKS:
  case IL_VDPOWIS:
    switch (opc) {
    case IL_VPOWI:
    case IL_VDPOWI:
      root = "powi";
      mth_fn = MTH_powi;
      break;
    case IL_VPOWK:
    case IL_VFPOWK:
      root = "powk";
      mth_fn = MTH_powk;
      break;
    case IL_VPOWIS:
    case IL_VDPOWIS:
      root = "powi1";
      mth_fn = MTH_powi1;
      break;
    case IL_VPOWKS:
    case IL_VFPOWKS:
      root = "powk1";
      mth_fn = MTH_powk1;
      break;
    default:
      break;
    }
    assert(IL_VECT(ILI_OPC(op1)), "addarth():expected vector opc", ILI_OPC(op1),
           ERR_Fatal);
    vdt1 = ili_get_vect_dtype(op1);
    if (IL_VECT(ILI_OPC(op2))) {
      vdt2 = ili_get_vect_dtype(op2);
    } else if (IL_TYPE(ILI_OPC(op2)) == ILTY_LOAD) {
      if (ILI_OPC(op2) == IL_LD)
        vdt2 = DT_INT;
      else if (ILI_OPC(op2) == IL_LDKR)
        vdt2 = DT_INT8;
      else {
#if DEBUG
        assert(0, "addarth(): unrecognized load type", ILI_OPC(op2), ERR_Fatal);
#endif
        vdt2 = DT_INT;
      }
    } else if (IL_TYPE(ILI_OPC(op2)) == ILTY_CONS) {
      if (ILI_OPC(op2) == IL_ICON)
        vdt2 = DT_INT;
      else if (ILI_OPC(op2) == IL_KCON)
        vdt2 = DT_INT8;
      else {
#if DEBUG
        assert(0, "addarth(): unrecognized constant type", ILI_OPC(op2),
               ERR_Fatal);
#endif
        vdt2 = DT_INT;
      }
    } else if (IL_TYPE(ILI_OPC(op2)) == ILTY_ARTH) {
      if (IL_RES(ILI_OPC(op2)) == ILIA_IR)
        vdt2 = DT_INT;
      else if (IL_RES(ILI_OPC(op2)) == ILIA_KR)
        vdt2 = DT_INT8;
      else {
#if DEBUG
        assert(0, "addarth(): unrecognized arth type", IL_TYPE(ILI_OPC(op2)),
               ERR_Fatal);
#endif
        vdt2 = DT_INT;
      }
    } else if (IL_TYPE(ILI_OPC(op2)) == ILTY_MOVE) {
      switch(IL_RES(ILI_OPC(op2)))
      {
        case ILIA_IR:
          vdt2 = DT_INT;
          break;
        case ILIA_KR:
          vdt2 = DT_INT8;
          break;
        default:
          assert(0, "addarth(): bad move result for operand 2", 
                 IL_RES(ILI_OPC(op2)), ERR_Fatal);
      }
    } else
      assert(0, "addarth(): bad type for operand 2", IL_TYPE(ILI_OPC(op2)),
             ERR_Fatal);
    mask_ili = ilip->opnd[2];
    if (ILI_OPC(mask_ili) == IL_NULL) /* no mask */
    {
      ilix = ad_func(IL_NONE, IL_GJSR,
                     vect_math(mth_fn, root, 2, DTypeILIOpnd(ilip, 3), opc, vdt1, vdt2, false),
                     2, op1, op2);
    } else /* need to generate call to mask version */
    {
      ilix = ad_func(IL_NONE, IL_GJSR,
                     vect_math(mth_fn, root, 3, DTypeILIOpnd(ilip, 3),   opc, vdt1, vdt2, true),
                     3, op1, op2, mask_ili);
    }
    return ad4altili(opc, op1, op2, mask_ili, ilip->opnd[3], ilix);
    /***** }  do not forget to update ili_get_vect_dtype() } *****/
    break;

#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ABS:
  case IL_FLOAT128CHS:
  case IL_FLOAT128RNDINT:
  case IL_FLOAT128TO:
  case IL_FLOAT128FROM:
  case IL_FLOAT128ADD:
  case IL_FLOAT128SUB:
  case IL_FLOAT128MUL:
  case IL_FLOAT128DIV:
  case IL_FLOAT128CMP:
    break;
#endif /* LONG_DOUBLE_FLOAT128 */

  case IL_FCEIL:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_ceil, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_FCEIL, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_FCEIL, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled",
             opc, ERR_Informational);
#endif
    break;

  case IL_DCEIL:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_ceil, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_DCEIL, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DCEIL, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled",
             opc, ERR_Informational);
#endif
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCEIL:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_ceil, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_QCEIL, "f pure", DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_QCEIL, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled", opc,
             ERR_Informational);
#endif
    break;
#endif

  case IL_FFLOOR:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_floor, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_FFLOOR, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_FFLOOR, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled",
             opc, ERR_Informational);
#endif
    break;

  case IL_DFLOOR:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_floor, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_DFLOOR, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DFLOOR, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled",
             opc, ERR_Informational);
#endif
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFLOOR:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_floor, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }

#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_QFLOOR, "f pure", DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_QFLOOR, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled", opc,
             ERR_Informational);
#endif
    break;
#endif

  case IL_AINT:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_aint, &funcsptr, 1, false, DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_AINT, "f pure", DT_FLOAT, 1, DT_FLOAT);
      ilix = ad_func(IL_spfunc, IL_QJSR, MTH_I_AINT, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled",
             opc, ERR_Informational);
#endif
    break;

  case IL_DINT:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_aint, &funcsptr, 1, false, DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_DINT, "f pure", DT_DBLE, 1, DT_DBLE);
      ilix = ad_func(IL_dpfunc, IL_QJSR, MTH_I_DINT, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled",
             opc, ERR_Informational);
#endif
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QINT:
    if (XBIT_NEW_MATH_NAMES) {
      fname = make_math(MTH_aint, &funcsptr, 1, false, DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, fname, 1, op1);
      ilix = ad1altili(opc, op1, ilix);
      return ilix;
    }
#if defined(TARGET_LLVM_ARM) || defined(TARGET_WIN)
    else {
      (void)mk_prototype(MTH_I_QINT, "f pure", DT_QUAD, 1, DT_QUAD);
      ilix = ad_func(IL_qpfunc, IL_QJSR, MTH_I_QINT, 1, op1);
      return ad1altili(opc, op1, ilix);
    }
#else
    else
      interr("addarth: old math name for ili not handled",
             opc, ERR_Informational);
#endif
     break;
#endif

  default:

#if DEBUG
    interr("addarth:ili not handled", opc, ERR_Informational);
#endif
    break;
  }

  newili.opc = opc;
  newili.opnd[0] = op1;
  newili.opnd[1] = op2;
  for (i = 2; i < IL_OPRS(opc); ++i)
    newili.opnd[i] = ilip->opnd[i];
  return get_ili(&newili);

add_icon:
  return ad_icon(res.numi[1]);

add_kcon:
  return ad1ili(IL_KCON, getcon(res.numi, DT_INT8));

add_rcon:
  res.numi[0] = 0;
  return ad1ili(IL_FCON, getcon(res.numi, DT_FLOAT));

add_dcon:
  return ad1ili(IL_DCON, getcon(res.numi, DT_DBLE));

#ifdef TARGET_SUPPORTS_QUADFP
add_qcon:
  return ad1ili(IL_QCON, getcon(qres.numi, DT_QUAD));
#endif
}

static int
gen_sincos(ILI_OP opc, int op1, ILI_OP sincos_opc, ILI_OP fopc, MTH_FN fn,
           DTYPE dt, ILI_OP dfr_opc)
{
  int ilix;
  char *fname;

#if defined(TARGET_X8664) || defined(TARGET_POWER)
  /* only if using new names */
  if (XBIT(164, 0x800000))
    if (!XBIT(15, 0x08)) {
      ilix = ad1ili(sincos_opc, op1);
      ilix = ad1ili(fopc, ilix);
      return ilix;
    }
#endif
  fname = make_math(fn, NULL, 1, false, dt, 1, dt);
  ilix = ad_func(dfr_opc, IL_QJSR, fname, 1, op1);
  ilix = ad1altili(opc, op1, ilix);
  return ilix;
}

#if defined(TARGET_X8664) || defined(TARGET_POWER)
static int
_newton_fdiv(int op1, int op2)
{
  int i, x0, tmp1, ilix;
  /*
   * Newton's appx for recip:
   *   x1 = (2.0 - x * x0) * x0
   */
  i = ad1ili(IL_FCON, stb.flt2);
  x0 = ad1ili(IL_RCPSS, op2);
  tmp1 = ad2ili(IL_FMUL, op2, x0);
  tmp1 = ad2ili(IL_FSUB, i, tmp1);
  tmp1 = ad2ili(IL_FMUL, tmp1, x0);
  ilix = ad2ili(IL_FMUL, op1, tmp1);
  return ilix;
}
#endif

static bool
do_newton_sqrt(void)
{
#if !defined(TARGET_LLVM_ARM)
  if (!flg.ieee && !XBIT(15, 0x20000000) /* not -Mfpapprox */
      && XBIT(15, 0x10) && !XBIT(15, 0x10000) &&
      (XBIT(15, 0x20) || TEST_MACH(MACH_INTEL_PENTIUM4)))
    return true;
#endif
  return false;
}

/** \brief Determine if 'val' is a power of 2; the range of the powers is 1 ...
 * max_pwr, inclusive.
 */
static int
_pwr2(INT val, int max_pwr)
{
  int i;
  INT v;
  v = 2; /* a single bit representing the powers of 2 */
  for (i = 1; i <= max_pwr; i++) {
    if ((v & val) == 0)
      v <<= 1; /* slide the bit over  */
    else if (v == val)
      return i;
  }
  return 0;
}

/** \brief Determine if the 64-bit value represented by (cv1<<32)+cv2 is a
 * power of 2; the range of the powers is 1 ... max_pwr, inclusive.
 */
static int
_kpwr2(INT cv1, INT cv2, int max_pwr)
{
  int i;
  int mp;

  if (max_pwr > 31)
    mp = 31;
  else
    mp = max_pwr;
  if (cv1 == 0) {
    return _pwr2(cv2, mp);
  }
  if (cv2)
    return 0;
  if (max_pwr < 32)
    return 0;
  if (cv1 == 1)
    return 32;
  i = _pwr2(cv1, max_pwr - 32);
  if (i)
    return i + 32;
  return 0;
}

/**
 * \brief constant folds the integer add ili (iadd)
 */
static int
red_iadd(int ilix, INT con)
{
  int lop, rop, New;
  ILI_OP opc;
  static INT val;
  static ILI newili;

  opc = ILI_OPC(ilix);
  switch (opc) {

  case IL_ICON:
    return ad_icon(CONVAL2G(ILI_OPND(ilix, 1)) + con);

  case IL_IADD:
  case IL_UIADD:
    lop = ILI_OPND(ilix, 1);
    rop = ILI_OPND(ilix, 2);
    New = red_iadd(rop, con);
    if (New != 0) {
      newili.opc = opc;
      newili.opnd[0] = lop;
      newili.opnd[1] = New;
      if (ILI_OPC(New) == IL_ICON) {
        val = CONVAL2G(ILI_OPND(New, 1));
        if (val == 0)
          return lop;
        if ((opc == IL_IADD) && (val < 0 && val != (INT)0x80000000)) {
          newili.opc = IL_ISUB;
          newili.opnd[1] = ad_icon(-val);
        }
      } else if (lop > New) {
        newili.opnd[0] = New;
        newili.opnd[1] = lop;
      }
      return get_ili(&newili);
    }
    New = red_iadd(lop, con);
    if (New != 0) {
      newili.opc = opc;
      if (New > rop) {
        newili.opnd[0] = rop;
        newili.opnd[1] = New;
      } else {
        newili.opnd[0] = New;
        newili.opnd[1] = rop;
      }
      return get_ili(&newili);
    }
    break;

  case IL_ISUB:
  case IL_UISUB:
    lop = ILI_OPND(ilix, 1);
    rop = ILI_OPND(ilix, 2);
    New = red_iadd(lop, con);
    if (New != 0) {
      if (ILI_OPC(New) == IL_ICON && ILI_OPND(New, 1) == stb.i0)
        return ad1ili(IL_INEG, rop);
      newili.opc = opc;
      newili.opnd[0] = New;
      newili.opnd[1] = rop;
      return get_ili(&newili);
    }
    if (opc == IL_ISUB && con > 0 && ILI_OPC(rop) == IL_ICON) {
      UINT uv = CONVAL2G(ILI_OPND(rop, 1));
      if ((uv + (UINT)con) > 0x80000000U) {
        /*
         * (rop + -con) can be smaller than INT_MIN on 64-bit; e.g.,
         * in mp_correct clmp, an induction init expression is
         * presented as (iistart - INT_MIN) + 1 and becomes
         * iistart - (INT_MIN - 1) (obviously, an invalid re-association)
         */
        break;
      }
    }
    New = red_iadd(rop, -con);
    if (New != 0) {
      newili.opc = opc;
      newili.opnd[0] = lop;
      newili.opnd[1] = New;
      if (ILI_OPC(New) == IL_ICON) {
        val = CONVAL2G(ILI_OPND(New, 1));
        if (val == 0)
          return lop;
        if ((opc == IL_ISUB) && (val < 0 && val != (INT)0x80000000)) {
          newili.opc = IL_IADD;
          newili.opnd[1] = ad_icon(-val);
        }
      }
      return get_ili(&newili);
    }
    break;
  default:;
  } /*****  end of switch(ILI_OPC(ilix))  *****/

  return 0;
}

/**
 * \brief constant folds the integer*8 add ili (kadd)
 */
static int
red_kadd(int ilix, INT con[2])
{
  int lop, rop, New;
  ILI_OP opc;
  INT tmp[2];
  static INT val[2];
  static ILI newili;

  opc = ILI_OPC(ilix);
  switch (opc) {
  default:
    break;

  case IL_KCON:
    val[0] = CONVAL1G(ILI_OPND(ilix, 1));
    val[1] = CONVAL2G(ILI_OPND(ilix, 1));
    add64(val, con, val);
    return ad1ili(IL_KCON, getcon(val, DT_INT8));

  case IL_KADD:
  case IL_UKADD:
    lop = ILI_OPND(ilix, 1);
    rop = ILI_OPND(ilix, 2);
    New = red_kadd(rop, con);
    if (New != 0) {
      newili.opc = opc;
      newili.opnd[0] = lop;
      newili.opnd[1] = New;
      if (ILI_OPC(New) == IL_KCON) {
        val[0] = CONVAL1G(ILI_OPND(New, 1));
        val[1] = CONVAL2G(ILI_OPND(New, 1));
        if (val[0] == 0 && val[1] == 0)
          return lop;
        if (opc == IL_KADD && val[0] < 0 &&
            !(val[0] == (INT)0x80000000 && val[1] == 0)) {
          newili.opc = IL_KSUB;
          neg64(val, val);
          newili.opnd[1] = ad1ili(IL_KCON, getcon(val, DT_INT8));
        }
      } else if (lop > New) {
        newili.opnd[0] = New;
        newili.opnd[1] = lop;
      }
      return get_ili(&newili);
    }
    New = red_kadd(lop, con);
    if (New != 0) {
      newili.opc = opc;
      if (New > rop) {
        newili.opnd[0] = rop;
        newili.opnd[1] = New;
      } else {
        newili.opnd[0] = New;
        newili.opnd[1] = rop;
      }
      return get_ili(&newili);
    }
    break;

  case IL_KSUB:
  case IL_UKSUB:
    lop = ILI_OPND(ilix, 1);
    rop = ILI_OPND(ilix, 2);
    New = red_kadd(lop, con);
    if (New != 0) {
      if (ILI_OPC(New) == IL_KCON && ILI_OPND(New, 1) == stb.k0)
        return ad1ili(IL_KNEG, rop);
      newili.opc = opc;
      newili.opnd[0] = New;
      newili.opnd[1] = rop;
      return get_ili(&newili);
    }
    neg64(con, tmp);
    New = red_kadd(rop, tmp);
    if (New != 0) {
      newili.opc = opc;
      newili.opnd[0] = lop;
      newili.opnd[1] = New;
      if (ILI_OPC(New) == IL_KCON) {
        val[0] = CONVAL1G(ILI_OPND(New, 1));
        val[1] = CONVAL2G(ILI_OPND(New, 1));
        if (val[0] == 0 && val[1] == 0)
          return lop;
        if (opc == IL_KSUB && val[0] < 0 &&
            !(val[0] == (INT)0x80000000 && val[1] == 0)) {
          newili.opc = IL_KADD;
          neg64(val, val);
          newili.opnd[1] = ad1ili(IL_KCON, getcon(val, DT_INT8));
        }
      }
      return get_ili(&newili);
    }
    break;
  } /*****  end of switch(ILI_OPC(ilix))  *****/

  return 0;
}

/**
 * \brief constant folds the address add ili (aadd)
 */
static int
red_aadd(int ilix, SPTR sym, ISZ_T off, int scale)
{
  SPTR lop;
  int rop, New, oldsc;
  SPTR vsym;
  ISZ_T voff;
  static ILI newili;

  switch (ILI_OPC(ilix)) {
  default:
    break;
  case IL_ACON:
    lop = ILI_SymOPND(ilix, 1);
    vsym = SymConval1(lop);
    if (scale >= 0)
      off <<= scale;
    else
      off = (double)off / ((INT)1 << (-scale));
    voff = ACONOFFG(lop) + off;
    if (sym == vsym) {
      if (sym == 0)
        return ad_aconi(voff);
      return ad_acon(vsym, voff);
    } else if (vsym == 0) {
      vsym = sym;
      return ad_acon(vsym, voff);
    } else if (sym == 0)
      return ad_acon(vsym, voff);
    break;

  case IL_AADD:
    lop = ILI_SymOPND(ilix, 1);
    rop = ILI_OPND(ilix, 2);
    oldsc = ILI_OPND(ilix, 3);
    New = red_aadd(lop, sym, off, scale);
    if (New != 0) {
      newili.opc = IL_AADD;
      newili.opnd[0] = New;
      newili.opnd[1] = rop;
      newili.opnd[2] = oldsc;
      return get_ili(&newili);
    }
    if (scale < oldsc)
      break;
    New = red_aadd(rop, sym, off, scale - oldsc);
    if (New != 0) {
      newili.opc = IL_AADD;
      if (ILI_OPC(New) == IL_ACON && ACONOFFG(ILI_OPND(New, 1)) == 0 &&
          CONVAL1G(ILI_OPND(New, 1)) == 0)
        return lop;
      newili.opnd[0] = lop;
      newili.opnd[1] = New;
      newili.opnd[2] = oldsc;
      return get_ili(&newili);
    }
    switch (ILI_OPC(rop)) {
    default:
      break;
    case IL_IAMV:
    case IL_KAMV:
      /*
       * Push the constant below the AADD of the IAMV/KAMV.
       */
      New = ad_acon(sym, off);
      newili.opc = IL_AADD;
      newili.opnd[0] = lop;
      newili.opnd[1] = New;
      newili.opnd[2] = scale;
      lop = sptrGetILI(&newili);
      newili.opnd[0] = lop;
      newili.opnd[1] = rop;
      newili.opnd[2] = oldsc;
      return get_ili(&newili);
    }
    break;

  case IL_ASUB:
    lop = ILI_SymOPND(ilix, 1);
    rop = ILI_OPND(ilix, 2);
    oldsc = ILI_OPND(ilix, 3);
    New = red_aadd(lop, sym, off, scale);
    if (New != 0) {
      newili.opc = IL_ASUB;
      newili.opnd[0] = New;
      newili.opnd[1] = rop;
      newili.opnd[2] = oldsc;
      return get_ili(&newili);
    }
    if (scale < oldsc)
      break;
    if (sym == 0) {
      New = red_aadd(rop, SPTR_NULL, -off, scale - oldsc);
      if (New != 0) {
        newili.opc = IL_ASUB;
        newili.opnd[0] = lop;
        newili.opnd[1] = New;
        newili.opnd[2] = oldsc;
        return get_ili(&newili);
      }
    }
    break;

  } /*****  end of switch(ILI_OPC(ilix))  *****/

  return 0;
}

/**
 * \brief combines the operand of damv's in an aadd expr
 *
 * Routine which simplifies an aadd expression where one of the operands is a
 * damv ILI.  This routine combines the operand of the damv with any other
 * damv's in opr by performing an iadd followed by a damv.
 *
 * \param opr   locates expression of an aadd
 * \param damv  damv ILI
 * \param scale scale of aadd - if >= 0 scale operand of damv, if < 0
 * - scale operand of opr if damv
 */
static int
red_damv(int opr, int damv, int scale)
{
  int tmp, oldsc;
  int o1;

  switch (ILI_OPC(opr)) {
  default:
    break;
  case IL_IAMV:
    if (scale >= 0) {
      o1 = ILI_OPND(damv, 1);
      if (IL_RES(ILI_OPC(o1)) != ILIA_KR) {
        tmp = ad2ili(IL_IMUL, o1, ad_icon((INT)(1 << scale)));
        return ad1ili(IL_IAMV, ad2ili(IL_IADD, (int)ILI_OPND(opr, 1), tmp));
      }
      tmp = ad2ili(IL_KMUL, o1, ad_kconi((INT)(1 << scale)));
      o1 = ad1ili(IL_IKMV, ILI_OPND(opr, 1));
      return ad1ili(IL_KAMV, ad2ili(IL_KADD, o1, tmp));
    }
    tmp = ad2ili(IL_IMUL, (int)ILI_OPND(opr, 1), ad_icon((INT)(1 << (-scale))));
    return ad1ili(IL_IAMV, ad2ili(IL_IADD, (int)ILI_OPND(damv, 1), tmp));

  case IL_KAMV:
    if (scale >= 0) {
      o1 = ILI_OPND(damv, 1);
      if (IL_RES(ILI_OPC(o1)) == ILIA_IR)
        o1 = ad1ili(IL_IKMV, o1);
      tmp = ad2ili(IL_KMUL, o1, ad_kconi((INT)(1 << scale)));
      return ad1ili(IL_KAMV, ad2ili(IL_KADD, (int)ILI_OPND(opr, 1), tmp));
    }
    tmp =
        ad2ili(IL_KMUL, (int)ILI_OPND(opr, 1), ad_kconi((INT)(1 << (-scale))));
    return ad1ili(IL_KAMV, ad2ili(IL_KADD, (int)ILI_OPND(damv, 1), tmp));

  case IL_AADD:
    /*
     * the case looked for is <acon> <+> <rop>
     */
    oldsc = ILI_OPND(opr, 3);
    tmp = red_damv((int)ILI_OPND(opr, 2), damv, scale - oldsc);
    if (tmp) {
      if (oldsc <= scale) {
        return ad3ili(IL_AADD, (int)ILI_OPND(opr, 1), tmp, oldsc);
      }
      return ad3ili(IL_AADD, (int)ILI_OPND(opr, 1), tmp, scale);
    }
    break;

  case IL_ASUB:
    /*
     * the case looked for is <lop> <-> <acon>
     */
    oldsc = ILI_OPND(opr, 3);
    tmp = red_damv((int)ILI_OPND(opr, 1), damv, scale - oldsc);
    if (tmp) {
      if (oldsc <= scale) {
        return ad3ili(IL_ASUB, tmp, (int)ILI_OPND(opr, 2), oldsc);
      }
      return ad3ili(IL_ASUB, tmp, (int)ILI_OPND(opr, 2), scale);
    }
    break;
  }
  return 0;
}

/**
 * Simplify
 * <pre>
 *    max( max(x, y), y )
 *    max( y, max(x, y) )
 * </pre>
 * as
 * <pre>
 *    max(x, y)
 * </pre>
 */
static int
red_minmax(ILI_OP opc, int op1, int op2)
{
  ILI newili;

  if (op1 == op2)
    return op1; /* max(x,x) == x */
  if (opc == ILI_OPC(op1)) {
    if (ILI_OPND(op1, 2) == op2)
      return op1;
  } else if (opc == ILI_OPC(op2)) {
    if (ILI_OPND(op2, 2) == op1)
      return op2;
  }
  newili.opc = opc;
  newili.opnd[0] = op1;
  newili.opnd[1] = op2;
  return get_ili(&newili);
}

/** \brief Transform -(c<OP>x<OP>y) into (-c)<OP>x<OP>y; OP is a mult or divide
 */
static int
red_negate(int old, ILI_OP neg_opc, int mult_opc, int div_opc)
{
  int op1, op2;
  int New;
  op1 = ILI_OPND(old, 1);
  op2 = ILI_OPND(old, 2);
  if (IL_TYPE(ILI_OPC(op2)) == ILTY_CONS) {
    op2 = ad1ili(neg_opc, op2);
    return ad2ili(ILI_OPC(old), op1, op2); /* could be mult or divide */
  }
  if (IL_TYPE(ILI_OPC(op1)) == ILTY_CONS) {
    if (!XBIT(15, 0x4) || 
	(div_opc != IL_FDIV && div_opc != IL_DDIV)) {
      /* don't do if mult by recip enabled */
      op1 = ad1ili(neg_opc, op1);
      return ad2ili(ILI_OPC(old), op1, op2); /* should only be a divide */
    }
  }
  if (ILI_OPC(op2) == mult_opc) {
    New = red_negate(op2, neg_opc, mult_opc, div_opc);
    if (New != op2) {
      return ad2ili(ILI_OPC(old), op1, New); /* could be mult or divide */
    }
  }
  if (ILI_OPC(op2) == div_opc) {
    New = red_negate(op2, neg_opc, mult_opc, div_opc);
    if (New != op2) {
      return ad2ili(ILI_OPC(old), op1, New); /* could be mult or divide */
    }
  }
  if (ILI_OPC(op1) == mult_opc) {
    New = red_negate(op1, neg_opc, mult_opc, div_opc);
    if (New != op1) {
      return ad2ili(ILI_OPC(old), New, op2); /* could be mult or divide */
    }
  }
  if (ILI_OPC(op1) == div_opc) {
    New = red_negate(op1, neg_opc, mult_opc, div_opc);
    if (New != op1) {
      return ad2ili(ILI_OPC(old), New, op2); /* could be mult or divide */
    }
  }
  return old;
}

static bool
_is_nand(SPTR sptr)
{
  int v, e, m;
  /*
   *  our fp cannoical form (big endian IEEE):
   *  struct {
   *      unsigned int ml;
   *      unsigned int mh:20;
   *      unsigned int e:11;
   *      unsigned int s:1;
   *  };
   * A NaN has an exponent field of all one's and a non-zero mantissa.
   */
  v = CONVAL1G(sptr);
  e = (v >> 20) & 0x7ff;
  if (e == 0x7ff) {
    m = v & 0xfffff;
    if (m || CONVAL2G(sptr))
      return true;
  }
  return false;
}

static int
addother(ILI *ilip)
{
  int ilix;
  int ncons;
  ILI_OP opc, opc1;
  int op1;
  int op2, cons2, con2v2;
  opc = ilip->opc;
  ncons = 0;
  switch (opc) {
  case IL_ISELECT:
  case IL_ASELECT:
  case IL_KSELECT:
  case IL_FSELECT:
  case IL_DSELECT:
  case IL_CSSELECT:
  case IL_CDSELECT:
    op1 = ilip->opnd[0];
    opc1 = ILI_OPC(op1);
    if (IL_TYPE(opc1) == ILTY_CONS) {
      cons2 = ILI_OPND(op1, 1);
      con2v2 = CONVAL2G(cons2);
      if (con2v2 == 0)
        return ilip->opnd[1];
      return ilip->opnd[2];
    }
    if (ilip->opnd[1] == ilip->opnd[2]) {
      if (!func_in(op1))
        return ilip->opnd[1];
    }
    if (opc1 == IL_ICMPZ) {
      int new_op1 = cmpz_of_cmp(ILI_OPND(op1, 1), CC_ILI_OPND(op1, 2));
      if (new_op1 >= 0)
        return ad3ili(opc, new_op1, ilip->opnd[1], ilip->opnd[2]);
    }
    break;
  case IL_DEALLOC:
    ilix = new_ili(ilip); /* Can't allow sharing; use new_ili() */
    op1 = ad_func(IL_NONE, IL_JSR, "_mp_free", 1, ilip->opnd[0]);
    ILI_ALT(ilix) = op1;
    iltb.callfg = 1;
    return ilix;
  case IL_DPDP2DCMPLXI0:
    op1 = ilip->opnd[0];
    if (ILI_OPC(op1) == IL_DCON) {
      INT numi[2];
      numi[0] = ILI_OPND(op1, 1);
      numi[1] = stb.dbl0;
      return ad1ili(IL_DCMPLXCON, getcon(numi, DT_DCMPLX));
    }
    break;
  case IL_DPDP2DCMPLX:
    op1 = ilip->opnd[0];
    op2 = ilip->opnd[1];
    if (ILI_OPC(op1) == IL_DCON && ILI_OPC(op2) == IL_DCON) {
      INT numi[2];
      numi[0] = ILI_OPND(op1, 1);
      numi[1] = ILI_OPND(op2, 1);
      return ad1ili(IL_DCMPLXCON, getcon(numi, DT_DCMPLX));
    }
    break;
  case IL_SPSP2SCMPLXI0:
    op1 = ilip->opnd[0];
    if (ILI_OPC(op1) == IL_FCON) {
      INT numi[2];
      numi[0] = CONVAL2G(ILI_OPND(op1, 1));
      numi[1] = 0;
      return ad1ili(IL_SCMPLXCON, getcon(numi, DT_CMPLX));
    }
    break;
  case IL_SPSP2SCMPLX:
    op1 = ilip->opnd[0];
    op2 = ilip->opnd[1];
    if (ILI_OPC(op1) == IL_FCON && ILI_OPC(op2) == IL_FCON) {
      INT numi[2];
      numi[0] = CONVAL2G(ILI_OPND(op1, 1));
      numi[1] = CONVAL2G(ILI_OPND(op2, 1));
      return ad1ili(IL_SCMPLXCON, getcon(numi, DT_CMPLX));
    }
    break;
  case IL_DCMPLX2REAL:
    op1 = ilip->opnd[0];
    if (ILI_OPC(op1) == IL_DPDP2DCMPLX || ILI_OPC(op1) == IL_DPDP2DCMPLXI0) {
      return ILI_OPND(op1, 1);
    } else if (ILI_OPC(op1) == IL_DCMPLXCON) {
      return ad1ili(IL_DCON, CONVAL1G(ILI_OPND(op1, 1)));
    }
    break;
  case IL_DCMPLX2IMAG:
    op1 = ilip->opnd[0];
    if (ILI_OPC(op1) == IL_DPDP2DCMPLX) {
      return ILI_OPND(op1, 2);
    } else if (ILI_OPC(op1) == IL_DPDP2DCMPLXI0) {
      return ad1ili(IL_DCON, stb.dbl0);
    } else if (ILI_OPC(op1) == IL_DCMPLXCON) {
      return ad1ili(IL_DCON, CONVAL2G(ILI_OPND(op1, 1)));
    }
    break;
  case IL_SCMPLX2REAL:
    op1 = ilip->opnd[0];
    if (ILI_OPC(op1) == IL_SPSP2SCMPLX || ILI_OPC(op1) == IL_SPSP2SCMPLXI0) {
      return ILI_OPND(op1, 1);
    } else if (ILI_OPC(op1) == IL_SCMPLXCON) {
      INT numi[2];
      numi[0] = 0;
      numi[1] = CONVAL1G(ILI_OPND(op1, 1));
      return ad1ili(IL_FCON, getcon(numi, DT_FLOAT));
    }
    break;
  case IL_SCMPLX2IMAG:
    op1 = ilip->opnd[0];
    if (ILI_OPC(op1) == IL_SPSP2SCMPLX) {
      return ILI_OPND(op1, 2);
    } else if (ILI_OPC(op1) == IL_SPSP2SCMPLXI0) {
      return ad1ili(IL_FCON, stb.flt0);
    } else if (ILI_OPC(op1) == IL_SCMPLXCON) {
      INT numi[2];
      numi[0] = 0;
      numi[1] = CONVAL2G(ILI_OPND(op1, 1));
      return ad1ili(IL_FCON, getcon(numi, DT_FLOAT));
    }
    break;

    /* Use IL_ATOMICRMWI to detect atomic support. */
  case IL_CMPXCHGI:
  case IL_CMPXCHGKR:
  case IL_CMPXCHGA:
  case IL_ATOMICRMWI:
  case IL_ATOMICRMWKR:
  case IL_ATOMICRMWA:
  case IL_ATOMICRMWSP:
  case IL_ATOMICRMWDP:
    /* Can't allow sharing; use new_ili() */
    ilix = new_ili(ilip);
    return ilix;
  case IL_FENCE:
    if (value_of_irlnk_operand(ilip->opnd[1], MO_SEQ_CST) == MO_RELAXED) {
      /* Fence has no effect. */
      return ad_free(ilip->opnd[1]);
    }
    break;
  default:
    break;
  }

  /* default return */
  return get_ili(ilip);

} /* addother */

/**
 * \brief adds branch ili
 */
static int
addbran(ILI *ilip)
{
  int op1, op2;
  CC_RELATION cc_op2;
  CC_RELATION new_cond;
  CC_RELATION cond;
  int lab, cmp_val;
  int tmp;

  union { /* constant value structure	 */
    INT numi[2];
    UINT numu[2];
    DBLE numd;
  } num1, num2;

  op1 = ilip->opnd[0];
  if (ilip->opc != IL_JMP) {
    /* purify UMR: second operand isn't set for IL_JMP */
    op2 = ilip->opnd[1];
    cc_op2 = (CC_RELATION) op2;
  }
  switch (ilip->opc) {
  default:
    break;

  case IL_LCJMPZ:
    /*
     * Fortran logical compare and jump:
     *   op2 = 1 (EQ), jump if op1 is false
     *   op2 = 2 (NE), jump if op1 is true
     */
    assert(cc_op2 == CC_EQ || cc_op2 == CC_NE, "addbran:bad stc of LCJMPZ", op2,
           ERR_Severe);
    switch (ILI_OPC(op1)) {

    case IL_ICON:
      if (cc_op2 == CC_EQ) {
        if ((CONVAL2G(ILI_OPND(op1, 1)) & 1) == 0)
          return ad1ili(IL_JMP, (int)ilip->opnd[2]);
      } else { /* NE constant optimization */
        if (CONVAL2G(ILI_OPND(op1, 1)))
          return ad1ili(IL_JMP, (int)ilip->opnd[2]);
      }
      RFCNTD(ilip->opnd[2]);
      return 0;

    case IL_ICMPZ: /* if the operand being tested is a compare */
    case IL_FCMPZ: /* ILI, switch to the ILI which does the */
    case IL_DCMPZ: /* integer compare with zero and branches */
    case IL_ACMPZ: /* if the condition is met */
    case IL_ICMP:
    case IL_FCMP:
    case IL_DCMP:
#ifdef TARGET_SUPPORTS_QUADFP
    case IL_QCMP:
#endif
    case IL_ACMP:
#if defined(TARGET_X8664)
    case IL_KCMPZ:
    case IL_KCMP:
#endif
      return ad3ili(IL_ICJMPZ, op1, op2, ilip->opnd[2]);

    case IL_NOT: /* remove the NOT and negate the condition */
      new_cond = (cc_op2 == CC_EQ) ? CC_NE : CC_EQ;
      return ad3ili(IL_LCJMPZ, (int)ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);

    default:
      break;

    } /*****  end of switch(ILI_OPC(op1))  *****/

/* generate sequence of AND and ICJMPZ ili */
    if (IL_RES(ILI_OPC(op1)) == ILIA_KR) {
#if defined(TARGET_X8664)
      tmp = ad_kcon(0, 1);
      op1 = ad2ili(IL_KAND, op1, tmp);
      return ad3ili(IL_KCJMPZ, op1, op2, ilip->opnd[2]);
#else
      op1 = ad1ili(IL_KIMV, op1);
#endif
    }
    op1 = ad2ili(IL_AND, op1, ad_icon(1));
    return ad3ili(IL_ICJMPZ, op1, op2, ilip->opnd[2]);

  case IL_ICJMPZ:
    switch (ILI_OPC(op1)) {
    default:
      break;
    case IL_ICON:
      cond = cc_op2;
      lab = ilip->opnd[2];
      cmp_val = icmp(CONVAL2G(ILI_OPND(op1, 1)), 0);
      goto fold_jmp;

    case IL_ICMPZ:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == 0)
        break;
      return ad3ili(IL_ICJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);

    case IL_UICMPZ:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == 0)
        break;
      return ad3ili(IL_UICJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);

    case IL_KCMPZ:
      /*
       * Make sure the original condition is still used when
       * referencing the alternate (a ICMPZ) ili.  Cannot fold
       * the condition twice!!!!
       */
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == CC_EQ || new_cond == CC_NE) {
        return ad3ili(IL_KCJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);
      }
      return ad3ili(IL_ICJMPZ, ILI_ALT(op1), op2, ilip->opnd[2]);
    case IL_UKCMPZ:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == 0)
        break;
      return ad3ili(IL_UKCJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);

    case IL_FCMPZ:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == 0)
        break;
      return ad3ili(IL_FCJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);

    case IL_DCMPZ:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == 0)
        break;
      return ad3ili(IL_DCJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);

#ifdef TARGET_SUPPORTS_QUADFP
    case IL_QCMPZ:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == 0)
        break;
      return ad3ili(IL_QCJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);
#endif

    case IL_ACMPZ:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 2), cc_op2);
      if (new_cond == 0)
        break;
      return ad3ili(IL_ACJMPZ, ILI_OPND(op1, 1), new_cond, ilip->opnd[2]);

    case IL_ICMP:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2);
      if (new_cond == 0)
        break;
      return ad4ili(IL_ICJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);

    case IL_UICMP:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2);
      if (new_cond == 0)
        break;
      return ad4ili(IL_UICJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);
    case IL_UKCMP:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2);
      if (new_cond == 0)
        break;
      return ad4ili(IL_UKCJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);
    case IL_KCMP:
      new_cond = combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2);
      if (new_cond == 0)
        break;
      return ad4ili(IL_KCJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);

    case IL_FCMP:
      new_cond =
          (!IEEE_CMP)
              ? combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2)
              : combine_ieee_ccs(CC_ILI_OPND(op1, 3), cc_op2);
      if (new_cond == 0)
        break;
      return ad4ili(IL_FCJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);

    case IL_DCMP:
      new_cond = (!IEEE_CMP)
                     ? combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2)
                     : combine_ieee_ccs(CC_ILI_OPND(op1, 3), cc_op2);
      if (new_cond == 0)
        break;
      return ad4ili(IL_DCJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);

#ifdef TARGET_SUPPORTS_QUADFP
    case IL_QCMP:
      new_cond = (!IEEE_CMP)
                     ? combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2)
                     : combine_ieee_ccs(CC_ILI_OPND(op1, 3), cc_op2);
      if (new_cond == 0)
        break;
      return ad4ili(IL_QCJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);
#endif

    case IL_ACMP:
      if ((new_cond = combine_int_ccs(CC_ILI_OPND(op1, 3), cc_op2)) ==
          0)
        break;
      return ad4ili(IL_ACJMP, ILI_OPND(op1, 1), ILI_OPND(op1, 2), new_cond,
                    ilip->opnd[2]);

    } /*****  end of switch(ILI_OPC(op1))  *****/

    break;

  case IL_ICJMP:

    /* 0/1  <rel>  i  -->  i  <rev(rel)>z  */

    cond = CCRelationILIOpnd(ilip, 2);
    if (ILI_OPC(op1) == IL_ICON) {
      if (ILI_OPND(op1, 1) == stb.i0)
        return ad3ili(IL_ICJMPZ, op2, commute_cc(cond), ilip->opnd[3]);
      if (ILI_OPND(op1, 1) == stb.i1) {
        if (cond == CC_LE)
          /*  1 LE x  --> x GT z */
          return ad3ili(IL_ICJMPZ, op2, CC_GT, ilip->opnd[3]);
        if (cond == CC_GT)
          /*  1 GT x  --> x LE z */
          return ad3ili(IL_ICJMPZ, op2, CC_LE, ilip->opnd[3]);
      }
      if (CONVAL2G(ILI_OPND(op1, 1)) >= 1 && is_zero_one(op2)) {
        /* low-quality range analysis */
        switch (cond) {
        default:
          break;
        case CC_EQ:
        case CC_LE:
          if (CONVAL2G(ILI_OPND(op1, 1)) > 1) {
            /* 2 <= x is false */
            if (func_in(op2))
              break;
            RFCNTD(ilip->opnd[3]);
            return 0;
          }
          /* 1 <= x becomes x != 0 */
          return ad3ili(IL_ICJMPZ, op2, CC_NE, ilip->opnd[3]);
        case CC_NE:
        case CC_GT:
          if (CONVAL2G(ILI_OPND(op1, 1)) > 1) {
            /* 2 > x is true */
            if (func_in(op2))
              break;
            return ad1ili(IL_JMP, ilip->opnd[3]);
          }
          /* 1 > x becomes x == 0 */
          return ad3ili(IL_ICJMPZ, op2, CC_EQ, ilip->opnd[3]);
        case CC_LT: /* 1 < x never true */
          if (func_in(op2))
            break;
          RFCNTD(ilip->opnd[3]);
          return 0;
        case CC_GE: /* 1 >= x always true */
          if (func_in(op2))
            break;
          return ad1ili(IL_JMP, ilip->opnd[3]);
        }
      }

      /*  constant <rel> constant ---> constant fold */
      if (ILI_OPC(op2) == IL_ICON) {
        lab = ilip->opnd[3];
        cmp_val = icmp(CONVAL2G(ILI_OPND(op1, 1)), CONVAL2G(ILI_OPND(op2, 1)));
        goto fold_jmp;
      }
    }

    /* i  <rel>  0/1  -->  i  <rel>z  */

    if (ILI_OPC(op2) == IL_ICON) {
      if (ILI_OPND(op2, 1) == stb.i0)
        return ad3ili(IL_ICJMPZ, op1, cond, (int)ilip->opnd[3]);
      if (ILI_OPND(op2, 1) == stb.i1) {
        if (cond == CC_GE)
          /*  x GE 1  --> x GT z */
          return ad3ili(IL_ICJMPZ, op1, CC_GT, (int)ilip->opnd[3]);
        if (cond == CC_LT)
          /*  x LT 1  --> x LE z */
          return ad3ili(IL_ICJMPZ, op1, CC_LE, (int)ilip->opnd[3]);
      }
      if (CONVAL2G(ILI_OPND(op2, 1)) >= 1 && is_zero_one(op1)) {
        /* low-quality range analysis */
        switch (cond) {
        default:
          break;
        case CC_EQ:
        case CC_GE:
          if (CONVAL2G(ILI_OPND(op2, 1)) > 1) {
            /* x >= 2 is false */
            if (func_in(op1))
              break;
            RFCNTD(ilip->opnd[3]);
            return 0;
          }
          /* x >= 1 becomes x != 0 */
          return ad3ili(IL_ICJMPZ, op1, CC_NE, ilip->opnd[3]);
        case CC_NE:
        case CC_LT:
          if (CONVAL2G(ILI_OPND(op2, 1)) > 1) {
            /* x < 2 is true */
            if (func_in(op1))
              break;
            return ad1ili(IL_JMP, ilip->opnd[3]);
          }
          /* x < 1 becomes x == 0 */
          return ad3ili(IL_ICJMPZ, op1, CC_EQ, ilip->opnd[3]);
        case CC_GT: /* x > 1 never true */
          if (func_in(op1))
            break;
          RFCNTD(ilip->opnd[3]);
          return 0;
        case CC_LE: /* x <= 1 always true */
          if (func_in(op1))
            break;
          return ad1ili(IL_JMP, ilip->opnd[3]);
        }
      }
    }
    goto cjmp_2; /* check if operands are identical */

  case IL_UICJMPZ:
    switch (ILI_OPC(op1)) {
    case IL_ICON:
      cond = cc_op2;
      lab = ilip->opnd[2];
      cmp_val = xucmp(CONVAL2G(ILI_OPND(op1, 1)), (INT)0);
      goto fold_jmp;

    default:
      switch (op2) {
      case CC_EQ:
        break;

      case CC_NE:
        break;

      case CC_LT:
        if (func_in(op1))
          break;
        RFCNTD(ilip->opnd[2]);
        return 0;

      case CC_GE:
        if (func_in(op1))
          break;
        return ad1ili(IL_JMP, (int)ilip->opnd[2]);

      case CC_LE:
        ilip->opnd[1] = CC_EQ;
        break;

      case CC_GT:
        ilip->opnd[1] = CC_NE;
        break;
      }
      break;

    } /*****  end of switch(ILI_OPC(op1))  *****/

    break;

  case IL_UICJMP:

    /* 0  <rel>  i  -->  i  <rev(rel)>z  */

    if (ILI_OPC(op1) == IL_ICON) {
      if (ILI_OPND(op1, 1) == stb.i0)
        return ad3ili(IL_UICJMPZ, op2, commute_cc(CCRelationILIOpnd(ilip, 2)),
                      (int)ilip->opnd[3]);

      /*  constant <rel> constant ---> constant fold */
      if (ILI_OPC(op2) == IL_ICON) {
        cond = CCRelationILIOpnd(ilip, 2);
        lab = ilip->opnd[3];
        cmp_val = xucmp(CONVAL2G(ILI_OPND(op1, 1)), CONVAL2G(ILI_OPND(op2, 1)));
        goto fold_jmp;
      }
    }

    /* i  <rel>  0  -->  i  <rel>z  */

    if (ILI_OPC(op2) == IL_ICON && ILI_OPND(op2, 1) == stb.i0)
      return ad3ili(IL_UICJMPZ, op1, (int)ilip->opnd[2], (int)ilip->opnd[3]);
    goto cjmp_2; /* check if operands are identical */

  case IL_KCJMPZ:
    if (ILI_OPC(op1) == IL_KCON) {
      cond = cc_op2;
      lab = ilip->opnd[2];
      GETVALI64(num1, ILI_OPND(op1, 1));
      GETVALI64(num2, stb.k0);
      cmp_val = cmp64(num1.numi, num2.numi);
      goto fold_jmp;
    }
    break;
  case IL_KCJMP:
    /* 0  <rel>  i  -->  i  <rev(rel)>z  */
    if (ILI_OPC(op1) == IL_KCON) {
      if (ILI_OPND(op1, 1) == stb.k0)
        return ad3ili(IL_KCJMPZ, op2, commute_cc(CCRelationILIOpnd(ilip, 2)),
                      ilip->opnd[3]);

      /*  constant <rel> constant ---> constant fold */
      if (ILI_OPC(op2) == IL_KCON) {
        cond = CCRelationILIOpnd(ilip, 2);
        lab = ilip->opnd[3];
        GETVALI64(num1, ILI_OPND(op1, 1));
        GETVALI64(num2, ILI_OPND(op2, 1));
        cmp_val = cmp64(num1.numi, num2.numi);
        goto fold_jmp;
      }
    }

    /* i  <rel>  0  -->  i  <rel>z  */
    if (ILI_OPC(op2) == IL_KCON && ILI_OPND(op2, 1) == stb.k0)
      return ad3ili(IL_KCJMPZ, op1, (int)ilip->opnd[2], (int)ilip->opnd[3]);
    goto cjmp_2; /* check if operands are identical */
  case IL_UKCJMPZ:
    switch (ILI_OPC(op1)) {
    case IL_KCON:
      cond = cc_op2;
      lab = ilip->opnd[2];
      GETVALI64(num1, ILI_OPND(op1, 1));
      GETVALI64(num2, stb.k0);
      cmp_val = ucmp64(num1.numu, num2.numu);
      goto fold_jmp;
    default:
      switch (op2) {
      case CC_EQ:
        break;
      case CC_NE:
        break;
      case CC_LT:
        if (func_in(op1))
          break;
        RFCNTD(ilip->opnd[2]);
        return 0;
      case CC_GE:
        if (func_in(op1))
          break;
        return ad1ili(IL_JMP, (int)ilip->opnd[2]);
      case CC_LE:
        ilip->opnd[1] = CC_EQ;
        break;
      case CC_GT:
        ilip->opnd[1] = CC_NE;
        break;
      }
      break;

    } /*****  IL_UKCJMPZ: end of switch(ILI_OPC(op1))  *****/
    break;
  case IL_UKCJMP:
    /* 0  <rel>  i  -->  i  <rev(rel)>z  */
    if (ILI_OPC(op1) == IL_KCON) {
      if (ILI_OPND(op1, 1) == stb.k0)
        return ad3ili(IL_UKCJMPZ, op2,
                      commute_cc(CCRelationILIOpnd(ilip, 2)),
                      ilip->opnd[3]);

      /*  constant <rel> constant ---> constant fold */
      if (ILI_OPC(op2) == IL_KCON) {
        cond = CCRelationILIOpnd(ilip, 2);
        lab = ilip->opnd[3];
        GETVALI64(num1, ILI_OPND(op1, 1));
        GETVALI64(num2, ILI_OPND(op2, 1));
        cmp_val = ucmp64(num1.numu, num2.numu);
        goto fold_jmp;
      }
    }
    /* i  <rel>  0  -->  i  <rel>z  */
    if (ILI_OPC(op2) == IL_KCON && ILI_OPND(op2, 1) == stb.k0)
      return ad3ili(IL_UKCJMPZ, op1, (int)ilip->opnd[2], (int)ilip->opnd[3]);
    goto cjmp_2; /* check if operands are identical */

  case IL_FCJMPZ:
    if (ILI_OPC(op1) == IL_FCON && IS_FLT0(ILI_OPND(op1, 1))) {
      if (op2 == CC_EQ || op2 == CC_GE || op2 == CC_LE || op2 == CC_NOTNE ||
          op2 == CC_NOTLT || op2 == CC_NOTGT)
        return ad1ili(IL_JMP, (int)ilip->opnd[2]);
      RFCNTD(ilip->opnd[2]);
      return 0;
    }
#if defined(TARGET_X86) && !defined(TARGET_LLVM_ARM)
    if (mach.feature[FEATURE_SCALAR_SSE]) { /* scalar sse code gen. don't use
                                               FCJMPZ */
      tmp = ad1ili(IL_FCON, stb.flt0);
      return ad4ili(IL_FCJMP, op1, tmp, op2, (int)ilip->opnd[2]);
    }
#endif
#ifndef TM_FCJMPZ
    tmp = ad1ili(IL_FCON, stb.flt0);
    return ad4ili(IL_FCJMP, op1, tmp, op2, (int)ilip->opnd[2]);
#endif
    break;

  case IL_FCJMP:
#if defined(TARGET_X86)
    if (mach.feature[FEATURE_SCALAR_SSE]) {
      /* scalar sse code gen. don't use FCJMPZ; it costs less to
       * 'compute' 0.0 rather than fetch from memory
       */
      goto nogen_fcjmpz;
    }
#endif
#ifdef TM_FCJMPZ
    if (ILI_OPC(op1) == IL_FCON && IS_FLT0(ILI_OPND(op1, 1)))
      return ad3ili(IL_FCJMPZ, op2,
                    commute_cc(CCRelationILIOpnd(ilip, 2)),
                    ilip->opnd[3]);
    if (ILI_OPC(op2) == IL_FCON && IS_FLT0(ILI_OPND(op2, 1)))
      return ad3ili(IL_FCJMPZ, op1, ilip->opnd[2], ilip->opnd[3]);
#endif
#if defined(TARGET_X86)
  nogen_fcjmpz:
#endif
    if (op1 == op2 && ILI_OPC(op2) == IL_FCON && !_is_nanf(ILI_OPND(op2, 1))) {
      cond = CCRelationILIOpnd(ilip, 2);
      if (cond == CC_EQ || cond == CC_GE || cond == CC_LE || cond == CC_NOTNE ||
          cond == CC_NOTLT || cond == CC_NOTGT)
        return ad1ili(IL_JMP, (int)ilip->opnd[3]);
      RFCNTD(ilip->opnd[3]);
      return 0;
    }
    if (!IEEE_CMP)
      goto cjmp_2; /* check if operands are identical */
    break;

  case IL_DCJMPZ:
    if (ILI_OPC(op1) == IL_DCON && IS_DBL0(ILI_OPND(op1, 1))) {
      if (op2 == CC_EQ || op2 == CC_GE || op2 == CC_LE || op2 == CC_NOTNE ||
          op2 == CC_NOTLT || op2 == CC_NOTGT)
        return ad1ili(IL_JMP, ilip->opnd[2]);
      RFCNTD(ilip->opnd[2]);
      return 0;
    }
#if defined(TARGET_X86)
    if (mach.feature[FEATURE_SCALAR_SSE]) {
      /* scalar sse code gen. don't use DCJMPZ */
      tmp = ad1ili(IL_DCON, stb.dbl0);
      return ad4ili(IL_DCJMP, op1, tmp, op2, (int)ilip->opnd[2]);
    }
#endif
#ifndef TM_DCJMPZ
    tmp = ad1ili(IL_DCON, stb.dbl0);
    return ad4ili(IL_DCJMP, op1, tmp, op2, (int)ilip->opnd[2]);
#endif
    break;

  case IL_DCJMP:
#if defined(TARGET_X86)
    if (mach.feature[FEATURE_SCALAR_SSE]) {
      /* scalar sse code gen. don't use DCJMPZ; it costs less to
       * 'compute' 0.0 rather than fetch from memory
       */
      goto nogen_dcjmpz;
    }
#endif
#ifdef TM_DCJMPZ
    if (ILI_OPC(op1) == IL_DCON && IS_DBL0(ILI_OPND(op1, 1)))
      return ad3ili(IL_DCJMPZ, op2, commute_cc(CCRelationILIOpnd(ilip, 2)),
                    ilip->opnd[3]);
    if (ILI_OPC(op2) == IL_DCON && IS_DBL0(ILI_OPND(op2, 1)))
      return ad3ili(IL_DCJMPZ, op1, ilip->opnd[2], ilip->opnd[3]);
#endif
#if defined(TARGET_X86)
  nogen_dcjmpz:
#endif
    if (op1 == op2 && (ILI_OPC(op2) == IL_DCON) &&
        !_is_nand(ILI_SymOPND(op2, 1))) {
      cond = CCRelationILIOpnd(ilip, 2);
      if (cond == CC_EQ || cond == CC_GE || cond == CC_LE || cond == CC_NOTNE ||
          cond == CC_NOTLT || cond == CC_NOTGT)
        return ad1ili(IL_JMP, (int)ilip->opnd[3]);
      RFCNTD(ilip->opnd[3]);
      return 0;
    }
    if (!IEEE_CMP)
      goto cjmp_2; /* check if operands are identical */
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMPZ:
    if (ILI_OPC(op1) == IL_QCON && IS_QUAD0(ILI_OPND(op1, 1))) {
      if (op2 == CC_EQ || op2 == CC_GE || op2 == CC_LE || op2 == CC_NOTNE ||
          op2 == CC_NOTLT || op2 == CC_NOTGT)
        return ad1ili(IL_JMP, ilip->opnd[2]);
      RFCNTD(ilip->opnd[2]);
      return 0;
    }
#if defined(TARGET_X86)
    if (mach.feature[FEATURE_SCALAR_SSE]) {
      /* scalar sse code gen. don't use QCJMPZ */
      tmp = ad1ili(IL_QCON, stb.quad0);
      return ad4ili(IL_QCJMP, op1, tmp, op2, (int)ilip->opnd[2]);
    }
#endif
#ifndef TM_QCJMPZ
    tmp = ad1ili(IL_QCON, stb.quad0);
    return ad4ili(IL_QCJMP, op1, tmp, op2, (int)ilip->opnd[2]);
#endif
    break;

  case IL_QCJMP:
#if defined(TARGET_X86)
    if (mach.feature[FEATURE_SCALAR_SSE]) {
      /* scalar sse code gen. don't use QCJMPZ; it costs less to
       * 'compute' 0.0 rather than fetch from memory
       */
      goto nogen_qcjmpz;
    }
#endif
#ifdef TM_QCJMPZ
    if (ILI_OPC(op1) == IL_QCON && IS_QUAD0(ILI_OPND(op1, 1)))
      return ad3ili(IL_QCJMPZ, op2, commute_cc(CCRelationILIOpnd(ilip, 2)),
                    ilip->opnd[3]);
    if (ILI_OPC(op2) == IL_QCON && IS_QUAD0(ILI_OPND(op2, 1)))
      return ad3ili(IL_QCJMPZ, op1, ilip->opnd[2], ilip->opnd[3]);
#endif
#if defined(TARGET_X86)
  nogen_qcjmpz:
#endif
    if (op1 == op2 && (ILI_OPC(op2) == IL_QCON) &&
        !_is_nand(ILI_SymOPND(op2, 1))) {
      cond = CCRelationILIOpnd(ilip, 2);
      if (cond == CC_EQ || cond == CC_GE || cond == CC_LE || cond == CC_NOTNE ||
          cond == CC_NOTLT || cond == CC_NOTGT)
        return ad1ili(IL_JMP, (int)ilip->opnd[3]);
      RFCNTD(ilip->opnd[3]);
      return 0;
    }
    if (!IEEE_CMP)
      goto cjmp_2; /* check if operands are identical */
    break;
#endif

  case IL_ACJMPZ:
    if (ILI_OPC(op1) == IL_ACON) {
      int sym;
      sym = CONVAL1G(ILI_OPND(op1, 1));
      if (sym == 0) {
        INT v;
        v = CONVAL2G(ILI_OPND(op1, 1));
        switch (op2) {
        case CC_EQ:
        case CC_LE:
          tmp = v == 0;
          break;
        case CC_NE:
        case CC_GT:
          tmp = v != 0;
          break;
        case CC_LT:
          tmp = 0;
          break;
        case CC_GE:
          tmp = 1;
          break;
        }
        if (tmp)
          return ad1ili(IL_JMP, ilip->opnd[2]);
        RFCNTD(ilip->opnd[2]);
        return 0;
      }
      /* comparing an address with NULL */
      switch (op2) {
      case CC_LT:
        RFCNTD(ilip->opnd[2]);
        return 0;
      case CC_EQ:
      case CC_LE:
        if (IS_LCL_OR_DUM(sym) || CCSYMG(sym)) {
          RFCNTD(ilip->opnd[2]);
          return 0;
        }
        break;
      case CC_GE:
        return ad1ili(IL_JMP, ilip->opnd[2]);
      default:
        if (IS_LCL_OR_DUM(sym) || CCSYMG(sym))
          return ad1ili(IL_JMP, ilip->opnd[2]);
        break;
      }
    }
    break;

  case IL_ACJMP:
    if (ILI_OPC(op1) == IL_ACON && CONVAL1G(ILI_OPND(op1, 1)) == 0 &&
        CONVAL2G(ILI_OPND(op1, 1)) == 0)
      return ad3ili(IL_ACJMPZ, op2,
                    commute_cc(CCRelationILIOpnd(ilip, 2)), ilip->opnd[3]);
    if (ILI_OPC(op2) == IL_ACON && CONVAL1G(ILI_OPND(op2, 1)) == 0 &&
        CONVAL2G(ILI_OPND(op2, 1)) == 0)
      return ad3ili(IL_ACJMPZ, op1, ilip->opnd[2], ilip->opnd[3]);
  cjmp_2:
    if (op1 == op2 && !func_in(op1)) {
      /*  i <rel> i  -->  jmp or fall thru  */
      cond = CCRelationILIOpnd(ilip, 2);
      if (cond == CC_EQ || cond == CC_GE || cond == CC_LE)
        return ad1ili(IL_JMP, ilip->opnd[3]);
      RFCNTD(ilip->opnd[3]);
      return 0;
    }
    break;

  case IL_JMPM:

    /* not cased for JMPMs in C because:
     * 1. JMPM ili does not provide enough information to associate jump
     *    table value with the case label (more than 1 JMPM can be generated
     *    for a single C switch; lower bound is not provided).  For Fortran,
     *    the range is simply [0, n-1]; precisely one JMPM is generated for
     *    a computed goto.
     * 2. fortran is well-behaved wrt completeness of the jump table
     *    (table in C may have holes),
     * 3. for fortran, default case is next block (default label is 4th
     *    operand).
     */
    if (ILI_OPC(op1) == IL_ICON) {
      int swarr;  /* cc sym representing table of labels */
      int n;      /* number of cases in JMPM */
      int tv;     /* constant index value for jmp */
      int v;      /* current index value of jmpm label */
      int lab;    /* eventual target of JMPM; 0 => default */
      SWEL *swel; /* curr SWEL ptr - linear for fortran */

      swarr = ilip->opnd[2];
#if DEBUG
      assert(ilip->opnd[3], "addbranJMPM 4th opnd zero", swarr, ERR_Severe);
      assert(ILI_OPC(op2) == IL_ICON, "addbranJMPM, range not icon", op2,
             ERR_Severe);
#endif
      n = CONVAL2G(ILI_OPND(op2, 1));
      tv = CONVAL2G(ILI_OPND(op1, 1));
      lab = 0;
      RFCNTD(ilip->opnd[3]); /* no need for default */
      swel = switch_base + SWELG(swarr);
      for (v = 0; v < n; v++, swel++) {
        RFCNTD(swel->clabel);
        if (v == tv)
          lab = swel->clabel;
      };
      if (lab) {
        RFCNTI(lab);
        return ad1ili(IL_JMP, lab);
      }
      return 0;
    }
    break;

  } /*****  end of switch(ilip->opc)  *****/

  return get_ili(ilip);

fold_jmp:
  switch (cond) {
  default:
    break;
  case CC_EQ:
    if (cmp_val == 0)
      return ad1ili(IL_JMP, lab);
    break;
  case CC_NE:
    if (cmp_val != 0)
      return ad1ili(IL_JMP, lab);
    break;
  case CC_LT:
    if (cmp_val < 0)
      return ad1ili(IL_JMP, lab);
    break;
  case CC_GE:
    if (cmp_val >= 0)
      return ad1ili(IL_JMP, lab);
    break;
  case CC_LE:
    if (cmp_val <= 0)
      return ad1ili(IL_JMP, lab);
    break;
  case CC_GT:
    if (cmp_val > 0)
      return ad1ili(IL_JMP, lab);
    break;
  }
  RFCNTD(lab);
  return 0;
}

bool is_floating_comparison_opcode(ILI_OP opc);

/** \brief adds a branch ili by complementing the condition
 *
 * This routine adds a branch ili whose condition is the complement of the
 * condition specified by the ili ilix.  lbl is the target of the branch.  The
 * complement is defined by the following: (Z, NZ), (LT, GE), LE, GT)
 */
int
compl_br(int ilix, int lbl)
{
  ILI New;
  int i;
  ILI_OP opc;

  opc = ILI_OPC(ilix);
  i = ilis[opc].oprs;
  New.opc = opc;
  New.opnd[--i] = lbl;
  if (is_floating_comparison_opcode(opc)) {
    New.opnd[i - 1] = complement_ieee_cc(CC_ILI_OPND(ilix, i));
  } else {
    New.opnd[i - 1] = complement_int_cc(CC_ILI_OPND(ilix, i));
  }
  while (--i > 0)
    New.opnd[i - 1] = ILI_OPND(ilix, i);
  return addili((ILI *)&New);
}

/**
 * \brief compare two INTs
 */
static INT
icmp(INT v1, INT v2)
{
  if (v1 < v2)
    return -1;
  if (v1 > v2)
    return 1;
  return 0;
}

/**
 * \brief convert a compare value to a logical value for a relation
 *
 * Converts the result (val) of comparison to a logical value based on the
 * relation rel. The possible values of a comparison and their meanings are:
 * <pre>
 *   -1  -  op1 < op2
 *    0  -  op1 = op2
 *    1  -  op1 > op2
 * </pre>
 * The values of rel and their meanings are enumerated in the switch
 *  statement.
 */
static INT
cmp_to_log(INT val, int rel)
{
  INT logval;

  switch (rel) {
  case CC_EQ:
  case CC_NOTNE:
    logval = (val & 1) ^ 1;
    break;
  case CC_NE:
  case CC_NOTEQ:
    logval = val & 1;
    break;
  case CC_LT:
  case CC_NOTGE:
    logval = ((unsigned)val) >> 31;
    break;
  case CC_GE:
  case CC_NOTLT:
    logval = (((unsigned)val) >> 31) ^ 1;
    break;
  case CC_LE:
  case CC_NOTGT:
    logval = icmp((INT)1, val);
    break;
  case CC_GT:
  case CC_NOTLE:
    logval = icmp((INT)1, val) ^ 1;
    break;
  default:
    interr("cmp_to_log: bad relation", rel, ERR_Severe);
    return 0;
  }
/*
 * At this point, logval's value is either 1 or 0 (C's definition of
 * logical value);  watch for fortran ...
 */
  logval = logval ? SCFTN_TRUE : SCFTN_FALSE;
  return logval;
}

/** \brief Subtract two integers and check for overflow */
static bool
isub_ovf(INT c1, INT c2, INT *r)
{
  *r = c1 - c2;
  /* overflow if signs of c1 and c2 are different and the sign of the
   * result is different than the sign of c1.
   */
  if (((c1 ^ c2) >> 31) && ((c1 ^ *r) >> 31))
    return true;
  return false;
}

/** \brief Add two integers and check for overflow */
static bool
iadd_ovf(INT c1, INT c2, INT *r)
{
  *r = c1 + c2;
  /* overflow if signs of c1 and c2 are the same and the sign of the
   * result is different.
   */
  if (((c1 ^ c2) >> 31) == 0 && ((c1 ^ *r) >> 31))
    return true;
  return false;
}

/**
 * \brief enter ili into ILI area by attempting to share
 */
static int
get_ili(ILI *ilip)
{
  int i, p;
  int indx, tab;
  ILI_OP opc = ilip->opc;
  int val = opc;
  int noprs = ilis[opc].oprs;

  /* compute the hash index for this ILI  */

  for (i = 0; i < noprs; i++)
    val ^= (ilip->opnd[i] >> (i * 4));
  indx = val % ILHSHSZ;
  /*
   * calculate which hash table to use which is based on the number of
   * operands
   */

  assert(noprs <= ILTABSZ, "get_ili: noprs > ILTABSZ", opc, ERR_Severe);
  tab = (noprs == 0) ? 0 : noprs - 1;
  /* search the hash links for this ILI  */

  for (p = ilhsh[tab][indx]; p != 0; p = ILI_HSHLNK(p)) {
    if (opc == ILI_OPC(p)) {
      for (i = 1; i <= noprs; i++)
        if (ilip->opnd[i - 1] != ILI_OPND(p, i))
          goto next;
      return p; /* F O U N D  */
    }
  next:;
  }

  /*
   * NOT FOUND -- if no more storage is available, check for zero use
   * counts. If one does not exist, get more storage
   */

  p = STG_NEXT_FREELIST(ilib);

  /*
   * NEW ENTRY - add the ili to the ili area and to its hash chain
   */
  BZERO(&ilib.stg_base[p], ILI, 1);
  ILI_OPCP(p, opc);
  for (i = 1; i <= noprs; i++)
    ILI_OPND(p, i) = ilip->opnd[i - 1];
  for (i = noprs + 1; i <= MAX_OPNDS; i++)
    ILI_OPND(p, i) = 0;
#if DEBUG
  for (i = 1; i <= noprs; ++i) {
    if (IL_ISLINK(opc, i)) {
      int opnd;
      opnd = ILI_OPND(p, i);
      if (opnd < 0 || opnd >= ilib.stg_size ||
          ILI_OPC(opnd) == GARB_COLLECTED) {
        interr("bad ili link in get_ili", opc, ERR_Severe);
      }
    }
  }
#endif

  ILI_HSHLNK(p) = ilhsh[tab][indx];
  ilhsh[tab][indx] = p;
  /*
   * Initialize nonzero fields of the ili - (here and in new_ili()).
   */
  return p;
}

/* wrapper of new_ili for external reference. */

int
get_ili_ns(ILI *ilip)
{
  return new_ili(ilip);
}

/**
   \brief enter ili into ILI area (no sharing)
 */
static int
new_ili(ILI *ilip)
{
  int i, p;
  ILI_OP opc;
  int noprs;

  opc = ilip->opc;
  noprs = ilis[opc].oprs;

#if DEBUG
  assert(noprs <= ILTABSZ, "new_ili: noprs > ILTABSZ", opc, ERR_Severe);
#endif

  /* NEW ENTRY */
  p = STG_NEXT_FREELIST(ilib);

  BZERO(&ilib.stg_base[p], ILI, 1);
  ILI_OPCP(p, opc);
  for (i = 1; i <= noprs; i++)
    ILI_OPND(p, i) = ilip->opnd[i - 1];
  for (i = noprs + 1; i <= MAX_OPNDS; i++)
    ILI_OPND(p, i) = 0;

  ILI_HSHLNK(p) = 0;
  /*
   * Initialize nonzero fields of the ili - (here and in get_ili()).
   */
  return p;
}

/* GARBAGE COLLECTION */

static void mark_ili(int);

/** \brief Sort the free list
 *
 * Sorting the free list removes a source of indeterminacy from the
 * compiler's intermediate representation that would otherwise be
 * introduced by hashing the ILIs.
 */
static int
sort_free_list(int head)
{
  int last, split[2];

  /* Split the list into two lists via alternation */
  split[0] = split[1] = 0;
  for (last = 0; head; last ^= 1) {
    int next = ILI_HSHLNK(head);
    ILI_HSHLNK(head) = split[last];
    split[last] = head;
    head = next;
  }

  /* Recursively sort the split lists */
  if (!split[1])
    return split[0];
  split[0] = sort_free_list(split[0]);
  split[1] = sort_free_list(split[1]);

  /* Merge the sorted split lists */
  for (last = 0; split[0] && split[1];) {
    int which = split[1] < split[0];
    int at = split[which];
    split[which] = ILI_HSHLNK(at);
    if (last)
      ILI_HSHLNK(last) = at;
    else
      head = at;
    last = at;
  }
  ILI_HSHLNK(last) = split[split[1] > 0];
  return head;
}

void
garbage_collect(void (*mark_function)(int))
{
  int i, j, p, q, t;

  /* first, go through and mark all the ili that are reachable from
   * the ILT.  Then, call mark_function to mark any ILI that may not
   * be reachable from the ILT.
   */
  if (DBGBIT(10, 2048))
    return;
#if DEBUG
  if (DBGBIT(10, 1024)) {
    fprintf(gbl.dbgfil, "garbage: before collect: avail: %d, free: %d\n",
            ilib.stg_avail, ilib.stg_free);
  }
#endif
  mark_ili(GARB_VISITED);
  if (mark_function != 0)
    (*mark_function)(GARB_VISITED);

  /* ILI #0, #1 is special */
  ILI_VISIT(0) = ILI_VISIT(1) = GARB_VISITED;
  /* next, go through the hash chains and delete anything that wasn't
   * marked reachable, putting the freed ili on the linked list.
   */
  for (i = 0; i < ILTABSZ; ++i)
    for (j = 0; j < ILHSHSZ; ++j) {
      q = 0;
      for (p = ilhsh[i][j]; p != 0;) {
        if (ILI_VISIT(p) == GARB_UNREACHABLE) {
          /* unreachable */
          if (q == 0)
            ilhsh[i][j] = ILI_HSHLNK(p);
          else
            ILI_HSHLNK(q) = ILI_HSHLNK(p);
          t = p;
          p = ILI_HSHLNK(p);
          STG_ADD_FREELIST(ilib, t);
          ILI_OPCP(t, GARB_COLLECTED);
          ILI_VISIT(t) = GARB_COLLECTED;
        } else {
          /* reachable */
          q = p;
          p = ILI_HSHLNK(p);
        }
      }
    }
  /* finally, go through all the ILI.  Those that have been collected
   * should be marked GARB_COLLECTED.  Those that are reachable should
   * be marked GARB_VISITED.  Those marked GARB_UNREACHABLE are
   * collected, they weren't originally in the hash chains */
  for (i = 0; i < ilib.stg_avail; ++i) {
    if (ILI_VISIT(i) == GARB_UNREACHABLE) {
      if (ILI_OPC(i) == GARB_COLLECTED) {
        /* this was put on the list in a previous garbage collect;
         * we don't want to put it on again. */
        ILI_VISIT(i) = 0;
        continue;
      }
      assert(ILI_HSHLNK(i) == 0, "garbage_collection: bad hashlnk", i,
             ERR_Fatal);
      STG_ADD_FREELIST(ilib, i);
      ILI_OPCP(i, GARB_COLLECTED);
    } else if (ILI_VISIT(i) == GARB_VISITED) {
      assert(ILI_OPC(i) != GARB_COLLECTED,
             "garbage_collection: bad opc for reachable ili", i, ERR_Fatal);
      ILI_VISIT(i) = 0;
    } else {
      /* this was collected */
      assert(ILI_VISIT(i) == GARB_COLLECTED,
             "garbage_collection: bad visit field", i, ERR_Fatal);
      assert(ILI_OPC(i) == GARB_COLLECTED,
             "garbage_collection: bad opc for collected ili", i, ERR_Fatal);
      ILI_VISIT(i) = 0;
    }
  }
#if DEBUG
  if (DBGBIT(10, 1024)) {
    fprintf(gbl.dbgfil, "garbage: after collect: avail: %d, freelist: %d\n",
            ilib.stg_avail, ilib.stg_free);
  }
  /* Do a check -- every ili should either have a valid opcode or should
     be on the free list */
  j = 0;
  for (q = ilib.stg_free; q != 0; q = ILI_HSHLNK(q)) {
    assert(ILI_OPC(q) == GARB_COLLECTED,
           "garbage_collection: bad opc for avail ili", i, ERR_Fatal);
    ++ILI_VISIT(q);
    ++j;
  }

  for (i = 0; i < ilib.stg_avail; ++i) {
    if (ILI_VISIT(i) == 0)
      assert(ILI_OPC(i) != GARB_COLLECTED,
             "garbage_collection: collected ILI not on free list", i,
             ERR_Fatal);
    else if (ILI_VISIT(i) == 1)
      assert(ILI_OPC(i) == GARB_COLLECTED,
             "garbage_collection: free ILI not marked collected", i, ERR_Fatal);
    else {
      interr("garbage_collection: ILI on free list more than once", i,
             ERR_Fatal);
    }
    ILI_VISIT(i) = 0;
  }
#endif

  if (!XBIT(15, 0x1000))
    ilib.stg_free = sort_free_list(ilib.stg_free);
}

static void mark_nme(int, int);
static void
mark_ilitree(int ili, int val)
{
  int i;
  ILI_OP opc;
  int noprs;

  if (ILI_VISIT(ili)) {
    assert(ILI_VISIT(ili) == val, "mark_ilitree: visit != val", ili, ERR_Fatal);
    return;
  }
  opc = ILI_OPC(ili);
  noprs = ilis[opc].oprs;

  ILI_VISIT(ili) = val;
  for (i = 1; i <= noprs; i++) {
    if (IL_ISLINK(opc, i))
      mark_ilitree(ILI_OPND(ili, i), val);
    else if (IL_OPRFLAG(opc, i) == ILIO_NME)
      mark_nme(ILI_OPND(ili, i), val);
  }
  if (opc == IL_SMOVE)
    mark_nme(ILI_OPND(ili, 4), val);
  if (ILI_ALT(ili))
    mark_ilitree(ILI_ALT(ili), val);
}

static void
mark_nme(int nme, int val)
{
  for (; nme; nme = NME_NM(nme)) {
    if (NME_TYPE(nme) == NT_ARR && NME_SUB(nme))
      mark_ilitree(NME_SUB(nme), val);
    else if (NME_TYPE(nme) == NT_IND && NME_SUB(nme))
      mark_ilitree(NME_SUB(nme), val);
  }
}

static void
mark_ili(int val)
{
  /* mark all the ili */
  int bihx, iltx;
  int ii;

  for (ii = 1; ii < ilib.stg_avail; ii++)
    ILI_VISIT(ii) = 0;

  for (bihx = gbl.entbih; bihx != 0; bihx = BIH_NEXT(bihx))
    for (iltx = BIH_ILTFIRST(bihx); iltx != 0; iltx = ILT_NEXT(iltx)) {
      int ilix = (int)ILT_ILIP(iltx);
      mark_ilitree(ilix, val);
    }
}

/** \brief Search the ili subtree located by ilix for functions.  */
bool
func_in(int ilix)
{
  int noprs,  /* number of lnk operands in ilix	 */
      i;      /* index variable			 */
  ILI_OP opc; /* ili opcode of ilix		 */

  if ((opc = ILI_OPC(ilix)) == IL_JSR || opc == IL_JSRA)
    return true;
  noprs = ilis[opc].oprs;
  for (i = 1; i <= noprs; i++) {
    if (IL_ISLINK(opc, i))
      if (func_in(ILI_OPND(ilix, i)))
        return true;
  }
  return false;
}

/** \brief Search the ili subtree located by ilix for QJSRs.
 *
 * This routine is used instead of ili_traverse of qjrsearch when the
 * visit_list is already being used (ili_traverse needs the visit_list).
 */
bool
qjsr_in(int ilix)
{
  int noprs,  /* number of lnk operands in ilix	 */
      i;      /* index variable			 */
  ILI_OP opc; /* ili opcode of ilix		 */

  opc = ILI_OPC(ilix);
  if (opc == IL_QJSR)
    return true;
  switch (ILI_OPC(ilix)) {
  default:
    break;
  case IL_FSINCOS:
  case IL_DSINCOS:
    return true;
  }
  if (ILI_ALT(ilix) && qjsr_in(ILI_ALT(ilix)))
    return true;
  noprs = ilis[opc].oprs;
  for (i = 1; i <= noprs; i++) {
    if (IL_ISLINK(opc, i))
      if (qjsr_in(ILI_OPND(ilix, i)))
        return true;
  }
  return false;
}

static int visit_list = 0;

bool
_find_ili(int ilix, int find_this)
{
  ILI_OP opc;
  int noprs, j;
  if (ilix == find_this) {
    ILI_VLIST(ilix) = -visit_list;
    visit_list = ilix;
    return true;
  }
  if (ILI_VLIST(ilix)) {
    if (ILI_VISIT(ilix) < 0)
      return true;
    return false;
  }
  opc = ILI_OPC(ilix);
  noprs = IL_OPRS(opc);
  for (j = 1; j <= noprs; ++j) {
    int opnd;
    if (IL_ISLINK(opc, j)) {
      opnd = ILI_OPND(ilix, j);
      if (_find_ili(opnd, find_this)) {
        ILI_VLIST(ilix) = -visit_list;
        visit_list = ilix;
        return true;
      }
    }
  }
  ILI_VLIST(ilix) = visit_list;
  visit_list = ilix;
  return false;
} /* _find_ili */

/**
   \brief determine if an ili occurs in an ili subtree
 */
bool
find_ili(int tree, int it)
{
  bool res;
  int v, nxt;
  visit_list = 1;
  ILI_VLIST(1) = 0;
  res = _find_ili(tree, it);
  for (v = visit_list; v; v = nxt) {
    nxt = ILI_VLIST(v);
    ILI_VLIST(v) = 0;
    if (nxt < 0)
      nxt = -nxt;
  }
  return res;
} /* find_ili */

/*
 * general ili rewrite mechanism which saves for each ili traversed, its
 * rewritten ili in the visit field (ILI_VISIT). This mechanism uses
 * ILI_VISIT to short circuit traversal and to ensure that if a proc ili
 * has been rewritten, multiple uses of that proc ili are replaced with
 * the same ili (share_proc_ili is false during this process).  Note that
 * rewr_ili saves the index to the tree that's rewritten; rewr_cln_ili
 * retrieves the index to clean up the visit fields (several trees can be
 * rewritten before a clean-up occurs).
 *
 * WARNING:  there exists one slight flaw with using the visit field to save
 * the rewritten ili.  addili, when it sees a branch with constant operands,
 * may return a value of 0 (also ==> the ili is not visited). rewr_cln_ili
 * uses the visit flag to determine if it must traverse!!!  Solve this slight
 * problem by setting the visit flag of each root in rewr_cln_ili since this
 * only happens for a terminal (branch).
 *
 *     int        rewr_ili() - routine to setup rewriting (rewr_)
 *     int        rewr_ili_nme() - modified routine to change an ili and an nme
 *     static int rewr_()    - postorder rewrite routine
 *     static int rewr_nm()  - recursive rewrite routine for nmes
 *     static bool has_nme()  - check if nme has any ref to nme to replace
 *     void       rewr_cln_ili() - routine to setup clearing ILI_VISIT
 *     void       rewr_cln() - recursively clean up ILI_VISIT
 */

static int rewr_(int);
static int rewr_nm(int);
static bool has_nme(int);
static void rewr_cln(int);
static void rewr_cln_nm(int);
static int rewr_old, rewr_new, rewr_old_nme, rewr_new_nme;
static int rewr_use = 1, rewr_def = 1;
static int need_to_rename = 0;
static int rewrite_nme_indirect = 0;

/* save list of ILI being rewritten */
static struct {
  int *base;
  int size;
  int cnt;
} rewrb = {NULL, 0, 0};

/* save old/new ILI pairs for rewriting */
typedef struct rewrtstruct {
  int oldili, newili, oldnme, newnme;
} rewrtstruct;

static struct {
  rewrtstruct *base;
  int size, cnt, all_acon;
} rewrt = {NULL, 0, 0, 0};

/*
 * save an old/new ILI for subsequent rewrite
 */
void
rewr_these_ili(int oldili, int newili)
{
  if (rewrt.size == 0) {
    rewrt.size = 16;
    NEW(rewrt.base, rewrtstruct, rewrt.size);
    rewrt.cnt = 0;
  } else {
    NEED(rewrt.cnt + 2, rewrt.base, rewrtstruct, rewrt.size, rewrt.size + 16);
  }

  rewrt.base[rewrt.cnt].oldili = oldili;
  rewrt.base[rewrt.cnt].newili = newili;
  rewrt.base[rewrt.cnt].oldnme = 0;
  rewrt.base[rewrt.cnt].newnme = 0;
  ++rewrt.cnt;
} /* rewr_these_ili */

/*
 * save an old/new ILI/NME for subsequent rewrite
 */
void
rewr_these_ili_nme(int oldili, int newili, int oldnme, int newnme)
{
  if (rewrt.size == 0) {
    rewrt.size = 16;
    NEW(rewrt.base, rewrtstruct, rewrt.size);
    rewrt.cnt = 0;
  } else {
    NEED(rewrt.cnt + 2, rewrt.base, rewrtstruct, rewrt.size, rewrt.size + 16);
  }

  rewrt.base[rewrt.cnt].oldili = oldili;
  rewrt.base[rewrt.cnt].newili = newili;
  rewrt.base[rewrt.cnt].oldnme = oldnme;
  rewrt.base[rewrt.cnt].newnme = newnme;
  ++rewrt.cnt;
} /* rewr_these_ili_nme */

/*
 * get most recent 'newnme' for the 'oldnme' given
 */
int
get_rewr_new_nme(int nmex)
{
  int j;
  for (j = rewrt.cnt; j > 0; --j) {
    if (nmex == rewrt.base[j - 1].oldnme)
      return rewrt.base[j - 1].newnme;
  }
  return 0;
} /* get_rewr_new_nme */

/*
 * save/restore rewrite count
 */
int
save_rewr_count(void)
{
  return rewrt.cnt;
} /* save_rewr_count */

void
restore_rewr_count(int c)
{
  rewrt.cnt = c;
} /* restore_rewr_count */

/*
 * rewrites the ili tree 'tree', changing 'old' into 'New'
 * saves 'tree' in 'rewrb'
 * must be followed by a call to rewr_cln_ili
 */
int
rewr_ili(int tree, int old, int New)
{
  int save_proc, save_all_acon;

#if DEBUG
  assert(tree > 0, "rewr_ili, bad tree", 0, ERR_Severe);
#endif
  if (rewrb.size == 0) {
    rewrb.size = 16;
    NEW(rewrb.base, int, rewrb.size);
    rewrb.cnt = 0;
  } else
    NEED(rewrb.cnt + 1, rewrb.base, int, rewrb.size, rewrb.size + 16);

  rewrb.base[rewrb.cnt++] = tree;

  save_proc = share_proc_ili;
  share_proc_ili = false;
  rewr_old = old;
  rewr_new = New;
  need_to_rename = 0;
  rewrite_nme_indirect = 0;
  rewr_old_nme = 0;
  rewr_new_nme = 0;
  if (old >= 1 && ILI_OPC(old) == IL_LDA) {
    rewr_old_nme = ILI_OPND(old, 2);
    need_to_rename = 1;
  }
  save_all_acon = rewrt.all_acon;
  if (old == -2) {
    rewrt.all_acon = 1;
  } else {
    rewrt.all_acon = 0;
  }
  New = rewr_(tree);
  share_proc_ili = save_proc;
  rewrt.all_acon = save_all_acon;
  need_to_rename = 0;
  rewrite_nme_indirect = 0;
  rewr_old_nme = 0;
  rewr_new_nme = 0;

  return New;
}

/** Rewrites the ili tree 'tree', changing 'oldili' into 'newili' and
 * the nme 'oldnme' into 'newnme'; saves 'tree' in 'rewrb'.
 *
 * Must be followed by a call to rewr_cln_ili.
 */
int
rewr_ili_nme(int tree, int oldili, int newili, int oldnme, int newnme,
             int douse, int dodef)
{
  int save_proc, New;

#if DEBUG
  assert(tree > 0, "rewr_ili_nme, bad tree", 0, ERR_Severe);
#endif
  if (rewrb.size) {
    NEED(rewrb.cnt + 1, rewrb.base, int, rewrb.size, rewrb.size + 16);
  } else {
    rewrb.size = 16;
    NEW(rewrb.base, int, rewrb.size);
    rewrb.cnt = 0;
  }
  rewrb.base[rewrb.cnt++] = tree;

  save_proc = share_proc_ili;
  share_proc_ili = false;
  rewr_old = oldili;
  rewr_new = newili;
  rewr_old_nme = oldnme;
  rewr_new_nme = newnme;
  rewrite_nme_indirect = 0;
  rewr_use = douse;
  rewr_def = dodef;
  rewrt.all_acon = 0;
  New = rewr_(tree);
  share_proc_ili = save_proc;
  rewr_use = 1;
  rewr_def = 1;
  rewrite_nme_indirect = 0;
  return New;
} /* rewr_ili_nme */

static int
rewr_indirect_nme(int nmex)
{
  int new_nmex;
  if (NME_TYPE(nmex) == NT_IND && NME_NM(nmex) == NME_NM(rewr_old_nme)) {
    if (NME_SUB(nmex)) {
      new_nmex =
          add_arrnme(NT_ARR, NME_NULL, rewr_new_nme, 0, NME_SUB(nmex), 0);
    } else {
      new_nmex = rewr_new_nme;
    }
    return new_nmex;
  }
  switch (NME_TYPE(nmex)) {
  case NT_MEM:
    new_nmex = rewr_indirect_nme(NME_NM(nmex));
    if (new_nmex != NME_NM(nmex))
      nmex = addnme(NME_TYPE(nmex), NME_SYM(nmex), new_nmex, NME_CNST(nmex));
    break;
  case NT_IND:
    new_nmex = rewr_indirect_nme(NME_NM(nmex));
    if (new_nmex != NME_NM(nmex)) {
      nmex = add_arrnme(NT_IND, NME_SYM(nmex), new_nmex, NME_CNST(nmex),
                        NME_SUB(nmex), NME_INLARR(nmex));
    }
    break;
  case NT_ARR:
    new_nmex = rewr_indirect_nme(NME_NM(nmex));
    if (new_nmex != NME_NM(nmex)) {
      nmex = add_arrnme(NT_ARR, NME_SYM(nmex), new_nmex, NME_CNST(nmex),
                        NME_SUB(nmex), NME_INLARR(nmex));
    }
    break;
  case NT_VAR:
  case NT_UNK:
    break;
  case NT_SAFE:
    new_nmex = rewr_indirect_nme(NME_NM(nmex));
    if (new_nmex != NME_NM(nmex))
      nmex = addnme(NT_SAFE, SPTR_NULL, new_nmex, 0);
    break;
  default:
#if DEBUG
    interr("rewr_indirect_nme: unexpected nme", nmex, ERR_Severe);
#endif
    break;
  }

  return nmex;
} /* rewr_indirect_nme */

/*
 * Given the ILI by which we replace the address of a symbol,
 * and the offset to the base of that symbol, build a tree for the offset
 * applied to the ILI
 */
static int
build_pointer_tree(int ilix, ISZ_T offset)
{
  return ad3ili(IL_AADD, ilix, ad_aconi(offset), 0);
} /* build_pointer_tree */

static int
rewr_(int tree)
{
  ILI_OP opc;
  int noprs;
  int opnd;
  int i, j, dontdef, dontuse;
  ILI newili;
  bool changes;

#if DEBUG
  assert(tree > 0, "rewr_, bad tree", 0, ERR_Severe);
#endif
  if (ILI_VISIT(tree))
    return ILI_VISIT(tree);
  if (rewrt.cnt == 0) {
    if (tree == rewr_old) {
      ILI_VISIT(tree) = rewr_new;
      return rewr_new;
    }
  } else {
    /* look through the entire table of 'old' */
    for (j = rewrt.cnt; j > 0; --j) {
      if (tree == rewrt.base[j - 1].oldili) {
        ILI_VISIT(tree) = rewrt.base[j - 1].newili;
        return ILI_VISIT(tree);
      }
    }
  }
  opc = ILI_OPC(tree);
  noprs = IL_OPRS(opc);
  changes = false;
  /* first, look for matching NMEs */
  dontdef = dontuse = 0;
  if (rewrt.all_acon && rewrt.cnt && opc == IL_ACON) {
    /* see if we are supposed to rewrite an IL_ACON based on the same symbol */
    /* if we have a rewrite IL_ACON(ST_CONST(symbol,0)) => IL_LDA(foo), then
     * when we see IL_ACON(ST_CONST(symbol,4)), we want to create
     *  IL_LDA(foo)+4 */
    int acon, sptr;
    acon = ILI_OPND(tree, 1);
    sptr = CONVAL1G(acon);
    for (j = rewrt.cnt; j > 0; --j) {
      int oldtree = rewrt.base[j - 1].oldili;
      if (ILI_OPC(oldtree) == opc) {
        int oldacon;
        oldacon = ILI_OPND(oldtree, 1);
        if (CONVAL1G(oldacon) == sptr && ACONOFFG(oldacon) == 0) {
          int newtree =
              build_pointer_tree(rewrt.base[j - 1].newili, ACONOFFG(acon));
          ILI_VISIT(tree) = newtree;
          return newtree;
        }
      }
    }
  }
  if (!rewr_use || !rewr_def) {
    if (IL_TYPE(opc) == ILTY_STORE) {
      dontdef = !rewr_def;
    }
    if (IL_TYPE(opc) == ILTY_LOAD) {
      dontuse = !rewr_use;
    }
  }
  newili.alt = 0;
  for (i = 1; i <= noprs; i++) {
    opnd = ILI_OPND(tree, i);
    if (IL_ISLINK(opc, i)) {
      int newopnd;
      if (dontuse && opnd == rewr_old) {
        newopnd = opnd;
      } else if (dontdef && opnd == rewr_old) {
        newopnd = opnd;
      } else {
        newopnd = rewr_(opnd);
        if (newopnd != opnd && IL_RES(ILI_OPC(opnd)) == ILIA_AR) {
          /* The problem is Fortran Cray pointers, where integer-valued
           * expressions are stored into integer pointers, which are then
           * used as pointer values.  The integer-valued pointer is often
           * converted from an address-valued expression (ACON or the like)
           * and here we want the address-valued expression or to move it
           * to an address-valued expression */
          if (ILI_OPC(newopnd) == IL_AKMV || ILI_OPC(newopnd) == IL_AIMV) {
            newopnd = ILI_OPND(newopnd, 1);	/* get the address value */
          } else if (IL_RES(ILI_OPC(newopnd)) == ILIA_KR) {
            newopnd = ad1ili(IL_KAMV, newopnd);
          } else if (IL_RES(ILI_OPC(newopnd)) == ILIA_IR) {
            newopnd = ad1ili(IL_IAMV, newopnd);
          }
        }
      }
      newili.opnd[i - 1] = newopnd;
      if (newopnd != opnd)
        changes = true;
    } else if (IL_OPRFLAG(opc, i) == ILIO_NME) {
      if (rewrt.cnt) {
        newili.opnd[i - 1] = opnd;
        for (j = rewrt.cnt; j > 0; --j) {
          if (opnd == rewrt.base[j - 1].oldnme) {
            newili.opnd[i - 1] = rewrt.base[j - 1].newnme;
            break;
          }
        }
        if (j == 0)
          newili.opnd[i - 1] = rewr_nm(opnd);
      } else if (dontuse && opnd == rewr_old_nme) {
        newili.opnd[i - 1] = opnd;
      } else if (dontdef && opnd == rewr_old_nme) {
        newili.opnd[i - 1] = opnd;
      } else {
        newili.opnd[i - 1] = opnd;
        if (need_to_rename) {
          if ((IL_TYPE(opc) == ILTY_STORE || IL_TYPE(opc) == ILTY_LOAD) &&
              has_nme(opnd)) {
            int opr_i_1;

            opr_i_1 = newili.opnd[i - 2];
            rewr_new_nme = 0;
            switch (ILI_OPC(opr_i_1)) {
            case IL_ACON:
              rewr_new_nme = build_sym_nme(SymConval1(ILI_SymOPND(opr_i_1, 1)),
                                           ACONOFFG(ILI_OPND(opr_i_1, 1)),
                                           (opc == IL_LDA || opc == IL_STA));
              break;
            case IL_LDA:
              rewr_new_nme = ILI_OPND(opr_i_1, 2);
              if (rewr_new_nme)
                rewr_new_nme =
                    build_sym_nme(basesym_of(rewr_new_nme), 0, (opc == IL_LDA || opc == IL_STA));
              break;
            default:
              break;
            }
            if ((!opnd && rewr_new_nme) ||
                (DTY(dt_nme(opnd)) == DTY(dt_nme(rewr_new_nme))))
              newili.opnd[i - 1] = rewr_new_nme;
            rewr_new_nme = 0;
          }
        } else {
          if (opnd == rewr_old_nme) {
            opnd = rewr_new_nme;
          } else if (rewrite_nme_indirect) {
            opnd = rewr_indirect_nme(opnd);
          }
          newili.opnd[i - 1] = rewr_nm(opnd);
        }
      }
      if (newili.opnd[i - 1] != opnd)
        changes = true;
    } else {
      newili.opnd[i - 1] = opnd;
    }
  }
  /*
   * if a rewr_ili() is passed an old_tree of 1 (=> IL_NULL), any
   * proc ili are rewritten even if their operands did not change.
   * situation which could occur is when ili are duplicated, incorrect
   * cse'ing could occur due to the sharing of proc ili.
   */
  if (changes || (IL_TYPE(opc) == ILTY_PROC && rewr_old == 1)) {
    int index;
    int newalt;
    newili.opc = opc;
    index = addili(&newili);
    /* if we're selectively changing loads/stores, don't set ILI_VISIT for
     * load/store */
    ILI_VISIT(tree) = index;
    /* addarth will add alt field  */
    if (ILI_ALT(tree) && !ILI_ALT(index)) {
      index = new_ili(&newili);
      newalt = rewr_(ILI_ALT(tree));
      if (newalt != index)
        ILI_ALT(index) = newalt;
      ILI_VISIT(tree) = index;
    }

    if ((IL_TYPE(opc) == ILTY_PROC || IL_TYPE(opc) == ILTY_DEFINE) &&
        ILI_ALT(tree) && !ILI_ALT(index)) {
      /* the old tree had an ALT field, the new one doesn't,
       * rewrite the old one and fill in the ILI_ALT field of the
       * new one */
      newalt = rewr_(ILI_ALT(tree));
      ILI_ALT(index) = newalt;
      if (ILI_ALT(index) == index)
        ILI_ALT(index) = 0;
    } else {
      switch (opc) {
      case IL_APURE:  /* pure function, no arguments, returning AR */
      case IL_IPURE:  /* pure function, no arguments, returning IR */
      case IL_APUREA: /* pure function, one AR argument, returning AR */
      case IL_APUREI: /* pure function, one IR argument, returning AR */
      case IL_IPUREA: /* pure function, one AR argument, returning IR */
      case IL_IPUREI: /* pure function, one IR argument, returning IR */
        newalt = rewr_(ILI_ALT(tree));
        ILI_ALT(index) = newalt;
        break;
      default:
        break;
      }
    }
  } else {
    ILI_VISIT(tree) = tree;
  }

  return ILI_VISIT(tree);
}

static bool
has_nme(int nme)
{
  if (nme == 0)
#if OPT_ZNME
    return true;
#else
    return false;
#endif
  if (nme == rewr_old_nme)
    return true;
  switch (NME_TYPE(nme)) {
  default:
    break;
  case NT_MEM:
  case NT_IND:
  case NT_ARR:
    return has_nme(NME_NM(nme));
  case NT_VAR:
  case NT_UNK:
  case NT_SAFE:
    break;
  }
  return false;
}

static int
rewr_nm(int nme)
{
  int new_nm, new_sub, j;
#if DEBUG
  if (nme < 0 || nme >= nmeb.stg_size) {
    interr("rewr_nm:bad names ptr", nme, ERR_Severe);
    return nme;
  }
#endif
  for (j = rewrt.cnt; j > 0; --j) {
    if (nme == rewrt.base[j - 1].oldnme) {
      return rewrt.base[j - 1].newnme;
    }
  }

  switch (NME_TYPE(nme)) {
  case NT_MEM:
    new_nm = rewr_nm((int)NME_NM(nme));
    if (new_nm != NME_NM(nme))
      return addnme(NME_TYPE(nme), NME_SYM(nme), new_nm, NME_CNST(nme));
    break;
  case NT_IND:
    new_nm = rewr_nm((int)NME_NM(nme));
    if (NME_SUB(nme))
      new_sub = rewr_((int)NME_SUB(nme));
    else
      new_sub = 0;
    if (new_nm != NME_NM(nme) || new_sub != NME_SUB(nme)) {
      nme = add_arrnme(NT_IND, NME_SYM(nme), new_nm, NME_CNST(nme), new_sub,
                       (int)NME_INLARR(nme));
    }
    break;
  case NT_ARR:
    new_nm = rewr_nm((int)NME_NM(nme));
    if (NME_SUB(nme))
      new_sub = rewr_((int)NME_SUB(nme));
    else
      new_sub = 0;
    if (new_nm != NME_NM(nme) || new_sub != NME_SUB(nme)) {
      const ILI_OP opc = ILI_OPC(new_sub);
      if (new_sub && IL_TYPE(opc) == ILTY_CONS) {
        ISZ_T off;
        if (IL_RES(opc) != ILIA_KR)
          off = CONVAL2G(ILI_OPND(new_sub, 1));
        else
          off = ACONOFFG(ILI_OPND(new_sub, 1));
        nme = add_arrnme(NT_ARR, SPTR_NULL, new_nm, off, new_sub,
                         (int)NME_INLARR(nme));
      } else if (new_sub == 0 && NME_CNST(nme)) {
        nme = add_arrnme(NT_ARR, NME_SYM(nme), new_nm, NME_CNST(nme), new_sub,
                         NME_INLARR(nme));
      } else {
        nme = add_arrnme(NT_ARR, NME_SYM(nme), new_nm, 0, new_sub,
                         NME_INLARR(nme));
      }
      return nme;
    }
    break;
  case NT_VAR:
  case NT_UNK:
    break;
  case NT_SAFE:
    new_nm = rewr_nm((int)NME_NM(nme));
    if (new_nm != NME_NM(nme))
      return addnme(NT_SAFE, SPTR_NULL, new_nm, 0);
    break;
  default:
#if DEBUG
    interr("rewr_nm:unexp. nme", nme, ERR_Severe);
#endif
    break;
  }

  return nme;
}

/*
 * reset the ILI_VISIT field that was set by calls to rewr_ili
 */
void
rewr_cln_ili(void)
{
  int i;

#if DEBUG
  assert(rewrb.cnt, "rewr_cln_ili: cnt is zero", 0, ERR_Severe);
#endif

  for (i = 0; i < rewrb.cnt; i++) {
    ILI_VISIT(rewrb.base[i]) = 1; /* solve '0' as the new ili problem */
    rewr_cln(rewrb.base[i]);
  }

  FREE(rewrb.base);
  rewrb.base = NULL;
  rewrb.size = rewrb.cnt = 0;

#if defined(MY_SCN) || defined(DEW)
  for (i = 1; i < ilib.stg_avail; i++)
    if (ILI_VISIT(i))
      interr("rewr_cln_ili: visit not zero", i, ERR_Severe);
#endif
  if (rewrt.base) {
    FREE(rewrt.base);
    rewrt.base = NULL;
    rewrt.size = rewrt.cnt = 0;
  }
}

static void
rewr_cln(int tree)
{
  ILI_OP opc;
  int noprs;
  int i;

#if DEBUG
  assert(tree > 0, "rewr_cln, bad tree", 0, ERR_Severe);
#endif
  if (ILI_VISIT(tree)) {
    ILI_VISIT(tree) = 0;
    opc = ILI_OPC(tree);
    noprs = IL_OPRS(opc);
    for (i = 1; i <= noprs; i++) {
      if (IL_ISLINK(opc, i))
        rewr_cln((int)ILI_OPND(tree, i));
      else if (IL_OPRFLAG(opc, i) == ILIO_NME)
        rewr_cln_nm((int)ILI_OPND(tree, i));
    }
    if (ILI_ALT(tree)) {
      rewr_cln(ILI_ALT(tree));
    }
  }
}

static void
rewr_cln_nm(int nme)
{
#if DEBUG
  if (nme < 0 || nme >= nmeb.stg_size) {
    interr("rewr_cln_nm:bad names ptr", nme, ERR_Severe);
    return;
  }
#endif

  switch (NME_TYPE(nme)) {
  case NT_MEM:
  case NT_SAFE:
    rewr_cln_nm((int)NME_NM(nme));
    break;
  case NT_IND:
  case NT_ARR:
    rewr_cln_nm((int)NME_NM(nme));
    if (NME_SUB(nme))
      rewr_cln((int)NME_SUB(nme));
    break;
  case NT_VAR:
  case NT_UNK:
    break;
  default:
#if DEBUG
    interr("rewr_cln_nm:unexp. nme", nme, ERR_Severe);
#endif
    break;
  }
}

static void
prsym(int sym, FILE *ff)
{
  char *p;
  p = getprint((int)sym);
  fprintf(ff, "%s", p);
  if (strncmp("..inline", p, 8) == 0)
    fprintf(ff, "%d", sym);
}

static ILI_OP
opc_comp_zero_one(int cmp_ili, int *invert_cc, int *label_op)
{
  int icon_ili;
  ILI_OP opc;
  int cc;

  *invert_cc = 0;
  opc = ILI_OPC(cmp_ili);
  switch (opc) {
  case IL_ICMP:
  case IL_ICJMP:
    cc = ILI_OPND(cmp_ili, 3);
    if (cc != CC_EQ && cc != CC_NE)
      return IL_NONE;
    icon_ili = ILI_OPND(cmp_ili, 2);
    if (ILI_OPC(icon_ili) != IL_ICON)
      return IL_NONE;
    if (ILI_OPND(icon_ili, 1) != stb.i1)
      return IL_NONE;
    *invert_cc = (cc == CC_NE);
    if (opc == IL_ICJMP)
      *label_op = ILI_OPND(cmp_ili, 4);
    break;
  case IL_ICMPZ:
  case IL_ICJMPZ:
    cc = ILI_OPND(cmp_ili, 2);
    if (cc != CC_EQ && cc != CC_NE)
      return IL_NONE;
    *invert_cc = (cc == CC_EQ);
    if (opc == IL_ICJMPZ)
      *label_op = ILI_OPND(cmp_ili, 3);
    break;
  default:
    return IL_NONE;
  }
  return opc;
}

/**
 * \brief returns simplified version of comparison ili
 * \param cmp_ili  Comparison ILI
 *
 * <pre>
 * simplify following eight cases:
 * (1)
 *          X = IL_xxCMP A, B, cmp_op
 *          Y = IL_ICMP X, 1, eq
 *      =>
 *          Y = IL_xxCMP A, B, cmp_op
 * (2)
 *          X = IL_xxCMPZ A, cmp_op
 *          Y = IL_ICMP X, 1, eq
 *      =>
 *          Y = IL_xxCMPZ A, cmp_op
 * (3)
 *          X = IL_xxCMP A, B, cmp_op
 *          Y = IL_ICMP X, 1, ne
 *      =>
 *          Y = IL_xxCMP A, B, ~cmp_op
 * (4)
 *          X = IL_xxCMPZ A, cmp_op
 *          Y = IL_ICMP X, 1, ne
 *      =>
 *          Y = IL_xxCMPZ A, ~cmp_op
 * (5)
 *          X = IL_xxCMP A, B, cmp_op
 *          Y = IL_ICMPZ X, eq
 *      =>
 *          Y = IL_xxCMP A, B, ~cmp_op
 * (6)
 *          X = IL_xxCMPZ A, cmp_op
 *          Y = IL_ICMPZ X, eq
 *      =>
 *          Y = IL_xxCMPZ A, ~cmp_op
 * (7)
 *          X = IL_xxCMP A, B, cmp_op
 *          Y = IL_ICMPZ X, ne
 *      =>
 *          Y = IL_xxCMP A, B, cmp_op
 * (8)
 *          X = IL_xxCMPZ A, cmp_op
 *          Y = IL_ICMPZ X, ne
 *      =>
 *          Y = IL_xxCMPZ A, cmp_op
 * </pre>
 * This algorithm is applied recursively to simplify chains of comparison to one
 * and zero
 */
int
simplified_cmp_ili(int cmp_ili)
{
  ILI_OP opc, new_opc, jump_opc;
  int new_ili;
  CC_RELATION new_cc;
  int invert_cc;
  int label_op;

  while ((
      opc = opc_comp_zero_one(cmp_ili, &invert_cc, &label_op))) {
    new_ili = ILI_OPND(cmp_ili, 1);
    new_opc = ILI_OPC(new_ili);
    switch (new_opc) {
    case IL_ACMP:
      jump_opc = IL_ACJMP;
      goto shared_bin_cmp;
    case IL_FCMP:
      jump_opc = IL_FCJMP;
      goto shared_bin_cmp;
    case IL_DCMP:
      jump_opc = IL_DCJMP;
      goto shared_bin_cmp;
#ifdef TARGET_SUPPORTS_QUADFP
    case IL_QCMP:
      jump_opc = IL_QCJMP;
      goto shared_bin_cmp;
#endif
    case IL_ICMP:
      jump_opc = IL_ICJMP;
      goto shared_bin_cmp;
    case IL_KCMP:
      jump_opc = IL_KCJMP;
      goto shared_bin_cmp;
    case IL_UICMP:
      jump_opc = IL_UICJMP;
      goto shared_bin_cmp;
    case IL_UKCMP:
      jump_opc = IL_UKCJMP;
    shared_bin_cmp:
      new_cc = CC_ILI_OPND(new_ili, 3);
      if (invert_cc) {
        if (is_floating_comparison_opcode(new_opc))
          new_cc = complement_ieee_cc(new_cc);
        else
          new_cc = complement_int_cc(new_cc);
      }
      if (opc == IL_ICJMP || opc == IL_ICJMPZ)
        cmp_ili = ad4ili(jump_opc, (int)ILI_OPND(new_ili, 1),
                         (int)ILI_OPND(new_ili, 2), (int)new_cc, (int)label_op);
      else
        cmp_ili = ad3ili(new_opc, (int)ILI_OPND(new_ili, 1),
                         (int)ILI_OPND(new_ili, 2), (int)new_cc);
      break;
    case IL_ACMPZ:
      jump_opc = IL_ACJMPZ;
      goto shared_una_cmp;
    case IL_FCMPZ:
      jump_opc = IL_FCJMPZ;
      goto shared_una_cmp;
    case IL_DCMPZ:
      jump_opc = IL_DCJMPZ;
      goto shared_una_cmp;
#ifdef TARGET_SUPPORTS_QUADFP
    case IL_QCMPZ:
      jump_opc = IL_QCJMPZ;
      goto shared_una_cmp;
#endif
    case IL_ICMPZ:
      jump_opc = IL_ICJMPZ;
      goto shared_una_cmp;
    case IL_KCMPZ:
      jump_opc = IL_KCJMPZ;
      goto shared_una_cmp;
    case IL_UICMPZ:
      jump_opc = IL_UICJMPZ;
      goto shared_una_cmp;
    case IL_UKCMPZ:
      jump_opc = IL_UKCJMPZ;
    shared_una_cmp:
      new_cc = CC_ILI_OPND(new_ili, 2);
      if (invert_cc) {
        if (is_floating_comparison_opcode(new_opc))
          new_cc = complement_ieee_cc(new_cc);
        else
          new_cc = complement_int_cc(new_cc);
      }
      if (opc == IL_ICJMP || opc == IL_ICJMPZ)
        cmp_ili = ad3ili(jump_opc, (int)ILI_OPND(new_ili, 1), (int)new_cc,
                         (int)label_op);
      else
        cmp_ili = ad2ili(new_opc, (int)ILI_OPND(new_ili, 1), (int)new_cc);
      break;
    default:
      return cmp_ili;
    }
  }
  return cmp_ili;
}

const char *
dump_msz(MSZ ms)
{
  const char *msz;
  switch (ms) {
  case MSZ_SBYTE:
    msz = "sb";
    break;
  case MSZ_SHWORD:
    msz = "sh";
    break;
  case MSZ_WORD:
    msz = "wd";
    break;
  case MSZ_SLWORD:
    msz = "sl";
    break;
  case MSZ_BYTE:
    msz = "bt";
    break;
  case MSZ_UHWORD:
    msz = "uh";
    break;
  case MSZ_PTR:
    msz = "pt";
    break;
  case MSZ_ULWORD:
    msz = "ul";
    break;
  case MSZ_DBLE:
    msz = "db";
    break;
#ifdef MSZ_I8
  case MSZ_I8:
    msz = "i8";
    break;
#endif
  case MSZ_UWORD:
    msz = "uw";
    break;
#ifdef MSZ_F10
  case MSZ_F10:
    msz = "ep";
    break;
#endif /* MSZ_F10 */
  default:
    interr("Bad msz to LD/ST", ms, ERR_Severe);
    msz = "??";
  }
  return msz;
}

/** Return print name of an ATOMIC_ORIGIN value */
static const char *
atomic_origin_name(ATOMIC_ORIGIN origin)
{
  switch (origin) {
  case AORG_CPLUS:
    return "C";
  case AORG_OPENMP:
    return "openmp";
  case AORG_OPENACC:
    return "openacc";
  }
}

static const char *
atomic_rmw_op_name(ATOMIC_RMW_OP op)
{
  switch (op) {
  case AOP_XCHG:
    return "xchg";
  case AOP_ADD:
    return "add";
  case AOP_SUB:
    return "sub";
  case AOP_AND:
    return "and";
  case AOP_OR:
    return "or";
  case AOP_XOR:
    return "xor";
  default:
    return "<bad ATOMIC_RMW_OP>";
  }
}

void
dump_atomic_info(FILE *f, ATOMIC_INFO info)
{
  if (f == NULL)
    f = stderr;
  fprintf(f, "{");
  fprintf(f, "%s", dump_msz(info.msz));
  if (info.scope == SS_SINGLETHREAD)
    fprintf(f, " singlethread");
  fprintf(f, " %s", atomic_origin_name(info.origin));
  fprintf(f, "}");
}

void
dump_ili(FILE *f, int i)
{
  int j, noprs;
  ILI_OP opc;
  static const char *cond[] = {"eq",    "ne",    "lt",    "ge",    "le",    "gt",
                               "noteq", "notne", "notlt", "notge", "notle", "notgt"};
  static const char *msz;

  if (f == NULL)
    f = stderr;
  opc = ILI_OPC(i);
  if (opc == GARB_COLLECTED) {
    fprintf(f, "%-4u **DELETED**\n", i);
    return;
  }
  assert(opc > 0 && opc < N_ILI, "dump_ili: bad opc", i, ERR_Severe);
  noprs = ilis[opc].oprs;
  fprintf(f, "%-4u %-9s  ", i, ilis[opc].name);
  for (j = 1; j <= noprs; j++) {
    int opn = ILI_OPND(i, j);
    switch (IL_OPRFLAG(opc, j)) {
    default:
      break;
    case ILIO_SYM:
    case ILIO_OFF:
      if (opn <= 0) {
        /* Special values permitted in a few cases. */
        bool okay;
        switch (opc) {
        case IL_ACCCOPYIN:
        case IL_ACCCOPY:
        case IL_ACCCOPYOUT:
        case IL_ACCCREATE:
        case IL_ACCDELETE:
        case IL_ACCPDELETE:
        case IL_ACCPCREATE:
        case IL_ACCPCOPY:
        case IL_ACCPCOPYIN:
        case IL_ACCPCOPYOUT:
        case IL_ACCPRESENT:
        case IL_ACCUSEDEVICE:
        case IL_ACCUSEDEVICEIFP:
          okay = true;
          break;
        case IL_GJSR:
        case IL_GJSRA:
          /* last operand is symbol that can be -1, 0, or label */
          okay = j == noprs && opn >= -1;
          break;
        case IL_ACCDEVICEPTR:
          /* last operand sym is unused and can be 0 */
          okay = j == noprs && opn == 0;
          break;
        default:
          okay = false;
        }
        if (!okay)
          interr("dump_ili:bad symbol table ptr", i, ERR_Fatal);
      }
      fprintf(f, " %5u~", opn);
      if (opn > 0) {
        if (opc != IL_ACON) {
          fprintf(f, "<");
          prsym(opn, f);
          fprintf(f, ">");
        } else {
          if (CONVAL1G(opn)) {
            fprintf(f, "<");
            prsym(CONVAL1G(opn), f);
          } else
            fprintf(f, "<%d", CONVAL1G(opn));
          fprintf(f, ",%" ISZ_PF "d>", ACONOFFG(opn));
        }
      }
      break;
    case ILIO_NME:
      dumpname(opn);
      break;
    case ILIO_STC:
      switch (opc) {
      case IL_ICMP:
      case IL_FCMP:
      case IL_DCMP:
      case IL_ACMP:
      case IL_ICMPZ:
      case IL_FCMPZ:
      case IL_DCMPZ:
      case IL_ACMPZ:
      case IL_ICJMP:
      case IL_FCJMP:
      case IL_DCJMP:
#ifdef TARGET_SUPPORTS_QUADFP
      case IL_QCJMP:
#endif
      case IL_ACJMP:
      case IL_ICJMPZ:
      case IL_FCJMPZ:
      case IL_DCJMPZ:
#ifdef TARGET_SUPPORTS_QUADFP
      case IL_QCJMPZ:
#endif
      case IL_ACJMPZ:
#ifdef TM_LPCMP
      case IL_ICLOOP:
      case IL_FCLOOP:
      case IL_DCLOOP:
      case IL_ACLOOP:
      case IL_ICLOOPZ:
      case IL_FCLOOPZ:
      case IL_DCLOOPZ:
      case IL_ACLOOPZ:
#endif
      case IL_UICMP:
      case IL_UICMPZ:
      case IL_UICJMP:
      case IL_UICJMPZ:
      case IL_KCJMP:
      case IL_KCJMPZ:
      case IL_KCMP:
      case IL_KCMPZ:
      case IL_UKCJMP:
      case IL_UKCJMPZ:
      case IL_UKCMP:
      case IL_UKCMPZ:
      case IL_LCJMPZ:
#ifdef IL_X87CMP
      case IL_X87CMP:
#endif
#ifdef IL_DOUBLEDOUBLECMP
      case IL_DOUBLEDOUBLECMP:
#endif
        fprintf(f, " %5s ", cond[opn - 1]);
        break;
      case IL_LD:
      case IL_ST:
      case IL_LDKR:
      case IL_STKR:
        msz = dump_msz(ConvertMSZ(opn));
        fprintf(f, " %5s ", msz);
        break;
      case IL_ATOMICRMWI:
      case IL_ATOMICRMWA:
      case IL_ATOMICRMWKR:
      case IL_ATOMICRMWSP:
      case IL_ATOMICRMWDP:
        if (j == 4)
          dump_atomic_info(f, atomic_info(i));
        else if (j == 5)
          fprintf(f, " %s", atomic_rmw_op_name(ConvertATOMIC_RMW_OP(opn)));
        else
          fprintf(f, " %d", (int)((short)opn));
        break;
      default:
        fprintf(f, " %6d", (int)((short)opn));
      }
      break;
    case ILIO_LNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      switch (ILI_OPC(opn)) {
      case IL_KERNELNEST:
        break;
      case IL_KERNELBLOCK:
        break;
      case IL_KERNELGRID:
        break;
      case IL_KERNELSTREAM:
        break;
      case IL_KERNELDEVICE:
        break;
      default:
        assert(IL_RES(ILI_OPC(opn)) != ILIA_TRM, "dump_ili: any link exp", i,
               ERR_Severe);
      }
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_IRLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_IR, "dump_ili: ir link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_KRLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_KR, "dump_ili: kr link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
#ifdef ILIO_PPLNK
    case ILIO_PPLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_PR, "dump_ili: pr link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
#endif
    case ILIO_ARLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_AR, "dump_ili: ar link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_SPLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_SP, "dump_ili: sp link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_DPLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
#ifdef IL_DASPSP
      if (opc != IL_DASPSP || IL_RES(ILI_OPC(opn)) != ILIA_CS) {
#endif
        assert(IL_RES(ILI_OPC(opn)) == ILIA_DP, "dump_ili: dp link exp", i,
               ERR_Severe);
#ifdef IL_DASPSP
      }
#endif
      fprintf(f, " %5u^", opn);
      break;
#ifdef ILIO_CSLNK
    case ILIO_QPLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_QP, "dump_ili: qp link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_CSLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_CS, "dump_ili: cs link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_CDLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_CD, "dump_ili: cd link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_CQLNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_CQ, "dump_ili: cq link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_128LNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_128, "dump_ili: 128 link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_256LNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_256, "dump_ili: 256 link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
    case ILIO_512LNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_512, "dump_ili: 512 link exp", i,
             ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
#ifdef LONG_DOUBLE_FLOAT128
    case ILIO_FLOAT128LNK:
      assert(opn > 0 && opn < ilib.stg_size, "dump_ili: bad ili lnk", i,
             ERR_Severe);
      assert(IL_RES(ILI_OPC(opn)) == ILIA_FLOAT128,
             "dump_ili: doubledouble link exp", i, ERR_Severe);
      fprintf(f, " %5u^", opn);
      break;
#endif
#endif

    case ILIO_IR:
      fprintf(f, " ir(%2d)", opn);
      break;
    case ILIO_KR:
#if defined(TARGET_X8664)
      fprintf(f, " kr(%2d)", opn);
#else
      fprintf(f, " kr(%d,%d)", KR_MSH(opn), KR_LSH(opn));
#endif
      break;
    case ILIO_AR:
      fprintf(f, " ar(%2d)", opn);
      break;
    case ILIO_SP:
      fprintf(f, " sp(%2d)", opn);
      break;
    case ILIO_DP:
      fprintf(f, " dp(%2d)", opn);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    /* just for debug to dump ili */
    case ILIO_QP:
      fprintf(f, " qp(%2d)", opn);
      break;
#endif
    case ILIO_CS:
      fprintf(f, " cs(%2d)", opn);
      break;
    case ILIO_CD:
      fprintf(f, " cd(%2d)", opn);
      break;
    case ILIO_XMM:
      fprintf(f, " xmm%d", opn);
      break;
    }
  }
  if (ILI_ALT(i))
    fprintf(f, " %5u^-alt", ILI_ALT(i));
  fprintf(f, "\n");
}

/* ****************************************************************** */
static void
dilitree(int i)
{
  int k, j, opn, noprs;
  ILI_OP opc;
  static int indent = 0;

  indent += 3;
  opc = ILI_OPC(i);
  noprs = ilis[opc].oprs;

  for (j = 1; j <= noprs; j++) {
    opn = ILI_OPND(i, j);
    switch (IL_OPRFLAG(opc, j)) {
    case ILIO_LNK:
      if (indent > 3) {
        indent -= 3;
        dilitree(opn);
        indent += 3;
        break;
      }
      FLANG_FALLTHROUGH;
    case ILIO_IRLNK:
    case ILIO_KRLNK:
    case ILIO_ARLNK:
    case ILIO_SPLNK:
    case ILIO_DPLNK:
#ifdef ILIO_CSLNK
    case ILIO_QPLNK:
    case ILIO_CSLNK:
    case ILIO_CDLNK:
    case ILIO_CQLNK:
    case ILIO_128LNK:
    case ILIO_256LNK:
    case ILIO_512LNK:
#ifdef LONG_DOUBLE_FLOAT128
    case ILIO_FLOAT128LNK:
#endif
#endif
#ifdef ILIO_PPLNK
    case ILIO_PPLNK:
#endif
      assert(opn > 0 && opn < ilib.stg_size, "dilitree: bad ili lnk", i,
             ERR_Severe);
      dilitree(opn);
      break;
    default:
      break;
    }
  }
  for (k = 1; k < indent; k++)
    fprintf(gbl.dbgfil, " ");

#if DEBUG
  dump_ili(gbl.dbgfil, i);
#endif

  indent -= 3;
}

/**
 * \brief ILI dump routine
 */
void
dmpili(void)
{
  int i, j, tmp, opn;

  if (gbl.dbgfil == NULL)
    gbl.dbgfil = stderr;

  fprintf(gbl.dbgfil, "\n\n***** ILI Area Dump *****\n\n");
  for (i = 1; i < ilib.stg_avail; i++) {
    dump_ili(gbl.dbgfil, i);
  }
  if (DBGBIT(10, 1))
    for (i = 0; i < ILTABSZ; i++) {
      fprintf(gbl.dbgfil, "\n\n***** ILI Hash Table%2d *****\n", i);
      for (j = 0; j < ILHSHSZ; j++)
        if ((opn = ilhsh[i][j]) != 0) {
          tmp = 0;
          fprintf(gbl.dbgfil, "%3d.", j);
          for (; opn != 0; opn = ILI_HSHLNK(opn)) {
            fprintf(gbl.dbgfil, " %5u^", opn);
            if ((++tmp) == 6) {
              tmp = 0;
              fprintf(gbl.dbgfil, "\n    ");
            }
          }
          if (tmp != 0)
            fprintf(gbl.dbgfil, "\n");
        }
    }
}

#if DEBUG
#define OT_UNARY 1
#define OT_BINARY 2
#define OT_LEAF 3

static void
prnme(int opn, int ili)
{

  switch (NME_TYPE(opn)) {
  default:
    break;
  case NT_VAR:
    prsym(NME_SYM(opn), gbl.dbgfil);
    break;
  case NT_MEM:
    prnme((int)NME_NM(opn), ili);
    if (NME_SYM(opn) == 0) {
      fprintf(gbl.dbgfil, ".real");
      break;
    }
    if (NME_SYM(opn) == 1) {
      fprintf(gbl.dbgfil, ".imag");
      break;
    }
    fprintf(gbl.dbgfil, "->%s", getprint((int)NME_SYM(opn)));
    break;
  case NT_IND:
    fprintf(gbl.dbgfil, "*(");
    prnme((int)NME_NM(opn), ili);
    if (NME_SYM(opn) == 0)
      if (NME_CNST(opn))
        fprintf(gbl.dbgfil, "%+" ISZ_PF "d)", NME_CNST(opn));
      else
        fprintf(gbl.dbgfil, ")");
    else {
      fprintf(gbl.dbgfil, "<");
      prilitree(ili);
      fprintf(gbl.dbgfil, ">");
    }
    if (NME_SUB(opn)) {
      fprintf(gbl.dbgfil, "[");
      prilitree(NME_SUB(opn));
      fprintf(gbl.dbgfil, "]");
    }
    break;
  case NT_ARR:
    prnme((int)NME_NM(opn), ili);
    fprintf(gbl.dbgfil, "[");
    if (NME_SYM(opn) == 0)
      fprintf(gbl.dbgfil, "%" ISZ_PF "d]", NME_CNST(opn));
    else if (NME_SUB(opn)) {
      prilitree(NME_SUB(opn));
      fprintf(gbl.dbgfil, "]");
    } else
      fprintf(gbl.dbgfil, "i]");
    break;
  case NT_SAFE:
    fprintf(gbl.dbgfil, "safe(");
    prnme((int)NME_NM(opn), ili);
    fprintf(gbl.dbgfil, ")");
    break;
  case NT_UNK:
    if (NME_SYM(opn))
      fprintf(gbl.dbgfil, "?vol");
    else
      fprintf(gbl.dbgfil, "?");
    break;
  }
}

static void
prcon(int sptr)
{
  char *p;
  for (p = getprint(sptr); *p == ' '; p++)
    ;
  fprintf(gbl.dbgfil, "%s", p);
}

static int
optype(ILI_OP opc)
{
  switch (opc) {
  case IL_INEG:
  case IL_UINEG:
  case IL_KNEG:
  case IL_UKNEG:
  case IL_SCMPLXNEG:
  case IL_DCMPLXNEG:
  case IL_FNEG:
  case IL_DNEG:
    return OT_UNARY;
  case IL_LD:
  case IL_LDKR:
  case IL_LDSP:
  case IL_LDDP:
  case IL_LDSCMPLX:
  case IL_LDDCMPLX:
  case IL_LDA:
  case IL_ICON:
  case IL_KCON:
  case IL_DCON:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCON:
#endif
  case IL_FCON:
  case IL_ACON:
    return OT_LEAF;
  default:
    return OT_BINARY;
  }
}

void
prilitree(int i)
{
  int k, j, noprs, o;
  ILI_OP opc;
  int n;
  const char *opval;
  static const char *ccval[] = {
      "??",    "==",    "!=",    "<",     ">=",    "<=",   ">",
      "noteq", "notne", "notlt", "notge", "notle", "notgt"};

  opc = ILI_OPC(i);
  noprs = ilis[opc].oprs;
  switch (opc) {
  case IL_IADD:
  case IL_KADD:
  case IL_UKADD:
  case IL_FADD:
  case IL_DADD:
  case IL_SCMPLXADD:
  case IL_DCMPLXADD:
  case IL_UIADD:
  case IL_AADD:
    opval = "+";
    goto binop;
  case IL_DSUB:
    opval = "-";
    goto binop;
  case IL_ISUB:
  case IL_KSUB:
  case IL_UKSUB:
  case IL_FSUB:
  case IL_SCMPLXSUB:
  case IL_DCMPLXSUB:
  case IL_UISUB:
  case IL_ASUB:
    opval = "-";
    goto binop;
  case IL_IMUL:
  case IL_KMUL:
  case IL_UKMUL:
  case IL_FMUL:
  case IL_DMUL:
  case IL_UIMUL:
  case IL_SCMPLXMUL:
  case IL_DCMPLXMUL:
    opval = "*";
    goto binop;
  case IL_SCMPLXDIV:
  case IL_DCMPLXDIV:
  case IL_DDIV:
  case IL_KDIV:
  case IL_FDIV:
  case IL_IDIV:
    opval = "/";
    goto binop;
  case IL_UIDIV:
    opval = "/_u";
    goto binop;
  case IL_UKDIV:
    opval = "/_u";
    goto binop;
  case IL_IDIVZ:
    opval = "/";
    goto binop;
  case IL_UIDIVZ:
    opval = "/_u";
    goto binop;
  case IL_KDIVZ:
    opval = "/";
    goto binop;
  case IL_UKDIVZ:
    opval = "/_u";
    goto binop;
  case IL_KAND:
  case IL_AND:
    opval = "&";
    goto binop;
  case IL_KOR:
  case IL_OR:
    opval = "|";
    goto binop;
  case IL_KXOR:
  case IL_XOR:
    opval = "^";
    goto binop;
  case IL_KMOD:
  case IL_MOD:
    opval = "%";
    goto binop;
  case IL_UIMOD:
    opval = "%_u";
    goto binop;
  case IL_MODZ:
    opval = "%";
    goto binop;
  case IL_UIMODZ:
    opval = "%_u";
    goto binop;
  case IL_KMODZ:
    opval = "%";
    goto binop;
  case IL_KUMODZ:
    opval = "%_u";
    goto binop;
  case IL_FMOD:
    n = 2;
    opval = "amod";
    goto intrinsic;
  case IL_DMOD:
    n = 2;
    opval = "dmod";
    goto intrinsic;
  case IL_LSHIFT:
  case IL_ULSHIFT:
  case IL_KLSHIFT:
    opval = "<<";
    goto binop;
  case IL_RSHIFT:
  case IL_URSHIFT:
  case IL_KURSHIFT:
    opval = ">>";
    goto binop;
  case IL_ARSHIFT:
  case IL_KARSHIFT:
    opval = "a>>";
    goto binop;

  case IL_IPOWI:
#ifdef IL_KPOWI
  case IL_KPOWI:
#endif
#ifdef IL_KPOWK
  case IL_KPOWK:
#endif
  case IL_DPOWK:
  case IL_DPOWD:
  case IL_DPOWI:
  case IL_FPOWF:
  case IL_FPOWI:
  case IL_FPOWK:
  case IL_SCMPLXPOW:
  case IL_DCMPLXPOW:
  case IL_SCMPLXPOWI:
  case IL_DCMPLXPOWI:
  case IL_SCMPLXPOWK:
  case IL_DCMPLXPOWK:
    opval = "**";
    goto binop;

  case IL_KCMP:
  case IL_UKCMP:
  case IL_DCMPLXCMP:
  case IL_SCMPLXCMP:
  case IL_ICMP:
  case IL_FCMP:
  case IL_DCMP:
  case IL_ACMP:
  case IL_UICMP:
#ifdef IL_X87CMP
  case IL_X87CMP:
#endif
#ifdef IL_DOUBLEDOUBLECMP
  case IL_DOUBLEDOUBLECMP:
#endif
    opval = ccval[ILI_OPND(i, 3)];
  binop:
    if ((o = optype(ILI_OPC(ILI_OPND(i, 1)))) != OT_UNARY && o != OT_LEAF) {
      fputc('(', gbl.dbgfil);
      prilitree(ILI_OPND(i, 1));
      fputc(')', gbl.dbgfil);
    } else
      prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, "%s", opval);
    if ((o = optype(ILI_OPC(ILI_OPND(i, 2)))) != OT_UNARY && o != OT_LEAF) {
      fputc('(', gbl.dbgfil);
      prilitree(ILI_OPND(i, 2));
      fputc(')', gbl.dbgfil);
    } else
      prilitree(ILI_OPND(i, 2));
    break;

  case IL_INEG:
  case IL_KNEG:
  case IL_UKNEG:
  case IL_SCMPLXNEG:
  case IL_DCMPLXNEG:
  case IL_DNEG:
  case IL_UINEG:
  case IL_FNEG:
    opval = "-";
    goto unop;
  case IL_NOT:
  case IL_UNOT:
  case IL_KNOT:
  case IL_UKNOT:
    opval = "!";
  unop:
    fprintf(gbl.dbgfil, "%s", opval);
    if ((o = optype(ILI_OPC(ILI_OPND(i, 1)))) != OT_UNARY && o != OT_LEAF) {
      fputc('(', gbl.dbgfil);
      prilitree(ILI_OPND(i, 1));
      fputc(')', gbl.dbgfil);
    } else
      prilitree(ILI_OPND(i, 1));
    break;
  case IL_ICMPZ:
  case IL_KCMPZ:
  case IL_FCMPZ:
  case IL_DCMPZ:
  case IL_ACMPZ:
  case IL_UICMPZ:
    opval = ccval[ILI_OPND(i, 2)];
    if ((o = optype(ILI_OPC(ILI_OPND(i, 1)))) != OT_UNARY && o != OT_LEAF) {
      fputc('(', gbl.dbgfil);
      prilitree(ILI_OPND(i, 1));
      fputc(')', gbl.dbgfil);
    } else
      prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, "%s0", opval);
    break;

  case IL_HFMAX:
  case IL_FMAX:
  case IL_DMAX:
  case IL_KMAX:
  case IL_IMAX:
    n = 2;
    opval = "max";
    goto intrinsic;
  case IL_HFMIN:
  case IL_FMIN:
  case IL_DMIN:
  case IL_KMIN:
  case IL_IMIN:
    n = 2;
    opval = "min";
    goto intrinsic;
  case IL_DBLE:
    n = 1;
    opval = "dble";
    goto intrinsic;
  case IL_SNGL:
    n = 1;
    opval = "sngl";
    goto intrinsic;
  case IL_SCMPLX2IMAG:
    n = 1;
    opval = "imag";
    goto intrinsic;
    break;
  case IL_DCMPLX2IMAG:
    n = 1;
    opval = "dimag";
    goto intrinsic;
    break;
  case IL_SCMPLX2REAL:
    n = 1;
    opval = "real";
    goto intrinsic;
    break;
  case IL_DCMPLX2REAL:
    n = 1;
    opval = "dreal";
    goto intrinsic;
    break;
  case IL_SPSP2SCMPLX:
    n = 2;
    opval = "cmplx";
    goto intrinsic;
    break;
  case IL_SPSP2SCMPLXI0:
    n = 1;
    opval = "cmplx";
    goto intrinsic;
    break;
  case IL_DPDP2DCMPLX:
    n = 2;
    opval = "dcmplx";
    goto intrinsic;
    break;
  case IL_DPDP2DCMPLXI0:
    n = 1;
    opval = "dcmplx";
    goto intrinsic;
    break;
  case IL_SCMPLXCONJG:
    n = 1;
    opval = "conjg";
    goto intrinsic;
    break;
  case IL_DCMPLXCONJG:
    n = 1;
    opval = "conjg";
    goto intrinsic;
    break;
  case IL_SCMPLXEXP:
    n = 1;
    opval = "cexp";
    goto intrinsic;
    break;
  case IL_DCMPLXEXP:
    n = 1;
    opval = "cdexp";
    goto intrinsic;
    break;
  case IL_SCMPLXCOS:
    n = 1;
    opval = "cexp";
    goto intrinsic;
    break;
  case IL_DCMPLXCOS:
    n = 1;
    opval = "cdexp";
    goto intrinsic;
    break;
  case IL_SCMPLXSIN:
    n = 1;
    opval = "csin";
    goto intrinsic;
    break;
  case IL_DCMPLXSIN:
    n = 1;
    opval = "cdsin";
    goto intrinsic;
    break;
  case IL_SCMPLXTAN:
    n = 1;
    opval = "ctan";
    goto intrinsic;
    break;
  case IL_DCMPLXTAN:
    n = 1;
    opval = "cdtan";
    goto intrinsic;
    break;
  case IL_SCMPLXACOS:
    n = 1;
    opval = "cacos";
    goto intrinsic;
    break;
  case IL_DCMPLXACOS:
    n = 1;
    opval = "cdacos";
    goto intrinsic;
    break;
  case IL_SCMPLXASIN:
    n = 1;
    opval = "casin";
    goto intrinsic;
    break;
  case IL_DCMPLXASIN:
    n = 1;
    opval = "cdasin";
    goto intrinsic;
    break;
  case IL_SCMPLXATAN:
    n = 1;
    opval = "catan";
    goto intrinsic;
    break;
  case IL_DCMPLXATAN:
    n = 1;
    opval = "cdatan";
    goto intrinsic;
    break;
  case IL_SCMPLXCOSH:
    n = 1;
    opval = "ccosh";
    goto intrinsic;
    break;
  case IL_DCMPLXCOSH:
    n = 1;
    opval = "cdcosh";
    goto intrinsic;
    break;
  case IL_SCMPLXSINH:
    n = 1;
    opval = "csinh";
    goto intrinsic;
    break;
  case IL_DCMPLXSINH:
    n = 1;
    opval = "cdsinh";
    goto intrinsic;
    break;
  case IL_SCMPLXTANH:
    n = 1;
    opval = "ctanh";
    goto intrinsic;
    break;
  case IL_DCMPLXTANH:
    n = 1;
    opval = "cdtanh";
    goto intrinsic;
    break;
  case IL_SCMPLXLOG:
    n = 1;
    opval = "clog";
    goto intrinsic;
    break;
  case IL_DCMPLXLOG:
    n = 1;
    opval = "cdlog";
    goto intrinsic;
    break;
  case IL_SCMPLXSQRT:
    n = 1;
    opval = "csqrt";
    goto intrinsic;
    break;
  case IL_DCMPLXSQRT:
    n = 1;
    opval = "cdsqrt";
    goto intrinsic;
    break;
  case IL_FIX:
  case IL_FIXK:
  case IL_FIXUK:
    n = 1;
    opval = "fix";
    goto intrinsic;
  case IL_DFIXK:
  case IL_DFIXUK:
    n = 1;
    opval = "dfix";
    goto intrinsic;
  case IL_UFIX:
    n = 1;
    opval = "fix";
    goto intrinsic;
  case IL_DFIX:
  case IL_DFIXU:
    n = 1;
    opval = "dfix";
    goto intrinsic;
  case IL_FLOAT:
    n = 1;
    opval = "float";
    goto intrinsic;
  case IL_FLOATK:
    n = 1;
    opval = "floatk";
    goto intrinsic;
  case IL_FLOATU:
    n = 1;
    opval = "floatu";
    goto intrinsic;
  case IL_FLOATUK:
    n = 1;
    opval = "floatuk";
    goto intrinsic;
  case IL_DEXP:
    n = 1;
    opval = "dexp";
    goto intrinsic;
  case IL_DFLOAT:
    n = 1;
    opval = "dfloat";
    goto intrinsic;
  case IL_DFLOATU:
    n = 1;
    opval = "dfloatu";
    goto intrinsic;
  case IL_DFLOATK:
    n = 1;
    opval = "dfloatk";
    goto intrinsic;
  case IL_DFLOATUK:
    n = 1;
    opval = "dfloatuk";
    goto intrinsic;
  case IL_DNEWT:
  case IL_FNEWT:
    n = 1;
    opval = "recip";
    goto intrinsic;
  case IL_NINT:
    n = 1;
    opval = "nint";
    goto intrinsic;
#ifdef IL_KNINT
  case IL_KNINT:
    n = 1;
    opval = "knint";
    goto intrinsic;
#endif
  case IL_IDNINT:
    n = 1;
    opval = "idnint";
    goto intrinsic;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_IQNINT:
    n = 1;
    opval = "iqnint";
    goto intrinsic;
#endif
#ifdef IL_KIDNINT
  case IL_KIDNINT:
    n = 1;
    opval = "kidnint";
    goto intrinsic;
#endif
  case IL_DABS:
    n = 1;
    opval = "abs";
    goto intrinsic;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QABS:
    n = 1;
    opval = "abs";
    goto intrinsic;
#endif
  case IL_FABS:
    n = 1;
    opval = "abs";
    goto intrinsic;
  case IL_KABS:
    n = 1;
    opval = "abs";
    goto intrinsic;
  case IL_IABS:
    n = 1;
    opval = "abs";
    goto intrinsic;
  case IL_FSQRT:
    n = 1;
    opval = "sqrt";
    goto intrinsic;
  case IL_SIGN:
    n = 2;
    opval = "sign";
    goto intrinsic;
  case IL_RCPSS:
    n = 1;
    opval = "rcpss";
    goto intrinsic;
  case IL_RSQRTSS:
    n = 1;
    opval = "rsqrtss";
    goto intrinsic;
  case IL_DSQRT:
    n = 1;
    opval = "dsqrt";
    goto intrinsic;
  case IL_DSIGN:
    n = 2;
    opval = "dsign";
    goto intrinsic;
#ifdef IL_FRSQRT
  case IL_FRSQRT:
    n = 1;
    opval = "frsqrt";
    goto intrinsic;
#endif
  case IL_FAND:
    n = 2;
    opval = "andps";
    goto intrinsic;
  case IL_ILEADZI:
    n = 2;
    opval = "leadz";
    goto intrinsic;
  case IL_ILEADZ:
  case IL_KLEADZ:
    n = 1;
    opval = "leadz";
    goto intrinsic;
  case IL_ITRAILZI:
    n = 2;
    opval = "trailz";
    goto intrinsic;
  case IL_ITRAILZ:
  case IL_KTRAILZ:
    n = 1;
    opval = "trailz";
    goto intrinsic;
  case IL_IPOPCNTI:
    n = 2;
    opval = "popcnt";
    goto intrinsic;
  case IL_IPOPCNT:
  case IL_KPOPCNT:
    n = 1;
    opval = "popcnt";
    goto intrinsic;
  case IL_IPOPPARI:
    n = 2;
    opval = "poppar";
    goto intrinsic;
  case IL_IPOPPAR:
  case IL_KPOPPAR:
    n = 1;
    opval = "poppar";
    goto intrinsic;
  case IL_CMPNEQSS:
    n = 2;
    opval = "cmpneqss";
    goto intrinsic;
  case IL_VA_ARG:
    n = 1;
    opval = IL_NAME(opc);
    goto intrinsic;
  case IL_ALLOC:
  case IL_DEALLOC:
    n = 1;
    opval = IL_NAME(opc);
    goto intrinsic;
  intrinsic:
    fprintf(gbl.dbgfil, "%s(", opval);
    for (j = 1; j <= n; ++j) {
      prilitree(ILI_OPND(i, j));
      if (j != n)
        fprintf(gbl.dbgfil, ",");
    }
    fprintf(gbl.dbgfil, ")");
    break;

  case IL_JMP:
    fprintf(gbl.dbgfil, "goto %d[bih%d]\n", ILI_OPND(i, 1),
            ILIBLKG(ILI_OPND(i, 1)));
    break;

  case IL_UKCJMP:
  case IL_KCJMP:
  case IL_ICJMP:
  case IL_FCJMP:
  case IL_DCJMP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMP:
#endif
  case IL_ACJMP:
  case IL_UICJMP:
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, " %s ", ccval[ILI_OPND(i, 3)]);
    prilitree(ILI_OPND(i, 2));
    fprintf(gbl.dbgfil, " goto %d[bih%d]\n", ILI_OPND(i, 4),
            ILIBLKG(ILI_OPND(i, 4)));
    break;
  case IL_KCJMPZ:
  case IL_UKCJMPZ:
  case IL_ICJMPZ:
  case IL_FCJMPZ:
  case IL_DCJMPZ:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMPZ:
#endif
  case IL_ACJMPZ:
  case IL_UICJMPZ:
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, " %s 0", ccval[ILI_OPND(i, 2)]);
    fprintf(gbl.dbgfil, " goto %d[bih%d]\n", ILI_OPND(i, 3),
            ILIBLKG(ILI_OPND(i, 3)));
    break;

  case IL_DFRKR:
  case IL_DFRIR:
  case IL_DFRSP:
  case IL_DFRDP:
  case IL_DFRCS:
  case IL_DFRAR:
    prilitree(ILI_OPND(i, 1));
    break;

  case IL_JSRA:
  case IL_GJSRA:
    fprintf(gbl.dbgfil, "*(");
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ")(");
    goto like_jsr;
  case IL_QJSR:
  case IL_JSR:
  case IL_GJSR:
    fprintf(gbl.dbgfil, "%s(", getprint(ILI_OPND(i, 1)));
  like_jsr:
    j = ILI_OPND(i, 2);
    k = 0;
    while (ILI_OPC(j) != 0) {
      if (k)
        fprintf(gbl.dbgfil, ", ");
      switch (ILI_OPC(j)) {
      case IL_DAKR:
      case IL_DAAR:
      case IL_DADP:
#ifdef IL_DA128
      case IL_DA128:
#endif
#ifdef IL_DA256
      case IL_DA256:
#endif
      case IL_DASP:
      case IL_DAIR:
#ifdef IL_DASPSP
      case IL_DASPSP:
      case IL_DACS:
      case IL_DACD:
#endif
        prilitree(ILI_OPND(j, 1));
        j = ILI_OPND(j, 3);
        break;
#ifdef IL_ARGRSRV
      case IL_ARGRSRV:
        fprintf(gbl.dbgfil, "%%%d", ILI_OPND(j, 1));
        j = ILI_OPND(j, 2);
        break;
#endif
      case IL_ARGKR:
      case IL_ARGIR:
      case IL_ARGSP:
      case IL_ARGDP:
      case IL_ARGAR:
#ifdef LONG_DOUBLE_FLOAT128
      case IL_FLOAT128ARG:
#endif
      case IL_GARG:
      case IL_GARGRET:
        prilitree(ILI_OPND(j, 1));
        j = ILI_OPND(j, 2);
        break;
      default:
        goto done;
      }
      k = 1;
    }
  done:
    fprintf(gbl.dbgfil, ")\n");
    break;

  case IL_MVKR:
    opval = "MVKR";
    fprintf(gbl.dbgfil, "%s(%d,%d) = ", opval, KR_MSH(ILI_OPND(i, 2)),
            KR_LSH(ILI_OPND(i, 2)));
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, "\n");
    break;
  case IL_MVIR:
    opval = "MVIR";
    goto mv_reg;
  case IL_MVSP:
    opval = "MVSP";
    goto mv_reg;
  case IL_MVDP:
    opval = "MVDP";
    goto mv_reg;
  case IL_MVAR:
    opval = "MVAR";
    goto mv_reg;
  mv_reg:
    fprintf(gbl.dbgfil, "%s(%2d) = ", opval, ILI_OPND(i, 2));
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, "\n");
    break;
  case IL_KRDF:
    opval = "KRDF";
    fprintf(gbl.dbgfil, "%s(%d,%d)", opval, KR_MSH(ILI_OPND(i, 1)),
            KR_LSH(ILI_OPND(i, 1)));
    break;
  case IL_IRDF:
    opval = "IRDF";
    goto df_reg;
  case IL_SPDF:
    opval = "SPDF";
    goto df_reg;
  case IL_DPDF:
    opval = "DPDF";
    goto df_reg;
  case IL_ARDF:
    opval = "ARDF";
    goto df_reg;
  df_reg:
    fprintf(gbl.dbgfil, "%s(%2d)", opval, ILI_OPND(i, 1));
    break;
  case IL_IAMV:
  case IL_AIMV:
  case IL_KAMV:
  case IL_AKMV:
    prilitree(ILI_OPND(i, 1));
    break;
  case IL_KIMV:
    fprintf(gbl.dbgfil, "_K2I(");
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ")");
    break;
  case IL_IKMV:
    fprintf(gbl.dbgfil, "_I2K(");
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ")");
    break;
  case IL_UIKMV:
    fprintf(gbl.dbgfil, "_UI2K(");
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ")");
    break;

  case IL_CSE:
  case IL_CSEKR:
  case IL_CSEIR:
  case IL_CSESP:
  case IL_CSEDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_CSEQP:
#endif
  case IL_CSEAR:
  case IL_CSECS:
  case IL_CSECD:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CSE:
#endif
    fprintf(gbl.dbgfil, "#<");
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ">#");
    break;
  case IL_FREEKR:
    opval = "FREEKR";
    goto pscomm;
  case IL_FREECS:
    opval = "FREECS";
    goto pscomm;
  case IL_FREECD:
    opval = "FREECD";
    goto pscomm;
  case IL_FREEDP:
    opval = "FREEDP";
    goto pscomm;
  case IL_FREESP:
    opval = "FREESP";
    goto pscomm;
  case IL_FREEAR:
    opval = "FREEAR";
    goto pscomm;
  case IL_FREEIR:
    opval = "FREEIR";
    goto pscomm;
  case IL_FREE:
    opval = "FREE";
    goto pscomm;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128FREE:
    opval = "FLOAT128FR";
    goto pscomm;
#endif /* LONG_DOUBLE_FLOAT128 */
  pscomm:
    fprintf(gbl.dbgfil, "%s = ", opval);
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ";\n");
    break;

  case IL_KCON:
  case IL_ICON:
  case IL_FCON:
  case IL_DCON:
  case IL_SCMPLXCON:
  case IL_DCMPLXCON:
    prcon(ILI_OPND(i, 1));
    break;

  case IL_ACON:
    j = ILI_OPND(i, 1);
    if (CONVAL1G(j)) {
      fprintf(gbl.dbgfil, "<");
      prsym(CONVAL1G(j), gbl.dbgfil);
    } else
      fprintf(gbl.dbgfil, "<%d", CONVAL1G(j));
    fprintf(gbl.dbgfil, ",%" ISZ_PF "d>", ACONOFFG(j));
    break;

  case IL_LD:
  case IL_LDSP:
  case IL_LDDP:
  case IL_LDSCMPLX:
  case IL_LDDCMPLX:
  case IL_LDKR:
  case IL_LDA:
    prnme(ILI_OPND(i, 2), ILI_OPND(i, 1));
    if (DBGBIT(10, 4096)) {
      fprintf(gbl.dbgfil, "<*");
      prilitree(ILI_OPND(i, 1));
      fprintf(gbl.dbgfil, "*>");
    }
    break;
  case IL_VLD:
  case IL_VLDU:
    prnme(ILI_OPND(i, 2), ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, "(V%d) ", ILI_OPND(i, 3));
    if (DBGBIT(10, 4096)) {
      fprintf(gbl.dbgfil, "<*");
      prilitree(ILI_OPND(i, 1));
      fprintf(gbl.dbgfil, "*>");
    }
    break;

  case IL_STKR:
  case IL_ST:
  case IL_STDP:
  case IL_STSP:
  case IL_STSCMPLX:
  case IL_STDCMPLX:
  case IL_DSTS_SCALAR:
  case IL_SSTS_SCALAR:
  case IL_STA:
    prnme(ILI_OPND(i, 3), ILI_OPND(i, 2));
    if (DBGBIT(10, 4096)) {
      fprintf(gbl.dbgfil, "<*");
      prilitree(ILI_OPND(i, 2));
      fprintf(gbl.dbgfil, "*>");
    }
    fprintf(gbl.dbgfil, " = ");
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ";\n");
    break;
  case IL_VST:
  case IL_VSTU:
    prnme(ILI_OPND(i, 3), ILI_OPND(i, 2));
    fprintf(gbl.dbgfil, "(V%d) ", ILI_OPND(i, 4));
    if (DBGBIT(10, 4096)) {
      fprintf(gbl.dbgfil, "<*");
      prilitree(ILI_OPND(i, 2));
      fprintf(gbl.dbgfil, "*>");
    }
    fprintf(gbl.dbgfil, " = ");
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ";\n");
    break;

  case IL_PREFETCHNTA:
  case IL_PREFETCHT0:
  case IL_PREFETCHW:
  case IL_PREFETCH:
    fprintf(gbl.dbgfil, "%s ", IL_MNEMONIC(opc));
    prilitree(ILI_OPND(i, 1));
    if (ILI_OPND(i, 3)) {
      dumpname(ILI_OPND(i, 3));
    }
    fprintf(gbl.dbgfil, ";\n");
    break;
  case IL_PDMV_LOWH:
  case IL_PDMV_HIGHH:
  case IL_PSLD_LOWH:
  case IL_PSLD_HIGHH:
  case IL_PSST_LOWH:
  case IL_PSST_HIGHH:
  case IL_PDLD_LOWH:
  case IL_PDLD_HIGHH:
  case IL_PDST_LOWH:
  case IL_PDLD:
  case IL_PDST:
  case IL_PDMUL:
  case IL_PDST_HIGHH:
    fprintf(gbl.dbgfil, "%s( ", IL_NAME(opc));
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, ", xmm%d", ILI_OPND(i, 2));
    fprintf(gbl.dbgfil, " )");
    if (ILI_OPND(i, 3)) {
      dumpname(ILI_OPND(i, 3));
    }
    fprintf(gbl.dbgfil, "\n");
    break;

  case IL_ISELECT:
  case IL_ASELECT:
  case IL_KSELECT:
  case IL_FSELECT:
  case IL_DSELECT:
  case IL_CSSELECT:
  case IL_CDSELECT:
    fprintf(gbl.dbgfil, "%s(", IL_NAME(opc));
    prilitree(ILI_OPND(i, 1));
    fprintf(gbl.dbgfil, " : ");
    prilitree(ILI_OPND(i, 2));
    fprintf(gbl.dbgfil, " : ");
    prilitree(ILI_OPND(i, 3));
    fprintf(gbl.dbgfil, ")\n");
    break;

  default:
    dilitree(i);
    break;
  }
}
#endif

void
dmpilitree(int i)
{
#if DEBUG
  assert(i > 0, "dmpilitree: invalid value for ili pointer:", i, ERR_Severe);
  if (DBGBIT(10, 512))
    prilitree(i);
  else
    dilitree(i);
#endif
}

#if DEBUG
void
ddilitree(int i, int flag)
{
  FILE *f = gbl.dbgfil;
  assert(i > 0, "ddilitree: invalid value for ili pointer:", i, ERR_Severe);
  gbl.dbgfil = stderr;
  if (flag) {
    prilitree(i);
    fprintf(gbl.dbgfil, "\n");
  } else
    dilitree(i);
  gbl.dbgfil = f;
}

void
_ddilitree(int i, int flag)
{
  FILE *f = gbl.dbgfil;
  assert(i > 0, "_ddilitree: invalid value for ili pointer:", i, ERR_Severe);
  if (f == NULL)
    gbl.dbgfil = stderr;
  if (flag) {
    prilitree(i);
    fprintf(gbl.dbgfil, "\n");
  } else
    dilitree(i);
  gbl.dbgfil = f;
}
#endif

static int
get_encl_function(int sptr)
{
  int enclsptr = ENCLFUNCG(sptr);
  while (enclsptr &&
         STYPEG(enclsptr) != ST_ENTRY
  ) {
    assert(enclsptr != ENCLFUNCG(enclsptr),
           "Invalid ENCLFUNC, cannot be recursive", sptr, ERR_Fatal);
    enclsptr = ENCLFUNCG(enclsptr);
  }
  return enclsptr;
}

/**
 * \brief Finds the corresponding host function of the device outlined function.

 */
static SPTR
find_host_function() {
  SPTR current_function = GBL_CURRFUNC;
  return current_function;
}

bool
is_llvm_local_private(int sptr) {
  const int enclsptr = get_encl_function(sptr);
  const SPTR current_function = find_host_function();
  if (enclsptr && !OUTLINEDG(enclsptr))
    return false;
  /* Some compiler generated private variables does not have ENCLFUNC set
   * especially if
   * generated in the back end i.e., by optimizer - assumed locally defined in
   * function.
   */
  if (!enclsptr && SCG(sptr) == SC_PRIVATE && OUTLINEDG(GBL_CURRFUNC))
    return true;
  return enclsptr && (SC_PRIVATE == SCG(sptr)) && (enclsptr == current_function);
}

bool
is_llvm_local(int sptr, int funcsptr)
{
  const int enclsptr = get_encl_function(sptr);
  if (enclsptr && !OUTLINEDG(enclsptr))
    return false;
  if (!enclsptr && SCG(sptr) == SC_PRIVATE && OUTLINEDG(GBL_CURRFUNC))
    return true;
  return enclsptr && (SC_PRIVATE == SCG(sptr)) && (enclsptr == GBL_CURRFUNC);
}

static int
ll_internref_ili(SPTR sptr)
{
  int mem, ili, nme, off;
  INT zoff;
  SPTR asym;

  off = 0;
  mem = get_sptr_uplevel_address(sptr);
  zoff = ADDRESSG(mem);

  if (SCG(aux.curr_entry->display) == SC_DUMMY) {
    asym = mk_argasym(aux.curr_entry->display);
    nme = addnme(NT_VAR, asym, 0, 0);
    ili = ad_acon(asym, 0);
    ili = ad2ili(IL_LDA, ili, nme);
  } else {
    nme = addnme(NT_VAR, aux.curr_entry->display, 0, (INT)0);
    ili = ad_acon(aux.curr_entry->display, (INT)0);
    ili = ad2ili(IL_LDA, ili, nme); /* load display struct */
  }
  nme = addnme(NT_VAR, aux.curr_entry->display, 0, (INT)0);

  if (zoff) {
    off = ad_aconi(zoff);
    ili = ad3ili(IL_AADD, ili, off, 0); /* add offset of sptr to display */
  }
  if (PASSBYVALG(sptr)) {
    return ili;
  }
  nme = addnme(NT_VAR, sptr, 0, 0);
  ili = ad2ili(IL_LDA, ili, nme); /* load sptr addr from display struct */
  ADDRCAND(ili, nme);
  return ili;
}

static int
ll_taskprivate_inhost_ili(SPTR sptr)
{
  int ilix, offset, basenm;

  /* This access happens only we copy into firstprivate data for task
   * in the host routine
   */
  SPTR taskAllocSptr = llTaskAllocSptr();
  if (taskAllocSptr != SPTR_NULL) {
    if (!ADDRESSG(sptr) && SCG(sptr) == SC_PRIVATE && TASKG(sptr)) {
      /* There are certain compiler generated temp variable that
       * is created much later such as forall loop variable when
       * we transform array assignment to loop or temp var
       * to hold temp value for array bounds. We would want to
       * make is local to a routine.
       * Reasons:
       *   1) if host routine is not outlined function,
       *      compiler will give error.
       *   2) it should be private for taskdup routine so that
       *      it does not share with other task.
       */
      return ad_acon(sptr, 0);
    }
    basenm = addnme(NT_VAR, taskAllocSptr, 0, 0);
    ilix = ad2ili(IL_LDA, ad_acon(taskAllocSptr, 0), basenm);
    ilix = ad3ili(IL_AADD, ilix, ad_aconi(ADDRESSG(sptr)), 0);
    return ilix;
  } else {
    offset = ll_get_uplevel_offset(sptr);
    ilix = ad_acon(aux.curr_entry->uplevel, 0);
    basenm = addnme(NT_VAR, aux.curr_entry->uplevel, 0, 0);
    ilix = ad2ili(IL_LDA, ilix, basenm);
  }
  return ilix;
}

static int
ll_uplevel_addr_ili(SPTR sptr, bool is_task_priv)
{
  int ilix, basenm, offset;
  bool isLocalPriv;

  isLocalPriv = is_llvm_local_private(sptr);
  if (flg.smp) {
    if (!is_task_priv && isLocalPriv) {
      return ad_acon(sptr, (INT)0);
    }
  }

  /* Certain variable: SC_STATIC is set in the backend but PARREF flag may
   * have been set in the front end already.
   */
  if (SCG(sptr) == SC_STATIC && !THREADG(sptr))
    return ad_acon(sptr, (INT)0);
  if (SCG(aux.curr_entry->uplevel) == SC_DUMMY) {
    SPTR asym = mk_argasym(aux.curr_entry->uplevel);
    int anme = addnme(NT_VAR, asym, 0, (INT)0);
    ilix = ad_acon(asym, 0);
    ilix = ad2ili(IL_LDA, ilix, anme);
    basenm = addnme(NT_VAR, aux.curr_entry->uplevel, 0, 0);
    if (TASKFNG(GBL_CURRFUNC) || ISTASKDUPG(GBL_CURRFUNC)) {
      int nme = addnme(NT_IND, aux.curr_entry->uplevel, basenm, 0);
      ilix = ad2ili(IL_LDA, ilix, nme); /* task[0] */
    }
  } else {
    /* aux.curr_entry->uplevel is local ptr = shared ptr address */
    ilix = ad_acon(aux.curr_entry->uplevel, 0);
    basenm = addnme(NT_VAR, aux.curr_entry->uplevel, 0, 0);
    ilix = ad2ili(IL_LDA, ilix, basenm);
  }
  if (TASKFNG(GBL_CURRFUNC) || ISTASKDUPG(GBL_CURRFUNC)) {
    if (TASKG(sptr)) {
      if (is_task_priv) {
        if (ISTASKDUPG(GBL_CURRFUNC)) {
          SPTR arg = ll_get_hostprog_arg(GBL_CURRFUNC, 1);
          ilix = ad_acon(arg, 0);
          basenm = addnme(NT_VAR, arg, 0, 0);
        } else {
          SPTR arg = ll_get_shared_arg(GBL_CURRFUNC);
          ilix = ad_acon(arg, 0);
          basenm = addnme(NT_VAR, arg, 0, 0);
        }
        offset = ADDRESSG(sptr);
      } else {
        if (ISTASKDUPG(GBL_CURRFUNC)) {
          SPTR arg;
          offset = llmp_task_get_privoff(
              sptr, llGetTask(OUTLINEDG(TASKDUPG(GBL_CURRFUNC))));
          if (offset) {
            arg = ll_get_hostprog_arg(GBL_CURRFUNC, 2);
            ilix = ad_acon(arg, 0);
            basenm = addnme(NT_VAR, arg, 0, 0);
          }
        } else
          return ad_acon(sptr, (INT)0);
      }
    } else {
      if (ISTASKDUPG(GBL_CURRFUNC)) {
        offset = llmp_task_get_privoff(
            sptr, llGetTask(OUTLINEDG(TASKDUPG(GBL_CURRFUNC))));
        if (!offset)
          offset = ll_get_uplevel_offset(sptr);
      } else
        offset = ll_get_uplevel_offset(sptr);
    }
  } else if (OMPTEAMPRIVATEG(sptr)) {
    offset = ADDRESSG(sptr);
  } else {
    offset = ll_get_uplevel_offset(sptr);
  }
  ilix = ad3ili(IL_AADD, ilix, ad_aconi(offset), 0);
  ilix = ad2ili(IL_LDA, ilix, addnme(NT_IND, sptr, basenm, 0));
  return ilix;
}

int
mk_charlen_parref_sptr(SPTR sptr)
{
  int ilix, basenm;
  INT offset;
  if (is_llvm_local_private(sptr)) {
    return ad_acon(sptr, 0);
  }
  ilix = ad_acon(aux.curr_entry->uplevel, 0);
  basenm = addnme(NT_VAR, aux.curr_entry->uplevel, 0, 0);
  ilix = ad2ili(IL_LDA, ilix, basenm);
  offset = ll_get_uplevel_offset(sptr);
  offset += 8;
  ilix = ad3ili(IL_AADD, ilix, ad_aconi(offset), 0);
  return ilix;
}

/**
 * \brief create ili representing the address of a variable
 */
int
mk_address(SPTR sptr)
{
  LLTask *task;
  bool is_task_priv;

  if (UPLEVELG(sptr) && (SCG(sptr) == SC_LOCAL || SCG(sptr) == SC_DUMMY)) {
    if (INTERNREFG(sptr) && gbl.internal > 1) {
      return ll_internref_ili(sptr);
    }
  }

  /* call to ll_taskprivate_inhost_ili should happen only when
   * we copy and allocate firstprivate data onto taskalloc ptr
   * in the host routine. Firstprivate initialization must be
   * done before the construct(Note that for native compiler,
   * we do it inside task and that may or may not cause problem
   * in the future).
   */
  if (flg.smp && TASKG(sptr) && !ISTASKDUPG(GBL_CURRFUNC) &&
      (!TASKFNG(GBL_CURRFUNC) || !is_llvm_local_private(sptr))) {
    return ll_taskprivate_inhost_ili(sptr);
  }

  /* Determine if sptr is a firstprivate variable in a task */
  if (ISTASKDUPG(GBL_CURRFUNC)) {
    int taskfn = TASKDUPG(GBL_CURRFUNC);
    task = llmp_task_get_by_fnsptr(taskfn);
    is_task_priv = task && !!llmp_task_get_private(task, sptr, taskfn);
  } else {
    task = TASKFNG(GBL_CURRFUNC) ? llmp_task_get_by_fnsptr(GBL_CURRFUNC) : NULL;
    is_task_priv = task && !!llmp_task_get_private(task, sptr, GBL_CURRFUNC);
  }

  /* Make an address for an outlined function variable */
#ifdef OMP_OFFLOAD_LLVM
  if(!(gbl.outlined && flg.omptarget && gbl.ompaccel_intarget))
#endif
  if ((PARREFG(sptr) || TASKG(sptr)) &&
      (gbl.outlined || ISTASKDUPG(GBL_CURRFUNC))) 
  {
  /* If it's a host function, we skip loading from uplevel and load them directly */
  if(((SCG(sptr) != SC_EXTERN && SCG(sptr) != SC_STATIC)) || THREADG(sptr))
    return ll_uplevel_addr_ili(sptr, is_task_priv);
  }
  if (SCG(sptr) == SC_DUMMY && !REDUCG(sptr)) {
    SPTR asym = mk_argasym(sptr);
    return ad_acon(asym, 0);
  }
#ifdef VLAG
  if ((VLAG(sptr) || SCG(sptr) == SC_BASED) && MIDNUMG(sptr)) {
    return ad2ili(IL_LDA, ad_acon(MIDNUMG(sptr), (INT)0),
                  addnme(NT_VAR, MIDNUMG(sptr), 0, (INT)0));
  }
#endif
  return ad_acon(sptr, (INT)0);
}

int
compute_address(SPTR sptr)
{
  int addr;
  addr = mk_address(sptr);
  if (SCG(sptr) == SC_DUMMY) {
    SPTR asym = mk_argasym(sptr);
    int anme = addnme(NT_VAR, asym, 0, 0);
    addr = ad2ili(IL_LDA, addr, anme);
  }
  return addr;
}

/*
 * General ili traversal routines; a list is used to keep track of
 * the ili which have been visited (located by visit_list).
 */

/** \brief Initialize for a traversal and call the traversal routine.
 *
 * The traversal routine sets the ILI_VISIT & ILI_VLIST fields.
 */
int
ili_traverse(int (*visit_f)(int), int ilix)
{
  visit_list = 1;
  ILI_VLIST(1) = 0;
  if (visit_f == NULL)
    /* initializing only; the visit routine is explicitly called */
    return 0;
  return (*visit_f)(ilix);
}

/** \brief Add an ili to visit_list and set its VISIT field.
 */
void
ili_visit(int ilix, int v)
{
  ILI_VISIT(ilix) = v;
  if (ILI_VLIST(ilix) == 0) {
    ILI_VLIST(ilix) = visit_list;
    visit_list = ilix;
  }
}

/** \brief Clear the VISIT & VLIST fields of the ili in the visit_list.
 */
void
ili_unvisit(void)
{
  int v, nxt;

  for (v = visit_list; v;) {
    nxt = ILI_VLIST(v);
    ILI_VLIST(v) = 0;
    ILI_VISIT(v) = 0;
    v = nxt;
  }
  visit_list = 0;
}

/** \brief Look for 'JSR' in this tree
 */
static int
_jsrsearch(int ilix)
{
  ILI_OP opc;
  int noprs, j;
  opc = ILI_OPC(ilix);
  switch (opc) {
  case IL_JSR:
  case IL_JSRA:
    ili_visit(ilix, 1);
    iltb.callfg = 1;
    return 1;
  default:
    break;
  }
  if (ILI_VISIT(ilix)) {
    return (ILI_VISIT(ilix) == 1);
  }
  noprs = IL_OPRS(opc);
  for (j = 1; j <= noprs; ++j) {
    if (IL_ISLINK(opc, j)) {
      const int opnd = ILI_OPND(ilix, j);
      if (_jsrsearch(opnd)) {
        ili_visit(ilix, 1);
        return 1;
      }
    }
  }
  ili_visit(ilix, 2);
  return 0;
}

/** \brief Search for "jsrs"
 */
int
jsrsearch(int ilix)
{
  iltb.callfg = 0;
  return _jsrsearch(ilix);
} /* jsrsearch */

int
alt_qjsr(int ilix)
{
  int altx, j, noprs;
  ILI_OP opc;

  if (ILI_ALT(ilix)) {
    altx = ILI_ALT(ilix);
    opc = ILI_OPC(altx);
    if (IL_TYPE(opc) == ILTY_DEFINE)
      return 1;

    /* In a few cases a QJSR's result ILI (namely a 'define' ILI such
     * as IL_DFRDP, etc) is not 'altx' itself but rather an operand of
     * 'altx'.  For example, if the C99 ABI is used on x86 then a
     * complex*16 function result is returned in a pair of registers,
     * %xmm0 and %xmm1, and the ILI 'altx' is
     * IL_DPDP2DCMPLX( IL_DFRDP(...), IL_DFRDP(...) ).  Also on x86-32
     * a 64-bit condition such as:
     *
     *     if (i8 .gt. 0)
     *
     * gives rise to ILIs such as:
     *
     * 49   QJSR          342~<__mth_i_kcmpz>    48^
     * 50   DFRIR          49^ ir( 1)
     * 51   ICMPZ          50^    gt
     * 52   KCMPZ          47^    gt     51^-alt
     *
     * The solution for now is to check whether any operand of 'altx'
     * is a define ILI.
     *
     * This deals with the cases that we're currently aware of, but if
     * a situation arises in which the define ILI is more deeply
     * nested in 'altx's ILI tree then we'll have to implement a
     * complete fix, e.g. by calling 'qjsr_in()'.  Note, if this
     * function fails to detect a QJSR ILI that is present in 'altx's
     * ILI tree then the x86 native CG may indicate this by generating
     * the warning ICE "gen_lilis: BIH_QJSR needs to be set for bih
     * <bih_number>".
     */
    noprs = IL_OPRS(opc);
    for (j = 1; j <= noprs; ++j) {
      if (IL_ISLINK(opc, j) &&
          IL_TYPE(ILI_OPC(ILI_OPND(altx, j))) == ILTY_DEFINE) {
        return 1;
      }
    }
  }

  switch (ILI_OPC(ilix)) {
  case IL_FSINCOS:
  case IL_DSINCOS:
    return 1;
  default:
    break;
  }
  return 0;
} /* end alt_qjsr(int ilix) */

/** \brief Look for 'QJSR' in this tree
 */
int
qjsrsearch(int ilix)
{
  ILI_OP opc;
  int noprs, j;
  opc = ILI_OPC(ilix);
  switch (opc) {
  case IL_QJSR:
    ili_visit(ilix, 1);
    return 1;
  default:;
  }
  if (alt_qjsr(ilix)) {
    ili_visit(ilix, 1);
    return 1;
  }
  if (ILI_VISIT(ilix)) {
    if (ILI_VISIT(ilix) == 1)
      return 1;
    return 0;
  }
  noprs = IL_OPRS(opc);
  for (j = 1; j <= noprs; ++j) {
    int opnd;
    if (IL_ISLINK(opc, j)) {
      opnd = ILI_OPND(ilix, j);
      if (qjsrsearch(opnd)) {
        ili_visit(ilix, 1);
        return 1;
      }
    }
  }
  ili_visit(ilix, 2);
  return 0;
}

/** \brief If a VECT ili, return its dtype, 0 otherwise.
 */

int
ili_get_vect_arg_count(int ilix)
{
  if (ILI_OPC(ilix) == IL_CSE)
    return ili_get_vect_arg_count(ILI_OPND(ilix, 1));
  if (IL_VECT(ILI_OPC(ilix))) {
    switch (ILI_OPC(ilix)) {
    case IL_VNEG:
    case IL_VCVTV:
    case IL_VCVTS:
    case IL_VNOT:
    case IL_VABS:
      return 2;
    case IL_VSQRT:
    case IL_VCOS:
    case IL_VSIN:
    case IL_VACOS:
    case IL_VASIN:
    case IL_VSINCOS:
    case IL_VTAN:
    case IL_VSINH:
    case IL_VCOSH:
    case IL_VTANH:
    case IL_VATAN:
    case IL_VEXP:
    case IL_VLOG:
    case IL_VLOG10:
    case IL_VRCP:
    case IL_VRSQRT:
    case IL_VLD:
    case IL_VLDU:
    case IL_VADD:
    case IL_VSUB:
    case IL_VMUL:
    case IL_VAND:
    case IL_VOR:
    case IL_VXOR:
    case IL_VCMPNEQ:
    case IL_VLSHIFTV:
    case IL_VRSHIFTV:
    case IL_VLSHIFTS:
    case IL_VRSHIFTS:
    case IL_VURSHIFTS:
    case IL_VMIN:
    case IL_VMAX:
      return 3;
    case IL_VDIV:
    case IL_VDIVZ:
    case IL_VMOD:
    case IL_VMODZ:
    case IL_VPOW:
    case IL_VPOWI:
    case IL_VPOWK:
    case IL_VPOWIS:
    case IL_VPOWKS:
    case IL_VFPOWK:
    case IL_VDPOWI:
    case IL_VFPOWKS:
    case IL_VDPOWIS:
    case IL_VATAN2:
    case IL_VST:
    case IL_VSTU:
    case IL_VFMA1:
    case IL_VFMA2:
    case IL_VFMA3:
    case IL_VFMA4:
    case IL_VPERMUTE:
    case IL_VCMP:
    case IL_VBLEND:
      return 4;
    default:
      break;
    }
  }
  return 0;
}

DTYPE
ili_get_vect_dtype(int ilix)
{
  if (ILI_OPC(ilix) == IL_CSE)
    return ili_get_vect_dtype(ILI_OPND(ilix, 1));
  if (!IL_VECT(ILI_OPC(ilix)))
    return DT_NONE;
  switch (ILI_OPC(ilix)) {
  case IL_VCON:
    return DTYPEG(ILI_OPND(ilix, 1));
  case IL_VNEG:
  case IL_VCVTV:
  case IL_VCVTS:
  case IL_VCVTR:
  case IL_VNOT:
  case IL_VABS:
    return DT_ILI_OPND(ilix, 2);
  case IL_VSQRT:
  case IL_VCOS:
  case IL_VSIN:
  case IL_VACOS:
  case IL_VASIN:
  case IL_VSINCOS:
  case IL_VTAN:
  case IL_VSINH:
  case IL_VCOSH:
  case IL_VTANH:
  case IL_VATAN:
  case IL_VEXP:
  case IL_VLOG:
  case IL_VLOG10:
  case IL_VRCP:
  case IL_VRSQRT:
  case IL_VFLOOR:
  case IL_VCEIL:
  case IL_VAINT:
  case IL_VLD:
  case IL_VLDU:
  case IL_VADD:
  case IL_VSUB:
  case IL_VMUL:
  case IL_VAND:
  case IL_VOR:
  case IL_VXOR:
  case IL_VCMPNEQ:
  case IL_VLSHIFTV:
  case IL_VRSHIFTV:
  case IL_VLSHIFTS:
  case IL_VRSHIFTS:
  case IL_VURSHIFTS:
  case IL_VMIN:
  case IL_VMAX:
    return DT_ILI_OPND(ilix, 3);
  case IL_VDIV:
  case IL_VDIVZ:
  case IL_VMOD:
  case IL_VMODZ:
  case IL_VPOW:
  case IL_VPOWI:
  case IL_VPOWK:
  case IL_VPOWIS:
  case IL_VPOWKS:
  case IL_VFPOWK:
  case IL_VDPOWI:
  case IL_VFPOWKS:
  case IL_VDPOWIS:
  case IL_VATAN2:
  case IL_VST:
  case IL_VSTU:
  case IL_VFMA1:
  case IL_VFMA2:
  case IL_VFMA3:
  case IL_VFMA4:
  case IL_VPERMUTE:
  case IL_VCMP:
  case IL_VBLEND:
    return DT_ILI_OPND(ilix, 4);
  default:
    interr("ili_get_vect_dtype missing case for ili opc", ILI_OPC(ilix),
           ERR_Severe);
  }
  return DT_NONE;
}

/** \brief Return MSZ_ for each datatype, or -1 if an aggregate type.
 *
 * Result is the memory reference size, not including padding.
 * For example, TY_X87 on x86 targets maps to F10.
 *
 * The logic is custom to each target.
 */
MSZ
mem_size(TY_KIND ty)
{
  MSZ msz = MSZ_UNDEF;
  switch (ty) {
  case TY_PTR:
    msz = MSZ_PTR;
    break;
  case TY_LOG8:
  case TY_INT8:
    msz = MSZ_I8;
    break;
  case TY_UINT8:
    msz = MSZ_I8;
    break;
  case TY_INT:
    msz = MSZ_WORD;
    break;
  case TY_FLOAT:
    msz = MSZ_F4;
    break;
  case TY_QUAD:
    DEBUG_ASSERT(size_of(DT_QUAD) == 8, "TY_QUAD assumed to be 8 bytes");
    msz = MSZ_F8;
    break;
  case TY_DBLE:
    msz = MSZ_F8;
    break;
  case TY_CMPLX:
    msz = MSZ_F8;
    break;
  case TY_DCMPLX:
    msz = MSZ_F16;
    break;
  case TY_BLOG:
  case TY_BINT:
    msz = MSZ_SBYTE;
    break;
  case TY_LOG:
    msz = MSZ_WORD;
    break;

  case TY_UINT:
    msz = MSZ_WORD;
    break;

  case TY_SINT:
    msz = MSZ_SHWORD;
    break;
  case TY_SLOG:
    msz = MSZ_SHWORD;
    break;
  case TY_USINT:
    msz = MSZ_UHWORD;
    break;
  case TY_FLOAT128:
    msz = MSZ_F16;
    break;
  case TY_128:
    msz = MSZ_F16;
    break;
  default:
    msz = MSZ_UNDEF;
  }
  return msz;
} /* mem_size */

static int
_ipowi(int x, int i)
{
  int f;

  /* special cases */

  if (i < 0) {
    if (x == 1)
      return 1;
    if (x == -1) {
      if (i & 1)
        return -1;
      return 1;
    }
    return 0;
  }

  if (i == 0)
    return 1;
  f = 1;
  while (1) {
    if (i & 1)
      f *= x;
    i >>= 1;
    if (i == 0)
      return f;
    x *= x;
  }
}

/** Raising an operand to a constant power >= 1.  generate ILI which maximize
 * cse's (i.e., generate a balanced tree).
 *
 * - opn -- operand (ILI) raised to power 'pwd'
 * -  pwr -- power (constant)
 * -  opc -- mult ILI opcode
 */
static int
_xpowi(int opn, int pwr, ILI_OP opc)
{
  int res;
  int p2; /* largest power of 2 such that 2**p2 <= opn**pwr */
  int n;

  if (pwr >= 2) {
    p2 = 0;
    n = pwr;
    while ((n >>= 1) > 0)
      p2++;

    n = 1 << p2; /* 2**p2 */
    res = opn;
    /* generate a balanced multiply tree whose height is p2 */
    while (p2-- > 0)
      res = ad2ili(opc, res, res);

    /* residual */
    n = pwr - n;
    if (n > 0) {
      int right;
      right = _xpowi(opn, n, opc);
      res = ad2ili(opc, res, right);
    }

    return res;
  }
  return opn;
}

#if defined(TARGET_X8664) || defined(TARGET_POWER) || !defined(TARGET_LLVM_ARM)
static int
_frsqrt(int x)
{
  int three;
  int mhalf;
  int x0, ilix;
  INT num[2];
  /*
   * Newton's appx for recip sqrt:
   *   x1 = (3.0*x0 - x*x0**3))/2.0
   * or
   *   x1 = (3.0 - x*x0*x0)*x0*.5
   * or
   *   x1 = (x*x0*x0 - 3.0)*x0*(-.5)
   */
  num[0] = 0;
  num[1] = 0x40400000;
  three = ad1ili(IL_FCON, getcon(num, DT_FLOAT));
  num[1] = 0xbf000000;
  mhalf = ad1ili(IL_FCON, getcon(num, DT_FLOAT));
  x0 = ad1ili(IL_RSQRTSS, x);
  ilix = ad2ili(IL_FMUL, x0, x0);
  ilix = ad2ili(IL_FMUL, x, ilix);
  ilix = ad2ili(IL_FSUB, ilix, three);
  ilix = ad2ili(IL_FMUL, ilix, x0);
  ilix = ad2ili(IL_FMUL, ilix, mhalf);
#if defined(IL_FRSQRT)
  ilix = ad1altili(IL_FRSQRT, x, ilix);
#endif
  return ilix;
}
#endif

ISZ_T
get_isz_conili(int ili)
{
  if (ILI_OPC(ili) == IL_KCON) {
    ISZ_T ii;
    ii = get_isz_cval(ILI_OPND(ili, 1));
    return ii;
  }
#if DEBUG
  assert(ILI_OPC(ili) == IL_ICON, "get_isz_conili, unexpected const ILI",
         ILI_OPC(ili), ERR_unused);
#endif
  return CONVAL2G(ILI_OPND(ili, 1));
}

/** \brief Select an integer constant ili, ICON or KCON.
 */
int
sel_icnst(ISZ_T val, int isi8)
{
  if (isi8) {
    return ad_kconi(val);
  }
  return ad_icon(val);
}

/** \brief Select an integer conversion: IR/KR to IR, or IR/KR to KR
 */
int
sel_iconv(int ili, int isi8)
{
  if (isi8) {
    if (IL_RES(ILI_OPC(ili)) != ILIA_KR) {
      return ad1ili(IL_IKMV, ili);
    }
    return ili;
  }
  if (IL_RES(ILI_OPC(ili)) == ILIA_KR)
    return ad1ili(IL_KIMV, ili);
  return ili;
}

/** \brief Select an IR or KR substract by 1
 */
int
sel_decr(int ili, int isi8)
{
  if (isi8)
    return ad2ili(IL_KSUB, ili, ad_kcon(0, 1));
  return ad2ili(IL_ISUB, ili, ad_icon((INT)1));
}

/** \brief Select an IR or KR conversion to AR
 */
int
sel_aconv(int ili)
{
  if (IL_RES(ILI_OPC(ili)) == ILIA_KR)
    return ad1ili(IL_KAMV, ili);
  return ad1ili(IL_IAMV, ili);
}

/*
 * The following routines replace code which relied on the ordering of
 * certain ili and a defined relationship between ILIA_MAX & ILI_AR, e.g.,
 *     opc >= IL_CSEIR && opc <= (IL_CSEAR+(ILIA_MAX-ILIA_AR)
 */

int
is_argili_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_ARGIR:
  case IL_ARGSP:
  case IL_ARGDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_ARGQP:
#endif
  case IL_ARGAR:
  case IL_ARGKR:
  case IL_GARG:
  case IL_GARGRET:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ARG:
#endif
    return 1;
  default:
    return 0;
  }
}

int
is_cseili_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_CSE:
  case IL_CSEIR:
  case IL_CSESP:
  case IL_CSEDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_CSEQP:
#endif
  case IL_CSEAR:
  case IL_CSECS:
  case IL_CSECD:
  case IL_CSEKR:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CSE:
#endif
    return 1;
  default:
    return 0;
  }
}

int
is_freeili_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_FREEIR:
  case IL_FREESP:
  case IL_FREEDP:
  case IL_FREECS:
  case IL_FREECD:
  case IL_FREEAR:
  case IL_FREEKR:
  case IL_FREE:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128FREE:
#endif
    return 1;
  default:
    break;
  }
  return 0;
}

int
is_mvili_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_MVIR:
  case IL_MVSP:
  case IL_MVDP:
  case IL_MVAR:
  case IL_MVKR:
#ifdef IL_MVSPX87
  case IL_MVSPX87:
  case IL_MVDPX87:
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128RETURN:
#endif
  case IL_MVQ:
  case IL_MV256:
    return 1;
  default:
    break;
  }
  return 0;
}

int
is_rgdfili_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_IRDF:
  case IL_SPDF:
  case IL_DPDF:
  case IL_ARDF:
  case IL_KRDF:
    return 1;
  default:
    break;
  }
  return 0;
}

int
is_daili_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_DAIR:
  case IL_DASP:
  case IL_DADP:
#ifdef IL_DA128
  case IL_DA128:
#endif
#ifdef IL_DA256
  case IL_DA256:
#endif
  case IL_DAAR:
  case IL_DAKR:
  case IL_DACS:
  case IL_DACD:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ARG:
#endif
    return 1;
  default:
    break;
  }
  return 0;
}

int
is_dfrili_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_DFRIR:
  case IL_DFRSP:
  case IL_DFRDP:
  case IL_DFRAR:
  case IL_DFRCS:
  case IL_DFRKR:
  case IL_DFRCD:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128RESULT:
#endif
#ifdef IL_DFRDPX87
  case IL_DFRDPX87:
  case IL_DFRSPX87:
#endif
  case IL_DFR128:
  case IL_DFR256:
    return 1;
  default:
    break;
  }
  return 0;
}

bool
is_integer_comparison_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_ICMP:
  case IL_ICMPZ:
  case IL_UICMP:
  case IL_UICMPZ:
  case IL_KCMP:
  case IL_KCMPZ:
  case IL_UKCMP:
  case IL_UKCMPZ:
  case IL_ACMP:
  case IL_ACMPZ:
  case IL_FCMP:
  case IL_FCMPZ:
  case IL_DCMP:
  case IL_DCMPZ:
  case IL_SCMPLXCMP:
  case IL_DCMPLXCMP:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CMP:
#endif
  case IL_ICJMP:
  case IL_ICJMPZ:
  case IL_UICJMP:
  case IL_UICJMPZ:
  case IL_KCJMP:
  case IL_KCJMPZ:
  case IL_UKCJMP:
  case IL_UKCJMPZ:
  case IL_ACJMP:
  case IL_ACJMPZ:
    return true;
  default:
    return false;
  }
}

bool
is_floating_comparison_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_FCMP:
  case IL_FCMPZ:
  case IL_DCMP:
  case IL_DCMPZ:
  case IL_SCMPLXCMP:
  case IL_DCMPLXCMP:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CMP:
#endif
  case IL_FCJMP:
  case IL_FCJMPZ:
  case IL_DCJMP:
  case IL_DCJMPZ:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMP:
  case IL_QCJMPZ:
#endif
    return true;
  default:
    return false;
  }
}

bool
is_unsigned_opcode(ILI_OP opc)
{
  switch (opc) {
  case IL_UINEG:
  case IL_UKNEG:
  case IL_UNOT:
  case IL_UKNOT:
  case IL_UFIX:
  case IL_UIADD:
  case IL_UKADD:
  case IL_UISUB:
  case IL_UKSUB:
  case IL_UIMUL:
  case IL_UKMUL:
  case IL_UIMULH:
  case IL_UKMULH:
  case IL_UIDIV:
  case IL_UKDIV:
  case IL_UIMOD:
  case IL_UIMIN:
  case IL_UKMIN:
  case IL_UIMAX:
  case IL_UKMAX:
  case IL_UICMP:
  case IL_UICMPZ:
  case IL_UKCMP:
  case IL_UKCMPZ:
  case IL_UICJMP:
  case IL_UICJMPZ:
  case IL_UKCJMP:
  case IL_UKCJMPZ:
    return true;
  default:
    return false;
  }
}

int
has_cse(int ilix)
{
  ILI_OP opc;
  int i;

  opc = ILI_OPC(ilix);
  if (is_cseili_opcode(opc))
    return 1;

  for (i = 1; i <= ilis[opc].oprs; i++) {
    if (IL_ISLINK(opc, i) && has_cse(ILI_OPND(ilix, i)))
      return 1;
  }
  return 0;
}

#if DEBUG && defined(TARGET_64BIT)
/*
 * recursive depth-first postorder traversal of the ili tree
 *  look for calls to various MULH routines
 *  KMULH - turn into IL_KMULH
 *  UKMULH - turn into IL_UKMULH
 *  IMULH - turn into upper-part of i32 multipy
 *  UIMULH - turn into upper-part of unsigned i32 multipy
 */
static int
mulhsearch(int ilix, int ilt)
{
  int ret, jsr, mulh, arg1, arg2;
  ILI_OP opc;
  if (ilix <= 1)
    return ilix;
  if (ILI_REPL(ilix) && ILI_VISIT(ilix) == ilt) {
    /* we have a replacement for this ilt */
    ret = ILI_REPL(ilix);
    return ret;
  }
  opc = ILI_OPC(ilix);
  ili_visit(ilix, ilt);
  if (opc >= IL_CSEIR && opc <= IL_CSEKR) {
    int link;
    link = ILI_OPND(ilix, 1);
    if (ILI_REPL(link)) {
      /* just insert IL_CSE.. to this link */
      ILI New;
      if (ILI_REPL(link) == link) {
        ILI_REPL(ilix) = ilix;
        return ilix;
      }
      BZERO(&New, ILI, 1);
      New.opc = opc;
      New.opnd[0] = ILI_REPL(link);
      ret = ILI_REPL(ilix) = addili(&New);
      return ret;
    }
  }
  mulh = 0;
  if (opc == IL_DFRKR) {
    /* is this a call to KMULH */
    jsr = ILI_OPND(ilix, 1);
    if (ILI_OPC(jsr) == IL_JSR) {
      int sym;
      sym = ILI_OPND(jsr, 1);
      if (strcmp(SYMNAME(sym), "KMULH") == 0) {
        arg1 = ILI_OPND(jsr, 2);
        if (arg1) {
          arg2 = ILI_OPND(arg1, 3);
          if (arg2) {
            mulh = 1;
          }
        }
      } else if (strcmp(SYMNAME(sym), "UKMULH") == 0) {
        arg1 = ILI_OPND(jsr, 2);
        if (arg1) {
          arg2 = ILI_OPND(arg1, 3);
          if (arg2) {
            mulh = 2;
          }
        }
      }
    }
  }
  if (opc == IL_DFRIR) {
    /* is this a call to MULH */
    jsr = ILI_OPND(ilix, 1);
    if (ILI_OPC(jsr) == IL_JSR) {
      int sym;
      sym = ILI_OPND(jsr, 1);
      if (strcmp(SYMNAME(sym), "IMULH") == 0) {
        arg1 = ILI_OPND(jsr, 2);
        if (arg1) {
          arg2 = ILI_OPND(arg1, 3);
          if (arg2) {
            mulh = 3;
          }
        }
      } else if (strcmp(SYMNAME(sym), "UIMULH") == 0) {
        arg1 = ILI_OPND(jsr, 2);
        if (arg1) {
          arg2 = ILI_OPND(arg1, 3);
          if (arg2) {
            mulh = 4;
          }
        }
      }
    }
  }
  if (mulh == 1) {
    ret = KMULSH(ILI_OPND(arg1, 1), ILI_OPND(arg2, 1));
    ILI_REPL(ilix) = ret;
  } else if (mulh == 2) {
    ret = KMULUH(ILI_OPND(arg1, 1), ILI_OPND(arg2, 1));
    ILI_REPL(ilix) = ret;
  } else if (mulh == 3) {
    int op1, op2;
    op1 = ad1ili(IL_IKMV, ILI_OPND(arg1, 1));
    op2 = ad1ili(IL_IKMV, ILI_OPND(arg2, 1));
    ret = MULSH(op1, op2);
    ILI_REPL(ilix) = ret;
  } else if (mulh == 4) {
    int op1, op2;
    op1 = ad1ili(IL_UIKMV, ILI_OPND(arg1, 1));
    op2 = ad1ili(IL_UIKMV, ILI_OPND(arg2, 1));
    ret = MULUH(op1, op2);
    ILI_REPL(ilix) = ret;
  } else {
    int noprs, j, changes, newalt;
    ILI New;
    BZERO(&New, ILI, 1);
    noprs = IL_OPRS(opc);
    New.opc = opc;
    changes = 0;
    for (j = 1; j <= noprs; ++j) {
      int opnd;
      opnd = ILI_OPND(ilix, j);
      if (!IL_ISLINK(opc, j)) {
        New.opnd[j - 1] = opnd;
      } else {
        New.opnd[j - 1] = mulhsearch(opnd, ilt);
        if (New.opnd[j - 1] != opnd)
          ++changes;
      }
    }
    newalt = New.alt = 0;
    if (ILI_ALT(ilix)) {
      int alt;
      alt = ILI_ALT(ilix);
      newalt = mulhsearch(alt, ilt);
      New.alt = alt; /* old alt here; new alt to be replaced later */
    }
    if (changes == 0) {
      ret = ilix;
    } else {
      ret = addili(&New);
      ILI_ALT(ret) = New.alt;
    }
  }
  return ret;
} /* mulhsearch */

/* turn KMULH call into IL_KMULH ili */
void
inline_mulh(void)
{
  int block, ili, callfg;
  bool save_share_proc_ili;
  for (ili = 0; ili < ilib.stg_avail; ++ili) {
    ILI_VISIT(ili) = ILI_REPL(ili) = 0;
  }
  save_share_proc_ili = share_proc_ili;
  share_proc_ili = false;

  (void)ili_traverse(NULL, 0);

  callfg = 0;
  for (block = gbl.entbih; block; block = BIH_NEXT(block)) {
    int ilt;
    rdilts(block);
    gbl.lineno = BIH_LINENO(block);
    gbl.findex = BIH_FINDEX(block);
    bihb.callfg = 0;
    iltb.callfg = 0;
    for (ilt = BIH_ILTFIRST(block); ilt; ilt = ILT_NEXT(ilt)) {
      ili_visit(1, 1);
      ILT_ILIP(ilt) = mulhsearch(ILT_ILIP(ilt), ilt);
      ili_unvisit();
      if (ILT_EX(ilt)) {
        /* jsrsearch() sets iltb.callfg to 0 or 1 */
        ili_visit(1, 1);
        ILT_EX(ilt) = jsrsearch(ILT_ILIP(ilt));
        ili_unvisit();
      }
      bihb.callfg |= iltb.callfg;
    }
    BIH_EX(block) = bihb.callfg;
    callfg |= bihb.callfg;
    wrilts(block);
    if (BIH_LAST(block))
      break;
  }

  BIH_EX(gbl.entbih) = callfg;

  share_proc_ili = save_share_proc_ili;
  for (ili = 0; ili < ilib.stg_avail; ++ili) {
    ILI_VISIT(ili) = ILI_REPL(ili) = 0;
  }
} /* inline_mulh */
#endif

int
mkfunc_avx(char *nmptr, int avxp)
{
  int sptr;
  sptr = _mkfunc(nmptr);
#ifdef AVXP
  /*
   * AVXP(sptr,avxp) ought to be sufficient, but am concerned
   * about clearing the AVX field for an existing 'avx' routine.
   */
  if (avxp) {
    AVXP(sptr, 1);
  }
#endif
  return sptr;
}

static int
_mkfunc(const char *name)
{
  int ss;
  if (!XBIT(15, 0x40))
    ss = mkfunc_cncall(name);
  else
    ss = mkfunc(name);
  return ss;
}

static int
DblIsSingle(SPTR dd)
{
  INT num[2];

  if (XBIT(15, 0x80))
    return 0;
  if (is_dbl0(dd)) {
    return ad1ili(IL_FCON, stb.flt0);
  }
  num[0] = CONVAL1G(dd);
  num[1] = CONVAL2G(dd);
  if ((num[1] & 0x1fffffff) == 0) {
    /* the mantissa does not exceed the mantissa of a single
     * precision value
     */
    unsigned uu;
    int de; /* exponent of double value */
    uu = num[0];
    de = (int)((uu >> 20) & 0x7ff) - 1023;
    if (de >= -126 && de <= 127) {
      /*
       * exponent is within the Emin & Emax for single precision.
       */
      unsigned ds, dm; /* sign, mantissa of dble value */
      int se;          /* exponent of single value */
      unsigned ss, sm; /* sign, mantissa of single value */
      INT v;

      uu = num[0];
      ds = uu >> 31;
      dm = (uu & 0xfffff);
      uu = num[1] >> 29;
      dm = (dm << 3) | uu;

      xsngl(num, &v);
      uu = v;
      ss = uu >> 31;
      se = (int)((uu >> 23) & 0xff) - 127;
      sm = uu & 0x7fffff;

      if (ss == ds && se == de && sm == dm) {
        static INT numi[2];
        numi[1] = v;
        return ad1ili(IL_FCON, getcon(numi, DT_FLOAT));
      }
    }
  }
  return 0;
}

/** \brief Check if the expression is shifting one left; if so, return its the
 * shift count.
 */
static int
_lshift_one(int ili)
{
  int op1;

  switch (ILI_OPC(ili)) {
  case IL_LSHIFT:
  case IL_ULSHIFT:
    op1 = ILI_OPND(ili, 1);
    if (ILI_OPC(op1) != IL_ICON)
      return 0;
    break;
  case IL_KLSHIFT:
    op1 = ILI_OPND(ili, 1);
    if (ILI_OPC(op1) != IL_KCON)
      return 0;
    break;
  default:
    return 0;
  }

  if (get_isz_conili(op1) == 1)
    return ILI_OPND(ili, 2);

  return 0;
}

/* If a new IL_(U)ICMPZ would test the result of another comparison,
 * it would be redundant.  Returns a nonnegative ILI index if the
 * proposed IL_(U)ICMPZ is not needed, or -1 otherwise.
 */
static int
cmpz_of_cmp(int op1, CC_RELATION cmpz_relation)
{
  int relation;

  /* IL_(U)ICMPZ of IL_(U)ICMPZ can be collapsed */
  while (ILI_OPC(op1) == IL_ICMPZ || ILI_OPC(op1) == IL_UICMPZ) {
    cmpz_relation = combine_int_ccs(CC_ILI_OPND(op1, 2), cmpz_relation);
    if (cmpz_relation == 0)
      return -1;
    op1 = ILI_OPND(op1, 1);
  }

  switch (ILI_OPC(op1)) {
  default:
    break;
  case IL_ICMP:
  case IL_UICMP:
  case IL_KCMP:
  case IL_UKCMP:
  case IL_ACMP:
    relation = combine_int_ccs(CC_ILI_OPND(op1, 3), cmpz_relation);
    if (relation == 0)
      break;
    return ad3ili(ILI_OPC(op1), ILI_OPND(op1, 1), ILI_OPND(op1, 2), relation);
  case IL_ICMPZ:
  case IL_UICMPZ:
  case IL_KCMPZ:
  case IL_UKCMPZ:
  case IL_ACMPZ:
    relation = combine_int_ccs(CC_ILI_OPND(op1, 2), cmpz_relation);
    if (relation == 0)
      break;
    return ad2ili(ILI_OPC(op1), ILI_OPND(op1, 1), relation);
  case IL_FCMP:
  case IL_DCMP:
  case IL_SCMPLXCMP:
  case IL_DCMPLXCMP:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CMP:
#endif
    if (IEEE_CMP)
      relation = combine_ieee_ccs(CC_ILI_OPND(op1, 3), cmpz_relation);
    else
      relation = combine_int_ccs(CC_ILI_OPND(op1, 3), cmpz_relation);
    if (relation == 0)
      break;
    return ad3ili(ILI_OPC(op1), ILI_OPND(op1, 1), ILI_OPND(op1, 2), relation);
  case IL_FCMPZ:
  case IL_DCMPZ:
    if (IEEE_CMP)
      relation = combine_ieee_ccs(CC_ILI_OPND(op1, 2), cmpz_relation);
    else
      relation = combine_int_ccs(CC_ILI_OPND(op1, 2), cmpz_relation);
    if (relation == 0)
      break;
    return ad2ili(ILI_OPC(op1), ILI_OPND(op1, 1), relation);
  }
  return -1;
}

/* Predicate: does an ILI produce a value 0 <= x <= 1 only? */
static bool
is_zero_one(int ili)
{
  return false;
}

int
ili_subscript(int sub)
{
  if (ILI_OPC(sub) == IL_KCON) {
    int cst = ILI_OPND(sub, 1);
    if (CONVAL2G(cst) >= 0) {
      if (CONVAL1G(cst) == 0)
        sub = ad_icon(CONVAL2G(cst));
    } else if (CONVAL1G(cst) == (INT)0xffffffff)
      sub = ad_icon(CONVAL2G(cst));
  }
  return sub;
}

int
ili_isdeleted(int ili)
{
  if (ILI_OPC(ili) == GARB_COLLECTED) {
    return 1;
  }
  return 0;
}

typedef struct {
  int ili, nme, dtype;
} argtype;
static argtype *args;
static int nargs, argsize;

void
initcallargs(int count)
{
  extern void init_arg_ili(int);
  NEW(args, argtype, count);
  nargs = 0;
  init_arg_ili(count);
  argsize = count;
} /* initcallargs */

void
addcallarg(int ili, int nme, int dtype)
{
  if (nargs >= argsize)
    interr("too many arguments in addcallarg", nargs, ERR_Fatal);
  args[nargs].ili = ili;
  args[nargs].nme = nme;
  args[nargs].dtype = dtype;
  ++nargs;
} /* addcallarg */

int
gencallargs(void)
{
  extern void add_arg_ili(int, int, int);
  extern int gen_arg_ili(void);
  extern void end_arg_ili(void);
  int i;
  for (i = 0; i < nargs; ++i) {
    add_arg_ili(args[i].ili, args[i].nme, args[i].dtype);
  }
  i = gen_arg_ili();
  FREE(args);
  args = NULL;
  argsize = 0;
  end_arg_ili();
  return i;
} /* gencallargs */

/*
 * generate the return DFR ILI
 */
int
genretvalue(int ilix, ILI_OP resultopc)
{
  switch (resultopc) {
  case IL_NONE:
    break; /* no return value */
  case IL_DFRAR:
    ilix = ad2ili(resultopc, ilix, AR_RETVAL);
    break;
  case IL_DFRIR:
    ilix = ad2ili(resultopc, ilix, IR_RETVAL);
    break;
  case IL_DFRKR:
    ilix = ad2ili(resultopc, ilix, KR_RETVAL);
    break;
  case IL_DFRSP:
    ilix = ad2ili(resultopc, ilix, SP_RETVAL);
    break;
  case IL_DFRDP:
    ilix = ad2ili(resultopc, ilix, DP_RETVAL);
    break;
  case IL_DFRCS:
    ilix = ad2ili(resultopc, ilix, CS_RETVAL);
    break;
  default:
    interr("genretvalue: illegal resultopc", resultopc, ERR_Severe);
  }
  return ilix;
} /* genretvalue */

#define SIZEOF(array) (sizeof(array) / sizeof(char *))

/* stc-kind: kind zero (normal)
 * kind one, comparison condition.
 * kind two, memory size
 */
int
ilstckind(ILI_OP opc, int opnum)
{
  switch (opc) {
  case IL_ICMP:
  case IL_FCMP:
  case IL_DCMP:
  case IL_ACMP:
  case IL_ICMPZ:
  case IL_FCMPZ:
  case IL_DCMPZ:
  case IL_ACMPZ:
  case IL_ICJMP:
  case IL_FCJMP:
  case IL_DCJMP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMP:
#endif
  case IL_ACJMP:
  case IL_ICJMPZ:
  case IL_FCJMPZ:
  case IL_DCJMPZ:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMPZ:
#endif
  case IL_ACJMPZ:
#ifdef TM_LPCMP
  case IL_ICLOOP:
  case IL_FCLOOP:
  case IL_DCLOOP:
  case IL_ACLOOP:
  case IL_ICLOOPZ:
  case IL_FCLOOPZ:
  case IL_DCLOOPZ:
  case IL_ACLOOPZ:
#endif
  case IL_UICMP:
  case IL_UICMPZ:
  case IL_UICJMP:
  case IL_UICJMPZ:
  case IL_KCJMP:
  case IL_KCJMPZ:
  case IL_KCMP:
  case IL_KCMPZ:
  case IL_UKCMP:
  case IL_UKCMPZ:
  case IL_UKCJMP:
  case IL_UKCJMPZ:
  case IL_SCMPLXCMP:
  case IL_DCMPLXCMP:
  case IL_LCJMPZ:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CMP:
#endif
    return 1;
  case IL_LD:
  case IL_ST:
  case IL_LDKR:
  case IL_STKR:
  case IL_LDSP:
  case IL_STSP:
  case IL_SSTS_SCALAR:
  case IL_LDDP:
  case IL_STDP:
  case IL_LDSCMPLX:
  case IL_LDDCMPLX:
  case IL_STSCMPLX:
  case IL_STDCMPLX:
  case IL_DSTS_SCALAR:

#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128LD:
  case IL_FLOAT128ST:
#endif
    return 2;
  default:
    return 0;
  }
} /* ilstckind */

const char *
scond(int c)
{
  static const char *cond[] = {"..",  "eq",  "ne",  "lt",  "ge",  "le", "gt",
                               "neq", "nne", "nlt", "nge", "nle", "ngt"};
  static char B[15];
  if (c <= 0 || (size_t)c >= SIZEOF(cond)) {
    snprintf(B, 15, "%d", c);
    return B;
  } else {
    return cond[c];
  }
} /* scond */

/*
 * move, if necessary, from I to K register
 */
int
ikmove(int ilix)
{
  if (ILI_OPC(ilix) == IL_KIMV)
    return ILI_OPND(ilix, 1);
  if (IL_RES(ILI_OPC(ilix)) == ILIA_IR) {
    ilix = ad1ili(IL_IKMV, ilix);
  }
  return ilix;
} /* ikmove */

/*
 * move, if necessary, from I to K register
 */
int
uikmove(int ilix)
{
  if (ILI_OPC(ilix) == IL_KIMV)
    return ILI_OPND(ilix, 1);
  if (IL_RES(ILI_OPC(ilix)) == ILIA_IR) {
    ilix = ad1ili(IL_UIKMV, ilix);
  }
  return ilix;
} /* uikmove */

/*
 * move, if necessary, from K to I register
 */
int
kimove(int ilix)
{
  if (ILI_OPC(ilix) == IL_IKMV)
    return ILI_OPND(ilix, 1);
  if (IL_RES(ILI_OPC(ilix)) == ILIA_KR) {
    ilix = ad1ili(IL_KIMV, ilix);
  }
  return ilix;
} /* kimove */

/*
 *  Condition code operations
 */
CC_RELATION
complement_int_cc(CC_RELATION cc)
{
  switch (cc) {
  case CC_EQ:
    return CC_NE;
  case CC_NE:
    return CC_EQ;
  case CC_LT:
    return CC_GE;
  case CC_GE:
    return CC_LT;
  case CC_LE:
    return CC_GT;
  case CC_GT:
    return CC_LE;
  default:
    interr("bad cc", cc, ERR_Severe);
    return CC_None;
  }
}

CC_RELATION
complement_ieee_cc(CC_RELATION cc)
{
  if (IEEE_CMP) {
    switch (cc) {
    case CC_EQ:
      return CC_NE;
    case CC_NE:
      return CC_EQ;
    case CC_LT:
      return CC_NOTLT;
    case CC_GE:
      return CC_NOTGE;
    case CC_LE:
      return CC_NOTLE;
    case CC_GT:
      return CC_NOTGT;
    case CC_NOTEQ:
      return CC_EQ;
    case CC_NOTNE:
      return CC_NE;
    case CC_NOTLT:
      return CC_LT;
    case CC_NOTGE:
      return CC_GE;
    case CC_NOTLE:
      return CC_LE;
    case CC_NOTGT:
      return CC_GT;
    default:
      interr("bad cc", cc, ERR_Severe);
      return CC_None;
    }
  }
  return complement_int_cc(cc);
}

CC_RELATION
commute_cc(CC_RELATION cc)
{
  switch (cc) {
  case CC_EQ:
    return CC_EQ;
  case CC_NE:
    return CC_NE;
  case CC_LT:
    return CC_GT;
  case CC_GE:
    return CC_LE;
  case CC_LE:
    return CC_GE;
  case CC_GT:
    return CC_LT;
  case CC_NOTEQ:
    return CC_NOTEQ;
  case CC_NOTNE:
    return CC_NOTNE;
  case CC_NOTLT:
    return CC_NOTGT;
  case CC_NOTGE:
    return CC_NOTLE;
  case CC_NOTLE:
    return CC_NOTGE;
  case CC_NOTGT:
    return CC_NOTLT;
  default:
    interr("bad cc", cc, ERR_Severe);
    return CC_None;
  }
}

CC_RELATION
combine_int_ccs(CC_RELATION binary_cc, CC_RELATION zero_cc)
{
  if (SCFTN_TRUE < 0) {
    zero_cc = commute_cc(zero_cc);
  }
  switch (zero_cc) {
  case CC_LT: /* {0,1} <  0: always false */
  case CC_GE: /* {0,1} >= 0: always true */
    /* don't fold, make jump unconditional or fall-through */
    return CC_None;
  case CC_NE:
  case CC_GT:
    switch (binary_cc) {
    case CC_EQ:
    case CC_NE:
    case CC_LT:
    case CC_GE:
    case CC_LE:
    case CC_GT:
      break;
    default:
      interr("bad binary_cc", binary_cc, ERR_Severe);
    }
    return binary_cc;
  case CC_EQ:
  case CC_LE:
    return complement_int_cc(binary_cc);
  default:
    interr("bad zero_cc", zero_cc, ERR_Severe);
    return CC_None;
  }
}

CC_RELATION
combine_ieee_ccs(CC_RELATION binary_cc, CC_RELATION zero_cc)
{
  if (SCFTN_TRUE < 0)
    zero_cc = commute_cc(zero_cc);
  switch (zero_cc) {
  case CC_LT: /* {0,1} <  0: always false */
  case CC_GE: /* {0,1} >= 0: always true */
    /* don't fold, make jump unconditional or fall-through */
    return CC_None;
  case CC_NE:
  case CC_GT:
    switch (binary_cc) {
    case CC_EQ:
    case CC_NE:
    case CC_LT:
    case CC_GE:
    case CC_LE:
    case CC_GT:
    case CC_NOTEQ:
    case CC_NOTNE:
    case CC_NOTLT:
    case CC_NOTGE:
    case CC_NOTLE:
    case CC_NOTGT:
      break;
    default:
      interr("bad binary_cc", binary_cc, ERR_Severe);
    }
    return binary_cc;
  case CC_EQ:
  case CC_LE:
    return complement_ieee_cc(binary_cc);
  default:
    interr("bad zero_cc", zero_cc, ERR_Severe);
    return CC_None;
  }
}

bool
cc_includes_equality(CC_RELATION cc)
{
  switch (cc) {
  case CC_EQ:
  case CC_GE:
  case CC_LE:
    return true;
  case CC_NE:
  case CC_LT:
  case CC_GT:
    return false;
  case CC_NOTEQ:
  case CC_NOTGE:
  case CC_NOTLE:
    return false;
  case CC_NOTNE:
  case CC_NOTLT:
  case CC_NOTGT:
    return false;
  default:
    interr("bad cc", cc, ERR_Severe);
    return false;
  }
}

bool
ccs_are_complementary(CC_RELATION cc1, CC_RELATION cc2)
{
  switch (cc1) {
  case CC_EQ:
    return cc2 == CC_NE || cc2 == CC_NOTEQ;
  case CC_NE:
    return cc2 == CC_EQ || cc2 == CC_NOTNE;
  case CC_LT:
    return cc2 == CC_GE || cc2 == CC_NOTLT;
  case CC_GE:
    return cc2 == CC_LT || cc2 == CC_NOTGE;
  case CC_LE:
    return cc2 == CC_GT || cc2 == CC_NOTLE;
  case CC_GT:
    return cc2 == CC_LE || cc2 == CC_NOTGT;
  case CC_NOTEQ:
    return cc2 == CC_EQ;
  case CC_NOTNE:
    return cc2 == CC_NE;
  case CC_NOTLT:
    return cc2 == CC_LT;
  case CC_NOTGE:
    return cc2 == CC_GE;
  case CC_NOTLE:
    return cc2 == CC_LE;
  case CC_NOTGT:
    return cc2 == CC_GT;
  default:
    return false;
  }
}

/* TODO: replace ll_ad_outlined_func with ad_func everywhere */
int
ll_ad_outlined_func(ILI_OP result_opc, ILI_OP call_opc, char *func_name,
                    int narg, int arg1, int arg2, int arg3)
{
  return ad_func(result_opc, call_opc, func_name, narg, arg1, arg1, arg3);
}

static bool
_is_nanf(int sptr)
{
  int v, e, m;
  /*
   *  our fp cannoical form (big endian IEEE):
   *  struct {
   *      unsigned int s:1;
   *      unsigned int e:8;
   *      unsigned int m:23;
   *  };
   * A NaN has an exponent field of all one's and a non-zero mantissa.
   */
  v = CONVAL2G(sptr);
  e = (v >> 23) & 0xff;
  if (e == 0xff) {
    m = v & 0x7fffff;
    if (m)
      return true;
  }
  return false;
}

/* For an given ILI that calls a function, or stores a value,
   determine if the call can throw an exception, or if the store
   stores the result of call that can throw an exception.
   If so, determine to where control goes if an exception is thrown.

   The possible return values are:
      0: never throws
      -1: might throw, but there are no associated cleanup actions in the
   caller.
      positive integer: index of a label symbol. */
int
ili_throw_label(int ilix)
{
  ILI_OP opc = ILI_OPC(ilix);
  if (IL_TYPE(opc) == ILTY_STORE) {
    /* See if it's a store of a function result. */
    ilix = ILI_OPND(ilix, 1);
    if (!is_dfrili_opcode(ILI_OPC(ilix)))
      return 0;
    ilix = ILI_OPND(ilix, 1);
    if (IL_TYPE(ILI_OPC(ilix)) != ILTY_PROC)
      return 0;
  }
  opc = ILI_OPC(ilix);
  DEBUG_ASSERT(IL_TYPE(opc) == ILTY_PROC,
               "ili_throw_label: not a store or proc");
  /* Look at the function call.  Extract the "alt" if not a GJSR/GJSRA */
  switch (opc) {
  default:
    interr("ili_throw_label: not a call", ILI_OPC(ilix), ERR_Fatal);
    return 0;
  case IL_QJSR:
    /* QJSR never throws */
    return 0;
  case IL_JSR:
  case IL_JSRA:
    ilix = ILI_ALT(ilix);
    if (!ilix)
      /* Call must have "alt" if it can throw */
      return 0;
    break;
  case IL_GJSR:
  case IL_GJSRA:
    break;
  }
  /* Now looking at the GJSR/GJSRA call, which will have the throw label. */
  opc = ILI_OPC(ilix);
  switch (opc) {
  default:
    interr("ili_throw_label: unexpected alt", opc, ERR_Fatal);
    return 0;
  case IL_GJSR:
    return ILI_OPND(ilix, 3);
  case IL_GJSRA:
    return ILI_OPND(ilix, 5);
  }
}

static char
dt_to_mthtype(char mtype)
{
  switch (mtype) {
  default:
    break;
  case DT_FLOAT:
    return 's';
  case DT_DBLE:
    return 'd';
  case DT_CMPLX:
    return 'c';
  case DT_DCMPLX:
    return 'z';
#ifdef TARGET_SUPPORTS_QUADFP
  case DT_QUAD:
    return 'q';
#endif
  }
  interr("iliutil.c:dt_to_mthtype, unexpected mtype", mtype, ERR_Severe);
  return '?';
}

/**
   \brief LLVM wrapper for make_math_name()
   \param buff (output), must be preallocated, should have a size > 32
 */
void
llmk_math_name(char *buff, int fn, int vectlen, bool mask, DTYPE res_dt)
{
  DEBUG_ASSERT(buff, "buffer must not be null");
  buff[0] = '@';
  strcpy(buff + 1, make_math_name((MTH_FN)fn, vectlen, mask, res_dt));
}

static bool override_abi = false;

char *
make_math_name(MTH_FN fn, int vectlen, bool mask, DTYPE res_dt)
{
  static char name[32]; /* return buffer */
  static const char *fn2str[] = {"acos", "asin",   "atan",  "atan2", "cos",
                                 "cosh", "div",    "exp",   "log",   "log10",
                                 "pow",  "powi",   "powk",  "powi1", "powk1",
                                 "sin",  "sincos", "sinh",  "sqrt",  "tan",
                                 "tanh", "mod",    "floor", "ceil",  "aint"};
  const char *fstr;
  char ftype = 'f';
  if (flg.ieee)
    ftype = 'p';
  else if (XBIT_NEW_RELAXEDMATH)
    ftype = 'r';
  fstr = "__%c%c_%s_%d%s";
  if (vectlen == 1 && (override_abi || XBIT_VECTORABI_FOR_SCALAR))
    /* use vector ABI for scalar routines */
    fstr = "__%c%c_%s_%dv%s";
  sprintf(name, fstr, ftype, dt_to_mthtype(res_dt), fn2str[fn], vectlen,
          mask ? "m" : "");
  return name;
}

char *
make_math_name_vabi(MTH_FN fn, int vectlen, bool mask, DTYPE res_dt)
{
  char *name;
  /*
   * Need an override for llvect since it may emit its own calls to the
   * scalar routines rather than emitting the corresponding scalar ILI.
   * For now, restrict the override to double complex!!!
   */
  ;
  if (res_dt == DT_DCMPLX)
    override_abi = true;
  name = make_math_name(fn, vectlen, mask, res_dt);
  override_abi = false;
  return name;
}

char *
make_math(MTH_FN fn, SPTR *fptr, int vectlen, bool mask, DTYPE res_dt,
          int nargs, int arg1_dt, ...)
{
  va_list vargs;
  char *fname;
  SPTR func;

  /* Note: standard says it is undefined behavior to pass an enum to va_start */
  va_start(vargs, arg1_dt);
  fname = make_math_name(fn, vectlen, mask, res_dt);

  // NB: we must pass argX_dt as an int, because passing an enum (DTYPE) via
  // varargs is undefined behavior
  if (nargs == 1) {
    func = mk_prototype(fname, "f pure", res_dt, 1, arg1_dt);
  } else {
    const int arg2_dt = va_arg(vargs, int);
    func = mk_prototype(fname, "f pure", res_dt, 2, arg1_dt, arg2_dt);
  }
  if (fptr)
    *fptr = func;
  va_end(vargs);
  return fname;
}

static int
atomic_encode_aux(MSZ msz, SYNC_SCOPE scope, ATOMIC_ORIGIN origin,
                  ATOMIC_RMW_OP op)
{
  union ATOMIC_ENCODER u;
  DEBUG_ASSERT(sizeof(u.info) <= sizeof(int),
               "need to reimplement atomic_encode");
  DEBUG_ASSERT((unsigned)origin <= (unsigned)AORG_MAX_DEF,
               "atomic_encode_ld_st: bad origin");
  DEBUG_ASSERT(scope == SS_SINGLETHREAD || scope == SS_PROCESS,
               "atomic_encode_ld_st: bad scope");
  u.encoding = 0;
  u.info.msz = msz;
  u.info.scope = scope;
  u.info.origin = origin;
  u.info.op = op;
  DEBUG_ASSERT((u.encoding & 0xFF) == msz,
               "ILI_MSZ_OF_LD and ILI_MSZ_OF_ST won't work");
  return u.encoding;
}

/** Encode the given information into an int that can be used
    as the operand for a ATOMICLDx, ATOMICSTx, or CMPXCHGx. */
int
atomic_encode(MSZ msz, SYNC_SCOPE scope, ATOMIC_ORIGIN origin)
{
  return atomic_encode_aux(msz, scope, origin, AOP_UNDEF);
}

/** Encode atomic info for an ATOMICRMW instruction. */
int
atomic_encode_rmw(MSZ msz, SYNC_SCOPE scope, ATOMIC_ORIGIN origin,
                  ATOMIC_RMW_OP op)
{
  DEBUG_ASSERT((unsigned)op <= (unsigned)AOP_MAX_DEF,
               "atomic_encode_ld_st: bad origin");
  return atomic_encode_aux(msz, scope, origin, op);
}

/* Routines atomic_decode and atomic_info_index are provided for sake of LILI
   clients.  ILI clients should use the more abstract routine atomic_info. */

/** Decode ATOMIC_INFO from an int that was created with atomic_encode
    or atomic_encode_rmw.  */
ATOMIC_INFO
atomic_decode(int encoding)
{
  ATOMIC_INFO result;
  union ATOMIC_ENCODER u;
  u.encoding = encoding;
  result.msz = (MSZ) u.info.msz;
  result.op = (ATOMIC_RMW_OP) u.info.op;
  result.origin = (ATOMIC_ORIGIN) u.info.origin;
  result.scope = (SYNC_SCOPE) u.info.scope;
  return result;
}

/** Get index of ATOMIC_INFO operand for a given ILI instruction. */
int
atomic_info_index(ILI_OP opc)
{
  /* Get index of operand that encodes the ATOMIC_INFO. */
  switch (opc) {
  default:
    assert(false, "atomic_info: not an atomic op", opc, ERR_Severe);
    return 0;
  case IL_CMPXCHGI:
  case IL_CMPXCHGKR:
  case IL_CMPXCHGA:
  case IL_ATOMICRMWI:
  case IL_ATOMICRMWKR:
  case IL_ATOMICRMWA:
  case IL_ATOMICRMWSP:
  case IL_ATOMICRMWDP:
  case IL_ATOMICSTI:
  case IL_ATOMICSTKR:
  case IL_ATOMICSTA:
  case IL_ATOMICSTSP:
  case IL_ATOMICSTDP:
    return 4;
  case IL_ATOMICLDI:
  case IL_ATOMICLDKR:
  case IL_ATOMICLDA:
  case IL_ATOMICLDSP:
  case IL_ATOMICLDDP:
    return 3;
  case IL_FENCE:
    return 1;
  }
}

/** Get ATOMIC_INFO associated with a given ILI instruction */
ATOMIC_INFO
atomic_info(int ilix)
{
  return atomic_decode(ILI_OPND(ilix, atomic_info_index(ILI_OPC(ilix))));
}

/** Return value of irlnk operand if its value is known, otherwise return
 * default_value. */
static INT
value_of_irlnk_operand(int ilix, int default_value)
{
  ILI_OP opc = ILI_OPC(ilix);
  DEBUG_ASSERT(IL_RES(opc) == ILIA_IR, "irlnk expected");
  if (opc == IL_ICON) {
    INT value = CONVAL2G(ILI_OPND(ilix, 1));
    return value;
  }
  return default_value;
}

/** Given an ILI expression for a memory order, return a MEMORY_ORDER
    at least as strong as what the expression specifies. */
static MEMORY_ORDER
memory_order_from_operand(int ilix)
{
  INT k = value_of_irlnk_operand(ilix, MO_SEQ_CST);
  if ((unsigned)k <= MO_MAX_DEF)
    return (MEMORY_ORDER)k;
  /* Behavior is undefined.  Be conservative and return strongest ordering. */
  return MO_SEQ_CST;
}

/**
   \brief add ili for CMPXCHGx

   Actually two ilis are created, one for the CMPXCHGx and a CMPXCHG_DST to hold
   the operands that won't fit.

   \param opc an IL_CMPXCHGx opcode
   \param ilix_val ILI expression for value to be stored if operation succeeds
   \param ilix_loc ILI address expression for location to be updated.
   \param nme NME for destination.
   \param stc_atomic_info information packed by routine atomic_encode.
   \param ilix_comparand value to compare against
   \param ilix_is_weak if 1 then operation is allowed to spuriously fail
   occasionally.
                       Should be 0 otherwise.
   \param ilix_success ILI expression for memory order on success
   \param ilix_failure ILI expression for memory order on failure

 */
int
ad_cmpxchg(ILI_OP opc, int ilix_val, int ilix_loc, int nme, int stc_atomic_info,
           int ilix_comparand, int ilix_is_weak, int ilix_success,
           int ilix_failure)
{
  int dst;
  DEBUG_ASSERT(IL_IS_CMPXCHG(opc), "ad_cmpxchg: opc must be a CMPXCHGx");
  dst = ad4ili(IL_CMPXCHG_DST, ilix_loc, ilix_is_weak, ilix_success,
               ilix_failure);
  return ad5ili(opc, ilix_val, dst, nme, stc_atomic_info, ilix_comparand);
}

/** Get the IL_CMPXCHG_DST instruction underlying a IL_CMPXCHGx instruction. */
static int
get_dst_of_cmpxchg(int ilix)
{
  int iliy;
  DEBUG_ASSERT(IL_IS_CMPXCHG(ILI_OPC(ilix)),
               "get_dst_of_cmpxchg: not a CMPXCHGx");
  iliy = ILI_OPND(ilix, 2);
  DEBUG_ASSERT(ILI_OPC(iliy) == IL_CMPXCHG_DST,
               "get_dst_of_cmpxchg: IL_CMPXCHG_DST expected");
  return iliy;
}

/** For a IL_CMPXCHGx operation, return true if it is allowed to
    spuriously fail occcasionally.  Returns false when in doubt. */
bool
cmpxchg_is_weak(int ilix)
{
  int iliy = get_dst_of_cmpxchg(ilix);
  return value_of_irlnk_operand(ILI_OPND(iliy, 2), 0) == 0;
}

/** For an IL_CMPXCHGx instruction, return address expression of location. */
int
cmpxchg_loc(int ilix)
{
  int iliy = get_dst_of_cmpxchg(ilix);
  return ILI_OPND(iliy, 1);
}

/* clang-format off */

/** \brief Table used by cmpxchg_memory_order. 

    Indexed by [succ][fail], it returns an weakened value for failure  
    per rules in C++11 standard and LLVM.  The weakenings guarantee that
    the failure value is no stronger than the succ value and has
    no "release" aspect to it.
 
    Element type is char to save space.
    
    this table is also used by accelerator CG, so DON'T make it as static
*/
char memory_order_fail_table[MO_MAX_DEF + 1][MO_MAX_DEF + 1] = {
    /* succ==relaxed */
    {MO_RELAXED, MO_RELAXED, MO_RELAXED, MO_RELAXED, MO_RELAXED, MO_RELAXED},
    /* succ==consume */
    {MO_RELAXED, MO_CONSUME, MO_CONSUME, MO_RELAXED, MO_CONSUME, MO_CONSUME},
    /* succ==acquire */
    {MO_RELAXED, MO_CONSUME, MO_ACQUIRE, MO_RELAXED, MO_ACQUIRE, MO_ACQUIRE},
    /* succ==release */
    {MO_RELAXED, MO_RELAXED, MO_RELAXED, MO_RELAXED, MO_RELAXED, MO_RELAXED},
    /* succ==acq_rel */
    {MO_RELAXED, MO_CONSUME, MO_ACQUIRE, MO_RELAXED, MO_ACQUIRE, MO_ACQUIRE},
    /* succ==seq_cst */
    {MO_RELAXED, MO_CONSUME, MO_ACQUIRE, MO_RELAXED, MO_ACQUIRE, MO_SEQ_CST},
};

/* clang-format on */

/** For an IL_CMPXCHGx instruction, return memory orders for
    success and failure cases. */
CMPXCHG_MEMORY_ORDER
cmpxchg_memory_order(int ilix)
{
  CMPXCHG_MEMORY_ORDER result;
  MEMORY_ORDER succ, fail;
  int dst = get_dst_of_cmpxchg(ilix);
  succ = memory_order_from_operand(ILI_OPND(dst, 3));
  fail = memory_order_from_operand(ILI_OPND(dst, 4));
  DEBUG_ASSERT((unsigned)succ <= (unsigned)MO_MAX_DEF,
               "max_memory_order: bad mo1");
  DEBUG_ASSERT((unsigned)fail <= (unsigned)MO_MAX_DEF,
               "max_memory_order: bad mo2");
  result.success = succ;
  result.failure = (MEMORY_ORDER)memory_order_fail_table[succ][fail];
  return result;
}

/** Get MEMORY_ORDER associated with given ILI instruction.
    If the memory order operand is not a constant, require
    a conservative bound.  If there are different memory
    orders for success/failure, return an upper bound.

    For IL_CMPXCHGx instructions, the memory order is always
    the "success" order, since the "failure" order is not allowed
    to be stronger. */
MEMORY_ORDER
memory_order(int ilix)
{
  int i;
  ILI_OP opc = ILI_OPC(ilix);
  DEBUG_ASSERT(IL_HAS_FENCE(opc), "opc missing fence attribute");
  switch (opc) {
  default:
    assert(false, "memory_order: unimplemented op", opc, ERR_Severe);
    return MO_UNDEF;
  case IL_CMPXCHGI:
  case IL_CMPXCHGKR:
  case IL_CMPXCHGA: {
    /** The "failure" order cannot be stronger than the "success" order,
        so return the success order. */
    int dst = get_dst_of_cmpxchg(ilix);
    return memory_order_from_operand(ILI_OPND(dst, 3));
  }
  case IL_ATOMICRMWI:
  case IL_ATOMICRMWKR:
  case IL_ATOMICRMWA:
  case IL_ATOMICRMWSP:
  case IL_ATOMICRMWDP:
  case IL_ATOMICSTI:
  case IL_ATOMICSTKR:
  case IL_ATOMICSTA:
  case IL_ATOMICSTSP:
  case IL_ATOMICSTDP:
    i = 5;
    break;
  case IL_ATOMICLDI:
  case IL_ATOMICLDKR:
  case IL_ATOMICLDA:
  case IL_ATOMICLDSP:
  case IL_ATOMICLDDP:
    i = 4;
    break;
  case IL_FENCE:
    i = 2;
    break;
  }
  return memory_order_from_operand(ILI_OPND(ilix, i));
}

bool
is_omp_atomic_ld(int ilix)
{
  ATOMIC_INFO info;
  switch (ILI_OPC(ilix)) {
  case IL_ATOMICLDI:
  case IL_ATOMICLDKR:
  case IL_ATOMICLDA:
  case IL_ATOMICLDSP:
  case IL_ATOMICLDDP:
    break;
  default:
    return false;
  }
  info = atomic_info(ilix);

  if (info.origin == AORG_OPENMP)
    return true;
  return false;
}

bool
is_omp_atomic_st(int ilix)
{
  ATOMIC_INFO info;
  switch (ILI_OPC(ilix)) {
  case IL_ATOMICSTI:
  case IL_ATOMICSTKR:
  case IL_ATOMICSTA:
  case IL_ATOMICSTSP:
  case IL_ATOMICSTDP:
    break;
  default:
    return false;
  }
  info = atomic_info(ilix);

  if (info.origin == AORG_OPENMP)
    return true;
  return false;
}

/*
 * a set of functions to create integer expressions,
 * that automatically detect long (KR) operands and adjust appropriately
 * if an ilix index is zero, treat it as a missing operand
 */

static int
ivconst(ISZ_T valconst)
{
  ISZ_T valbig, valhigh;
  int ilix;
  valbig = 0xffffffff00000000LL;
  valhigh = valbig & valconst; /* high order bits */
  if (valhigh == 0 || valhigh == valbig) {
    ilix = ad_icon(valconst);
  } else {
    ilix = ad_kconi(valconst);
  }
  return ilix;
} /* ivconst */

/*
 * imul_const_ili(con, ilix), detect when con == 0 or con == 1
 *  return 0 if the value would be the constant zero
 *  if ilix==0, return the constant, either as a 4- or 8-byte int
 */
int
imul_const_ili(ISZ_T valconst, int valilix)
{
  int ilix;
  if (valconst == 0)
    return 0;
  if (valconst == 1)
    return valilix;
  if (valilix == 0) {
    ilix = ivconst(valconst);
  } else if (IL_RES(ILI_OPC(valilix)) == ILIA_KR) {
    ilix = ad_kconi(valconst);
    ilix = ad2ili(IL_KMUL, ilix, valilix);
  } else {
    ilix = ad_icon(valconst);
    ilix = ad2ili(IL_IMUL, ilix, valilix);
  }
  return ilix;
} /* imul_const_ili */

int
imul_ili_ili(int leftx, int rightx)
{
  int ilix;
  if (IL_RES(ILI_OPC(leftx)) == ILIA_KR || IL_RES(ILI_OPC(rightx)) == ILIA_KR) {
    ilix = ad2ili(IL_KMUL, ikmove(leftx), ikmove(rightx));
  } else {
    ilix = ad2ili(IL_IMUL, leftx, rightx);
  }
  return ilix;
} /* imul_ili_ili */

/*
 * if valilix==0, return valconst
 */
int
iadd_const_ili(ISZ_T valconst, int valilix)
{
  int ilix;
  if (valconst == 0)
    return valilix;
  if (valilix == 0) {
    ilix = ivconst(valconst);
  }
  if (IL_RES(ILI_OPC(valilix)) == ILIA_KR) {
    if (!valilix) {
      ilix = ad_kconi(valconst);
    } else if (valconst > 0) {
      ilix = ad_kconi(valconst);
      ilix = ad2ili(IL_KADD, ilix, valilix);
    } else {
      ilix = ad_kconi(-valconst);
      ilix = ad2ili(IL_KSUB, valilix, ilix);
    }
  } else {
    if (!valilix) {
      ilix = ad_icon(valconst);
    } else if (valconst > 0) {
      ilix = ad_icon(valconst);
      ilix = ad2ili(IL_IADD, ilix, valilix);
    } else {
      /* -c + i ==> i-c */
      ilix = ad_icon(-valconst);
      ilix = ad2ili(IL_ISUB, valilix, ilix);
    }
  }
  return ilix;
} /* iadd_const_ili */

int
iadd_ili_ili(int leftx, int rightx)
{
  int ilix;
  if (leftx < 0 || rightx < 0)
    interr("iadd_ili_ili argument error", 0, ERR_Fatal);
  if (leftx == 0)
    return rightx;
  if (rightx == 0)
    return leftx;
  if (IL_RES(ILI_OPC(leftx)) == IL_RES(ILI_OPC(rightx))) {
    /* both KR or both IR */
    /* look for trivial simplifications */
    if (ILI_OPC(rightx) == IL_KNEG || ILI_OPC(rightx) == IL_INEG) {
      /* a + (-b) */
      return isub_ili_ili(leftx, ILI_OPND(rightx, 1));
    }
    if (ILI_OPC(leftx) == IL_KNEG || ILI_OPC(leftx) == IL_INEG) {
      /* (-a) + b */
      return isub_ili_ili(rightx, ILI_OPND(leftx, 1));
    }
    if (ILI_OPC(leftx) == IL_KSUB || ILI_OPC(leftx) == IL_ISUB ||
        ILI_OPC(leftx) == IL_UKSUB || ILI_OPC(leftx) == IL_UISUB) {
      if (ILI_OPND(leftx, 2) == rightx) {
        /* (a-b) + b */
        return ILI_OPND(leftx, 1);
      }
    }
    if (ILI_OPC(rightx) == IL_KSUB || ILI_OPC(rightx) == IL_ISUB ||
        ILI_OPC(rightx) == IL_UKSUB || ILI_OPC(rightx) == IL_UISUB) {
      if (ILI_OPND(rightx, 2) == rightx) {
        /* b + (a-b) */
        return ILI_OPND(rightx, 1);
      }
    }
    if (IL_RES(ILI_OPC(leftx)) == ILIA_KR) {
      ilix = ad2ili(IL_KADD, leftx, rightx);
    } else {
      ilix = ad2ili(IL_IADD, leftx, rightx);
    }
  } else {
    /* either one KR, move to KR */
    ilix = ad2ili(IL_KADD, ikmove(leftx), ikmove(rightx));
  }
  return ilix;
} /* iadd_ili_ili */

int
isub_ili_ili(int leftx, int rightx)
{
  int ilix;
  if (leftx < 0 || rightx < 0)
    interr("isub_ili_ili argument error", 0, ERR_Fatal);
  if (rightx == 0)
    return leftx;
  if (leftx == 0) {
    /* negate */
    if (ILI_OPC(rightx) == IL_KNEG || ILI_OPC(rightx) == IL_INEG)
      return ILI_OPND(rightx, 1);
    if (IL_RES(ILI_OPC(rightx)) == ILIA_KR) {
      return ad1ili(IL_KNEG, rightx);
    } else {
      return ad1ili(IL_INEG, rightx);
    }
  }
  if (leftx == rightx)
    /* a - b */
    return 0;
  if (IL_RES(ILI_OPC(leftx)) == IL_RES(ILI_OPC(rightx))) {
    /* look for trivial simplifications */
    if (ILI_OPC(rightx) == IL_KNEG || ILI_OPC(rightx) == IL_INEG) {
      /* a - (-b) */
      return iadd_ili_ili(leftx, ILI_OPND(rightx, 1));
    }
    if (ILI_OPC(leftx) == IL_KADD || ILI_OPC(leftx) == IL_IADD ||
        ILI_OPC(leftx) == IL_UKADD || ILI_OPC(leftx) == IL_UIADD) {
      if (ILI_OPND(leftx, 2) == rightx) {
        /* (a+b) - b */
        return ILI_OPND(leftx, 1);
      }
      if (ILI_OPND(leftx, 1) == rightx) {
        /* (a+b) - a */
        return ILI_OPND(leftx, 2);
      }
    }
    if (ILI_OPC(leftx) == IL_KSUB || ILI_OPC(leftx) == IL_ISUB ||
        ILI_OPC(leftx) == IL_UKSUB || ILI_OPC(leftx) == IL_UISUB) {
      if (ILI_OPND(leftx, 1) == rightx) {
        /* (a-b) - a */
        return isub_ili_ili(0, ILI_OPND(leftx, 2));
      }
    }
    if (ILI_OPC(rightx) == IL_KADD || ILI_OPC(rightx) == IL_IADD ||
        ILI_OPC(rightx) == IL_UKADD || ILI_OPC(rightx) == IL_UIADD) {
      if (ILI_OPND(rightx, 1) == leftx) {
        /* a - (a+b) */
        return isub_ili_ili(0, ILI_OPND(rightx, 2));
      }
      if (ILI_OPND(rightx, 2) == leftx) {
        /* a - (b+a) */
        return isub_ili_ili(0, ILI_OPND(rightx, 1));
      }
    }
    if (ILI_OPC(rightx) == IL_KSUB || ILI_OPC(rightx) == IL_ISUB ||
        ILI_OPC(rightx) == IL_UKSUB || ILI_OPC(rightx) == IL_UISUB) {
      if (ILI_OPND(rightx, 1) == leftx) {
        /* a - (a-b) */
        return ILI_OPND(rightx, 2);
      }
    }
    if (IL_RES(ILI_OPC(leftx)) == ILIA_KR) {
      ilix = ad2ili(IL_KSUB, leftx, rightx);
    } else {
      ilix = ad2ili(IL_ISUB, leftx, rightx);
    }
  } else {
    /* either one KR, move both to KR */
    ilix = ad2ili(IL_KSUB, ikmove(leftx), ikmove(rightx));
  }
  return ilix;
} /* isub_ili_ili */

/*
 * ilix / const
 */
int
idiv_ili_const(int valilix, ISZ_T valconst)
{
  int ilix;
  if (valconst == 1)
    return valilix;
  if (valilix < 0)
    interr("div_ili_const argument error", 0, ERR_Fatal);
  if (valconst == -1)
    return isub_ili_ili(0, valilix);
  if (IL_RES(ILI_OPC(valilix)) == ILIA_KR) {
    ilix = ad_kconi(valconst);
    ilix = ad2ili(IL_KDIV, valilix, ilix);
  } else {
    ilix = ad_icon(valconst);
    ilix = ad2ili(IL_IDIV, valilix, ilix);
  }
  return ilix;
} /* idiv_ili_const */

int
idiv_ili_ili(int leftx, int rightx)
{
  int ilix;
  if (leftx < 0 || rightx < 0)
    interr("div_ili_ili argument error", 0, ERR_Fatal);
  if (IL_RES(ILI_OPC(leftx)) == ILIA_KR || IL_RES(ILI_OPC(rightx)) == ILIA_KR) {
    ilix = ad2ili(IL_KDIV, ikmove(leftx), ikmove(rightx));
  } else {
    ilix = ad2ili(IL_IDIV, leftx, rightx);
  }
  return ilix;
} /* idiv_ili_ili */

int
imax_ili_ili(int leftx, int rightx)
{
  int ilix;
  if (leftx < 0 || rightx < 0)
    interr("max_ili_ili argument error", 0, ERR_Fatal);
  if (IL_RES(ILI_OPC(leftx)) == ILIA_KR || IL_RES(ILI_OPC(rightx)) == ILIA_KR) {
    ilix = ad2ili(IL_KMAX, ikmove(leftx), ikmove(rightx));
  } else {
    ilix = ad2ili(IL_IMAX, leftx, rightx);
  }
  return ilix;
} /* imax_ili_ili */

int
imin_ili_ili(int leftx, int rightx)
{
  int ilix;
  if (leftx < 0 || rightx < 0)
    interr("min_ili_ili argument error", 0, ERR_Fatal);
  if (IL_RES(ILI_OPC(leftx)) == ILIA_KR || IL_RES(ILI_OPC(rightx)) == ILIA_KR) {
    ilix = ad2ili(IL_KMIN, ikmove(leftx), ikmove(rightx));
  } else {
    ilix = ad2ili(IL_IMIN, leftx, rightx);
  }
  return ilix;
} /* imin_ili_ili */

