/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief SMP expander routines
 */

#include "expsmp.h"
#include "exputil.h"
#include "exp_rte.h"
#include "dtypeutl.h"
#include "expatomics.h"
#include "regutil.h"
#include "machreg.h"
#include "ilm.h"
#include "ilmtp.h"
#include "ili.h"
#define EXPANDER_DECLARE_INTERNAL
#include "expand.h"
#include "machar.h"
#include "ccffinfo.h"
#include "kmpcutil.h"
#include "outliner.h"
#include "mp.h"
#include "mach.h"
#include "assem.h"
#include "llutil.h"
#include "llassem.h"
#include "ll_ftn.h"
#include "llmputil.h"
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
#include "ompaccel.h"
#include "tgtutil.h"
#endif
#include "symfun.h"

#ifdef __cplusplus
inline SPTR GetPARUPLEVEL(SPTR sptr) {
  return static_cast<SPTR>(PARUPLEVELG(sptr));
}
#undef PARUPLEVELG
#define PARUPLEVELG GetPARUPLEVEL
#endif

static int incrOutlinedCnt(void);
static int decrOutlinedCnt(void);
#ifdef FLANG2_EXPSMP_UNUSED
static int getOutlinedTemp(char *, int);
#endif
static int isUnnamedCs(int);
static int addMpUnp(void);
static int addMpUnv(void);
static int addMpBcsNest(void);
static int addMpEcsNest(void);
static int allocThreadprivate(SPTR sym, int *tmpthr);

#define mk_prototype mk_prototype_llvm

static int availIreg; /* next available integer register for jsr */
static int availFreg; /* next available floating point register for jsr */
static int maxIreg;   /* max # of integer registers used by jsr */
static int maxFreg;   /* max # of floating point registers used by jsr */

static int outlinedCnt; /* counter to record of outlined function */
static int parCnt;      /* counter to record parallel regions */
static int parsectCnt;  /* counter to record parallel sections */
static int critCnt;     /* counter for critical sections */
static int taskbih;     /* mark where task allocation should be */
static int taskCnt;     /* counter for task regions  */
static int taskLoopCnt; /* counter for taskloop regions  */
static int taskBv;      /* bit values for flag for BTASK & TASKREG:
                         *   0x01 -- untied
                         *   0x02 -- if clause present
                         *   0x04 -- orphaned (dynamic, not lexically,
                         parallel)
                         *   0x08 -- nested task
                         *   0x10 -- forced defer (CUDA)
                         *   0x20 -- final task
                         *   0x40 -- execute immediately
                         */
static int taskdup;
static int taskIfv;        /* value of if clause for BTASK & TASKREG */
static SPTR taskFlags;     ///< value of final clause for BTASK & TASKREG
static SPTR taskFnsptr;    ///< store task func sptr
static SPTR taskAllocSptr; ///< store the return value from kmpc_alloc
static int maxOutlinedCnt; /* maximum parCnt for a function */
static int sumOutlinedCnt; /* sum of parCnts of functions already
                            * processed.  'sumOutlinedCnt+parCnt' can be
                            * the suffix of the name of a temp created for
                            * a outlined region within a function so that
                            * the temp is:
                            * 1) unique across functions,
                            * 2) reused across parallel regions within
                            *    a function.
                            */
static SPTR scopeSptr;
static int *mppgbih;
static int mppgcnt;
static int mppgBihSiz;

static struct {
  int lb_ili;
  int ub_ili;
  int st_ili;
  int lastitr;
  int flags;
  INT offset;
  int tasklpargs[10]; /* ili in order as enum tasklooparg below */
} taskLpInfo;

enum taskloooparg {
  TASKLPARG_TASK = 0,
  TASKLPARG_IF_VAL,
  TASKLPARG_LB,
  TASKLPARG_UB,
  TASKLPARG_ST,
  TASKLPARG_NOGROUP,
  TASKLPARG_SCHED,
  TASKLPARG_GRAINSIZE,
  TASKLPARG_TASKDUP,
  TASKLPARG_MAX // must be last
};

#define TASK_LB taskLpInfo.lb_ili
#define TASK_LPVAR_OFFSET taskLpInfo.offset
#define TASK_UB taskLpInfo.ub_ili
#define TASK_ST taskLpInfo.st_ili
#define TASK_LASTITR taskLpInfo.lastitr

/* arguments to __kmpc_taskloop excepts ident and gtid */
#define TASKLPARGS taskLpInfo.tasklpargs
#define TASKLP_TASK taskLpInfo.tasklpargs[TASKLPARG_TASK]
#define TASKLP_IF taskLpInfo.tasklpargs[TASKLPARG_IF_VAL]
#define TASKLP_LB taskLpInfo.tasklpargs[TASKLPARG_LB]
#define TASKLP_UB taskLpInfo.tasklpargs[TASKLPARG_UB]
#define TASKLP_ST taskLpInfo.tasklpargs[TASKLPARG_ST]
#define TASKLP_NOGROUP taskLpInfo.tasklpargs[TASKLPARG_NOGROUP]
#define TASKLP_SCHED taskLpInfo.tasklpargs[TASKLPARG_SCHED]
#define TASKLP_GRAINSIZE taskLpInfo.tasklpargs[TASKLPARG_GRAINSIZE]
#define TASKLP_TASKDUP taskLpInfo.tasklpargs[TASKLPARG_TASKDUP]

typedef struct SectionsWrk_t {
  SPTR lb;  /* start at 0 */
  SPTR ub;  /* number of sections */
  SPTR st;  /* stride 1 */
  SPTR last; /* flag for last section */
  int cnt;  /* running count */
  int bbih; /* start block for sections */
} SectionsWrk_t;

static SectionsWrk_t sectionsWrk;

#define SECT_UB sectionsWrk.lb
#define SECT_LB sectionsWrk.ub
#define SECT_ST sectionsWrk.st
#define SECT_LAST sectionsWrk.last
#define SECT_CNT sectionsWrk.cnt
#define SECT_BBIH sectionsWrk.bbih

#define MP_NOT_IMPLEMENTED(_str) error(375, ERR_Fatal, 0, _str, NULL)

/**
   \brief For use with generating an array filed with copyprivate addresses.
 */
typedef struct sptrListT {
  SPTR o_sptr;
  SPTR sptr; ///< either base sptr or TPpxxx thread private common block vector
  int size_ili;
  int vec_size_ili;
  bool is_common_block;
  struct sptrListT *next;
  int cplus_assign_rou;
} sptrListT;

/* called once per function */
void
exp_smp_init(void)
{
  parCnt = 0;
  parsectCnt = 0;
  outlinedCnt = 0;
  critCnt = 0;
  expb.lcpu2 = 0;
  expb.lcpu3 = 0;
  expb.ncpus2 = 0;
  maxOutlinedCnt = 0;
  mppgBihSiz = 16;
  NEW(mppgbih, int, mppgBihSiz);
  mppgcnt = 0;
  taskCnt = 0;
  taskLoopCnt = 0;
}

void
exp_smp_fini(void)
{
  sumOutlinedCnt = +maxOutlinedCnt;
  FREE(mppgbih);
}

static SPTR
getPrivateTemp(DTYPE dtype)
{
  static int count;
  SPTR sptr = getnewccsym('s', count++, ST_VAR);
  SCP(sptr, SC_PRIVATE);
  DTYPEP(sptr, dtype);
  ENCLFUNCP(sptr, GBL_CURRFUNC);
  return sptr;
}

static void
expSmpSectionInit(void)
{
  SECT_LB = getPrivateTemp(DT_UINT);
  SECT_UB = getPrivateTemp(DT_UINT);
  SECT_LAST = getPrivateTemp(DT_UINT);
  SECT_ST = getPrivateTemp(DT_UINT);
  SECT_CNT = 0;
  if (!gbl.outlined) {
    SCP(SECT_LB, SC_AUTO);
    SCP(SECT_UB, SC_AUTO);
    SCP(SECT_LAST, SC_AUTO);
    SCP(SECT_ST, SC_AUTO);
  }
  SECT_BBIH = expb.curbih;
}

static void
expSmpSectionEnd(void)
{
  SECT_LB = getPrivateTemp(DT_UINT);
  SECT_UB = getPrivateTemp(DT_UINT);
  SECT_LAST = getPrivateTemp(DT_UINT);
  SECT_ST = getPrivateTemp(DT_UINT);
  SECT_CNT = 0;
  SECT_BBIH = 0;
}

static int
sectionCreateBlock(int nextLabel, SPTR lb, SPTR ub, int myVal)
{
  int ili, ubVal, lbVal;

  myVal = ad_icon(myVal);
  lbVal = ad3ili(IL_LD, ad_acon(lb, 0), addnme(NT_VAR, lb, 0, 0), MSZ_WORD);
  ubVal = ad3ili(IL_LD, ad_acon(ub, 0), addnme(NT_VAR, ub, 0, 0), MSZ_WORD);

  ili = ad4ili(IL_UICJMP, myVal, ubVal, CC_GT, nextLabel);
  RFCNTI(nextLabel);
  chk_block(ili);

  ili = ad4ili(IL_UICJMP, myVal, lbVal, CC_LT, nextLabel);
  RFCNTI(nextLabel);

  return ili;
}

#ifdef FLANG2_EXPSMP_UNUSED
static int
sectionCreateLastblock(int nextLabel, SPTR lastValSym, int myVal)
{
  int ili, lastVal;

  myVal = ad_icon(myVal);
  lastVal = ad3ili(IL_LD, ad_acon(lastValSym, 0),
                   addnme(NT_VAR, lastValSym, 0, 0), MSZ_WORD);

  ili = ad4ili(IL_UICJMP, myVal, lastVal, CC_EQ, nextLabel);
  RFCNTI(nextLabel);
  return ili;
}
#endif

void
section_create_endblock(SPTR endLabel)
{
  /* call kmpc_for_static_fini */
  int ili;

  wr_block();
  cr_block();
  ili = ll_make_kmpc_for_static_fini();
  exp_label(endLabel);
  iltb.callfg = 1;
  chk_block(ili);
  ili = ll_make_kmpc_barrier();
  iltb.callfg = 1;
  chk_block(ili);
  BIH_LABEL(expb.curbih) = endLabel;
  ILIBLKP(endLabel, expb.curbih);
}

/* set:       1 to set, 0 to restore
 * eampp:     if it is eampp, then subtract its value(1) from mppgcnt
 */
#define SET_MPPBIH 1
#define RESTORE_MPPBIH 0
#define IS_PREVMPPG 1
#define IS_NOTPREVMPPG 0
#define USE_NEXTBIH 1
#define NOTUSE_NEXTBIH 0

static void
resetMppBih(int set, int eampp)
{
  static int savebih;
  static int savex14;
  int bih;

  if (mppgcnt == 0)
    return;
  if (set) {
    savebih = expb.curbih;
    bih = mppgbih[mppgcnt - eampp];
    if (savebih == bih) {
      savebih = 0;
      return;
    }
    savex14 = flg.x[14];
    flg.x[14] |= 0x1000; /* don't split at calls */
    wr_block();
    expb.curbih = bih;
    rdilts(expb.curbih);
    expb.curilt = ILT_PREV(0);
  } else if (savebih) { /* only reset if we save it */
    wrilts(mppgbih[mppgcnt - eampp]);
    expb.curbih = savebih;
    rdilts(expb.curbih);
    expb.curilt = ILT_PREV(0);
    flg.x[14] = savex14;
  }
}

static void
resetTaskBih(int set)
{
  static int savebih;
  static int savex14;

  if (taskbih == 0)
    return;
  if (set) {
    savebih = expb.curbih;
    savex14 = flg.x[14];
    flg.x[14] |= 0x1000;
    wr_block();
    expb.curbih = taskbih;
    rdilts(expb.curbih);
    expb.curilt = ILT_PREV(0);
  } else {
    wrilts(taskbih);
    expb.curbih = savebih;
    rdilts(expb.curbih);
    expb.curilt = ILT_PREV(0);
    flg.x[14] = savex14;
  }
}

static void
sptrListAdd(sptrListT **list, SPTR sptr, int size_ili, bool is_cmblk,
            int cplus_assign_rou, int vec_size_ili, SPTR o_sptr)
{
  sptrListT *node = (sptrListT *)malloc(sizeof(sptrListT));

  node->o_sptr = o_sptr;
  node->sptr = sptr;
  node->next = *list;
  node->is_common_block = is_cmblk;
  node->size_ili = size_ili;
  node->vec_size_ili = vec_size_ili; /* used for COPYIN_CL of arrays */
  node->cplus_assign_rou = cplus_assign_rou;
  *list = node;
}

static void
sptrListFree(sptrListT **list)
{
  sptrListT *n = *list;
  while (n) {
    sptrListT *next = n->next;
    free(n);
    n = next;
  }
  *list = NULL;
}

/* Returns an ili of a temporary variable that conatins size information
 * The runtime for instance, _mp_copypriv_kmpc, expects size_t* for size.
 *
 * 'bytes' is the actual byte size and not an sptr or ili.
 */
static int
genSizeAcon(int size_ili)
{
  int ili;
  SPTR tmp;
  int nme;
  const DTYPE dtype = (TARGET_PTRSIZE == 8) ? DT_INT8 : DT_INT;

  tmp = getPrivateTemp(dtype);
  SCP(tmp, SC_AUTO);

  ili = ad_acon(tmp, 0);
  nme = addnme(NT_VAR, tmp, 0, 0);
  ADDRTKNP(tmp, 1);

  if (TARGET_PTRSIZE == 8) {
    ili = ad4ili(IL_STKR, size_ili, ili, nme, MSZ_I8);
  } else {
    size_ili = ad1ili(IL_KIMV, size_ili);
    ili = ad4ili(IL_ST, size_ili, ili, nme, MSZ_WORD);
  }
  chk_block(ili);

  return ad_acon(tmp, 0);
}

/* Given a sptr list, create an array of pairs:
 * (size, address) where:
 * 'size' - Pointer to a temporary variable containing the byte size of
 *          sptr. (size_t *)
 * 'address' - Address of sptr. (void *).
 *
 * These pairs are represented in an array where
 * the even indices are the size pointers and the odd indices the
 * addresses.  The sentinel/terminator is the all-zero pair.
 * [(sz0,addr0), (sz1,addr1), ... (0x0, 0x0)].
 *
 * We represent these as an array, which is more convenient to manage
 * internally.  The runtime routine _mp_copypriv_kmpc expects this format.
 *
 * Returns: The sptr of this majestic array that we so masterfully create here.
 */
static SPTR
makeCopyprivArray(const sptrListT *list, bool pass_size_addresses)
{
  int i, ili, nme, n_elts;
  SPTR array;
  DTYPE dtype;
  int basenme;
  static int id;
  const sptrListT *node;

  /* Count the number of items in the list */
  n_elts = 0;
  for (node = list; node; node = node->next)
    ++n_elts;

  /* We represent each entry as a pair for each private variable (each node in
   * sptr_list): (size, sptr)
   *
   * +2 for the last node, the sentinel (null node), which tells the
   * runtime it has reached the end of the array.  Each node is 2 array elts.
   */
  n_elts = (n_elts * 2) + 2;

  /* Create the array dtype: each element is word size */
  array = getnewccsym('a', ++id, ST_ARRAY);
  {
    ADSC *adsc;
    INT con[2] = {0, n_elts};

    dtype = get_array_dtype(1, DT_CPTR);
    adsc = AD_DPTR(dtype);
    AD_LWBD(adsc, 0) = stb.i1;
    AD_UPBD(adsc, 0) = getcon(con, DT_INT);
    AD_NUMELM(adsc) = AD_UPBD(adsc, 0);
  }

  DTYPEP(array, dtype);
  SCP(array, SC_AUTO);

  /* Build the list: (size, sptr) pairs. */
  basenme = addnme(NT_VAR, array, 0, 0);
  for (node = list, i = 0; node; node = node->next, ++i) {
    int sptr_nme, sptr_ili;

    if (node->is_common_block || THREADG(node->sptr)) {
/* This is thread private so obtain address from the TP vector */
      if (node->is_common_block)
        ref_threadprivate(node->sptr, &sptr_ili, &sptr_nme);
      else
        ref_threadprivate_var(node->sptr, &sptr_ili, &sptr_nme, 1);
    } else {
      /* Else, this is not thread private */
      sptr_nme = addnme(NT_VAR, node->sptr, 0, 0);
      sptr_ili = mk_address(node->sptr);
    }

    /* array[i] = size */
    nme = add_arrnme(NT_ARR, array, basenme, 0, ad_icon(i), false);
    if (pass_size_addresses) { /* why do I need to pass address? */
      ili = genSizeAcon(node->size_ili);
      ili = ad3ili(IL_STA, ili, ad_acon(array, i * TARGET_PTRSIZE), nme);
    } else {
      ili = ad4ili(IL_ST, node->size_ili, ad_acon(array, i * TARGET_PTRSIZE),
                   nme, TARGET_PTRSIZE == 8 ? MSZ_I8 : MSZ_WORD);
    }
    chk_block(ili);

    /* array[i+1] = local (stack based) sptr */
    ++i;
    nme = add_arrnme(NT_ARR, array, basenme, 0, ad_icon(i), false);
    ili = ad3ili(IL_STA, sptr_ili, ad_acon(array, i * TARGET_PTRSIZE), nme);
    chk_block(ili);
  }

  /* Terminate the array with a sentinel that the runtime will recognize */
  nme = add_arrnme(NT_ARR, array, basenme, 0, ad_icon(i), false);
  ili = ad3ili(IL_STA, ad_aconi(0), ad_acon(array, i * TARGET_PTRSIZE), nme);
  chk_block(ili);

  ++i;
  nme = add_arrnme(NT_ARR, array, basenme, 0, ad_icon(i), false);
  ili = ad3ili(IL_STA, ad_aconi(0), ad_acon(array, i * TARGET_PTRSIZE), nme);
  chk_block(ili);

  return array;
}

static int
mkMemcpy(void)
{
  int func;
  func = mk_prototype("memcpy", NULL, DT_CPTR, 3, DT_CPTR, DT_CPTR, DT_UINT8);
  SCP(func, SC_EXTERN);
  func = mkfunc("memcpy");
  return func;
}

/**
   \brief ...
   \param arglist
   \param opc     IL_DAIR/IL_DAAR/IL_DADP/IL_DASP/IL_ARGxx (x86)
   \param argili

   Add argument expression \p argili to existing argument list \p arglist using
   opcode \p opc. If \p arglist = 0, begin a new list.
 */
static int
jsrAddArg(int arglist, ILI_OP opc, int argili)
{
  int rg;
  int ili;

  if (arglist == 0) {
    arglist = ad1ili(IL_NULL, 0);
    availIreg = 0;
    availFreg = 0;
  }
  /*
   * WARNING: For the x86, this implies that the standard call mechanism is
   * being used.  If there are multiple arguments, they need to be pushed
   * on the stack in reverse order (first jsrAddArg() call is for the last
   * argument, ...).
   */
  switch (opc) {
  case IL_ARGAR:
    ili = ad3ili(IL_ARGAR, argili, arglist, 0);
    return ili;
  case IL_ARGIR:
  case IL_ARGKR:
  case IL_ARGSP:
  case IL_ARGDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_ARGQP:
#endif
    ili = ad2ili(opc, argili, arglist);
    return ili;
  default:
    /* allow arguments to be passed in registers and on the stack */
    break;
  }
  assert(is_daili_opcode(opc), "jsrAddArg: invalid opcode", opc, ERR_Fatal);
  if (opc == IL_DAIR || opc == IL_DAAR || opc == IL_DAKR) {
    rg = IR(availIreg++);
  } else {
    if (opc == IL_DADP && (availFreg & 1))
      availFreg++;
    rg = SP(availFreg);
    availFreg++;
    if (opc == IL_DADP)
      availFreg++;
  }

  ili = ad3ili(opc, argili, rg, arglist);
  return ili;
}

/** \brief Return the ili of a call to a function with name fname, and argument
 * list argili. If argili = 0, argument list is empty.
 *
 * \param fname  function name
 * \param opc    IL_QJSR/IL_JSR
 * \param argili argument list
 */
static int
makeCall(const char *fname, ILI_OP opc, int argili)
{
  int ili;
  bool old_share_proc, old_share_qjsr;

  if (argili == 0) {
    argili = ad1ili(IL_NULL, 0);
    availIreg = 0;
    availFreg = 0;
  }

  old_share_proc = share_proc_ili;
  old_share_qjsr = share_qjsr_ili;
  share_proc_ili = false;
  share_qjsr_ili = false;
  ili = ad2ili(opc, mkfunc(fname), argili);
  share_proc_ili = old_share_proc;
  share_qjsr_ili = old_share_qjsr;

  if (availFreg > 0 && availFreg < 4)
    availFreg = 4;
  if (availIreg > maxIreg)
    maxIreg = availIreg;
  if (availFreg > maxFreg)
    maxFreg = availFreg;

  return ili;
}

static void
addCopyinInplace(const sptrListT *list)
{
  int i, ili, n_elts, dest_nme, argili, call;
  int master_ili;
  SPTR lab;
  int altili, func;
  SPTR sptr;
  int indirect_load;
  const sptrListT *node;

  n_elts = 0;
  lab = getlab();
  for (node = list, i = 0; node; node = node->next, ++i) {
    int sptr_nme, sptr_ili;

    sptr = node->o_sptr;
    indirect_load = 0;
    if (STYPEG(sptr) == ST_CMBLK) {
      sptr = CMEMFG(sptr);
      if (!sptr)
        continue;
    } else if (SCG(sptr) == SC_BASED && POINTERG(sptr)) {
      if (ALLOCATTRG(sptr)) {
        indirect_load = 1;
      }
      sptr = MIDNUMG(sptr);
    }
    /* what we have here it TPxx, need to find the symbol it points to */
    /* master copy - should be passed from previous region */
    master_ili = mk_address(sptr);

    /* current threadprivate copy */
    sptr_ili = llGetThreadprivateAddr(node->sptr);
    if (indirect_load == 1) {
      sptr_nme = addnme(NT_VAR, sptr, 0, 0);
      sptr_ili = ad2ili(IL_LDA, sptr_ili, sptr_nme);
      master_ili = ad2ili(IL_LDA, master_ili, sptr_nme);
    }
    dest_nme = ILI_OPND(sptr_ili, 2);

    if (n_elts == 0) {
      ili = ad4ili(IL_ACJMP, sptr_ili, master_ili, CC_EQ, lab);
      RFCNTI(lab);
      chk_block(ili);
      n_elts = 1;
    }

    /* now do a copy */
    altili = 0;
    {
      func = mkMemcpy();
      argili = jsrAddArg(0, IL_ARGKR, sel_iconv(node->size_ili, 1));
      argili = jsrAddArg(argili, IL_ARGAR, master_ili);
      argili = jsrAddArg(argili, IL_ARGAR, sptr_ili);
      call = makeCall("memcpy", IL_JSR, argili);
      argili = ad1ili(IL_NULL, 0);
      argili =
          ad4ili(IL_GARG, sel_iconv(node->size_ili, 1), argili, DT_INT8, 0);
      argili = ad4ili(IL_GARG, master_ili, argili, DT_CPTR, 0);
      argili = ad4ili(IL_GARG, sptr_ili, argili, DT_CPTR, 0);
      altili = ad3ili(IL_GJSR, func, argili, 0);
    }
    ILI_ALT(call) = altili;
    iltb.callfg = 1;
    chk_block(call);
  }
  if (n_elts) {
    wr_block();
    cr_block();

    /* create a block */
    BIH_LABEL(expb.curbih) = lab;
    ILIBLKP(lab, expb.curbih);
    ili = ll_make_kmpc_barrier();
    iltb.callfg = 1;
    chk_block(ili);

    wr_block();
    cr_block();
  }
}

static void
makeCopyprivArray_tls(const sptrListT *list)
{
  int i, ili, n_elts, basenme, argili, call;
  int master_ili;
  SPTR lab;
  int altili, master_nme, func;
  SPTR sptr;
  const sptrListT *node;

  n_elts = 0;
  lab = getlab();
  for (node = list, i = 0; node; node = node->next, ++i) {
    int sptr_ili;

    sptr = MIDNUMG(node->sptr);
    if (STYPEG(sptr) == ST_CMBLK) {
      sptr = CMEMFG(node->o_sptr);
      if (!sptr)
        continue;
    } else if (SCG(sptr) == SC_BASED && POINTERG(sptr)) {
      sptr = MIDNUMG(sptr);
    }
    master_nme = addnme(NT_VAR, sptr, 0, (INT)0);
    master_ili = mk_address(sptr);

    basenme = addnme(NT_VAR, node->sptr, 0, (INT)0);
    sptr_ili = ad2ili(IL_LDA, ad_acon(node->sptr, (INT)0), basenme);
    if (n_elts == 0) {
      ili = ad4ili(IL_ACJMP, sptr_ili, master_ili, CC_EQ, lab);
      RFCNTI(lab);
      chk_block(ili);
      n_elts = 1;
    }

    /* now do a copy */
    altili = 0;
    {
      func = mkMemcpy();
      argili = jsrAddArg(0, IL_ARGIR, sel_iconv(node->size_ili, 0));
      argili = jsrAddArg(argili, IL_ARGAR, master_ili);
      argili = jsrAddArg(argili, IL_ARGAR, sptr_ili);
      call = makeCall("memcpy", IL_JSR, argili);
      argili = ad1ili(IL_NULL, 0);
      argili =
          ad4ili(IL_GARG, sel_iconv(node->size_ili, 1), argili, DT_INT8, 0);
      argili = ad4ili(IL_GARG, master_ili, argili, DT_CPTR, 0);
      argili = ad4ili(IL_GARG, sptr_ili, argili, DT_CPTR, 0);
      altili = ad3ili(IL_GJSR, func, argili, 0);
    }
    ILI_ALT(call) = altili;
    iltb.callfg = 1;
    chk_block(call);
  }
  if (n_elts) {
    wr_block();
    cr_block();

    /* create a block */
    BIH_LABEL(expb.curbih) = lab;
    ILIBLKP(lab, expb.curbih);
    ili = ll_make_kmpc_barrier();
    iltb.callfg = 1;
    chk_block(ili);

    wr_block();
    cr_block();
  }
}

static int
findEnlabBih(int func)
{
  int bih;
  bih = BIH_NEXT(BIHNUMG(func));
  return bih;
}

static void
setTaskloopVars(SPTR lb, SPTR ub, SPTR stride, SPTR lastitr)
{
  int nme, basenm, baseili, ili;
  SPTR arg;
  int oldbih;
  ILI_OP ld, st;
  MSZ msz;

  oldbih = expb.curbih;
  /* This code is in an outlined taskloop routine.
   * Load taskloop vars from arg1 to local/private vars.
   */
  arg = ll_get_hostprog_arg(GBL_CURRFUNC, 2);
  basenm = addnme(NT_VAR, arg, 0, 0);
  baseili = ad_acon(arg, 0);
  baseili = mk_address(arg);
  arg = mk_argasym(arg);
  basenm = addnme(NT_VAR, arg, 0, (INT)0);
  baseili = ad2ili(IL_LDA, baseili, basenm);
  nme = addnme(NT_IND, lb, basenm, 0);
  ili = ad3ili(IL_AADD, baseili, ad_aconi(TASK_LPVAR_OFFSET), 0);
  ldst_msz(DT_INT8, &ld, &st, &msz);
  ili = ad3ili(ld, ili, nme, msz);
  ldst_msz(DTYPEG(lb), &ld, &st, &msz);
  if (msz != MSZ_I8)
    ili = kimove(ili);
  ili = ad4ili(st, ili, mk_address(lb), addnme(NT_VAR, lb, 0, 0), msz);
  chk_block(ili);

  nme = addnme(NT_IND, ub, basenm, 0);
  ili = ad3ili(IL_AADD, baseili,
               ad_aconi(TASK_LPVAR_OFFSET + zsize_of(DT_INT8)), 0);
  ldst_msz(DT_INT8, &ld, &st, &msz);
  ili = ad3ili(ld, ili, nme, msz);
  ldst_msz(DTYPEG(ub), &ld, &st, &msz);
  if (msz != MSZ_I8)
    ili = kimove(ili);
  ili = ad4ili(st, ili, mk_address(ub), addnme(NT_VAR, ub, 0, 0), msz);
  chk_block(ili);

  if (STYPEG(stride) != ST_CONST) {
    nme = addnme(NT_IND, stride, basenm, 0);
    ili = ad3ili(IL_AADD, baseili,
                 ad_aconi(TASK_LPVAR_OFFSET + (zsize_of(DT_INT8) * 2)), 0);
    ldst_msz(DT_INT8, &ld, &st, &msz);
    ili = ad3ili(ld, ili, nme, msz);
    ldst_msz(DTYPEG(stride), &ld, &st, &msz);
    if (msz != MSZ_I8)
      ili = kimove(ili);
    ili =
        ad4ili(st, ili, mk_address(stride), addnme(NT_VAR, stride, 0, 0), msz);
    chk_block(ili);
  }

  if (lastitr && STYPEG(lastitr) != ST_CONST) {
    nme = addnme(NT_IND, lastitr, basenm, 0);
    ldst_msz(DT_INT, &ld, &st, &msz);
    ili = ad3ili(IL_AADD, baseili,
                 ad_aconi(TASK_LPVAR_OFFSET + (zsize_of(DT_INT8) * 3)), 0);
    ili = ad3ili(ld, ili, nme, msz);
    ldst_msz(DTYPEG(lastitr), &ld, &st, &msz);
    ili = ad4ili(st, ili, ad_acon(lastitr, 0), addnme(NT_VAR, lastitr, 0, 0),
                 msz);
    chk_block(ili);
  }
  if (oldbih == expb.curbih) {
    wr_block();
    cr_block();
  }
}

/* isn't there some standard routine I can subsititue for this? */
static int
getElemSize(DTYPE dtype)
{

  DTYPE dd = dtype;

  while (dd && (DTY(dd) == TY_ARRAY)) {
    dd = DTySeqTyElement(dd);
  }
  if (DTY(dd) == TY_STRUCT)
    return DTyAlgTySize(dd);
  return 0;
}

static void
clearTaskloopInfo(void)
{
  INT offset = TASK_LPVAR_OFFSET;
  BZERO(&taskLpInfo, char, sizeof(taskLpInfo));
  TASK_LPVAR_OFFSET = offset;
}

static int
genIntStore(SPTR sym, int rhs)
{
  int ili;
  int nme;

  ili = ad_acon(sym, 0);
  nme = addnme(NT_VAR, sym, 0, 0);
  ili = ad4ili(IL_ST, rhs, ili, nme, MSZ_WORD);
  return ili;
}

static int
genIntLoad(SPTR sym)
{
  int ili;
  int nme;

  ili = ad_acon(sym, 0);
  nme = addnme(NT_VAR, sym, 0, 0);
  ili = ad3ili(IL_LD, ili, nme, MSZ_WORD);
  return ili;
}

void
exp_smp(ILM_OP opc, ILM *ilmp, int curilm)
{
#ifdef IM_BPAR
  int ili = 0;
  SPTR sym;
  SPTR sptr;
  int offset, savebih;
  int off;
  int addr, stili;
  int argilm;
  SPTR tpv;
  int pv;
  int savex14;
  const char *doschedule;
  int semaphore;
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
  int dotarget;
  SPTR beg_label;
#endif
  static int assign_rou = 0; /* C++ only, lets avoid more ifdefs */
  int element_size;
  loop_args_t loop_args;
  LLTask *task;
  bool is_cmblk;
  static sptrListT *copysptr_list = NULL;
  static SPTR uplevel_sptr;
  static SPTR single_thread;
  static SPTR in_single;
  static SPTR targetfunc_sptr = SPTR_NULL;
  SPTR nlower, nupper, nstride;
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
  static int target_ili_num_threads = 0;
  static int target_ili_num_teams= 0;
  static int target_ili_thread_limit= 0;
#endif
  int sz;
  ISZ_T size, num_elements;
  static int isTargetDevice = 0;
  switch (opc) {
  case IM_BPAR:
  case IM_BPARN:
  case IM_BPARD:
  case IM_BPARA:
  case IM_EPAR:
  case IM_EPARD:
  case IM_BTASK:
  case IM_BTASKLOOP:
  case IM_ETASK:
  case IM_ETASKLOOP:
  case IM_BTARGET:
  case IM_ETARGET:
  case IM_BTEAMS:
  case IM_BTEAMSN:
  case IM_ETEAMS:
  case IM_TASKLOOPVARS:
  case IM_TASKLOOPREG:
#ifdef IM_TASKPRIVATE
  case IM_TASKPRIVATE:
#endif
  case IM_TASKFIRSTPRIV:
  case IM_ETASKLOOPREG:
  case IM_BTASKDUP:
  case IM_ETASKDUP:
    break;
  default:
    ll_rewrite_ilms(-1, curilm, 0);
    break;
  }
  switch (opc) {
  case IM_BMPSCOPE:
    if (ll_ilm_is_rewriting())
      break;
    scopeSptr = ILM_SymOPND(ilmp, 1);
#ifdef PARUPLEVELG
    uplevel_sptr = PARUPLEVELG(scopeSptr);
#else
    uplevel_sptr = 0;
#endif
    break;
  case IM_EMPSCOPE:
    break;
  case IM_BPAR:
  case IM_BPARN:
  case IM_BPARA:
#ifdef OMP_OFFLOAD_LLVM
      if (flg.omptarget && gbl.ompaccel_intarget) {
        exp_ompaccel_bpar(ilmp, curilm, uplevel_sptr, scopeSptr, incrOutlinedCnt);
        break;
      }
#endif
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    incrOutlinedCnt();
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    if (gbl.outlined)
      expb.sc = SC_PRIVATE;
    if (outlinedCnt == 1) {
      int isPar = ILI_OF(ILM_OPND(ilmp, 1));
      SPTR par_label, end_label;
      int iliarg, nthreads, proc_bind;
      {
        sptr = ll_make_outlined_func_wopc((SPTR)uplevel_sptr, scopeSptr, opc);
        if (!PARENCLFUNCG(scopeSptr))
          PARENCLFUNCP(scopeSptr, sptr);
      }
        ll_write_ilm_header(sptr, curilm);
      iliarg = ll_load_outlined_args(scopeSptr, sptr, gbl.outlined);

      /* if (isPar == 0)
             call omp_do_func(.....)
             goto do_end;
         par_label:
             call kmpc_fork_call (....., omp_do_func,.... )
         do_label:
       */

      par_label = getlab();
      end_label = getlab();

      isPar = ad3ili(IL_ICJMPZ, isPar, CC_EQ, par_label);
      RFCNTI(par_label);
      chk_block(isPar);

      ili = ll_make_kmpc_serialized_parallel();
      iltb.callfg = 1;
      chk_block(ili);

      ili = ll_make_outlined_call2(sptr, iliarg);
      iltb.callfg = 1;
      chk_block(ili);

      ili = ll_make_kmpc_end_serialized_parallel();
      iltb.callfg = 1;
      chk_block(ili);

      ili = ad1ili(IL_JMP, end_label);
      RFCNTI(end_label);
      chk_block(ili);

      wr_block();
      cr_block();
      exp_label(par_label);
      proc_bind = 0;
      if (opc == IM_BPARA) {
        int flag = ILM_OPND(ilmp, 3);
        if (flag & 0x2) {
          nthreads = ILI_OF(ILM_OPND(ilmp, 2));
          ili = ll_make_kmpc_push_num_threads(nthreads);
          iltb.callfg = 1;
          chk_block(ili);
        }
        if (flag & 0x1) {
          proc_bind = ILM_OPND(ilmp, 4);
        }
      }
      if (opc == IM_BPARN) {
        nthreads = ILI_OF(ILM_OPND(ilmp, 2));
        ili = ll_make_kmpc_push_num_threads(nthreads);
        iltb.callfg = 1;
        chk_block(ili);
      }
      if (proc_bind) {
        proc_bind = ad_icon(proc_bind);
        ili = ll_make_kmpc_push_proc_bind(proc_bind);
        iltb.callfg = 1;
        chk_block(ili);
      }
        ili = ll_make_kmpc_fork_call(sptr, 1, &iliarg, OPENMP, -1);
      iltb.callfg = 1;
      chk_block(ili);

      exp_label(end_label);

      ccff_info(MSGOPENMP, "OMP001", gbl.findex, gbl.lineno,
                "Parallel region activated", NULL);

    } else if (outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
    }

    break;
  case IM_BPARD:
    /* lexically nested begin parallel */
    incrOutlinedCnt();
    if (outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    if (outlinedCnt == 1) {
      int isPar = ILI_OF(ILM_OPND(ilmp, 1));
      SPTR par_label, end_label;
      int iliarg, proc_bind;
      sptr = ll_make_outlined_func((SPTR)uplevel_sptr, scopeSptr);
      if (!PARENCLFUNCG(scopeSptr))
        PARENCLFUNCP(scopeSptr, sptr);
      ll_write_ilm_header(sptr, curilm);
      iliarg = ll_load_outlined_args(scopeSptr, sptr, gbl.outlined);

      /* if (isPar == 0)
             call omp_do_func(.....)
             goto do_end;
         par_label:
             call kmpc_fork_call (....., omp_do_func,.... )
         do_label:
       */

      par_label = getlab();
      end_label = getlab();

      isPar = ad3ili(IL_ICJMPZ, isPar, CC_EQ, par_label);
      RFCNTI(par_label);
      chk_block(isPar);

      ili = ll_make_kmpc_serialized_parallel();
      iltb.callfg = 1;
      chk_block(ili);

      ili = ll_make_outlined_call2(sptr, iliarg);
      iltb.callfg = 1;
      chk_block(ili);

      ili = ll_make_kmpc_end_serialized_parallel();
      iltb.callfg = 1;
      chk_block(ili);

      ili = ad1ili(IL_JMP, end_label);
      RFCNTI(end_label);
      chk_block(ili);

      proc_bind = ILM_OPND(ilmp, 2);
      if (proc_bind) {
        proc_bind = ad_icon(proc_bind);
        ili = ll_make_kmpc_push_proc_bind(proc_bind);
        iltb.callfg = 1;
        chk_block(ili);
      }

      wr_block();
      cr_block();
      exp_label(par_label);
      ili = ll_make_kmpc_fork_call(sptr, 1, &iliarg, OPENMP, -1);
      iltb.callfg = 1;
      chk_block(ili);

      wr_block();
      cr_block();
      exp_label(end_label);
    }
    ccff_info(MSGOPENMP, "OMP001", gbl.findex, gbl.lineno,
              "Parallel region activated", NULL);
    break;
  case IM_BTEAMS:
  case IM_BTEAMSN:
#ifdef OMP_OFFLOAD_LLVM
      if(flg.omptarget && gbl.ompaccel_intarget) {
        exp_ompaccel_bteams(ilmp, curilm, outlinedCnt, uplevel_sptr, scopeSptr, incrOutlinedCnt);
        break;
      }
#endif
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    incrOutlinedCnt();
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    if (gbl.outlined)
      expb.sc = SC_PRIVATE;
    if (outlinedCnt == 1) {
      SPTR par_label;
      int iliarg, nteams, n_limit;
      {
        sptr = ll_make_outlined_func_wopc((SPTR)uplevel_sptr, scopeSptr, opc);
      }
      if (!PARENCLFUNCG(scopeSptr))
        PARENCLFUNCP(scopeSptr, sptr);
        ll_write_ilm_header(sptr, curilm);
      iliarg = ll_load_outlined_args(scopeSptr, sptr, gbl.outlined);

      par_label = getlab();

      wr_block();
      cr_block();
      exp_label(par_label);
      if (opc == IM_BTEAMSN) {
        nteams = ILI_OF(ILM_OPND(ilmp, 1));
        n_limit = ILI_OF(ILM_OPND(ilmp, 2));
        ili = ll_make_kmpc_push_num_teams(nteams, n_limit);
        iltb.callfg = 1;
        chk_block(ili);
      }
      ili = ll_make_kmpc_fork_teams(sptr, 1, &iliarg);
      iltb.callfg = 1;
      chk_block(ili);

      ccff_info(MSGOPENMP, "OMP022", gbl.findex, gbl.lineno,
                "Teams region activated", NULL);

    } else if (outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
    }

    break;

  case IM_BTARGET:
#ifdef OMP_OFFLOAD_LLVM
      if (flg.omptarget) {
        exp_ompaccel_btarget(ilmp, curilm, uplevel_sptr, scopeSptr, incrOutlinedCnt, &targetfunc_sptr, &isTargetDevice);
        break;
      }
#endif
    /* lexically nested begin parallel */
    incrOutlinedCnt();
    if (outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    if (outlinedCnt == 1) {
      isTargetDevice = ILI_OF(ILM_OPND(ilmp, 1));
      targetfunc_sptr = ll_make_outlined_func_wopc((SPTR)uplevel_sptr, scopeSptr, opc);
      if (!PARENCLFUNCG(scopeSptr))
        PARENCLFUNCP(scopeSptr, targetfunc_sptr);
      ll_write_ilm_header(targetfunc_sptr, curilm);
    }
    ccff_info(MSGOPENMP, "OMP020", gbl.findex, gbl.lineno,
              "Target region activated", NULL);
    break;
  case IM_ETARGET:
    if (outlinedCnt == 1) {
      ilm_outlined_pad_ilm(curilm);
    }
    decrOutlinedCnt();
    if (outlinedCnt >= 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    if (gbl.outlined)
      expb.sc = SC_AUTO;

#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    SPTR end_label;
#endif
    int iliarg;

#ifdef OMP_OFFLOAD_LLVM
    if(flg.omptarget) {
      assert(targetfunc_sptr != SPTR_NULL,
           "Outlined function of target region is not found.", GBL_CURRFUNC, ERR_Fatal);
      // In Flang, We outline the region once, and offload it in the device
      // We don't generate outlined function for the host. so we don't have host fallback.
      exp_ompaccel_etarget(ilmp, curilm, targetfunc_sptr, outlinedCnt, (SPTR) uplevel_sptr, decrOutlinedCnt);
      break;
    }
#endif

    {
      assert(targetfunc_sptr != SPTR_NULL,
           "Outlined function of target region is not found.", GBL_CURRFUNC, ERR_Fatal);
      // When OpenMP Offload is not enabled, we simply call host outlined function.
      iliarg = ll_load_outlined_args(scopeSptr, targetfunc_sptr, gbl.outlined);
      ili = ll_make_outlined_call2(targetfunc_sptr, iliarg);
      iltb.callfg = 1;
      chk_block(ili);
      wr_block();
      cr_block();
    }
    targetfunc_sptr = SPTR_NULL;
    ccff_info(MSGOPENMP, "OMP021", gbl.findex, gbl.lineno,
              "Target region terminated", NULL);
    break;
  case IM_EPAR:
#ifdef OMP_OFFLOAD_LLVM
      if (flg.omptarget && gbl.ompaccel_intarget) {
        exp_ompaccel_epar(ilmp, curilm, outlinedCnt, decrOutlinedCnt);
        break;
      }
#endif
    if (outlinedCnt == 1) {
      ilm_outlined_pad_ilm(curilm);
    }
    decrOutlinedCnt();
    if (outlinedCnt >= 1)
      ll_rewrite_ilms(-1, curilm, 0);

    if (gbl.outlined)
      expb.sc = SC_AUTO;
    ccff_info(MSGOPENMP, "OMP002", gbl.findex, gbl.lineno,
              "Parallel region terminated", NULL);
    break;
  case IM_EPARD:
    if (outlinedCnt == 1) {
      ilm_outlined_pad_ilm(curilm);
    }
    decrOutlinedCnt();
    if (outlinedCnt >= 1)
      ll_rewrite_ilms(-1, curilm, 0);
    ccff_info(MSGOPENMP, "OMP002", gbl.findex, gbl.lineno,
              "Parallel region terminated", NULL);
    break;
  case IM_ETEAMS:
#ifdef OMP_OFFLOAD_LLVM
      if (flg.omptarget) {
        exp_ompaccel_eteams(ilmp, curilm, outlinedCnt, decrOutlinedCnt);
        break;
      }
#endif
    if (outlinedCnt == 1) {
      ilm_outlined_pad_ilm(curilm);
    }
    decrOutlinedCnt();
    if (outlinedCnt >= 1)
      ll_rewrite_ilms(-1, curilm, 0);

    if (gbl.outlined)
      expb.sc = SC_AUTO;
    ccff_info(MSGOPENMP, "OMP023", gbl.findex, gbl.lineno,
              "Teams region terminated", NULL);
    break;
  case IM_BCS:
    /*
     * It's required that the front-end does not generate nested
     * critical sections (static only).  Keeping the semaphore variable
     * around for the IM_ECS depends on this.  If nested critical sections
     * need to be supported, then need to add a field to the IM_BCS and
     * IM_ECS ilms which will be the semaphore variable created by the
     * front-ends.
     */
    if (ll_ilm_is_rewriting())
      break;
    critCnt++;
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    bihb.csfg = BIH_CS(expb.curbih) = true;
    ili = addMpBcsNest();
    iltb.callfg = 1;
    chk_block(ili);
    ccff_info(MSGOPENMP, "OMP003", gbl.findex, gbl.lineno,
              "Begin critical section", NULL);
    break;
  case IM_ECS:
    if (ll_ilm_is_rewriting())
      break;
    critCnt--;
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    BIH_CS(expb.curbih) = true;
    ili = addMpEcsNest();
    iltb.callfg = 1;
    chk_block(ili);
    wr_block();
    cr_block();
    if (critCnt <= 0)
      bihb.csfg = 0;
    ccff_info(MSGOPENMP, "OMP004", gbl.findex, gbl.lineno,
              "End critical section", NULL);
    break;
  case IM_P:
    if (ll_ilm_is_rewriting())
      break;
    critCnt++;
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    bihb.csfg = BIH_CS(expb.curbih) = true;
    sym = ILM_SymOPND(ilmp, 1);
    if (!XBIT(69, 0x40) || !isUnnamedCs(sym)) {
      ili = add_mp_p(sym);
    } else {
      ili = addMpUnp();
    }
    iltb.callfg = 1;
    chk_block(ili);
    semaphore = MIDNUMG(sym);
    ccff_info(MSGOPENMP, "OMP012", gbl.findex, gbl.lineno,
              "Begin critical section (%semaphore)", "semaphore=%s",
              SYMNAME(semaphore), NULL);
    break;
  case IM_V:
    if (ll_ilm_is_rewriting())
      break;
    critCnt--;
    BIH_FT(expb.curbih) = true;
    BIH_QJSR(expb.curbih) = true;
    BIH_NOMERGE(expb.curbih) = true;
    BIH_CS(expb.curbih) = true;
    sym = ILM_SymOPND(ilmp, 1);
    if (!XBIT(69, 0x40) || !isUnnamedCs(sym)) {
      ili = add_mp_v(sym);
    } else {
      ili = addMpUnv();
    }
    iltb.callfg = 1;
    chk_block(ili);
    wr_block();
    cr_block();
    if (critCnt <= 0)
      bihb.csfg = 0;
    semaphore = MIDNUMG(sym);
    ccff_info(MSGOPENMP, "OMP013", gbl.findex, gbl.lineno,
              "End critical section (%semaphore)", "semaphore=%s",
              SYMNAME(semaphore), NULL);
    break;
  case IM_MPSCHED: {
    if (!ll_ilm_is_rewriting()) {
      const SPTR lower = ILM_SymOPND(ilmp, 1);
      const SPTR upper = ILM_SymOPND(ilmp, 2);
      const SPTR stride = ILM_SymOPND(ilmp, 3);
      const SPTR last = ILM_SymOPND(ilmp, 4);
      const DTYPE dtype = ILM_DTyOPND(ilmp, 5);
      ili = ll_make_kmpc_dispatch_next(lower, upper, stride, last, dtype);
      iltb.callfg = 1;
      chk_block(ili);
      ILM_RESULT(curilm) = ili;
    }
    break;
  }
  case IM_MPBORDERED: {
    if (!ll_ilm_is_rewriting()) {
      BIH_NOMERGE(expb.curbih) = true;
      critCnt++;
      bihb.csfg = BIH_CS(expb.curbih) = true;
      ili = ll_make_kmpc_ordered();
      iltb.callfg = 1;
      chk_block(ili);
    }
    break;
  }
  case IM_MPEORDERED: {
    if (!ll_ilm_is_rewriting()) {
      ili = ll_make_kmpc_end_ordered();
      iltb.callfg = 1;
      BIH_CS(expb.curbih) = true;
      chk_block(ili);
      wr_block();
      critCnt--;
      if (critCnt <= 0)
        bihb.csfg = 0;
    }
    break;
  }
  case IM_MPTASKLOOP:
    if (ll_ilm_is_rewriting())
      break;

    {
      SPTR lb = ILM_SymOPND(ilmp, 1);
      SPTR ub = ILM_SymOPND(ilmp, 2);
      SPTR st = ILM_SymOPND(ilmp, 3);
      SPTR lastitr = ILM_SymOPND(ilmp, 4);

      ENCLFUNCP(lb, taskFnsptr);
      ENCLFUNCP(ub, taskFnsptr);
      ENCLFUNCP(st, taskFnsptr);
      TASK_LASTITR = lastitr;
      if (lastitr) {
        ENCLFUNCP(lastitr, taskFnsptr);
      }
      setTaskloopVars(lb, ub, st, lastitr);
    }

    break;
  case IM_MPLOOP: {
    int sched;
    if (outlinedCnt >= 1)
      break;
#ifdef OMP_OFFLOAD_LLVM
      if (flg.omptarget && gbl.ompaccel_intarget) {
        exp_ompaccel_mploop(ilmp, curilm);
        break;
      }
#endif
    loop_args.sched = (kmpc_sched_e)ILM_OPND(ilmp, 7);
    sched = mp_sched_to_kmpc_sched(loop_args.sched);
    nlower = ILM_SymOPND(ilmp, 1);
    nupper = ILM_SymOPND(ilmp, 2);
    nstride = ILM_SymOPND(ilmp, 3);
    if (!XBIT(183, 0x100000)) {
      nlower = getccsym_copy(nlower);
      nupper = getccsym_copy(nupper);
      nstride = getccsym_copy(nstride);
      ENCLFUNCP(nlower, GBL_CURRFUNC);
      ENCLFUNCP(nupper, GBL_CURRFUNC);
      ENCLFUNCP(nstride, GBL_CURRFUNC);
      exp_add_copy(nlower, ILM_SymOPND(ilmp, 1));
      exp_add_copy(nupper, ILM_SymOPND(ilmp, 2));
      exp_add_copy(nstride, ILM_SymOPND(ilmp, 3));
    }
    loop_args.lower = nlower;
    loop_args.upper = nupper;
    loop_args.stride = nstride;
    loop_args.chunk = ILM_SymOPND(ilmp, 4);
    loop_args.last = ILM_SymOPND(ilmp, 5);
    loop_args.dtype = ILM_DTyOPND(ilmp, 6);
    switch (sched) {
    case KMP_SCH_STATIC:
    case KMP_SCH_STATIC_CHUNKED:
      if ((ILM_OPND(ilmp, 7) & 0xff00) == MP_SCH_CHUNK_1) {
        doschedule = "static cyclic";
        ccff_info(MSGOPENMP, "OMP014", gbl.findex, gbl.lineno,
                  "Parallel loop activated with %schedule schedule",
                  "schedule=%s", doschedule, NULL);
      }
      FLANG_FALLTHROUGH;

    case KMP_DISTRIBUTE_STATIC_CHUNKED:
    case KMP_DISTRIBUTE_STATIC:
    case KMP_DISTRIBUTE_STATIC_CHUNKED_CHUNKONE:
      ili = ll_make_kmpc_for_static_init(&loop_args);
      break;
    default:
      ili = ll_make_kmpc_dispatch_init(&loop_args);
    }
    iltb.callfg = 1;
    chk_block(ili);
    BIH_NOMERGE(expb.curbih) = true;
    if (!XBIT(183, 0x100000)) {
      exp_add_copy(ILM_SymOPND(ilmp, 1), nlower);
      exp_add_copy(ILM_SymOPND(ilmp, 2), nupper);
      exp_add_copy(ILM_SymOPND(ilmp, 3), nstride);
    }

    /* constant propagation stop when it sees function call. We may have some
     * stride that needs to propagate for computation of tripcount. */
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }

    break;
  }
  case IM_MPDISTLOOP: {
    int sched;
    if (outlinedCnt >= 1)
      break;
    loop_args.lower = ILM_SymOPND(ilmp, 1);
    loop_args.upper = ILM_SymOPND(ilmp, 2);
    loop_args.stride = ILM_SymOPND(ilmp, 3);
    loop_args.chunk = ILM_SymOPND(ilmp, 4);
    loop_args.last = ILM_SymOPND(ilmp, 5);
    loop_args.upperd = ILM_SymOPND(ilmp, 6);
    loop_args.dtype = ILM_DTyOPND(ilmp, 7);
    loop_args.sched = (kmpc_sched_e)ILM_OPND(ilmp, 8);
    sched = mp_sched_to_kmpc_sched(loop_args.sched);
    switch (sched) {
    case KMP_SCH_STATIC:
    case KMP_SCH_STATIC_CHUNKED:
    case KMP_DISTRIBUTE_STATIC_CHUNKED:
    case KMP_DISTRIBUTE_STATIC:
      ili = ll_make_kmpc_dist_for_static_init(&loop_args);
      break;
    default:
      ili = ll_make_kmpc_dist_dispatch_init(&loop_args);
    }
    iltb.callfg = 1;
    chk_block(ili);
    BIH_NOMERGE(expb.curbih) = true;

    /* constant propagation stop when it sees function call. We may have some
     * stride that needs to propagate for computation of tripcount. */
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }

    break;
  }
  case IM_MPLOOPFINI: {
#ifdef OMP_OFFLOAD_LLVM
      if (flg.omptarget) {
        exp_ompaccel_mploopfini(ilmp, curilm, outlinedCnt);
        break;
      }
#endif
    if (outlinedCnt >= 1)
      break;
    const int sched = mp_sched_to_kmpc_sched(ILM_OPND(ilmp, 2));
    if (sched == KMP_ORD_STATIC || sched == KMP_ORD_DYNAMIC_CHUNKED) {
      ili = ll_make_kmpc_dispatch_fini(ILM_DTyOPND(ilmp, 1));
      iltb.callfg = 1;
      chk_block(ili);
    } else if (sched == KMP_SCH_STATIC || sched == KMP_SCH_STATIC_CHUNKED ||
               sched == KMP_DISTRIBUTE_STATIC ||
               sched == KMP_DISTRIBUTE_STATIC_CHUNKED) {
      ili = ll_make_kmpc_for_static_fini();
      iltb.callfg = 1;
      chk_block(ili);
    }
    break;
  }
  case IM_BPDO:
  case IM_EPDO:
    break;

  case IM_PDO:
    if (outlinedCnt >= 1)
      break;
    sym = ILM_SymOPND(ilmp, 1);
    if (ILIBLKG(sym))
      BIH_PARLOOP(ILIBLKG(sym)) = 1;
    switch (ILM_OPND(ilmp, 2) & 0xff) {
    case 6: /* distribute static schedule */
    case 0: /* static schedule */
      switch (ILM_OPND(ilmp, 2) & 0xff00) {
      case 0:
        doschedule = "static block";
        break;
      case MP_SCH_CHUNK_1:
        doschedule = "static cyclic";
        break;
      case MP_SCH_BLK_CYC:
        doschedule = "static block-cyclic";
        break;
      case MP_SCH_BLK_ALN:
        /* also PARALN */
        doschedule = "static block";
        break;
      default:
        doschedule = "";
        break;
      }
      break;
    case 1:
      doschedule = " dynamic";
      break;
    case 2:
      doschedule = " guided";
      break;
    case 3:
      doschedule = " interleaved"; /* not used */
      break;
    case 4:
      doschedule = " runtime schedule";
      break;
    case 5:
      doschedule = " auto schedule";
      break;
    default:
#if DEBUG
      interr("exp_smp: IM_PDO unknown schedule", ILM_OPND(ilmp, 2) & 0xff,
             ERR_Severe);
#endif
      doschedule = " static";
    }
    if ((ILM_OPND(ilmp, 2) & 0xff) == 6) {
      ccff_info(MSGOPENMP, "OMP024", gbl.findex, gbl.lineno,
                "Distribute loop activated with %schedule schedule",
                "schedule=%s", doschedule, NULL);
      break;
    }
    ccff_info(MSGOPENMP, "OMP014", gbl.findex, gbl.lineno,
              "Parallel loop activated with %schedule schedule", "schedule=%s",
              doschedule, NULL);
    break;
  case IM_BARRIER:
    if (outlinedCnt >= 1)
      break;
    else if (!XBIT(183, 0x2000)) { /* If kmpc enabled */
      ili = ll_make_kmpc_barrier();
      iltb.callfg = 1;
    }
    chk_block(ili);
    ccff_info(MSGOPENMP, "OMP015", gbl.findex, gbl.lineno, "Barrier", NULL);
    break;
  case IM_BSECTIONS:
    if (!ll_ilm_is_rewriting()) {
      ccff_info(MSGOPENMP, "OMP005", gbl.findex, gbl.lineno, "Begin sections",
                NULL);

      if (flg.opt != 0) {
        wr_block();
        cr_block();
      }
      parsectCnt++;
      BIH_PARSECT(expb.curbih) = bihb.parsectfg = true;
      expSmpSectionInit();
      wr_block();
      cr_block();
    }
    break;
  case IM_MASTER:
    if (outlinedCnt >= 1)
      break;
    ccff_info(MSGOPENMP, "OMP008", gbl.findex, gbl.lineno,
              "Begin master region", NULL);

    parsectCnt++;
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    sym = ILM_SymOPND(ilmp, 1);
    ili = ll_make_kmpc_master();
    ili = ad3ili(IL_ICJMPZ, ili, CC_EQ, sym);
    iltb.callfg = 1;
    BIH_PARSECT(expb.curbih) = bihb.parsectfg = true;
    chk_block(ili);
    break;
  case IM_SECTION:
    if (!ll_ilm_is_rewriting()) {
      ccff_info(MSGOPENMP, "OMP006", gbl.findex, gbl.lineno, "New section",
                NULL);

      if (SECT_CNT == 0) { /* first section make call */
        /* we should know lower bound but don't know upper bound */
        /* make a call to static_for_init here - we will fill upper bound later
         */
        int *args;
        wr_block();
        cr_block();
        ili = ad4ili(IL_ST, ad_icon(0), ad_acon(SECT_LB, 0),
                     addnme(NT_VAR, SECT_LB, 0, 0), MSZ_WORD);
        chk_block(ili);
        ili = ad4ili(IL_ST, ad_icon(1), ad_acon(SECT_ST, 0),
                     addnme(NT_VAR, SECT_ST, 0, 0), MSZ_WORD);
        chk_block(ili);
        args = ll_make_sections_args(SECT_LB, SECT_UB, SECT_ST, SECT_LAST);
        ili = ll_make_kmpc_for_static_init_args(DT_UINT, args);
        iltb.callfg = 1;
        chk_block(ili);
      }

      /*
       * if (lb != cnt)
       *   jmp to next label
       */
      wr_block();
      cr_block();
      exp_label(ILM_SymOPND(ilmp, 3));
      BIH_LABEL(expb.curbih) = ILM_SymOPND(ilmp, 3);
      ILIBLKP(BIH_LABEL(expb.curbih), expb.curbih);

      ili = sectionCreateBlock(ILM_OPND(ilmp, 2), SECT_LB, SECT_UB, SECT_CNT);
      chk_block(ili);
      RFCNTI(ILM_OPND(ilmp, 2));

      ++SECT_CNT;
    }
    break;
  case IM_LSECTION:
    if (!ll_ilm_is_rewriting()) {
      wr_block();
      cr_block();
      exp_label(ILM_SymOPND(ilmp, 3));
      BIH_LABEL(expb.curbih) = ILM_SymOPND(ilmp, 3);
      ILIBLKP(BIH_LABEL(expb.curbih), expb.curbih);
      wr_block();
      cr_block();

      /* now assign the upper bound to SECT_UB */
      savebih = expb.curbih;
      savex14 = flg.x[14];
      flg.x[14] |= 0x1000;
      wr_block();
      expb.curbih = SECT_BBIH;
      rdilts(expb.curbih);
      expb.curilt = ILT_PREV(0);
      ili = ad4ili(IL_ST, ad_icon(SECT_CNT - 1), ad_acon(SECT_UB, 0),
                   addnme(NT_VAR, SECT_UB, 0, 0), MSZ_WORD);
      expb.curilt = addilt(expb.curilt, ili);
      wrilts(SECT_BBIH);
      expb.curbih = savebih;
      rdilts(expb.curbih);
      expb.curilt = ILT_PREV(0);
      flg.x[14] = savex14;
    }
    break;
  case IM_ESECTIONS:
    if (!ll_ilm_is_rewriting()) {
      expSmpSectionEnd();
      ccff_info(MSGOPENMP, "OMP007", gbl.findex, gbl.lineno, "End sections",
                NULL);
      goto esect_shared;
    }
    break;
  case IM_CANCEL:
    if (!ll_ilm_is_rewriting()) {
      int ifcancel = ILI_OF(ILM_OPND(ilmp, 3));
      int cancel_kind = ILM_OPND(ilmp, 2);
      int label = ILM_OPND(ilmp, 1);

      SPTR cancel_label = getlab();
      ifcancel = ad3ili(IL_ICJMPZ, ifcancel, CC_EQ, cancel_label);
      RFCNTI(cancel_label);
      chk_block(ifcancel);

      ili = ll_make_kmpc_cancel(ad_icon(cancel_kind));
      ifcancel = ad3ili(IL_ICJMPZ, ili, CC_NE, label);
      iltb.callfg = 1;
      chk_block(ifcancel);

      wr_block();
      cr_block();
      exp_label(cancel_label);
      ccff_info(MSGOPENMP, "OMP026", gbl.findex, gbl.lineno, "Cancel", NULL);
    }
    break;
  case IM_CANCELPOINT:
    if (!ll_ilm_is_rewriting()) {
      int cancel_kind = ILM_OPND(ilmp, 2);
      int label = ILM_OPND(ilmp, 1);
      ili = ll_make_kmpc_cancellationpoint(ad_icon(cancel_kind));
      ili = ad3ili(IL_ICJMPZ, ili, CC_NE, label);
      iltb.callfg = 1;
      chk_block(ili);
      ccff_info(MSGOPENMP, "OMP027", gbl.findex, gbl.lineno,
                "Cancellation point", NULL);
    }
    break;
  case IM_SINGLE:
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    if (!ll_ilm_is_rewriting()) {
      parsectCnt++;
      ccff_info(MSGOPENMP, "OMP010", gbl.findex, gbl.lineno,
                "Begin single region", NULL);
      single_thread = getPrivateTemp(DT_INT);
      in_single = getPrivateTemp(DT_INT);
      ili = genIntStore(single_thread, ad_icon(-1));
      chk_block(ili);
      ili = genIntStore(in_single, ad_icon(0));
      chk_block(ili);
      if (!gbl.outlined) {
        SCP(single_thread, SC_AUTO);
        SCP(in_single, SC_AUTO);
      }
      ili = ll_make_kmpc_single();
      sym = ILM_SymOPND(ilmp, 2);
      ili = ad3ili(IL_ICJMPZ, ili, CC_EQ, sym), iltb.callfg = 1;
      BIH_PARSECT(expb.curbih) = bihb.parsectfg = true;
      chk_block(ili);
    }
    break;
  case IM_EMASTER:
    if (outlinedCnt >= 1)
      break;
    ili = ll_make_kmpc_end_master();
    iltb.callfg = 1;
    chk_block(ili);
    ccff_info(MSGOPENMP, "OMP009", gbl.findex, gbl.lineno, "End master region",
              NULL);
    goto esect_shared;
  case IM_ESINGLE:
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }
    if (!ll_ilm_is_rewriting()) {
      int threadili;
      threadili = ll_get_gtid_val_ili();
      ili = genIntStore(single_thread, threadili);
      chk_block(ili);
      ili = genIntStore(in_single, ad_icon(1));
      chk_block(ili);
      ili = ll_make_kmpc_end_single();
      iltb.callfg = 1;
      chk_block(ili);
      ccff_info(MSGOPENMP, "OMP011", gbl.findex, gbl.lineno,
                "End single region", NULL);
    } else {
      break;
    }

  esect_shared:
    BIH_PARSECT(expb.curbih) = true;
    exp_label(ILM_SymOPND(ilmp, 1));
    parsectCnt--;
    if (parsectCnt <= 0)
      bihb.parsectfg = false;
    break;

  /* C, FORTRAN */
  case IM_BCOPYIN:
  case IM_ECOPYIN:
    if (!ll_ilm_is_rewriting()) {
      if (opc == IM_ECOPYIN) {
        if (XBIT(69, 0x80)) {
          makeCopyprivArray_tls(copysptr_list);
          sptrListFree(&copysptr_list);
          break;
        }

        addCopyinInplace(copysptr_list);
        sptrListFree(&copysptr_list);
        break;
      }
    }
    break;

  case IM_COPYIN:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    /* variable/common block to be copied */
    sym = ILM_SymOPND(ilmp, 1);
    tpv = MIDNUMG(sym);

    if (STYPEG(sym) == ST_CMBLK) {
      /* entire common block is being copied */
      size = SIZEG(sym);
      off = ad1ili(IL_ICON, stb.i0);
    } else if (SCG(sym) == SC_CMBLK) {
      /* a variable of the common block is being copied */
      size = size_of(DTYPEG(sym));
      /* locate common block */
      sym = MIDNUMG(sym);
      tpv = MIDNUMG(sym);
    }
    else if (SCG(sym) == SC_BASED && POINTERG(sym)) {
      pv = MIDNUMG(sym);
      if (SCG(pv) == SC_CMBLK) {
        /* f90 pointer or allocatable common block member:
         *
         * MIDNUM locates the user/compiler-created pointer;
         * its MIDNUM locates the common block.
         */
        int sdsptr;
        size = size_of(DTYPEG(pv));
        sdsptr = SDSCG(sym); /* $sd */
        if (sdsptr) {
          size += size_of(DT_ADDR);        /* $o */
          size += size_of(DTYPEG(sdsptr)); /* $sd */
        }
        ADDRTKNP(sym, 1);
        tpv = MIDNUMG(MIDNUMG(pv));
      } else {
        /* f90 pointer or allocatable:
         *
         * MIDNUM locates the user/compiler-created pointer;
         * its MIDNUM locates the variable's thread pointer vector
         * Could compute the size of that variable or just of the
         * pointer dtype ...
         */
        size = size_of(DT_ADDR);
#if DEBUG
        assert(size == size_of(DTYPEG(pv)),
               "COPYIN size incorrect for SC_BASED sym", sym, ERR_Fatal);
#endif
        ADDRTKNP(sym, 1);
        tpv = MIDNUMG(pv);
      }
    }
    else if (SCG(sym) == SC_BASED) {
      /* Cray pointee:
       * MIDNUM locates the variable's thread pointer vector and
       * its MIDNUM locates the user/compiler-created pointer that's
       * actually copied.  Could compute the size of that variable
       * or just of the pointer dtype ...
       */
      size = size_of(DT_ADDR);
#if DEBUG
      assert(size == size_of(DTYPEG(MIDNUMG(MIDNUMG(sym)))),
             "COPYIN size incorrect for SC_BASED sym", sym, ERR_Fatal);
#endif
      ADDRTKNP(sym, 1);
    } else {
      /* regular user var being copied */
      size = size_of(DTYPEG(sym));
      ADDRTKNP(sym, 1);
    }
    /* false: Because we want to always use the vector/cache (tpv)
     * and not the data item from the cache.
     */
    sz = ad_kconi(size);
    sptrListAdd(&copysptr_list, tpv, sz, false, 0, 0, sym);
    break;

#ifdef IM_COPYIN_A
  case IM_COPYIN_A:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    /* allocatable to be copied */
    sym = ILM_SymOPND(ilmp, 1);
    /* MIDNUM locates the user/compiler-created pointer; its MIDNUM locates the
     * variable's thread pointer vector */
    pv = MIDNUMG(sym);
    if (SCG(sym) == SC_BASED && POINTERG(sym)) {
      if (SCG(pv) == SC_CMBLK) {
        /* f90 pointer or allocatable common block member:
         *
         * MIDNUM(sym)locates the user/compiler-created pointer (pv)
         * which is a member of the common block.  Its MIDNUM locates
         * the common block.
         */
        pv = MIDNUMG(pv); /* locate common block */
      }
    }
    sz = ILI_OF(ILM_OPND(ilmp, 2));
    ADDRTKNP(sym, 1);
    tpv = MIDNUMG(pv);

    sptrListAdd(&copysptr_list, tpv, sz, false, 0, 0, sym);
    break;
#endif

  case IM_BCOPYPRIVATE:
  case IM_ECOPYPRIVATE:
    if (!ll_ilm_is_rewriting()) {
      if (opc == IM_ECOPYPRIVATE) {
        SPTR sptr_addr = makeCopyprivArray(copysptr_list, true);
        addr = sptr_addr;
        stili = genIntLoad(in_single);

        /* c++ will set up assign_rou from IM_COPYPRIVATE_CL (_P) */
        if (!assign_rou) {
          assign_rou = ad_acon(mkfunc("_mp_copypriv_kmpc"), 0);
        }

        ili = ll_make_kmpc_copyprivate(sptr_addr, stili, assign_rou);

        assign_rou = 0;
        iltb.callfg = 1;
        chk_block(ili);
        sptrListFree(&copysptr_list);
      }
    }
    break;
  case IM_COPYPRIVATE_CL_P:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    /* C++ ONLY class copyprivate */
    argilm = ILM_OPND(ilmp, 2);
    sym = ILM_SymOPND((ILM *)(ilmb.ilm_base + argilm), 1);
    assign_rou = ad_acon(ILM_SymOPND(ilmp, 3), 0);
    if (DTY(DTYPEG(sym)) == TY_ARRAY) {
      element_size = getElemSize(DTYPEG(sym));
      num_elements = extent_of(DTYPEG(sym));
      size = num_elements * element_size; /* Total size required for
                                                        llvm memcpy */
    } else {
      size = size_of(DTYPEG(sym)); /* used for POD */
    }
    sz = ad_kconi(size);
    sptrListAdd(&copysptr_list, sym, sz, false, assign_rou, 0, sym);
    ADDRTKNP(sym, 1);
    break;

#ifdef IM_COPYPRIVATE_PA
  case IM_COPYPRIVATE_PA:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    argilm = ILM_OPND(ilmp, 2);
    sym = ILM_SymOPND((ILM *)(ilmb.ilm_base + argilm), 1);
    ili = ILI_OF(ILM_OPND(ilmp, 3));
    ili = sel_iconv(ili, 1);

    sptrListAdd(&copysptr_list, sym, ili, false, 0, 0, sym);
    break;
#endif

  case IM_COPYPRIVATE_P:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    argilm = ILM_OPND(ilmp, 2);
    sym = ILM_SymOPND((ILM *)(ilmb.ilm_base + argilm), 1);
    sz = 0;
    if (SCG(sym) == SC_DUMMY && DTY(DTYPEG(sym)) != TY_PTR &&
        (DDTG(DTYPEG(sym)) == DT_ASSCHAR)) {
      sz = charlen(sym);
    }
    if (sz == 0) {
      size = size_of(DTYPEG(sym));
      sz = ad_kconi(size);
    }
    sptrListAdd(&copysptr_list, sym, sz, false, 0, 0, sym);
    break;

  case IM_COPYPRIVATE_CL:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    /* C++ ONLY class copyprivate */
    /* variable/class to be copied out */
    sym = ILM_SymOPND(ilmp, 2);
    assign_rou = ad_acon(ILM_SymOPND(ilmp, 3), 0);
    if (DTY(DTYPEG(sym)) == TY_ARRAY) {
      element_size = getElemSize(DTYPEG(sym));
      num_elements = extent_of(DTYPEG(sym));
      size = num_elements * element_size; /* Total size required for
                                             llvm memcpy */
    } else {
      size = size_of(DTYPEG(sym)); /* used for POD */
    }
    sz = ad_kconi(size);
    sptrListAdd(&copysptr_list, sym, sz, false, assign_rou, 0, sym);
    ADDRTKNP(sym, 1);
    break;

  case IM_COPYPRIVATE:
    if (ll_ilm_is_rewriting()) {
      break;
    }

    /* variable/common block to be copied out */
    sym = ILM_SymOPND(ilmp, 2);
    is_cmblk = false;

    if (STYPEG(sym) == ST_CMBLK) {
      /* Entire common block */
      size = SIZEG(sym);
      sym = MIDNUMG(sym);
      is_cmblk = true;
    } else if (SCG(sym) == SC_CMBLK) {
      /* Var in common block */
      size = size_of(DTYPEG(sym));
      sym = MIDNUMG(sym);
      sym = MIDNUMG(sym);
      is_cmblk = true;
    } else {
      size = size_of(DTYPEG(sym));
    }
    sz = ad_kconi(size);
    sptrListAdd(&copysptr_list, sym, sz, is_cmblk, 0, 0, sym);
    ADDRTKNP(sym, 1);
    break;

#ifdef IM_FLUSH
  case IM_FLUSH:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    ili = ll_make_kmpc_flush();
    iltb.callfg = 1;
    chk_block(ili);
    break;
#endif
  case IM_TASKGROUP:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    ili = ll_make_kmpc_taskgroup();
    iltb.callfg = 1;
    chk_block(ili);
    break;
  case IM_ETASKGROUP:
    if (ll_ilm_is_rewriting()) {
      break;
    }
    ili = ll_make_kmpc_end_taskgroup();
    iltb.callfg = 1;
    chk_block(ili);
    break;

  case IM_BTASK:
  case IM_BTASKLOOP:
    incrOutlinedCnt();
    if (outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    taskCnt++;
    wr_block();
    cr_block();

    if (gbl.outlined)
      expb.sc = SC_PRIVATE;
    else
      expb.sc = SC_AUTO;
    taskBv = ILM_OPND(ilmp, 2);
    taskIfv = ILI_OF(ILM_OPND(ilmp, 3));
    taskAllocSptr = getnewccsym('z', GBL_CURRFUNC, ST_VAR);
    SCP(taskAllocSptr, expb.sc);
    DTYPEP(taskAllocSptr, DT_CPTR);
    taskFlags = getPrivateTemp(DT_INT);
    SCP(taskFlags, expb.sc);
    /* Note: kmpc(5.0) does not use mergeable and priority flags */
    if (taskBv & MP_TASK_FINAL) {
      const int kmpc_flags = mp_to_kmpc_tasking_flags(taskBv);

      /* Expand the 'final' expression */
      const SPTR lab = getlab();
      RFCNTI(lab);
      ili = ad3ili(IL_ICJMPZ, ILI_OF(ILM_OPND(ilmp, 4)), CC_EQ, lab);
      chk_block(ili);

      /* In the branch: update the flags variable */
      ili = genIntStore(taskFlags, ad_icon(kmpc_flags));
      chk_block(ili);
      wr_block();
      exp_label(lab);
    } else {
      if (taskBv & MP_TASK_UNTIED) {
        ili = ad_icon(0);
      } else {
        ili = ad_icon(1);
      }
      ili = genIntStore(taskFlags, ili);
      chk_block(ili);
    }
    wr_block();

    /* mark for __kmpc_task_alloc */
    taskbih = expb.curbih;

    cr_block();
    /* mark block to place taskloop vars */
    NEED(mppgcnt + 1, mppgbih, int, mppgBihSiz, mppgBihSiz + 16);
    mppgbih[mppgcnt] = expb.curbih;
    mppgcnt++;

    /* create task here because we want to set ENCLFUNC for all private
     * variables, including loop variables(for taskloop)*/
    task = llGetTask(scopeSptr);
    taskFnsptr = ll_make_outlined_task(uplevel_sptr, scopeSptr);
    llmp_task_set_fnsptr(task, taskFnsptr);
    if (!PARENCLFUNCG(scopeSptr))
      PARENCLFUNCP(scopeSptr, taskFnsptr);
    if (opc == IM_BTASKLOOP) {
      /* Reserve space for taskloop vars & lastiter on task_alloc ptr.  */
      TASK_LPVAR_OFFSET = llmp_task_add_loopvar(task, 4, DT_INT8);
      taskdup = 0;
      taskLoopCnt++;

      if (taskBv & MP_TASK_IF) {
        int tmp0, tmp1;
        tmp0 = ad_icon(0);
        taskIfv = sel_iconv(taskIfv, 0);
        ili = ad3ili(IL_ICMP, taskIfv, tmp0, CC_EQ);
        tmp1 = ad_icon(1);
        TASKLP_IF = ad3ili(IL_ISELECT, ili, tmp1, tmp0);
      } else {
        TASKLP_IF = ad_icon(1);
      }
      if (taskBv & MP_TASK_NOGROUP) {
        TASKLP_NOGROUP = ad_icon(1);
      } else {
        TASKLP_NOGROUP = ad_icon(0);
      }
      if (taskBv & MP_TASK_GRAINSIZE) {
        TASKLP_SCHED = ad_icon(1);
      } else if (taskBv & MP_TASK_NUM_TASKS) {
        TASKLP_SCHED = ad_icon(2);
      } else {
        TASKLP_SCHED = ad_icon(0);
      }
      TASKLP_GRAINSIZE = ILI_OF(ILM_OPND(ilmp, 6));

      ccff_info(MSGOPENMP, "OMP028", gbl.findex, gbl.lineno, "Begin taskloop",
                NULL);
      ll_write_ilm_header(taskFnsptr, curilm);
    } else {
      ccff_info(MSGOPENMP, "OMP016", gbl.findex, gbl.lineno, "Begin task",
                NULL);
      ll_write_ilm_header(taskFnsptr, curilm);
    }

    expb.sc = SC_PRIVATE;
    break;
#ifdef IM_BTASKDUP
  case IM_BTASKDUP:
    if (taskCnt != 1 || outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    /* for normal task: Stop writint to temp parfile and evaluate
     *                  ILMs.
     * for taskloop: write ilms between IM_BTASKDUP and IM_ETASKDUP
     *               to taskdup routine.  Also don't write to
     *               temp parfile because we want to evaluate
     *               ILMS in between in host routine too.
     */
    outlinedCnt = 0;
    if (taskdup == 0) { /* allow nested taskdup */
      /* make sure we write after taskAlloc is allocated */
      wr_block();
      cr_block();

      if (taskLoopCnt) {
        ilm_outlined_pad_ilm(curilm);
        unsetRewritingILM();
        start_taskdup(taskFnsptr, curilm);
      } else {
        ilm_outlined_pad_ilm(curilm);
        unsetRewritingILM();
      }
      if (gbl.outlined)
        expb.sc = SC_PRIVATE;
      else
        expb.sc = SC_AUTO;
    } else {
      llWriteNopILM(-1, curilm, 0);
    }
    taskdup++;
    break;
  case IM_ETASKDUP:
    if (taskCnt != 1 || outlinedCnt > 0) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    --taskdup;
    if (taskdup == 0) {
      if (taskLoopCnt) {
        stop_taskdup(taskFnsptr, curilm);
      }
      restartRewritingILM(curilm);
      outlinedCnt = 1;
      if (gbl.outlined)
        expb.sc = SC_PRIVATE;
    } else {
      llWriteNopILM(-1, curilm, 0);
    }
    break;
#endif

#ifdef IM_TASKFIRSTPRIV
  case IM_TASKFIRSTPRIV:
    if (taskCnt != 1 || outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    {
      /* Must set ADDRESSG field in caller.
       * The reason to do it in caller is that
       * currently some allocataion is done before
       * we emit IM_TASKFIRSTPRIV.
       * We then can get its address in callee
       * for the allocation.
       */
      task = llGetTask(scopeSptr);
      sym = ILM_SymOPND(ilmp, 1);
      sptr = ILM_SymOPND(ilmp, 2);
      offset = llmp_task_add_private(task, sym, sptr);
      ADDRESSP(sptr, offset);
      ENCLFUNCP(sptr, taskFnsptr);
      llWriteNopILM(-1, curilm, 0);
    }
    break;
#endif
#ifdef IM_TASKPRIVATE
  case IM_TASKPRIVATE:
    if (taskCnt != 1 || outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    {
      task = llGetTask(scopeSptr);
      sym = ILM_OPND(ilmp, 1);
      sptr = ILM_OPND(ilmp, 2);
      offset = llmp_task_add_private(task, sym, sptr);
      ADDRESSP(sptr, offset);
      ENCLFUNCP(sptr, taskFnsptr);
      llWriteNopILM(-1, curilm, 0);
    }
    break;
#endif

#ifdef IM_TASKLOOPVARS
  case IM_TASKLOOPVARS:
    if (taskLoopCnt != 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    /* store the ilm rewrite so that we can work on taskloop bounds/stride */
    if (outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
    } else if (outlinedCnt == 1) {
      outlinedCnt = 0;
      ilm_outlined_pad_ilm(curilm);
      unsetRewritingILM();
      if (gbl.outlined)
        expb.sc = SC_PRIVATE;
      else
        expb.sc = SC_AUTO;
      resetMppBih(SET_MPPBIH, IS_PREVMPPG);
    }
    break;
#endif
  case IM_TASKLOOPREG:
    if (taskLoopCnt != 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    if (outlinedCnt > 0) {
      ll_rewrite_ilms(-1, curilm, 0);
    } else {
      restartRewritingILM(curilm);
      outlinedCnt = 1;
      TASK_LB = ILI_OF(ILM_OPND(ilmp, 1));
      TASK_UB = ILI_OF(ILM_OPND(ilmp, 2));
      TASK_ST = ILI_OF(ILM_OPND(ilmp, 3));
      if (gbl.outlined)
        expb.sc = SC_PRIVATE;
      resetMppBih(RESTORE_MPPBIH, IS_PREVMPPG);
    }
    break;

  case IM_ETASKLOOPREG:
    if (taskCnt != 1 || outlinedCnt > 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    if (outlinedCnt == 1) {
      llWriteNopILM(-1, curilm, 0);
    }
    break;

  case IM_ETASKLOOP:
    if (outlinedCnt == 1) {
      /* do following so that we get the ILM right in case ILM is
       * written in the same ILM block.
       */
      ilm_outlined_pad_ilm(curilm);
      ccff_info(MSGOPENMP, "OMP029", gbl.findex, gbl.lineno, "End taskloop",
                NULL);
      taskLoopCnt--;
    }
    decrOutlinedCnt();
    if (outlinedCnt >= 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
    goto shared_etask;

  case IM_ETASK:
    if (outlinedCnt == 1) {
      ilm_outlined_pad_ilm(curilm);
      ccff_info(MSGOPENMP, "OMP017", gbl.findex, gbl.lineno, "End task", NULL);
    }
    decrOutlinedCnt();
    if (outlinedCnt >= 1) {
      ll_rewrite_ilms(-1, curilm, 0);
      break;
    }
  shared_etask:
    /* Insert kmpc_task_alloc here because default firstprivate assignment can
     * be done after IM_ETASKREG/ETASKLOOPREG and we need to collect
     * the size of all firstprivate vars and pass to kmpc.
     */
    if (gbl.outlined)
      expb.sc = SC_PRIVATE;
    else
      expb.sc = SC_AUTO;
    {
      SPTR lab;
      int end_lab;
      SPTR s_scope;

      if (opc == IM_ETASKLOOP) {
        /* must be called after decrOutlinedCnt so that outlined
         * function ILMs are done emitted before taskdup is emitted.
         */
        finish_taskdup_routine(curilm, taskFnsptr,
                               TASK_LPVAR_OFFSET + (zsize_of(DT_INT8) * 3));
      }

      resetTaskBih(SET_MPPBIH);
      /* Load args first */
      s_scope = scopeSptr;
      scopeSptr = (SPTR)OUTLINEDG(taskFnsptr);     // ???
      taskAllocSptr = ll_make_kmpc_task_arg(
          taskAllocSptr, taskFnsptr, scopeSptr, taskFlags);
      ll_load_outlined_args(scopeSptr, taskFnsptr, false);
      /* Load taskloop vars and store onto task_alloc ptr
       * Also get its address on task_alloc ptr to pass
       * to __kmpc_taskloop.
       */
      if (opc == IM_ETASKLOOP) {
        int nme, task_ili;
        ILI_OP ld, st;
        MSZ msz;
        INT offset = 0;

        ili = ad_acon(taskAllocSptr, offset);
        nme = addnme(NT_VAR, taskAllocSptr, (INT)0, 0);
        task_ili = ad2ili(IL_LDA, ili, nme);
        ldst_msz(DT_INT8, &ld, &st, &msz);

        offset = ad_aconi(TASK_LPVAR_OFFSET);
        ili = ad3ili(IL_AADD, task_ili, offset, 0);
        TASKLP_LB = ili;
        ili = ad4ili(st, TASK_LB, ili, nme, msz);
        chk_block(ili);

        offset = ad_aconi(TASK_LPVAR_OFFSET + zsize_of(DT_INT8));
        ili = ad3ili(IL_AADD, task_ili, offset, 0);
        TASKLP_UB = ili;
        ili = ad4ili(st, TASK_UB, ili, nme, msz);
        chk_block(ili);

        offset = ad_aconi(TASK_LPVAR_OFFSET + (zsize_of(DT_INT8) * 2));
        ili = ad3ili(IL_AADD, task_ili, offset, 0);
        ili = ad4ili(st, TASK_ST, ili, nme, msz);
        TASKLP_ST = TASK_ST;

        iltb.callfg = 1; /* Call task */
        chk_block(ili);
      }

      resetTaskBih(RESTORE_MPPBIH);
      scopeSptr = s_scope;

      /* If 'if' clause is used, this is the false branch, if (0) then... */
      end_lab = ILM_OPND(ilmp, 1);
      if (opc == IM_ETASK) {
        if (taskBv & MP_TASK_IF) {
          lab = getlab();
          RFCNTI(lab);
          ili = ad3ili(IL_ICJMPZ, taskIfv, CC_NE, lab);
          chk_block(ili);

          iltb.callfg = 1; /* Begin */
          ili = ll_make_kmpc_task_begin_if0(taskAllocSptr);
          chk_block(ili);

          iltb.callfg = 1; /* Call task */
          ili = ll_make_outlined_task_call(taskFnsptr, taskAllocSptr);
          chk_block(ili);

          iltb.callfg = 1; /* End */
          ili = ll_make_kmpc_task_complete_if0(taskAllocSptr);
          chk_block(ili);

          /* Create and jump to an end label at the end of the task */
          RFCNTI(end_lab);
          ili = ad1ili(IL_JMP, end_lab);
          chk_block(ili);

          exp_label(lab);
        }
      }
      if (opc == IM_ETASK) {
        /* Make api call */
        ili = ll_make_kmpc_task(taskAllocSptr);
      } else {
        TASKLP_TASK = ad2ili(IL_LDA, ad_acon(taskAllocSptr, 0),
                             addnme(NT_VAR, taskAllocSptr, 0, 0));
        /* FIXME: if there is no firstprivate and lastprivate
                  don't pass taskdup - performance issue maybe?
         */
        if (TASKDUPG(taskFnsptr)) {
          ll_process_routine_parameters(TASKDUPG(taskFnsptr));
          TASKLP_TASKDUP = ad_acon(TASKDUPG(taskFnsptr), 0);
        } else {
          TASKLP_TASKDUP = 0;
        }
        ili = ll_make_kmpc_taskloop(TASKLPARGS);
        clearTaskloopInfo();
      }
      iltb.callfg = 1;
      chk_block(ili);
    }
    wr_block();
    cr_block();

    /* reset once done processing a task - need this for mk_address to work */
    taskAllocSptr = SPTR_NULL;

    taskCnt--;
    mppgcnt--;
    taskbih = 0;

    exp_label(ILM_SymOPND(ilmp, 1));
    break;

  case IM_TASKWAIT:
    if (ll_ilm_is_rewriting())
      break;
    ccff_info(MSGOPENMP, "OMP018", gbl.findex, gbl.lineno, "Taskwait", NULL);
    ili = ll_make_kmpc_task_wait();
    iltb.callfg = 1;
    chk_block(ili);
    break;

  case IM_TASKYIELD:
    if (ll_ilm_is_rewriting())
      break;
    ccff_info(MSGOPENMP, "OMP019", gbl.findex, gbl.lineno, "Taskyield", NULL);
    ili = ll_make_kmpc_task_yield();
    iltb.callfg = 1;
    chk_block(ili);
    break;

  case IM_BMPPG:
    if (ll_ilm_is_rewriting())
      break;

    /* create a block for kmpc_task_alloc */

    NEED(mppgcnt + 1, mppgbih, int, mppgBihSiz, mppgBihSiz + 16);
    mppgbih[mppgcnt] = expb.curbih;
    mppgcnt++;

    /* for task call */
    wr_block();
    cr_block();
    break;

  case IM_EMPPG:
    if (ll_ilm_is_rewriting())
      break;
    mppgcnt--;
    break;

  case IM_BAMPPG:
    if (ll_ilm_is_rewriting())
      break;

    resetMppBih(SET_MPPBIH, IS_PREVMPPG);

    break;

  case IM_EAMPPG:
    if (ll_ilm_is_rewriting())
      break;

    resetMppBih(RESTORE_MPPBIH, IS_PREVMPPG);
    break;
  case IM_BTARGETUPDATE:
  case IM_BTARGETDATA:
  case IM_TARGETENTERDATA:
  case IM_TARGETEXITDATA:
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if(!flg.omptarget)
      break;
    dotarget = ILI_OF(ILM_OPND(ilmp, 1));
    beg_label = getlab();
    end_label = getlab();

    dotarget = ad3ili(IL_ICJMPZ, dotarget, CC_EQ, end_label);
    RFCNTI(end_label);
    chk_block(dotarget);

    wr_block();
    cr_block();
    exp_label(beg_label);

    if(!IS_OMP_DEVICE_CG)
      exp_ompaccel_targetdata(ilmp, curilm, opc);

    exp_label(end_label);
#endif
    break;
  case IM_ETARGETUPDATE:
  case IM_ETARGETDATA:
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if(!flg.omptarget || IS_OMP_DEVICE_CG) break;
    OMPACCEL_TINFO *targetinfo;
    wr_block();
    cr_block();
    if(opc == IM_ETARGETDATA) {
      targetinfo = ompaccel_tinfo_current_get_targetdata();
      int ili = ll_make_tgt_target_data_end(OMPACCEL_DEFAULT_DEVICEID, targetinfo);
      chk_block(ili);
    }
    iltb.callfg = 1;
    wr_block();
    cr_block();
#endif
    break;
  case IM_BDISTRIBUTE:
    if (ll_ilm_is_rewriting())
      ccff_info(MSGOPENMP, "OMP024", gbl.findex, gbl.lineno,
                "Distribute loop activated", NULL);
    break;
  case IM_EDISTRIBUTE:
    if (ll_ilm_is_rewriting())
      ccff_info(MSGOPENMP, "OMP025", gbl.findex, gbl.lineno,
                "Distribute loop terminated", NULL);
    break;
    break;

  case IM_MP_ATOMIC:
    if (ll_ilm_is_rewriting())
      break;
    wr_block();
    cr_block();
    bihb.csfg = BIH_CS(expb.curbih) = true;
    break;
  case IM_MP_ENDATOMIC:
    if (ll_ilm_is_rewriting())
      break;
    wr_block();
    cr_block();
    bihb.csfg = BIH_CS(expb.curbih) = false;
    break;

  case IM_MP_ATOMICREAD:
    if (ll_ilm_is_rewriting())
      break;
    ILM_RESULT(curilm) = exp_mp_atomic_read(ilmp);
    break;

  case IM_MP_ATOMICWRITE:
    if (ll_ilm_is_rewriting())
      break;
    exp_mp_atomic_write(ilmp);
    break;
  case IM_MP_ATOMICUPDATE:
    if (ll_ilm_is_rewriting())
      break;
    exp_mp_atomic_update(ilmp);
    break;
  case IM_MP_ATOMICCAPTURE:
    if (ll_ilm_is_rewriting())
      break;
    exp_mp_atomic_capture(ilmp);
    break;
  case IM_TASKLASTPRIV:
    if (ISTASKDUPG(GBL_CURRFUNC)) {
      INT offset;
      int offset_sptr, ioffset, acon, load, nme;
      SPTR secarg = ll_get_hostprog_arg(GBL_CURRFUNC, 1);
      SPTR lastitr = ll_get_hostprog_arg(GBL_CURRFUNC, 3);
      offset_sptr = ILM_OPND(ilmp, 1);
      offset = get_isz_cval(offset_sptr);
      /* load from 3rd argument(int litr) into 1st argument at offset */
      acon = ad_acon(lastitr, 0);
      load = ad3ili(IL_LD, acon, addnme(NT_VAR, lastitr, 0, 0), MSZ_WORD);

      nme = addnme(NT_VAR, secarg, 0, 0);
      acon = mk_address(secarg);
      ioffset = ad_aconi(offset);
      acon = ad3ili(IL_AADD, acon, ioffset, 0);
      ili =
          ad4ili(IL_ST, load, acon, addnme(NT_IND, lastitr, nme, 0), MSZ_WORD);

      chk_block(ili);
    }
    break;

#ifdef IM_TASKREG
  case IM_TASKREG:
    break;
#endif
    /* unused: to be removed in future release */
#ifdef IM_ETASKREG
  case IM_ETASKREG:
    break;
#endif
#ifdef IM_ETASKFIRSTPRIV
  case IM_ETASKFIRSTPRIV:
    break;
#endif
#ifdef OMP_OFFLOAD_LLVM
    case IM_MP_REDUCTIONITEM:
      if (flg.omptarget && gbl.ompaccel_intarget)
        exp_ompaccel_reductionitem(ilmp, curilm);
      break;
    case IM_MP_BREDUCTION:
    case IM_MP_EREDUCTION:
      break;
    case IM_MP_TARGETLOOPTRIPCOUNT:
      if(flg.omptarget)
        exp_ompaccel_looptripcount(ilmp, curilm);
      break;
    case IM_MP_MAP:
      if(flg.omptarget && !(IS_OMP_DEVICE_CG || gbl.ompaccel_intarget))
        exp_ompaccel_map(ilmp, curilm, outlinedCnt);
      break;
    case IM_MP_EMAP:
    if(flg.omptarget && !(IS_OMP_DEVICE_CG || gbl.ompaccel_intarget)) {
      exp_ompaccel_emap(ilmp, curilm);
    }
    break;
    case IM_MP_TARGETMODE:
      if(flg.omptarget) {
        ompaccel_tinfo_set_mode_next_target((OMP_TARGET_MODE)ILM_OPND(ilmp, 1));
        target_ili_num_teams = ILI_OF(ILM_OPND(ilmp, 2));
        target_ili_thread_limit = ILI_OF(ILM_OPND(ilmp, 3));
        target_ili_num_threads = ILI_OF(ILM_OPND(ilmp, 4));
      }
    break;
#endif /* end #ifdef OMP_OFFLOAD_LLVM */
    default:
      interr("exp_smp: unsupported opc", opc, ERR_Severe);
      break;
  }

#endif /* end #ifdef IM_BPAR */
}

#ifdef FLANG2_EXPSMP_UNUSED
/* opc: IL_DFRDP / IL_DFRSP, depending on result type of call */
/* Return the ili of a call to a function that returns a result, using ili
 * callili, followed by freeing of the appropriate argument registers with
 * opcode opc. */
static int
makeCallResult(ILI_OP opc, int callili)
{
  int rg;
  int ili;

  assert(IL_DFRIR <= opc && opc <= IL_DFRAR, "makeCallResult: invalid opcode",
         opc, ERR_Fatal);

  switch (opc) {
  case IL_DFRIR:
    rg = IR_RETVAL;
    break;
  case IL_DFRSP:
    rg = SP_RETVAL;
    break;
  case IL_DFRDP:
    rg = DP_RETVAL;
    break;
  case IL_DFRAR:
    rg = AR_RETVAL;
    break;
  default:
    interr("makeCallResult: invalid register free opcode", opc, ERR_Fatal);
  }
  ili = ad2ili(opc, callili, rg);

  return ili;
}
#endif

SPTR
lcpu_temp(SC_KIND sc)
{
  SPTR sym;
  char name[10];
  static int lcpu_cnt = 0; /* counter for lcpu temporaries */

  strcpy(name, ".lcp");
  sprintf(&name[4], "%05d", lcpu_cnt);
  lcpu_cnt++;
  sym = getcctemp_sc(name, ST_VAR,
                     sc); /* lcpu variable, 1 per critical section */
  DTYPEP(sym, DT_INT);
  return sym;
}

SPTR
ncpus_temp(SC_KIND sc)
{
  SPTR sym;
  char name[10];
  static int ncpus_cnt = 0; /* counter for ncpus temporaries */

  strcpy(name, ".ncp");
  sprintf(&name[4], "%05d", ncpus_cnt);
  ncpus_cnt++;
  sym = getcctemp_sc(name, ST_VAR,
                     sc); /* ncpus variable, 1 per critical section */
  DTYPEP(sym, DT_INT);
  return sym;
}

static int
addMpBcsNest(void)
{
  int ili;
  ili = makeCall("_mp_bcs_nest_red", IL_JSR, 0);
  return ili;
}

static int
addMpEcsNest(void)
{
  int ili;
  ili = makeCall("_mp_ecs_nest_red", IL_JSR, 0);
  return ili;
}

int
add_mp_p(SPTR semaphore)
{
  int ili;
  ili = ll_make_kmpc_critical(semaphore);
  return ili;
}

int
add_mp_v(SPTR semaphore)
{
  int ili;
  ili = ll_make_kmpc_end_critical(semaphore);
  return ili;
}

int
add_mp_penter(int ispar)
{
  return 0;
}

int
add_mp_pexit(void)
{
  return 0;
}

int
add_mp_ncpus(void)
{
  return 0;
}

int
add_mp_ncpus3(void)
{
  return 0;
}

int
add_mp_lcpu(void)
{
  return 0;
}

int
add_mp_barrier2(void)
{
  return 0;
}

/* for compiler generated routines that have referenced the threadprivate
   variables, but do not need the kmpc_threadprivate_cached set up
 */
void
clear_tplnk(void)
{
  int sym;
  for (sym = gbl.threadprivate; sym > NOSYM; sym = TPLNKG(sym)) {
    TPLNKP(sym, 0);
    THPRVTOPTP(sym, 0); /* so much trouble clear this too */
  }
  gbl.threadprivate = NOSYM;
}

/** \brief Generate any mp-specific prologue for a function.
 */
void
exp_mp_func_prologue(bool process_tp)
{
  SPTR sym;
  int tmpthread;
  int func;
  int cond_ili = 0;
  int bih = 0;

#ifdef CUDAG
  if (CUDAG(GBL_CURRFUNC) == CUDA_GLOBAL || CUDAG(GBL_CURRFUNC) == CUDA_DEVICE)
    return;
#endif
  if (process_tp) {
    for (sym = gbl.threadprivate; sym > NOSYM; sym = TPLNKG(sym)) {
      /* For each threadprivate common, must 'declare' the threads'
       * copies by calling:
       * _kmpc_threadprivate_cached(&cmn_block, &cmn_vector, size(cmn_block))
       */
      tmpthread = allocThreadprivate(sym, &cond_ili);
      if (gbl.outlined)
        func = gbl.currsub;
      else
        func =
            gbl.entries; /* this does not really work for entry because for llvm
                          * entry are done very late and in separate function.
                          */
      for (func = gbl.currsub; func != NOSYM; func = SYMLKG(func)) {
        if (EXPDBG(8, 256))
          fprintf(gbl.dbgfil, "---_kmpc_threadprivate_cached: in %s ---\n",
                  SYMNAME(func));

        bih = expb.curbih = findEnlabBih(func);
        rdilts(expb.curbih); /* get block after entry */
        expb.curilt = 0;
        iltb.callfg = 1;
        chk_block(tmpthread);
        wrilts(expb.curbih);
      }
      THPRVTOPTP(sym, 0);
    }
  }

  if (has_multiple_entries(GBL_CURRFUNC)) {
    if (bih)
      ll_save_gtid_val(bih);
    for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
      bih = expb.curbih = findEnlabBih(func);
      if (bih)
        ll_save_gtid_val(bih);
    }
  } else {
    ll_save_gtid_val(bih);
  }
}

static int
decrOutlinedCnt(void)
{
  outlinedCnt--;
  if (outlinedCnt == 0) {
      ll_write_ilm_end();
    unsetRewritingILM();
  }
  return outlinedCnt;
}

static int
incrOutlinedCnt(void)
{
  parCnt++;
  if (parCnt > maxOutlinedCnt)
    maxOutlinedCnt = parCnt;
  outlinedCnt++;
  return outlinedCnt;
}

#ifdef FLANG2_EXPSMP_UNUSED
static int
getOutlinedTemp(char *pfx, int dtype)
{
  char name[32];
  int sym;

  sprintf(name, "%s%05d", pfx, maxOutlinedCnt + outlinedCnt);
  sym = getcctemp_sc(name, ST_VAR, expb.sc);
  DTYPEP(sym, DT_INT);
  return sym;
}
#endif

static int
isUnnamedCs(int sem)
{

  if (strcmp(SYMNAME(MIDNUMG(sem)), "__cs_unspc") == 0)
    return 1;
  return 0;
}

static int
addMpUnp(void)
{
  int ili;
  ili = ll_make_kmpc_critical(SPTR_NULL);
  return ili;
}

static int
addMpUnv(void)
{
  int ili;
  ili = ll_make_kmpc_end_critical(SPTR_NULL);
  return ili;
}

int
_make_mp_get_threadprivate(int data_ili, int size_ili, int cache_ili)
{
  int argili, ili, con;
  int null_arg;
  INT tmp[2];
  tmp[0] = 0;
  tmp[1] = 0;
  con = getcon(tmp, DT_INT);
  null_arg = ad1ili(IL_ACON, con);

  argili = jsrAddArg(0, IL_ARGAR, cache_ili);
  mk_prototype("_mp_get_threadprivate", NULL, DT_CPTR, 5, DT_CPTR, DT_INT,
               DT_CPTR, DT_INT8, DT_CPTR);
  size_ili = sel_iconv(size_ili, 1);
  argili = jsrAddArg(argili, IL_ARGKR, size_ili);
  argili = jsrAddArg(argili, IL_ARGAR, data_ili);
  argili = jsrAddArg(argili, IL_ARGIR, ll_get_gtid_val_ili());
  argili = jsrAddArg(argili, IL_ARGAR, null_arg);
  ili = makeCall("_mp_get_threadprivate", IL_QJSR, argili);
  ili = genretvalue(ili, IL_DFRAR);
  return ili;
}

/** \brief C and Fortran threadprivate : for simple POD */
static int
allocThreadprivate(SPTR sym, int *tmpthr)
{
  SPTR cm;
  int size;
  int adr_vector;
  int adr_cm;

  cm = MIDNUMG(sym); /* corresponding common block  or threadprivate var */
  if (STYPEG(cm) == ST_CMBLK) {
    adr_cm = ad_acon(CMEMFG(cm), 0); /* &cmn_block */
    size = ad_icon((INT)SIZEG(cm));
  }
  else if (SCG(cm) == SC_BASED && POINTERG(cm)) {
    /*
     * Cannot rely on the SYMLK chain appearing as
     *     $p -> $o -> $sd
     * Apparently, these links only occur for the
     * pointer's internal variables if the pointer
     * does not have the SAVE attribute.  Without
     * these fields, the correct size of the threads'
     * copies cannot be computed.
     * Just explicitly look for the internal pointer
     * and descriptor. If the descriptor is present,
     * can assume that there is an offest variable which
     * only needs to be accounted for in the size
     * computation of the threads' copies.
     * Setup up the MIDNUM fields as follows where
     * foo is the symtab entry which has the POINTER
     * flag set:
     *    foo    -> foo$p
     *    TPpfoo -> foo
     *    foo$p  -> TPpfoo
     *    foo$sd -> TPpfoo
     * Note that foo's SDSC -> foo$sd.
     * Before we had:
     *    foo    -> TPpfoo
     *    TPpfoo -> foo$p
     * which is a problem for computing the size
     * when starting with TPpfoo.
     */
    SPTR tptr;
    int sdsptr;
    tptr = MIDNUMG(cm);
    adr_cm = ad_acon(tptr, 0); /* &tp_var */
    size = size_of(DTYPEG(tptr));
    sdsptr = SDSCG(cm); /* $sd */
    if (sdsptr) {
      size += size_of(DT_ADDR);        /* $o */
      size += size_of(DTYPEG(sdsptr)); /* $sd */
    }
    size = ad_icon(size);
  }
  else if (DTY(DTYPEG(cm)) == TY_PTR) {
    /*
     * Given the above code for POINTER, this code is
     * probably dead, but leave it just in case.
     */
    adr_cm = ad_acon(cm, 0); /* &tp_var */
    size = size_of(DTYPEG(cm));
    if (SYMLKG(cm) != NOSYM) {
      size += size_of(DTYPEG(SYMLKG(cm))); /* $o */
      if (SYMLKG(SYMLKG(cm)) != NOSYM) {
        size += size_of(DTYPEG(SYMLKG(SYMLKG(cm)))); /* $sd */
      }
    }
    size = ad_icon(size);
  } else if (SCG(sym) == SC_BASED) {
    adr_cm = ad_acon(cm, 0); /* &tp_var */
    size = ad_icon(size_of(DTYPEG(cm)));
  } else {
    adr_cm = ad_acon(cm, 0); /* &tp_var */
    size = ad_icon(size_of(DTYPEG(cm)));
  }
  adr_vector = ad_acon(sym, 0); /* &cmn_vector/tp_vector */

  if (tmpthr) {
    int tili;
    SPTR tsym;
    ll_set_new_threadprivate(sym);
    tsym = THPRVTOPTG(sym);
    if (XBIT(69, 0x80)) { /* experiment flag */
      tili = _make_mp_get_threadprivate(adr_cm, size, adr_vector);
      *tmpthr = ad3ili(IL_STA, tili, adr_vector, addnme(NT_VAR, sym, 0, 0));
    } else {
      tili = ll_make_kmpc_threadprivate_cached(adr_cm, size, adr_vector);
      *tmpthr =
          ad3ili(IL_STA, tili, ad_acon(tsym, 0), addnme(NT_VAR, tsym, 0, 0));
    }
  }
  return *tmpthr;
}

int
get_threadprivate_origsize(int sym)
{
  int cm;
  int size;

  if (SCG(sym) == SC_CMBLK)
    sym = MIDNUMG(sym); /* get the original common block */

  sym = MIDNUMG(sym);
  cm = MIDNUMG(sym);
  if (STYPEG(cm) == ST_CMBLK) {
    size = ad_icon((INT)SIZEG(cm));
  }
  else if (SCG(cm) == SC_BASED && POINTERG(cm)) {
    int tptr;
    int sdsptr;
    tptr = MIDNUMG(cm);
    size = size_of(DTYPEG(tptr));
    sdsptr = SDSCG(cm); /* $sd */
    if (sdsptr) {
      size += size_of(DT_ADDR);        /* $o */
      size += size_of(DTYPEG(sdsptr)); /* $sd */
    }
    size = ad_icon(size);
  }
  else if (DTY(DTYPEG(cm)) == TY_PTR) {
    size = size_of(DTYPEG(cm));
    if (SYMLKG(cm) != NOSYM) {
      size += size_of(DTYPEG(SYMLKG(cm))); /* $o */
      if (SYMLKG(SYMLKG(cm)) != NOSYM) {
        size += size_of(DTYPEG(SYMLKG(SYMLKG(cm)))); /* $sd */
      }
    }
    size = ad_icon(size);
  } else if (SCG(sym) == SC_BASED) {
    size = ad_icon(size_of(DTYPEG(cm)));
  } else {
    size = ad_icon(size_of(DTYPEG(cm)));
  }

  return size;
}

SPTR
llTaskAllocSptr(void)
{
  return taskAllocSptr;
}

LLTask *
llGetTask(int scope)
{
  int sptr = scope;
  if (!scope)
    sptr = scopeSptr;
  assert(sptr, "No scope for task found ", sptr, ERR_Fatal);
  LLTask *task = llmp_get_task(sptr);
  if (!task)
    task = llmp_create_task(sptr);
  assert(task, "No task associated to this scope sptr", sptr, ERR_Fatal);
  return task;
}
