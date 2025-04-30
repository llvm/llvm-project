/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief extract regions into subroutines; add uplevel references as
   arguments
 */

#include "outliner.h"
#include "error.h"
#include "semant.h"
#include "llassem.h"
#include "exputil.h"
#include "ilmtp.h"
#include "ilm.h"
#include "expand.h"
#include "kmpcutil.h"
#include "machreg.h"
#include "mp.h"
#include "ll_structure.h"
#include "llmputil.h"
#include "llutil.h"
#include "expsmp.h"
#include "dtypeutl.h"
#include "ll_ftn.h"
#include "cgllvm.h"
#include "regutil.h"
#include "symfun.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "asprintf.h"
#if !defined(TARGET_WIN)
#include <unistd.h>
#else
#include <io.h>
#define ftruncate _chsize
#endif
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
#include "ompaccel.h"
#endif
#ifdef OMP_OFFLOAD_LLVM
static bool isReplacerEnabled = false;
static int op1Pld = 0;
#endif

#define MAX_PARFILE_LEN 15

FILE *orig_ilmfil;
FILE *par_file1 = NULL;
FILE *par_file2 = NULL;
FILE *par_curfile = NULL; /* current tempfile for ilm rewrite */

static FILE *savedILMFil = NULL;
static char parFileNm1[MAX_PARFILE_LEN]; /* temp ilms file: pgipar1XXXXXX */
static char parFileNm2[MAX_PARFILE_LEN]; /* temp ilms file: pgipar2XXXXXX */
static bool hasILMRewrite;               /* if set, tempfile is not empty. */
static bool isRewritingILM;              /* if set, write ilm to tempfile */
static int funcCnt = 0;   /* keep track how many outlined region */
static int llvmUniqueSym; /* keep sptr of unique symbol */
static SPTR uplevelSym;
static SPTR gtid;
static bool writeTaskdup; /* if set, write IL_NOP to TASKDUP_FILE */
static int pos;

/* store taskdup ILMs */
static struct taskdupSt {
  ILM_T *file;
  int sz;
  int avl;
} taskdup;

#define TASKDUP_FILE taskdup.file
#define TASKDUP_SZ taskdup.sz
#define TASKDUP_AVL taskdup.avl
static void allocTaskdup(int);

/* Forward decls */
static void resetThreadprivate(void);

/* Check shall we eliminate outlined or not */
static bool eliminate_outlining(ILM_OP opc);

/* Generate a name for outlined function */
static char *ll_get_outlined_funcname(int fileno, int lineno, bool isompaccel, ILM_OP opc);

#define DT_VOID_NONE DT_NONE

#define MXIDLEN 250

/* Dump the values being stored in the uplevel argument */
static void
dumpUplevel(int uplevel_sptr)
{
  int i;
  FILE *fp = gbl.dbgfil ? gbl.dbgfil : stdout;

  fprintf(fp, "********* UPLEVEL Struct *********\n");
  for (i = DTyAlgTyMember(DTYPEG(uplevel_sptr)); i > NOSYM; i = SYMLKG(i))
    fprintf(fp, "==> %s %s\n", SYMNAME(i), stb.tynames[DTY(DTYPEG(i))]);
  fprintf(fp, "**********\n\n");
}

void
dump_parsyms(int sptr, int isTeams)
{
  int i;
  const LLUplevel *up;
  FILE *fp = gbl.dbgfil ? gbl.dbgfil : stdout;
  //TODO Add more OpenMP regions
  const char* ompRegion = isTeams ? "Teams" : "Parallel";
  assert(STYPEG(sptr) == ST_BLOCK, "Invalid OpenMP region sptr", sptr,
         ERR_Fatal);

  up = llmp_get_uplevel(sptr);
  fprintf(fp,
          "\n********** OUTLINING: %s Region "
          "%d (%d variables) **********\n",
          ompRegion, sptr, up->vals_count);

  for (i = 0; i < up->vals_count; ++i) {
    const int var = up->vals[i];
    fprintf(fp, "==> %d) %d (%s) (stype:%d, sc:%d)\n", i + 1, var, SYMNAME(var),
            STYPEG(var), SCG(var));
  }
}
const char* ilmfile_states[] = {"ORIGINAL", "PARFILE1", "PARFILE2" };
const char* outliner_state_names[] = {"Inactive", "Parfile1", "ParFile2", "SwitchParFiles", "Reset", "Error"};

static const char*
get_file_state(FILE *ilmfile) {
  if(ilmfile == orig_ilmfil )
    return ilmfile_states[0];
  else if(ilmfile == par_file1 )
    return ilmfile_states[1];
  else if(ilmfile == par_file2 )
    return ilmfile_states[2];
  else
    //orig_ilmfil is not set yet, so the state is original.
    return ilmfile_states[0];
}

/* Outliner State */
static outliner_states_t outl_state = outliner_not_active;

void
set_outliner_state(outliner_states_t next)
{
  if(DBGBIT(233, 0x100))
    fprintf(gbl.dbgfil, "[Outliner] Compiling [%50s], State: [%10s] -> [%10s] \n", SYMNAME(GBL_CURRFUNC), outliner_state_names[outl_state], outliner_state_names[next]);
  outl_state = next;
}
static void
dump_ilmfile_state(FILE *previous_file)
{
  FILE *dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if(DBGBIT(233, 0x200)) {
    fprintf(dfile , "[Outliner] ILM File:\t[%10s] --> [%10s]\n", get_file_state(previous_file), get_file_state(gbl.ilmfil));
  }
}
void dump_outliner() {
  FILE *dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile , "State: %10s\n", outliner_state_names[outl_state]);
  fprintf(dfile , "ILM File: %10s\n", get_file_state(gbl.ilmfil));
  fprintf(dfile , "Saving ILMs into parfile: %10s\n", isRewritingILM ? "Yes" : "No");
}

static void
set_ilmfile(FILE *file)
{
  FILE *prev = gbl.ilmfil;
  gbl.ilmfil = file;
  dump_ilmfile_state(prev);
}

static int
genNullArg()
{
  int con, ili;
  INT tmp[2];

  tmp[0] = 0;
  tmp[1] = 0;
  con = getcon(tmp, DT_INT);
  ili = ad1ili(IL_ACON, con);
  return ili;
}

static ISZ_T
ll_parent_vals_count(int stblk_sptr)
{
  const LLUplevel *up_parent;
  const LLUplevel *up = llmp_get_uplevel(stblk_sptr);
  ISZ_T sz = 0;
  if (up && up->parent) {
    up_parent = llmp_get_uplevel(up->parent);
    while (up_parent) {
      sz = sz + up_parent->vals_count;
      if (up_parent->parent) {
        up_parent = llmp_get_uplevel(up_parent->parent);
      } else {
        break;
      }
    }
  }
  return sz;
}
/* Returns size in bytes for task shared variable addresses */

ISZ_T
getTaskSharedSize(SPTR scope_sptr)
{
  ISZ_T sz;
  const LLUplevel *up;
  const SPTR uplevel_sptr = (SPTR)PARUPLEVELG(scope_sptr);
  sz = 0;
  if (gbl.internal >= 1)
    sz = sz + 1;
  if (llmp_has_uplevel(uplevel_sptr)) {
    sz = ll_parent_vals_count(uplevel_sptr);
    up = llmp_get_uplevel(uplevel_sptr);
    if (up) {
      sz = sz + up->vals_count;
    }
  }
  sz = sz * size_of(DT_CPTR);
  return sz;
}

/* Returns a dtype for arguments referenced by stblk_sptr */
DTYPE
ll_make_uplevel_type(SPTR stblk_sptr)
{
  int i, j;
  DTYPE dtype;
  int nmems, count, presptr;
  const LLUplevel *up;
  KMPC_ST_TYPE *meminfo = NULL;
  ISZ_T sz;

  up = llmp_get_uplevel(stblk_sptr);
  count = nmems = up->vals_count;

  if (gbl.internal >= 1)
    nmems = nmems + 1;

  /* Add members */
  if (nmems)
    meminfo = (KMPC_ST_TYPE *)calloc(nmems, sizeof(KMPC_ST_TYPE));
  i = 0;
  if (gbl.internal >= 1) {
    meminfo[i].name = strdup(SYMNAME(aux.curr_entry->display));
    meminfo[i].dtype = DT_CPTR;
    meminfo[i].byval = false;
    meminfo[i].psptr = aux.curr_entry->display;
    i++;
  }
  presptr = 0;
  for (j = 0; j < count; ++j) {
    int sptr = up->vals[j];
    meminfo[i].name = strdup(SYMNAME(sptr));
    meminfo[i].dtype = DT_CPTR;
    meminfo[i].byval = false;
    meminfo[i].psptr = sptr;
    ++i;
  }
  sz = ll_parent_vals_count(stblk_sptr) * size_of(DT_CPTR);
  if (sz == 0 && !nmems)
    return DT_CPTR;
  dtype = ll_make_kmpc_struct_type(nmems, NULL, meminfo, sz);

  /* Cleanup */
  for (i = 0; i < nmems; ++i)
    free(meminfo[i].name);
  if (meminfo)
    free(meminfo);
  meminfo = NULL;

  return dtype;
}

/**
   This symbol is used only for its name, if none is found, a unique name is
   generated.
 */
int
llvm_get_unique_sym(void)
{
  return llvmUniqueSym;
}

static const char*
get_opc_name(ILM_OP opc)
{
  switch(opc) {
    case IM_BTARGET:
      return "TARGET";
    break;
    case IM_BTEAMS:
    case IM_BTEAMSN:
      return "TEAMS";
    break;
    case IM_BPAR:
    case IM_BPARA:
    case IM_BPARD:
    case IM_BPARN:
      return "PARALLEL";
    break;
    case IM_BTASK:
      return "TASK";
    break;
    default:
      return "NOPC";
    break;
  }
}

static char *
ll_get_outlined_funcname(int fileno, int lineno, bool isompaccel, ILM_OP opc) {
  char *name;
  const int maxDigitLen = 10; /* Len of 2147483647 */
  int nmSize;
  int r;
  char *name_currfunc = getsname(GBL_CURRFUNC);
  const char *prefix = "";
  int plen;
  const char *host_prefix = "__nv_";
  const char *device_prefix = "nvkernel_";
  if(isompaccel) {
    prefix = device_prefix;
  } else {
    funcCnt++;
    prefix = host_prefix;
  }
  if(gbl.outlined) {
    {
      plen = strlen(host_prefix);
      name_currfunc = strtok(&name_currfunc[plen], "_");
    }
  }
  nmSize = (3 * maxDigitLen) + 5 + strlen(name_currfunc) + 1;
  name = (char *)malloc(nmSize + strlen(prefix));
  r = snprintf(name, nmSize, "%s%s_F%dL%d_%d", prefix, name_currfunc, fileno, lineno, funcCnt);
  assert(r < nmSize, "buffer overrun", r, ERR_Fatal);
  return name;
}

/**
   \p argili is in order
 */
int
ll_make_outlined_garg(int nargs, int *argili, DTYPE *arg_dtypes)
{
  int i, gargl = ad1ili(IL_NULL, 0);
  if (arg_dtypes != NULL) {
    for (i = nargs - 1; i >= 0; --i) {
      if (argili[i]) /* Null if this is a varargs ellipsis */ {
        if (arg_dtypes[i] == 0)
          gargl = ad4ili(IL_GARG, argili[i], gargl, DT_CPTR, 0);
        else
          gargl = ad4ili(IL_GARG, argili[i], gargl, arg_dtypes[i], 0);
      }
    }
  } else {
    for (i = nargs - 1; i >= 0; --i)
      if (argili[i]) /* Null if this is a varargs ellipsis */
        gargl = ad4ili(IL_GARG, argili[i], gargl, DT_CPTR, 0);
  }
  return gargl;
}

int
ll_make_outlined_gjsr(int func_sptr, int nargs, int arg1, int arg2, int arg3)
{
  int gjsr;
  int garg;
  int arglist[10];

  arglist[0] = arg1;
  arglist[1] = arg2;
  arglist[2] = arg3;

  garg = ll_make_outlined_garg(3, arglist, NULL);
  gjsr = ad3ili(IL_GJSR, func_sptr, garg, 0);

  return gjsr;
}

int
ll_ad_outlined_func2(ILI_OP result_opc, ILI_OP call_opc, int sptr, int nargs,
                     int *args)
{
  int i, rg, argl, ilix;
  int *argsp = args;

  rg = 0;
  argl = ad1ili(IL_NULL, 0);
  for (i = 0; i < nargs; i++) {
    int arg = *argsp++;
    if (!arg) /* If varargs ellipses */
      continue;
    switch (IL_RES(ILI_OPC(arg))) {
    case ILIA_AR:
      argl = ad3ili(IL_ARGAR, arg, argl, 0);
      rg++;
      break;
    case ILIA_IR:
      argl = ad3ili(IL_ARGIR, arg, argl, 0);
      rg++;
      break;
    case ILIA_SP:
      argl = ad3ili(IL_ARGSP, arg, argl, 0);
      rg++;
      break;
    case ILIA_DP:
      argl = ad3ili(IL_ARGDP, arg, argl, 0);
      rg += 2;
      break;
    case ILIA_KR:
      argl = ad3ili(IL_ARGKR, arg, argl, 0);
      rg += 2;
      break;
    default:
      interr("ll_ad_outlined_func2: illegal arg", arg, ERR_Severe);
      break;
    }
  }

  ilix = ad2ili(call_opc, sptr, argl);
  if (result_opc)
    ilix = genretvalue(ilix, result_opc);

  return ilix;
}

/* right now, the last argument is the uplevel struct */
SPTR
ll_get_shared_arg(SPTR func_sptr)
{
  int paramct, dpdscp;
  SPTR sym;

  paramct = PARAMCTG(func_sptr);
  dpdscp = DPDSCG(func_sptr);

  while (paramct--) {
    sym = (SPTR)aux.dpdsc_base[dpdscp++];
    if (ISTASKDUPG(func_sptr) && paramct == 2)
      break;
  }
  return sym;
}

void
ll_make_ftn_outlined_params(int func_sptr, int paramct, DTYPE *argtype)
{
  int count = 0;
  int sym;
  char name[MXIDLEN + 2];
  int dpdscp = aux.dpdsc_avl;

  PARAMCTP(func_sptr, paramct);
  DPDSCP(func_sptr, dpdscp);
  aux.dpdsc_avl += paramct;
  NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
       aux.dpdsc_size + paramct + 100);

  while (paramct--) {
    sprintf(name, "%sArg%d", SYMNAME(func_sptr), count++);
    sym = getsymbol(name);
    SCP(sym, SC_DUMMY);
    if (*argtype == DT_CPTR) { /* either i8* or actual type( pass by value). */
      DTYPEP(sym, DT_INT8);
    } else {
      DTYPEP(sym, *argtype);
      PASSBYVALP(sym, 1);
    }
    argtype++;
    STYPEP(sym, ST_VAR);
    aux.dpdsc_base[dpdscp++] = sym;
  }
}

/**
   This is a near duplicate of ll_make_ftn_outlined_params but handles by value
   for fortran.
 */
static void
llMakeFtnOutlinedSignature(int func_sptr, int n_params,
                           const KMPC_ST_TYPE *params)
{
  int i, sym;
  char name[MXIDLEN + 2];
  int count = 0;
  int dpdscp = aux.dpdsc_avl;

  PARAMCTP(func_sptr, n_params);
  DPDSCP(func_sptr, dpdscp);
  aux.dpdsc_avl += n_params;
  NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
       aux.dpdsc_size + n_params + 100);

  for (i = 0; i < n_params; ++i) {
    DTYPE dtype = params[i].dtype;
    const int byval = params[i].byval;

    sprintf(name, "%sArg%d", SYMNAME(func_sptr), count++);
    sym = getsymbol(name);
    SCP(sym, SC_DUMMY);

    if (dtype == DT_CPTR) {
      dtype = DT_INT8;
    }

    DTYPEP(sym, dtype);
    STYPEP(sym, ST_VAR);
    PASSBYVALP(sym, byval);
    aux.dpdsc_base[dpdscp++] = sym;
  }
}

/* Update ACC information such that our OpenACC code generator will be aware of
 * this routine.
 *
 * fnsptr: Function sptr
 */
void
update_acc_with_fn_flags(int fnsptr, int flags)
{
}
void
update_acc_with_fn(int fnsptr)
{
}

static SPTR
llGetSym(char *name, DTYPE dtype)
{
  SPTR gtid;
  if (!name)
    return SPTR_NULL;
  gtid = getsymbol(name);
  DTYPEP(gtid, dtype);
  SCP(gtid, SC_AUTO);
  ENCLFUNCP(gtid, GBL_CURRFUNC);
  STYPEP(gtid, ST_VAR);
  /* to prevent llassem.c setting it to SC_STATIC for Fortran */
  CCSYMP(gtid, 1);
  return gtid;
}

/* Return the ili representing the global thread id:
 * This value is generated from:
 * 1) Calling the kmpc api directly: kmpc_global_thread_num
 * 2) Using the 1st formal parameter if this is a microtask (i.e., outlined
 * function called by kmpc_fork_call).
 * 3) Using the 1st parameter if this is a task.
 *    where 'this' is gbl.curr_func.
 *
 * * If this is a task, the 1st formal param represents the gtid: i32 gtid.
 * * If this is an outlined func, the 1st formal represents gtid: i32* gtid.
 */
int
ll_get_gtid_val_ili(void)
{
  int ili, nme;
  char *name;

  if (!gtid) {
    name = (char *)malloc(strlen(getsname(GBL_CURRFUNC)) + 10);
    sprintf(name, "%s%s", "__gtid_", getsname(GBL_CURRFUNC));
    gtid = llGetSym(name, DT_INT);
    if (flg.omptarget)
      PDALNP(gtid, 3);
    sym_is_refd(gtid);
    free(name);
  }
  ili = ad_acon(gtid, 0);
  nme = addnme(NT_VAR, gtid, 0, 0);
  ili = ad3ili(IL_LD, ili, nme, MSZ_WORD);
  return ili;
}

int
ll_get_gtid_addr_ili(void)
{
  int ili;
  char *name;

  if (!gtid) {
    name = (char *)malloc(strlen(getsname(GBL_CURRFUNC)) + 10);
    sprintf(name, "%s%s", "__gtid_", getsname(GBL_CURRFUNC));
    gtid = llGetSym(name, DT_INT);
    if (flg.omptarget)
      PDALNP(gtid, 3);
    sym_is_refd(gtid);
    free(name);
  }
  ili = ad_acon(gtid, 0);
  return ili;
}

static int
llLoadGtid(void)
{
  int ili, nme, rhs;
  SPTR gtid = ll_get_gtid();

  if (!gtid)
    return 0;

  if (gbl.outlined) {
    SPTR arg = ll_get_hostprog_arg(GBL_CURRFUNC, 1);
    int nme = addnme(NT_VAR, arg, 0, 0);
    int ili = ad_acon(arg, 0);
    if (!TASKFNG(GBL_CURRFUNC)) {
      ili = mk_address(arg);
      nme = addnme(NT_VAR, arg, 0, (INT)0);
      arg = mk_argasym(arg);
      ili = ad2ili(IL_LDA, ili, addnme(NT_VAR, arg, 0, (INT)0));
    }
    rhs = ad3ili(IL_LD, ili, nme, MSZ_WORD);
  } else {
    rhs = ll_make_kmpc_global_thread_num();
  }
  ili = ad_acon(gtid, 0);
  nme = addnme(NT_VAR, gtid, 0, 0);
  ili = ad4ili(IL_ST, rhs, ili, nme, MSZ_WORD);
  ASSNP(gtid, 1);

  return ili;
}

int
ll_save_gtid_val(int bih)
{
  int ili;
#ifdef CUDAG
  if ((CUDAG(GBL_CURRFUNC) & CUDA_GLOBAL) || CUDAG(GBL_CURRFUNC) == CUDA_DEVICE)
    return 0;
#endif

  if (ll_get_gtid()) {
    if (!bih) {
      bih = expb.curbih = BIH_NEXT(BIHNUMG(GBL_CURRFUNC));
    }
    rdilts(bih); /* get block after entry */
    expb.curilt = 0;
    iltb.callfg = 1;
    ili = llLoadGtid();
    if (ili)
      chk_block(ili);
    wrilts(bih);
  }
  return 0;
}

/* Return the uplevel argument from the current function */
int
ll_get_uplevel_arg(void)
{
  int uplevel;

  if (!gbl.outlined && !ISTASKDUPG(GBL_CURRFUNC))
    return 0;

  uplevel = ll_get_shared_arg(GBL_CURRFUNC);
  return uplevel;
}

SPTR
ll_create_task_sptr(void)
{
  SPTR base = getnewccsym('z', GBL_CURRFUNC, ST_VAR);
  SCP(base, SC_AUTO);
  DTYPEP(base, DT_CPTR);
  return base;
}

int *
ll_make_sections_args(SPTR lbSym, SPTR ubSym, SPTR stSym, SPTR lastSym)
{
  static int args[9];

  args[8] = genNullArg();            /* i32* ident     */
  args[7] = ll_get_gtid_val_ili();   /* i32 tid        */
  args[6] = ad_icon(KMP_SCH_STATIC); /* i32 schedule   */
  args[5] = ad_acon(lastSym, 0);     /* i32* plastiter */
  args[4] = ad_acon(lbSym, 0);       /* i32* plower    */
  args[3] = ad_acon(ubSym, 0);       /* i32* pupper    */
  args[2] = ad_acon(stSym, 0);       /* i32* pstridr   */
  args[1] = ad_icon(1);              /* i32 incr       */
  args[0] = ad_icon(0);              /* i32 chunk      */
  ADDRTKNP(lbSym, 1);
  ADDRTKNP(ubSym, 1);
  ADDRTKNP(stSym, 1);
  ADDRTKNP(lastSym, 1);
  return args;
}

/* Create the prototype for an outlined function or task.
 * An outlined function is:  void (int32*, int32*, ...);
 * An outlined task is:      int32 (int32, void*);
 *
 * We actually treat these as:
 * An outlined function is:  void (int32*, int32*, void*);
 * An outlined task is:      void (int32, void*); Return is ignored.
 */
static const KMPC_ST_TYPE funcSig[3] = {
    {NULL, DT_INT,  false, 0},
    {NULL, DT_CPTR, false, 0},
    {NULL, DT_CPTR, false, 0} /* Pass ptr directly */
};

static const KMPC_ST_TYPE taskSig[2] = {
    {NULL, DT_INT,  true,  0},
    {NULL, DT_CPTR, false, 0} /* Pass ptr directly */
};

static const KMPC_ST_TYPE taskdupSig[3] = {
    {NULL, DT_CPTR, false, 0},
    {NULL, DT_CPTR, false, 0},
    {NULL, DT_INT,  true,  0}
};

void
setOutlinedPragma(int func_sptr, int saved)
{
}

static SPTR
makeOutlinedFunc(int stblk_sptr, int scope_sptr, bool is_task, bool istaskdup, bool isompaccel, ILM_OP opc) {
  char *nm;
  SPTR func_sptr;
  DTYPE ret_dtype;
  int n_args;
  const KMPC_ST_TYPE *args;

  /* Get the proper prototype dtypes */
  ret_dtype = DT_VOID_NONE;
  if (is_task) {
    args = taskSig;
    n_args = 2;
  } else if (istaskdup) {
    args = taskdupSig;
    n_args = 3;
  } else {
    args = funcSig;
    n_args = 3;
  }

  if (DBGBIT(45, 0x8) && stblk_sptr)
    dump_parsyms(stblk_sptr, FALSE);

  /* Create the function sptr */
  nm = ll_get_outlined_funcname(gbl.findex, gbl.lineno, isompaccel, opc);
  func_sptr = getsymbol(nm);
  TASKFNP(func_sptr, is_task);
  ISTASKDUPP(func_sptr, istaskdup);
  OUTLINEDP(func_sptr, scope_sptr);
  FUNCLINEP(func_sptr, gbl.lineno);

/* Set return type and  parameters for function dtype */
  STYPEP(func_sptr, ST_ENTRY);
  DTYPEP(func_sptr, ret_dtype);
  DEFDP(func_sptr, 1);
  SCP(func_sptr, SC_STATIC);
  llMakeFtnOutlinedSignature(func_sptr, n_args, args);
  ADDRTKNP(func_sptr, 1);
/* In Auto Offload mode, we generate every outlining function in the host and device code.
    * We build single style ILI for host and device.
    */
  update_acc_with_fn(func_sptr);

  if(DBGBIT(233,2))
    fprintf(gbl.dbgfil, "[Outliner] #%s region is outlined for %10s \t%30s() \tin %s()\n",
        get_opc_name(opc),
        isompaccel ? "Device" : "Host", SYMNAME(func_sptr), SYMNAME(GBL_CURRFUNC));
  return func_sptr;
}

SPTR
ll_make_outlined_func_target_device(SPTR stblk_sptr, SPTR scope_sptr, ILM_OP opc) {
  SPTR sptr = SPTR_NULL;
  if (!eliminate_outlining(opc)) {
    // Create a func sptr for omp target device
    sptr = ll_make_outlined_omptarget_func(stblk_sptr, scope_sptr, opc);
    // Create ABI for the func sptr
    ll_load_outlined_args(scope_sptr, sptr, gbl.outlined);
  }
  return sptr;
}

SPTR
ll_make_outlined_omptarget_func(SPTR stblk_sptr, SPTR scope_sptr, ILM_OP opc)
{
  return makeOutlinedFunc(stblk_sptr, scope_sptr, false, false, true, opc);
}

SPTR
ll_make_outlined_func_wopc(SPTR stblk_sptr, SPTR scope_sptr, ILM_OP opc)
{
  return makeOutlinedFunc(stblk_sptr, scope_sptr, false, false, false, opc);
}

SPTR
ll_make_outlined_func(SPTR stblk_sptr, SPTR scope_sptr)
{
  return makeOutlinedFunc(stblk_sptr, scope_sptr, false, false, false, N_ILM);
}

SPTR
ll_make_outlined_task(SPTR stblk_sptr, SPTR scope_sptr)
{
  return makeOutlinedFunc(stblk_sptr, scope_sptr, true, false, false, N_ILM);
}

static int
llMakeTaskdupRoutine(int task_sptr)
{
  int dupsptr;

  dupsptr = makeOutlinedFunc(0, 0, false, true, false, N_ILM);
  TASKDUPP(task_sptr, dupsptr);
  TASKDUPP(dupsptr, task_sptr);
  ISTASKDUPP(dupsptr, 1);
  return dupsptr;
}

static outliner_states_t
outliner_nextstate()
{
  if(hasILMRewrite) {
    if(outl_state == outliner_not_active)
      set_outliner_state(outliner_active_host_par1);
    else if(gbl.ilmfil == par_file1)
      set_outliner_state(outliner_active_host_par2);
    else if(gbl.ilmfil == par_file2)
      set_outliner_state(outliner_active_switchfile);
  }
  else if(outl_state == outliner_not_active || outl_state == outliner_reset)
    set_outliner_state(outliner_not_active);
  else
    set_outliner_state(outliner_reset);
  return outl_state;
}

int
ll_reset_parfile(void)
{
  /* Process outliner state */
  outliner_nextstate();

  if (!savedILMFil)
    savedILMFil = gbl.ilmfil;
  int returnflag = 1;

  switch (outl_state) {
  case outliner_not_active:
    returnflag = 0;
    break;
  case outliner_active_host_par1:
    gbl.eof_flag = 0;
    orig_ilmfil = gbl.ilmfil;
    set_ilmfile(par_file1);
    par_curfile = par_file2;
    hasILMRewrite = 0;
    (void)fseek(gbl.ilmfil, 0L, 0);
    (void)fseek(par_curfile, 0L, 0);
    break;
  case outliner_active_host_par2:
    set_ilmfile(par_file2);
    gbl.eof_flag = 0;
    par_curfile = par_file1;
    ftruncate(fileno(par_file1), 0);
    hasILMRewrite = 0;
    (void)fseek(gbl.ilmfil, 0L, 0);
    (void)fseek(par_curfile, 0L, 0);
    break;
  case outliner_active_switchfile:
    set_ilmfile(par_file1);
    gbl.eof_flag = 0;
    par_curfile = par_file2;
    ftruncate(fileno(par_file2), 0);
    hasILMRewrite = 0;
    (void)fseek(gbl.ilmfil, 0L, 0);
    (void)fseek(par_curfile, 0L, 0);
    break;
  case outliner_reset:
    if (orig_ilmfil)
      set_ilmfile(orig_ilmfil);
    ftruncate(fileno(par_file1), 0);
    ftruncate(fileno(par_file2), 0);
    (void)fseek(par_file1, 0L, 0);
    (void)fseek(par_file2, 0L, 0);
    par_curfile = par_file1;
    reset_kmpc_ident_dtype();
    resetThreadprivate();
    returnflag = 0;
    /* Set state again */
    outliner_nextstate();

    break;
  default:
    assert(0, "Unknown outliner state", outl_state, ERR_Fatal);
  }
  return returnflag;
}

int
ll_reset_parfile_(void)
{
  static FILE *orig_ilmfil = 0;
  if (!savedILMFil)
    savedILMFil = gbl.ilmfil;
  if (hasILMRewrite) {
    if (gbl.ilmfil == par_file1) {
      gbl.ilmfil = par_file2;
      gbl.eof_flag = 0;
      par_curfile = par_file1;
      ftruncate(fileno(par_file1), 0);
      hasILMRewrite = 0;
      (void)fseek(gbl.ilmfil, 0L, 0);
      (void)fseek(par_curfile, 0L, 0);
      return 1;
    } else if (gbl.ilmfil == par_file2) {
      gbl.ilmfil = par_file1;
      gbl.eof_flag = 0;
      par_curfile = par_file2;
      ftruncate(fileno(par_file2), 0);
      hasILMRewrite = 0;
      (void)fseek(gbl.ilmfil, 0L, 0);
      (void)fseek(par_curfile, 0L, 0);
      return 1;
    } else {
      gbl.eof_flag = 0;
      orig_ilmfil = gbl.ilmfil;
      gbl.ilmfil = par_file1;
      par_curfile = par_file2;
      hasILMRewrite = 0;
      (void)fseek(gbl.ilmfil, 0L, 0);
      (void)fseek(par_curfile, 0L, 0);
      return 1;
    }
  } else {
    if (orig_ilmfil)
      gbl.ilmfil = orig_ilmfil;
    ftruncate(fileno(par_file1), 0);
    ftruncate(fileno(par_file2), 0);
    (void)fseek(par_file1, 0L, 0);
    (void)fseek(par_file2, 0L, 0);
    par_curfile = par_file1;
    reset_kmpc_ident_dtype();
    resetThreadprivate();
    return 0;
  }
  return 0;
}

static int
llGetILMLen(int ilmx)
{
  int opcx, len;
  ILM *ilmpx;

  opcx = ILM_OPC(ilmpx = (ILM *)(ilmb.ilm_base + ilmx));
  len = ilms[opcx].oprs + 1;
  if (IM_VAR(opcx))
    len += ILM_OPND(ilmpx, 1); /* include the number of
                                * variable operands */
  return len;
}

/* collect static variable for Fortran and collect threadprivate for C/C++(need
 * early)*/
static void
llCollectSymbolInfo(ILM *ilmpx)
{
  int flen, len, opnd;
  SPTR sptr;
  int opc;

  opc = ILM_OPC(ilmpx);
  flen = len = ilms[opc].oprs + 1;
  if (IM_VAR(opc)) {
    len += ILM_OPND(ilmpx, 1); /* include the variable opnds */
  }
  /* is this a variable reference */
  for (opnd = 1; opnd <= flen; ++opnd) {
    if (IM_OPRFLAG(opc, opnd) == OPR_SYM) {
      sptr = ILM_SymOPND(ilmpx, opnd);
      if (sptr > 0 && sptr < stb.stg_avail) {
        switch (STYPEG(sptr)) {
        case ST_VAR:
        case ST_ARRAY:
        case ST_STRUCT:
        case ST_UNION:
          switch (SCG(sptr)) {
          case SC_AUTO:
            if (!CCSYMG(sptr) && (DINITG(sptr) || SAVEG(sptr))) {
              SCP(sptr, SC_STATIC);
              sym_is_refd(sptr);
            }
            break;
          case SC_STATIC:
            if (!CCSYMG(sptr)) {
              sym_is_refd(sptr);
            }
            break;
          default:;
          }
          break;
        default:;
        }
      }
    }
  }
}

int
ll_rewrite_ilms(int lineno, int ilmx, int len)
{
  int nw, i;
  ILM *ilmpx;
  ILM_T nop = IM_NOP;

  if (writeTaskdup) {
    if (len == 0)
      len = llGetILMLen(ilmx);
    ilmpx = (ILM *)(ilmb.ilm_base + ilmx);
    if (ilmx == 0 || pos == 0 || pos < ilmx) {
      pos = ilmx;
      allocTaskdup(len);
      memcpy((TASKDUP_FILE + TASKDUP_AVL), (char *)ilmpx, len * sizeof(ILM_T));
      TASKDUP_AVL += len;
    }
  }

  if (!isRewritingILM) /* only write when this flag is set */
    return 0;

  /* if we are writing to taskdup routine, we are going to
   * write IL_NOP to outlined function.  One reason is that
   * we don't want to evaluate and set when see BMPPG/EMPPG
   */
  if (writeTaskdup) {
    if (len == 0)
      len = llGetILMLen(ilmx);
    if (ilmx == 0) {
      ilmpx = (ILM *)(ilmb.ilm_base + ilmx);
      nw = fwrite((char *)ilmpx, sizeof(ILM_T), len, par_curfile);
#if DEBUG
#endif
    } else {
      i = ilmx;
      while (len) {
        nw = fwrite((char *)&nop, sizeof(ILM_T), 1, par_curfile);
#if DEBUG
        assert(nw, "error write to temp file in ll_rewrite_ilms", nw,
               ERR_Fatal);
#endif
        len--;
      };
    }
    return 1;
  }

  if (len == 0) {
    len = llGetILMLen(ilmx);
  }
  ilmpx = (ILM *)(ilmb.ilm_base + ilmx);
  if (!gbl.outlined)
    llCollectSymbolInfo(ilmpx);
  {
    {
#ifdef OMP_OFFLOAD_LLVM
      /* ompaccel symbol replacer */
      if (flg.omptarget) {
        if (isReplacerEnabled) {
          ILM_T opc = ILM_OPC(ilmpx);
          if (op1Pld) {
            if (opc == IM_ELEMENT) {
              ILM_OPND(ilmpx, 2) = op1Pld;
            }
            op1Pld = 0;
          }
          if (opc == IM_BCS) {
            ompaccel_symreplacer(true);
          } else if (opc == IM_BCS) {
            ompaccel_symreplacer(false);
          } else if (ILM_OPC(ilmpx) == IM_ELEMENT && gbl.ompaccel_intarget ) {
            /* replace dtype for allocatable arrays */
            ILM_OPND(ilmpx, 3) =
                ompaccel_tinfo_current_get_dev_dtype(DTYPE(ILM_OPND(ilmpx, 3)));
          } else if (ILM_OPC(ilmpx) == IM_PLD && gbl.ompaccel_intarget) {
            /* replace host sptr with device sptrs, PLD keeps sptr in 2nd index
             */
            op1Pld = ILM_OPND(ilmpx, 1);
            ILM_OPND(ilmpx, 2) =
                ompaccel_tinfo_current_get_devsptr(ILM_SymOPND(ilmpx, 2));
          } else if(gbl.ompaccel_intarget) {
            /* replace host sptr with device sptrs */
            ILM_OPND(ilmpx, 1) =
                ompaccel_tinfo_current_get_devsptr(ILM_SymOPND(ilmpx, 1));
          }
        }
      }
#endif

      nw = fwrite((char *)ilmpx, sizeof(ILM_T), len, par_curfile);
#if DEBUG
      assert(nw, "error write to temp file in ll_rewrite_ilms", nw, ERR_Fatal);
#endif
    }
  }
  return 1;
}

/*
 * 0 BOS            4     1     6
 * 4 ENTRY        207           ;sub
 *
 * 0 BOS            4     1     5
 * 4 ENLAB
 */

void
ll_write_ilm_header(int outlined_sptr, int curilm)
{
  int nw, len, noplen;
  ILM_T t[6];
  ILM_T t2[6];
  ILM_T t3[4];

  if (!par_curfile)
    par_curfile = par_file1;

  t[0] = IM_BOS;
  t[1] = gbl.lineno;
  t[2] = gbl.findex;
  t[3] = 6;
  t[4] = IM_ENTRY;
  t[5] = outlined_sptr;

  t2[0] = IM_BOS;
  t2[1] = gbl.lineno;
  t2[2] = gbl.findex;
  t2[3] = 5;
  t2[4] = IM_ENLAB;
  t2[5] = 0;

  t3[0] = IM_BOS;
  t3[1] = gbl.lineno;
  t3[2] = gbl.findex;
  t3[3] = ilmb.ilmavl;

  setRewritingILM();
  hasILMRewrite = 1;

  nw = fwrite((char *)t, sizeof(ILM_T), 6, par_curfile);
  nw = fwrite((char *)t2, sizeof(ILM_T), 5, par_curfile);

  len = llGetILMLen(curilm);
  noplen = curilm + len;
  len = ilmb.ilmavl - (curilm + len);
  if (len) {
    nw = fwrite((char *)t3, sizeof(ILM_T), 4, par_curfile);
    llWriteNopILM(gbl.lineno, 0, noplen - 4);
  }
#if DEBUG
#endif
}

/*
 * read outlined ilm header to get outlined function sptr so that we can set
 * gbl.currsub to it.   Fortran check gbl.currsub early in the init.
 */
static int
llReadILMHeader()
{
  int nw, outlined_sptr = 0;
  ILM_T t[6];

  if (!gbl.ilmfil)
    return 0;

  nw = fread((char *)t, sizeof(ILM_T), 6, gbl.ilmfil);

  if (nw)
    outlined_sptr = t[5];

  return outlined_sptr;
}

/*
 * 0 BOS           14     1     5
 * 4 END
 */
void
ll_write_ilm_end(void) {
  int nw;
  ILM_T t[6];

  t[0] = IM_BOS;
  t[1] = gbl.lineno;
  t[2] = gbl.findex;
  t[3] = 5;
  t[4] = IM_END;

  nw = fwrite((char *)t, sizeof(ILM_T), 5, par_curfile);
}

void
llWriteNopILM(int lineno, int ilmx, int len)
{
  int nw, i, tlen;
  ILM_T nop = IM_NOP;

  if (writeTaskdup) {
    tlen = len;
    if (tlen == 0)
      tlen = llGetILMLen(ilmx);
    if (tlen)
      allocTaskdup(tlen);
    while (tlen) {
      memcpy((TASKDUP_FILE + TASKDUP_AVL), (char *)&nop, sizeof(ILM_T));
      TASKDUP_AVL += 1;
      tlen--;
    };
  }

  if (!isRewritingILM) /* only write when this flag is set */
    return;

  if (len == 0)
    len = llGetILMLen(ilmx);
  i = ilmx;
  while (len) {
    nw = fwrite((char *)&nop, sizeof(ILM_T), 1, par_curfile);
#if DEBUG
    assert(nw, "error write to temp file in ll_rewrite_ilms", nw, ERR_Fatal);
#endif
    len--;
  };
}

void
ilm_outlined_pad_ilm(int curilm)
{
  int len;
  llWriteNopILM(-1, curilm, 0);
  len = llGetILMLen(curilm);
  len = ilmb.ilmavl - (curilm + len);
  if (len) {
    llWriteNopILM(-1, curilm, len);
  }
}

static SPTR
createUplevelSptr(SPTR uplevel_sptr)
{
  static int n;
  LLUplevel *up;
  SPTR uplevelSym = getccssym("uplevelArgPack", ++n, ST_STRUCT);
  DTYPE uplevel_dtype = ll_make_uplevel_type(uplevel_sptr);
  up = llmp_get_uplevel(uplevel_sptr);
  llmp_uplevel_set_dtype(up, uplevel_dtype);
  DTYPEP(uplevelSym, uplevel_dtype);
  if (gbl.outlined)
    SCP(uplevelSym, SC_PRIVATE);
  else
    SCP(uplevelSym, SC_AUTO);

  /* set alignment of last argument for GPU "align 8". */
  if (DTY(uplevel_dtype) == TY_STRUCT)
    DTySetAlgTyAlign(uplevel_dtype, 7);

  if (DTY(DTYPEG(uplevelSym)) == TY_STRUCT)
    DTySetAlgTyAlign(DTYPEG(uplevelSym), 7);

  return uplevelSym;
}

/* Create a new local uplevel variable and perform a shallow copy of the
 * original uplevel_sptr to the new uplevel sptr.
 */
static SPTR
cloneUplevel(SPTR fr_uplevel_sptr, SPTR to_uplevel_sptr, bool is_task)
{
  int ilix, dest_nme, src_nme;
  const SPTR new_uplevel = createUplevelSptr(to_uplevel_sptr);
  ISZ_T count = ll_parent_vals_count(to_uplevel_sptr);

  if (gbl.internal >= 1)
    count = count + 1;

/* rm_smove will convert SMOVEI into SMOVE.  When doing this
 * rm_smove will remove one ILI so we need to add an ili, so that it is
 * removed when rm_smove executes.
 */
  if (DTYPEG(fr_uplevel_sptr) == DT_ADDR) {
    src_nme = addnme(NT_VAR, fr_uplevel_sptr, 0, 0);
    ilix = ad2ili(IL_LDA, ad_acon(fr_uplevel_sptr, 0), src_nme);
  } else {
    int ili = mk_address(fr_uplevel_sptr);
    SPTR arg = mk_argasym(fr_uplevel_sptr);
    src_nme = addnme(NT_VAR, arg, 0, (INT)0);
    ilix = ad2ili(IL_LDA, ili, src_nme);
  }

/* For nested tasks: the ilix will reference the task object pointer.
 * So in that case we just loaded the task, and will need to next load the
 * uplevel stored at offset zero in that task object, that is what this load
 * does.
 * For Fortran, we store the uplevel in a temp address(more or less like homing)
 *              so we need to make sure to have another load so that when
 *              rm_smove remove one ILI, it gets to the correct address.
 */
  if (DTYPEG(fr_uplevel_sptr) != DT_ADDR)
    if (TASKFNG(GBL_CURRFUNC)) {
      ilix = ad2ili(IL_LDA, ilix, 0); /* task[0] */
    }

  /* Copy the uplevel to the local version of the uplevel */
  if (is_task) {
    int to_ili;
    SPTR taskAllocSptr = llTaskAllocSptr();
    dest_nme = addnme(NT_VAR, taskAllocSptr, 0, 0);
    dest_nme = addnme(NT_IND, SPTR_NULL, dest_nme, 0);
    to_ili = ad2ili(IL_LDA, ad_acon(taskAllocSptr, 0), dest_nme);
    to_ili = ad2ili(IL_LDA, to_ili, dest_nme);
    ilix = ad5ili(IL_SMOVEJ, ilix, to_ili, src_nme, dest_nme, ((int)count) * TARGET_PTRSIZE);
  } else {
    dest_nme = addnme(NT_VAR, new_uplevel, 0, 0);
    ilix = ad5ili(IL_SMOVEJ, ilix, ad_acon(new_uplevel, 0), src_nme, dest_nme,
                  ((int)count) * TARGET_PTRSIZE);
  }
  chk_block(ilix);

  return new_uplevel;
}

static int
loadCharLen(SPTR lensym)
{
  int ilix = mk_address(lensym);
  if (DTYPEG(lensym) == DT_INT8)
    ilix = ad3ili(IL_LDKR, ilix, addnme(NT_VAR, lensym, 0, 0), MSZ_I8);
  else
    ilix = ad3ili(IL_LD, ilix, addnme(NT_VAR, lensym, 0, 0), MSZ_WORD);
  return ilix;
}

static int
toUplevelAddr(SPTR taskAllocSptr, SPTR uplevel, int offset)
{
  int ilix, nme = 0, addr;
  if (taskAllocSptr != SPTR_NULL) {
    ilix = ad_acon(taskAllocSptr, 0);
    nme = addnme(NT_VAR, taskAllocSptr, 0, 0);
    addr = ad2ili(IL_LDA, ilix, nme);
    addr = ad2ili(IL_LDA, addr, addnme(NT_IND, taskAllocSptr, nme, 0));
    if (offset != 0)
      addr = ad3ili(IL_AADD, addr, ad_aconi(offset), 0);
  } else {
    if (TASKFNG(GBL_CURRFUNC) && DTYPEG(uplevel) == DT_ADDR) {
      ilix = ad_acon(uplevel, 0);
      addr = ad2ili(IL_LDA, ilix, nme); // FIXME: initialize nme
    } else {
      addr = ad_acon(uplevel, offset);
    }
  }
  return addr;
}

static void
handle_nested_threadprivate(LLUplevel *parent, SPTR uplevel, SPTR taskAllocSptr,
                            int nme)
{
  int i, sym, ilix, addr, val;
  SPTR sptr;
  int offset;
  if (parent && parent->vals_count) {
    for (i = 0; i < parent->vals_count; ++i) {
      sptr = (SPTR)parent->vals[i];
      if (THREADG(sptr)) {
        if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
          sptr = MIDNUMG(sptr);
        }
        offset = ll_get_uplevel_offset(sptr);
        sym = getThreadPrivateTp(sptr);
        val = llGetThreadprivateAddr(sym);
        addr = toUplevelAddr(taskAllocSptr, uplevel, offset);
        ilix = ad4ili(IL_STA, val, addr, nme, MSZ_PTR);
        chk_block(ilix);
      }
    }
  }
}

/* Given the member list of a struct datatype and an offset, returns the member
 * within the struct whose ADDRESSG matches the offset.
 */
static SPTR
member_with_offset(SPTR member, int offset)
{
  for (SPTR m = member; m > NOSYM && ADDRESSG(m) <= offset; m = SYMLKG(m)) {
    if (ADDRESSG(m) == offset)
      return m; /* found the matching member */
  }
  return SPTR_NULL; /* trouble. */
} /* member_with_offset */

/* Generate load instructions to load just the fields of the uplevel table for
 * this function.
 * uplevel:        sptr to the uplevel table for this nest of regions.
 * base:           Base index into aux table.
 * count:          Number of sptrs to consecutively store in uplevel.
 *
 * Returns the ili for the sequence of store ilis.
 */
static int
loadUplevelArgsForRegion(SPTR uplevel, SPTR taskAllocSptr, int count,
                         int uplevel_stblk_sptr)
{
  int i, addr, ilix, offset, val, nme, encl, based;
  DTYPE dtype;
  SPTR lensptr, member;
  bool do_load, byval;
  ISZ_T addition;
  const LLUplevel *up = NULL;
  if (llmp_has_uplevel(uplevel_stblk_sptr)) {
    up = llmp_get_uplevel(uplevel_stblk_sptr);
  }
  offset = 0;
  if (taskAllocSptr != SPTR_NULL) {
    nme = addnme(NT_VAR, taskAllocSptr, 0, 0);
    nme = addnme(NT_IND, taskAllocSptr, nme, 0);
  } else {
    nme = addnme(NT_VAR, uplevel, 0, 0);
  }
  /* load display argument from host routine */
  if (gbl.internal >= 1) {
    SPTR sptr = aux.curr_entry->display;
    if (gbl.outlined) {
      ADDRTKNP(sptr, 1);
      val = mk_address(sptr);
      val = ad2ili(IL_LDA, val, addnme(NT_VAR, sptr, 0, (INT)0));
    } else if (gbl.internal == 1) {
      ADDRTKNP(sptr, 1);
      val = mk_address(sptr);
    } else {
      sptr = mk_argasym(sptr);
      val = mk_address(sptr);
      val = ad2ili(IL_LDA, val, addnme(NT_VAR, sptr, 0, (INT)0));
    }
    if (taskAllocSptr != SPTR_NULL) {
      addr = toUplevelAddr(taskAllocSptr, (SPTR)uplevel_stblk_sptr, 0);
    } else {
      addr = ad_acon(uplevel, offset);
    }
    ilix = ad4ili(IL_STA, val, addr, nme, MSZ_PTR);
    chk_block(ilix);
    offset += size_of(DT_CPTR);
  }
  addition = ll_parent_vals_count(uplevel_stblk_sptr) * size_of(DT_CPTR);
  offset = offset + addition;
  if (up)
    count = up->vals_count;

  lensptr = SPTR_NULL;
  byval = 0;
  dtype = DTYPEG(uplevel);
  member = DTyAlgTyMember(dtype);
  for (i = 0; i < count; ++i) {
    SPTR sptr = (SPTR)up->vals[i]; // ???

    based = 0;
    if (!sptr && !lensptr) {
      // We put a placeholder in the front end for character len.
      offset += size_of(DT_CPTR);
      continue;
    }

    /* Load the uplevel pointer and get the offset where the pointer to the
     * member should be placed.
     */
    if (!lensptr && need_charlen(DTYPEG(sptr))) {
      lensptr = CLENG(sptr);
    }
    if (lensptr && !sptr) {
      val = loadCharLen(lensptr);
      byval = 1;
      sptr = lensptr;
    } else if (SCG(sptr) == SC_DUMMY) {
      SPTR asym = mk_argasym(sptr);
      int anme = addnme(NT_VAR, asym, 0, (INT)0);
      val = mk_address(sptr);
      val = ad2ili(IL_LDA, val, anme);
    } else if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
      /* For adjustable len char the $p does not have clen field so we need
       * to reference it from the SC_BASED.
       */
      based = sptr;
      sptr = MIDNUMG(sptr);
      val = mk_address(sptr);
#if DO_NOT_DUPLICATE_LOAD_THEN_FIX_ME
      offset += size_of(DT_CPTR);
      continue;
#endif
      if (SCG(sptr) == SC_DUMMY) {
        SPTR asym = mk_argasym(sptr);
        int anme = addnme(NT_VAR, asym, 0, (INT)0);
        val = mk_address(sptr);
        val = ad2ili(IL_LDA, val, anme);
      } else if (THREADG(sptr)) {
        int sym = getThreadPrivateTp(sptr);
        val = llGetThreadprivateAddr(sym);
      }
    } else if (THREADG(sptr)) {
      /* Special handling for copyin threadprivate var - we put it in uplevel
       * structure so that we get master threadprivate copy and pass down to
       * its team.
       */
      int sym = getThreadPrivateTp(sptr);
      val = llGetThreadprivateAddr(sym);
    } else {
      val = mk_address(sptr);
    }
    addr = toUplevelAddr(taskAllocSptr, uplevel, offset);

    /* Skip non-openmp ST_BLOCKS stop at closest one (uplevel is set) */
    encl = ENCLFUNCG(sptr);
    if (STYPEG(encl) != ST_ENTRY && STYPEG(encl) != ST_PROC) {
      while (encl && ((STYPEG(ENCLFUNCG(encl)) != ST_ENTRY) ||
                      (STYPEG(ENCLFUNCG(encl)) != ST_PROC))) {
        if (PARUPLEVELG(encl)) /* Only OpenMP blocks use this */
          break;
        encl = ENCLFUNCG(encl);
      }
    }

    /* Private and encl is an omp block not expanded, then do not load */
    if (encl && PARUPLEVELG(encl) && SCG(sptr) == SC_PRIVATE &&
        (STYPEG(encl) == ST_BLOCK)) {
      if (!PARENCLFUNCG(encl)) {
        offset += size_of(DT_CPTR);
        continue;
      } else {
        if ((STYPEG(ENCLFUNCG(encl)) != ST_ENTRY)) {
          offset += size_of(DT_CPTR);
          lensptr = SPTR_NULL;
          continue;
        }
      }
    }

    /* Determine if we should call a store */
    do_load = false;
    if (THREADG(sptr)) {
      do_load = true;
    } else if (!gbl.outlined && SCG(sptr) != SC_PRIVATE) {
      do_load = true; /* Non-private before outlined func - always load */
      sym_is_refd(sptr);
      if (SCG(sptr) == SC_STATIC) {
        if (based)
          ADDRTKNP(based, 1);
        else
          ADDRTKNP(sptr, 1);
        offset += size_of(DT_CPTR);
        continue;
      }
    } else if (gbl.outlined && is_llvm_local_private(sptr)) {
      do_load = true;
    }

    if (do_load) {
      int mnmex;
      if (based) {
        /* PARREFLOAD is set if ADDRTKN of based was false */
        PARREFLOADP(based, !ADDRTKNG(based));
        ADDRTKNP(based, 1);
      } else {
        /* PARREFLOAD is set if ADDRTKN of sptr was false */
        PARREFLOADP(sptr, !ADDRTKNG(sptr));
        /* prevent optimizer to remove store instruction */
        ADDRTKNP(sptr, 1);
      }
      if (!XBIT(69, 0x80000)) {
        mnmex = nme;
      } else {
        member = member_with_offset(member, offset);
        if (!member) {
          mnmex = nme;
        } else {
          mnmex = addnme(NT_MEM, member, nme, 0);
        }
      }
      if (lensptr && byval) {
        if (CHARLEN_64BIT) {
          val = sel_iconv(val, 1);
          ilix = ad4ili(IL_STKR, val, addr, mnmex, MSZ_I8);
        } else {
          val = sel_iconv(val, 0);
          ilix = ad4ili(IL_ST, val, addr, mnmex, MSZ_WORD);
        }
        lensptr = SPTR_NULL;
        byval = 0;
      } else {
        ilix = ad4ili(IL_STA, val, addr, nme, MSZ_PTR);
      }
      chk_block(ilix);
    }
    //TODO ompaccel optimize load offset for team-private.
    offset += size_of(DT_CPTR);
  }
  /* Special handling for threadprivate copyin, we need to copy the
   * address of current master copy to its slaves.
   */
  if (count == 0) {
    handle_nested_threadprivate(
        llmp_outermost_uplevel((SPTR)uplevel_stblk_sptr), uplevel,
        taskAllocSptr, nme);
  }
  return ad_acon(uplevel, 0);
}

/* Either:
 *
 * 1) Create an instance of the uplevel argument for the outlined call that
 * expects scope_blk_sptr.
 *
 * 2) Create the uplevel table and pass that as an arg.
 *
 */
int
ll_load_outlined_args(int scope_blk_sptr, SPTR callee_sptr, bool clone)
{
  DTYPE uplevel_dtype;
  SPTR uplevel, taskAllocSptr = SPTR_NULL;
  int base, count, ilix, newcount;
  const SPTR uplevel_sptr = (SPTR)PARUPLEVELG(scope_blk_sptr); // ???
  bool is_task = false;
  bool pass_uplevel_byval = false;
  /* If this is not the parent for a nest of funcs just return uplevel tbl ptr
   * which was passed to this function as arg3.
   */
  base = 0;
  count =
      PARSYMSG(uplevel_sptr) ? llmp_get_uplevel(uplevel_sptr)->vals_count : 0;
  newcount = count;
  if (gbl.internal >= 1) {
    if (count == 0 && PARSYMSG(uplevel_sptr) == 0) {
      const int key = llmp_get_next_key();
      llmp_create_uplevel_bykey(key);
      PARSYMSP(uplevel_sptr, key);
    }
    newcount = count + 1;
  }

  is_task = TASKFNG(callee_sptr) ? true : false;
  if (is_task) {
    taskAllocSptr = llTaskAllocSptr();
  }
  if (gbl.outlined) {
    uplevelSym = uplevel = aux.curr_entry->uplevel;
    ll_process_routine_parameters(callee_sptr);
    sym_is_refd(callee_sptr);
    /* Clone: See comment in this function's description above. */
    if (ll_parent_vals_count(uplevel_sptr) != 0) {
      if(!pass_uplevel_byval)
        uplevel = cloneUplevel(uplevel, uplevel_sptr, is_task);
      uplevelSym = uplevel;
    } else if (newcount) {
      /* nothing to copy in parent */
      uplevelSym = uplevel = createUplevelSptr(uplevel_sptr);
      uplevel_dtype = DTYPEG(uplevelSym);
      REFP(uplevel, 1);
    }
  } else { /* Else: is the parent and we need to create an uplevel table */
    if (newcount == 0) { /* No items to pass via uplevel, just pass null  */
      ll_process_routine_parameters(callee_sptr);
      return ad_aconi(0);
    }
    /* Create an uplevel instance and give it a custom struct type */
    uplevelSym = uplevel = createUplevelSptr(uplevel_sptr);
    uplevel_dtype = DTYPEG(uplevelSym);
    REFP(uplevel, 1); /* don't want it to go in sym_is_refd */

    DTYPEP(uplevel, uplevel_dtype);

/* Align uplevel for GPU "align 8" */
    ll_process_routine_parameters(callee_sptr);
    sym_is_refd(callee_sptr);
    /* set alignment of last argument for GPU "align 8". It may not be the same
     * as uplevel if this is task */
    if (DTY(uplevel_dtype) == TY_STRUCT)
      DTySetAlgTyAlign(uplevel_dtype, 7);

    if (DTY(DTYPEG(uplevel)) == TY_STRUCT)
      DTySetAlgTyAlign(DTYPEG(uplevel), 7);

    /* Debug */
    if (DBGBIT(45, 0x8))
      dumpUplevel(uplevel);
  }
  if(pass_uplevel_byval) {
    ilix = ad3ili(IL_LDA, ad_acon(uplevel, 0), addnme(NT_VAR, uplevel, 0, 0),
                  MSZ_PTR);
  } else
    ilix =
        loadUplevelArgsForRegion(uplevel, taskAllocSptr, newcount, uplevel_sptr);
  if (TASKFNG(GBL_CURRFUNC) && DTYPEG(uplevel) == DT_ADDR)
    ilix = ad2ili(IL_LDA, ilix, addnme(NT_VAR, uplevel, 0, 0));

  return ilix;
}

int
ll_get_uplevel_offset(int sptr)
{
  DTYPE dtype;
  SPTR mem;

  if (gbl.outlined || ISTASKDUPG(GBL_CURRFUNC)) {
    int scope_sptr;
    int uplevel_stblk;
    LLUplevel *uplevel;

    if (ISTASKDUPG(GBL_CURRFUNC)) {
      scope_sptr = OUTLINEDG(TASKDUPG(GBL_CURRFUNC));
    } else {
      scope_sptr = OUTLINEDG(GBL_CURRFUNC);
    }
    uplevel_stblk = PARUPLEVELG(scope_sptr);
  redo:
    uplevel = llmp_get_uplevel(uplevel_stblk);

    dtype = uplevel->dtype;
    for (mem = DTyAlgTyMember(dtype); mem > 1; mem = SYMLKG(mem))
      if (PAROFFSETG(mem) == sptr)
        return ADDRESSG(mem);
    if (uplevel->parent) {
      uplevel_stblk = uplevel->parent;
      goto redo;
    }
  }

  return ADDRESSG(sptr);
}

int
ll_make_outlined_call(int func_sptr, int arg1, int arg2, int arg3)
{
  int ilix, altili, argili;
  const int nargs = 3;
  char *funcname = SYMNAME(func_sptr);

  argili = ad_aconi(0);
  ilix = ll_ad_outlined_func(IL_NONE, IL_JSR, funcname, nargs, argili, argili,
                             arg3);

  altili = ll_make_outlined_gjsr(func_sptr, nargs, argili, argili, arg3);
  ILI_ALT(ilix) = altili;

  return ilix;
}

/* whicharg starts from 1 to narg - 1 */
SPTR
ll_get_hostprog_arg(int func_sptr, int whicharg)
{
  int paramct, dpdscp;
  SPTR sym;

  paramct = PARAMCTG(func_sptr);
  dpdscp = DPDSCG(func_sptr);
  sym = (SPTR)aux.dpdsc_base[dpdscp + (whicharg - 1)]; // ???

  return sym;
}

int
ll_make_outlined_call2(int func_sptr, int uplevel_ili)
{
  int ilix, altili;
  int nargs = 3;
  int arg1, arg2, arg3, args[3];

  if (!gbl.outlined) {
    /* orphaned outlined function */
    arg1 = args[2] = ll_get_gtid_addr_ili();
    arg2 = args[1] = genNullArg();
    arg3 = args[0] = uplevel_ili;
  } else {
    /* The first and second arguments are from host program */
    SPTR sarg1 = ll_get_hostprog_arg(GBL_CURRFUNC, 1);
    SPTR sarg2 = ll_get_hostprog_arg(GBL_CURRFUNC, 2);
    arg3 = args[0] = uplevel_ili;
    arg1 = args[2] = mk_address(sarg1);
    arg2 = args[1] = mk_address(sarg2);
  }
  ilix = ll_ad_outlined_func2(IL_NONE, IL_JSR, func_sptr, nargs, args);

  altili = ll_make_outlined_gjsr(func_sptr, nargs, arg1, arg2, arg3);
  ILI_ALT(ilix) = altili;

  return ilix;
}

/* Call an outlined task.
 * func_sptr: Outlined function representing a task.
 * task_sptr: Allocated kmpc task struct
 */
int
ll_make_outlined_task_call(int func_sptr, SPTR task_sptr)
{
  int altili, ilix;
  int arg1, arg2, args[2] = {0};

  arg1 = args[1] = ll_get_gtid_val_ili();
  arg2 = args[0] =
      ad2ili(IL_LDA, ad_acon(task_sptr, 0), addnme(NT_VAR, task_sptr, 0, 0));
  ilix = ll_ad_outlined_func2(IL_NONE, IL_JSR, func_sptr, 2, args);

  altili = ll_make_outlined_gjsr(func_sptr, 2, arg1, arg2, 0);
  ILI_ALT(ilix) = altili;

  return ilix;
}

void
llvm_set_unique_sym(int sptr)
{
  if (!llvmUniqueSym) { /* once set - don't overwrite it */
    llvmUniqueSym = sptr;
  }
}

void
ll_set_outlined_currsub(bool isILMrecompile)
{
  int scope_sptr;
  static long gilmpos;
  static SPTR prev_func_sptr;
  if(!isILMrecompile)
    gilmpos = ftell(gbl.ilmfil);
  gbl.currsub = (SPTR)llReadILMHeader(); // ???
  if(!isILMrecompile)
  prev_func_sptr = gbl.currsub;
  scope_sptr = OUTLINEDG(gbl.currsub);
  if (scope_sptr && gbl.currsub)
    ENCLFUNCP(scope_sptr, PARENCLFUNCG(scope_sptr));
  gbl.rutype = RU_SUBR;
  if(DBGBIT(233,2) && gbl.currsub) {
    FILE *fp = gbl.dbgfil ? gbl.dbgfil : stdout;
    fprintf(fp, "[Outliner] GBL_CURRFUNC is set %s\n", SYMNAME(gbl.currsub));
  }
  fseek(gbl.ilmfil, gilmpos, 0);
}

/* should be call when the host program is done */
static void
resetThreadprivate(void)
{
  int sym, next_tp;
  for (sym = gbl.threadprivate; sym > NOSYM; sym = next_tp) {
    next_tp = TPLNKG(sym);
    TPLNKP(sym, 0);
  }
  gbl.threadprivate = NOSYM;
}

SPTR
ll_get_gtid(void)
{
  return gtid;
}

void
ll_reset_gtid(void)
{
  gtid = SPTR_NULL;
}

void
ll_reset_outlined_func(void)
{
  uplevelSym = SPTR_NULL;
}

SPTR
ll_get_uplevel_sym(void)
{
  return uplevelSym;
}

static void
llRestoreSavedILFil()
{
  if (savedILMFil)
    gbl.ilmfil = savedILMFil;
}

void
ll_open_parfiles()
{
  strcpy(parFileNm1, "pgipar1XXXXXX");
  strcpy(parFileNm2, "pgipar2XXXXXX");
#if defined(TARGET_WIN)
  char* result1 = _mktemp(parFileNm1);
  char* result2 = _mktemp(parFileNm2);
  if (result1 != NULL && result2 != NULL) {
    fopen_s( &par_file1, result1, "w" );
    fopen_s( &par_file2, result2, "w" );
  }
#else
  int fd1, fd2;
  fd1 = mkstemp(parFileNm1);
  fd2 = mkstemp(parFileNm2);
  par_file1 = fdopen(fd1, "w+");
  par_file2 = fdopen(fd2, "w+");
#endif
  if (!par_file1)
    errfatal((error_code_t)4);
  if (!par_file2)
    errfatal((error_code_t)4);
}

void
ll_unlink_parfiles()
{
  llRestoreSavedILFil();
  if (par_file1)
    unlink(parFileNm1);
  if (par_file2)
    unlink(parFileNm2);
  par_file1 = NULL;
  par_file2 = NULL;
}

/* START: OUTLINING MCONCUR */
void
llvmSetExpbCurIlt(void)
{
  expb.curilt = ILT_PREV(0);
}

int
llvmGetExpbCurIlt(void)
{
  return expb.curilt;
}

SPTR
llvmAddConcurEntryBlk(int bih)
{
  int newbih;
  int asym, ili_uplevel, nme, ili;
  SPTR display_temp = SPTR_NULL;

  /* add entry block */
  newbih = addnewbih(bih, bih, bih);
  rdilts(newbih);
  expb.curbih = newbih;
  BIHNUMP(GBL_CURRFUNC, expb.curbih);
  expb.curilt = addilt(ILT_PREV(0), ad1ili(IL_ENTRY, GBL_CURRFUNC));
  wrilts(newbih);
  BIH_LABEL(newbih) = GBL_CURRFUNC;
  BIH_EN(newbih) = 1;

  gbl.outlined = 1;
  gbl.entbih = newbih;

  reset_kmpc_ident_dtype();

  reg_init(GBL_CURRFUNC);

  aux.curr_entry->uplevel = ll_get_shared_arg(GBL_CURRFUNC);
  asym = mk_argasym(aux.curr_entry->uplevel);
  ADDRESSP(asym, ADDRESSG(aux.curr_entry->uplevel)); /* propagate ADDRESS */
  MEMARGP(asym, 1);

  if (gbl.internal > 1) {
    rdilts(newbih);
    display_temp = getccsym('S', gbl.currsub, ST_VAR);
    SCP(display_temp, SC_PRIVATE);
    ENCLFUNCP(display_temp, GBL_CURRFUNC);
    DTYPEP(display_temp, DT_ADDR);
    sym_is_refd(display_temp);

    ili = ad_acon(display_temp, 0);
    nme = addnme(NT_VAR, display_temp, 0, 0);

    ili_uplevel = mk_address(aux.curr_entry->uplevel);
    nme = addnme(NT_VAR, aux.curr_entry->uplevel, 0, 0);
    ili_uplevel = ad2ili(IL_LDA, ili_uplevel, nme);
    ili_uplevel =
        ad2ili(IL_LDA, ili_uplevel, addnme(NT_IND, display_temp, nme, 0));

    ili = ad2ili(IL_LDA, ili, addnme(NT_IND, display_temp, nme, 0));
    nme = addnme(NT_VAR, display_temp, 0, 0);
    ili = ad3ili(IL_STA, ili_uplevel, ili, nme);
    expb.curilt = addilt(expb.curilt, ili);
    wrilts(newbih);

    flg.recursive = true;
  }

  newbih = addnewbih(bih, bih,
                     bih); /* add empty block  - make entry block separate */
  return display_temp;
}

void
llvmAddConcurExitBlk(int bih)
{
  int newbih;

  newbih = addnewbih(BIH_PREV(bih), bih, bih);
  rdilts(newbih);
  expb.curbih = newbih;
  expb.curilt = addilt(ILT_PREV(0), ad1ili(IL_EXIT, GBL_CURRFUNC));
  wrilts(newbih);
  BIH_XT(newbih) = 1;
  BIH_LAST(newbih) = 1;
  BIH_FT(newbih) = 0;
  expb.arglist = 0;
  expb.flags.bits.callfg = 0;
  mkrtemp_end();
}

/* END: OUTLINING MCONCUR */

/* START: TASKDUP(kmp_task_t* task, kmp_task_t* newtask, int lastitr)
 * write all ilms between IM_BTASKLOOP and IM_ETASKLOOP
 * to a taskdup routine.  Mostly use for firstprivate and
 * last iteration variables copy/constructor.
 * writeTaskdup is set when we see IM_BTASKLOOP and unset when
 * we see IM_TASKLOOPREG. It then will be set again after IM_ETASKLOOPREG
 * until IM_ETASKLOOP(C/C++ may have firstprivate initialization
 * later).  Currently only private data allocation & initialization
 * are expected in those ilms.  In future, if there are other ilms
 * in the mix, the we may need to provide some delimits to mark
 * where to start write and end.
 */

void
start_taskdup(int task_fnsptr, int curilm)
{
  int len, noplen;
  ILM_T t3[6];
  writeTaskdup = true;
  t3[0] = IM_BOS;
  t3[1] = gbl.lineno;
  t3[2] = gbl.findex;
  t3[3] = ilmb.ilmavl;
  if (!TASKDUPG(task_fnsptr)) {
    int dupsptr = llMakeTaskdupRoutine(task_fnsptr);
    ILM_T t[6];
    ILM_T t2[6];

    t[0] = IM_BOS;
    t[1] = gbl.lineno;
    t[2] = gbl.findex;
    t[3] = 6;
    t[4] = IM_ENTRY;
    t[5] = dupsptr;

    t2[0] = IM_BOS;
    t2[1] = gbl.lineno;
    t2[2] = gbl.findex;
    t2[3] = 5;
    t2[4] = IM_ENLAB;
    t2[5] = 0;

    allocTaskdup(6);
    memcpy((TASKDUP_FILE + TASKDUP_AVL), (char *)t, 6 * sizeof(ILM_T));
    TASKDUP_AVL += 6;

    allocTaskdup(5);
    memcpy((TASKDUP_FILE + TASKDUP_AVL), (char *)t2, 5 * sizeof(ILM_T));
    TASKDUP_AVL += 5;
  }
  pos = 0;
  len = llGetILMLen(curilm);
  noplen = curilm + len;
  len = ilmb.ilmavl - (curilm + len);
  if (len) {
    allocTaskdup(4);
    memcpy((TASKDUP_FILE + TASKDUP_AVL), (char *)t3, 4 * sizeof(ILM_T));
    TASKDUP_AVL += 4;
    llWriteNopILM(gbl.lineno, 0, noplen - 4);
  }
}

void
restartRewritingILM(int curilm)
{
  int len, noplen, nw;
  ILM_T t[6];

  t[0] = IM_BOS;
  t[1] = gbl.lineno;
  t[2] = gbl.findex;
  t[3] = ilmb.ilmavl;
  pos = 0;
  len = llGetILMLen(curilm);
  noplen = curilm + len;
  len = ilmb.ilmavl - (curilm + len);
  setRewritingILM();
  if (len) {
    nw = fwrite((char *)t, sizeof(ILM_T), 4, par_curfile);
#if DEBUG
#endif
    llWriteNopILM(gbl.lineno, 0, noplen - 4);
  }
}

void
stop_taskdup(int task_fnsptr, int curilm)
{
  /* write IL_NOP until the end of ILM blocks */
  ilm_outlined_pad_ilm(curilm);
  writeTaskdup = false;
  pos = 0;
}

static void
clearTaskdup()
{
  FREE(TASKDUP_FILE);
  TASKDUP_AVL = 0;
  TASKDUP_SZ = 0;
  TASKDUP_FILE = NULL;
}

static void
copyLastItr(int fnsptr, INT offset)
{
  ILM_T *ptr;
  int offset_sptr;
  int total_ilms = 0;
  INT tmp[2];
  tmp[0] = 0;
  tmp[1] = offset;
  offset_sptr = getcon(tmp, DT_INT);

  allocTaskdup(6);
  ptr = (TASKDUP_FILE + TASKDUP_AVL);

  *ptr++ = IM_BOS;
  *ptr++ = gbl.lineno;
  *ptr++ = gbl.findex;
  *ptr++ = 6;
  total_ilms = total_ilms + 4;

  *ptr++ = IM_TASKLASTPRIV;
  *ptr++ = offset_sptr;
  total_ilms = total_ilms + 2;

  TASKDUP_AVL += total_ilms;
}

void
finish_taskdup_routine(int curilm, int fnsptr, INT offset)
{
  int nw;
  ILM_T t[6];

  if (!TASKDUP_AVL)
    return;

  t[0] = IM_BOS;
  t[1] = gbl.lineno;
  t[2] = gbl.findex;
  t[3] = 5;
  t[4] = IM_END;
  if (offset) {
    copyLastItr(fnsptr, offset);
  }
  /* write taskdup ilms to file */
  if (TASKDUP_AVL) {
    allocTaskdup(6);
    memcpy((TASKDUP_FILE + TASKDUP_AVL), (char *)t, 6 * sizeof(ILM_T));
    TASKDUP_AVL += 6;

    nw = fwrite((char *)TASKDUP_FILE, sizeof(ILM_T), TASKDUP_AVL, par_curfile);
#ifdef DEBUG
#endif
  }
  clearTaskdup();
  writeTaskdup = false;
  hasILMRewrite = 1;
  pos = 0;
}

static void
allocTaskdup(int len)
{
  NEED((TASKDUP_AVL + len + 20), TASKDUP_FILE, ILM_T, TASKDUP_SZ,
       (TASKDUP_AVL + len + 20));
}

/* END: TASKDUP routine */

void
unsetRewritingILM()
{
  isRewritingILM = 0;
}

void
setRewritingILM()
{
  isRewritingILM = 1;
}

bool
ll_ilm_is_rewriting(void)
{
  return isRewritingILM;
}

int
ll_has_more_outlined()
{
  return hasILMRewrite;
}

int
llvm_ilms_rewrite_mode(void)
{
  if (gbl.ilmfil == par_file1 || gbl.ilmfil == par_file2)
    return 1;
  return 0;
}

/* used by Fortran only.  If gbl.ilmfil points to tempfile, then
 * we are processing ILMs in that file.  This function is called
 * after we emit we call schedule of current function and we are
 * trying to decide if we should continue processing the current
 * file or the next tempfile.
 */
int
llProcessNextTmpfile()
{
  if (gbl.ilmfil == par_file1 || gbl.ilmfil == par_file2)
    return 0;
  return hasILMRewrite;
}

int
mk_function_call(DTYPE ret_dtype, int n_args, DTYPE *arg_dtypes, int *arg_ilis,
                 SPTR func_sptr)
{
  int i, ilix, altilix, gargs, *garg_ilis = ALLOCA (int, n_args);
  DTYPE *garg_types = ALLOCA (DTYPE, n_args);

  DTYPEP(func_sptr, ret_dtype);
  // SCP(outlined_func_sptr, SC_EXTERN);
  STYPEP(func_sptr, ST_PROC);
  // CCSYMP(outlined_func_sptr, 1); /* currently we make all CCSYM func varargs
  // in Fortran. */
  CFUNCP(func_sptr, 1);
  // ll_make_ftn_outlined_params(outlined_func_sptr, n_args, arg_dtypes);
  ll_process_routine_parameters(func_sptr);

  // sym_is_refd(outlined_func_sptr);

  ilix = ll_ad_outlined_func2((ILI_OP)0, IL_JSR, func_sptr, n_args, arg_ilis);

  /* Create the GJSR */
  for (i = n_args - 1; i >= 0; --i) { /* Reverse the order */
    garg_ilis[i] = arg_ilis[n_args - 1 - i];
    garg_types[i] = arg_dtypes[n_args - 1 - i];
  }
  gargs = ll_make_outlined_garg(n_args, garg_ilis, garg_types);
  altilix = ad3ili(IL_GJSR, func_sptr, gargs, 0);

  /* Add gjsr as an alt to the jsr */
  if (0)
    ILI_ALT(ILI_OPND(ilix, 1)) = altilix;
  else
    ILI_ALT(ilix) = altilix;

  return ilix;
}

static bool
eliminate_outlining(ILM_OP opc)
{
  return false;
}

bool
outlined_is_eliminated(ILM_OP opc)
{
  return false;
}

bool
outlined_need_recompile() {
  return false;
}

#ifdef OMP_OFFLOAD_LLVM
void
llMakeFtnOutlinedSignatureTarget(SPTR func_sptr, OMPACCEL_TINFO *current_tinfo)
{
  int i, count = 0, dpdscp = aux.dpdsc_avl;

  PARAMCTP(func_sptr, current_tinfo->n_symbols);
  DPDSCP(func_sptr, dpdscp);
  aux.dpdsc_avl += current_tinfo->n_symbols;
  NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
       aux.dpdsc_size + current_tinfo->n_symbols + 100);

  for (i = 0; i < current_tinfo->n_symbols; ++i) {
    SPTR sptr = current_tinfo->symbols[i].host_sym;
    SPTR sym = ompaccel_create_device_symbol(sptr, count);
    count++;
    current_tinfo->symbols[i].device_sym = sym;
    OMPACCDEVSYMP(sym, TRUE);
    aux.dpdsc_base[dpdscp++] = sym;
  }
}

int
ll_make_outlined_ompaccel_call(SPTR parent_func_sptr, SPTR outlined_func)
{

  int nargs, nme, ili, i;
  SPTR sptr;
  OMPACCEL_TINFO *omptinfo;
  omptinfo = ompaccel_tinfo_get(outlined_func);
  nargs = omptinfo->n_symbols;
  int args[nargs];
  DTYPE arg_dtypes[nargs];

  DTYPEP(outlined_func, DT_NONE);
  STYPEP(outlined_func, ST_PROC);
  CFUNCP(outlined_func, 1);
  for (i = 0; i < nargs; ++i) {
    sptr = omptinfo->symbols[i].host_sym;
    nme = addnme(NT_VAR, sptr, 0, (INT)0);
    ili = mk_address(sptr);
    if (!PASSBYVALG(sptr))
      args[nargs - i - 1] = ad2ili(IL_LDA, ili, nme);
    else {
      if (DTY(DTYPEG(sptr)) == TY_PTR) {
        args[nargs - i - 1] = ad2ili(IL_LDA, ili, nme);
      } else {
        if (DTYPEG(sptr) == DT_INT8)
          args[nargs - i - 1] = ad3ili(IL_LDKR, ili, nme, MSZ_I8);
        else if (DTYPEG(sptr) == DT_DBLE)
          args[nargs - i - 1] = ad3ili(IL_LDDP, ili, nme, MSZ_F8);
        else
          args[nargs - i - 1] = ad3ili(IL_LD, ili, nme, MSZ_WORD);
      }
    }
    arg_dtypes[nargs - i - 1] = DTYPEG(sptr);
  }

  int call_ili =
      mk_function_call(DT_NONE, nargs, arg_dtypes, args, outlined_func);

  return call_ili;
}

static int ompaccel_isreductionregion = 0;
void
ompaccel_notify_reduction(bool enable)
{
  if (XBIT(232, 4))
    return;
  if (enable)
    ompaccel_isreductionregion++;
  else
    ompaccel_isreductionregion--;
  if (DBGBIT(61, 4) && gbl.dbgfil != NULL) {
    if (enable)
      fprintf(gbl.dbgfil, "[ompaccel] Skip codegen of omp cpu reduction - ON   "
                          "################################### \n");
    else
      fprintf(gbl.dbgfil, "[ompaccel] Skip codegen of omp cpu reduction - OFF  "
                          "################################### \n");
  }
}
bool
ompaccel_is_reduction_region()
{
  return ompaccel_isreductionregion;
}

void
ompaccel_symreplacer(bool enable)
{
  if (XBIT(232, 2))
    return;
  isReplacerEnabled = enable;
  if (DBGBIT(61, 2) && gbl.dbgfil != NULL) {
    if (enable)
      fprintf(
          gbl.dbgfil,
          "[ompaccel] Replacer - ON   ################################### \n");
    else
      fprintf(
          gbl.dbgfil,
          "[ompaccel] Replacer - OFF  ################################### \n");
  }
}

INLINE static SPTR
create_target_outlined_func_sptr(SPTR scope_sptr, bool iskernel)
{
  char *nm = ll_get_outlined_funcname(gbl.findex, gbl.lineno, 0, IM_BTARGET);
  SPTR func_sptr = getsymbol(nm);
  TASKFNP(func_sptr, FALSE);
  ISTASKDUPP(func_sptr, FALSE);
  OUTLINEDP(func_sptr, scope_sptr);
  FUNCLINEP(func_sptr, gbl.lineno);
  STYPEP(func_sptr, ST_ENTRY);
  DTYPEP(func_sptr, DT_VOID_NONE);
  DEFDP(func_sptr, 1);
  SCP(func_sptr, SC_STATIC);
  ADDRTKNP(func_sptr, 1);
  if (iskernel)
    OMPACCFUNCKERNELP(func_sptr, 1);
  else
    OMPACCFUNCDEVP(func_sptr, 1);
  return func_sptr;
}

INLINE static SPTR
ompaccel_copy_arraydescriptors(SPTR arg_sptr)
{
  SPTR device_symbol;
  DTYPE dtype;
  char *name;
  NEW(name, char, MXIDLEN);
  sprintf(name, "Arg_%s", SYMNAME(arg_sptr));
  device_symbol = getsymbol(name);
  SCP(device_symbol, SC_DUMMY);

  // check whether it is allocatable or not
  ADSC *new_ad;
  ADSC *org_ad = AD_DPTR(DTYPEG(arg_sptr));
  TY_KIND atype = DTY(DTYPE(DTYPEG(arg_sptr) + 1));
  int numdim = AD_NUMDIM(org_ad);
  dtype = get_array_dtype(numdim, (DTYPE)atype);

  new_ad = AD_DPTR(dtype);
  AD_NUMDIM(new_ad) = numdim;
  AD_SCHECK(new_ad) = AD_SCHECK(org_ad);
  AD_ZBASE(new_ad) = ompaccel_tinfo_current_get_devsptr((SPTR)AD_ZBASE(org_ad));
  AD_NUMELM(new_ad) =
      ompaccel_tinfo_current_get_devsptr((SPTR)AD_NUMELM(org_ad));
  // todo ompaccel maybe zero, maybe an array?
  // check global in the module?
  AD_SDSC(new_ad) = ompaccel_tinfo_current_get_devsptr((SPTR)AD_SDSC(org_ad));

  if (numdim >= 1 && numdim <= 7) {
    int i;
    for (i = 0; i < numdim; ++i) {
      AD_LWBD(new_ad, i) =
          ompaccel_tinfo_current_get_devsptr((SPTR)AD_LWBD(org_ad, i));
      AD_UPBD(new_ad, i) =
          ompaccel_tinfo_current_get_devsptr((SPTR)AD_UPBD(org_ad, i));
      AD_MLPYR(new_ad, i) =
          ompaccel_tinfo_current_get_devsptr((SPTR)AD_MLPYR(org_ad, i));
    }
  }

  DTYPEP(device_symbol, dtype);

  STYPEP(device_symbol, STYPEG(arg_sptr));
  SCP(device_symbol, SCG(arg_sptr));
  POINTERP(device_symbol, POINTERG(arg_sptr));
  ADDRTKNP(device_symbol, ADDRTKNG(arg_sptr));
  ALLOCATTRP(device_symbol, ALLOCATTRG(arg_sptr));
  NOCONFLICTP(device_symbol, NOCONFLICTG(arg_sptr));
  ASSNP(device_symbol, ASSNG(arg_sptr));
  DCLDP(device_symbol, DCLDG(arg_sptr));
  PARREFP(device_symbol, PARREFG(arg_sptr));
  ORIGDIMP(device_symbol, ORIGDIMG(arg_sptr));
  ORIGDUMMYP(device_symbol, ORIGDUMMYG(arg_sptr));
  MEMARGP(device_symbol, MEMARGG(arg_sptr));
  ASSUMSHPP(device_symbol, ASSUMSHPG(arg_sptr));

  int org_midnum = MIDNUMG(arg_sptr);
  SPTR dev_midnum = ompaccel_tinfo_current_get_devsptr((SPTR)org_midnum);
  MIDNUMP(device_symbol, dev_midnum);

  PARREFP(dev_midnum, PARREFG(org_midnum));
  ADDRTKNP(dev_midnum, ADDRTKNG(org_midnum));
  ASSNP(dev_midnum, ASSNG(org_midnum));
  CCSYMP(dev_midnum, CCSYMG(org_midnum));
  NOCONFLICTP(dev_midnum, NOCONFLICTG(org_midnum));
  PTRSAFEP(dev_midnum, PTRSAFEG(org_midnum));
  PARREFLOADP(dev_midnum, PARREFLOADG(org_midnum));
  PTRSAFEP(dev_midnum, PTRSAFEG(org_midnum));
  REFP(dev_midnum, REFG(org_midnum));
  VARDSCP(dev_midnum, VARDSCG(org_midnum));

  return device_symbol;
}

SPTR
ll_make_outlined_ompaccel_func(SPTR stblk_sptr, SPTR scope_sptr, bool iskernel)
{
  const LLUplevel *uplevel;
  SPTR func_sptr, arg_sptr;
  int n_args = 0, max_nargs, i;
  OMPACCEL_TINFO *current_tinfo;

  uplevel = llmp_has_uplevel(stblk_sptr);
  max_nargs = uplevel != NULL ? uplevel->vals_count : 0;
  /* Create function symbol for target region */
  func_sptr = create_target_outlined_func_sptr(scope_sptr, iskernel);

  /* Create target info for the outlined function */
  current_tinfo = ompaccel_tinfo_create(func_sptr, max_nargs);
  for (i = 0; i < max_nargs; ++i) {
    arg_sptr = (SPTR)uplevel->vals[i];
    if (!arg_sptr && !ompaccel_tinfo_current_is_registered(arg_sptr))
      continue;
    if (SCG(arg_sptr) == SC_PRIVATE)
      continue;
    if (DESCARRAYG(arg_sptr))
      continue;

    if (!iskernel && !OMPACCDEVSYMG(arg_sptr))
      arg_sptr = ompaccel_tinfo_parent_get_devsptr(arg_sptr);
    ompaccel_tinfo_current_add_sym(arg_sptr, SPTR_NULL, 0);

    n_args++;
  }

  llMakeFtnOutlinedSignatureTarget(func_sptr, current_tinfo);

  ompaccel_symreplacer(true);
  if (isReplacerEnabled) {
    /* Data dtype replication for allocatable arrays */
    for (i = 0; i < ompaccel_tinfo_current_get()->n_quiet_symbols; ++i) {
      ompaccel_tinfo_current_get()->quiet_symbols[i].device_sym =
          ompaccel_copy_arraydescriptors(
              ompaccel_tinfo_current_get()->quiet_symbols[i].host_sym);
    }
    for (i = 0; i < ompaccel_tinfo_current_get()->n_symbols; ++i) {
      if (SDSCG(ompaccel_tinfo_current_get()->symbols[i].host_sym))
        ompaccel_tinfo_current_get()->symbols[i].device_sym =
            ompaccel_copy_arraydescriptors(
                ompaccel_tinfo_current_get()->symbols[i].host_sym);
    }
  }
  ompaccel_symreplacer(false);

  return func_sptr;
}
#endif /* End #ifdef OMP_OFFLOAD_LLVM */
