/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Fortran expander routines

   For processing ILMs dealing with the run-time environment, e.g., expanding
   calls, expanding entries, and handling structure assignments.
 */

#include "exp_rte.h"
#include "error.h"
#include "llassem.h"
#include "ll_ftn.h"
#include "outliner.h"
#include "cgmain.h"
#include "expatomics.h"
#include "exp_rte.h"
#include "exputil.h"
#include "regutil.h"
#include "machreg.h"
#include "exp_ftn.h"
#include "expsmp.h"
#include "expreg.h"
#include "semutil0.h"
#include "ilm.h"
#include "ilmtp.h"
#include "ili.h"
#define EXPANDER_DECLARE_INTERNAL
#include "expand.h"
#include "machar.h"
#include "mach.h"
#include "rtlRtns.h"
#include "dtypeutl.h"
#include "upper.h"
#include "symfun.h"

static int exp_strx(int, STRDESC *, STRDESC *);
static int exp_strcpy(STRDESC *, STRDESC *);
static bool strovlp(STRDESC *, STRDESC *);
static STRDESC *getstr(int);
static STRDESC *getstrconst(const char *, int);
static STRDESC *storechartmp(STRDESC *str, int mxlenili, int clenili);
static char *getcharconst(STRDESC *);
static int ftn_strcmp(char *, char *, int, int);
static int getstrlen64(STRDESC *);
static void pp_entries(void);
static void pp_entries_mixedstrlen(void);
static void pp_params(SPTR func);
static void pp_params_mixedstrlen(int);
static void cp_memarg(int, INT, int);
static void cp_byval_mem_arg(SPTR argsptr);
static SPTR allochartmp(int lenili);
static int block_str_move(STRDESC *, STRDESC *);
static int getchartmp(int ili);
static void _exp_smove(int, int, int, int, DTYPE);

#ifdef FLANG2_EXPRTE_UNUSED
static int has_desc_arg(int, int);
#endif
static int check_desc(int, int);
static void check_desc_args(int);
static int exp_type_bound_proc_call(int arg, SPTR descno, int vtoff,
                                    int arglnk);
static bool is_asn_closure_call(int sptr);
static bool is_proc_desc_arg(int ili);
static bool process_end_of_list(SPTR func, SPTR osym, int *nlens,
                                DTYPE argdtype);

static int get_chain_pointer_closure(SPTR sdsc);
static int add_last_arg(int arglnk, int displnk);
static int add_arglnk_closure(SPTR sdsc);
static int add_gargl_closure(SPTR sdsc);

#define CLASS_NONE 0
#define CLASS_INT4 4
#define CLASS_INT8 8
#define CLASS_MEM 13

#define MAX_PASS_STRUCT_SIZE 16

#define mk_prototype mk_prototype_llvm

#ifdef TARGET_SUPPORTS_QUADFP
#define IS_INTERNAL_PROC_CALL(opc)                                  \
  (opc == IM_PCALLA || opc == IM_PCHFUNCA || opc == IM_PNCHFUNCA || \
   opc == IM_PKFUNCA || opc == IM_PLFUNCA || opc == IM_PIFUNCA ||   \
   opc == IM_PRFUNCA || opc == IM_PDFUNCA || opc == IM_PCFUNCA ||   \
   opc == IM_PQFUNCA || opc == IM_PCDFUNCA || opc == IM_PCQFUNCA || \
   opc == IM_PPFUNCA)
#else
#define IS_INTERNAL_PROC_CALL(opc)                                  \
  (opc == IM_PCALLA || opc == IM_PCHFUNCA || opc == IM_PNCHFUNCA || \
   opc == IM_PKFUNCA || opc == IM_PLFUNCA || opc == IM_PIFUNCA ||   \
   opc == IM_PRFUNCA || opc == IM_PDFUNCA || opc == IM_PCFUNCA ||   \
   opc == IM_PCDFUNCA || opc == IM_PPFUNCA)
#endif

static SPTR exp_call_sym; /**< sptr subprogram being called */
static SPTR fptr_iface;   /**< sptr of function pointer's interface */
static SPTR allocharhdr;
static int *parg; /**< pointer to area for dummy arg processing */

typedef struct {
  INT mem_off;  /**< next offset in the memory arg area */
  short retgrp; /**< return group # for a function */
  /** function ret variable for return group -- there is a sub-table in the
      finfo table which is indexed by the return group index (0 - retgrp_cnt-1).
      This field is valid only for the sub-table. */
  SPTR fval;
  /** register descriptor for the case where the function is bind(C) and the
      return value is a small structure returned in memory; 0 otherwise */
  int ret_sm_struct;
  int ret_align; /**< if returning small struct, this is its alignment */
} finfo_t;

static finfo_t *pfinfo; /**< table of finfo for the entries */
static int nentries;    /**< number of entries for the subprogram */
static int smove_flag;
static int mscall_flag;
static int alloca_flag;
static int retgrp_cnt; /**< number of return counts */
static SPTR retgrp_var; /**< local variable holding return group value */

/** variable used to locate the beginning of the memory argument area */
static SPTR memarg_var;

#ifdef __cplusplus
inline SPTR convertSPTR(int i) {
  return static_cast<SPTR>(i);
}
inline SPTR sptr_mk_address(SPTR sym) {
  return static_cast<SPTR>(mk_address(sym));
}
inline SPTR GetVTable(SPTR sym) {
  return static_cast<SPTR>(VTABLEG(sym));
}
#undef VTABLEG
#define VTABLEG GetVTable
inline SPTR GetIface(SPTR sym) {
  return static_cast<SPTR>(IFACEG(sym));
}
#undef IFACEG
#define IFACEG GetIface
#else
#define convertSPTR(X)  X
#define sptr_mk_address mk_address
#endif

static bool
strislen1(STRDESC *str)
{
  return str->liscon && str->lval == 1;
}

static bool
strislen0(STRDESC *str)
{
  return str->liscon && str->lval == 0;
}

static int
getstraddr(STRDESC *str)
{
  if (str->aisvar)
    return ad1ili(IL_ACON, str->aval);
  return str->aval;
}

static int
getstrlen(STRDESC *str)
{
  if (str->liscon)
    return ad_icon(str->lval);
  return str->lval;
}

static int
getstrlen64(STRDESC *str)
{
  int il;
  il = getstrlen(str);
  if (IL_RES(ILI_OPC(il)) != ILIA_KR)
    il = ad1ili(IL_IKMV, il);
  return il;
}

/*
 * Generating GSMOVE ILI is under XBIT(2,0x800000). When the XBIT is not
 * set, _exp_smove() will proceed as before; in particular, chk_block() is
 * called to add terminal ILI to the block current to the expander.   When
 * the XBIT is set, the GSMOVE ili are transformed sometime after the expander,
 * but we still want the code in _exp_smove() to do the work. However, we
 * cannot call chk_block() to add the terminal ILI; we must use 'addilt'.
 * So, define and use a function pointer, p_chk_block, which calls either
 * chk_block() or a new local addilit routine, gsmove_chk_block().  In this
 * case, the current ilt is saved as the file static, gsmove_ilt.
 */
static void (*p_chk_block)(int) = chk_block;
static void gsmove_chk_block(int);
static int gsmove_ilt;

/* aux.curr_entry->flags description:
 *     Initialized to 0 by exp_end
 * NA      0x1  -  need to save argument registers  (set by exp_end).
 * NA      0x2  -  r1 is not needed (set by scheduler)
 * NA      0x4  -  function contained varargs or is passed memory arguments
 *                 (set by exp_end)
 * NA      0x8  -  fast linkage
 *       0x100  -  must set up the frame (set by exp_end)
 *       0x200  -  AVX only: we can 32-byte align the stack if it is
 *                   beneficial to do so (set by exp_end)
 *       0x400  -  AVX only: we MUST 32-byte align the stack, e.g. because
 *                   a 32-byte aligned load or store has been generated
 *                   which assumes that the stack is 32-byte aligned.
 *  0x40000000  -  mscall seen
 *  0x80000000  -  alloca called.
 */

int
is_passbyval_dummy(int sptr)
{
  if (BYVALDEFAULT(GBL_CURRFUNC))
    return 1;
  if (PASSBYVALG(sptr))
    return 1;
  return 0;
}

/* Visual Studio cDEC$ ATTRIBUTES are very specific about when a character
   argument is passed by value, passed by ref with a length, passed
   by ref without a length.  This routine returns true if the argument
   is pass by reference with a length
 */
int
needlen(int sym, int func)
{
  if (sym <= 0)
    return false;

  if (func <= 0)
    return false;

  if (sym == FVALG(func)) {

    /* special case for functions returning character :
       always need a length This can not be modified
       any ATTRIBUTES.
     */
    return true;
  }

  if (PASSBYVALG(sym)) {
    return false;
  }
  if (STDCALLG(func) || CFUNCG(func)) {
    if (PASSBYREFG(sym)) {
      return false;
    }

    if (PASSBYREFG(func)) {
      return true;
    }

    /* plain func= c/stdcall is pass by value */
    return false;
  }
  return true;
}

static void
create_llvm_display_temp(void)
{
  DTYPE dtype;
  SPTR display_temp, asym;

  if (!gbl.internal)
    return;

  display_temp = getccsym('S', expb.gentmps++, ST_VAR);

  if (gbl.outlined) {
    SCP(display_temp, SC_PRIVATE);
    if (gbl.internal >= 1)
      load_uplevel_addresses(display_temp);
  } else if (gbl.internal == 1) {
    dtype = DTYPEG(display_temp);
    if (DTY(dtype) != TY_STRUCT)
      dtype = make_uplevel_arg_struct();
    DTYPEP(display_temp, dtype);
    SCP(display_temp, SC_LOCAL);
    ADDRTKNP(display_temp, 1);
    sym_is_refd(display_temp);
    aux.curr_entry->display = display_temp;

    if (!gbl.outlined) {
      /* now load address of local variable on to this array */
      load_uplevel_addresses(display_temp);
    }
    return;
  } else {
    SCP(display_temp, SC_DUMMY);
    dtype = DTYPEG(display_temp);
    if (DTY(dtype) != TY_STRUCT)
      dtype = make_uplevel_arg_struct();
    asym = mk_argasym(display_temp);
    ADDRESSP(asym, ADDRESSG(display_temp)); /* propagate ADDRESS */
    MEMARGP(asym, 1);
  }
  DTYPEP(display_temp, DT_ADDR);
  sym_is_refd(display_temp);
  aux.curr_entry->display = display_temp;
}

/***************************************************************/

/**
 * Expand entry, main, sub, or func.  For an unnamed program, PROGRAM,
 * SUBROUTINE, or FUNCTION, sym is 0; otherwise, sym is the ENTRY name.
 */
void
exp_header(SPTR sym)
{
  if (sym == SPTR_NULL) {
    smove_flag = 0;
    mscall_flag = 0;
    if (WINNT_CALL)
      mscall_flag = 1;
    alloca_flag = 0;
    sym = gbl.currsub;
    allocharhdr = SPTR_NULL;
    memarg_var = SPTR_NULL;
    expb.arglcnt.next = expb.arglcnt.start = expb.arglcnt.max;
    aux.curr_entry->ent_save = SPTR_NULL;
    if (gbl.rutype != RU_PROG) {
      if ((!WINNT_CALL && !CREFG(sym)) || NOMIXEDSTRLENG(sym))
        pp_entries();
      else
        pp_entries_mixedstrlen();
    }
    mkrtemp_init();
  } else {
    if (flg.smp && OUTLINEDG(sym) && BIHNUMG(sym)) {
      return;
    }
    flsh_block();
    cr_block();
  }

  /* get expb.curbih for this entry and save in symtab */

  BIHNUMP(sym, expb.curbih);

  /* generate ILI for entry operator */

  expb.curilt = addilt(0, ad1ili(IL_ENTRY, sym));
  /*
   * Store into the bih for this block the entry ST item and define
   * the pointer to the auxilary Entry information and the BIH index
   * for the current function.
   */
  BIH_LABEL(expb.curbih) = sym;
#ifdef OUTLINEDG
  gbl.outlined = ((OUTLINEDG(sym)) ? true : false);
#endif

  if (sym == gbl.currsub)
    reg_init(sym); /* init reg info and set stb.curr_entry */
  if (gbl.internal >= 1) {
    /* always create display variable for gbl.internal */
    create_llvm_display_temp();
  }

  if (gbl.outlined) {
    SPTR asym;
    int ili_uplevel;
    SPTR tmpuplevel;
    int nme, ili;
    bihb.parfg = 1;
    aux.curr_entry->uplevel = ll_get_shared_arg(sym);
    asym = mk_argasym(aux.curr_entry->uplevel);
    ADDRESSP(asym, ADDRESSG(aux.curr_entry->uplevel)); /* propagate ADDRESS */
    MEMARGP(asym, 1);

    /* if I am the task_routine(arg1, task*) */
    if (TASKFNG(sym)) {
      bihb.taskfg = 1;

      /* Set up local variable and store the address where first shared
       * variable is stored.
       */
      tmpuplevel = getccsym('S', expb.gentmps++, ST_VAR);
      SCP(tmpuplevel, SC_PRIVATE);
      DTYPEP(tmpuplevel, DT_ADDR);
      sym_is_refd(tmpuplevel);
      ENCLFUNCP(tmpuplevel, GBL_CURRFUNC);

      /* aux.curr_entry->uplevel = arg2[0] */
      /* 2 levels of indirection.
       * 1st: Fortran specific where we load address of
       *      argument from address constant variable.
       *      We store the address of argument into
       *      address constant at the beginning of routine.
       *      We should one day revisit if it is applicable anymore.
       *      Or if we should just do the same as C.
       *      We would now have an address of task
       * 2nd: Load first element from task which should be the
       *      address on task_sptr where first shared var address
       *      is stored.
       */
      ili_uplevel = mk_address(aux.curr_entry->uplevel);
      nme = addnme(NT_VAR, asym, 0, 0);
      ili_uplevel = ad2ili(IL_LDA, ili_uplevel, nme); /* .Cxxx = (task) */
      nme = addnme(NT_IND, aux.curr_entry->uplevel, nme, 0);
      ili_uplevel = ad2ili(IL_LDA, ili_uplevel, nme); /* taskptr = .Cxxx */

      ili = ad_acon(tmpuplevel, 0);
      nme = addnme(NT_VAR, tmpuplevel, 0, 0);
      ili = ad3ili(IL_STA, ili_uplevel, ili, nme);
      chk_block(ili);
      aux.curr_entry->uplevel = tmpuplevel;
    }
  } else if (ISTASKDUPG(sym)) {
    SPTR asym;
    int ili_uplevel;
    SPTR tmpuplevel;
    int nme, ili;
    aux.curr_entry->uplevel = ll_get_hostprog_arg(sym, 2);
    asym = mk_argasym(aux.curr_entry->uplevel);
    ADDRESSP(asym, ADDRESSG(aux.curr_entry->uplevel)); /* propagate ADDRESS */
    MEMARGP(asym, 1);

    bihb.taskfg = 1;

    /* Set up local variable and store the address of shared variable
     * from second argument: taskdup(nexttask, task, lastitr)
     * So that we don't need to do multiple indirect access when
     * we want to access shared variable.
     */
    tmpuplevel = getccsym('S', expb.gentmps++, ST_VAR);
    SCP(tmpuplevel, SC_PRIVATE);
    DTYPEP(tmpuplevel, DT_ADDR);
    sym_is_refd(tmpuplevel);
    ENCLFUNCP(tmpuplevel, GBL_CURRFUNC);

    /* now load address from arg2[0] to tmpuplevel */
    ili_uplevel = mk_address(aux.curr_entry->uplevel);
    nme = addnme(NT_VAR, asym, 0, 0);

    /* 2 levels of indirection.
     * 1st: Fortran specific where we load address of
     *      argument from address constant variable.
     *      We store the address of argument into
     *      address constant at the beginning of routine.
     *      We should one day revisit if it is applicable anymore.
     *      Or if we should just do the same as C.
     *      We would now have an address of task
     * 2nd: Load first element from task which should be the
     *      address on task_sptr where the first shared var
     *      address is stored.
     */
    ili_uplevel = ad2ili(IL_LDA, ili_uplevel, nme); /* .Cxxx = (task) */
    nme = addnme(NT_IND, aux.curr_entry->uplevel, nme, 0);
    ili_uplevel = ad2ili(IL_LDA, ili_uplevel, nme); /* taskptr = .Cxxx */

    ili = ad_acon(tmpuplevel, 0);
    nme = addnme(NT_VAR, tmpuplevel, 0, 0);
    ili = ad3ili(IL_STA, ili_uplevel, ili, nme);
    chk_block(ili);
    aux.curr_entry->uplevel = tmpuplevel;
  } else {
    bihb.parfg = 0;
    bihb.taskfg = 0;
    aux.curr_entry->uplevel = SPTR_NULL;
  }

  BIH_EN(expb.curbih) = 1;
  gbl.entbih = expb.curbih;
  if (gbl.rutype != RU_PROG) {
    if ((!WINNT_CALL && !CREFG(sym)) || NOMIXEDSTRLENG(sym))
      pp_params(sym);
    else
      pp_params_mixedstrlen(sym);
  }

  if (gbl.internal && gbl.outlined && aux.curr_entry->display) {
    /* do this after aux->curr_entry.display is created:  */
    int ili_uplevel;
    int nme;
    int ili = ad_acon(aux.curr_entry->display, 0);
    aux.curr_entry->uplevel = ll_get_shared_arg(sym);
    ili_uplevel = mk_address(aux.curr_entry->uplevel);
    nme = addnme(NT_VAR, aux.curr_entry->uplevel, 0, 0);
    ili_uplevel = ad2ili(IL_LDA, ili_uplevel, nme);
    ili_uplevel = ad2ili(IL_LDA, ili_uplevel,
                         addnme(NT_IND, aux.curr_entry->display, nme, 0));
    ili = ad2ili(IL_LDA, ili, addnme(NT_IND, aux.curr_entry->display, nme, 0));
    nme = addnme(NT_VAR, aux.curr_entry->display, 0, 0);
    ili = ad3ili(IL_STA, ili_uplevel, ili, nme);
    chk_block(ili);
    flg.recursive = true;
  }
  if (flg.debug || XBIT(120, 0x1000) || XBIT(123, 0x400)) {
    /*
     * Since the debug code is produced, the entry block will have
     * line number of 0.  The block following the entry block will
     * have the entry's line number.  This block represents the entry
     * to the function as seen by the debugger.
     */
    BIH_LINENO(expb.curbih) = 0;
    wr_block();
    cr_block();
    BIH_LINENO(expb.curbih) = gbl.lineno;
  } else {
    wr_block(); /* make entry block separate */
    cr_block();
  }
}

/*
 * WARNING: there are nomixedstrlen and mixedstrlen functions to preprocess
 * entries.
 */
static void
pp_entries(void)
{
  int func;
  int nargs;
  int *dpdscp;
  int sym;
  int curpos;
  int pos;
  int lenpos;
  int argpos;
  int finfox;
  int byvalue;
  /*
   * Preprocess the entries in the subprogram to determine for which
   * entries arguments must be copied due to the arguments occupying
   * different positions.  The entry and the arguments which must
   * be copied are flagged (COPYPRMS flag).  Also, for a character
   * argument whose length is passed, a symbol table entry is created
   * to represent its length (the arg's CLEN field will locate the length
   * ST item).
   *
   * A unique list (table) is created (located by parg) of the arguments
   * and lengths for character arguments which appear in all of the entries.
   * While a function is processed, a section of the table is divided into
   * two tables:  the first table is used for the arguments and the second
   * table is used for lengths.  argpos is an index into the table and
   * locates the position of the most recent unique argument; lenpos indicates
   * the position of the most recent character length.
   *
   * Note that the ADDRESS field is temporarily used to record the
   * argument's position in the list created for all the arguments.
   * An argument is entered into the list only once even though it
   * may occur in more than one entry.
   */

  /* compute number of entries and total number of arguments */
  finfox = retgrp_cnt = nentries = nargs = 0;
  for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
    nargs += PARAMCTG(func);
    nentries++;
  }
  /*
   * assume all arguments are character arguments;  note that the first
   * argument is in position 1.  Allocate space for the table used to
   * record arguments and lengths and space for the finfo table (to be
   * used by pp_params).
   */
  nargs = 2 * nargs + 1;
  parg = (int *)getitem(1, sizeof(int) * nargs);

  pfinfo = (finfo_t *)getitem(1, sizeof(finfo_t) * nentries);
  BZERO(pfinfo, finfo_t, nentries);

  argpos = 0;
  for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
    int savlenpos, i, total_words;

    total_words = 0;
    MIDNUMP(func, finfox++); /* remember index to func's finfo */
    nargs = PARAMCTG(func);
    dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
    curpos = 0;
    if (gbl.rutype != RU_FUNC)
      goto scan_args;
    /*
     * enter the function return variable into the group return table
     * (table is shared with the finfo table) if not already there.
     */
    for (i = 0; i < retgrp_cnt; i++)
      if (pfinfo[i].fval == FVALG(func)) {
        pfinfo[MIDNUMG(func)].retgrp = i;
        if (EXPDBG(8, 256))
          fprintf(gbl.dbgfil, "%s shares group %d\n", SYMNAME(func), i);
        goto check_type;
      }
    pfinfo[retgrp_cnt].fval = FVALG(func);
    pfinfo[MIDNUMG(func)].retgrp = retgrp_cnt;
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "%s enters group %d, %s\n", SYMNAME(func), retgrp_cnt,
              SYMNAME(FVALG(func)));
    retgrp_cnt++;

  check_type:
    switch (DTY(DTYPEG(func))) {
    case TY_CHAR:
    case TY_NCHAR:
      /* NOTE: if function returns char, then all entries return char
       */
      if (func == gbl.currsub) {
        sym = dpdscp[nargs - 1];
        parg[1] = sym;
        if (needlen(sym, func) &&
            (DTYPEG(func) == DT_ASSCHAR || DTYPEG(func) == DT_DEFERCHAR ||
             DTYPEG(func) == DT_DEFERNCHAR || DTYPEG(func) == DT_ASSNCHAR)) {
          int clen = CLENG(sym);
          if (clen == 0 || !REDUCG(clen)) {
            clen = getdumlen();
            CLENP(sym, clen);
          }
          parg[2] = clen;
          ADDRESSP(clen, 2);
        } else
          parg[2] = -sym;
        ADDRESSP(sym, 1);
        argpos = 2;
      }
      curpos = 2;
      nargs--;
      total_words += 2;
      break;
    case TY_CMPLX:
    case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
#endif
      /* for complex functions, an extra argument is the first argument
       * which is also used to return the result.
       */
      if (CFUNCG(func) || CMPLXFUNC_C) {
        break;
      }
      curpos = 1;
      sym = dpdscp[nargs - 1];
      pos = ADDRESSG(sym) & 0xffff;
      if (pos == 0) {
        parg[++argpos] = sym;
        ADDRESSP(sym, argpos);
        pos = argpos;
      }
      if (pos != curpos) {
        COPYPRMSP(func, 1);
        COPYPRMSP(sym, 1);
      }
      nargs--;
      total_words++;
      break;
    default:
      break;
    }

  scan_args:
    savlenpos = lenpos = argpos + nargs;

    while (nargs--) {
      int osym;
      DTYPE dt;
      curpos++;
      sym = *dpdscp;
      osym = sym;

      if (((DTY(DTYPEG(sym))) == TY_STRUCT) ||
          ((DTY(DTYPEG(sym))) == TY_ARRAY) || ((DTY(DTYPEG(sym))) == TY_UNION))
        /* no passbyvalue arrays, structs */
        byvalue = 0;
      else
        byvalue = BYVALDEFAULT(func);

      if (PASSBYVALG(sym))
        byvalue = 1;
      if (PASSBYREFG(sym))
        byvalue = 0;

      if (SCG(sym) == SC_BASED && MIDNUMG(sym) && XBIT(57, 0x80000) &&
          SCG(MIDNUMG(sym)) == SC_DUMMY) {
        /* for char, we put pointee in argument list so as to get
         * the char length here, but we really pass the pointer
         * use the actual pointer */
        sym = MIDNUMG(sym);
      }
      dpdscp++;
      pos = ADDRESSG(sym) & 0xffff;
      if (pos == 0) {
        parg[++argpos] = sym;
        ADDRESSP(sym, argpos);
        pos = argpos;
      }
      if (pos != curpos) {
        COPYPRMSP(func, 1);
        COPYPRMSP(sym, 1);
      }
      total_words++;
      dt = DDTG(DTYPEG(osym));

      if (byvalue) {
        if (DTY(dt) == TY_DBLE || DTY(dt) == TY_INT8 || DTY(dt) == TY_LOG8 ||
            DTY(dt) == TY_CMPLX)
          total_words++;
        else if (DTY(dt) == TY_DCMPLX || DTY(dt) == TY_QUAD)
          total_words += 3;
        else if ((DTY(dt) == TY_STRUCT && (size_of(DTYPEG(osym)) > 4)) || DTY(dt) == TY_QCMPLX)
          total_words += size_of(DTYPEG(osym)) / 4 - 1;
      }

      /*
       * save length if character
       */
      if ((DTYG(DTYPEG(osym)) == TY_CHAR || DTYG(DTYPEG(osym)) == TY_NCHAR) &&
          needlen(osym, func)) {
        parg[++lenpos] = osym;
        total_words++;
      }
    }
    /*
     * all arguments have been processed for func; process the lengths
     * which have been saved in the table.  Since there could be a gap
     * between the arguments and the lengths, the lengths which are seen
     * for the first time are moved up to follow the arguments.
     */
    while (savlenpos < lenpos) {
      int osym;

      savlenpos++;
      curpos++;
      sym = parg[savlenpos];
      osym = sym;
      if (SCG(sym) == SC_BASED && MIDNUMG(sym) && XBIT(57, 0x80000) &&
          SCG(MIDNUMG(sym)) == SC_DUMMY) {
        /* for char, we put pointee in argument list so as to get
         * the char length here, but we really pass the pointer
         * use the actual pointer */
        sym = MIDNUMG(sym);
      }
      pos = (ADDRESSG(sym) >> 16) & 0xffff;
      if (pos == 0) {
        ++argpos;
        ADDRESSP(sym, argpos << 16 | ADDRESSG(sym));
        if (needlen(sym, func) && (DDTG(DTYPEG(osym)) == DT_ASSCHAR ||
                                   DDTG(DTYPEG(osym)) == DT_DEFERCHAR ||
                                   DDTG(DTYPEG(osym)) == DT_DEFERNCHAR ||
                                   DDTG(DTYPEG(osym)) == DT_ASSNCHAR)) {
          int clen;
          clen = CLENG(osym);
          if (clen == 0) {
            clen = getdumlen();
            CLENP(osym, clen);
            parg[argpos] = clen;
          } else if (REDUCG(clen)) {
            parg[argpos] = clen;
          } else {
            /* adjustable length dummy */
            parg[argpos] = -sym;
            AUTOBJP(osym, 1); /* mark as adjustable length */
          }
        } else
          parg[argpos] = -sym;
        pos = argpos;
      }
      if (pos != curpos &&
          (DDTG(DTYPEG(osym)) == DT_ASSCHAR ||
           DDTG(DTYPEG(osym)) == DT_DEFERCHAR ||
           DDTG(DTYPEG(osym)) == DT_DEFERNCHAR ||
           DDTG(DTYPEG(osym)) == DT_ASSNCHAR)
          && !AUTOBJG(osym)
      ) {
        sym = CLENG(osym);
#if DEBUG
        assert(sym != 0, "pp_entries: 0 clen", parg[savlenpos], ERR_Severe);
#endif
        parg[pos] = sym;
        COPYPRMSP(sym, 1);
        COPYPRMSP(func, 1);
      }
    }
#if defined(TARGET_WIN)
    if (MSCALLG(func)) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "%s total_words %d\n", SYMNAME(func), total_words);
      if (total_words > 0) {
        ARGSIZEP(func, total_words * 4);
      } else if (total_words == 0)
        ARGSIZEP(func, -1);
    }
#endif
  }
  for (pos = 1; pos <= argpos; pos++) {
    sym = parg[pos];
    if (sym > 0) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "%4d: %s   %s\n", pos, SYMNAME(sym),
                COPYPRMSG(sym) ? "<copied>" : "");
      ADDRESSP(sym, 0);
    } else if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "%4d: length of %s\n", pos, SYMNAME(-sym));
  }

  if (retgrp_cnt > 1) {
    retgrp_var = getccsym('F', 0, ST_VAR);
    SCP(retgrp_var, SC_LOCAL);
    DTYPEP(retgrp_var, DT_INT);
  }
}

/*
 * WARNING: there are nomixedstrlen and mixedstrlen functions to preprocess
 * entries.
 */
static void
pp_entries_mixedstrlen(void)
{
  int func;
  int nargs;
  int *dpdscp;
  SPTR sym;
  int curpos;
  int pos;
  int argpos;
  int finfox;
  int byvalue = 0;
  /*
   * Preprocess the entries in the subprogram to determine for which
   * entries arguments must be copied due to the arguments occupying
   * different positions.  The entry and the arguments which must
   * be copied are flagged (COPYPRMS flag).  Also, for a character
   * argument whose length is passed, a symbol table entry is created
   * to represent its length (the arg's CLEN field will locate the length
   * ST item).
   *
   * A unique list (table) is created (located by parg) of the arguments
   * and lengths for character arguments which appear in all of the entries.
   * While a function is processed, a section of the table is divided into
   * two tables:  the first table is used for the arguments and the second
   * table is used for lengths.  argpos is an index into the table and
   * locates the position of the most recent unique argument; lenpos indicates
   * the position of the most recent character length.
   *
   * Note that the ADDRESS field is temporarily used to record the
   * argument's position in the list created for all the arguments.
   * An argument is entered into the list only once even though it
   * may occur in more than one entry.
   */

  /* compute number of entries and total number of arguments */
  finfox = retgrp_cnt = nentries = nargs = 0;
  for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
    nargs += PARAMCTG(func);
    nentries++;
  }
  if (nentries > 1) {
    sym = getccsym('Q', expb.gentmps++, ST_VAR);
    SCP(sym, SC_LOCAL);
    DTYPEP(sym, DT_INT);
    aux.curr_entry->ent_save = sym;
    ADDRTKNP(sym, 1); /* so optimizer won't delete */
  }
  /*
   * assume all arguments are character arguments;  note that the first
   * argument is in position 1.  Allocate space for the table used to
   * record arguments and lengths and space for the finfo table (to be
   * used by pp_params).
   */
  nargs = 2 * nargs + 1;
  parg = (int *)getitem(1, sizeof(int) * nargs);

  pfinfo = (finfo_t *)getitem(1, sizeof(finfo_t) * nentries);
  BZERO(pfinfo, finfo_t, nentries);

  argpos = 0;
  for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
    int i, total_words;

    total_words = 0;
    MIDNUMP(func, finfox++); /* remember index to func's finfo */
    nargs = PARAMCTG(func);
    dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
    curpos = 0;
    if (gbl.rutype != RU_FUNC)
      goto scan_args;
    /*
     * enter the function return variable into the group return table
     * (table is shared with the finfo table) if not already there.
     */
    for (i = 0; i < retgrp_cnt; i++)
      if (pfinfo[i].fval == FVALG(func)) {
        pfinfo[MIDNUMG(func)].retgrp = i;
        if (EXPDBG(8, 256))
          fprintf(gbl.dbgfil, "%s shares group %d\n", SYMNAME(func), i);
        goto check_type;
      }
    pfinfo[retgrp_cnt].fval = FVALG(func);
    pfinfo[MIDNUMG(func)].retgrp = retgrp_cnt;
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "%s enters group %d, %s\n", SYMNAME(func), retgrp_cnt,
              SYMNAME(FVALG(func)));
    retgrp_cnt++;

  check_type:
    switch (DTY(DTYPEG(func))) {
    case TY_CHAR:
    case TY_NCHAR:
      /* NOTE: if function returns char, then all entries return char
       */
      if (func == gbl.currsub) {
        sym = convertSPTR(dpdscp[nargs - 1]);
        parg[1] = sym;
        if ((DTYPEG(func) == DT_ASSCHAR || DTYPEG(func) == DT_DEFERCHAR ||
             DTYPEG(func) == DT_DEFERNCHAR || DTYPEG(func) == DT_ASSNCHAR)) {
          int clen = CLENG(sym);
          if (clen == 0 || !REDUCG(clen)) {
            clen = getdumlen();
            CLENP(sym, clen);
          }
          parg[2] = CLENG(sym);
          ADDRESSP(parg[2], 2);
        } else
          parg[2] = -sym;
        ADDRESSP(sym, 1);
        argpos = 2;
      }
      curpos = 2;
      nargs--;
      total_words++;
      /* character length */
      if (needlen(sym, func)) {
        total_words++;
      }

      break;
    case TY_CMPLX:
    case TY_DCMPLX:
      /* for complex functions, an extra argument is the first argument
       * which is also used to return the result.
       */
      curpos = 1;
      sym = convertSPTR(dpdscp[nargs - 1]);
      pos = ADDRESSG(sym) & 0xffff;
      if (pos == 0) {
        parg[++argpos] = sym;
        ADDRESSP(sym, argpos);
        pos = argpos;
      }
      if (pos != curpos) {
        COPYPRMSP(func, 1);
        COPYPRMSP(sym, 1);
      }
      nargs--;
      total_words++;
      break;
    default:
      break;
    }

  scan_args:
    while (nargs--) {
      int osym;
      DTYPE dt;
      curpos++;
      sym = convertSPTR(*dpdscp);
      osym = sym;

      if (((DTY(DTYPEG(sym))) == TY_STRUCT) ||
          ((DTY(DTYPEG(sym))) == TY_ARRAY) || ((DTY(DTYPEG(sym))) == TY_UNION))
        /* no passbyvalue arrays, structs */
        byvalue = 0;
      else
        byvalue = BYVALDEFAULT(func);

      if (PASSBYVALG(sym))
        byvalue = 1;
      if (PASSBYREFG(sym))
        byvalue = 0;

      if (SCG(sym) == SC_BASED && MIDNUMG(sym) && XBIT(57, 0x80000) &&
          SCG(MIDNUMG(sym)) == SC_DUMMY) {
        /* char pointers, we put the pointee on the argument
         * list so as to get the char length, but we really pass
         * the pointer.
         * replace by the actual pointer */
        sym = MIDNUMG(sym);
      }
      dpdscp++;
      pos = ADDRESSG(sym) & 0xffff;
      if (pos == 0) {
        parg[++argpos] = sym;
        ADDRESSP(sym, argpos);
        pos = argpos;
      }
      if (pos != curpos) {
        COPYPRMSP(func, 1);
        COPYPRMSP(sym, 1);
      }
      total_words++;
      dt = DDTG(DTYPEG(osym));

      if (byvalue) {
        if (DTY(dt) == TY_DBLE || DTY(dt) == TY_INT8 || DTY(dt) == TY_LOG8 ||
            DTY(dt) == TY_CMPLX)
          total_words++;
        else if (DTY(dt) == TY_DCMPLX || DTY(dt) == TY_QUAD)
          total_words += 3;
        else if ((DTY(dt) == TY_STRUCT && (size_of(DTYPEG(osym)) > 4)) || DTY(dt) == TY_QCMPLX)
          total_words += size_of(DTYPEG(osym)) / 4 - 1;
      }

      /*
       * save length if character
       */
      if (DTY(dt) == TY_CHAR || DTY(dt) == TY_NCHAR) {
        curpos++;
        pos = (ADDRESSG(sym) >> 16) & 0xffff;
        if (pos == 0) {
          pos = ++argpos;
          ADDRESSP(sym, argpos << 16 | ADDRESSG(sym));
          if (needlen(sym, func) &&
              (dt == DT_ASSCHAR || dt == DT_ASSNCHAR || dt == DT_DEFERCHAR ||
               dt == DT_DEFERNCHAR)) {
            int clen;
            clen = CLENG(osym);
            if (clen == 0) {
              clen = getdumlen();
              CLENP(osym, clen);
              parg[argpos] = CLENG(osym);
            } else if (REDUCG(clen)) {
              parg[argpos] = clen;
            } else {
              /* adjustable length dummy */
              parg[argpos] = -sym;
              AUTOBJP(osym, 1); /* mark as adjustable length */
            }
          } else
            parg[argpos] = -sym;
        }
        if (pos != curpos &&
            (dt == DT_ASSCHAR || dt == DT_ASSNCHAR || dt == DT_DEFERCHAR ||
             dt == DT_DEFERNCHAR)
            && !AUTOBJG(osym)
        ) {
          sym = CLENG(osym);
#if DEBUG
          assert(sym != 0, "pp_entries_mixedstrlen: 0 clen", parg[pos],
                 ERR_Severe);
#endif
          COPYPRMSP(sym, 1);
          COPYPRMSP(func, 1);
        }
        if (needlen(sym, func)) {
          total_words++;
        }
      }
    }
    if (WINNT_CALL) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "%s total_words %d\n", SYMNAME(func), total_words);
      if (total_words > 0) {
        ARGSIZEP(func, total_words * 4);
      } else if (total_words == 0)
        ARGSIZEP(func, -1);
    }
  }
  for (pos = 1; pos <= argpos; pos++) {
    sym = convertSPTR(parg[pos]);
    if (sym > 0) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "%4d: %s   %s\n", pos, SYMNAME(sym),
                COPYPRMSG(sym) ? "<copied>" : "");
      ADDRESSP(sym, 0);
    } else if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "%4d: length of %s\n", pos, SYMNAME(-sym));
  }

  if (retgrp_cnt > 1) {
    retgrp_var = getccsym('F', 0, ST_VAR);
    SCP(retgrp_var, SC_LOCAL);
    DTYPEP(retgrp_var, DT_INT);
  }
}

SPTR
getdumlen(void)
{
  SPTR sym = getccsym('U', expb.chardtmps++, ST_VAR);
  if (CHARLEN_64BIT) {
    DTYPEP(sym, DT_INT8);
  } else {
    DTYPEP(sym, DT_INT);
  }
  SCP(sym, SC_DUMMY);
  REDUCP(sym, 1);     /* mark temp as char len dummy */
  PASSBYVALP(sym, 1); /* Char len dummies are passed by value */
  return sym;
}

SPTR
gethost_dumlen(int arg, ISZ_T address)
{
  SPTR sym = getccsym('U', arg, ST_VAR);
  if (CHARLEN_64BIT) {
    DTYPEP(sym, DT_INT8);
  } else {
    DTYPEP(sym, DT_INT);
  }
  SCP(sym, SC_DUMMY);
  ADDRESSP(sym, address);
  REDUCP(sym, 1); /* mark temp as char len dummy */
  UPLEVELP(sym, 1);
  PASSBYVALP(sym, 1);
  pop_sym(sym); /* don't let this symbol conflict with getdumlen() */
  return sym;
}

static int
exp_type_bound_proc_call(int arg, SPTR descno, int vtoff, int arglnk)
{

  SPTR sym;
  int ili;
  int type_offset, vft_offset, func_offset, sz;
  int jsra_mscall_flag;

  sym = descno;

  if (XBIT(68, 0x1)) {
    type_offset = 72;
    vft_offset = 80;
  } else {
    type_offset = 40;
    vft_offset = 48;
  }
  func_offset = 8 * (vtoff - 1);
  sz = MSZ_I8;
  ADDRTKNP(sym, 1);
  if (SCG(sym) == SC_EXTERN) {
    int ili2;
    SPTR tmp = getccsym_sc('Q', expb.gentmps++, ST_VAR, SC_LOCAL);

    DTYPEP(tmp, DT_ADDR);

    ili = ad1ili(IL_ACON, get_acon(sym, 0));

    ili2 = ad1ili(IL_ACON, get_acon(tmp, 0));

    ili = ad3ili(IL_STA, ili, ili2, NME_UNK);
    chk_block(ili);

    ili = ad2ili(IL_LDA, ili2, NME_UNK);
    ili = ad3ili(IL_AADD, ili, ad_aconi(vft_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
    ili = ad3ili(IL_AADD, ili, ad_aconi(func_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
  } else if (SCG(sym) != SC_DUMMY) {
    ili = mk_address(sym);
    ili = ad3ili(IL_AADD, ili, ad_aconi(type_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
    ili = ad3ili(IL_AADD, ili, ad_aconi(vft_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
    ili = ad3ili(IL_AADD, ili, ad_aconi(func_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
  } else {
    if (!TASKDUPG(gbl.currsub) && CONTAINEDG(gbl.currsub) && INTERNREFG(sym)) {
      ili = mk_address(sym);
    } else {
      const SPTR asym = mk_argasym(sym);
      const int addr = mk_address(sym);
      ili = ad2ili(IL_LDA, addr, addnme(NT_VAR, asym, 0, 0));
    }
    ili = ad3ili(IL_AADD, ili, ad_aconi(type_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
    ili = ad3ili(IL_AADD, ili, ad_aconi(vft_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
    ili = ad3ili(IL_AADD, ili, ad_aconi(func_offset), 0);
    ili = ad2ili(IL_LDA, ili, NME_UNK);
  }

  if (!MSCALLG(arg))
    jsra_mscall_flag = 0;
  else
    jsra_mscall_flag = 0x1;

  return ad4ili(IL_JSRA, ili, arglnk, jsra_mscall_flag, fptr_iface);
}

#ifdef FLANG2_EXPRTE_UNUSED
static int
has_desc_arg(int func, int sptr)
{

  int argsym, nargs, *dpdscp, i;
  dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
  nargs = PARAMCTG(func);

  for (i = 0; i < nargs; ++i) {
    argsym = dpdscp[i];
    if (SDSCG(sptr) == argsym)
      return 1;
  }
  return 0;
}
#endif

static int
check_desc(int func, int sptr)
{
  /* Called by check_desc_args() below. Swaps traditional descriptor arguments
   * with type descriptor arguments when they're out of order.
   */

  int nargs, *dpdscp, desc, *scratch;
  int pos, pos2, pos3, argsym, i, seenCC, seenDesc, seenSym, seenClass;
  int swap_from, swap_to, j, pos4, rslt;

  rslt = 0;
  desc = SDSCG(sptr);
  if (!desc)
    return 0;

  dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
  nargs = PARAMCTG(func);

  for (seenSym = seenDesc = seenCC = seenClass = pos = pos2 = pos3 = pos4 = i =
           0;
       i < nargs; ++i) {
    argsym = dpdscp[i];

    if (!seenSym &&
        (!SDSCG(argsym) || (SCG(SDSCG(argsym)) != SC_DUMMY &&
                            (!CLASSG(argsym) || FVALG(func) == argsym)))) {
      ++pos4;
    }
    if (argsym == sptr) {
      pos = i;
      seenSym = 1;
    } else if (argsym == desc) {
      pos2 = i;
      seenDesc = 1;
    }
    if (!pos3 && CCSYMG(argsym) && seenSym) {
      pos3 = i;
      seenCC = 1;
    }
    if (CLASSG(argsym)) {
      seenClass = 1;
    }
  }

  if (seenCC && seenDesc && seenSym && seenClass) {

    NEW(scratch, int, nargs);
    assert(scratch, "check_desc: out of memory!", 0, ERR_Fatal);
    swap_from = pos2;
    swap_to = pos3 + (pos - pos4);
    scratch[swap_to] = dpdscp[swap_from];
    for (j = i = 0; i < nargs && j < nargs;) {
      if (j == swap_to) {
        ++j;
        continue;
      }
      if (i == swap_from) {
        ++i;
        continue;
      }
      scratch[j] = dpdscp[i];
      ++j;
      ++i;
    }

    for (i = 0; i < nargs; ++i) {
      dpdscp[i] = scratch[i];
    }
    FREE(scratch);
    rslt = 1;
  }
  return rslt;
}

static void
check_desc_args(int func)
{
  /* Reorder arguments if we're mixing traditional descriptor arguments w/
   * type descriptor arguments since they get emitted at different times
   * in the front end.
   */
  int i, nargs, *dpdscp, argsym, swap;
  dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
  nargs = PARAMCTG(func);

  swap = 0;
  for (i = 0; i < nargs; ++i) {
    argsym = dpdscp[i];
    if (0 && CCSYMG(argsym))
      break;
    if (SDSCG(argsym)) {
      DESCARRAYP(SDSCG(argsym), 1); /* needed by type bound procedures */
      if (STYPEG(argsym) == ST_PROC) {
        /* needed when we have procedure dummy arguments with character
         * arguments
         */
        IS_PROC_DESCRP(SDSCG(argsym), 1);
      }

      if (check_desc(func, argsym))
        swap = 1;
    }
  }
}

bool
func_has_char_args(SPTR func)
{
  int i, nargs, *dpdscp;
  DTYPE argdtype;

  dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
  nargs = PARAMCTG(func);

  for (i = 0; i < nargs; ++i) {
    const SPTR argsym = convertSPTR(dpdscp[i]);
    argdtype = DTYPEG(argsym);
    if (DTYG(argdtype) == TY_CHAR || DTYG(argdtype) == TY_NCHAR)
      return true;
  }

  return false;
}

INLINE static int
check_struct(DTYPE dtype)
{
  if (ll_check_struct_return(dtype))
    return CLASS_INT4; /* something not CLASS_MEM */
  return CLASS_MEM;
}

static int
check_return(DTYPE retdtype)
{
  if (DTY(retdtype) == TY_STRUCT || DTY(retdtype) == TY_UNION ||
      DT_ISCMPLX(retdtype))
    return check_struct(retdtype);
  if (retdtype == DT_INT8) /* could be the fval of a C_PTR function */
    return CLASS_INT8;
  return CLASS_INT4; /* something not CLASS_MEM */
}

INLINE static void
align_struct_tmp(int sptr)
{
#if defined(X86_64)
  if (DTY(DTYPEG(sptr)) == TY_STRUCT && PDALNG(sptr) == 4) {
    return;
  }
#endif

  switch (alignment(DTYPEG(sptr))) {
  case 0:
  case 1:
  case 3:
    PDALNP(sptr, 2);
    break;
  case 7:
    PDALNP(sptr, 3);
    break;
  case 15:
    PDALNP(sptr, 4);
    break;
  case 31:
    PDALNP(sptr, 5);
    break;
  default:
#if DEBUG
    interr("align_struct_tmp: unexpected alignment", alignment(DTYPEG(sptr)),
           ERR_Severe);
#endif
    break;
  }
}

/**
   \brief Does the bind(c) function return the struct in register(s)?
   \param func_sym   the function's symbol
 */
bool
bindC_function_return_struct_in_registers(int func_sym)
{
  DEBUG_ASSERT(CFUNCG(func_sym), "function not bind(c)");
  return check_return(DTYPEG(func_sym)) != CLASS_MEM;
}

static void
handle_bindC_func_ret(int func, finfo_t *pf)
{
  int retdesc;
  int retsym = pf->fval;
  const DTYPE retdtype = DTYPEG(retsym);

  ADDRTKNP(retsym, 1);
  retdesc = check_return(retdtype);
  if (retdesc == CLASS_MEM) {
    /* Large struct: the address is passed in as an argument */
    SCP(retsym, SC_DUMMY);
    return;
  }
  align_struct_tmp(retsym);
  pf->ret_sm_struct = retdesc;
  pf->ret_align = alignment(retdtype);
}

static bool
process_end_of_list(SPTR func, SPTR osym, int *nlens, DTYPE argdtype)
{
  if ((needlen(osym, func) &&
       (DTYG(argdtype) == TY_CHAR || DTYG(argdtype) == TY_NCHAR)) ||
      (IS_PROC_DESCRG(osym) && !HAS_OPT_ARGSG(func) &&
       func_has_char_args(func))) {
    parg[*nlens] = osym;
    *nlens += 1;
    return true;
  }

  return false;
}

/*
 * WARNING: there are nomixedstrlen and mixedstrlen functions to preprocess
 * parameters.
 */
static void
pp_params(SPTR func)
{
  SPTR argsym;
  int asym;
  DTYPE argdtype;
  int nargs;
  int *dpdscp;
  int nlens;
  int byvalue;
  finfo_t *pf;

  check_desc_args(func);

  if (EXPDBG(8, 256))
    fprintf(gbl.dbgfil, "---pp_params: %s ---\n", SYMNAME(func));
  pf = &pfinfo[MIDNUMG(func)]; /* pfinfo alloc'd and init'd by pp_entries */
  argdtype = DTYPEG(func);
  dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
  nargs = PARAMCTG(func);
  nlens = 0;
  byvalue = 0;
  pf->mem_off = 8; /* offset for 1st dummy arg */
  if (gbl.rutype != RU_FUNC)
    goto scan_args;

  if (CFUNCG(func) || (CMPLXFUNC_C && DT_ISCMPLX(argdtype))) {
    handle_bindC_func_ret(func, &pfinfo[pf->retgrp]);
  }

  switch (DTY(argdtype)) {
  case TY_CHAR:
  case TY_NCHAR:
    /*
     * If this is a function which returns character, the first
     * two arguments are for the return length. The last entry in
     * the function's dpdsc auxiliary area is the "return" symbol.
     */
    argsym = convertSPTR(dpdscp[nargs - 1]);
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "func returns char, through %s\n", SYMNAME(argsym));
    MEMARGP(argsym, 1);
    ADDRESSP(argsym, 8);
    asym = mk_argasym(argsym);
    ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
    MEMARGP(asym, 1);
    argsym = CLENG(argsym);
    if (argsym) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "func return len in %s\n", SYMNAME(argsym));
      MEMARGP(argsym, 1);
      ADDRESSP(argsym, 12);
      asym = mk_argasym(argsym);
      ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
      MEMARGP(asym, 1);
    }
    pf->mem_off = 16; /* offset for 1st dummy arg */
    nargs--;
    break;
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
    /*
     * If this is a function which returns complex, the first arg is
     * also for the return value.  The last entry in the function's
     * dpdsc auxiliary area is the "return" symbol.
     */
    if (!CFUNCG(func) && !CMPLXFUNC_C) {
      argsym = convertSPTR(dpdscp[nargs - 1]);
      MEMARGP(argsym, 1);
      ADDRESSP(argsym, 8);
      asym = mk_argasym(argsym);
      ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
      MEMARGP(asym, 1);
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "func also returns complex, through %s\n",
                SYMNAME(argsym));
      pf->mem_off = 12; /* offset for 1st dummy arg */
      nargs--;
    }
    break;
  default:
    break;
  }
scan_args:
  /*
   * scan through all of the arguments of the function to compute
   * how (register or memory area) and where (reg # or offset) the
   * arguments area passed.  Also, generate the the ili if the argument
   * must be copied.  If a register argument is not copied, it is recorded
   * in the entry's finfo table; if the arg has been copied, a register
   * is still "assigned" but it is not recorded (slot is zero).
   *
   * The only concern for now is arguments which are addresses; the
   * exception is the lengths of character args (actually only those
   * which are passed length). If compiler is enhanced to allow value
   * parameters, presumably there will be some way to distinguish these
   * from reference arguments (i.e., a symbol table flag).
   */
  while (nargs--) {
    SPTR osym;
    argsym = convertSPTR(*dpdscp++);
    osym = argsym;
    argdtype = DTYPEG(osym);
    if (IS_PROC_DESCRG(osym) && !HAS_OPT_ARGSG(func) &&
        process_end_of_list(func, osym, &nlens, argdtype)) {
      continue;
    }
    if (((DTY(DTYPEG(argsym))) == TY_STRUCT) ||
        ((DTY(DTYPEG(argsym))) == TY_ARRAY) ||
        ((DTY(DTYPEG(argsym))) == TY_UNION))
      /* no passbyvalue arrays, structs */
      byvalue = 0;
    else
      byvalue = BYVALDEFAULT(func);

    if (PASSBYVALG(argsym))
      byvalue = 1;
    if (PASSBYREFG(argsym))
      byvalue = 0;
    if (SCG(argsym) == SC_BASED && MIDNUMG(argsym) && XBIT(57, 0x80000) &&
        SCG(MIDNUMG(argsym)) == SC_DUMMY) {
      /* for char, we put pointee in argument list so as to get
       * the char length here, but we really pass the pointer
       * use the actual pointer */
      argsym = MIDNUMG(argsym);
    }
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "%s in mem area at %d\n", SYMNAME(argsym),
              pf->mem_off);
    if (COPYPRMSG(argsym))
      cp_memarg(argsym, pf->mem_off, DT_ADDR);
    else if (DTY(argdtype) == TY_STRUCT) {
      REFP(MIDNUMG(argsym), 1);
      cp_memarg(argsym, pf->mem_off, DT_ADDR);
    } else {
      MEMARGP(argsym, 1);
      asym = mk_argasym(argsym);
      MEMARGP(asym, 1);
    }
    if (byvalue) {
      if (argdtype == DT_DBLE || argdtype == DT_INT8 || argdtype == DT_LOG8 ||
          argdtype == DT_CMPLX)
        pf->mem_off += 8;
      else if (argdtype == DT_DCMPLX || argdtype == DT_QUAD)
        pf->mem_off += 16;
      else if (DTY(argdtype) == TY_STRUCT || argdtype == DT_QCMPLX)
        pf->mem_off += size_of(argdtype);
      else
        pf->mem_off += 4;
      if (DTY(DTYPEG(argsym)) == TY_STRUCT) {
        int src_addr, n;
        int src_nme;
        int dest_addr;
        int dest_nme;
        SPTR newsptr = get_byval_local(argsym);
        dest_addr = ad_acon(newsptr, 0);
        dest_nme = addnme(NT_VAR, newsptr, 0, 0);
        src_addr = ad_acon(argsym, 0);
        src_nme = NME_VOL;
        n = size_of(DTYPEG(newsptr));
        chk_block(ad5ili(IL_SMOVEJ, src_addr, dest_addr, src_nme, dest_nme,
                         n));
      }
    } else {
      pf->mem_off += 4;
    }
    process_end_of_list(func, osym, &nlens, argdtype);

    if ((!HOMEDG(argsym) && (SCG(argsym) == SC_DUMMY)) &&
        (!PASSBYREFG(argsym)) &&
        (PASSBYVALG(argsym) ||
         (BYVALDEFAULT(func) && (((DTY(DTYPEG(argsym))) != TY_ARRAY) &&
                                 ((DTY(DTYPEG(argsym))) != TY_STRUCT) &&
                                 ((DTY(DTYPEG(argsym))) != TY_UNION))))) {
      if (!gbl.outlined && !ISTASKDUPG(GBL_CURRFUNC))
        cp_byval_mem_arg(argsym);
      PASSBYVALP(argsym, 1);
    }
  }
  /*
   * go through the list of character arguments. Here we only care
   * about processing those which have passed length; we still need
   * to keep track of the registers and the offset into the memory
   * argument area for those char arguments which are declared with
   * constant lengths.
   */
  dpdscp = parg;
  while (nlens--) {
    argsym = convertSPTR(*dpdscp);
    if (SCG(argsym) == SC_BASED && MIDNUMG(argsym) && XBIT(57, 0x80000) &&
        SCG(MIDNUMG(argsym)) == SC_DUMMY) {
      /* for char, we put pointee in argument list so as to get
       * the char length here, but we really pass the pointer
       * use the actual pointer */
      *dpdscp = MIDNUMG(argsym);
    }
    dpdscp++;
    argdtype = DTYPEG(argsym);
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "%s.len in mem area at %d\n", SYMNAME(argsym),
              pf->mem_off);
    if (
        (!HAS_OPT_ARGSG(func) && IS_PROC_DESCRG(argsym)) ||
        (
            !AUTOBJG(argsym) &&
            (argsym = CLENG(argsym)))) {
      if (COPYPRMSG(argsym))
        cp_memarg(argsym, pf->mem_off, expb.charlen_dtype);
      else {
        MEMARGP(argsym, 1);
        asym = mk_argasym(argsym);
        MEMARGP(asym, 1);
      }
    }
    pf->mem_off += 4;
  }
}

/*
 * WARNING: there are nomixedstrlen and mixedstrlen functions to preprocess
 * parameters.
 */
static void
pp_params_mixedstrlen(int func)
{
  SPTR argsym;
  int asym;
  DTYPE argdtype;
  int nargs;
  int *dpdscp;
  int nlens;
  int byvalue;
  finfo_t *pf;

  check_desc_args(func);

  if (EXPDBG(8, 256))
    fprintf(gbl.dbgfil, "---pp_params_mixedstrlen: %s ---\n", SYMNAME(func));
  pf = &pfinfo[MIDNUMG(func)]; /* pfinfo alloc'd and init'd by pp_entries */
  argdtype = DTYPEG(func);
  dpdscp = (int *)(aux.dpdsc_base + DPDSCG(func));
  nargs = PARAMCTG(func);
  nlens = 0;
  byvalue = 0;

  pf->mem_off = 8; /* offset for 1st dummy arg */
  if (gbl.rutype != RU_FUNC)
    goto scan_args;
  switch (DTY(argdtype)) {
  case TY_CHAR:
  case TY_NCHAR:
    /*
     * If this is a function which returns character, the first
     * two arguments are for the return length. The last entry in
     * the function's dpdsc auxiliary area is the "return" symbol.
     */
    argsym = convertSPTR(dpdscp[nargs - 1]);
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "func returns char, through %s\n", SYMNAME(argsym));
    MEMARGP(argsym, 1);
    ADDRESSP(argsym, 8);
    asym = mk_argasym(argsym);
    ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
    MEMARGP(asym, 1);
    argsym = CLENG(argsym);
    if (argsym) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "func return len in %s\n", SYMNAME(argsym));
      MEMARGP(argsym, 1);
      ADDRESSP(argsym, 12);
      asym = mk_argasym(argsym);
      ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
      MEMARGP(asym, 1);
    }
    pf->mem_off = 16; /* offset for 1st dummy arg */
    nargs--;
    break;
  case TY_CMPLX:
  case TY_DCMPLX:
    /*
     * If this is a function which returns complex, the first arg is
     * also for the return value.  The last entry in the function's
     * dpdsc auxiliary area is the "return" symbol.
     */
    argsym = convertSPTR(dpdscp[nargs - 1]);
    MEMARGP(argsym, 1);
    ADDRESSP(argsym, 8);
    asym = mk_argasym(argsym);
    ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
    MEMARGP(asym, 1);
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "func also returns complex, through %s\n",
              SYMNAME(argsym));
    pf->mem_off = 12; /* offset for 1st dummy arg */
    nargs--;
    break;
  default:
    break;
  }
scan_args:
  /*
   * scan through all of the arguments of the function to compute
   * how (register or memory area) and where (reg # or offset) the
   * arguments area passed.  Also, generate the the ili if the argument
   * must be copied.  If a register argument is not copied, it is recorded
   * in the entry's finfo table; if the arg has been copied, a register
   * is still "assigned" but it is not recorded (slot is zero).
   *
   * The only concern for now is arguments which are addresses; the
   * exception is the lengths of character args (actually only those
   * which are passed length). If compiler is enhanced to allow value
   * parameters, presumably there will be some way to distinguish these
   * from reference arguments (i.e., a symbol table flag).
   */
  while (nargs--) {
    int osym;
    argsym = convertSPTR(*dpdscp++);
    osym = argsym;
    if (((DTY(DTYPEG(argsym))) == TY_STRUCT) ||
        ((DTY(DTYPEG(argsym))) == TY_ARRAY) ||
        ((DTY(DTYPEG(argsym))) == TY_UNION))
      /* no passbyvalue arrays, structs */
      byvalue = 0;
    else
      byvalue = BYVALDEFAULT(func);

    if (PASSBYVALG(argsym))
      byvalue = 1;
    if (PASSBYREFG(argsym))
      byvalue = 0;
    if (SCG(argsym) == SC_BASED && MIDNUMG(argsym) && XBIT(57, 0x80000) &&
        SCG(MIDNUMG(argsym)) == SC_DUMMY) {
      /* char pointers, we put the pointee on the argument
       * list so as to get the char length, but we really pass
       * the pointer.
       * replace by the actual pointer */
      argsym = MIDNUMG(argsym);
    }
    argdtype = DTYPEG(osym);
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "%s in mem area at %d\n", SYMNAME(argsym),
              pf->mem_off);
    if (COPYPRMSG(argsym)) {
      cp_memarg(argsym, pf->mem_off, DT_ADDR);
    } else if (DTY(argdtype) == TY_STRUCT) {
      REFP(MIDNUMG(argsym), 1);
      cp_memarg(argsym, pf->mem_off, DT_ADDR);
    } else {
      MEMARGP(argsym, 1);
      ADDRESSP(argsym, pf->mem_off);
      asym = mk_argasym(argsym);
      ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
      MEMARGP(asym, 1);
    }
    if (byvalue) {
      if (argdtype == DT_DBLE || argdtype == DT_INT8 || argdtype == DT_LOG8 ||
          argdtype == DT_CMPLX)
        pf->mem_off += 8;
      else if (argdtype == DT_DCMPLX || argdtype == DT_QUAD)
        pf->mem_off += 16;
      else if (DTY(argdtype) == TY_STRUCT || argdtype == DT_QCMPLX)
        pf->mem_off += size_of(argdtype);
      else
        pf->mem_off += 4;
    } else {
      pf->mem_off += 4;
    }

    /*
     * character length.
     */
    if ((DTYG(argdtype) == TY_CHAR || DTYG(argdtype) == TY_NCHAR) &&
        needlen(argsym, func)) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "%s.len in mem area at %d\n", SYMNAME(argsym),
                pf->mem_off);
      if (
          !AUTOBJG(argsym) &&
          (argsym = CLENG(osym))) {
        if (COPYPRMSG(argsym))
          cp_memarg(argsym, pf->mem_off, expb.charlen_dtype);
        else {
          MEMARGP(argsym, 1);
          ADDRESSP(argsym, pf->mem_off);
          asym = mk_argasym(argsym);
          ADDRESSP(asym, ADDRESSG(argsym)); /* propagate ADDRESS */
          MEMARGP(asym, 1);
        }
      }
      pf->mem_off += 4;
    }
    if ((!HOMEDG(argsym) && (SCG(argsym) == SC_DUMMY)) &&
        (!PASSBYREFG(argsym)) &&
        (PASSBYVALG(argsym) ||
         (BYVALDEFAULT(func) && ((DTY(DTYPEG(argsym))) != TY_ARRAY)))) {
      cp_byval_mem_arg(argsym);
      PASSBYVALP(argsym, 1);
    }

  } /* end while */
}

#ifdef FLANG2_EXPRTE_UNUSED
static int
get_frame_off(INT off)
{
  int ili;

  /* Compute the address of the memory argument by relying on
   * a dummy symbol whose address is the first memory argument
   * immediately upon entry, i.e., after the return address has been pushed
   * on the stack by the call instruction  but before any manipulation
   * of %rbp by the cg.
   * The actual address computation will consist of an ACON whose
   * symbol is the dummy symbol and whose offset is relative to
   * the dummy symbol.
   */
  if (memarg_var == 0) {
    memarg_var = getccsym('Q', expb.gentmps++, ST_VAR);
    SCP(memarg_var, SC_DUMMY);
    DTYPEP(memarg_var, DT_CPTR);
    REDUCP(memarg_var, 1); /* mark sym --> no further indirection */
    HOMEDP(memarg_var, 0);
    ADDRTKNP(memarg_var, 1);
  }
  ili = ad_acon(memarg_var, off - MEMARG_OFFSET);
  return ili;
}
#endif

/* from exp_c.c */
static void
ldst_size(DTYPE dtype, ILI_OP *ldo, ILI_OP *sto, int *siz)
{
  *ldo = IL_LD;
  *sto = IL_ST;

  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_CHAR:
    *siz = MSZ_SBYTE;
    break;
  case TY_SINT:
  case TY_SLOG:
  case TY_NCHAR:
    *siz = MSZ_SHWORD;
    break;
  case TY_FLOAT:
  case TY_CMPLX:
    *siz = MSZ_F4;
    *ldo = IL_LDSP;
    *sto = IL_STSP;
    break;
  case TY_INT8:
    *siz = MSZ_I8;
    *ldo = IL_LDKR;
    *sto = IL_STKR;
    break;
  case TY_DBLE:
  case TY_DCMPLX:
    *siz = MSZ_F8;
    *ldo = IL_LDDP;
    *sto = IL_STDP;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
  case TY_QCMPLX:
    *siz = MSZ_F16;
    *ldo = IL_LDQP;
    *sto = IL_STQP;
    break;
#endif
  case TY_PTR:
    *siz = MSZ_WORD;
    *ldo = IL_LDA;
    *sto = IL_STA;
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
    }
    break;
  case TY_BLOG:
    *siz = MSZ_SBYTE;
    break;
  case TY_INT:
  default:
    *siz = MSZ_WORD;
  }
  switch (*siz) {
  case MSZ_FWORD:
    *ldo = IL_LDSP;
    *sto = IL_STSP;
    break;
  case MSZ_DFLWORD:
    *ldo = IL_LDDP;
    *sto = IL_STDP;
    break;
  }
} /* ldst_size */

/***************************************************************/
/*        F o r t r a n   S t r i n g   S u p p o r t          */
/***************************************************************/

/* for the character*1 load/store optimization, need a names entry
 * for use in the load/store ili which is sufficient for cg to
 * correctly schedule the loads/stores when loads/stores of overlaid
 * data (MAPs, see tpr 564) are present.  NME_UNK is insufficient
 * since cg does not always consider NME_UNK to conflict with all
 * others.  The macro NME_STR1 is used when the optimization occurs;
 * it's defined to be the actual nme which is used.  'Precise' nmes
 * aren't used since the optimization phases do not expect to see
 * Fortran character variables.
 */
#define NME_STR1 NME_VOL

/* copy an argument passed by value to it's identically named
   compiler created SC_LOCAL
   this is used only for args not passed in registers
*/
static void
cp_byval_mem_arg(SPTR argsptr)
{
  SPTR newsptr;
  ILI_OP ldo, sto;
  int ms_siz;
  int ilix;
  int val, val_nme;
  int addr, addr_nme;
  DTYPE dtype = DTYPEG(argsptr);

  ldst_size(dtype, &ldo, &sto, &ms_siz);
  newsptr = get_byval_local(argsptr);
  HOMEDP(argsptr, 1);
  MEMARGP(argsptr, 0);

  if (DTY(dtype) != TY_STRUCT) {
    if (dtype != DT_CMPLX && dtype != DT_DCMPLX) {
      val = ad_acon(argsptr, 0);
      val_nme = addnme(NT_VAR, argsptr, 0, 0);
      ilix = ad3ili(ldo, val, val_nme, ms_siz);
      addr = ad_acon(newsptr, 0);
      if (dtype == DT_CHAR || dtype == DT_NCHAR) {
        addr_nme = NME_STR1;
      } else {
        addr_nme = addnme(NT_VAR, newsptr, 0, 0);
      }
      ilix = ad4ili(sto, ilix, addr, addr_nme, ms_siz);
      chk_block(ilix);
    } else {
      int val_nme2, addr_nme2, sz;
      sz = size_of(dtype);
      /* copy the real part */
      val = ad_acon(argsptr, 0);
      val_nme = addnme(NT_VAR, argsptr, 0, 0);
      val_nme2 = addnme(NT_MEM, SPTR_NULL, val_nme, 0);
      ilix = ad3ili(ldo, val, val_nme2, ms_siz);
      addr = ad_acon(newsptr, 0);
      addr_nme = addnme(NT_VAR, newsptr, 0, 0);
      addr_nme2 = addnme(NT_MEM, SPTR_NULL, addr_nme, 0);
      ilix = ad4ili(sto, ilix, addr, addr_nme2, ms_siz);
      chk_block(ilix);
      val = ad_acon(argsptr, sz / 2);
      val_nme2 = addnme(NT_MEM, NOSYM, val_nme, sz / 2);
      ilix = ad3ili(ldo, val, val_nme2, ms_siz);
      addr = ad_acon(newsptr, sz / 2);
      addr_nme2 = addnme(NT_MEM, NOSYM, addr_nme, sz / 2);
      ilix = ad4ili(sto, ilix, addr, addr_nme2, ms_siz);
      chk_block(ilix);
    }
  }
  if (gbl.internal == 1) {
    sym_is_refd(argsptr);
    HOMEDP(argsptr, 0);
  }
}

/** \brief Copy an argument from the memory area to the local area; this
 * routine is only called from pp_params (the arg needs to be copied).
 */
static void
cp_memarg(int sym, INT off, int dtype)
{
  int asym;

  HOMEDP(sym, 1);
  MEMARGP(sym, 0);
  switch (dtype) {
  case DT_INT:
    /* TODO: store by value arg into memory */
    break;
  case DT_INT8:
    /* TODO: store by value arg into memory */
    break;
  case DT_ADDR:
    /* TODO: store by value arg into memory */
    asym = mk_argasym(sym);
    HOMEDP(asym, 1);
    MEMARGP(asym, 0);
    break;
  default:
    asym = 0;
    interr("unrec dtype in cp_memarg", dtype, ERR_Severe);
    break;
  }
  if (gbl.internal == 1 && asym != 0)
    arg_is_refd(asym);
  if (EXPDBG(8, 256))
    fprintf(gbl.dbgfil, "%s must be copied from MEM+%d\n", SYMNAME(sym), off);
}

/***************************************************************/

int
exp_alloca(ILM *ilmp)
{
  int op1, op2;

  alloca_flag = 1;
  op1 = ILI_OF(ILM_OPND(ilmp, 1)); /* nelems */
  op2 = ILI_OF(ILM_OPND(ilmp, 2)); /* nbytes */
  /** sptr = ILM_OPND(ilmp, 3);  sym and currently ignored **/
  /** tmp  = ILM_OPND(ilmp, 4);  stc and currently ignored **/
  /*
   * final size must be a multiple of 16:
   *     (nelems*nbytes + 15) & ~0xfL
   */
  op2 = ikmove(op2);
  op1 = ad2ili(IL_KMUL, op1, op2);
  if (!XBIT(54, 0x10)) {
    /**  runtime adjusts the size  **/
    (void)mk_prototype("__builtin_aa", "pure", DT_ADDR, 1, DT_INT8);
  } else {
    op1 = ad2ili(IL_KADD, op1, ad_kconi(15));
    op1 = ad2ili(IL_KAND, op1, ad_kcon(0xffffffff, 0xfffffff0));
  }
  op2 = ad1ili(IL_NULL, 0);
  op2 = ad2ili(IL_ARGKR, op1, op2);
  if (!XBIT(54, 0x10))
    op1 = ad2ili(IL_JSR, mkfunc("__builtin_aa"), op2);
  else
    op1 = ad2ili(IL_JSR, mkfunc("__builtin_alloca"), op2);
  return ad2ili(IL_DFRAR, op1, AR_RETVAL);
}

/***************************************************************/

static void gen_funcret(finfo_t *);

void
exp_end(ILM *ilmp, int curilm, bool is_func)
{
  int tmp;
  int func;
  int sym;
  finfo_t *pf;
  int exit_bih;

  if (expb.retlbl != 0) {
    exp_label(expb.retlbl);
    expb.retlbl = SPTR_NULL;
  }
  if (allocharhdr) {
    /* if character temps were allocated, need to free the
     * list of allocated areas created by the run-time.
     */
    int ld;

    /*  ftn_str_free(allocharhdr) */
    ld = ad_acon(allocharhdr, 0);
    ld = ad2ili(IL_LDA, ld, addnme(NT_VAR, allocharhdr, 0, 0));
    sym = frte_func(mkfunc, mkRteRtnNm(RTE_str_free));
    tmp = ad1ili(IL_NULL, 0);
    tmp = ad3ili(IL_ARGAR, ld, tmp, 0);
    tmp = ad2ili(IL_JSR, sym, tmp);
    iltb.callfg = 1;
    chk_block(tmp);
  }

  exp_restore_mxcsr();

  if (is_func) {
    SPTR exit_lab;
    SPTR next_lab;
    int load_retgrp;
    int currgrp;

    if (retgrp_cnt > 1) {
      load_retgrp = ad3ili(IL_LD, ad_acon(retgrp_var, 0),
                           addnme(NT_VAR, retgrp_var, 0, 0), MSZ_WORD);
      exit_lab = getlab();
    } else {
      exit_lab = SPTR_NULL;
    }
    /*
     * generate test, move, branch for all but the first return
     * group.
     */
    for (currgrp = 1; currgrp < retgrp_cnt; currgrp++) {
      /*  generate code sequence for a group as follows:
       *      if (load_retgrp != currgrp) got to next_lab;
       *      result <---  load  currgrp's fval;
       *      goto exit_lab;
       *  next_lab:
       */
      next_lab = getlab();
      RFCNTI(next_lab);
      tmp = ad4ili(IL_ICJMP, load_retgrp, ad_icon(currgrp), 2, next_lab);
      chk_block(tmp);
      gen_funcret(&pfinfo[currgrp]);
      RFCNTI(exit_lab);
      tmp = ad1ili(IL_JMP, exit_lab);
      chk_block(tmp);
      exp_label(next_lab);
    }
    /*  generate move for last block  */
    gen_funcret(&pfinfo[0]);
    if (exit_lab)
      exp_label(exit_lab);
  }
  if (gbl.arets) {
    int addr;
    int nme;
    int move;

    addr = ad_acon(expb.aret_tmp, 0);
    nme = addnme(NT_VAR, expb.aret_tmp, 0, 0);
    tmp = ad3ili(IL_LD, addr, nme, MSZ_WORD);
    move = ad2ili(IL_MVIR, tmp, IR_ARET);
    chk_block(move);
  }
  if (flg.opt >= 1 && expb.curilt != 0) {
    flsh_block(); /* at the higher opt levels, the exit	 */
    cr_block();   /* block is a stand-alone block	 */
  }
  /* xon/xoff stuff goes here */

  /* exit debug stuff goes here */

  tmp = ad1ili(IL_EXIT, gbl.currsub);
  expb.curilt = addilt(expb.curilt, tmp);
  BIH_XT(expb.curbih) = 1;
  BIH_LAST(expb.curbih) = 1;
  exit_bih = expb.curbih;
  wr_block();
  BIH_EX(gbl.entbih) = expb.flags.bits.callfg;
  BIH_SMOVE(gbl.entbih) = smove_flag;
  aux.curr_entry->flags = 0;
  if (mscall_flag)
    aux.curr_entry->flags |= 0x40000000;
  if (alloca_flag)
    aux.curr_entry->flags |= 0x80000000;
  /*
   * scan through all the entries to store return group value if necessary.
   */
  if (gbl.rutype == RU_PROG)
    goto exp_end_ret;
  for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
    if (EXPDBG(8, 256))
      fprintf(gbl.dbgfil, "---exp_end: %s ---\n", SYMNAME(func));
    expb.curbih = BIHNUMG(func);
    BIH_EX(expb.curbih) = expb.flags.bits.callfg; /* ALL entry bihs */
    BIH_SMOVE(expb.curbih) = smove_flag;
    if (retgrp_cnt > 1) {
      pf = &pfinfo[MIDNUMG(func)];
      rdilts(expb.curbih); /* get entry block */
      expb.curilt = ILT_PREV(0);
      tmp = ad_icon(pf->retgrp);
      tmp = ad4ili(IL_ST, tmp, ad_acon(retgrp_var, 0),
                   addnme(NT_VAR, retgrp_var, 0, 0), MSZ_WORD);
      chk_block(tmp);
      wrilts(expb.curbih);
    }
  }
  /*
   * For multiple entries using the WINNT calling convention, must store
   * the number of bytes passed to each entry in a temporary. This store
   * must appear in the prologue of each entry -- the code generator will
   * load the temporary and use its value to pop the arguments from the
   * stack.  A sufficient test for generating the store is if the temporary
   * was created (saved in aux.curr_entry->ent_save),
   */
  if (aux.curr_entry->ent_save) {
    int addr, nme;
    addr = ad_acon(aux.curr_entry->ent_save, 0);
    nme = addnme(NT_VAR, aux.curr_entry->ent_save, 0, 0);
    for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
      expb.curbih = BIHNUMG(func);
      rdilts(expb.curbih); /* get entry block */
      expb.curilt = ILT_PREV(0);
      if (ARGSIZEG(func) < 0)
        tmp = ad_icon(0);
      else
        tmp = ad_icon(ARGSIZEG(func));
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "---storing %d in %s ---\n",
                CONVAL2G(ILI_OPND(tmp, 1)), SYMNAME(aux.curr_entry->ent_save));
      tmp = ad4ili(IL_ST, tmp, addr, nme, MSZ_WORD);
      chk_block(tmp);
      wrilts(expb.curbih);
    }
  }

  freearea(1); /* duumy arg processing (alloc'd in pp_entries) */

exp_end_ret:
  if (allocharhdr) {
    /* if character temps were allocated, need to initialize the
     * head of a list of allocated areas created by the run-time.
     */
    int st;

    tmp = ad_acon(SPTR_NULL, 0);
    st = ad_acon(allocharhdr, 0);
    st = ad3ili(IL_STA, tmp, st, addnme(NT_VAR, allocharhdr, 0, 0));
    for (func = gbl.entries; func != NOSYM; func = SYMLKG(func)) {
      if (EXPDBG(8, 256))
        fprintf(gbl.dbgfil, "---init allocharhdr: %s in %s---\n",
                SYMNAME(allocharhdr), SYMNAME(func));
      expb.curbih = BIH_NEXT(BIHNUMG(func));
      rdilts(expb.curbih); /* get block after entry block */
      expb.curilt = 0;
      /*  allocharhdr = NULL; */
      chk_block(st);
      wrilts(expb.curbih);
    }
  }

  /* emit any mp initialization for the function & its entries */
  exp_mp_func_prologue(true);

  if (!XBIT(121, 0x01) ||                  /* -Mnoframe isn't specified */
      (flg.debug && !XBIT(123, 0x400)) ||  /* -debug is set */
      (flg.profile && XBIT(129, 0x800)) || /* -Minstrument */
      XBIT(34, 0x200) ||                   /* -Mconcur */
      flg.smp ||                           /* -mp */
      alloca_flag ||                       /* alloca present */
      (gbl.internal        /* contains an internal subprogram or is an
                            * internal subprogram. */
       && !gbl.cudaemu) || /* Don't use a frame pointer when emulating
                            * CUDA device code. */
      gbl.vfrets || /* contains variable format expressions */
      /* linux main now aligns the stack - so can't allow -Mnoframe */
      (XBIT(119, 0x8000000) && gbl.rutype == RU_PROG) ||
      /* -Msmartalloc=huge[:n] */
      (XBIT(129, 0x10000000) && gbl.rutype == RU_PROG) ||
      aux.curr_entry->ent_save > 0 /* is this a fortran routine with
                                    * multiple entries and mscall */
  )
    aux.curr_entry->flags |= 0x100; /* bit set ==> must use frame pointer */

  /* we can't afford a third global register unless -Mnoframe is allowed */
  if (aux.curr_entry->flags & 0x100)
    mr_reset_numglobals(1); /* must use frame - reduce nglobals by 1 */
  else
    mr_reset_numglobals(0); /* -Mnoframe ok */

  /* only perform floating-point caching at -O2 or higher */
  if (flg.opt < 2 || XBIT(8, 0x400) || XBIT(8, 0x1000) || flg.ieee ||
      XBIT(6, 0x100) || XBIT(6, 0x200))
    mr_reset_frglobals();

  if (DOREG1) { /* assign registers for opt level 1  */
    expb.curbih = exit_bih;
    reg_assign1();
  }
  /*
   * for opt levels 0 and 1, check if this function is a terminal
   * routine.
   */
  if (flg.opt <= 1)
    chk_terminal_func(gbl.entbih, expb.curbih);

  /* chk_savears(expb.curbih) needed? */

  /* final stuff to cleanup at the end of a function  */
  expb.arglist = 0;
  expb.flags.bits.callfg = 0;
  mkrtemp_end();
}

static void
gen_bindC_retval(finfo_t *fp)
{
  const SPTR fval = fp->fval;
  const int retv = ad_acon(fval, 0);
  const int nme = addnme(NT_VAR, fval, 0, 0);
  int ilix = retv;

  if (fp->ret_sm_struct) {
    ilix = ad2ili(IL_MVAR, retv, RES_IR(0));
    ADDRTKNP(fval, 1);
  } else {
    switch (IL_RES(ILI_OPC(ilix))) {
    case ILIA_AR:
      ilix = ad2ili(IL_LDA, ilix, nme);
      ilix = ad2ili(IL_MVAR, ilix, RES_IR(0));
      break;
    case ILIA_IR:
      ilix = ad2ili(IL_MVIR, ilix, RES_IR(0));
      break;
    case ILIA_SP:
      if (ILI_OPC(ilix) != IL_LDSP && ILI_OPC(ilix) != IL_FCON) {
        const SPTR sfval = fp->fval;
        ilix = ad4ili(IL_STSP, ilix, ad_acon(sfval, 0),
                      addnme(NT_VAR, sfval, 0, 0), MSZ_F4);
        chk_block(ilix);
        ilix = ad3ili(IL_LDSP, ad_acon(sfval, 0),
                      addnme(NT_VAR, sfval, 0, 0), MSZ_F4);
      }
      ilix = ad2ili(IL_MVSP, ilix, RES_XR(0));
      break;
    case ILIA_DP:
      if (ILI_OPC(ilix) != IL_LDDP && ILI_OPC(ilix) != IL_DCON) {
        const SPTR sfval = fp->fval;
        ilix = ad4ili(IL_STDP, ilix, ad_acon(sfval, 0),
                      addnme(NT_VAR, sfval, 0, 0), MSZ_F8);
        chk_block(ilix);
        ilix = ad3ili(IL_LDDP, ad_acon(sfval, 0),
                      addnme(NT_VAR, sfval, 0, 0), MSZ_F8);
      }
      if (ILI_OPC(ilix) == IL_LD256) {
        ilix = ad2ili(IL_MV256, ilix, RES_XR(0)); /*m256*/
      } else if (ILI_OPC(ilix) != IL_LDQ) {
        ilix = ad2ili(IL_MVDP, ilix, RES_XR(0));
      } else {
        ilix = ad2ili(IL_MVQ, ilix, RES_XR(0)); /*m128*/
      }
      break;
    case ILIA_KR:
      ilix = ad2ili(IL_MVKR, ilix, RES_IR(0));
      break;
    default:
      interr("expand:illegal return expr", retv, ERR_Severe);
      break;
    }
  }
  if (EXPDBG(8, 256))
    fprintf(gbl.dbgfil, "gen_retval %d @ %d\n", ilix, gbl.lineno);
  /*
   * check what is in the current block to see if the block has to be
   * written out
   */
  chk_block(ilix);
}

static void
gen_funcret(finfo_t *fp)
{
  int addr;
  int nme;
  int ili1;
  int move;
  SPTR fval = fp->fval;
  int fvaltyp = DTY(DTYPEG(fval));

  if (CFUNCG(gbl.currsub) || (CMPLXFUNC_C && TY_ISCMPLX(fvaltyp))) {
    gen_bindC_retval(fp);
    return;
  }
  addr = ad_acon(fval, 0);
  nme = addnme(NT_VAR, fval, 0, 0);
  /*
   *  if it's possible that fvar has storage SC_DUMMY AND we need
   *  to generate a load, then we need a LDA:
   *     if (SCG(fval) == SC_DUMMY)
   *         addr = ad2ili(IL_LDA, addr, nme);
   */
  switch (fvaltyp) {
  case TY_CHAR:
  case TY_NCHAR:
    return;
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
    if (!CFUNCG(gbl.currsub) && !CMPLXFUNC_C)
      return;
    move = ad2ili(IL_MVAR, addr, RES_IR(0));
    ADDRTKNP(fval, 1);
    if (XBIT(121, 0x400)) {
      int gret;
      gret = ad3ili(IL_RETURN, addr, DTYPEG(fval), nme);
      ILI_ALT(move) = gret;
    }
    break;
  case TY_REAL:
    ili1 = ad3ili(IL_LDSP, addr, nme, MSZ_F4);
    move = ad2ili(IL_MVSP, ili1, FR_RETVAL);
    break;
  case TY_DBLE:
    ili1 = ad3ili(IL_LDDP, addr, nme, MSZ_F8);
    move = ad2ili(IL_MVDP, ili1, FR_RETVAL);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ili1 = ad3ili(IL_LDQP, addr, nme, MSZ_F16);
    move = ad2ili(IL_MVQ, ili1, FR_RETVAL);
    break;
#endif
  case TY_BINT:
  case TY_BLOG:
    ili1 = ad3ili(IL_LD, addr, nme, MSZ_SBYTE);
    move = ad2ili(IL_MVIR, ili1, IR_RETVAL);
    break;
  case TY_SINT:
  case TY_SLOG:
    ili1 = ad3ili(IL_LD, addr, nme, MSZ_SHWORD);
    move = ad2ili(IL_MVIR, ili1, IR_RETVAL);
    break;
  case TY_INT:
  case TY_LOG:
    ili1 = ad3ili(IL_LD, addr, nme, MSZ_WORD);
    move = ad2ili(IL_MVIR, ili1, IR_RETVAL);
    break;
  case TY_INT8:
  case TY_LOG8:
    ili1 = ad3ili(IL_LDKR, addr, nme, MSZ_I8);
    move = ad2ili(IL_MVKR, ili1, KR_RETVAL);
    break;
  default:
    interr("gen_funcret: illegal dtype, sym", fval, ERR_Severe);
    return;
  }

  chk_block(move);
}

/***************************************************************/

static SWEL *sw_array; /**< linear form of the switch list, incl default */
static int sw_temp;    /**< acon ili of temp holding value of switch val */
static int sw_val;     /**< ili of the original switch value; becomes a load
                            of a temp if it's necessary to temp store value */
static void genswitch(INT, INT);

/**
   \brief expand a computed go to

   this processing is similiar to the processing of a switch by pgc.  The
   exception is that the switch list is already ordered as a table in increasing
   order.  pgc must first create a table of the switch values.
 */
void
exp_cgoto(ILM *ilmp, int curilm)
{
  INT i;
  INT n; /* # of cases */
  INT cval;

  sw_val = ILI_OF(ILM_OPND(ilmp, 1));
  sw_temp = 0;
  i = ILM_OPND(ilmp, 2); /* index from switch_base locating default */
  sw_array = switch_base + i;
  n = sw_array[0].val;
#if DEBUG
  if (flg.dbg[10] != 0) {
    fprintf(gbl.dbgfil,
            "\n\n Switch: %-5d  line: %-5d  n: %-5d  default: %-5d\n", curilm,
            gbl.lineno, n, sw_array[0].clabel);
    for (i = 1; i <= n; i++) {
      fprintf(gbl.dbgfil, " %10d    %5d~\n", sw_array[i].val,
              sw_array[i].clabel);
    }
  }
#endif
  assert(n != 0, "exp_cgoto: cnt is zero, at ilm", curilm, ERR_Severe);
  if (ILI_OPC(sw_val) == IL_ICON) {
    /*
     * switch value is a constant -- search switch list for the equal
     * value and generate a jump to that label.  If not found, the jump
     * to the default will take place
     */
    cval = CONVAL2G(ILI_OPND(sw_val, 1));
    i = 1; /* first in switch list	 */
    do {
      if (cval == sw_array[i].val)
        chk_block(ad1ili(IL_JMP, sw_array[i].clabel));
      else
        RFCNTD(sw_array[i].clabel);
    } while (++i <= n);
    chk_block(ad1ili(IL_JMP, sw_array[0].clabel));
    return;
  }
  genswitch(1, n);
}

/**
   \param lb  lower bound of switch array
   \param ub  upper bound of switch array
 */
static void
genswitch(INT lb, INT ub)
{
  UINT ncases;
  UINT range;
  int i;

  ncases = ub - lb + 1;
  range = sw_array[ub].val - sw_array[lb].val + 1;
#if DEBUG
  if (flg.dbg[10])
    fprintf(gbl.dbgfil, "genswitch: lb: %d, ub: %d\n", lb, ub);
#endif
  if (ncases >= 6 && range <= (3 * ncases)) {
    /*
     * Use a memory table of addresses for the switch. The JMPM
     * ili is created which fetches to branch address from a table
     * in memory based on the value of the switch expression.
     * This value is normalized to 0 (first entry contains the first
     * case label)
     */
    int ilix;
    SWEL *swhdr;
    /*
     * First, locate beginning of the switch list for this range in
     * the original area.  Also, terminate the last element in the list.
     */
    swhdr = &sw_array[lb];
    sw_array[ub].next = 0;
    ilix = ad_icon(range);
    /*
     * for TARGET_LLVM, pairs of case values and labels are present to
     * the llvm switch instruction, we should not be normalizing the
     * switch expression to zero.
     */
    ilix = ad4ili(IL_JMPM, sw_val, ilix,
                  mk_swtab(range, swhdr, sw_array[0].clabel, 1),
                  sw_array[0].clabel);
    chk_block(ilix);
    if (ILT_ILIP(expb.curilt) != ilix) {
      /*
       * An ILT was not created for the JMPM -- the previous ILT is an
       * unconditional branch.  go through and decrement all of the
       * use counts for the switch labels
       */
      RFCNTD(sw_array[0].clabel);
      for (i = lb; i <= ub; i++)
        RFCNTD(sw_array[i].clabel);
      wr_block(); /* end this ilt block */
    }
  } else if (ncases > 4) {
    int m, first;
    SPTR label;
    /*
     * perform a binary search of the switch array:
     * generate ili of the form
     *
     *   if (sw_val > sw_array[m].val) goto label;
     *       switch for table[lb .. m]
     * label:
     *       switch for table[m+1 .. ub]
     *
     * Note that a new block must be created for the switch on the
     * upper half of the table; the switch value must be temp stored
     * in the current block.
     */
    RFCNTI(sw_array[0].clabel); /* default label has another use */
    m = (lb + ub) / 2;
    if (sw_temp == 0) {
      int nme;
      /*
       * need to temp store the switch value in this block, and the
       * first use will be a cse of the original value
       */
      const SPTR sym = mkrtemp_sc(sw_val, expb.sc);
      sw_temp = ad_acon(sym, 0);
      nme = addnme(NT_VAR, sym, 0, 0);
      chk_block(ad4ili(IL_ST, sw_val, sw_temp, nme, MSZ_WORD));
      first = ad1ili(IL_CSEIR, sw_val);
      sw_val = ad3ili(IL_LD, sw_temp, nme, MSZ_WORD);
    } else /* use the load of the temporary containing the switch value */
      first = sw_val;
    label = getlab();
    RFCNTI(label);
    chk_block(ad4ili(IL_ICJMP, first, ad_icon(sw_array[m].val), 6, label));
    genswitch(lb, m);
    exp_label(label);
    genswitch(m + 1, ub);
  } else {
    int first, next, i;
    /*
     * generate a sequence of "if (sw_val == case value) goto case label"
     * followed by a JMP to the default label.
     */
    if (sw_temp) {
      /*
       * since the switch value has been temp stored, use the load
       * of the temp for all cases.
       */
      first = next = sw_val;
    } else if (ncases > 1 && flg.opt != 1) {
      /*
       * for this situation, the switch will generate multiple blocks.
       * Therefore, in the block evaluating sw_val, a temp store of
       * sw_val must occur and in ensuing blocks, the switch expression
       * will be fetched from the temporary.
       */
      int nme;
      const SPTR sym = mkrtemp_sc(sw_val, expb.sc);
      sw_temp = ad_acon(sym, 0);
      nme = addnme(NT_VAR, sym, 0, 0);
      chk_block(ad4ili(IL_ST, sw_val, sw_temp, nme, MSZ_WORD));
      /*
       * The first case occurs in the same block as the store, so just
       * use a cse of the original switch value for the first case.
       */
      first = ad1ili(IL_CSEIR, sw_val);
      next = sw_val = ad3ili(IL_LD, sw_temp, nme, MSZ_WORD);
    } else {
      /*
       * Since all of the conditional branches will fit in the current
       * block, the first branch uses sw_val and subsequent branches
       * will use a cse of sw_val.
       */
      first = sw_val;
      next = ad1ili(IL_CSEIR, sw_val);
    }

    /* generate first compare */

    chk_block(ad4ili(IL_ICJMP, first, ad_icon(sw_array[lb].val), 1,
                     sw_array[lb].clabel));

    /* generate compares for the remaining cases */

    for (i = lb + 1; i <= ub; i++) {
      chk_block(ad4ili(IL_ICJMP, next, ad_icon(sw_array[i].val), 1,
                       sw_array[i].clabel));
    }

    /* generate the default jump */

    chk_block(ad1ili(IL_JMP, sw_array[0].clabel));
  }
}

static int agotostart;

void
exp_build_agoto(int *tab, int mx)
{
  int i;
  SWEL *swelp;

  if (mx <= 0)
    return;
  /*
   * AGOTOs will be treated like CGOTOs so an extra entry in the
   * switch table is needed for te default label.
   */
  agotostart = getswel(mx + 1);
  /*
   * switch_base[agotostart].clabel is reserved for the default
   */
  switch_base[agotostart].val = mx;
  switch_base[agotostart].next = agotostart + 1;
  swelp = 0; /* quite possible use before def */
  for (i = 1; i <= mx; i++) {
    swelp = switch_base + (agotostart + i);
    swelp->clabel = convertSPTR(tab[i - 1]);
    RFCNTI(swelp->clabel);
    swelp->val = i;
    swelp->next = (agotostart + i + 1);
  }
  swelp->next = 0;
}

/** \brief Expand a goto
 *
 * for TARGET_LLVM, we are not performing an indirect branch, so expand
 * an assigned goto into a computed goto -- the labels appearing in the
 * ASSIGN statements and their respective computed goto index values have
 * already been collected into a switch_base table whose starting index
 * is agotostart.
 */
void
exp_agoto(ILM *ilmp, int curilm)
{
  INT i;
  INT n; /* # of cases */

  sw_val = kimove(ILI_OF(ILM_OPND(ilmp, 2)));
  sw_temp = 0;
  i = agotostart; /* index from switch_base locating default */
  sw_array = switch_base + i;
  n = sw_array[0].val;
  sw_array[0].clabel = getlab();
  RFCNTI(sw_array[0].clabel);
#if DEBUG
  if (flg.dbg[10] != 0) {
    fprintf(gbl.dbgfil,
            "\n\n Switch: %-5d  line: %-5d  n: %-5d  default: %-5d\n", curilm,
            gbl.lineno, n, sw_array[0].clabel);
    for (i = 1; i <= n; i++) {
      fprintf(gbl.dbgfil, " %10d    %5d~\n", sw_array[i].val,
              sw_array[i].clabel);
    }
  }
#endif
  assert(n != 0, "exp_agoto: cnt is zero, at ilm", curilm, ERR_Severe);
  genswitch(1, n);
  exp_label(sw_array[0].clabel);
}

/***************************************************************/

/* structure to hold argument list from which argili chain is
 * later built.
 */
typedef struct {
  int ili_type;
  int ili_arg;
  int dtype; // currently use only for byvalue struct args
} arg_info;

typedef struct {
  int ilix;
  int dtype;
  int val_flag; /* 0 or 1, aka NME_VOL */
  int nme;
} garg_info;

static arg_info *arg_ili; /* pointers to argument chain info */
static int arg_entry;     /* # of argument entries in call chain */
static int charargs;      /* # of character arguments */
static int *len_ili;      /* pointers to character length ili */
static garg_info *garg_ili;

/*
 * structure to provide communication between exp_call and the
 * routines to generate ili for arguments.
 */
typedef struct {
  int mem_area; /* sym of memory arg area */
  int mem_nme;  /* nme of memory arg area */
  INT mem_off;  /* size and next available offset */
  int lnk;      /* list of define reg ili of args in regs */
  char ireg;    /* next integer reg to use for args */
  char freg;    /* next fp register to use for args */
} ainfo_t;

static void from_addr_and_length(STRDESC *s, ainfo_t *ainfo_ptr);
static void arg_ir(int, ainfo_t *);
static void arg_kr(int, ainfo_t *);
static void arg_ar(int, ainfo_t *, int);
static void arg_sp(int, ainfo_t *);
static void arg_dp(int, ainfo_t *);
#ifdef TARGET_SUPPORTS_QUADFP
static void arg_qp(int, ainfo_t *);
#endif
static void arg_charlen(int, ainfo_t *);
static void arg_length(STRDESC *, ainfo_t *);

static void
init_ainfo(ainfo_t *ap)
{
  ap->lnk = ad1ili(IL_NULL, 0);
}

static void
end_ainfo(ainfo_t *ap)
{
  /* NOTHING TO DO */
}

void
init_arg_ili(int n)
{
  /* allocate enough space to accomodate the arguments, character lengths
   * if they're passed immediately after their arguments, and any function
   * return arguments.
   */
  NEW(arg_ili, arg_info, 2 * n + 3);
  charargs = 0;
  BZERO(arg_ili, arg_info, 2 * n + 3);
  NEW(len_ili, int, n + 1);
  arg_entry = 0;
  BZERO(len_ili, int, n + 1);
  if (XBIT(121, 0x800)) {
    /***** %val(complex) => 2 GARG arguments of component type *****/
    NEW(garg_ili, garg_info, 2 * n + 1);
    BZERO(garg_ili, garg_info, 2 * n + 1);
  }
}

void
end_arg_ili(void)
{
  FREE(arg_ili);
  FREE(len_ili);
  if (XBIT(121, 0x800)) {
    FREE(garg_ili);
  }
}

static void
add_to_args(int type, int argili)
{
  arg_ili[arg_entry].ili_type = type;
  arg_ili[arg_entry].ili_arg = argili;
  ++arg_entry;
}

static void
add_struct_byval_to_args(int type, int argili, int dtype)
{
  arg_ili[arg_entry].dtype = dtype;
  add_to_args(type, argili);
}

/* for 'by-value' arguments */
void
add_arg_ili(int ilix, int nme, int dtype)
{
  switch (IL_RES(ILI_OPC(ilix))) {
  case ILIA_IR:
    add_to_args(IL_ARGIR, ilix);
    break;
  case ILIA_KR:
    add_to_args(IL_ARGKR, ilix);
    break;
  case ILIA_SP:
    add_to_args(IL_ARGSP, ilix);
    break;
  case ILIA_DP:
    add_to_args(IL_ARGDP, ilix);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case ILIA_QP:
    add_to_args(IL_ARGQP, ilix);
    break;
#endif
  case ILIA_AR:
    add_to_args(IL_ARGAR, ilix);
    break;
  case ILIA_CS:
    add_to_args(IL_ARGSP, ilix);
    break;
  case ILIA_CD:
    add_to_args(IL_ARGDP, ilix);
    break;

  default:
    interr("exp_call:bad ili for BYVAL", ilix, ERR_Severe);
  }
} /* add_arg_ili */

static void
put_arg_ili(int i, ainfo_t *ainfo)
{

  switch (arg_ili[i].ili_type) {
  case IL_ARGIR:
    arg_ir(arg_ili[i].ili_arg, ainfo);
    break;
  case IL_ARGKR:
    arg_kr(arg_ili[i].ili_arg, ainfo);
    break;
  case IL_ARGAR:
    arg_ar(arg_ili[i].ili_arg, ainfo, arg_ili[i].dtype);
    break;
  case IL_ARGSP:
    arg_sp(arg_ili[i].ili_arg, ainfo);
    break;
  case IL_ARGDP:
    arg_dp(arg_ili[i].ili_arg, ainfo);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_ARGQP:
    arg_qp(arg_ili[i].ili_arg, ainfo);
    break;
#endif
  default:
    interr("exp_call: ili arg type not cased", arg_ili[i].ili_arg, ERR_Severe);
    break;
  }
}

static void
process_desc_args(ainfo_t *ainfo)
{
  int i;
  for (i = arg_entry - 1; i >= 0; --i) {
    int ili = arg_ili[i].ili_arg;
    if (is_proc_desc_arg(ili)) {
      put_arg_ili(i, ainfo);
    }
  }
}

int
gen_arg_ili(void)
{
  ainfo_t ainfo;
  int i;

  init_ainfo(&ainfo);

  if (charargs > 0 && !HAS_OPT_ARGSG(exp_call_sym))
    process_desc_args(&ainfo);

  /*  go through the list of character length ili which have been
   *  saved up and add them as arguments to the call.
   */
  for (i = charargs - 1; i >= 0; --i) {
    arg_charlen(len_ili[i], &ainfo);
  }

  /*  now go through the list of all stored arguments and add them
   *  to the argument chain for this call
   */
  for (i = arg_entry - 1; i >= 0; --i) {
    int ili = arg_ili[i].ili_arg;
    if (charargs > 0 && !HAS_OPT_ARGSG(exp_call_sym) && is_proc_desc_arg(ili))
      continue;
    put_arg_ili(i, &ainfo);
  }

  end_ainfo(&ainfo);
  return ainfo.lnk;
} /* gen_arg_ili */

static void
pass_char_arg(int type, int argili, int lenili)
{
  int len_opc;

  len_opc = IL_ARGKR;
  add_to_args(type, argili);

  if (!XBIT(125, 0x40000)) {
    if (IL_RES(ILI_OPC(lenili)) != ILIA_KR) {
      lenili = ad1ili(IL_IKMV, lenili);
    }
  } else
    len_opc = IL_ARGIR;

  if ((MSCALLG(exp_call_sym) || CREFG(exp_call_sym)) &&
      !NOMIXEDSTRLENG(exp_call_sym))
    add_to_args(len_opc, lenili);
  else
    len_ili[charargs++] = lenili;
}

#define IILM_OPC(i) ilmb.ilm_base[i]
#define IILM_OPND(i, j) ilmb.ilm_base[i + j]
#define FUNCPTR_BINDC 0x1
#ifdef __cplusplus
inline DTYPE IILM_DTyOPND(int i, int j) {
  return static_cast<DTYPE>(IILM_OPND(i, j));
}
#else
#define IILM_DTyOPND IILM_OPND
#endif

/* Returns the sptr for the tmp representing the SFUNC's return */
static int
struct_ret_tmp(int ilmx)
{
  ILM *ilmpx;
  int ilmxt;

  ilmpx = (ILM *)(ilmb.ilm_base + ilmx);

  assert(ILM_OPC(ilmpx) == IM_LOC || ILM_OPC(ilmpx) == IM_FARG ||
             ILM_OPC(ilmpx) == IM_FARGF,
         "struct_ret_tmp bad SFUNC", ilmx, ERR_Severe);
  ilmxt = ILM_OPND(ilmpx, 1);
  ilmpx = (ILM *)(ilmb.ilm_base + ilmxt);
  assert(ILM_OPC(ilmpx) == IM_BASE, "struct_ret_tmp bad SFUNC not base", ilmx,
         ERR_Severe);
  return ILM_OPND(ilmpx, 1); /* get sptr of temp */
}

static int
check_cstruct_return(DTYPE retdtype)
{
  int size;
  if (DTY(retdtype) == TY_STRUCT) {
    size = size_of(retdtype);
    if (size <= MAX_PASS_STRUCT_SIZE)
      return 1;
    return 0;
  }
  return 1;
}

static void
cmplx_to_mem(int real, int imag, DTYPE dtype, int *addr, int *nme)
{
  int load;
  ILI_OP store;
  int size, msz;
  int r_op1, i_op1, i_op2;
  SPTR tmp;

  assert(DT_ISCMPLX(dtype), "cmplx_to_mem: not complex dtype", dtype,
         ERR_Severe);
  if (DTY(dtype) == TY_CMPLX) {
    if (XBIT(70, 0x40000000) && !imag) {
      load = IL_LDSCMPLX;
      store = IL_STSCMPLX;
      msz = MSZ_F8;
    } else {
      load = IL_LDSP;
      store = IL_STSP;
      msz = MSZ_F4;
    }
  } else {
    if (XBIT(70, 0x40000000) && !imag) {
      load = IL_LDDCMPLX;
      store = IL_STDCMPLX;
      msz = MSZ_F16;
    } else {
      load = IL_LDDP;
      store = IL_STDP;
      msz = MSZ_F8;
    }
  }
  if (!XBIT(70, 0x40000000)) {
    size = size_of(dtype) / 2;
  } else {
    if (!imag)
      size = size_of(dtype);
    else
      size = size_of(dtype) / 2;
    if (ILI_OPC(real) == load) {
      r_op1 = ILI_OPND(real, 1);
      if (ILI_OPC(r_op1) == IL_ACON) {
        *addr = ILI_OPND(real, 1);
        *nme = ILI_OPND(real, 2);
        return;
      }
    }
  }

  if (ILI_OPC(real) == load && ILI_OPC(imag) == load) {
    /* Direct load? */
    r_op1 = ILI_OPND(real, 1);
    i_op1 = ILI_OPND(imag, 1);
    if (ILI_OPC(r_op1) == IL_ACON && ILI_OPC(i_op1) == IL_ACON) {
      r_op1 = ILI_OPND(r_op1, 1);
      i_op1 = ILI_OPND(i_op1, 1);
      if (CONVAL1G(r_op1) == CONVAL1G(i_op1) &&
          ACONOFFG(r_op1) + size == ACONOFFG(i_op1)) {
        *addr = ILI_OPND(real, 1);
        *nme = NME_NM(ILI_OPND(real, 2));
        return;
      }
    }

    /* Indirect load? */
    r_op1 = ILI_OPND(real, 1);
    i_op1 = ILI_OPND(imag, 1);
    if (ILI_OPC(i_op1) == IL_AADD) {
      i_op2 = ILI_OPND(i_op1, 2);
      i_op1 = ILI_OPND(i_op1, 1);
      if (i_op1 == r_op1 && ILI_OPC(i_op2) == IL_ACON &&
          CONVAL1G(ILI_OPND(i_op2, 1)) == 0 &&
          ACONOFFG(ILI_OPND(i_op2, 1)) == size) {
        *addr = r_op1;
        *nme = NME_NM(ILI_OPND(real, 2));
        return;
      }
      /*
       * TBD - can do better to detect subscripted references:
       */
    }
  }
  tmp = mkrtemp_cpx_sc(dtype, expb.sc);
  *addr = ad_acon(tmp, 0);
  *nme = addnme(NT_VAR, tmp, 0, 0);
  loc_of(*nme);
  if (XBIT(70, 0x40000000) && !imag) {
    if (dtype == DT_CMPLX)
      chk_block(ad4ili(IL_STSCMPLX, real, *addr, *nme, msz));
    else
      chk_block(ad4ili(IL_STDCMPLX, real, *addr, *nme, msz));
  } else {
    chk_block(ad4ili(store, real, *addr, addnme(NT_MEM, SPTR_NULL, *nme, 0), msz));
    chk_block(ad4ili(store, imag,
                     ad3ili(IL_AADD, *addr, ad_aconi(size), 0),
                     addnme(NT_MEM, NOSYM, *nme, size), msz));
  }
}

/**
 * \brief get the chain pointer argument from a descriptor.
 *
 * \param arglnk is a chain of argument ILI for a call-site.
 *
 * \param sdsc is the descriptor that has the chain pointer.
 *
 * \return  an IL_LDA ili chain that contains the ILI that loads the chain
 *          pointer from the descriptor.
 */
static int
get_chain_pointer_closure(SPTR sdsc)
{
  int nme, cp, cp_offset;

  if (XBIT(68, 0x1)) {
    cp_offset = 72;
  } else {
    cp_offset = 40;
  }
  nme = addnme(NT_VAR, sdsc, 0, 0);
  if (SCG(sdsc) != SC_DUMMY) {
    if (PARREFG(sdsc)) {
      /**
       * In LLVM, pointer descriptor is not visible in the outlined func.
       * Use mk_address() which fetches the uplevel ref
       */ 
      int addr = mk_address(sdsc);
      int ili = ad2ili(IL_LDA, addr, nme);
      cp = ad3ili(IL_AADD, ili, ad_aconi(cp_offset), 0);
    } else {
      cp = ad_acon(sdsc, cp_offset);
      cp = ad2ili(IL_LDA, cp, nme);
    }
  } else {
    SPTR asym = mk_argasym(sdsc);
    int addr = mk_address(sdsc);
    int ili = ad2ili(IL_LDA, addr, addnme(NT_VAR, asym, 0, 0));
    cp = ad3ili(IL_AADD, ili, ad_aconi(cp_offset), 0);
    if (!INTERNREFG(sdsc) && !PARREFG(sdsc))
      cp = ad2ili(IL_LDA, cp, nme);
  }

  return cp;
}

static int
add_last_arg(int arglnk, int displnk)
{
  int i;

  if (ILI_OPC(arglnk) == IL_NULL)
    return displnk;

  for (i = arglnk; i > 0 && ILI_OPC(ILI_OPND(i, 2)) != IL_NULL;
       i = ILI_OPND(i, 2)) {
    // do nothing
  }

  ILI_OPND(i, 2) = displnk;
  return arglnk;
}

static int
add_arglnk_closure(SPTR sdsc)
{
  int i;

  i = get_chain_pointer_closure(sdsc);
  i = ad3ili(IL_ARGAR, i, ad1ili(IL_NULL, 0), ad1ili(IL_NULL, 0));
  return i;
}

static int
add_gargl_closure(SPTR sdsc)
{
  int i;

  i = get_chain_pointer_closure(sdsc);
  i = ad4ili(IL_GARG, i, ad1ili(IL_NULL, 0), DT_ADDR, NME_VOL);
  return i;
}

static bool
is_asn_closure_call(int sptr)
{
  if (sptr > NOSYM && STYPEG(sptr) == ST_PROC && CCSYMG(sptr) &&
      strcmp(SYMNAME(sptr), mkRteRtnNm(RTE_asn_closure)) == 0) {
    return true;
  }
  return false;
}

static bool
is_proc_desc_arg(int ili)
{
  SPTR sym;
  if (ILI_OPC(ili) == IL_ACON) {
    sym = SymConval1(ILI_SymOPND(ili, 1));
  } else if (IL_TYPE(ILI_OPC(ili)) == ILTY_LOAD) {
    int op1 = ILI_OPND(ili,1);
    if (ILI_OPC(op1) == IL_ACON) {
      sym = SymConval1(ILI_SymOPND(op1, 1));
    } else {
      sym = NME_SYM(ILI_OPND(ili,2));
     }
  } else {
    sym = SPTR_NULL;
  }
  if (sym > NOSYM && IS_PROC_DESCRG(sym)) {
      return true;
  }
  return false;
}

void
exp_call(ILM_OP opc, ILM *ilmp, int curilm)
{
  int nargs;   /* # args */
  int ililnk;  /* ili link */
  int argili;  /* ili for arg */
  int argili2; /* ili for arg */
  int gargili; /* ili for arg */
  int ilix;    /* ili pointer */
  ILM *ilmlnk; /* current ILM operand */
  int ilm1;
  SPTR sym;    /* symbol pointers */
  INT skip;   /* distance to imag part of a complex */
  int basenm; /* base nm entry */
  int i;      /* temps */
  STRDESC *str1;
  int argopc;
  int cfunc;
  int cfunc_nme;
  DTYPE dtype, dtype1;
  int val_flag;
  int arglnk;
  int func_addr;
  int vtoff;
  int descno = 0;
  int gargl, gi, gjsr, ngargs, garg_disp;
  int gfch_addr, gfch_len; /* character function return */
  int jsra_mscall_flag;
  int funcptr_flags;
  int retdesc;
  int struct_tmp;
  int chain_pointer_arg = 0;
  int result_arg = 0;

  nargs = ILM_OPND(ilmp, 1); /* # args */
  func_addr = 0;
  funcptr_flags = 0;
  switch (opc) {
  case IM_CALL:
    exp_call_sym = ILM_SymOPND(ilmp, 2); /* external reference  */
    /* Q&D for the absence of prototypes/signatures for our run-time
     * routines. -- 9/19/14, do it for user functions too!
     */
    dtype1 = DTYPEG(exp_call_sym);
    DTYPEP(exp_call_sym, DT_NONE);
    break;
  case IM_CHFUNC:
  case IM_NCHFUNC:
  case IM_KFUNC:
  case IM_LFUNC:
  case IM_IFUNC:
  case IM_RFUNC:
  case IM_DFUNC:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_QFUNC:
#endif
  case IM_CFUNC:
  case IM_CDFUNC:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQFUNC:
#endif
  case IM_PFUNC:
  case IM_SFUNC:
    exp_call_sym = ILM_SymOPND(ilmp, 2); /* external reference  */
    break;
  case IM_CALLA:
  case IM_PCALLA:
  case IM_CHFUNCA:
  case IM_PCHFUNCA:
  case IM_NCHFUNCA:
  case IM_PNCHFUNCA:
  case IM_KFUNCA:
  case IM_PKFUNCA:
  case IM_LFUNCA:
  case IM_PLFUNCA:
  case IM_IFUNCA:
  case IM_PIFUNCA:
  case IM_RFUNCA:
  case IM_PRFUNCA:
  case IM_DFUNCA:
  case IM_PDFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_PQFUNCA:
#endif
  case IM_CFUNCA:
  case IM_PCFUNCA:
  case IM_CDFUNCA:
  case IM_PCDFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQFUNCA:
  case IM_PCQFUNCA:
#endif
  case IM_PFUNCA:
  case IM_PPFUNCA:
    funcptr_flags = ILM_OPND(ilmp, 2);
    exp_call_sym = SPTR_NULL; /* via procedure ptr */
    if (!IS_INTERNAL_PROC_CALL(opc)) {
      ilm1 = ILM_OPND(ilmp, 3);
    } else {
      ilm1 = ILM_OPND(ilmp, 4);
      descno = ILM_OPND(ilmp, 3);
    }
    func_addr = ILI_OF(ilm1);
    ilmlnk = (ILM *)(ilmb.ilm_base + ilm1);
    switch (ILM_OPC(ilmlnk)) {
    case IM_PLD:
      exp_call_sym = ILM_SymOPND(ilmlnk, 2);
      break;
    case IM_BASE:
      exp_call_sym = ILM_SymOPND(ilmlnk, 1);
      break;
    case IM_MEMBER:
      exp_call_sym = ILM_SymOPND(ilmlnk, 2);
      break;
    default:
      interr("exp_call: Procedure pointer not found", ilm1, ERR_unused);
      break;
    }
    break;
  case IM_VCALLA:
    descno = 5;
    goto vcalla_common;
  case IM_CHVFUNCA:
    descno = 5;
    goto vcalla_common;
  case IM_NCHVFUNCA:
    descno = 5;
    goto vcalla_common;
  case IM_KVFUNCA:
    descno = 5;
    goto vcalla_common;
  case IM_LVFUNCA:
    descno = 5;
    goto vcalla_common;
  case IM_IVFUNCA:
    descno = 5;
    goto vcalla_common;
  case IM_RVFUNCA:
    descno = 5;
    goto vcalla_common;
  case IM_DVFUNCA:
    descno = 5;
    goto vcalla_common;
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_QVFUNCA:
    descno = 5;
    goto vcalla_common;
#endif
  case IM_CVFUNCA:
    descno = 5;
    goto vcalla_common;
  case IM_CDVFUNCA:
    descno = 5;
    goto vcalla_common;
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQVFUNCA:
    descno = 5;
    goto vcalla_common;
#endif
  case IM_PVFUNCA:
    descno = 5;
  vcalla_common:
    exp_call_sym = SPTR_NULL; /* via type bound proc */
    descno = ILM_OPND(ilmp, descno);
    ilm1 = ILM_OPND(ilmp, 3);
    /* external reference  */
    exp_call_sym = ILM_SymOPND(ilmp, 3);
    vtoff = VTOFFG(TBPLNKG(exp_call_sym));
    if (VTABLEG(exp_call_sym))
      exp_call_sym = VTABLEG(exp_call_sym);
    else if (IFACEG(exp_call_sym))
      exp_call_sym = IFACEG(exp_call_sym);
    break;
  default:
    exp_call_sym = ILM_SymOPND(ilmp, 2); /* external reference  */
    interr("exp_call: Bad Function opc", opc, ERR_Severe);
  }

  init_arg_ili(nargs);

  if (opc == IM_LFUNC && nargs == 1) {
    if (CCSYMG(exp_call_sym) &&
        strcmp(SYMNAME(exp_call_sym), mkRteRtnNm(RTE_present)) == 0) {
      int opc1;
      /* F90 PRESENT() call; is this a missing optional argument? */
      ilm1 = ILM_OPND(ilmp, 3);
      opc1 = ILM_OPC((ILM *)(ilmb.ilm_base + ilm1));
      if (opc1 == IM_BASE) {
        if (optional_missing(NME_OF(ilm1))) {
          /* treat like zero */
          replace_by_zero(opc, ilmp, curilm);
          return;
        } else if (optional_present(NME_OF(ilm1))) {
          /* treat like one */
          replace_by_one(opc, ilmp, curilm);
          return;
        }
      } else if (IM_TYPE(opc1) == IMTY_CONS) {
        /* inlined optional argument, constant actual argument */
        /* treat like zero */
        replace_by_one(opc, ilmp, curilm);
        return;
      }
    }
  }

  gfch_addr = 0;
  switch (opc) {
  case IM_CHFUNC:
  case IM_NCHFUNC:
  case IM_CHFUNCA:
  case IM_NCHFUNCA:
  case IM_PCHFUNCA:
  case IM_PNCHFUNCA:
    /*
     * for a function returning character, the first 2 arguments
     * are the address of a char temporary created by the semantic
     * analyzer and its length, respectively.
     */

    if ((opc == IM_CHFUNC) || (opc == IM_NCHFUNC)) {
      ilm1 = ILM_OPND(ilmp, 3);
    } else if (opc == IM_PCHFUNCA || opc == IM_PNCHFUNCA) {
      ilm1 = ILM_OPND(ilmp, 5);
    } else {
      ilm1 = ILM_OPND(ilmp, 4);
    }
    if (IILM_OPC(ilm1) == IM_FARG)
      ilm1 = IILM_OPND(ilm1, 1);
    else if (IILM_OPC(ilm1) == IM_FARGF)
      ilm1 = IILM_OPND(ilm1, 1);
    gfch_addr = ILM_RESULT(ilm1);
    gfch_len = ILM_CLEN(ilm1);
    add_to_args(IL_ARGAR, gfch_addr);

    /* always add the character function length to the argument list:
       do not modify this with STDCALL, REFERENCE, VALUE
       the information required to do this has been lost at this
       call point : the sptr is different .  We don't have
       FVALG() or the parameter list
     */
    if (CHARLEN_64BIT) {
      gfch_len = sel_iconv(gfch_len, 1);
      add_to_args(IL_ARGKR, gfch_len);
    } else {
      add_to_args(IL_ARGIR, gfch_len);
    }
    if ((opc == IM_CHFUNC) || (opc == IM_NCHFUNC)) {
      i = 4; /* ilm pointer to first arg */
    } else {
      i = 5; /* ilm pointer to first arg */
    }
    break;
  case IM_CFUNC:
  case IM_CDFUNC:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQFUNC:
#endif
    i = 3;
    goto share_cfunc;
  case IM_PCFUNCA:
  case IM_PCDFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_PCQFUNCA:
#endif
    i = 5;
    goto share_cfunc;
  case IM_CFUNCA:
  case IM_CDFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQFUNCA:
#endif
    i = 4;
  share_cfunc:
    ilm1 = ILM_OPND(ilmp, i);
    dtype = IILM_DTyOPND(ilm1, 2);
    if (IILM_OPC(ilm1) == IM_FARG || IILM_OPC(ilm1) == IM_FARGF)
      ilm1 = IILM_OPND(ilm1, 1);
    cfunc = ILM_RESULT(ilm1);
    cfunc_nme = NME_OF(ilm1);
    if (CFUNCG(exp_call_sym) || (funcptr_flags & FUNCPTR_BINDC) ||
        CMPLXFUNC_C) {
      ADDRTKNP(IILM_OPND(ilm1, 1), 1);
      if (opc == IM_CFUNCA || opc == IM_CDFUNCA
#ifdef TARGET_SUPPORTS_QUADFP
          || opc == IM_CQFUNCA
#endif
         ) {
        ilm1 = ILM_OPND(ilmp, i);
      } else {
        ilm1 = ILM_OPND(ilmp, (i + 2));
      }
      if (XBIT(121, 0x800)) {
        garg_ili[0].ilix = cfunc;
        garg_ili[0].dtype = dtype;
        garg_ili[0].nme = cfunc_nme;
      }
      nargs--;
      i++;
    }
    break;
  case IM_CHVFUNCA:
  case IM_NCHVFUNCA:
    /*
     * for a function returning character, the first 2 arguments
     * are the address of a char temporary created by the semantic
     * analyzer and its length, respectively.
     */

    ilm1 = ILM_OPND(ilmp, 6);
    if (IILM_OPC(ilm1) == IM_FARG)
      ilm1 = IILM_OPND(ilm1, 1);
    else if (IILM_OPC(ilm1) == IM_FARGF)
      ilm1 = IILM_OPND(ilm1, 1);
    gfch_addr = ILM_RESULT(ilm1);
    gfch_len = ILM_CLEN(ilm1);
    add_to_args(IL_ARGAR, ILM_RESULT(ilm1));

    /* always add the character function length to the argument list:
       do not modify this with STDCALL, REFERENCE, VALUE
       the information required to do this has been lost at this
       call point : the sptr is different .  We don't have
       FVALG() or the parameter list
     */
    if (CHARLEN_64BIT) {
      gfch_len = sel_iconv(gfch_len, 1);
      add_to_args(IL_ARGKR, gfch_len);
    } else {
      add_to_args(IL_ARGIR, ILM_CLEN(ilm1));
    }
    i = 7; /* ilm pointer to first arg */
    break;
  case IM_CVFUNCA:
  case IM_CDVFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQVFUNCA:
#endif
    ilm1 = ILM_OPND(ilmp, 6);
    if (IILM_OPC(ilm1) == IM_FARG)
      ilm1 = IILM_OPND(ilm1, 1);
    else if (IILM_OPC(ilm1) == IM_FARGF)
      ilm1 = IILM_OPND(ilm1, 1);
    cfunc = ILM_RESULT(ilm1);
    cfunc_nme = NME_OF(ilm1);
    i = 6; /* ilm pointer to first arg */
    if (CMPLXFUNC_C)
      goto share_cfunc;
    break;
  case IM_VCALLA:
  case IM_KVFUNCA:
  case IM_LVFUNCA:
  case IM_IVFUNCA:
  case IM_RVFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_QVFUNCA:
#endif
  case IM_DVFUNCA:
  case IM_PVFUNCA:
    i = 6;
    break;
  case IM_SFUNC:
    /* eventually, delete retdesc;  XBIT(121,0x800) is the default and there
     * is always a return temp.
     */
    retdesc = check_cstruct_return(DTYPEG(exp_call_sym));
    struct_tmp = struct_ret_tmp(ILM_OPND(ilmp, 3));
    ilm1 = ILM_OPND(ilmp, 3);
    if (IILM_OPC(ilm1) == IM_FARG || IILM_OPC(ilm1) == IM_FARGF)
      ilm1 = IILM_OPND(ilm1, 1);
    cfunc = ILM_RESULT(ilm1);
    cfunc_nme = NME_OF(ilm1);
    nargs--;
    i = 4;
    if (XBIT(121, 0x800)) {
      add_struct_byval_to_args(IL_ARGAR, cfunc, DTYPEG(struct_tmp));
      garg_ili[0].ilix = cfunc;
      garg_ili[0].dtype = DTYPEG(struct_tmp);
      garg_ili[0].nme = cfunc_nme;
    }
    ilm1 = ILM_OPND(ilmp, i);
    break;

  case IM_IFUNCA:
  case IM_RFUNCA:
  case IM_DFUNCA:
  case IM_QFUNCA:
  case IM_M256FUNCA:
  case IM_M256VFUNCA:
  case IM_LFUNCA:
  case IM_PFUNCA:
  case IM_KFUNCA:
  case IM_CALLA:
    i = 4; /* ilm pointer to first arg */
    break;
  case IM_PCALLA:
  case IM_PIFUNCA:
  case IM_PRFUNCA:
  case IM_PDFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_PQFUNCA:
#endif
  case IM_PLFUNCA:
  case IM_PPFUNCA:
  case IM_PKFUNCA:
    descno = ILM_OPND(ilmp, 3);
    i = 5;
    break; /* ilm pointer to first arg */
  default:
    i = 3; /* ilm pointer to first arg */
    break;
  }

  ngargs = 0;
  if (XBIT(121, 0x800)) {
    ngargs = nargs;
  }
  gi = 1;
  while (nargs--) {
    bool pass_len = true;
    ilm1 = ILM_OPND(ilmp, i);
    dtype = DT_ADDR;
    val_flag = 0;
    if (IILM_OPC(ilm1) == IM_FARG) {
      dtype = IILM_DTyOPND(ilm1, 2);
      ilm1 = IILM_OPND(ilm1, 1);
    } else if (IILM_OPC(ilm1) == IM_FARGF) {
      dtype = IILM_DTyOPND(ilm1, 2);
      if (IILM_OPND(ilm1, 3) & 0x1) {
        /* corresponding formal is a CLASS(*) */
        pass_len = false;
      }
      ilm1 = IILM_OPND(ilm1, 1);
    }
    gargili = ILM_RESULT(ilm1);
    if (!result_arg)
      result_arg = gargili;
    ilmlnk = (ILM *)(ilmb.ilm_base + ilm1);
    /* ilmlnk is ith argument */
    switch (argopc = ILM_OPC(ilmlnk)) {
    case IM_PARG:
      /* special ILM for passing an object with the pointer attribute.
       * need to pass the address of the object's pointer variable
       * and a character length if the scalar/element type is character.
       */
      ilm1 = ILM_OPND(ilmlnk, 1); /* locate address of object's pointer */
      loc_of(NME_OF(ilm1));
      argili = ILI_OF(ilm1);
      ilm1 = ILM_OPND(ilmlnk, 2); /* BASE ILM of the object */
      if (ILM_RESTYPE(ilm1) != ILM_ISCHAR || !pass_len) {
        add_to_args(IL_ARGAR, argili);
      } else {
        pass_char_arg(IL_ARGAR, argili, ILM_CLEN(ilm1));
      }
      gargili = argili;
      break;
    case IM_BYVAL:
      ilm1 = ILM_OPND(ilmlnk, 1); /* operand of BYVAL */
      gargili = ILM_RESULT(ilm1);
      /* dtype of by-value argument */
      dtype = ILM_DTyOPND(ilmlnk, 2);
      val_flag = NME_VOL;
      ilmlnk = (ILM *)(ilmb.ilm_base + ilm1);
      argopc = ILM_OPC(ilmlnk);
      if (IM_TYPE(argopc) == IMTY_REF) {
        /* call by reference */
        loc_of(NME_OF(ilm1));
      }
      if (!DT_ISBASIC(dtype)) {
        argili = ILI_OF(ilm1);
        switch (IL_RES(ILI_OPC(argili))) {
        case ILIA_IR:
          argili = ad1ili(IL_IAMV, argili);
          add_to_args(IL_ARGAR, argili);
          break;
        case ILIA_KR:
          argili = ad1ili(IL_KAMV, argili);
          add_to_args(IL_ARGAR, argili);
          break;
        default:
          if (DTY(dtype) == TY_STRUCT) {
            add_struct_byval_to_args(IL_ARGAR, argili, dtype);
          } else {
            add_to_args(IL_ARGAR, argili);
          }
          break;
        }
        break;
      } else {
        if (ILI_OPC(gargili) == IL_DFRAR) {
          /* if argument of BYVAL is function call, then don't set val_flag */
          int ili = ILI_OPND(gargili, 1);
          if (ILI_OPC(ili) == IL_JSR)
            val_flag = 0;
        }
      }
      if (ILM_RESTYPE(ilm1) == ILM_ISCMPLX ||
          ILM_RESTYPE(ilm1) == ILM_ISDCMPLX || dtype == DT_CMPLX ||
          dtype == DT_DCMPLX) {
        int res, mem_msz, msz;
        ILI_OP st_opc, ld_opc, arg_opc;
        argili = ILM_RRESULT(ilm1);
        if (ILM_RESTYPE(ilm1) == ILM_ISCMPLX)
          arg_opc = IL_ARGSP;
        else
          arg_opc = IL_ARGDP;

        if (XBIT(70, 0x40000000)) {
          int rili;
          int addr, nme;
          /* llvm doesn't care for following arg ilis because it looks at garg.
           * we add each component to arg so that we don't get dump ili error
           * because we don't have ili for whole complex argument(except
           * DASPSP).
           */
          rili = ILM_RESULT(ilm1);
          gargili = rili;
          if (dtype == DT_CMPLX) {
            arg_opc = IL_ARGSP;
            argili = ad1ili(IL_SCMPLX2IMAG, rili);
            add_to_args(arg_opc, argili);
            argili = ad1ili(IL_SCMPLX2REAL, rili);
            add_to_args(arg_opc, argili);
          } else {
            arg_opc = IL_ARGDP;
            argili = ad1ili(IL_DCMPLX2IMAG, rili);
            add_to_args(arg_opc, argili);
            argili = ad1ili(IL_DCMPLX2REAL, rili);
            add_to_args(arg_opc, argili);
          }
          cmplx_to_mem(ILM_RESULT(ilm1), 0, dtype, &addr, &nme);
          gargili = addr;
          loc_of(nme);
          break;
        }

        add_to_args(arg_opc, argili);
#if   defined(IL_GJSR) && defined(USE_LLVM_CMPLX) /* New functionality */
        res = ILI_OPND(ILM_RESULT(ilm1), 1);
        basenm = 0;
        dtype = ILM_RESTYPE(ilm1) == ILM_ISCMPLX ? DT_CMPLX : DT_DCMPLX;
        ld_opc = dtype == DT_CMPLX ? IL_LDSCMPLX : IL_LDDCMPLX;
        msz = dtype == DT_CMPLX ? MSZ_F8 : MSZ_F16;
        mem_msz = dtype == DT_CMPLX ? MSZ_F4 : MSZ_F8;
        if (!ILIA_ISAR(IL_RES(ILI_OPC(res)))) {
          /* Not an address, so we need to add a temp store */
          st_opc = dtype == DT_CMPLX ? IL_STSP : IL_STDP;
          skip = dtype == DT_CMPLX ? size_of(DT_FLOAT) : size_of(DT_DBLE);
          sym = mkrtemp_cpx_sc(dtype, expb.sc);
          ADDRTKNP(sym, 1);
          basenm = addnme(NT_VAR, sym, 0, 0);

          /* Real component */
          res = ad_acon(sym, 0);
          ilix = ILM_RRESULT(ilm1);
          ilix = ad4ili(st_opc, ilix, res,
                        addnme(NT_MEM, SPTR_NULL, basenm, 0), mem_msz);
          chk_block(ilix);

          /* Imag component */
          ilix = ILM_IRESULT(ilm1);
          ilix = ad4ili(st_opc, ilix, ad_acon(sym, skip),
                        addnme(NT_MEM, NOSYM, basenm, skip), mem_msz);
          chk_block(ilix);
        }
        gargili = ad3ili(ld_opc, res, basenm, msz);
#endif /* GJSR && USE_LLVM_CMPLX (End of new functionality) */
        argili = ILM_IRESULT(ilm1);
        add_to_args(arg_opc, argili);
        break;
      }
      if (DTY(dtype) == TY_CHAR) {
        /*
         * NOTE that character scalar arguments may appear
         * as the operand to BYVAL -- need to ensure the
         * argument is in memory.
         */
        str1 = getstr(ilm1);
        if (str1->next)
          str1 = storechartmp(str1, ILM_MXLEN(ilm1), ILM_CLEN(ilm1));
        argili = getstraddr(str1);
        argili = ad3ili(IL_LD, argili, NME_STR1, MSZ_BYTE);
        gargili = argili;
      }
      else if (DTY(dtype) == TY_NCHAR) {
        /*
         * NOTE that character scalar arguments may appear
         * as the operand to BYVAL -- need to ensure the
         * argument is in memory.
         */
        str1 = getstr(ilm1);
        if (str1->next)
          str1 = storechartmp(str1, ILM_MXLEN(ilm1), ILM_CLEN(ilm1));
        argili = getstraddr(str1);
        argili = ad3ili(IL_LD, argili, NME_STR1, MSZ_UHWORD);
        gargili = argili;
      }
      else {
        /*
         * SIMPLE scalar types!
         * NOTE that character scalar arguments may already bei
         * passed as an integer via ICHAR.
         */
        /* word expression by value */
        argili = ILM_RESULT(ilm1);
      }
      add_arg_ili(argili, 0, 0);
      break;

    case IM_DPSCON: /* short constant by value */
      dtype = DT_INT;
      argili = ad_icon(ILM_OPND(ilmlnk, 1));
      /* store all the argument entries so we can process
       * them in the same order as C
       */
      add_to_args(IL_ARGIR, argili);
      gargili = argili;
      break;

    case IM_DPNULL: /* null character string */
      dtype = DT_CHAR;
      argili = ad_acon(SPTR_NULL, 0);
      if (pass_len) {
        argili2 = ad_icon(0);
        pass_char_arg(IL_ARGAR, argili, argili2);
      } else
        add_to_args(IL_ARGAR, argili);
      gargili = argili;
      break;

    case IM_DPVAL:
      ilm1 = ILM_OPND(ilmlnk, 1); /* operand of DPVAL */
      gargili = ILM_RESULT(ilm1);
      val_flag = NME_VOL;
      if (ILM_RESTYPE(ilm1) == ILM_ISCMPLX) {
        dtype = DT_REAL;
        argili = ILM_RRESULT(ilm1);
        add_to_args(IL_ARGSP, argili);
        if (XBIT(121, 0x800)) {
          garg_ili[gi].ilix = gargili;
          garg_ili[gi].dtype = dtype;
          garg_ili[gi].val_flag = NME_VOL;
          gi++;
          ngargs++;
        }
        argili = ILM_IRESULT(ilm1);
        gargili = argili;
        add_to_args(IL_ARGSP, argili);
        break;
      }
      if (ILM_RESTYPE(ilm1) == ILM_ISDCMPLX) {
        dtype = DT_DBLE;
        argili = ILM_RRESULT(ilm1);
        add_to_args(IL_ARGDP, argili);
        if (XBIT(121, 0x800)) {
          garg_ili[gi].ilix = gargili;
          garg_ili[gi].dtype = dtype;
          garg_ili[gi].val_flag = NME_VOL;
          gi++;
          ngargs++;
        }
        argili = ILM_IRESULT(ilm1);
        gargili = argili;
        add_to_args(IL_ARGDP, argili);
        break;
      }
      /* word expression by value */
      argili = ILM_RESULT(ilm1);
      switch (IL_RES(ILI_OPC(argili))) {
      case ILIA_IR:
        add_to_args(IL_ARGIR, argili);
        dtype = DT_INT;
        break;
      case ILIA_KR:
        add_to_args(IL_ARGKR, argili);
        dtype = DT_INT8;
        break;
      case ILIA_SP:
        add_to_args(IL_ARGSP, argili);
        dtype = DT_REAL;
        break;
      case ILIA_DP:
        add_to_args(IL_ARGDP, argili);
        dtype = DT_DBLE;
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case ILIA_QP:
        add_to_args(IL_ARGQP, argili);
        dtype = DT_QUAD;
        break;
#endif
      case ILIA_AR:
        add_to_args(IL_ARGAR, argili);
        dtype = DT_ADDR;
        break;
      case ILIA_CS:
        /* this happens when frontend put DPVAL on top of COMPLEX ILM
         * For example: print *, complex
         * Not really sure if we have any other cases.
         */
        dtype = DT_REAL;
        argili = ad1ili(IL_SCMPLX2REAL, ILM_RESULT(ilm1));
        add_to_args(IL_ARGSP, argili);
        if (XBIT(121, 0x800)) {
          garg_ili[gi].ilix = argili;
          garg_ili[gi].dtype = dtype;
          gi++;
          ngargs++;
        }
        argili = ad1ili(IL_SCMPLX2IMAG, ILM_RESULT(ilm1));
        gargili = argili;
        add_to_args(IL_ARGSP, argili);
        break;
      case ILIA_CD:
        dtype = DT_DBLE;
        argili = ad1ili(IL_DCMPLX2REAL, ILM_RESULT(ilm1));
        add_to_args(IL_ARGDP, argili);
        if (XBIT(121, 0x800)) {
          garg_ili[gi].ilix = argili;
          garg_ili[gi].dtype = dtype;
          gi++;
          ngargs++;
        }
        argili = ad1ili(IL_DCMPLX2IMAG, ILM_RESULT(ilm1));
        gargili = argili;
        add_to_args(IL_ARGDP, argili);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case ILIA_CQ:
        dtype = DT_QUAD;
        argili = ad1ili(IL_QCMPLX2REAL, ILM_RESULT(ilm1));
        add_to_args(IL_ARGQP, argili);
        if (XBIT(121, 0x800)) {
          garg_ili[gi].ilix = argili;
          garg_ili[gi].dtype = dtype;
          gi++;
          ngargs++;
        }
        argili = ad1ili(IL_QCMPLX2IMAG, ILM_RESULT(ilm1));
        gargili = argili;
        add_to_args(IL_ARGQP, argili);
        break;
#endif
      default:
        interr("exp_call:bad ili for DPVAL", argili, ERR_Severe);
      }
      break;

    case IM_DPREF:                /* %REF(expression) */
      ilm1 = ILM_OPND(ilmlnk, 1); /* operand of DPREF */
      gargili = ILM_RESULT(ilm1);
      ilmlnk = (ILM *)(ilmb.ilm_base + ilm1);
      /*
       * If the argument to %ref is character, only the address
       * of the expression is used (no length is needed).
       * Otherwise, DPREF is handled just like the default case.
       */
      if (ILM_RESTYPE(ilm1) == ILM_ISCHAR) {
        str1 = getstr(ilm1);
        if (str1->next)
          str1 = storechartmp(str1, ILM_MXLEN(ilm1), ILM_CLEN(ilm1));
        argili = getstraddr(str1);
        add_to_args(IL_ARGAR, argili);
        break;
      }
      argopc = ILM_OPC(ilmlnk);
      if (argopc == IM_PLD) {
        argili = ILM_RESULT(ilm1);
        add_to_args(IL_ARGAR, argili);
        break;
      }
      goto argdefault;

    case IM_DPREF8:               /* pass integer*8 as integer*4 */
      ilm1 = ILM_OPND(ilmlnk, 1); /* operand of DPREF8 */
      gargili = ILM_RESULT(ilm1);
      ilmlnk = (ILM *)(ilmb.ilm_base + ilm1);
      argopc = ILM_OPC(ilmlnk);
      goto argdefault;

    case IM_PLD:
      if (ILM_RESTYPE(ilm1) != ILM_ISCHAR) {
        argili = ILM_RESULT(ilm1);
        add_to_args(IL_ARGAR, argili);
        break;
      }
      /* else fall thru for handling character */
      FLANG_FALLTHROUGH;

    default:
      gargili = ILM_RESULT(ilm1);
      /* As for function which return assumed-length character,
         use A_CALL and process its second argument(length of character) here */
      if ((opc == IM_CALL) && (dtype1 == DT_ASSCHAR) && (nargs == (ngargs - 2))) {
        if (CHARLEN_64BIT) {
          gargili = sel_iconv(gargili, 1);
          add_to_args(IL_ARGKR, gargili);
        } else {
          add_to_args(IL_ARGIR, gargili);
        }
        break;
      }
      if (ILM_RESTYPE(ilm1) == ILM_ISCHAR) {
        str1 = getstr(ilm1);
        if (str1->next)
          str1 = storechartmp(str1, ILM_MXLEN(ilm1), ILM_CLEN(ilm1));
        argili = getstraddr(str1);
        if (pass_len) {
          pass_char_arg(IL_ARGAR, argili, getstrlen(str1));
        } else {
          add_to_args(IL_ARGAR, argili);
        }
        gargili = argili;
        break;
      }
    argdefault:
      if (IM_TYPE(argopc) == IMTY_REF) {
        /* call by reference */
        loc_of(NME_OF(ilm1));
        argili = ILI_OF(ilm1);
      } else if (IM_TYPE(argopc) == IMTY_CONS) {
        argili = ad_acon(ILM_SymOPND(ilmlnk, 1), 0);
      } else {
        /* general expression */
        if (ILM_RESTYPE(ilm1) == ILM_ISCMPLX) {
          sym = mkrtemp_cpx_sc(DT_CMPLX, expb.sc);
        } else if (ILM_RESTYPE(ilm1) == ILM_ISDCMPLX) {
          sym = mkrtemp_cpx_sc(DT_DCMPLX, expb.sc);
        } else
          sym = mkrtemp_sc(ILM_RESULT(ilm1), expb.sc);
        ADDRTKNP(sym, 1);
        /* generate store into temp */
        argili = ad_acon(sym, 0);
        basenm = addnme(NT_VAR, sym, 0, 0);
        if (ILM_RESTYPE(ilm1) == ILM_ISCMPLX) {
          skip = size_of(DT_FLOAT);
          ilix = ILM_RRESULT(ilm1);
          ilix = ad4ili(IL_STSP, ilix, argili,
                        addnme(NT_MEM, SPTR_NULL, basenm, 0), MSZ_F4);
          chk_block(ilix);
          ilix = ILM_IRESULT(ilm1);
          ilix = ad4ili(IL_STSP, ilix, ad_acon(sym, skip),
                        addnme(NT_MEM, NOSYM, basenm, skip), MSZ_F4);
          chk_block(ilix);
        } else if (ILM_RESTYPE(ilm1) == ILM_ISDCMPLX) {
          skip = size_of(DT_DBLE);
          ilix = ILM_RRESULT(ilm1);
          ilix = ad4ili(IL_STDP, ilix, argili,
                        addnme(NT_MEM, SPTR_NULL, basenm, 0), MSZ_F8);
          chk_block(ilix);
          ilix = ILM_IRESULT(ilm1);
          ilix = ad4ili(IL_STDP, ilix, ad_acon(sym, skip),
                        addnme(NT_MEM, NOSYM, basenm, skip), MSZ_F8);
          chk_block(ilix);
        } else {
          ilix = ILM_RESULT(ilm1);
          switch (IL_RES(ILI_OPC(ilix))) {
          case ILIA_IR:
            ilix = ad4ili(IL_ST, ilix, argili, basenm, MSZ_WORD);
            break;
          case ILIA_KR:
            ilix = ad4ili(IL_STKR, ilix, argili, basenm, MSZ_I8);
            break;
          case ILIA_AR:
            ilix = ad3ili(IL_STA, ilix, argili, basenm);
            break;
          case ILIA_SP:
            ilix = ad4ili(IL_STSP, ilix, argili, basenm, MSZ_F4);
            break;
          case ILIA_DP:
            ilix = ad4ili(IL_STDP, ilix, argili, basenm, MSZ_F8);
            break;
#ifdef TARGET_SUPPORTS_QUADFP
          case ILIA_QP:
            ilix = ad4ili(IL_STQP, ilix, argili, basenm, MSZ_F16);
            break;
#endif
          case ILIA_CS:
            ilix = ad4ili(IL_STSCMPLX, ilix, argili, basenm, MSZ_F8);
            break;
          case ILIA_CD:
            ilix = ad4ili(IL_STDCMPLX, ilix, argili, basenm, MSZ_F16);
            break;
#ifdef TARGET_SUPPORTS_QUADFP
          case ILIA_CQ:
            ilix = ad4ili(IL_STQCMPLX, ilix, argili, basenm, MSZ_F32);
            break;
#endif
          default:
            // in exp_call for IM_SFUNC, we decide to save IL_JSR
            // in the ILI_OF(or ILM_RESULT) field.
            // Check here if that is the case
            if (ILI_ALT(ilix)) {
              int alt_call = ILI_ALT(ilix);
              int ili_opnd = ILI_OPND(alt_call, 2);
              if (ILI_OPC(ili_opnd) == IL_GARGRET) {
                DTYPE dtype = ILI_DTyOPND(ili_opnd, 3);
                int nme = ILI_OPND(ili_opnd, 4);
                chk_block(ilix);
                ilix = ILI_OPND(ili_opnd, 1);
                /* copy from ilix to argili */
                _exp_smove(basenm, nme, argili, ilix, dtype);
                ilix = 0;
                break;
              }
            }
            interr("exp_call: ili ret type not cased", argili, ERR_Severe);
          }
          if (ilix > 0)
            chk_block(ilix);
        }
      } /* else general expression */
      if (CSTRUCTRETG(exp_call_sym) && nargs + 1 == ILM_OPND(ilmp, 1)) {
        /* if this is call to a bind C rtn that returns a C struct on the
         * stack, the dtype needs to be set to 1 or prevent the code
         * generator from aligning the stack argument area.  This happens
         * only for 32 bit compiles.  The CSTRUCTRETG is ignored by the
         * 64 bit compilers.
         */
        add_struct_byval_to_args(IL_ARGAR, argili, 1);
      } else
      {
        add_to_args(IL_ARGAR, argili);
      }
      gargili = argili;
      break;
    } /* switch */
    if (XBIT(121, 0x800)) {
      garg_ili[gi].ilix = gargili;
      garg_ili[gi].dtype = dtype;
      garg_ili[gi].val_flag = val_flag;
    }
    i++;
    gi++;
  } /* for each arg */

  arglnk = gen_arg_ili();
  garg_disp = 0;

  if (gbl.internal &&
      (CONTAINEDG(exp_call_sym) || is_asn_closure_call(exp_call_sym))) {
    int disp;
    int nme;
    /* calling contained procedure from
     *   outlined program
     *   host program
     *   another internal procedure
     */
    if (gbl.outlined) {
      nme = addnme(NT_VAR, aux.curr_entry->display, 0, 0);
      disp = mk_address(aux.curr_entry->display);
      disp = ad2ili(IL_LDA, disp, nme);
    } else if (gbl.internal == 1) {
      disp = ad_acon(aux.curr_entry->display, 0);
    } else {
      if (SCG(aux.curr_entry->display) == SC_DUMMY) {
        const SPTR sdisp = mk_argasym(aux.curr_entry->display);
        nme = addnme(NT_VAR, sdisp, 0, 0);
        disp = mk_address(sdisp);
        disp = ad2ili(IL_LDA, disp, nme);
      } else {
        /* Should not get here - something is wrong */
        const SPTR sdisp = sptr_mk_address(aux.curr_entry->display);
        disp = ad2ili(IL_LDA, sdisp, addnme(NT_VAR, sdisp, 0, 0));
      }
    }
    if (!XBIT(121, 0x800)) {
      chain_pointer_arg =
          ad3ili(IL_ARGAR, disp, ad1ili(IL_NULL, 0), ad1ili(IL_NULL, 0));
    }

    if (XBIT(121, 0x800))
      garg_disp = disp;
  }

  /* generate call */
  if (XBIT(121, 0x800)) {
    int dt;
    gargl = ad1ili(IL_NULL, 0);
    if (charargs) {
      /* when character arguments are present, place any procedure descriptor
       * arguments at the end of the argument list.
       */
      for (gi = ngargs; gi >= 1; gi--) {
        if (!HAS_OPT_ARGSG(exp_call_sym) &&
            is_proc_desc_arg(garg_ili[gi].ilix)) {
          ilix = ad4ili(IL_GARG, garg_ili[gi].ilix, gargl, garg_ili[gi].dtype,
                        garg_ili[gi].val_flag);
          gargl = ilix;
        }
      }
      if (IL_RES(ILI_OPC(len_ili[0])) != ILIA_KR)
        dt = DT_INT;
      else
        dt = DT_INT8;
      for (i = charargs - 1; i >= 0; i--) {
        ilix = ad4ili(IL_GARG, len_ili[i], gargl, dt, NME_VOL);
        gargl = ilix;
      }
    }
    for (gi = ngargs; gi >= 1; gi--) {
      if (charargs && !HAS_OPT_ARGSG(exp_call_sym) &&
          is_proc_desc_arg(garg_ili[gi].ilix)) {
        /* already processed the procedure descriptor argument in this case */
        continue;
      }

      // IM_CALL falls in the default case and a BIND(C) func returning struct
      // should use IL_GARGRET.
      if (opc == IM_CALL && CSTRUCTRETG(exp_call_sym) &&
          garg_ili[gi].ilix == result_arg) {
        ilix = ad4ili(IL_GARGRET, garg_ili[gi].ilix, gargl, garg_ili[gi].dtype,
                      garg_ili[gi].val_flag);
      } else {
        ilix = ad4ili(IL_GARG, garg_ili[gi].ilix, gargl, garg_ili[gi].dtype,
                      garg_ili[gi].val_flag);
      }
      gargl = ilix;
    }
    if (gfch_addr) {
      if (IL_RES(ILI_OPC(gfch_len)) != ILIA_KR)
        dt = DT_INT;
      else
        dt = DT_INT8;
      ilix = ad4ili(IL_GARG, gfch_len, gargl, dt, NME_VOL);
      gargl = ilix;
      ilix = ad4ili(IL_GARG, gfch_addr, gargl, DT_ADDR, 0);
      gargl = ilix;
    }
    if (garg_ili[0].ilix) {
      ilix = ad4ili(IL_GARGRET, garg_ili[0].ilix, gargl, garg_ili[0].dtype,
                    garg_ili[0].nme);
      gargl = ilix;
    }
    if (garg_disp) {
      ilix = ad4ili(IL_GARG, garg_disp, ad1ili(IL_NULL, 0), DT_ADDR, 0);
      if (ILI_OPC(gargl) == IL_NULL)
        gargl = ilix;
      else
        add_last_arg(gargl, ilix);
    }
  }
  if (chain_pointer_arg != 0) {
    if (ILI_OPC(arglnk) == IL_NULL)
      arglnk = chain_pointer_arg;
    else
      add_last_arg(arglnk, chain_pointer_arg);
  }
  fptr_iface = SPTR_NULL;
  if (exp_call_sym) {
    DTYPE dt;
    fptr_iface = exp_call_sym;
    switch (STYPEG(fptr_iface)) {
    case ST_ENTRY:
    case ST_PROC:
      break;
    default:
      dt = DTYPEG(fptr_iface);
      if (DTY(dt) == TY_PTR && DTY(DTySeqTyElement(dt)) == TY_PROC) {
        fptr_iface = DTyInterface(DTySeqTyElement(dt));
      } else {
        fptr_iface = SPTR_NULL;
      }
      break;
    }
  }
  if (func_addr) {
    if (!MSCALLG(exp_call_sym))
      jsra_mscall_flag = 0;
    else
      jsra_mscall_flag = 0x1;
    if (IS_INTERNAL_PROC_CALL(opc)) {
      SPTR sptr_descno = (SPTR) descno;
      arglnk = add_last_arg(arglnk, add_arglnk_closure(sptr_descno));
      if (XBIT(121, 0x800)) {
        gargl = add_last_arg(gargl, add_gargl_closure(sptr_descno));
      }
    }
    ililnk = ad4ili(IL_JSRA, func_addr, arglnk, jsra_mscall_flag, fptr_iface);
    if (XBIT(121, 0x800)) {
      gjsr = ad4ili(IL_GJSRA, func_addr, gargl, jsra_mscall_flag, fptr_iface);
      ILI_ALT(ililnk) = gjsr;
    }
  } else if (SCG(exp_call_sym) != SC_DUMMY) {
    switch (opc) {
    case IM_VCALLA:
    case IM_CHVFUNCA:
    case IM_NCHVFUNCA:
    case IM_KVFUNCA:
    case IM_LVFUNCA:
    case IM_IVFUNCA:
    case IM_RVFUNCA:
    case IM_DVFUNCA:
    case IM_CVFUNCA:
    case IM_CDVFUNCA:
#ifdef TARGET_SUPPORTS_QUADFP
    case IM_CQVFUNCA:
#endif
    case IM_PVFUNCA: {
      SPTR sptr_descno = (SPTR) descno;
      ililnk = exp_type_bound_proc_call(exp_call_sym, sptr_descno, vtoff, arglnk);
      if (XBIT(121, 0x800)) {
        if (!MSCALLG(exp_call_sym))
          jsra_mscall_flag = 0;
        else
          jsra_mscall_flag = 0x1;
        gjsr = ad4ili(IL_GJSRA, ILI_OPND(ililnk, 1), gargl, jsra_mscall_flag,
                      fptr_iface);
        ILI_ALT(ililnk) = gjsr;
      }
    } break;
    default:
      ililnk = ad2ili(IL_JSR, exp_call_sym, arglnk);
      if (XBIT(121, 0x800)) {
        gjsr = ad2ili(IL_GJSR, exp_call_sym, gargl);
        ILI_ALT(ililnk) = gjsr;
      }
    }
  } else {
    SPTR asym = mk_argasym(exp_call_sym);
    int addr = mk_address(exp_call_sym);
    /* Currently we don't set CONTAINEDG for outlined function - no need too */
    if (!((INTERNREFG(exp_call_sym) && CONTAINEDG(gbl.currsub)) ||
          (gbl.outlined && PARREFG(exp_call_sym))))
      addr = ad2ili(IL_LDA, addr, addnme(NT_VAR, asym, 0, 0));
    if (!MSCALLG(exp_call_sym))
      jsra_mscall_flag = 0;
    else
      jsra_mscall_flag = 0x1;

    ililnk = ad4ili(IL_JSRA, addr, arglnk, jsra_mscall_flag, fptr_iface);
    if (XBIT(121, 0x800)) {
      gjsr = ad4ili(IL_GJSRA, addr, gargl, jsra_mscall_flag, fptr_iface);
      ILI_ALT(ililnk) = gjsr;
    }
  }
  iltb.callfg = 1;
  switch (opc) {
  case IM_CALL:
  case IM_CHFUNC:
  case IM_NCHFUNC:
  case IM_CALLA:
  case IM_PCALLA:
  case IM_VCALLA:
  case IM_CHVFUNCA:
  case IM_NCHVFUNCA:
  case IM_CHFUNCA:
  case IM_NCHFUNCA:
  case IM_PCHFUNCA:
  case IM_PNCHFUNCA:
    chk_block(ililnk);
    break;
  case IM_KFUNC:
  case IM_KFUNCA:
  case IM_PKFUNCA:
  case IM_KVFUNCA:
    ililnk = ad2ili(IL_DFRKR, ililnk, KR_RETVAL);
    ILI_OF(curilm) = ililnk;
    break;
  case IM_LFUNC:
  case IM_IFUNC:
  case IM_LFUNCA:
  case IM_IFUNCA:
  case IM_PLFUNCA:
  case IM_PIFUNCA:
  case IM_LVFUNCA:
  case IM_IVFUNCA:
    ILI_OF(curilm) = ad2ili(IL_DFRIR, ililnk, IR_RETVAL);
    break;
  case IM_RFUNC:
  case IM_RFUNCA:
  case IM_PRFUNCA:
  case IM_RVFUNCA:
    ILI_OF(curilm) = ad2ili(IL_DFRSP, ililnk, FR_RETVAL);
    break;
  case IM_DFUNC:
  case IM_DFUNCA:
  case IM_PDFUNCA:
  case IM_DVFUNCA:
    ILI_OF(curilm) = ad2ili(IL_DFRDP, ililnk, FR_RETVAL);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_QFUNC:
  case IM_QFUNCA:
  case IM_PQFUNCA:
  case IM_QVFUNCA:
    ILI_OF(curilm) = ad2ili(IL_DFRQP, ililnk, FR_RETVAL);
    break;
#endif
  case IM_CFUNC:
  case IM_CFUNCA:
  case IM_PCFUNCA:
  case IM_CVFUNCA:
    chk_block(ililnk);
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDSCMPLX, cfunc, cfunc_nme, MSZ_F8);
    } else {
      ILM_RRESULT(curilm) = ad3ili(IL_LDSP, cfunc, addnme(NT_MEM, SPTR_NULL, cfunc_nme, 0), MSZ_F4);
      ILM_IRESULT(curilm) = ad3ili(IL_LDSP, ad3ili(IL_AADD, cfunc, ad_aconi(4), 0), addnme(NT_MEM, NOSYM, cfunc_nme, 4), MSZ_F4);
      ILM_RESTYPE(curilm) = ILM_ISCMPLX;
    }

    break;
  case IM_CDFUNC:
  case IM_CDFUNCA:
  case IM_PCDFUNCA:
  case IM_CDVFUNCA:
    chk_block(ililnk);
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDDCMPLX, cfunc, cfunc_nme, MSZ_F16);
    } else {
      ILM_RRESULT(curilm) = ad3ili(IL_LDDP, cfunc, addnme(NT_MEM, SPTR_NULL, cfunc_nme, 0), MSZ_F8);
      ILM_IRESULT(curilm) =
          ad3ili(IL_LDDP, ad3ili(IL_AADD, cfunc, ad_aconi(8), 0),
                 addnme(NT_MEM, NOSYM, cfunc_nme, 8), MSZ_F8);
      ILM_RESTYPE(curilm) = ILM_ISDCMPLX;
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IM_CQFUNC:
  case IM_CQFUNCA:
  case IM_PCQFUNCA:
  case IM_CQVFUNCA:
    chk_block(ililnk);
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDQCMPLX, cfunc, cfunc_nme, MSZ_F32);
    } else {
      ILM_RRESULT(curilm) = ad3ili(IL_LDQP, cfunc, addnme(NT_MEM, SPTR_NULL, cfunc_nme, 0), MSZ_F16);
      ILM_IRESULT(curilm) =
          ad3ili(IL_LDQP, ad3ili(IL_AADD, cfunc, ad_aconi(16), 0),
                 addnme(NT_MEM, NOSYM, cfunc_nme, 16), MSZ_F16);
      ILM_RESTYPE(curilm) = ILM_ISQCMPLX;
    }
    break;
#endif
  case IM_PFUNC:
  case IM_PFUNCA:
  case IM_PPFUNCA:
  case IM_PVFUNCA:
    ILI_OF(curilm) = ad2ili(IL_DFRAR, ililnk, AR_RETVAL);
    ILM_NME(curilm) = NME_UNK;
    break;
  case IM_SFUNC:
    if (XBIT(121, 0x800)) {
      /* set the result to the JSR so that its result (hidden) argument can be
       * replaced:
      chk_block(ililnk);
      ILI_OF(curilm) = cfunc;
       */
      ILI_OF(curilm) = ililnk;
      ILM_NME(curilm) = cfunc_nme;
      break;
    }
    /* the rest will soon be deleted */
    if (retdesc == 1) {
      if (sizeof(DTYPEG(exp_call_sym)) <= 4) {
        ililnk = ad2ili(IL_DFRIR, ililnk, IR_RETVAL);
        ililnk = ad4ili(IL_STKR, ililnk, cfunc, cfunc_nme, MSZ_WORD);
      } else {
        ililnk = ad2ili(IL_DFRKR, ililnk, KR_RETVAL);
        ililnk = ad4ili(IL_STKR, ililnk, cfunc, cfunc_nme, MSZ_I8);
      }
      chk_block(ililnk);

      ILI_OF(curilm) = cfunc;
      ILM_NME(curilm) = cfunc_nme;

    } else {
      /* callee should copy result into hidden argument */
      ililnk = ad2ili(IL_DFRAR, ililnk, AR_RETVAL);
      ILM_NME(curilm) = cfunc_nme;
    }
    break;
  default:
    interr("exp_call: bad function opc", opc, ERR_Severe);
  }
  end_arg_ili();
}

/**
   \param ext        name of routine to call
   \param res_dtype  function return type

   Generate a sequence of ili for the current ilm which is an "arithmetic".
   This sequence essentially looks like a normal call, except where we can,
   arguments are passed by value.

   The requirements are:
   1.  The ilm looks like an "arithmetic" ILM where there's a fixed number
       of operands (determined by the ilms info).
   2.  The result is returned in a temporary.
   3.  The address of the result is the first argument in the call.
   4.  The operands are fully evaluated (no reference ilms).
   5.  Character arguments are not seen.

   For now this only works for complex/double complex ILMs which are QJSRs in
   the "standard" fortran.
 */
void
exp_qjsr(const char *ext, DTYPE res_dtype, ILM *ilmp, int curilm)
{
  int nargs;
  int ililnk;  /* ili link */
  int ilix;    /* ili pointer */
  ILM *ilmlnk; /* current ILM operand */
  int ilm1;
  int i;      /* temps */
  static ainfo_t ainfo;
  SPTR res; /* sptr of function result temporary */
  int res_addr;
  int res_nme;
  int extsym;

  if (DT_ISCMPLX(res_dtype)) {
    res = mkrtemp_arg1_sc(res_dtype, expb.sc);
    res_addr = ad_acon(res, 0);
    res_nme = addnme(NT_VAR, res, 0, 0);
    ADDRTKNP(res, 1);
  } else {
    interr("exp_qjsr, illegal dtype", res_dtype, ERR_Severe);
    return;
  }
  nargs = ilms[ILM_OPC(ilmp)].oprs;
  extsym = mkfunc(ext);
#ifdef ARG1PTRP
  ARG1PTRP(extsym, 1);
#endif
  init_ainfo(&ainfo);

  i = nargs;
  while (nargs--) {
    ilm1 = ILM_OPND(ilmp, i);
    ilmlnk = (ILM *)(ilmb.ilm_base + ilm1); /* ith operand */
    switch (ILM_RESTYPE(ilm1)) {
    case ILM_ISCHAR:
      interr("exp_qjsr: char arg not allowed", ilm1, ERR_Severe);
      break;
    case ILM_ISCMPLX:
      arg_sp(ILM_IRESULT(ilm1), &ainfo);
      arg_sp(ILM_RRESULT(ilm1), &ainfo);
      break;
    case ILM_ISDCMPLX:
      arg_dp(ILM_IRESULT(ilm1), &ainfo);
      arg_dp(ILM_RRESULT(ilm1), &ainfo);
      break;
    default:
      ilix = ILM_RESULT(ilm1);
      switch (IL_RES(ILI_OPC(ilix))) {
      case ILIA_IR:
        arg_ir(ilix, &ainfo);
        break;
      case ILIA_AR:
        arg_ar(ilix, &ainfo, 0);
        break;
      case ILIA_SP:
        arg_sp(ilix, &ainfo);
        break;
      case ILIA_DP:
        arg_dp(ilix, &ainfo);
        break;
      case ILIA_KR:
        arg_kr(ilix, &ainfo);
        break;
#ifdef ILIA_CS
      case ILIA_CS:
        ilix = ad1ili(IL_SCMPLX2IMAG, ILM_RESULT(ilm1));
        arg_sp(ilix, &ainfo);
        ilix = ad1ili(IL_SCMPLX2REAL, ILM_RESULT(ilm1));
        arg_sp(ilix, &ainfo);
        break;
      case ILIA_CD:
        ilix = ad1ili(IL_DCMPLX2IMAG, ILM_RESULT(ilm1));
        arg_dp(ilix, &ainfo);
        ilix = ad1ili(IL_DCMPLX2REAL, ILM_RESULT(ilm1));
        arg_dp(ilix, &ainfo);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case ILIA_CQ:
        ilix = ad1ili(IL_QCMPLX2IMAG, ILM_RESULT(ilm1));
        arg_qp(ilix, &ainfo);
        ilix = ad1ili(IL_QCMPLX2REAL, ILM_RESULT(ilm1));
        arg_qp(ilix, &ainfo);
        break;
#endif
#endif
      default:
        interr("exp_qjsr: ili ret type not cased", ilix, ERR_Severe);
        break;
      }
    }
    i--;
  } /* for each arg */

  arg_ar(res_addr, &ainfo, 0);
  ililnk = ad2ili(IL_JSR, extsym, ainfo.lnk);
  iltb.callfg = 1;
  chk_block(ililnk);

  if (res_dtype == DT_CMPLX) {
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDSCMPLX, res_addr, res_nme, MSZ_F8);
    } else {
      ILM_RRESULT(curilm) = ad3ili(IL_LDSP, res_addr, addnme(NT_MEM, SPTR_NULL, res_nme, 0), MSZ_F4);
      ILM_IRESULT(curilm) =
          ad3ili(IL_LDSP, ad3ili(IL_AADD, res_addr, ad_aconi(4), 0),
                 addnme(NT_MEM, NOSYM, res_nme, 4), MSZ_F4);
      ILM_RESTYPE(curilm) = ILM_ISCMPLX;
    }
#ifdef TARGET_SUPPORTS_QUADFP
  } else if (res_dtype == DT_QCMPLX) {
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDQCMPLX, res_addr, res_nme, MSZ_F32);
    } else {
      ILM_RRESULT(curilm) = ad3ili(IL_LDQP, res_addr, addnme(NT_MEM, SPTR_NULL, res_nme, 0), MSZ_F16);
      ILM_IRESULT(curilm) =
          ad3ili(IL_LDQP, ad3ili(IL_AADD, res_addr, ad_aconi(16), 0),
                 addnme(NT_MEM, NOSYM, res_nme, 16), MSZ_F16);
      ILM_RESTYPE(curilm) = ILM_ISQCMPLX;
    }
#endif
  } else {
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDDCMPLX, res_addr, res_nme, MSZ_F16);
    } else {

      ILM_RRESULT(curilm) = ad3ili(IL_LDDP, res_addr, addnme(NT_MEM, SPTR_NULL, res_nme, 0), MSZ_F8);
      ILM_IRESULT(curilm) =
          ad3ili(IL_LDDP, ad3ili(IL_AADD, res_addr, ad_aconi(8), 0),
                 addnme(NT_MEM, NOSYM, res_nme, 8), MSZ_F8);
      ILM_RESTYPE(curilm) = ILM_ISDCMPLX;
    }
  }

  end_ainfo(&ainfo);
}

/**
   \param ext        name of routine to call
   \param res_dtype  function return type

   Same as exp_qjsr() except that if the result is complex, its pointer argument
   is passed as the last argument instead of the first argument.  This is
   necessary to keep double arguments properly aligned on the stack.

   For now this only works for complex/double complex ILMs which are QJSRs in
   the "standard" fortran.
 */
void
exp_zqjsr(char *ext, DTYPE res_dtype, ILM *ilmp, int curilm)
{
  int nargs;
  int ililnk;  /* ili link */
  int ilix;    /* ili pointer */
  ILM *ilmlnk; /* current ILM operand */
  int ilm1;
  int i;      /* temps */
  static ainfo_t ainfo;
  SPTR res; /* sptr of function result temporary */
  int res_addr;
  int res_nme;
  int extsym;

  if (DT_ISCMPLX(res_dtype)) {
    res = mkrtemp_cpx_sc(res_dtype, expb.sc);
    res_addr = ad_acon(res, 0);
    res_nme = addnme(NT_VAR, res, 0, 0);
    ADDRTKNP(res, 1);
  } else {
    interr("exp_zqjsr, illegal dtype", res_dtype, ERR_Severe);
    return;
  }
  nargs = ilms[ILM_OPC(ilmp)].oprs;
  extsym = mkfunc(ext);
  init_ainfo(&ainfo);
  arg_ar(res_addr, &ainfo, 0);

  i = nargs;
  while (nargs--) {
    ilm1 = ILM_OPND(ilmp, i);
    ilmlnk = (ILM *)(ilmb.ilm_base + ilm1); /* ith operand */
    switch (ILM_RESTYPE(ilm1)) {
    case ILM_ISCHAR:
      interr("exp_zqjsr: char arg not allowed", ilm1, ERR_Severe);
      break;
    case ILM_ISCMPLX:
      arg_sp(ILM_IRESULT(ilm1), &ainfo);
      arg_sp(ILM_RRESULT(ilm1), &ainfo);
      break;
    case ILM_ISDCMPLX:
      arg_dp(ILM_IRESULT(ilm1), &ainfo);
      arg_dp(ILM_RRESULT(ilm1), &ainfo);
      break;
    default:
      ilix = ILM_RESULT(ilm1);
      switch (IL_RES(ILI_OPC(ilix))) {
      case ILIA_IR:
        arg_ir(ilix, &ainfo);
        break;
      case ILIA_AR:
        arg_ar(ilix, &ainfo, 0);
        break;
      case ILIA_SP:
        arg_sp(ilix, &ainfo);
        break;
      case ILIA_DP:
        arg_dp(ilix, &ainfo);
        break;
      case ILIA_KR:
        arg_kr(ilix, &ainfo);
        break;
#ifdef ILIA_CS
      case ILIA_CS:
        ilix = ad1ili(IL_SCMPLX2IMAG, ILM_RESULT(ilm1));
        arg_sp(ilix, &ainfo);
        ilix = ad1ili(IL_SCMPLX2REAL, ILM_RESULT(ilm1));
        arg_sp(ilix, &ainfo);
        break;
      case ILIA_CD:
        ilix = ad1ili(IL_DCMPLX2IMAG, ILM_RESULT(ilm1));
        arg_dp(ilix, &ainfo);
        ilix = ad1ili(IL_DCMPLX2REAL, ILM_RESULT(ilm1));
        arg_dp(ilix, &ainfo);
        break;
#endif
      default:
        interr("exp_zqjsr: ili ret type not cased", ilix, ERR_Severe);
        break;
      }
    }
    i--;
  } /* for each arg */

  ililnk = ad2ili(IL_JSR, extsym, ainfo.lnk);
  iltb.callfg = 1;
  chk_block(ililnk);

  if (res_dtype == DT_CMPLX) {
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDSCMPLX, res_addr, res_nme, MSZ_F8);
    } else {

      ILM_RRESULT(curilm) =
          ad3ili(IL_LDSP, res_addr, addnme(NT_MEM, SPTR_NULL, res_nme, 0), MSZ_F4);
      ILM_IRESULT(curilm) =
          ad3ili(IL_LDSP, ad3ili(IL_AADD, res_addr, ad_aconi(4), 0),
                 addnme(NT_MEM, NOSYM, res_nme, 4), MSZ_F4);
      ILM_RESTYPE(curilm) = ILM_ISCMPLX;
    }
  } else {
    if (XBIT(70, 0x40000000)) {
      ILM_RESULT(curilm) = ad3ili(IL_LDDCMPLX, res_addr, res_nme, MSZ_F16);
    } else {
      ILM_RRESULT(curilm) =
          ad3ili(IL_LDDP, res_addr, addnme(NT_MEM, SPTR_NULL, res_nme, 0), MSZ_F8);
      ILM_IRESULT(curilm) = ad3ili(IL_LDDP, ad3ili(IL_AADD, res_addr, ad_aconi(8), 0),
                 addnme(NT_MEM, NOSYM, res_nme, 8), MSZ_F8);
      ILM_RESTYPE(curilm) = ILM_ISDCMPLX;
    }
  }

  end_ainfo(&ainfo);
}

static void
arg_ir(int ilix, ainfo_t *ap)
{
  ilix = sel_iconv(ilix, 0);
  ap->lnk = ad2ili(IL_ARGIR, ilix, ap->lnk);
}

static void
arg_kr(int ilix, ainfo_t *ap)
{
  ilix = sel_iconv(ilix, 1);
  ap->lnk = ad2ili(IL_ARGKR, ilix, ap->lnk);
}

static void
arg_ar(int ilix, ainfo_t *ap, int dtype)
{
  ap->lnk = ad3ili(IL_ARGAR, ilix, ap->lnk, dtype);
}

static void
arg_sp(int ilix, ainfo_t *ap)
{
  ap->lnk = ad2ili(IL_ARGSP, ilix, ap->lnk);
}

static void
arg_dp(int ilix, ainfo_t *ap)
{
  ap->lnk = ad2ili(IL_ARGDP, ilix, ap->lnk);
}

#ifdef TARGET_SUPPORTS_QUADFP
static void arg_qp(int ilix, ainfo_t *ap)
{
  ap->lnk = ad2ili(IL_ARGQP, ilix, ap->lnk);
}
#endif

static void
arg_charlen(int ilix, ainfo_t *ap)
{
  if (IL_RES(ILI_OPC(ilix)) != ILIA_KR)
    arg_ir(ilix, ap);
  else
    arg_kr(ilix, ap);
}

static void
arg_length(STRDESC *str, ainfo_t *ap)
{
  if (!XBIT(125, 0x40000))
    arg_kr(getstrlen64(str), ap);
  else
    arg_ir(getstrlen(str), ap);
}

/***************************************************************/

/** Expand an smove ILM.
    \param destilm: ilm of receiving struct/union
    \param srcilm:  ilm of sending struct/union
    \param dtype: data type of struct/union
  */
void
expand_smove(int destilm, int srcilm, DTYPE dtype)
{
  int dest_nme;  /* names entry				*/
  int src_nme;   /* names entry				*/
  int dest_addr; /* pointer to ili for destination addr	*/
  int src_addr;  /* pointer to ili for source addr	*/

  dest_nme = NME_OF(destilm);
  src_nme = NME_OF(srcilm);
  if (flg.opt > 1) {
    loc_of(dest_nme); /* implicit LOC          */
    loc_of(src_nme);
  }
  dest_addr = ILI_OF(destilm);
  src_addr = ILI_OF(srcilm);
  if (USE_GSMOVE) {
    int ilix;
    ilix = ad5ili(IL_GSMOVE, src_addr, dest_addr, src_nme, dest_nme, dtype);
    chk_block(ilix);
  } else {
    _exp_smove(dest_nme, src_nme, dest_addr, src_addr, dtype);
  }
}

/** \brief Transform the GSMOVE ILI created by expand_smove()
 */
void
exp_remove_gsmove(void)
{
  int bihx, iltx, ilix;
  p_chk_block = gsmove_chk_block;
  for (bihx = gbl.entbih; bihx; bihx = BIH_NEXT(bihx)) {
    int next_ilt;
    bool any_gsmove = false;
    rdilts(bihx);
    for (iltx = ILT_NEXT(0); iltx;) {
      next_ilt = ILT_NEXT(iltx);
      ilix = ILT_ILIP(iltx);
      if (ILI_OPC(ilix) == IL_GSMOVE) {
        int src_addr = ILI_OPND(ilix, 1);
        int dest_addr = ILI_OPND(ilix, 2);
        int src_nme = ILI_OPND(ilix, 3);
        int dest_nme = ILI_OPND(ilix, 4);
        DTYPE dtype = ILI_DTyOPND(ilix, 5);
        any_gsmove = true;
        gsmove_ilt = iltx;
        _exp_smove(dest_nme, src_nme, dest_addr, src_addr, dtype);
        ILT_NEXT(gsmove_ilt) = next_ilt;
        ILT_PREV(next_ilt) = gsmove_ilt;
        delilt(iltx);
      }
      iltx = next_ilt;
    }
    wrilts(bihx);
    if (DBGBIT(10, 2) && any_gsmove) {
      fprintf(gbl.dbgfil, "\n***** After remove gsmove *****\n");
      dump_one_block(gbl.dbgfil, bihx, NULL);
    }
  }
  p_chk_block = chk_block;
}

static void
_exp_smove(int dest_nme, int src_nme, int dest_addr, int src_addr, DTYPE dtype)
{
  ISZ_T n; /* number of bytes left to copy		*/
  int i;
  INT offset; /* number of bytes from begin addr 	*/

  n = size_of(dtype);
  if (0 && !XBIT(2, 0x1000000)) {
    chk_block(ad5ili(IL_SMOVEJ, src_addr, dest_addr, src_nme, dest_nme, n));
    smove_flag = 1; /* structure move in this function */
    return;
  }
  offset = 0;

/*  for large structs, copy as much as possible using an smovl/smoveq instr: */
#define SMOVE_CHUNK 8
#define TEST_BOUND 96
  if (n > TEST_BOUND) {

    if (XBIT(2, 0x200000)) {
      p_chk_block(ad4ili(IL_SMOVE, src_addr, dest_addr,
                         ad_aconi(n / SMOVE_CHUNK), dest_nme));
    } else {
      p_chk_block(ad5ili(IL_SMOVEJ, src_addr, dest_addr, src_nme,
                         dest_nme, n));
    }
    smove_flag = 1; /* structure move in this function */
    offset = (n / SMOVE_CHUNK) * SMOVE_CHUNK;
    n = n - offset;
    if (n > 0) {
      /* add CSE's to prevent addresses from being recalculated: */
      src_addr = ad1ili(IL_CSEAR, src_addr);
      dest_addr = ad1ili(IL_CSEAR, dest_addr);
    }
  }

  /*  generate loads and stores for the parts of the structs remaining: */

#define START_AT 0 /*  loop for skip size == 8, 4, 2, 1  */
  for (i = START_AT; i < 4; i++) {
    static struct {
      short siz;
      short skip;
    } info[4] = {{MSZ_I8, 8}, {MSZ_WORD, 4}, {MSZ_UHWORD, 2}, {MSZ_UBYTE, 1}};

    int siz = info[i].siz;
    int skip = info[i].skip;

    while (n >= skip) {
      int ilip, ilix; /* temporary ili pointers */

      /*  add load and store ili:  */

      ilip = ad_aconi(offset);
      ilix = ad3ili(IL_AADD, src_addr, ilip, 0);
      if (siz == MSZ_I8)
        ilix = ad3ili(IL_LDKR, ilix, src_nme, siz);
      else
        ilix = ad3ili(IL_LD, ilix, src_nme, siz);
      ilip = ad3ili(IL_AADD, dest_addr, ilip, 0);
      if (siz == MSZ_I8)
        ilip = ad4ili(IL_STKR, ilix, ilip, dest_nme, siz);
      else
        ilip = ad4ili(IL_ST, ilix, ilip, dest_nme, siz);
      p_chk_block(ilip);

      offset += skip;
      n -= skip;
      if (n > 0) {
        src_addr = ad1ili(IL_CSEAR, src_addr);
        dest_addr = ad1ili(IL_CSEAR, dest_addr);
      }
    }
  }
}

/***************************************************************/

/**
   \param to    ilm of receiving struct/union
   \param from  ilm of sending struct/union
   \param dtype data type of struct/union
 */
void
exp_szero(ILM *ilmp, int curilm, int to, int from, int dtype)
{
  int nme;   /* names entry				*/
  int addr,  /* address ili where value stored	*/
      expr,  /* ili of value being stored		*/
      sym;   /* ST item				*/
  int tmp;

  nme = NME_OF(to);
  addr = ILI_OF(to);
  expr = ILI_OF(from);
  loc_of(nme);
  tmp = ad1ili(IL_NULL, 0);
  tmp = ad3ili(IL_ARGAR, addr, tmp, 0);
  tmp = ad2ili(IL_ARGIR, expr, tmp);
  sym = mkfunc("__c_bzero");
  chk_block(ad2ili(IL_JSR, sym, tmp)); /* temporary */
}

void
exp_fstring(ILM_OP opc, ILM *ilmp, int curilm)
{
  int ili1;
  int sym;
  int op1, op2;
  int tmp;
  INT val[2];
  int addr, highsub, lowsub;
  int hsubili, lsubili;
  bool any_kr;
  STRDESC *str1, *str2;
  int ilm1;

  switch (opc) {
  case IM_ICHAR: /* char to integer */
    tmp = MSZ_BYTE;
    FLANG_FALLTHROUGH;
  case IM_INCHAR: /* nchar to integer */
    if (opc == IM_INCHAR)
      tmp = MSZ_UHWORD;
    ilm1 = ILM_OPND(ilmp, 1);
    str1 = getstr(ilm1);
    if (!str1->next)
      ili1 = ILI_OF(ilm1); /* char result */
    else {
      if (str1->liscon && str1->lval >= 1) {
        ;
      } else {
        str1 = storechartmp(str1, ILM_MXLEN(ilm1), ILM_CLEN(ilm1));
      }
      ili1 = getstraddr(str1);
    }
    if (ILI_OPC(ili1) == IL_ACON && opc != IM_INCHAR &&
        STYPEG(sym = CONVAL1G(ILI_OPND(ili1, 1))) == ST_CONST) {
/* constant char str */
#if DEBUG
      assert(DTY(DTYPEG(sym)) == TY_CHAR, "non char op of ICHAR", ili1,
             ERR_Severe);
#endif
      op1 = CONVAL1G(sym);               /* names area idx containing string */
      op2 = CONVAL2G(ILI_OPND(ili1, 1)); /* offset */
      tmp = stb.n_base[op1 + op2] & 0xff;
      ILM_RESULT(curilm) = ad_icon(tmp);
    } else
      ILM_RESULT(curilm) = ad3ili(IL_LD, ili1, NME_STR1, tmp);
    return;

  case IM_CHAR: /* integer to char */
    val[0] = getchartmp(ad_icon(1));
    val[1] = 0;
    tmp = getcon(val, DT_ADDR);
    op1 = ILI_OF(ILM_OPND(ilmp, 1));
    if (IL_RES(ILI_OPC(op1)) == ILIA_KR)
      op1 = ad1ili(IL_KIMV, op1);
    ili1 = ad4ili(IL_ST, op1, ad1ili(IL_ACON, tmp), NME_STR1, MSZ_BYTE);
    chk_block(ili1);
    ILM_RESULT(curilm) = ad1ili(IL_ACON, tmp);
    ILM_RESTYPE(curilm) = ILM_ISCHAR;
    if (CHARLEN_64BIT) {
      ILM_CLEN(curilm) = ILM_MXLEN(curilm) = ad_kconi(1);
    } else {
      ILM_CLEN(curilm) = ILM_MXLEN(curilm) = ad_icon(1);
    }
    return;

  case IM_NCHAR: /* integer to kanji char */
    val[0] = getchartmp(ad_icon(2));
    val[1] = 0;
    tmp = getcon(val, DT_ADDR);
    ili1 = ad4ili(IL_ST, ILI_OF(ILM_OPND(ilmp, 1)), ad1ili(IL_ACON, tmp),
                  NME_STR1, MSZ_UHWORD);
    chk_block(ili1);
    ILM_RESULT(curilm) = ad1ili(IL_ACON, tmp);
    ILM_RESTYPE(curilm) = ILM_ISCHAR;
    if (CHARLEN_64BIT) {
      ILM_CLEN(curilm) = ILM_MXLEN(curilm) = ad_kconi(1);
    } else {
      ILM_CLEN(curilm) = ILM_MXLEN(curilm) = ad_icon(1);
    }
    return;

  case IM_SST: /* string store */
  case IM_NSST:
    str1 = getstr(ILM_OPND(ilmp, 1));
    str2 = getstr(ILM_OPND(ilmp, 2));
#if DEBUG
    assert(str1->cnt == 1, "string store into concat", curilm, ERR_Severe);
#endif
    /* special case string store into single char */
    if (strislen1(str1)) {
      int tmp = MSZ_BYTE;
      if (strislen0(str2)) {
        if (opc != IM_NSST) {
          str2 = getstrconst(" ", 1);
        } else {
          goto bldfcall;
        }
      }
      if (opc == IM_NSST)
        tmp = MSZ_UHWORD;
      ili1 = ad3ili(IL_LD, getstraddr(str2), NME_STR1, tmp);
      ili1 = ad4ili(IL_ST, ili1, getstraddr(str1), NME_STR1, tmp);
      chk_block(ili1);
      return;
    }
  bldfcall:
    /* build function call */
    ili1 = exp_strcpy(str1, str2);
    iltb.callfg = 1;
    chk_block(ili1);
    return;

  case IM_SPSEUDOST: /* string pseudo store */
  case IM_NSPSEUDOST:
    /* for now, just force the character expression into a temporary
     * and pass on the information for the temp.
     */
    str2 = getstr(ILM_OPND(ilmp, 2));
    ili1 = ad_icon(ILM_OPND(ilmp, 1));
    str1 = storechartmp(str2, ili1, ili1);
    ILM_RESULT(curilm) = getstraddr(str1);
    ILM_RESTYPE(curilm) = ILM_ISCHAR;
    if (CHARLEN_64BIT) {
      ILM_CLEN(curilm) = ILM_MXLEN(curilm) = sel_iconv(ili1, 1);
    } else {
      ILM_CLEN(curilm) = ILM_MXLEN(curilm) = ili1;
    }
    return;

  case IM_LEN: /* length of string */
  case IM_NLEN:
    ili1 = ILM_CLEN(ILM_OPND(ilmp, 1));
    if (IL_RES(ILI_OPC(ili1)) == ILIA_KR)
      ili1 = ad1ili(IL_KIMV, ili1);
    ILM_RESULT(curilm) = ili1;
#if DEBUG
    assert(ILM_RESULT(curilm) != 0, "IM_LEN:len ili 0", curilm, ERR_Severe);
#endif
    return;
  case IM_KLEN: /* length of string */
    ili1 = ILM_CLEN(ILM_OPND(ilmp, 1));
    if (IL_RES(ILI_OPC(ili1)) != ILIA_KR)
      ili1 = ad1ili(IL_IKMV, ili1);
    ILM_RESULT(curilm) = ili1;
    return;

  case IM_SUBS: /* substring */
  case IM_NSUBS:
    /*-
     * addr = addr + lowsub - 1
     * len = highsub - lowsub + 1
     * maxlen = len (if const.) else maxlen
     */
    addr = ILM_OPND(ilmp, 1);
    lowsub = ILM_OPND(ilmp, 2);
    highsub = ILM_OPND(ilmp, 3);
    lsubili = ILI_OF(lowsub);
    hsubili = ILI_OF(highsub);

    if (CHARLEN_64BIT)
      any_kr = true;
    else
      any_kr = (IL_RES(ILI_OPC(lsubili)) == ILIA_KR) ||
               (IL_RES(ILI_OPC(hsubili)) == ILIA_KR);
    if (any_kr) {
      if (IL_RES(ILI_OPC(lsubili)) != ILIA_KR)
        lsubili = ad1ili(IL_IKMV, lsubili);
      if (IL_RES(ILI_OPC(hsubili)) != ILIA_KR)
        hsubili = ad1ili(IL_IKMV, hsubili);
      ili1 = ad2ili(IL_KSUB, lsubili, ad_kconi(1));
      if (opc == IM_NSUBS)
        ili1 = ad2ili(IL_KMUL, ili1, ad_kconi(2));
      ili1 = ad1ili(IL_KAMV, ili1);
      ILI_OF(curilm) = ad3ili(IL_AADD, ILI_OF(addr), ili1, 0);
      ili1 = ad2ili(IL_KSUB, hsubili, lsubili);
      ili1 = ad2ili(IL_KADD, ili1, ad_kconi(1));
      if (!CHARLEN_64BIT)
        ili1 = ad1ili(IL_KIMV, ili1);
    } else {
      ili1 = ad2ili(IL_ISUB, lsubili, ad_icon(1));
      if (opc == IM_NSUBS)
        ili1 = ad2ili(IL_IMUL, ili1, ad_icon(2));
      ili1 = ad1ili(IL_IAMV, ili1);
      ILI_OF(curilm) = ad3ili(IL_AADD, ILI_OF(addr), ili1, 0);
      ili1 = ad2ili(IL_ISUB, hsubili, lsubili);
      ili1 = ad2ili(IL_IADD, ili1, ad_icon(1));
      if (CHARLEN_64BIT)
        ili1 = sel_iconv(ili1, 1);
    }

    if (IL_TYPE(ILI_OPC(ili1)) == ILTY_CONS) {
      if (get_isz_cval(ILI_OPND(ili1, 1)) < 0)
        ili1 = ad_icon(0);
      if (CHARLEN_64BIT)
        ili1 = sel_iconv(ili1, 1);
      ILM_CLEN(curilm) = ili1;
      ILM_MXLEN(curilm) = ili1;
    } else {
      if (CHARLEN_64BIT) {
        ILM_CLEN(curilm) = ad2ili(IL_KMAX, ili1, ad_kconi(0));
        if (ILM_MXLEN(addr))
          ILM_MXLEN(curilm) = sel_iconv(ILM_MXLEN(addr), 1);
        else
          ILM_MXLEN(curilm) = 0;
      } else {
        ILM_CLEN(curilm) = ad2ili(IL_IMAX, ili1, ad_icon(0));
        ILM_MXLEN(curilm) = ILM_MXLEN(addr);
      }
    }
    ILM_RESTYPE(curilm) = ILM_ISCHAR;
    return;

  case IM_SCAT: /* concatenation */
  case IM_NSCAT:
    op1 = ILM_OPND(ilmp, 1);
    op2 = ILM_OPND(ilmp, 2);
    if (CHARLEN_64BIT) {
      ILM_CLEN(curilm) = ad2ili(IL_KADD, sel_iconv(ILM_CLEN(op1), 1),
                                sel_iconv(ILM_CLEN(op2), 1));
    } else {
      ILM_CLEN(curilm) =
          ad2ili(IL_IADD, ILM_CLEN(op1), ILM_CLEN(op2));
    }
    if (ILM_MXLEN(op1) && ILM_MXLEN(op2)) {
      if (CHARLEN_64BIT) {
        ILM_MXLEN(curilm) = ad2ili(IL_KADD, sel_iconv(ILM_MXLEN(op1), 1),
                                   sel_iconv(ILM_MXLEN(op2), 1));
      } else {
        ILM_MXLEN(curilm) =
            ad2ili(IL_IADD, ILM_MXLEN(op1), ILM_MXLEN(op2));
      }
    } else {
      ILM_MXLEN(curilm) = 0;
    }
    ILM_RESULT(curilm) = 0; /* FIXME? */
    ILM_RESTYPE(curilm) = ILM_ISCHAR;
    return;

  case IM_SCMP:
  case IM_NSCMP:
    /* set indicator for the referencing relational ILM --
     * indicates a string compare is handled by calling
     * ftn_strcmp.
     */
    ILM_RESTYPE(curilm) = ILM_ISCHAR;
    FLANG_FALLTHROUGH;
  case IM_INDEX:
  case IM_KINDEX:
  case IM_NINDEX:
    /* if either arg is SCAT generate tmp and store into it */
    str1 = getstr(ILM_OPND(ilmp, 1));
    if (str1->next)
      str1 = storechartmp(str1, ILM_MXLEN(ILM_OPND(ilmp, 1)),
                          ILM_CLEN(ILM_OPND(ilmp, 1)));
    str2 = getstr(ILM_OPND(ilmp, 2));
    if (str2->next)
      str2 = storechartmp(str2, ILM_MXLEN(ILM_OPND(ilmp, 2)),
                          ILM_CLEN(ILM_OPND(ilmp, 2)));
    if (opc == IM_SCMP) {
      char *p1, *p2;

      p1 = getcharconst(str1);
      p2 = getcharconst(str2);
      if (p1 != NULL & p2 != NULL) {
        val[0] = ftn_strcmp(p1, p2, str1->lval, str2->lval);
        ILM_RESULT(curilm) = ad_icon(val[0]);
        return;
      }
      if (strislen1(str1) && strislen1(str2)) {
        /* special case str cmp of single chars: generate a ICMP ili
         * with a load the two character items.  This ili is save as
         * the result of the SCMP and will be picked up as a special
         * case by the relational ILM referencing this ILM (due to the
         * ILM_RESTYPE of ILM_ISCHAR).
         */
        op1 = ad3ili(IL_LD, getstraddr(str1), NME_STR1, MSZ_BYTE);
        op2 = ad3ili(IL_LD, getstraddr(str2), NME_STR1, MSZ_BYTE);
        ILM_RESULT(curilm) = ad3ili(IL_ICMP, op1, op2, CC_EQ);
        return;
      }
    }
    /* gen call to strcmp or stridx routine */
    iltb.callfg = 1;
    ili1 = exp_strx(opc, str1, str2);
    ILM_RESULT(curilm) = ili1;
    return;

  default:
    interr("unrecognized fstr ILM", opc, ERR_Severe);
    break;
  }
}

static int
exp_strx(int opc, STRDESC *str1, STRDESC *str2)
{
  int sym;
  int ili1;
  const char *str_index_nm;
  const char *nstr_index_nm;
  const char *strcmp_nm;
  const char *nstrcmp_nm;
  const char *ftn_str_kindex_nm;

  if (CHARLEN_64BIT) {
    str_index_nm = mkRteRtnNm(RTE_str_index_klen);
    nstr_index_nm = mkRteRtnNm(RTE_nstr_index_klen);
    strcmp_nm = mkRteRtnNm(RTE_strcmp_klen);
    nstrcmp_nm = mkRteRtnNm(RTE_nstrcmp_klen);
    ftn_str_kindex_nm = "ftn_str_kindex_klen";
  } else {
    str_index_nm = mkRteRtnNm(RTE_str_index);
    nstr_index_nm = mkRteRtnNm(RTE_nstr_index);
    strcmp_nm = mkRteRtnNm(RTE_strcmp);
    nstrcmp_nm = mkRteRtnNm(RTE_nstrcmp);
    ftn_str_kindex_nm = "ftn_str_kindex";
  }

  if (str1->dtype == TY_NCHAR)
    sym = frte_func(mkfunc, opc == IM_NSCMP ? nstrcmp_nm : nstr_index_nm);
  else if (opc == IM_KINDEX)
    sym = mkfunc(ftn_str_kindex_nm);
  else
    sym = frte_func(mkfunc, opc == IM_SCMP ? strcmp_nm : str_index_nm);
  ili1 = ad1ili(IL_NULL, 0);
  /* str1 & str2 lens */
  if (!XBIT(125, 0x40000)) {
    ili1 = ad2ili(IL_ARGKR, getstrlen64(str2), ili1);
    ili1 = ad2ili(IL_ARGKR, getstrlen64(str1), ili1);
  } else {
    ili1 = ad2ili(IL_ARGIR, getstrlen(str2), ili1);
    ili1 = ad2ili(IL_ARGIR, getstrlen(str1), ili1);
  }
  /* str1 & str2 addrs */
  ili1 = ad3ili(IL_ARGAR, getstraddr(str2), ili1, 0);
  ili1 = ad3ili(IL_ARGAR, getstraddr(str1), ili1, 0);
  /* JSR */
  ili1 = ad2ili(IL_JSR, sym, ili1);
  if (opc == IM_KINDEX)
    ili1 = ad2ili(IL_DFRKR, ili1, KR_RETVAL);
  else
    ili1 = ad2ili(IL_DFRIR, ili1, IR_RETVAL);
  return ili1;
}

static void
from_addr_and_length(STRDESC *s, ainfo_t *ainfo_ptr)
{
  if (s->next)
    from_addr_and_length(s->next, ainfo_ptr);
  arg_length(s, ainfo_ptr);
  arg_ar(getstraddr(s), ainfo_ptr, 0);
}

static int
exp_strcpy(STRDESC *str1, STRDESC *str2)
{
  int sym;
  int n;
  int ili1;
  static ainfo_t ainfo;
  const char *str_copy_nm;
  const char *nstr_copy_nm;
  if (CHARLEN_64BIT) {
    str_copy_nm = mkRteRtnNm(RTE_str_copy_klen);
    nstr_copy_nm = mkRteRtnNm(RTE_nstr_copy_klen);
  } else {
    str_copy_nm = mkRteRtnNm(RTE_str_copy);
    nstr_copy_nm = mkRteRtnNm(RTE_nstr_copy);
  }

  init_ainfo(&ainfo);

  if (str1->dtype == TY_CHAR) {
    if (!strovlp(str1, str2)) {
/*
 * single source, no overlap
 */
#define STR_MOVE_THRESH 16
      if (!XBIT(125, 0x800) && str1->liscon && str2->liscon &&
          str1->lval <= STR_MOVE_THRESH) {
        /*
         * perform a 'block move' of the rhs to the lhs -- the move
         * will move a combination of 8 (64-bit only) 4, 2, and 1
         * bytes.  Note that this same code appears in the 32-bit
         * and 64-bit compilers, thus the check of TARGET_X8632.
         */
        if (str1->lval > str2->lval) {
          char *p2;
          p2 = getcharconst(str2);
          if (p2) {
            /*
             * if the rhs is a constant shorter than the lhs,
             * need to create a new constant padded with
             * blanks.  Pad the constant to make its length
             * a multiple of a number specific to the arch
             * (8 for 64-bit, and 4 for 32-bit).
             */
            ISZ_T len;
            ISZ_T md;
            ISZ_T pad;
            char b[STR_MOVE_THRESH + 1];
            char *str;

            str = b;
            len = (ISZ_T)str2->lval;
            while (len-- > 0) {
              *str++ = *p2++;
            }
            md = (8 - (str2->lval & 0x7)) & 0x7;
            if (XBIT(125, 0x1000) || str2->lval + md > str1->lval) {
              pad = str1->lval - str2->lval;
            } else {
              pad = md;
            }
            len = str2->lval + pad;
            while (pad-- > 0) {
              *str++ = ' ';
            }
            str2 = getstrconst(b, len);
          }
        }
        ili1 = block_str_move(str1, str2);
        return ili1;
      }
      sym = frte_func(mkfunc_cncall, mkRteRtnNm(RTE_str_cpy1));

      /* from addr and length */
      arg_length(str2, &ainfo);
      arg_ar(getstraddr(str2), &ainfo, 0);

      /* to addr and length */
      arg_length(str1, &ainfo);
      arg_ar(getstraddr(str1), &ainfo, 0);

      /* JSR */
      ili1 = ad2ili(IL_JSR, sym, ainfo.lnk);
      end_ainfo(&ainfo);
      return ili1;
    }
  }

  if (str1->dtype == TY_NCHAR)
    sym = frte_func(mkfunc, nstr_copy_nm);
  else
    sym = frte_func(mkfunc, str_copy_nm);
  VARARGP(sym, 1);
  n = str2->cnt;

  /* from addrs and lengths, need to recurse */
  from_addr_and_length(str2, &ainfo);

  /* to addr and length */
  arg_length(str1, &ainfo);
  arg_ar(getstraddr(str1), &ainfo, 0);

  arg_ir(ad_icon(n), &ainfo); /* # from strings */
  /* JSR */
  ili1 = ad2ili(IL_JSR, sym, ainfo.lnk);
  end_ainfo(&ainfo);
  return ili1;
}

static int
block_str_move(STRDESC *str1, STRDESC *str2)
{
  int len;
  int bfill;
  int nb;
  ISZ_T off;
  int addr1, addr2;
  int a1, a2;
  int ili1;

  ili1 = 0;
  len = str1->lval;
  if (len <= str2->lval)
    bfill = 0;
  else {
    bfill = len - str2->lval;
    len = str2->lval;
  }
  addr1 = getstraddr(str1);
  addr2 = getstraddr(str2);
  off = 0;
  while (true) {
    if (ili1)
      chk_block(ili1);
    if (len > 7) {
      nb = 8;
    } else if (len > 3) {
      nb = 4;
    } else if (len > 1) {
      nb = 2;
    } else {
      nb = 1;
    }
    a1 = ad3ili(IL_AADD, addr1, ad_aconi(off), 0);
    a2 = ad3ili(IL_AADD, addr2, ad_aconi(off), 0);
    switch (nb) {
    case 8:
      ili1 = ad3ili(IL_LDKR, a2, NME_STR1, MSZ_I8);
      ili1 = ad4ili(IL_STKR, ili1, a1, NME_STR1, MSZ_I8);
      break;
    case 4:
      ili1 = ad3ili(IL_LD, a2, NME_STR1, MSZ_WORD);
      ili1 = ad4ili(IL_ST, ili1, a1, NME_STR1, MSZ_WORD);
      break;
    case 2:
      ili1 = ad3ili(IL_LD, a2, NME_STR1, MSZ_UHWORD);
      ili1 = ad4ili(IL_ST, ili1, a1, NME_STR1, MSZ_UHWORD);
      break;
    default:
      ili1 = ad3ili(IL_LD, a2, NME_STR1, MSZ_BYTE);
      ili1 = ad4ili(IL_ST, ili1, a1, NME_STR1, MSZ_BYTE);
      break;
    }
    len -= nb;
    if (len <= 0)
      break;
    off += nb;
  }
  if (bfill) {
    len = bfill;
    off = str2->lval;
    while (true) {
      if (ili1)
        chk_block(ili1);
      if (len > 7) {
        ili1 = ad_kcon(0x20202020, 0x20202020);
        nb = 8;
      } else if (len > 3) {
        ili1 = ad_icon(0x20202020);
        nb = 4;
      } else if (len > 1) {
        ili1 = ad_icon(0x2020);
        nb = 2;
      } else {
        ili1 = ad_icon(0x20);
        nb = 1;
      }
      a1 = ad3ili(IL_AADD, addr1, ad_aconi(off), 0);
      switch (nb) {
      case 8:
        ili1 = ad4ili(IL_STKR, ili1, a1, NME_STR1, MSZ_I8);
        break;
      case 4:
        ili1 = ad4ili(IL_ST, ili1, a1, NME_STR1, MSZ_WORD);
        break;
      case 2:
        ili1 = ad4ili(IL_ST, ili1, a1, NME_STR1, MSZ_UHWORD);
        break;
      default:
        ili1 = ad4ili(IL_ST, ili1, a1, NME_STR1, MSZ_BYTE);
        break;
      }
      len -= nb;
      if (len <= 0)
        break;
      off += nb;
    }
  }
  return ili1;
}

/** \brief Determine if it's possible that the lhs & rhs of a string/character
 * assignment can overlap.
 *
 * Note that for now, an assumed-size char lhs is not a candidate since its
 * STRDESC is not marked 'asivar'.
 */
static bool
strovlp(STRDESC *lhs, STRDESC *rhs)
{
  int rsym;
  int lsym;

  if (rhs->next != NULL) /* single rhs only */
    return true;
  if (!rhs->aisvar) /* rhs must be simple var or constant */
    return true;
  rsym = CONVAL1G(rhs->aval);
  if (rsym == 0)
    return true;
  if (STYPEG(rsym) == ST_CONST)
    /* constants never overlaps */
    return false;
  if (!lhs->aisvar) /* lhs must be simple var */
    return true;
  lsym = CONVAL1G(lhs->aval);
  if (lsym == 0)
    return true;
  if (lsym != rsym) /* lhs & rhs variables must be different */
    return false;
  return true;
}

static char *
getcharconst(STRDESC *str)
{
  int asym;
  int sym;
  char *p;

  if (!str->aisvar || !str->liscon)
    return NULL;
  asym = str->aval;
  sym = CONVAL1G(asym);
  if (sym == 0 || STYPEG(sym) != ST_CONST)
    return NULL;
  p = stb.n_base + (CONVAL1G(sym) + CONVAL2G(asym));
  return p;
}

/*
 * fortran compare of strings a1 & a2; returns:
 *    0 => strings are the same
 *   -1 => a1 lexically less than a2
 *    1 => a1 lexically greater than a2
 * If the lengths of the strings are not equal, the short string is blank
 * padded.
 */
static int
_fstrcmp(char *a1, char *a2, int len)
{
  while (len > 0) {
    if (*a1 != *a2) {
      if (*a1 > *a2)
        return 1;
      return -1;
    }
    ++a1;
    ++a2;
    --len;
  }
  return 0;
}

static int
ftn_strcmp(char *a1, char *a2, int a1_len, int a2_len)
{
  int retv;

  if (a1_len == a2_len)
    return _fstrcmp(a1, a2, a1_len);

  if (a1_len > a2_len) {
    /* first compare the first a2_len characters of the strings */
    retv = _fstrcmp(a1, a2, a2_len);
    if (retv)
      return retv;
    a1 += a2_len;
    a1_len -= a2_len;
    /*
     * if the last (a1_len - a2_len) characters of a1 are blank, then the
     * strings are equal; otherwise, compare the first non-blank char. to
     * blank
     */
    while (a1_len > 0) {
      if (*a1 != ' ') {
        if (*a1 > ' ')
          return 1;
        return -1;
      }
      ++a1;
      --a1_len;
    }
  } else {
    /* a2_len > a1_len */
    /* first compare the first a1_len characters of the strings */
    retv = _fstrcmp(a1, a2, a1_len);
    if (retv)
      return retv;
    a2 += a1_len;
    a2_len -= a1_len;
    /*
     * if the last (a2_len - a1_len) characters of a2 are blank, then the
     * strings are equal; otherwise, compare the first non-blank char. to
     * blank
     */
    while (a2_len > 0) {
      if (*a2 != ' ') {
        if (' ' > *a2)
          return 1;
        return -1;
      }
      ++a2;
      --a2_len;
    }
  }
  return 0;
}

/**
   \param ili   max size ili
 */
static int
getchartmp(int ili)
{
  DTYPE dtype;
  SPTR sym = getccsym('T', expb.chartmps++, ST_VAR);
  SCP(sym, expb.sc);

  if (ili && IL_TYPE(ILI_OPC(ili)) == ILTY_CONS)
    dtype = get_type(2, TY_CHAR, CONVAL2G(ILI_OPND(ili, 1)));
  else
    return allochartmp(ili);
  DTYPEP(sym, dtype);
  return sym;
}

/**
   \param lenili   length ili
 */
static SPTR
allochartmp(int lenili)
{
  SPTR sym;
  int sptr1;
  int ili;
  ainfo_t ainfo;
  const char *str_malloc_nm;
  if (CHARLEN_64BIT) {
    str_malloc_nm = mkRteRtnNm(RTE_str_malloc_klen);
  } else {
    str_malloc_nm = mkRteRtnNm(RTE_str_malloc);
  }

  if (allocharhdr == 0) {
    /* create a symbol to represent the head of list of allocated
     * areas created by the run-time (ftn_str_malloc()).  This variable
     * will be initialized in each entry and the list of allocated areas
     * will be freed at the end of each subprogram.
     */
    allocharhdr = getccsym('T', expb.chartmps++, ST_VAR);
    SCP(allocharhdr, SC_LOCAL);
    DTYPEP(allocharhdr, DT_ADDR);
    ADDRTKNP(allocharhdr, 1);
  }
  sym = getccsym('T', expb.chartmps++, ST_VAR);
  SCP(sym, SC_LOCAL);

  init_ainfo(&ainfo);
  /*  space <- ftn_str_malloc(lenili, &allocharhdr) */
  sptr1 = frte_func(mkfunc, str_malloc_nm);
  /***** remember that arguments are in reverse order *****/
  arg_ar(ad_acon(allocharhdr, 0), &ainfo, 0);
  arg_ir(lenili, &ainfo);
  /* JSR */
  DTYPEP(sptr1, DT_ADDR);
  ili = ad2ili(IL_JSR, sptr1, ainfo.lnk);
  ili = ad2ili(IL_DFRAR, ili, AR(0));
  ili = ad3ili(IL_STA, ili, ad_acon(sym, 0), addnme(NT_VAR, sym, 0, 0));
  end_ainfo(&ainfo);
  iltb.callfg = 1;
  chk_block(ili);

  DTYPEP(sym, DT_ADDR);
  return sym;
}

static STRDESC *
getstr(int ilm)
{
  ILM *ilmp;
  int addrili, lenili, opc;
  STRDESC *list1, *list2, *item;

  /* get string descriptor for string ILM */
  ilmp = (ILM *)(ilmb.ilm_base + ilm);
  if (ILM_OPC(ilmp) == IM_SCAT || ILM_OPC(ilmp) == IM_NSCAT) {
    list1 = getstr(ILM_OPND(ilmp, 1));
    list2 = getstr(ILM_OPND(ilmp, 2));
    item = list1;
    list1->cnt += list2->cnt;
    while (list1->next)
      list1 = list1->next;
    list1->next = list2;
    if (ILM_OPC(ilmp) == IM_NSCAT)
      item->dtype = TY_NCHAR;
  } else {
    item = (STRDESC *)getitem(STR_AREA, sizeof(STRDESC));
    addrili = ILM_RESULT(ilm);
    lenili = ILM_CLEN(ilm);
    if (IL_TYPE(ILI_OPC(addrili)) == ILTY_CONS &&
        SCG(CONVAL1G(ILI_OPND(addrili, 1))) != SC_DUMMY) {
      item->aisvar = true;
      item->aval = ILI_OPND(addrili, 1);
    } else {
      item->aisvar = false;
      item->aval = addrili;
    }
    if (IL_TYPE(ILI_OPC(lenili)) == ILTY_CONS) {
      item->liscon = true;
      item->lval = CONVAL2G(ILI_OPND(lenili, 1));
    } else {
      item->liscon = false;
      item->lval = lenili;
    }
    item->next = 0;
    item->cnt = 1;
    item->dtype = TY_CHAR;
    opc = ILM_OPC(ilmp);
    if (opc == IM_NCHAR || opc == IM_NSUBS || opc == IM_NCHFUNC ||
        opc == IM_NSPSEUDOST)
      item->dtype = TY_NCHAR;
    else if ((ilm = getrval(ilm))) { /* returns sptr or 0 */
      DTYPE dtype = DTYPEG(ilm);
      if (DTY(dtype) == TY_ARRAY)
        dtype = DTySeqTyElement(dtype);
      if (DTY(dtype) == TY_NCHAR)
        item->dtype = TY_NCHAR;
    }
  }

  return item;
}

static STRDESC *
getstrconst(const char *str, int len)
{
  SPTR s0;
  STRDESC *item;

  s0 = getstring(str, len);
  item = (STRDESC *)getitem(STR_AREA, sizeof(STRDESC));
  item->aisvar = true;
  item->aval = get_acon(s0, 0);
  item->liscon = true;
  item->lval = len;
  item->next = 0;
  item->cnt = 1;
  item->dtype = TY_CHAR;
  return item;
}

static STRDESC *
storechartmp(STRDESC *str, int mxlenili, int clenili)
{
  INT val[2];
  STRDESC *item;
  int ilix;
  int msz;
  int lenili;

  msz = MSZ_BYTE;
  if (mxlenili)
    lenili = mxlenili;
  else
    lenili = clenili;
  if (str->dtype == TY_NCHAR) {
    ilix = ad_icon(2L);
    lenili = ad2ili(IL_IMUL, ilix, lenili);
    msz = MSZ_UHWORD;
  }
  item = (STRDESC *)getitem(STR_AREA, sizeof(STRDESC));
  val[1] = 0;
  if (mxlenili) {
    val[0] = getchartmp(lenili);
    item->aval = getcon(val, DT_ADDR);
    item->aisvar = true;
  } else {
    SPTR sym = allochartmp(lenili);
    ilix = ad_acon(sym, 0);
    ilix = ad2ili(IL_LDA, ilix, addnme(NT_VAR, sym, 0, 0));
    item->aval = ilix;
    item->aisvar = false;
  }
  if (IL_TYPE(ILI_OPC(clenili)) == ILTY_CONS) {
    item->liscon = true;
    item->lval = CONVAL2G(ILI_OPND(clenili, 1));
  } else {
    item->liscon = false;
    item->lval = clenili;
  }

  item->dtype = str->dtype;
  item->next = 0;
  item->cnt = 1;
  if (strislen1(item)) {
    ilix = ad3ili(IL_LD, getstraddr(str), NME_STR1, msz);
    ilix = ad4ili(IL_ST, ilix, getstraddr(item), NME_STR1, msz);
    chk_block(ilix);
    return (item);
  }
  /* generate call to store str into item */
  iltb.callfg = 1;
  chk_block(exp_strcpy(item, str));
  return (item);
}

/**
 * \brief return ili for character length of passed length dummy.
 */
int
charlen(SPTR sym)
{
  SPTR lensym;
  int addr;

  lensym = CLENG(sym);
  if (!INTERNREFG(lensym) && gbl.internal > 1 && INTERNREFG(sym)) {
    /* Its len is passed by value in aux.curr_entry->display after sym */
    addr = mk_charlen_address(sym);
  } else if (PARREFG(lensym) && PASSBYVALG(lensym) && gbl.outlined) {
    addr = mk_charlen_parref_sptr(sym);
  } else
  {
    addr = mk_address(lensym);
  }
  if (DTYPEG(lensym) == DT_INT8)
    return ad3ili(IL_LDKR, addr, addnme(NT_VAR, lensym, 0, 0), MSZ_I8);
  return ad3ili(IL_LD, addr, addnme(NT_VAR, lensym, 0, 0), MSZ_WORD);
}

/**
 * \brief Return ili for character addr of passed length dummy.
 */
int
charaddr(SPTR sym)
{
  SPTR asym;
  int addr;

  assert(SCG(sym) == SC_DUMMY, "charaddr: sym not dummy", sym, ERR_Severe);
  asym = mk_argasym(sym);
  addr = mk_address(sym);

  /* We already do a load address in mk_address */
  if (INTERNREFG(sym) && gbl.internal > 1)
    return addr;
  if (PARREFG(sym) && SCG(sym) == SC_DUMMY && gbl.outlined)
    return addr;
  return ad2ili(IL_LDA, addr, addnme(NT_VAR, asym, 0, 0));
}

/********************************************************************/

/**
   \param entbih    bih of the entry block
   \param exitbih   bih of the exit block

   Check if this function is a terminal routine (one that does not call any
   other routines).  If so, the necessary changes will be made to the entry and
   exit blocks.  This optimization depends on the target machine and its
   execution environment.  It is appropriate when the target does not have
   instructions to manipulate the stack; multiple instructions have to be
   generated to allocate stack space, manipulate the frame and stack pointers,
   and to check for overflow and underflow.

   When the terminal function optimization is appropriate, the following
   applies:

   1.  exceptions and global registers are not used:
       a.  if the routine is terminal, static space is used in lieu of the
           stack.  ILIs QENTRY and QEXIT are used.
       b.  otherwise, faster entry and exit routines, c_i_qentry and
           c_i_qexit, are used.  The ENTRY ili is modified to locate
           c_i_qentry, and a new EXIT ili locating c_i_qexit is generated.
   2.  otherwise, the ENTRY and EXIT ILIs are left as is.

   When the terminal function optimization is not appropriate, the following
   applies:

   1.  if exceptions and global registers are not used, the faster entry and
       exit routines, c_i_qentry and c_i_qexit, are used.  The ILIs QENTRY and
       QEXIT are used.
   2.  otherwise, the ENTRY and EXIT ILIs are left as is.

   The -q 0 256 switch forces full entry and exit to be used.
   The -q 0 4096 switch forces QENTRY and QEXIT to be used for all
   routines.
 */
void
chk_terminal_func(int entbih, int exitbih)
{
  aux.curr_entry->auto_array = 0;
}

/*------------------------------------------------------------------*/

/**
   \param ir number of integer regs used as arguments
   \param fr number of floating point regs used as arguments

   Perform the necessary adjustments regarding the number of argument registers
   used by a jsr/qsr added after the expand phase and before the optimizer
   (i.e., by the vectorizer).  An argument to the current function must be
   stored in memory if it has been marked by expand as a register argument and
   if its register is used by the jsr/qjr.  Also, the available set of
   arg/scratch registers that can be used as globals by the optimizer must be
   updated.
 */
void
exp_reset_argregs(int ir, int fr)
{
}

/**
 * \brief Create & add an ILT for an ILI when transforming GSMOVE ILI
 */
static void
gsmove_chk_block(int ili)
{
  gsmove_ilt = addilt(gsmove_ilt, ili);
}

/*------------------------------------------------------------------*/

#undef ILM_OPC
#undef ILM_OPND
#define ILM_OPC(i) ilmb.ilm_base[i]
#define ILM_OPND(i, n) ilmb.ilm_base[i + n]
#ifdef __cplusplus
inline SPTR ILM_SymOPND(int i, int n) {
  return static_cast<SPTR>(ILM_OPND(i, n));
}
#else
#define ILM_SymOPND ILM_OPND
#endif

void
AssignAddresses(void)
{
  int opc;
  reset_global_ilm_position();
  do {
    int ilmx, len;
    int numilms = rdilms();
    if (numilms == 0)
      break;
    for (ilmx = 0; ilmx < numilms; ilmx += len) {
      int flen, opnd;
      opc = ILM_OPC(ilmx);
      flen = len = ilms[opc].oprs + 1;
      if (IM_VAR(opc)) {
        len += ILM_OPND(ilmx, 1);
      }
      /* is this a variable reference */
      for (opnd = 1; opnd <= flen; ++opnd) {
        if (IM_OPRFLAG(opc, opnd) == OPR_SYM) {
          SPTR sptr = ILM_SymOPND(ilmx, opnd);
          if (sptr > SPTR_NULL && sptr < stb.stg_avail) {
            switch (STYPEG(sptr)) {
            case ST_CONST:
              sym_is_refd(sptr);
              break;
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
              default:
                break;
              }
              break;
            default:
              break;
            }
          }
        }
      }
    }
  } while (opc != IM_END && opc != IM_ENDF);
  reset_global_ilm_position();
}
