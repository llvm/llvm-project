/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Fortran routines called at the end of semantic processing
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "error.h"
#include "semstk.h"
#include "soc.h"
#include "dinit.h"
#include "machar.h"
#include "state.h"
#include "ast.h"
#include "rte.h"
#include "rtlRtns.h"

static void do_common_blocks(void);
static LOGICAL is_in_currsub(int sptr);
static void expand_common_pointers(int);
static void reorder_common_pointers(int);
static void fix_args(int);
static void check_derived_type_in_comm(int common);

static void do_access(void);
static LOGICAL chk_evar(int);
static void equivalence(int, int);
static void add_socs(int, ISZ_T, ISZ_T);
static void do_nml(void);
static void do_save(void);
static void do_sequence(void);
static void nml_equiv(int socp);
static void dinit_name(int sptr);
static void put_name(int sptr);
static void misc_checks(void);

static void vol_equiv(int socp);

/*  define data used for equivalence processing  */

typedef struct {
  int cmblk;   /* pointer to common block, or 0, or -1 */
  int memlist; /* list of variables in this psect */
} PSECT;

static PSECT *psect_base;
static int psect_num;     /* next psect number to be assigned */
static int psect_size;    /* size of currently allocated psect array */
static LOGICAL in_module; /* gbl.currsub is a MODULE */

/*------------------------------------------------------------------*/
#define NO_PTR XBIT(49, 0x8000)
#define NO_CHARPTR XBIT(58, 0x1)
#define NO_DERIVEDPTR XBIT(58, 0x40000)

/** \brief Increment a type bound procedure's (tbp's) pass object argument
  * position (its INVOBJ field) if we are adding a result variable as the
  * function's first argument.
  *
  * The pass object's argument position of a tbp is stored in the INVOBJ
  * field. We need to increment it when we are about to add the result
  * variable as the first argument in the function (e.g., pointer and
  * allocatable results). This function is passed a symbol table pointer
  * (sptr) to a function. We check to see if there are any tbps that use
  * the function as an implementation (i.e., the RHS of the => in a tbp
  * declaration). If so, we check whether the first argument is already a
  * result variable. If it is not, we increment the pass object argument
  * position (INVOBJ field). If there is already a result variable, we
  * skip it. If a derived type inherits from a type defined in a
  * use associated module, it can have function tbps that already have
  * had their argument lists set-up. That's why we can't just arbitrarily
  * increment the INVOBJ field of a tbp.
  *
  * If the addit field is set, then this function is being called by
  * ipa_semfin(). It handles a special case for IPA in which the result
  * argument may already have been set in the semfin() function during
  * the first compilation, but the INVOBJ has not yet been set.
  *
  * If a tbp has an explicit pass(arg) attribute defined, then
  * we can just update the INVOBJ field by searching the argument list for
  * the specified pass argument.
  *
  * \param sptr is the function symbol to search.
  *
  * \param addit is true when this is a special case for IPA (see the verbose
  * description of this function).
  *
 */
static void
incr_invobj_for_retval_add(int impl_sptr, LOGICAL addit)
{
  int sptr2;

  for (sptr2 = 1; sptr2 < stb.stg_avail; ++sptr2) {
    int bind_sptr;
    if (STYPEG(sptr2) == ST_MEMBER && CLASSG(sptr2) &&
        VTABLEG(sptr2) == impl_sptr && !NOPASSG(sptr2) &&
        (bind_sptr = BINDG(sptr2)) > NOSYM && STYPEG(bind_sptr) == ST_PROC &&
        !INVOBJINCG(bind_sptr)) {
      int invobj = INVOBJG(bind_sptr);
      if (invobj == 0) {
        invobj = find_dummy_position(impl_sptr, PASSG(sptr2));
        if (invobj == 0) {
          if ((addit && PASSG(sptr2) <= NOSYM) || PARAMCTG(impl_sptr) > 0)
            invobj = 1;
        }
      }
      if (invobj > 0 && (PARAMCTG(impl_sptr) < 1 ||
                         !RESULTG(aux.dpdsc_base[DPDSCG(impl_sptr)]))) {
        INVOBJP(bind_sptr, invobj + 1);
        INVOBJINCP(bind_sptr, TRUE);
      }
    }
  }
}

#define USER_GNRIC_OR_OPR(sptr) \
  (sptr > stb.firstusym &&      \
   (STYPEG(sptr) == ST_USERGENERIC || STYPEG(sptr) == ST_OPERATOR))

static void
merge_generics(void)
{
  int sptr;
  int sptr1;
  int sptr_genr_curscope;
  int sptr_alias_currscope;

  if (sem.pgphase != PHASE_CONTAIN && !sem.use_seen)
    return;

  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    sptr_genr_curscope = 0;
    if (!USER_GNRIC_OR_OPR(sptr))
      continue;
    if (test_scope(sptr) == -1)
      continue;
    if (SCOPEG(sptr) == stb.curr_scope) {
      sptr_genr_curscope = sptr;
    }

    /* if there is more that one user generic by this name, then they must be
     * merged into
     * a single generic in the current scope
     */
    sptr_alias_currscope = 0;
    for (sptr1 = first_hash(sptr); sptr1 && sptr1 != NOSYM;
         sptr1 = HASHLKG(sptr1)) {
      if (sptr1 < stb.firstusym)
        continue;
      if (NMPTRG(sptr1) != NMPTRG(sptr))
        continue;
      if (IGNOREG(sptr) || (PRIVATEG(sptr1) && SCOPEG(sptr1) != stb.curr_scope))
        continue;
      if (test_scope(sptr1) == -1)
        continue;

      if (sptr1 == sptr_genr_curscope || sptr1 == sptr)
        continue;
      if (STYPEG(sptr1) == ST_ALIAS && USER_GNRIC_OR_OPR(sptr) &&
          SCOPEG(sptr1) == stb.curr_scope) {
        if (sptr_alias_currscope) {
          /* more than one alias in current scope */
          IGNOREP(sptr1, 1);
        } else {
          sptr_alias_currscope = sptr1; /* alias inserted by do_access */
        }
      }
      if (!USER_GNRIC_OR_OPR(sptr1))
        continue;

      if (!sptr_genr_curscope) {
        /* use the generic in the current scope */
        if (SCOPEG(sptr1) == stb.curr_scope) {
          sptr_genr_curscope = sptr1;
        }
      } else if (SCOPEG(sptr1) == stb.curr_scope &&
                 PRIVATEG(sptr_genr_curscope) && !PRIVATEG(sptr1)) {
        /* if more than one generic in current scope, prefer a non-PRIVATE */
        copy_specifics(sptr_genr_curscope, sptr1);
        IGNOREP(sptr_genr_curscope, 1);
        sptr_genr_curscope = sptr1;
      }

      IGNOREP(sptr, sptr != sptr_genr_curscope);
      IGNOREP(sptr1, sptr1 != sptr_genr_curscope);

      if (!sptr_genr_curscope) {
        sptr_genr_curscope = declsym_newscope(sptr, STYPEG(sptr), DTYPEG(sptr));
      }

      if (sptr != sptr_genr_curscope) {
        copy_specifics(sptr, sptr_genr_curscope);
      }
      if (sptr1 != sptr_genr_curscope) {
        copy_specifics(sptr1, sptr_genr_curscope);
      }

      if (sptr_alias_currscope) {
        SYMLKP(sptr_alias_currscope, sptr_genr_curscope);
        PRIVATEP(sptr_genr_curscope, 0);
      }
    }
  }
}

static void
inject_arg(int func_sptr, int arg_sptr, int position)
{
  int old_args = PARAMCTG(func_sptr);
  int new_dsc = ++aux.dpdsc_avl;

  aux.dpdsc_avl += old_args + 1;
  NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size, aux.dpdsc_avl + 50);
  memcpy(&aux.dpdsc_base[new_dsc], &aux.dpdsc_base[DPDSCG(func_sptr)],
         old_args * sizeof *aux.dpdsc_base);
  memmove(&aux.dpdsc_base[new_dsc + position + 1],
          &aux.dpdsc_base[new_dsc + position],
          (old_args - position) * sizeof *aux.dpdsc_base);
  aux.dpdsc_base[new_dsc + position] = arg_sptr;
  SCP(arg_sptr, SC_DUMMY);
  DPDSCP(func_sptr, new_dsc);
  PARAMCTP(func_sptr, old_args + 1);
}

static LOGICAL
have_class_args_been_fixed_already(int func_sptr)
{
  int dscptr = DPDSCG(func_sptr);
  int count = PARAMCTG(func_sptr);
  int j;

  for (j = 0; j < count; ++j) {
    int arg_sptr = aux.dpdsc_base[dscptr + j];
    if (CLASSG(arg_sptr) && CCSYMG(arg_sptr))
      return TRUE;
  }
  return FALSE;
}

static LOGICAL
add_class_arg_descr_arg(int func_sptr, int arg_sptr, int new_arg_position)
{
  if (!CCSYMG(arg_sptr) && CLASSG(arg_sptr)) {
    if (!needs_descriptor(arg_sptr)) {
      /* add type descriptor argument */
      static int tmp = 0;
      int new_arg_sptr = getccsym_sc('O', tmp++, ST_VAR, SC_DUMMY);
      DTYPE dtype = get_array_dtype(1, astb.bnd.dtype);
      ADD_LWBD(dtype, 0) = 0;
      ADD_LWAST(dtype, 0) = astb.bnd.one;
      ADD_NUMELM(dtype) = ADD_UPBD(dtype, 0) = ADD_UPAST(dtype, 0) =
        mk_isz_cval(get_descriptor_len(0), astb.bnd.dtype);
      CLASSP(new_arg_sptr, 1);
      DTYPEP(new_arg_sptr, dtype);
      inject_arg(func_sptr, new_arg_sptr, new_arg_position);
      PARENTP(arg_sptr, new_arg_sptr);
      if (PARREFG(arg_sptr))
        set_parref_flag2(new_arg_sptr, arg_sptr, 0);
      return TRUE;
    }
    if (!SDSCG(arg_sptr)) {
      /* FS#19541 - create normal descr dummy now */
      int descr_sptr = sym_get_arg_sec(arg_sptr);
      SDSCP(arg_sptr, descr_sptr);
      CCSYMP(descr_sptr, TRUE);
    }
  }
  return FALSE;
}

static void
prepend_func_result_as_first_arg(int func_sptr)
{
  int fval_sptr = FVALG(func_sptr);

  if (fval_sptr > NOSYM && DPDSCG(func_sptr) > 0 &&
      aux.dpdsc_base[DPDSCG(func_sptr) + 0] != fval_sptr) {

    /* Push the function result variable into the argument list as
     * its new first argument.
     */
    incr_invobj_for_retval_add(func_sptr, FALSE);
    inject_arg(func_sptr, fval_sptr, 0 /* first argument position */);

    /* If fix_class_args() has already been run, and if it would have
    * added a type descriptor argument for the new argument that we
    * just prepended to convey the function result (i.e., it's
    * a polymorphic pointer), then we need to create the new argument's
    * type descriptor argument and insert it into the list at the right
    * position.
    */
    if (have_class_args_been_fixed_already(func_sptr)) {
      int last_real_arg_position = PARAMCTG(func_sptr);
      while (--last_real_arg_position > 0) {
        int arg_sptr =
            aux.dpdsc_base[DPDSCG(func_sptr) + last_real_arg_position];
        if (!CLASSG(arg_sptr) || !CCSYMG(arg_sptr))
          break;
      }
      add_class_arg_descr_arg(func_sptr, fval_sptr, last_real_arg_position + 1);
    }
  }
}

/** \brief Finalize semantic processing.
 */
void
semfin(void)
{
  int sptr, dtype;
  int last_lineno;
  INT arg;
  int i;
  int agoto;

  last_lineno = gbl.lineno; /* presumably, line # of the END statement */
  gbl.nowarn = FALSE;       /* warnings may be inhibited for second parse */

  if (sem.which_pass) {
    if (gbl.rutype == RU_PROG)
      flg.recursive = FALSE; /* ensure static locals for the main */
    else if (flg.smp || flg.accmp)
      flg.recursive = TRUE; /* no static locals */
  }
  /* Do not want to go from the contained routine to its module.
   * As a general rule, the SCOPE field of a module routine is
   * set to its ST_ALIAS.  However, there are cases (see fs17256)
   * where its SCOPE field is set directly to its module. */
  if (SCOPEG(gbl.currsub) && STYPEG(SCOPEG(gbl.currsub)) != ST_MODULE)
    push_scope_level(SCOPEG(gbl.currsub), SCOPE_NORMAL);
  else
    push_scope_level(gbl.currsub, SCOPE_NORMAL);

  if (sem.which_pass || IN_MODULE) {
    do_dinit(); /* process dinits which were deferred */
  }

  gbl.lineno = 0;

  in_module = (STYPEG(gbl.currsub) == ST_MODULE);

  gbl.entries =
      (gbl.rutype == RU_BDATA) ? NOSYM : (gbl.currsub ? gbl.currsub : NOSYM);

  if (sem.which_pass) {
#if DEBUG
    if (DBGBIT(3, 1024)) {
      fprintf(gbl.dbgfil, "dscptr area before modification\n");
      for (i = 0; i < aux.dpdsc_avl; i++) {
        arg = aux.dpdsc_base[i];
        fprintf(gbl.dbgfil, "dscptr[%d] = %d  (%s)\n", i, arg,
                (arg ? SYMNAME(arg) : ""));
      }
    }
#endif

    /* walk thru all of the dummy arguments of the entries in the
     * subprogram to fix stypes of the args which were not referenced.
     * Expand the parameter descriptor for an entry which returns a
     * derived type and/or has derived-type arguments.
     */
    for (sptr = gbl.entries; sptr != NOSYM; sptr = SYMLKG(sptr)) {
      ENDLINEP(sptr, last_lineno);
      gbl.lineno = FUNCLINEG(sptr);
      if (gbl.rutype == RU_FUNC) {
        (void)ref_entry(sptr);
      }
      if (STYPEG(sptr) != ST_MODULE)
        fix_args(sptr);
    }

#if DEBUG
    if (DBGBIT(3, 1024)) {
      fprintf(gbl.dbgfil, "dscptr area after modification\n");
      for (i = 0; i < aux.dpdsc_avl; i++) {
        arg = aux.dpdsc_base[i];
        fprintf(gbl.dbgfil, "dscptr[%d] = %d  (%s)\n", i, arg,
                (arg ? SYMNAME(arg) : ""));
      }
    }
#endif

    /* If this is a function subprogram, loop thru entries to check
     * data type and do some stuff for character functions:
     */
    if (gbl.rutype == RU_FUNC) {
      int ent_dtype; /* dtype of ENTRY */

      sptr = gbl.entries;
      dtype = DTYPEG(sptr);
      for (; sptr != NOSYM; sptr = SYMLKG(sptr)) {
        gbl.lineno = FUNCLINEG(sptr);
        ent_dtype = DTYPEG(sptr);
        if (POINTERG(sptr))
          PTRARGP(sptr, 1);
        /*Constraint: A function name must not be declared with an asterisk
         *type-param-value if the function is an internal or module
         *function,array-valued, pointer-valued, or recursive.
         */
        if (ASSUMLENG(sptr) &&
            (POINTERG(sptr) || RECURG(sptr) || DTY(ent_dtype) == TY_ARRAY ||
             gbl.internal > 1)) {
          error(48, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(gbl.entries));
        }
        if (DTYG(dtype) == TY_DERIVED) {
          if (DTYG(ent_dtype) != DTYG(dtype)) {
            error(45, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(gbl.entries));
            continue;
          }
        }
        switch (DTY(dtype)) {
        case TY_ARRAY:
          /*
           * If an array function, all entries must return arrays of the
           * same type and shape; make the temporary the first argument.
           */
          prepend_func_result_as_first_arg(sptr);
          if (DTY(ent_dtype) != TY_ARRAY ||
              DTY(ent_dtype + 1) != DTY(dtype + 1) ||
              !conformable(ent_dtype, dtype))
            error(45, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(gbl.entries));
          gbl.rutype = RU_SUBR;
          SCP(FVALG(sptr), SC_DUMMY);
          STYPEP(FVALG(sptr), ST_ARRAY);
          DTYPEP(sptr, DT_NONE);
          if (ASUMSZG(FVALG(sptr)))
            error(155, 3, gbl.lineno,
                  "Array function result may not be assumed-size -",
                  SYMNAME(sptr));
          break;
        case TY_CHAR:
        case TY_NCHAR: /* kanji */
                       /*
                        * Character Functions must return the same type.
                        */
          if (dtype != ent_dtype)
            error(45, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(gbl.entries));
          if (!POINTERG(sptr) && ADJLENG(FVALG(sptr))) {
            prepend_func_result_as_first_arg(sptr);
            gbl.rutype = RU_SUBR;
            DTYPEP(sptr, DT_NONE);
            SCP(FVALG(sptr), SC_DUMMY);
            break;
          }
          goto pointer_check;
        case TY_DCMPLX:
          if (DTY(ent_dtype) != TY_DCMPLX) {
            error(45, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(gbl.entries));
            break;
          }
          goto pointer_check;
        default:
          if (DTY(ent_dtype) == TY_DCMPLX || DTY(ent_dtype) == TY_CHAR ||
              DTY(ent_dtype) == TY_NCHAR)
            error(45, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(gbl.entries));
        pointer_check:
          STYPEP(FVALG(sptr), ST_VAR);
          if (POINTERG(FVALG(sptr)) || ALLOCATTRG(FVALG(sptr))) {
            /* We convert a pointer-valued function into a subroutine whose
             * first dummy argument is the result now, really late in
             * semantic analysis.
             * Check the attributes of fval instead of the attributes of entry,
             * because only the first entry can get all attributes defined by
             * fval through copy_type_to_entry(semant.c).
             */
            prepend_func_result_as_first_arg(sptr);
            gbl.rutype = RU_SUBR;
            DTYPEP(sptr, DT_NONE);
            SCP(FVALG(sptr), SC_DUMMY);
          }
          break;
        }
      }
    }

    /* Check for undefined labels */

    gbl.lineno = 0;
    agoto = 0;
    for (sptr = sem.flabels; sptr; sptr = SYMLKG(sptr)) {
      int fmt;
      if (!DEFDG(sptr))
        errlabel(113, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      else if ((fmt = FMTPTG(sptr))) {
        if (!DINITG(fmt))
          errlabel(218, 3, gbl.lineno, SYMNAME(sptr), "is not a FORMAT");
        else if (TARGETG(sptr))
          errlabel(218, 3, gbl.lineno, SYMNAME(sptr),
                   "must be a branch target statement");
        if (RFCNTG(sptr))
          REFP(fmt, 1);
        if (ASSNG(sptr)) {
          (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_loc), DT_ADDR);
        }
      } else if (ASSNG(sptr)) {
        agoto++;
        AGOTOP(sptr, agoto);
      }
    }
  } else {
    for (sptr = gbl.entries; sptr != NOSYM; sptr = SYMLKG(sptr)) {
      int dpdsc, paramct, i;
      if (STYPEG(sptr) != ST_MODULE) {
        paramct = PARAMCTG(sptr);
        dpdsc = DPDSCG(sptr);
        for (i = 0; i < paramct; ++i) {
          int arg;
          arg = aux.dpdsc_base[dpdsc + i];
          if (ASSUMSHPG(arg) && !XBIT(54, 2) &&
              !(XBIT(58, 0x400000) && TARGETG(arg))) {
            SDSCS1P(arg, 1);
          }
        }
      }
    }
  }

  do_common_blocks();

  /* Process PUBLIC/PRIVATE data */

  do_access();

  merge_generics();

  /* Process data from EQUIVALENCE statements */

  if (sem.eqvlist != 0)
    do_equiv();

  /* Process data from SAVE statements */

  if (sem.savloc || sem.savall)
    do_save();

  /* Process data from NAMELIST statements */

  do_nml();

  /* Process data from [NO]SEQUENCE statements */

  flg.sequence = TRUE;
  flg.hpf = FALSE;
  do_sequence();

  if (sem.which_pass) {
    /* fixup argument area for array-valued functions */

    for (sptr = aux.list[ST_PROC]; sptr != NOSYM; sptr = SLNKG(sptr)) {
#if DEBUG
      /* aux.list[ST_PROC] must be terminated with NOSYM, not 0 */
      assert(sptr > 0, "semfin: corrupted aux.list[ST_PROC]", sptr, 4);
#endif
      dtype = DTYPEG(sptr);
      if (PARAMCTG(sptr)) {
        fix_args(sptr);
        fix_class_args(sptr);
      }
      if (POINTERG(sptr))
        PTRARGP(sptr, 1);
      if (DTY(dtype) == TY_ARRAY) {
        /*
         * If an array function, all entries must return arrays of the
         * same type and shape; make the temporary the first argument.
         */
        STYPEP(FVALG(sptr), ST_ARRAY);
        prepend_func_result_as_first_arg(sptr);
        FUNCP(sptr, 0);
        if (ASUMSZG(FVALG(sptr)))
          error(155, 3, gbl.lineno,
                "Array function result may not be assumed-size -",
                SYMNAME(sptr));
      } else {
        STYPEP(FVALG(sptr), ST_VAR);
        if (POINTERG(sptr) || ALLOCATTRG(FVALG(sptr)) ||
            allocatable_member(FVALG(sptr)) || ADJLENG(FVALG(sptr))) {
          prepend_func_result_as_first_arg(sptr);
          (void)ref_entry(sptr);
          IGNOREP(FVALG(sptr), TRUE);
          FUNCP(sptr, 0);
          DTYPEP(sptr, DT_NONE);
        }
      }
    }
    /* fixing up procedure pointer dtype that contain interfaces and convert
     * from function to subroutine.
     */
    for (i = 0; i < sem.typroc_avail; i++) {
      int fval;
      int procdt, iface;
      procdt = sem.typroc_base[i];
      iface = DTY(procdt + 2);
      fval = FVALG(iface);
      if (iface && fval) {
        dtype = DTY(procdt + 1); /* result type */
        if (DTY(dtype) == TY_ARRAY || POINTERG(iface) || ALLOCATTRG(fval) ||
            allocatable_member(fval)) {
          if (iface) {
            prepend_func_result_as_first_arg(iface);
            (void)ref_entry(iface);
            IGNOREP(FVALG(iface), TRUE);
            FUNCP(iface, 0);
            DTYPEP(iface, DT_NONE);
          }
          /* insert function result -- there is a space reserved for it */
          DTY(procdt + 3) += 1; /* PARAMCT */
          DTY(procdt + 4) -= 1; /* DPDSC */
          aux.dpdsc_base[DTY(procdt + 4)] = fval;
        }
      }
    }
  }

  misc_checks();

  if (sem.which_pass == 0 && !in_module) {
    df_dinit_end();
  }

  gbl.lineno = last_lineno;
  queue_tbp(0, 0, 0, 0, TBP_COMPLETE_FIN);
  if (sem.which_pass) {
    for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
      fixup_reqgs_ident(sptr);
    }
  }
  pop_scope_level(SCOPE_NORMAL);
}

static void
check_derived_type_in_comm(int common)
{
  SPTR member_sptr;
  if (STYPEG(common) != ST_CMBLK || MODCMNG(common))
    return;
  for (member_sptr = CMEMFG(common); member_sptr > NOSYM;
       member_sptr = SYMLKG(member_sptr)) {
    SPTR dt_sptr;
    DTYPE dtype = DDTG(DTYPEG(member_sptr));
    if (DTY(dtype) != TY_DERIVED )
      continue;
    dt_sptr = get_struct_tag_sptr(dtype);
    if (STYPEG(dt_sptr) == ST_TYPEDEF && BASETYPEG(dt_sptr) > DT_NONE) {
      dtype = BASETYPEG(dt_sptr);
      dt_sptr = get_struct_tag_sptr(dtype);
    }
    if (!CFUNCG(dt_sptr) && !SEQG(dt_sptr))
      error(S_0155_OP1_OP2, ERR_Severe, LINENOG(member_sptr),
            "Derived type shall have the BIND attribute or "
            "the SEQUENCE attribute in COMMON -",
            SYMNAME(member_sptr));
    if (allocatable_member(dt_sptr)) {
      error(S_0155_OP1_OP2, ERR_Severe, LINENOG(member_sptr),
            "Derived type cannot have allocatable attribute in COMMON -",
             SYMNAME(member_sptr));
    } else if (has_init_value(dt_sptr)) {
      error(S_0155_OP1_OP2, ERR_Severe, LINENOG(member_sptr),
            "Derived type cannot have default initialization in COMMON -",
            SYMNAME(member_sptr));
    } else {
    }
  }
}

/*
 * Put pointer member pointer/offset/descriptor into common block.
 * Assign addresses to common block elements and compute size of
 * common blocks:
 */
static void
do_common_blocks(void)
{
  int sptr;

  for (sptr = gbl.cmblks; sptr != NOSYM; sptr = SYMLKG(sptr)) {
    int std_err, member, ssptr;
    ISZ_T size;
    int aln_n = 1;

    if (!XBIT(49, 0x10000000)) {
      expand_common_pointers(sptr);
    } else {
      reorder_common_pointers(sptr);
    }

    check_derived_type_in_comm(sptr);

    for (member = CMEMFG(sptr); member != NOSYM; member = SYMLKG(member)) {
      if (EQVG(member) && SOCPTRG(member)) {
        /* this was already processed, probably part of
         * a module common block, and we are in a contained function */
        int socptr;
        for (socptr = SOCPTRG(member); socptr; socptr = SOC_NEXT(socptr)) {
          int socsptr = SOC_SPTR(socptr);
          if (!EQVG(socsptr)) {
            ISZ_T diff = ADDRESSG(member) - ADDRESSG(socsptr);
            ADDRESSP(member, diff);
            break;
          }
        }
      }
    }
    std_err = 0;
    size = 0;
    for (member = CMEMFG(sptr); member != NOSYM; member = SYMLKG(member)) {
      ISZ_T next_off, msz;
      int addr, dtype;
      const char *errmsg = 0;

      if (EQVG(member))
        continue;
      addr = alignment_of_var(member);
      next_off = size;
      size = ALIGN(size, addr);
      if (!CCSYMG(sptr) && !HCCSYMG(sptr) && next_off != size &&
          sem.which_pass == 1) {
        error(63, ERR_Informational, LINENOG(member), SYMNAME(sptr),
              SYMNAME(member));
      }
      ADDRESSP(member, size);
      REFP(member, 1);
      dtype = DTYPEG(member);
      msz = 0;

      if (STYPEG(member) == ST_ARRAY) {
        /* NEC 301 / tpr 2583
         * Added check for deferred shape array in `if' below.
         * Deferred shape is set for common block members that
         * are aligned or distributed.
         */
        if (ALLOCG(member) && !POINTERG(member) && !HCCSYMG(sptr) &&
            !ADD_DEFER(dtype)) {
          errmsg = "- an allocatable array cannot be in COMMON";
        } else if (ADJARRG(member)) {
          errmsg = "- an adjustable array cannot be in COMMON";
        } else if ((DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) &&
                   ADJLENG(member)) {
          errmsg = "- an adjustable-length character array cannot be in COMMON";
        }
      } else if ((DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) &&
                 ADJLENG(member)) {
        errmsg =
            "- an adjustable-length character variable cannot be in COMMON";
      }
      if (ALLOCATTRG(member)) {
        errmsg = "- an allocatable object cannot be in COMMON";
      }
      if (errmsg) {
        if (is_in_currsub(sptr)) {
          error(84, ERR_Severe, LINENOG(member), SYMNAME(member), errmsg);
        }
        msz = 0;
      } else {
        msz = size_of_var(member);
      }

      size += pad_cmn_mem(member, msz, &aln_n);

      if (DTYG(dtype) == TY_CHAR) {
        std_err |= 1;
      } else if (DTYG(dtype) == TY_NCHAR) {
        std_err |= 4;
      } else {
        std_err |= 2;
      }
      if (VOLG(sptr)) {  /* note: common may not be volatile but */
        VOLP(member, 1); /* a member may */
      }
    }
    for (member = CMEMFG(sptr); member != NOSYM; member = SYMLKG(member)) {
      if (EQVG(member) && SOCPTRG(member)) {
        /* finish up: set address of equivalenced member relative
         * to address of its overlap member */
        int socptr;
        for (socptr = SOCPTRG(member); socptr; socptr = SOC_NEXT(socptr)) {
          int socsptr = SOC_SPTR(socptr);
          if (!EQVG(socsptr)) {
            ISZ_T diff = ADDRESSG(member) + ADDRESSG(socsptr);
            ADDRESSP(member, diff);
            break;
          }
        }
      }
    }
    SIZEP(sptr, size);
    if (sem.savall) {
      SAVEP(sptr, 1);
    }
    if (is_in_currsub(sptr)) {
      if (flg.standard) {
        if (std_err != 1 && std_err != 2) {
          error(182, ERR_Warning, LINENOG(sptr), SYMNAME(sptr), CNULL);
        }
      } else if (std_err & 4 && std_err != 4) {
        error(184, ERR_Warning, LINENOG(sptr), SYMNAME(sptr), CNULL);
      }
      /* check for name conflict between common name and program unit
       * or other entry points */
      for (ssptr = first_hash(sptr); ssptr >= stb.firstusym;
           ssptr = HASHLKG(ssptr)) {
        if (NMPTRG(ssptr) != NMPTRG(sptr))
          continue;
        if (IGNOREG(ssptr))
          continue;
        if (ssptr == gbl.currsub || STYPEG(ssptr) == ST_ENTRY) {
          /* conflict between common block and entry point name */
          error(166, ERR_Severe, LINENOG(sptr), SYMNAME(sptr), CNULL);
        }
      }
    }
  }
}

/* is the scope this symbol the currsub */
static LOGICAL
is_in_currsub(int sptr)
{
  int scope = SCOPEG(sptr);
  while (STYPEG(scope) == ST_ALIAS) {
    scope = SYMLKG(scope);
  }
  return scope == gbl.currsub;
}

static void
expand_common_pointers(int sptr)
{
  /*
   * Expand POINTER members in the common by placing the pointer/offset
   * descriptor with respect to the order of the member's  appearance
   * in the common block -- this is standard f90/f95/f2003 behavior.
   */
  int member;
  int nextmember, lastmember, nextlastmember, firstpointer;

  firstpointer = 0;
  lastmember = 0;
  for (member = CMEMFG(sptr); member != NOSYM;
       lastmember = nextlastmember, member = nextmember) {
    nextlastmember = member;
    nextmember = SYMLKG(member);
    if (STYPEG(member) == ST_IDENT || STYPEG(member) == ST_UNKNOWN)
      STYPEP(member, ST_VAR);

    if (SDSCG(member) == 0 && !F90POINTERG(member) &&
        (POINTERG(member) || ALLOCG(member))) {
      get_static_descriptor(member);
      get_all_descriptors(member);
      SCP(member, SC_BASED);
    }
    if (POINTERG(member)) {
      int ptr, off, sdsc, added;
      added = 0;
      ptr = MIDNUMG(member);
      if (ptr && SCG(ptr) != SC_CMBLK) {
        SCP(ptr, SC_CMBLK);
        CMBLKP(ptr, sptr);
        if (lastmember)
          SYMLKP(lastmember, ptr);
        else
          firstpointer = ptr;
        lastmember = ptr;
        added = 1;
      }
      off = PTROFFG(member);
      if (off && SCG(off) != SC_CMBLK) {
        SCP(off, SC_CMBLK);
        CMBLKP(off, sptr);
        if (lastmember)
          SYMLKP(lastmember, off);
        else
          firstpointer = off;
        lastmember = off;
        added = 1;
      }
      sdsc = SDSCG(member);
      if (sdsc && SCG(sdsc) != SC_CMBLK) {
        SCP(sdsc, SC_CMBLK);
        CMBLKP(sdsc, sptr);
        if (lastmember)
          SYMLKP(lastmember, sdsc);
        else
          firstpointer = sdsc;
        lastmember = sdsc;
        added = 1;
      }
      if (added) {
        /* remove base variable from common block? leave it? */
        int dtype, dty;
        int useptr = 1;
        dtype = DTYPEG(member);
        dty = DTYG(dtype);
        if (NO_PTR) {
          useptr = 0;
        } else if ((dty == TY_NCHAR || dty == TY_CHAR) && NO_CHARPTR) {
          useptr = 0;
        } else if (dty == TY_DERIVED && NO_DERIVEDPTR) {
          useptr = 0;
        }
        if (useptr) {
          /* remove the base variable from the common block */
          SYMLKP(lastmember, nextmember);
          nextlastmember = lastmember;
          CMBLKP(member, 0);
          SYMLKP(member, NOSYM);
          SCP(member, SC_BASED);
        } else {
          SYMLKP(lastmember, member);
        }
      }
    }
  }
  /* link list of pointer/offset/descriptor at from of common block */
  if (firstpointer)
    CMEMFP(sptr, firstpointer);
  CMEMLP(sptr, lastmember);
}

static void
reorder_common_pointers(int sptr)
{
  /*
   * Expand POINTER members in the common by placing the pointer/offset
   * descriptor near the beginning of common block because of alignment
   * restrictions  This is not standard f90/f95/f2003 behavior, but
   * ok for HPF since storage association rules are allowed to be violated.
   */
  int member, nextmember, lastmember, nextlastmember, firstpointer, lastpointer;

  firstpointer = lastpointer = 0;
  lastmember = 0;
  for (member = CMEMFG(sptr); member != NOSYM;
       lastmember = nextlastmember, member = nextmember) {
    nextlastmember = member;
    nextmember = SYMLKG(member);
    if (STYPEG(member) == ST_IDENT || STYPEG(member) == ST_UNKNOWN)
      STYPEP(member, ST_VAR);
    if (SDSCG(member) == 0 && !F90POINTERG(member) &&
        (POINTERG(member) || ALLOCG(member))) {
      get_static_descriptor(member);
      get_all_descriptors(member);
      SCP(member, SC_BASED);
    }
    if (POINTERG(member)) {
      int ptr, off, sdsc, added;
      added = 0;
      ptr = MIDNUMG(member);
      if (ptr && SCG(ptr) != SC_CMBLK) {
        SCP(ptr, SC_CMBLK);
        CMBLKP(ptr, sptr);
        if (lastpointer)
          SYMLKP(lastpointer, ptr);
        else
          firstpointer = ptr;
        lastpointer = ptr;
        added = 1;
      }
      off = PTROFFG(member);
      if (off && SCG(off) != SC_CMBLK) {
        SCP(off, SC_CMBLK);
        CMBLKP(off, sptr);
        if (lastpointer)
          SYMLKP(lastpointer, off);
        else
          firstpointer = off;
        lastpointer = off;
        added = 1;
      }
      sdsc = SDSCG(member);
      if (sdsc && SCG(sdsc) != SC_CMBLK) {
        SCP(sdsc, SC_CMBLK);
        CMBLKP(sdsc, sptr);
        if (lastpointer)
          SYMLKP(lastpointer, sdsc);
        else
          firstpointer = sdsc;
        lastpointer = sdsc;
        added = 1;
      }
      if (added) {
        /* remove base variable from common block? leave it? */
        int dtype, dty;
        int useptr = 1;
        dtype = DTYPEG(member);
        dty = DTYG(dtype);
        if (NO_PTR) {
          useptr = 0;
        } else if ((dty == TY_NCHAR || dty == TY_CHAR) && NO_CHARPTR) {
          useptr = 0;
        } else if (dty == TY_DERIVED && NO_DERIVEDPTR) {
          useptr = 0;
        }
        if (useptr) {
          /* remove the base variable from the common block */
          if (lastmember) {
            SYMLKP(lastmember, nextmember);
          } else {
            CMEMFP(sptr, nextmember);
          }
          nextlastmember = lastmember;
          CMBLKP(member, 0);
          SYMLKP(member, NOSYM);
          SCP(member, SC_BASED);
        }
      }
    }
  }
  /* link list of pointer/offset/descriptor at from of common block */
  if (lastpointer) {
    SYMLKP(lastpointer, CMEMFG(sptr));
    CMEMFP(sptr, firstpointer);
    if (lastmember == 0)
      lastmember = lastpointer;
  }
  CMEMLP(sptr, lastmember);
}

/** \brief Deallocate data structures for semantic analysis.
 */
void
semfin_free_memory(void)
{
  if (sem.doif_base == NULL)
    return;
  FREE(sem.doif_base);
  sem.doif_base = NULL;
  FREE(sem.stsk_base);
  sem.stsk_base = NULL;
  FREE(switch_base);
  switch_base = NULL;
  FREE(sem.interf_base);
  sem.interf_base = NULL;
  FREE(sem.scope_stack);
  sem.scope_stack = NULL;
  FREE(sem.typroc_base);
  sem.typroc_base = NULL;
  FREE(sem.iface_base);
  sem.iface_base = NULL;
  freearea(3); /* free area used for stmt function,
                * [NO]SEQUENCE info, and access info
                *
                * NOTE: 9/17/97, area 8 is used for stmt
                * functions -- need to keep just in case
                * the defs appear in a containing subprogram.
                */
  freearea(1); /* DOINFO records */
}

/** \brief Add type descriptor arguments to a specified function if they have
           not already been added.
    \param sptr is the symbol table pointer of the specified function.
 */
void
fix_class_args(int func_sptr)
{
  if (!have_class_args_been_fixed_already(func_sptr)) {
    /* type descriptors have not yet been added, so now we add them */
    int orig_count = PARAMCTG(func_sptr);
    int new_arg_position = orig_count;
    int j;
    for (j = 0; j < orig_count; ++j) {
      int arg_sptr = aux.dpdsc_base[DPDSCG(func_sptr) + j];
      if (add_class_arg_descr_arg(func_sptr, arg_sptr, new_arg_position))
        ++new_arg_position;
    }
  }
}

static void
fix_args(int sptr)
{
  /* walk thru all of the dummy arguments of the entries in the
   * subprogram to fix stypes of the args which were not referenced or
   * to replace a derived argument with its components.
   */
  int arg;
  int dscptr, i;
  /*
   * use a true pointer for locating the arguments; don't reallocate
   * aux.dpsdc_base between this assignment and its uses.
   */
  dscptr = DPDSCG(sptr);
  for (i = 0; i < PARAMCTG(sptr); ++i) {
    arg = aux.dpdsc_base[dscptr + i];
    /*  watch for alternate return specifier */
    if (arg) {
#if DEBUG
      assert(SCG(arg) == SC_DUMMY, "fix_args: arg not dummy", arg, 3);
#endif
      switch (STYPEG(arg)) {
      case ST_UNKNOWN:
      case ST_IDENT:
        STYPEP(arg, ST_VAR);
        break;
      case ST_ARRAY:
        if (ELEMENTALG(sptr)) {
          errsev(461);
          continue;
        }
        break;
      case ST_PROC:
        /* don't DCLCHK if used as a subroutine */
        if (ELEMENTALG(sptr)) {
          errsev(463);
        }
        if (FUNCG(arg) == 0) {
          if (!SDSCG(arg) && IS_PROC_DUMMYG(arg)) {
           get_static_descriptor(arg);
          }
          continue;
        }
        break;
      default:
        break;
      }
      if (ASSNG(arg) && INTENTG(arg) == INTENT_IN) {
        error(194, 2, gbl.lineno, SYMNAME(arg), CNULL);
        INTENTP(arg, INTENT_DFLT);
      }

      if (sptr == gbl.currsub && ALLOCATTRG(arg) &&
          INTENTG(arg) == INTENT_OUT) {
        gen_conditional_dealloc_for_sym(arg, ENTSTDG(sptr));
      }
      if (!SDSCG(arg) && IS_PROC_DUMMYG(arg)) { 
        get_static_descriptor(arg);
      } else if (POINTERG(arg)) {
        if (ELEMENTALG(sptr)) {
          errsev(462);
        }
        PTRARGP(sptr, 1);
        if (!SDSCG(arg) && !F90POINTERG(arg)) {
          /* only unreferenced dummies should get here.
             we could give an informational message.
           */
          get_static_descriptor(arg);
          get_all_descriptors(arg);
        }
      }
    }
  }
  if (FVALG(sptr)) {
    arg = FVALG(sptr);
    if (POINTERG(arg)) {
      if (ELEMENTALG(sptr)) {
        errsev(462);
      }
      PTRARGP(sptr, 1);
      if (!SDSCG(arg) && !F90POINTERG(arg)) {
        /* unreferenced return value.
           we could give an informational message.
         */
        get_static_descriptor(arg);
        get_all_descriptors(arg);
      }
    }
  }

}

void
llvm_fix_args(int sptr)
{
  fix_args(sptr);
}

static int
gen_accl_alias(int sptr, ACCL *accessp)
{
  int osptr = sptr;

  sptr = insert_sym(accessp->sptr);
  STYPEP(sptr, ST_ALIAS);
  SCOPEP(sptr, stb.curr_scope);
  IGNOREP(sptr, 0);
  if (STYPEG(osptr) == ST_ALIAS) {
    SYMLKP(sptr, SYMLKG(osptr));
  } else {
    SYMLKP(sptr, osptr);
  }
  return sptr;
}

static void
do_access(void)
{
  int sptr, encl;
  int nsyms;
  int stype;
  ACCL *accessp;

  if (sem.accl.type == 'v') {
    /*  scan entire symbol table to find variables to mark private */
    nsyms = stb.stg_avail - 1;
    for (sptr = stb.firstusym; sptr <= nsyms; ++sptr) {
      stype = STYPEG(sptr);
      switch (stype) {
      case ST_IDENT:
      case ST_VAR:
      case ST_ARRAY:
      case ST_STRUCT:
      case ST_UNION:
      /*
      ** PUBLIC/PRIVATE attribute *is* allowed for common block variables! **

                      if (SCG(sptr) == SC_CMBLK)
                          break;
      */
      case ST_UNKNOWN:
      case ST_NML:
      case ST_PROC:
      case ST_PARAM:
      case ST_TYPEDEF:
      case ST_OPERATOR:
      case ST_MODPROC:
      case ST_CMBLK:
      case ST_USERGENERIC:
        encl = ENCLFUNCG(sptr);
        if (encl && STYPEG(encl) == ST_MODULE && encl != gbl.currsub)
          break;
        if ((stype == ST_PROC || stype == ST_OPERATOR ||
             stype == ST_USERGENERIC) &&
            CLASSG(sptr) && VTOFFG(sptr))
          break; /* tbp PRIVATE set in derived type */
        if (is_procedure_ptr(sptr))
          break; /* FS#21906: proc ptr PRIVATE set at declaration */
        PRIVATEP(sptr, 1);
        break;
      case ST_ALIAS:
        encl = SCOPEG(sptr);
        if (encl && STYPEG(encl) == ST_MODULE && encl != gbl.currsub)
          break;
        PRIVATEP(sptr, 1);
        break;
      case ST_MODULE:
        if (sptr == gbl.currsub) {
          /* the module being defined contains  PRIVATE */
          PRIVATEP(sptr, 1);
        }
        break;
      default:
        break;
      }
    }
  }
  /*
   * traverse access list and process any variables which appeared with
   * the access attribute
   */
  for (accessp = sem.accl.next; accessp != NULL; accessp = accessp->next) {
    int rsptr;
    if (accessp->oper == 'o') {
      rsptr = sym_in_scope(accessp->sptr, OC_OPERATOR, &sptr, NULL, 0);
    } else {
      rsptr = sym_in_scope(accessp->sptr, OC_OTHER, &sptr, NULL, 0);
    }
    /* the original symbol may have been from a module
     * or be overloaded with a predefined name */
    if (sptr < stb.firstosym) {
      if (in_module) {
        if (TYPDG(accessp->sptr)) {
          /* can't issue public/private for intrinsics */
          error(155, 2, gbl.lineno, "PUBLIC/PRIVATE attribute ignored for",
                SYMNAME(sptr));
          continue;
        } else if (DCLDG(accessp->sptr) && STYPEG(accessp->sptr) != ST_MEMBER) {
          /* type declared, make a variable of this type */
          sptr = insert_sym(accessp->sptr);
          STYPEP(sptr, ST_VAR);
          SCOPEP(sptr, stb.curr_scope);
          IGNOREP(sptr, 0);
          SYMLKP(sptr, 0);
          DCLDP(sptr, 1);
          DTYPEP(sptr, DTYPEG(accessp->sptr));
        } else {
          /* otherwise, treat like a new symbol */
          sptr = insert_sym(accessp->sptr);
          if (in_module) {
            STYPEP(sptr, ST_UNKNOWN);
          } else {
            STYPEP(sptr, ST_IDENT);
          }
          SCOPEP(sptr, stb.curr_scope);
          IGNOREP(sptr, 0);
          SYMLKP(sptr, 0);
        }
      }
    } else if (sptr < stb.firstusym ||
               ((SCOPEG(sptr) && SCOPEG(sptr) != gbl.currsub &&
                 STYPEG(SCOPEG(sptr)) == ST_MODULE) &&
                (STYPEG(sptr) != ST_ALIAS || SCOPEG(sptr) != stb.curr_scope))) {
      /* insert an ST_ALIAS for that symbol */
      int osptr;
      osptr = gen_accl_alias(sptr, accessp);
      PRIVATEP(osptr, accessp->type == 'v');
      continue;
    }
    stype = STYPEG(sptr);
    switch (stype) {
    case ST_UNKNOWN:
      if (in_module) {
        if (sem.none_implicit) {
          /* can't be a variable, wouldn't be an unknown */
          SPTR sptr2 = findByNameStypeScope(SYMNAME(sptr), ST_INTRIN, 0);
          if (sptr2 > NOSYM && sptr != sptr2) {
            STYPEP(sptr, ST_ALIAS);
            PRIVATEP(sptr, accessp->type == 'v');
            SYMLKP(sptr, sptr2);
            SCOPEP(sptr, stb.curr_scope);
            break;
          }
          STYPEP(sptr, ST_MODPROC);
        } else {
          /* assume it's a variable to start out with */
          STYPEP(sptr, ST_IDENT);
        }
        SYMLKP(sptr, 0);
      }
      PRIVATEP(sptr, accessp->type == 'v');
      break;
    case ST_ALIAS:
      PRIVATEP(sptr, accessp->type == 'v');
      break;
    case ST_IDENT:
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
    /*
    ** PUBLIC/PRIVATE attribute *is* allowed for common block variables! **

                if (SCG(sptr) == SC_CMBLK) {
                    error(155, 2, gbl.lineno,
                        "PUBLIC/PRIVATE attribute ignored for common block
    member",
                        SYMNAME(sptr));
                    break;
                }
                PRIVATEP(sptr, accessp->type == 'v');
                break;
    */
    case ST_NML:
    case ST_PROC:
    case ST_ENTRY:
    case ST_PARAM:
    case ST_TYPEDEF:
    case ST_OPERATOR:
    case ST_CMBLK:
      if (STYPEG(sptr) == ST_PROC && GSAMEG(sptr)) {
        /* FS#20565 & FS#20566: Need to set public/private on the
         * generic name, not the procedure.
         */
        PRIVATEP(GSAMEG(sptr), accessp->type == 'v');
      } else {
        PRIVATEP(sptr, accessp->type == 'v');
      }
      /* make sure the $ac of this sptr also has the same access */

      if (PARAMG(sptr)) {
        if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
          PRIVATEP(CONVAL1G(sptr), accessp->type == 'v');
        } else if (DTY(DTYPEG(sptr)) == TY_DERIVED) {
          PRIVATEP(CONVAL1G(sptr), accessp->type == 'v');
        }
      }
      break;

    case ST_USERGENERIC:
      PRIVATEP(sptr, accessp->type == 'v');
      if (GTYPEG(sptr))
        PRIVATEP(GTYPEG(sptr), PRIVATEG(sptr));
      break;

    case ST_MODPROC:
      PRIVATEP(sptr, accessp->type == 'v');
      if (GSAMEG(sptr))
        PRIVATEP(GSAMEG(sptr), accessp->type == 'v');
      break;

    case ST_PD:
    case ST_GENERIC:
    case ST_INTRIN:
      sptr = refsym(sptr, OC_OTHER);
      PRIVATEP(sptr, accessp->type == 'v');
      break;

    default:
      error(155, 3, gbl.lineno, "PUBLIC/PRIVATE cannot be applied to",
            SYMNAME(sptr));
      break;
    }
  }
  if (IN_MODULE && in_module) {
    /* save public state */
    if (sem.accl.type == 'v') {
      /* default is private */
      sem.mod_public_flag = 0;
    } else {
      sem.mod_public_flag = 1;
    }
    /* look for PUBLIC symbols that are declared with a PRIVATE type */
    nsyms = stb.stg_avail - 1;
    for (sptr = stb.firstusym; sptr <= nsyms; ++sptr) {
      stype = STYPEG(sptr);
      switch (stype) {
      case ST_VAR:
      case ST_ARRAY:
      case ST_STRUCT:
      case ST_UNION:
      case ST_PROC:
      case ST_PARAM:
      case ST_OPERATOR:
      case ST_MODPROC:
        break;
      case ST_TYPEDEF:
        break;
      default:
        break;
      }
    }
  }
}

/* ******************************************************************/

void
do_equiv(void)
{
  int evp, first_evp;
  int sptr, ps;
  ISZ_T addr, size, temp;
  ISZ_T loc_addr, s_addr;
  LOGICAL first_ok, saveflg, dinitflg;
  int loc_list;
  int first_save; /* first saved local variable in list */
  int last_save;  /* last saved local variable in list */
  int maxa;       /* maximum alignment used for equiv'd variables */
  int a;          /* alignment of variable */

  /* Allocate space for PSECT records */
  psect_size = 100;
  NEW(psect_base, PSECT, psect_size);
  psect_num = 1;

#if DEBUG
  if (DBGBIT(3, 8))
    fprintf(gbl.dbgfil, "EQUIVALENCE LIST");
#endif

  first_save = last_save = 0;

  /*  loop thru equivalence list, performing error checking and
   *  equivalence operations:
   */
  first_evp = 0;
  for (evp = sem.eqvlist; evp != 0; evp = EQV(evp).next) {
    if (EQV(evp).is_first < 0) {
      /* already handled when imported */
      first_evp = 0;
    } else if (EQV(evp).is_first > 0) { /* first member of group */
      first_evp = evp;
      first_ok = chk_evar(evp);
    } else if (first_evp != 0 && chk_evar(evp) && first_ok) {
      equivalence(first_evp, evp);
      /*
       *  if the psect represented by first_evp was eliminated
       *  (merged into evp), use evp for subsequent equivalences
       *  in this group instead of first_evp:
       */
      if (psect_base[EQV(first_evp).ps].cmblk == -1)
        first_evp = evp;
    }
  }
  /*
   *  loop thru psects and
   *  (1) issue error if any element of a psect is not aligned correctly
   *  (2) assign addresses to symbols in local psects:
   */
  if (soc.size == 0) {
    soc.size = 1000;
    NEW(soc.base, SOC_ITEM, soc.size);
  }

  s_addr = loc_addr = 0; /* first available local variable address */
  loc_list = NOSYM;      /* list of equivalenced locals */
  dinitflg = FALSE;
  for (ps = 1; ps < psect_num; ++ps) {
    LOGICAL dinitd;
    LOGICAL vold;
    LOGICAL nmld;
    int cmblk = psect_base[ps].cmblk;

    if (cmblk == -1) /* ignore deleted psects */
      continue;
    for (sptr = psect_base[ps].memlist; sptr != NOSYM; sptr = SYMLKG(sptr)) {
      /*
       * storage overlap chains are terminated by 0; clean up the SOCPTR
       * fields since they were used temporarily to locate the ps index
       * of the equivalenced symbols.
       */
      assert(sptr, "equiv:bsym", 0, 3);
      SOCPTRP(sptr, 0);
    }
    maxa = size = 0;
    saveflg = sem.savall | (!(flg.recursive & 1));
    nmld = vold = dinitd = FALSE;
    for (sptr = psect_base[ps].memlist; sptr != NOSYM; sptr = SYMLKG(sptr)) {
      assert(sptr, "equiv:bsym", 1, 3);
      saveflg |= SAVEG(sptr);
      dinitd |= DINITG(sptr);
      saveflg |= dinitd;
      vold |= VOLG(sptr);
      nmld |= NMLG(sptr);
      addr = ADDRESSG(sptr);
      temp = size_of((int)DTYPEG(sptr));
      if (addr + temp > size)
        size = addr + temp;
      a = alignment((int)DTYPEG(sptr));
      if (a & addr)
        error(62, 2, gbl.lineno, SYMNAME(sptr), CNULL);
      if (a > maxa)
        maxa = a;
      add_socs(sptr, addr, temp);
      if (cmblk > 0 && SCG(sptr) != SC_CMBLK) {
        /* add sptr to common block psect */
        SCP(sptr, SC_CMBLK);
        SYMLKP(CMEMLG(cmblk), sptr);
        CMEMLP(cmblk, sptr);
        SYMLKP(sptr, NOSYM);
      }
    }
    if (vold) {
      for (sptr = psect_base[ps].memlist; sptr != NOSYM; sptr = SYMLKG(sptr))
        if (VOLG(sptr) && SOCPTRG(sptr))
          vol_equiv((int)SOCPTRG(sptr));
    }
    if (nmld) {
      for (sptr = psect_base[ps].memlist; sptr != NOSYM; sptr = SYMLKG(sptr))
        if (NMLG(sptr) && SOCPTRG(sptr))
          nml_equiv((int)SOCPTRG(sptr));
    }
    if (cmblk != 0) /* common block psect */
      /*  common block may have increased in size  */
      SIZEP(cmblk, size);
    else if (!in_module) { /* local psect */
      addr = ((saveflg | nmld) ? s_addr : loc_addr);
      addr = ALIGN(addr, maxa); /* round up addr to max boundary */
      if ((sptr = psect_base[ps].memlist) != NOSYM)
        for (;; sptr = SYMLKG(sptr)) {
          assert(sptr, "equiv:bsym", 2, 3);
          ADDRESSP(sptr, ADDRESSG(sptr) + addr);
          REFP(sptr, 1);
          if (SYMLKG(sptr) == NOSYM) /* NOTE: last sptr needs to */
            break;                   /* saved for next section */
        }
      if (saveflg | nmld) {
        /*  link psect list into end of saved variables list  */
        if (last_save)
          SYMLKP(last_save, psect_base[ps].memlist);
        else
          first_save = psect_base[ps].memlist;
        last_save = sptr;
        s_addr = addr + size;
        dinitflg |= dinitd;
      } else {
        /*  link psect list into front of referenced locals list */
        SYMLKP(sptr, loc_list);
        loc_list = psect_base[ps].memlist;
        loc_addr = addr + size;
      }
    }
  }

  /*  for the equivalenced locals, assign the target addresses to the
   *  variables and add to the gbl.locals list.
   */
  fix_equiv_locals(loc_list, loc_addr);

  /* for the equivalence locals which were saved and/or dinitd, assign
   * the target addresses to the variables and classify as SC_STATIC.
   */

  if (first_save)
    fix_equiv_statics(first_save, s_addr, dinitflg);

  FREE(psect_base);
#if DEBUG
  if (DBGBIT(3, 8))
    fprintf(gbl.dbgfil, "\nEQUIVALENCE LIST END\n");
#endif
}

/*
 * Check that a variable or array reference in an equivalence
 * list is a valid reference.  Return TRUE if okay, FALSE otherwise.
 */
static LOGICAL
chk_evar(int evp)
{
  int sptr, ps, dim, cmblk;
  int ss, j, numss, dty, ssast, savelineno;
  ADSC *ad;
  ISZ_T offset;
#define EVARERR(n, m)                          \
  {                                            \
    error(n, 3, gbl.lineno, SYMNAME(sptr), m); \
    return FALSE;                              \
  }

  /* Get symbol & check if an error occured earlier */
  sptr = EQV(evp).sptr;
  if (sptr == 0)
    return (FALSE);
  if (gbl.internal > 1 && !INTERNALG(sptr))
    return FALSE;
  ss = EQV(evp).subscripts;
  savelineno = gbl.lineno;
  gbl.lineno = EQV(evp).lineno;

#if DEBUG
  if (DBGBIT(3, 8)) {
    if (EQV(evp).is_first)
      fprintf(gbl.dbgfil, "\nline(%5d) ", EQV(evp).lineno);
    else
      fprintf(gbl.dbgfil, "\n            ");
    fprintf(gbl.dbgfil, "%s", SYMNAME(sptr));
    if (ss > 0) {
      numss = EQV_NUMSS(ss);
      for (j = 0; j < numss; ++j) {
        if (j)
          fprintf(gbl.dbgfil, ",");
        else
          fprintf(gbl.dbgfil, "(");
        ssast = EQV_SS(ss, j);
        if (A_TYPEG(ssast) == A_ID || A_TYPEG(ssast) == A_CNST) {
          fprintf(gbl.dbgfil, "sym %d (%d)", A_SPTRG(ssast),
                  CONVAL2G(A_SPTRG(ssast)));
        } else {
          fprintf(gbl.dbgfil, "unknownast[%d]", ssast);
        }
      }
      fprintf(gbl.dbgfil, ")");
    }
    fprintf(gbl.dbgfil, " (%" ISZ_PF "d)", EQV(evp).byte_offset);
    fprintf(gbl.dbgfil, ",");
    fprintf(gbl.dbgfil, "\n");
  }
#endif

  /*  check for variables which are illegal in equivalences  */

  if (SCG(sptr) == SC_DUMMY)
    EVARERR(57, CNULL);
  if (SCG(sptr) == SC_BASED)
    EVARERR(116, "(EQUIVALENCE)");
  dty = DTYPEG(sptr);
  if (DTY(dty) == TY_STRUCT || DTY(dty) == TY_UNION)
    EVARERR(60, CNULL);
  if (DTY(dty) == TY_DERIVED) {
    int tag;
    /* see if the derived type has the SEQUENCE attribute */
    tag = DTY(dty + 3);
    if (tag == 0 || !SEQG(tag)) {
      EVARERR(444, CNULL);
    }
  }

  offset = 0;
  if (STYPEG(sptr) == ST_IDENT || STYPEG(sptr) == ST_UNKNOWN)
    STYPEP(sptr, ST_VAR);
  if (STYPEG(sptr) == ST_VAR) {
    if (DTY(DTYPEG(sptr)) == TY_CHAR || DTY(DTYPEG(sptr)) == TY_NCHAR) {
      /* Check if char variable was referenced as an array */
      if (ss > 0) {
        if (EQV(evp).byte_offset)
          EVARERR(76, CNULL);
        if (EQV_NUMSS(ss) != 1)
          EVARERR(76, CNULL);
        ssast = EQV_SS(ss, 0);
        if (A_TYPEG(ssast) == A_ID || A_TYPEG(ssast) == A_CNST) {
          EQV(evp).byte_offset = CONVAL2G(A_SPTRG(ssast));
          if (flg.standard)
            error(76, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        } else {
          EQV(evp).byte_offset = 0;
          /* error already issued */
          /*error(155, 3, gbl.lineno, SYMNAME(sptr), "- nonconstant equivalence
           * subscript" );*/
        }
      }
    } else {
      if (ss > 0 || EQV(evp).byte_offset)
        EVARERR(76, CNULL);
    }
  } else if (STYPEG(sptr) == ST_ARRAY) {
    if (ALLOCG(sptr)) {
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- an allocatable array cannot be equivalenced");
      gbl.lineno = savelineno;
      return FALSE;
    }
    if (ADJARRG(sptr)) {
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- an adjustable array cannot be equivalenced");
      gbl.lineno = savelineno;
      return FALSE;
    }
    if (ss > 0) {
      int err = 0;
      ad = AD_PTR(sptr);
      numss = EQV_NUMSS(ss);
      for (dim = 0; dim < numss; ++dim) {
        if (dim >= AD_NUMDIM(ad))
          EVARERR(78, CNULL);
        ssast = EQV_SS(ss, dim);
        if (A_TYPEG(ssast) == A_ID || A_TYPEG(ssast) == A_CNST) {
          offset += (CONVAL2G(A_SPTRG(ssast)) -
                     get_int_cval(sym_of_ast(AD_LWAST(ad, dim)))) *
                    get_int_cval(sym_of_ast(AD_MLPYR(ad, dim)));
        } else {
          /* error already issued */
          /*error(155, 3, gbl.lineno, SYMNAME(sptr),
                      "- nonconstant equivalence subscript" );*/
          err = 1;
        }
      }
      if (dim != AD_NUMDIM(ad)) {
        if (dim == 1) {
          if (flg.standard)
            error(78, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        } else
          EVARERR(78, CNULL);
      } else if (flg.standard && err == 0) {
        for (dim = 0; dim < numss; ++dim) {
          int val;
          val = CONVAL2G(A_SPTRG(EQV_SS(ss, dim)));
          if (val < get_int_cval(sym_of_ast(AD_LWAST(ad, dim))) ||
              val > get_int_cval(sym_of_ast(AD_UPAST(ad, dim))))
            error(80, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        }
      }
      offset *= size_of((int)DDTG(DTYPEG(sptr)));
    }
  } else
    EVARERR(84, CNULL);

  if (EQV(evp).byte_offset) {
    if (DTYG(DTYPEG(sptr)) == TY_CHAR)
      EQV(evp).byte_offset--;
    else if (DTYG(DTYPEG(sptr)) == TY_NCHAR)
      EQV(evp).byte_offset = 2 * (EQV(evp).byte_offset - 1);
    else
      EVARERR(75, CNULL);
  }
  /*
   *  assign to EQV(evp).byte_offset, the total byte offset from the
   *  beginning of the psect:
   */
  EQV(evp).byte_offset += (offset + ADDRESSG(sptr));

  /*  allocate a new psect if necessary  */
  if (SC_ISCMBLK(SCG(sptr))) {
    cmblk = CMBLKG(sptr); /* sym pointer to common block name */
    ps = CMBLKG(cmblk);
  } else {
    /*  local variable  */
    cmblk = 0;
    ps = SOCPTRG(sptr);
  }
  if (ps == 0) { /* allocate new psect */
    ps = psect_num++;
    NEED(psect_num, psect_base, PSECT, psect_size, psect_size + 100);
    psect_base[ps].cmblk = cmblk;
    if (cmblk) {
      CMBLKP(cmblk, ps);
      psect_base[ps].memlist = CMEMFG(cmblk);
    } else {
      assert(SYMLKG(sptr) == 0 || SYMLKG(sptr) == NOSYM, "chk_evar:b slnk",
             sptr, 2);
      SOCPTRP(sptr, ps);
      psect_base[ps].memlist = sptr;
    }
  }
  EQV(evp).ps = ps; /* save psect number */
  gbl.lineno = savelineno;
  return TRUE;
}

static void
equivalence(int evp, int evp2)
{
  int ps, ps2;
  ISZ_T offset, offset2;
  int sptr, sptr2;
  int pstemp;

  ps = EQV(evp).ps;
  ps2 = EQV(evp2).ps;
  offset = EQV(evp).byte_offset;
  offset2 = EQV(evp2).byte_offset;
  sptr = EQV(evp).sptr;
  sptr2 = EQV(evp2).sptr;

  if (DBGBIT(3, 8))
    fprintf(gbl.dbgfil, ">>>>> equivalence of %s/psect(%d):%" ISZ_PF
                        "d and %s/psect(%d):%" ISZ_PF "d\n",
            SYMNAME(sptr), ps, offset, SYMNAME(sptr2), ps2, offset2);

  if (in_module) {
    if ((DTYG(DTYPEG(sptr)) == TY_CHAR && DTYG(DTYPEG(sptr2)) != TY_CHAR) ||
        (DTYG(DTYPEG(sptr2)) == TY_CHAR && DTYG(DTYPEG(sptr)) != TY_CHAR) ||
        (DTYG(DTYPEG(sptr)) == TY_NCHAR || DTYG(DTYPEG(sptr2)) == TY_NCHAR))
      error(310, 3, gbl.lineno,
            "Cannot EQUIVALENCE non-character and character",
            "in the specification part of a MODULE");
  } else if (flg.standard) {
    if (DTYG(DTYPEG(sptr)) == TY_CHAR && DTYG(DTYPEG(sptr2)) != TY_CHAR)
      error(183, 2, gbl.lineno, SYMNAME(sptr2), SYMNAME(sptr));
    else if (DTYG(DTYPEG(sptr2)) == TY_CHAR && DTYG(DTYPEG(sptr)) != TY_CHAR)
      error(183, 2, gbl.lineno, SYMNAME(sptr), SYMNAME(sptr2));
    else if (DTYG(DTYPEG(sptr)) == TY_NCHAR || DTYG(DTYPEG(sptr2)) == TY_NCHAR)
      error(183, 2, gbl.lineno, SYMNAME(sptr), SYMNAME(sptr2));
  } else {
    if (DTYG(DTYPEG(sptr)) == TY_NCHAR && DTYG(DTYPEG(sptr2)) != TY_NCHAR)
      error(185, 2, gbl.lineno, SYMNAME(sptr2), SYMNAME(sptr));
    else if (DTYG(DTYPEG(sptr2)) == TY_NCHAR && DTYG(DTYPEG(sptr)) != TY_NCHAR)
      error(185, 2, gbl.lineno, SYMNAME(sptr), SYMNAME(sptr2));
  }

  if (ps == ps2) {
    /*  redundant equivalence - must not be inconsistent  */
    if (offset != offset2)
      error(59, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(sptr2));
  } else {
    /*  decide whether to merge ps into ps2, or vice versa  */
    offset = offset2 - offset;
    if (offset < 0 || (offset == 0 && psect_base[ps].cmblk)) {
      /*  ps2 will be merged ... switch ps and ps2  */
      offset = -offset;
      pstemp = ps;
      ps = ps2;
      ps2 = pstemp;
    }
    /*
     *  not allowed to equivalence two common blocks, and -
     *  not allowed to extend common block backwards:
     */
    if (psect_base[ps].cmblk) {
      if (psect_base[ps2].cmblk)
        error(58, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(sptr2));
      else
        error(61, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(sptr2));
      return;
    }
    /*
     *  eliminate ps - update the addresses of its members, and
     *  insert its member list after the first member of ps2:
     */
    for (sptr = psect_base[ps].memlist;; sptr = SYMLKG(sptr)) {
      assert(sptr, "equiv:bsym", 3, 3);
      ADDRESSP(sptr, ADDRESSG(sptr) + offset);
      SOCPTRP(sptr, ps2);          /* assign new psect number */
      if (psect_base[ps2].cmblk) { /* update true psect */
        SCP(sptr, SC_CMBLK);
        CMBLKP(sptr, psect_base[ps2].cmblk);
        if (DINITG(sptr))
          DINITP(psect_base[ps2].cmblk, 1);
      }
      /* hack - mark symbol as being added to a common block or other
       * memory area due to an equivalence
       */
      EQVP(sptr, 1);
      if (SYMLKG(sptr) == NOSYM) /* NOTE: last sptr is needed */
        break;                   /* for the ensuing code */
    }
    sptr2 = psect_base[ps2].memlist; /* first member of ps2 */
    SYMLKP(sptr, SYMLKG(sptr2));
    SYMLKP(sptr2, psect_base[ps].memlist);
    psect_base[ps].cmblk = -1;
  }

}

/*
 * add elements to SOC lists for those elements following sptr in psect
 * list which overlap sptr
 *     sptr:  equivalenced symbol
 *     addr:  address (relative) of sptr
 *     size:  size in bytes of sptr
 */
static void
add_socs(int sptr, ISZ_T addr, ISZ_T size)
{
  int sptr2;
  ISZ_T addr2;

  for (sptr2 = SYMLKG(sptr); sptr2 != NOSYM; sptr2 = SYMLKG(sptr2)) {
    assert(sptr2, "equiv:bsym", 4, 3);
    addr2 = ADDRESSG(sptr2);
    if (addr <= addr2) {
      if (addr + size <= addr2)
        continue;
    } else if (addr >= addr2 + size_of((int)DTYPEG(sptr2)))
      continue;

    /*  add item to Storage Overlap Chain for both sptr and sptr2  */

    NEED(soc.avail + 2, soc.base, SOC_ITEM, soc.size, soc.size + 1000);
    SOC_SPTR(soc.avail) = sptr2;
    SOC_NEXT(soc.avail) = SOCPTRG(sptr);
    SOCPTRP(sptr, soc.avail);
    SEQP(sptr, 1);
    soc.avail++;
    SOC_SPTR(soc.avail) = sptr;
    SOC_NEXT(soc.avail) = SOCPTRG(sptr2);
    SOCPTRP(sptr2, soc.avail);
    SEQP(sptr2, 1);
    soc.avail++;
    if (DBGBIT(3, 8))
      fprintf(gbl.dbgfil, " %s overlaps %s\n", SYMNAME(sptr), SYMNAME(sptr2));
  }

}

/**
   \brief set VOL of all symbols which are equivalenced (closure of socs)
 */
static void
vol_equiv(int socp)
{
  int sptr;
  int p;

  sptr = SOC_SPTR(socp);
  if (VOLG(sptr))
    return;
  VOLP(sptr, 1);
  p = socp;
  while ((p = SOC_NEXT(p))) {
    vol_equiv(p);
    if (socp == p)
      break;
    socp = p;
  }
}

/**
   \brief set NML of all symbols which are equivalenced (closure of socs)
 */
static void
nml_equiv(int socp)
{
  int sptr;
  int p;

  sptr = SOC_SPTR(socp);
  if (NMLG(sptr))
    return;
  NMLP(sptr, 1);
  p = socp;
  while ((p = SOC_NEXT(p))) {
    nml_equiv(p);
    if (socp == p)
      break;
    socp = p;
  }
}

/* ******************************************************************/

static int nml;         /* current namelist group */
static LOGICAL nml_err; /* any errors in the namelist groups */
static int nml_size;    /* size of the namelist group array */
static LOGICAL new_nml; /* for adjustable array */

static void _put(INT);
#define PUT(n) (_put((INT)(n)))
#define PUTA(n) (dinit_put(DINIT_LABEL, (INT)(n)))

static void nml_traverse(int, void (*p)(int));
static void nml_check_item(int);
static void nml_emit_desc(int);

static void
do_nml(void)
{
  int sptr, item, cnt, nmlinmodule;
  int plist;
  LOGICAL ref_nml;

  ref_nml = FALSE;
  new_nml = FALSE;
  for (nml = sem.nml; nml != NOSYM; nml = SYMLKG(nml)) {
    /* set 'nmlinmodule' if this namelist was from a module */
    nmlinmodule = ENCLFUNCG(nml);
    if (!nmlinmodule || STYPEG(nmlinmodule) != ST_MODULE) {
      nmlinmodule = 0;
    }
    /* always generate error messages, compute size */
    nml_err = FALSE;
    nml_size = 3; /* namelen, name, count */
    cnt = 0;      /* number of items in group */
    plist = ADDRESSG(nml);
    for (item = CMEMFG(nml); item; item = NML_NEXT(item)) {
      sptr = NML_SPTR(item);
      gbl.lineno = NML_LINENO(item);
      nml_traverse(sptr, nml_check_item);
      if (nml_err)
        continue;

      /* VALID namelist symbol */

      if (!in_module && SCG(sptr) == SC_NONE) {
        /*
         * When the namelist declaration appears a MODULE, we know that
         * the items are 'global' and the items' storage class will be
         * defined by module.c:fix_module_common().  Clearly, making the
         * items SC_LOCAL is incorrect.
         */
        SCP(sptr, SC_LOCAL);
      }
      ASSNP(sptr, 1);
      cnt++;
    }
    PLLENP(plist, nml_size);
    if ((REFG(nml) == 0 && !in_module && gbl.internal != 1) || nml_err ||
        nmlinmodule)
      continue;
    /*
     * Create data initialized character variables for the names of
     * the namelist group and its members if character constants aren't
     * allowed as arguments to RTE_loc().
     */
    if (XBIT(49, 0x100000)) {
      dinit_name(nml);
      for (item = CMEMFG(nml); item; item = NML_NEXT(item)) {
        sptr = NML_SPTR(item);
        dinit_name(sptr);
      }
    }
    /*
     * data initialize the descriptor of the namelist group which is
     * addressed by the group's associated plist - this descriptor
     * is defined by the PGI Fortran I/O spec.
     */
    dinit_put(DINIT_NML, (INT)plist);
    put_name(nml); /* name of namelist group */
    PUT(cnt);
    /*
     * scan through all of the items in the group and create a descriptor
     * for each item.
     */
    new_nml = TRUE; /* set for adjustable array */
    for (item = CMEMFG(nml); item; item = NML_NEXT(item)) {
      sptr = NML_SPTR(item);
      nml_traverse(sptr, nml_emit_desc);
      new_nml = FALSE;
    }
    new_nml = FALSE;
    DINITP(plist, 1);
#ifdef USE_MPC
    /* Need to be done before sym_is_refd on the plist */
    etls_privatize(nml);
#endif
    sym_is_refd(plist);
    dinit_put(DINIT_END, 0);
    ref_nml = TRUE;
  }
  if (ref_nml)
    (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_loc), DT_ADDR);

}

static void
nml_sym_is_refd(int sptr)
{
  if (sptr > 0) {
    if (STYPEG(sptr) == ST_MEMBER || ALLOCG(sptr) || POINTERG(sptr))
      return;
    if (STYPEG(sptr) == ST_ARRAY && ADJARRG(sptr))
      return;
    sym_is_refd(sptr);
  }
}

static void
do_nml_sym_is_refd(void)
{
  int sptr, item, nml;

  for (nml = sem.nml; nml != NOSYM; nml = SYMLKG(nml)) {
    for (item = CMEMFG(nml); item; item = NML_NEXT(item)) {
      sptr = NML_SPTR(item);
      nml_traverse(sptr, nml_sym_is_refd);
    }
  }
}

static void
_put(INT n)
{
  if (size_of(DT_PTR) == 8) {
    n = cngcon(n, DT_INT4, DT_INT8);
    dinit_put(DT_INT8, n);
  } else
    dinit_put(DT_INT4, n);
}

#if defined(PARENTG)

static void
nml_traverse_parenttype(int dtype, void (*visitf)(int))
{
  int possible_ext = 1;
  int parent, m;
  for (m = DTY(dtype + 1); m != NOSYM; m = SYMLKG(m)) {
    parent = PARENTG(m);
    /* check extended type , traverse member instead */
    if (possible_ext && parent && parent == m && DTY(DTYPEG(m) == TY_DERIVED)) {
      nml_traverse_parenttype(DTYPEG(m), visitf);

    } else {
      nml_traverse(m, visitf);
    }
    possible_ext = 0;
  }
}
#endif

/* nml traversal in linear order */
static void
nml_traverse(int sptr, void (*visitf)(int))
{
  int dtype, ty, possible_ext, parent, i;
  possible_ext = 1;

  (*visitf)(sptr);
  if (STYPEG(sptr) == ST_MEMBER && (POINTERG(sptr) || ALLOCG(sptr)))
    /* don't traverse the member with the POINTER or ALLOCATABLE
     * attribute for fear of self-referential structures -- these
     * are illegal, an error will be reported, but nml_traverse()
     * would infinitely recurse without this check.
     */
    return;
  dtype = DDTG(DTYPEG(sptr)); /* get element dtype if array */
  ty = DTY(dtype);
  i = dtype_has_defined_io(dtype) & (DT_IO_FWRITE | DT_IO_FREAD);
  if (ty == TY_DERIVED && !i) {
    int m;
    for (m = DTY(dtype + 1); m != NOSYM; m = SYMLKG(m)) {
#ifdef PARENTG
      parent = PARENTG(m);
      /* check extended type , traverse member instead */
      if (possible_ext && parent && parent == m &&
          DTY(DTYPEG(m)) == TY_DERIVED) {
        nml_traverse_parenttype(DTYPEG(m), visitf);

      } else {
#endif
        nml_traverse(m, visitf);
#if defined(PARENTG)
      }
#endif
      possible_ext = 0;
    }
    (*visitf)(0); /* and to mark the end of the members */
  }
}

/* check for a valid namelist item and compute its descriptor size */
static void
nml_check_item(int sptr)
{
  int dtype, ty, ndims, dtio, i;

  if (sptr <= 0) {
    /* end of derived type members */
    nml_size++;
    return;
  }

  nml_size += 5; /* namelen, name, address, datatype, charlen */
  dtype = DTYPEG(sptr);
  if ((POINTERG(sptr) || ALLOCG(sptr)) && STYPEG(sptr) != ST_MEMBER) {
    ndims = 1;
  } else if (DTY(dtype) == TY_ARRAY) {
    ndims = ADD_NUMDIM(dtype);
    dtype = DTY(dtype + 1);
  } else
    ndims = 0;

  /* defined io: 0, readptr, writeptr, dtv, v_list,
   *             dtv$sd, v_list$sd, iotype$cl
   * dtv is already counted
   */
  dtio = 0;
  i = dtype_has_defined_io(dtype) & (DT_IO_FWRITE | DT_IO_FREAD);
  if (i) {
    dtio = 7;
  }
  nml_size += 1 + dtio + 2 * ndims; /* ndims, [lower, upper]... */

  ty = DTY(dtype);
  if (ty >= TY_STRUCT && ty != TY_DERIVED) {
    error(108, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(nml));
    nml_err = TRUE;
    return;
  }

  switch (STYPEG(sptr)) {
  case ST_UNKNOWN:
  case ST_IDENT:
    STYPEP(sptr, ST_VAR);
    FLANG_FALLTHROUGH;
  case ST_VAR:
    if (SCG(sptr) == SC_DUMMY) {
      if (DTY(DDTG(dtype)) != TY_CHAR)
        break;
      if (!ASSUMLENG(sptr))
        break;
    } else if (SCG(sptr) != SC_BASED)
      break;
    if (DTY(DDTG(dtype)) != TY_CHAR)
      break;
    if ((DDTG(dtype)) == DT_DEFERCHAR || (DDTG(dtype)) == DT_DEFERNCHAR)
      break;
    if (!ASSUMLENG(sptr))
      break;
    /** assumed-size char not allowed **/
    error(108, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(nml));
    nml_err = TRUE;
    break;
  case ST_ARRAY:
    /** assumed-size arrays not allowed **/
    if (SCG(sptr) == SC_NONE && ASUMSZG(sptr)) {
      error(108, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(nml));
      nml_err = TRUE;
    }
    break;
  case ST_MEMBER:
    break;
  default:
    error(108, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(nml));
    nml_err = TRUE;
    break;
  }

}

static int
gen_vlist(void)
{
  int sptr, vlist_ast;
  ADSC *ad;

  /* make array of size 0 */
  /* set it as array size 0 first */

  int dtype;
  if (XBIT(124, 0x10))
    dtype = get_array_dtype(1, DT_INT8);
  else
    dtype = get_array_dtype(1, DT_INT);
  ad = AD_DPTR(dtype);
  AD_LWAST(ad, 0) = astb.i1;
  AD_LWBD(ad, 0) = astb.i1;
  AD_UPAST(ad, 0) = astb.i0;
  AD_UPBD(ad, 0) = astb.i0;
  AD_MLPYR(ad, 0) = astb.i1;

  sptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, dtype, SC_LOCAL);
  ALLOCP(sptr, 1);
  get_static_descriptor(sptr);
  get_all_descriptors(sptr);
  vlist_ast = mk_id(sptr);
  DESCUSEDP(sptr, 1);
  ARGP(sptr, 1);

  return vlist_ast;
}

static ITEM *
gen_dtio_arglist(int sptr, int vlist_ast)
{
  ITEM *p, *arglist;
  int ast_type, iostat_ast, iomsg_ast, unit_ast;
  int iotype_ast;
  int tsptr, tdtype;
  int argdtyp;
  if (XBIT(124, 0x10))
    argdtyp = DT_INT8;
  else
    argdtyp = DT_INT;

  /* dtv , must be scalar*/
  tsptr = sptr;
  p = (ITEM *)getitem(0, sizeof(ITEM));
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  p->next = ITEM_END;
  p->next = NULL;
  if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
    tdtype = DDTG(DTYPEG(sptr));
    tsptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, tdtype, SC_LOCAL);
  }
  p->ast = mk_id(tsptr);
  arglist = p;
  SST_ASTP(p->t.stkp, p->ast);
  SST_DTYPEP(p->t.stkp, DTYPEG(tsptr));
  SST_SYMP(p->t.stkp, tsptr);
  SST_PARENP(p->t.stkp, 0);
  /* need to check if this is S_IDENT or S_SCONST  */
  SST_IDP(p->t.stkp, S_IDENT);
  SST_SHAPEP(p->t.stkp, A_SHAPEG(p->ast));

  /* make fake unit */
  if (A_DTYPEG(astb.i0) != argdtyp)
    unit_ast = mk_convert(astb.i0, argdtyp);
  else
    unit_ast = astb.i0;
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  SST_ASTP(p->t.stkp, unit_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(unit_ast));
  ast_type = A_TYPEG(unit_ast);
  SST_SHAPEP(p->t.stkp, 0);
  SST_IDP(p->t.stkp, S_CONST);
  SST_SYMP(p->t.stkp, A_SPTRG(unit_ast));
  SST_LSYMP(p->t.stkp, 0);
  SST_CVALP(p->t.stkp, CONVAL2G(A_SPTRG(unit_ast)));
  p->ast = unit_ast;

  /* fake iotype */
  iotype_ast = mk_cnst(getstring("NAMELIST", strlen("NAMELIST")));
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  p->ast = iotype_ast;
  SST_ASTP(p->t.stkp, iotype_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(iotype_ast));
  SST_SYMP(p->t.stkp, A_SPTRG(iotype_ast));
  SST_PARENP(p->t.stkp, 0);
  SST_SHAPEP(p->t.stkp, 0);
  SST_IDP(p->t.stkp, S_CONST);

  /* v_list */
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  p->next = NULL;
  p->ast = vlist_ast;
  SST_ASTP(p->t.stkp, vlist_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(vlist_ast));
  SST_SYMP(p->t.stkp, A_SPTRG(vlist_ast));
  SST_PARENP(p->t.stkp, 0);
  SST_SHAPEP(p->t.stkp, 0);
  SST_IDP(p->t.stkp, S_IDENT);

  /* fake iostat */
  if (A_DTYPEG(astb.i0) != argdtyp)
    iostat_ast = mk_convert(astb.i0, argdtyp);
  else
    iostat_ast = astb.i0;
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  p->ast = iostat_ast;
  SST_ASTP(p->t.stkp, iostat_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(iostat_ast));
  SST_SYMP(p->t.stkp, A_SPTRG(iostat_ast));
  SST_IDP(p->t.stkp, S_CONST);
  SST_PARENP(p->t.stkp, 0);
  SST_SHAPEP(p->t.stkp, 0);

  /* fake iomsg */
  sptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, DT_CHAR, SC_LOCAL);
  iomsg_ast = mk_id(sptr);
  p->next = (ITEM *)getitem(0, sizeof(ITEM));
  p = p->next;
  p->t.stkp = (SST *)getitem(0, sizeof(SST));
  p->next = ITEM_END;
  p->ast = iomsg_ast;
  SST_ASTP(p->t.stkp, iomsg_ast);
  SST_DTYPEP(p->t.stkp, A_DTYPEG(iomsg_ast));
  SST_SYMP(p->t.stkp, A_SPTRG(iomsg_ast));
  SST_IDP(p->t.stkp, S_IDENT);
  SST_PARENP(p->t.stkp, 0);
  SST_SHAPEP(p->t.stkp, 0);

  return arglist;
}

static int static_cnt = 0;
/* emit a descriptor for a namelist item.  For derived types, the descriptors
 * for members immediately follow the derived type's descriptor.  The
 * last member is followed by a single word whose value is 0.
 */
static void
nml_emit_desc(int sptr)
{
  int cnt, dtype, ndims, dttype, i;
  ADSC *ad;

  if (new_nml == TRUE) {
    static_cnt = 3; /* nml header (name, size, len)*/
    new_nml = FALSE;
  }

  if (sptr <= 0) {
    /* end of derived type members */
    PUT(0);
    ++static_cnt;
    return;
  }

  if (SCG(sptr) == SC_LOCAL) {
    if (DINITG(sptr) || SAVEG(sptr))
      SCP(sptr, SC_STATIC); /* ensure item's addr is static */
  }

  put_name(sptr); /* name of item in group */
  static_cnt = static_cnt + 2;

  if (ALLOCG(sptr) || POINTERG(sptr)) {
    if (SDSCG(sptr) == 0) {
      if (ALLOCATTRG(sptr)) {
        get_static_descriptor(sptr);
        DESCUSEDP(sptr, 1);
        if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
          if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_MODULE) {
            get_all_descriptors(sptr);
          } else {
            trans_mkdescr(sptr);
            NODESCP(sptr, 0);
            SECDSCP(DESCRG(sptr), SDSCG(sptr));
          }
        }
      } else {
        DESCUSEDP(sptr, 1);
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
      }
      ALLOCDESCP(sptr, 1);
      SCP(sptr, SC_BASED);
    }
    if (!MIDNUMG(sptr)) {
      PUTA(sptr);        /* item's address */
      ADDRTKNP(sptr, 1); /* item appears as an argument */
    } else {
      ADDRTKNP(MIDNUMG(sptr), 1); /* item appears as an argument */
      PUTA(MIDNUMG(sptr));        /* item's address */
    }
    ++static_cnt;
  } else if (STYPEG(sptr) != ST_MEMBER) {
    ADDRTKNP(sptr, 1); /* item appears as an argument */
    PUTA(sptr);        /* item's address */
    ++static_cnt;
  } else {
    PUT(ADDRESSG(sptr)); /* member's offset */
    ++static_cnt;
  }
  dtype = DTYPEG(sptr);
  if (DTY(dtype) != TY_ARRAY) {
    ndims = 0;
  } else { /* ST_ARRAY */
    ad = AD_PTR(sptr);
    ndims = AD_NUMDIM(ad);
    dtype = DTY(dtype + 1);
  }
  PUT(dtype_to_arg(dtype));
  ++static_cnt;
  if ((DDTG(dtype)) == DT_DEFERCHAR || (DDTG(dtype)) == DT_DEFERNCHAR) {
    PUT(0); /* character length */
    ++static_cnt;
  } else if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
    int clen = string_length(dtype);
    PUT(clen); /* character length */
    ++static_cnt;
  } else if (DTY(dtype) == TY_DERIVED) {
    PUT(DTY(dtype + 2)); /* size of the derived type */
    ++static_cnt;
  } else {
    PUT(0);
    ++static_cnt;
  }

  /* IMPORTANT: If more data is added between sptr and after ndims,
   *            an update needs to be done in lower_data_stmt()
   *            in lowerilm.c for adjustable array because
   *            it counts many data between sptr and ndims
   *            to start lower/upperbounds information.
   */

  dttype = DDTG(DTYPEG(sptr));
  i = dtype_has_defined_io(dttype) & (DT_IO_FWRITE | DT_IO_FREAD);
  if (SDSCG(sptr) && (POINTERG(sptr) || ALLOCG(sptr)))
    if (DTY(dttype) == TY_DERIVED && i) {
      PUT(-2); /* number of dimensions */
    } else {
      PUT(-1); /* number of dimensions */
    }
  else {
    if (DTY(dttype) == TY_DERIVED && i) {
      PUT(ndims + 30); /* number of dimensions+30 */
    } else {
      PUT(ndims); /* number of dimensions */
    }
  }
  ++static_cnt;
  if (ndims && !POINTERG(sptr) && !ALLOCG(sptr) && !ADJARRG(sptr)) {
    cnt = 0;
    /* lower and upper bounds for each dimension */
    do {
      PUT(get_int_cval(sym_of_ast(AD_LWAST(ad, cnt))));
      PUT(get_int_cval(sym_of_ast(AD_UPAST(ad, cnt))));
      static_cnt = static_cnt + 2;
      cnt++;
    } while (--ndims);
  } else if (ndims && ADJARRG(sptr)) {
    int dt = DTYPEG(sptr);
    int subs[MAXRANK];
    cnt = 0;
    if (SCG(sptr) != SC_DUMMY) {
      /*
       * Namelist of automatic array - its pointer is to be stored at
       *  nml [static_cnt-3]
       */
      int from, astplist, ast, dest;
      from = mk_id(sptr);
      from = mk_unop(OP_LOC, from, DT_PTR);
      subs[0] = mk_cval(static_cnt - 3, DT_INT);
      astplist = mk_id(ADDRESSG(nml));
      dest = mk_subscr(astplist, subs, 1, DT_PTR);
      ast = mk_assn_stmt(dest, from, DTYPEG(dest));
      add_stmt_after(ast, 0);
    }
    do {
      if (ADD_LWBD(dt, cnt)) {
        int from, astplist, ast, dest;
        ++static_cnt;
        from = mk_id(sym_of_ast(AD_LWAST(ad, cnt)));
        subs[0] = mk_cval(static_cnt, DT_INT);
        astplist = mk_id(ADDRESSG(nml));
        dest = mk_subscr(astplist, subs, 1, DT_PTR);
        ast = mk_assn_stmt(dest, from, DTYPEG(dest));
        PUT(-99);
        add_stmt_after(ast, 0);
      } else {
        ++static_cnt;
        PUT(get_int_cval(sym_of_ast(AD_LWAST(ad, cnt))));
      }

      if (ADD_UPBD(dt, cnt)) {
        int from, astplist, ast, dest;
        ++static_cnt;
        from = mk_id(sym_of_ast(AD_UPAST(ad, cnt)));
        astplist = mk_id(ADDRESSG(nml));
        subs[0] = mk_cval(static_cnt, DT_INT);
        dest = mk_subscr(astplist, subs, 1, DT_PTR);
        ast = mk_assn_stmt(dest, from, DTYPEG(dest));
        add_stmt_after(ast, 0);
        PUT(-99);
      } else {
        PUT(get_int_cval(sym_of_ast(AD_UPAST(ad, cnt))));
        ++static_cnt;
      }
      cnt++;
    } while (--ndims);
  } else if (POINTERG(sptr) || ALLOCATTRG(sptr)) {
    PUT(ndims); /* number of dimensions */
    ++static_cnt;
    PUTA(SDSCG(sptr)); /* item's descriptor address */
    ++static_cnt;
    ADDRTKNP(SDSCG(sptr), 1);
  }

  /* defined io */
  i = dtype_has_defined_io(dttype) & (DT_IO_FWRITE | DT_IO_FREAD);
  if (DTY(dttype) == TY_DERIVED && i) {
    int rsptr, wsptr, vlist, vlistsd, dtvsd;
    ITEM *arglist;

    vlist = gen_vlist();
    arglist = gen_dtio_arglist(sptr, vlist);

    rsptr = resolve_defined_io(0, arglist->t.stkp, arglist);
    wsptr = resolve_defined_io(1, arglist->t.stkp, arglist);
#if DEBUG
    if (rsptr == 0 && wsptr == 0) {
      printf("ERROR can't find either read or write user defined io\n");
    }
#endif
    vlistsd = SDSCG(A_SPTRG(vlist));
    dtvsd = SDSCG(sptr);
    ADDRTKNP(vlistsd, 1);
    ADDRTKNP(vlistsd, 1);
    ADDRTKNP(MIDNUMG(A_SPTRG(vlist)), 1);

    PUTA(-98); /* derived type with defined io */
    if (CLASSG(rsptr) && TBPLNKG(rsptr)) {
      /* FS#21015: Read is a type bound procedure. Need to resolve it to a
       * static routine.
       */
      rsptr = get_implementation(TBPLNKG(rsptr), rsptr, 0, 0);
    }
    PUTA(rsptr); /* read funcptr address */
    if (CLASSG(wsptr) && TBPLNKG(wsptr)) {
      /* FS#21015: Write is a type bound procedure. Need to resolve it to a
       * static routine.
       */
      wsptr = get_implementation(TBPLNKG(wsptr), wsptr, 0, 0);
    }
    PUTA(wsptr);                   /* write funcptr address */
    PUTA(sptr);                    /* dtv address */
    PUTA(0);                       /* dtv$sd address */
    PUTA(MIDNUMG(A_SPTRG(vlist))); /* v_list address */
    PUTA(vlistsd);                 /* v_list$sd address */
    static_cnt += 7;
  }
}

/*
 * Create a character variable which is data initialized with the name
 * of the symbol.
 */
static void
dinit_name(int sptr)
{
  char *name;
  int sym_name;
  int new_var;

  name = SYMNAME(sptr);
  sym_name = getstring(local_sname(name), strlen(name));
  new_var = getcctmp('t', sym_name, ST_UNKNOWN, DTYPEG(sym_name));
  if (STYPEG(new_var) == ST_UNKNOWN) {
    STYPEP(new_var, ST_VAR);
    DINITP(new_var, 1);
    sym_is_refd(new_var);
    dinit_put(DINIT_LOC, new_var);
    dinit_put(DINIT_STR, (INT)sym_name);
    dinit_put(DINIT_END, (INT)0);
  }
}

/*
 * emit the length and the address of a character string constant which
 * is the name of this symbol.  In order ensure that the character string is
 * initialized by the Assembler, sym_is_refd is called.
 */
static void
put_name(int sptr)
{
  char *name;
  int sym_name;

  name = SYMNAME(sptr);
  PUT(strlen(name));
  sym_name = getstring(local_sname(name), strlen(name));
  sym_is_refd(sym_name);
  if (XBIT(49, 0x100000)) {
    int new_var;
    new_var = getcctmp('t', sym_name, ST_UNKNOWN, DTYPEG(sym_name));
    sym_name = new_var;
  }
  dinit_put(DINIT_LABEL, (INT)sym_name);
}

/*------------------------------------------------------------------*/

static LOGICAL
in_local_scope(int sym, int local_scope)
{
  int scp = SCOPEG(sym);
  if (scp && STYPEG(scp) == ST_ALIAS)
    scp = SYMLKG(scp);
  return scp == local_scope;
}

static void
do_save(void)
{
  SPTR sptr;
  int nsyms;
  int stype;
  SPTR local_scope = gbl.currsub ? gbl.currsub : stb.curr_scope;

  /*  scan entire symbol table to find variables to add to .save. */

  nsyms = stb.stg_avail - 1;
  for (sptr = stb.firstusym; sptr <= nsyms; ++sptr) {
    stype = STYPEG(sptr);
    if (!ST_ISVAR(stype))
      continue;
    if (stype == ST_ARRAY && (ADJARRG(sptr) || RUNTIMEG(sptr)) &&
        (SCG(sptr) == SC_NONE || SCG(sptr) == SC_LOCAL) &&
        !CCSYMG(sptr) && !HCCSYMG(sptr)) {
      /* automatic array */
      if (SAVEG(sptr))
        error(39, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      continue;
    }
    if (SCG(sptr) == SC_LOCAL && (SAVEG(sptr) || in_save_scope(sptr)) &&
        !REFG(sptr) && !CCSYMG(sptr) && !(HCCSYMG(sptr) && ALLOCG(sptr)) &&
        in_local_scope(sptr, local_scope)) {
      int dt_dtype = DDTG(DTYPEG(sptr));
      if (
          (DTY(dt_dtype) == TY_CHAR || DTY(dt_dtype) == TY_NCHAR) &&
          !A_ALIASG(DTY(dt_dtype + 1))) {
        /* non-constant length character string */
        if (SAVEG(sptr))
          error(39, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      }
      else {
        SCP(sptr, SC_STATIC);
        SAVEP(sptr, 1);
        /* see if the DINIT flag is going to be set */
        if (DTY(dt_dtype) == TY_DERIVED && DTY(dt_dtype + 5) &&
            !POINTERG(sptr) && !ALLOCG(sptr) && !ADJARRG(sptr)) {
          DINITP(sptr, 1);
        }
        sym_is_refd(sptr);
      }
    } else if (SCG(sptr) == SC_BASED && ALLOCATTRG(sptr) &&
               in_save_scope(sptr) && in_local_scope(sptr, local_scope)) {
      SAVEP(sptr, 1);
    }
  }

}

static void
do_sequence(void)
{
  SPTR sptr;
  int nsyms;
  int stype;
  SEQL *seqp;

  if ((sem.seql.type == 0 && flg.sequence) || sem.seql.type == 's') {
    /*  scan entire symbol table to find variables to mark
     *  sequential
     */
    nsyms = stb.stg_avail - 1;
    for (sptr = stb.firstusym; sptr <= nsyms; ++sptr) {
      stype = STYPEG(sptr);
      if (ST_ISVAR(stype)) {
        if (SOCPTRG(sptr) == 0 && !ASUMSZG(sptr) && !ASSUMSHPG(sptr))
          SEQP(sptr, 1);
      } else if (stype == ST_CMBLK) {
        SEQP(sptr, 1);
      } else if (stype == ST_MEMBER) {
        SEQP(sptr, 1);
      }
    }
  } else if (sem.seql.type == 'n') {
    /*  scan entire symbol table to find variables to mark
     *  nonsequential
     */
    nsyms = stb.stg_avail - 1;
    for (sptr = stb.firstusym; sptr <= nsyms; ++sptr) {
      stype = STYPEG(sptr);
      if (ST_ISVAR(stype)) {
        if (SOCPTRG(sptr) == 0 && !ASUMSZG(sptr) && !ASSUMSHPG(sptr))
          SEQP(sptr, 0);
      } else if (stype == ST_CMBLK) {
        SEQP(sptr, 0);
      } else if (stype == ST_MEMBER) {
        SEQP(sptr, 0);
      }
    }
  }
  /*
   * traverse sequence list and process any common blocks which
   * appeared in the sequence statements
   */
  for (seqp = sem.seql.next; seqp != NULL; seqp = seqp->next) {
    sptr = seqp->sptr;
    stype = STYPEG(sptr);
    if (stype == ST_CMBLK) {
      if (seqp->type == 's')
        SEQP(sptr, 1);
      else
        SEQP(sptr, 0);
    }
  }
  /*
   * traverse common blocks and propagate storage association to the
   * members.
   */
  for (sptr = gbl.cmblks; sptr != NOSYM; sptr = SYMLKG(sptr)) {
    int elsym;

    if (SEQG(sptr))
      for (elsym = CMEMFG(sptr); elsym != NOSYM; elsym = SYMLKG(elsym))
        SEQP(elsym, 1);
  }
  /*
   * traverse sequence list and process any variables which
   * appeared in the sequence statements
   */
  for (seqp = sem.seql.next; seqp != NULL; seqp = seqp->next) {
    sptr = seqp->sptr;
    stype = STYPEG(sptr);
    if (ST_ISVAR(stype)) {
      if (seqp->type == 's') {
        SEQP(sptr, 1);
      } else {
        if (SOCPTRG(sptr) || ASUMSZG(sptr))
          error(155, 3, gbl.lineno, SYMNAME(sptr),
                "cannot appear in a NOSEQUENCE statement");
        else if (SCG(sptr) == SC_CMBLK && SEQG(CMBLKG(sptr)))
          error(155, 3, gbl.lineno,
                "Nonsequential variable in sequential common block -",
                SYMNAME(sptr));
        else
          SEQP(sptr, 0);
      }
    } else if (stype == ST_IDENT) {
      if (seqp->type == 's')
        SEQP(sptr, 1);
    } else if (stype != ST_CMBLK)
      error(155, 3, gbl.lineno, SYMNAME(sptr),
            "cannot appear in a [NO]SEQUENCE statement");
  }

}

/*------------------------------------------------------------------*/
/* return TRUE if the expression at 'ast' is composed of constants
 * the special symbol 'hpf_np$', dummy arguments, common variables, or
 * module variables, or is data initialized */

static LOGICAL available_internal;
static LOGICAL _available(int ast);

static LOGICAL
_available_size(int ast)
{
  int sptr, i, ss, ndim, asd, narg, argt, lop, firstarg;
  if (!ast)
    return TRUE;
  switch (A_TYPEG(ast)) {
  case A_ID:
    /* check for named parameter, or hpf_np$ */
    sptr = A_SPTRG(ast);
    if (STYPEG(sptr) == ST_CONST || STYPEG(sptr) == ST_PARAM)
      return TRUE;
    switch (SCG(sptr)) {
    case SC_CMBLK:
    case SC_NONE:
    case SC_LOCAL:
    case SC_DUMMY:
    case SC_STATIC:
      return TRUE;
    case SC_EXTERN:
    case SC_BASED:
    case SC_PRIVATE:
      break;
    }
    if (HCCSYMG(sptr)) /* compiler temp, must assume it'll get filled*/
      return TRUE;
    if (DINITG(sptr))
      return TRUE;
    if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_MODULE)
      return TRUE;
    if (available_internal && !INTERNALG(sptr))
      return TRUE;
    break;
  case A_MEM:
    return _available_size(A_PARENTG(ast));
  case A_SUBSCR:
    if (!_available_size(A_LOPG(ast))) {
      return FALSE;
    }
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      ss = ASD_SUBS(asd, i);
      if (!_available(ss)) {
        return FALSE;
      }
    }
    return TRUE;
  case A_TRIPLE:
    if (!_available(A_LBDG(ast)))
      return FALSE;
    if (!_available(A_UPBDG(ast)))
      return FALSE;
    if (!_available(A_STRIDEG(ast)))
      return FALSE;
    return TRUE;
  case A_CNST:
    return TRUE;
  case A_BINOP:
    if (_available_size(A_LOPG(ast)) && _available_size(A_ROPG(ast))) {
      return TRUE;
    }
    break;
  case A_UNOP:
    if (ast == astb.ptr0)
      return TRUE;
    if (ast == astb.ptr1)
      return TRUE;
    if (ast == astb.ptr0c)
      return TRUE;
    FLANG_FALLTHROUGH;
  case A_PAREN:
  case A_CONV:
    if (_available_size(A_LOPG(ast))) {
      return TRUE;
    }
    break;
  case A_FUNC:
    lop = A_LOPG(ast);
    if (!HCCSYMG(A_SPTRG(lop))) {
      return FALSE;
    }
    FLANG_FALLTHROUGH;
  case A_INTR:
    firstarg = 0;
    narg = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    if (A_TYPEG(ast) == A_INTR) {
      switch (A_OPTYPEG(ast)) {
      case I_SIZE:
      case I_LBOUND:
      case I_UBOUND:
        firstarg = 1;
        if (!_available_size(ARGT_ARG(argt, 0))) {
          return FALSE;
        }
        break;
      }
    }
    for (i = firstarg; i < narg; ++i) {
      if (!_available(ARGT_ARG(argt, i))) {
        return FALSE;
      }
    }
    return TRUE;
  } /* switch */
  return FALSE;
} /* _available_size */

static LOGICAL
_available(int ast)
{
  int sptr, i, ss, ndim, asd, narg, argt, lop, firstarg;
  if (!ast)
    return TRUE;
  switch (A_TYPEG(ast)) {
  case A_ID:
    /* check for named parameter, or hpf_np$ */
    sptr = A_SPTRG(ast);
    if (sptr == gbl.sym_nproc)
      return TRUE;
    if (STYPEG(sptr) == ST_CONST || STYPEG(sptr) == ST_PARAM)
      return TRUE;
    if (SCG(sptr) == SC_CMBLK)
      return TRUE;
    if (SCG(sptr) == SC_DUMMY)
      return TRUE;
    if (SCG(sptr) == SC_BASED) {
      if (POINTERG(sptr) && MIDNUMG(sptr)) {
        if (SCG(MIDNUMG(sptr)) == SC_CMBLK)
          return TRUE;
        if (SCG(MIDNUMG(sptr)) == SC_DUMMY)
          return TRUE;
      }
    }
    if (HCCSYMG(sptr)) /* compiler temp, must assume it'll get filled*/
      return TRUE;
    if (DINITG(sptr))
      return TRUE;
    if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_MODULE)
      return TRUE;
    if (available_internal && !INTERNALG(sptr))
      return TRUE;
    break;
  case A_MEM:
    return _available(A_PARENTG(ast));
  case A_SUBSCR:
    if (!_available(A_LOPG(ast))) {
      return FALSE;
    }
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      ss = ASD_SUBS(asd, i);
      if (!_available(ss)) {
        return FALSE;
      }
    }
    return TRUE;
  case A_TRIPLE:
    if (!_available(A_LBDG(ast)))
      return FALSE;
    if (!_available(A_UPBDG(ast)))
      return FALSE;
    if (!_available(A_STRIDEG(ast)))
      return FALSE;
    return TRUE;
  case A_CNST:
    return TRUE;
  case A_BINOP:
    if (_available(A_LOPG(ast)) && _available(A_ROPG(ast))) {
      return TRUE;
    }
    break;
  case A_UNOP:
    if (ast == astb.ptr0)
      return TRUE;
    if (ast == astb.ptr1)
      return TRUE;
    if (ast == astb.ptr0c)
      return TRUE;
    FLANG_FALLTHROUGH;
  case A_PAREN:
  case A_CONV:
    if (_available(A_LOPG(ast))) {
      return TRUE;
    }
    break;
  case A_FUNC:
    lop = A_LOPG(ast);
    if (!HCCSYMG(A_SPTRG(lop))) {
      return FALSE;
    }
    FLANG_FALLTHROUGH;
  case A_INTR:
    firstarg = 0;
    narg = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    if (A_TYPEG(ast) == A_INTR) {
      switch (A_OPTYPEG(ast)) {
      case I_SIZE:
      case I_LBOUND:
      case I_UBOUND:
        firstarg = 1;
        if (!_available_size(ARGT_ARG(argt, 0))) {
          return FALSE;
        }
        break;
      }
    }
    for (i = firstarg; i < narg; ++i) {
      if (!_available(ARGT_ARG(argt, i))) {
        return FALSE;
      }
    }
    return TRUE;
  } /* switch */
  return FALSE;
} /* _available */

/** \brief Check that sptr is declared if IMPLICIT NONE is set.

    Be careful about the situation where IMPLICIT NONE is in the host,
    but there are IMPLICIT statements in the contained subprogram.
 */
void
CheckDecl(int sptr)
{
  /* if symbol was declared, no problem */
  if (DCLDG(sptr))
    return;
#ifdef CLASSG
  if (STYPEG(sptr) == ST_ENTRY && CLASSG(sptr))
    return; /* forward reference to a type bound procedure is OK */
#endif
  /*
   *in a contained subprogram, if no IMPLICIT NONE in the
   * subprogram, and the symbol was implicitly typed due to
   * an IMPLICIT statement in the contained subprogram, no problem
   */
  if (gbl.internal > 1 && (sem.none_implicit & 0x08) == 0 &&
      was_implicit(sptr) != 0)
    return;
  /*
   * Similar to above, but in a contained subprogram of a module
   */
  if (IN_MODULE && (sem.none_implicit & 0x08) == 0 && was_implicit(sptr) != 0)
    return;
  /*
   * in a module subprogram, no IMPLICIT NONE in the module subprogram
   * (must be in the module itself), and symbol was implicitly
   * typed due to an IMPLICIT statement in the module subprogram,
   * no problem
   */
  if (gbl.internal <= 1 && sem.mod_cnt == 2 &&
      (sem.none_implicit & 0x04) == 0 && was_implicit(sptr))
    return;
  /* Subroutine reference in a module, could be defined later */
  if (sem.mod_cnt > 0 && STYPEG(sptr) == ST_PROC && sem.which_pass == 0)
    return;

  error(38, !XBIT(124, 0x20000) ? 3 : 2, gbl.lineno, SYMNAME(sptr), CNULL);
  DCLDP(sptr, 1);
} /* CheckDecl */

// Construct scope of array in bounds_contain_automatics processing.
static SPTR array_construct_scope;

static LOGICAL
search_for_auto(int ast, int *auto_found)
{
  int sptr;
  int i;

  if (A_TYPEG(ast) == A_ID) {
    sptr = A_SPTRG(ast);
    if (sptr && SCG(sptr) == SC_LOCAL && SCOPEG(sptr) == gbl.currsub &&
        DT_ISINT(DTYPEG(sptr)) && !HCCSYMG(sptr) && !PASSBYVALG(sptr) &&
        (!array_construct_scope || ENCLFUNCG(sptr) == array_construct_scope)) {
      *auto_found = TRUE;
    }
  }

  /* don't look at func args */
  if (A_TYPEG(ast) == A_FUNC || A_TYPEG(ast) == A_INTR) {
    int argt = A_ARGSG(ast);
    for (i = 0; i < A_ARGCNTG(ast); i++) {
      if (ARGT_ARG(argt, i)) {
        ast_visit(ARGT_ARG(argt, i), 1);
      }
    }
  }
  return *auto_found;
}

static LOGICAL
bnd_contains_auto(int ast)
{
  LOGICAL auto_found = FALSE;
  ast_visit(1, 1);
  ast_traverse(ast, search_for_auto, NULL, &auto_found);
  ast_unvisit();
  return auto_found;
}

static LOGICAL
bounds_contain_automatics(int sptr)
{
  int dtype = DTYPEG(sptr);
  ADSC *ad = AD_DPTR(dtype);
  int ndim = AD_NUMDIM(ad);
  int i;

  array_construct_scope = CONSTRUCTSYMG(sptr) ? ENCLFUNCG(sptr) : 0;
  for (i = 0; i < ndim; i++) {
    if (AD_LWBD(ad, i) && bnd_contains_auto(AD_LWBD(ad, i)))
      return TRUE;
    if (AD_UPBD(ad, i) && bnd_contains_auto(AD_UPBD(ad, i)))
      return TRUE;
  }
  return FALSE;
}

static void
append_to_adjarr_list(int sptr)
{
  int i;

  for (i = gbl.p_adjarr; i > NOSYM; i = SYMLKG(i)) {
    if (i == sptr) {
      return;
    }
  }

  SYMLKP(sptr, gbl.p_adjarr);
  gbl.p_adjarr = sptr;
}

static void
append_to_adjstr_list(int sptr)
{
  int i;

  for (i = gbl.p_adjstr; i > NOSYM; i = ADJSTRLKG(i)) {
    if (i == sptr) {
      return;
    }
  }

  ADJSTRLKP(sptr, gbl.p_adjstr);
  gbl.p_adjstr = sptr;
}

static void
misc_checks(void)
{
  int sptr;
  int nsyms;
  int stype;
  ITEM *itemp;
  int dtype;

  /*  scan entire symbol table */

  nsyms = stb.stg_avail - 1;
  for (sptr = stb.firstusym; sptr <= nsyms; ++sptr) {
    stype = STYPEG(sptr);
    /* if sptr is adjustable or assumed-size array, or assumed-size
       character identifier, check that it is a dummy argument */
    switch (stype) {
    case ST_IDENT:
      if (gbl.internal == 1 && SCG(sptr) == SC_NONE && ADJLENG(sptr)) {
        /* unreferenced symbol in host subprogram; set storage class */
        STYPEP(sptr, ST_VAR);
      }
      FLANG_FALLTHROUGH;
    case ST_ARRAY:
    case ST_VAR:
      if (gbl.internal == 1 && SCG(sptr) == SC_NONE) {
        /* unreferenced symbol in host subprogram; set storage class */
        sem_set_storage_class(sptr);
      }
      if (XBIT(58, 0x10000) && !F90POINTERG(sptr) && SDSCG(sptr) == 0 &&
          gbl.internal == 1 && SCG(sptr) == SC_BASED &&
          (POINTERG(sptr) || ALLOCG(sptr) || ADJARRG(sptr) || RUNTIMEG(sptr) ||
           ALLOCATTRG(sptr))) {
        /* need descriptor for contained subprograms */
        get_static_descriptor(sptr);
        if (POINTERG(sptr)) {
          get_all_descriptors(sptr);
        } else {
          trans_mkdescr(sptr);
          SECDSCP(DESCRG(sptr), SDSCG(sptr));
          if (ALLOCATTRG(sptr) && !SAVEG(sptr) && !in_save_scope(sptr))
            add_auto_dealloc(sptr);
        }
      }
      if (SCG(sptr) == SC_DUMMY && IGNORE_TKRG(sptr) && !ignore_tkr_all(sptr) && !ASSUMRANKG(sptr)) {
        if ((ASSUMSHPG(sptr) && (IGNORE_TKRG(sptr) & IGNORE_C) == 0) ||
            POINTERG(sptr) || ALLOCATTRG(sptr)) {
          error(155, 3, gbl.lineno, "IGNORE_TKR may not be specified for",
                SYMNAME(sptr));
        }
      }
      if (STYPEG(sptr) == ST_ARRAY && !IGNOREG(sptr) && !HCCSYMG(sptr) &&
          !DEVICEG(sptr) && (SCG(sptr) == SC_NONE || SCG(sptr) == SC_LOCAL) &&
          bounds_contain_automatics(sptr)) {
        error(310, 3, LINENOG(sptr),
              "Adjustable array can not have automatic bounds specifiers -",
              SYMNAME(sptr));
      }

      if (SCG(sptr) == SC_DUMMY && PASSBYVALG(sptr)) {
        PASSBYVALP(MIDNUMG(sptr),
                   0); /* clear byval flag on local (copy of arg) */
      }
      if (ADJARRG(sptr) && !IGNOREG(sptr)) {
        append_to_adjarr_list(sptr);
      }
      if (ADJLENG(sptr) && !IGNOREG(sptr)) {
        append_to_adjstr_list(sptr);
      }
#ifdef PTRRHSG
      if (!in_module && TARGETG(sptr) && !PTRRHSG(sptr)) {
        if (ALLOCATTRG(sptr)) {
          int ptr;
          ptr = MIDNUMG(sptr);
          if (ptr)
            switch (SCG(ptr)) {
            case SC_LOCAL:
            case SC_STATIC:
            case SC_PRIVATE:
              if (!gbl.internal || INTERNALG(sptr))
                TARGETP(sptr, 0);
              break;
            default:;
            }
        } else if (!POINTERG(sptr))
          switch (SCG(sptr)) {
          case SC_LOCAL:
          case SC_STATIC:
          case SC_PRIVATE:
            if (!gbl.internal || INTERNALG(sptr))
              TARGETP(sptr, 0);
            break;
          default:;
          }
      }
#endif
      /* does it need data initialization? */
      dtype = DTYPEG(sptr);
      dtype = DDTG(dtype);
      if (sem.which_pass && !IGNOREG(sptr) &&
          (gbl.internal <= 1 || INTERNALG(sptr)) &&
          (ENCLFUNCG(sptr) == 0 || CONSTRUCTSYMG(sptr) ||
           STYPEG(ENCLFUNCG(sptr)) == ST_MODULE) &&
          DTY(dtype) == TY_DERIVED &&
          (get_struct_initialization_tree(dtype) || CLASSG(sptr)) &&
          !CCSYMG(sptr) &&
          !POINTERG(sptr) && !ALLOCG(sptr) && !ADJARRG(sptr) &&
          !HCCSYMG(sptr)) {
        if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_MODULE) {
          /*
           * a derived type module variable has component
           * initializers, so its inits have already been processed
           */
          break;
        }
        if (SCG(sptr) == SC_NONE && !REFG(sptr) &&
            has_finalized_component(sptr)) {
            /* unreferenced derived type with final component needs to be
             * initialized since its final subroutine will still get called.
             */
            sem_set_storage_class(sptr);
        }
        if (gbl.rutype == RU_PROG || SCG(sptr) == SC_STATIC ||
            (SCG(sptr) == SC_LOCAL && (SAVEG(sptr) || in_save_scope(sptr)))) {
          build_typedef_init_tree(sptr, dtype);
          SAVEP(sptr, 1);
          DINITP(sptr, 1);
        } else if (SCG(sptr) == SC_LOCAL || RESULTG(sptr) ||
                   (SCG(sptr) == SC_DUMMY && INTENTG(sptr) == INTENT_OUT)) {
          init_derived_type(sptr, 0, 0);
        }
        if (SCG(sptr) == SC_LOCAL && !SAVEG(sptr) && !in_save_scope(sptr) &&
            ALLOCFLDG(DTY(dtype + 3))) {
          add_auto_dealloc(sptr);
        }
      }
      else if (RESULTG(sptr) && ALLOCATTRG(sptr) &&
               FVALG(gbl.currsub) == sptr) {
        int ast;
        ast = add_nullify_ast(mk_id(sptr));
        (void)add_stmt_after(ast, 0);
      }
      // force implicitly save for local threadprivate
      if (gbl.rutype == RU_PROG && sem.which_pass && THREADG(sptr)) {
        int midnum = 0;
        if (SCG(sptr) == SC_BASED) {
           midnum = MIDNUMG(sptr);
        }
        if (midnum && SCG(midnum) == SC_LOCAL) {
          int sdsc = SDSCG(sptr);
          int ptroff = PTROFFG(sptr);
          SAVEP(midnum, 1);
          if (sdsc) {
            SAVEP(sdsc, 1);
          }
          if (ptroff) {
            SAVEP(ptroff, 1);
          }
        }
      }
      if (gbl.rutype != RU_PROG && sem.which_pass && THREADG(sptr) &&
          !CCSYMG(sptr)) {
        if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_MODULE)
          continue;
        if (!DINITG(sptr) && !SAVEG(sptr) && !in_save_scope(sptr) &&
            (((ALLOCG(sptr) || ALLOCATTRG(sptr) || POINTERG(sptr)) &&
              (!MIDNUMG(sptr) || SCG(MIDNUMG(sptr)) != SC_CMBLK)) ||
             SCG(sptr) != SC_CMBLK))
          error(155, ERR_Severe, gbl.lineno,
                "THREADPRIVATE variable must have the SAVE attribute -",
                SYMNAME(sptr));
      }
      break;
    case ST_CMBLK:
      if (CMEMFG(sptr) == 0 && THREADG(sptr) && !CCSYMG(sptr))
        error(155, 3, gbl.lineno, "THREADPRIVATE common block is empty -",
              SYMNAME(sptr));
      break;
    }
#ifdef DEVCOPYG
    if (DEVCOPYG(sptr) && STYPEG(sptr) == ST_UNKNOWN)
      error(535, 3, gbl.lineno, SYMNAME(sptr), 0);
#endif
    if (stype == ST_ARRAY && ASUMSZG(sptr) && SCG(sptr) != SC_DUMMY &&
        SCG(sptr) != SC_BASED && !CCSYMG(sptr) && !HCCSYMG(sptr)) {
        error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    } else if (stype == ST_ARRAY && ASSUMSHPG(sptr) && SCG(sptr) != SC_DUMMY &&
               !CCSYMG(sptr) && !HCCSYMG(sptr))
      error(196, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    else if (stype == ST_IDENT && ALLOCATTRG(sptr) && !ALLOCG(sptr)) {
      /* FS#3849.  In semant.c, we allow ALLOCATTR to be set on
       * ST_IDENT symbols to avoid false errors when the ALLOCATABLE
       * statement precedes the DIMENSION statement.  But by this
       * time, an ST_IDENT symbol should not have ALLOCATTR set
       * unless ALLOC is set also.
       */
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- must be a deferred shape array");
    } else if (!CCSYMG(sptr) && !HCCSYMG(sptr) &&
               (stype == ST_VAR || stype == ST_ARRAY || stype == ST_IDENT) &&
               stype != ST_CONST && stype != ST_ENTRY &&
               SCG(sptr) != SC_DUMMY && ASSUMLENG(sptr) &&
               (DTYPEG(sptr) == DT_ASSCHAR || DTYPEG(sptr) == DT_ASSNCHAR ||
                (DTY(DTYPEG(sptr)) == TY_ARRAY &&
                 (DDTG(DTYPEG(sptr)) == DT_ASSCHAR ||
                  DDTG(DTYPEG(sptr)) == DT_ASSNCHAR))))
      error(89, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    else if ((stype == ST_IDENT || stype == ST_VAR || stype == ST_ARRAY) &&
             OPTARGG(sptr) && !HCCSYMG(sptr) && SCG(sptr) != SC_DUMMY)
      error(84, 3, gbl.lineno, SYMNAME(sptr), "- must be a dummy argument");
    else if ((stype == ST_VAR || stype == ST_ARRAY || stype == ST_IDENT) &&
             POINTERG(sptr) && HCCSYMG(sptr) && SCG(sptr) == SC_NONE)
      SCP(sptr, SC_BASED);
    if (stype == ST_PROC) {
      if (!HCCSYMG(sptr) && !CCSYMG(sptr) && !CFUNCG(sptr)) {
        if (WINNT_CALL)
          MSCALLP(sptr, 1);
#ifdef CREFP
        if (WINNT_CREF && !STDCALLG(sptr))
          CREFP(sptr, 1);
        if (WINNT_NOMIXEDSTRLEN)
          NOMIXEDSTRLENP(sptr, 1);
#endif
      }
      /*
       * tprs 3223, 3266, 3267, 3268: watch out for a dummy subroutine
       * which does not have DT.
       */
      if (SCG(sptr) == SC_DUMMY && DTYPEG(sptr) == DT_NONE &&
          FVALG(sptr) == 0 && TYPDG(sptr))
        DTYPEP(sptr, DT_INT);
    }

    if (sem.none_implicit) {
      /* check that variable has a type if:
       *  1. IMPLICIT NONE
       *  2. not a temp
       *  3. not marked as ignored
       *  4. not from containing procedure
       *  5. not from USEd module */
      int encl;
      encl = ENCLFUNCG(sptr);
      if (!HCCSYMG(sptr) && !CCSYMG(sptr) && !IGNOREG(sptr) &&
          (gbl.internal <= 1 || INTERNALG(sptr)) &&
          (encl == 0 || CONSTRUCTSYMG(sptr))) {
        switch (STYPEG(sptr)) {
        case ST_IDENT:
          if (SCG(sptr) != SC_NONE)
            break;
          FLANG_FALLTHROUGH;
        case ST_VAR:
        case ST_ARRAY:
        case ST_PARAM:
        case ST_STFUNC:
          DCLCHK(sptr);
          break;
        case ST_ENTRY:
          if (gbl.rutype == RU_FUNC) {
            if (FVALG(sptr)) {
              DCLCHK(FVALG(sptr));
            } else {
              DCLCHK(sptr);
            }
          }
          break;
        case ST_PROC:
          if (FUNCG(sptr)) {
            if (FVALG(sptr)) {
              DCLCHK(FVALG(sptr));
            } else {
              DCLCHK(sptr);
            }
          }
          break;
        default:
          break;
        }
      }
      /* set DCLD if this is a module variable,
       * since IMPLICIT NONE may not have been set in the module */
      if (encl && STYPEG(encl) == ST_MODULE && encl != gbl.currsub) {
        switch (STYPEG(sptr)) {
        case ST_VAR:
        case ST_ARRAY:
        case ST_PARAM:
          DCLDP(sptr, 1);
          break;
        case ST_PROC:
          if (FUNCG(sptr)) {
            DCLDP(sptr, 1);
          }
          break;
        default:
          break;
        }
      }
    }
  }

  /* FS3913:  Now it's safe to call sym_is_refd() for namelist items.
   * Items whose types have component initializations were initialized
   * above, so they'll correctly receive offsets into the initialized
   * data area now.
   *
   * When in a module, there could still be variables which are still
   * SC_NONE and we defer to module.c:fix_module_common() to set.
   * So we do not want do_nml_sym_is_refd() -> sym_is_refd() to occur.
   */
  if (!nml_err && !in_module)
    do_nml_sym_is_refd();

  for (itemp = sem.intent_list; itemp != NULL; itemp = itemp->next) {
    sptr = itemp->t.sptr;
    if (SCG(sptr) != SC_DUMMY) {
      error(134, 3, itemp->ast, "- intent specified for nondummy argument",
            SYMNAME(sptr));
    } else if (STYPEG(sptr) == ST_PROC) {
      error(134, 3, itemp->ast,
            "- intent specified for dummy subprogram argument", SYMNAME(sptr));
    }
  }
  /* TPR 1692: set this to NULL now, because semant_init() (which also
   * initialize the intent_list) is not called between contained subprograms
   * within another subprogram */
  sem.intent_list = NULL;

}

static void
presence_test(LOGICAL *tested_presence, int *after_std, SPTR sptr)
{
  if (!*tested_presence && SCG(sptr) == SC_DUMMY && OPTARGG(sptr)) {
    /*
     * Have an OPTIONAL INTENT(OUT) argument; need to
     * guard the initialization with "if (PRESENT(...))"
     */
    int present, aif;
    (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(sptr));
    aif = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(aif, present);
    *after_std = add_stmt_after(aif, *after_std);
    *tested_presence = TRUE;
  }
}

void
init_derived_type(SPTR sptr, int parent_ast, int wherestd)
{
  DTYPE dtype = DTYPEG(sptr);
  SPTR tagsptr;

  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  tagsptr = get_struct_tag_sptr(dtype);
  if (tagsptr > NOSYM) {
    int std = CONSTRUCTSYMG(sptr) ? BLOCK_ENTRY_STD(sptr) : wherestd;
    LOGICAL need_ENDIF = FALSE;
    int new_ast = 0;

    if (SCG(sptr) == SC_DUMMY &&
        !ALLOCATTRG(sptr) &&
        (ALLOCFLDG(tagsptr) || allocatable_member(tagsptr)) &&
        !RESULTG(sptr) &&
        FVALG(gbl.currsub) != sptr) {
      presence_test(&need_ENDIF, &std, sptr);
      std = gen_dealloc_for_sym(sptr, std);
    }

    if (CLASSG(sptr)) {
      int descr_ast = find_descriptor_ast(sptr, parent_ast);
      if (descr_ast <= 0) {
        SPTR desc_sptr = get_static_type_descriptor(sptr);
        if (desc_sptr > NOSYM)
          descr_ast = mk_id(desc_sptr);
      }
      if (descr_ast > 0) {
        int func_ast = mk_id(sym_mkfunc_nodesc(mkRteRtnNm(RTE_init_from_desc),
                                               DT_NONE));
        int argt = mk_argt(3);
        new_ast = mk_func_node(A_CALL, func_ast, 3, argt);
        ARGT_ARG(argt, 0) = mk_id(sptr);
        ARGT_ARG(argt, 1) = descr_ast;
        ARGT_ARG(argt, 2) =
          mk_unop(OP_VAL, mk_cval(rank_of_sym(sptr), DT_INT4), DT_INT4);
      }
    }

    if (new_ast == 0) {
      /* Not using RTE_init_from_desc; initialize via prototype assignment */
      SPTR prototype = get_dtype_init_template(dtype);
      if (prototype > NOSYM)
        new_ast = mk_assn_stmt(mk_id(sptr), mk_id(prototype), dtype);
    }

    if (new_ast > 0) {
      presence_test(&need_ENDIF, &std, sptr);
      std = add_stmt_after(new_ast, std);
    }
    if (need_ENDIF)
      add_stmt_after(mk_stmt(A_ENDIF, 0), std);
  }
}

/*------------------------------------------------------------------*/

void rw_semant_state(RW_ROUTINE, RW_FILE)
{
  int nw;

  RW_SCALAR(sem.none_implicit);
  symutl.none_implicit = sem.none_implicit;
  RW_SCALAR(stb.curr_scope);
  RW_SCALAR(sem.scope_level);
  if (!sem.scope_stack) {
    fseek(fd, sizeof(SCOPESTACK) * (sem.scope_level + 1), 1);
  } else {
    if (ISREAD()) {
      NEED(sem.scope_level + 1, sem.scope_stack, SCOPESTACK, sem.scope_size,
           sem.scope_level + 10);
    }
    RW_FD(sem.scope_stack, SCOPESTACK, sem.scope_level + 1);
  }
  RW_SCALAR(sem.eqvlist);
  RW_SCALAR(sem.eqv_avail);
  if (sem.eqvlist > 0) {
    if (ISREAD()) {
      NEED(sem.eqv_avail, sem.eqv_base, EQVV, sem.eqv_size, sem.eqv_avail + 50);
    }
    RW_FD(sem.eqv_base, EQVV, sem.eqv_avail);
  }
  RW_SCALAR(sem.eqv_ss_avail);
  if (sem.eqv_ss_avail > 1) {
    if (ISREAD()) {
      NEED(sem.eqv_ss_avail, sem.eqv_ss_base, int, sem.eqv_ss_size,
           sem.eqv_ss_avail + 50);
    }
    RW_FD(sem.eqv_ss_base, int, sem.eqv_ss_avail);
  }
}
