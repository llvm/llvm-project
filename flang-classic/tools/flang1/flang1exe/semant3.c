/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file semant3.c
    \brief This file contains part 3 of the compiler's semantic actions
    (also known as the semant3 phase).
*/

#include "gbldefs.h"
#include "gramsm.h"
#include "gramtk.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "semstk.h"
#include "ast.h"
#include "dinit.h"
#include "pragma.h"
#include "fdirect.h"
#include "rte.h"
#include "hpfutl.h"
#include "lower.h"
#include "rtlRtns.h"
#include "pd.h"

static LOGICAL alloc_error = FALSE;
static int alloc_source;
static int orig_alloc_source;
static int typed_alloc;
static int do_label;
static int construct_name;
static int last_std;

#ifdef FLANG_SEMANT3_UNUSED
static void add_nullify(int);
#endif
static void check_do_term();
static int gen_logical_if_expr(SST *);
static int dealloc_tmp(int);
static void chk_and_rewrite_cmplxpart_assn(SST *rhs, SST *lhs);
static LOGICAL init_exprs_idx_dependent(int, int);
static int gen_derived_arr_init(int arr_dtype, int strt_std, int end_std);
static int convert_to_block_forall(int old_forall_ast);

static int find_non_tbp(char *);
#ifdef FLANG_SEMANT3_UNUSED
static int gen_sourced_allocation(int astdest, int astsrc);
#endif

static int construct_association(int lhs_sptr, SST *rhs, int stmt_dtype,
                                 LOGICAL is_class);
static void end_association(int sptr);
static int get_sst_named_whole_variable(SST *rhs);
static int get_derived_type(SST *, LOGICAL);

#define IN_OPENMP_ATOMIC (sem.mpaccatomic.ast && !(sem.mpaccatomic.is_acc))

/** \brief semantic actions - part 3.
 *  \param rednum   reduction number
 *  \param top      top of stack after reduction
 */
void
semant3(int rednum, SST *top)
{
  SPTR sptr, sptr1, sptr2;
  DTYPE dtype, dtype2;
  int shape, stype, symi;
  int i = 0, j, msg;
  int begin, count;
  int opc;
  int lab;
  SST *e1, sst;
  INT rhstop;
  ITEM *itemp, *itemp1;
  INT conval;
  int doif;
  DOINFO *doinfo;
  char name[16];
  int dum;
  SWEL *swel;
  int ast, ast1, ast2, ast3, lop;
  int astlab;
  int astli;
  int std;
  char *np;
  const char *s;
  int numdim;
  int (*p_cmp)(int, int);
  int arith_if_expr;
  int bef;
  int rhs_ast;
  int o_ast;
  int mold_or_src;
  FtnRtlEnum rtlRtn;
  TYPE_LIST *types, *prev, *curr;
  switch (rednum) {

  /* ------------------------------------------------------------------ */
  /*
   *      <simple stmt> ::= <assignment> |
   */
  case SIMPLE_STMT1:
    /* <assignment> sets sem.pgphase */
    break;
  /*
   *      <simple stmt> ::= <assigned GOTO> |
   */
  case SIMPLE_STMT2:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <GOTO assignment> |
   */
  case SIMPLE_STMT3:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <computed GOTO> |
   */
  case SIMPLE_STMT4:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <arith IF> |
   */
  case SIMPLE_STMT5:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <call>     |
   */
  case SIMPLE_STMT6:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <return>   |
   */
  case SIMPLE_STMT7:
    goto executable_shared;
  /*
   *      <simple stmt> ::= CONTINUE   |
   */
  case SIMPLE_STMT8:
    ast = mk_stmt(A_CONTINUE, 0);
    SST_ASTP(LHS, ast);
    goto executable_shared;
  /*
   *      <simple stmt> ::= <stop stmt> |
   */
  case SIMPLE_STMT9:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <pause stmt> |
   */
  case SIMPLE_STMT10:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <allocation stmt> |
   */
  case SIMPLE_STMT11:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <IO stmt>
   */
  case SIMPLE_STMT12:
    if (flg.smp || flg.accmp) {
      ast = begin_call(A_CALL, sym_mkfunc_nodesc("_mp_ecs_nest", DT_NONE), 0);
      SST_ASTP(LHS, ast);
    } else if (XBIT(125, 0x1)) {
      /*
       * unconditionally call the routine which marks the end
       * of an i/o critical section.  Note that conditional calls
       * are necessary when END and/or ERR are present.
       */
      ast =
          begin_call(A_CALL, sym_mkfunc(mkRteRtnNm(RTE_f90io_end), DT_NONE), 0);
      SST_ASTP(LHS, ast);
    }
    goto executable_shared;
  /*
   *	<simple stmt> ::= <exit stmt> |
   */
  case SIMPLE_STMT13:
    goto executable_shared;
  /*
   *	<simple stmt> ::= <cycle stmt>
   */
  case SIMPLE_STMT14:
    goto executable_shared;
  /*
   *	<simple stmt> ::= <pointer assignment> |
   */
  case SIMPLE_STMT15:
    goto executable_shared;
  /*
   *	<simple stmt> ::= <nullify stmt> |
   */
  case SIMPLE_STMT16:
    goto executable_shared;
  /*
   *      <simple stmt> ::= <where clause> <assignment> |
   */
  case SIMPLE_STMT17:
    ast = mk_stmt(A_WHERE, 0);
    A_IFEXPRP(ast, SST_ASTG(RHS(1)));
    A_IFSTMTP(ast, SST_ASTG(RHS(2)));
    add_stmt(ast);
    gen_deallocate_arrays(); /* dealloc temp arrays generated for <simple stmt>
                              */
    SST_ASTP(LHS, 0);
    if (sem.doif_depth > 0 && DI_ID(sem.doif_depth) == DI_WHERE)
      --sem.doif_depth;
    goto executable_shared;
  /*
   *	<simple stmt> ::= <forall clause> <forall assn stmt> |
   */
  case SIMPLE_STMT18:
    ast = SST_ASTG(RHS(1));
    A_IFSTMTP(ast, SST_ASTG(RHS(2)));
    if (STD_NEXT(DI_FORALL_LASTSTD(sem.doif_depth)) &&
        STD_NEXT(DI_FORALL_LASTSTD(sem.doif_depth)) != astb.std.stg_avail) {
      ast = convert_to_block_forall(ast);
      SST_ASTP(LHS, ast); /* ast is ENDFORALL */
    }
    if (flg.smp) {
      DI_NOSCOPE_FORALL(sem.doif_depth) = 0;
      check_no_scope_sptr();
    }
    doif = sem.doif_depth--;
    direct_loop_end(DI_LINENO(doif), gbl.lineno);
    sem.pgphase = PHASE_EXEC;
    for (symi = DI_IDXLIST(doif); symi; symi = SYMI_NEXT(symi))
      pop_sym(SYMI_SPTR(symi));
    goto forall_shared;
  /*
   *	<simple stmt> ::= <smp stmt>
   */
  case SIMPLE_STMT19:
    if (gbl.currsub) {
      if (PUREG(gbl.currsub))
        error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
              "- PURE subprograms may not contain OpenMP directives");
      else if (ELEMENTALG(gbl.currsub))
        error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
              "- ELEMENTAL subprograms may not contain OpenMP directives");
    }
    FLANG_FALLTHROUGH;
  /*
   *	<simple stmt> ::= <pragma stmt> |
   */
  case SIMPLE_STMT20:
  /*  fall thru  */
  /*
   *      <simple stmt> ::= <accel stmt>
   */
  case SIMPLE_STMT21:
  /*  fall thru  */
  /*
   *	<simple stmt> ::= <kernel stmt>
   */
  case SIMPLE_STMT22:
  /*  fall thru  */
  /*
   *	<simple stmt> ::= <error stop stmt>
   */
  case SIMPLE_STMT23:
    /*  fall thru  */

  executable_shared:
    sem.pgphase = PHASE_EXEC;
    break;
  /* ------------------------------------------------------------------ */
  /*
   *      <assignment> ::= <psfunc> <var ref> <psfunc> = <expression>
   */
  case ASSIGNMENT1:
    if (SST_IDG(RHS(2)) == S_LVALUE || SST_IDG(RHS(2)) == S_EXPR)
      sptr1 = SST_LSYMG(RHS(2));
    else if (SST_IDG(RHS(2)) == S_DERIVED || SST_IDG(RHS(2)) == S_IDENT)
      sptr1 = SST_SYMG(RHS(2));
    else if (SST_IDG(RHS(2)) == S_SCONST) {
      (void)mkarg(RHS(2), &dum);
      sptr1 = SST_SYMG(RHS(2));
    } else {
      sptr1 = 0;
    }
    if (sptr1) {
      if (sem.new_param_dt) {
        dtype = SST_DTYPEG(RHS(5));
        if (sem.new_param_dt == dtype && eq_dtype(DTYPEG(sptr1), dtype)) {
          /* LHS is compatible with RHS. Use LHS PDT */
          SST_DTYPEP(RHS(5), DTYPEG(sptr1));
        }
      }
      ast = SST_ASTG(RHS(5));
      if (ast && ALLOCATTRG(sptr1) && A_TYPEG(ast) == A_FUNC) {
        sptr = sym_of_ast(A_LOPG(ast));
        if (sptr && STYPEG(sptr) == ST_PROC) {
          ALLOCASNP(sptr, 1);
        }
      }
    }

    dtype = SST_DTYPEG(RHS(5));
    if (SST_IDG(RHS(5)) == S_IDENT) {
      sptr = SST_SYMG(RHS(5));
    } else if (SST_IDG(RHS(5)) == S_LVALUE) {
      sptr = SST_LSYMG(RHS(5));
    } else {
      sptr = 0;
    }
    if (STYPEG(sptr) == ST_PROC && IS_TBP(sptr)) {
      ast = SST_ASTG(RHS(5));
      if (!XBIT(68, 0x80) && A_TYPEG(ast) != A_FUNC) {
        /* Disallow type bound procedure extension noted below by default */
        char name[MAXIDLEN];
        char *pos;
        strcpy(name, SYMNAME(sptr));
        pos = strstr(name, "$tbp");
        if (pos != NULL) {
          *pos = '\0';
        }
        error(1216, 3, gbl.lineno, name, CNULL);
      } else if ((A_TYPEG(ast) == A_FUNC || XBIT(68, 0x80)) && sem.tbp_arg) {
        /* Process passed object argument if we have a type bound procedure
         * call (e.g., z = x%foo()). If XBIT(68, 0x80) is enabled, we treat
         * expressions such as z = x%foo the same as z = x%foo() (this is as an
         * extension).
         */
        int mem;
        SST *sp;
        ITEM *tbpArg;
        sptr1 = 0;

        get_implementation(TBPLNKG(sptr), sptr, 0, &sptr1);
        if (!NOPASSG(sptr1)) {
          tbpArg = pop_tbp_arg();
          sp = tbpArg->t.stkp;
          if (SST_IDG(sp) == S_LVALUE || SST_IDG(sp) == S_EXPR)
            sptr1 = SST_LSYMG(sp);
          else if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
            sptr1 = SST_SYMG(sp);
          else if (SST_IDG(sp) == S_SCONST) {
            (void)mkarg(sp, &dum);
            sptr1 = SST_SYMG(sp);
          } else {
            sptr1 = 0;
#if DEBUG
            interr("semant3: bad tbp passed object argument", SST_IDG(sp), 3);
#endif
          }
          dtype = DTYPEG(sptr1);
          if (DTY(dtype) == TY_ARRAY)
            dtype = DTY(dtype + 1);
          sptr = get_implementation(dtype, sptr, 0, &mem);
          SST_SYMP(RHS(5), sptr);
          SST_LSYMP(RHS(5), sptr);
          SST_DTYPEP(RHS(5), DTYPEG(sptr));
          (void)mkvarref(RHS(5), tbpArg);
        }
      }
    }

    SST_ASTP(LHS, 0); /* initialize to zero */
    if (SST_IDG(RHS(2)) == S_CONST) {
      sptr = SST_ERRSYMG(RHS(2));
      error(72, 3, gbl.lineno, "constant or parameter",
            (sptr > 0 && sptr < stb.stg_avail) ? SYMNAME(sptr) : CNULL);
      break;
    }
    if (SST_IDG(RHS(2)) == S_LVALUE || SST_IDG(RHS(2)) == S_IDENT) {
      /* check for assignment to array parameter */
      sptr = SST_SYMG(RHS(2));
      if (sptr) {
        if (STYPEG(sptr) == ST_ENTRY || STYPEG(sptr) == ST_PROC)
          /* avoid using PARAMG with these types of symbols --
           * PARAM overlays INMODULE.
           */
          ;
        else if (PARAMG(sptr)) {
          if (PARAMVALG(sptr)) {
            /* get the original parameter symbol */
            if (A_SPTRG(PARAMVALG(sptr)))
              sptr = A_SPTRG(PARAMVALG(sptr));
          }
          error(72, 3, gbl.lineno, "array or derived type parameter",
                (sptr > 0 && sptr < stb.stg_avail) ? SYMNAME(sptr) : CNULL);
          break;
        }
      }
    }

    if (SST_IDG(RHS(2)) == S_STFUNC) {
      /* a statement function reference */
      sptr = SST_SYMG(RHS(2));
      stype = STYPEG(sptr);
      if (IS_INTRINSIC(stype)) {
        if ((sptr = newsym(sptr)) <= NOSYM) {
          /* Symbol frozen as an intrinsic, ignore here */
          break;
        }
      } else if (stype != ST_IDENT && stype != ST_UNKNOWN) {
        error(43, 3, gbl.lineno, "symbol", SYMNAME(sptr));
        break;
      }
      if (SST_IDG(RHS(5)) == S_CONST && SST_DTYPEG(RHS(5)) == DT_WORD) {
        if (DT_ISWORD(SST_DTYPEG(RHS(2)))) {
          SST_DTYPEP(RHS(5), SST_DTYPEG(RHS(2)));
        } else {
          SST_DTYPEP(RHS(5), DT_INT);
          error(426, 3, gbl.lineno, NULL, CNULL);
        }
      }
      dtype = DTYPEG(sptr);
      if (dtype != SST_DTYPEG(RHS(5)))
        cngtyp(RHS(5), dtype);
      if (!sem.stfunc_error)
        ast = define_stfunc(sptr, SST_ENDG(RHS(2)), RHS(5));
      ast = 0; /* don't add A_STFUNC ast to STD */
      sem.pgphase = PHASE_SPEC;
      sem.in_stfunc = FALSE;
    } else if (is_intrinsic_opr(OP_ST, LHS, RHS(2), RHS(5), 0)) {
      ast = SST_ASTG(LHS);
      sem.pgphase = PHASE_EXEC;
      if (sem.doif_depth > 0) {
        doif = sem.doif_depth;
        if (DI_ID(doif) == DI_WHERE || DI_ID(doif) == DI_ELSEWHERE) {
          error(440, 3, gbl.lineno, CNULL, CNULL);
        }
      }
    } else if (SST_IDG(RHS(5)) == S_EXPR &&
               A_TYPEG(SST_ASTG(RHS(5))) == A_INTR &&
               A_OPTYPEG(SST_ASTG(RHS(5))) == I_NULL) {
      ast = SST_ASTG(RHS(5));
      if (SST_IDG(RHS(2)) == S_IDENT) {
        sptr = SST_SYMG(RHS(2));
      } else {
        sptr = memsym_of_ast(SST_ASTG(RHS(2)));
      }
      if (!ALLOCATTRG(sptr)) {
        error(467, 3, gbl.lineno, NULL, CNULL);
      }
      mkexpr1(RHS(2));
      ast = mk_assn_stmt(SST_ASTG(RHS(2)), ast, SST_DTYPEG(RHS(2)));
      *LHS = *RHS(2);
    } else {
      if (SST_IDG(RHS(5)) == S_SCONST) {
        dtype = SST_DTYPEG(RHS(5));
        if (SST_IDG(RHS(2)) == S_IDENT && SST_DTYPEG(RHS(2)) == dtype) {
          /* sptr = SCONST */
          set_assn(SST_SYMG(RHS(2)));
          mklvalue(RHS(2), 1);
          sptr = init_derived_w_acl(SST_SYMG(RHS(2)), SST_ACLG(RHS(5)));
          ast = 0; /* don't add A_SCONST ast to STD */

          *LHS = *RHS(2);
          sem.pgphase = PHASE_SPEC;
          SST_ASTP(LHS, ast);
          break; /* get out */
        } else {
          /* tmp = SCONST */
          int first_acl_std = astb.std.stg_avail;
          if (SST_ACLG(RHS(5)))
            sptr = init_derived_w_acl(0, SST_ACLG(RHS(5)));
          if (sem.doif_depth && DI_ID(sem.doif_depth) == DI_FORALL &&
              init_exprs_idx_dependent(first_acl_std, astb.std.stg_avail)) {

            /* generate a tmp array of derived type and initialize it
             * in forall loops */
            ast = gen_derived_arr_init(DTYPEG(sptr1), first_acl_std,
                                       astb.std.stg_avail);

            /* the above tmp array of derived type becomes the src of
             * original forall assignment
             */
            SST_IDP(RHS(5), S_EXPR);
            SST_ASTP(RHS(5), ast);
          } else {
            mkident(RHS(5));
            SST_SYMP(RHS(5), sptr);
          }
        }
      }
      if (SST_IDG(RHS(2)) == S_DERIVED) {
        sptr = SST_SYMG(RHS(2));
      }

      /* iso_c derived type c_ptr = c_loc(x) or
                        c_funptr =  c_funcloc(x)
          turns into (scalar, LHS) c_ptr  = %LOC(x)

      */
      if (SST_IDG(RHS(2)) == S_LVALUE || SST_IDG(RHS(2)) == S_IDENT) {
        /* check for assignment to array parameter */
        sptr = SST_SYMG(RHS(2));
        rhs_ast = SST_ASTG((RHS(5)));
        if (SST_ACLG(RHS(5)))
          rhs_ast = 0;
        else
          rhs_ast = SST_ASTG((RHS(5)));
        if (is_iso_cptr(DTYPEG(sptr)) && is_iso_cloc(rhs_ast)) {
          if (A_OPTYPEG(rhs_ast) != I_C_FUNLOC &&
              A_TYPEG(ARGT_ARG(A_ARGSG(rhs_ast), 0)) == A_ID &&
              !TARGETG(A_SPTRG(ARGT_ARG(A_ARGSG(rhs_ast), 0))) &&
              !POINTERG(A_SPTRG(ARGT_ARG(A_ARGSG(rhs_ast), 0))))
            /* TARGET attribute is required for c_loc param */
            errwarn(468);
          else {
            /* for C_FUNLOC, the argument must either be
             * interoperable (i.e., have the BIND attribute)
             * or is a procedure pointer associated with an
             * interoperable procedure.
             ******  TBD  ******
             */
            ;
          }
          if (A_OPTYPEG(rhs_ast) == I_C_LOC &&
              A_TYPEG(ast1 = ARGT_ARG(A_ARGSG(rhs_ast), 0)) == A_ID &&
              (DTY(dtype = A_DTYPEG(ast1)) == TY_CHAR ||
               DTY(dtype) == TY_NCHAR) &&
              A_ALIASG(DTY(dtype + 1)) != 0 &&
              CONVAL2G(A_SPTRG(A_ALIASG(DTY(dtype + 1)))) > 1)
            /* CHAR arg to C_LOC must have length of one */
            errwarn(469);

          SST_IDP(RHS(5), S_EXPR);
          SST_ASTP(RHS(5),
                   mk_unop(OP_LOC, ARGT_ARG(A_ARGSG(rhs_ast), 0), DT_PTR));
          SST_SHAPEP(RHS(5), 0);
          (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_loc), DT_ADDR);
        }
      }
      if (SST_IDG(RHS(2)) == S_LVALUE || SST_IDG(RHS(2)) == S_EXPR) {
        sptr = SST_LSYMG(RHS(2));
      } else {
        sptr = SST_SYMG(RHS(2));
      }
      if (CLASSG(sptr) && !MONOMORPHICG(sptr) && !ALLOCATTRG(sptr)) {
        error(1217, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL);
      }
      if (STYPEG(sptr) == ST_ENTRY && FVALG(sptr)) {
        char buffer[256];
        (void)snprintf(buffer, sizeof(buffer), "- should use %s",
                       SYMNAME(FVALG(sptr)));
        error(72, 3, gbl.lineno, SYMNAME(sptr), buffer);
        sptr = FVALG(sptr);
      }
      chk_and_rewrite_cmplxpart_assn(RHS(2), RHS(5));

      /* really an assignment statement */
      if (FVALG(gbl.currsub) != sptr &&
          strcmp(SYMNAME(FVALG(sptr)), SYMNAME(gbl.currsub)) == 0 &&
          strcmp(SYMNAME(sptr), SYMNAME(gbl.currsub)) == 0 &&
          strcmp(SYMNAME(sptr), SYMNAME(FVALG(gbl.currsub))) == 0) {
        /* replace with correct LHS symbol */
        sptr = gbl.currsub;
        if (SST_IDG(RHS(2)) == S_LVALUE || SST_IDG(RHS(2)) == S_EXPR) {
          SST_LSYMP(RHS(2), sptr);
        } else {
          SST_SYMP(RHS(2), sptr);
        }
      }
      if ((!ALLOCATTRG(sptr1) || !XBIT(54, 0x1)) && !POINTERG(sptr1) &&
          has_finalized_component(sptr1)) {
        /* LHS has finalized component(s). Need to finalize them before
         * (re-)assigning to them. If LHS is allocatable and we're using
         * F2003 allocatation semantics, then finalization
         * is performed with (automatic) deallocation. If the result is
         * pointer, then we do not finalize the object (the language spec
         * indicates that it processor dependent whether such objects are
         * finalized).
         */
        int std = add_stmt(mk_stmt(A_CONTINUE, 0));
        int parent = SST_ASTG(RHS(2));
        if (A_TYPEG(parent) != A_MEM &&
            (A_TYPEG(parent) != A_SUBSCR || A_TYPEG(A_LOPG(parent)) != A_MEM)) {
          parent = 0;
        }
        gen_finalization_for_sym(sptr1, std, parent);
      }
      if (use_opt_atomic(sem.doif_depth) && sem.mpaccatomic.seen &&
          !sem.mpaccatomic.is_acc) {
        sem.mpaccatomic.accassignc++;
        ast = do_openmp_atomics(RHS(2), RHS(5));
        if (ast) {
          ast = assign(RHS(2), RHS(5));
          add_stmt(ast);
        }
        ast = 0;
        SST_ASTP(LHS, ast);
        sem.pgphase = PHASE_EXEC;
        goto end_stmt;
      } else if (sem.mpaccatomic.seen && IN_OPENMP_ATOMIC) {
        validate_omp_atomic(RHS(2), RHS(5));
        if (sem.mpaccatomic.action_type != ATOMIC_CAPTURE)
          sem.mpaccatomic.seen = FALSE;
      }

      ast = assign(RHS(2), RHS(5));
      *LHS = *RHS(2);
      /* assign() will return 0 if the rhs is an array-valued function
       * for which the lhs becomes the result argument.
       */
      if (ast) {
        if (sem.doif_depth > 0) {
          doif = sem.doif_depth;
          if (DI_ID(doif) == DI_WHERE || DI_ID(doif) == DI_ELSEWHERE) {
            int lhs = A_DESTG(ast);
            shape = A_SHAPEG(lhs);
            if (!shape || SHD_NDIM(shape) != DI_SHAPEDIM(doif))
              error(155, 3, gbl.lineno, NULL,
                    "Array assignment "
                    "and block WHERE mask expression do not conform");
          }
        }
      }
      sem.pgphase = PHASE_EXEC;

      /* if the statement is still in the OpenACC atomic region.
         we have to check whether this assignment stmt is illegal
         to present in the atomic region.

         If sem.mpaccatomic.pending is reset, an assignment
         statement has been processed.

         In ATOMIC CAPTURE region, two or more assignment statements
         are allowed.
         by daniel tian
      */

      if (sem.mpaccatomic.is_acc == TRUE)
        sem.mpaccatomic.accassignc++;

      if (sem.atomic[0])
        sem.atomic[2] = TRUE;
      if (sem.mpaccatomic.pending &&
          sem.mpaccatomic.action_type != ATOMIC_CAPTURE) {
        sem.mpaccatomic.apply = TRUE;
        sem.mpaccatomic.pending = FALSE;
      }
    }
  end_stmt:
    if (A_TYPEG(SST_ASTG(LHS)) == A_MEM) {
      sptr = memsym_of_ast(SST_ASTG(LHS));
      if (!USEKINDG(sptr) && KINDG(sptr)) {
        error(155, 3, gbl.lineno, "Illegal assignment to type parameter",
              SYMNAME(sptr));
      }
    }
    SST_ASTP(LHS, ast);
    break;

  /*
   *      <psfunc> ::=
   */
  case PSFUNC1:
    /*
     * This rule exists for the sole purpose of marking that we are
     * on the left of an = sign and that there is a potential statement
     * function definition.
     * This makes <var ref> processing easier.
     */
    if (!sem.psfunc) {
      sem.psfunc = TRUE; /* Toggle */
    } else {
      sem.psfunc = FALSE; /* Toggle */
      BZERO((char *)(&sem.arrfn), char, sizeof(sem.arrfn));
      /*
       * collect information for an array-valued function if one
       * appears in the rhs.  Will be looking for opportunities
       * for optimizing A = func(...) where the temp passed to
       * the function for the return value is replaced by A.
       */
      sem.arrfn.try = 1;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <assigned GOTO> ::= GOTOX <ident> <optional comma> ( <label list> ) |
   */
  case ASSIGNED_GOTO1:
    count = SST_COUNTG(RHS(5));
    goto assigned_goto;
  /*
   *      <assigned GOTO> ::= GOTOX <ident>
   */
  case ASSIGNED_GOTO2:
    count = 0;
  assigned_goto:
    check_do_term();
    /*
     * Ideally, <ident> would be <var ref>; however the grammar is not
     * LR(1) because of '(' lookahead.  So, this code duplicates most
     * of the actions performed by <var ref> ::= <ident>.
     */
    if (not_in_forall("Assigned GOTO"))
      break;
    sptr = refsym((int)SST_SYMG(RHS(2)), OC_OTHER);
    if (STYPEG(sptr) == ST_PARAM) {
      error(107, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      break;
    }
    SST_SYMP(RHS(2), sptr);
    /* Pick up the data type from the symbol table entry which was
     * either: 1) explicitly set by the user, or 2) has the current
     * default value.
     */
    if (DTYPEG(sptr) == DT_NONE) {
      /* This is only okay if identifier is an intrinsic,
       * generic, or predeclared.  This means the function was
       * used as an identifier without parenthesized arguments.
       */
      if (IS_INTRINSIC(STYPEG(sptr)))
        setimplicit(sptr);
    }
    dtype = DTYPEG(sptr);
    if (!legal_labelvar(dtype)) {
      error(107, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      break;
    }
    SST_DTYPEP(RHS(2), dtype);
    if (SCG(sptr) == SC_NONE)
      SCP(sptr, SC_LOCAL);
    if (flg.xref)
      xrefput(sptr, 'r');
    /*
     * For 64-byte targets, create a temp variable iff the type
     * of the user variable is integer*4.
     * When targeting llvm, always create a temp variable of
     * ptr-size integer type.
     */
    sptr = mklabelvar(RHS(2));
    if (count) {
      start_astli();
      for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
        astli = add_astli();
        ASTLI_AST(astli) = mk_label(itemp->t.sptr);
      }
      astli = ASTLI_HEAD;
    } else
      astli = 0;
    ast = mk_stmt(A_AGOTO, 0);
    lop = SST_ASTG(RHS(2));
    A_LOPP(ast, lop);
    A_LISTP(ast, astli);
    SST_ASTP(LHS, ast);
    gbl.asgnlbls = -1;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <label list> ::= <label list> , <reflabel> |
   */
  case LABEL_LIST1:
    rhstop = 3;
    goto label_list;
  /*
   *      <label list> ::= <reflabel>
   */
  case LABEL_LIST2:
    rhstop = 1;
  label_list:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.sptr = SST_SYMG(RHS(rhstop));
    if (rhstop == 1) {
      SST_BEGP(LHS, itemp);
      SST_COUNTP(LHS, 1);
    } else {
      SST_ENDG(RHS(1))->next = itemp;
      SST_COUNTP(LHS, SST_COUNTG(RHS(1)) + 1);
    }
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<reflabel> ::= <label>
   */
  case REFLABEL1:
    RFCNTI(SST_SYMG(RHS(1)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <label> ::= <integer>
   */
  case LABEL1:
    conval = SST_CVALG(RHS(1));
    sptr = declref(getsymf(".L%05ld", (long)conval), ST_LABEL, 'r');
    if (conval == 0) {
      error(18, 3, gbl.lineno, "0", CNULL);
      DEFDP(sptr, 1);
    } else if (conval > 99999) {
      error(18, 3, gbl.lineno, "- length exceeds 5 digits", CNULL);
      DEFDP(sptr, 1);
    }
    SST_SYMP(LHS, sptr);
    /*
     * link into list of referenced labels if not already
     * there
     */
    if (SYMLKG(sptr) == NOSYM) {
      SYMLKP(sptr, sem.flabels);
      sem.flabels = sptr;
    }
    SST_ASTP(LHS, mk_label(sptr));
    SST_PARENP(LHS, 0);
    if (scn.stmtyp != TK_ASSIGN)
      /* mark the label as one which must label a branch target
       * statement.  This flag can be set in this production since
       * format labels are not parsed as <label>.
       */
      TARGETP(sptr, 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <GOTO assignment> ::= ASSIGN <reflabel> TO <var ref>
   */
  case GOTO_ASSIGNMENT1:
    if (not_in_forall("ASSIGN statement"))
      break;
    dtype = SST_DTYPEG(RHS(4));
    if (SST_IDG(RHS(4)) != S_IDENT || !legal_labelvar(dtype)) {
      error(107, 3, gbl.lineno, "in ASSIGN statement", CNULL);
      break;
    }
    sptr = SST_SYMG(RHS(2));
    (void)mklvalue(RHS(4), 1); /* <var ref> is a scalar variable */
    (void)mklabelvar(RHS(4));
    /*
     * For now, prevent deleting the ASSIGN labels of basic blocks by
     * the backend.  Currently, deletion could occur because the RFCNTs
     * of the set of the possible target labels are not updated by the
     * backend at each GOTO I statement.
     */
    VOLP(sptr, 1);
    sptr1 = A_SPTRG(SST_ASTG(RHS(4)));
    ASSNP(sptr, 1); /* label appears in an ASSIGN */
    set_assn(sptr1);
    ast = mk_stmt(A_ASNGOTO, 0);
    A_DESTP(ast, SST_ASTG(RHS(4)));
    A_SRCP(ast, SST_ASTG(RHS(2)));
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <computed GOTO> ::= GOTOX ( <label list> ) <optional comma> <etmp exp>
   */
  case COMPUTED_GOTO1:
    if (not_in_forall("Computed GOTO"))
      break;
    chk_scalartyp(RHS(6), DT_INT, TRUE);
    /* allocate and initialize cgoto list header; this is equivalent to
     * pgc's switch list.
     */
    begin = sem.switch_avl++; /* relative ptr to header */
    NEED(sem.switch_avl, switch_base, SWEL, sem.switch_size,
         sem.switch_size + 300);
    swel = switch_base + begin;
    count = SST_COUNTG(RHS(3));
    sptr = getlab(); /* "default" label */
    RFCNTI(sptr);
    swel->val = count;
    swel->clabel = sptr;
    conval = 0;
    start_astli();
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      /*
       * allocate switch list element; it will automatically be in
       * sorted order
       */
      i = sem.switch_avl++;
      swel->next = i;
      NEED(sem.switch_avl, switch_base, SWEL, sem.switch_size,
           sem.switch_size + 300);
      swel = switch_base + i;
      swel->val = ++conval;
      swel->clabel = itemp->t.sptr;
      astli = add_astli();
      ASTLI_AST(astli) = mk_label(itemp->t.sptr);
    }
    swel->next = 0;

    ast = mk_stmt(A_CGOTO, 0);
    ast2 = SST_ASTG(RHS(6));
    A_LOPP(ast, ast2);
    A_LISTP(ast, ASTLI_HEAD);
    SST_ASTP(LHS, ast);

    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <arith IF> ::= <arith> <reflabel> , <reflabel> <opt aif label>
   */
  case ARITH_IF1:
    check_do_term();
    if (not_in_forall("Arithmetic IF"))
      break;
    if (SST_ISNONDECC(RHS(1)))
      cngtyp(RHS(1), DT_INT);
    mkexpr(RHS(1));
    dtype = SST_DTYPEG(RHS(1));
    if (!TY_ISINT(DTY(dtype)) && !TY_ISREAL(DTY(dtype))) {
      errsev(102);
      /* Now refine error message with another error message */
      if (DTY(dtype) == TY_CHAR)
        errsev(147);
      else if (DTY(dtype) == TY_STRUCT)
        errsev(149);
      else if (DTY(dtype) == TY_ARRAY)
        errsev(83);
      SST_ASTP(LHS, 0);
      break;
    }
    if (TY_ISLOG(DTY(dtype))) {
      cngtyp(RHS(1), DT_INT);
    }
    arith_if_expr = check_etmp(RHS(1));
    if (SST_ASTG(RHS(5)) == 0) {
      if (TY_ISLOG(DTY(dtype))) {
        /*  extension: indirect logical if;  generate the sequence
         *    if (<expr>) goto l1
         *    goto l2
         */
        if (flg.standard)
          error(170, 2, gbl.lineno, "Indirect logical IF statement", CNULL);
        ast = mk_stmt(A_IF, 0);
        A_IFEXPRP(ast, arith_if_expr);
        ast2 = mk_stmt(A_GOTO, 0);
        A_L1P(ast2, SST_ASTG(RHS(2)));
        A_IFSTMTP(ast, ast2);
        (void)add_stmt(ast);
        ast = mk_stmt(A_GOTO, 0);
        A_L1P(ast, SST_ASTG(RHS(4)));
        SST_ASTP(LHS, ast);
        break;
      }
      /*  extension: 2-way arithmetic if; generate an arithmetic if
       *  of the form
       *    goto (<expr>) l1, l2, l1
       */
      if (flg.standard)
        error(170, 2, gbl.lineno, "Two-way arithmetic IF statement", CNULL);
      SST_ASTP(RHS(5), SST_ASTG(RHS(2)));
    }
    ast = mk_stmt(A_AIF, 0);
    A_IFEXPRP(ast, arith_if_expr);
    A_L1P(ast, SST_ASTG(RHS(2)));
    A_L2P(ast, SST_ASTG(RHS(4)));
    A_L3P(ast, SST_ASTG(RHS(5)));
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt aif label> ::= |
   */
  case OPT_AIF_LABEL1:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<opt aif label> ::= , <reflabel>
   */
  case OPT_AIF_LABEL2:
    SST_ASTP(LHS, SST_ASTG(RHS(2)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <arith> ::= <if construct> <etmp lp> <expression> )
   */
  case ARITH1:
    *LHS = *RHS(3);
    if (construct_name)
      error(305, 3, gbl.lineno, "Arithmetic IF statement",
            stb.n_base + construct_name);
    break;

  /* ------------------------------------------------------------------ */
  /*
          <call> ::= CALL <cvar ref> |
  */
  case CALL1:
    SST_ASTP(LHS, SST_ASTG(RHS(2)));
    break;
  /*
   *	<call> ::= CALL <ident> <chevron>  |
   */
  case CALL2:
  /*
   *	<call> ::= CALL <ident> <chevron> ( ) |
   */
  case CALL3:
    itemp = ITEM_END;
    goto cuda_call_shared;
  /*
   *	<call> ::= CALL <ident> <chevron> ( <arg list> )
   */
  case CALL4:
    itemp = (ITEM *)SST_BEGG(RHS(5));
  cuda_call_shared:
    SST_ASTP(LHS, 0);
    if (!cuda_enabled("<<<"))
      break;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <cvar ref> ::= <ident>  |
   */
  case CVAR_REF1:
    itemp = ITEM_END;
    sptr = refsym((int)SST_SYMG(RHS(1)), OC_OTHER);
    if (STYPEG(sptr) == ST_PROC && CLASSG(sptr) && VTOFFG(sptr)) {
      /* Type bound procedure name may be overloaded by an extern or
       * module procedure. Try to resolve it here, otherwise, declare it
       * and assume it gets resolved later...
       */
      int sptr2 = find_non_tbp(SYMNAME(sptr));
      if (!sptr2) {
        sptr = insert_sym(sptr);
        sptr = declsym(sptr, ST_PROC, FALSE);
      } else {
        sptr = sptr2;
      }
    }
    SST_SYMP(RHS(1), sptr);
    goto cvar_ref_common;
  /*
   *      <cvar ref> ::= <var primary ssa> ( )  |
   */
  case CVAR_REF2:
    ast = SST_ASTG(RHS(1));
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      sptr1 = memsym_of_ast(ast);
      sptr = BINDG(sptr1);
      break;
    default:
      sptr = 0;
    }
    itemp = ITEM_END;
    if (VTOFFG(sptr) &&
        (sem.tbp_arg || NOPASSG(sptr1) || STYPEG(sptr) == ST_USERGENERIC ||
         STYPEG(sptr) == ST_OPERATOR)) {

      if (!NOPASSG(sptr1) && sem.tbp_arg) {
        itemp = pop_tbp_arg();
      } else {
        int mem, dty, func;
        dty = TBPLNKG(sptr);
        func = get_implementation(dty, sptr, 0, &mem);
        if (STYPEG(BINDG(mem)) == ST_OPERATOR ||
            STYPEG(BINDG(mem)) == ST_USERGENERIC) {
          if (generic_tbp_has_pass_and_nopass(dty, sptr)) {
            ITEM *itemp2;
            int sp;
            e1 = (SST *)getitem(0, sizeof(SST));
            sp = sym_of_ast(ast);
            SST_SYMP(e1, sp);
            SST_DTYPEP(e1, DTYPEG(sp));
            mkident(e1);
            mkexpr(e1);
            itemp2 = (ITEM *)getitem(0, sizeof(ITEM));
            itemp2->t.stkp = e1;
            itemp2->next = ITEM_END;
            itemp = itemp2;
          }
        }
      }
      if (STYPEG(sptr) == ST_OPERATOR || STYPEG(sptr) == ST_USERGENERIC) {
        int mem, dty;
        dty = TBPLNKG(sptr);
        get_implementation(dty, sptr, 0, &mem);
        subr_call2(RHS(1), itemp, !PRIVATEG(mem));
      } else {
        subr_call(RHS(1), itemp);
      }
      break;
    }
    itemp = ITEM_END;
    goto cvar_ref_common;
  /*
   *      <cvar ref> ::= <var primary ssa> ( <ssa list> )  |
   */
  case CVAR_REF3:
    itemp = SST_BEGG(RHS(3));
    ast = SST_ASTG(RHS(1));
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      sptr1 = memsym_of_ast(ast);
      sptr = BINDG(sptr1);
      sptr2 = pass_sym_of_ast(ast);
      break;
    default:
      sptr = 0;
    }
    if (VTOFFG(sptr)) {
      int imp, mem;

      imp = get_implementation(TBPLNKG(sptr), sptr, 0, &mem);
      if (!sem.tbp_arg && imp && !NOPASSG(mem)) {
        ITEM *itemp2;
        int sp;
        e1 = (SST *)getitem(0, sizeof(SST));
        sp = sym_of_ast(ast);
        SST_SYMP(e1, sp);
        SST_DTYPEP(e1, DTYPEG(sp));
        mkident(e1);
        mkexpr(e1);
        itemp2 = (ITEM *)getitem(0, sizeof(ITEM));
        itemp2->t.stkp = e1;
        itemp2->next = ITEM_END;
        push_tbp_arg(itemp2);
      } else if (!sem.tbp_arg && imp &&
                 (STYPEG(sptr) == ST_OPERATOR ||
                  STYPEG(sptr) == ST_USERGENERIC)) {
        int mem, dty;
        dty = TBPLNKG(sptr);
        get_implementation(dty, sptr, 0, &mem);
        if (STYPEG(BINDG(mem)) == ST_OPERATOR ||
            STYPEG(BINDG(mem)) == ST_USERGENERIC) {
          if (generic_tbp_has_pass_and_nopass(dty, sptr)) {
            ITEM *itemp2;
            int sp, mem2;
            int iface, paramct, arg_cnt;

            mem2 = get_generic_tbp_pass_or_nopass(dty, sptr, 1);
            proc_arginfo(VTABLEG(mem2), &paramct, 0, &iface);
            for (arg_cnt = 0, itemp2 = itemp; itemp2 != ITEM_END;
                 itemp2 = itemp2->next) {
              ++arg_cnt;
            }
            if (arg_cnt >= paramct) {
              subr_call(RHS(1), itemp);
              break;
            }
            e1 = (SST *)getitem(0, sizeof(SST));
            sp = sym_of_ast(ast);
            SST_SYMP(e1, sp);
            SST_DTYPEP(e1, DTYPEG(sp));
            mkident(e1);
            mkexpr(e1);
            itemp2 = (ITEM *)getitem(0, sizeof(ITEM));
            itemp2->t.stkp = e1;
            itemp2->next = ITEM_END;
            push_tbp_arg(itemp2);
          }
        }
      }

      if (sem.tbp_arg && !NOPASSG(sptr1)) {
        int argno, arg;
        ITEM *itemp2, *curr, *prev;
        itemp2 = pop_tbp_arg();
        if ((STYPEG(sptr) == ST_USERGENERIC || STYPEG(sptr) == ST_OPERATOR)) {
          /* FS#20696: Resolve generic before getting argno */
          ITEM *itemp2;
          int arg_cnt;
          int mem, mem2;
          for (arg_cnt = 0, itemp2 = itemp; itemp2 != ITEM_END;
               itemp2 = itemp2->next) {
            ++arg_cnt;
          }
          mem = get_generic_member2(TBPLNKG(sptr), sptr, arg_cnt, NULL);
          mem2 = get_specific_member(TBPLNKG(sptr), VTABLEG(mem));
          argno = get_tbp_argno(BINDG(mem2), TBPLNKG(sptr));
        } else {
          argno = get_tbp_argno(sptr, TBPLNKG(BINDG(memsym_of_ast(ast))));
        }
        if (!argno) {
          sptr = get_generic_tbp_pass_or_nopass(
              TBPLNKG(BINDG(memsym_of_ast(ast))), sptr, 0);
          if (!sptr)
            break; /* error -- probably no interface specified */
          sptr = VTABLEG(sptr);
          argno = get_tbp_argno(sptr, TBPLNKG(BINDG(memsym_of_ast(ast))));
          if (!argno)
            break; /* error -- probably no interface specified */
        }

        if (itemp == ITEM_END) {
          itemp = itemp2;
          itemp->next = ITEM_END;
        } else {
          for (arg = 1, curr = prev = itemp; arg <= argno;) {
            if (arg == argno) {
              itemp2->next = curr;
              if (argno == 1) {
                itemp = itemp2;
              } else {
                prev->next = itemp2;
              }
              break;
            }
            if (curr->next == ITEM_END) {
              /* put pass arg at end of list. This may happen if an optional
               * arg precedes the pass arg.
               */
              itemp2->next = ITEM_END;
              curr->next = itemp2;
              break;
            }
            ++arg;
            prev = curr;
            if (curr == ITEM_END) {
              interr("semant3: bad item list for <cvar ref> ", 0, 3);
              break;
            }
            curr = curr->next;
          }
        }
      }
      if (STYPEG(sptr) == ST_USERGENERIC || STYPEG(sptr) == ST_OPERATOR)
        subr_call2(RHS(1), itemp, 1);
      else {
        sptr1 = memsym_of_ast(ast);
        sptr1 = BINDG(sptr1); /* Get binding name of the type bound procedure */
        if (STYPEG(sptr1) == ST_USERGENERIC || STYPEG(sptr1) == ST_OPERATOR) {
          /* We have a generic type bound procedure. We only need to check the
           * access on the generic tbp definition; not the access on the
           * individual type bound procedures in the generic set.
           */
          subr_call2(RHS(1), itemp, !PRIVATEG(sptr1));
        } else {
          subr_call(RHS(1), itemp);
        }
      }
      break;
    }
    goto cvar_ref_common;
  /*
   *      <cvar ref> ::= <var primary> . <id>
   */
  case CVAR_REF4:
    (void)mkexpr(RHS(1));
    dtype = SST_DTYPEG(RHS(1));
    if (flg.standard)
      error(179, 2, gbl.lineno, SYMNAME(DTY(dtype + 1)), CNULL);
    if (DTY(dtype) != TY_STRUCT) {
      error(141, 3, gbl.lineno, "RECORD", ".");
      break;
    }
    sptr1 = SST_SYMG(RHS(3));
    i = NMPTRG(sptr1);
    ast = SST_ASTG(RHS(1));
    ast = mkmember(dtype, ast, i);
    if (ast) {
      sptr = A_SPTRG(A_MEMG(ast));
      SST_IDP(LHS, S_LVALUE);
      SST_LSYMP(LHS, sptr);
      SST_ASTP(LHS, ast);
    } else {
      /* <id> is not a member of this record */
      error(142, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
      break;
    }
    itemp = ITEM_END;
    goto cvar_ref_common;
  /*
   *	<cvar ref> ::= <var primary> % <id> |
   */
  case CVAR_REF5:
    rhstop = 3;
    (void)mkexpr(RHS(1));
    goto cvar_ref_component_shared;
  /*
   *	<cvar ref> ::= <var primary> %LOC
   */
  case CVAR_REF6:
    (void)mkexpr(RHS(1));
    rhstop = 2;
    SST_SYMP(RHS(2), getsymbol("loc"));
  cvar_ref_component_shared:
    ast = A_ORIG_EXPRG(SST_ASTG(RHS(1)));
    if (ast != 0) {
      /* The <var primary> is part of a procedure call, not a polymorphic
       * array expression. So, we restore the original expression.
       */
      SST_ASTP(RHS(1), ast);
    }
    itemp = ITEM_END;
    if (SST_IDG(RHS(1)) == S_IDENT || SST_IDG(RHS(1)) == S_DERIVED)
      sptr = SST_SYMG(RHS(1));
    else {
      if (SST_IDG(RHS(1)) == S_EXPR && A_TYPEG(SST_ASTG(RHS(1))) == A_FUNC) {
        error(155, 3, gbl.lineno,
              "Illegal context for the component reference to",
              SYMNAME(SST_SYMG(RHS(rhstop))));
      }
      sptr = SST_LSYMG(RHS(1));
    }
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    if (DTY(dtype) == TY_DERIVED) {
      int mem;
      sptr1 = SST_SYMG(RHS(rhstop));
      sptr1 = resolve_sym_aliases(sptr1);
      switch (A_TYPEG(SST_ASTG(RHS(1)))) {
      case A_ID:
      case A_LABEL:
      case A_ENTRY:
      case A_SUBSCR:
      case A_SUBSTR:
      case A_MEM:
        break;
      default:
        goto normal_cvar_ref_component;
      }
      mem = memsym_of_ast(SST_ASTG(RHS(1)));
      dtype = DTYPEG(mem);
      if (DTY(dtype) == TY_ARRAY)
        dtype = DTY(dtype + 1);
      mem = 0;
      sptr1 = get_implementation(dtype, sptr1, 0, &mem);
      if (sptr1 > NOSYM)
        sptr1 = BINDG(mem);

      if (sptr1 &&
          (STYPEG(sptr1) == ST_PROC || STYPEG(sptr1) == ST_USERGENERIC) &&
          IS_TBP(sptr1)) {

        int old_sptr1;
        old_sptr1 = SST_SYMG(RHS(rhstop));

        e1 = (SST *)getitem(0, sizeof(SST));
        *e1 = *RHS(1);
        itemp = (ITEM *)getitem(0, sizeof(ITEM));
        itemp->t.stkp = e1;
        itemp->next = ITEM_END;

        SST_SYMP(LHS, sptr1);

        if (!NOPASSG(mem))
          push_tbp_arg(itemp);

        i = NMPTRG(mem);
        ast = SST_ASTG(RHS(1));
        ast = mkmember(dtype, ast, i);
        if (ast) {
          ast = rewrite_ast_with_new_dtype(ast, DTYPEG(VTABLEG(mem)));
          sptr = A_SPTRG(A_MEMG(ast));
          SST_IDP(LHS, S_LVALUE);
          SST_LSYMP(LHS, sptr1);
          SST_SYMP(LHS, sptr1);
          SST_SHAPEP(LHS, 0);
          SST_DTYPEP(LHS, DTYPEG(VTABLEG(mem)));
          SST_ASTP(LHS, ast);
          SST_PARENP(LHS, 0);
        } else {
          error(142, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
        }
        subr_call(LHS, itemp);
        break;
      }
    }
  normal_cvar_ref_component:
    dtype = SST_DTYPEG(RHS(1));
    dtype = DDTG(dtype);
    ast = SST_ASTG(RHS(1));
    if (DTY(dtype) != TY_DERIVED) {
      error(141, 3, gbl.lineno, "Derived Type object", "%");
      break;
    }
    i = NMPTRG(SST_SYMG(RHS(rhstop)));
    ast = mkmember(dtype, ast, i);
    if (ast) {
      sptr1 = A_SPTRG(A_MEMG(ast));
      if (PRIVATEG(sptr1) && test_private_dtype(ENCLDTYPEG(sptr1))) {
        error(155, 3, gbl.lineno,
              "Attempt to use private component:", SYMNAME(sptr1));
      }
      SST_IDP(LHS, S_LVALUE);
      SST_LSYMP(LHS, sptr1);
      SST_ASTP(LHS, ast);
    } else {
      /* <id> is not a member of this record */
      sptr1 = SST_SYMG(RHS(rhstop));
      error(142, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
      break;
    }
  /*  fall thru  */
  cvar_ref_common:
    if (not_in_forall("CALL statement"))
      break;
    if (SST_IDG(RHS(1)) == S_IDENT) {
      sptr = SST_SYMG(RHS(1));
      if (!is_procedure_ptr(sptr)) {
        subr_call(RHS(1), itemp);
      } else {
        if (sem.parallel)
          (void)mkarg(RHS(1), &dum);
        ptrsubr_call(RHS(1), itemp);
      }
    } else {
      ptrsubr_call(RHS(1), itemp);
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<chevron> ::= '<<<' <expression list> '>>>'
   */
  case CHEVRON1:
    *LHS = *RHS(2);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<expression list> ::= <expression list> , <expression> |
   */
  case EXPRESSION_LIST1:
    rhstop = 3;
    goto common_expr_list;
  /*
   *	<expression list> ::= * |
   */
  case EXPRESSION_LIST2:
    rhstop = 1;
    SST_IDP(RHS(rhstop), S_CONST);
    SST_ASTP(RHS(rhstop), mk_cval(-1, DT_INT));
    SST_CVALP(RHS(rhstop), -1);
    SST_DTYPEP(RHS(rhstop), DT_INT);
    goto common_expr_list;
  /*
   *	<expression list> ::= <expression>
   */
  case EXPRESSION_LIST3:
    rhstop = 1;
  common_expr_list:
    e1 = (SST *)getitem(0, sizeof(SST));
    *e1 = *RHS(rhstop);
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.stkp = e1;
    if (rhstop == 1)
      SST_BEGP(LHS, itemp);
    else
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <arg list> ::= <arg list> , <arg> |
   */
  case ARG_LIST1:
    rhstop = 3;
    goto common_arg_list;
  /*
   *      <arg list> ::= <arg>
   */
  case ARG_LIST2:
    rhstop = 1;
  common_arg_list:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.stkp = SST_E1G(RHS(rhstop));
    if (rhstop == 1)
      SST_BEGP(LHS, itemp);
    else
      SST_ENDG(RHS(1))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <arg> ::= <expression>  |
   */
  case ARG1:
    rhstop = 1;
    goto common_arg;
  /*
   *      <arg> ::= <id name> = <expression>  |
   */
  case ARG2:
    e1 = (SST *)getitem(0, sizeof(SST));
    *e1 = *RHS(1);
    SST_IDP(e1, S_KEYWORD);
    SST_E1P(LHS, e1);
    SST_E3P(e1, (SST *)getitem(0, sizeof(SST)));
    *(SST_E3G(e1)) = *RHS(3);
    SST_E2P(LHS, 0);
    break;
  /*
   *      <arg> ::= * <reflabel>  |
   */
  case ARG3:
    rhstop = 2;
    goto common_arg;
  /*
   *      <arg> ::= & <reflabel>  |
   */
  case ARG4:
    if (flg.standard)
      errwarn(181);
    rhstop = 2;
  common_arg:
    e1 = (SST *)getitem(0, sizeof(SST));
    *e1 = *RHS(rhstop);
    SST_E1P(LHS, e1);
    if (rhstop == 2)
      SST_IDP(e1, S_LABEL);
    else
      SST_E2P(LHS, 0);
    break;
  /*
   *      <arg> ::= <arg builtin>
   */
  case ARG5:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <return> ::= RETURN |
   */
  case RETURN1:
    check_do_term();
    if (sem.parallel || sem.task || sem.teams) {
      error(155, 3, gbl.lineno,
            "Cannot branch out of parallel/teams/task region", CNULL);
    }
    if (not_in_forall("RETURN"))
      break;
    ast = mk_stmt(A_RETURN, 0);
    if (gbl.arets)
      A_LOPP(ast, astb.i0);
    else
      A_LOPP(ast, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <return> ::= RETURN <expression>
   */
  case RETURN2:
    if (not_in_forall("RETURN"))
      break;
    if (sem.parallel || sem.task || sem.teams) {
      error(155, 3, gbl.lineno,
            "Cannot branch out of parallel/teams/task region", CNULL);
    }
    if (gbl.rutype != RU_SUBR)
      errsev(159);
    else if (gbl.arets) {
      chk_scalartyp(RHS(2), DT_INT, TRUE);
    } else
      errsev(158);
    ast = mk_stmt(A_RETURN, 0);
    A_LOPP(ast, SST_ASTG(RHS(2)));
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <stop stmt> ::= STOP <stop pause>
   */
  case STOP_STMT1:
    check_do_term();
    if (not_in_forall("STOP"))
      break;
    if (gbl.currsub && PUREG(gbl.currsub))
      error(155, 2, gbl.lineno, SYMNAME(gbl.currsub),
            "- PURE subprograms may not contain STOP statements");
    if (sem.doif_depth > 0 && DI_IN_NEST(sem.doif_depth, DI_DOCONCURRENT))
      error(1050, ERR_Severe, gbl.lineno, "STOP in", CNULL); // 2018-C1137
    ast1 = SST_TMPG(RHS(2));
    ast2 = SST_ASTG(RHS(2));
    if (XBIT(54, 0x10)) {
      rtlRtn = RTE_stopa;
      goto pause_shared;
    }
    rtlRtn = RTE_stop08a;
    sptr = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
    NODESCP(sptr, 1);
    ast = begin_call(A_CALL, sptr, 2);
    add_arg(ast1);
    add_arg(ast2);
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <quiet clause> ::= QUIET = <expression>
   */
  case QUIET_CLAUSE1:
    if (DTY(SST_DTYPEG(RHS(3))) != TY_LOG) {
      error(1208, ERR_Severe, gbl.lineno, NULL, NULL);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <error stop stmt> ::= ERRORSTOP <error stop pause> |
   */
  case ERROR_STOP_STMT1:
    if (not_in_forall("ERRORSTOP"))
      break;

    ast3 =
        mk_unop(OP_REF, (DTY(DT_INT) == TY_INT8) ? astb.k0 : astb.i0, DT_PTR);
    goto errorstop_shared;
  /*
   *    <error stop stmt> ::= ERRORSTOP <error stop pause> , <quiet clause>
   */
  case ERROR_STOP_STMT2:
    if (not_in_forall("ERRORSTOP"))
      break;

    check_do_term();
    ast3 = SST_ASTG(RHS(6));

  errorstop_shared:
    ast1 = SST_TMPG(RHS(2));
    ast2 = SST_ASTG(RHS(2));

    rtlRtn = DTY(A_DTYPEG(ast2)) == TY_CHAR ? RTE_errorstop08a_char
                                            : RTE_errorstop08a_int;
    sptr = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
    NODESCP(sptr, 1);
    ast = begin_call(A_CALL, sptr, 2);

    if (DTY(A_DTYPEG(ast2)) == TY_CHAR) {
      add_arg(ast3); // QUIET= value
      add_arg(ast2); // output string / integer stop-code value
    } else {
      add_arg(ast1); // stop-code integer value
      add_arg(ast3); // QUIET= value
    }

    SST_ASTP(LHS, ast);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *    <error stop pause> ::=     |
   */
  case ERROR_STOP_PAUSE1:
    SST_ASTP(LHS, astb.ptr0c); /* null pointer */
    ast1 =
        mk_unop(OP_REF, (DTY(DT_INT) == TY_INT8) ? astb.k1 : astb.i1, DT_PTR);
    ast2 = astb.ptr0c;
    SST_TMPP(LHS, ast1);
    SST_ASTP(LHS, ast2);
    break;
  /*
   *    <error stop pause> ::= <expression>
   */
  case ERROR_STOP_PAUSE2:
    ast1 =
        mk_unop(OP_REF, (DTY(DT_INT) == TY_INT8) ? astb.k1 : astb.i1, DT_PTR);
    if (DTY(SST_DTYPEG(RHS(1))) == TY_CHAR) {
      (void)mkarg(RHS(1), &dum);
      ast2 = SST_ASTG(RHS(1));
    } else if (DT_ISINT(SST_DTYPEG(RHS(1)))) {
      if ((DTY(SST_DTYPEG(RHS(1))) == TY_INT && !XBIT(124, 0x10)) ||
          (DTY(SST_DTYPEG(RHS(1))) == TY_INT8 && XBIT(124, 0x10))) {
        (void)mkarg(RHS(1), &dum);
        ast1 = ast2 = SST_ASTG(RHS(1));
      } else {
        error(1209, ERR_Severe, gbl.lineno, NULL, NULL);
        break;
      }
    } else {
      error(1207, ERR_Severe, gbl.lineno, NULL, NULL);
      break;
    }
    SST_TMPP(LHS, ast1);
    SST_ASTP(LHS, ast2);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <pause stmt> ::= PAUSE <stop pause>
   */
  case PAUSE_STMT1:
    if (not_in_forall("PAUSE"))
      break;
    if (gbl.currsub && PUREG(gbl.currsub))
      error(155, 2, gbl.lineno, SYMNAME(gbl.currsub),
            "- PURE subprograms may not contain PAUSE statements");
    ast2 = SST_ASTG(RHS(2));
    if (A_TYPEG(ast2) == A_CNST && DTY(A_DTYPEG(ast2)) != TY_CHAR) {
      ast2 = astb.ptr0c;
      errsev(87);
    }
  pause_shared:
    sptr = sym_mkfunc(mkRteRtnNm(RTE_pausea), DT_NONE);
    NODESCP(sptr, 1);
    ast = begin_call(A_CALL, sptr, 1);
    add_arg(ast2);
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <stop pause> ::=    |
   */
  case STOP_PAUSE1:
    SST_ASTP(LHS, astb.ptr0c); /* null pointer */
    ast1 =
        mk_unop(OP_REF, (DTY(DT_INT) == TY_INT8) ? astb.k0 : astb.i0, DT_PTR);
    SST_TMPP(LHS, ast1);
    break;
  /*
   *      <stop pause> ::= <expression>
   */
  case STOP_PAUSE2:
    ast1 =
        mk_unop(OP_REF, (DTY(DT_INT) == TY_INT8) ? astb.k0 : astb.i0, DT_PTR);
    if (SST_IDG(RHS(1)) == S_CONST) {
      if (DTY(SST_DTYPEG(RHS(1))) == TY_CHAR) {
        (void)mkarg(RHS(1), &dum);
        ast2 = SST_ASTG(RHS(1));
      } else {
        /* <expression> should be a constant integer */
        (void)mkarg(RHS(1), &dum);
        ast1 = SST_ASTG(RHS(1));
        i = chkcon(RHS(1), DT_INT, TRUE);
        /* 64-bit hack */
        if (DTY(DT_INT) == TY_INT8)
          i = get_int_cval(i);
        snprintf(name, sizeof(name), "%5ld", (long)i);
        ast2 = mk_cnst(getstring(name, 5));
      }
    } else {
      if (DTY(SST_DTYPEG(RHS(1))) == TY_CHAR) {
        (void)mkarg(RHS(1), &dum);
        ast2 = SST_ASTG(RHS(1));
      } else {
        ast2 = astb.ptr0c;
        (void)mkarg(RHS(1), &dum);
        ast1 = SST_ASTG(RHS(1));
      }
      if (flg.standard) {
        error(170, 2, gbl.lineno,
              "Non-constant character expression in STOP or PAUSE", CNULL);
      }
    }
    SST_TMPP(LHS, ast1);
    SST_ASTP(LHS, ast2);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <GOTO stmt> ::= GOTO <reflabel>
   */
  case GOTO_STMT1:
    check_do_term();
    if (not_in_forall("GOTO"))
      break;
    ast = mk_stmt(A_GOTO, 0);
    A_L1P(ast, SST_ASTG(RHS(2)));
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <IF clause> ::= <if construct> <etmp lp> <expression> )
   */
  case IF_CLAUSE1:
    if (not_in_forall("logical IF"))
      break;
    ast2 = gen_logical_if_expr(RHS(3));
    /* begin converting single statement if to block if */
    ast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ast, ast2);
    (void)add_stmt(ast);
    SST_ASTP(LHS, ast);
    if (construct_name)
      error(305, 3, gbl.lineno, "IF statement", stb.n_base + construct_name);
    sem.use_etmps = TRUE;
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<if construct> ::= IF |
   */
  case IF_CONSTRUCT1:
    construct_name = 0;
    sem.pgphase = PHASE_EXEC; /* set now, since may have IF (...) stmt */
    sem.stats.nodes++;
    break;
  /*
   *	<if construct> ::= <check construct> : IF
   */
  case IF_CONSTRUCT2:
    sem.pgphase = PHASE_EXEC; /* set now, since may have IF (...) stmt */
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<check construct> ::= <named construct>
   */
  case CHECK_CONSTRUCT1:
    np = scn.id.name + SST_CVALG(RHS(1));
    sptr = block_local_sym(getsymbol(np));
    sptr = declsym(sptr, ST_CONSTRUCT, TRUE);
    FUNCLINEP(sptr, gbl.lineno);
    DCLDP(sptr, true);
    construct_name = NMPTRG(sptr);
    i = sem.doif_depth;
    while (i > 0) {
      doif = i--;
      if (DI_NAME(doif) == construct_name) {
        error(306, 3, gbl.lineno, stb.n_base + construct_name, CNULL);
        construct_name = 0;
        break;
      }
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <control stmt> ::= <IF clause> <simple stmt>  |
   */
  case CONTROL_STMT1:
    /* complete converting single statement if to block if */
    ast = SST_ASTG(RHS(2));
    /* if ast is zero, statements were already added */
    if (ast)
      (void)add_stmt(ast);
    gen_deallocate_arrays(); /* dealloc temp arrarys generated for
                              * <simple stmt>
                              */
    gen_dealloc_etmps();     /* dealloc if expression temps */
    ast = mk_stmt(A_ENDIF, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <control stmt> ::= <if construct> <etmp lp> <expression> )
   * GOTO
   * <reflabel> |
   */
  case CONTROL_STMT2:
    ast2 = gen_logical_if_expr(RHS(3));
    ast = mk_stmt(A_IF, 0);
    A_IFEXPRP(ast, ast2);
    ast2 = mk_stmt(A_GOTO, 0);
    A_L1P(ast2, SST_ASTG(RHS(6)));
    A_IFSTMTP(ast, ast2);
    SST_ASTP(LHS, ast);
    if (construct_name)
      error(305, 3, gbl.lineno, "IF statement", stb.n_base + construct_name);
    break;
  /*
   *      <control stmt> ::= <if construct> <etmp lp> <expression> )
   * THEN  |
   */
  case CONTROL_STMT3:
    NEED_DOIF(doif, DI_IF);
    DI_NAME(doif) = construct_name;
    ast2 = gen_logical_if_expr(RHS(3));
    ast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ast, ast2);
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <control stmt> ::= <elseif> <expression> ) THEN <construct
   * name>  |
   */
  case CONTROL_STMT4:
    if (SST_ASTG(RHS(1)) == 0)
      /* error was detected */
      break;
    /*
     * For <elseif>, we've generated
     *         goto exit_label
     *     endif
     *
     * NOTE: if the last statement was a 'goto', 'return', 'stop',
     * the 'goto exit_label' was not generated.
     *
     * Now, we need to add
     *     if (<expression>) then
     */
    doif = sem.doif_depth;
    ast2 = gen_logical_if_expr(RHS(2));
    ast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ast, ast2);
    SST_ASTP(LHS, ast);
    if (construct_name && DI_NAME(doif) != construct_name)
      err307("IF-THEN and ELSEIF", DI_NAME(doif), construct_name);
    break;
  /*
   *      <control stmt> ::= ELSE <construct name>  |
   */
  case CONTROL_STMT5:
    doif = sem.doif_depth;
    if (doif && DI_ID(doif) == DI_IF) {
      DI_ID(doif) = DI_IFELSE;
    } else {
      sem_err105(doif);
      SST_ASTP(LHS, 0);
      break;
    }
    if (construct_name && DI_NAME(doif) != construct_name)
      err307("IF-THEN and ELSE", DI_NAME(doif), construct_name);
    ast = mk_stmt(A_ELSE, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <control stmt> ::= ENDIF <construct name> |
   */
  case CONTROL_STMT6:
    doif = sem.doif_depth--;
    if (!doif || (DI_ID(doif) != DI_IF && DI_ID(doif) != DI_IFELSE)) {
      sem_err105(doif);
      SST_ASTP(LHS, 0);
      if (!doif)
        sem.doif_depth = 0;
      break;
    }
    if (DI_NAME(doif) != construct_name)
      err307("IF-THEN and ENDIF", DI_NAME(doif), construct_name);
    ast = mk_stmt(A_ENDIF, 0);
    if (DI_EXIT_LABEL(doif)) {
      (void)add_stmt(ast);
      scn.currlab = DI_EXIT_LABEL(doif);
      ast = mk_stmt(A_CONTINUE, 0);
      (void)add_stmt(ast);
      ast = 0;
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <control stmt> ::= <do begin> <loop control> |
   */
  case CONTROL_STMT7:
    SST_ASTP(LHS, SST_ASTG(RHS(2)));
    break;
  /*
   *      <control stmt> ::= <do begin>
   */
  case CONTROL_STMT8:
    (void)not_in_forall("DO statement");
    /* no loop control; generate 'dowhile(.true.)' */
    NEED_DOIF(doif, DI_DOWHILE);
    DI_DO_LABEL(doif) = do_label;
    doinfo = get_doinfo(1);
    doinfo->index_var = 0; /* marks doinfo for a DOWHILE */
    DI_DOINFO(doif) = doinfo;
    DI_NAME(doif) = construct_name;
    if (scn.currlab)
      DI_TOP_LABEL(doif) = scn.currlab;
    else
      scn.currlab = DI_TOP_LABEL(doif) = getlab();
    ast = mk_stmt(A_CONTINUE, 0);
    (void)add_stmt(ast);
    direct_loop_enter();
    ast2 = mk_cval(SCFTN_TRUE, DT_LOG);
    ast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ast, ast2);
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <control stmt> ::= ENDDO <construct name> |
   */
  case CONTROL_STMT9:
  share_do:
    SST_ASTP(LHS, 0);
    if (sem.doif_depth <= 0) {
      error(104, ERR_Severe, gbl.lineno, "- mismatched ENDDO", CNULL);
      (void)add_stmt(mk_stmt(A_ENDDO, 0));
      break;
    }
    doif = sem.doif_depth;
    if (scn.currlab && DI_DO_LABEL(doif) == scn.currlab)
      /*
       * the enddo is labeled and the label matches the do label.
       */
      ;
    else if ((DI_ID(doif) != DI_DO && DI_ID(doif) != DI_DOWHILE &&
              DI_ID(doif) != DI_DOCONCURRENT) ||
             DI_DO_LABEL(doif)) {
      error(104, 3, gbl.lineno, "- mismatched ENDDO", CNULL);
      SST_ASTP(LHS, 0);
      break;
    }
    if (DI_NAME(doif) != construct_name)
      err307("DO [CONCURRENT|WHILE] and ENDDO", DI_NAME(doif), construct_name);
    doinfo = DI_DOINFO(doif);
    do_end(doinfo);
    direct_loop_end(DI_LINENO(doif), gbl.lineno);
    if (doinfo->distloop == LP_PARDO_OTHER) {
      --sem.doif_depth; /* skip past DI_PARDO */
      goto share_do;
    } else if (sem.doif_depth > 1 && DI_ID(sem.doif_depth) == DI_PARDO) {
      doinfo = DI_DOINFO(sem.doif_depth - 1);
      if (DI_ID(sem.doif_depth - 1) == DI_DO &&
          doinfo->distloop == LP_DISTPARDO) {
        --sem.doif_depth; /* skip past DI_PARDO */
        goto share_do;
      }
    }
    break;
  /*
   *      <control stmt> ::= <where clause> |
   */
  case CONTROL_STMT10:
    ast = mk_stmt(A_WHERE, 0);
    A_IFEXPRP(ast, SST_ASTG(RHS(1)));
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <control stmt> ::= <elsewhere clause> |
   */
  case CONTROL_STMT11:
    break;

  /*
   *     "<elsewhere clause> ::= ELSEWHERE <construct name> |
   */
  case ELSEWHERE_CLAUSE1:
    if (sem.doif_depth > 0) {
      doif = sem.doif_depth;
      if (DI_ID(doif) == DI_WHERE) {
        DI_ID(doif) = DI_ELSEWHERE;
        DI_MASKED(doif) = 0;
        if (construct_name && DI_NAME(doif) != construct_name)
          err307("WHERE and ELSEWHERE", DI_NAME(doif), construct_name);
      } else {
        error(104, 3, gbl.lineno, "- mismatched ELSEWHERE", CNULL);
      }
    } else
      error(104, 3, gbl.lineno, "- mismatched ELSEWHERE", CNULL);
    SST_ASTP(LHS, mk_stmt(A_ELSEWHERE, 0));
    break;

  /*
   *      <elsewhere clause> ::= ELSEWHERE ( <mask expr> ) <construct
   * name>
   */
  case ELSEWHERE_CLAUSE2:
    i = 0;
    if (sem.doif_depth > 0) {
      doif = sem.doif_depth;
      if (DI_ID(doif) == DI_WHERE) {
        DI_ID(doif) = DI_ELSEWHERE;
        DI_MASKED(doif) = 1;
        if (construct_name && DI_NAME(doif) != construct_name)
          err307("WHERE and ELSEWHERE", DI_NAME(doif), construct_name);
        i = DI_NAME(doif);
      } else {
        error(104, 3, gbl.lineno, "- mismatched ELSEWHERE", CNULL);
      }
    } else
      error(104, 3, gbl.lineno, "- mismatched ELSEWHERE", CNULL);

    add_stmt(mk_stmt(A_ELSEWHERE, 0));

    shape = A_SHAPEG(SST_ASTG(RHS(3)));
    NEED_DOIF(doif, DI_WHERE);
    DI_NAME(doif) = i;
    if (shape)
      DI_SHAPEDIM(doif) = SHD_NDIM(shape);
    *LHS = *RHS(3);

    ast = mk_stmt(A_WHERE, 0);
    A_IFEXPRP(ast, SST_ASTG(RHS(1)));
    SST_ASTP(LHS, ast);

    break;
  /*
   *      <control stmt> ::= ENDWHERE <construct name> |
   */
  case CONTROL_STMT12:
    if (sem.doif_depth > 0) {
      doif = sem.doif_depth;
      if (DI_ID(doif) != DI_WHERE && DI_ID(doif) != DI_ELSEWHERE)
        error(104, 3, gbl.lineno, "- mismatched ENDWHERE", CNULL);
      else {
        SST_ASTP(LHS, mk_stmt(A_ENDWHERE, 0));
        while (--sem.doif_depth && DI_ID(sem.doif_depth) == DI_ELSEWHERE &&
               DI_MASKED(sem.doif_depth)) {
          add_stmt(mk_stmt(A_ENDWHERE, 0));
        }
        if (construct_name && DI_NAME(doif) != construct_name)
          err307("WHERE and ENDWHERE", DI_NAME(doif), construct_name);
      }
    } else
      error(104, 3, gbl.lineno, "- mismatched ENDWHERE", CNULL);
    break;
  /*
   *      <control stmt> ::= <forall clause> |
   */
  case CONTROL_STMT13:
    ast = SST_ASTG(RHS(1));
  forall_shared:
    if (last_std == STD_PREV(0))
      last_std = 0;
    else
      last_std = STD_NEXT(last_std);
    A_SRCP(ast, last_std);
    break;
  /*
   *      <control stmt> ::= ENDFORALL <construct name> |
   */
  case CONTROL_STMT14:
    if (flg.smp) {
      DI_NOSCOPE_FORALL(sem.doif_depth) = 0;
      clear_no_scope_sptr();
    }
    doif = sem.doif_depth;
    if (doif <= 0 || DI_ID(doif) != DI_FORALL) {
      error(104, 3, gbl.lineno, "- mismatched ENDFORALL", CNULL);
    } else {
      for (symi = DI_IDXLIST(doif); symi; symi = SYMI_NEXT(symi))
        pop_sym(SYMI_SPTR(symi));
      direct_loop_end(DI_LINENO(doif), gbl.lineno);
      --sem.doif_depth;
    }
    SST_ASTP(LHS, mk_stmt(A_ENDFORALL, 0));
    break;
  /*
   *	<control stmt> ::= <case begin> |
   */
  case CONTROL_STMT15:
    (void)mkexpr(RHS(1));
    dtype = SST_DTYPEG(RHS(1));
    sptr2 = 0; /* allocated char temp for case expr */
    if (DT_ISINT(dtype) || DT_ISLOG(dtype) || DTY(dtype) == TY_CHAR ||
        DTY(dtype) == TY_NCHAR) {
      ast = check_etmp(RHS(1));
      if (A_TYPEG(ast) == A_ID && sem.p_dealloc &&
          A_SPTRG(ast) == sem.p_dealloc->t.sptr) {
        /* need to remove the temp from the list for which
         * deallocates are generated at the end of the statement;
         * this temp needs to be deallocated at each case construct.
         */
        sem.p_dealloc = sem.p_dealloc->next;
        sptr2 = A_SPTRG(ast);
      } else if (!A_ISLVAL(A_TYPEG(ast)) || A_CALLFGG(ast)) {
        ast = sem_tempify(RHS(1));
        (void)add_stmt(ast);
        ast = A_DESTG(ast);
        if (sem.p_dealloc && A_SPTRG(ast) == sem.p_dealloc->t.sptr) {
          /* need to remove the temp from the list for which
           * deallocates are generated at the end of the statement;
           * this temp needs to be deallocated at each case construct.
           */
          sem.p_dealloc = sem.p_dealloc->next;
          sptr2 = A_SPTRG(ast);
        }
      } else if (scn.currlab) {
        ast2 = mk_stmt(A_CONTINUE, 0);
        (void)add_stmt(ast2);
      }
    } else {
      error(310, 3, gbl.lineno,
            "SELECTCASE expression must be "
            "integer, logical, or character",
            CNULL);
      ast = astb.i0;
      dtype = 0;
    }
    NEED_DOIF(doif, DI_CASE);
    DI_NAME(doif) = construct_name;
    DI_CASE_EXPR(doif) = ast;
    DI_DTYPE(doif) = dtype;
    DI_ALLO_CHTMP(doif) = sptr2;
    SST_ASTP(LHS, 0);

    /* Create an empty list for the case values; swel_hd locates an
     * empty
     * list item whose next field will filled in to locate the first
     * item
     * in the list.
     */
    begin = sem.switch_avl++; /* relative ptr to header */
    NEED(sem.switch_avl, switch_base, SWEL, sem.switch_size,
         sem.switch_size + 300);
    DI_SWEL_HD(doif) = begin;
    switch_base[begin].next = 0;

    break;
  /*
   *	<control stmt> ::= <case> <elp> <case value list> ) <construct
   *name> |
   */
  case CONTROL_STMT16:
    doif = SST_CVALG(RHS(1));
    if (doif == 0 || SST_ASTG(RHS(3)) == 0) {
      SST_ASTP(LHS, 0);
      break;
    }
    if (construct_name && DI_NAME(doif) != construct_name)
      err307("SELECTCASE and CASE", DI_NAME(doif), construct_name);
    if (DI_DEFAULT_SEEN(doif) && !DI_DEFAULT_COMPLETE(doif)) {
      /*
       * get first STD of default; if nothing appears in the default
       * block, then this will be 0
       */
      int s1 = DI_BEG_DEFAULT(doif); /* STD preceding the default */

      DI_DEFAULT_COMPLETE(doif) = 1;

      /* need to save the default block */
      DI_BEG_DEFAULT(doif) = STD_NEXT(s1);

      /*
       * get the last STD of default; if nothing appears in the default
       * block, then this will be the STD which precedes the
       * CASEDEFAULT.
       */
      DI_END_DEFAULT(doif) = SST_ASTG(RHS(2)); /* last  STD of default */
      /* Unlink the list of stds representing the default block */
      STD_NEXT(s1) = 0;
      STD_PREV(0) = s1;
    }
    if (DI_PENDING(doif)) {
      if (DI_EXIT_LABEL(doif) == 0)
        DI_EXIT_LABEL(doif) = getlab();
      ast = mk_stmt(A_GOTO, 0);
      astlab = mk_label(DI_EXIT_LABEL(doif));
      A_L1P(ast, astlab);
      RFCNTI(DI_EXIT_LABEL(doif));
      (void)add_stmt(ast);
      ast = mk_stmt(A_ENDIF, 0);
      (void)add_stmt(ast);
      DI_PENDING(doif) = 0;
    }

    /* generate an IF-THEN */
    ast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ast, SST_ASTG(RHS(3)));
    DI_PENDING(doif) = 1;
    SST_ASTP(LHS, ast);
    if (DI_ALLO_CHTMP(doif)) {
      (void)add_stmt(ast);
      dealloc_tmp(DI_ALLO_CHTMP(doif));
      SST_ASTP(LHS, 0);
    }
    break;
  /*
   *	<control stmt> ::= CASEDEFAULT <construct name> |
   */
  case CONTROL_STMT17:
    SST_ASTP(LHS, 0); /* nothing to generate now */
    doif = sem.doif_depth;
    if (!doif || DI_ID(doif) != DI_CASE) {
      sem_err105(doif);
      break;
    }
    if (construct_name && DI_NAME(doif) != construct_name)
      err307("SELECTCASE and CASEDEFAULT", DI_NAME(doif), construct_name);

    if (DI_DEFAULT_SEEN(doif)) {
      error(310, 3, gbl.lineno,
            "At most one CASEDEFAULT may appear in a case construct", CNULL);
      break;
    }

    if (DI_PENDING(doif)) {
      if (DI_EXIT_LABEL(doif) == 0)
        DI_EXIT_LABEL(doif) = getlab();
      ast = mk_stmt(A_GOTO, 0);
      astlab = mk_label(DI_EXIT_LABEL(doif));
      A_L1P(ast, astlab);
      RFCNTI(DI_EXIT_LABEL(doif));
      (void)add_stmt(ast);
      ast = mk_stmt(A_ENDIF, 0);
      (void)add_stmt(ast);
      DI_PENDING(doif) = 0;
    }
    DI_DEFAULT_SEEN(doif) = 1;
    DI_BEG_DEFAULT(doif) = STD_PREV(0);
    if (DI_ALLO_CHTMP(doif))
      dealloc_tmp(DI_ALLO_CHTMP(doif));
    break;
  /*
   *	<control stmt> ::= ENDSELECT <construct name>
   */
  case CONTROL_STMT18:
    sem.select_type_seen = 0;
    doif = sem.doif_depth;
    if (doif > 0 && DI_ID(doif) == DI_SELECT_TYPE) {
      sem.doif_depth--;
      if (DI_NAME(doif) != construct_name)
        err307("SELECT TYPE and END SELECT", DI_NAME(doif), construct_name);
      if ((sptr = DI_ACTIVE_SPTR(doif)) > NOSYM)
        pop_sym(sptr);
      if (!DI_IS_WHOLE(doif) && (sptr = DI_SELECTOR(doif)) > NOSYM)
        pop_sym(sptr);

      if (DI_SELECT_TYPE_LIST(doif)) {
        /* Generate the type and class comparisons */
        int tag1, sdsc2, sdsc3, argt, flag;
        int flag_con, argt_cnt, zero, fsptr, tmp, beg_std, std2, rslt;
        int intrin_type;

        sptr = DI_SELECTOR(doif);
        tag1 = dtype = DTYPEG(sptr);
        if (DTY(tag1) == TY_ARRAY)
          tag1 = DTY(dtype + 1);
        tag1 = DTY(tag1 + 3);
        flag = 0;
        if (POINTERG(sptr)) {
          flag |= 1;
        } else if (ALLOCATTRG(sptr)) {
          flag |= 2;
        }
        if (flag) {
          sdsc3 = get_static_type_descriptor(tag1);
          argt_cnt = 6;
        } else {
          sdsc3 = 0;
          argt_cnt = 5;
        }
        flag_con = mk_cval1(flag, DT_INT);
        flag_con = mk_unop(OP_VAL, flag_con, DT_INT);
        zero = mk_cval1(0, DT_INT);
        zero = mk_unop(OP_VAL, zero, DT_INT);
        tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, DT_INT, sem.sc);

        beg_std = DI_TYPE_BEG(doif);
        rslt = mk_cval1(-1, DT_INT);
        for (curr = DI_SELECT_TYPE_LIST(doif); curr;) {
          int dtype2 = curr->dtype;
          if (DTY(dtype2) == TY_DERIVED) {
            int tag2 = DTY(dtype2 + 3);
            sdsc2 = get_static_type_descriptor(tag2);
            sdsc2 = mk_id(sdsc2);
            intrin_type = 0;
          } else {
            intrin_type = 1;
            sdsc2 = dtype_to_arg(dtype2);
            sdsc2 = mk_cval1(sdsc2, DT_INT);
            sdsc2 = mk_unop(OP_VAL, sdsc2, DT_INT);
          }
          argt = mk_argt(argt_cnt);
          ARGT_ARG(argt, 0) = mk_id(sptr);
          ARGT_ARG(argt, 1) = mk_id(SDSCG(sptr));
          ARGT_ARG(argt, 2) = zero;
          ARGT_ARG(argt, 3) = sdsc2;
          ARGT_ARG(argt, 4) = flag_con;
          if (argt_cnt == 6) {
            ARGT_ARG(argt, 5) = mk_id(sdsc3);
          }
          if (curr->is_class) {
            if (XBIT(68, 0x1)) {
              fsptr =
                  sym_mkfunc_nodesc(mkRteRtnNm(RTE_extends_type_of), DT_LOG);
            } else
              fsptr =
                  sym_mkfunc_nodesc(mkRteRtnNm(RTE_extends_type_of), DT_LOG);
          } else {
            if (XBIT(68, 0x1)) {
              fsptr = sym_mkfunc_nodesc(
                  (!intrin_type ? mkRteRtnNm(RTE_same_type_as)
                                : mkRteRtnNm(RTE_same_intrin_type_as)),
                  DT_LOG);

            } else
              fsptr = sym_mkfunc_nodesc(
                  (!intrin_type ? mkRteRtnNm(RTE_same_type_as)
                                : mkRteRtnNm(RTE_same_intrin_type_as)),
                  DT_LOG);
          }

          ast = mk_id(fsptr);
          ast = mk_func_node(A_FUNC, ast, argt_cnt, argt);
          A_DTYPEP(ast, stb.user.dt_log);
          ast = mk_assn_stmt(mk_id(tmp), ast, DT_INT);
          std2 = add_stmt_before(ast, beg_std);
          ast = mk_binop(OP_EQ, A_DESTG(ast), rslt, DT_INT);

          ast1 = mk_stmt(A_IF, 0);
          A_IFEXPRP(ast1, ast);
          ast2 = mk_stmt(A_GOTO, 0);
          RFCNTI(curr->label);
          ast = mk_label(curr->label);
          A_L1P(ast2, ast);
          A_IFSTMTP(ast1, ast2);
          add_stmt_after(ast1, std2);

          prev = curr;
          curr = curr->next;
          FREE(prev);
        }
      }

      DI_SELECT_TYPE_LIST(doif) = 0;

      if (DI_CLASS_DEFAULT_LABEL(doif)) {

        /* Add branch to CLASS DEFAULT */
        RFCNTI(DI_CLASS_DEFAULT_LABEL(doif));
        ast1 = mk_stmt(A_GOTO, 0);
        ast2 = mk_label(DI_CLASS_DEFAULT_LABEL(doif));
        A_L1P(ast1, ast2);
        add_stmt_before(ast1, DI_TYPE_BEG(doif));
      }

      /* Add label at end select type */
      ast = mk_stmt(A_CONTINUE, 0);
      std = add_stmt(ast);
      STD_LABEL(std) = DI_END_SELECT_LABEL(doif);

      SST_ASTP(LHS, ast);
      break;
    }

    /* Not a SELECT TYPE construct; must be SELECT CASE. */
    lab = scn.currlab;
    scn.currlab = 0;
    doif = sem.doif_depth--;
    if (!doif || DI_ID(doif) != DI_CASE) {
      sem_err105(doif);
      SST_ASTP(LHS, 0);
      if (!doif)
        sem.doif_depth = 0;
      break;
    }
    if (DI_NAME(doif) != construct_name)
      err307("SELECTCASE and ENDSELECT", DI_NAME(doif), construct_name);
    if (DI_DEFAULT_SEEN(doif) && DI_BEG_DEFAULT(doif)) {
      /*  CASE DEFAULT present */
      if (DI_PENDING(doif)) {
        /*  CASE still open */
        ast = mk_stmt(A_ELSE, 0); /* enclose default in an ELSE */
        if (DI_END_DEFAULT(doif)) {
          /* default block saved; append them to the current std */
          int s1;

          s1 = add_stmt(ast);
          STD_PREV(DI_BEG_DEFAULT(doif)) = s1;
          STD_NEXT(s1) = DI_BEG_DEFAULT(doif);
          STD_PREV(0) = DI_END_DEFAULT(doif);
          STD_NEXT(DI_END_DEFAULT(doif)) = 0;
        } else
          (void)add_stmt_after(ast, DI_BEG_DEFAULT(doif));
      }
    } else if (DI_ALLO_CHTMP(doif)) {
      if (DI_PENDING(doif)) {
        /*  CASE still open */
        ast = mk_stmt(A_ELSE, 0); /* enclose default in an ELSE */
        (void)add_stmt(ast);
      }
      dealloc_tmp(DI_ALLO_CHTMP(doif));
    }
    if (DI_PENDING(doif)) {
      ast = mk_stmt(A_ENDIF, 0);
      (void)add_stmt(ast);
    }
    if (DI_EXIT_LABEL(doif)) {
      scn.currlab = DI_EXIT_LABEL(doif);
      ast = mk_stmt(A_CONTINUE, 0);
      (void)add_stmt(ast);
    }
    if (lab) {
      scn.currlab = lab;
      ast = mk_stmt(A_CONTINUE, 0);
      (void)add_stmt(ast);
    }

    SST_ASTP(LHS, 0);
    break;

  /*
   *      <control stmt> ::= <associate stmt> |
   */
  case CONTROL_STMT19:
    break;
  /*
   *      <control stmt> ::= ENDASSOCIATE <construct name>
   */
  case CONTROL_STMT20:
    doif = sem.doif_depth;
    if (doif <= 0 || DI_ID(doif) != DI_ASSOC) {
      error(155, 3, gbl.lineno,
            "END ASSOCIATE without matching ASSOCIATE found", CNULL);
      break;
    }
    sem.doif_depth--;
    if (construct_name && DI_NAME(doif) != construct_name)
      error(155, 3, gbl.lineno, "Invalid construct name for END ASSOCIATE -",
            stb.n_base + construct_name);
    for (itemp1 = DI_ASSOCIATIONS(doif); (itemp = itemp1);) {
      itemp1 = itemp->next;
      end_association(itemp->t.sptr);
      FREE(itemp);
    }
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <associate stmt> ::=    ASSOCIATE ( <association list> )
   *      <associate stmt> ::= <check construct> : ASSOCIATE (
   * <association list> )
   */
  case ASSOCIATE_STMT1:
    construct_name = 0;
    FLANG_FALLTHROUGH;
  case ASSOCIATE_STMT2:
    rhstop = rednum == ASSOCIATE_STMT1 ? 3 : 5;
    itemp = SST_BEGG(RHS(rhstop));
    NEED_DOIF(doif, DI_ASSOC);
    DI_NAME(doif) = construct_name;
    DI_ASSOCIATIONS(doif) = itemp;
    /* Bring the name(s) into scope now for the contained block. */
    for (; itemp; itemp = itemp->next) {
      push_sym(itemp->t.sptr);
    }
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <association list> ::=  <association> |
   *      <association list> ::= <association list> , <association>
   */
  case ASSOCIATION_LIST1:
  case ASSOCIATION_LIST2:
    rhstop = rednum == ASSOCIATION_LIST1 ? 1 : 3;
    itemp1 = rednum == ASSOCIATION_LIST2 ? SST_BEGG(RHS(1)) : NULL;
    sptr = SST_TMPG(RHS(rhstop)); /* saved <id> from <association> below */
    SST_TMPP(RHS(rhstop), 0);
    sptr = construct_association(sptr, RHS(rhstop), 0 /* no forced dtype */,
                                 FALSE);
    pop_sym(sptr); /* it gets pushed back at the end of the ASSOCIATE
                      statement */
    for (itemp = itemp1; itemp; itemp = itemp->next) {
      if (strcmp(SYMNAME(itemp->t.sptr), SYMNAME(sptr)) == 0) {
        error(155, 3, gbl.lineno, "Duplicate name in ASSOCIATE", SYMNAME(sptr));
        break;
      }
    }
    NEW(itemp, ITEM, 1);
    BZERO(itemp, ITEM, 1);
    itemp->t.sptr = sptr;
    itemp->next = itemp1;
    SST_ASTP(LHS, 0);
    SST_BEGP(LHS, itemp);
    break;
  /*
   *      <association> ::=       <id> '=>' <expression>
   */
  case ASSOCIATION1:
    sptr = SST_SYMG(RHS(1));
    *LHS = *RHS(3);
    SST_TMPP(LHS, sptr);
    break;
  /*
   *      <control stmt> ::= <select type stmt> |
   */
  case CONTROL_STMT21:
    break;
  /*
   *      <control stmt> ::= <type guard stmt>
   */
  case CONTROL_STMT22:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <select type stmt> ::= SELECTTYPE ( <assoc or selector> ) |
   *      <select type stmt> ::= <check construct> : SELECTTYPE ( <assoc
   * or
   * selector> )
   */
  case SELECT_TYPE_STMT1:
    construct_name = 0;
    FLANG_FALLTHROUGH;
  case SELECT_TYPE_STMT2:
    rhstop = rednum == SELECT_TYPE_STMT1 ? 3 : 5;
    NEED_DOIF(doif, DI_SELECT_TYPE);
    DI_NAME(doif) = construct_name;
    DI_SELECTOR(doif) = SST_SYMG(RHS(rhstop));
    DI_IS_WHOLE(doif) = SST_TMPG(RHS(rhstop)); /* whole variable selector */
    ast = mk_stmt(A_CONTINUE, 0);
    add_stmt(ast);
    DI_TYPE_BEG(doif) = STD_LAST;
    SST_ASTP(LHS, ast);
    sptr = getlab();
    DI_END_SELECT_LABEL(doif) = sptr;
    DEFDP(sptr, 1);
    SST_ASTP(LHS, 0);
    sem.select_type_seen++;
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <assoc or selector> ::= <association> |
   *      <assoc or selector> ::= <expression>
   */
  case ASSOC_OR_SELECTOR1:
  case ASSOC_OR_SELECTOR2:
    if (rednum == ASSOC_OR_SELECTOR1) {
      sptr = SST_TMPG(RHS(1));
      SST_TMPP(LHS, FALSE); /* selector is not a whole variable */
    } else if ((sptr = get_sst_named_whole_variable(RHS(1))) > NOSYM) {
      SST_TMPP(LHS, FALSE); /* selector is not whole variable */
    } else {
      error(155, 3, gbl.lineno,
            "A SELECT TYPE selector without an "
            "associate-name must be a named variable",
            CNULL);
    }
    if (sptr > NOSYM) {
      if (!SST_TMPG(LHS)) {
        /* If selector is not a whole variable, create a temporary
         * pointer
         * or allocatable for it.
         */
        sptr =
            construct_association(sptr, RHS(1), 0 /* no forced dtype */, FALSE);
      }
      if (!CLASSG(sptr))
        error(155, 3, gbl.lineno, "Non-polymorphic selector in SELECT TYPE - ",
              SYMNAME(sptr));
    }
    SST_SYMP(LHS, sptr);
    break;
  /*
   *      <type guard stmt> ::= <typeis stmt> |
   */
  case TYPE_GUARD_STMT1:
    break;
  /*
   *      <type guard stmt> ::= <classis stmt> |
   */
  case TYPE_GUARD_STMT2:
    break;
  /*
   *      <type guard stmt> ::= <classdefault stmt>
   */
  case TYPE_GUARD_STMT3:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <typeis stmt> ::= TYPEIS ( <typespec> ) <construct name>
   *      <classis stmt> ::= CLASSIS ( <typespec> ) <construct name>
   */
  case TYPEIS_STMT1:
  case CLASSIS_STMT1:
    sem.select_type_seen = 0;
    doif = sem.doif_depth;

    if (doif <= 0 || DI_ID(doif) != DI_SELECT_TYPE) {
      if (rednum == TYPEIS_STMT1)
        error(155, 4, gbl.lineno, "TYPE IS statement is not within SELECT TYPE",
              0);
      else
        error(155, 4, gbl.lineno,
              "CLASS IS statement is not within SELECT TYPE", 0);
      break;
    }

    if (construct_name && DI_NAME(doif) != construct_name) {
      if (rednum == TYPEIS_STMT1)
        err307("SELECT TYPE and TYPE IS", DI_NAME(doif), construct_name);
      else
        err307("SELECT TYPE and CLASS IS", DI_NAME(doif), construct_name);
    }

    if ((sptr = DI_ACTIVE_SPTR(doif)) > NOSYM) {
      pop_sym(sptr); /* end previous active binding for this SELECT TYPE */
      DI_ACTIVE_SPTR(doif) = 0;
    }

    /* Add branch around */
    RFCNTI(DI_END_SELECT_LABEL(doif));
    ast1 = mk_stmt(A_GOTO, 0);
    ast2 = mk_label(DI_END_SELECT_LABEL(doif));
    A_L1P(ast1, ast2);
    std = add_stmt(ast1);

    dtype = SST_DTYPEG(RHS(3));

    dtype2 = DTYPEG(DI_SELECTOR(doif)); /* type of the SELECT TYPE selector */
    if (dtype2 && DTY(dtype2) == TY_ARRAY)
      dtype2 = DTY(dtype2 + 1); /* ... less any dimensions */
    if (DTY(dtype2) == TY_DERIVED && !eq_dtype2(dtype2, dtype, TRUE)) {
      int tag = DTY(dtype2 + 3);
      if (!UNLPOLYG(tag)) {
        if (rednum == TYPEIS_STMT1) {
          error(155, 4, gbl.lineno,
                "Type specified in TYPE IS must be an "
                "extension of type",
                SYMNAME(tag));
        } else {
          error(155, 4, gbl.lineno,
                "Type specified in CLASS IS must be an "
                "extension of type",
                SYMNAME(tag));
        }
      }
    }

    /* Add type info to the type list */
    NEW(types, TYPE_LIST, 1);
    types->is_class = rednum == CLASSIS_STMT1;
    types->dtype = dtype;

    /* create branch to label */
    sptr = getlab();
    types->label = sptr;
    DEFDP(sptr, 1);

    ast = mk_stmt(A_CONTINUE, 0);
    std = add_stmt_after(ast, std);
    STD_LABEL(std) = types->label;

    /* Check for duplicate class/type is statement */
    for (curr = DI_SELECT_TYPE_LIST(doif); curr; curr = curr->next) {
      if (curr->is_class == (rednum == CLASSIS_STMT1) &&
          eq_dtype2(curr->dtype, types->dtype, 0)) {
        if (rednum == TYPEIS_STMT1) {
          error(155, 3, gbl.lineno, "Duplicate TYPE IS",
                (DTY(types->dtype) == TY_DERIVED)
                    ? SYMNAME(DTY(types->dtype + 3))
                    : target_name(types->dtype));
        } else {
          error(155, 3, gbl.lineno, "Duplicate CLASS IS",
                SYMNAME(DTY(types->dtype + 3)));
        }
        break;
      }
    }

    if (rednum == TYPEIS_STMT1) {
      /* insert type comparison in front of list for type is */
      types->next = DI_SELECT_TYPE_LIST(doif);
      DI_SELECT_TYPE_LIST(doif) = types;
    } else {
      /* insert class comparison in "sorted" order for class is.
       * The class comparison must get inserted after all type
       * comparisons but before any class comparison that's more general
       * than the one we're inserting.
       */
      for (prev = curr = DI_SELECT_TYPE_LIST(doif); curr; curr = curr->next) {
        if (curr->is_class && eq_dtype2(curr->dtype, types->dtype, 1)) {
          if (curr != DI_SELECT_TYPE_LIST(doif)) {
            prev->next = types;
            prev = types->next = curr;
          } else {
            prev = types->next = DI_SELECT_TYPE_LIST(doif);
            DI_SELECT_TYPE_LIST(doif) = types;
          }
          break;
        }
        prev = curr;
      }
      if (!prev) {
        types->next = 0;
        DI_SELECT_TYPE_LIST(doif) = types;
      } else if (!curr) {
        types->next = 0;
        prev->next = types;
      }
    }

    /* Set up a new pointer of the specified type, with the dimensions
     * of the SELECT TYPE selector.
     */
    dtype2 = DTYPEG(DI_SELECTOR(doif)); /* type of the SELECT TYPE */
    if (dtype2 && DTY(dtype2) == TY_ARRAY) {
      int element_dtype = dtype;
      dtype = dup_array_dtype(dtype2);
      DTY(dtype + 1) = element_dtype;
    }
    mkident(LHS);
    sptr = DI_SELECTOR(doif);
    SST_SYMP(LHS, sptr);
    SST_ASTP(LHS, mk_id(sptr));
    sptr = construct_association(sptr, LHS, dtype, rednum == CLASSIS_STMT1);
    DI_ACTIVE_SPTR(doif) = sptr;
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<typespec> ::= <intrinsic type> |
   */
  case TYPESPEC1:
    SST_DTYPEP(LHS, sem.gdtype);
    break;
  /*
   *	<typespec> ::= <derived type spec>
   */
  case TYPESPEC2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<derived type spec> ::= <type name> |
   */
  case DERIVED_TYPE_SPEC1:
    break;
  /*
   *	<derived type spec> ::= <pdt>
   */
  case DERIVED_TYPE_SPEC2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<type name> ::= <ident>
   */
  case TYPE_NAME1:
    dtype = get_derived_type(RHS(1), FALSE);
    if (dtype == 0) {
      if (scn.stmtyp == TK_CLASSIS)
        error(155, 4, gbl.lineno,
              "Type specified in CLASS IS must be an "
              "extensible type",
              NULL);
      if (scn.stmtyp == TK_TYPEIS)
        error(155, 4, gbl.lineno,
              "Length type parameter in TYPE IS must "
              "be assumed (*)",
              NULL);
    }
    SST_DTYPEP(LHS, dtype);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pdt> ::= <type name> ( <pdt param list> )
   */
  case PDT1:
    dtype = SST_DTYPEG(RHS(1));
    if (dtype != 0) {
      /* TODO - 'resolve' PDT */
      error(155, 3, gbl.lineno, "Unimplemented feature -",
            "PDT appearing as a type spec in an expression");
    }
    SST_DTYPEP(LHS, dtype);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pdt param list> ::= <pdt param list> , <pdt param> |
   */
  case PDT_PARAM_LIST1:
    break;
  /*
   *	<pdt param list> ::= <pdt param>
   */
  case PDT_PARAM_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pdt param> ::= <expression> |
   */
  case PDT_PARAM1:
    break;
  /*
   *	<pdt param> ::= <id name> = <expression> |
   */
  case PDT_PARAM2:
    break;
  /*
   *	<pdt param> ::= : |
   */
  case PDT_PARAM3:
    break;
  /*
   *	<pdt param> ::= <id name> = : |
   */
  case PDT_PARAM4:
    break;
  /*
   *	<pdt param> ::= * |
   */
  case PDT_PARAM5:
    break;
  /*
   *	<pdt param> ::= <id name> = *
   */
  case PDT_PARAM6:
    break;
  /* ------------------------------------------------------------------
   */
  /*
   *      <classdefault stmt> ::= CLASSDEFAULT <construct name>
   */
  case CLASSDEFAULT_STMT1:
    sem.select_type_seen = 0;
    doif = sem.doif_depth;
    if (doif <= 0 || DI_ID(doif) != DI_SELECT_TYPE) {
      error(155, 4, gbl.lineno,
            "CLASS DEFAULT statement is not within SELECT TYPE", 0);
      break;
    }
    if (DI_CLASS_DEFAULT_LABEL(doif)) {
      error(155, 3, gbl.lineno, "Duplicate CLASS DEFAULT in SELECT TYPE",
            CNULL);
    }
    if (construct_name && DI_NAME(doif) != construct_name)
      err307("SELECT TYPE and CLASS DEFAULT", DI_NAME(doif), construct_name);
    if ((sptr = DI_ACTIVE_SPTR(doif)) > NOSYM) {
      pop_sym(sptr); /* end previous active binding for this SELECT TYPE */
      DI_ACTIVE_SPTR(doif) = 0;
    }

    /* Add branch around */
    RFCNTI(DI_END_SELECT_LABEL(doif));
    ast1 = mk_stmt(A_GOTO, 0);
    ast2 = mk_label(DI_END_SELECT_LABEL(doif));
    A_L1P(ast1, ast2);
    add_stmt(ast1);

    /* class default label */
    sptr = getlab();
    DI_CLASS_DEFAULT_LABEL(doif) = sptr;
    DEFDP(sptr, 1);
    ast = mk_stmt(A_CONTINUE, 0);
    std = add_stmt(ast);
    STD_LABEL(std) = sptr;
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<elseif> ::= ELSEIF <etmp lp>
   */
  case ELSEIF1:
    sem.stats.nodes++;
    doif = sem.doif_depth;
    if (!doif || DI_ID(doif) != DI_IF) {
      sem_err105(doif);
      SST_ASTP(LHS, 0);
      break;
    }
    /*
     * Translate
     *     elseif (<expr>) then
     * to:
     *         goto exit_label
     *     endif
     *     if (<expr>) then
     *
     * At this time generate the 'goto' and 'endif' -- need to do
     * this now so that any allocates generated by <expr> will
     * appear after the endif.
     * NOTE:  If the last statement was a 'goto', 'return', 'stop',
     * no need to add the 'goto exit_label'
     */
    std = STD_PREV(0);
    ast = STD_AST(std);
    if (ast)
      switch (A_TYPEG(ast)) {
      case A_STOP:
      case A_RETURN:
      case A_GOTO:
        break;
      default:
        if (DI_EXIT_LABEL(doif) == 0)
          DI_EXIT_LABEL(doif) = getlab();
        ast = mk_stmt(A_GOTO, 0);
        astlab = mk_label(DI_EXIT_LABEL(doif));
        A_L1P(ast, astlab);
        RFCNTI(DI_EXIT_LABEL(doif));
        (void)add_stmt(ast);
        break;
      }
    ast = mk_stmt(A_ENDIF, 0);
    (void)add_stmt(ast);
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <do begin> ::= <do construct> <label> |
   */
  case DO_BEGIN1:
    do_label = SST_SYMG(RHS(2));
    break;
  /*
   *      <do begin> ::= <do construct>
   */
  case DO_BEGIN2:
    do_label = 0;
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<do construct> ::= DO |
   */
  case DO_CONSTRUCT1:
    construct_name = 0;
    break;
  /*
   *	<do construct> ::= <check construct> : DO
   */
  case DO_CONSTRUCT2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<loop control> ::= <opt comma> <var ref> = <etmp exp> , <etmp *exp>
   *	                   <etmp e3> |
   */
  case LOOP_CONTROL1:
    sem.index_sym_to_pop = SPTR_NULL;
    sptr = mklvalue(RHS(2), 0);
    dtype = DTYPEG(sptr);
    if (!DT_ISREAL(dtype) && !DT_ISINT(dtype)) {
      error(94, 3, gbl.lineno, SYMNAME(sptr), "- DO index variable");
      dtype = DT_INT;
    }
    /* treat logical like integer */
    switch (dtype) {
    default:
      break;
    case DT_BLOG:
      dtype = DT_BINT;
      break;
    case DT_SLOG:
      dtype = DT_SINT;
      break;
    case DT_LOG4:
      dtype = DT_INT4;
      break;
    case DT_LOG8:
      dtype = DT_INT8;
      break;
    }
    chk_scalartyp(RHS(4), dtype, FALSE);
    chk_scalartyp(RHS(6), dtype, FALSE);
    if (SST_ASTG(RHS(7)) == 0)
      /* <e3> was not specified */
      SST_ASTP(RHS(7), astb.i1);
    chk_scalartyp(RHS(7), dtype, FALSE);
    doinfo = get_doinfo(1);
    doinfo->index_var = sptr;
    doinfo->init_expr = SST_ASTG(RHS(4));
    doinfo->limit_expr = SST_ASTG(RHS(6));
    doinfo->step_expr = SST_ASTG(RHS(7));

  // <concurrent control> branches here for directive processing.
  do_directive_processing:
    /*
     * and write ast for DO begin.
     */
    if (sem.collapsed_acc_do) {
      int east = mk_stmt(A_PRAGMA, 0);
      int sptr_ast = mk_id(sptr);
      A_PRAGMATYPEP(east, PR_ACCLOOPPRIVATE);
      A_PRAGMASCOPEP(east, PR_NOSCOPE);
      A_LOPP(east, sptr_ast);
      add_stmt(east);
      --sem.collapsed_acc_do;
      if (sem.expect_acc_do)
        --sem.expect_acc_do;
    } else if (sem.expect_cuf_do) {
      int east = mk_stmt(A_PRAGMA, 0);
      int sptr_ast = mk_id(sptr);
      A_PRAGMATYPEP(east, PR_CUFLOOPPRIVATE);
      A_PRAGMASCOPEP(east, PR_NOSCOPE);
      A_LOPP(east, sptr_ast);
      add_stmt(east);
      --sem.expect_cuf_do;
    }
    if (sem.expect_do) {
      sem.expect_do = FALSE;
      do_lastval(doinfo);
      sem.expect_simd_do = FALSE; /* do_lastval check sem.expect_simd_do */
      if (1) {
        /* only distribute the work if in the outermost
         * parallel region or not in a parallel region.
         */
        sem.collapse_depth = sem.collapse;
        if (sem.collapse_depth < 2) {
          sem.collapse_depth = 0;
          ast = do_parbegin(doinfo);
        } else {
          doinfo->collapse = sem.collapse_depth;
          ast = collapse_begin(doinfo);
        }
      } else {
        sem.collapse_depth = 0;
        ast = do_begin(doinfo);
        DI_DOINFO(sem.doif_depth) = 0; /* remove any chunk info */
      }
    } else if (sem.expect_dist_do) {
      /* Distribute parallel loop construct.
       * Create a distribute loop, then parallel loop.
       * Collapse will apply to parallel loop.
       */
      sem.expect_dist_do = FALSE;
      do_lastval(doinfo);
      ast = do_distbegin(doinfo, do_label, construct_name);
      SST_ASTP(LHS, ast);
      do_label = 0;
#ifdef OMP_OFFLOAD_LLVM
      if (DI_ID(sem.doif_depth - 3) == DI_TARGTEAMSDISTPARDO) {
        int dovar = mk_id(doinfo->index_var);
        ast1 = DI_BTARGET(3);
        ast2 = mk_stmt(A_MP_TARGETLOOPTRIPCOUNT, 0);
        A_LOOPTRIPCOUNTP(ast1, ast2);
        A_DOVARP(ast2, dovar);
        A_M1P(ast2, doinfo->init_expr);
        A_M2P(ast2, doinfo->limit_expr);
        A_M3P(ast2, doinfo->step_expr);
      }
#endif
      break;
    } else if (sem.expect_simd_do) {
      /* Note: set sem.expect_simd_do = FALSE after calling to
       * do_lastvalbecause do_lastval check this flag.
       */
      do_lastval(doinfo);
      sem.expect_simd_do = FALSE;
      sem.collapse_depth = sem.collapse;
      if (sem.collapse_depth < 2) {
        sem.collapse_depth = 0;
        ast = do_begin(doinfo);
      } else {
        doinfo->collapse = sem.collapse_depth;
        ast = collapse_begin(doinfo);
      }
    } else if (!sem.collapse_depth) {
      ast = do_begin(doinfo);
    } else {
      doinfo->collapse = sem.collapse_depth;
      ast = collapse_add(doinfo);
    }
    if (ast)
      std = add_stmt(ast);
    if (rednum == LOOP_CONTROL1) {
      NEED_DOIF(doif, DI_DO);
      DI_DO_POPINDEX(doif) = sem.index_sym_to_pop;
    } else {
      NEED_DOIF(doif, DI_DOCONCURRENT);
      symi = add_symitem(sptr, 0);
      if (doif == 1 || DI_ID(doif - 1) != DI_DOCONCURRENT ||
          DI_CONC_SYMAVL(doif - 1) != sem.doconcurrent_symavl) {
        // First (outermost) control var=triplet.
        DI_CONC_SYMAVL(doif) = sem.doconcurrent_symavl;
        DI_CONC_BLOCK_SYM(doif) = sptr =
            getccsym('b', sem.blksymnum++, ST_BLOCK);
        CCSYMP(sptr, true);
        STARTLINEP(sptr, gbl.lineno);
        ENCLFUNCP(sptr, sem.construct_sptr ? sem.construct_sptr : gbl.currsub);
        sem.construct_sptr = sptr;
        DI_CONC_SYMS(doif) = symi;
      } else {
        // Second or subsequent (inner) control var=triplet.
        // Some fields will only be set for an innermost loop.
        assert(doif > 1 && DI_ID(doif - 1) == DI_DOCONCURRENT,
               "missing outer doconcurrent doif slot", 0, ERR_Severe);
        sem.doif_base[doif] = sem.doif_base[doif - 1];
        SYMI_NEXT(DI_CONC_LAST_SYM(doif)) = symi;
      }
      DI_CONC_COUNT(doif)++;
      DI_CONC_LAST_SYM(doif) = symi;
      if (ast)
        STD_BLKSYM(std) = DI_CONC_BLOCK_SYM(doif);
    }
    DI_DO_LABEL(doif) = do_label;
    DI_DO_AST(doif) = ast;
    DI_DOINFO(doif) = doinfo;
    DI_NAME(doif) = construct_name;
    direct_loop_enter();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<loop control> ::= <dowhile> <etmp lp> <expression> ) |
   */
  case LOOP_CONTROL2:
    ast2 = gen_logical_if_expr(RHS(3));
    ast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ast, ast2);
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<loop control> ::= <doconcurrent> <concurrent header>
   *                       <concurrent locality>
   */
  case LOOP_CONTROL3:
    // Set the DO CONCURRENT body marker to the last header, mask, or
    // locality assignment std.  Shift to its successor before use.
    DI_CONC_BODY_STD(sem.doif_depth) = STD_LAST;
    sem.doconcurrent_symavl = SPTR_NULL;
    sem.doconcurrent_dtype = DT_NONE;
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<dowhile> ::= <opt comma> WHILE
   */
  case DOWHILE1:
    NEED_DOIF(doif, DI_DOWHILE);
    DI_DO_LABEL(doif) = do_label;
    doinfo = get_doinfo(1);
    doinfo->index_var = 0; /* marks doinfo for a DOWHILE */
    DI_DOINFO(doif) = doinfo;
    DI_NAME(doif) = construct_name;
    if (scn.currlab)
      DI_TOP_LABEL(doif) = scn.currlab;
    else
      scn.currlab = DI_TOP_LABEL(doif) = getlab();
    ast = mk_stmt(A_CONTINUE, 0);
    (void)add_stmt(ast);
    direct_loop_enter();
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<doconcurrent> ::= <opt comma> CONCURRENT
   */
  case DOCONCURRENT1:
    sem.doconcurrent_symavl = stb.stg_avail;
    sem.doconcurrent_dtype = DT_NONE;
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<concurrent header> ::= ( <concurrent type> <concurrent list>
   *	                        <opt mask expr> )
   */
  case CONCURRENT_HEADER1:
    // <concurrent type> processing was done upstream.
    ast1 = SST_ASTG(RHS(4));
    doif = sem.doif_depth;
    switch (DI_ID(doif)) {
    case DI_DOCONCURRENT:
      // <concurrent list> processing was done upstream; process mask.
      if (ast1) {
        if (A_SHAPEG(ast1)) {
          error(1042, ERR_Severe, gbl.lineno, "DO CONCURRENT", CNULL);
        } else {
          ast = mk_stmt(A_IFTHEN, 0);
          A_IFEXPRP(ast, ast1);
          DI_CONC_MASK_STD(doif) = add_stmt(ast);
        }
      }
      SST_ASTP(LHS, 0);
      break;
    case DI_FORALL:
      // Generate a FORALL ast.
      if (ast1 && A_SHAPEG(ast1)) {
        error(1042, ERR_Severe, gbl.lineno, "FORALL", CNULL);
        ast1 = 0;
      }
      start_astli();
      for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
        astli = add_astli();
        ASTLI_SPTR(astli) = itemp->t.sptr;
        ASTLI_TRIPLE(astli) = itemp->ast;
      }
      DI_FORALL_AST(doif) = ast = mk_stmt(A_FORALL, 0);
      A_LISTP(ast, ASTLI_HEAD);
      A_IFEXPRP(ast, ast1);
      SST_ASTP(LHS, ast);
      break;
    default:
      interr("semant3: invalid doif id", DI_ID(doif), ERR_Severe);
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<concurrent type> ::= |
   */
  case CONCURRENT_TYPE1:
    break;
  /*
   *	<concurrent type> ::= <type spec> ::
   */
  case CONCURRENT_TYPE2:
    dtype = SST_DTYPEG(RHS(1));
    if (sem.doconcurrent_symavl) {
      sem.doconcurrent_dtype = dtype;
    } else {
      assert(sem.doif_depth && DI_ID(sem.doif_depth) == DI_FORALL,
             "semant3: expecting doconcurrent or forall control", 0,
             ERR_Severe);
      DI_FORALL_DTYPE(sem.doif_depth) = dtype;
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<concurrent list> ::= <concurrent list> , <concurrent control> |
   *	<concurrent list> ::= <concurrent control>
   */
  case CONCURRENT_LIST1:
  case CONCURRENT_LIST2:
    doif = sem.doif_depth;
    switch (DI_ID(doif)) {
    case DI_DOCONCURRENT:
      // <concurrent control> processing was done upstream.
      break;
    case DI_FORALL:
      count = rednum == CONCURRENT_LIST1 ? 3 : 1; // RHS symbol count
      sptr = SST_SYMG(RHS(count));
      if (count == 3)
        for (itemp = SST_BEGG(RHS(1)); itemp != ITEM_END; itemp = itemp->next)
          if (itemp->t.sptr == sptr) // repeat use of index var
            error(1053, ERR_Severe, gbl.lineno, "FORALL", CNULL);
      itemp = (ITEM *)getitem(0, sizeof(ITEM));
      itemp->next = ITEM_END;
      itemp->t.sptr = sptr;              // forall variable
      itemp->ast = SST_ASTG(RHS(count)); // forall triplet
      if (count == 1)
        SST_BEGP(LHS, itemp);
      else
        SST_ENDG(RHS(1))->next = itemp;
      SST_ENDP(LHS, itemp);
      DI_IDXLIST(doif) = add_symitem(itemp->t.sptr, DI_IDXLIST(doif));
      break;
    default:
      interr("semant3: invalid doif id", DI_ID(doif), ERR_Severe);
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<concurrent control> ::= <var ref> = <expression> : <expression>
   *                             <opt stride>
   */
  case CONCURRENT_CONTROL1:
    if (sem.doconcurrent_symavl) {
      sptr = mklvalue(RHS(1), 0); // 0 = do index var
      dtype = DTYPEG(sptr);
      if (!DT_ISINT(dtype) || DT_ISLOG(dtype)) {
        error(94, ERR_Severe, gbl.lineno, SYMNAME(sptr), // 2018-C1122
              "- DO CONCURRENT index variable");
        dtype = DT_INT;
      }
      (void)chk_scalar_inttyp(RHS(3), dtype, "DO CONCURRENT lower bound");
      (void)chk_scalar_inttyp(RHS(5), dtype, "DO CONCURRENT upper bound");
      if (SST_IDG(RHS(6)))
        (void)chk_scalar_inttyp(RHS(6), dtype, "DO CONCURRENT stride");
      doinfo = get_doinfo(1);
      doinfo->index_var = sptr;
      doinfo->init_expr = SST_ASTG(RHS(3));
      doinfo->limit_expr = SST_ASTG(RHS(5));
      if (!SST_ASTG(RHS(6)))
        SST_ASTP(RHS(6), astb.i1);
      doinfo->step_expr = SST_ASTG(RHS(6));
      goto do_directive_processing;
    } else {
      assert(sem.doif_depth && DI_ID(sem.doif_depth) == DI_FORALL,
             "semant3: expecting doconcurrent or forall control", 0,
             ERR_Severe);
      sptr = mklvalue(RHS(1), 5); // 5 = forall index var
      dtype = DTYPEG(sptr);
      if (!DT_ISINT(dtype) || DT_ISLOG(dtype)) {
        error(94, ERR_Severe, gbl.lineno, SYMNAME(sptr), // 2018-C1122
              "- FORALL index variable");
        dtype = DT_INT;
      }
      (void)chk_scalar_inttyp(RHS(3), dtype, "FORALL lower bound");
      (void)chk_scalar_inttyp(RHS(5), dtype, "FORALL upper bound");
      if (SST_IDG(RHS(6)) != S_NULL)
        (void)chk_scalar_inttyp(RHS(6), dtype, "FORALL stride");
      ast = mk_triple((int)SST_ASTG(RHS(3)), (int)SST_ASTG(RHS(5)),
                      (int)SST_ASTG(RHS(6)));
      SST_ASTP(LHS, ast);
      SST_SYMP(LHS, sptr);
      if (flg.smp) {
        add_no_scope_sptr(sptr, sptr, gbl.lineno);
        DI_NOSCOPE_FORALL(sem.doif_depth) = 1;
        is_dovar_sptr(sptr);
      }
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<opt mask expr> ::=  |
   */
  case OPT_MASK_EXPR1:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<opt mask expr> ::= , <mask expr>
   */
  case OPT_MASK_EXPR2:
    *LHS = *RHS(2);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<concurrent locality> ::=
   *	<concurrent locality> ::= <locality spec list>
   */
  case CONCURRENT_LOCALITY1:
  case CONCURRENT_LOCALITY2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<locality spec list> ::= <locality spec list> <locality spec> |
   *	<locality spec list> ::= <locality spec>
   */
  case LOCALITY_SPEC_LIST1:
  case LOCALITY_SPEC_LIST2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<locality spec> ::= <locality kind> ( <locality name list> ) |
   */
  case LOCALITY_SPEC1:
    break;
  /*
   *	<locality spec> ::= DEFAULT ( NONE )
   */
  case LOCALITY_SPEC2:
    if (DI_CONC_NO_DEFAULT(sem.doif_depth))              // repeat DEFAULT(NONE)
      error(1047, ERR_Severe, gbl.lineno, CNULL, CNULL); // 2018-C1127
    DI_CONC_NO_DEFAULT(sem.doif_depth) = true;
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<locality kind> ::= LOCAL |
   */
  case LOCALITY_KIND1:
    DI_CONC_KIND(sem.doif_depth) = TK_LOCAL;
    break;
  /*
   *	<locality kind> ::= LOCAL_INIT |
   */
  case LOCALITY_KIND2:
    DI_CONC_KIND(sem.doif_depth) = TK_LOCAL_INIT;
    break;
  /*
   *	<locality kind> ::= SHARED
   */
  case LOCALITY_KIND3:
    DI_CONC_KIND(sem.doif_depth) = TK_SHARED;
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<locality name list> ::= <locality name list> , <ident> |
   *	<locality name list> ::= <ident>
   */
  case LOCALITY_NAME_LIST1:
  case LOCALITY_NAME_LIST2:
    count = rednum == LOCALITY_NAME_LIST1 ? 3 : 1; // RHS symbol count
    sptr = SST_SYMG(RHS(count));
    doif = sem.doif_depth;
    msg = 0;
    switch (STYPEG(sptr)) {
    case ST_PD:
      if (DI_CONC_KIND(doif) != TK_LOCAL) {
        msg = 1044; // 2018-C1124 -- sptr is not a variable
        break;
      }
      FLANG_FALLTHROUGH;
    case ST_UNKNOWN: // potential keyword as an identifier
    case ST_IDENT:
    case ST_VAR:
    case ST_ARRAY:
      if (sym_in_sym_list(sptr, DI_CONC_SYMS(doif))) {
        for (i = DI_CONC_COUNT(doif), symi = DI_CONC_SYMS(doif); i && !msg;
             --i, symi = SYMI_NEXT(symi))
          if (sptr == SYMI_SPTR(symi))
            msg = 1045; // 2018-C1125 -- sptr is an index var
        if (!msg)
          msg = 1046; // 2018-C1126 -- repeat appearance for sptr
      }
      break;
    default:
      msg = 1044; // 2018-C1124 -- sptr is not a variable
    }
    if (msg) {
      if (!sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(doif))) {
        error(msg, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL);
        DI_CONC_ERROR_SYMS(doif) = add_symitem(sptr, DI_CONC_ERROR_SYMS(doif));
      }
      break;
    }
    DCLCHK(sptr);
    switch (DI_CONC_KIND(doif)) {
    case TK_LOCAL_INIT:
      if (sptr >= DI_CONC_SYMAVL(doif)) {
        error(1062, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL);
        DI_CONC_ERROR_SYMS(doif) = add_symitem(sptr, DI_CONC_ERROR_SYMS(doif));
        break;
      }
      FLANG_FALLTHROUGH;
    case TK_LOCAL:
      if (sptr < DI_CONC_SYMAVL(doif)) {
        // sptr is external to the loop; get a construct instance.
        if (STYPEG(sptr) == ST_PD) {
          int dcld = DCLDG(sptr);
          sptr = insert_sym(sptr);
          CONSTRUCTSYMP(sptr, true);
          DCLDP(sptr, dcld);
        } else {
          sptr = insert_dup_sym(sptr);
        }
        if (SDSCG(sptr)) {
          MIDNUMP(sptr, 0);
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
        }
      }
      BINDP(sptr, 0);
      INTENTP(sptr, 0);
      INTERNALP(sptr, gbl.internal > 1);
      PASSBYVALP(sptr, 0);
      PROTECTEDP(sptr, 0);
      SAVEP(sptr, 0);
      SST_IDP(&sst, S_IDENT);
      SST_DTYPEP(&sst, DTYPEG(sptr));
      SST_SYMP(&sst, sptr);
      s = NULL;
      if (ALLOCATTRG(sptr))
        s = "has the ALLOCATABLE attribute";
      else if (INTENTG(sptr) == INTENT_IN)
        s = "has the INTENT(IN) attribute";
      else if (OPTARGG(sptr))
        s = "has the OPTIONAL attribute";
      else if (has_finalized_component(sptr))
        s = "has finalizable type";
      else if (ARGG(sptr) && CLASSG(sptr) && !POINTERG(sptr))
        s = "is a nonpointer polymorphic dummy argument";
      else if (ASUMSZG(sptr))
        s = "is an assumed shape array";
      else if (mklvalue(&sst, 1) == 0)
        s = "is not permitted in a variable definition context";
      if (s) {
        if (!sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(doif))) {
          error(1048, ERR_Severe, gbl.lineno, SYMNAME(sptr), s); // 2018-C1128
          DI_CONC_ERROR_SYMS(doif) =
              add_symitem(sptr, DI_CONC_ERROR_SYMS(doif));
        }
        break;
      }
      if (DI_CONC_KIND(doif) == TK_LOCAL)
        break;
      // LOCAL_INIT assignment
      if (POINTERG(sptr)) {
        SST_ASTP(RHS(count), mk_id(SST_SYMG(RHS(count))));
        (void)add_stmt(assign_pointer(&sst, RHS(count)));
      } else {
        SST_ASTP(&sst, 0);
        (void)add_stmt(assign(&sst, RHS(count)));
      }
      ASSNP(sptr, 0);
      break;
    case TK_SHARED:
      break;
    default:
      interr("semant3: invalid locality", DI_CONC_KIND(doif), ERR_Severe);
    }
    // DI_CONC_SYMS list contains all index vars followed by LOCAL,
    // LOCAL_INIT, and SHARED vars in declaration order.
    assert(DI_CONC_LAST_SYM(doif), "missing DO CONCURRENT index name(s)", 0,
           ERR_Warning);
    SYMI_NEXT(DI_CONC_LAST_SYM(doif)) = symi = add_symitem(sptr, 0);
    DI_CONC_LAST_SYM(doif) = symi;
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <where clause> ::= <where construct> ( <mask expr> )
   */
  case WHERE_CLAUSE1:
    shape = A_SHAPEG(SST_ASTG(RHS(3)));
    NEED_DOIF(doif, DI_WHERE);
    DI_NAME(doif) = construct_name;
    if (shape)
      DI_SHAPEDIM(doif) = SHD_NDIM(shape);
    *LHS = *RHS(3);
    sem.pgphase = PHASE_EXEC; /* set now, since may have where(...) stmt */
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<where construct> ::= WHERE |
   */
  case WHERE_CONSTRUCT1:
    construct_name = 0;
    break;
  /*
   *	<where construct> ::= <check construct> : WHERE
   */
  case WHERE_CONSTRUCT2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <mask expr> ::= <expression>
   */
  case MASK_EXPR1:
    (void)mkexpr(RHS(1));
    if (!DT_ISLOG(DDTG(SST_DTYPEG(RHS(1))))) {
      if (scn.stmtyp == TK_FORALL)
        error(155, 3, gbl.lineno,
              "The FORALL mask expression must be type logical", CNULL);
      else if (scn.stmtyp == TK_WHERE)
        error(155, 3, gbl.lineno, "The WHERE expression must be type logical",
              CNULL);
      else
        error(155, 3, gbl.lineno, "The mask expression must be type logical",
              CNULL);
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <allocation stmt> ::= ALLOCATE ( <alloc list> <alloc cntl> ) |
   */
  case ALLOCATION_STMT1:
    if (alloc_error) {
      alloc_error = FALSE;
      SST_ASTP(LHS, 0);
      break;
    }
    rhstop = 3;
    dtype = 0;
    typed_alloc = 0;
    check_alloc_clauses(SST_BEGG(RHS(rhstop)), SST_BEGG(RHS(rhstop + 1)),
                        &alloc_source, &mold_or_src);
    orig_alloc_source = alloc_source;
    if (!alloc_error && !alloc_source) {
      for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
        int sptr2, dtype2;
        sptr2 = memsym_of_ast(itemp->ast);
        dtype2 = DTYPEG(sptr2);
        if (DTY(dtype2) == TY_DERIVED && ABSTRACTG(DTY(dtype2 + 3))) {
          error(155, 3, gbl.lineno, "illegal use of abstract type",
                SYMNAME(sptr2));
        } else {
          switch (A_TYPEG(itemp->ast)) {
          case A_ID:
          case A_MEM:
            if (DTY(dtype2) == TY_ARRAY && (ALLOCG(sptr2) || POINTERG(sptr2))) {
              error(198, 3, gbl.lineno, "Illegal use of", SYMNAME(sptr2));
              alloc_error = TRUE;
            }
            break;
          }
        }
      }
    }
  shared_alloc_stmt:
    if (not_in_forall("ALLOCATE"))
      break;
    if (alloc_error) {
      alloc_error = FALSE;
      SST_ASTP(LHS, 0);
      break;
    }
    /*
     * Check specification clauses and, if a SOURCE specification is
     * present, get the source ast.
     */
    check_alloc_clauses(SST_BEGG(RHS(rhstop)), SST_BEGG(RHS(rhstop + 1)),
                        &alloc_source, &mold_or_src);
    orig_alloc_source = alloc_source;
    if (alloc_source) {
      /* Allocated object receives the type of the source. */
      if (A_TYPEG(alloc_source) == A_SUBSCR) {
        /* FS#17729: to avoid lower error - bad OP type */
        alloc_source = A_LOPG(alloc_source);
      }
      dtype = A_DTYPEG(alloc_source);
      if (DTY(dtype) == TY_ARRAY) {
        dtype = dup_array_dtype(dtype);
      }
      if (typed_alloc) {
        error(155, 3, gbl.lineno,
              "SOURCE= and type-spec appearing in same"
              " ALLOCATE statement is not allowed",
              NULL);
      }
      if ((DTY(dtype) == TY_ARRAY && DTY(DTY(dtype + 1)) == TY_DERIVED &&
           UNLPOLYG(DTY(DTY(dtype + 1) + 3))) ||
          (DTY(dtype) == TY_DERIVED && UNLPOLYG(DTY(dtype + 3)))) {
        /* TBD - should string_expr_length() handle this
         * case and return 0 instead of an ICE????
         */
        sem.gcvlen = 0;
      } else if (DTY(DDTG(dtype)) != TY_CHAR && DTY(DDTG(dtype)) != TY_NCHAR) {
        sem.gcvlen = 0;
      } else
        sem.gcvlen = string_expr_length(alloc_source);
    }
    for (itemp = SST_BEGG(RHS(rhstop)); itemp != ITEM_END;
         itemp = itemp->next) {
      if (alloc_source) {
        /* FS#19472: Support F2008 sourced allocation convention of
         * destination array inheriting bounds of source array when
         * bounds are not explicitly specified.
         */
        int dest_ast = itemp->ast;
        DTYPE dest_dtype = A_DTYPEG(dest_ast);

        if (A_TYPEG(dest_ast) != A_SUBSCR && is_array_dtype(dest_dtype)) {
          /* An array is being allocated with shape assumed from the
           * MOLD= or SOURCE= expression.
           */
          int lb_asts[MAXRANK], ub_asts[MAXRANK];
          int rank = get_ast_bounds(lb_asts, ub_asts, orig_alloc_source, dtype);
          if (rank < 1) {
            /* MOLD= or SOURCE= is scalar, so set all the bounds to 1:1.
             */
            rank = ADD_NUMDIM(dest_dtype);
            for (i = 0; i < rank; ++i) {
              lb_asts[i] = ub_asts[i] = astb.bnd.one;
            }
          }
          itemp->ast = add_bounds_subscripts(itemp->ast, rank, lb_asts, ub_asts,
                                             DDTG(dest_dtype));
        }
      }
      if (alloc_source && !sem.gcvlen &&
          (DDTG(A_DTYPEG(itemp->ast)) == DT_DEFERCHAR ||
           DDTG(A_DTYPEG(itemp->ast)) == DT_DEFERCHAR)) {
        /* FS#20580: Generate error here instead of ICE
         * in gen_alloc_dealloc()
         */
        error(155, 4, gbl.lineno,
              "ALLOCATE Source Specification is incompatible "
              "with type of object ",
              SYMNAME(memsym_of_ast(itemp->ast)));
      }
      bef = STD_PREV(0); /* std preceding the allocate */
      ast =
          gen_alloc_dealloc(TK_ALLOCATE, itemp->ast, SST_BEGG(RHS(rhstop + 1)));
      if (!alloc_source && !typed_alloc) {
        int dtype2, sptr2;
        sptr2 = memsym_of_ast(itemp->ast);
        dtype2 = DTYPEG(sptr2);
        if (DTY(dtype2) == TY_DERIVED && UNLPOLYG(DTY(dtype2 + 3))) {
          error(155, 3, gbl.lineno,
                "ALLOCATE with unlimited polymorphic "
                "object requires type-spec or SOURCE=",
                NULL);
        }
      }

      if (dtype && typed_alloc) {
        /* Store dtype to allocate object. This comes from type
         * allocation production below. We later generate the type
         * assignment in rewrite_calls of func.c
         */
        DTYPE dty, base;
        SPTR dest, tmp, sdsc;
        int arg0, arg1;
        switch (A_TYPEG(itemp->ast)) {
        case A_ID:
        case A_LABEL:
        case A_ENTRY:
        case A_SUBSCR:
        case A_SUBSTR:
        case A_MEM:
          dty = DTYPEG(memsym_of_ast(itemp->ast));
          if (DTY(dty) == TY_ARRAY) {
            /* FS#19369: Need to create an array dtype for typed
             * allocation.
             */
            ADSC *ad;
            dty = dup_array_dtype(dty);
            DTY(dty + 1) = dtype;
            base = dtype;
            dtype = dty;
            ad = AD_DPTR(dty);
            if (AD_DEFER(ad)) {
              /* FS#20849: Make sure size_of() processes this dtype
               * when we generate the len argument to RTE_instance()
               * by setting AD_DEFER to 0.
               */
              AD_DEFER(ad) = 0;
            }
          } else {
            base = dtype;
          }
        }
        ast = rewrite_ast_with_new_dtype(ast, dtype);
        itemp->ast = rewrite_ast_with_new_dtype(itemp->ast, dtype);
        dest = memsym_of_ast(itemp->ast);
        if (SDSCG(dest) && DTY(DDTG(dtype)) == TY_CHAR && is_unl_poly(dest)) {
          /* FS#20580: Set up destination descriptor where
           * unlimited polymorphic object is getting allocated
           * with a string.
           */
          int val, assn, dast;

          if (string_length(DDTG(dtype))) {
            val = mk_cval1(string_length(DDTG(dtype)), DT_INT);
          } else {
            val = DTY(DDTG(dtype) + 1);
          }

          if (val) {
            dast = check_member(itemp->ast, get_byte_len(SDSCG(dest)));
            assn = mk_assn_stmt(dast, val, DT_INT);
            add_stmt(assn);
          }
        } else if (CLASSG(dest)) {
          /* For typed allocation of a normal polymorphic object, make sure
           * we allocate the size of the specified type by creating a
           * temporary object of that type and use sourced allocation
           * to ensure proper size. Sourced allocation is accomplished by
           * storing the temp in the A_ALLOC ast's A_START field below.
           */
          tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, base, sem.sc);
          A_STARTP(ast, mk_id(tmp));

          /* We also need to set the type field of the ast's type
           * descriptor. arg0 is the destination descriptor argument for the
           * set_type call.
           */
          sdsc = STYPEG(dest) == ST_MEMBER ? get_member_descriptor(dest) : 0;
          if (sdsc > NOSYM) {
            arg0 = check_member(itemp->ast, mk_id(sdsc));
          } else {
            arg0 = mk_id(get_type_descr_arg(gbl.currsub, dest));
          }

          /* source descriptor argument for set_type call */
          if (DTY(DDTG(base)) == TY_DERIVED) {
            /* derived type uses a regular type descriptor */
            arg1 = mk_id(get_static_type_descriptor(tmp));
          } else {
            /* non-derived type uses its runtime intrinsic type code */
            int type_code = dtype_to_arg(DTY(DDTG(base)));
            arg1 = mk_cval1(type_code, DT_INT);
            arg1 = mk_unop(OP_VAL, arg1, DT_INT);
          }

          /* generate set_type call */
          ast = mk_set_type_call(arg0, arg1, DTY(DDTG(base)) != TY_DERIVED);
          std = add_stmt(ast);
        }
      }
      if (!alloc_error) {
        if (itemp->t.flitmp && bef != itemp->t.flitmp->last) {
          /* Move the bounds assignments immediately before the object's
           * allocate.
           */
          i = STD_PREV(itemp->t.flitmp->first); /* before first assn */
          j = STD_NEXT(itemp->t.flitmp->last);  /* after last assn */
          /*  unlink bounds assignments  */
          STD_NEXT(i) = j;
          STD_PREV(j) = i;
          if (STD_LABEL(itemp->t.flitmp->first)) {
            /* if the first assignment is labeled, move the label
             * as well.
             */
            STD_LABEL(j) = STD_LABEL(itemp->t.flitmp->first);
            STD_LABEL(itemp->t.flitmp->first) = 0;
          }
          /*  insert the bounds assigments */
          j = STD_NEXT(bef);                      /* allocate */
          STD_NEXT(bef) = itemp->t.flitmp->first; /* before allocate */
          STD_PREV(itemp->t.flitmp->first) = bef;
          STD_NEXT(itemp->t.flitmp->last) = j; /* after last assn */
          STD_PREV(j) = itemp->t.flitmp->last;
        }
        if (!alloc_source && DTY(DDTG(A_DTYPEG(itemp->ast))) == TY_DERIVED) {
          gen_derived_type_alloc_init(itemp);
        }
        if (alloc_source) {
          /* Allocated object receives the value of the source. */
          int src, dest, dest_dtype, src_dtype, tag1, tag2;
          int sdsc_mem, src_sdsc_ast, dest_sdsc_ast, argt;
          FtnRtlEnum fidx;

          src_sdsc_ast = 0;
          ast = rewrite_ast_with_new_dtype(ast, dtype);
          alloc_source = orig_alloc_source;
          switch (A_TYPEG(alloc_source)) { /* FS#19312 */
          case A_ID:
          case A_LABEL:
          case A_ENTRY:
          case A_SUBSCR:
          case A_SUBSTR:
          case A_MEM:
            src = memsym_of_ast(alloc_source);
            src_dtype = DTYPEG(src);
            ADDRTKNP(src, 1); /* TBD - do not optimize away the source */
            break;
          default:
            src = 0;
            src_dtype = A_DTYPEG(alloc_source);
          }
          dest = memsym_of_ast(itemp->ast);
          if (mold_or_src == TK_SOURCE) {
            dest_dtype = DTYPEG(dest);
            if (!is_unl_poly(dest) && !is_unl_poly(src) &&
                (dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR ||
                 dest_dtype == DT_DEFERCHAR || dest_dtype == DT_DEFERNCHAR ||
                 DDTG(dtype) == DT_DEFERCHAR || DDTG(dtype) == DT_DEFERNCHAR ||
                 DDTG(dest_dtype) == DT_DEFERCHAR ||
                 DDTG(dest_dtype) == DT_DEFERNCHAR)) {
              ast = mk_assn_stmt(itemp->ast, alloc_source, dtype);
              add_stmt(ast);
            } else {
              if (DTY(src_dtype) == TY_DERIVED) {
                tag1 = DTY(src_dtype + 3);
              } else {
                int dt = src_dtype;
                tag1 = 0;
                if (DTY(dt) == TY_ARRAY) {
                  dt = DTY(dt + 1);
                  if (DTY(dt) == TY_DERIVED) {
                    tag1 = DTY(dt + 3);
                  }
                }
              }
              if (DTY(dest_dtype) == TY_DERIVED) {
                tag2 = DTY(dest_dtype + 3);
              } else {
                int dt = dest_dtype;
                tag2 = 0;
                if (DTY(dt) == TY_ARRAY) {
                  dt = DTY(dt + 1);
                  if (DTY(dt) == TY_DERIVED) {
                    tag2 = DTY(dt + 3);
                  }
                }
              }
              src_dtype = A_DTYPEG(orig_alloc_source);
              if (((CLASSG(src) && !CLASSG(dest)) ||
                   !eq_dtype2(DTYPEG(dest), DTYPEG(src), CLASSG(dest)))) {

                /* for CLASS we need to copy to a temp if it is an
                 * section array
                 * for vector array - it is also OK to do it here(if it
                 * let
                 * it through it will be done at rewrite_sub_args but
                 * only
                 * vector array).
                 */
                if ((CLASSG(src) && !CLASSG(dest)) || CLASSG(dest)) {
                  if (A_TYPEG(orig_alloc_source) == A_SUBSCR) {
                    if (DTY(src_dtype) == TY_ARRAY) {
                      int ptr_ast;
                      int eldtype, asn_ast;
                      int subscr[MAXRANK];
                      int temp_arr;

                      eldtype = DDTG(src_dtype);
                      temp_arr = mk_assign_sptr(orig_alloc_source, "a", subscr,
                                                eldtype, &ptr_ast);
                      asn_ast =
                          mk_assn_stmt(ptr_ast, orig_alloc_source, eldtype);

                      if (ALLOCG(temp_arr))
                        gen_alloc_dealloc(TK_ALLOCATE, ptr_ast, 0);
                      add_stmt(asn_ast);
                      check_and_add_auto_dealloc_from_ast(itemp->ast);

                      if (ALLOCG(temp_arr))
                        check_and_add_auto_dealloc_from_ast(ptr_ast);

                      alloc_source = ptr_ast;
                      src = temp_arr;
                    } else {
                      int ptr, ptr_ast, callast;

                      ptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, src_dtype,
                                        sem.sc);
                      POINTERP(ptr, TRUE);
                      ptr_ast = mk_id(ptr);
                      ast = rewrite_ast_with_new_dtype(ast, src_dtype);
                      callast = add_ptr_assign(ptr_ast, orig_alloc_source, 0);
                      add_stmt(callast);
                      alloc_source = ptr_ast;
                      CLASSP(ptr, CLASSG(src));

                      set_descriptor_rank(TRUE);
                      get_static_descriptor(ptr);
                      set_descriptor_rank(FALSE);
                      get_all_descriptors(ptr);

                      ALLOCDESCP(ptr, TRUE);
                      src_sdsc_ast = mk_id(SDSCG(ptr));
                      sym_is_refd(ptr);
                      sym_is_refd(SDSCG(ptr));
                      CCSYMP(SDSCG(ptr), TRUE);
                      src = ptr;
                    }
                  }
                }

                if (A_TYPEG(orig_alloc_source) == A_SUBSCR &&
                    A_SHAPEG(orig_alloc_source))
                  numdim = SHD_NDIM(A_SHAPEG(orig_alloc_source));
                else if (DTY(src_dtype) == TY_ARRAY) {
                  numdim = DTY((src_dtype) + 2)
                               ? ADD_NUMDIM(src_dtype)
                               : SHD_NDIM(A_SHAPEG(orig_alloc_source));
                } else {
                  numdim = 0;
                }
                if ((DTY(src_dtype) == TY_ARRAY &&
                     DTY(dest_dtype) != TY_ARRAY) ||
                    (DTY(src_dtype) == TY_ARRAY &&
                     DTY(dest_dtype) == TY_ARRAY &&
                     numdim != ADD_NUMDIM(dest_dtype))) {
                  error(155, 3, gbl.lineno,
                        "ALLOCATE object must have same rank as"
                        " SOURCE= expression",
                        SYMNAME(dest));
                } else if (UNLPOLYG(tag1) && !UNLPOLYG(tag2)) {
                  int orig_tag = A_DTYPEG(orig_alloc_source);
                  if (DTY(orig_tag) == TY_ARRAY)
                    orig_tag = DTY(orig_tag + 1);
                  if (DTY(orig_tag) == TY_DERIVED &&
                      UNLPOLYG(DTY(orig_tag + 3))) {
                    error(155, 3, gbl.lineno,
                          "ALLOCATE Source Specification is"
                          " incompatible with type of object ",
                          SYMNAME(dest));
                  }
                }
              }
              if (CLASSG(src) || UNLPOLYG(tag2) || has_poly_mbr(src, 1) ||
                  has_type_parameter(DTYPEG(src)) || CLASSG(dest)) {
                /* polymorphic source or destination,
                 * call RTE_poly_asn() or RTE_poly_asn_src_intrin()
                 */

                int new_sym, dty2, dest_ast;
                int flag_con = 2;
                dty2 = dtype;
                dest_ast = 0;
                fidx = RTE_poly_asn;

                if (DTY(dty2) == TY_ARRAY) {
                  dty2 = DTY(dty2 + 1);
                  if (DTY(dest_dtype) == TY_ARRAY) {
                    int dty = DTY(dest_dtype + 1);
                    if (DTY(dty) == TY_DERIVED && UNLPOLYG(DTY(dty + 3)) &&
                        dty2 != DT_DEFERCHAR) {
                      new_sym = get_unl_poly_sym(dty2);
                      chkstruct(DTYPEG(new_sym));
                      dest_dtype = dup_array_dtype(dest_dtype);
                      DTY(dest_dtype + 1) = DTYPEG(new_sym);
                      DTYPEP(dest, dest_dtype);
                      dest_ast =
                          rewrite_ast_with_new_dtype(itemp->ast, dest_dtype);
                      itemp->ast = dest_ast;
                      get_all_descriptors(memsym_of_ast(dest_ast));
                    }
                  }
                }
                if (!src_sdsc_ast && CLASSG(src) && STYPEG(src) == ST_MEMBER) {
                  int unl_poly_src;
                  sdsc_mem = get_member_descriptor(src);
                  /* TBD: Is the choice of descriptor really dependent
                   * on a class(*) src?
                   */

                  unl_poly_src = src_dtype;
                  if (DTY(unl_poly_src) == TY_ARRAY)
                    unl_poly_src = DTY(unl_poly_src + 1);
                  if (DTY(unl_poly_src) == TY_DERIVED &&
                      UNLPOLYG(DTY(unl_poly_src + 3)) &&
                      STYPEG(SDSCG(src)) == ST_MEMBER) {
                    unl_poly_src = 1;
                  } else {
                    unl_poly_src = 0;
                  }
                  src_sdsc_ast =
                      check_member(alloc_source, (SDSCG(src) && unl_poly_src)
                                                     ? mk_id(SDSCG(src))
                                                     : mk_id(sdsc_mem));

                } else {
                  src_sdsc_ast = 0;
                }

                if (CLASSG(dest) && STYPEG(dest) == ST_MEMBER) {
                  int unl_poly_dest;
                  sdsc_mem = get_member_descriptor(dest);
                  /* TBD: Is the choice of descriptor really dependent
                   * on a class(*) dest?
                   */
                  unl_poly_dest = dest_dtype;
                  if (DTY(unl_poly_dest) == TY_ARRAY)
                    unl_poly_dest = DTY(unl_poly_dest + 1);
                  if (DTY(unl_poly_dest) == TY_DERIVED &&
                      UNLPOLYG(DTY(unl_poly_dest + 3)) /*&& !src_sdsc_ast*/
                      && STYPEG(SDSCG(dest)) == ST_MEMBER) {
                    unl_poly_dest = 1;
                  } else {
                    unl_poly_dest = 0;
                  }
                  dest_sdsc_ast =
                      check_member(itemp->ast, (SDSCG(dest) && unl_poly_dest)
                                                   ? mk_id(SDSCG(dest))
                                                   : mk_id(sdsc_mem));
                } else {
                  dest_sdsc_ast = 0;
                }

                if (!dest_sdsc_ast && !SDSCG(dest)) {
                  get_static_type_descriptor(dest);
                }

                if (!src_sdsc_ast && !SDSCG(src)) {
                  get_static_type_descriptor(src);
                }
                if ((!needs_descriptor(src) && !SDSCG(src) &&
                     (DTY(src_dtype) != TY_DERIVED &&
                      (DTY(src_dtype) != TY_ARRAY ||
                       DTY(src_dtype + 1) != TY_DERIVED))) ||
                    (!CLASSG(src) && (DTY(src_dtype) != TY_DERIVED &&
                                      (DTY(src_dtype) != TY_ARRAY ||
                                       DTY(src_dtype + 1) != TY_DERIVED))) ||
                    (!src_sdsc_ast && !get_type_descr_arg2(gbl.currsub, src))) {
                  if (DTYG(src_dtype) == TY_CHAR && is_unl_poly(dest)) {
                    fidx = RTE_poly_asn_src_intrin;
                    src_sdsc_ast = mk_cval1(dtype_to_arg(DT_CHAR), DT_INT);
                    src_sdsc_ast = mk_unop(OP_VAL, src_sdsc_ast, DT_INT);
                    flag_con = 0;
                  } else if (DTY(src_dtype) == TY_ARRAY ||
                             (SDSCG(dest) && DTY(src_dtype) == TY_CHAR &&
                              is_unl_poly(dest))) {
                    if (!SDSCG(src)) {
                      src_sdsc_ast = mk_cval1(0, DT_INT);
                      src_sdsc_ast = mk_unop(OP_VAL, src_sdsc_ast, DT_INT);
                    } else {
                      if (STYPEG(src) == ST_MEMBER) {
                        sdsc_mem = get_member_descriptor(src);
                        if (sdsc_mem <= NOSYM) {
                          sdsc_mem = SDSCG(src);
                        }
                      } else {
                        sdsc_mem = SDSCG(src);
                      }
                      src_sdsc_ast =
                          STYPEG(sdsc_mem) == ST_MEMBER
                              ? check_member(alloc_source, mk_id(sdsc_mem))
                              : mk_id(sdsc_mem);
                    }
                  } else if (A_TYPEG(alloc_source) == A_FUNC) {
                    /* FS#21130: get descriptor for return variable */
                    int func, rtn;
                    func = sym_of_ast(A_LOPG(alloc_source));
                    rtn = FVALG(func);
                    src_sdsc_ast = mk_id(get_type_descr_arg2(gbl.currsub, rtn));
                  } else {
                    /* FS#20859: Pass in "fake" scalar descriptor */
                    int tmp;
                    int tag = mk_cval(dtype_to_arg(src_dtype), astb.bnd.dtype);
                    tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, astb.bnd.dtype,
                                      sem.sc);
                    tmp = mk_id(tmp);
                    add_stmt(mk_assn_stmt(tmp, tag, astb.bnd.dtype));
                    src_sdsc_ast = tmp;
                    flag_con = 0;
                  }
                } else if (!CLASSG(src) && !ALLOCDESCG(src) && SDSCG(src) &&
                           (DTY(src_dtype) == TY_DERIVED ||
                            (DTY(src_dtype) == TY_ARRAY &&
                             DTY(src_dtype + 1) == TY_DERIVED))) {
                  int t, d;
                  if (DTY(src_dtype) == TY_ARRAY) {
                    d = DTY(src_dtype + 1);
                  } else {
                    d = src_dtype;
                  }
                  t = DTY(d + 3);
                  src_sdsc_ast = mk_id(get_static_type_descriptor(t));
                }

                if (SDSCG(dest) && is_unl_poly(src) && is_unl_poly(dest) &&
                    SDSCG(src)) {
                  int dast =
                      (dest_sdsc_ast != 0) ? dest_sdsc_ast : mk_id(SDSCG(dest));
                  int sast = (src_sdsc_ast != 0)
                                 ? src_sdsc_ast
                                 : mk_id(get_type_descr_arg(gbl.currsub, src));

                  gen_init_unl_poly_desc(dast, sast, 0);
                } else if (SDSCG(dest) && DTY(DDTG(src_dtype)) == TY_CHAR &&
                           is_unl_poly(dest)) {

                  /* FS#20580: Set up destination descriptor where
                   * unlimited polymorphic object is getting allocated
                   * with a string.
                   */
                  int val, assn, dty, dast, sast;

                  if (string_length(DDTG(src_dtype))) {
                    val = mk_cval1(string_length(DDTG(src_dtype)), DT_INT);
                  } else {
                    dty = DDTG(src_dtype);
                    val = DTY(dty + 1);
                  }
                  if (val) {
                    dast =
                        check_member(dest_sdsc_ast, get_byte_len(SDSCG(dest)));
                    assn = mk_assn_stmt(dast, val, DT_INT);
                    add_stmt(assn);
                  } else if (SDSCG(src)) {
                    dast =
                        check_member(dest_sdsc_ast, get_byte_len(SDSCG(dest)));
                    sast = check_member(src_sdsc_ast, get_byte_len(SDSCG(src)));
                    assn = mk_assn_stmt(dast, sast, DT_INT);
                    add_stmt(assn);
                  }

                  val = mk_cval1(35, DT_INT);
                  dast = check_member(dest_sdsc_ast, get_desc_tag(SDSCG(dest)));
                  assn = mk_assn_stmt(dast, val, DT_INT);
                  add_stmt(assn);

                  if (DTY(src_dtype) != TY_ARRAY) {
                    val = mk_cval1(0, DT_INT);
                    dast =
                        check_member(dest_sdsc_ast, get_desc_rank(SDSCG(dest)));
                    assn = mk_assn_stmt(dast, val, DT_INT);
                    add_stmt(assn);
                    dast =
                        check_member(dest_sdsc_ast, get_desc_lsize(SDSCG(dest)));
                    assn = mk_assn_stmt(dast, val, DT_INT);
                    add_stmt(assn);

                    dast =
                        check_member(dest_sdsc_ast, get_desc_gsize(SDSCG(dest)));
                    assn = mk_assn_stmt(dast, val, DT_INT);
                    add_stmt(assn);

                    val = mk_cval1(ty_to_lib[TY_CHAR], DT_INT);
                    dast = check_member(dest_sdsc_ast, get_kind(SDSCG(dest)));
                    assn = mk_assn_stmt(dast, val, DT_INT);
                    add_stmt(assn);
                  }
                }

                if ((dest_ast && A_TYPEG(dest_ast) == A_SUBSCR) ||
                    (!dest_ast && A_TYPEG(itemp->ast) == A_SUBSCR)) {
                  /* FS#19293 - for subscripted destination, use
                   * the A_LOPG to avoid lerror ICE
                   */
                  dest_ast = (dest_ast) ? A_LOPG(dest_ast) : A_LOPG(itemp->ast);
                }
                argt = mk_argt(5);
                flag_con = mk_cval1(flag_con, DT_INT);
                flag_con = mk_unop(OP_VAL, flag_con, DT_INT);
                ARGT_ARG(argt, 4) = flag_con;

                ARGT_ARG(argt, 0) = (!dest_ast) ? itemp->ast : dest_ast;
                ARGT_ARG(argt, 1) =
                    (!dest_sdsc_ast)
                        ? mk_id(get_type_descr_arg(gbl.currsub, dest))
                        : dest_sdsc_ast;
                ARGT_ARG(argt, 2) = alloc_source;
                ARGT_ARG(argt, 3) =
                    (!src_sdsc_ast)
                        ? mk_id(get_type_descr_arg(gbl.currsub, src))
                        : src_sdsc_ast;
                ast = mk_id(sym_mkfunc_nodesc(mkRteRtnNm(fidx), DT_NONE));
                ast = mk_func_node(A_CALL, ast, 5, argt);
                add_stmt(ast);
              } else {
                ast = mk_assn_stmt(itemp->ast, alloc_source, dtype);
                add_stmt(ast);
              }
            }
          } else if (CLASSG(dest) || DTY(DDTG(src_dtype)) == TY_DERIVED) {
            DTYPE src_dty = DDTG(src_dtype);
            SPTR arg0, arg1;
            int std;
            int dtype_arg_ast;
            SPTR sdsc;
            bool intrin_type = false;

            /* Special cases for mold= (i.e., polymorphic allocatable or
             * derived type allocatable).
             *
             * For mold=, we just set up shape, type, and type parameters of
             * of expression without copying the value.
             * If destination is polymorphic or a derived typed object,
             * set up its type descriptor. If it's a derived typed object,
             * we also need to initialize values that are default
             * initialized.
             */

            sdsc = STYPEG(dest) == ST_MEMBER ? get_member_descriptor(dest) : 0;
            if (sdsc > NOSYM) {
              arg0 = check_member(itemp->ast, mk_id(sdsc));
            } else {
              arg0 = mk_id(get_type_descr_arg(gbl.currsub, dest));
            }

            sdsc = STYPEG(src) == ST_MEMBER ? get_member_descriptor(src) : 0;
            if (sdsc > NOSYM && DTY(src_dty) == TY_DERIVED) {
              arg1 = check_member(alloc_source, mk_id(sdsc));
            } else if (DTY(src_dty) != TY_DERIVED &&
                       (dtype_arg_ast = dtype_to_arg(src_dty)) > 0) {
              arg1 = mk_unop(OP_VAL, mk_cval1(dtype_arg_ast, DT_INT), DT_INT);
              intrin_type = true;
            } else {
              arg1 = mk_id(get_type_descr_arg(gbl.currsub, src));
            }

            ast = mk_set_type_call(arg0, arg1, intrin_type);
            std = add_stmt(ast);

            if (DTY(src_dty) == TY_DERIVED) {
              /* initialize any default initialized values, type
               * parameters, etc. using the "init_from_desc" runtime routine.
               */
              int func_ast = mk_id(
                  sym_mkfunc_nodesc(mkRteRtnNm(RTE_init_from_desc), DT_NONE));
              int argt = mk_argt(3);

              ast = mk_func_node(A_CALL, func_ast, 3, argt);

              /* 1st arg is destination's address. Must be the base address */
              ARGT_ARG(argt, 0) = A_TYPEG(itemp->ast) == A_SUBSCR
                                      ? A_LOPG(itemp->ast)
                                      : itemp->ast;

              /* 2nd arg is destination's descriptor */
              ARGT_ARG(argt, 1) = arg0;

              /* 3rd arg takes the rank if we have a traditional descriptor,
               * otherwise, pass in 0 */
              ARGT_ARG(argt, 2) =
                  mk_unop(OP_VAL,
                          mk_cval(SDSCG(dest) && !CLASSG(SDSCG(dest))
                                      ? rank_of_sym(dest)
                                      : 0,
                                  DT_INT4),
                          DT_INT4);

              add_stmt_after(ast, std);
            }
          }
        }

        if (!alloc_source && has_type_parameter(A_DTYPEG(A_SRCG(ast)))) {
          /* Handle type parameters */
          gen_type_initialize_for_sym(sym_of_ast(A_SRCG(ast)), -1, 1,
                                      sem.new_param_dt);
        }
        check_and_add_auto_dealloc_from_ast(itemp->ast);
      }
    }
    SST_ASTP(LHS, 0);
    sem.gcvlen = 0;
    break;
  /*
   *	<allocation stmt> ::=
   *           ALLOCATE ( <alloc type> :: <alloc list> <alloc cntl> ) |
   */
  case ALLOCATION_STMT2:
    rhstop = 5;
    typed_alloc = 0;
    dtype = SST_DTYPEG(RHS(3));
    if (dtype <= 0) {
      error(155, 3, gbl.lineno,
            "Undefined type specifier in ALLOCATE statement", CNULL);
    } else {
      if (0 && DTY(dtype) != TY_DERIVED && DTY(dtype) != TY_CHAR) {
        /* TBD - We'll probably want to support intrinsic types
         * when we support unlimited polymorphic entities.
         */
        error(155, 3, gbl.lineno,
              "Unimplemented feature - specifying "
              "a non-extensible type in"
              " an ALLOCATE statement",
              CNULL);
      } else if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
        for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
          int sptr2, dtype2;
          if (itemp->ast == 0) {
            error(155, 3, gbl.lineno, "invalid ALLOCATE statement", CNULL);
            continue;
          }
          sptr2 = memsym_of_ast(itemp->ast);
          dtype2 = DTYPEG(sptr2);
          if (dtype2 == DT_DEFERCHAR || dtype2 == DT_DEFERNCHAR)
            goto shared_alloc_stmt;
          else if (DDTG(dtype2) == DT_DEFERCHAR ||
                   DDTG(dtype2) == DT_DEFERNCHAR)
            goto shared_alloc_stmt;
          if (DTY(dtype2) == TY_ARRAY)
            dtype2 = DTY(dtype2 + 1);
          if (!UNLPOLYG(DTY(DDTG(dtype2) + 3)) &&
              !eq_dtype2(dtype, dtype2, CLASSG(sptr2))) {
            error(155, 3, gbl.lineno,
                  "ALLOCATE Type-Specification is incompatible with "
                  "type of object ",
                  SYMNAME(sptr2));
          } else {
            /* allocate object with dtype as its dynamic type */
            typed_alloc = 1;
            goto shared_alloc_stmt;
          }
        }
      } else {
        for (itemp = SST_BEGG(RHS(5)); itemp != ITEM_END; itemp = itemp->next) {
          int sptr2, dtype2;
          if (itemp->ast == 0) {
            error(155, 3, gbl.lineno, "invalid ALLOCATE statement", CNULL);
            continue;
          }
          sptr2 = memsym_of_ast(itemp->ast);
          dtype2 = DTYPEG(sptr2);
          if (DTY(dtype2) == TY_ARRAY)
            dtype2 = DTY(dtype2 + 1);
          if (!eq_dtype2(dtype2, dtype, CLASSG(sptr2)) &&
              (DTY(DDTG(dtype2)) != TY_DERIVED ||
               !UNLPOLYG(DTY(DDTG(dtype2) + 3)))) { /* FS#17728 */
            error(155, 3, gbl.lineno,
                  "ALLOCATE Type-Specification is incompatible with "
                  "type of object ",
                  SYMNAME(sptr2));
          } else {
            /* allocate object with dtype as its dynamic type */
            typed_alloc = 1;
            goto shared_alloc_stmt;
          }
        }
      }
    }
    dtype = 0;
    goto shared_alloc_stmt;
  /*
   *      <allocation stmt> ::= DEALLOCATE ( <alloc list> <alloc cntl> )
   */
  case ALLOCATION_STMT3:
    if (alloc_error) {
      alloc_error = FALSE;
      SST_ASTP(LHS, 0);
      break;
    }
    check_dealloc_clauses(SST_BEGG(RHS(3)), SST_BEGG(RHS(4)));
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next)
      (void)gen_alloc_dealloc(TK_DEALLOCATE, itemp->ast, SST_BEGG(RHS(4)));
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<alloc type> ::= <type spec>
   */
  case ALLOC_TYPE1:
    if (SST_IDG(RHS(1)) == S_IDENT) {
      dtype = get_derived_type(RHS(1), TRUE);
      SST_DTYPEP(LHS, dtype);
    } else
      SST_DTYPEP(LHS, sem.gdtype);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <alloc list> ::= <alloc list> , <alloc object> |
   */
  case ALLOC_LIST1:
    rhstop = 3;
    goto alloc_list;
  /*
   *      <alloc list> ::= <alloc object>
   */
  case ALLOC_LIST2:
    rhstop = 1;
  alloc_list:
    if (scn.stmtyp == TK_ALLOCATE)
      sem.stats.allocs++;
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = SST_ASTG(RHS(rhstop));
    itemp->t.flitmp = 0;
    if (!alloc_error && SST_IDG(RHS(rhstop)) == S_EXPR &&
        SST_FIRSTG(RHS(rhstop))) {
      /* a deferred shape array is allocated. pass up the
       * first and last STDs of the bounds assignments.
       */
      itemp->t.flitmp = (FLITM *)getitem(0, sizeof(FLITM));
      itemp->t.flitmp->first = SST_FIRSTG(RHS(rhstop));
      itemp->t.flitmp->last = SST_LASTG(RHS(rhstop));
    }
    if (rhstop == 1)
      /* adding first item to list */
      SST_BEGP(LHS, itemp);
    else
      /* adding subsequent items to list */
      SST_ENDG(LHS)->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<alloc object> ::= <alloc obj>
   */
  case ALLOC_OBJECT1:
    /*
     * <alloc obj> may be:
     * S_IDENT:	SST_SYMG is the ST entry of the identifier
     * S_LVALUE:	object has an explicit shape; SST_SYMG is the ST
     *entry
     *		of the identifier; SST_DBEGG is the shape list
     * S_DERIVED:	object is a derived object; SST_SYMG is the
     *component;
     *		SST_MNOFFG is the offset of the mangled component.
     */
    ast = 0;
    o_ast = 0;
    switch (SST_IDG(RHS(1))) {
    case S_IDENT:
      sptr = SST_SYMG(RHS(1));
      if (POINTERG(sptr) || (ALLOCATTRG(sptr) && STYPEG(sptr) == ST_MEMBER) ||
          XBIT(54, 0x1) || DTYPEG(sptr) == DT_DEFERCHAR ||
          DTYPEG(sptr) == DT_DEFERNCHAR)
        switch (STYPEG(sptr)) {
        case ST_UNKNOWN:
        case ST_IDENT:
          STYPEP(sptr, ST_VAR);
          FLANG_FALLTHROUGH;
        case ST_VAR:
          if (SDSCG(sptr) == 0 && !F90POINTERG(sptr)) {
            if (SCG(sptr) == SC_NONE)
              SCP(sptr, SC_BASED);
            get_static_descriptor(sptr);
            get_all_descriptors(sptr);
          }
          break;
        default:
          break;
        }
      goto alloc_ident;
    case S_DERIVED:
      ast = SST_ASTG(RHS(1));
      if (A_TYPEG(ast) == A_MEM) {
        sptr = A_SPTRG(A_MEMG(ast));
      } else {
        sptr = SST_SYMG(RHS(1));
      }
      if (SST_DBEGG(RHS(1)))
        goto alloc_explicit_shape;
    alloc_ident:
      SST_ASTP(LHS, 0);
      if (is_procedure_ptr(sptr)) {
        error(198, 3, gbl.lineno, "Illegal use of", SYMNAME(sptr));
        alloc_error = TRUE;
        break;
      }
      if (scn.stmtyp == TK_ALLOCATE) {
      } else if (SCG(sptr) == SC_NONE) {
        if (DTY(DTYPEG(sptr)) == TY_ARRAY && ALLOCG(sptr))
          SCP(sptr, SC_BASED);
      }
      if (!POINTERG(sptr) && !ALLOCATTRG(sptr)) {
        if (SCG(sptr) == SC_BASED && MIDNUMG(sptr) && !CCSYMG(MIDNUMG(sptr)) &&
            !HCCSYMG(MIDNUMG(sptr)))
          /* cray pointee need not be allocatable/pointer */
          ;
        else if (SCG(sptr) == SC_CMBLK && ALLOCG(sptr))
          /* allocatable common block */
          ;
        else {
          error(198, 3, gbl.lineno, "Illegal use of", SYMNAME(sptr));
          alloc_error = TRUE;
          break;
        }
      }
      if (STYPEG(sptr) == ST_ARRAY && ASUMSZG(sptr)) {
        error(198, 3, gbl.lineno, "Illegal use of assumed-size array",
              SYMNAME(sptr));
        alloc_error = TRUE;
        break;
      }
      /*
       * pass up sptr of based object and the address of the associated
       * pointer variable.
       */
      sptr1 = ref_based_object(sptr);
      mkident(LHS);
      SST_SYMP(LHS, sptr);
      if (ast == 0) {
        ast = mk_id(sptr);
      }
      o_ast = ast;
      SST_ASTP(LHS, ast);
      if (POINTERG(sptr) && chk_pointer_intent(sptr, 0)) {
        alloc_error = TRUE;
      }
      break;
    case S_LVALUE:
      ast = SST_ASTG(RHS(1));
      sptr = SST_SYMG(RHS(1));
      if (!SST_DBEGG(RHS(1)))
        goto alloc_ident;
    alloc_explicit_shape:
      SST_ASTP(LHS, 0);
      if (DTY(DTYPEG(sptr)) != TY_ARRAY) {
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- must be an allocatable array");
        alloc_error = TRUE;
        break;
      }
      if (!ALLOCG(sptr) && !POINTERG(sptr)) {
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- must be an allocatable array");
        alloc_error = TRUE;
        break;
      }
      if (SCG(sptr) == SC_NONE)
        SCP(sptr, SC_BASED);
      if (!POINTERG(sptr) && !ALLOCATTRG(sptr)) {
        if (SCG(sptr) == SC_BASED && MIDNUMG(sptr) && !CCSYMG(MIDNUMG(sptr)) &&
            !HCCSYMG(MIDNUMG(sptr)))
          /* cray pointee need not be allocatable/pointer */
          ;
        else if (SCG(sptr) == SC_CMBLK && ALLOCG(sptr))
          /* allocatable common block */
          ;
        else {
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- must be an allocatable array");
          alloc_error = TRUE;
          break;
        }
      }
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        sem.arrdim.ndim = 0;
        for (itemp = SST_DBEGG(RHS(1)); itemp != ITEM_END;
             itemp = itemp->next) {
          SST *stkp;

          if (sem.arrdim.ndim >= 7) {
            error(47, 3, gbl.lineno, CNULL, CNULL);
            alloc_error = TRUE;
            break;
          }
          stkp = itemp->t.stkp;
          switch (SST_IDG(stkp)) {
          case S_KEYWORD:
            errsev(79);
            alloc_error = TRUE;
            break;
          case S_TRIPLE:
            e1 = SST_E1G(stkp);
            if (scn.stmtyp == TK_ALLOCATE) {
              if (SST_IDG(e1) == S_NULL) {
                error(34, 3, gbl.lineno, ":", CNULL);
                alloc_error = TRUE;
                break;
              }
              sem.bounds[sem.arrdim.ndim].lowtype = S_EXPR;
              sem.bounds[sem.arrdim.ndim].lowb =
                  chk_scalartyp(e1, astb.bnd.dtype, FALSE);
              sem.bounds[sem.arrdim.ndim].lwast = SST_ASTG(e1);
              e1 = SST_E2G(stkp);
              if (SST_IDG(e1) == S_NULL) {
                error(34, 3, gbl.lineno, ":", CNULL);
                alloc_error = TRUE;
                break;
              }
              sem.bounds[sem.arrdim.ndim].uptype = S_EXPR;
              sem.bounds[sem.arrdim.ndim].upb =
                  chk_scalartyp(e1, astb.bnd.dtype, FALSE);
              sem.bounds[sem.arrdim.ndim].upast = SST_ASTG(e1);
            } else {
              if (SST_IDG(e1) == S_NULL) {
                sem.bounds[sem.arrdim.ndim].lowtype = S_CONST;
                sem.bounds[sem.arrdim.ndim].lowb = 1;
                sem.bounds[sem.arrdim.ndim].lwast = astb.bnd.one;
              } else {
                sem.bounds[sem.arrdim.ndim].lowtype = S_EXPR;
                sem.bounds[sem.arrdim.ndim].lowb =
                    chk_scalartyp(e1, astb.bnd.dtype, FALSE);
                sem.bounds[sem.arrdim.ndim].lwast = SST_ASTG(e1);
              }
              e1 = SST_E2G(stkp);
              if (SST_IDG(e1) == S_NULL) {
                sem.bounds[sem.arrdim.ndim].uptype = S_EXPR;
                sem.bounds[sem.arrdim.ndim].upb =
                    ADD_UPBD(DTYPEG(sptr), sem.arrdim.ndim);
                sem.bounds[sem.arrdim.ndim].upast =
                    ADD_UPAST(DTYPEG(sptr), sem.arrdim.ndim);
              } else {
                sem.bounds[sem.arrdim.ndim].uptype = S_EXPR;
                sem.bounds[sem.arrdim.ndim].upb =
                    chk_scalartyp(e1, astb.bnd.dtype, FALSE);
                sem.bounds[sem.arrdim.ndim].upast = SST_ASTG(e1);
              }
            }
            e1 = SST_E3G(stkp);
            if (SST_IDG(e1) != S_NULL) {
              error(34, 3, gbl.lineno, ":", CNULL);
              alloc_error = TRUE;
            }
            break;
          default:
            sem.bounds[sem.arrdim.ndim].lowtype = S_CONST;
            sem.bounds[sem.arrdim.ndim].lowb = 1;
            sem.bounds[sem.arrdim.ndim].lwast = astb.bnd.one;
            sem.bounds[sem.arrdim.ndim].uptype = S_EXPR;
            sem.bounds[sem.arrdim.ndim].upb =
                chk_scalartyp(stkp, astb.bnd.dtype, FALSE);
            sem.bounds[sem.arrdim.ndim].upast = SST_ASTG(stkp);
            break;
          }
          sem.arrdim.ndim++;
        }
        if (alloc_error)
          break;
        dtype = DTYPEG(sptr);
        if (AD_NUMDIM(AD_DPTR(dtype)) != sem.arrdim.ndim) {
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- incorrect number of shape specifiers");
          alloc_error = TRUE;
          break;
        }
      }
      (void)ref_based_object(sptr);
      if (A_TYPEG(ast) == A_MEM && A_SHAPEG(A_PARENTG(ast))) {
        error(198, 3, gbl.lineno, "Array-valued", "derived-type parent");
        alloc_error = TRUE;
        break;
      }
      if (POINTERG(sptr) && chk_pointer_intent(sptr, ast)) {
        alloc_error = TRUE;
        break;
      }
      bef = STD_PREV(0); /* std before any bounds assignments */
      o_ast = ast;
      ast = gen_defer_shape(sptr, ast, 0);
      SST_ASTP(LHS, ast);
      SST_IDP(LHS, S_EXPR);
      SST_SYMP(LHS, sptr);
      SST_FIRSTP(LHS, 0);
      SST_LASTP(LHS, 0);
      if (STD_PREV(0) && bef != STD_PREV(0)) {
        /* Record the first and last STDs of the bounds assignments.
         * The assignments will be moved immediately before the
         * corresponding allocate.
         */
        SST_FIRSTP(LHS, STD_NEXT(bef));
        SST_LASTP(LHS, STD_PREV(0));
      }
      break;
    default:
      alloc_error = TRUE;
      break;
    }
    if (!alloc_error && is_protected(sym_of_ast(o_ast))) {
      err_protected(sym_of_ast(o_ast),
                    "appear in a ALLOCATE or DEALLOCATE statement");
    }
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <alloc obj> ::= <alloc obj> ( <ssa list> ) :: <ident>
   */
  case ALLOC_OBJ5:
    if (SST_IDG(RHS(1)) == S_IDENT) {
      int sptr2 = SST_SYMG(RHS(6));
      int dtype2 = DTYPEG(sptr2);
      if (DTY(dtype2) == TY_ARRAY)
        dtype2 = DTY(dtype2 + 1);
      sptr = SST_SYMG(RHS(1));
      if (STYPEG(sptr) == ST_USERGENERIC && GTYPEG(sptr)) {
        sptr = GTYPEG(sptr);
      }
      if (STYPEG(sptr) == ST_TYPEDEF && DTY(dtype2) == TY_DERIVED) {
        int mem, mem2, mem3;
        int offset = 0;
        int seen_keyword = 0;
        int dtype3 = 0;
        dtype = DTYPEG(sptr);

        for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
          e1 = itemp->t.stkp;
          if (SST_IDG(e1) == S_KEYWORD) {
            seen_keyword = 1;
            np = scn.id.name + SST_CVALG(e1);
            mem2 = get_parm_by_name(np, dtype);
            if (mem2 <= NOSYM) {
              error(155, 3, gbl.lineno,
                    "Too many type parameter specifiers for", SYMNAME(sptr));
              continue;
            }
            offset = KINDG(mem2);
            goto alloc_tp_comm;
          } else if (SST_IDG(e1) == S_CONST || SST_IDG(e1) == S_EXPR ||
                     SST_IDG(e1) == S_IDENT) {
            if (seen_keyword) {
              error(155, 3, gbl.lineno,
                    "A non keyword = type parameter "
                    "specifier cannot follow a keyword = "
                    "type parameter specifier",
                    NULL);
              continue;
            }
            if (SST_IDG(e1) == S_IDENT)
              mkexpr(e1);
            ++offset;
            mem2 = get_parm_by_number(offset, dtype);
            if (mem2 <= NOSYM) {
              error(155, 3, gbl.lineno,
                    "Too many type parameter specifiers for", SYMNAME(sptr));
              continue;
            }
          alloc_tp_comm:
            mem = get_parm_by_name(SYMNAME(mem2), dtype2);

            if (mem <= NOSYM || ASZG(mem)) {
              error(155, 3, gbl.lineno, "Invalid type specifier for",
                    SYMNAME(sptr));
              continue;
            }
            if (DEFERLENG(mem)) {
              if (!dtype3) {
                dtype3 = create_parameterized_dt(dtype2, 1);
              }
              mem3 = get_parm_by_name(SYMNAME(mem), dtype3);
              if (mem3 > NOSYM) {
                ast = (!seen_keyword) ? SST_ASTG(e1) : SST_ASTG(SST_E3G(e1));
                LENP(mem3, ast);
                if (A_TYPEG(ast) == A_CNST) {
                  KINDP(mem3, CONVAL2G(A_SPTRG(ast)));
                }
              }

              for (mem2 = DTY(dtype + 1), mem3 = DTY(dtype3 + 1);
                   mem3 > NOSYM && mem2 > NOSYM;
                   mem3 = SYMLKG(mem3), mem2 = SYMLKG(mem2)) {

                int mem_dtype = DTYPEG(mem2);
                int mem_dtype3 = DTYPEG(mem3);

                while (mem3 > NOSYM &&
                       strcmp(SYMNAME(mem2), SYMNAME(mem3)) != 0) {
                  /* skip over internal descriptors/pointers */
                  mem3 = SYMLKG(mem3);
                }
                if (mem3 <= NOSYM) {
                  interr("semant3: unexpected component in allocation of"
                         " derived type with deferred type parameter",
                         0, 3);
                  break;
                }

                if (DTY(mem_dtype) == TY_ARRAY) {
                  /* TBD: Can we always dup the array type, or should
                   * we check mem_dtype for use of deferred type
                   * parameter
                   * (mem) first????
                   */
                  DTYPEP(mem3, dup_array_dtype(mem_dtype));
                }

                if (DTY(mem_dtype) == TY_CHAR || DTY(mem_dtype) == TY_NCHAR) {
                  ast = DTY(mem_dtype + 1);
                  DTY(mem_dtype3 + 1) = ast;
                }
              }
              continue;
            }

            if (!seen_keyword && KINDG(mem) >= 0 &&
                (SST_IDG(e1) != S_CONST || SST_CVALG(e1) != KINDG(mem))) {
              error(155, 3, gbl.lineno, "Invalid type specifier for",
                    SYMNAME(sptr));
              continue;
            } else if (seen_keyword) {
              ast = SST_ASTG(SST_E3G(e1));
              if (KINDG(mem) >= 0 && (A_TYPEG(ast) != A_CNST ||
                                      SST_CVALG(SST_E3G(e1)) != KINDG(mem))) {
                error(155, 3, gbl.lineno, "Invalid type specifier for",
                      SYMNAME(sptr));
                continue;
              }
            }

            if (LENG(mem) && cmp_len_parms(LENG(mem), SST_ASTG(e1)) == 0) {
              error(155, 3, gbl.lineno, "Invalid type specifier for",
                    SYMNAME(sptr));
              continue;
            }
          }
        }

        if (dtype3) {
          mkexpr(RHS(6));
          sem.new_param_dt = dtype3;
          put_length_type_param(dtype3, 0);
        }
      }
    }
    rhstop = 6;
    goto alloc_comm;

  /*
   *	<alloc obj> ::= <ident> |
   */
  case ALLOC_OBJ1:
    rhstop = 1;
  alloc_comm:
    sptr = refsym((int)SST_SYMG(RHS(rhstop)), OC_OTHER);
    if (STYPEG(sptr) == ST_ENTRY) {
      if (gbl.rutype != RU_FUNC)
        error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      sptr = ref_entry(sptr);
    }
    if (sem.parallel || sem.task || sem.target || sem.teams || sem.orph)
      sptr = sem_check_scope(sptr, sptr);
    SST_SYMP(LHS, sptr);
    if (SCG(sptr) == SC_NONE)
      sem_set_storage_class(sptr);
    SST_SHAPEP(LHS, 0);
    SST_MNOFFP(LHS, 0);
    break;
  /*
   *	<alloc obj> ::= <alloc obj> ( <ssa list> ) |
   */
  case ALLOC_OBJ2:
    itemp = SST_BEGG(RHS(3));
    switch (SST_IDG(RHS(1))) {
    case S_IDENT:
      SST_IDP(LHS, S_LVALUE);
      /* problem, since SST_DBEG overlaps with SST_LSYM */
      /* SST_SYMP(LHS, SST_SYMG(RHS(1))); -- already in position */
      ast = mk_id(SST_SYMG(RHS(1)));
      SST_ASTP(LHS, ast);
      SST_DBEGP(LHS, itemp);
      SST_DENDP(LHS, itemp);
      break;
    case S_DERIVED:
      SST_DBEGP(LHS, itemp);
      SST_DENDP(LHS, itemp);
      break;
    case S_LVALUE:
      SST_DBEGP(LHS, itemp);
      SST_DENDP(LHS, itemp);
      FLANG_FALLTHROUGH;
    default:
      break;
    }
    break;
  /*
   *	<alloc obj> ::= <alloc obj> % <id>
   */
  case ALLOC_OBJ3:
    rhstop = 3;
    goto alloc_component_shared;
  /*
   *	<alloc obj> ::= <alloc obj> %LOC
   */
  case ALLOC_OBJ4:
    rhstop = 2;
    SST_SYMP(RHS(2), getsymbol("loc"));
  alloc_component_shared:
    ast = 0;
    switch (SST_IDG(RHS(1))) {
    case S_IDENT:
      (void)mkexpr(RHS(1));
      sptr = SST_SYMG(RHS(1));
      break;
    case S_DERIVED:
      sptr = SST_SYMG(RHS(1));
      if (SST_DBEGG(RHS(1))) {
        itemp = SST_DBEGG(RHS(1));
        SST_IDP(RHS(1), S_IDENT);
        (void)mkvarref(RHS(1), itemp);
      }
      ast = SST_ASTG(RHS(1));
      break;
    case S_LVALUE:
      sptr = SST_SYMG(RHS(1));
      ast = SST_ASTG(RHS(1));
      if (SST_DBEGG(RHS(1))) {
        itemp = SST_DBEGG(RHS(1));
        SST_LSYMP(RHS(1), sptr);
        SST_DTYPEP(RHS(1), DTYPEG(sptr));
        (void)mkvarref(RHS(1), itemp);
        ast = SST_ASTG(RHS(1));
      }
      break;
    default:
      interr("alloc obj %%", SST_IDG(RHS(1)), 3);
      sptr = SST_SYMG(RHS(1));
      break;
    }
    {
      dtype = DDTG(DTYPEG(sptr));
      if (ast == 0)
        ast = mk_id(sptr);
    }
    if (DTY(dtype) != TY_DERIVED) {
      error(141, 3, gbl.lineno, "Derived Type object", "%");
      SST_IDP(LHS, S_NULL);
      break;
    }
    i = NMPTRG(SST_SYMG(RHS(rhstop)));
    ast = mkmember(dtype, ast, i);
    if (ast) {
      sptr1 = A_SPTRG(A_MEMG(ast));
      dtype = DTYPEG(sptr1);
      if (PRIVATEG(sptr1) && test_private_dtype(ENCLDTYPEG(sptr1))) {
        error(155, 3, gbl.lineno,
              "Attempt to use private component:", SYMNAME(sptr1));
      }
      SST_MNOFFP(LHS, 0);
      SST_IDP(LHS, S_LVALUE);
      SST_SYMP(LHS, sptr1); /* must use SYMP, not LSYMP */
      SST_SHAPEP(LHS, A_SHAPEG(ast));
      SST_DTYPEP(LHS, dtype);
      SST_DBEGP(LHS, 0);
    } else {
      /* <id> is not a member of this record */
      sptr1 = SST_SYMG(RHS(rhstop));
      error(142, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
      SST_IDP(LHS, S_NULL);
    }
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <alloc cntl> ::=    |
   */
  case ALLOC_CNTL1:
    SST_BEGP(LHS, 0);
    break;
  /*
   *	<alloc cntl> ::= <alloc cntl list>
   */
  case ALLOC_CNTL2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<alloc cntl list> ::= <alloc cntl list> <alloc cntl item> |
   */
  case ALLOC_CNTL_LIST1:
    SST_ENDG(RHS(1))->next = SST_BEGG(RHS(2));
    SST_ENDP(LHS, SST_ENDG(RHS(2)));
    break;
  /*
   *	<alloc cntl list> ::= <alloc cntl item>
   */
  case ALLOC_CNTL_LIST2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<alloc cntl item> ::=  , PINNED = <var ref> |
   */
  case ALLOC_CNTL_ITEM1:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.conval = TK_PINNED;
    itemp->ast = 0;
    SST_ENDP(LHS, itemp);
    SST_BEGP(LHS, itemp);
    if (cuda_enabled("pinned")) {
    }
    alloc_error = TRUE;
    break;
  /*
   *	<alloc cntl item> ::= , STAT = <var ref> ,
   */
  case ALLOC_CNTL_ITEM2:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.conval = TK_STAT;
    itemp->ast = 0;
    SST_ENDP(LHS, itemp);
    SST_BEGP(LHS, itemp);
    if (is_varref(RHS(4))) {
      mkarg(RHS(4), &dum);
      itemp->ast = SST_ASTG(RHS(4));
      if (DT_ISINT(dum)) {
        sptr = sym_of_ast(itemp->ast);
        ADDRTKNP(sptr, 1);
        break;
      }
    }
    error(198, 3, gbl.lineno,
          "STAT specifier must be a scalar integer variable", CNULL);
    alloc_error = TRUE;
    break;
  /*
   *	<alloc cntl item> ::= , ERRMSG  = <var ref> |
   */
  case ALLOC_CNTL_ITEM3:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.conval = TK_ERRMSG;
    itemp->ast = 0;
    SST_ENDP(LHS, itemp);
    SST_BEGP(LHS, itemp);
    if (is_varref(RHS(4))) {
      mkarg(RHS(4), &dum);
      itemp->ast = SST_ASTG(RHS(4));
      if (DTY(dum) == TY_CHAR) {
        sptr = sym_of_ast(itemp->ast);
        ADDRTKNP(sptr, 1);
        break;
      }
    }
    error(198, 3, gbl.lineno,
          "ERRMSG specifier must be a scalar character string variable", CNULL);
    alloc_error = TRUE;
    break;
  /*
   *	<alloc cntl item> ::= , SOURCE = <expression> |
   */
  case ALLOC_CNTL_ITEM4:
    if (alloc_error)
      break;
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.conval = TK_SOURCE;
    itemp->ast = 0;
    SST_ENDP(LHS, itemp);
    SST_BEGP(LHS, itemp);
    mkarg(RHS(4), &dum);
    itemp->ast = SST_ASTG(RHS(4));

    /* Store len of character in case the alloctable object is DEFERCHAR
     */
    if (DDTG(A_DTYPEG(itemp->ast) == DT_CHAR) ||
        DDTG(A_DTYPEG(itemp->ast) == DT_NCHAR))
      sem.gcvlen = size_ast_of(itemp->ast, A_DTYPEG(itemp->ast));
    break;
  /*
   *	<alloc cntl item> ::= , MOLD = <expression> |
   */
  case ALLOC_CNTL_ITEM5:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.conval = TK_MOLD;
    itemp->ast = 0;
    SST_ENDP(LHS, itemp);
    SST_BEGP(LHS, itemp);
    mkarg(RHS(4), &dum);
    itemp->ast = SST_ASTG(RHS(4));

    /* Store len of character in case the alloctable object is DEFERCHAR
     */
    if (DDTG(A_DTYPEG(itemp->ast) == DT_CHAR) ||
        DDTG(A_DTYPEG(itemp->ast) == DT_NCHAR))
      sem.gcvlen = size_ast_of(itemp->ast, A_DTYPEG(itemp->ast));
    break;
  /*
   *	<alloc cntl item> ::= , ALIGN = <expression>
   */
  case ALLOC_CNTL_ITEM6:
    (void)chktyp(RHS(4), DT_INT8, TRUE);
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.conval = TK_ALIGN;
    itemp->ast = SST_ASTG(RHS(4));
    SST_ENDP(LHS, itemp);
    SST_BEGP(LHS, itemp);
    if (flg.standard)
      error(170, 2, gbl.lineno, "ALIGN= specifier in an ALLOCATE", CNULL);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *      <forall clause> ::= <forall begin> <concurrent header>
   */
  case FORALL_CLAUSE1:
    *LHS = *RHS(2);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<forall begin> ::= <forall construct>
   */
  case FORALL_BEGIN1:
    sem.pgphase = PHASE_EXEC; /* set now, since may have forall(...) stmt */
    NEED_DOIF(doif, DI_FORALL);
    DI_NAME(doif) = construct_name;
    DI_FORALL_SYMAVL(doif) = stb.stg_avail;
    DI_FORALL_LASTSTD(sem.doif_depth) = STD_PREV(0);
    last_std = sem.last_std;
    direct_loop_enter();
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<forall construct> ::= FORALL |
   */
  case FORALL_CONSTRUCT1:
    construct_name = 0;
    break;
  /*
   *	<forall construct> ::= <check construct> : FORALL
   */
  case FORALL_CONSTRUCT2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<null stmt> ::=
   */
  case NULL_STMT1:
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<exit stmt> ::= EXIT <construct name>
   */
  case EXIT_STMT1:
    if (not_in_forall("EXIT"))
      break;
    goto exit_cycle_shared;

  /* ------------------------------------------------------------------
   */
  /*
   *	<construct name> ::=  |
   */
  case CONSTRUCT_NAME1:
    construct_name = 0;
    break;
  /*
   *	<construct name> ::= <id>
   */
  case CONSTRUCT_NAME2:
    construct_name = NMPTRG(SST_SYMG(RHS(1)));
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<cycle stmt> ::= CYCLE <construct name>
   */
  case CYCLE_STMT1:
    if (not_in_forall("CYCLE"))
      break;
    if (sem.collapse && DI_ID(sem.doif_depth) == DI_DO &&
        DI_DOINFO(sem.doif_depth)) {
      if (construct_name && construct_name != DI_NAME(sem.doif_depth)) {
        error(155, 3, gbl.lineno,
              "CYCLE may only apply to the innermost associated loop", CNULL);
      } else {
        i = sem.collapse;
        for (j = sem.doif_depth; j > 0; j--) {
          if (DI_ID(j) != DI_DO) {
            if (DI_ID(j) != DI_PARDO && DI_ID(j) != DI_PDO &&
                DI_ID(j) != DI_SIMD && DI_ID(j) != DI_DISTPARDO &&
                DI_ID(j) != DI_TEAMSDISTPARDO && DI_ID(j) != DI_TEAMSDIST &&
                DI_ID(j) != DI_TARGTEAMSDIST && DI_ID(j) != DI_DISTRIBUTE &&
                DI_ID(j) != DI_TARGTEAMSDISTPARDO && DI_ID(j) != DI_TASKLOOP)
              i = 0;
            break;
          }
          i--;
          if (!i)
            break;
        }
        if (i)
          error(155, 3, gbl.lineno,
                "CYCLE may only apply to the innermost associated loop", CNULL);
      }
    }
  exit_cycle_shared:
    check_do_term();
    ast = 0;
    for (doif = sem.doif_depth; doif > 0; --doif) {
      int label;
      if (DI_ID(doif) == DI_DOCONCURRENT && scn.stmtyp == TK_EXIT) {
        error(1050, ERR_Severe, gbl.lineno, "EXIT from", CNULL); // 2018-C1166
        break;
      }
      if (construct_name && construct_name != DI_NAME(doif))
        continue;
      if (DI_ID(doif) != DI_DO && DI_ID(doif) != DI_DOWHILE &&
          DI_ID(doif) != DI_DOCONCURRENT &&
          (!construct_name || scn.stmtyp == TK_CYCLE))
        continue;
      if (scn.stmtyp == TK_EXIT) {
        if (DI_EXIT_LABEL(doif) == 0)
          DI_EXIT_LABEL(doif) = getlab();
        label = DI_EXIT_LABEL(doif);
      } else { /* CYCLE statement */
        if (DI_CYCLE_LABEL(doif) == 0)
          DI_CYCLE_LABEL(doif) = getlab();
        label = DI_CYCLE_LABEL(doif);
      }
      ast = mk_stmt(A_GOTO, 0);
      astlab = mk_label(label);
      A_L1P(ast, astlab);
      RFCNTI(label);
      break;
    }
    if (doif <= 0) {
      if (construct_name)
        error(304, 3, gbl.lineno, " named ", stb.n_base + construct_name);
      else
        error(304, 3, gbl.lineno, CNULL, CNULL);
    }
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<case begin> ::= SELECTCASE <etmp lp> <expression> ) |
   */
  case CASE_BEGIN1:
    if (not_in_forall("SELECT CASE"))
      break;
    construct_name = 0;
    rhstop = 3;
    *LHS = *RHS(3);
    break;
  /*
   *	<case begin> ::= <check construct> : SELECTCASE <etmp lp>
   *<expression> )
   */
  case CASE_BEGIN2:
    if (not_in_forall("SELECT CASE"))
      break;
    *LHS = *RHS(5);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<case> ::= CASE
   */
  case CASE1:
    sem.stats.nodes++;
    doif = sem.doif_depth;
    if (!doif || DI_ID(doif) != DI_CASE) {
      sem_err105(doif);
      doif = 0;
    }
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<case value list> ::= <case value list> , <case value> |
   */
  case CASE_VALUE_LIST1:
    ast = SST_ASTG(RHS(1));
    if (ast) {
      ast2 = SST_ASTG(RHS(3));
      if (ast2) {
        ast = mk_binop(OP_LOR, ast, ast2, DT_LOG);
        SST_ASTP(LHS, ast);
      } else
        SST_ASTP(LHS, 0);
    }
    break;
  /*
   *	<case value list> ::= <case value>
   */
  case CASE_VALUE_LIST2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<case value> ::= <case expr>   |
   */
  case CASE_VALUE1:
    if (SST_ASTG(LHS) == 0)
      break;
    doif = sem.doif_depth;
    ast = DI_CASE_EXPR(doif);
    if (DT_ISLOG(A_DTYPEG(ast)))
      opc = OP_LEQV;
    else
      opc = OP_EQ;
    ast1 = SST_ASTG(RHS(1));
    ast = mk_binop(opc, ast, ast1, DT_LOG);
    SST_ASTP(LHS, ast);
    add_case_range(doif, A_SPTRG(A_ALIASG(ast1)), 0);
    break;
  /*
   *	<case value> ::= <case expr> : |
   */
  case CASE_VALUE2:
    if (SST_ASTG(LHS) == 0)
      break;
    doif = sem.doif_depth;
    ast = DI_CASE_EXPR(doif);
    if (DT_ISLOG(DI_DTYPE(doif))) {
      error(310, 3, gbl.lineno, "Range of case values not allowed for logical",
            CNULL);
      SST_ASTP(LHS, 0);
      break;
    }
    ast1 = SST_ASTG(RHS(1));
    ast = mk_binop(OP_GE, ast, ast1, DT_LOG);
    SST_ASTP(LHS, ast);
    add_case_range(doif, A_SPTRG(A_ALIASG(ast1)), 1);
    break;
  /*
   *	<case value> ::= : <case expr> |
   */
  case CASE_VALUE3:
    if (SST_ASTG(RHS(2)) == 0)
      break;
    doif = sem.doif_depth;
    ast = DI_CASE_EXPR(doif);
    if (DT_ISLOG(DI_DTYPE(doif)))
      error(310, 3, gbl.lineno, "Range of case values not allowed for logical",
            CNULL);
    ast1 = SST_ASTG(RHS(2));
    ast = mk_binop(OP_LE, ast, ast1, DT_LOG);
    SST_ASTP(LHS, ast);
    add_case_range(doif, A_SPTRG(A_ALIASG(ast1)), -1);
    break;
  /*
   *	<case value> ::= <case expr> : <case expr>
   */
  case CASE_VALUE4:
    if (SST_ASTG(LHS) == 0 || SST_ASTG(RHS(3)) == 0)
      break;
    doif = sem.doif_depth;
    ast = DI_CASE_EXPR(doif);
    if (DT_ISLOG(DI_DTYPE(doif))) {
      error(310, 3, gbl.lineno, "Range of case values not allowed for logical",
            CNULL);
      SST_ASTP(LHS, 0);
      break;
    }
    if (DI_DTYPE(doif) == DT_INT8)
      p_cmp = _i8_cmp;
    else if (DT_ISINT(DI_DTYPE(doif)))
      p_cmp = _i4_cmp;
    else if (DTY(DI_DTYPE(doif)) == TY_CHAR)
      p_cmp = _char_cmp;
    else
      p_cmp = _nchar_cmp;
    ast1 = SST_ASTG(RHS(1));
    ast2 = SST_ASTG(RHS(3));
    sptr1 = A_SPTRG(A_ALIASG(ast1));
    sptr2 = A_SPTRG(A_ALIASG(ast2));
    if ((*p_cmp)(sptr1, sptr2) > 0) {
      error(310, 2, gbl.lineno, "Empty case value range", CNULL);
      sptr2 = 0; /* don't record range in SWEL list */
    }
    i = mk_binop(OP_GE, ast, ast1, DT_LOG);
    ast = mk_binop(OP_LE, ast, ast2, DT_LOG);
    ast = mk_binop(OP_LAND, i, ast, DT_LOG);
    SST_ASTP(LHS, ast);
    if (sptr2)
      add_case_range(doif, sptr1, sptr2);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<case expr> ::= <expression>
   */
  case CASE_EXPR1:
    if (sem.doif_depth <= 0) {
      SST_ASTP(LHS, 0);
      break;
    }
    doif = sem.doif_depth;
    if (DI_ID(doif) != DI_CASE || DI_DTYPE(doif) == 0) {
      SST_ASTP(LHS, 0);
      break;
    }
    mkexpr(RHS(1));
    ast = SST_ASTG(RHS(1));
    if (A_ALIASG(ast) == 0) {
      error(310, 3, gbl.lineno, "CASE value must be a constant", CNULL);
      SST_ASTP(LHS, 0);
      break;
    }
    if (!cmpat_dtype(DI_DTYPE(doif), A_DTYPEG(ast))) {
      error(310, 3, gbl.lineno, "The type of the CASE value does not match",
            "the type of the SELECTCASE expression");
      SST_ASTP(LHS, 0);
      break;
    }
    /* don't bother converting constant if character */
    if (DTY(DI_DTYPE(doif)) != TY_CHAR && DTY(DI_DTYPE(doif)) != TY_NCHAR)
      (void)chktyp(RHS(1), DI_DTYPE(doif), TRUE);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<pointer assignment> ::= <psfunc> <var ref> <psfunc> '=>'
   *<expression>
   */
  case POINTER_ASSIGNMENT1:
    ast = assign_pointer(RHS(2), RHS(5));
    SST_ASTP(LHS, ast);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<nullify stmt> ::= NULLIFY ( <nullify list> )
   */
  case NULLIFY_STMT1:
    if (not_in_forall("NULLIFY"))
      break;
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<nullify list> ::= <nullify list> , <nullify object> |
   */
  case NULLIFY_LIST1:
    break;
  /*
   *	<nullify list> ::= <nullify object>
   */
  case NULLIFY_LIST2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<nullify object> ::= <var ref>
   */
  case NULLIFY_OBJECT1:
    if (is_varref(RHS(1))) {
      mkarg(RHS(1), &dum);
      ast = SST_ASTG(RHS(1));
      sptr = find_pointer_variable(ast);
      if (sptr == 0) {
        error(155, 3, gbl.lineno, "Illegal NULLIFY statement -",
              "non-POINTER object");
        break;
      }
      if (!POINTERG(sptr)) {
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- must have the POINTER attribute");
        break;
      }
      if (chk_pointer_intent(sptr, ast))
        break;
      sptr1 = intast_sym[I_NULLIFY];
      {
        (void)add_stmt(add_nullify_ast(ast));
      }
      if (is_protected(sym_of_ast(ast))) {
        err_protected(sym_of_ast(ast), "appear in a NULLIFY statement");
      }
      break;
    }
    error(155, 3, gbl.lineno, "Illegal NULLIFY statement -",
          "non-POINTER object");
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<forall assn stmt> ::= <assignment> |
   */
  case FORALL_ASSN_STMT1:
    break;
  /*
   *	<forall assn stmt> ::= <pointer assignment>
   */
  case FORALL_ASSN_STMT2:
    break;

  /* ------------------------------------------------------------------
   */
  /*
   *	<pragma stmt> ::= PREFETCH <var ref list> |
   */
  case PRAGMA_STMT1:
    for (itemp = SST_BEGG(RHS(2)); itemp != ITEM_END; itemp = itemp->next) {
      mklvalue(itemp->t.stkp, 3);
      ast = mk_stmt(A_PREFETCH, 0);
      A_LOPP(ast, SST_ASTG(itemp->t.stkp));
      A_OPTYPEP(ast, 0);
      (void)add_stmt(ast);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<pragma stmt> ::= DISTRIBUTEPOINT |
   */
  case PRAGMA_STMT2:
    ast = 0;
    doif = sem.doif_depth;
    if (doif > 0 && DI_ID(doif) == DI_DO) {
      ast = mk_stmt(A_ENDDO, 0);
      (void)add_stmt(ast);
      ast = mk_stmt(A_TYPEG(DI_DO_AST(doif)), 0);
      BCOPY(astb.stg_base + ast, astb.stg_base + DI_DO_AST(doif), AST, 1);
      DI_DO_AST(doif) = ast;
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<pragma stmt> ::= DISTRIBUTE
   */
  case PRAGMA_STMT3:
    /* this only exists to create the token, TK_DISTRIBUTE, so that
     * scan() can find DISTRIBUTE POINT separated by white space
     */
    break;

  /* ------------------------------------------------------------------
   */
  default:
    interr("semant3:bad rednum", rednum, 3);
  }
}

void
set_construct_name(int name)
{
  construct_name = name;
}

int
get_construct_name(void)
{
  return construct_name;
}

#ifdef FLANG_SEMANT3_UNUSED
static void
add_nullify(int sptr)
{
  int ast = add_nullify_ast(mk_id(sptr));
  (void)add_stmt(ast);
}
#endif

/* error checking: this is called at a statement that cannot
 * terminate a DO statement. */
static void
check_do_term()
{
  int doif = sem.doif_depth;
  if (doif) {
    if (DI_ID(doif) == DI_DO || DI_ID(doif) == DI_DOWHILE ||
        DI_ID(doif) == DI_DOCONCURRENT) {
      if (DI_DO_LABEL(doif) == scn.currlab && scn.currlab != 0) {
        /* make sure this is not conditional */
        int std, ast = 0;
        std = STD_PREV(0);
        if (std)
          ast = STD_AST(std);
        if (ast && A_TYPEG(ast) != A_IFTHEN) {
          /* we have an error situation. */
          if (flg.standard) {
            errsev(308);
          } else {
            errwarn(308);
          }
        }
      }
    }
  }
}

enum { DC_HEADER = 1, DC_MASK, DC_BODY }; // DO CONCURRENT loop components

/** \brief       Check a DO CONCURRENT loop ast for constraint violations.
 *  \param ast   ast to examine
 *  \param doif  address of DO CONCURRENT doif stack slot index
 */
static LOGICAL
check_doconcurrent_ast(int ast, int *doif)
{
  SPTR sptr = SPTR_NULL;
  int argt, astli, i, symi;
  char *name;
  const char *s;
  static const char *finalize_name = NULL, *dev_finalize_name = NULL;
  static const char *ieee_name[] = {"ieee_exceptions", "ieee_get_flag",
                                    "ieee_get_halting_mode",
                                    "ieee_set_halting_mode"};

  switch (A_TYPEG(ast)) {
  case A_ID:
    sptr = sym_of_ast(ast);
    if (sptr <= NOSYM || CCSYMG(sptr) || HCCSYMG(sptr))
      break;
    switch (DI_CONC_KIND(*doif)) {
    case DC_HEADER:
      // Check for an index or LOCAL var reference in a limit/step expression.
      if (sptr >= DI_CONC_SYMAVL(*doif) &&
          !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
        error(1043, ERR_Severe, gbl.lineno, // 2018-C1123 2018-C1129
              "limit or step expression", SYMNAME(sptr));
        DI_CONC_ERROR_SYMS(*doif) =
            add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
      }
      break;
    case DC_MASK:
      // Check for a LOCAL var reference in a mask expression.
      if (sptr < DI_CONC_SYMAVL(*doif))
        break;
      // Skip past index vars.
      for (i = DI_CONC_COUNT(*doif), symi = DI_CONC_SYMS(*doif); i; --i)
        symi = SYMI_NEXT(symi);
      if (sym_in_sym_list(sptr, symi)) { // 2018-C1129
        error(1043, ERR_Severe, gbl.lineno, "mask expression", SYMNAME(sptr));
        DI_CONC_ERROR_SYMS(*doif) =
            add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
      }
      break;
    case DC_BODY:
      // Check for an impure call.
      if (CONSTRUCTSYMG(sptr) && sptr >= DI_CONC_SYMAVL(*doif) &&
          has_impure_finalizer(sptr) &&
          !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
        error(488, ERR_Severe, gbl.lineno, // 2018-C1139
              "Final subroutine for DO CONCURRENT reference", SYMNAME(sptr));
        DI_CONC_ERROR_SYMS(*doif) =
            add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
      }
      // Check for reference to a var that is not in any locality spec list.
      if (!DI_CONC_NO_DEFAULT(*doif))
        break;
      if (sptr >= DI_CONC_SYMAVL(*doif))
        break;
      if (STYPEG(sptr) != ST_IDENT && STYPEG(sptr) != ST_VAR &&
          STYPEG(sptr) != ST_ARRAY)
        break;
      if (!sym_in_sym_list(sptr, DI_CONC_SYMS(*doif)) &&
          !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
        error(1049, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL); // 2018-C1130
        DI_CONC_ERROR_SYMS(*doif) =
            add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
      }
    }
    break;

  case A_CALL:
    if (DI_CONC_KIND(*doif) == DC_HEADER)
      break;
    sptr = memsym_of_ast(A_LOPG(ast));
    name = SYMNAME(sptr);
    // Check for a prohibited ieee_exceptions routine call.  2018-C1141
    if (strncmp(SYMNAME(ENCLFUNCG(sptr)), ieee_name[0], 15) == 0)
      for (i = 1; i < sizeof(ieee_name) / sizeof(char *); ++i)
        if (strncmp(name, ieee_name[i], strlen(ieee_name[i])) == 0) {
          error(1052, ERR_Severe, gbl.lineno, ieee_name[i], CNULL);
          return false;
        }
    // Check for an impure call.
    if (VTABLEG(sptr)) {
      sptr = VTABLEG(sptr);
      name = SYMNAME(sptr);
    }
    s = NULL;
    if (!finalize_name) {
      finalize_name = mkRteRtnNm(RTE_finalize);
    }
    if (strcmp(name, finalize_name) == 0 ||
        (dev_finalize_name && strcmp(name, dev_finalize_name) == 0)) {
      sptr = memsym_of_ast(ARGT_ARG(A_ARGSG(ast), 0));
      name = SYMNAME(sptr);
      if (DI_CONC_KIND(*doif) == DC_MASK)
        s = "Final subroutine for DO CONCURRENT mask reference"; // 2018-C1121
      else
        s = "Final subroutine for DO CONCURRENT reference"; // 2018-C1139
    } else if (!CCSYMG(sptr) && !HCCSYMG(sptr) && is_impure(sptr)) {
      if (DI_CONC_KIND(*doif) == DC_MASK)
        s = "Subroutine call in DO CONCURRENT mask expression"; // 2018-C1121
      else
        s = "Subroutine call in DO CONCURRENT construct"; // 2018-C1139
    }
    if (s) {
      error(488, ERR_Severe, gbl.lineno, s, name);
      break;
    }
    // Check for an alternate return branch out of the loop.
    argt = A_ARGSG(ast);
    for (i = 0; i < A_ARGCNTG(ast); ++i) {
      if (A_TYPEG(ARGT_ARG(argt, i)) == A_LABEL) {
        sptr = A_SPTRG(ARGT_ARG(argt, i));
        if (!sym_in_sym_list(sptr, DI_CONC_LABEL_SYMS(*doif)) &&
            !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
          error(1050, ERR_Severe, gbl.lineno, // 2018-C1138
                "Alternate return branch out of", CNULL);
          DI_CONC_ERROR_SYMS(*doif) =
              add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
        }
      }
    }
    break;

  case A_ICALL:
    // Check for deallocation of a polymorphic arg to move_alloc.
    sptr = memsym_of_ast(A_LOPG(ast));
    if (PDNUMG(sptr) == PD_move_alloc) {
      sptr = memsym_of_ast(ARGT_ARG(A_ARGSG(ast), 1)); // TO dest arg
      if (ALLOCATTRG(sptr) && CLASSG(sptr))
        error(1051, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL); // 2018-C1140
      break;
    }
    // Check for an impure call.
    name = SYMNAME(sptr);
    if (INKINDG(sptr) == IK_SUBROUTINE && *name != '.') {
      error(488, ERR_Severe, gbl.lineno, // 2018-C1139
            "Intrinsic subroutine call in DO CONCURRENT", name);
      break;
    }
    FLANG_FALLTHROUGH;
  case A_FUNC:
  case A_INTR:
    if (DI_CONC_KIND(*doif) == DC_HEADER)
      break;
    if (!sptr)
      sptr = memsym_of_ast(A_LOPG(ast));
    if (VTABLEG(sptr))
      sptr = VTABLEG(sptr);
    name = SYMNAME(sptr);
    if (CCSYMG(sptr) || HCCSYMG(sptr) || STYPEG(sptr) == ST_PD ||
        STYPEG(sptr) == ST_GENERIC)
      break;
    // Check for an impure call.
    if (is_impure(sptr)) {
      if (DI_CONC_KIND(*doif) == DC_MASK)
        error(488, ERR_Severe, gbl.lineno, // 2018-C1121
              "Subprogram call in DO CONCURRENT mask expression", name);
      else
        error(488, ERR_Severe, gbl.lineno, // 2018-C1139
              "Subprogram call in DO CONCURRENT", name);
      break;
    }
    // Check for deallocation of a polymorphic entity.
    if (FUNCG(sptr) && ALLOCATTRG(FVALG(sptr)) && CLASSG(FVALG(sptr))) {
      error(1051, ERR_Severe, gbl.lineno, name, CNULL); // 2018-C1140
      break;
    }
    break;

  case A_AIF:
  case A_GOTO:
    // Check for loop branch exits.
    sptr = A_SPTRG(A_L1G(ast));
    if (!sym_in_sym_list(sptr, DI_CONC_LABEL_SYMS(*doif)) &&
        !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
      error(1050, ERR_Severe, gbl.lineno, "Branch out of", CNULL); // 2018-C1138
      DI_CONC_ERROR_SYMS(*doif) = add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
      break;
    }
    sptr = A_SPTRG(A_L2G(ast));
    if (sptr && !sym_in_sym_list(sptr, DI_CONC_LABEL_SYMS(*doif)) &&
        !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
      error(1050, ERR_Severe, gbl.lineno, "Branch out of", CNULL); // 2018-C1138
      DI_CONC_ERROR_SYMS(*doif) = add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
      break;
    }
    sptr = A_SPTRG(A_L3G(ast));
    if (sptr && !sym_in_sym_list(sptr, DI_CONC_LABEL_SYMS(*doif)) &&
        !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
      error(1050, ERR_Severe, gbl.lineno, "Branch out of", CNULL); // 2018-C1138
      DI_CONC_ERROR_SYMS(*doif) = add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
    }
    break;

  case A_AGOTO:
    // Check for loop branch exits.
    if (!DI_CONC_LABEL_SYMS(*doif)) {
      // If there are no loop labels, any branch must be a loop exit.
      error(1050, ERR_Severe, gbl.lineno, "Branch out of", CNULL); // 2018-C1138
      break;
    }
    FLANG_FALLTHROUGH;
  case A_CGOTO:
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      sptr = A_SPTRG(ASTLI_AST(astli));
      if (sptr && !sym_in_sym_list(sptr, DI_CONC_LABEL_SYMS(*doif)) &&
          !sym_in_sym_list(sptr, DI_CONC_ERROR_SYMS(*doif))) {
        error(1050, ERR_Severe, gbl.lineno, "Branch out of", CNULL); // C1138
        DI_CONC_ERROR_SYMS(*doif) =
            add_symitem(sptr, DI_CONC_ERROR_SYMS(*doif));
        break;
      }
    }
    break;

  case A_RETURN:
    error(1050, ERR_Severe, gbl.lineno, "RETURN in", CNULL); // 2018-C1136
    break;

  case A_ASN:
    // Check for deallocation of a polymorphic entity.
    // Visit user assignments; skip compiler-created initializations.
    // Finalizers for assignments are explicit calls, handled under A_CALL.
    if (ast_is_sym(A_SRCG(ast))) {
      sptr = sym_of_ast(A_SRCG(ast));
      if (CCSYMG(sptr) || HCCSYMG(sptr))
        break;
    }
    sptr = memsym_of_ast(A_DESTG(ast));
    if (CCSYMG(sptr) || HCCSYMG(sptr))
      break;
    if (ALLOCATTRG(sptr) && CLASSG(sptr))
      error(1051, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL); // 2018-C1140
    break;

  case A_ALLOC:
    // Check for deallocation of a polymorphic entity.
    // Check for an impure call.
    // Visit deallocate; skip allocate.
    if (A_TKNG(ast) == TK_ALLOCATE)
      break;
    sptr = memsym_of_ast(A_SRCG(ast));
    if (CLASSG(sptr))
      error(1051, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL); // 2018-C1140
    if (has_impure_finalizer(sptr))
      error(488, ERR_Severe, gbl.lineno, // 2018-C1139
            "Final subroutine of DEALLOCATE object", SYMNAME(sptr));
    break;
  }

  return false;
}

/** \brief       Check a DO CONCURRENT loop for constraint violations.
 *  \param doif  DO CONCURRENT doif stack slot index
 */
void
check_doconcurrent(int doif)
{
  int i, std;
  int savelineno = gbl.lineno;

  // Visit header code.  (The checks to perform vary by loop component.)
  DI_CONC_KIND(doif) = DC_HEADER;
  gbl.lineno = DI_LINENO(doif);
  for (i = doif - DI_CONC_COUNT(doif) + 1; i <= doif; ++i) {
    DOINFO *doinfo = DI_DOINFO(i);
    ast_visit(1, 1);
    if (A_TYPEG(doinfo->init_expr) != A_CNST)
      ast_traverse(doinfo->init_expr, check_doconcurrent_ast, NULL, &doif);
    if (A_TYPEG(doinfo->limit_expr) != A_CNST)
      ast_traverse(doinfo->limit_expr, check_doconcurrent_ast, NULL, &doif);
    if (A_TYPEG(doinfo->step_expr) != A_CNST)
      ast_traverse(doinfo->step_expr, check_doconcurrent_ast, NULL, &doif);
    ast_unvisit();
  }

  // Visit mask code.
  std = DI_CONC_MASK_STD(doif);
  if (std) {
    DI_CONC_KIND(doif) = DC_MASK;
    gbl.lineno = STD_LINENO(std);
    ast_visit(1, 1);
    ast_traverse(STD_AST(std), check_doconcurrent_ast, NULL, &doif);
    ast_unvisit();
  }

  // Body marker is the last header or mask std; adjust it.
  DI_CONC_BODY_STD(doif) = STD_NEXT(DI_CONC_BODY_STD(doif));

  // Get a list of labels defined in the loop, including cycle and end labels.
  if (DI_CYCLE_LABEL(doif))
    DI_CONC_LABEL_SYMS(doif) = add_symitem(DI_CYCLE_LABEL(doif), 0);
  if (scn.currlab)
    DI_CONC_LABEL_SYMS(doif) =
        add_symitem(scn.currlab, DI_CONC_LABEL_SYMS(doif));
  for (std = DI_CONC_BODY_STD(doif); std; std = STD_NEXT(std))
    if (STD_LABEL(std))
      DI_CONC_LABEL_SYMS(doif) =
          add_symitem(STD_LABEL(std), DI_CONC_LABEL_SYMS(doif));

  // Visit body code.
  DI_CONC_KIND(doif) = DC_BODY;
  ast_visit(1, 1);
  for (std = DI_CONC_BODY_STD(doif); std; std = STD_NEXT(std)) {
    gbl.lineno = STD_LINENO(std);
    ast_traverse(STD_AST(std), check_doconcurrent_ast, NULL, &doif);
  }
  ast_unvisit();

  gbl.lineno = savelineno;
}

/*
 * Generate the ast for a scalar logical if expression.  If any temps
 * were allocated for the expression, need to assign the expression to
 * a scalar temp, deallocate the temps, and return the AST for the new
 * temp.
 */
static int
gen_logical_if_expr(SST *stkp)
{
  int ast;
  switch (SST_DTYPEG(stkp)) {
  case DT_LOG4:
  case DT_LOG8:
    mkexpr1(stkp);
    break;
  default:
    (void)chk_scalartyp(stkp, DT_LOG, FALSE);
  }
  ast = check_etmp(stkp);
  return ast;
}

static int
dealloc_tmp(int sptr)
{
  int ast;
  ast = gen_alloc_dealloc(TK_DEALLOCATE, mk_id(sptr), 0);
  return ast;
}

/** \brief Return TRUE if derived type has a polymorphic member.

    If flag is set, then it also returns TRUE if it has a pointer or
    allocatable derived type member with a type descriptor embedded in
   the type
    itself.
 */
int
has_poly_mbr(int sptr, int flag)
{
  int dtype, mem;

  if (!sptr)
    return 0;

  dtype = DTYPEG(sptr);
  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);
  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      if (has_poly_mbr(mem, flag))
        return 1;
    } else if (CLASSG(mem)) {
      if (is_tbp_or_final(mem))
        continue; /* skip tbp */
      return 1;
    } else if (flag && ALLOCDESCG(mem)) {
      return 1;
    }
  }
  return 0;
}

static int
find_non_tbp(char *symname)
{
  int hash, hptr, len;
  len = strlen(symname);
  HASH_ID(hash, symname, len);
  for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
    if (STYPEG(hptr) == ST_PROC && strcmp(SYMNAME(hptr), symname) == 0 &&
        !CLASSG(hptr)) {
      return hptr;
    }
  }
  return 0;
}

void
push_tbp_arg(ITEM *item)
{
  ITEM **items;
  int i;

  items = (ITEM **)getitem(0, sizeof(ITEM *) * (sem.tbp_arg_cnt + 1));
  items[0] = item;
  for (i = 0; i < sem.tbp_arg_cnt; ++i)
    items[i + 1] = sem.tbp_arg[i];
  sem.tbp_arg = items;
  sem.tbp_arg_cnt += 1;
}

ITEM *
pop_tbp_arg(void)
{
  ITEM *item;

  if (sem.tbp_arg_cnt > 0) {
    item = sem.tbp_arg[0];
    sem.tbp_arg_cnt -= 1;
    if (sem.tbp_arg_cnt > 0)
      sem.tbp_arg += 1;
    else if (sem.tbp_arg_cnt == 0)
      sem.tbp_arg = 0;
    return item;
  }
  return 0;
}

void
err307(const char *msg, int diname, int namedc)
{
  char *nm;
  if (diname)
    nm = stb.n_base + diname;
  else
    nm = stb.n_base + namedc;
  error(307, 3, gbl.lineno, msg, nm);
}

static SST *
gen_fcn_sst(const char *nm, int dtype)
{
  int sptr;
  SST *fcn_sst;

  sptr = getsymbol(nm);
  if (IS_INTRINSIC(STYPEG(sptr))) {
    setimplicit(sptr);
  }
  fcn_sst = (SST *)getitem(0, sizeof(SST));
  BZERO(fcn_sst, SST, 1);
  SST_IDP(fcn_sst, S_IDENT);
  SST_SYMP(fcn_sst, sptr);
  SST_DTYPEP(fcn_sst, dtype);

  return fcn_sst;
}

static SST *
rewrite_cmplxpart_rhs(const char *i_cmplxnm, SST *realpart, SST *imagpart,
                      int dtype)
{
  SST *i_cmplx_fcn;
  ITEM *i_cmplx_arg;

  i_cmplx_fcn = gen_fcn_sst(i_cmplxnm, dtype);
  i_cmplx_arg = (ITEM *)getitem(0, sizeof(ITEM));
  BZERO(i_cmplx_arg, ITEM, 1);
  i_cmplx_arg->t.stkp = realpart;
  i_cmplx_arg->next = (ITEM *)getitem(0, sizeof(ITEM));
  BZERO(i_cmplx_arg->next, ITEM, 1);
  i_cmplx_arg->next->t.stkp = imagpart;
  i_cmplx_arg->next->next = ITEM_END;
  mkvarref(i_cmplx_fcn, i_cmplx_arg);

  return i_cmplx_fcn;
}

static SST *
gen_cmplxpart_intr(const char *i_partnm, SST *part, int dtype)
{
  SST *fcn;
  ITEM *arg;

  arg = (ITEM *)getitem(0, sizeof(ITEM));
  BZERO(arg, ITEM, 1);
  arg->t.stkp = (SST *)getitem(0, sizeof(SST));
  BZERO(arg->t.stkp, SST, 1);
  *arg->t.stkp = *part;
  arg->next = ITEM_END;
  fcn = gen_fcn_sst(i_partnm, dtype);
  mkvarref(fcn, arg);

  return fcn;
}

static void
chk_and_rewrite_cmplxpart_assn(SST *lhs, SST *rhs)
{
  int ast;
  int sptr;
  int part; /* 1==> real, 2==>imag */
  int dtype;
  const char *i_realnm;
  const char *i_imagnm;
  const char *i_cmplxnm;
  SST *fcn;
  SST *i_cmplx_fcn;

  if ((ast = SST_ASTG(lhs)) && A_TYPEG(ast) == A_MEM &&
      DT_ISCMPLX(A_DTYPEG(A_PARENTG(ast))) && A_TYPEG(A_MEMG(ast)) == A_ID) {
    sptr = A_SPTRG(A_MEMG(ast));
    if (strcmp(SYMNAME(sptr), "re") == 0) {
      part = 1;
    } else if (strcmp(SYMNAME(sptr), "im") == 0) {
      part = 2;
    } else {
      return;
    }
    dtype = A_DTYPEG(ast);
    switch (DTY(dtype)) {
    case TY_REAL:
      i_realnm = "real";
      i_imagnm = "imag";
      i_cmplxnm = "cmplx";
      break;
    case TY_DBLE:
      i_realnm = "dreal";
      i_imagnm = "dimag";
      i_cmplxnm = "dcmplx";
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      i_realnm = "qreal";
      i_imagnm = "qimag";
      i_cmplxnm = "qcmplx";
      break;
#endif
    default:
      interr("chk_and_rewrite_cmplxpart_assn: unexpected type", DTY(dtype), 3);
    }

    /* rewrite rhs: "<cplx var>%<cmplx part> ="  to "<cplx var> =" */
    SST_IDP(lhs, S_EXPR);
    SST_ASTP(lhs, A_PARENTG(ast));
    SST_DTYPEP(lhs, A_DTYPEG(A_PARENTG(ast)));
    SST_PARENP(lhs, 0);

    if (part == 1) {
      /* orig assn: <cplx var>%re = <expr>
       * rewrite rhs to:            cmplx(<expr>, imag(<cplx var>))
       * giving:    <cplx var>    = cmplx(<expr>, imag(<cplx var>))
       */
      fcn = gen_cmplxpart_intr(i_imagnm, lhs, dtype);
      i_cmplx_fcn =
          rewrite_cmplxpart_rhs(i_cmplxnm, rhs, fcn, A_TYPEG(SST_ASTG(lhs)));

    } else {
      /* orig assn: <cplx var>%im = <expr>
       * rewrite rhs to:            cmplx(real(<cmplx var>), <expr>)
       * giving:    <cplx var>    = cmplx(real(<cmplx var>), <expr>)
       */
      fcn = gen_cmplxpart_intr(i_realnm, lhs, dtype);
      i_cmplx_fcn =
          rewrite_cmplxpart_rhs(i_cmplxnm, fcn, rhs, A_TYPEG(SST_ASTG(lhs)));
    }
    *rhs = *i_cmplx_fcn;
    SST_IDP(lhs, S_LVALUE);
  }
}

static LOGICAL
rhs_expr_idx_dependent(int ast, int *id_dep)
{
  if (A_TYPEG(ast) == A_ID) {
    if (sym_in_sym_list(A_SPTRG(ast), DI_IDXLIST(sem.doif_depth))) {
      *id_dep = TRUE;
      return TRUE;
    }
  }
  return FALSE;
}

static LOGICAL
init_exprs_idx_dependent(int strt_std, int end_std)
{
  int ast;
  LOGICAL is_dep = FALSE;
  int i;

  for (i = strt_std; i && i < end_std; i = STD_NEXT(i)) {
    ast = STD_AST(i);
    if (A_TYPEG(ast) == A_ASN) {
      ast_visit(1, 1);
      ast_traverse(A_SRCG(ast), rhs_expr_idx_dependent, NULL, &is_dep);
      ast_unvisit();
      if (is_dep) {
        break;
      }
    }
  }
  return is_dep;
}

static int
new_member_expr(int new_parent, int mem)
{
  int ast;

  ast = mem;
  if (A_TYPEG(mem) == A_SUBSCR) {
    ast = new_member_expr(new_parent, A_LOPG(mem));
  } else if (A_TYPEG(A_PARENTG(ast)) == A_MEM) {
    ast = new_member_expr(new_parent, A_PARENTG(ast));
    ast = mk_member(ast, A_MEMG(mem), A_DTYPEG(ast));
  } else {
    ast = mk_member(new_parent, A_MEMG(ast), A_DTYPEG(ast));
  }

  return ast;
}

static int
gen_derived_arr_init(int arr_dtype, int strt_std, int end_std)
{
  int sptr;
  int subscr[MAXRANK] = {0, 0, 0, 0, 0, 0, 0};
  int idx_item;
  int ndim;
  int prev_std;

  int asgn_ast;
  int dest_ast;
  int subscr_ast;
  int i;
  int std;

  sptr = get_arr_temp(arr_dtype, FALSE, FALSE, FALSE);
  subscr_ast = mk_id(sptr);

  /* gen subscr'd temp */
  ndim = AD_NUMDIM(AD_DPTR(arr_dtype));
  for (i = ndim - 1, idx_item = DI_IDXLIST(sem.doif_depth); i >= 0;
       idx_item = SYMI_NEXT(idx_item), i--) {
    subscr[i] = mk_id(SYMI_SPTR(idx_item));
  }
  subscr_ast = mk_subscr(subscr_ast, subscr, ndim, DDTG(arr_dtype));

  /* build forall's that assign init rhs to subscr'd tmp */
  prev_std = 0;
  for (i = strt_std; i && i < end_std; i = STD_NEXT(i)) {
    asgn_ast = STD_AST(i);
    dest_ast = A_DESTG(asgn_ast);
    if (prev_std) {
      delete_stmt(prev_std);
    }

    assert(A_TYPEG(asgn_ast) == A_ASN, "expected initialization assignment",
           asgn_ast, 4);
    assert(A_TYPEG(A_DESTG(asgn_ast)) == A_MEM ||
               A_TYPEG(A_DESTG(asgn_ast)) == A_SUBSCR,
           "unexpected LHS expression in initialization assignment",
           A_MEMG(dest_ast), 4);

    dest_ast = new_member_expr(subscr_ast, A_DESTG(asgn_ast));

    A_DESTP(asgn_ast, dest_ast);
    std = add_stmt_before(asgn_ast, i);
    prev_std = i;
  }
  delete_stmt(prev_std);

  return subscr_ast;
}

static int
convert_to_block_forall(int old_forall_ast)
{
  int i;
  int ast;
  int prev_std;
  int curr_last_std = astb.std.stg_avail;
  ITEM *dealloc_list = ITEM_END;
  ITEM *itemp;
  ITEM **p_dealloc;

  ast = mk_stmt(A_FORALL, 0);
  A_IFEXPRP(ast, A_IFEXPRG(old_forall_ast));
  A_SRCP(ast, A_SRCG(old_forall_ast));
  A_LISTP(ast, A_LISTG(old_forall_ast));
  add_stmt(ast);

  prev_std = 0;
  i = DI_FORALL_LASTSTD(sem.doif_depth) == 0
          ? STD_NEXT(0)
          : STD_NEXT(DI_FORALL_LASTSTD(sem.doif_depth));
  for (; i && i < curr_last_std; i = STD_NEXT(i)) {
    if (A_TYPEG(STD_AST(i)) == A_COMMENT) {
      continue;
    }
    if (prev_std) {
      delete_stmt(prev_std);
    }
    ast = STD_AST(i);
    if (A_TYPEG(ast) == A_ALLOC && A_TKNG(ast) == TK_ALLOCATE) {
      itemp = (ITEM *)getitem(1, sizeof(ITEM));
      itemp->t.sptr = sym_of_ast(A_SRCG(ast));
      itemp->next = dealloc_list;
      dealloc_list = itemp;
    }
    add_stmt(ast);
    prev_std = i;
  }
  if (A_IFSTMTG(old_forall_ast)) {
    add_stmt(A_IFSTMTG(old_forall_ast));
  }
  delete_stmt(prev_std);

  /* if an allocate was moved into the forall loop, generate the dealloc
   * and
   * remove
   * the item from the sem.p_dealloc list */
  for (itemp = dealloc_list; itemp != ITEM_END; itemp = itemp->next) {
    ast = gen_alloc_dealloc(TK_DEALLOCATE, mk_id(itemp->t.sptr), 0);
    for (p_dealloc = &sem.p_dealloc;
         p_dealloc && (*p_dealloc)->next != ITEM_END;
         p_dealloc = &((*p_dealloc)->next)) {
      if (A_SPTRG((*p_dealloc)->ast) == itemp->t.sptr) {
        *p_dealloc = (*p_dealloc)->next;
      }
    }
  }

  return mk_stmt(A_ENDFORALL, 0);
}

/** \brief Generate a call to init_unl_poly_desc which initializes a descriptor
 *         for an unlimited polymorphic object with another descriptor.
 *
 *  \param dest_sdsc_ast is the AST of the destination's descriptor.
 *  \param src_sdsc_ast is the AST of the source descriptor.
 *  \param std is the statement descriptor to insert the call, or 0 to use
 *         the default statement descriptor.
 */
void
gen_init_unl_poly_desc(int dest_sdsc_ast, int src_sdsc_ast, int std)
{
  int fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_init_unl_poly_desc), DT_NONE);
  int argt = mk_argt(3);
  int val = mk_cval1(43, DT_INT);
  int ast = mk_id(fsptr);
  ARGT_ARG(argt, 0) = dest_sdsc_ast;
  ARGT_ARG(argt, 1) = src_sdsc_ast;
  val = mk_unop(OP_VAL, val, DT_INT);
  ARGT_ARG(argt, 2) = val;
  ast = mk_func_node(A_CALL, ast, 3, argt);
  if (std == 0) {
    add_stmt(ast);
  } else {
    add_stmt_after(ast, std);
  }
}

#ifdef FLANG_SEMANT3_UNUSED
static int
gen_sourced_allocation(int astdest, int astsrc)
{

  int dest_dtype, astdest2, src_dtype, std2;
  int subs[MAXRANK];
  int fsptr, argt;
  int dtyper, func_ast;
  int ast, i, asttmp;
  int sptrsrc, sptrdest;

  switch (A_TYPEG(astdest)) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    sptrdest = memsym_of_ast(astdest);
    break;
  default:
    sptrdest = 0;
  }
again:
  switch (A_TYPEG(astsrc)) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    sptrsrc = memsym_of_ast(astsrc);
    break;
  case A_BINOP:
    astsrc = A_LOPG(astsrc);
    goto again;
  default:
    sptrsrc = 0;
  }

  ast = mk_stmt(A_CONTINUE, 0);
  std2 = add_stmt(ast);

  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_ALLOCATE);
  A_SRCP(ast, astdest);
  A_STARTP(ast, astsrc);

  dest_dtype = A_DTYPEG(astdest);
  src_dtype = A_DTYPEG(astsrc);
  astdest2 = 0;
  if (DTY(src_dtype) == TY_ARRAY) {
    if (!SDSCG(sptrsrc)) {
      ADSC *ad = AD_DPTR(src_dtype);
      int ndims = AD_NUMDIM(ad);
      for (i = 0; i < ndims; ++i) {
        subs[i] = mk_triple(AD_LWAST(ad, i), AD_UPAST(ad, i), 0);
      }
      astdest2 = mk_subscr(astdest, subs, ndims, dest_dtype);
      A_SRCP(ast, astdest2);
    } else {
      /* Source has a descriptor, so use size intrinsic */
      if (XBIT(68, 0x1) && XBIT(68, 0x2))
        dtyper = DT_INT8;
      else
        dtyper = stb.user.dt_int;

      if (XBIT(68, 0x1))
        fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_sizeDsc), dtyper);
      else
        fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_sizeDsc), dtyper);
      argt = mk_argt(2);
      ARGT_ARG(argt, 0) = astb.ptr0;
      ARGT_ARG(argt, 1) = check_member(astsrc, mk_id(SDSCG(sptrsrc)));
      func_ast = mk_id(fsptr);
      func_ast = mk_func_node(A_FUNC, func_ast, 2, argt);
      A_DTYPEP(func_ast, dtyper);
      asttmp = mk_id(get_temp(DT_INT));
      func_ast = mk_assn_stmt(asttmp, func_ast, dtyper);
      std2 = add_stmt_after(func_ast, std2);
      subs[0] = mk_triple(astb.i1, asttmp, 0);
      astdest2 = mk_subscr(astdest, subs, 1, dest_dtype);
      A_SRCP(ast, astdest2);
    }
  }

  std2 = add_stmt_after(ast, std2);

  return std2;
}
#endif

/* Predicate: is the right-hand side of an association in an ASSOCIATE
 * or SELECT TYPE statement suitable for a direct association (as
 * opposed to
 * an association with a temporary containing the captured value of an
 * expression)?  If so, assignments through the association's left-hand
 * side
 * name must be supported and their values will persist after the
 * association,
 * since a temporary will (and must) not have been created.
 *
 * This test should basically work out to be the same test as for an
 * actual
 * argument association to a pass-by-descriptor argument, or for a valid
 * right-hand side of a hypothetical variant of pointer assignment that
 * ignored
 * TARGET and POINTER attributes.  Arrays with a vector-valued subscript
 * cannot
 * be not accepted.
 */

static LOGICAL
is_associatable_variable_ast(int ast)
{
  return is_data_ast(ast) && is_variable_ast(ast) &&
         !has_vector_subscript_ast(ast);
}

static LOGICAL
is_associatable_variable_sst(SST *rhs)
{
  if (rhs) {
    switch (SST_IDG(rhs)) {
    case S_IDENT:
      return !is_procedure_ptr(SST_SYMG(rhs));
    case S_LVALUE:
    case S_EXPR:
      return is_associatable_variable_ast(SST_ASTG(rhs));
    }
  }
  return FALSE;
}

/* Implement an association for one of these constructs:
 *
 *   ASSOCIATE(a => variable)
 *   ASSOCIATE(a => expression)
 *   SELECT TYPE(whole variable)   - optional usage
 *   SELECT TYPE(a => variable)
 *   SELECT TYPE(a => expression)
 *   CLASS IS(polymorphic type)
 *   TYPE IS(monomorphic or intrinsic type)
 *
 * The association is implemented as a compiler-generated temporary
 * that is intended to work exactly as if the construct is being
 * rewritten in terms of a POINTER or ALLOCATABLE local variable.
 * POINTERs are used for the cases above where the right-hand side could
 * be legally passed as an actual argument by reference (i.e., a pointer
 * can be associated with it, less any check for the TARGET attribute),
 * and ALLOCATABLEs are used otherwise.  The distinction matters for
 * more
 * than just the run-time expense of allocating and filling a temporary,
 * because the program must be allowed to modify storage via the
 * association
 * and have those modifications persist after the association ends in
 * those
 * cases where it makes sense.
 */
static int
construct_association(int lhs_sptr, SST *rhs, int stmt_dtype, LOGICAL is_class)
{
  int rhs_ast, rhs_dtype;
  int lhs_dtype, lhs_element_dtype, lhs_ast;
  LOGICAL set_up_a_pointer;
  int rhs_sptr = 0;
  LOGICAL is_array = FALSE;
  LOGICAL is_lhs_runtime_length_char;
  LOGICAL is_lhs_unl_poly;
  int rhs_descriptor_ast = 0;
  LOGICAL does_lhs_need_runtime_type;
  int lhs_length_ast = 0;

  if (!(rhs_ast = SST_ASTG(rhs))) {
    mkexpr(rhs);
    rhs_ast = SST_ASTG(rhs);
  }

  rhs_dtype = A_DTYPEG(rhs_ast);
  rhs_sptr = get_ast_sptr(rhs_ast);

  if (STYPEG(lhs_sptr) != 0) {
    /* Shadow any current instance of the association name. */
    lhs_sptr = insert_sym_first(lhs_sptr);
  }

  lhs_dtype = stmt_dtype > 0 ? stmt_dtype : rhs_dtype;
  if (is_array_dtype(lhs_dtype)) {
    int rank = get_ast_rank(rhs_ast);
    is_array = rank > 0;
    if (is_array) {
      lhs_element_dtype = array_element_dtype(lhs_dtype);
      lhs_element_dtype = change_assumed_char_to_deferred(lhs_element_dtype);
      lhs_dtype = get_array_dtype(rank, lhs_element_dtype);
      ADD_DEFER(lhs_dtype) = TRUE;
      ADD_NOBOUNDS(lhs_dtype) = TRUE;
    }
  } else {
    lhs_dtype = change_assumed_char_to_deferred(lhs_dtype);
    lhs_element_dtype = lhs_dtype;
  }
  is_lhs_runtime_length_char = is_dtype_runtime_length_char(lhs_dtype);

  DTYPEP(lhs_sptr, lhs_dtype);
  STYPEP(lhs_sptr, is_array ? ST_ARRAY : ST_VAR);
  POINTERP(lhs_sptr, TRUE);
  DCLDP(lhs_sptr, TRUE); /* dodge spurious errors with IMPLICIT NONE */
  SCOPEP(lhs_sptr, stb.curr_scope);
  ADDRTKNP(lhs_sptr,
           TRUE); /* do not remove pointer assignment in optimization */
  SCP(lhs_sptr, SC_BASED);

  /* N.B. We do NOT set CCSYM on the lhs_sptr temporary.  Doing so would
   * causes overly strong assumptions to be made about aliasing that
   * lead to
   * tests (e.g. oop481) to fail with optimization enabled.
   */

  set_up_a_pointer = is_associatable_variable_sst(rhs);
  if (set_up_a_pointer) {
    /* Implement the association with a pointer. */
    int rhs_base_sptr = sym_of_ast(rhs_ast);
    rhs_descriptor_ast = find_descriptor_ast(rhs_sptr, rhs_ast);
    ALLOCP(lhs_sptr, is_array);
    PTRVP(lhs_sptr, TRUE);
    F90POINTERP(lhs_sptr, TRUE);
    if (rhs_base_sptr > NOSYM) {
      ADDRTKNP(rhs_base_sptr, TRUE);
      PTRRHSP(rhs_base_sptr, TRUE);
      PTRSAFEP(rhs_base_sptr, FALSE);
    }
#if DEBUG
    assert(rhs_sptr > NOSYM, "construct_association: no rhs sptr", 0, 3);
#endif
    if (rhs_sptr > NOSYM) {
      ADDRTKNP(rhs_sptr, TRUE);
      PTRRHSP(rhs_sptr, TRUE);
      PTRSAFEP(rhs_sptr, FALSE);
      VOLP(lhs_sptr, VOLG(rhs_sptr));
      DEVICEP(lhs_sptr, DEVICEG(rhs_sptr));

      /* Associations can become pointer targets if and only if their
       * right-hand sides would be legal pointer targets.
       */
      TARGETP(lhs_sptr, TARGETG(rhs_sptr) || !stmt_dtype && POINTERG(rhs_sptr));
    }
  } else {
    /* Implement the association as an allocatable. */
    ALLOCP(lhs_sptr, TRUE);
    ALLOCATTRP(lhs_sptr, TRUE);
    NOALLOOPTP(lhs_sptr, TRUE);
  }

  /* Set the polymorphism flag if the LHS is meant to be polymorphic. */
  if (stmt_dtype) {
    if (is_class) {
      /* CLASS IS statement */
      CLASSP(lhs_sptr, TRUE);
    } else {
      /* TYPE IS statement */
      if (rhs_sptr > NOSYM && CLASSG(rhs_sptr)) {
        /* For TYPE IS that's specializing a polymorphic type, the LHS
         * is
         * also marked polymorphic so that its descriptor will be set up
         * so as to be able to address the RHS' data.
         * But the new MONOMORPHIC flag is also set to inhibit what
         * would
         * otherwise be a spurious error message when the temp is used
         * in a
         * context requiring monomorphism.  This is bogus and I don't
         * understand why it works, but it does.
         */
        CLASSP(lhs_sptr, TRUE);
        MONOMORPHICP(lhs_sptr, TRUE);
      }
    }
  } else {
    /* ASSOCIATE or SELECT TYPE selector association */
    if (rhs_sptr > NOSYM && CLASSG(rhs_sptr))
      CLASSP(lhs_sptr, TRUE);
  }

  /* There should be a utility routine to encapsulate the determination
   * of whether a pointer or allocatable requires a descriptor, rather
   * than using this ALLOCDESC flag that's set all over the front-end.
   */
  does_lhs_need_runtime_type = CLASSG(lhs_sptr) || has_tbp(lhs_dtype);
  if (is_array || is_lhs_runtime_length_char || does_lhs_need_runtime_type)
    ALLOCDESCP(lhs_sptr, TRUE);

  /* If the LHS's descriptor needs a pointer to a type descriptor,
   * this call will set a static flag in rte.c to allocate space for
   * it when get_static_descriptor() is called.  Goodness only knows
   * why this interface is called set_descriptor_rank() or why
   * this information can't just be passed as arguments to
   * get_static_descriptor().
   */
  set_descriptor_rank(does_lhs_need_runtime_type);
  get_static_descriptor(lhs_sptr);
  set_descriptor_rank(FALSE /* to reset the hidden API state :-P */);
  get_all_descriptors(lhs_sptr);
  if (sem.parallel || sem.target || sem.task) {
    if (SDSCG(lhs_sptr)) {
      SCP(SDSCG(lhs_sptr), SC_PRIVATE);
    }
    if (MIDNUMG(lhs_sptr)) {
      SCP(MIDNUMG(lhs_sptr), SC_PRIVATE);
    }
    if (PTROFFG(lhs_sptr)) {
      SCP(PTROFFG(lhs_sptr), SC_PRIVATE);
    }
  }

  lhs_ast = mk_id(lhs_sptr); /* must follow descriptor creation */
  is_lhs_unl_poly = is_unl_poly(lhs_sptr);

  /* When the left-hand side is CLASS(*), special initialization of
   * its descriptor is required.
   */
  if (is_lhs_unl_poly) {
#if DEBUG
    assert(SDSCG(lhs_sptr) > NOSYM, "unl poly lhs has no sdsc", lhs_sptr, 4);
    assert(rhs_descriptor_ast > 0, "no rhs descr for unl poly lhs", lhs_sptr,
           4);
#endif
    gen_init_unl_poly_desc(mk_id(SDSCG(lhs_sptr)), rhs_descriptor_ast, 0);
  }
  if (set_up_a_pointer) {
    /* Construct association by means of a pointer to extant data, no
     * temporary */
    int assignment_ast =
        add_ptr_assign(lhs_ast, rhs_ast, 0 /* no STD to precede */);
    add_stmt(assignment_ast);
  } else {
    /* Construct association by means of an allocated temporary */
    int alloc_lhs_ast =
        add_shapely_subscripts(lhs_ast, rhs_ast, lhs_dtype, DDTG(lhs_dtype));
    int alloc_ast = mk_stmt(A_ALLOC, 0);
    int assignment_ast = mk_assn_stmt(lhs_ast, rhs_ast, rhs_dtype);
    A_TKNP(alloc_ast, TK_ALLOCATE);
    A_SRCP(alloc_ast, alloc_lhs_ast);
    A_FIRSTALLOCP(alloc_ast, TRUE);
    A_STARTP(alloc_ast, rhs_ast);
    add_stmt(alloc_ast);
    add_stmt(assignment_ast);
  }

  /* For TYPE IS statements with intrinsic types, set the LHS type
   * directly.
   */
  if (SDSCG(lhs_sptr) > NOSYM && stmt_dtype && !is_class &&
      DTY(lhs_element_dtype) != TY_DERIVED) {
    int type_code = dtype_to_arg(lhs_element_dtype);
    int assignment_ast = mk_assn_stmt(get_kind(SDSCG(lhs_sptr)),
                                      mk_cval1(type_code, DT_INT), DT_INT);
    add_stmt(assignment_ast);
  }

  /* Generate code to initialize, when necessary, the byte length field
   * in the left-hand side's descriptor, if it exists.
   */
  if (is_lhs_runtime_length_char || is_lhs_unl_poly || is_array)
    lhs_length_ast = symbol_descriptor_length_ast(lhs_sptr, 0 /*no AST*/);
  if (lhs_length_ast > 0) {
    SPTR size_sptr = stmt_dtype > DT_NONE && !is_class /* TYPE IS */ &&
                             !is_lhs_runtime_length_char
                         ? lhs_sptr
                         : NOSYM;
    int rhs_length_ast = get_value_length_ast(
        rhs_dtype, rhs_ast, size_sptr, lhs_element_dtype, rhs_descriptor_ast);
    if (rhs_length_ast > 0)
      add_stmt(mk_assn_stmt(lhs_length_ast, rhs_length_ast, astb.bnd.dtype));
  }

  return lhs_sptr;
}

static void
end_association(int sptr)
{
  pop_sym(sptr);
  if (ALLOCATTRG(sptr)) {
    int ast = mk_stmt(A_ALLOC, 0);
    int sptr_ast = mk_id(sptr);
    A_TKNP(ast, TK_DEALLOCATE);
    A_SRCP(ast, sptr_ast);
    add_stmt(ast);
  }
  /* TODO: deallocate memory on branches out of the construct, too! */
}

/* If a semantic stack entry names a whole variable that's in scope,
 * return its symbol table index.
 */
static int
get_sst_named_whole_variable(SST *rhs)
{
  int sptr = 0;

  switch (SST_IDG(rhs)) {
  case S_IDENT:
    sptr = SST_SYMG(rhs);
    break;
  case S_LVALUE:
  case S_EXPR: {
    int ast = SST_ASTG(rhs);
    if (A_TYPEG(ast) == A_ID)
      sptr = A_SPTRG(ast);
    break;
  }
  }
  if (sptr > NOSYM && test_scope(sptr) < 0)
    return 0; /* not in scope */
  return sptr;
}

static int
get_derived_type(SST *sst, LOGICAL abstract_type_not_allowed)
{
  int dtype, sptr;
  dtype = 0;
  sptr = refsym(SST_SYMG(sst), OC_OTHER);
  if (sptr != 0) {
    if (STYPEG(sptr) == ST_USERGENERIC && GTYPEG(sptr) > NOSYM)
      sptr = GTYPEG(sptr);
    if (STYPEG(sptr) == ST_TYPEDEF)
      dtype = DTYPEG(sptr);
  }
  if (dtype == 0) {
    error(155, 3, gbl.lineno, "Derived type has not been declared -",
          SYMNAME(SST_SYMG(sst)));
    if (scn.stmtyp == TK_CLASSIS)
      error(155, 4, gbl.lineno,
            "Type specified in CLASS IS must be an "
            "extensible type",
            NULL);
    if (scn.stmtyp == TK_TYPEIS)
      error(155, 4, gbl.lineno,
            "Length type parameter in TYPE IS must "
            "be assumed (*)",
            NULL);
  } else if (abstract_type_not_allowed && ABSTRACTG(DTY(dtype + 3))) {
    error(155, 3, gbl.lineno, "illegal use of abstract type", SYMNAME(sptr));
  }
  return dtype;
}
