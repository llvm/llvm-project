/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Fortran utility routines used by Semantic Analyzer to process
           user-defined generics including overloaded operators
 */

#include "gbldefs.h"
#include "global.h"
#include "gramtk.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "semstk.h"
#include "pd.h"
#include "machar.h"
#include "ast.h"
#include "state.h"

static int silent_error_mode = 0;
#undef E155
#define E155(s1, s2)      \
  if (!silent_error_mode) \
  error(155, 3, gbl.lineno, s1, s2)

static int resolve_generic(int, SST *, ITEM *);
static long *args_match(int, int, int, ITEM *, LOGICAL, LOGICAL);
static LOGICAL tkr_match(int, SST *, int, LOGICAL);
static LOGICAL kwd_match(ITEM *, int, char *);
static void get_type_rank(SST *, int *, int *);
static ITEM *make_list(SST *, SST *);
static int resolve_operator(int, SST *, SST *);
static int find_operator(int, SST *, SST *, LOGICAL);
static bool queue_generic_tbp_once(SPTR gnr);
static bool is_conflicted_generic(SPTR, SPTR);

/* macros used by the arg scoring routines */
#define UNIT_SZ 3 /**< bits necessary to hold the max *_MATCH value */
#define NBR_DISTANCE_ELM_BITS ((sizeof(long) * 8 - 1) / UNIT_SZ)
#define DISTANCE_BIT(i) (i % NBR_DISTANCE_ELM_BITS)
#define DISTANCE_ELM(distance, i) (distance[i / NBR_DISTANCE_ELM_BITS])

/* constants returned by by args_match and tkr_match */
#define INF_DISTANCE ((long)-1)
#define MIN_DISTANCE 0
/* also returned by tkr_match */
#define EXACT_MATCH 0
#define EXTND_MATCH 4
/* UNIT_SZ (above) must be the number of bits necessary to hold the max  *_MATCH
 * value */

#define MAN_MAN_MATCH 0
#define MAN_DEV_MATCH 1
#define MAN_HOST_MATCH 2

static int resolved_to_self = 0;

/*
 * Table used to record and return the ST_OPERATOR symbols corresponding
 * to the intrinsic and the assignment operators.
 */
static struct optabstruct {
  int opr;          /* if non-zero, locates the ST_OPERATOR symbol */
  const char *name; /* name of the corresponding ST_OPERATOR symbol */
} optab[] = {
    {0, ""},       /* OP_NEG	0 */
    {0, "+"},      /* OP_ADD	1 */
    {0, "-"},      /* OP_SUB	2 */
    {0, "*"},      /* OP_MUL	3 */
    {0, "/"},      /* OP_DIV	4 */
    {0, "**"},     /* OP_XTOI	5 */
    {0, ""},       /* OP_XTOX	6 */
    {0, ""},       /* OP_CMP	7 */
    {0, ""},       /* OP_AIF	8 */
    {0, ""},       /* OP_LD	9 */
    {0, "="},      /* OP_ST	10 */
    {0, ""},       /* OP_FUNC	11 */
    {0, ""},       /* OP_CON	12 */
    {0, "//"},     /* OP_CAT	13 */
    {0, ""},       /* OP_LOG	14 */
    {0, ".eqv."},  /* OP_LEQV	15 */
    {0, ".neqv."}, /* OP_LNEQV	16 */
    {0, ".or."},   /* OP_LOR	17 */
    {0, ".and."},  /* OP_LAND	18 */
    {0, "=="},     /* OP_EQ	19 */
    {0, ">="},     /* OP_GE	20 */
    {0, ">"},      /* OP_GT	21 */
    {0, "<="},     /* OP_LE	22 */
    {0, "<"},      /* OP_LT	23 */
    {0, "!="},     /* OP_NE	24 */
    {0, ".not."},  /* OP_LNOT	25 */
    {0, ""},       /* OP_LOC	26 */
    {0, ""},       /* OP_REF	27 */
    {0, ""},       /* OP_VAL	28 */
};
#define OPTABSIZE 29

/** \brief Determines if we should (re)generate generic type bound procedure
 *  (tbp) bindings based on scope. This should only be done once per scope.
 *
 *  \param gnr is the SPTR of the symbol to check or 0 if N/A.
 *
 *  \return true if we should (re)generate generic tbp bindings, else false.
 */
static bool
queue_generic_tbp_once(SPTR gnr)
{
  if (GNCNTG(gnr) == 0 || gbl.internal > 1) {
    static int generic_tbp_scope = 0;
    bool rslt = (generic_tbp_scope != stb.curr_scope);
    generic_tbp_scope = stb.curr_scope;
    return rslt;
  }
  return false;
}

/** \brief Determines if two generic procedures from different
     modules are conflicted or not. 
 *
 *  \param found_sptrgen is the first generic procedure sptr.
 *  \param func_sptrgen is the second generic procedure sptr.
 *
 *  \return true if the func_sptrgen and found_sptrgen are not conflicted, else
 *   false.
 */
static bool
is_conflicted_generic(SPTR func_sptrgen, SPTR found_sptrgen) {
  return func_sptrgen != found_sptrgen &&
         (PRIVATEG(func_sptrgen) != PRIVATEG(found_sptrgen) ||
         NOT_IN_USEONLYG(func_sptrgen) != NOT_IN_USEONLYG(found_sptrgen));
}

void
check_generic(int gnr)
{
  if (STYPEG(gnr) == ST_USERGENERIC) {
    ;
  } else {
#if DEBUG
    assert(STYPEG(gnr) == ST_OPERATOR, "check_generic, expected ST_OPERATOR",
           STYPEG(gnr), 3);
#endif
  }
}

int
generic_tbp_call(int gnr, SST *stktop, ITEM *list, ITEM *chevlist)
{
  int sptr;

#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "user generic, call %s\n", SYMNAME(gnr));
#endif
  if (queue_generic_tbp_once(gnr)) {
    queue_tbp(0, 0, 0, 0, TBP_COMPLETE_GENERIC);
  }

  if (list == NULL)
    list = ITEM_END;
  sptr = resolve_generic(gnr, stktop, list);
  return sptr;
}

void
generic_call(int gnr, SST *stktop, ITEM *list, ITEM *chevlist)
{
  int sptr;

#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "user generic, call %s\n", SYMNAME(gnr));
#endif
  if (list == NULL)
    list = ITEM_END;
  sptr = resolve_generic(gnr, stktop, list);
  if (sptr == 0) {
    SST_ASTP(stktop, 0);
    return;
  }
#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "user generic resolved to %s\n", SYMNAME(sptr));
#endif
  SST_SYMP(stktop, -sptr);

    subr_call2(stktop, list, 1);

}

int
generic_tbp_func(int gnr, SST *stktop, ITEM *list)
{
  int sptr;

#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "user generic %s\n", SYMNAME(gnr));
#endif

  if (queue_generic_tbp_once(gnr)) {
    queue_tbp(0, 0, 0, 0, TBP_COMPLETE_GENERIC);
  }

  if (list == NULL)
    list = ITEM_END;
  sptr = resolve_generic(gnr, stktop, list);
  return sptr;
}

int
generic_func(int gnr, SST *stktop, ITEM *list)
{
  int sptr;

#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "user generic %s\n", SYMNAME(gnr));
#endif
  if (list == NULL)
    list = ITEM_END;
  sptr = resolve_generic(gnr, stktop, list);
  if (sptr == 0) {
    SST_IDP(stktop, S_CONST);
    SST_DTYPEP(stktop, DT_INT);
    return 3;
  }
  if (sptr == -1) {
    /*  the generic resolve to a structure constructor  */
    return 1;
  }
#if DEBUG
  if (DBGBIT(3, 256)) {
    fprintf(gbl.dbgfil, "user generic resolved to %s\n", SYMNAME(sptr));
    if (sptr < stb.firstosym)
      fprintf(gbl.dbgfil, "USING intrinsic generic\n");
  }
#endif
  mkident(stktop);
  SST_SYMP(stktop, sptr);
  SST_DTYPEP(stktop, DTYPEG(sptr));
  if (sptr < stb.firstosym) {
    if (STYPEG(sptr) == ST_PD)
      return ref_pd(stktop, list);
    return ref_intrin(stktop, list);
  }
  SST_ASTP(stktop, mk_id(sptr));
  return func_call2(stktop, list, 1);
}

static long *
set_distance_to(long value, long *distance, int sz)
{
  int i;

  for (i = 0; i < sz; i++)
    distance[i] = value;

  return distance;
}

/* compare distance, distance; return
 * -1 if distance1 < distance2
 *  0 if distance1 == distance2
 *  1 if distance1 > distance2
 */
static int
cmp_arg_score(long *distance1, long *distance2, int sz)
{
  int i;
  if (*distance1 != INF_DISTANCE && *distance2 == INF_DISTANCE) {
    return -1;
  } else if (*distance1 == INF_DISTANCE && *distance2 != INF_DISTANCE) {
    return 1;
  } else if (*distance1 == INF_DISTANCE && *distance2 == INF_DISTANCE) {
    return 0;
  }

  for (i = 0; i < sz; ++i) {
    if (distance1[i] < distance2[i])
      return -1;
    else if (distance1[i] > distance2[i])
      return 1;
  }
  return 0;
}

static int
find_best_generic(int gnr, ITEM *list, int arg_cnt, int try_device,
                  LOGICAL chk_elementals)
{
  int gndsc, nmptr;
  int sptr;
  int sptrgen;
  int found;
  int bind;
  int found_bind;
  int func;
  long *argdistance;
  long *min_argdistance = 0;
  int distance_sz;
  LOGICAL gnr_in_active_scope;
  int dscptr;
  int paramct, curr_paramct;
  SPTR found_sptrgen, func_sptrgen;

  /* find the generic's max nbr of formal args and use it to compute
   * the size of the arg distatnce data item.
   */
  paramct = 0;
  for (sptr = first_hash(gnr); sptr > NOSYM; sptr = HASHLKG(sptr)) {
    sptrgen = sptr;
    while (STYPEG(sptrgen) == ST_ALIAS)
      sptrgen = SYMLKG(sptrgen);
    for (gndsc = GNDSCG(sptrgen); gndsc; gndsc = SYMI_NEXT(gndsc)) {
      func = SYMI_SPTR(gndsc);
      while (STYPEG(func) == ST_MODPROC || STYPEG(func) == ST_ALIAS) {
        /* Need to get the actual routine symbol in order to
         * access the arguments and number of arguments of the routine.
         */
        func = SYMLKG(func);
      }
      dscptr = DPDSCG(func);
      curr_paramct = PARAMCTG(func);
      if (curr_paramct > paramct) {
        paramct = curr_paramct;
      }
    }
  }
  /* initialize arg distance data item */
  distance_sz = paramct / NBR_DISTANCE_ELM_BITS + 1;
  NEW(min_argdistance, long, distance_sz);
  (void)set_distance_to(INF_DISTANCE, min_argdistance, distance_sz);

  nmptr = NMPTRG(gnr);

  found = 0;
  found_bind = 0;
  for (sptr = first_hash(gnr); sptr > NOSYM; sptr = HASHLKG(sptr)) {
    gnr_in_active_scope = FALSE;
    sptrgen = sptr;
    if (NMPTRG(sptrgen) != nmptr)
      continue;
    if (PRIVATEG(sptr) && gbl.currmod && SCOPEG(sptr) != gbl.currmod)
      continue;
    while (STYPEG(sptrgen) == ST_ALIAS)
      sptrgen = SYMLKG(sptrgen);
    if (STYPEG(sptrgen) != ST_USERGENERIC)
      continue;
    /* is the original symbol (sptr, not sptrgen) in an active scope */
    if (test_scope(sptr) >= 0 ||
        (STYPEG(SCOPEG(sptr)) == ST_MODULE && !PRIVATEG(SCOPEG(sptr)))) {
      gnr_in_active_scope = TRUE;
    }
    if (!gnr_in_active_scope && !CLASSG(sptrgen))
      continue;
    if (GNCNTG(sptrgen) == 0 && GTYPEG(sptrgen)) {
      continue; /* Could be an overloaded type */
    }
    if (queue_generic_tbp_once(sptrgen)) {
      queue_tbp(0, 0, 0, 0, TBP_COMPLETE_GENERIC);
    }
    if (GNCNTG(sptrgen) == 0 && !IS_TBP(sptrgen)) {
      /* Ignore if generic tbp overloads sptrgen. This might be
       * an overloaded intrinsic. We check for an overloaded intrinsic
       * below.
       */

      E155("Empty generic procedure -", SYMNAME(sptr));
    }

    for (gndsc = GNDSCG(sptrgen); gndsc; gndsc = SYMI_NEXT(gndsc)) {
      func = SYMI_SPTR(gndsc);
      func_sptrgen = sptrgen;
      if (IS_TBP(func)) {
        /* For generic type bound procedures, use the implementation
         * of the generic bind name for the argument comparison.
         */
        int mem, dty;
        bind = func;
        dty = TBPLNKG(func /*sptrgen*/);
        func = get_implementation(dty, func, 0, &mem);
        if (STYPEG(BINDG(mem)) == ST_OPERATOR ||
            STYPEG(BINDG(mem)) == ST_USERGENERIC) {
          mem = get_specific_member(dty, func);
          func = VTABLEG(mem);
          bind = BINDG(mem);
        }
        if (!func)
          continue;
        mem = get_generic_member(dty, bind);
        if (NOPASSG(mem) && generic_tbp_has_pass_and_nopass(dty, BINDG(mem)))
          continue;
        if (mem && PRIVATEG(mem) && SCOPEG(stb.curr_scope) != SCOPEG(mem))
          continue;
      } else
        bind = 0;
      if (STYPEG(func) == ST_MODPROC) {
        func = SYMLKG(func);
        if (func == 0)
          continue;
      }
      if (STYPEG(func) == ST_ALIAS)
        func = SYMLKG(func);
      if (chk_elementals && ELEMENTALG(func)) {
        argdistance =
            args_match(func, arg_cnt, distance_sz, list, TRUE, try_device == 1);
      } else {
        argdistance = args_match(func, arg_cnt, distance_sz, list, FALSE,
                                 try_device == 1);
      }
      if (found && func && found != func && *min_argdistance != INF_DISTANCE &&
          !PRIVATEG(SCOPEG(func)) &&
          !is_conflicted_generic(func_sptrgen, found_sptrgen) &&
          cmp_arg_score(argdistance, min_argdistance, distance_sz) == 0) {
        int len;
        char *name, *name_cpy;
        len = strlen(SYMNAME(gnr)) + 1;
        name_cpy = getitem(0, len);
        strcpy(name_cpy, SYMNAME(gnr));
        name = strchr(name_cpy, '$');
        if (name)
          *name = '\0';
        E155("Ambiguous interfaces for generic procedure", name_cpy);
        FREE(argdistance);
        break;
      } else if (cmp_arg_score(argdistance, min_argdistance, distance_sz) ==
                 -1) {
        FREE(min_argdistance);
        min_argdistance = argdistance;
        found = func;
        found_bind = bind;
        found_sptrgen = sptrgen;
      } else {
        FREE(argdistance);
      }
    }
  }
  FREE(min_argdistance);
  found = (found_bind) ? found_bind : found;
  return found;
}

/*
 * Possible return values:
 * -1  : generic resolves to a struct constructor
 *  0  : error
 * >0  : sptr of the 'specific'
 */
static int
resolve_generic(int gnr, SST *stktop, ITEM *list)
{
  int nmptr;
  int arg_cnt;
  ITEM *itemp;
  SST *sp;
  int sptr;
  int found;
  int try_device = 0;

  arg_cnt = 0;
  for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
    arg_cnt++;
    sp = itemp->t.stkp;
    if (SST_IDG(sp) == S_TRIPLE) {
      /* form is e1:e2:e3 */
      error(76, 3, gbl.lineno, SYMNAME(gnr), CNULL);
      return 0;
    }
    if (SST_IDG(sp) == S_ACONST) {
      mkexpr(sp);
    }
  }
#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "resolve_generic: %s, count %d\n", SYMNAME(gnr),
            arg_cnt);
#endif

  nmptr = NMPTRG(gnr);
/* search HASH list for all user generics of the same name */
  {
    if ((found = find_best_generic(gnr, list, arg_cnt, try_device, FALSE))) {
      return found;
    }
  }

  if ((found = find_best_generic(gnr, list, arg_cnt, 0, TRUE))) {
    return found;
  }

  /* search HASH list for intrinsic generic of the same name */
  for (sptr = gnr; sptr; sptr = HASHLKG(sptr)) {
    if (NMPTRG(sptr) == nmptr && IS_INTRINSIC(STYPEG(sptr)) &&
        sptr < stb.firstosym) {
      return sptr;
    }
  }
  if (STYPEG(gnr) == ST_ENTRY || STYPEG(gnr) == ST_PROC) {
    /* allow specific name to be used also */
    return gnr;
  }
  if (CLASSG(gnr)) {
    char *name_cpy, *name;
    name_cpy = getitem(0, strlen(SYMNAME(gnr)) + 1);
    strcpy(name_cpy, SYMNAME(gnr));
    name = strchr(name_cpy, '$');
    if (name)
      *name = '\0';
    E155("Could not resolve generic type bound procedure", name_cpy);
  }
  if (GTYPEG(gnr)) {
    ACL *aclp, *hd, *tl;
    /*
     * build the ACL list from the list of arguments
     */
    hd = tl = NULL;
    for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
      sp = itemp->t.stkp;
      if (SST_IDG(sp) == S_ACONST || SST_IDG(sp) == S_SCONST) {
        aclp = SST_ACLG(sp);
      } else {
        /* put in ACL */
        aclp = GET_ACL(15);
        aclp->id = AC_EXPR;
        aclp->repeatc = aclp->size = 0;
        aclp->next = NULL;
        aclp->subc = NULL;
        aclp->u1.stkp = sp;
      }
      if (!hd) {
        hd = aclp;
      } else {
        tl->next = aclp;
      }
      tl = aclp;
    }
    sptr = GTYPEG(gnr);
    /* create head AC_SCONST for element list */
    aclp = GET_ACL(15);
    aclp->id = AC_SCONST;
    aclp->next = NULL;
    aclp->subc = hd;
    aclp->dtype = DTYPEG(sptr);
    SST_IDP(stktop, S_SCONST);
    SST_DTYPEP(stktop, aclp->dtype);
    SST_ACLP(stktop, aclp);
    chk_struct_constructor(aclp);
    SST_SYMP(stktop, sptr);
    return -1; /* generic resolves to a struct constructor */
  }
  if (CLASSG(gnr)) {
    char *name_cpy, *name;
    name_cpy = getitem(0, strlen(SYMNAME(gnr)) + 1);
    strcpy(name_cpy, SYMNAME(gnr));
    name = strchr(name_cpy, '$');
    if (name)
      *name = '\0';
    E155("Could not resolve generic type bound procedure", name_cpy);
  } else
    E155("Could not resolve generic procedure", SYMNAME(gnr));
  return 0;
}

/*
 * check if arguments passed to a generic match the arguments of the given
 * specific.
 */
static long *
args_match(int ext, int count, int distance_sz, ITEM *list, LOGICAL elemental,
           LOGICAL usedevcopy)
{
  int dscptr;
  int paramct;
  int actual_cnt;
  int i;
  char *kwd_str; /* where keyword string for 'ext' is stored */
  long arg_distance;
  long *distance;

  NEW(distance, long, distance_sz);

  dscptr = DPDSCG(ext);
  paramct = PARAMCTG(ext);

  if (count == 0 && paramct == 0)
    return set_distance_to(MIN_DISTANCE, distance, distance_sz);
  if (count > paramct)
    return set_distance_to(INF_DISTANCE, distance, distance_sz);
  kwd_str = make_kwd_str(ext);
  if (!kwd_match(list, paramct, kwd_str)) {
    FREE(kwd_str);
    return set_distance_to(INF_DISTANCE, distance, distance_sz);
  }
  FREE(kwd_str);

  (void)set_distance_to(MIN_DISTANCE, distance, distance_sz);
  for (i = 0, actual_cnt = 0; i < paramct && actual_cnt < count;
       i++, dscptr++) {
    SST *sp;
    int dum;
    int actual;
    int arg;
    sp = ARG_STK(i);
    if (sp) {
      (void)chkarg(sp, &dum);
      XFR_ARGAST(i);
    }
    actual = ARG_AST(i);
    arg = *(aux.dpdsc_base + dscptr);
    if (arg) {
      if (actual) {
        actual_cnt++;
        arg_distance = tkr_match(arg, sp, actual, elemental);
        if (arg_distance == INF_DISTANCE) {
          return set_distance_to(INF_DISTANCE, distance, distance_sz);
        } else {
          DISTANCE_ELM(distance, i) =
              (DISTANCE_ELM(distance, i) << UNIT_SZ) + arg_distance;
        }
      } else {
        DISTANCE_ELM(distance, i) =
            (DISTANCE_ELM(distance, i) << UNIT_SZ) + MIN_DISTANCE;
      }
    } else if (actual == 0 || A_TYPEG(actual) != A_LABEL) {
      /* alternate returns */
      return set_distance_to(INF_DISTANCE, distance, distance_sz);
    }
  }

  return distance;
}

/* Check TYPE-KIND-RANK */
static int
tkr_match(int formal, SST *opnd, int actual, int elemental)
{
  int ddum, dact, elddum, eldact;
  int rank;
  int sptr;
  int mng_match;
  LOGICAL formal_assumesz = FALSE;

  if (!ignore_tkr(formal, IGNORE_M) && ast_is_sym(actual)) {
    sptr = memsym_of_ast(actual);
    if ( (ALLOCATTRG(formal) && !ALLOCATTRG(sptr)) ||
         (POINTERG(formal) && !POINTERG(sptr)) ) {
      return INF_DISTANCE;
    }
  }

  mng_match = 0;
  ddum = DTYPEG(formal);
  elddum = DDTG(ddum);
  get_type_rank(opnd, &dact, &rank);
  eldact = DDTG(dact);
  if (elemental) {
    dact = eldact;
    rank = 0;
  }
  if (STYPEG(formal) == ST_PROC) {
    if (actual == 0)
      return INF_DISTANCE;
    /* actual must be an ID that is another PROC or ENTRY */
    if (A_TYPEG(actual) != A_ID)
      return INF_DISTANCE;
    sptr = A_SPTRG(actual);
    if (STYPEG(sptr) != ST_PROC && STYPEG(sptr) != ST_ENTRY &&
        !IS_INTRINSIC(STYPEG(sptr)))
      return INF_DISTANCE;
  } else if (A_TYPEG(actual) == A_ID && (STYPEG(A_SPTRG(actual)) == ST_PROC || 
             STYPEG(A_SPTRG(actual)) == ST_ENTRY) && 
             !IS_INTRINSIC(STYPEG(A_SPTRG(actual)))) {
        /* formal is not an ST_PROC, so return INF_DISTANCE */
        return INF_DISTANCE;
  }
  if (!ignore_tkr(formal, IGNORE_R)) {
    if (DTY(ddum) == TY_ARRAY) {
      if (AD_NUMDIM(AD_DPTR(ddum)) != rank) {
        if (rank && AD_ASSUMSZ(AD_DPTR(ddum)) &&
            AD_NUMDIM(AD_DPTR(ddum)) == 1) {
          formal_assumesz = TRUE;
        } else {
          return INF_DISTANCE;
        }
      }
    } else /* formal is not an array */
        if (rank)
      return INF_DISTANCE;
  }

  if (STYPEG(formal) == ST_PROC) {
    if (IS_INTRINSIC(STYPEG(sptr))) {
      setimplicit(sptr);
      dact = DTYPEG(sptr);
      /* TBD: should EXPST be set??? */
    }
    if (ddum == 0) {
      /* formal has no datatype; was the actual really typed? */
      if (DCLDG(sptr) && DTYPEG(sptr)) /* actual was given a datatype */
        return INF_DISTANCE;
      return EXACT_MATCH + mng_match;
    }
    if (dact == 0) {
      /* actual has no datatype; was the formal explicitly typed? */
      if (DCLDG(formal) && DTYPEG(formal)) /* formal was declared */
        return INF_DISTANCE;
      return EXACT_MATCH + mng_match;
    }
    if (!DCLDG(formal) && !FUNCG(formal) && !DCLDG(sptr) && !FUNCG(sptr))
      /* formal & actual are subroutines?? */
      return EXACT_MATCH + mng_match;
  }

  /* check if type and kind of the data types match */
  if (DTY(elddum) != DTY(eldact)) {
    /* element TY_ values are not the same */
    if (ignore_tkr(formal, IGNORE_K)) {
      if (same_type_different_kind(elddum, eldact))
        return EXACT_MATCH + mng_match;
    } else if (ignore_tkr(formal, IGNORE_T) &&
               different_type_same_kind(elddum, eldact))
      return EXACT_MATCH + mng_match;
  }
  if (ignore_tkr(formal, IGNORE_T)) {
    if (ignore_tkr(formal, IGNORE_K))
      return EXACT_MATCH + mng_match;
    /* cannot ignore the kind, so it must be the same! */
    if (different_type_same_kind(elddum, eldact))
      return EXACT_MATCH + mng_match;
  }

  /* check for an exact match first */
  if (tk_match_arg(ddum, dact, FALSE)) {
    return formal_assumesz ? EXTND_MATCH + mng_match : EXACT_MATCH + mng_match;
  } else if (tk_match_arg(ddum, dact, CLASSG(formal))) {
    return EXTND_MATCH + mng_match;
  } else if (DTY(elddum) == TY_DERIVED && UNLPOLYG(DTY(elddum + 3))) {
    /* Dummy argument is declared CLASS(*), so it can
     * take any rank compatible actual argument.
     */
    return formal_assumesz ? EXTND_MATCH + mng_match : EXACT_MATCH + mng_match;
  }
  return INF_DISTANCE;
}

static LOGICAL
kwd_match(ITEM *list,  /* list of arguments */
          int cnt,     /* maximum number of arguments allowed for intrinsic */
          char *kwdarg /* string defining position and keywords of arguments*/
          )
{
  SST *stkp;
  int pos;
  int i;
  char *kwd, *np;
  int kwd_len;
  char *actual_kwd; /* name of keyword used with the actual arg */
  int actual_kwd_len;
  LOGICAL kwd_present;

  /*
   * NOTE:  'variable' arguments (see get_kwd_args in semfunc2.c)
   *        will not be seen for user-defined interfaces.
   */

  kwd_present = FALSE;
  sem.argpos = (argpos_t *)getitem(0, sizeof(argpos_t) * cnt);

  for (i = 0; i < cnt; i++) {
    ARG_STK(i) = NULL;
    ARG_AST(i) = 0;
  }

  for (pos = 0; list != ITEM_END; list = list->next, pos++) {
    stkp = list->t.stkp;
    if (SST_IDG(stkp) == S_KEYWORD) {
      kwd_present = TRUE;
      actual_kwd = scn.id.name + SST_CVALG(stkp);
      actual_kwd_len = strlen(actual_kwd);
      kwd = kwdarg;
      for (i = 0; TRUE; i++) {
#if DEBUG
        assert(*kwd != '#', "kwd_match, unexp. #", pos, 3);
#endif
        if (*kwd == '*')
          kwd++;
        kwd_len = 0;
        for (np = kwd; TRUE; np++, kwd_len++)
          if (*np == ' ' || *np == '\0')
            break;
        if (kwd_len == actual_kwd_len &&
            strncmp(kwd, actual_kwd, actual_kwd_len) == 0)
          break;
        if (*np == '\0')
          return FALSE;
        kwd = np + 1; /* skip over blank */
      }
      if (ARG_STK(i))
        return FALSE;
      stkp = SST_E3G(stkp);
      ARG_STK(i) = stkp;
      ARG_AST(i) = SST_ASTG(stkp);
    } else {
      if (ARG_STK(pos)) {
        kwd = kwdarg;
        for (i = 0; TRUE; i++) {
          if (*kwd == '*' || *kwd == ' ')
            kwd++;
          if (*kwd == '\0')
            return FALSE;
          kwd_len = 0;
          for (np = kwd; TRUE; np++) {
            if (*np == ' ' || *np == '\0')
              break;
            kwd_len++;
          }
          if (i == pos)
            break;
          kwd = np;
        }
        return FALSE;
      }
      ARG_STK(pos) = stkp;
      ARG_AST(pos) = SST_ASTG(stkp);
    }
  }

  /* determine if required argument is not present */

  kwd = kwdarg;
  for (pos = 0; pos < cnt; pos++, kwd = np) {
    if (*kwd == ' ')
      kwd++;
    if (*kwd == '#' || *kwd == '!')
      break;
    kwd_len = 0;
    for (np = kwd; TRUE; np++) {
      if (*np == ' ' || *np == '\0')
        break;
      kwd_len++;
    }
    if (*kwd == '*')
      continue;
    if (ARG_STK(pos) == NULL)
      return FALSE;
  }

  return TRUE;
}

int
defined_operator(int opr, SST *stktop, SST *lop, SST *rop)
{
  int sptr;
  ITEM *list;
  int i;

#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "user operator %s\n", SYMNAME(opr));
#endif
  if (queue_generic_tbp_once(0))
    queue_tbp(0, 0, 0, 0, TBP_COMPLETE_GENERIC);
  if (STYPEG(opr) != ST_OPERATOR) {
    i = findByNameStypeScope(SYMNAME(opr), ST_OPERATOR, stb.curr_scope);
    if (i) {
      opr = i;
    }
  }
  sptr = resolve_operator(opr, lop, rop);
  if (sptr == 0) {
    SST_IDP(stktop, S_CONST);
    SST_DTYPEP(stktop, DT_INT);
    return 1;
  }
#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "user operator resolved to %s\n", SYMNAME(sptr));
#endif

  list = make_list(lop, rop);
  mkident(stktop);
  SST_SYMP(stktop, sptr);
  SST_DTYPEP(stktop, DTYPEG(sptr));
  SST_ASTP(stktop, mk_id(sptr));
  return func_call2(stktop, list, 1);
}

static int
resolve_operator(int opr, SST *lop, SST *rop)
{
  int func;
#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "resolve_operator: %s, count %d\n", SYMNAME(opr),
            rop == NULL ? 1 : 2);
#endif
  func = find_operator(opr, lop, rop, FALSE);
  if (func != 0) {
    return func;
  }
  /* Redo the search, this time allow type matching for elemental subprograms */
  func = find_operator(opr, lop, rop, TRUE);
  if (func != 0) {
    return func;
  }

  /* Overloading did not occur; issue error message only if this is not
   * an intrinsic operator.
   */
  if (INKINDG(opr) == 0) {
    if (GNCNTG(opr) == 0) {
      E155("Empty operator -", SYMNAME(opr));
    } else {
      E155("Could not resolve operator", SYMNAME(opr));
    }
  }
  return 0;
}

static int
find_operator(int opr, SST *lop, SST *rop, LOGICAL elemental)
{
  int sptr;
  int opnd_cnt = rop == NULL ? 1 : 2;
  int nmptr = NMPTRG(opr);
  for (sptr = first_hash(opr); sptr; sptr = HASHLKG(sptr)) {
    int gndsc;
    int sptrgen = sptr;
    if (NMPTRG(sptrgen) != nmptr)
      continue;
    if (STYPEG(sptrgen) == ST_ALIAS)
      sptrgen = SYMLKG(sptrgen);
    if (STYPEG(sptrgen) != ST_OPERATOR)
      continue;
    /* is the ST_OPERATOR or ST_ALIAS in an active scope */
    if (test_scope(sptr) < 0 && !CLASSG(sptrgen))
      continue;

    for (gndsc = GNDSCG(sptrgen); gndsc; gndsc = SYMI_NEXT(gndsc)) {
      int dscptr;
      int paramct;
      int bind;
      int func = SYMI_SPTR(gndsc);
      if (IS_TBP(func)) {
        /* For generic type bound procedures, use the implementation
         * of the generic bind name for the argument comparison.
         */
        int mem, dty;
        bind = func;
        dty = TBPLNKG(func);
        func = get_implementation(dty, func, 0, &mem);
        if (STYPEG(BINDG(mem)) == ST_OPERATOR ||
            STYPEG(BINDG(mem)) == ST_USERGENERIC) {
          mem = get_specific_member(dty, func);
          func = VTABLEG(mem);
          bind = BINDG(mem);
        }
        if (!func)
          continue;
        mem = get_generic_member(dty, bind);
        if (mem && PRIVATEG(mem) && SCOPEG(stb.curr_scope) != SCOPEG(mem))
          continue;
      } else {
        bind = 0;
      }
      if (STYPEG(func) == ST_MODPROC) {
        func = SYMLKG(func);
        if (func == 0)
          continue;
      }
      if (STYPEG(func) == ST_ALIAS)
        func = SYMLKG(func);
      paramct = PARAMCTG(func);

      if (paramct != opnd_cnt) {
        if (!bind) {
          continue;
        } else {
          dscptr = DPDSCG(func);
          if (paramct == 2 && opnd_cnt == 1) {
            int arg = *(aux.dpdsc_base + dscptr + 1);
            if (!CCSYMG(arg) || !CLASSG(arg))
              continue;
          } else if (paramct == 4 && opnd_cnt == 2) {
            int arg = *(aux.dpdsc_base + dscptr + 2);
            if (!CCSYMG(arg) || !CLASSG(arg))
              continue;
            arg = *(aux.dpdsc_base + dscptr + 3);
            if (!CCSYMG(arg) || !CLASSG(arg))
              continue;
          } else {
            continue;
          }
        }
      }
      dscptr = DPDSCG(func);
      if (!elemental || ELEMENTALG(func)) {
        int arg = *(aux.dpdsc_base + dscptr);
        if (arg && (tkr_match(arg, lop, 0, elemental) == INF_DISTANCE))
          continue;
        if (rop != NULL) {
          int arg = *(aux.dpdsc_base + dscptr + 1);
          if (arg && (tkr_match(arg, rop, 0, elemental) == INF_DISTANCE))
            continue;
        }
        return bind ? bind : func;
      }
    }
  }
  return 0; // not found
}

void
init_intrinsic_opr(void)
{
  int i;

  for (i = 0; i <= OP_VAL; i++)
    optab[i].opr = 0;
}

void
bind_intrinsic_opr(int val, int opr)
{
  optab[val].opr = opr;
  INKINDP(opr, 1);  /* intrinsic or assignment operator */
  PDNUMP(opr, val); /* OP_... value */
}

static int
tkn_alias_sym(int tkn_alias)
{
  int sym;
  switch (tkn_alias) {
  case TK_XORX:
    sym = getsymbol("x");
    break;
  case TK_XOR:
    sym = getsymbol("xor");
    break;
  case TK_ORX:
    sym = getsymbol("o");
    break;
  case TK_NOTX:
    sym = getsymbol("n");
    break;
  default:
    interr("tkn_alias_sym: no token", 0, 3);
    sym = getsymbol("..zz");
  }
  return sym;
}

int
get_intrinsic_oprsym(int val, int tkn_alias)
{
  int sym;
  if (!tkn_alias)
    sym = getsymbol(optab[val].name);
  else
    sym = tkn_alias_sym(tkn_alias);
  return sym;
}

int
get_intrinsic_opr(int val, int tkn_alias)
{
  int opr;
  opr = get_intrinsic_oprsym(val, tkn_alias);
  opr = declsym(opr, ST_OPERATOR, FALSE);
  bind_intrinsic_opr(val, opr);

  return opr;
}

LOGICAL
is_intrinsic_opr(int val, SST *stktop, SST *lop, SST *rop, int tkn_alias)
{
  /*  tkn_alias is currently not referenced */
  int opr;
  int func;
  ITEM *list;
  int rank, dtype;
  char buf[100];

  opr = optab[val].opr;
  if (opr) {
    func = resolve_operator(opr, lop, rop);
    if (!func && /*IN_MODULE*/ sem.mod_cnt && sem.which_pass) {
      if (queue_generic_tbp_once(0))
        queue_tbp(0, 0, 0, 0, TBP_COMPLETE_GENERIC);
      func = resolve_operator(opr, lop, rop);
    }
    if (CLASSG(func) && IS_TBP(func)) {
      int ast, mem, inv;
      get_implementation(TBPLNKG(func), func, 0, &mem);
      if (NOPASSG(mem)) {
        if (val != OP_ST) {
          E155("Type bound procedure with NOPASS attribute not valid "
               "for generic operator",
               SYMNAME(opr));
        } else {
          E155("Type bound procedure with NOPASS attribute not valid "
               "for generic assignment",
               SYMNAME(opr));
        }
        inv = 0;
      } else {
        inv = get_tbp_argno(func, TBPLNKG(func));
      }
      if (inv < 1 || inv > 2) {
        if (val != OP_ST) {
          E155("Invalid type bound procedure in generic set "
               "for generic operator",
               SYMNAME(opr));
        } else {
          E155("Invalid type bound procedure in generic set "
               "for generic assignment",
               SYMNAME(opr));
        }
        inv = 0;
      }
      list = make_list(lop, rop);
      if (rop != NULL && (inv == 1 || inv == 2)) {
        if (SST_IDG(rop) == S_SCONST) {
          /* Support operator look up with structure
           * constructor argument on RHS.
           */
          int tmp = getccsym_sc('d', sem.dtemps++, ST_VAR, SC_LOCAL);
          DTYPEP(tmp, SST_DTYPEG(rop));
          ast = mk_id(tmp);
        } else if (inv == 1) {
          mkexpr(lop);
          ast = SST_ASTG(lop);
          if (A_TYPEG(ast) == A_INTR) {
            mkexpr(rop);
            ast = SST_ASTG(rop);
          }

        } else {
          mkexpr(rop);
          ast = SST_ASTG(rop);
          if (A_TYPEG(ast) == A_INTR) {
            mkexpr(lop);
            ast = SST_ASTG(lop);
          }
        }
      } else {
        mkexpr(lop);
        ast = SST_ASTG(lop);
      }
      ast = mkmember(TBPLNKG(func), ast, NMPTRG(mem));
      SST_ASTP(stktop, ast);
      SST_SYMP(stktop, -func);
      if (val == OP_ST)
        subr_call2(stktop, list, 1);
      else
        func_call2(stktop, list, 1);
      return TRUE;
    }
    if (func != 0) {
#if DEBUG
      if (DBGBIT(3, 256))
        fprintf(gbl.dbgfil, "intrinsic operator resolved to %s\n",
                SYMNAME(func));
#endif
      list = make_list(lop, rop);
      mkident(stktop);
      SST_SYMP(stktop, -func);
      if (val == OP_ST)
        subr_call2(stktop, list, 1);
      else {
        SST_ASTP(stktop, mk_id(func));
        SST_DTYPEP(stktop, DTYPEG(func));
        func_call2(stktop, list, 1);
      }
      return TRUE;
    }
  }

  /* Check for illegal use of an operator on a derived type. */
  if (val == OP_ST) /* Assignment is ok. */
    return FALSE;
  get_type_rank(lop, &dtype, &rank);
  if (DTYG(dtype) == TY_DERIVED) {
    /*
     * (reference f20848) - long ago, semgnr.c spelled .ne. as "!=".
     * As a consequence, operator(/=) would show up as != in the
     * symbol table and propagated to .mod files, such as
     * iso_c_binding.  Fixing semgnr means that we will fail to
     * process '!=' from mod files; interf.c needs to change '!=' to '/=';
     * and the mod file version needs to incremented.  SO, just hack the
     * error message when appropriate.
     */
    if (strcmp(optab[val].name, "!="))
      sprintf(buf, "operator %s on a derived type", optab[val].name);
    else
      sprintf(buf, "operator %s on a derived type", "/=");
    error(99, 3, gbl.lineno, buf, CNULL);
  } else if (rop != NULL) {
    get_type_rank(rop, &dtype, &rank);
    if (DTYG(dtype) == TY_DERIVED) {
      if (strcmp(optab[val].name, "!="))
        sprintf(buf, "operator %s on a derived type", optab[val].name);
      else
        sprintf(buf, "operator %s on a derived type", "/=");
      error(99, 3, gbl.lineno, buf, CNULL);
    }
  }
  return FALSE;
}

static void
get_type_rank(SST *stkptr, int *dt_p, int *rank_p)
{
  int dtype;
  int sptr;
  int shape;

  dtype = 0;
  shape = 0;
  switch (SST_IDG(stkptr)) {
  case S_IDENT:
    sptr = SST_SYMG(stkptr);
    switch (STYPEG(sptr)) {
    case ST_INTRIN:
    case ST_GENERIC:
    case ST_PD:
      if (!EXPSTG(sptr)) {
        /* Not a frozen intrinsic, so assume its a variable */
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        /* need storage class (local) */
        sem_set_storage_class(sptr);
        SST_SYMP(stkptr, sptr);
        dtype = DTYPEG(sptr);
      }
      break;
    case ST_UNKNOWN:
    case ST_IDENT:
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_ENTRY:
    case ST_USERGENERIC:
    case ST_PROC:
      dtype = DTYPEG(sptr);
      break;
    default:
      break;
    }
    break;
  case S_LVALUE:
  case S_LOGEXPR:
  case S_EXPR:
    dtype = SST_DTYPEG(stkptr);
    shape = SST_SHAPEG(stkptr);
    break;
  case S_CONST:
  case S_SCONST:
  case S_ACONST:
    dtype = SST_DTYPEG(stkptr);
    break;
  case S_STFUNC:
  case S_DERIVED:
    dtype = DTYPEG(SST_SYMG(stkptr));
    break;
  default:
    break;
  }

  *dt_p = dtype;
  *rank_p = 0;

  if (dtype) {
    if (shape)
      *rank_p = SHD_NDIM(shape);
    else if (DTY(dtype) == TY_ARRAY)
      *rank_p = AD_NUMDIM(AD_DPTR(dtype));
  }

}

static ITEM *
make_list(SST *lop, SST *rop)
{
  ITEM *list;

  list = (ITEM *)getitem(0, sizeof(ITEM));
  list->t.stkp = (SST *)getitem(0, sizeof(SST));
  *list->t.stkp = *lop;

  if (rop != NULL) {
    ITEM *tmp;
    tmp = (ITEM *)getitem(0, sizeof(ITEM));
    tmp->t.stkp = (SST *)getitem(0, sizeof(SST));
    *tmp->t.stkp = *rop;
    list->next = tmp;
    tmp->next = ITEM_END;
  } else
    list->next = ITEM_END;

  return list;
}

void rw_gnr_state(RW_ROUTINE, RW_FILE)
{
  int nw;
  RW_FD(optab, struct optabstruct, OPTABSIZE);
} /* rw_gnr_state */

static void
defined_io_error(const char *proc, int is_unformatted, const char *msg, int func)
{

  char *buf;

  buf = getitem(0, strlen("for defined WRITE(UNFORMATTED), in subroutine") +
                       strlen(msg) + 1);
  sprintf(buf, "for defined %s(%s), %s in subroutine",
          (strcmp(proc, ".read") == 0) ? "READ" : "WRITE",
          (is_unformatted) ? "UNFORMATTED" : "FORMATTED", msg);

  error(155, 3, gbl.lineno, buf, SYMNAME(func));
}

static void
check_defined_io2(const char *proc, int silentmode, int chk_dtype)
{
  int gnr, sptr, sptrgen;
  LOGICAL gnr_in_active_scope;
  int gndsc, nmptr;
  int func, paramct, dpdsc, iface;
  int psptr, dtype, tag;
  int is_unformatted, func2;
  int seen_error, dtv_dtype;
  int extensible, found;
  int bind, dt_int;
  int second_arg_error;

  if (!proc)
    return;
  if (XBIT(124, 0x10)) {
    dt_int = DT_INT8; /* -i8 */
  } else {
    dt_int = DT_INT;
  }
  if (chk_dtype) {
    if (DTY(chk_dtype) == TY_ARRAY)
      chk_dtype = DTY(chk_dtype + 1);
    if (DTY(chk_dtype) != TY_DERIVED)
      return;
  }
  gnr = getsymbol(proc);
  found = 0;
  if (STYPEG(gnr) == ST_USERGENERIC) {
    gnr_in_active_scope = FALSE;
    nmptr = NMPTRG(gnr);
    for (sptr = first_hash(gnr); sptr > NOSYM; sptr = HASHLKG(sptr)) {
      sptrgen = sptr;
      second_arg_error = seen_error = 0;
      dtv_dtype = 0;
      extensible = 0;
      if (NMPTRG(sptrgen) != nmptr)
        continue;
      if (STYPEG(sptrgen) == ST_ALIAS)
        sptrgen = SYMLKG(sptrgen);
      if (STYPEG(sptrgen) != ST_USERGENERIC)
        continue;
      /* is the original symbol (sptr, not sptrgen) in an active scope */
      if (test_scope(sptr)) {
        gnr_in_active_scope = TRUE;
      }
      if (!gnr_in_active_scope && !CLASSG(sptrgen))
        continue;
      if (GNCNTG(sptrgen) == 0 && GTYPEG(sptrgen))
        continue;
      if (queue_generic_tbp_once(sptrgen)) {
        queue_tbp(0, 0, 0, 0, TBP_COMPLETE_GENERIC);
      }

      for (gndsc = GNDSCG(sptrgen); gndsc; gndsc = SYMI_NEXT(gndsc)) {
        func = SYMI_SPTR(gndsc);
        is_unformatted = 0;

        if (IS_TBP(func)) {
          /* For generic type bound procedures, use the implementation
           * of the generic bind name for the argument comparison.
           */
          int mem, dty;
          bind = func;
          dty = TBPLNKG(func);

          func = get_implementation(dty, func, 0, &mem);
          if (STYPEG(BINDG(mem)) == ST_OPERATOR ||
              STYPEG(BINDG(mem)) == ST_USERGENERIC) {
            mem = get_specific_member(dty, func);
            func = VTABLEG(mem);
            bind = BINDG(mem);
          }
          if (!func)
            continue;
          mem = get_generic_member(dty, bind);
          if (NOPASSG(mem) && generic_tbp_has_pass_and_nopass(dty, BINDG(mem)))
            continue;
          if (mem && PRIVATEG(mem) && SCOPEG(stb.curr_scope) != SCOPEG(mem))
            continue;
        } else
          bind = 0;

        for (func2 = (!bind) ? first_hash(func) : first_hash(bind);
             func2 > NOSYM; func2 = HASHLKG(func2)) {
          if (!test_scope(func2))
            continue;
          if (UNFMTG(func2)) {
            is_unformatted = 1;
            break;
          }
        }
        if (FVALG(func)) {
          seen_error++;
          if (!silentmode) {
            if (is_unformatted) {
              if (strcmp(proc, ".read") == 0) {
                error(155, 3, gbl.lineno,
                      "The generic set for a defined"
                      "READ(UNFORMATTED) contains non-subroutine",
                      SYMNAME(func));
              } else {
                error(155, 3, gbl.lineno,
                      "The generic set for a defined"
                      "WRITE(UNFORMATTED) contains non-subroutine",
                      SYMNAME(func));
              }
            } else {
              if (strcmp(proc, ".read") == 0) {
                error(155, 3, gbl.lineno,
                      "The generic set for a defined"
                      "READ(FORMATTED) contains non-subroutine",
                      SYMNAME(func));
              } else {
                error(155, 3, gbl.lineno,
                      "The generic set for a defined"
                      "WRITE(FORMATTED) contains non-subroutine",
                      SYMNAME(func));
              }
            }
          }
          continue;
        }
        paramct = dpdsc = iface = 0;
        if (STYPEG(func) == ST_MODPROC) {
          func = SYMLKG(func);
          if (func <= NOSYM)
            continue;
        }
        if (STYPEG(func) == ST_ALIAS) {
          func = SYMLKG(func);
          if (func <= NOSYM)
            continue;
        }
        if (STYPEG(func) != ST_PROC && STYPEG(func) != ST_ENTRY)
          continue;

        proc_arginfo(func, &paramct, &dpdsc, &iface);
        if (!dpdsc)
          continue;

        if (paramct > 4) {
          psptr = *(aux.dpdsc_base + dpdsc + (paramct - 1));
          if (CLASSG(psptr) && CCSYMG(psptr)) {
            --paramct; /* don't count type descriptor arg */
          }
        }

        if (is_unformatted && paramct == 4) {
          psptr = *(aux.dpdsc_base + dpdsc);
          dtype = DTYPEG(psptr);
          if (DTY(dtype) == TY_ARRAY)
            dtype = DTY(dtype + 1);
          if (DTY(dtype) != TY_DERIVED) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "first argument must be a derived type", func);
            continue;
          }
          dtv_dtype = dtype;
          tag = DTY(dtype + 3);
          if (!CLASSG(psptr) && !CFUNCG(tag) && !SEQG(tag)) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "first argument with extensible type"
                               " must be declared CLASS",
                               func);
          }
          if (CLASSG(psptr) && !CFUNCG(tag) && !SEQG(tag)) {
            extensible = 1;
          }
          if (!all_len_parms_assumed(dtype)) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "all length type parameters must be assumed"
                               " for derived type argument 1",
                               func);
          }
          if (INTENTG(psptr) != INTENT_INOUT && INTENTG(psptr) != INTENT_IN) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "first argument must be declared INTENT(IN)"
                               " or INTENT(INOUT)",
                               func);
          }

          psptr = *(aux.dpdsc_base + dpdsc + 1);
          dtype = DTYPEG(psptr);
          if (DT_ISINT(dtype)) {
            dt_int = dtype;
          }
          if (dtype != dt_int) {
            seen_error++;
            second_arg_error = 1;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "second argument must be declared INTEGER",
                               func);
          }
          if (INTENTG(psptr) != INTENT_IN) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "second argument must be declared"
                               " INTENT(IN)",
                               func);
          }
          psptr = *(aux.dpdsc_base + dpdsc + 2);
          dtype = DTYPEG(psptr);
          if (dtype != dt_int) {
            seen_error++;
            if (!silentmode) {
              if (second_arg_error) {
                defined_io_error(proc, is_unformatted,
                                 "third argument must be declared INTEGER",
                                 func);
              } else {
                defined_io_error(proc, is_unformatted,
                                 "second and third argument must be declared "
                                 "INTEGER",
                                 func);
              }
            }
          }
          if (INTENTG(psptr) != INTENT_OUT) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "third argument must be declared "
                               "INTENT(INOUT)",
                               func);
          }
          psptr = *(aux.dpdsc_base + dpdsc + 3);
          dtype = DTYPEG(psptr);
          if (dtype != DT_ASSCHAR) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "fourth argument must be declared "
                               "CHARACTER(LEN=*)",
                               func);
          }
          if (INTENTG(psptr) != INTENT_INOUT) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "fourth argument must be declared "
                               "INTENT(INOUT)",
                               func);
          }
          if (!seen_error) {
            /* set UFIO flag on the tag */
            if (strcmp(proc, ".read") == 0) {
              UFIOP(tag, (DT_IO_UREAD | UFIOG(tag)));
            } else {
              UFIOP(tag, (DT_IO_UWRITE | UFIOG(tag)));
            }
            if (chk_dtype && eq_dtype2(dtv_dtype, chk_dtype, extensible)) {
              int tag2;
              tag2 = DTY(chk_dtype + 3);
              found++;
              if (strcmp(proc, ".read") == 0) {
                UFIOP(tag2, (DT_IO_UREAD | UFIOG(tag2)));
              } else {
                UFIOP(tag2, (DT_IO_UWRITE | UFIOG(tag2)));
              }
              UFIOP(tag2, (UFIOG(tag2) & ~(DT_IO_NONE)));
            }
          }
        } else if (!is_unformatted && paramct == 6) {
          psptr = *(aux.dpdsc_base + dpdsc);
          dtype = DTYPEG(psptr);
          if (DTY(dtype) == TY_ARRAY)
            dtype = DTY(dtype + 1);
          if (DTY(dtype) != TY_DERIVED) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "first argument must be a derived type", func);
            continue;
          }
          dtv_dtype = dtype;
          tag = DTY(dtype + 3);
          if (!CLASSG(psptr) && !CFUNCG(tag) && !SEQG(tag)) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "first argument with extensible type"
                               " must be declared CLASS",
                               func);
          }
          if (CLASSG(psptr) && !CFUNCG(tag) && !SEQG(tag)) {
            extensible = 1;
          }
          if (!all_len_parms_assumed(dtype)) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "all length type parameters must be assumed"
                               " for derived type argument 1",
                               func);
          }
          if (INTENTG(psptr) != INTENT_INOUT && INTENTG(psptr) != INTENT_IN) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "first argument must be declared INTENT(IN)"
                               " or INTENT(INOUT)",
                               func);
          }

          psptr = *(aux.dpdsc_base + dpdsc + 1);
          dtype = DTYPEG(psptr);
          if (DT_ISINT(dtype)) {
            dt_int = dtype;
          }
          if (dtype != dt_int) {
            seen_error++;
            second_arg_error = 1;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "second argument must be declared INTEGER",
                               func);
          }
          if (INTENTG(psptr) != INTENT_IN) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "second argument must be declared"
                               " INTENT(IN)",
                               func);
          }
          psptr = *(aux.dpdsc_base + dpdsc + 2);
          dtype = DTYPEG(psptr);
          if (dtype != DT_ASSCHAR) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "third argument must be declared "
                               "CHARACTER(LEN=*)",
                               func);
          }
          if (INTENTG(psptr) != INTENT_IN) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "third argument must be declared INTENT(IN)",
                               func);
          }
          psptr = *(aux.dpdsc_base + dpdsc + 3);
          dtype = DTYPEG(psptr);
          if (DTY(dtype) != TY_ARRAY || DTY(dtype + 1) != dt_int ||
              !ASSUMSHPG(psptr) || rank_of_sym(psptr) != 1) {
            seen_error++;
            if (!silentmode) {
              if (!second_arg_error) {
                defined_io_error(proc, is_unformatted,
                                 "second argument must be declared INTEGER",
                                 func);
              }
              defined_io_error(proc, is_unformatted,
                               "fourth argument must be a rank 1 assumed"
                               " shape array of type INTEGER",
                               func);
            }
          }
          if (INTENTG(psptr) != INTENT_IN) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "fourth argument must be declared INTENT(IN)",
                               func);
          }
          psptr = *(aux.dpdsc_base + dpdsc + 4);
          dtype = DTYPEG(psptr);
          if (dtype != dt_int) {
            seen_error++;
            if (!silentmode) {
              if (second_arg_error) {
                defined_io_error(proc, is_unformatted,
                                 "fifth argument must be declared INTEGER",
                                 func);
              } else {
                defined_io_error(proc, is_unformatted,
                                 "second and fifth argument must be declared "
                                 "INTEGER",
                                 func);
              }
            }
          }
          if (INTENTG(psptr) != INTENT_OUT) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "fifth argument must be declared "
                               "INTENT(OUT)",
                               func);
          }
          psptr = *(aux.dpdsc_base + dpdsc + 5);
          dtype = DTYPEG(psptr);
          if (dtype != DT_ASSCHAR) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "sixth argument must be declared "
                               "CHARACTER(LEN=*)",
                               func);
          }
          if (INTENTG(psptr) != INTENT_INOUT) {
            seen_error++;
            if (!silentmode)
              defined_io_error(proc, is_unformatted,
                               "sixth argument must be declared "
                               "INTENT(INOUT)",
                               func);
          }

          if (!seen_error) {
            /* set UFIO flag on the tag */
            if (strcmp(proc, ".read") == 0) {
              UFIOP(tag, (DT_IO_FREAD | UFIOG(tag)));
            } else {
              UFIOP(tag, (DT_IO_FWRITE | UFIOG(tag)));
            }
            if (chk_dtype && eq_dtype2(dtv_dtype, chk_dtype, extensible)) {
              int tag2;
              tag2 = DTY(chk_dtype + 3);
              found++;
              if (strcmp(proc, ".read") == 0) {
                UFIOP(tag2, (DT_IO_FREAD | UFIOG(tag2)));
              } else {
                UFIOP(tag2, (DT_IO_FWRITE | UFIOG(tag2)));
              }
              UFIOP(tag2, (UFIOG(tag2) & ~(DT_IO_NONE)));
            }
          }
        } else {
          seen_error++;
          if (!silentmode)
            defined_io_error(proc, is_unformatted, "invalid argument list",
                             func);
        }
      }
    }
  }
  if (!found && chk_dtype) {
    tag = DTY(chk_dtype + 3);
    if (!UFIOG(tag)) {
      UFIOP(tag, DT_IO_NONE);
    }
  }
}

/** \brief Return a bit mask indicating which I/O routines are defined for a
           derived type.
 */
int
dtype_has_defined_io(int dtype)
{
  int tag;

  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);
  if (DTY(dtype) != TY_DERIVED)
    return 0;

  tag = DTY(dtype + 3);

  if (!UFIOG(tag)) {
    check_defined_io2(".read", 1, dtype);
    check_defined_io2(".write", 1, dtype);
  }
  return UFIOG(tag);
}

void
check_defined_io(void)
{

  check_defined_io2(".write", 0, 0);
  check_defined_io2(".read", 0, 0);
}

/**
   \param read_or_write  0 specifies read, 1 specifies write
   \param stktop         SST we're processing.
   \param list           argument list for read/write
   \return
   <pre>
   = -1 : error (resolves to struct constructor -- should never happen)
   = 0  : error or no I/O subroutine
   \> 0  : sptr of the 'specific' defined I/O subroutine
   </pre>
 */
int
resolve_defined_io(int read_or_write, SST *stktop, ITEM *list)
{
  int i;
  int gnr = getsymbol(read_or_write ? ".write" : ".read");

  if (STYPEG(gnr) != ST_USERGENERIC) {
    return 0;
  }

  resolved_to_self = 0;
  silent_error_mode = 1;
  i = resolve_generic(gnr, stktop, list);
  silent_error_mode = 0;
  if (resolved_to_self) {
    if (i > NOSYM && !RECURG(gbl.currsub)) {
      error(155, 3, gbl.lineno,
            "Subroutines that participate in recursive"
            " defined I/O operations must be declared RECURSIVE -",
            SYMNAME(gbl.currsub));
    }
    resolved_to_self = 0;
  }
  return i;
}

void
add_overload(int gnr, int func)
{
  int gnidx;
  if (sem.defined_io_type == 2 || sem.defined_io_type == 4) {
    UNFMTP(func, 1);
  }
  gnidx = add_symitem(func, GNDSCG(gnr));
  GNDSCP(gnr, gnidx);
  GNCNTP(gnr, GNCNTG(gnr) + 1);
#if DEBUG
  if (DBGBIT(3, 256))
    fprintf(gbl.dbgfil, "overload %s --> %s, symi_base+%d\n", SYMNAME(gnr),
            SYMNAME(func), gnidx);
#endif
}

void
copy_specifics(int fromsptr, int tosptr)
{
  int symi_src;

  assert((STYPEG(fromsptr) == ST_OPERATOR || STYPEG(fromsptr) == ST_USERGENERIC) &&
         (STYPEG(tosptr) == ST_OPERATOR || STYPEG(tosptr) == ST_USERGENERIC),
         "copy_specifics src or dest not user generic or operator", 0, 3);

  for (symi_src = GNDSCG(fromsptr); symi_src; symi_src = SYMI_NEXT(symi_src)) {
    /* don't copy if the specific is already in the generic's list */
    /* TODO: is comparison of sptrs good enough or is comparison
     * of nmptr and signature necessary?
     */
    int src = SYMI_SPTR(symi_src);
    if (!sym_in_sym_list(src, GNDSCG(tosptr))) {
      add_overload(tosptr, src);
    }
  }
}
