/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Fortran Semantic action routines to resolve symbol references as to
           overloading class.

    This module hides the walking of hash chains and overloading class checks.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "gramtk.h"
#include "semant.h"
#include "ast.h"
#include "rte.h"
#include "interf.h"

static int find_in_host(int);
static void internref_bnd(int);
static int add_private_allocatable(int, int);
static void check_parref(int, int, int);

static LOGICAL checking_scope = FALSE;

static LOGICAL
isGenericOrProcOrModproc(SPTR sptr)
{
  SPTR localSptr = STYPEG(sptr) == ST_ALIAS ? SYMLKG(sptr) : sptr;
  switch (STYPEG(localSptr)) {
  case ST_PROC:
  case ST_MODPROC:
  case ST_USERGENERIC:
    return TRUE;
  default:
    return FALSE;
  }
}

static LOGICAL
isSameNameGenericOrProcOrModproc(SPTR sptr1, SPTR sptr2)
{
  if (GSAMEG(sptr2) && isGenericOrProcOrModproc(sptr1) &&
      isGenericOrProcOrModproc(sptr2)) {
    return NMPTRG(sptr1) == NMPTRG(GSAMEG(sptr2));
  }
  return FALSE;
}

static int
getEnclFunc(SPTR sptr)
{
  int currencl;
  int enclsptr;
  currencl = enclsptr = ENCLFUNCG(sptr);
  while (enclsptr && STYPEG(enclsptr) != ST_ENTRY) {
    currencl = enclsptr;
    enclsptr = ENCLFUNCG(enclsptr);
  }

  if (currencl)
    return SCOPEG(currencl);
  return 0;
}

static LOGICAL
isLocalPrivate(SPTR sptr)
{
  int scope = getEnclFunc(sptr);

  if (scope && STYPEG(scope) == ST_ENTRY && scope != gbl.currsub)
    return FALSE;

  /* have to return TRUE if ENCLFUNC nor SCOPE is set */
  return TRUE;
}

/** \brief Look for a symbol with same name as first and in an active scope.
    \param first              the symbol to match by name
    \param overloadclass      0 or the value of stb.ovclass to match
    \param paliassym          return the symbol the result is an alias of
    \param plevel             return the scope nest at which symbol was found
    \param multiple_use_error if true, report error if name is USE-associated
                              from two different modules
    \return symbol (or alias) if found, else 0
 */
int
sym_in_scope(int first, OVCLASS overloadclass, int *paliassym, int *plevel,
             int multiple_use_error)
{
  SPTR sptrloop, bestsptr, bestsptrloop, cc_alias;
  int bestsl, bestuse, bestuse2, bestusecount, bestuse2count;

  // Construct names have their own OC_CONSTRUCT overloading class (name
  // space), but are actually in the OC_OTHER class along with variables names
  // and a number of other names.  The compiler should flag multiple uses of
  // a name in this class, even though it looks benign to allow overloading
  // between construct names and other names in this class in most cases.
  // Another consideration is that construct scoping information should
  // (probably) also be taken into account, even though that information is
  // not always available for symbols here.  As a compromise to exact error
  // reporting, look for name clashes of this sort primarily (but not
  // exclusively) for symbols at the top construct scope level, to reduce
  // scoping complications.  As with prior code without an alt_overloadclass
  // check, this continues to permit both false positives and false negatives.
  // These should be rare and easy to work around by not overloading names.
  OVCLASS alt_overloadclass =
    overloadclass == OC_CONSTRUCT && sem.doif_depth == 0 ? OC_OTHER : 0;

  if (paliassym)
    *paliassym = 0;
  if (plevel)
    *plevel = 0;
  bestsptr = bestsptrloop = 0;
  bestuse = bestuse2 = bestusecount = bestuse2count = 0;
  bestsl = -1;
  for (sptrloop = first_hash(first); sptrloop; sptrloop = HASHLKG(sptrloop)) {
    int want_scope, usecount, sptrlink;
    SCOPESTACK *scope;
    int sptr = sptrloop;
    if (NMPTRG(sptr) != NMPTRG(first))
      continue;
    switch (STYPEG(sptr)) {
    case ST_ISOC:
    case ST_CRAY:
      /* predefined symbol, but not active in this routine */
      continue;
    case ST_MODPROC:
    case ST_PROC:
    case ST_IDENT:
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
    case ST_DESCRIPTOR:
    case ST_TYPEDEF:
      if (HIDDENG(sptr))
        continue;
      /* make sure it is in current function scope */
      if (gbl.internal > 1 && SCG(sptr) == SC_PRIVATE && ENCLFUNCG(sptr)) {
        if (!isLocalPrivate(sptr))
          continue;
      }
      break;
    default:;
    }
    if (sem.scope_stack == NULL) {
      /* must be after the parser, such as in static-init */
      if (overloadclass == 0 || STYPEG(sptr) == ST_UNKNOWN ||
          stb.ovclass[STYPEG(sptr)] == overloadclass) {
        if (STYPEG(sptr) == ST_ALIAS)
          sptr = SYMLKG(sptr);
        if (paliassym != NULL)
          *paliassym = sptrloop;
        if (plevel != NULL)
          *plevel = -1;
        return sptr;
      }
      continue;
    }

    /* in a current scope? */
    want_scope = SCOPEG(sptr);
    if (want_scope == 0 && STYPEG(sptr) == ST_MODULE) {
      /* see if there is a USE clause for this module, use that level */
      SCOPESTACK *scope = next_scope_kind_sptr(0, SCOPE_USE, sptr);
      if (scope != 0) {
        want_scope = scope->sptr; /* treat module as scoped at itself */
      }
    }
    if (want_scope == 0) {
      if (STYPEG(sptr) == ST_ALIAS)
        sptr = SYMLKG(sptr);
      if (bestsl == -1) {
        bestsl = 0;
        bestsptr = sptr;
        bestsptrloop = sptrloop;
      }
      continue;
    }
    cc_alias = 0;
    if (STYPEG(sptr) == ST_ALIAS && DCLDG(sptr) &&
        NMPTRG(sptr) == NMPTRG(SYMLKG(sptr))) {
      /* from a 'use module, only: i' statement;
       * compiler inserts an alias 'i' in this scope,
       * but the alias in this scope has no meaning; look at the
       * original symbol 'i'.
       * This is very different from 'use module, only: j=>i',
       * where the alias 'j' in this scope DOES have meaning
       *
       * But (fs16195), do keep track of alias as an additional
       * check of the except list.
       */
      cc_alias = sptr;
      sptr = SYMLKG(sptr);
    }
    sptrlink = sptr;
    while ((STYPEG(sptrlink) == ST_ALIAS || STYPEG(sptrlink) == ST_MODPROC) &&
           SYMLKG(sptrlink)) {
      sptrlink = SYMLKG(sptrlink);
    }
    usecount = 0;
    /* look in the scope stack for an active scope containing this symbol */
    scope = 0;
    while ((scope = next_scope(scope)) != 0) {
      /* past a SCOPE_NORMAL, association is HOST-association,
       * not USE-association */
      if (scope->kind == SCOPE_NORMAL)
        ++usecount;
      if (scope->sptr == want_scope ||
          /* module procedures are 'scoped' at module level.
           * treat as if they are scoped here */
          scope->sptr == sptrloop || 
          (scope->sptr && want_scope < stb.stg_avail && 
           scope->sptr == find_explicit_interface(want_scope))) {
        LOGICAL found = is_except_in_scope(scope, sptr) ||
                        is_except_in_scope(scope, cc_alias);
        if (scope->Private &&
            ((STYPEG(sptr) != ST_PROC && STYPEG(sptr) != ST_OPERATOR &&
              STYPEG(sptr) != ST_USERGENERIC) ||
             (!VTOFFG(sptr) && !TBPLNKG(sptr)) ||
             (IS_TBP(sptr) && PRIVATEG(sptr)))) {
          found = TRUE; /* in a private USE */
        } else if (scope->kind == SCOPE_USE &&
                   (PRIVATEG(sptr) || 
                    PRIVATEG(sptrloop))) {
          /* FE creates an alias when processing the case like:
                'use mod_name, only : i'. 
             So, if found sptrloop is a type of ST_ALIAS, we need to check whether
             current module is a submod of ENCLFUNCG(sptrloop). If yes, then this
             variable is accessible. 
           */
          if (STYPEG(sptrloop) == ST_ALIAS && ANCESTORG(gbl.currmod))
            found = ENCLFUNCG(sptrloop) != ANCESTORG(gbl.currmod);
          else
            found = TRUE; /* private module variable */
          /* private module variables are visible to inherited submodules*/
          if (is_used_by_submod(gbl.currsub, sptr))
            return sptr;
        }
        if (!found) { /* not found in 'except' list */
          if (STYPEG(sptr) == ST_ALIAS)
            sptr = SYMLKG(sptr);
          if (overloadclass == 0 || STYPEG(sptrlink) == ST_UNKNOWN ||
              stb.ovclass[STYPEG(sptrlink)] == overloadclass ||
              (stb.ovclass[STYPEG(sptrlink)] == alt_overloadclass &&
              !CONSTRUCTSYMG(sptrlink))) {
            int sl = get_scope_level(scope);
            if (sl > bestsl &&
                (scope->kind != SCOPE_BLOCK || sptr >= scope->symavl)) {
              if (scope->kind == SCOPE_USE &&
                  STYPEG(sptrlink) != ST_USERGENERIC &&
                  STYPEG(sptrlink) != ST_ENTRY && !VTOFFG(sptrlink) &&
                  !TBPLNKG(sptrlink)) {
                if (bestuse && bestuse2 == 0 && bestsptr != sptrlink) {
                  bestuse2 = bestuse;
                  bestuse2count = bestusecount;
                }
                bestuse = scope->sptr;
                bestusecount = usecount;
              } else {
                bestuse = 0;
              }
              bestsl = sl;
              bestsptr = sptrlink;
              bestsptrloop = sptrloop;
            } else if (bestuse && scope->kind == SCOPE_USE &&
                       /* for submodule, use-association overwrites host-association*/
                       STYPEG(scope->sptr) == ST_MODULE && 
                       ANCESTORG(gbl.currmod) != scope->sptr &&
                       scope->sptr != bestuse &&
                       STYPEG(sptrlink) != ST_USERGENERIC &&
                       STYPEG(sptrlink) != ST_ENTRY && !VTOFFG(sptrlink) &&
                       !TBPLNKG(sptrlink) && bestsptr != sptrlink) {
              bestuse2 = scope->sptr;
              bestuse2count = usecount;
            }
          }
        }
      }
      if (scope->closed && scope->kind != SCOPE_INTERFACE) {
        if (!bestsptr && scope->kind == SCOPE_NORMAL && scope->import) {
          if (sym_in_sym_list(sptr, scope->import)) {
            if (STYPEG(sptr) == ST_ALIAS)
              sptr = SYMLKG(sptr);
            return sptr;
          }
        }
        break; /* can't go farther out anyway */
      }
    }
  }

  if (bestuse && bestuse2 && multiple_use_error && bestuse != bestuse2 &&
      !isSameNameGenericOrProcOrModproc(bestsptr, bestsptrloop) &&
      bestusecount == bestuse2count && sem.which_pass == 1) {
    /* oops; this name is USE-associated from two
     * different modules */
    char msg[200];
    sprintf(msg,
            "is use-associated from modules %s and %s,"
            " and cannot be accessed",
            SYMNAME(bestuse), SYMNAME(bestuse2));
    error(155, 3, gbl.lineno, SYMNAME(first), msg);
  }
  if (paliassym != NULL)
    *paliassym = bestsptrloop;
  if (plevel != NULL)
    *plevel = bestsl;
  return bestsptr;
} /* sym_in_scope */

/** \brief IMPORT symbol from host scope -- not to be confused with interf
           import stuff.
 */
void
sem_import_sym(int s)
{
  int sptr;
  int smi;
  SCOPESTACK *scope;

  sptr = find_in_host(s);
  while (sptr > NOSYM && STYPEG(sptr) == ST_ALIAS && SYMLKG(sptr) > NOSYM &&
         strcmp(SYMNAME(sptr), SYMNAME(SYMLKG(sptr))) == 0)
    sptr = SYMLKG(sptr); /* FS#17251 - need to resolve alias */
  if (sptr <= NOSYM) {
    error(155, 3, gbl.lineno, "Cannot IMPORT", SYMNAME(s));
    return;
  }
  /*
   *   <zero or more> SCOPE_USE
   */
  scope = next_scope_kind(0, SCOPE_NORMAL);
  smi = add_symitem(sptr, scope->import);
  scope->import = smi;
}

/*
 * The current context is:
 * interface
 *    ...
 *    subroutine/function  ...
 *        INPORT FROMHOST  <<< current context, find FROMHOST>>>>
 *    endsubroutine/endfunction
 *    ...
 * endinterface
 *
 * There should be three scope entries corresponding to this context:
 *
 * scope_level-2 : SCOPE_INTERFACE
 * scope_level-1 : SCOPE_NORMAL
 *   <zero or more> SCOPE_USE
 * scope_level   : SCOPE_SUBPROGRAM
 *
 */
static int
find_in_host(int s)
{
  int cap;
  int sptr;
  SCOPESTACK *scope, *iface_scope;

  /*
   * First check for the minimal scope entries.
   */
  cap = sem.scope_level - 3 * (sem.interface - 1);
  if (cap < 4)
    return -1;

  scope = get_scope(cap);
  if (scope->kind != SCOPE_SUBPROGRAM) {
    return -1;
  }
  scope = next_scope_kind(scope, SCOPE_NORMAL);
  if (scope == 0 || get_scope_level(scope) < 4) {
    return -1;
  }

  iface_scope = next_scope(scope);
  if (iface_scope->kind != SCOPE_INTERFACE) {
    return -1;
  }

  /*
   * Find symbol suitable for IMPORT from the hash list.
   */
  for (sptr = first_hash(s); sptr; sptr = HASHLKG(sptr)) {
    if (NMPTRG(sptr) != NMPTRG(s))
      continue;
    if (stb.ovclass[STYPEG(sptr)] != OC_OTHER)
      continue;
    /*
     * Now, search the scope entries.
     */
    /*
     * Now, search the scope entries starting below the scope for the interface
     */
    scope = iface_scope;
    while ((scope = next_scope(scope)) != 0) {
      if (scope->sptr == SCOPEG(sptr)) {
        LOGICAL ex;
        if (scope->except) {
          ex = is_except_in_scope(scope, sptr);
        } else if (scope->Private) {
          for (ex = scope->only; ex; ex = SYMI_NEXT(ex)) {
            int sptr2 = SYMI_SPTR(ex);
            if (sptr2 == sptr)
              return sptr;
            /* FS#14811  Check for symbol in GNDSC list. */
            if (STYPEG(sptr2) == ST_ALIAS)
              sptr2 = SYMLKG(sptr2);
            if (sym_in_sym_list(sptr2, GNDSCG(sptr))) {
              return sptr;
            }
          }
          ex = TRUE; /* in a private USE */
        } else {
          ex = FALSE;
        }
        if (!ex) /* not on a 'except' list */
          return sptr;
      }
      if (scope->closed)
        break; /* can't go farther out anyway */
    }
  }
  return -1;
}

int
test_scope(int sptr)
{
  int sl;
  for (sl = sem.scope_level; sl >= 0; --sl) {
    SCOPESTACK *scope = get_scope(sl);
    if (scope->sptr == SCOPEG(sptr)) {
      int ex = is_except_in_scope(scope, sptr);
      if (scope->Private) {
        for (ex = scope->only; ex; ex = SYMI_NEXT(ex)) {
          int sptr2 = SYMI_SPTR(ex);
          if (sptr2 == sptr)
            return sl;
          /* FS#14811  Check for symbol in GNDSC list. */
          if (STYPEG(sptr2) == ST_ALIAS)
            sptr2 = SYMLKG(sptr2);
          if (sym_in_sym_list(sptr2, GNDSCG(sptr))) {
            return sl;
          }
        }
        ex = 1; /* in a private USE */
      } else if (scope->kind == SCOPE_USE && scope->sptr != gbl.currmod &&
                 PRIVATEG(sptr)) {
        ex = 1; /* private module variable */
      }
      if (ex == 0) /* not on a 'except' list */
        return sl;
    }
    if (scope->closed)
      break; /* can't go farther out anyway */
  }
  return -1;
} /* test_scope */

/* **********************************************************************/

/** \brief Look up symbol having a specific symbol type.

    If a symbol is found in the same overloading class and has
    the same symbol type, it is returned to the caller.
    If a symbol is found in the same overloading class, the action
    of declref depends on the stype of the existing symbol and
    value of the argument def:
    1.  if symbol is an unfrozen intrinsic and def is 'd' (define),
        its intrinsic property is removed and a new symbol is declared,
    2.  if def is 'd', a multiple declaration error occurs, or
    3.  if def is not 'd', an 'illegal use' error occurs

    If an error occurs or a matching symbol is not found, one is
    created and its symbol type is initialized.
 */
int
declref(int first, SYMTYPE stype, int def)
{
  int sptr1, sptr;

  sptr = sym_in_scope(first, stb.ovclass[stype], NULL, NULL, 0);
  if (sptr) {
    SYMTYPE st = STYPEG(sptr);
    if (st == ST_UNKNOWN && sptr == first)
      goto return1; /* stype not set yet, set it */
    if ((int)SCOPEG(sptr) != stb.curr_scope && def == 'd')
      goto return0;
    if (stype != st) {
      if (def == 'd') {
        /* Redeclare of intrinsic symbol is okay unless frozen */
        if (IS_INTRINSIC(st)) {
          if ((sptr1 = newsym(sptr)) != 0)
            sptr = sptr1;
          goto return1;
        }
        /* multiple declaration */
        error(44, 3, gbl.lineno, SYMNAME(first), CNULL);
      } else
        /* illegal use of symbol */
        error(84, 3, gbl.lineno, SYMNAME(first), CNULL);
      goto return0;
    }
    goto return2; /* found, return it */
  }
return0:
  sptr = insert_sym(first); /* create new one if def or illegal use */
return1:
  STYPEP(sptr, stype);
  SCOPEP(sptr, stb.curr_scope);
  if (!sem.interface)
    IGNOREP(sptr, 0);
return2:
  if (flg.xref)
    xrefput(sptr, def);
  return sptr;
}

/* If we see an ENTRY in a module with the same name as a variable
 * in the module, we must change the variable into an ENTRY.
 * We must remove the variable from the module common
 * (actually, to simplify things, we replace it with another variable)
 * and change the sptr to be an ENTRY.  We can't add another ENTRY to
 * the end, because postprocessing of the symbols added by this subprogram
 * assumes that all new symbols are undeclared in the module specification
 * part, and changes things like the PRIVATE/PUBLIC bit. */
static int
replace_variable(int sptr, SYMTYPE stype)
{
  int newsptr;
  ACCL *accessp;
  newsptr = insert_sym(sptr);
  STYPEP(newsptr, stype);
  DTYPEP(newsptr, DTYPEG(sptr));
  /* add 'private' or 'public' for this symbol */
  accessp = (ACCL *)getitem(3, sizeof(ACCL));
  accessp->next = sem.accl.next;
  sem.accl.next = accessp;
  accessp->sptr = newsptr;
  accessp->oper = ' ';
  if (PRIVATEG(sptr)) {
    accessp->type = 'v';
  } else {
    accessp->type = 'u';
  }
  HIDDENP(sptr, 1);
  module_must_hide_this_symbol(sptr);
  return newsptr;
} /* replace_variable */

static void
set_internref_flag(int sptr)
{
  INTERNREFP(sptr, 1);
  if (DTY(DTYPEG(sptr)) == TY_ARRAY || POINTERG(sptr) || ALLOCATTRG(sptr) ||
      IS_PROC_DUMMYG(sptr) || ADJLENG(sptr)) {
    int descr, sdsc, midnum, devcopy;
    int cvlen = 0;
    descr = DESCRG(sptr);
    sdsc = SDSCG(sptr);
    midnum = MIDNUMG(sptr);
    devcopy = DEVCOPYG(sptr);
    // adjustable char arrays can exist as single vars or array of arrays
    if (STYPEG(sptr) == ST_VAR || STYPEG(sptr) == ST_ARRAY ||
        STYPEG(sptr) == ST_IDENT)
      cvlen = CVLENG(sptr);
    if (descr)
      INTERNREFP(descr, 1);
    if (sdsc)
      INTERNREFP(sdsc, 1);
    if (midnum)
      INTERNREFP(midnum, 1);
    if (cvlen)
      INTERNREFP(cvlen, 1);
    if (devcopy)
      INTERNREFP(devcopy, 1);
  }
  if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
    ADSC *ad;
    ad = AD_DPTR(DTYPEG(sptr));
    if (AD_ADJARR(ad) || ALLOCATTRG(sptr) || ASSUMSHPG(sptr)) {
      int i, ndim;
      ndim = AD_NUMDIM(ad);
      for (i = 0; i < ndim; i++) {
        internref_bnd(AD_LWAST(ad, i));
        internref_bnd(AD_UPAST(ad, i));
        internref_bnd(AD_MLPYR(ad, i));
        internref_bnd(AD_EXTNTAST(ad, i));
      }
      internref_bnd(AD_NUMELM(ad));
      internref_bnd(AD_ZBASE(ad));
    }
  }
  if (SCG(sptr) == SC_DUMMY && CLASSG(sptr)) {
    int parent = PARENTG(sptr);
    if (parent && CLASSG(parent))
      set_internref_flag(parent);
  }
}

static void
internref_bnd(int ast)
{
  if (ast && A_TYPEG(ast) == A_ID) {
    int sptr;
    sptr = A_SPTRG(ast);
    INTERNREFP(sptr, 1);
  }
}

void
set_internref_stfunc(int ast, int* extra_arg)
{
  if (ast && A_TYPEG(ast) == A_ID) {
    int sptr;
    sptr = A_SPTRG(ast);
    if (SCOPEG(sptr) && SCOPEG(sptr) != gbl.currsub)
      set_internref_flag(sptr);
  }
}

/** \brief Declare a new symbol.

    An error can occur if the symbol is already in the symbol table.<br>
    If the symbol types match: treat as in error if \a errflg is true; otherwise
    return the symbol.<br>
    If they don't match: if symbol is an intrinsic attempt to remove symbol's
    intrinsic property; otherwise it is an error.
 */
int
declsym(int first, SYMTYPE stype, LOGICAL errflg)
{
  SYMTYPE st;
  int sptr1, sptr, sptralias, oldsptr, level;
  int symi;

  sptr = sym_in_scope(first, stb.ovclass[stype], &sptralias, &level, 0);
  if (sptr) {
    if (STYPEG(sptr) == ST_ENTRY && FVALG(sptr))
      sptr = FVALG(sptr);
    st = STYPEG(sptr);
    if (st == ST_UNKNOWN && sptr == first && gbl.internal &&
        sptr < stb.firstusym)
      goto return0; /* New symbol at this scope. */
    if ((st == ST_UNKNOWN || 
         (st == ST_MODPROC && !SEPARATEMPG(sptr) && sem.interface)) && 
        sptr == first && sptr >= stb.firstusym)
      goto return1; /* Brand new symbol, return it. */
    if ((int)SCOPEG(sptr) == stb.curr_scope && st == ST_IDENT &&
        stb.ovclass[st] == stb.ovclass[stype]) {
      /* Found an ST_IDENT in the same overloading class */
      goto return1; /* OK (?) */
    }
    if (stype == ST_USERGENERIC) {
      if ((STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_MODPROC) &&
          GSAMEG(sptr)) {
        /* Looking for a generic, found a subroutine by the same name.
         * Get the generic
         */
        sptr = GSAMEG(sptr);
        st = STYPEG(sptr);
      } else if (STYPEG(sptr) == ST_TYPEDEF) {
        oldsptr = sptr;
        sptr = insert_sym(first);
        GTYPEP(sptr, oldsptr);
        goto return1;
      }
    }
    if (stype == ST_ENTRY && st == ST_USERGENERIC) {
      /* looking for a subroutine (modproc) found a generic, look  for a
       * modproc with the same name */
      for (sptr1 = first_hash(sptr); sptr1 > NOSYM; sptr1 = HASHLKG(sptr1)) {
        if (NMPTRG(sptr) == NMPTRG(sptr1) && STYPEG(sptr1) == ST_USERGENERIC &&
            GSAMEG(sptr1)) {
          sptr = GSAMEG(sptr1);
          st = STYPEG(sptr);
          break;
        }
      }
    }
    if (stype == ST_MODPROC && IN_MODULE_SPEC) {
      for (sptr1 = first_hash(sptr); sptr1 > NOSYM; sptr1 = HASHLKG(sptr1)) {
        if (NMPTRG(sptr) == NMPTRG(sptr1) && STYPEG(sptr1) == ST_MODPROC &&
            SCOPEG(sptr1) == gbl.currmod) {
          sptr = sptr1;
          st = STYPEG(sptr);
          break;
        }
      }
    }
    if (SCOPEG(sptr) == 0) { /* predeclared, overwrite it */
      oldsptr = sptr;
      sptr = insert_sym(first);
      if (st != ST_MODULE && DCLDG(oldsptr)) {
        DCLDP(sptr, 1);
        DTYPEP(sptr, DTYPEG(oldsptr));
        DCLDP(oldsptr, 0);
      }
      goto return1;
    }
    if (stype == st) {
      if (st == ST_GENERIC && sptr < stb.firstusym) {
        if ((sptr1 = newsym(sptr)) != 0) {
          sptr = sptr1;
          goto return1;
        }
      }
      /* is this a symbol that was host-associated?
       * if so, declare a new symbol */
      if (level > 0) {
        if (get_scope_level(next_scope_kind(0, SCOPE_NORMAL)) > level) {
          /* declare a new symbol; the level at which
           * the existing symbol was found is outside the
           * current scope */
          goto return0;
        }
      }
      /* Possible attempt to multiply define symbol */
      if (errflg) {
        if (stype == ST_ENTRY && sem.interface == 1) {
          /* interface for a subprogram appears in the
           * the subprogram; just create another instance
           * of the ST_ENTRY.
           */
          sptr = insert_sym(first);
          STYPEP(sptr, stype);
          SCOPEP(sptr, stb.curr_scope);
          return sptr;
        }
        if (stype == ST_IDENT && STYPEG(first) == ST_ENTRY) {
          if (SCOPEG(first) == 0 && stb.curr_scope) {
            /* host (outer) routine with same-named
             * identifier in inner scope
             */
            sptr = insert_sym(first);
            STYPEP(sptr, stype);
            SCOPEP(sptr, stb.curr_scope);
            return sptr;
          }
        }

        if (sptr == first && (int)SCOPEG(sptr) != stb.curr_scope && sem.interface == 1) {
          sptr = insert_sym(first);
          STYPEP(sptr, stype);
          SCOPEP(sptr, stb.curr_scope);
          return sptr;
        }
        error(44, 3, gbl.lineno, SYMNAME(first), CNULL);
        goto return0;
      }
      goto return2;
    }
    /* stype != st */
    if (sem.interface && stype == ST_ENTRY && st == ST_PROC &&
        (int)SCOPEG(sptr) == stb.curr_scope) {
      /* nested interface for a subprogram which is an
       * argument to the current subprogram; make it an
       * entry and return it;
       */

      if (SCG(sptr) == SC_DUMMY) {
        STYPEP(sptr, stype);
      }
      return sptr;
    }
    /* Redeclare of intrinsic symbol is okay unless frozen */
    if (IS_INTRINSIC(st)) {
      if (EXPSTG(sptr) && stype == ST_GENERIC) {
        /* used intrinsic before (in an initializatn?),
           now want to use name as a generic.
           Should be o.k. */
        sptr = sptr1 = insert_sym(first);
        goto return1;
      }
      if ((sptr1 = newsym(sptr)) != 0)
        sptr = sptr1;
      goto return1;
    }
    if (st == ST_USERGENERIC) {
      if (GSAMEG(sptr) == 0 && (stype == ST_ENTRY || stype == ST_MODPROC)) {
        sptr1 = insert_sym(first);
        if (ENCLFUNCG(sptr) && STYPEG(ENCLFUNCG(sptr)) == ST_MODULE &&
            ENCLFUNCG(sptr) != gbl.currmod) {
          /* user generic was from a USE MODULE statement */
        } else {
          GSAMEP(sptr, sptr1);
          GSAMEP(sptr1, sptr);
          /* find MODPROC and fix up its SYMLK if necessary */
          for (symi = GNDSCG(sptr); symi; symi = SYMI_NEXT(symi)) {
            int sptr_modproc = SYMI_SPTR(symi);
            if (NMPTRG(sptr1) != NMPTRG(sptr_modproc))
              continue;
            if (STYPEG(sptr_modproc) == ST_MODPROC && !SYMLKG(sptr_modproc)) {
              SYMLKP(sptr_modproc, sptr1);
              export_append_sym(sptr_modproc);
            }
            break;
          }
        }
        sptr = sptr1;
        goto return1;
      }
    }
    if (stype == ST_ENTRY && st == ST_MODPROC && IN_MODULE &&
        sem.interface == 0 && SYMLKG(sptr) == 0) {
      sptr1 = insert_sym(first);
      SYMLKP(sptr, sptr1);
      export_append_sym(sptr);
      if (GSAMEG(sptr)) {
        GSAMEP(sptr1, GSAMEG(sptr));
        GSAMEP(GSAMEG(sptr), sptr1);
      }
      if (PRIVATEG(sptr)) {
        PRIVATEP(sptr1, 1);
      }
      sptr = sptr1;
      goto return1;
    }
    if (stype == ST_ENTRY && STYPEG(sptralias) == ST_ALIAS && sem.mod_sym &&
        st == ST_PROC && ENCLFUNCG(sptr) == sem.mod_sym) {
      /* the existing symbol is the interface (ST_PROC) for
       * a module contained subprogram.
       */
      /*pop_sym(sptr);*/ /* Hide the module subprogram symbol */
      IGNOREP(sptr, 1);
      HIDDENP(sptr, 1);
      sptr = sptralias;
      goto return1;
    }
    if (stype == ST_ENTRY && sptralias == sptr && sem.mod_sym &&
        st == ST_PROC && ENCLFUNCG(sptr) == sem.mod_sym) {
      /* the existing symbol is the interface (ST_PROC) for
       * a module contained subprogram; no ST_ALIAS added
       * for native-mode.
       */
      IGNOREP(sptr, 1); /* hide the subprogram symbol */
      oldsptr = sptr;
      /* create new one if def or illegal use */
      sptr = insert_sym(first);
      /* make sure this is the first symbol on the hash list */
      pop_sym(sptr);
      push_sym(sptr);
      INMODULEP(sptr, INMODULEG(oldsptr));
      goto return1;
    }
    if (stype == ST_ENTRY && sptralias == sptr &&
        SCOPEG(sptr) == stb.curr_scope && st == ST_PROC &&
        ENCLFUNCG(sptr) == sem.mod_sym && !INTERNALG(sptr)) {
      /* the existing symbol was added for a CALL, and now we see
       * an ENTRY of that name.
       */
      SCP(sptr, SC_NONE);
      goto return1;
    }
    /* is this a symbol that was host-associated?
     * if so, declare a new symbol */
    if (level >= 0) {
      if (get_scope_level(next_scope_kind(0, SCOPE_NORMAL)) > level) {
        /* declare a new symbol; the level at which
         * the existing symbol was found is outside the
         * current scope */
        goto return0;
      }
    }
    if (stype == ST_ENTRY && st == ST_PROC) {
      goto return0;
    }
    /* if we are declaring a MODULE PROCEDURE, but we have found
     * a name from an USE or outer associated scope level, create a new
     * symbol */
    if (stype == ST_MODPROC) {
      switch (get_scope(level)->kind) {
      case SCOPE_OUTER:
        goto return0;
      case SCOPE_MODULE:
        if (STYPEG(sptr) == ST_PROC) {
          oldsptr = sptr;
          sptr = insert_sym(first); /* create new one */
          SYMLKP(sptr, oldsptr);    /* resolve ST_MODPROC */
          goto return1;
        }
        break;
      case SCOPE_USE:
        if (STYPEG(sptr) == ST_PROC) {
          oldsptr = sptr;
          sptr = insert_sym(first); /* create new one */
          SYMLKP(sptr, oldsptr);    /* resolve ST_MODPROC */
          goto return1;
        }
        goto return0;
      default:;
      }
    }
    /* if we are in a module, creating a module subprogram,
     * and the old symbol is a 'variable', override the variable. */
    if (sem.mod_sym && sem.which_pass == 0 && gbl.internal == 0 &&
        stype == ST_ENTRY && st == ST_VAR) {
      sptr = replace_variable(sptr, stype);
      goto return1;
    }
    error(43, 3, gbl.lineno, "symbol", SYMNAME(first));
  }

return0:
  sptr = insert_sym(first); /* create new one if def or illegal use */
return1:
  STYPEP(sptr, stype);
  SCOPEP(sptr, stb.curr_scope);
  if (!sem.interface)
    IGNOREP(sptr, 0);
return2:
  if (flg.xref)
    xrefput(sptr, 'd');
#ifdef GSCOPEP
  if (sem.which_pass && gbl.internal <= 1 &&
      internal_proc_has_ident(sptr, gbl.currsub)) {
    GSCOPEP(sptr, 1);
  }
#endif
  if (gbl.internal > 1 && first == sptr) {
    set_internref_flag(sptr);
  }

  return sptr;
}

/** \brief Look up a symbol having the given overloading class.

    If the symbol with the overloading class is found its sptr is returned.  If
    no symbol with the given overloading class is found, a new sptr is returned.
 */
int
refsym(int first, OVCLASS oclass)
{
  int sptr, sl;

  sptr = sym_in_scope(first, oclass, NULL, NULL, 1);
  if (sptr) {
    SYMTYPE st = STYPEG(sptr);
    if (st == ST_UNKNOWN && sptr == first)
      goto return1;
    if (stb.ovclass[st] == oclass) {
      /* was this a reference to the return value? */
      if (st == ST_ENTRY && !RESULTG(sptr) &&
          (gbl.rutype == RU_FUNC || (sptr == gbl.outersub && FVALG(sptr)))) {
        /* always a reference to the result variable */
        sl = sptr;
        sptr = ref_entry(sptr);
        if (FVALG(sl) == sptr) {
          if (gbl.internal > 1) {
            set_internref_flag(sptr);
          }
          if ((sem.parallel || sem.task || sem.target || sem.teams)) {
            set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
          }
        }

      } else if (SCG(sptr) == SC_DUMMY) {
        if (gbl.internal > 1) {
          if (SCOPEG(sptr) && SCOPEG(sptr) == SCOPEG(gbl.currsub))
            set_internref_flag(sptr);
        }
        if ((sem.parallel || sem.task || sem.target || sem.teams)) {
          set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
        }
      }
      goto returnit;
    }
  }

  /* Symbol in given overloading class not found, create new one */
  sptr = insert_sym(first);
return1:
  if (flg.xref)
    xrefput(sptr, 'd');
  if (!sem.interface)
    IGNOREP(sptr, 0);
returnit:
  if (gbl.internal > 1 && first == sptr) {
    if (STYPEG(sptr) == ST_PROC && SCG(sptr) == SC_DUMMY)
      set_internref_flag(sptr);
    else if (STYPEG(sptr) != ST_PROC && STYPEG(sptr) != ST_STFUNC)
      set_internref_flag(sptr);
  }
  return sptr;
}

/** \brief Similar to refsym() except that the current scope is taken into
           consideration.

    If the symbol with the overloading class is found its sptr is returned.
    If no symbol with the given overloading class is found, a new sptr is
    returned.
 */
int
refsym_inscope(int first, OVCLASS oclass)
{
  int sptr, level;

  sptr = sym_in_scope(first, oclass, NULL, &level, 0);
  if (sptr) {
    SYMTYPE st = STYPEG(sptr);
    /* if this is the symbol just created for this subprogram, use it */
    if (st == ST_UNKNOWN && sptr == first && sptr >= stb.firstusym)
      goto return1;
    if (stb.ovclass[st] == oclass) {
      if (gbl.currsub == sptr)
        /* && (int)SCOPEG(sptr) == (stb.curr_scope - 1)) */
        goto returnit;
      /* is this a symbol that was host-associated?
       * if so, declare a new symbol */
      if (level > 0) {
        int sl;
        for (sl = sem.scope_level; sl > level; --sl) {
          if (sem.scope_stack[sl].kind == SCOPE_NORMAL) {
            /* declare a new symbol; the level at which
             * the existing symbol was found is outside the
             * current scope */
            goto return0;
          }
        }
      } else if (level == 0 && st == ST_MODULE &&
                 sptr == sem.mod_sym       /* is the current module */
                 && sptr != stb.curr_scope /* not in outer host scope */
      ) {
        /* context is a module which is being defined but not in its
         * module specification part -- the symbol is being declared
         * in a scope contained within the module.
         */
        goto return0;
      }
      if (ENCLFUNCG(sptr) && STYPEG(ENCLFUNCG(sptr)) == ST_MODULE &&
          ENCLFUNCG(sptr) != gbl.currmod) {
        /* see if the scope level makes this host associated */
        if (level < 0)
          goto return0;
        /* use associated symbol */
        if (IGNOREG(sptr) || PRIVATEG(sptr) ||
            (st == ST_PROC && PRIVATEG(SCOPEG(sptr))) ||
            ((st == ST_USERGENERIC || st == ST_OPERATOR) &&
             TBPLNKG(sptr)) /* FS#20696: needed for overloading */
        )
          goto return0; /* create new symbol */
        if (oclass == OC_CMBLK || 
            /* Check whether the gbl.currmod and ENCLFUNCG(sptr) share
               with the same ancestor, if yes then use host-association
             */
            (oclass == OC_OTHER && 
             (ANCESTORG(gbl.currmod) ? 
              ANCESTORG(gbl.currmod) : gbl.currmod) == 
             (ANCESTORG(ENCLFUNCG(sptr)) ? 
              ANCESTORG(ENCLFUNCG(sptr)) : ENCLFUNCG(sptr))))
          goto return0;
        error(155, 3, gbl.lineno, SYMNAME(sptr),
              "is use associated and cannot be redeclared");
        goto return0;
      }
      if (gbl.internal > 1 && !INTERNALG(sptr)) {
        /* This is a non-internal symbol in an internal subprogram. */
        if (IS_INTRINSIC(STYPEG(sptr)))
          goto returnit; // tentative intrinsic; may be overridden later
        goto return0; // declare a new symbol
      }
      /*if ((int)SCOPEG(sptr) == stb.curr_scope)*/
      goto returnit;
      /* break;	don't create new symbol */
    }
  }

return0:
  /* Symbol in given overloading class not found, create new one */
  sptr = insert_sym(first);
return1:
  SCOPEP(sptr, stb.curr_scope);
  if (!sem.interface)
    IGNOREP(sptr, 0);
  if (flg.xref)
    xrefput(sptr, 'd');
returnit:
  if (gbl.internal > 1 && first == sptr) {
    set_internref_flag(sptr);
  }
  return sptr;
}

void
enforce_denorm(void)
{
  int first, sptr;

  if (!sem.ieee_features || STYPEG(gbl.currsub) == ST_MODULE)
    return;
  first = lookupsymbol("ieee_denormal");
  if (!first)
    return;
  sptr = sym_in_scope(first, OC_OTHER, NULL, NULL, 1);
  if (sptr && STYPEG(sptr) == ST_PARAM && SCOPEG(sptr) &&
      strcmp(SYMNAME(SCOPEG(sptr)), "ieee_features") == 0) {
    gbl.denorm = TRUE;
    return;
  }
}

/** \brief Look up symbol matching overloading class of given symbol type.
    \param first  the symbol to match by name
    \param oclass the overloading class to match
    \param alias  if true and the symbol is an `ST_ALIAS`, return the
                  dereferenced symbol
    \return The symbol whose overloading class matches the overloading class of
            the symbol type given.  If no symbol is found in the given
            overloading class one is created.
 */
int
getocsym(int first, OVCLASS oclass, LOGICAL alias)
{
  int sptr, sptralias;

  sptr = sym_in_scope(first, oclass, &sptralias, NULL, 0);
  if (!alias)
    sptr = sptralias;
  if (sptr) {
    SYMTYPE st = STYPEG(sptr);
    if (st == ST_UNKNOWN && sptr == first)
      goto return1;
    if (stb.ovclass[st] == oclass)
      goto returnit; /* found it! */
  }

  /* create new symbol if undefined or illegal use */
  sptr = insert_sym(first);
return1:
  if (flg.xref)
    xrefput(sptr, 'd');
  if (!sem.interface)
    IGNOREP(sptr, 0);
returnit:
  return sptr;
}

/* declobject - certain symbols which are non-data objects (e.g.,
 *              TEMPLATE and PROCESSOR).  In these cases, it's legal to
 *              specify the object's shape before the actual object type.
 *              The symbol representing the object is returned.
 */
int
declobject(int sptr, SYMTYPE stype)
{
  sptr = refsym(sptr, OC_OTHER); /* all objects (data, non-data) */
  if (STYPEG(sptr) == ST_ARRAY && !DCLDG(sptr) && SCG(sptr) == SC_NONE) {
    ADSC *ad;
    ad = AD_DPTR(DTYPEG(sptr));
    if (AD_ASSUMSZ(ad) || AD_DEFER(ad))
      error(30, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    STYPEP(sptr, stype);
    if (flg.xref)
      xrefput(sptr, 'd');
  } else
    sptr = declsym(sptr, stype, TRUE);

  return sptr;
}

/** \brief Reset fields of intrinsic or generic symbol, sptr, to zero in
           preparation for changing its symbol type by the Semantic Analyzer.

   If the symbol type of the symbol has been 'frozen', issue an error message
   and notify the caller by returning a zero symbol pointer.
 */
int
newsym(int sptr)
{
  int sp2, sp1;

  if (EXPSTG(sptr)) {
    /* Symbol previously frozen as an intrinsic */
    error(43, 3, gbl.lineno, "intrinsic", SYMNAME(sptr));
    return 0;
  }
  /*
   * try to find another sym in the same overloading class; we need to
   * try this first since there could be multiple occurrences of an
   * intrinsic and therefore the sptr appears more than once in the
   * semantic stack.  E.g.,
   *    call sub (sin, sin)
   * NOTE that in order for this to work we need to perform another getsym
   * to start at the beginning of the hash links for symbols whose names
   * are the same.
   */
  sp1 = getsym(LOCAL_SYMNAME(sptr), strlen(SYMNAME(sptr)));
  sp2 = getocsym(sp1, OC_OTHER, FALSE);
  if (sp2 != sptr)
    return sp2;
  /*
   * Create a new symbol with the same name:
   */
  error(35, 1, gbl.lineno, SYMNAME(sptr), CNULL);
  sp2 = insert_sym(sp1);

  /* transfer dtype if it was explicitly declared for sptr:  */

  if (DCLDG(sptr)) {
    DTYPEP(sp2, DTYPEG(sptr));
    DCLDP(sp2, 1);
    DCLDP(sptr, 0);
    ADJLENP(sp2, ADJLENG(sptr));
    ADJLENP(sptr, 0);
  }

  return sp2;
}

/*---------------------------------------------------------------------*/

/** \brief Reference a symbol when it's known the context requires an
           identifier.

    If an error occurs (e.g., symbol which is frozen as an intrinsic),
    a new symbol is created so that processing can continue.  If the symbol
    found is ST_UNKNOWN, its stype is changed to ST_IDENT.
 */
int
ref_ident(int sptr)
{
  int sym;

  sym = refsym(sptr, OC_OTHER);
  if (IS_INTRINSIC(STYPEG(sym))) {
    sym = newsym(sym);
    if (sym == 0)
      sym = insert_sym(sptr);
  }
  if (STYPEG(sym) == ST_UNKNOWN)
    STYPEP(sym, ST_IDENT);

  return sym;
}

int
ref_ident_inscope(int sptr)
{
  int sym;

  sym = refsym_inscope(sptr, OC_OTHER);
  if (IS_INTRINSIC(STYPEG(sym))) {
    sym = newsym(sym);
    if (sym == 0)
      sym = insert_sym(sptr);
  }
  if (STYPEG(sym) == ST_UNKNOWN)
    STYPEP(sym, ST_IDENT);

  return sym;
}

/*---------------------------------------------------------------------*/

/** \brief Reference a symbol when it's known the context requires storage,
   e.g.,
           a variable or the result of a function.

   If an error occurs (e.g., symbol which is frozen as an intrinsic), a new
   symbol is created so that processing can continue.  If the symbol found is
   ST_UNKNOWN, its stype is changed to ST_IDENT.
 */
int
ref_storage(int sptr)
{
  int sym;

  sym = ref_ident(sptr);
  switch (STYPEG(sym)) {
  case ST_ENTRY:
    if (gbl.rutype == RU_FUNC && !RESULTG(sptr)) {
      sym = ref_entry(sptr);
    }
    break;
  case ST_IDENT:
    if (DTY(DTYPEG(sym)) == TY_ARRAY)
      STYPEP(sym, ST_ARRAY);
    else
      STYPEP(sym, ST_VAR);
    break;
  default:
    break;
  }

  return sym;
}

int
ref_storage_inscope(int sptr)
{
  int sym;

  sym = refsym_inscope(sptr, OC_OTHER);
  if (IS_INTRINSIC(STYPEG(sym))) {
    sym = newsym(sym);
    if (sym == 0)
      sym = insert_sym(sptr);
  }
  if (STYPEG(sym) == ST_UNKNOWN)
    STYPEP(sym, ST_IDENT);
  switch (STYPEG(sym)) {
  case ST_ENTRY:
    if (gbl.rutype == RU_FUNC && !RESULTG(sym)) {
      sym = ref_entry(sym);
    }
    break;
  case ST_IDENT:
    if (DTY(DTYPEG(sym)) == TY_ARRAY)
      STYPEP(sym, ST_ARRAY);
    else
      STYPEP(sym, ST_VAR);
    break;
  default:
    break;
  }

  return sym;
}

/*---------------------------------------------------------------------*/

/** \brief Reference a symbol when it's known the context requires an integer
           scalar variable.

    If an error occurs (e.g., symbol which is frozen as an intrinsic),
    a new symbol is created so that processing can continue.  If the symbol
    found is ST_UNKNOWN, its stype is changed to ST_IDENT.
 */
int
ref_int_scalar(int sptr)
{
  int sym;

  sym = refsym(sptr, OC_OTHER);
  if (IS_INTRINSIC(STYPEG(sym))) {
    sym = newsym(sym);
    if (sym == 0)
      sym = insert_sym(sptr);
  }
  if (STYPEG(sym) == ST_UNKNOWN)
    STYPEP(sym, ST_IDENT);
  if (STYPEG(sym) == ST_PARAM || !DT_ISINT(DTYPEG(sptr)))
    error(84, 3, gbl.lineno, SYMNAME(sptr),
          "-must be an integer scalar variable");

  return sym;
}

/** \brief Mark a compiler-created temp as static.
 */
static void
mark_static(int astx)
{
  if (A_TYPEG(astx) == A_ID || A_TYPEG(astx) == A_SUBSCR ||
      A_TYPEG(astx) == A_MEM) {
    int sptr;
    sptr = sym_of_ast(astx);
    if (CCSYMG(sptr) || HCCSYMG(sptr)) {
      SCP(sptr, SC_STATIC);
      SAVEP(sptr, 1);
    }
  }
} /* mark_static */

/** \brief Reference a based object.

    Since it's possible to have more than one level of 'based' storage, need to
    scan through the MIDNUM fields until the "pointer" variable is found.  Along
    the way, it may be necessary to fix the stypes of the based variables and to
    create xref 'r' records.  Also, the storage class of the 'pointer' variable
    is fixed if necessary.  The symbol table index of the 'pointer' variable is
    returned.
 */
int
ref_based_object(int sptr)
{
  int sptr1;
  sptr1 = ref_based_object_sc(sptr, SC_LOCAL);
  return sptr1;
}

int
ref_based_object_sc(int sptr, SC_KIND sc)
{
  int sptr1;
#if DEBUG
  assert(SCG(sptr) == SC_BASED || POINTERG(sptr) || ALLOCATTRG(sptr) ||
             (SCG(sptr) == SC_CMBLK && ALLOCG(sptr)),
         "ref_based_object: sptr not based", sptr, 3);
#endif
  if (flg.xref)
    xrefput(sptr, 'r');

  if (DTY(DTYPEG(sptr)) != TY_ARRAY) {
    /* test for scalar pointer */
    if (POINTERG(sptr) && SDSCG(sptr) == 0 && !F90POINTERG(sptr)) {
      if (SCG(sptr) == SC_NONE)
        SCP(sptr, SC_BASED);
      get_static_descriptor(sptr);
      get_all_descriptors(sptr);
    }
  }

  /*
   * for an allocatable array, it's possible that the array is not
   * associated with a pointer (did not appear in a POINTER statement).
   * Create a compiler temporary to represent the pointer variable.
   */
  if (MIDNUMG(sptr) <= NOSYM && !F90POINTERG(sptr)) {
    if (F77OUTPUT) {
      sptr1 = sym_get_ptr(sptr);
    } else {
      sptr1 = getccsym('Z', sptr, ST_VAR);
      DTYPEP(sptr1, DT_PTR);
    }
    MIDNUMP(sptr, sptr1);
    /*
     * if an allocatable array is saved, need to ensure that all of its
     * associated temporary variables are marked save -- e.g., the internal
     * pointer variable, its bounds' variables, its zero-base temporary,
     * etc.
     */
    if (SAVEG(sptr) ||
        (in_save_scope(sptr) && !CCSYMG(sptr) && !HCCSYMG(sptr))) {
      ADSC *ad;
      int i, numdim;

      SCP(sptr1, SC_STATIC);
      SAVEP(sptr1, 1);

      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        ad = AD_PTR(sptr);
        numdim = AD_NUMDIM(ad);

        mark_static(AD_NUMELM(ad));
        mark_static(AD_ZBASE(ad));
        for (i = 0; i < numdim; i++) {
          mark_static(AD_LWAST(ad, i));
          mark_static(AD_UPAST(ad, i));
          mark_static(AD_MLPYR(ad, i));
          mark_static(AD_EXTNTAST(ad, i));
        }
      }
    }
    else if (GSCOPEG(sptr)) {
      fixup_reqgs_ident(sptr);
    }
    else
      SCP(sptr1, sc);
  }
  sptr1 = sptr;
  while (TRUE) {
    if (STYPEG(sptr1) == ST_IDENT)
      STYPEP(sptr1, ST_VAR);
    sptr1 = MIDNUMG(sptr1);
    if (SAVEG(sptr))
      SAVEP(sptr1, 1);
    if (flg.xref)
      xrefput(sptr1, 'r');
    if (SCG(sptr1) != SC_BASED)
      break;
#if DEBUG
    assert(sptr1 > NOSYM, "ref_based_object: bad list", sptr, 0);
#endif
  }
  if (SCG(sptr1) == SC_NONE)
    SCP(sptr1, sc);
  if (gbl.internal > 1 && SCOPEG(sptr) == SCOPEG(gbl.currsub)) {
    set_internref_flag(sptr);
  }
  if (flg.smp)
    check_parref(sptr, sptr, sptr);
  return sptr1;
}

/** \brief Reference the first symbol of the given overloading class in the
           current scope. If not found, zero is returned.
 */
int
refocsym(int first, OVCLASS oclass)
{
  int sptr;

  sptr = sym_in_scope(first, oclass, NULL, NULL, 0);
  if (sptr) {
    SYMTYPE st = STYPEG(sptr);
    if (stb.ovclass[st] == oclass) {
      if (st == ST_ALIAS)
        return DTYPEG(sptr); /* should this be SYMLKG? */
      return sptr;
    }
  }
  /*
   * error -  symbol used in wrong overloading class, except may be
   * function call, so no message:
   */
  return 0;
}

int
sym_skip_construct(int first)
{
  if (first > NOSYM && STYPEG(first) == ST_CONSTRUCT) {
    int sptr = first;
    while ((sptr = HASHLKG(sptr)) > NOSYM) {
      if (NMPTRG(sptr) == NMPTRG(first))
        return sptr;
    }
  }
  return first;
}

/** \brief Return a symbol local to the current BLOCK if applicable.
    \param sptr symbol (index), possibly declared in an enclosing scope.
    \return symbol in the current BLOCK if in a BLOCK (which may be the
            original \sptr); otherwise return original \sptr.

    This function should be called when a symbol might be declared in an
    ancestor subprogram or block, but should be redeclared as a symbol in the
    current BLOCK.
 */
SPTR
block_local_sym(SPTR sptr)
{
  if (sem.block_scope) {
    if ((sptr < sem.scope_stack[sem.block_scope].symavl && !INSIDE_STRUCT))
      // New block symbol hides a symbol in an enclosing scope.
      sptr = insert_sym(sptr);
    if (sptr >= sem.scope_stack[sem.block_scope].symavl &&
        !CONSTRUCTSYMG(sptr)) {
      CONSTRUCTSYMP(sptr, true);
      ENCLFUNCP(sptr, sem.construct_sptr);
    }
  }
  return sptr;
}

/** \brief Declare a symbol in the most current scope; if one already exists
           return it.
 */
int
declsym_newscope(int sptr, SYMTYPE stype, int dtype)
{
  sptr = getocsym(sptr, stb.ovclass[stype], FALSE);
  if (STYPEG(sptr) != stype || SCOPEG(sptr) != stb.curr_scope) {
    if (STYPEG(sptr) != ST_UNKNOWN)
      sptr = insert_sym(sptr);
    /* enter symbol into a separate scope */
    STYPEP(sptr, stype);
    SCOPEP(sptr, stb.curr_scope);
    DTYPEP(sptr, dtype);
    DCLDP(sptr, 1);
    if (gbl.internal > 1)
      INTERNALP(sptr, 1);
  }
  return sptr;
}

static void
nullify_member_after(int ast, int std, int sptr)
{
  int dtype = DTYPEG(sptr);
  int sptrmem, aast, mem_sptr_id;

  for (sptrmem = DTY(DDTG(dtype) + 1); sptrmem > NOSYM;
       sptrmem = SYMLKG(sptrmem)) {
    if (ALLOCATTRG(sptrmem)) {
      aast = mk_id(sptrmem);
      mem_sptr_id = mk_member(ast, aast, DTYPEG(sptrmem));
      std = add_stmt_after(add_nullify_ast(mem_sptr_id), std);
    }
    if (is_tbp_or_final(sptrmem)) {
      /* skip tbp */
      continue;
    }
    if (dtype != DTYPEG(sptrmem) && !POINTERG(sptrmem) &&
        allocatable_member(sptrmem)) {
      aast = mk_id(sptrmem);
      mem_sptr_id = mk_member(ast, aast, DTYPEG(sptrmem));
      nullify_member_after(mem_sptr_id, std, sptrmem);
    }
  }
}
/*---------------------------------------------------------------------*/

/** \brief Declare a private symbol which may be based on the attributes of
           an existing symbol.

    If the symbol doesn't exist (its stype is ST_UNKNOWN), it's assumed that the
    private variable will be a scalar variable.
 */
int
decl_private_sym(int sptr)
{
  int sptr1;
  SYMTYPE stype;
  char *name;
  int new = 0;
  int rgn_level;
  /*
   * First, retrieve the first symbol in the hash list whose name is the same.
   * Then, use refsym to retrieve the first symbol whose overloading class
   * is the same.  This is all necessary because a private symbol could
   * have already been created ahead of the existing symbol (sptr).
   */
  name = SYMNAME(sptr);
  sptr1 = getsymbol(name);
  sptr = refsym(sptr1, stb.ovclass[STYPEG(sptr)]);
  if (SCOPEG(sptr) == sem.scope_stack[sem.scope_level].sptr)
    return sptr; /* a variable can appear in more than 1 clause */
  if (checking_scope && sem.scope_stack[sem.scope_level].kind == SCOPE_PAR) {
    rgn_level = sem.scope_stack[sem.scope_level].rgn_scope;
    if (SCOPEG(sptr) == sem.scope_stack[rgn_level].sptr) {
      return sptr; /* a variable can appear in more than 1 clause */
    }
  }
  if (ALLOCG(sptr) || POINTERG(sptr)) {
    new = insert_sym(sptr1);
    STYPEP(new, STYPEG(sptr));
    if (DTY(DTYPEG(sptr)) == TY_ARRAY)
      DTYPEP(new, dup_array_dtype(DTYPEG(sptr)));
    else
      DTYPEP(new, DTYPEG(sptr));
    ALLOCP(new, ALLOCG(sptr));
    POINTERP(new, POINTERG(sptr));
    ALLOCATTRP(new, ALLOCATTRG(sptr));
    SCP(new, SC_BASED);
    set_descriptor_sc(SC_PRIVATE);
    get_static_descriptor(new);
    get_all_descriptors(new);
    new = add_private_allocatable(sptr, new);
    set_descriptor_sc(SC_LOCAL);
    if (ADJLENG(sptr)) {
      int cvlen = CVLENG(sptr);
      if (cvlen == 0) {
        cvlen = sym_get_scalar(SYMNAME(sptr), "len", DT_INT);
        CVLENP(sptr, cvlen);
        if (SCG(sptr) == SC_DUMMY)
          CCSYMP(cvlen, 1);
      }
      CVLENP(new, cvlen);
      ADJLENP(new, 1);
    }
    goto return_it;
  }
  stype = STYPEG(sptr);
  switch (stype) {
  case ST_UNKNOWN:
    new = sptr;
    STYPEP(new, ST_VAR);
    break;
  case ST_IDENT:
  case ST_VAR:
    new = insert_sym(sptr1);
    STYPEP(new, ST_VAR);
    DTYPEP(new, DTYPEG(sptr));
    if (ADJLENG(sptr)) {
      new = add_private_allocatable(sptr, new);
      goto return_it;
    } else if (ASSUMLENG(sptr)) {
      new = add_private_allocatable(sptr, new);
      goto return_it;
    }
    if (allocatable_member(sptr)) {
      if (checking_scope && sem.scope_stack[sem.scope_level].end_prologue != 0)
        nullify_member_after(
            mk_id(new), sem.scope_stack[sem.scope_level].end_prologue, sptr1);
      else
        nullify_member_after(mk_id(new), STD_PREV(0), sptr1);
    }
    break;
  case ST_STRUCT:
  case ST_UNION:
    new = insert_sym(sptr1);
    STYPEP(new, stype);
    DTYPEP(new, DTYPEG(sptr));
    break;
  case ST_ARRAY:
    new = insert_sym(sptr1);
    STYPEP(new, ST_ARRAY);
    DTYPEP(new, DTYPEG(sptr));
    if (SCG(sptr) == SC_DUMMY) {
      if (ASUMSZG(sptr))
        error(155, 3, gbl.lineno,
              "Assumed-size arrays cannot be specified as private",
              SYMNAME(sptr));
    }
    if (SCG(sptr) == SC_BASED && MIDNUMG(sptr) && !CCSYMG(MIDNUMG(sptr)) &&
        !HCCSYMG(MIDNUMG(sptr))) {
      /* Cray pointee: just copy ADJARR flag (fixes tpr3374) */
      ADJARRP(new, ADJARRG(sptr));
    } else if (ADJARRG(sptr) || RUNTIMEG(sptr) || ADJLENG(sptr)) {
      /*
       * The private copy of an adjustable/automatic array is an
       * allocated array.  The bounds information of the adjustable array
       * and its private copy is the same.  The private array will
       * be allocated from the heap; need to save the sptr of the
       * private copy so that it can be deallocated at the end
       * of the parallel construct.
       */
      new = add_private_allocatable(sptr, new);
      goto return_it;
    } else if (ASSUMSHPG(sptr)) {
      /*
       * The private copy of an assumed-shape array is an allocated
       * array.  The bounds information of the assume-shape array
       * will be assigned to its private copy.  The private array will
       * be allocated from the heap; need to save the sptr of the
       * private copy so that it can be deallocated at the end
       * of the parallel construct.
       */
      ADSC *ad;
      int i, ndim;
      int dt;

      ad = AD_DPTR(DTYPEG(sptr));
      ndim = AD_NUMDIM(ad);
      for (i = 0; i < ndim; i++) {
        int lb;
        lb = AD_LWBD(ad, i);
        if (A_ALIASG(lb)) {
          sem.bounds[i].lowtype = S_CONST;
          sem.bounds[i].lowb = get_isz_cval(A_SPTRG(lb));
        } else {
          sem.bounds[i].lowtype = S_EXPR;
          sem.bounds[i].lowb = lb;
        }
        sem.bounds[i].lwast = lb;
        sem.bounds[i].uptype = S_EXPR;
        sem.bounds[i].upb = AD_UPBD(ad, i);
        sem.bounds[i].upast = AD_UPBD(ad, i);
      }
      sem.arrdim.ndim = ndim;
      sem.arrdim.ndefer = 0;
      dt = mk_arrdsc();
      DTY(dt + 1) = DTY(DTYPEG(sptr) + 1);
      DTYPEP(new, dt);
      new = add_private_allocatable(sptr, new);
      goto return_it;
    }
    break;
  default:
    sptr = new = insert_sym(sptr1);
    STYPEP(new, ST_VAR);
    break;
  }

  if (SCG(sptr) != SC_BASED)
    SCP(new, sem.sc);
  else {
    int stp;
    stp = decl_private_sym(MIDNUMG(sptr));
    MIDNUMP(new, stp);
    SCP(new, SC_BASED);
  }
  if (sem.task && SCG(new) == SC_PRIVATE) {
    int i;
    for (i = sem.doif_depth; i; i--) {
      switch (DI_ID(i)) {
      default:
        break;
      case DI_TASK:
      case DI_TASKLOOP:
        TASKP(new, 1);
        goto td_exit;
      case DI_PAR:
      case DI_PARDO:
      case DI_PARSECTS:
        goto td_exit;
      }
    }
  td_exit:;
  }
return_it:
  if (checking_scope && sem.scope_stack[sem.scope_level].kind == SCOPE_PAR) {
    rgn_level = sem.scope_stack[sem.scope_level].rgn_scope;
    SCOPEP(new, sem.scope_stack[rgn_level].sptr);
  } else
    SCOPEP(new, sem.scope_stack[sem.scope_level].sptr);
  ENCLFUNCP(new, BLK_SYM(sem.scope_level));
  CCSYMP(new, CCSYMG(sptr));
  DCLDP(new, 1); /* so DCLCHK is quiet */
  TARGETP(new, TARGETG(sptr));
  if (flg.smp) {
    if (!ENCLFUNCG(new)) {
      ENCLFUNCP(new, BLK_SCOPE_SPTR(sem.scope_level));
    }
    set_private_encl(sptr, new);
    if (sem.task && SCG(new) == SC_BASED) {
      set_private_taskflag(new);
    }
  }
  return new;
}

static void
check_adjustable_array(int sptr)
{
  if (STYPEG(sptr) == ST_ARRAY && ADJARRG(sptr) && SCG(sptr) != SC_DUMMY) {
    if (!POINTERG(sptr) && !ALLOCATTRG(sptr) && !MIDNUMG(sptr)) {
      int pvar = sym_get_ptr(sptr);

      SCP(pvar, SCG(sptr));
      SCOPEP(pvar, SCOPEG(sptr));
      ENCLFUNCP(pvar, ENCLFUNCG(sptr));
      MIDNUMP(sptr, pvar);
      PTRSAFEP(MIDNUMG(sptr), 1);
    }
  }
}

static int
add_private_allocatable(int old, int new)
{
  /*
   * The private copy of an adjustable/automatic array is an allocated
   * array.  The bounds information of the original object and its private
   * copy are the same.  The private array will be allocated from the heap;
   * need to save the sptr of the private copy so that it can be deallocated
   * at the end of the parallel construct.
   *
   * NOTE, need to distinguish:
   * 1.  allocatables - conditionally allocate/deallocate
   * 2.  pointer      - no allocate/deallocate
   * 3.  other (adj., automatic) - unconditionally  allocate/deallocate
   */
  ITEM *itemp;
  int pvar;
  int allo_obj;
  int where;

  SCP(new, SC_BASED);
  if (!POINTERG(old) && !ALLOCATTRG(old)) {
    pvar = getccsym('Z', new, ST_VAR);
    DTYPEP(pvar, DT_PTR);
    SCP(pvar, sem.sc);
    SCOPEP(pvar, sem.scope_stack[sem.scope_level].sptr);
    ENCLFUNCP(pvar, BLK_SYM(sem.scope_level));
    MIDNUMP(new, pvar);
  }
  if (ADJLENG(old)) {
    int cvlen = CVLENG(old);
    if (cvlen == 0) {
      cvlen = sym_get_scalar(SYMNAME(old), "len", DT_INT);
      CVLENP(old, cvlen);
      if (SCG(old) == SC_DUMMY)
        CCSYMP(cvlen, 1);
    }
    CVLENP(new, cvlen);
    ADJLENP(new, 1);
    if (flg.smp) {
      if (SCG(old) == SC_BASED)
        ref_based_object(old);
      set_parref_flag(cvlen, cvlen, BLK_UPLEVEL_SPTR(sem.scope_level));
      set_parref_flag(old, old, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
  } else if (STYPEG(new) != ST_ARRAY && ASSUMLENG(old)) {
    /* 1) we don't know the size of assumlen char at compile time
     * 2) make private copy adjustable len char
     * 3) make CVLEN a private copy for convenience.
     */
    int ast;
    int oldlen = ast_intr(I_LEN, astb.bnd.dtype, 1, mk_id(old));
    int cvlen = sym_get_scalar(SYMNAME(old), "len", DT_INT);
    ast = mk_assn_stmt(mk_id(cvlen), oldlen, DT_INT);
    (void)add_stmt(ast);
    CVLENP(new, cvlen);
    ADJLENP(new, 1);
    SCP(cvlen, SCG(MIDNUMG(new)));
    ENCLFUNCP(cvlen, ENCLFUNCG(new));
    SCOPEP(cvlen, sem.scope_stack[sem.scope_level].sptr);
    if (SCG(new) == SC_DUMMY)
      CCSYMP(cvlen, 1);
    if (flg.smp) {
      if (SCG(old) == SC_BASED)
        ref_based_object(old);
      set_parref_flag(old, old, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
  }
  allo_obj = mk_id(new); /* base symbol of allocation */
  if (STYPEG(new) == ST_ARRAY) {
    int dt;
    if (ADJARRG(old) || RUNTIMEG(old))
      ADJARRP(new, 1);
    dt = DTYPEG(new);
    if (ASSUMSHPG(old)) {
      ADJARRP(new, 1);
      ADD_NOBOUNDS(dt) = 1;
    }
    if (ADD_NOBOUNDS(dt)) {
      /*
       * an adjustable array with this flag set is an automatic
       * array.  Need to use the bounds of the array in the allocation
       * so that lower() will correctly assign the .A temporaries.
       */
      int numdim;
      int subs[MAXRANK];
      int i;
      if (ALLOCATTRG(old)) {
        /*
         * an allocatable inherits its bounds from the original;
         * switch to the dtype of the original to get the correct
         * bounds.
         */
        dt = DTYPEG(old);
      }
      numdim = ADD_NUMDIM(dt);
      for (i = 0; i < numdim; i++) {
        int lb, ub;

        lb = ADD_LWAST(dt, i);
        if (!lb)
          lb = astb.bnd.one;
        ub = ADD_UPAST(dt, i);
        if (!ub)
          ub = astb.bnd.one;
        subs[i] = mk_triple(lb, ub, 0);
      }
      allo_obj = mk_subscr(allo_obj, subs, numdim, dt);
    }
  }
  if (checking_scope) {
    /* If checking_scope (handling variables in a PARALLEL directive
     * with a DEFAULT(PRIVATE) or DEFAULT(FIRSTPRIVATE) clause),
     * we have to do the allocation in the prologue of the PARALLEL
     * directive.  We saved "end_prologue" in do_default_clause()
     * in semsmp.c
     */
    where = sem.scope_stack[sem.scope_level].end_prologue;
    if (where == 0)
      interr("add_private_allocatable - can't find prologue", 0, 3);
  } else
    where = STD_PREV(0); /* Just add to the end. */

  if (!POINTERG(old)) {
    itemp = (ITEM *)getitem(1, sizeof(ITEM));
    itemp->t.sptr = new;
    itemp->next = DI_ALLOCATED(sem.doif_depth);
    DI_ALLOCATED(sem.doif_depth) = itemp;
    if (ALLOCATTRG(old)) {
      where = add_stmt_after(add_nullify_ast(mk_id(new)), where);
      where = gen_conditional_alloc(mk_id(old), allo_obj, where);
    } else
      where = gen_conditional_alloc(0, allo_obj, where);
    if (checking_scope)
      sem.scope_stack[sem.scope_level].end_prologue = where;
    if (flg.smp) {
      if (SCG(old) == SC_BASED)
        ref_based_object(old);
      set_parref_flag(old, old, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
  }

  return new;
}

static void
check_parref(int sym, int new, int orig)
{
  /* Only set parref in parallel, task, or target.
   * Target should cover teams and distribute.
   */
  if (!(sem.parallel || sem.task || sem.target))
    return;

  if (sym == new) { /* no new private var created - set parref flag */
    check_adjustable_array(sym);
    if (STYPEG(orig) == ST_PROC && FVALG(orig) == new &&SCG(orig) == SC_EXTERN)
      return;
    if (sem.scope_stack[sem.scope_level].par_scope == PAR_SCOPE_SHARED)
      set_parref_flag(sym, new, BLK_UPLEVEL_SPTR(sem.scope_level));
    else if (is_sptr_in_shared_list(sym))
      set_parref_flag(sym, new, BLK_UPLEVEL_SPTR(sem.scope_level));
    else if (SCG(sym) == SC_DUMMY && (sem.parallel || sem.teams || sem.target))
      /* case where dummy argument is omp do loop upper bound */
      set_parref_flag(sym, new, BLK_UPLEVEL_SPTR(sem.scope_level));
    else if (SCOPEG(sym) && sem.scope_level && SCOPEG(sym) < sem.scope_level)
      set_parref_flag(sym, new, BLK_UPLEVEL_SPTR(sem.scope_level));
    else if (sem.task)
      set_parref_flag(sym, new, BLK_UPLEVEL_SPTR(sem.scope_level));
  }
}

/** \brief Check the current parallel scope of a variable.
    \param sym  Represents the variable which actually references storage.
    \param orig Identifier to which the sym refers; for entries, orig will be
                the ST_ENTRY and sym will be its FVAL.

    If the current scope is default private, need to ensure that any variables
    which have not been declared in this scope are declared as private
    variables.  If the current scope is 'none', then need to ensure that the
    variables were actually declared in this scope.

    Note that at this time, ST_UNKNOWN and ST_IDENT symbols should be have
    been resolved.
 */
int
sem_check_scope(int sym, int orig)
{
  int new;
  int no_scope;

  checking_scope = TRUE;
  new = sym;
  if (sem.parallel || sem.task || sem.target || sem.teams
      || sem.orph
  ) {
    /* Cray pointees are special cases:
     * 1.  the pointee is unaffected by the DEFAULT clause.
     * 2.  the pointer's scope is determined at the point of the
     *     pointee's use.
     *
     * For a Cray pointee, need to recursively check its pointer.
     * Check the scope of each pointer and create a private copy if
     * one is needed.
     */
    switch (STYPEG(orig)) {
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
      if (SCG(new) == SC_BASED && MIDNUMG(new) && !CCSYMG(MIDNUMG(new)) &&
          !HCCSYMG(MIDNUMG(new))) {
        int ptr;
        ptr = MIDNUMG(new);
        ptr = refsym(ptr, OC_OTHER);
        ptr = sem_check_scope(ptr, ptr);
        if (ptr != MIDNUMG(new)) {
          /* A new pointer was created, create a new pointee. */
          checking_scope = TRUE;
          new = decl_private_sym(new);
        }
        goto returnit;
      }
      break;
    default:
      break;
    }
    if (sem.scope_stack[sem.scope_level].par_scope != PAR_SCOPE_SHARED) {
      int s;
      switch (STYPEG(orig)) {
      case ST_ENTRY:
      case ST_VAR:
      case ST_ARRAY:
      case ST_STRUCT:
      case ST_UNION:
        if (STYPEG(new) != ST_ENTRY) {
          if (SCG(new) == SC_CMBLK) {
            if (THREADG(CMBLKG(new)))
              goto returnit;
          } else if (THREADG(new)) {
            goto returnit;
          }
          if (CCSYMG(new) || HCCSYMG(new))
            goto returnit;
          if (STYPEG(new) == ST_ARRAY && SCG(new) == SC_DUMMY && ASUMSZG(new)) {
            goto returnit;
          }
        }
        for (s = sem.scope_stack[sem.scope_level].rgn_scope;
             s <= sem.scope_level; s++) {
          if (SCOPEG(new) == sem.scope_stack[s].sptr)
            goto sym_ok;
        }
        no_scope = 0;
        if (!sem.ignore_default_none &&
            sem.scope_stack[sem.scope_level].par_scope == PAR_SCOPE_NONE) {
          no_scope = 1;
        }
        if (STYPEG(new) == ST_ENTRY)
          goto returnit;
        if (sem.scope_stack[sem.scope_level].par_scope ==
            PAR_SCOPE_TASKNODEFAULT) {
          if (sem.parallel) {
            /*
             * for a task appearing within the lexical extent
             * of a parallel region, only the private objects are
             * firstprivate
             */
            if (SCG(new) == SC_BASED && MIDNUMG(new)) {
              int ss, ptr;
              ss = new;
              ptr = MIDNUMG(new);
              while (TRUE) {
                if (SCG(ptr) != SC_PRIVATE)
                  goto returnit;
                if (SCG(ptr) != SC_BASED)
                  break;
                ss = ptr;
              }
            } else if (SCG(new) != SC_PRIVATE)
              goto returnit;
          } else {
            /*
             * for an orphaned task, all non-static objects are
             * firstprivate
             */
            if (SCG(new) == SC_CMBLK || SCG(new) == SC_STATIC || SAVEG(new))
              goto returnit;
            if (SCG(new) == SC_BASED && MIDNUMG(new)) {
              int ss, ptr;
              ss = new;
              ptr = MIDNUMG(new);
              while (TRUE) {
                if (SCG(ptr) == SC_STATIC || SCG(ptr) == SC_CMBLK)
                  goto returnit;
                if (SCG(ptr) != SC_BASED)
                  break;
                ss = ptr;
              }
            }
          }
        }
        new = decl_private_sym(new);
        if (no_scope) {
          add_no_scope_sptr(sym, new, gbl.lineno);
        }
        if (sem.scope_stack[sem.scope_level].par_scope ==
            PAR_SCOPE_FIRSTPRIVATE)
          add_assign_firstprivate(new, sym);
        else if (sem.scope_stack[sem.scope_level].par_scope ==
                 PAR_SCOPE_TASKNODEFAULT)
          add_assign_firstprivate(new, sym);
        break;
      default:
        break;
      }
    sym_ok:;
    }
  }
returnit:
  check_parref(sym, new, orig);
  checking_scope = FALSE;
  return new;
}
