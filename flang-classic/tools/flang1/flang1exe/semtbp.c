/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
      \file semtbp.c
      \brief This file contains semantic processing routines for
        type bound procedures (tbps), generic tbps, derived  type I/O
        generic tbps, and final subroutines.
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
#include "pragma.h"
#include "rte.h"
#include "pd.h"
#include "interf.h"
#include "fdirect.h"
#include "fih.h"

/** \brief Macro for clearing/zeroing out stale dtypes and sptrs in tbp_queue */
#define IS_CLEAR_STALE_RECORDS_TASK(task) (task == TBP_CLEAR_STALE_RECORDS)

/** \brief Macro for tasks that clear the tbp queue */
#define IS_CLEAR_TBP_TASK(task) (task == TBP_CLEAR || task == TBP_CLEAR_ERROR)

/** \brief Macro for tasks that add tbp to tbo queue */
#define IS_ADD_TBP_TASK(task)                              \
  (task == TBP_ADD_SIMPLE || task == TBP_ADD_IMPL ||       \
   task == TBP_CHECK_CHILD || task == TBP_CHECK_PRIVATE || \
   task == TBP_CHECK_PUBLIC)

/** \brief Macro for tasks that add tbps to their associated derived types */
#define IS_ADD_TBP_TO_DTYPE_TASK(task) (task == TBP_ADD_TO_DTYPE)

/** \brief Macro for task that copy inherited tbps from parent type to
    child type */
#define IS_INHERIT_TBP_TASK(task) (task == TBP_INHERIT)

/** \brief Macro for tasks that resolve tbp binding and implementation names */
#define IS_RESOLVE_TBP_TASK(task)                                \
  (task == TBP_COMPLETE_ENDMODULE || task == TBP_COMPLETE_FIN || \
   task == TBP_COMPLETE_END || task == TBP_COMPLETE_ENDTYPE ||   \
   task == TBP_COMPLETE_GENERIC || task == TBP_FORCE_RESOLVE)

/** \brief Macro for task that adds user specified interface to tbp */
#define IS_ADD_TBP_INTERFACE_TASK(task) (task == TBP_ADD_INTERFACE)

/** \brief Macro for tasks that adds pass attribute to tbp */
#define IS_ADD_TBP_PASS_ARG_TASK(task) (task == TBP_PASS)

/** \brief Macro for task that adds nopass attribute to tbp */
#define IS_ADD_TBP_NOPASS_ARG_TASK(task) (task == TBP_NOPASS)

/** \brief Macro for task that adds non_overridable attribute to tbp */
#define IS_ADD_NON_OVERRIDABLE_TBP_TASK(task) (task == TBP_NONOVERRIDABLE)

/** \brief Macro for task that adds private attribute to tbp */
#define IS_ADD_PRIVATE_TBP_TASK(task) (task == TBP_PRIVATE)

/** \brief Macro for task that adds public attribute to tbp */
#define IS_ADD_PUBLIC_TBP_TASK(task) (task == TBP_PUBLIC)

/** \brief Macro for task that checks for tbps in a particular derived type */
#define IS_TBP_STATUS_TASK(task) (task == TBP_STATUS)

/** \brief Macro for task that adds deferred attribute to tbp */
#define IS_ADD_DEFERRED_TBP_TASK(task) (task == TBP_DEFERRED)

/** \brief Macro for task that adds explicit external interface for tbp's
    implementation */
#define IS_ADD_EXPLICIT_IFACE_TBP_TASK(task) (task == TBP_IFACE)

/** \brief Macro for task that adds a final subroutine to tbp queue */
#define IS_ADD_FINAL_SUB_TASK(task) (task == TBP_ADD_FINAL)

/* Forward declaration of internal type bound procedure ADT */
struct tbp;
typedef struct tbp TBP;

/** \brief Enum used to hold private or public accessibility of tbp */
typedef enum tbpAccessTypes {
  DEFAULT_ACCESS_TBP = 0, /**< no access attribute specified */
  PRIVATE_ACCESS_TBP,     /**< private access attribute specified */
  PUBLIC_ACCESS_TBP       /**< public access attribute specified */
} tbpAccess;

static int initTbp(tbpTask task);
static int clearStaleRecs(tbpTask task);
static int enqueueTbp(int sptr, int bind, int offset, int dtype, tbpTask task);
static int addToDtype(int dtype, tbpTask task);
static int inheritTbps(int dtype, tbpTask task);
static int resolveTbps(int dtype, tbpTask task);
static int semCheckTbp(tbpTask task, TBP *curr, char *impName);
static int initFinalSub(tbpTask task, TBP *curr);
static int resolveImp(int dtype, tbpTask task, TBP *curr, char *impName);
static int resolveBind(tbpTask task, TBP *curr, char *impName);
static int addTbpInterface(int sptr, int dtype, tbpTask task);
static int addPassArg(int sptr, int bind, int dtype, tbpTask task);
static int addNoPassArg(int bind, int dtype, tbpTask task);
static int addNonOverridableTbp(int bind, int dtype, tbpTask task);
static int addPrivateAttribute(int bind, int dtype, tbpTask task);
static int addPublicAttribute(int bind, int dtype, tbpTask task);
static int dtypeHasTbp(int dtype, tbpTask task);
static int addDeferredTbp(int bind, int dtype, tbpTask task);
static int addExplicitIface(int sptr, tbpTask task);
static int addFinalSubroutine(SPTR sptr, DTYPE dtype, tbpTask task);
static void fixupImp(int next_sptr, TBP *curr);
static int resolvePass(tbpTask task, TBP *curr, char *impName);
static int resolveGeneric(tbpTask task, TBP *curr);
static void completeTbp(TBP *curr);
static int requiresOverloading(int sym, TBP *curr, tbpTask task);
#if DEBUG
static void checkForStaleTbpEntries(void);
#endif

/** \brief Internal type bound procedure ADT */
struct tbp {
  char *impName;  /**< implementation name */
  char *bindName; /**< binding name */
  int offset;     /**< offset into "virtual function table" */
  int dtype;      /**< derived type dtype holding this tbp */
  int dtPass;     /**< dtype of the pass argument */
  int impSptr;    /**< procedure implementation symbol table pointer */
  int bindSptr;   /**< binding name symbol table pointer */
  int memSptr;    /**< symbol table pointer of ST_MEMBER that holds this tbp */
  int isIface;    /**< set if impName specifies an interface-name */
  int lineno;     /**< source lineno of declaration */
  int pass;       /**< symbol table pointer for the PASS object argument */
  int hasNopass;  /**< set when the nopass attribute is specified for tbp */
  int isNonOver; /**< set when non_overridable attribute is specified for tbp */
  tbpAccess access; /**< set to specified access attribute */
  int isDeferred;   /**< set if tbp has deferred attribute */
  int isExtern; /**< set if explicit external interface specified for impSptr */
  int isInherited;  /**< set if we inherited this type bound procedure */
  int genericType;  /**< set to a generic stype if tbp is a generic tbp; else 0
                       */
  int isOverloaded; /**< set if this tbp overloads a parent tbp */
  int isFinal;      /**< set if this is a final subroutine */
  int isFwdRef;     /**< set when we declare sptr to ST_ENTRY for 1st. time */
  int isDefinedIO;  /**< set when this is a defined I/O generic tbp*/
  struct tbp *next; /**< next tbp in list */
};

/** \brief The internal tbp queue */
static TBP *tbpQueue = NULL;

/** \brief Main function for processing type bound procedures (tbps),
  * generic type bound procedures, derived type I/O generic type bound
  * procedures, and final subroutines.
  *
  * \param sptr is the symbol table pointer of the implementation name of tbp
  * \param bind symbol table pointer to the binding name of tbp
  * \param offset is the integer offset of tbp in its "virtual function table"
  * \param dtype is the enclosing dtype of the type bound procedure
  * \param task is the type of processing to be performed (see tbpTask
  *        definition in semant.h)
  */
int
queue_tbp(int sptr, int bind, int offset, int dtype, tbpTask task)
{
  if (IS_CLEAR_TBP_TASK(task)) {

    /* init/clear entries in tbpQueue */

    return initTbp(task);

  } else if (IS_CLEAR_STALE_RECORDS_TASK(task)) {
    
    /* delete records of stale dtypes. Zero out stale sptr fields */

    return clearStaleRecs(task);

  }else if (IS_ADD_TBP_TASK(task)) {

    /* add entry to tbpQueue */

    return enqueueTbp(sptr, bind, offset, dtype, task);

  } else if (IS_ADD_TBP_TO_DTYPE_TASK(task)) {

    /* add tbps to their encapsulating derived types */

    return addToDtype(dtype, task);

  } else if (IS_INHERIT_TBP_TASK(task)) {

    /* inherit (copy) tbps from parent to child (when applicable) */

    return inheritTbps(dtype, task);

  } else if (IS_RESOLVE_TBP_TASK(task)) {

    /* resolve each tbp's implementation name symbol, binding name symbol,
     * and semantically check its arguments, interface, etc.
     */

    return resolveTbps(dtype, task);

  } else if (IS_ADD_TBP_INTERFACE_TASK(task)) {

    /* Add interface-name if user specified it with the procedure */

    return addTbpInterface(sptr, dtype, task);

  } else if (IS_ADD_TBP_PASS_ARG_TASK(task)) {

    /* add pass argument for tbp if it has not already been added */

    return addPassArg(sptr, bind, dtype, task);

  } else if (IS_ADD_TBP_NOPASS_ARG_TASK(task)) {

    /* set nopass attribute on tbp */

    return addNoPassArg(bind, dtype, task);

  } else if (IS_ADD_NON_OVERRIDABLE_TBP_TASK(task)) {

    /* set non_overridable attribute on tbp */

    return addNonOverridableTbp(bind, dtype, task);

  } else if (IS_ADD_PRIVATE_TBP_TASK(task)) {

    /* Add specified private attribute to tbp */

    return addPrivateAttribute(bind, dtype, task);

  } else if (IS_ADD_PUBLIC_TBP_TASK(task)) {

    /* Add specified public attribute to tbp */

    return addPublicAttribute(bind, dtype, task);

  } else if (IS_TBP_STATUS_TASK(task)) {

    /* Check to see if there exists a tbp for specified dtype */

    return dtypeHasTbp(dtype, task);

  } else if (IS_ADD_DEFERRED_TBP_TASK(task)) {

    /* Add deferred attribute to tbp */

    return addDeferredTbp(bind, dtype, task);

  } else if (IS_ADD_EXPLICIT_IFACE_TBP_TASK(task)) {

    /* Check to see if tbp uses an implementation that is specified as
     * an explicit interface to an external routine.
     */

    return addExplicitIface(sptr, task);

  } else if (IS_ADD_FINAL_SUB_TASK(task)) {

    /* add final subroutine to tbpQueue */

    return addFinalSubroutine(sptr, dtype, task);
  }
  return 0;
}

/** \brief Called by resolveTbps(). This function initializes a final
   * subroutine.
   *
   * We also perform some intial semantic checking.
   *
   * \param task is the task that invoked this function
   * \param curr is tbp that we're currently processing
   * \return integer -1 if we processed a final subroutine, otherwise 0.
   */
static int
initFinalSub(tbpTask task, TBP *curr)
{

  int sym, paramct, dpdsc, psptr, dtype, mem, rank, tag;

  if (curr->isFinal) {
    sym = paramct = dpdsc = 0;
    if (!sem.mod_cnt) {
      error(155, ERR_Fatal, gbl.lineno, "FINAL subroutine must be a module"
                                " procedure with one dummy argument -",
            SYMNAME(curr->impSptr));
      return -1;
    }
    FINALP(curr->memSptr, -1);
    curr->impSptr = getsymbol(curr->impName);
    VTABLEP(curr->memSptr, curr->impSptr);
    if (!STYPEG(curr->impSptr)) {
      STYPEP(curr->impSptr, ST_ENTRY);
      SCOPEP(curr->impSptr, stb.curr_scope);
    }
    /* FINAL subroutine is always public. See Fortran 2018 7.5.6.1 NOTE 1. */
    PRIVATEP(curr->impSptr, 0);
    if (task == TBP_COMPLETE_ENDMODULE || task == TBP_COMPLETE_FIN) {
      proc_arginfo(curr->impSptr, &paramct, &dpdsc, &sym);
      if (task == TBP_COMPLETE_FIN && (!sym || !dpdsc || !paramct ||
          (SCOPEG(sym) != stb.curr_scope && sem.which_pass == 0))) {
          return -1;
      }
      if (!INMODULEG(curr->impSptr) || !dpdsc || paramct != 1 || FVALG(sym)) {
        error(155, ERR_Fatal, gbl.lineno, "FINAL subroutine must be a module"
                                  " procedure with one dummy argument -",
              SYMNAME(curr->impSptr));
      }
      psptr = *(aux.dpdsc_base + dpdsc);
      if (CLASSG(psptr)) {
        error(155, ERR_Fatal, gbl.lineno, "Dummy argument for FINAL subroutine"
                                  " must not be polymorphic -",
              SYMNAME(curr->impSptr));
      }
      if (ALLOCATTRG(psptr)) {
        error(155, ERR_Fatal, gbl.lineno, "Dummy argument for FINAL subroutine"
                                  " must not be allocatable -",
              SYMNAME(curr->impSptr));
      }
      if (POINTERG(psptr)) {
        error(155, ERR_Fatal, gbl.lineno, "Dummy argument for FINAL subroutine"
                                  " must not be a pointer -",
              SYMNAME(curr->impSptr));
      }
      if (OPTARGG(psptr)) {
        error(155, ERR_Fatal, gbl.lineno, "Dummy argument for FINAL subroutine"
                                  " must not be optional -",
              SYMNAME(curr->impSptr));
      }
      if (INTENTG(psptr) == INTENT_OUT) {
        error(155, ERR_Fatal, gbl.lineno, "Dummy argument for FINAL subroutine"
                                  " must not be INTENT(OUT) -",
              SYMNAME(curr->impSptr));
      }
      if (PASSBYVALG(psptr)) {
        error(155, ERR_Fatal, gbl.lineno, "Dummy argument for FINAL subroutine"
                                  " must not have VALUE attribute -",
              SYMNAME(curr->impSptr));
      }
      dtype = DTYPEG(psptr);
      if (is_array_dtype(dtype)) {
        dtype = array_element_dtype(dtype);
        FINALP(curr->memSptr, rank_of_sym(psptr) + 1);
      } else {
        FINALP(curr->memSptr, 1);
      }
      tag = get_struct_tag_sptr(curr->dtype);
      if (dtype != curr->dtype) {
        error(155, ERR_Fatal, curr->lineno, "Type for FINAL subroutine dummy "
                                    "argument does not match derived type",
              SYMNAME(tag));
      }
      for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
        if (CLASSG(mem) && (rank = FINALG(mem)) && mem != curr->memSptr) {
          if (rank == FINALG(curr->memSptr) &&
              ELEMENTALG(VTABLEG(mem)) == ELEMENTALG(curr->impSptr)) {
            error(155, ERR_Fatal, curr->lineno,
                  "Dummy argument for FINAL"
                  " subroutine has same rank as a dummy argument for"
                  " another FINAL subroutine in the same derived type -",
                  SYMNAME(curr->impSptr));
            break;
          }
        }
      }
      sym = getccsym_sc('d', sem.dtemps++, ST_VAR, SC_NONE);
      DTYPEP(sym, curr->dtype);
      set_descriptor_class(1);
      set_final_descriptor(1);
      get_static_descriptor(sym);
      set_descriptor_class(0);
      set_final_descriptor(0);
    }
    return -1;
  }

  return 0;
}

/** \brief Called by resolveTbps(). This function will check the validity
   * of the type bound procedure (tbp), curr, including its arguments and
   * interface.
   *
   * \param task is the task that invoked this function
   * \param curr is tbp that we're currently processing
   * \param impName is a C string that contains the implementation name suitable
   *  for printing in an error message
   *
   * \return integer 0 if no errors found, > 0 if error found, -1 if this
   * function chose to not check the tbp for various reasons
   * (see comments in function code).
   */
static int
semCheckTbp(tbpTask task, TBP *curr, char *impName)
{
  int sym, sym2, i, tag, dty;
  int errCnt, parent;

  if (task == TBP_COMPLETE_ENDMODULE || task == TBP_COMPLETE_ENDTYPE ||
      task == TBP_COMPLETE_END || task == TBP_FORCE_RESOLVE) {
    /* Check the arguments/interface of type bound procedure */
    int dpdsc, paramct, psptr, pass;
    int dpdsc2, paramct2, psptr2, pass2;
    int imp, mem;
    SPTR iface = IFACEG(curr->memSptr);

    errCnt = 0;
    sym = iface ? iface : curr->impSptr;
    if (curr->isInherited) {
      /* use inherited (parent's) sym if this type bound
       * procedure is not overloaded.
       */
      parent = curr->dtype;
      parent = DTYPEG(PARENTG(get_struct_tag_sptr(parent)));
      mem = 0;
      get_implementation(parent, iface ? sym : curr->bindSptr, 0, &mem);
      if (mem)
        sym = VTABLEG(mem);
    }
    get_next_hash_link(sym, 0);
  check_next_sptr_3:
    proc_arginfo(sym, &paramct, &dpdsc, &sym);
    if (!dpdsc && STYPEG(sym) == ST_ENTRY) {
      sym2 = findByNameStypeScope(SYMNAME(sym), ST_PROC, 0);
      if (sym2) {
        /* Replace sym with sym2's interface */
        proc_arginfo(sym2, &paramct, &dpdsc, &sym);
      }
    }
    if (!dpdsc) {
      sym = get_next_hash_link(sym, 1);
      if (sym)
        goto check_next_sptr_3;
      if (task == TBP_COMPLETE_ENDTYPE ||
          (STYPEG(sym) != ST_PROC && STYPEG(sym) != ST_ENTRY))
        return -1; /* may not have seen interface yet */
      if (!sem.which_pass && IN_MODULE && task == TBP_COMPLETE_ENDMODULE)
        return -1; /* may not have seen interface yet */
      if (!sem.which_pass && !IN_MODULE && task == TBP_COMPLETE_END)
        return -1; /* may not have seen interface yet */
      error(155, 4, gbl.lineno, "Missing interface for type bound procedure",
            impName);
      ++errCnt;
    }
    pass = 0;
    if (!curr->hasNopass) {
      if (curr->pass) {
        for (i = 0; dpdsc && i < paramct; dpdsc++) {
          psptr = *(aux.dpdsc_base + dpdsc);
          dty = DTYPEG(psptr);
          if (strcmp(SYMNAME(curr->pass), SYMNAME(psptr)) == 0) {
            pass = i + 1;
            if (!CLASSG(psptr)) {

              sym = get_next_hash_link(sym, 1);
              if (sym)
                goto check_next_sptr_3;

              error(155, 3, gbl.lineno, "PASS argument must be declared"
                                        " CLASS in",
                    impName);
              ++errCnt;
            }
            if (ALLOCATTRG(psptr) || POINTERG(psptr) || DTY(dty) == TY_ARRAY) {
              sym = get_next_hash_link(sym, 1);
              if (sym)
                goto check_next_sptr_3;
              error(155, 3, gbl.lineno, "PASS argument must be scalar,"
                                        " nonpointer, and nonallocatable in",
                    impName);
              ++errCnt;
            }
            if (DTY(dty) == TY_ARRAY) {
              dty = DTY(dty + 1);
            }

            if (PASSBYVALG(psptr)) {
              sym = get_next_hash_link(sym, 1);
              if (sym)
                goto check_next_sptr_3;
              error(155, 3, gbl.lineno, "PASS argument cannot have"
                                        " the VALUE attribute in",
                    impName);
              ++errCnt;
            }
            break;
          }
          ++i;
        }
        if (!pass) {
          error(155, 3, gbl.lineno, "Invalid argument name for PASS attribute"
                                    " in type bound procedure",
                impName);
          ++errCnt;
        }
      } else if (!curr->genericType) {
        psptr = (paramct) ? *(aux.dpdsc_base + dpdsc) : 0;
        dty = DTYPEG(psptr);
        if (paramct && !CLASSG(psptr)) {
          sym = get_next_hash_link(sym, 1);
          if (sym)
            goto check_next_sptr_3;
          error(155, 3, gbl.lineno, "PASS argument must be declared"
                                    " CLASS in",
                impName);
          ++errCnt;
        }
        if (ALLOCATTRG(psptr) || POINTERG(psptr) || DTY(dty) == TY_ARRAY) {
          sym = get_next_hash_link(sym, 1);
          if (sym)
            goto check_next_sptr_3;
          error(155, 3, gbl.lineno, "PASS argument must be scalar,"
                                    " nonpointer, and nonallocatable in",
                impName);
          ++errCnt;
        }
        if (DTY(dty) == TY_ARRAY) {
          dty = DTY(dty + 1);
        }

        if (PASSBYVALG(psptr)) {
          sym = get_next_hash_link(sym, 1);
          if (sym)
            goto check_next_sptr_3;
          error(155, 3, gbl.lineno, "PASS argument cannot have"
                                    " the VALUE attribute in",
                impName);
          ++errCnt;
        }
      }
    }
    tag = get_struct_tag_sptr(curr->dtype);
    if (curr->genericType == ST_OPERATOR) {
      /* Check generic interface against rules for Defined Operators/
       * Defined Assignments.
       */
      mem = get_specific_member(curr->dtype, VTABLEG(curr->memSptr));
      imp = VTABLEG(mem);
      if (imp) {
        char *buf;
        int len, imp2;
        proc_arginfo(imp, &paramct, &dpdsc, &imp2);
        if (!dpdsc && STYPEG(imp2) == ST_ENTRY) {
          sym2 = findByNameStypeScope(SYMNAME(imp2), ST_PROC, 0);
          if (sym2) {
            proc_arginfo(sym2, &paramct, &dpdsc, 0);
          }
        }
        if (!dpdsc) {
          if (task == TBP_COMPLETE_ENDTYPE)
            return -1; /* may not have seen interface yet */
          error(155, 4, gbl.lineno, "Missing interface for type bound"
                                    " procedure",
                impName);
          ++errCnt;
        } else if (strcmp(SYMNAME(curr->bindSptr), "=") == 0) {
          /* check interface of imp against rules for
           * Defined Assignments.
           */
          if (paramct != 2) {
            len = strlen("Defined assignment '") + strlen(SYMNAME(imp)) +
                  strlen("' in type '") + strlen(SYMNAME(tag)) +
                  strlen("' must have exactly two dummy arguments") + 1;
            buf = getitem(0, len);
            sprintf(buf, "Defined assignment '%s' in type '%s'"
                         " must have exactly two dummy arguments",
                    SYMNAME(imp), SYMNAME(tag));
            error(155, 3, gbl.lineno, buf, NULL);
            ++errCnt;
          } else {
            psptr = *(aux.dpdsc_base + dpdsc);
            psptr2 = *(aux.dpdsc_base + (dpdsc + 1));
            if (INTENTG(psptr) != INTENT_OUT &&
                INTENTG(psptr) != INTENT_INOUT) {
              len =
                  strlen("First dummy argument in defined "
                         "assignment '") +
                  strlen(SYMNAME(imp)) + strlen("' of type '") +
                  strlen(SYMNAME(tag)) +
                  strlen("' must have INTENT(OUT) or INTENT(INOUT) attribute") +
                  1;
              buf = getitem(0, len);
              sprintf(buf, "First dummy argument in defined "
                           "assignment '%s' of type '%s' must have "
                           "INTENT(OUT) or INTENT(INOUT) attribute",
                      SYMNAME(imp), SYMNAME(tag));
              error(155, 3, gbl.lineno, buf, NULL);
              ++errCnt;
            }
            if (INTENTG(psptr2) != INTENT_IN) {
              len = strlen("Second dummy argument in defined "
                           "assignment '") +
                    strlen(SYMNAME(imp)) + strlen("' of type '") +
                    strlen(SYMNAME(tag)) +
                    strlen("' must have INTENT(IN) attribute") + 1;
              buf = getitem(0, len);
              sprintf(buf, "Second dummy argument in defined "
                           "assignment '%s' of type '%s' must have "
                           "INTENT(IN) attribute",
                      SYMNAME(imp), SYMNAME(tag));
              error(155, 3, gbl.lineno, buf, NULL);
              ++errCnt;
            }
            if (OPTARGG(psptr) || OPTARGG(psptr2)) {
              len = strlen("Optional arguments in defined "
                           "assignment '") +
                    strlen(SYMNAME(imp)) + strlen("' of type '") +
                    strlen(SYMNAME(tag)) + strlen("' not allowed") + 1;
              buf = getitem(0, len);
              sprintf(buf, "Optional arguments in defined "
                           "assignment '%s' of type '%s' not allowed",
                      SYMNAME(imp), SYMNAME(tag));
              error(155, 3, gbl.lineno, buf, NULL);
              ++errCnt;
            }
          }
        } else {
          /* check interface of imp against rules for
           * Defined Operations.
           */
          if (paramct < 1 || paramct > 2) {
            len = strlen("Defined operation '") + strlen(SYMNAME(imp)) +
                  strlen("' in type '") + strlen(SYMNAME(tag)) +
                  strlen("' must have one or two dummy "
                         "arguments") +
                  1;
            buf = getitem(0, len);
            sprintf(buf, "Defined operation '%s' in type '%s'"
                         " must have one or two dummy arguments",
                    SYMNAME(imp), SYMNAME(tag));
            error(155, 3, gbl.lineno, buf, NULL);
            ++errCnt;
          } else {
            psptr = *(aux.dpdsc_base + dpdsc);
            if (paramct == 2)
              psptr2 = *(aux.dpdsc_base + (dpdsc + 1));
            else
              psptr2 = 0;
            if (OPTARGG(psptr) || (psptr2 && OPTARGG(psptr2))) {
              len = strlen("Optional arguments in defined "
                           "operation '") +
                    strlen(SYMNAME(imp)) + strlen("' of type '") +
                    strlen(SYMNAME(tag)) + strlen("' not allowed") + 1;
              buf = getitem(0, len);
              sprintf(buf, "Optional arguments in defined "
                           "operation '%s' of type '%s' not allowed",
                      SYMNAME(imp), SYMNAME(tag));
              error(155, 3, gbl.lineno, buf, NULL);
              ++errCnt;
            }
            if (INTENTG(psptr) != INTENT_IN ||
                (psptr2 && INTENTG(psptr2) != INTENT_IN)) {
              len = strlen("Dummy arguments in defined "
                           "operation '") +
                    strlen(SYMNAME(imp)) + strlen("' of type '") +
                    strlen(SYMNAME(tag)) +
                    strlen("' must have INTENT(IN) attribute") + 1;
              buf = getitem(0, len);
              sprintf(buf, "Dummy arguments in defined "
                           "operation '%s' of type '%s' must have "
                           "INTENT(IN) attribute",
                      SYMNAME(imp), SYMNAME(tag));
              error(155, 3, gbl.lineno, buf, NULL);
              ++errCnt;
            }
          }
        }
      }
    }
    parent = DTYPEG(PARENTG(tag));
    if (parent && !curr->genericType) {
      /* Check interface with parent's interface */
      mem = 0;
      imp = get_implementation(parent, curr->bindSptr, 0, &mem);
      if (imp && mem && !NONOVERG(mem)) {
        if ((NOPASSG(mem) && !NOPASSG(curr->memSptr)) ||
            (NOPASSG(curr->memSptr) && !NOPASSG(mem))) {
          error(155, 3, gbl.lineno, "PASS/NOPASS must be consistent with"
                                    " parent's type bound procedure in",
                impName);
          ++errCnt;
        }
        proc_arginfo(imp, &paramct2, &dpdsc2, &imp);
        proc_arginfo(sym, &paramct, &dpdsc, &sym);
        if (PASSG(mem) && PASSG(curr->memSptr) &&
            strcmp(SYMNAME(PASSG(mem)), SYMNAME(PASSG(curr->memSptr)))) {
          error(155, 3, gbl.lineno,
                "PASS argument name must match "
                "parent's PASS argument name in type bound procedure",
                impName);
          ++errCnt;
        } else if ((!NOPASSG(mem) && !PASSG(mem)) ||
                   (!NOPASSG(curr->memSptr) && !PASSG(curr->memSptr))) {
          psptr2 = *(aux.dpdsc_base + dpdsc2);
          psptr = *(aux.dpdsc_base + dpdsc);
          if (strcmp(SYMNAME(psptr2), SYMNAME(psptr))) {
            error(155, 3, gbl.lineno,
                  "PASS argument name/position must match "
                  "parent's PASS argument name/position in type bound "
                  "procedure",
                  impName);
            ++errCnt;
          }
        }
        pass2 = find_dummy_position(imp, PASSG(mem));
        if (PASSG(mem) && PASSG(curr->memSptr) && pass != pass2) {
          error(155, 3, gbl.lineno,
                "PASS argument is incompatible with"
                " parent's PASS argument in type bound procedure",
                impName);
          ++errCnt;
        }
        if (!curr->genericType) {
          get_next_hash_link(imp, 0);
          get_next_hash_link(sym, 0);
        check_next_sptr_4:
          proc_arginfo(imp, &paramct2, &dpdsc2, &imp);
          proc_arginfo(sym, &paramct, &dpdsc, &sym);
          /* check tbp overriding rules */
          if (!dpdsc || !dpdsc2 || paramct2 != paramct ||
              (FVALG(imp) && !FVALG(sym)) || (FVALG(sym) && !FVALG(imp)) ||
              (!PUREG(sym) && PUREG(imp)) ||
              ELEMENTALG(sym) != ELEMENTALG(imp)) {
            if (!dpdsc2) {
              imp = get_next_hash_link(imp, 1);
              if (imp)
                goto check_next_sptr_4;
            } else if (!dpdsc) {
              sym = get_next_hash_link(sym, 1);
              if (sym)
                goto check_next_sptr_4;
            }
            error(155, 3, gbl.lineno,
                  "Interface is not compatible "
                  "with parent's interface for type bound procedure",
                  impName);
            ++errCnt;
          } else if (FVALG(imp) && FVALG(sym)) {
            psptr = FVALG(imp);
            psptr2 = FVALG(sym);
            if (CLASSG(psptr) != CLASSG(psptr2) ||
                ALLOCATTRG(psptr) != ALLOCATTRG(psptr2) ||
                POINTERG(psptr) != POINTERG(psptr2)) {
              error(155, 3, gbl.lineno,
                    "Result is not compatible "
                    "with parent's result for type bound procedure",
                    impName);
              ++errCnt;
            } else if (!eq_dtype2(DTYPEG(psptr), DTYPEG(psptr2), 0)) {
              if (DTY(DTYPEG(psptr)) == TY_CHAR &&
                  DTY(DTYPEG(psptr2)) == TY_CHAR) {
                if (ADJLENG(psptr) != ADJLENG(psptr2)) {
                  error(155, 3, gbl.lineno,
                        "Character result differs "
                        "with parent's character result for type bound "
                        "procedure",
                        impName);
                  ++errCnt;
                }
              } else {
                error(155, 3, gbl.lineno,
                      "Result is not compatible "
                      "with parent's result for type bound procedure",
                      impName);
                ++errCnt;
              }
            }
          }
          for (i = 0; i < paramct; ++dpdsc, ++dpdsc2, ++i) {
            psptr2 = *(aux.dpdsc_base + dpdsc2);
            psptr = *(aux.dpdsc_base + dpdsc);
            if ((pass && i == (pass - 1)) ||
                (i == 0 &&
                 ((!NOPASSG(mem) && !PASSG(mem)) || (pass == 0 && pass2 == 1) ||
                  (pass2 == 0 && pass == 1)))) {

              if (INTENTG(psptr) != INTENTG(psptr2) ||
                  OPTARGG(psptr) != OPTARGG(psptr2) ||
                  ALLOCATTRG(psptr) != ALLOCATTRG(psptr2) ||
                  PASSBYVALG(psptr) != PASSBYVALG(psptr2) ||
                  ASYNCG(psptr) != ASYNCG(psptr2) ||
                  VOLG(psptr) != VOLG(psptr2) ||
                  CLASSG(psptr) != CLASSG(psptr2) ||
                  POINTERG(psptr) != POINTERG(psptr2) ||
                  TARGETG(psptr) != TARGETG(psptr2)) {
                /* check characteristics of dummy args */
                error(155, 3, gbl.lineno,
                      "Interface is not compatible "
                      "with parent's interface for type bound procedure",
                      impName);
                ++errCnt;
              }
              return errCnt;
            }
            if (STYPEG(psptr) == ST_PROC && STYPEG(psptr2) == ST_PROC) {
              if (!cmp_interfaces_strict(psptr, psptr2, IGNORE_IFACE_NAMES)) {
                error(155, 3, gbl.lineno,
                      "Interface is not compatible with "
                      "parent's interface for type bound procedure",
                      impName);
                ++errCnt;
              }
            } else if (sem.which_pass &&
                       !eq_dtype2(DTYPEG(psptr), DTYPEG(psptr2), 0)) {
              error(155, 3, gbl.lineno,
                    "Interface is not compatible with "
                    "parent's interface for type bound procedure",
                    impName);
              ++errCnt;
            } else if (sem.which_pass &&
                       (INTENTG(psptr) != INTENTG(psptr2) ||
                        OPTARGG(psptr) != OPTARGG(psptr2) ||
                        ALLOCATTRG(psptr) != ALLOCATTRG(psptr2) ||
                        PASSBYVALG(psptr) != PASSBYVALG(psptr2) ||
                        ASYNCG(psptr) != ASYNCG(psptr2) ||
                        VOLG(psptr) != VOLG(psptr2) ||
                        CLASSG(psptr) != CLASSG(psptr2) ||
                        POINTERG(psptr) != POINTERG(psptr2) ||
                        TARGETG(psptr) != TARGETG(psptr2))) {
              /* check characteristics of dummy args */
              error(155, 3, gbl.lineno,
                    "Interface is not compatible "
                    "with parent's interface for type bound procedure",
                    impName);
              ++errCnt;
            } else if (strcmp(SYMNAME(psptr), SYMNAME(psptr2)) != 0) {
              error(155, 3, gbl.lineno,
                    "Dummy argument names "
                    "must be the same as those in parent's interface "
                    "for type bound procedure",
                    impName);
              ++errCnt;
            }
          }
        }
      }
    }
  } else
    return -1; /* ignore, called with invalid task */

  return errCnt;
}

/** \brief Set curr->impSptr field to correct implementation symbol.
  *
  * This function is called by resolvePass() when we determine that we
  * have the wrong implementation symbol for the curr->impSptr field. This
  * can happen when curr->impSptr was initially set to a forward reference,
  * ST_ENTRY, instead of a ST_PROC.
  *
  * \param next_sptr is the correct symbol table pointer for the implementation
  * \param curr is the current type bound procedure record that we're
  *        processing.
  */
static void
fixupImp(int next_sptr, TBP *curr)
{

  if (next_sptr != curr->impSptr) {
    curr->impSptr = next_sptr;
    ADDRTKNP(next_sptr, 1);
    VTABLEP(curr->memSptr, curr->impSptr); /* implementation name */
    MSCALLP(curr->impSptr, getMscall());
#ifdef CREFP
    CREFP(curr->impSptr, getCref());
#endif

    if (curr->isIface) {
      IFACEP(curr->memSptr, curr->impSptr);
    }

    if (DTYPEG(curr->impSptr)) {
      /* Set member type to return type of function to ensure
       * correct I/O
       */
      DTYPEP(curr->memSptr, DTYPEG(curr->impSptr));
    }

    CLASSP(curr->impSptr, 1);

    if (!curr->isExtern && IN_MODULE)
      INMODULEP(curr->impSptr, 1);
  }
}

/** \brief Resolve the PASS object/argument of a type bound procedure (tbp).
  *
  * This includes the default and explicit pass objects. We also set NOPASS
  * on tbp derived type component of nopass was specified.
  *
  * \param task is the task that called this function.
  * \param curr is the current tbp record.
  * \param impName is a C string that represents the implementation-name
  *        that we are processing.
  *
  * \return 0 if successful, > 0 if an error occurred.
  */
static int
resolvePass(tbpTask task, TBP *curr, char *impName)
{

  int errCnt;
  int paramct, dpdsc;
  errCnt = 0;

  if ((task == TBP_COMPLETE_ENDMODULE ||
       (!IN_MODULE && task == TBP_COMPLETE_END)) &&
      !curr->isInherited && !curr->genericType) {
    if (curr->pass) {
      int psptr, arg;
      int next_sptr;
      next_sptr = curr->impSptr;
      get_next_hash_link(curr->impSptr, 0);
    check_next_sptr_1:
      proc_arginfo(next_sptr, &paramct, &dpdsc, NULL);
      if (dpdsc && paramct) {
        for (arg = 0; arg < paramct; ++arg) {
          psptr = *(aux.dpdsc_base + (dpdsc + arg));
          if (strcmp(SYMNAME(psptr), SYMNAME(curr->pass)) == 0) {
            int dty;
            dty = DTYPEG(psptr);
            if (DTY(dty) == TY_ARRAY)
              dty = DTY(dty + 1);
            if (!eq_dtype(curr->dtype, dty)) {
              char *buf;
              int tag, len;
              next_sptr = get_next_hash_link(curr->impSptr, 1);
              if (next_sptr)
                goto check_next_sptr_1;
              tag = get_struct_tag_sptr(curr->dtype);
              len =
                  strlen("Passed object dummy argument '' in type bound") +
                  strlen(" procedure") +
                  strlen(" '' of type '' must be declared with data type ''") +
                  strlen(SYMNAME(curr->pass)) + strlen(impName) +
                  (2 * strlen(SYMNAME(tag))) + 1;
              buf = getitem(0, len);
              sprintf(buf, "Passed object dummy argument '%s' in type bound "
                           "procedure"
                           " '%s' of type '%s' must be declared with data type"
                           " '%s'",
                      SYMNAME(curr->pass), impName, SYMNAME(tag), SYMNAME(tag));
              error(155, 3, gbl.lineno, buf, NULL);
              ++errCnt;
            } else {
              fixupImp(next_sptr, curr);
            }
            break;
          }
        }
      }
    } else if (!curr->pass && !curr->hasNopass) {
      int psptr, dty;
      int next_sptr;
      paramct = dpdsc = 0;
      next_sptr = curr->impSptr;
      get_next_hash_link(curr->impSptr, 0);
    check_next_sptr_2:
      proc_arginfo(next_sptr, &paramct, &dpdsc, NULL);
      if (dpdsc) {
        psptr = (paramct) ? *(aux.dpdsc_base + dpdsc) : 0;
        dty = DTYPEG(psptr);
        if (DTY(dty) == TY_ARRAY)
          dty = DTY(dty + 1);
        if (paramct && !eq_dtype(curr->dtype, dty)) {
          char *buf;
          int tag, len;

          next_sptr = get_next_hash_link(next_sptr, 1);
          if (next_sptr)
            goto check_next_sptr_2;
          tag = get_struct_tag_sptr(curr->dtype);
          len = strlen("Passed object dummy argument '' in type bound") +
                strlen(" procedure") +
                strlen(" '' of type '' must be declared with data type ''") +
                strlen(SYMNAME(psptr)) + strlen(impName) +
                (2 * strlen(SYMNAME(tag))) + 1;
          buf = getitem(0, len);
          sprintf(buf,
                  "Passed object dummy argument '%s' in type bound "
                  "procedure"
                  " '%s' of type '%s' must be declared with data type '%s'",
                  SYMNAME(psptr), impName, SYMNAME(tag), SYMNAME(tag));
          error(155, 3, gbl.lineno, buf, NULL);
          ++errCnt;
        } else if (!paramct) {
          char *buf;
          int len;
          SPTR tag = get_struct_tag_sptr(curr->dtype);
          len = strlen("Missing passed object dummy argument in type") +
                strlen(" bound procedure") +
                strlen(" '' of type '' (or NOPASS attribute is needed)") +
                strlen(impName) + strlen(SYMNAME(tag)) + 1;
          buf = getitem(0, len);
          sprintf(buf, "Missing passed object dummy argument in type"
                       " bound procedure"
                       " '%s' of type '%s' (or NOPASS attribute is needed)",
                  impName, SYMNAME(tag));
          error(155, 3, gbl.lineno, buf, NULL);
          ++errCnt;
        } else {
          fixupImp(next_sptr, curr);
        }
      }
    }
  }
  PASSP(curr->memSptr, curr->pass);
  NOPASSP(curr->memSptr, curr->hasNopass);

  return errCnt;
}

/** \brief Resolve generic symbols for generic type bound procedures.
  *
  * We also add specific type bound procedures to generic sets in this
  * function.
  *
  * \param task is the task that called this function.
  * \param curr is the current tbp record.
  *
  * \return 0 if successful, > 0 if an error occurred.
  */
static int
resolveGeneric(tbpTask task, TBP *curr)
{
  int errCnt;
  TBP *curr2;

  errCnt = 0;
  if (curr->genericType &&
      (task == TBP_COMPLETE_ENDMODULE || task == TBP_COMPLETE_END ||
       task == TBP_COMPLETE_GENERIC || task == TBP_FORCE_RESOLVE)) {
    for (curr2 = tbpQueue; curr2; curr2 = curr2->next) {
      if (curr != curr2 && curr2->genericType && curr2->dtype == curr->dtype &&
          strcmp(curr->bindName, curr2->bindName) == 0 &&
          PRIVATEG(curr->memSptr) != PRIVATEG(curr2->memSptr)) {
        char *buf;
        SPTR tag = get_struct_tag_sptr(curr->dtype);
        int len;
        len = strlen("Inconsistent access specified with one or more"
                     " generic-bindings for ") +
              strlen(curr->bindName) + strlen(" in derived type ") +
              strlen(SYMNAME(tag)) + 1;

        buf = getitem(0, len);
        sprintf(buf, "Inconsistent access specified with one or more"
                     " generic-bindings for %s in derived type %s",
                curr->bindName, SYMNAME(tag));

        error(155, 3, gbl.lineno, buf, NULL);

        ++errCnt;
        break;
      }
    }
    if (!curr->isInherited) {
      sem.defined_io_type = curr->isDefinedIO;
      add_overload(curr->bindSptr, curr->impSptr);
      sem.defined_io_type = 0;
    }
    if ((task == TBP_COMPLETE_ENDMODULE || task == TBP_COMPLETE_END ||
         task == TBP_FORCE_RESOLVE) &&
        (STYPEG(curr->bindSptr) == ST_USERGENERIC ||
         STYPEG(curr->bindSptr) == ST_OPERATOR)) {
      int mem;
      char *buf, *name;
      int len, tag, once;
      once = 0;
    gen_again:
      /* for a generic TBP, impSptr is a binding name */
      get_implementation(curr->dtype, curr->impSptr, 0, &mem);
      if (STYPEG(BINDG(mem)) == ST_OPERATOR ||
          STYPEG(BINDG(mem)) == ST_USERGENERIC) {
        mem = get_specific_member(curr->dtype, curr->impSptr);
      }
      if (!mem || STYPEG(BINDG(mem)) != ST_PROC) {
        char *buf2, *buf3;
        if (!sem.which_pass)
          return -1;
        else if (!once) {
          /* Generic bindings may not have been set up.
           * So, try calling queue_tbp() again and set up generic
           * bindings.
           */
          queue_tbp(0, 0, 0, 0, TBP_COMPLETE_GENERIC);
          once = 1;
          goto gen_again;
        }
        buf2 = getitem(0, strlen(SYMNAME(curr->bindSptr)) + 1);
        strcpy(buf2, SYMNAME(curr->bindSptr));
        name = strchr(buf2, '$');
        if (name)
          *name = '\0';
        buf3 = getitem(0, strlen(SYMNAME(curr->impSptr) + 1));
        strcpy(buf3, SYMNAME(curr->impSptr));
        name = strchr(buf3, '$');
        if (name)
          *name = '\0';
        tag = get_struct_tag_sptr(curr->dtype);
        len = strlen("Generic set contains an inherited non-specific"
                     " type bound procedure for ") +
              strlen(buf2) + strlen(" in type called ") + strlen(buf3) + 1;
        buf = getitem(0, len);
        sprintf(buf, "Generic set contains a%s non-specific type bound "
                     "procedure for %s in type %s called %s",
                (curr->isInherited) ? "n inherited" : "", buf2, SYMNAME(tag),
                buf3);
        error(155, 3, gbl.lineno, buf, CNULL);
        ++errCnt;
      }
      if (STYPEG(curr->bindSptr) == ST_OPERATOR && NOPASSG(mem)) {
        tag = get_struct_tag_sptr(curr->dtype);
        len = strlen("NOPASS attribute not allowed for generic type"
                     " bound procedure assignment ") +
              strlen(SYMNAME(curr->bindSptr)) + strlen("in type") + 1;
        buf = getitem(0, len);
        if (strcmp(SYMNAME(curr->bindSptr), "=") != 0) {
          sprintf(buf, "NOPASS attribute not allowed for generic type"
                       " bound procedure operator %s in type",
                  SYMNAME(curr->bindSptr));
        } else {
          sprintf(buf, "NOPASS attribute not allowed for generic type"
                       " bound procedure assignment %s in type",
                  SYMNAME(curr->bindSptr));
        }
        error(155, 3, gbl.lineno, buf, SYMNAME(tag));
        ++errCnt;
      }
      curr->hasNopass = NOPASSG(mem);
      NOPASSP(curr->memSptr, curr->hasNopass);
    }
  }
  return errCnt;
}

/** \brief Called by resolveTbps(). This function resolves the field
   * curr->bindName to the "in scope" binding name symbol table pointer.
   *
   * In a type bound procedure (tbp) expression, we have the following:
   *
   * procedure :: foo => bar
   *
   * In this case, foo is the binding name and bar is the implementation
   * name.
   *
   * \param task is the task that invoked this function
   * \param curr is the tbp record that we're currently processing
   * \param impName is a C string holding the tbp's implementation name that
   * is suitable for printing in an error message
   *
   * \return integer 0 upon success, > 0 if an error occurred, and < 0 if we
   * ignore this tbp.
   */
static int
resolveBind(tbpTask task, TBP *curr, char *impName)
{

  int sym, sym2, errCnt;

  /* complete tbp binding name */

  errCnt = 0;
  sym = getsymbol(curr->bindName);
  while (STYPEG(sym) == ST_ALIAS)
    sym = SYMLKG(sym);
  if (curr->genericType && STYPEG(sym) == ST_USERGENERIC && GNCNTG(sym) &&
      STYPEG(stb.curr_scope) == ST_MODULE && !curr->isInherited &&
      !eq_dtype2(curr->dtype, TBPLNKG(sym), 1) &&
      !eq_dtype2(TBPLNKG(sym), curr->dtype, 1)) {
    sym = insert_sym(sym);
    sym = declsym(sym, curr->genericType, FALSE);
  } else if (TBPLNKG(sym) && TBPLNKG(sym) != curr->dtype &&
             (curr->genericType || !same_ancestor(TBPLNKG(sym), curr->dtype)) &&
             (STYPEG(sym) == ST_PROC || curr->genericType) ) {
    /* This tbp binding name is used in a different context
     * (an unrelated derived type with a different pass argument)
     */
    int old_sym = sym;
    sym = insert_sym(sym);

    if (curr->genericType) {
      dup_sym(sym, stb.stg_base + old_sym);
      if (!same_ancestor(TBPLNKG(sym), curr->dtype)) {
        GNDSCP(sym, 0);
        GNCNTP(sym, 0);
      }
    } else {
      sym = declsym(sym, (curr->genericType) ? curr->genericType : ST_PROC,
                    FALSE);
    }

    VTOFFP(sym, curr->offset);
    CLASSP(sym, 1);
    TBPLNKP(sym, curr->dtype);
    SCOPEP(sym, stb.curr_scope);
    ENCLFUNCP(sym, 0);
    STYPEP(sym, (curr->genericType) ? curr->genericType : ST_PROC);
  } else

      if ((curr->genericType && STYPEG(sym) != ST_USERGENERIC &&
           STYPEG(sym) != ST_OPERATOR) ||
          (!curr->genericType && STYPEG(sym) != ST_PROC)) {
    IGNOREP(sym, 1);
    sym = insert_sym(sym);
    sym =
        declsym(sym, (curr->genericType) ? curr->genericType : ST_PROC, FALSE);
  } else if (!curr->genericType && SCOPEG(sym) != stb.curr_scope &&
             task != TBP_COMPLETE_GENERIC) {
    sym2 = insert_sym(sym);
    STYPEP(sym2, STYPEG(sym));
    sym = sym2;
  } else if (curr->genericType && STYPEG(sym) == ST_OPERATOR) {
    if (SCOPEG(sym) != stb.curr_scope && PRIVATEG(SCOPEG(sym)))
      sym = sym_in_scope(sym, OC_OPERATOR, NULL, NULL, 0);
  }
  curr->bindSptr = sym;

  BINDP(curr->memSptr, sym);
  VTOFFP(sym, curr->offset);
  CLASSP(sym, 1);
  NONOVERP(curr->memSptr, curr->isNonOver);
  if (curr->access == PRIVATE_ACCESS_TBP) {
    PRIVATEP(curr->memSptr, 1);
  } else if (curr->access == PUBLIC_ACCESS_TBP) {
    PRIVATEP(curr->memSptr, 0);
  }
  return errCnt;
}

/** \brief Called by resolveImp() to determine if the implementation symbol is
 *  overloading another symbol that may be in scope through module use
 *  association.
 *
 * Special case for overloading symbol. Execute when
 * we hit a "contains" in a module and symbol is overloadeded or
 * its stype is not set to a ST_PROC, ST_ENTRY, or ST_MODPROC
 * and therefore, needs to be overloaded.
 *
 * \param sym is symbol table pointer of the implementation name that we are
 *        checking.
 * \param curr is a pointer to the type bound procedure (tbp) record that we
 *        are checking.
 * \param task is the TBP task that called this function.
 *
 * \return integer > 0 if sym requires overloading, else 0.
 */
static int
requiresOverloading(int sym, TBP *curr, tbpTask task)
{

  if (task != TBP_COMPLETE_FIN)
    return 0;
  if (curr->isDeferred)
    return 0;
  if (STYPEG(sym) == ST_MODPROC)
    return 0;
  if (!IN_MODULE)
    return 0;
  if (curr->isOverloaded)
    return 1;
  if (STYPEG(sym) != ST_PROC && STYPEG(sym) != ST_ENTRY && PRIVATEG(sym))
    return 1;

  return 0;
}

/** \brief Called by resolveTbps(). This function resolves the field
   * curr->impName to the "in scope" implementation symbol table pointer.
   *
   * That is, we are resolving the implementation name to the symbol table
   * pointer of the implementation procedure.
   *
   * In a tbp expression, we have the following:
   *
   * procedure :: foo => bar
   *
   * In this case, foo is the binding name and bar is the implementation
   * name.
   *
   * \param dtype is the derived type record that we're currently processing
   * \param task is the task that invoked this routine
   * \param curr is the tbp record that we're currently processing
   * \param impName is a C string holding the tbp's implementation name that's
   * suitable for printing in an error message
   *
   * \return integer 0 upon success or > 0 if an error occurred.
   */
static int
resolveImp(int dtype, tbpTask task, TBP *curr, char *impName)
{
  int sym, sym2, errCnt, scope, inmod;

  /* complete tbp implementation */
  errCnt = 0;
  scope = 0;
  sym = 0;

  if (curr->memSptr > 0 && !curr->genericType) {
    scope = SCOPEG(curr->memSptr);
    sym2 = findByNameStypeScope(curr->impName, ST_PROC, scope);
    if (sym2) {
      sym = sym2;
    }
  }
  if (sym == 0)
    sym = getsymbol(curr->impName);

  if (STYPEG(sym) && curr->isInherited && curr->dtPass && !curr->genericType) {
    sym2 = get_implementation(curr->dtPass, curr->bindSptr, 0, NULL);
    if (sym2 && !ABSTRACTG(sym2))
      sym = sym2;
  }

  while (STYPEG(sym) == ST_ALIAS)
    sym = SYMLKG(sym);

  if (task == TBP_COMPLETE_GENERIC && !curr->genericType &&
      STYPEG(sym) == ST_ENTRY && !CLASSG(sym)) {
    int sym2 = findByNameStypeScope(SYMNAME(sym), ST_PROC, 0);
    if (sym2)
      sym = sym2;
  } else if ((!curr->genericType && STYPEG(sym) && STYPEG(sym) != ST_PROC &&
             STYPEG(sym) != ST_ENTRY &&
             STYPEG(sym) != ST_MODPROC) || (!IN_MODULE && !STYPEG(sym))) { 
    int sym2 = findByNameStypeScope(SYMNAME(sym), ST_PROC, 0);
    if (sym2)
      sym = sym2;
  }

  if (curr->isIface && !curr->isDeferred) {
    error(155, 3, gbl.lineno, "DEFERRED attribute required for type bound "
                              "procedure using interface name",
          impName);
    ++errCnt;
  }

  if (!sem.which_pass && STYPEG(sym) == ST_MODPROC && SYMLKG(sym) <= NOSYM) {
    /* Add ignore task on pass 0 to prevent any bogus warnings */
    IGNOREP(sym, 1);
  }

  scope = SCOPEG(sym);
  if (!sem.which_pass && (!STYPEG(sym) ||  
      (!curr->isInherited && PRIVATEG(sym) && IS_PROC(STYPEG(sym)) && 
       scope != stb.curr_scope && SCOPEG(scope) != stb.curr_scope))) {
    SPTR orig_sptr = sym;
    curr->isFwdRef = 1;
    sym = insert_sym(sym);
    if (!STYPEG(orig_sptr)) {
      sym = declsym(sym, ST_ENTRY, FALSE);
    } else {
      sym = declsym_newscope(sym, ST_ENTRY, 0);
    }
    SCP(sym, SC_EXTERN);
    IGNOREP(sym, 1); /* ignore forward reference */
  } else if (sem.which_pass && curr->isFwdRef) {
    while (STYPEG(sym) == ST_ALIAS) {
      sym = SYMLKG(sym);
    }
  } else if (STYPEG(sym) == ST_PD) {
    sym = insert_sym(sym);
    sym = declsym(sym, ST_ENTRY, FALSE);
    SCP(sym, SC_EXTERN);
  } else {
    while (STYPEG(sym) == ST_ALIAS) {
      sym = SYMLKG(sym);
    }
    if (requiresOverloading(sym, curr, task)) {
      if (curr->access != DEFAULT_ACCESS_TBP) {
        sym = insert_sym(sym);
        sym = declsym(sym, ST_ENTRY, FALSE);
      } else {
        inmod = INMODULEG(sym);
        sym = insert_sym(sym);
        sym = declsym(sym, ST_PROC, FALSE);
        INMODULEP(sym, inmod);
      }
      SCP(sym, SC_EXTERN);
    } else if (sem.which_pass && !curr->genericType &&
               (STYPEG(sym) == ST_OPERATOR || STYPEG(sym) == ST_USERGENERIC)) {
      sym = findByNameStypeScope(SYMNAME(sym), ST_PROC, 0);
    } else if (!curr->isInherited && !curr->genericType && PRIVATEG(sym) &&
               !curr->hasNopass && STYPEG(sym) == ST_PROC) {
      /* in case there's another implementation w/ same name in scope,
       * we want to create a new symbol so we don't generate bogus
       * error messages later.
       */
      int paramct, dpdsc, psptr, dty, pass, i;
      proc_arginfo(sym, &paramct, &dpdsc, 0);
      if (curr->pass && dpdsc) {
        pass = -1;
        for (i = 0; i < paramct; ++i) {
          psptr = *(aux.dpdsc_base + dpdsc + i);
          if (strcmp(SYMNAME(psptr), SYMNAME(curr->pass)) == 0) {
            pass = i;
            break;
          }
        }
        if (pass < 0) {
          sym = insert_sym(sym);
          sym = declsym(sym, ST_ENTRY, FALSE);
          SCP(sym, SC_EXTERN);
        }
      } else {
        pass = 0;
      }
      if (dpdsc && pass >= 0) {
        psptr = *(aux.dpdsc_base + dpdsc + pass);
        dty = DTYPEG(psptr);
        if (dty != curr->dtype) {
          if (DCLDG(sym)) {
            /* Since we're processing symbols at a module contains
             * statement (i.e., task == TBP_COMPLETE_FIN), we can conclude
             * that we have an overloaded symbol from another module here.
             * So, we need to declare an ST_ENTRY (i.e., forward declaration)
             * that's in the current scope.
             */
            sym = insert_sym(sym);
            sym = declsym(sym, ST_ENTRY, FALSE);
            SCP(sym, SC_EXTERN);
          }
        }
      }
    }
  }

  if (STYPEG(sym) == ST_OPERATOR || STYPEG(sym) == ST_USERGENERIC) {
    int sym2;
    sym2 = refocsym(sym, stb.ovclass[ST_PROC]);
    if (sym2) {
      sym = sym2;
    }
  }

  if (STYPEG(sym) != ST_ENTRY && STYPEG(sym) != ST_PROC &&
      STYPEG(sym) != ST_MODPROC && STYPEG(sym) != ST_OPERATOR &&
      STYPEG(sym) != ST_USERGENERIC) {
    /* Possible overloaded symbol */
    ACCL *accessp;
    int access_specified;
    for (access_specified = 0, accessp = sem.accl.next; accessp != NULL;
         accessp = accessp->next) {
      if (accessp->sptr == sym &&
          (accessp->type == 'u' || accessp->type == 'v')) {
        access_specified = 1;
        break;
      }
    }
    if (!access_specified) {
      sym = insert_sym(sym);
      sym = declsym(sym, ST_ENTRY, FALSE);
      SCP(sym, SC_EXTERN);
    }
  }

  if (curr->isIface && STYPEG(sym) == ST_ENTRY) {
    sym2 = findByNameStypeScope(SYMNAME(sym), ST_PROC, -1);
    if (sym2) {
      sym = sym2;
    }
  }

  curr->impSptr = sym;
  ADDRTKNP(sym, 1);

  if (curr->isInherited &&
      (STYPEG(sym) == ST_PROC || STYPEG(sym) == ST_ENTRY)) {
    sym2 = findByNameStypeScope(SYMNAME(sym), ST_PROC, -1);
    if (!sym2)
      sym2 = findByNameStypeScope(SYMNAME(sym), ST_PROC, 0);
    if (sym2)
      curr->impSptr = sym2;
  }

  VTABLEP(curr->memSptr, curr->impSptr); /* implementation name */
  MSCALLP(curr->impSptr, getMscall());
#ifdef CREFP
  CREFP(curr->impSptr, getCref());
#endif

  if (curr->isIface) {
    IFACEP(curr->memSptr, curr->impSptr);
  }

  if ((task != TBP_COMPLETE_FIN || SCOPEG(curr->impSptr) != stb.curr_scope ||
       curr->isDeferred) &&
      DTYPEG(curr->impSptr)) {
    /* Set member type to return type of function to ensure
     * correct I/O
     */
    DTYPEP(curr->memSptr, DTYPEG(curr->impSptr));
  }

  if (!curr->isExtern && IN_MODULE)
    INMODULEP(curr->impSptr, 1);

  return errCnt;
}

/** \brief Called by resolveTbps() to perform some final processing of the
  *        type bound procedure (tbp).
  *
  * \param curr is a pointer to the current tbp record.
  */
static void
completeTbp(TBP *curr)
{
  /* complete tbp derived type tag dtype  */
  SPTR sym = get_struct_tag_sptr(curr->dtype);
  CLASSP(sym, 1);

  /* complete pass arg */
  sym = get_struct_tag_sptr(curr->dtPass);
  CLASSP(sym, 1);
  TBPLNKP(curr->bindSptr, (STYPEG(curr->bindSptr) == ST_USERGENERIC ||
                           STYPEG(curr->bindSptr) == ST_OPERATOR)
                              ? curr->dtype
                              : curr->dtPass);

  if (curr->isInherited && !curr->isOverloaded) {
    /* Need to get parent's implementation */
    int bind, imp, mem;
    bind = BINDG(curr->memSptr);
    imp = get_implementation(TBPLNKG(curr->bindSptr), bind, 0, &mem);
    if (imp && !ABSTRACTG(imp)) {
      VTABLEP(curr->memSptr, imp);
    }
  }
}

/** \brief Resolve the implementation and binding name symbols associated with
   * each type bound procedure (tbp) in a derived type. Also the bulk of the
   * semantic checking on the tbps is performed here.
   *
   * In a tbp expression, we have the following:
   *
   * procedure :: foo => bar
   *
   * In this case, foo is the binding name and bar is the implementation
   * name.
   *
   * \param dtype is the derived type record that we're currently processing
   * \param task is the task that invoked this function
   *
   * \return integer > 0 if successful, else 0.
   */
static int
resolveTbps(int dtype, tbpTask task)
{
  TBP *curr;
  char *name, *nameCpy;
  int addit, i, sym;

  if (IS_RESOLVE_TBP_TASK(task)) {

    int errCnt;
    addit = 0;
    errCnt = 0;
    /* Visit each type bound procedure in queue, initialize it,
     * resolve the implementation name symbol, resolve the binding name, then
     * semantically check the tbp's arguments and interface.
     */
    for (curr = tbpQueue; curr; curr = curr->next) {
      if (curr->dtype && IN_MODULE && !XBIT(68, 0x4) &&
          STYPEG(stb.curr_scope) == ST_MODULE) {
        /* This ensures that we generate type descriptors for all
         * types that require them in the mod object file.
         */
        sym = get_struct_tag_sptr(curr->dtype);
        get_static_type_descriptor(sym);
      }

      i = initFinalSub(task, curr);
      if (i < 0)
        continue;

      if (!curr->impName) {
        curr->dtype = 0;
        continue;
      }
      if (STYPEG(stb.curr_scope) != ST_MODULE && task != TBP_COMPLETE_ENDTYPE &&
          task != TBP_COMPLETE_GENERIC && task != TBP_FORCE_RESOLVE) {
        continue;
      }
      nameCpy = getitem(0, strlen(curr->impName) + 1);
      strcpy(nameCpy, curr->impName);
      name = strchr(nameCpy, '$');
      if (name)
        *name = '\0';

      errCnt += resolveImp(dtype, task, curr, nameCpy);

      i = resolveBind(task, curr, nameCpy);
      if (i < 0)
        continue;
      errCnt += i;

      errCnt += resolvePass(task, curr, nameCpy);

      errCnt += resolveGeneric(task, curr);

      completeTbp(curr);

      addit = 1;

      i = semCheckTbp(task, curr, nameCpy);
      if (i < 0)
        continue;
      errCnt += i;
    }
    if (errCnt && task == TBP_COMPLETE_ENDMODULE) {
      /* Must clear out all entries to avoid bogus errors due to first pass
       * aborting...
       */
      queue_tbp(0, 0, 0, 0, TBP_CLEAR_ERROR);
    }
    return addit;
  }
  return 0;
}

/** \brief "Copy" inherited type bound procedures (tbps) from parent type to
  * child type.
  *
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invokved this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
inheritTbps(int dtype, tbpTask task)
{

  TBP *curr, *newTbp, *curr2;
  char *name, *nameCpy;
  int len, addit, sym, found, found_parent;
  SPTR tag;
  DTYPE parent;

  if (!IS_INHERIT_TBP_TASK(task))
    return 0;

  tag = get_struct_tag_sptr(dtype);
  parent = DTYPEG(PARENTG(tag));

  addit = 0;
  if (parent) {
    found_parent = 0;
    for (curr = tbpQueue; curr; curr = curr->next) {
      if (curr->dtype == parent && curr->impName) {
        found = 0;
        found_parent = 1;
        if (curr->isFinal)
          continue; /* do not inherit final subroutines */
        for (curr2 = tbpQueue; curr2; curr2 = curr2->next) {
          if (curr2->dtype == dtype) {
            if (strcmp(curr->bindName, curr2->bindName) == 0) {
              if (curr->isNonOver) {
                error(155, 3, gbl.lineno,
                      "Overriding type bound "
                      "procedure with NON_OVERRIDABLE attribute",
                      CNULL);
              } else {
                curr2->isOverloaded = 1;
                curr2->offset = curr->offset;
              }
              if ((curr->access != PRIVATE_ACCESS_TBP) &&
                  (curr2->access == PRIVATE_ACCESS_TBP ||
                   (curr2->access == DEFAULT_ACCESS_TBP &&
                    sem.tbp_access_stmt))) {
                error(155, 3, gbl.lineno,
                      "Cannot override PUBLIC type bound "
                      "procedure with PRIVATE type bound procedure",
                      CNULL);
              } else {
                found = 1;
              }
              break;
            }
          }
        }
        if (found && curr2->genericType && curr->genericType &&
            !curr2->isInherited && strcmp(curr->impName, curr2->impName) == 0) {
          char *buf;
          nameCpy = getitem(0, strlen(curr2->impName) + 1);
          strcpy(nameCpy, curr2->impName);
          name = strchr(nameCpy, '$');
          if (name)
            *name = '\0';
          len = strlen("Ambiguous type bound procedure ") + strlen(nameCpy) +
                strlen(" in generic set for ") + strlen(curr2->impName) +
                strlen(" of type") + 1;
          buf = getitem(0, len);
          sprintf(buf, "Ambiguous type bound procedure %s in generic set "
                       "for %s of type",
                  nameCpy, curr2->bindName);
          error(155, 3, gbl.lineno, buf, SYMNAME(tag));

        } else if (!found) {
          int any_sym = findByNameStypeScope(curr->impName, ST_PROC, 0);
          int inscope_sym = findByNameStypeScope(curr->impName, ST_PROC, -1);

          if (any_sym && !inscope_sym) {
            if (ABSTRACTG(get_struct_tag_sptr(parent)))
              continue;
          }

          if (curr->isDeferred && !ABSTRACTG(get_struct_tag_sptr(dtype))) {
            nameCpy = getitem(0, strlen(curr->bindName) + 1);
            strcpy(nameCpy, curr->bindName);
            name = strchr(nameCpy, '$');
            if (name)
              *name = '\0';
            error(155, 2, gbl.lineno, "No overriding procedure specified for"
                                      " DEFERRED type bound procedure",
                  nameCpy);
          }
          addit = 1;
          NEW(newTbp, TBP, 1);
          memcpy(newTbp, curr, sizeof(TBP));
          name = curr->impName;
          len = strlen(name) + 1;
          NEW(nameCpy, char, len);
          strcpy(nameCpy, name);
          newTbp->impName = nameCpy;

          name = curr->bindName;
          len = strlen(name) + 1;
          NEW(nameCpy, char, len);
          strcpy(nameCpy, name);
          newTbp->bindName = nameCpy;

          newTbp->dtype = dtype;
          newTbp->dtPass = curr->dtPass;

          newTbp->isIface = curr->isIface;
          newTbp->isDeferred = curr->isDeferred;
          newTbp->lineno = curr->lineno;
          newTbp->isInherited = 1;

          newTbp->next = tbpQueue;
          tbpQueue = newTbp;
          newTbp->offset = curr->offset;
          if (!VTOFFG(tag)) {
            VTOFFP(tag, VTOFFG(PARENTG(tag)));
          }
        }
      }
    }
    if (found_parent)
      return addit;
    for (sym = DTY(parent + 1); sym > NOSYM; sym = SYMLKG(sym)) {
      found = 0;
      if (FINALG(sym))
        continue; /* do not inherit final subroutines */
      if (CCSYMG(sym) && CLASSG(sym)) {
        name = SYMNAME(BINDG(sym));
        for (curr2 = tbpQueue; curr2; curr2 = curr2->next) {
          if (curr2->dtype == dtype) {
            if (strcmp(name, curr2->bindName) == 0 &&
                (VTABLEG(sym) || IFACEG(sym) ||
                 (PRIVATEG(sym) && !curr2->access && !sem.tbp_access_stmt))) {
              if (NONOVERG(sym)) {
                error(155, 3, gbl.lineno,
                      "Overriding type bound "
                      "procedure with NON_OVERRIDABLE attribute",
                      CNULL);
              } else {
                curr2->isOverloaded = 1;
              }
              if (!PRIVATEG(sym) && (curr2->access == PRIVATE_ACCESS_TBP ||
                                     (curr2->access == DEFAULT_ACCESS_TBP &&
                                      sem.tbp_access_stmt))) {
                error(155, 3, gbl.lineno,
                      "Cannot override PUBLIC type bound "
                      "procedure with PRIVATE type bound procedure",
                      CNULL);
              } else {
                found = 1;
              }
              break;
            }
          }
        }
        if (!found) {
          addit = 1;
          if (IFACEG(sym) && !ABSTRACTG(get_struct_tag_sptr(dtype))) {
            nameCpy = getitem(0, strlen(SYMNAME(BINDG(sym))) + 1);
            strcpy(nameCpy, SYMNAME(BINDG(sym)));
            name = strchr(nameCpy, '$');
            if (name)
              *name = '\0';
            error(155, 2, gbl.lineno, "No overriding procedure specified for"
                                      " DEFERRED type bound procedure",
                  nameCpy);
          }

          NEW(newTbp, TBP, 1);
          BZERO(newTbp, TBP, 1);
          name = SYMNAME(VTABLEG(sym));
          len = strlen(name) + 1;
          NEW(nameCpy, char, len);
          strcpy(nameCpy, name);
          newTbp->impName = nameCpy;

          name = SYMNAME(BINDG(sym));
          len = strlen(name) + 1;
          NEW(nameCpy, char, len);
          strcpy(nameCpy, name);
          newTbp->bindName = nameCpy;

          newTbp->dtype = dtype;
          newTbp->dtPass = TBPLNKG(BINDG(sym));

          newTbp->lineno = -1;

          newTbp->next = tbpQueue;
          tbpQueue = newTbp;
          newTbp->offset = VTOFFG(BINDG(sym));
          if (!VTOFFG(tag)) {
            VTOFFP(tag, VTOFFG(PARENTG(tag)));
          }

          newTbp->impSptr = VTABLEG(sym);
          newTbp->bindSptr = BINDG(sym);
          newTbp->pass = PASSG(sym);
          newTbp->hasNopass = NOPASSG(sym);
          newTbp->isNonOver = NONOVERG(sym);
          newTbp->access =
              (PRIVATEG(sym)) ? PRIVATE_ACCESS_TBP : PUBLIC_ACCESS_TBP;
          newTbp->isIface = IFACEG(sym);
          newTbp->isDeferred = IFACEG(sym);
          newTbp->isInherited = 1;
          newTbp->genericType = (STYPEG(newTbp->bindSptr) != ST_PROC &&
                                 STYPEG(newTbp->bindSptr) != ST_ENTRY)
                                    ? STYPEG(newTbp->bindSptr)
                                    : 0;
        }
      }
    }
  }
  return addit;
}
/** \brief Add type bound procedures (tbps) to derived type.
  *
  * Note that what gets added to the derived type here is a stub.
  * The symbol table pointers for the binding name, implementation name,
  * interface, etc. are resolved later in the function resolveTbps().
  *
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function.
  *
  * \return integer > 0 if successful, else 0
  */
static int
addToDtype(int dtype, tbpTask task)
{
  TBP *curr, *newTbp, *curr2;
  char *nameCpy, *nameCpy2;
  int addit;
  static int tmp = 0;

  if (IS_ADD_TBP_TO_DTYPE_TASK(task)) {

    addit = 0;
    for (curr = tbpQueue; curr;) {
      if (curr->genericType && curr->isInherited && curr->dtype == dtype) {
        /* look for tbp that overloads a tbp associated with a generic
         * tbp (or operator)
         */
        nameCpy = SYMNAME(curr->impSptr);
        for (curr2 = tbpQueue; curr2; curr2 = curr2->next) {
          if (curr2 == curr || curr2->dtype != dtype)
            continue;
          nameCpy2 = curr2->bindName;
          if (!curr2->genericType && !curr2->isInherited &&
              strcmp(nameCpy, nameCpy2) == 0) {
            curr->impSptr = 0;
            break;
          }
        }
      }
      if (curr->dtype == dtype) {
        int sym = DTY(dtype + 1);
        int mem;
        if (!curr->bindName) {
          addit = 1;
          break;
        }
        if (sym > NOSYM) {
          for (; SYMLKG(sym) > NOSYM; sym = SYMLKG(sym))
            ;
        }
        do {
          mem = getsymf("%s$%d", curr->bindName, tmp++);
          ENCLDTYPEP(mem, curr->dtype);
          if (STYPEG(mem) == ST_UNKNOWN) {
            STYPEP(mem, ST_MEMBER);
            CCSYMP(mem, 1);
            IGNOREP(mem, 1);
            SCOPEP(mem, stb.curr_scope);
            break;
          }
          if (SCOPEG(mem) == stb.curr_scope) {
            break;
          }
        } while (1);
        DTYPEP(mem, DT_INT4);

        curr->memSptr = mem;
        CLASSP(mem, 1);
        if (curr->isFinal) {
          FINALP(curr->memSptr, -1);
          if (!sem.which_pass) {
            /* Set placeholder for first semantic pass in case there
             * any structure parameter initializers in the module.
             */
            VTABLEP(mem, NOSYM);
          }
        } else if (!sem.which_pass) {
          /* Set placeholders for first semantic pass in case there
           * are any structure parameter initializers in the module.
           */
          VTABLEP(mem, NOSYM);
          BINDP(mem, NOSYM);
        }
        if (sym != NOSYM) {
          SYMLKP(sym, mem);
        } else {
          DTY(dtype + 1) = mem;
        }

        if (curr->access == PRIVATE_ACCESS_TBP ||
            (sem.tbp_access_stmt && curr->access != PUBLIC_ACCESS_TBP)) {
          PRIVATEP(mem, 1);
        }

        SYMLKP(mem, NOSYM);
        addit = 1;
      }
      curr = curr->next;
    }
    if (!addit) {
      /* add dtype record without any type bound procedures */
      NEW(newTbp, TBP, 1);
      BZERO(newTbp, TBP, 1);

      newTbp->dtype = dtype;

      newTbp->isIface = 0;
      newTbp->lineno = gbl.lineno;

      newTbp->next = tbpQueue;
      tbpQueue = newTbp;

      addit = 1;
    }

    return addit;
  }
  return 0;
}

/** \brief Add type bound procedure (tbp) to tbp queue, tbpQueue.
   *
   * If task is TBP_ADD_SIMPLE, then we are adding a tbp that was declared as
   * "procedure tbp".
   *
   * If task is TBP_ADD_IMPL, then we are adding a tbp that was declared as
   * "bind => implementation".
   *
   * If task is TBP_CHECK_CHILD, then we are also checking the validity of a
   * child tbp.
   *
   * If task is TBP_CHECK_PUBLIC, then we are also checking the validity of a
   * public declared tbp.
   *
   * If task is TBP_CHECK_PRIVATE, then we are also checking the validity of a
   * private declared tbp.
   *
   * \param sptr is the symbol table pointer of the implementation name
   * \param bind is the symbol table pointer of the binding name
   * \param offset is tbp's offset in dtype's "virtual function table"
   * \param dtype is the derived type record that we're currently processing
   * \param task is the task that invoked this function
   *
   * \return integer > 0 if successful, else 0
   */
static int
enqueueTbp(int sptr, int bind, int offset, int dtype, tbpTask task)
{

  TBP *curr, *prev, *newTbp, *curr2;
  char *name, *nameCpy;
  int len, inserted;

  if (IS_ADD_TBP_TASK(task)) {

    if (sptr && bind && (task == TBP_CHECK_CHILD || task == TBP_CHECK_PRIVATE ||
                         task == TBP_CHECK_PUBLIC) &&
        strcmp(SYMNAME(sptr), SYMNAME(bind)) == 0 &&
        sem.generic_tbp != ST_OPERATOR) {
      error(155, 3, gbl.lineno,
            "Generic type bound procedure has same name as specific type"
            " bound procedure -",
            SYMNAME(bind));
    }
    /* check for existing record first */
    for (name = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->dtype == dtype && curr->impName && curr->isIface &&
          curr->lineno == gbl.lineno &&
          strcmp(curr->impName, SYMNAME(sptr)) == 0) {
        if (task == TBP_ADD_IMPL) {
          error(155, 3, gbl.lineno,
                "Must specify an interface-name or a procedure-name, not both "
                "in type bound procedure -",
                SYMNAME(bind));
        }
        name = curr->impName;
        break;
      } else if (curr->dtype == dtype && curr->bindName && bind &&
                 (!curr->genericType ||
                  (task != TBP_CHECK_CHILD && task != TBP_CHECK_PRIVATE &&
                   task != TBP_CHECK_PUBLIC))) {

        if (strcmp(curr->bindName, SYMNAME(bind)) == 0) {
          nameCpy = getitem(0, strlen(curr->bindName) + 1);
          strcpy(nameCpy, curr->bindName);
          name = strchr(nameCpy, '$');
          if (name)
            *name = '\0';
          error(155, 3, gbl.lineno, "Redefinition of type bound procedure -",
                nameCpy);
          name = 0;
          break;
        }
      } else if (curr->dtype == dtype && curr->impName && curr->bindName &&
                 curr->genericType &&
                 (task == TBP_CHECK_CHILD || task == TBP_CHECK_PRIVATE ||
                  task == TBP_CHECK_PUBLIC) &&
                 strcmp(curr->bindName, SYMNAME(bind)) == 0) {
        nameCpy = getitem(0, strlen(curr->impName) + 1);
        strcpy(nameCpy, curr->impName);
        name = strchr(nameCpy, '$');
        if (name)
          *name = '\0';
        if (strcmp(nameCpy, SYMNAME(sptr)) == 0) {
          return 0;
        }
      }

      name = 0;
    }
    if (!name) {
      /* add record */

      NEW(newTbp, TBP, 1);
      BZERO(newTbp, TBP, 1);

      name = SYMNAME(sptr);

      len = strlen(name) + strlen("$tbp") + 1;
      NEW(nameCpy, char, len);
      if (!sem.generic_tbp) {
        strcpy(nameCpy, name);
      } else {
        sprintf(nameCpy, "%s$tbp", name);
      }

      newTbp->impName = nameCpy;

      name = SYMNAME(bind);
      len = strlen(name) + strlen("$tbpg") + 1;
      NEW(nameCpy, char, len);
      if (sem.generic_tbp == ST_OPERATOR ||
          (strlen(name) > 4 && strcmp("$tbp", name + (strlen(name) - 4)) == 0)){
        strcpy(nameCpy, name);
      } else if (sem.generic_tbp == ST_USERGENERIC &&
               (strcmp(name, ".read") == 0 || strcmp(name, ".write") == 0)) {
        strcpy(nameCpy, name); /* special case for defined I/O */
        newTbp->isDefinedIO = sem.defined_io_type;
      } else if (sem.generic_tbp == ST_USERGENERIC) {
        sprintf(nameCpy, "%s$tbpg", name);
      } else {
        sprintf(nameCpy, "%s$tbp", name);
      }
      newTbp->bindName = nameCpy;

      newTbp->dtype = dtype;
      newTbp->dtPass = TBPLNKG(bind);

      newTbp->isIface = 0;
      newTbp->lineno = gbl.lineno;

      newTbp->offset = offset;

      newTbp->genericType = sem.generic_tbp;

      if (task == TBP_CHECK_PRIVATE)
        newTbp->access = PRIVATE_ACCESS_TBP;
      else if (task == TBP_CHECK_PUBLIC)
        newTbp->access = PUBLIC_ACCESS_TBP;
      inserted = 0;
      if (newTbp->genericType == ST_OPERATOR && isalpha(newTbp->bindName[0])) {
        for (prev = curr2 = tbpQueue; curr2;) {
          prev = curr2;
          curr2 = curr2->next;
          if (!prev->genericType && newTbp->genericType &&
              prev->access != PRIVATE_ACCESS_TBP &&
              newTbp->dtype == prev->dtype && !prev->isDeferred &&
              strcmp(newTbp->impName, prev->bindName) == 0) {
            /* insert generic after its tbp definition */
            prev->next = newTbp;
            newTbp->next = curr2;
            inserted = 1;
            break;
          }
        }
      }
      if (!inserted) {
        newTbp->next = tbpQueue;
        tbpQueue = newTbp;
      }
    } else {
      /* update record */
      name = SYMNAME(bind);
      len = strlen(name) + strlen("$tbp") + 1;
      NEW(nameCpy, char, len);
      if (strlen(name) > 4 && strcmp("$tbp", name + (strlen(name) - 4)) == 0)
        strcpy(nameCpy, name);
      else
        sprintf(nameCpy, "%s$tbp", name);
      curr->bindName = nameCpy;

      curr->dtPass = TBPLNKG(bind);

      curr->offset = offset;

      curr->lineno = gbl.lineno;

      curr->genericType = sem.generic_tbp;

      if (task == TBP_CHECK_PRIVATE)
        curr->access = PRIVATE_ACCESS_TBP;
      else if (task == TBP_CHECK_PUBLIC)
        curr->access = PUBLIC_ACCESS_TBP;
    }
    return 1;
  }
  return 0;
}

/** \brief Initialize/clear entries in type bound procedure (tbp) queue,
  * tbpQueue.
  *
  * \param task is the task that invoked this function
  *
  * \return an integer > 0 if successful, else 0
  */
static int
initTbp(tbpTask task)
{

  TBP *curr;

  if (!IS_CLEAR_TBP_TASK(task)) {
    return 0;
  }
  if (!sem.which_pass && task == TBP_CLEAR) {
    for (curr = tbpQueue; curr; curr = curr->next) {
      if (curr->genericType == ST_OPERATOR ||
          curr->genericType == ST_USERGENERIC || curr->isFinal) {
        /* do not remove these until the second pass since they may be
         * used in module procedures (and the processing is done early).
         */
        return 0;
      }
    }
  }
  for (curr = tbpQueue; curr;) {
    TBP *del = curr;
    FREE(curr->impName);
    FREE(curr->bindName);
    curr = curr->next;
    FREE(del);
  }
  tbpQueue = NULL;
  return 1;
}

/** \brief Clear records in type bound procedure (tbp) queue that have a stale
  * dtype field. Zero out any other stale dtype/sptr fields.
  *
  * \param task is the task that invoked this function
  *
  * \return number of records/fields cleared
  */
static int
clearStaleRecs(tbpTask task)
{
  TBP *curr, *prev;
  int numUpdated = 0; 
  
  if (IS_CLEAR_STALE_RECORDS_TASK(task)) {
    for (prev = curr = tbpQueue; curr; ) {
      if (curr->dtype >= stb.dt.stg_avail) {
        /* stale dtype field, so delete entire entry in tbpQueue */
        if (curr == tbpQueue) {
          prev = tbpQueue = curr->next;
          FREE(curr);
          curr = tbpQueue;
        } else {
          prev->next = curr->next;
          FREE(curr);
          curr = prev->next;
        }
        ++numUpdated;
      } else {
        /* check for stale fields in current tbpQueue entry */
        if (curr->dtPass >= stb.dt.stg_avail) {
          curr->dtPass = DT_NONE;
          ++numUpdated;
        }
        if (curr->impSptr >= stb.stg_avail) {
          curr->impSptr = 0;
          ++numUpdated;
        }
        if (curr->bindSptr >= stb.stg_avail) {
          curr->bindSptr = 0;
          ++numUpdated;
        }
        if (curr->memSptr >= stb.stg_avail) {
          curr->memSptr = 0;
          ++numUpdated;
        }
        prev = curr;
        curr = curr->next;
      }
    }
  }
#if DEBUG
  checkForStaleTbpEntries();
#endif
  return numUpdated;
}


/** \brief Add pass argument for type bound procedure (tbp) binding name, bind,
  *  if it has not already been added.
  *
  * \param sptr is the symbol table pointer for the implementation name
  * \param bind is the symbol table pointer for the binding name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invokved this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
addPassArg(int sptr, int bind, int dtype, tbpTask task)
{

  TBP *curr;
  int addit;

  if (IS_ADD_TBP_PASS_ARG_TASK(task)) {
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->bindName &&
          strncmp(curr->bindName, SYMNAME(bind),
                  strlen(curr->bindName) - strlen("$tbp")) == 0) {
        if (curr->dtype == dtype && !curr->pass) {
          curr->pass = sptr;
          addit = 1;
          break;
        }
      }
    }
    return addit;
  }
  return 0;
}

/** \brief Add specified private attribute to type bound procedure (tbp)
  * definition.
  *
  * \param bind is the symbol table pointer of the binding name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
addPrivateAttribute(int bind, int dtype, tbpTask task)
{

  int addit;
  TBP *curr;
  if (IS_ADD_PRIVATE_TBP_TASK(task)) {
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->bindName &&
          strncmp(curr->bindName, SYMNAME(bind),
                  strlen(curr->bindName) - strlen("$tbp")) == 0) {
        if (curr->dtype == dtype) {
          curr->access = PRIVATE_ACCESS_TBP;
          addit = 1;
          break;
        }
      }
    }
    return addit;
  }
  return 0;
}

/** \brief Add specified public attribute to type bound procedure (tbp)
  * definition.
  *
  * \param bind is the symbol table pointer of the binding name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
addPublicAttribute(int bind, int dtype, tbpTask task)
{

  int addit;
  TBP *curr;
  if (IS_ADD_PUBLIC_TBP_TASK(task)) {
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->bindName &&
          strncmp(curr->bindName, SYMNAME(bind),
                  strlen(curr->bindName) - strlen("$tbp")) == 0) {
        if (curr->dtype == dtype) {
          curr->access = PUBLIC_ACCESS_TBP;
          addit = 1;
          break;
        }
      }
    }
    return addit;
  }
  return 0;
}

/** \brief Returns true if there exists a type bound procedure (tbp) for
  * a specified derived type, dtype.
  *
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invokved this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
dtypeHasTbp(int dtype, tbpTask task)
{
  TBP *curr;

  if (IS_TBP_STATUS_TASK(task)) {
    for (curr = tbpQueue; curr; curr = curr->next) {
      if (curr->dtype == dtype)
        return 1;
    }
  }
  return 0;
}

/** \brief Add deferred attribute to type bound procedure (tbp).
  *
  * \param bind is the symbol table pointer of a binding name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
addDeferredTbp(int bind, int dtype, tbpTask task)
{

  int addit;
  TBP *curr;

  if (IS_ADD_DEFERRED_TBP_TASK(task)) {
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->bindName &&
          strncmp(curr->bindName, SYMNAME(bind),
                  strlen(curr->bindName) - strlen("$tbp")) == 0) {
        if (curr->dtype == dtype) {
          curr->isDeferred = 1;
          addit = 1;
          break;
        }
      }
    }
    return addit;
  }
  return 0;
}

/** \brief Called when type bound procedure (tbp) definition uses an explicit
  * interface of an external routine for its implementation.
  *
  * \param sptr is the symbol table pointer of the implementation name
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0
*/
static int
addExplicitIface(int sptr, tbpTask task)
{
  int addit;
  TBP *curr;
  if (IS_ADD_EXPLICIT_IFACE_TBP_TASK(task)) {
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->impName && strcmp(curr->impName, SYMNAME(sptr)) == 0) {
        curr->isExtern = 1;
        addit = 1;
      }
    }
    return addit;
  }
  return 0;
}

/** \brief Add final subroutine to type bound procedure queue, tbpQueue.
  *
  * \param sptr is the symbol table pointer of the implementation name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0.
  */
static int
addFinalSubroutine(SPTR sptr, DTYPE dtype, tbpTask task)
{
  int addit, len;
  char *name, *nameCpy;
  TBP *curr, *newTbp;
  SPTR tag;

  if (IS_ADD_FINAL_SUB_TASK(task)) {
    tag = get_struct_tag_sptr(dtype);
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->dtype == dtype && curr->impName &&
          strcmp(curr->impName, SYMNAME(sptr)) == 0) {
        error(155, 3, gbl.lineno, "Duplicate final subroutine in",
              SYMNAME(tag));
        addit = 0;
        break;
      }
    }
    if (!curr) {

      NEW(newTbp, TBP, 1);
      BZERO(newTbp, TBP, 1);
      name = SYMNAME(sptr);
      len = strlen(name) + 1;
      NEW(nameCpy, char, len);
      strcpy(nameCpy, name);
      newTbp->impName = nameCpy;

      newTbp->dtype = dtype;
      newTbp->isIface = 1;

      newTbp->lineno = gbl.lineno;
      newTbp->isFinal = 1;

      newTbp->next = tbpQueue;
      tbpQueue = newTbp;
      addit = 1;

      NEW(nameCpy, char, len);
      strcpy(nameCpy, name);
      newTbp->bindName = nameCpy;
    }
    return addit;
  }
  return 0;
}

/** \brief Set non_overridable attribute on type bound procedure (tbp).
  *
  * \param bind is the symbol table pointer of the binding name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
addNonOverridableTbp(int bind, int dtype, tbpTask task)
{

  int addit;
  TBP *curr;

  if (IS_ADD_NON_OVERRIDABLE_TBP_TASK(task)) {
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->bindName &&
          strncmp(curr->bindName, SYMNAME(bind),
                  strlen(curr->bindName) - strlen("$tbp")) == 0) {
        if (curr->dtype == dtype) {
          curr->isNonOver = 1;
          addit = 1;
          break;
        }
      }
    }
    return addit;
  }
  return 0;
}

/** \brief Set nopass attribute on type bound procedure (tbp).
  *
  * \param bind is the symbol table pointer of the binding name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
addNoPassArg(int bind, int dtype, tbpTask task)
{

  int addit;
  TBP *curr;

  if (IS_ADD_TBP_NOPASS_ARG_TASK(task)) {
    for (addit = 0, curr = tbpQueue; curr; curr = curr->next) {
      if (curr->bindName &&
          strncmp(curr->bindName, SYMNAME(bind),
                  strlen(curr->bindName) - strlen("$tbp")) == 0) {
        if (curr->dtype == dtype) {
          curr->hasNopass = 1;
          addit = 1;
          break;
        }
      }
    }
    return addit;
  }
  return 0;
}

/** \brief Add interface-name if user specified it with the type bound
  * procedure (tbp) definition.
  *
  * For example,
  *
  * procedure (interface-name), deferred :: binding-name
  *
  * \param sptr is the symbol table pointer of implementation name
  * \param dtype is the derived type record that we're currently processing
  * \param task is the task that invoked this function
  *
  * \return integer > 0 if successful, else 0
  */
static int
addTbpInterface(int sptr, int dtype, tbpTask task)
{

  char *nameCpy;
  int len;
  TBP *newTbp;
  char *name;

  if (IS_ADD_TBP_INTERFACE_TASK(task)) {
    NEW(newTbp, TBP, 1);
    BZERO(newTbp, TBP, 1);
    name = SYMNAME(sptr);
    len = strlen(name) + 1;
    NEW(nameCpy, char, len);
    strcpy(nameCpy, name);
    newTbp->impName = nameCpy;

    newTbp->dtype = dtype;
    newTbp->isIface = 1;

    newTbp->lineno = gbl.lineno;

    newTbp->next = tbpQueue;
    tbpQueue = newTbp;

    return 1;
  }
  return 0;
}

#if DEBUG
/** \brief issues an internal compiler warning if it finds any stale 
  * dtype/sptr entries in the type bound procedure (tbp) queue. 
  *
  */
static void
checkForStaleTbpEntries(void)
{
  TBP *curr;

  for (curr = tbpQueue; curr; curr = curr->next) {
    if (curr->dtype >= stb.dt.stg_avail) {
      assert(FALSE, "checkForStaleTbpEntries: "
             "tbpQueue entry references stale dtype", curr->dtype,
             ERR_Warning);
    }
    if (curr->dtPass >= stb.dt.stg_avail) {
      assert(FALSE, "checkForStaleTbpEntries: "
             "tbpQueue entry references stale dtPass dtype", curr->dtPass,
             ERR_Warning);
    }
    if (curr->impSptr >= stb.stg_avail) {
      assert(FALSE, "checkForStaleTbpEntries: "
             "tbpQueue entry references stale impSptr", curr->impSptr,
             ERR_Warning);
    }
    if (curr->bindSptr >= stb.stg_avail) {
      assert(FALSE, "checkForStaleTbpEntries: "
             "tbpQueue entry references stale bindSptr", curr->bindSptr,
             ERR_Warning);
    }
    if (curr->memSptr >= stb.stg_avail) {
      assert(FALSE, "checkForStaleTbpEntries: "
             "tbpQueue entry references stale memSptr", curr->memSptr,
             ERR_Warning);
    }
  }
}
#endif
