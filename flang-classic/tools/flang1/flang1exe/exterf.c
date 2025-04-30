/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Routines for exporting symbols to .mod files and to IPA.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "machar.h"
#include "semant.h"
#include "ast.h"
#include "dinit.h"
#include "soc.h"
#include "lz.h"
#define TRACEFLAG 48
#define TRACEBIT 4
#define TRACESTRING "export"
#include "trace.h"

#define INSIDE_INTERF
#include "interf.h"
#include "fih.h"

#include "dpm_out.h"

#define COMPILER_OWNED_MODULE XBIT(58, 0x100000)

/* ------------------------------------------------------------------ */
/* ----------------------- Export Utilities ------------------------- */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*   Write symbols to export file  */
/* This is used for:
 *   module interface files
 *   interprocedural analysis
 *   procedure inlining
 *   static variable initialization
 */

/*  getitem area for module temp storage; pick an area not used by
 *  the caller of export()/import().
 */
#define MOD_AREA 18

/*  getitem area for appending symbols to the mod file; pick an area not
 *  used by semant and export()/import().
 */
#define APPEND_AREA 19

typedef struct itemx { /* generic item record */
  int val;
  struct itemx *next;
} ITEMX;

typedef struct xitemx { /* generic item record */
  int val;
  struct xitemx *next;
  int exceptlist;
} XITEMX;

static char *symbol_flag; /* flags for symbols being exported */
static int symbol_flag_size;
static int symbol_flag_lowest_const = 0;
static char *dtype_flag; /* flags for data types being exported */
static int dtype_flag_size;
static char *ast_flag; /* flags for asts being exported */
static int ast_flag_size;
static int ast_flag_lowest_const = 0;
static char *eqv_flag; /* flags for equivalences being exported */
static XITEMX *public_module_list = NULL; /* queue of modules in public part */
static ITEMX *private_module_list = NULL; /* other modules */

static ITEMX *append_list; /* list of symbols to be appended to mod file */

static LOGICAL for_module = FALSE;
static LOGICAL for_inliner = FALSE;
static int sym_module = 0; /* if we are exporting a module,
                              or a subprogram within a module */
static LOGICAL for_contained = FALSE;
static LOGICAL exporting_module = FALSE;
static lzhandle *outlz;
static int exportmode = 0;
#define MAX_FNAME_LEN 2050

static int out_platform = MOD_ANY;

EXPORTB exportb;

static void queue_symbol(int);
static void rqueue_ast(int ast, int *unused);
static void queue_ast(int ast);
static void queue_dtype(int dtype);
static void export_dtypes(int, int);
static void export_outer_derived_dtypes(int limit);
static void export_dt(int);
static void export_symbol(int);
static void export_one_ast(int);
static void export_iso_c_libraries(void);
static void export_iso_fortran_env_libraries(void);
static void export_ieee_arith_libraries(void);
static void export_one_std(int);
static void queue_one_std(int std);
static void all_stds(void (*)(int));
#ifdef FLANG_EXTERF_UNUSED
static void export_parameter_info(ast_visit_fn);
#endif
static void export_data_file(int);
static void export_component_init(int);
static void export_data_file_asts(ast_visit_fn, int, int, int);
static void export_component_init_asts(ast_visit_fn, int, int);
#ifdef FLANG_EXTERF_UNUSED
static void export_equiv_asts(int, ast_visit_fn);
static void export_dist_info(int, ast_visit_fn);
static void export_align_info(int, ast_visit_fn);
static void export_external_equiv();
#endif
static void export_equivs(void);

#ifdef FLANG_EXTERF_UNUSED
static void export_dinit_file(void (*)(int), void (*)(int, INT), int);
static void export_dinit_record(int, INT);
#endif
static int dtype_skip(int dtype);

#ifdef FLANG_EXTERF_UNUSED
/* return 1 if the base type is double/complex/other 8-byte-type */
static int
doubletype(int sptr)
{
  int dtype, dty;
  dtype = DTYPEG(sptr);
  dty = DTY(dtype);
  if (dty == TY_ARRAY) {
    dtype = DTY(dtype + 1);
    dty = DTY(dtype);
  }
  switch (dty) {
  case TY_DWORD:
  case TY_INT8:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_LOG8:
    return 1;
  }
  return 0;
} /* doubletype */
#endif

void
export_public_module(int module, int exceptlist)
{
  XITEMX *p;

  /* if an equivalent entry is in the list, don't add a duplicate */
  for (p = public_module_list; p; p = p->next) {
    if (p->val == module) {
      if (same_sym_list(p->exceptlist, exceptlist)) {
        return;
      }
    }
  }

  p = (XITEMX *)getitem(MOD_AREA, sizeof(XITEMX));
  p->val = module;
  p->next = public_module_list;
  p->exceptlist = exceptlist;
  public_module_list = p;
} /* export_public_module */

static lzhandle *
export_header(FILE *fd, const char *export_name, int compress)
{
  lzhandle *lz;

  if (XBIT(124, 0x10)) {
    out_platform = out_platform | MOD_I8;
  }
  if (XBIT(124, 0x8)) {
    out_platform = out_platform | MOD_R8;
  }
  if (XBIT(68, 0x1)) {
    out_platform = out_platform | MOD_LA;
  }
  if (COMPILER_OWNED_MODULE)
    out_platform = out_platform | MOD_PG;

  fprintf(fd, "V%d :0x%x %s\n", IVSN, out_platform, export_name);
  fprintf(fd, "%d %s S%d %d\n", (unsigned)strlen(gbl.src_file), gbl.src_file,
          stb.firstosym, compress);

  lz = lzinitfile(fd, 0 /*compress*/);
  lzprintf(lz, "%s\n", gbl.datetime);

  /* do the public and private libraries */
  if (for_module || for_inliner || for_contained) {
    XITEMX *pub;
    ITEMX *p;
    for (pub = public_module_list; pub; pub = pub->next) {
      int i, count;
      int base = CMEMFG(pub->val);
      lzprintf(lz, "use %s public", SYMNAME(pub->val));
      count = 0;
      for (i = pub->exceptlist; i; i = SYMI_NEXT(i))
        ++count;
      lzprintf(lz, " %d", count);
      for (i = pub->exceptlist; i; i = SYMI_NEXT(i)) {
        lzprintf(lz, " %d", SYMI_SPTR(i) - base);
      }
      if (imported_directly(SYMNAME(pub->val), pub->exceptlist)) {
        lzprintf(lz, " direct\n");
      } else {
        lzprintf(lz, " indirect\n");
      }
    }
    for (p = private_module_list; p; p = p->next) {
      lzprintf(lz, "use %s private\n", SYMNAME(p->val));
    }
  }
  lzprintf(lz, "enduse\n");
  return lz;
} /* export_header */

static void export(FILE *export_fd, char *export_name, int cleanup)
{
  int sptr;
  int ast;
  XITEMX *pub;

  Trace(("****** Exporting ******"));
#if DEBUG
  if (DBGBIT(5, 16384))
    symdmp(gbl.dbgfil, DBGBIT(5, 8));
#endif

  symbol_flag_size = stb.stg_avail + 1;
  symbol_flag_lowest_const = stb.stg_avail;
  NEW(symbol_flag, char, symbol_flag_size);
  BZERO(symbol_flag, char, stb.stg_avail + 1);

  dtype_flag_size = stb.dt.stg_avail + 1;
  NEW(dtype_flag, char, dtype_flag_size);
  BZERO(dtype_flag, char, dtype_flag_size);

  ast_flag_size = astb.stg_avail + 1;
  ast_flag_lowest_const = astb.stg_avail;
  NEW(ast_flag, char, ast_flag_size);
  BZERO(ast_flag, char, ast_flag_size);

  NEW(eqv_flag, char, sem.eqv_avail + 1);
  BZERO(eqv_flag, char, sem.eqv_avail + 1);

  for (pub = public_module_list; pub; pub = pub->next) {
    symbol_flag[pub->val] = 1;
  }
  if (for_module) {
    symbol_flag[sym_module] = 1;
  }

  exportb.hmark.maxsptr = stb.firstosym;
  ast_visit(1, 1);
  if (for_module || for_inliner) {
    for (sptr = stb.firstosym; sptr < stb.stg_avail; sptr++) {
      switch (STYPEG(sptr)) {
      case ST_CMBLK:
        if (for_module) {
          if (!IGNOREG(sptr) && SCOPEG(sptr) == sym_module) {
            FROMMODP(sptr, 1);
            queue_symbol(sptr);
          }
        }
        break;
      case ST_ENTRY:
        if (!for_module) {
          if (!IGNOREG(sptr)) {
            if (!for_inliner ||
                (sptr == gbl.currsub || SCOPEG(sptr) == SCOPEG(gbl.currsub)))
              queue_symbol(sptr);
          }
        }
        break;
      case ST_UNKNOWN:
      case ST_PARAM:
      case ST_ARRDSC:
      case ST_OPERATOR:
      case ST_TYPEDEF:
      case ST_STAG:
      case ST_MEMBER:
      case ST_MODULE:
      case ST_MODPROC:
      case ST_ALIAS:
        if (for_module) {
          if (!IGNOREG(sptr) &&
              (STYPEG(sptr) != ST_UNKNOWN || SCG(sptr) != SC_NONE) &&
              (SCOPEG(sptr) == sym_module || STYPEG(sptr) == ST_OPERATOR)) {
            if (STYPEG(sptr) == ST_TYPEDEF)
              FROMMODP(sptr, 1);
            queue_symbol(sptr);
          }
        } else if (for_inliner) {
          if (!IGNOREG(sptr) && sptr >= stb.firstusym) {
            if (sptr == gbl.currsub || SCOPEG(sptr) == SCOPEG(gbl.currsub))
              queue_symbol(sptr);
          }
        }
        break;
      case ST_USERGENERIC:
      case ST_PROC:
        if (for_module) {
          if (!IGNOREG(sptr) && SCOPEG(sptr) == sym_module) {
            queue_symbol(sptr);
          }
        } else if (for_inliner) {
          if (!IGNOREG(sptr) && sptr >= stb.firstusym) {
            if (sptr == gbl.currsub || SCOPEG(sptr) == SCOPEG(gbl.currsub))
              queue_symbol(sptr);
          }
        }
        break;
      case ST_LABEL:
      case ST_BLOCK:
        if (for_module) {
          if (!IGNOREG(sptr) && SCOPEG(sptr) == sym_module)
            queue_symbol(sptr);
        } else if (for_inliner) {
          if (!IGNOREG(sptr) && sptr >= stb.firstusym) {
            if (sptr == gbl.currsub || SCOPEG(sptr) == SCOPEG(gbl.currsub))
              queue_symbol(sptr);
          }
        }
        break;
      case ST_NML:
        if (for_module) {
          if (!IGNOREG(sptr) && SCOPEG(sptr) == sym_module) {
            queue_symbol(sptr);
          }
        } else if (exporting_module) {
          queue_symbol(sptr);
        }
        break;
      case ST_ARRAY:
      case ST_DESCRIPTOR:
      case ST_VAR:
      case ST_STRUCT:
      case ST_UNION:
        if (STYPEG(sptr) == ST_DESCRIPTOR && CLASSG(sptr) &&
            SCG(sptr) == SC_EXTERN && sem.mod_dllexport) {
          /* need to export type descriptor */
          DLLP(sptr, DLL_EXPORT);
        }
        if (for_module) {
          if (!IGNOREG(sptr) && SCOPEG(sptr) == sym_module) {
            queue_symbol(sptr);
          }
        }
        break;
      case ST_IDENT:
        if (for_module) {
          if (SCG(sptr) == SC_DUMMY && SCOPEG(SCOPEG(sptr)) != sym_module &&
              TBP_BOUND_TO_SMPG(SCOPEG(sptr))) {
            queue_symbol(sptr);
          }
        }
        break;
      default:
        break;
      }
    }
  }

  exportb.hmark.dt = DT_MAX + 1;

  {
    /* queue up all variables ever used, and
     * all alignment descriptors and distribution descriptors used in
     * realign/redistribute statements */
    if (!for_module)
      all_stds(queue_one_std);
    if (for_module) {
      int evp, evpfirst;
      for (evpfirst = sem.eqvlist; evpfirst; evpfirst = EQV(evpfirst).next) {
        if (EQV(evpfirst).is_first) {
          LOGICAL found = FALSE;
          evp = evpfirst;
          do {
            if (SCOPEG(EQV(evp).sptr) == sym_module ||
                symbol_flag[EQV(evp).sptr]) {
              found = TRUE;
              break;
            }
            evp = EQV(evp).next;
          } while (evp && !EQV(evp).is_first);
          if (found) {
            evp = evpfirst;
            do {
              int ss, numss, j;
              eqv_flag[evp] = 1;
              queue_symbol(EQV(evp).sptr);
              queue_ast(EQV(evp).substring);
              ss = EQV(evp).subscripts;
              numss = EQV_NUMSS(ss);
              /* depends on EQV_NUMSS(0) == 0, set in semant.c */
              for (j = 0; j < numss; ++j) {
                if (EQV_SS(ss, j))
                  queue_ast(EQV_SS(ss, j));
              }
              evp = EQV(evp).next;
            } while (evp && !EQV(evp).is_first);
          }
        }
      }
      export_data_file_asts(rqueue_ast, 1, 1, 0);
      export_component_init_asts(rqueue_ast, 1, 1);
    } else {
      int evp;
      for (evp = sem.eqvlist; evp; evp = EQV(evp).next) {
        int ss, numss, j;
        eqv_flag[evp] = 1;
        queue_symbol(EQV(evp).sptr);
        queue_ast(EQV(evp).substring);
        ss = EQV(evp).subscripts;
        numss = EQV_NUMSS(ss);
        /* depends on EQV_NUMSS(0) == 0, set in semant.c */
        for (j = 0; j < numss; ++j) {
          if (EQV_SS(ss, j))
            queue_ast(EQV_SS(ss, j));
        }
      }
    }
  }
  ast_unvisit();

  outlz = export_header(export_fd, export_name, 0);

  if (for_module) {
    export_iso_c_libraries();
    export_iso_fortran_env_libraries();
    export_ieee_arith_libraries();
  }

  export_dtypes(0, 0);

  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    if (symbol_flag[sptr])
      export_symbol(sptr);
  }

  for (ast = astb.firstuast; ast < astb.stg_avail; ++ast) {
    if (ast >= ast_flag_size || ast_flag[ast])
      export_one_ast(ast);
  }
  {
    exportb.hmark.ast = astb.firstuast;
    exportb.hmark.maxast = astb.stg_avail - 1;
    if (!for_module)
      all_stds(export_one_std);
    export_equivs();
  }

  if (cleanup) {
    freearea(MOD_AREA);
    public_module_list = NULL;
    private_module_list = NULL;
  }

  /* symbols, etc., will be added for the module subprograms */
  append_list = NULL;

  lzprintf(outlz, "Z\n");
  {
    export_data_file(0);
    export_component_init(cleanup);
  }
  lzprintf(outlz, "Z\n");

  FREE(eqv_flag);
  FREE(ast_flag);
  ast_flag_size = 0;
  FREE(dtype_flag);
  dtype_flag_size = 0;
  FREE(symbol_flag);
  symbol_flag_size = 0;
  lzfinifile(outlz);
  outlz = NULL;
  fflush(export_fd);
} /* export */

void
export_iso_c_libraries(void)
{
  int first, last, sptr;

  if (exportb.iso_c_library) {
    Trace(("Exporting ISO_C Library"));
    iso_c_lib_stat(&first, &last, ST_ISOC);
    for (sptr = first; sptr <= last; ++sptr) {
      if (STYPEG(sptr) == ST_INTRIN) {
        lzprintf(outlz, "B %d %s %s\n", sptr, "iso_c_binding", SYMNAME(sptr));
      }
    }
    sptr = lookupsymbol("c_sizeof");
    lzprintf(outlz, "B %d %s %s\n", sptr, "iso_c_binding", SYMNAME(sptr));
  }
} /* export_iso_c_libraries */

void
export_iso_fortran_env_libraries(void)
{
  int sptr;

  if (exportb.iso_fortran_env_library) {
    sptr = lookupsymbol("compiler_options");
    lzprintf(outlz, "B %d %s %s\n", sptr, "iso_fortran_env", SYMNAME(sptr));
    sptr = lookupsymbol("compiler_version");
    lzprintf(outlz, "B %d %s %s\n", sptr, "iso_fortran_env", SYMNAME(sptr));
  }
}

void
export_ieee_arith_libraries(void)
{
  int sptr;

  if (exportb.ieee_arith_library) {
    Trace(("Exporting IEEE_ARITH Library"));
    sptr = get_ieee_arith_intrin("ieee_selected_real_kind");
    lzprintf(outlz, "B %d %s %s\n", sptr, "ieee_arithmetic", SYMNAME(sptr));
  }
}

void
export_inline(FILE *export_fd, char *export_name, char *file_name)
{
  int internal;
  for_inliner = TRUE;
  if (gbl.internal > 1) {
    internal = INTERNALG(gbl.currsub);
    INTERNALP(gbl.currsub, 1);
  }
  export(export_fd, export_name, 1);
  fclose(export_fd);
  if (gbl.internal > 1) {
    INTERNALP(gbl.currsub, internal);
  }
  for_inliner = FALSE;
  sym_module = 0;
} /* export_inline */

/** \brief Save the module file for use when exporting contained subprograms */
void
export_module(FILE *module_file, char *export_name, int modulesym, int cleanup)
{
  Trace(("Exporting module name %s", export_name));
  for_module = TRUE;
  sym_module = modulesym;
  exporting_module = TRUE;
  export(module_file, export_name, cleanup);
  exporting_module = FALSE;
  for_module = FALSE;
  sym_module = 0;
}

void
export_append_sym(int sym)
{
  ITEMX *p;

  Trace(("export append symbol %d %s", sym, SYMNAME(sym)));
  p = (ITEMX *)getitem(APPEND_AREA, sizeof(ITEMX));
  p->val = sym;
  p->next = append_list;
  append_list = p;
}

static ITEMX
    *host_append_list; /* list of symbols to be appended to host file */

static void
mark_idstr(int ast, int *unused)
{
  if (A_TYPEG(ast) == A_ID && SCG(A_SPTRG(ast)) != SC_DUMMY) {
    A_IDSTRP(ast, 1);
  }
}

void
mark_dtype_ast_idstr(int dtype)
{
  int i;
  int ndim;

  if (DTY(dtype) == TY_CHAR) {
    if (DTY(dtype + 1)) {
      ast_traverse(DTY(dtype + 1), NULL, mark_idstr, NULL);
    }
  } else if (DTY(dtype) == TY_ARRAY && DTY(dtype + 2) > 0) {
    ndim = ADD_NUMDIM(dtype);
    for (i = 0; i < ndim; ++i) {
      if (ADD_LWBD(dtype, i)) {
        ast_traverse(ADD_LWBD(dtype, i), NULL, mark_idstr, NULL);
      }
      if (ADD_UPBD(dtype, i)) {
        ast_traverse(ADD_UPBD(dtype, i), NULL, mark_idstr, NULL);
      }
    }
  }
}

void
export_append_host_sym(int sym)
{
  ITEMX *p;

  Trace(("export append symbol %d %s", sym, SYMNAME(sym)));
  p = (ITEMX *)getitem(APPEND_AREA, sizeof(ITEMX));
  p->val = sym;
  p->next = host_append_list;
  host_append_list = p;
}

void
export_fix_host_append_list(int (*newsym)(int))
{
  ITEMX *p;
  int oldv;
  for (p = host_append_list; p != NULL; p = p->next) {
    oldv = p->val;
    p->val = newsym(oldv);
    Trace(("fix host symbol %d to %d", oldv, p->val));
  }
} /* export_fix_host_append_list */

static void
export_some_init()
{
  symbol_flag_size = stb.stg_avail + 1;
  symbol_flag_lowest_const = stb.stg_avail;
  NEW(symbol_flag, char, symbol_flag_size);
  BZERO(symbol_flag, char, stb.stg_avail + 1);

  dtype_flag_size = stb.dt.stg_avail + 1;
  NEW(dtype_flag, char, dtype_flag_size);
  BZERO(dtype_flag, char, dtype_flag_size);

  ast_flag_size = astb.stg_avail + 1;
  ast_flag_lowest_const = astb.stg_avail;
  NEW(ast_flag, char, ast_flag_size);
  BZERO(ast_flag, char, ast_flag_size);

  NEW(eqv_flag, char, sem.eqv_avail + 1);
  BZERO(eqv_flag, char, sem.eqv_avail + 1);

} /* export_some_init */

static void
export_some_procedure(int sptr)
{
  int fval, cnt, dpdsc;
  STYPEP(sptr, ST_PROC);
  for (cnt = PARAMCTG(sptr), dpdsc = DPDSCG(sptr); cnt; --cnt, ++dpdsc) {
    int arg = aux.dpdsc_base[dpdsc];
    IGNOREP(arg, 1);
  }
  fval = FVALG(sptr);
  if (fval) {
    dpdsc = DPDSCG(sptr);
    DTYPEP(sptr, DTYPEG(fval));
    /* If PARAMCT equals zero, it is illegal to access the beginning of the
     * argument list of the function.
     */
    if (PARAMCTG(sptr) > 0 && aux.dpdsc_base[dpdsc] == FVALG(sptr)) {
      DPDSCP(sptr, dpdsc + 1);
      PARAMCTP(sptr, PARAMCTG(sptr) - 1);
    }
    IGNOREP(fval, 1);
  }
} /* export_some_procedure */

static void
export_some_args(int sptr, int limitsptr)
{
  int fval, cnt, dpdsc;
  for (cnt = PARAMCTG(sptr), dpdsc = DPDSCG(sptr); cnt; --cnt, ++dpdsc) {
    int arg = aux.dpdsc_base[dpdsc];
    if (arg < limitsptr) {
      export_symbol(arg);
    }
  }
  fval = FVALG(sptr);
  if (fval) {
    if (fval < limitsptr) {
      export_symbol(fval);
    }
  }
} /* export_some_args */

static void
export_some_fini(int limitsptr, int limitast)
{
  int sptr, ast;
  for (sptr = symbol_flag_lowest_const; sptr < limitsptr; ++sptr) {
    if (symbol_flag[sptr] && STYPEG(sptr) == ST_CONST) {
      export_symbol(sptr);
    }
  }
  for (sptr = limitsptr; sptr < stb.stg_avail; ++sptr) {
    if (symbol_flag[sptr])
      export_symbol(sptr);
  }

  for (ast = ast_flag_lowest_const; ast < limitast; ++ast) {
    if (ast_flag[ast] && A_TYPEG(ast) == A_CNST) {
      export_one_ast(ast);
    }
  }
  for (ast = limitast; ast < astb.stg_avail; ++ast) {
    if (ast >= ast_flag_size || ast_flag[ast])
      export_one_ast(ast);
  }

  export_equivs();

  FREE(eqv_flag);
  FREE(ast_flag);
  ast_flag_size = 0;
  FREE(dtype_flag);
  dtype_flag_size = 0;
  FREE(symbol_flag);
  freearea(MOD_AREA);
  public_module_list = NULL;
  private_module_list = NULL;
  lzprintf(outlz, "Z\n");
} /* export_some_fini */

/* If the type of a contained subprogram return value or argument is a
 * fixed length string, the dtype length (dtype+1) is an ast that will
 * not be exported if the dtype happens to match the dtype of some data
 * item or literal in the host.  Stash the string DTY and length in the
 * symbol table entry so the dtype can be reconstructed when imported.
 */
static void
fixup_host_symbol_dtype(int sptr)
{
  int dtype = DTYPEG(sptr);
  if ((DTY(dtype) == TY_CHAR &&
       (dtype != DT_ASSCHAR || dtype != DT_DEFERCHAR)) ||
      (DTY(dtype) == TY_NCHAR &&
       (dtype != DT_ASSNCHAR || dtype != DT_DEFERNCHAR))) {
    int clen = DTY(dtype + 1);
    if (A_ALIASG(clen)
        /* If CLASS is set, then do not clear CVLEN since it's overloaded by
         * VTOFF and VTABLE which are used with type bound procedures. We
         * may need to revisit this when we implement unlimited polymorphic
         * types.
         */
        && (!CLASSG(sptr) ||
            (STYPEG(sptr) != ST_MEMBER && STYPEG(sptr) != ST_PROC))) {
      DTYPEP(sptr, 0);
      clen = CONVAL2G(A_SPTRG(A_ALIASG(clen)));
      /* HACK clen < 0 ==> TY_NCHAR */
      if (DTY(dtype) == TY_NCHAR) {
        clen = -clen;
      }
      CVLENP(sptr, clen);
    }
  } else if (DTY(dtype) == TY_ARRAY && ADJARRG(sptr)) {
    /* similar to above condition if the bound is host symbol
     * symbol will not be exported.
     */
    if (DTY(dtype + 2) > 0) {
      ast_visit(1, 1);
      mark_dtype_ast_idstr(dtype);
      ast_unvisit();
    }
  }
}

void
export_host_subprogram(FILE *host_file, int host_sym, int limitsptr,
                       int limitast, int limitdtype)
{
  ITEMX *p;
  Trace(("write host subprogram %d %s", host_sym, SYMNAME(host_sym)));
  if (host_file == NULL) {
    interr("no file to which to export contained subprogram", 0, 3);
  }
  if (sem.mod_cnt) {
    sym_module = sem.mod_sym;
  }
  for_contained = TRUE;
  export_some_init();
  Trace(
      ("limits are sptr=%d, ast=%d, dty=%d", limitsptr, limitast, limitdtype));

  for (p = host_append_list; p != NULL; p = p->next) {
    export_some_procedure(p->val);
    INTERNALP(p->val, 1);
  }
  for (p = host_append_list; p != NULL; p = p->next) {
    fixup_host_symbol_dtype(p->val);
    ast_visit(1, 1);
    queue_symbol(p->val);
    ast_unvisit();
  }

  outlz = export_header(host_file, "host file", 0);

  export_outer_derived_dtypes(limitdtype);
  if (gbl.internal && FVALG(gbl.currsub) &&
      (DTY(DTYPEG(FVALG(gbl.currsub))) != TY_ARRAY ||
       !ADD_DEFER(DTYPEG(FVALG(gbl.currsub))))) {
    ast_visit(1, 1);
    mark_dtype_ast_idstr(DTYPEG(FVALG(gbl.currsub)));
    ast_unvisit();
  }
  export_dtypes(limitdtype, 0);

  for (p = host_append_list; p != NULL; p = p->next) {
    if (p->val < limitsptr) {
      export_symbol(p->val);
    }
    export_some_args(p->val, limitsptr);
  }

  export_some_fini(limitsptr, limitast);
  lzfinifile(outlz);
  outlz = NULL;
  fflush(host_file);
  sym_module = 0;
  for_contained = FALSE;
} /* export_host_subprogram */

void
export_module_subprogram(FILE *subprog_file, int subprog_sym, int limitsptr,
                         int limitast, int limitdtype)
{
  ITEMX *p;
  int sptr;
  Trace(("write module subprogram %d %s", subprog_sym, SYMNAME(subprog_sym)));
  if (subprog_file == NULL) {
    interr("no file to which to export contained subprogram", 0, 3);
  }
  export_some_init();
  Trace(
      ("limits are sptr=%d, ast=%d, dty=%d", limitsptr, limitast, limitdtype));

  ENCLFUNCP(subprog_sym, sem.mod_sym);
  if (STYPEG(subprog_sym) == ST_ALIAS) {
    ENCLFUNCP(SYMLKG(subprog_sym), sem.mod_sym);
  }
  sym_module = sem.mod_sym;
  for_contained = TRUE;
  for (sptr = subprog_sym; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    export_some_procedure(sptr);
    INMODULEP(sptr, 1);
    ast_visit(1, 1);
    queue_symbol(sptr);
    ast_unvisit();
  }
  for (p = append_list; p != NULL; p = p->next) {
    ast_visit(1, 1);
    queue_symbol(p->val);
    ast_unvisit();
  }

  /*
   * Ensure that certain symbols are ignored by the compiler when
   * read from the module file; these symbol need to have their
   * IGNORE & HIDDEN flags set when exported. Typically, these
   * symbols were discovered from the specification of the dummy
   * arguments and are 'local' to the contained subprogram.
   */
  for (sptr = stb.stg_avail - 1; sptr > limitsptr; sptr--) {
    if (symbol_flag[sptr])
      switch (STYPEG(sptr)) {
      case ST_IDENT:
      case ST_VAR:
      case ST_ARRAY:
      case ST_STRUCT:
      case ST_UNION:
      case ST_CMBLK:
      case ST_PARAM:
        if (SCOPEG(sptr) && (SCOPEG(sptr) != sym_module) && !CFUNCG(sptr)) {
          /*
           * If symbol doesn't have module scope, assume it's
           * local.  Another way of determine if the symbol is
           * local:
           * -  the symbol's SCOPE is subprog_sym, or
           * -  if SCOPE of subprog_sym is an ST_ALIAS, the symbol's
           *    SCOPE is the alias.
           * CFUNCG : externally visable "C" style variable, type
           * or common block
           */
          HIDDENP(sptr, 1);
          IGNOREP(sptr, 1);
          Trace(("Ignore %d(%s) in %d(%s)", sptr, SYMNAME(sptr), subprog_sym,
                 SYMNAME(subprog_sym)));
        }
        FLANG_FALLTHROUGH;
      case ST_TYPEDEF:
        if (SCOPEG(sptr) && (SCOPEG(sptr) != sym_module)) {
          /*
           * If symbol doesn't have module scope, assume it's
           * local.  Another way of determine if the symbol is
           * local:
           * -  the symbol's SCOPE is subprog_sym, or
           * -  if SCOPE of subprog_sym is an ST_ALIAS, the symbol's
           *    SCOPE is the alias.
           */
          HIDDENP(sptr, 1);
          IGNOREP(sptr, 1);
          Trace(("Ignore %d(%s) in %d(%s)", sptr, SYMNAME(sptr), subprog_sym,
                 SYMNAME(subprog_sym)));
        }
        break;
      case ST_ENTRY:
      case ST_PROC:
        if (sem.mod_dllexport && ENCLFUNCG(sptr) == gbl.currmod) {
          DLLP(sptr, DLL_EXPORT);
        }
        break;

      default:
        break;
      }
  }

  outlz = export_header(subprog_file, "module-contained subprogram file", 0);

  export_dtypes(limitdtype, 1);

  for (sptr = subprog_sym; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    if (sptr < limitsptr) {
      export_symbol(sptr);
      export_some_args(sptr, limitsptr);
    }
  }
  for (p = append_list; p != NULL; p = p->next) {
    if (STYPEG(p->val) == ST_MODPROC) {
      export_symbol(p->val);
    }
  }
  for (p = append_list; p != NULL; p = p->next) {
    if (STYPEG(p->val) != ST_MODPROC) {
      export_symbol(p->val);
    }
  }
  append_list = NULL;

  export_some_fini(limitsptr, limitast);
  lzfinifile(outlz);
  outlz = NULL;
  fflush(subprog_file);
  sym_module = 0;
  for_contained = FALSE;
} /* export_module_subprogram */

void
exterf_init()
{
  freearea(APPEND_AREA);
  append_list = NULL;
  host_append_list = NULL;
} /* exterf_init */

void
exterf_init_host()
{
  host_append_list = NULL;
} /* exterf_init_host */

static VAR *
export_ivl_asts(VAR *ivl, ast_visit_fn astproc)
{
  do {
    if (ivl->u.varref.subt) {
      export_ivl_asts(ivl->u.varref.subt, astproc);
    } else {
      ast_traverse(ivl->u.varref.ptr, NULL, astproc, NULL);
    }
    ivl = ivl->next;
  } while (ivl != NULL && ivl->id == Varref);
  return ivl;
} /* export_ivl_asts */

static void
export_ict_asts(ACL *ict, ast_visit_fn astproc, int queuesym, int queuedtype,
                int domarkdtype)
{
  for (; ict != NULL; ict = ict->next) {
    if (queuesym && ict->sptr)
      queue_symbol(ict->sptr);
    if (ict->id == AC_IDO) {
      if (ict->u1.doinfo->init_expr)
        ast_traverse(ict->u1.doinfo->init_expr, NULL, astproc, NULL);
      if (ict->u1.doinfo->limit_expr)
        ast_traverse(ict->u1.doinfo->limit_expr, NULL, astproc, NULL);
      if (ict->u1.doinfo->step_expr)
        ast_traverse(ict->u1.doinfo->step_expr, NULL, astproc, NULL);
      if (ict->u1.doinfo->count)
        ast_traverse(ict->u1.doinfo->count, NULL, astproc, NULL);
      if (queuesym && ict->u1.doinfo->index_var)
        queue_symbol(ict->u1.doinfo->index_var);
    }
    if (queuedtype) {
      if (ict->dtype)
        queue_dtype(ict->dtype);
      if (ict->ptrdtype)
        queue_dtype(ict->ptrdtype);
    }
    if (!ict->subc) {
      if (ict->id == AC_IEXPR) {
        int dtype = ict->dtype;
        if (DTY(dtype) == TY_DERIVED) {
          if (queuesym && DTY(dtype + 3))
            queue_symbol(DTY(dtype + 3));
        }
        if (queuedtype)
          queue_dtype(ict->dtype);
        export_ict_asts(ict->u1.expr->lop, astproc, queuesym, queuedtype,
                        domarkdtype);
        if (BINOP(ict->u1.expr)) {
          export_ict_asts(ict->u1.expr->rop, astproc, queuesym, queuedtype,
                          domarkdtype);
        }
      } else {
        if (queuedtype)
          queue_dtype(ict->dtype);
        if (ict->u1.ast > 0 && ict->u1.ast <= astb.stg_avail)
          ast_traverse(ict->u1.ast, NULL, astproc, NULL);
      }
      if (ict->repeatc) {
        ast_traverse(ict->repeatc, NULL, astproc, NULL);
      }
    } else {
      int dtype = ict->dtype;
      if (DTY(dtype) == TY_DERIVED) {
        if (queuesym && DTY(dtype + 3))
          queue_symbol(DTY(dtype + 3));
      }
      if (ict->repeatc) {
        ast_traverse(ict->repeatc, NULL, astproc, NULL);
      }
      if (queuedtype)
        queue_dtype(ict->dtype);
      export_ict_asts(ict->subc, astproc, queuesym, queuedtype, domarkdtype);
    }
  }
} /* export_ict_asts */

static void
export_ivl_ict_asts(VAR *ivl, ACL *ict, ast_visit_fn astproc, int queuesym,
                    int queuedtype, int domarkdtype)
{
  /* ignore structures except for IPA */
  if (!exportmode && ivl == NULL && ict->subc != NULL)
    return;
  if (!ivl) {
    if (queuesym && ict->sptr)
      queue_symbol(ict->sptr);
  } else {
    VAR *next;
    for (; ivl != NULL; ivl = next) {
      next = ivl->next;
      switch (ivl->id) {
      case Dostart:
        ast_traverse(ivl->u.dostart.indvar, NULL, astproc, NULL);
        ast_traverse(ivl->u.dostart.lowbd, NULL, astproc, NULL);
        ast_traverse(ivl->u.dostart.upbd, NULL, astproc, NULL);
        if (ivl->u.dostart.step) {
          ast_traverse(ivl->u.dostart.step, NULL, astproc, NULL);
        }
        break;
      case Doend:
        break;
      case Varref:
        next = export_ivl_asts(ivl, astproc);
        break;
      default:
        break;
      }
    }
  }
  export_ict_asts(ict, astproc, queuesym, queuedtype, domarkdtype);
} /* export_ivl_ict_asts */

static void
export_data_file_asts(ast_visit_fn astproc, int queuesym, int queuedtype,
                      int domarkdtype)
{
  int nw, lineno, fileno;
  VAR *ivl;
  ACL *ict;
  if (astb.df == NULL)
    return;
  nw = fseek(astb.df, 0L, 0);
#if DEBUG
  assert(nw == 0, "export_data_file_asts: rewind error", nw, 4);
#endif
  while (1) {
    nw = fread(&lineno, sizeof(lineno), 1, astb.df);
    if (nw == 0)
      break;
#if DEBUG
    assert(nw == 1, "export_data_file_asts: lineno error", nw, 4);
#endif
    nw = fread(&fileno, sizeof(fileno), 1, astb.df);
    if (nw == 0)
      break;
#if DEBUG
    assert(nw == 1, "export_dinit_file: fileno error", nw, 4);
#endif
    nw = fread(&ivl, sizeof(ivl), 1, astb.df);
    if (nw == 0)
      break;
#if DEBUG
    assert(nw == 1, "export_data_file_asts: ivl error", nw, 4);
#endif
    nw = fread(&ict, sizeof(ict), 1, astb.df);
#if DEBUG
    assert(nw == 1, "export_data_file_asts: ict error", nw, 4);
#endif
    export_ivl_ict_asts(ivl, ict, astproc, queuesym, queuedtype, domarkdtype);
  } /* while */
} /* export_data_file_asts */

static void
export_component_init_asts(ast_visit_fn astproc, int queuesym, int queuedtype)
{
  int dtype;

  for (dtype = DT_MAX + 1; dtype < stb.dt.stg_avail;) {
    if (DTY(dtype) == TY_DERIVED) {
      ACL *ict = (ACL *)get_getitem_p(DTY(dtype + 5));
      if (ict) {
        export_ict_asts(ict, astproc, queuesym, queuedtype, 0);
      }
    }
    dtype += dtype_skip(dtype);
  }
}

static VAR *
export_ivl(VAR *ivl)
{
  do {
    int more = 0;
    if (ivl->next)
      more = 1;
    if (ivl->u.varref.subt) {
      lzprintf(outlz, "W %d %d\n", ivl->u.varref.dtype, more);
      export_ivl(ivl->u.varref.subt);
    } else {
      lzprintf(outlz, "V %d %d %d %d\n", ivl->u.varref.ptr, ivl->u.varref.dtype,
               ivl->u.varref.id, more);
    }
    ivl = ivl->next;
  } while (ivl != NULL && ivl->id == Varref);
  return ivl;
} /* export_ivl */

static void
export_ict(ACL *ict)
{
  for (; ict != NULL; ict = ict->next) {
    int more = 0, nosubc = 0;
    if (ict->next)
      more = 1;
    if (ict->subc == NULL)
      nosubc = 1;
    switch (ict->id) {
    case AC_IDENT:
      lzprintf(outlz, "I %d %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->repeatc, ict->u1.ast, more);
      break;
    case AC_CONST:
      lzprintf(outlz, "C %d %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->repeatc, ict->u1.ast, more);
      break;
    case AC_AST:
      lzprintf(outlz, "A %d %d %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->repeatc, (int)ict->is_const, ict->u1.ast,
               more);
      break;
    case AC_ACONST:
      lzprintf(outlz, "R %d %d %d %d\n", ict->sptr, ict->dtype, ict->ptrdtype,
               more);
      export_ict(ict->subc);
      break;
    case AC_SCONST:
      lzprintf(outlz, "S %d %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->repeatc, more, nosubc);
      export_ict(ict->subc);
      break;
    case AC_IDO:
      lzprintf(outlz, "O %d %d %d %d %d\n", ict->u1.doinfo->index_var,
               ict->u1.doinfo->init_expr, ict->u1.doinfo->limit_expr,
               ict->u1.doinfo->step_expr, more);
      export_ict(ict->subc);
      break;
    case AC_REPEAT:
      lzprintf(outlz, "P %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->u1.ast, more);
      break;
    case AC_VMSUNION:
      lzprintf(outlz, "U %d %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->repeatc, ict->u1.ast, more);
      export_ict(ict->subc);
      break;
    case AC_TYPEINIT:
      lzprintf(outlz, "T %d %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->repeatc, ict->u1.ast, more);
      export_ict(ict->subc);
      break;
    case AC_VMSSTRUCT:
      lzprintf(outlz, "V %d %d %d %d %d %d\n", ict->sptr, ict->dtype,
               ict->ptrdtype, ict->repeatc, ict->u1.ast, more);
      export_ict(ict->subc);
      break;
    case AC_IEXPR:
      lzprintf(outlz, "X %d %d %d %d %d %d\n", ict->u1.expr->op, ict->sptr,
               ict->dtype, ict->ptrdtype, ict->repeatc, more);
      if (ict->u1.expr->lop)
        export_ict(ict->u1.expr->lop);
      else
        lzprintf(outlz, "N\n");
      if (BINOP(ict->u1.expr)) {
        if (ict->u1.expr->rop)
          export_ict(ict->u1.expr->rop);
        else
          lzprintf(outlz, "N\n");
      }
      break;
    case AC_ICONST:
      lzprintf(outlz, "L %d %d\n", ict->u1.i, more);
      break;
    default:
      interr("Attempt to export an unknown initializer type\n", ict->id, 3);
      return;
    }
  }
} /* export_ict */

static void
export_ivl_ict(int lineno, VAR *ivl, ACL *ict, int dostructures)
{
  /* ignore structures */
  if (ivl == NULL && ict->subc != NULL && !dostructures)
    return;

  if (for_module && ivl) {
    /* put out initializations for named constants ONLY */
    if (ivl->next) {
      /* data statement, can't be a named constant */
      return;
    } else if (ivl->id == Varref) {
      int sptr = sym_of_ast(ivl->u.varref.ptr);
      if (!PARAMG(sptr)) {
        return;
      }
    }
  }

  if (ivl == NULL) {
    lzprintf(outlz, "J %d 0 1\n", lineno);
  } else {
    VAR *next;
    lzprintf(outlz, "J %d 1 1\n", lineno);
    for (; ivl != NULL; ivl = next) {
      int more = 0;
      next = ivl->next;
      if (next)
        more = 1;
      switch (ivl->id) {
      case Dostart:
        lzprintf(outlz, "D %d %d %d %d %d\n", ivl->u.dostart.indvar,
                 ivl->u.dostart.lowbd, ivl->u.dostart.upbd, ivl->u.dostart.step,
                 more);
        break;
      case Doend:
        lzprintf(outlz, "E %d\n", more);
        break;
      case Varref:
        next = export_ivl(ivl);
        break;
      default:
        break;
      }
    }
  }
  export_ict(ict);
} /* export_ivl_ict */

static void
export_component_init(int cleanup)
{
  int dtype, flag;
  flag = 2;
  if (cleanup)
    flag = 1;

  for (dtype = DT_MAX + 1; dtype < stb.dt.stg_avail;) {
    if (DTY(dtype) == TY_DERIVED) {
      ACL *ict = (ACL *)get_getitem_p(DTY(dtype + 5));
      if (ict && (ict->ci_exprt & flag) == 0) {
        export_ict(ict);
        ict->ci_exprt |= flag;
      }
    }
    dtype += dtype_skip(dtype);
  }
}

static void
export_data_file(int dostructures)
{
  int nw, lineno, fileno;
  VAR *ivl;
  ACL *ict;
  if (astb.df == NULL)
    return;
  nw = fseek(astb.df, 0L, 0);
  while (1) {
    nw = fread(&lineno, sizeof(lineno), 1, astb.df);
    if (nw == 0)
      break;
    nw = fread(&fileno, sizeof(fileno), 1, astb.df);
    if (nw == 0)
      break;
    nw = fread(&ivl, sizeof(ivl), 1, astb.df);
    if (nw == 0)
      break;
    nw = fread(&ict, sizeof(ict), 1, astb.df);
    export_ivl_ict(lineno, ivl, ict, dostructures);
  } /* while */
} /* export_data_file */

/* ----------------------------------------------------------- */
static void
rqueue_ast(int ast, int *unused)
{
  int s, i, cnt;
  if (!ast)
    return;
  if (ast < ast_flag_size) {
    if (ast_flag[ast])
      return;
    ast_flag[ast] = 1;
  }
  switch (A_TYPEG(ast)) {
  case A_ID:
    if (A_ALIASG(ast) && A_ALIASG(ast) != ast)
      queue_ast(A_ALIASG(ast));
    FLANG_FALLTHROUGH;
  case A_CNST:
    if (ast < ast_flag_lowest_const)
      ast_flag_lowest_const = ast;
    FLANG_FALLTHROUGH;
  case A_LABEL:
  case A_INIT:
    if (A_SPTRG(ast) && A_SPTRG(ast) < symbol_flag_size)
      queue_symbol(A_SPTRG(ast));
    if (A_DTYPEG(ast) && A_SPTRG(ast) < symbol_flag_size)
      queue_dtype(A_DTYPEG(ast));
    break;
  case A_ALLOC:
    if (A_DTYPEG(ast))
      queue_dtype(A_DTYPEG(ast));
    break;
  case A_FUNC:
  case A_INTR:
    if (A_DTYPEG(ast))
      queue_dtype(A_DTYPEG(ast));
    s = A_SHAPEG(ast);
    if (s) {
      cnt = SHD_NDIM(s);
      for (i = 0; i < cnt; ++i) {
        int bound;
        if ((bound = SHD_LWB(s, i)))
          queue_ast(bound);
        if ((bound = SHD_UPB(s, i)))
          queue_ast(bound);
        if ((bound = SHD_STRIDE(s, i)))
          queue_ast(bound);
      }
    }
    break;
  case A_FORALL:
  case A_IF:
  case A_IFTHEN:
  case A_ELSEIF:
  case A_DOWHILE:
  case A_AIF:
  case A_WHERE:
    queue_ast(A_IFSTMTG(ast));
    break;
  case A_MP_TARGET:
  case A_MP_TARGETDATA:
    queue_ast(A_IFPARG(ast));
    queue_ast(A_LOPG(ast));
    break;
  case A_MP_TARGETUPDATE:
  case A_MP_TARGETEXITDATA:
  case A_MP_TARGETENTERDATA:
    queue_ast(A_IFPARG(ast));
    break;
  case A_MP_ENDTARGETDATA:
  case A_MP_ENDTARGET:
    queue_ast(A_LOPG(ast));
    break;
  case A_MP_PARALLEL:
    queue_ast(A_IFPARG(ast));
    queue_ast(A_NPARG(ast));
    queue_ast(A_LOPG(ast));
    queue_ast(A_ENDLABG(ast));
    queue_ast(A_PROCBINDG(ast));
    break;
  case A_MP_TEAMS:
    queue_ast(A_NTEAMSG(ast));
    queue_ast(A_THRLIMITG(ast));
    queue_ast(A_LOPG(ast));
    break;
  case A_MP_BMPSCOPE:
    queue_ast(A_STBLKG(ast));
    break;
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
    queue_ast(A_LOPG(ast));
    queue_symbol(A_MEMG(ast));
    break;
  case A_MP_CANCEL:
    queue_ast(A_IFPARG(ast));
    FLANG_FALLTHROUGH;
  case A_MP_SECTIONS:
  case A_MP_CANCELLATIONPOINT:
    queue_ast(A_ENDLABG(ast));
    break;
  case A_MP_PDO:
    queue_ast(A_DOLABG(ast));
    queue_ast(A_DOVARG(ast));
    queue_ast(A_LASTVALG(ast));
    queue_ast(A_M1G(ast));
    queue_ast(A_M2G(ast));
    queue_ast(A_M3G(ast));
    queue_ast(A_CHUNKG(ast));
    queue_ast(A_ENDLABG(ast));
    break;
  case A_MP_ATOMICREAD:
    queue_ast(A_SRCG(ast));
    break;
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    queue_ast(A_LOPG(ast));
    queue_ast(A_ROPG(ast));
    break;
  case A_MP_PRE_TLS_COPY:
  case A_MP_COPYIN:
  case A_MP_COPYPRIVATE:
    queue_ast(A_ROPG(ast));
    queue_symbol(A_SPTRG(ast));
    break;
  case A_MP_TASK:
    queue_ast(A_IFPARG(ast));
    queue_ast(A_FINALPARG(ast));
    queue_ast(A_PRIORITYG(ast));
    queue_ast(A_LOPG(ast));
    queue_ast(A_ENDLABG(ast));
    break;
  case A_MP_TASKLOOP:
    queue_ast(A_IFPARG(ast));
    queue_ast(A_FINALPARG(ast));
    queue_ast(A_PRIORITYG(ast));
    queue_ast(A_LOPG(ast));
    break;
  case A_MP_TASKLOOPREG:
    queue_ast(A_M1G(ast));
    queue_ast(A_M2G(ast));
    queue_ast(A_M3G(ast));
    break;
  case A_MP_TASKFIRSTPRIV:
    queue_ast(A_LOPG(ast));
    queue_ast(A_ROPG(ast));
    break;
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_ENDPARALLEL:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_ENDSECTIONS:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
    queue_ast(A_LOPG(ast));
    break;
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_BARRIER:
  case A_MP_ENDPDO:
  case A_MP_BCOPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_BPDO:
  case A_MP_ETASKDUP:
  case A_MP_ETASKLOOPREG:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_EMPSCOPE:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_FLUSH:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
    break;
  default:
    if (A_DTYPEG(ast))
      queue_dtype(A_DTYPEG(ast));
    break;
  }
} /* rqueue_ast */

static void
queue_ast(int ast)
{
  if (ast)
    ast_traverse(ast, NULL, rqueue_ast, NULL);
} /* queue_ast */

static void
queue_dtype(int dtype)
{
  int ndim, i, sptr;
  int paramct;

  if (dtype < DT_MAX)
    return;

  if (dtype < dtype_flag_size) {
    if (dtype_flag[dtype])
      return;
    dtype_flag[dtype] = 1;
  }

  switch (DTY(dtype)) {
  case TY_PTR:
    queue_dtype(DTY(dtype + 1));
    break;
  case TY_ARRAY:
    queue_dtype(DTY(dtype + 1));
    if (DTY(dtype + 2) > 0) {
      ndim = ADD_NUMDIM(dtype);
      for (i = 0; i < ndim; ++i) {
        queue_ast(ADD_LWBD(dtype, i));
        queue_ast(ADD_UPBD(dtype, i));
        queue_ast(ADD_LWAST(dtype, i));
        queue_ast(ADD_UPAST(dtype, i));
        queue_ast(ADD_EXTNTAST(dtype, i));
        queue_ast(ADD_MLPYR(dtype, i));
      }
      queue_ast(ADD_ZBASE(dtype));
      queue_ast(ADD_NUMELM(dtype));
    }
    break;
  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    /* mark all members */
    for (sptr = DTY(dtype + 1); sptr > NOSYM; sptr = SYMLKG(sptr)) {
#ifdef PARENTG
      int parent = PARENTG(sptr);
      if (parent == sptr && sptr >= symbol_flag_size) {
        /* this can occur when user declares a type extension that is
         * local to a particular module procedure and we're now
         * exporting all dtypes at an "end module" statement. The
         * parent symbol is local to the module procedure but not
         * the other parts of the module. In this case, we do not queue
         * the dtype. FS#20816
         */
        return;
      }
#endif
      queue_symbol(sptr);
    }
    /* mark tag (structure name) */
    if (DTY(dtype + 3))
      queue_symbol(DTY(dtype + 3));
    break;
  case TY_CHAR:
  case TY_NCHAR:
    queue_ast(DTY(dtype + 1));
    break;
  case TY_PROC:
    queue_dtype(DTY(dtype + 1));
    if (DTY(dtype + 2)) /* interface */
      queue_symbol(DTY(dtype + 2));
    paramct = DTY(dtype + 3);
    if (paramct) {
      int *dscptr;
      for (dscptr = aux.dpdsc_base + DTY(dtype + 4); paramct > 0; paramct--) {
        queue_symbol(*dscptr);
        dscptr++;
      }
    }
    if (DTY(dtype + 5)) /* FVAL */
      queue_symbol(DTY(dtype + 5));

    break;
  }
} /* queue_dtype */

static void
add_to_private_mod_list(int sptr)
{
  ITEMX *p;
  for (p = private_module_list; p; p = p->next) {
    if (sptr == p->val) {
      return;
    }
  }
  p = (ITEMX *)getitem(MOD_AREA, sizeof(ITEMX));
  p->val = sptr;
  p->next = private_module_list;
  private_module_list = p;
}

/*  this symbol is referenced either directly or indirectly
 *  for the current function.  Arrange to have it written to
 *  output file:
 */
static void
queue_symbol(int sptr)
{
  int i, member;
  int stype, dtype;
  int dscptr;
  static LOGICAL recur_flag = FALSE;

#if DEBUG
  assert(sptr > 0, "queue_symbol, bad sptr", sptr, 2);
  if (sptr >= symbol_flag_size) {
    interr("queue_symbol, symbol_flag subscript too large", sptr, 4);
  }
#endif
  stype = STYPEG(sptr);
  if (stype == ST_UNKNOWN && !for_module && sptr == gbl.sym_nproc) {
    return;
  }
  if (symbol_flag[sptr])
    return;
  symbol_flag[sptr] = 1;

  /*  don't need to process predefined symbols:  */
  if (sptr < stb.firstosym)
    return;

  if (for_module || for_inliner || for_contained ||
      (exportmode && XBIT(66, 0x20000000))) {
    int scope, scope2;
    scope = SCOPEG(sptr);
    for (scope2 = scope; scope2; scope2 = SCOPEG(scope2)) {
      if (STYPEG(scope2) == ST_MODULE) {
        scope = scope2;
      }
      if ((STYPEG(scope2) == ST_ENTRY && scope2 != sptr) ||
          (STYPEG(scope2) == ST_ALIAS && STYPEG(SYMLKG(scope2)) == ST_ENTRY &&
           SYMLKG(scope2) != sptr)) {
        scope = scope2;
        break;
      }
      if ((STYPEG(scope2) == ST_PROC && scope2 != sptr) ||
          (STYPEG(scope2) == ST_ALIAS && STYPEG(SYMLKG(scope2)) == ST_PROC &&
           SYMLKG(scope2) != sptr)) {
        scope = scope2;
        break;
      }
      if (SCOPEG(scope2) == scope2)
        break;
    }
    if (for_inliner && (sptr == gbl.currsub || SCOPEG(sptr) == stb.curr_scope ||
                        SCG(sptr) == SC_DUMMY)) {
      /* export symbols from this subprogram as normal */
    } else if (sptr == gbl.currsub) {
    } else if (scope >= stb.firstosym && scope != sym_module &&
               STYPEG(scope) == ST_MODULE && stype != ST_MODULE) {
      /* putting out a "R " record. */
      queue_symbol(scope);
      switch (stype) {
      case ST_USERGENERIC:
      case ST_OPERATOR:
        for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr)) {
          int ds = SYMI_SPTR(dscptr);
          if (SCOPEG(ds) == stb.curr_scope) {
            queue_symbol(SYMI_SPTR(dscptr));
          }
        }
        break;
#ifdef ENCLDTYPEG
      case ST_MEMBER:
        /* enqueue the derived type tag */
        dtype = ENCLDTYPEG(sptr);
        if (DTY(dtype + 3))
          queue_symbol(DTY(dtype + 3));
        break;
#endif
      }
      return;
    }
  }
  dtype = DTYPEG(sptr);
  if (dtype)
    queue_dtype(dtype);

  /* Process newly added symbol */
  switch (stype) {
  case ST_MODULE:
    if (sptr != sym_module && !exportmode) {
      add_to_private_mod_list(sptr);
    }
    break;
  case ST_UNKNOWN:
  case ST_LABEL:
  case ST_STFUNC:
    break;
  case ST_ARRDSC:
    if (SECDSCG(sptr))
      queue_symbol(SECDSCG(sptr));
    if (ARRAYG(sptr))
      queue_symbol(ARRAYG(sptr));
    break;
  case ST_TYPEDEF:
  case ST_STAG:
    if (BASETYPEG(sptr)) {
      queue_dtype(BASETYPEG(sptr));
    }
    if (PARENTG(sptr)) {
      queue_symbol(PARENTG(sptr));
    }
    if (SDSCG(sptr) && CLASSG(SDSCG(sptr))) {
      queue_symbol(SDSCG(sptr));
    }
    if (TYPDEF_INITG(sptr) > NOSYM) {
      queue_symbol(TYPDEF_INITG(sptr));
    }
    break;
  case ST_IDENT:
    if (DESCRG(sptr))
      queue_symbol(DESCRG(sptr));
    if (ADJARRG(sptr) && SYMLKG(sptr) != NOSYM)
      queue_symbol(SYMLKG(sptr));
    if (ADJLENG(sptr) && ADJSTRLKG(sptr) && ADJSTRLKG(sptr) != NOSYM)
      queue_symbol(ADJSTRLKG(sptr));

    if (SDSCG(sptr))
      queue_symbol(SDSCG(sptr));
#ifdef DEVCOPYG
    if (DEVCOPYG(sptr))
      queue_symbol(DEVCOPYG(sptr));
#endif
    break;

  case ST_CONST:
    if (sptr < symbol_flag_lowest_const)
      symbol_flag_lowest_const = sptr;
    if (DTY(DTYPEG(sptr)) == TY_PTR) /* address constant */
      if (CONVAL1G(sptr)) {
        queue_symbol((int)CONVAL1G(sptr));
      }
    switch (DTY(DTYPEG(sptr))) {
    case TY_DCMPLX:
    case TY_QCMPLX:
      queue_symbol((int)CONVAL1G(sptr));
      queue_symbol((int)CONVAL2G(sptr));
      break;
    case TY_HOLL:
      queue_symbol((int)CONVAL1G(sptr));
      break;
    case TY_NCHAR:
      queue_symbol((int)CONVAL1G(sptr));
      break;
    default:
      break;
    }
    break;

  case ST_ENTRY:
  case ST_PROC:
    if (STYPEG(sptr) == ST_PROC) {
      if (ASSOC_PTRG(sptr) > NOSYM) {
        queue_symbol(ASSOC_PTRG(sptr));
      }
      if (PTR_TARGETG(sptr) > NOSYM) {
        queue_symbol(PTR_TARGETG(sptr));
      }
      if (IS_PROC_DUMMYG(sptr) && SDSCG(sptr)) {
        queue_symbol(SDSCG(sptr));
      }
    }
    if (FVALG(sptr)) {
      queue_symbol(FVALG(sptr));
    }
    if (ALTNAMEG(sptr)) {
      queue_symbol(ALTNAMEG(sptr));
    }
    if (GSAMEG(sptr))
      queue_symbol((int)GSAMEG(sptr));
    dscptr = DPDSCG(sptr);
    for (i = PARAMCTG(sptr); i > 0; i--) {
      int arg;
      arg = aux.dpdsc_base[dscptr];
      if (arg) {
        queue_symbol(arg);
      }
      dscptr++;
    }
    if (CLASSG(sptr) && TBPLNKG(sptr)) {
      queue_dtype(TBPLNKG(sptr));
    }
    break;

  case ST_PARAM:
    if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
      if (CONVAL1G(sptr)) {
        queue_symbol(CONVAL1G(sptr));
      }
    } else if (DTY(DTYPEG(sptr)) == TY_DERIVED) {
      if (CONVAL1G(sptr))
        queue_symbol(CONVAL1G(sptr));
      if (PARAMVALG(sptr))
        queue_ast(PARAMVALG(sptr));
    } else {
      if (!TY_ISWORD(DTY(DTYPEG(sptr)))) {
        queue_symbol(CONVAL1G(sptr));
      }
      queue_ast(CONVAL2G(sptr));
    }
    break;

  case ST_MEMBER:
    queue_symbol(SYMLKG(sptr));
    if (PSMEMG(sptr))
      queue_symbol(PSMEMG(sptr));
    if (VARIANTG(sptr))
      queue_symbol(VARIANTG(sptr));
    if (MIDNUMG(sptr))
      queue_symbol(MIDNUMG(sptr));
    if (SDSCG(sptr))
      queue_symbol(SDSCG(sptr));
    if (PTROFFG(sptr))
      queue_symbol(PTROFFG(sptr));
    if (DESCRG(sptr))
      queue_symbol(DESCRG(sptr));
    if (ENCLDTYPEG(sptr))
      queue_dtype(ENCLDTYPEG(sptr));
    if (PARENTG(sptr))
      queue_symbol(PARENTG(sptr));
    if (VTABLEG(sptr))
      queue_symbol(VTABLEG(sptr));
    if (PASSG(sptr))
      queue_symbol(PASSG(sptr));
    if (IFACEG(sptr))
      queue_symbol(IFACEG(sptr));
    if (BINDG(sptr))
      queue_symbol(BINDG(sptr));
    if (LENG(sptr) && LENPARMG(sptr))
      queue_ast(LENG(sptr));
    if (INITKINDG(sptr) && PARMINITG(sptr))
      queue_ast(PARMINITG(sptr));
    if (KINDASTG(sptr))
      queue_ast(KINDASTG(sptr));
    break;

    /* ELSE, FALL THROUGH: */

  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_VAR:
  case ST_STRUCT:
  case ST_UNION:
    if (!recur_flag) {
      if (CFUNCG(sptr)) {
        /* externally visible C_BIND var, struct */
        queue_symbol(sptr);

        if (ALTNAMEG(sptr)) {
          queue_symbol(ALTNAMEG(sptr));
        }
      } else if (SCG(sptr) == SC_CMBLK) {
#if DEBUG
        assert(STYPEG(CMBLKG(sptr)) == ST_CMBLK, "q_s:CMBLK?", sptr, 2);
#endif
        queue_symbol((int)CMBLKG(sptr));
      }
    }

    if (MIDNUMG(sptr))
      queue_symbol(MIDNUMG(sptr));
    if (SDSCG(sptr))
      queue_symbol(SDSCG(sptr));
    if (PTROFFG(sptr))
      queue_symbol(PTROFFG(sptr));
    if (DESCRG(sptr))
      queue_symbol(DESCRG(sptr));
    if (PARAMVALG(sptr))
      queue_ast(PARAMVALG(sptr));
    if (CVLENG(sptr))
      queue_symbol(CVLENG(sptr));
    if (ADJARRG(sptr) && SYMLKG(sptr) != NOSYM)
      queue_symbol(SYMLKG(sptr));
    if (ADJLENG(sptr) && ADJSTRLKG(sptr) && ADJSTRLKG(sptr) != NOSYM)
      queue_symbol(ADJSTRLKG(sptr));
    if (STYPEG(sptr) == ST_DESCRIPTOR && PARENTG(sptr) && CLASSG(sptr)) {
      queue_dtype(PARENTG(sptr));
    }
#ifdef DEVCOPYG
    if (DEVCOPYG(sptr))
      queue_symbol(DEVCOPYG(sptr));
#endif
#ifdef DSCASTG
    if (STYPEG(sptr) != ST_DESCRIPTOR && DSCASTG(sptr))
      queue_ast(DSCASTG(sptr));
#endif
    break;

  case ST_CMBLK:
    /*  process all elements of the common block:  */
    recur_flag = TRUE;
    for (member = CMEMFG(sptr); member > NOSYM; member = SYMLKG(member)) {
      queue_symbol(member);
    }
    recur_flag = FALSE;
    if (ALTNAMEG(sptr)) {
      queue_symbol(ALTNAMEG(sptr));
    }
    break;

  case ST_NML:
    Trace(("exporting namelist %d/%s", sptr, SYMNAME(sptr)));
    queue_symbol(ADDRESSG(sptr));
    /*  process all elements of the namelist  */
    recur_flag = TRUE;
    for (member = CMEMFG(sptr); member; member = NML_NEXT(member)) {
      queue_symbol(NML_SPTR(member));
    }
    recur_flag = FALSE;
    break;
  case ST_PLIST:
    Trace(("exporting Plist %d/%s", sptr, SYMNAME(sptr)));
    break;

  case ST_ALIAS:
    queue_symbol((int)SYMLKG(sptr));
    if (GSAMEG(sptr))
      queue_symbol((int)GSAMEG(sptr));
    break;

  case ST_USERGENERIC:
    if (GTYPEG(sptr)) {
      /* FS#17726 - export overloaded type */
      queue_symbol((int)GTYPEG(sptr));
    }
    FLANG_FALLTHROUGH;
  case ST_OPERATOR:
    if (GSAMEG(sptr))
      queue_symbol((int)GSAMEG(sptr));
    for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr)) {
      queue_symbol(SYMI_SPTR(dscptr));
    }
    if (CLASSG(sptr) && TBPLNKG(sptr)) {
      queue_dtype(TBPLNKG(sptr));
    }
    break;

  case ST_MODPROC:
    /*
     * Need to queue the module procedure's ST_ENTRY or ST_ALIAS if
     * a module is appending to generic defined in another module.
     */
    if (SYMLKG(sptr)) {
      queue_symbol(SYMLKG(sptr));
    }
    if (GSAMEG(sptr))
      queue_symbol((int)GSAMEG(sptr));
    /* module procedure descriptor */
    for (dscptr = SYMIG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr))
      queue_symbol(SYMI_SPTR(dscptr));
    break;

  case ST_BLOCK:
    if (STARTLABG(sptr))
      queue_symbol(STARTLABG(sptr));
    if (ENDLABG(sptr))
      queue_symbol(ENDLABG(sptr));
    break;

  default:
    Trace(("Illegal symbol %d/%s in queue_symbol, type=%d", sptr, SYMNAME(sptr),
           STYPEG(sptr)));
    interr("queue_symbol: unexpected symbol type", sptr, 3);
  }
  if (ENCLFUNCG(sptr)) {
    queue_symbol(ENCLFUNCG(sptr));
  }
  if ((int)(SCOPEG(sptr)) >= stb.firstosym) {
    queue_symbol(SCOPEG(sptr));
  }

  /* queue up variables in the storage overlap list, if necessary */
  switch (STYPEG(sptr)) {
  case ST_IDENT:
  case ST_VAR:
  case ST_ARRAY:
  case ST_STRUCT:
  case ST_UNION:
    if (SOCPTRG(sptr)) {
      int p;
      for (p = SOCPTRG(sptr); p; p = SOC_NEXT(p)) {
        queue_symbol(SOC_SPTR(p));
      }
    }
    break;
  default:
    break;
  }
} /* queue_symbol */

/* ----------------------------------------------------------- */

static int
dtype_skip(int dtype)
{
  return dlen(DTY(dtype));
} /* dtype_skip */

/*
 * write out necessary info for this data type:
 */
static void
export_dt(int dtype)
{
  int paramct;

  lzprintf(outlz, "D %d %d", dtype, (int)DTY(dtype));

  switch (DTY(dtype)) {
  case TY_PTR:
    lzprintf(outlz, " %d", (int)DTY(dtype + 1));
    break;

  case TY_ARRAY:
    /*  print dtype and array descriptor entry */
    lzprintf(outlz, " %d", (int)DTY(dtype + 1));
    if (DTY(dtype + 2)) {
      ADSC *ad;
      int i, ndims;

      if (DTY(dtype + 2) <= 0) {
        lzprintf(outlz, " 0");
      } else {
        ad = AD_DPTR(dtype);
        ndims = AD_NUMDIM(ad);
        lzprintf(outlz, " %d", ndims);
        lzprintf(outlz, " %d", AD_ZBASE(ad));
        lzprintf(outlz, " %d", AD_NUMELM(ad));
        lzprintf(outlz, " %d", AD_ASSUMSHP(ad));
        lzprintf(outlz, " %d", AD_DEFER(ad));
        lzprintf(outlz, " %d", AD_ADJARR(ad));
        lzprintf(outlz, " %d", AD_ASSUMSZ(ad));
        lzprintf(outlz, " %d", AD_NOBOUNDS(ad));

        /* separate line per dimension */
        for (i = 0; i < ndims; i++) {
          lzprintf(outlz, "\n %d", AD_LWBD(ad, i));
          lzprintf(outlz, " %d", AD_UPBD(ad, i));
          lzprintf(outlz, " %d", AD_MLPYR(ad, i));
          lzprintf(outlz, " %d", AD_LWAST(ad, i));
          lzprintf(outlz, " %d", AD_UPAST(ad, i));
          lzprintf(outlz, " %d", AD_EXTNTAST(ad, i));
        }
      }
    } else /* 'null' descriptor */
      lzprintf(outlz, " %d", 0);
    break;
  case TY_UNION:
  case TY_STRUCT:
  case TY_DERIVED:
    /*  print dtype and  descriptor entry */
    lzprintf(outlz, " %d %d %d %d", (int)DTY(dtype + 1), (int)DTY(dtype + 2),
             (int)DTY(dtype + 3), (int)DTY(dtype + 4));
    break;

  case TY_CHAR:
  case TY_NCHAR:
    lzprintf(outlz, " %d", (int)DTY(dtype + 1));
    break;

  case TY_PROC:
    lzprintf(outlz, " %d", DTY(dtype + 1));
    lzprintf(outlz, " %d", DTY(dtype + 2)); /* interface */
    paramct = DTY(dtype + 3);               /* PARAMCT */
    lzprintf(outlz, " %d", paramct);
    if (paramct) {
      int *dscptr;
      for (dscptr = aux.dpdsc_base + DTY(dtype + 4); paramct > 0; paramct--) {
        lzprintf(outlz, " %d", *dscptr);
        dscptr++;
      }
    }
    lzprintf(outlz, " %d", DTY(dtype + 5)); /* FVAL */
    break;

  default:
    interr("export_dt: illegal dtype", dtype, 3);
  }

  lzprintf(outlz, "\n");
}

/*  write out necessary info for all data types created in the module
 *  specification
 */
static void
export_dtypes(int start, int ignore)
{
  int dtype;
  if (start < DT_MAX + 1)
    start = DT_MAX + 1;

  for (dtype = DT_MAX + 1; dtype < stb.dt.stg_avail;) {
    if ((dtype >= dtype_flag_size || dtype_flag[dtype]) &&
        (dtype >= start || DTY(dtype) == TY_CHAR)) {
      if (ignore) {
        int mem;
        switch (DTY(dtype)) {
        case TY_DERIVED:
        case TY_UNION:
        case TY_STRUCT:
          if (DTY(dtype + 3) && !CFUNCG(DTY(dtype + 3))) {
            IGNOREP(DTY(dtype + 3), 1);
            HIDDENP(DTY(dtype + 3), 1);
          }
          for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
            if (!CFUNCG(mem)) {
              IGNOREP(mem, 1);
              HIDDENP(mem, 1);
            }
          }
          break;
        }
      }
      export_dt(dtype);
    }
    dtype += dtype_skip(dtype);
  }
}

/*  write out necessary info for this data type:  */
static void
export_derived_dt(int dtype)
{
  int sptr, scope;
  switch (DTY(dtype)) {
  case TY_UNION:
  case TY_STRUCT:
  case TY_DERIVED:
    sptr = DTY(dtype + 3);
    if (sptr == 0)
      return;
    scope = SCOPEG(sptr);
    if (scope == 0)
      return;
    if (STYPEG(scope) == ST_MODULE) {
      /*  print dtype and  descriptor entry */
      int base = CMEMFG(scope);
      lzprintf(outlz, "d %d %d %d %s %s\n", dtype, STYPEG(sptr), sptr - base,
               SYMNAME(scope), SYMNAME(sptr));
    } else {
      lzprintf(outlz, "e %d %d %d %s %s\n", dtype, STYPEG(sptr), STYPEG(scope),
               SYMNAME(scope), SYMNAME(sptr));
    }
    break;
  }
}

static void
export_outer_derived_dtypes(int limit)
{
  int dtype;

  for (dtype = 0; dtype < limit;) {
    if (dtype >= dtype_flag_size || dtype_flag[dtype]) {
      export_derived_dt(dtype);
    }
    dtype += dtype_skip(dtype);
  }
} /* export_outer_derived_dtypes */

/* ----------------------------------------------------------- */

/*
 * write out necessary info for this symbol:
 */
static void
export_symbol(int sptr)
{
  int i;
  int dtype;
  char *strptr;
  int stringlen;
  SYM *wp;
  int dscptr;
  int nml, scope, stype, flags, bit;

  scope = SCOPEG(sptr);
  stype = STYPEG(sptr);
  if (!exportmode && stype == ST_UNKNOWN && sptr == gbl.sym_nproc) {
    return;
  }
  if (for_module || for_inliner || for_contained ||
      (exportmode && XBIT(66, 0x20000000))) {
    int scope2, cs;
    for (scope2 = scope; scope2; scope2 = SCOPEG(scope2)) {
      if (STYPEG(scope2) == ST_MODULE) {
        scope = scope2;
      }
      if ((STYPEG(scope2) == ST_ENTRY && scope2 != sptr) ||
          (STYPEG(scope2) == ST_ALIAS && STYPEG(SYMLKG(scope2)) == ST_ENTRY &&
           SYMLKG(scope2) != sptr)) {
        scope = scope2;
        break;
      }
      if ((STYPEG(scope2) == ST_PROC && scope2 != sptr) ||
          (STYPEG(scope2) == ST_ALIAS && STYPEG(SYMLKG(scope2)) == ST_PROC &&
           SYMLKG(scope2) != sptr)) {
        scope = scope2;
        break;
      }
      if (SCOPEG(scope2) == scope2)
        break;
    }
    cs = SCOPEG(gbl.currsub);
    if (for_inliner &&
        (sptr == gbl.currsub || SCOPEG(sptr) == cs || SCG(sptr) == SC_DUMMY)) {
      /* export symbols from this subprogram as normal */
    } else if (sptr == gbl.currsub) {
    } else if ((scope >= stb.firstosym && scope != sym_module &&
                STYPEG(scope) == ST_MODULE && !ISSUBMODULEG(sptr))) {
      /* this symbol is from a USEd module */
      if (stype != ST_MODULE && stype != ST_UNKNOWN) {
        int dscptr, dsccnt;
        int base = CMEMFG(scope);
        int offset = sptr - base + 1;
        if (base == 0) {
          offset = 0;
        }
        lzprintf(outlz, "R %d %d %d %s %s", sptr, stype, offset, SYMNAME(scope),
                 SYMNAME(sptr));
        /* may have additional overloaded names */
        switch (stype) {
        case ST_MEMBER:
#ifdef ENCLDTYPEG
          dtype = ENCLDTYPEG(sptr);
          if (DTY(dtype + 3)) {
            lzprintf(outlz, " %s", SYMNAME(DTY(dtype + 3)));
          } else {
            lzprintf(outlz, " .");
          }
#endif
          lzprintf(outlz, "\n");
          break;
        case ST_USERGENERIC:
        case ST_OPERATOR:
          lzprintf(outlz, "\n");
          dsccnt = 0;
          for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr)) {
            int ds = SYMI_SPTR(dscptr);
            if (SCOPEG(ds) == stb.curr_scope) {
              ++dsccnt;
            }
          }
          if (dsccnt) {
            lzprintf(outlz, "O %d %d", sptr, dsccnt);
            for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr)) {
              int ds = SYMI_SPTR(dscptr);
              if (SCOPEG(ds) == stb.curr_scope) {
                lzprintf(outlz, " %d", ds);
              }
            }
            lzprintf(outlz, "\n");
          }
          break;
        default:
          lzprintf(outlz, "\n");
          break;
        }
      }
      return;
    }
    if (for_inliner && sptr < stb.firstusym && sptr >= stb.firstosym) {
      lzprintf(outlz, "C %d %d %s\n", sptr, STYPEG(sptr), SYMNAME(sptr));
      return;
    }
    if (stype == ST_MODULE && sptr != sym_module && !for_inliner &&
        /* No return when this module has a separate module procedure that
         * implements a type bound procedure. We need to export modules
         * sptr next.
         */
        !HAS_TBP_BOUND_TO_SMPG(sptr) && ANCESTORG(sym_module) != sptr) {
      return;
    }
  }

  if ((STYPEG(sptr) == ST_ALIAS || STYPEG(sptr) == ST_PROC ||
       STYPEG(sptr) == ST_ENTRY) &&
      ISSUBMODULEG(sptr))
    INMODULEP(sptr, TRUE);

  /* BYTE-ORDER INDEPENDENT */
  wp = stb.stg_base + sptr;
  lzprintf(outlz, "S %d", sptr);
  if (exportmode)
    lzprintf(outlz, " %d", HASHLKG(sptr));
  lzprintf(outlz, " %d %d %d %d %d %d %d %d %d", stb.stg_base[sptr].stype,
           stb.stg_base[sptr].sc, stb.stg_base[sptr].b3, stb.stg_base[sptr].b4,
           stb.stg_base[sptr].dtype, stb.stg_base[sptr].symlk,
           stb.stg_base[sptr].scope, stb.stg_base[sptr].nmptr,
           stb.stg_base[sptr].palign);

#undef PUTFIELD
#undef PUTISZ_FIELD
#define PUTFIELD(f) lzprintf(outlz, " %d", stb.stg_base[sptr].f)
#define PUTISZ_FIELD(f) lzprintf(outlz, " %" ISZ_PF "d", stb.stg_base[sptr].f)
#define ADDBIT(f)                                                              \
  if (stb.stg_base[sptr].f)                                                    \
    flags |= bit;                                                              \
  bit <<= 1;

  flags = 0;
  bit = 1;
  ADDBIT(f1);
  ADDBIT(f2);
  ADDBIT(f3);
  ADDBIT(f4);
  ADDBIT(f5);
  ADDBIT(f6);
  ADDBIT(f7);
  ADDBIT(f8);
  ADDBIT(f9);
  ADDBIT(f10);
  ADDBIT(f11);
  ADDBIT(f12);
  ADDBIT(f13);
  ADDBIT(f14);
  ADDBIT(f15);
  ADDBIT(f16);
  ADDBIT(f17);
  ADDBIT(f18);
  ADDBIT(f19);
  ADDBIT(f20);
  ADDBIT(f21);
  ADDBIT(f22);
  ADDBIT(f23);
  ADDBIT(f24);
  ADDBIT(f25);
  ADDBIT(f26);
  ADDBIT(f27);
  ADDBIT(f28);
  ADDBIT(f29);
  ADDBIT(f30);
  ADDBIT(f31);
  ADDBIT(f32);
  lzprintf(outlz, " %x", flags);
  flags = 0;
  bit = 1;
  ADDBIT(f33);
  ADDBIT(f34);
  ADDBIT(f35);
  ADDBIT(f36);
  ADDBIT(f37);
  ADDBIT(f38);
  ADDBIT(f39);
  ADDBIT(f40);
  ADDBIT(f41);
  ADDBIT(f42);
  ADDBIT(f43);
  ADDBIT(f44);
  ADDBIT(f45);
  ADDBIT(f46);
  ADDBIT(f47);
  ADDBIT(f48);
  ADDBIT(f49);
  ADDBIT(f50);
  ADDBIT(f51);
  ADDBIT(f52);
  ADDBIT(f53);
  ADDBIT(f54);
  ADDBIT(f55);
  ADDBIT(f56);
  ADDBIT(f57);
  ADDBIT(f58);
  ADDBIT(f59);
  ADDBIT(f60);
  ADDBIT(f61);
  ADDBIT(f62);
  ADDBIT(f63);
  ADDBIT(f64);
  lzprintf(outlz, " %x", flags);

  /*
   * New flags & fields were added for IVSN 26.  Prefix the new set of
   * flags & fields with ' A'. interf will check for this prefix, and if
   * not present, the .mod file must be the previous version and interf
   * will not attempt to read these fields.
   *
   * START ---------- IVSN 26 flags & fields
   */
  lzprintf(outlz, " A");
  flags = 0;
  bit = 1;
  ADDBIT(f65);
  ADDBIT(f66);
  ADDBIT(f67);
  ADDBIT(f68);
  ADDBIT(f69);
  ADDBIT(f70);
  ADDBIT(f71);
  ADDBIT(f72);
  ADDBIT(f73);
  ADDBIT(f74);
  ADDBIT(f75);
  ADDBIT(f76);
  ADDBIT(f77);
  ADDBIT(f78);
  ADDBIT(f79);
  ADDBIT(f80);
  ADDBIT(f81);
  ADDBIT(f82);
  ADDBIT(f83);
  ADDBIT(f84);
  ADDBIT(f85);
  ADDBIT(f86);
  ADDBIT(f87);
  ADDBIT(f88);
  ADDBIT(f89);
  ADDBIT(f90);
  ADDBIT(f91);
  ADDBIT(f92);
  ADDBIT(f93);
  ADDBIT(f94);
  ADDBIT(f95);
  ADDBIT(f96);
  lzprintf(outlz, " %x", flags);
  PUTFIELD(w34);
  PUTFIELD(w35);
  PUTFIELD(w36);
  /*
   * END   ---------- IVSN 26 flags & fields
   */

  /*
   * New flags & fields were added for IVSN 28.  Prefix the new set of
   * flags & fields with ' B'. interf will check for this prefix, and if
   * not present, the .mod file must be the previous version and interf
   * will not attempt to read these fields.
   *
   * START ---------- IVSN 28 flags & fields
   */
  lzprintf(outlz, " B");
  flags = 0;
  bit = 1;
  ADDBIT(f97);
  ADDBIT(f98);
  ADDBIT(f99);
  ADDBIT(f100);
  ADDBIT(f101);
  ADDBIT(f102);
  ADDBIT(f103);
  ADDBIT(f104);
  ADDBIT(f105);
  ADDBIT(f106);
  ADDBIT(f107);
  ADDBIT(f108);
  ADDBIT(f109);
  ADDBIT(f110);
  ADDBIT(f111);
  ADDBIT(f112);
  ADDBIT(f113);
  ADDBIT(f114);
  ADDBIT(f115);
  ADDBIT(f116);
  ADDBIT(f117);
  ADDBIT(f118);
  ADDBIT(f119);
  ADDBIT(f120);
  ADDBIT(f121);
  ADDBIT(f122);
  ADDBIT(f123);
  ADDBIT(f124);
  ADDBIT(f125);
  ADDBIT(f126);
  ADDBIT(f127);
  ADDBIT(f128);
  lzprintf(outlz, " %x", flags);
  PUTFIELD(lineno);
  PUTFIELD(w39);
  PUTFIELD(w40);
  /*
   * END   ---------- IVSN 28 flags & fields
   */

  PUTFIELD(w9);
  PUTISZ_FIELD(w10);
  PUTFIELD(w11);
  PUTFIELD(w12);
  PUTFIELD(w13);
  PUTISZ_FIELD(w14);
  PUTFIELD(w15);
  PUTFIELD(w16);
  PUTFIELD(w17);
  PUTFIELD(w18);
  PUTFIELD(w19);
  PUTFIELD(w20);
  PUTFIELD(w21);
  PUTFIELD(w22);
  PUTFIELD(w23);
  PUTFIELD(w24);
  PUTFIELD(w25);
  PUTFIELD(w26);
  PUTFIELD(w27);
  PUTFIELD(w28);
  PUTFIELD(uname);
  PUTFIELD(w30);
  PUTFIELD(w31);
  PUTFIELD(w32);
#undef ADDBIT
#undef PUTFIELD
#undef PUTISZ_FIELD

  switch (stype) {
  case ST_CONST:
    dtype = DTYPEG(sptr);
    lzprintf(outlz, " %d", (int)DTY(dtype)); /* contant's TY_ value */
    switch (DTY(dtype)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_INT8:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
    case TY_LOG8:
    case TY_REAL:
    case TY_DBLE:
    case TY_QUAD:
    case TY_CMPLX:
    case TY_NCHAR:
    case TY_DCMPLX:
    case TY_QCMPLX:
      if (NMPTRG(sptr)) {
        lzprintf(outlz, " %s", SYMNAME(sptr));
      }
      break;

    case TY_CHAR:
      strptr = stb.n_base + CONVAL1G(sptr);
      stringlen = string_length(DTYPEG(sptr));
      lzprintf(outlz, " %d", stringlen);
      for (i = 0; i < stringlen; i++)
        lzprintf(outlz, " %x", ((int)*strptr++));
      break;
    }
    break;

  case ST_UNKNOWN:
  case ST_IDENT:
  case ST_PARAM:
  case ST_MEMBER:
  case ST_UNION:
  case ST_STRUCT:
  case ST_VAR:
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_CMBLK:
  case ST_ALIAS:
  case ST_ARRDSC:
  case ST_TYPEDEF:
  case ST_STAG:
  case ST_LABEL:
  case ST_MODULE:
  case ST_STFUNC:
  case ST_INTRIN: /* for new intrinsics with OPTYPE NEW_INTRIN */
    lzprintf(outlz, " %s", SYMNAME(sptr));
    break;

  case ST_ENTRY:
  case ST_PROC:
    lzprintf(outlz, " %s", SYMNAME(sptr));
    {
      if ((i = PARAMCTG(sptr))) {
        /* output parameter descriptor */
        lzprintf(outlz, "\n");
        lzprintf(outlz, "F %d %d", sptr, i);
        dscptr = DPDSCG(sptr);
        while (TRUE) {
          lzprintf(outlz, " %d", aux.dpdsc_base[dscptr]);
          if (--i == 0)
            break;
          dscptr++;
        }
      } else {
        /* No args, but possibly an array or pointer return
           val. Create an 'F sptr 0' record. So Fix up will
           occur and DPDSC field gets filled in. */
        if (DPDSCG(sptr)) {
          lzprintf(outlz, "\n");
          lzprintf(outlz, "F %d %d", sptr, i);
        }
      }
    }
    break;

  case ST_USERGENERIC:
  case ST_OPERATOR:
    lzprintf(outlz, " %s", SYMNAME(sptr));
    if ((i = GNCNTG(sptr))) {
      /* output generic descriptor */
      lzprintf(outlz, "\n");
      lzprintf(outlz, "O %d %d", sptr, i);
      for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr))
        lzprintf(outlz, " %d", SYMI_SPTR(dscptr));
    }
    break;

  case ST_MODPROC:
    lzprintf(outlz, " %s", SYMNAME(sptr));
    if ((dscptr = SYMIG(sptr))) {
      /* module procedure descriptor */
      lzprintf(outlz, "\n");
      lzprintf(outlz, "Q %d", sptr);
      for (; dscptr; dscptr = SYMI_NEXT(dscptr))
        lzprintf(outlz, " %d", SYMI_SPTR(dscptr));
      lzprintf(outlz, " 0");
    }
    break;

  case ST_NML:
    lzprintf(outlz, " %s", SYMNAME(sptr));
    for (nml = CMEMFG(sptr); nml; nml = NML_NEXT(nml)) {
      lzprintf(outlz, "\nN %d %d", NML_SPTR(nml), NML_LINENO(nml));
    }
    lzprintf(outlz, "\nN -1 -1");
    break;

  case ST_PLIST:
  case ST_CONSTRUCT:
  case ST_BLOCK:
    lzprintf(outlz, " %s", SYMNAME(sptr));
    break;

  default:
    interr("export_symbol: illegal symbol type", sptr, 3);
  }

  lzprintf(outlz, "\n");

  /* output the storage overlap list, if necessary */
  switch (stype) {
  case ST_IDENT:
  case ST_VAR:
  case ST_ARRAY:
  case ST_STRUCT:
  case ST_UNION:
    if (SOCPTRG(sptr)) {
      int p;
      lzprintf(outlz, "L %d", sptr);
      for (p = SOCPTRG(sptr); p; p = SOC_NEXT(p)) {
        lzprintf(outlz, " %d", SOC_SPTR(p));
      }
      lzprintf(outlz, " -1\n");
    }
    break;
  default:
    break;
  }

  switch (stype) {
  case ST_IDENT:
  case ST_VAR:
    /* If the string dtype information was stashed in the
     * this symbol table entry (see fixup_host_symbol_dtype),
     * the information is no longer needed so clear it (shouldn't
     * be necessary but just to be safe). */
    dtype = DTYPEG(sptr);
    if ((DTY(dtype) == TY_CHAR &&
         (dtype != DT_ASSCHAR || dtype != DT_DEFERCHAR)) ||
        (DTY(dtype) == TY_NCHAR &&
         (dtype != DT_ASSNCHAR || dtype != DT_DEFERNCHAR))) {
      int clen = DTY(dtype + 1);
      if (A_ALIASG(clen)
          /* If CLASS is set, then do not clear CVLEN since it's overloaded by
           * VTOFF and VTABLE which are used with type bound procedures. We
           * may need to revisit this when we implement unlimited polymorphic
           * types.
           */
          &&
          (!CLASSG(sptr) ||
           (STYPEG(sptr) != ST_MEMBER && STYPEG(sptr) != ST_PROC &&
            STYPEG(sptr) != ST_USERGENERIC && STYPEG(sptr) != ST_OPERATOR))) {
        CVLENP(sptr, 0);
      }
    }
  }
}

/* ----------------------------------------------------------- */

/*  write out necessary info for a single ast  */
static void
export_one_ast(int ast)
{
  int bit, flags;
  int a;
  int i, s, n;
  int cnt;
  lzprintf(outlz, "A %d %d", ast, A_TYPEG(ast));
  flags = 0;
  bit = 1;
#define ADDBIT(fl)                                                             \
  if (astb.stg_base[ast].fl)                                                   \
    flags |= bit;                                                              \
  bit <<= 1;
  ADDBIT(f1);
  ADDBIT(f2);
  ADDBIT(f3);
  ADDBIT(f4);
  ADDBIT(f5);
  ADDBIT(f6);
  ADDBIT(f7);
  ADDBIT(f8);
#undef ADDBIT
  lzprintf(outlz, " %x", flags);
  lzprintf(outlz, " %d", astb.stg_base[ast].shape);
  lzprintf(outlz, " %d %d %d %d", astb.stg_base[ast].hshlk,
           astb.stg_base[ast].w3, astb.stg_base[ast].w4, astb.stg_base[ast].w5);
  lzprintf(outlz, " %d %d %d %d", astb.stg_base[ast].w6, astb.stg_base[ast].w7,
           astb.stg_base[ast].w8, astb.stg_base[ast].w9);
  lzprintf(outlz, " %d %d %d %d", astb.stg_base[ast].w10,
           astb.stg_base[ast].hw21, astb.stg_base[ast].hw22,
           astb.stg_base[ast].w12);
  lzprintf(outlz, " %d %d %d %d", astb.stg_base[ast].opt1,
           astb.stg_base[ast].opt2, astb.stg_base[ast].repl,
           astb.stg_base[ast].visit);
  /* IVSN 30 */
  lzprintf(outlz, " %d", astb.stg_base[ast].w18);
  lzprintf(outlz, " %d", astb.stg_base[ast].w19);

  if (A_TYPEG(ast) == A_ID && A_IDSTRG(ast)) {
    lzprintf(outlz, " %s", SYMNAME(A_SPTRG(ast)));
  }
  lzprintf(outlz, "\n");

  switch (A_TYPEG(ast)) {
  case A_FUNC:
  case A_INTR:
    if (!exportmode || gbl.internal > 1 || XBIT(66, 0x20000000)) {
      s = A_SHAPEG(ast);
      if (s) {
        n = SHD_NDIM(s);
        lzprintf(outlz, "T %d", n);
        for (i = 0; i < n; i++)
          lzprintf(outlz, " %d %d %d", SHD_LWB(s, i), SHD_UPB(s, i),
                   SHD_STRIDE(s, i));
        lzprintf(outlz, "\n");
      }
    }
    FLANG_FALLTHROUGH;
  case A_CALL:
  case A_ICALL:
  case A_ENDMASTER:
    a = A_ARGSG(ast);
    if (a) {
      cnt = A_ARGCNTG(ast);
      lzprintf(outlz, "W %d", cnt);
      for (i = 0; i < cnt; i++)
        lzprintf(outlz, " %d", ARGT_ARG(a, i));
      lzprintf(outlz, "\n");
    }
    break;
  case A_SUBSCR:
    a = A_ASDG(ast);
    cnt = ASD_NDIM(a);
    lzprintf(outlz, "X %d", cnt);
    for (i = 0; i < cnt; i++)
      lzprintf(outlz, " %d", ASD_SUBS(a, i));
    lzprintf(outlz, "\n");
    break;
  case A_CGOTO:
  case A_AGOTO:
  case A_FORALL:
    a = A_LISTG(ast);
    lzprintf(outlz, "Y");
    while (a) {
      lzprintf(outlz, " %d %d", ASTLI_SPTR(a), ASTLI_TRIPLE(a));
      a = ASTLI_NEXT(a);
    }
    lzprintf(outlz, " -1\n");
    break;
  }
} /* export_one_ast */

/* ----------------------------------------------------------- */

static void
queue_one_std(int std)
{
  if (STD_AST(std))
    queue_ast(STD_AST(std));
  if (STD_LABEL(std))
    queue_symbol(STD_LABEL(std));
} /* queue_one_std */

static void
export_one_std(int std)
{
  int bit, flags;
  flags = 0;
  bit = 1;
#define ADDBIT(f)                                                              \
  if (astb.std.stg_base[std].flags.bits.f)                                     \
    flags |= bit;                                                              \
  bit <<= 1;
  ADDBIT(ex);
  ADDBIT(st);
  ADDBIT(br);
  ADDBIT(delete);
  ADDBIT(ignore);
  ADDBIT(split);
  ADDBIT(minfo);
  ADDBIT(local);
  ADDBIT(pure);
  ADDBIT(par);
  ADDBIT(cs);
  ADDBIT(parsect);
  ADDBIT(orig);
#undef ADDBIT
  lzprintf(outlz, "V %d %d %d %d %x", std, STD_AST(std), STD_LABEL(std),
           STD_LINENO(std), flags);
  if (exportmode) {
    lzprintf(outlz, " %d", STD_TAG(std));
  }
  lzprintf(outlz, "\n");
} /* export_one_std */

static void
all_stds(void (*callproc)(int))
{
  int std;
  for (std = STD_NEXT(0); std; std = STD_NEXT(std))
    (*callproc)(std);
}

#ifdef FLANG_EXTERF_UNUSED
/* export a single record to the interf file */
static void
export_dinit_record(int rectype, INT recval)
{
  lzprintf(outlz, "I %d %x\n", rectype, recval);
} /* export_dinit_record */
#endif

#ifdef FLANG_EXTERF_UNUSED
/*
 * go through data initialization file.
 * call symproc for symbols in that file that will be saved
 */
static void
export_dinit_file(void (*symproc)(int), void (*recproc)(int, INT),
                  int do_fmt_nml)
{
  DREC *p;
  dinit_fseek(0);
  while ((p = dinit_read())) {
    int ptype;
    INT pcon;
    int sptr;
    ptype = p->dtype;
    pcon = p->conval;
    switch (ptype) {
    case DINIT_FMT: /* skip the format */
      if (do_fmt_nml) {
        sptr = pcon;
        if (symproc)
          (*symproc)(sptr);
      } else {
        while ((p = dinit_read()) && p->dtype != DINIT_END)
          ;
      }
      break;
    case DINIT_NML: /* skip the namelist unless this is a module */
      if (exporting_module || do_fmt_nml) {
        if (recproc)
          (*recproc)(ptype, pcon);
        sptr = pcon;
        if (symproc)
          (*symproc)(sptr);
      } else {
        while ((p = dinit_read()) && p->dtype != DINIT_END)
          ;
      }
      break;

    case DINIT_END:
    case DINIT_ENDTYPE:  /* skip this */
    case DINIT_STARTARY: /* skip this also */
    case DINIT_ENDARY:   /* skip this also */
    case 0:              /* alignment record */
    case DINIT_ZEROES:   /* skip it */
    case DINIT_OFFSET:   /* unexpected */
    case DINIT_REPEAT:   /* repeat count */
      if (recproc)
        (*recproc)(ptype, pcon);
      break;
    case DINIT_STR:     /* string value */
    case DINIT_LABEL:   /* take address, as for namelist */
    case DINIT_TYPEDEF: /* save the typedef symbol */
    case DINIT_LOC:     /* initialize this variable */
      if (recproc)
        (*recproc)(ptype, pcon);
      sptr = pcon;
      if (symproc)
        (*symproc)(sptr);
      break;
    default:
      if (recproc)
        (*recproc)(ptype, pcon);
      if (symproc) {
        switch (DTY(ptype)) {
        case TY_DBLE:
        case TY_QUAD:
        case TY_CMPLX:
        case TY_DCMPLX:
        case TY_QCMPLX:
        case TY_INT8:
        case TY_LOG8:
        case TY_CHAR:
        case TY_NCHAR:
          /* save sptr */
          sptr = pcon;
          (*symproc)(sptr);
          break;
        case TY_INT:   /* actual constant value stays the same */
        case TY_SINT:  /* actual constant value stays the same */
        case TY_BINT:  /* actual constant value stays the same */
        case TY_LOG:   /* actual constant value stays the same */
        case TY_SLOG:  /* actual constant value stays the same */
        case TY_BLOG:  /* actual constant value stays the same */
        case TY_FLOAT: /* actual constant value stays the same */
        case TY_PTR:   /* should not happen */
        default:       /* should not happen */
          break;
        }
      }
    } /* switch */
  }
  dinit_fseek_end();
} /* export_dinit_file */
#endif

#ifdef FLANG_EXTERF_UNUSED
/* go through symbols; if we find one that is a parameter, export
 * the ASTs for its value */
static void
export_parameter_info(ast_visit_fn astproc)
{
  int sptr;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; sptr++) {
    if (STYPEG(sptr) == ST_PARAM && DTY(DTYPEG(sptr)) != TY_ARRAY) {
      int ast = CONVAL2G(sptr);
      if (ast)
        ast_traverse(ast, NULL, astproc, NULL);
    }
  }
} /* export_parameter_info */
#endif

#ifdef FLANG_EXTERF_UNUSED
static int
externalequiv(int evp)
{
  do {
    switch (SCG(EQV(evp).sptr)) {
    case SC_CMBLK:
    case SC_STATIC:
      return TRUE;
    default:;
    }
    evp = EQV(evp).next;
  } while (evp != 0 && EQV(evp).is_first == 0);
  return FALSE;
} /* externalequiv */
#endif

#ifdef FLANG_EXTERF_UNUSED
static void
export_equiv_asts(int queuesym, ast_visit_fn astproc)
{
  int evp, evnext;
  for (evp = sem.eqvlist; evp != 0; evp = evnext) {
    evnext = EQV(evp).next;
    /* beginning of an equivalence block */
    /* and some static variable in it */
    if (EQV(evp).is_first && externalequiv(evp)) {
      do {
        int ss, numss, j;
        if (queuesym)
          queue_symbol(EQV(evp).sptr);
        /* 0 or ast index for substring */
        if (EQV(evp).substring) {
          ast_traverse(EQV(evp).substring, NULL, astproc, NULL);
        }
        ss = EQV(evp).subscripts;
        numss = EQV_NUMSS(ss);
        /* depends on EQV_NUMSS(0) == 0, set in semant.c */
        for (j = 0; j < numss; ++j) {
          if (EQV_SS(ss, j))
            ast_traverse(EQV_SS(ss, j), NULL, astproc, NULL);
        }
        evp = EQV(evp).next;
      } while (evp != 0 && EQV(evp).is_first == 0);
      evnext = evp;
    }
  }
} /* export_equiv_asts */
#endif

static void
export_equiv_item(int evp)
{
  int ss, numss, j;
  lzprintf(outlz, "E %d %d %d %d", PRIVATEG(EQV(evp).sptr), EQV(evp).lineno,
           EQV(evp).sptr, (EQV(evp).is_first == 0) ? 0 : 1);
  /* 0 or ast index for substring */
  lzprintf(outlz, " %d", (int)EQV(evp).substring);
  ss = EQV(evp).subscripts;
  numss = EQV_NUMSS(ss);
  /* depends on EQV_NUMSS(0) == 0, set in semant.c */
  for (j = 0; j < numss; ++j) {
    lzprintf(outlz, " %d", EQV_SS(ss, j));
  }
  lzprintf(outlz, " -1\n"); /*  end of subscripts */
} /* export_equiv_item */

#ifdef FLANG_EXTERF_UNUSED
static void
export_external_equiv()
{
  int evp, evnext;
  for (evp = sem.eqvlist; evp != 0; evp = evnext) {
    evnext = EQV(evp).next;
    /* beginning of an equivalence block */
    /* and some static variable in it */
    if (EQV(evp).is_first && externalequiv(evp)) {
      do {
        export_equiv_item(evp);
        evp = EQV(evp).next;
      } while (evp != 0 && EQV(evp).is_first == 0);
      evnext = evp;
    }
  }
} /* export_external_equiv */
#endif

static void
export_equivs(void)
{
  int evp;
  for (evp = sem.eqvlist; evp != 0; evp = EQV(evp).next) {
    if (eqv_flag[evp])
      export_equiv_item(evp);
  }
}

/* ----------------------------------------------------------- */

/*
 * set STD_TAG field
 */
static int max_tag = 0;
void
set_tag()
{
  int std;
  for (std = STD_NEXT(0); std > 0; std = STD_NEXT(std)) {
    ++max_tag;
    STD_TAG(std) = max_tag;
  }
} /* set_tag */
