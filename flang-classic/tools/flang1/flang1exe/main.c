/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file main.c
    \brief main program and initialization routines for fortran front-end
*/
#include "gbldefs.h"
#include <stdbool.h>
#include "flang/ArgParser/arg_parser.h"
#include "error.h"
#if !defined(TARGET_WIN)
#include <unistd.h>
#endif
#include <time.h>
#include "global.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "version.h"
#include "inliner.h"
#include "interf.h"
#include "semant.h"
#include "dinit.h"

#include "ast.h"
#include "lower.h"
#include "dbg_out.h"
#include "ccffinfo.h"
#include "mach.h"
#include "fdirect.h"
#include "optimize.h"
#include "transfrm.h"
#include "extern.h"
#include "commopt.h"
#include "scan.h"
#include "hlvect.h"
#include "ilidir.h" /* for ili_lpprg_init */

#define IPA_ENABLED                  0
#define IPA_NO_ASM                   0
#define IPA_COLLECTION_ENABLED       0
#define IPA_INHERIT_ENABLED          0
#define IPA_FUTURE_INHERIT_DISABLED  0
#define IPA_REPORT_ENABLED           0

/* static prototypes */

static void reptime(void);
static void do_debug(const char *phase);
static void cleanup(void);
static void init(int argc, char *argv[]);
static void datastructure_reinit(void);
static void do_set_tp(const char *tp);
static void fini(void);
static void mkDwfInfoFilename(void);

/* ******************************************************************** */

/* Below are definitions/static variables required by main function */
static int saveoptflag;
static int savevectflag;
static int savex8flag;
static int saverecursive;
static LOGICAL has_accel_code = FALSE;
static action_map_t *phase_dump_map;
#if DEBUG
static int debugfunconly = -1;
#endif
static LOGICAL ipa_import_mode = FALSE;
static char *ipa_export_file = NULL;
static BIGUINT ipa_import_offset = 0;
static const char *who[] = {"init",      "parser",   "bblock",
                            "vectorize", "optimize", "schedule",
                            "assemble",  "xref",     "unroll"};
#define _N_WHO (sizeof(who) / sizeof(char *))
static INT xtimes[_N_WHO];
static LOGICAL postprocessing = TRUE;

/* Feature names for Fortran front-end */
#if defined(TARGET_WIN_X8664)
static const char *feature = "pgfortran";
#elif defined(TARGET_OSX_X8664)
static const char *feature = "pgfortran";
#elif defined(OSF86)
static const char *feature = "pgi-f95-osf32";
#elif defined(TARGET_LLVM_POWER)
static const char *feature = "pgfortran";
#else
static const char *feature = "flang";
#endif

/** Product name in debug output
 */
#define DNAME "F90"

#if DEBUG
static int dodebug = 0;
#define TR(str)               \
  if (dodebug) {              \
    fprintf(gbl.dbgfil, str); \
    fflush(gbl.dbgfil);       \
  }
#define TR1(str)      \
  if (DBGBIT(0, 512)) \
  dump_stg_stat(str)
#define DUMP(a) do_debug(a)
#else
#define TR(str)
#define TR1(str)
#define DUMP(a)
#endif /* DEBUG */

#define NO_FLEXLM

/** \brief Fortran front-end main entry
    \param argc number of command-line arguments
    \pram argv array of command-line argument strings
*/
int
main(int argc, char *argv[])
{
  int savescope, savecurrmod = 0;
  get_rutime();
  init(argc, argv); /* initialize */
  if (gbl.fn == NULL)
    gbl.fn = gbl.src_file;

#if DEBUG
  if (debugfunconly > 0)
    dodebug = 0;
#endif

  saveoptflag = flg.opt;
  savevectflag = flg.vect;
  savex8flag = flg.x[8];
  saverecursive = flg.recursive;

  if (IPA_INHERIT_ENABLED && (flg.opt >= 2 || IPA_COLLECTION_ENABLED)) {
    ipa_init();
  }

  gbl.findex = addfile(gbl.fn, NULL, 0, 0, 0, 1, 0);

  if (ipa_export_file && ipa_import_mode) {
    ipa_import_open(ipa_export_file, ipa_import_offset);
  }
  if (IPA_ENABLED && ipa_export_file && !ipa_import_mode) {
    /* export the program unit for IPA recompilation */
    ipa_export_open(ipa_export_file);
  }

  if (gbl.srcfil == NULL) {
    if (!ipa_import_mode) {
      finish();
    }
  }
  do { /* loop once for each user program unit */
#if DEBUG
    if (debugfunconly > 0) {
      if (debugfunconly == gbl.func_count + 1)
        dodebug = 1;
      else
        dodebug = 0;
    }
#endif
    reinit();
    errini();
    if (ipa_export_file && ipa_import_mode && gbl.func_count == 0) {
      ipa_import_highpoint();
    }
    if (IPA_ENABLED && ipa_export_file && !ipa_import_mode &&
        gbl.func_count == 0) {
      ipa_export_highpoint();
    }
    xtimes[0] += get_rutime();
    if (ipa_export_file && ipa_import_mode) {
      ipa_import();
      if (gbl.eof_flag & 2)
        break;
    } else {
      TR(DNAME " PARSER begins\n")
      parser(); /* parse and do semantic analysis */
      set_tag();
    }
    gbl.func_count++;
    ccff_open_unit_f90();
    if (gbl.internal <= 1) {
      gbl.outersub = 0;
      gbl.outerentries = 0;
    }
    savescope = stb.curr_scope;
    if (gbl.currsub) {
      if (SCOPEG(gbl.currsub)) {
        stb.curr_scope = SCOPEG(gbl.currsub);
        if (STYPEG(stb.curr_scope) != ST_ALIAS ||
            SYMLKG(stb.curr_scope) != gbl.currsub) {
          stb.curr_scope = gbl.currsub;
        }
      } else {
        stb.curr_scope = gbl.currsub;
      }
    }
    TR1("- after semant");
    xtimes[1] += get_rutime();
    DUMP("parser");
    if (gbl.rutype == RU_BDATA) {
      /* a module? */
      if (has_cuda_data())
        has_accel_code = TRUE;
    }
    if (gbl.currsub == 0) {
      if (IPA_ENABLED && ipa_export_file && !ipa_import_mode) {
        /* export the program unit for IPA recompilation */
        ipa_export_endmodule();
      }
      continue; /* end of a module */
    }
    if (CUDAG(gbl.currsub) & (CUDA_GLOBAL | CUDA_DEVICE)) {
      /* remember that this routine needs a constructor */
      has_accel_code = TRUE;
    }
    savecurrmod = gbl.currmod;
#if DEBUG
    if (DBGBIT(5, 1))
      symdmp(gbl.dbgfil, DBGBIT(5, 8));
    if (DBGBIT(5, 2))
      dump_std();
    if (DBGBIT(5, 16))
      dmp_dtype();
#endif
    if (IPA_ENABLED && ipa_export_file && !ipa_import_mode) {
      /* export the program unit for IPA recompilation */
      ipa_export();
    }

#if DEBUG
    if (DBGBIT(4, 256))
      dump_ast();
#endif
    if (IPA_INHERIT_ENABLED && gbl.rutype != RU_BDATA) {
      ipa_startfunc(gbl.currsub);
      ipa_header1(gbl.currsub);
      ipa_header2(gbl.currsub);
    }
    postprocessing = FALSE;
    if (gbl.maxsev < 3 && !DBGBIT(2, 4)) {
      postprocessing = TRUE;

      flg.ipa |= 0x20;
      if (XBIT(57, 0x2000) && !flg.inliner) {
        /* try to eliminate unused common blocks here */
        eliminate_unused_variables(1);
        DUMP("staticunused");
      }
      /* by default, generate data initialization inline. */
      if (gbl.rutype != RU_BDATA) {
        direct_rou_load(gbl.currsub);
        if (flg.opt > 1 && !XBIT(47, 0x40000000)) {
          if (sem.stats.allocs > 800 && sem.stats.nodes > 1000) {
            direct_rou_setopt(gbl.currsub, 1);
            /*
             * Also, inhibit sectfloat() which is enabled with
             * -fast or -O2 or greater.
             */
            flg.x[70] &= (~0x400);
          }
        }
        ili_lpprg_init();

        TR(DNAME " BBLOCK begins\n");
        has_accel_code |= bblock();
        TR1("- after bblock");
        DUMP("bblock");
        if (flg.inliner) {
          TR(DNAME " INLINER begins\n");
#if DEBUG
          if (flg.x[29] == 0 || flg.x[29] == gbl.func_count)
#endif
            inliner();
          DUMP("inliner");
          TR1("- after inliner");
        }

        if (flg.opt >= 2 && XBIT(50, 0x40)) {
          unconditional_branches();
          DUMP("unconditional");
        }
        if (flg.opt >= 2 && !XBIT(47, 0x20)) {
          TR(DNAME " OPTIMIZE_ALLOC begins\n");
          optimize_alloc();
          DUMP("optalloc");
          TR1("- after optimize_alloc");
        }

        if (IPA_ENABLED) {
          ipasave();
          if (IPA_NO_ASM) {
            ipasave_endfunc();
            direct_rou_end();
            continue;
          }
        }
        if (IPA_INHERIT_ENABLED && !IPA_FUTURE_INHERIT_DISABLED) {
          if (!IPA_ENABLED) {
            fill_ipasym();
          }
          ipa();
          DUMP("ipa");
          if (IPA_Vestigial) {
            ipasave_endfunc();
            if (gbl.internal == 1) {
              ipa_set_vestigial_host(); /* interf.c */
              save_host_state(0x2 + (ipa_import_mode ? 0x20 : 0));
              gbl.outersub = gbl.currsub;
              gbl.outerentries = gbl.entries;
            }
            (void)summary(FALSE, FALSE);
            continue;
          }
        }

        /* infer array alignments */
        TR(DNAME " PROCESS_ALIGN begins\n");
        trans_process_align();
        TR1("- after process_align");
        DUMP("process-align");

        if (flg.opt >= 2) {
          if (XBIT(53, 2)) {
            points_to_anal();
            DUMP("pointsto");
          }
          pstride_analysis();
          DUMP("pstride");
        }

        if (!XBIT(49, 1)) {
          TR(DNAME " TRANSFORMER begins\n");
          transform();
          DUMP("transform");
          TR1("- after transform");

          forall_init();

          if (!XBIT(49, 0x20)) {
            if (flg.opt >= 2 && !XBIT(47, 0x02)) {
              TR(DNAME " COMMUNICATIONS pre-OPTIMIZER begins\n");
              comm_optimize_pre();
              DUMP("comm-analyze-pre");
              TR1("- after comm pre-optimizer");
            }
            TR(DNAME " COMMUNICATIONS ANALYZER begins\n");
            comm_analyze();
            DUMP("comm-analyze");
            TR1("- after comm_analyze");

            TR(DNAME " CALL ANALYZER begins\n");
            call_analyze();
            DUMP("call-analyze");
            TR1("- after call_analyze");
            if (flg.opt >= 2 && !XBIT(47, 0x01)) {
              TR(DNAME " COMMUNICATIONS post-OPTIMIZER begins\n");
              comm_optimize_post();
              DUMP("comm-optimize-post");
              TR1("- after comm post-optimizer");
            }
            if (flg.opt >= 2 && !XBIT(47, 0x08)) {
              TR(DNAME " COMMUNICATIONS hoisting begins\n");
              comm_invar();
              DUMP("comm-invar");
              TR1("- after comm_invar");
            }
            TR(DNAME " COMMUNICATIONS GENERATOR begins\n");
            comm_generator();
            DUMP("comm-generator");
            TR1("- after comm_generator");
          }
          TR(DNAME " CONVERT_FORALL begins\n");
          convert_forall();
          DUMP("convert-forall");
          TR1("- after convert_forall");

          TR(DNAME " CONVERT_OUTPUT begins\n");
          convert_output();
          TR1("- after convert_output");
          DUMP("convert-output");
        }
        if (XBIT(70, 0x400) || XBIT(47, 0x400000)
                ) {
          optimize(1);
          DUMP("optimize0");
          TR1("- after optimize0");
        }
        if (XBIT(70, 0x400)) {
          sectfloat();
          DUMP("sectfloat");
        }
        if (XBIT(47, 0x400000) || flg.opt >= 2 || XBIT(163, 1)
                ) {
          sectinline();
          DUMP("sectinline");
        }
        if (XBIT(70, 0x18)) {
          linearize_arrays();
          DUMP("linearize");
        }
        if (!XBIT(70, 0x40)) {
          DUMP("bredundss");
          redundss();
          DUMP("redundss");
        }
        if (flg.opt >= 2 && !XBIT(47, 0x1000)) {
          TR(DNAME " OPTIMIZER begins\n");
          optimize(0);
          DUMP("optimize");
          TR1("- after optimize");
        }
        if (IPA_ENABLED) {
          ipasave_endfunc();
        }
        if (IPA_REPORT_ENABLED) {
          ipa_report();
        }

        direct_rou_end();
        if (flg.opt >= 2 && XBIT(53, 2)) {
          fini_points_to_all();
        }
      } else { /* gbl.rutype == RU_BDATA */
        direct_rou_load(gbl.currsub);
        if (IPA_ENABLED) {
          ipasave();
        }
        merge_commons();
        if (XBIT(55, 2)) {
          cleanup();
          goto skip_compile;
        }
        /* block data must be transformed so that common blocks
         * get handled -- lfm
         */
        /* infer array alignments */
        TR("Blkdata -- " DNAME " PROCESS_ALIGN begins\n");
        trans_process_align();
        DUMP("process-align");
        TR1("- after process_align");
        if (!XBIT(49, 1)) {
          TR("Blkdata -- " DNAME " TRANSFORMER begins\n");
          transform();
          DUMP("transform");
          TR1("- after transform");
        }
      }
#if DEBUG
      if (XBIT(57, 0x100)) {
        renumber_lines();
      }
#endif
      if (XBIT(57, 0x2000)) {
        DUMP("bunused");
        eliminate_unused_variables(2);
        DUMP("unused");
      }
      DUMP("before-output");
      lower(0);
      if (gbl.internal == 1) {
        save_host_state(0x2 + (ipa_import_mode ? 0x20 : 0));
      }
      DUMP("output");
      if (gbl.rutype != RU_BDATA && flg.opt >= 2 && XBIT(53, 2)) {
        fini_pstride_analysis();
      }
#if DEBUG
      if (DBGBIT(5, 4))
        symdmp(gbl.dbgfil, DBGBIT(5, 8));
      if (DBGBIT(5, 16))
        dmp_dtype();
#endif
    } else { /* if( gbl.maxsev < 3 && !DBGBIT(2, 4) ) */
      if (gbl.internal == 1) {
        save_host_state(0x2);
      }
    } /* if( gbl.maxsev < 3 && !DBGBIT(2, 4) ) */

    if (flg.xref) {
      xref(); /* write cross reference map */
      xtimes[7] += get_rutime();
    }
    skip_compile:
    (void)summary(FALSE, FALSE);
    errini();

    if (gbl.internal == 1) {
      gbl.outersub = gbl.currsub;
      gbl.outerentries = gbl.entries;
    }
    stb.curr_scope = savescope;
    ccff_close_unit_f90();
  } while (!gbl.eof_flag);
  finish(); /* finish does not return */
  return 0; /* never reached */
}

/* ************************************************************** */

/*
 * static structures/variables used in command line processing/init() function:
 */

#define __ATOI(s, p, l, r) _atoi(s, p, l)
static char *objectfile = NULL;
static const char *outfile_name = NULL;
LOGICAL fpp_ = FALSE;
static int preproc = -1; /* not specified */

/* ***************************************************************** */

/*
 * Various types of AST dumpers, wrapper functions
 */

static void
dump_stds(void)
{
  dstds(0, 0);
}

static void
dump_sstds(void)
{
  dsstds(0, 0);
}

static void
dump_stdps(void)
{
  dstdps(0, 0);
}

/** \brief Dump symbols
 */
static void
dump_symbols(void)
{
  dsyms(0, 0);
}

/** \brief Dump all symbols
 */
static void
dump_all_symbols(void)
{
  dsyms(1, 0);
}

/** \brief Dump symbols from current source file
 */
static void
dump_current_symbols(void)
{
  dsyms(stb.firstosym, 0);
}

/** \brief Yet another symbol table dumper
 */
static void
dump_old_symbols(void)
{
  symdmp(gbl.dbgfil, 0);
}

/** \brief Dump memory area
 */
static void
report_area(void)
{
  reportarea(0);
}

static const char *current_phase;

/** \brief Dump stg statistics
 */
static void
dump_stg_stats(void)
{
  dump_stg_stat(current_phase);
}

/**
 * \brief Initialize Fortran frontend at the beginning of compilation.
 */
static void
init(int argc, char *argv[])
{
  int argindex;
  int nosuffixcheck = 0;
  char *sourcefile = NULL;
  const char *stboutfile = NULL;
  const char *listfile = NULL;
  const char *cppfile = NULL;
  const char *tempfile = NULL;
  const char *asmfile = NULL;
  const char *dbgfile = NULL;
  const char *file_suffix;
  int copy_curr_file = 1;
  int i;
  LOGICAL errflg = FALSE;
  FILE *fd;
  static struct {
    const char *nm; /* name, 0 = end of list */
    int form;       /* 0 = fixed, 1 = form */
    int fpp;        /* 0 = don't preprocess, 1 = preprocess */
  } suffixes[] = {
          {".hpf", 0, 0}, {".f", 0, 0},   {".F", 0, 1},
          {".f90", 1, 0}, {".F90", 1, 1}, {".f95", 1, 0},
          {".F95", 1, 1}, {".for", 0, 0}, {".fpp", 0, 1},
          {0, 0, 0},
  };
  time_t now;

  /* Fill xflags array with zeros */
  memset(flg.x, 0, sizeof(flg.x));

  flg.freeform = -1;
  file_suffix = ".f90"; /* default suffix for source files */
  /*
   * initialize error and symbol table modules in case error messages are
   * issued:
   */
  errini();
  gbl.curr_file = NULL;
  gbl.fn = NULL;

#if defined(TARGET_SUPPORTS_QUADFP)
  /* See tools/flang1/utils/machar/machar.n for info on machine type 3.
     This should be called before dtypeinfo is initialized by init_chartab. */
  flg.x[45] = 3;
#endif

  sym_init();
  interf_init();
  BZERO(&sem, SEM, 1);

  /* fill in date and time */
  time(&now);
  strftime(gbl.datetime, sizeof gbl.datetime, "%m/%d/%Y  %H:%M:%S",
           localtime(&now));

  gbl.ipaname = NULL;
  argindex = 0;

  flg.x[79] = 16; /* Hardwire XBIT(79,16): CSE DP loads for a distance of 16 */

  flg.x[27] = -1; /* overlap not set */

  if (argc < 2)
    goto empty_cl;

  const char *tp;            /* Target architecture */
  const char *omptp = NULL;  /* OpenMP Target architecture */
  int vect_val;              /* Vectorizer settings */
  const char *modexport_val; /* Modexport file name */
  const char *modindex_val;  /* Modindex file name */
  char **module_dirs;  /* Null-terminated list of module directories */
  bool arg_preproc;    /* Argument to turn preprocessor on and off */
  bool arg_freeform;   /* Argument to force free-form source */
  bool arg_extend;     /* Argument to force line extension */
  bool arg_reentrant;  /* Argument to enable generating reentrant code */

  /* Create a datastructure of various dump actions and their names */
  action_map_t *dump_map; /* Deallocated after arguments are parsed */
  create_action_map(&dump_map);
  add_action(dump_map, "ast", dump_ast);
  add_action(dump_map, "dtype", dumpdts);
  add_action(dump_map, "std", dump_stds);
  add_action(dump_map, "sstd", dump_sstds);
  add_action(dump_map, "stdp", dump_stdps);
  add_action(dump_map, "sym", dump_symbols);
  add_action(dump_map, "syms", dump_symbols);
  add_action(dump_map, "symtab", dump_symbols);
  add_action(dump_map, "allsym", dump_all_symbols);
  add_action(dump_map, "stats", dump_stg_stats);
  add_action(dump_map, "area", report_area);
  add_action(dump_map, "olddtype", dmp_dtype);
  add_action(dump_map, "odtype", dmp_dtype);
  add_action(dump_map, "oldsym", dump_old_symbols);
  add_action(dump_map, "osym", dump_current_symbols);
  add_action(dump_map, "hsym", dump_current_symbols);
  add_action(dump_map, "hsyms", dump_current_symbols);
  add_action(dump_map, "common", dcommons);
  add_action(dump_map, "commons", dcommons);
  add_action(dump_map, "nast", dumpasts);
  add_action(dump_map, "stdtree", dumpstdtrees);
  add_action(dump_map, "stdtrees", dumpstdtrees);
  add_action(dump_map, "shape", dumpshapes);
  add_action(dump_map, "aux", dumplists);
  /* Initialize the map that will be used by dump handler later */
  create_action_map(&phase_dump_map);

  arg_parser_t *arg_parser;

  create_arg_parser(&arg_parser, true);

  /* Register two ways for supplying source file argument */
  register_filename_arg(arg_parser, &(gbl.src_file));
  register_string_arg(arg_parser, "src", &(gbl.src_file), NULL);
  /* Output file (.ilm) */
  register_combined_bool_string_arg(arg_parser, "output", (bool *)&(flg.output),
                                    &outfile_name);
  /* Other files to input or output */
  register_string_arg(arg_parser, "stbfile", &stboutfile, NULL);
  register_string_arg(arg_parser, "modexport", &modexport_val, NULL);
  register_string_arg(arg_parser, "modindex", &modindex_val, NULL);
  register_string_arg(arg_parser, "qfile", &dbgfile, NULL);

  /* Optimization level */
  register_integer_arg(arg_parser, "opt", &(flg.opt), 1);

  /* Debug */
  register_boolean_arg(arg_parser, "debug", (bool *)&(flg.debug), 0);
  register_integer_arg(arg_parser, "ieee", &(flg.ieee), 0);

  /* Allocate space for command line macro definitions */
  flg.def = (char **)getitem(8, argc * sizeof(char *));
  flg.undef = (char **)getitem(8, argc * sizeof(char *));
  flg.idir = (char **)getitem(8, argc * sizeof(char *));
  module_dirs = (char **)getitem(8, argc * sizeof(char *));
  /* Command line macro definitions */
  register_string_list_arg(arg_parser, "def", flg.def);
  register_string_list_arg(arg_parser, "undef", flg.undef);
  register_string_list_arg(arg_parser, "idir", flg.idir);
  register_string_list_arg(arg_parser, "moddir", module_dirs);

  /* x flags */
  register_xflag_arg(arg_parser, "x", flg.x);
  register_yflag_arg(arg_parser, "y", flg.x);
  /* Debug flags */
  register_qflag_arg(arg_parser, "q", flg.dbg,
                     (sizeof(flg.dbg) / sizeof(flg.dbg[0])));
  register_action_map_arg(arg_parser, "qq", phase_dump_map, dump_map);

  /* Other flags */
  register_boolean_arg(arg_parser, "mp", (bool *)&(flg.smp), false);
  register_string_arg(arg_parser, "fopenmp-targets", &omptp, NULL);
  register_boolean_arg(arg_parser, "preprocess", &arg_preproc, true);
  register_boolean_arg(arg_parser, "reentrant", &arg_reentrant, false);
  register_integer_arg(arg_parser, "terse", &(flg.terse), 1);
  register_inform_level_arg(arg_parser, "inform",
                            (inform_level_t *)&(flg.inform), LV_Inform);
  register_boolean_arg(arg_parser, "hpf", (bool *)&(flg.hpf), true);
  register_boolean_arg(arg_parser, "static", (bool *)&(flg.doprelink), true);
  register_boolean_arg(arg_parser, "quad", (bool *)&(flg.quad), true);
  register_boolean_arg(arg_parser, "qp", (bool *)&(flg.qp), true);
  register_boolean_arg(arg_parser, "freeform", &arg_freeform, false);
  register_string_arg(arg_parser, "tp", &tp, NULL);
  register_string_arg(arg_parser, "stdinc", &(flg.stdinc), (char *)1);
  register_integer_arg(arg_parser, "vect", &(vect_val), 0);
  register_boolean_arg(arg_parser, "standard", (bool *)&(flg.standard), false);
  register_boolean_arg(arg_parser, "save", (bool *)&(flg.save), false);
  register_boolean_arg(arg_parser, "extend", &arg_extend, false);
  register_boolean_arg(arg_parser, "recursive", (bool *)&(flg.recursive),
                       false);
  register_string_arg(arg_parser, "cmdline", &(flg.cmdline), NULL);
  register_boolean_arg(arg_parser, "es", (bool *)&(flg.es), false);
  register_boolean_arg(arg_parser, "pp", (bool *)&(flg.p), false);
  register_boolean_arg(arg_parser, "list-macros", &flg.list_macros, false);

  /* Set values form command line arguments */
  parse_arguments(arg_parser, argc, argv);

  /* Direct debug output */
  if (was_value_set(arg_parser, &(flg.dbg)) ||
      was_value_set(arg_parser, phase_dump_map)) {
#if DEBUG
    dodebug = 1;
#endif
    if (dbgfile) {
      gbl.dbgfil = fopen(dbgfile, "w");
      if (gbl.dbgfil == NULL)
        errfatal(5);
    } else if ((flg.dbg[0] & 1) || gbl.src_file == NULL) {
      gbl.dbgfil = stderr;
    } else {
      if (ipa_import_mode) {
        tempfile = mkfname(gbl.src_file, file_suffix, ".qdbh");
      } else {
        tempfile = mkfname(gbl.src_file, file_suffix, ".qdbf");
        if ((gbl.dbgfil = fopen(tempfile, "w")) == NULL)
          errfatal(5);
      }
    }
  }

  /* Set preporocessor and Fortran source form
   * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   * FIXME this logic needs to be moved to where those values are consumed
   */
  if (was_value_set(arg_parser, &arg_preproc)) {
    /* If the argument was present on command line set the value, otherwise
     * keep "undefined" -1 */
    preproc = arg_preproc;
  }
  if (was_value_set(arg_parser, &arg_freeform)) {
    /* If the argument was present on command line set the value, otherwise
     * keep "undefined" -1 */
    flg.freeform = arg_freeform;
  }

  /* Enable reentrant code */
  if (was_value_set(arg_parser, &arg_reentrant)) {
    if (arg_reentrant) {
      flg.x[7] |= 0x2;      /* inhibit terminal func optz. */
      flg.recursive = TRUE; /* no static locals */
    } else {
      flg.x[7] &= ~(0x2);
      flg.recursive = FALSE;
    }
  }

  /* Free memory */
  destroy_arg_parser(&arg_parser);
  destroy_action_map(&dump_map);

  /* Now do some postprocessing
   * ^^^^^^^^^^^^^^^^^^^^^^^^^^
   */

  /* Check optimization level */
  if (flg.opt > 4) {
    fprintf(stderr, "%s-W-Opt levels greater than 4 are not supported\n", version.lang);
  }
  /* -nostatic postprocessing */
  if (!flg.doprelink)
    flg.ipa |= 0x50; /* don't defer initialization, issue errors */

  /* Postprocess target architecture */
  do_set_tp(tp);
#ifdef OMP_OFFLOAD_LLVM
  if(omptp != NULL)
    flg.omptarget = TRUE;
#endif
  /* Vectorizer settings */
  flg.vect |= vect_val;
  if (flg.vect & 0x10)
    flg.x[19] &= ~0x10;
  if (flg.vect & 0x20)
    flg.x[19] &= ~8;
  set_yflag(34, 0x30);

  /* modexport file name */
  mod_combined_name(modexport_val);
  /* modindex file name */
  mod_combined_index(modindex_val);

  /* Extend source file lines */
  if (arg_extend)
    flg.extend_source = 132;

  /* Set module directory linked list
   * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   * FIXME this is bad, have different implementations for different string
   * lists, that needs to stop.
   */
  char **module_dir = module_dirs;
  while (module_dir && *module_dir) {
    moddir_list *mdl;
    NEW(mdl, moddir_list, 1);
    mdl->module_directory = *module_dir;
    mdl->next = NULL;
    if (module_directory_list == NULL) {
      module_directory_list = mdl;
    } else {
      moddir_list *link;
      for (link = module_directory_list; link->next; link = link->next)
        ;
      link->next = mdl;
    }
    ++module_dir;
  }

  flg.genilm = TRUE;
  /* set -x 49 0x400000, F90-style output */
  set_xflag(49, 0x400000);
  /* set -x 58 0x10000, handle all pointers */
  set_xflag(58, 0x10000);
  gbl.is_f90 = TRUE;
  /* set -x 58 0x40, and reset hpf flag, no static-init */
  set_xflag(58, 0x40);
  flg.defaulthpf = flg.hpf = FALSE;
  flg.defaultsequence = flg.sequence = TRUE;
  /* set -x 58 0x20000, allocate temps only as big as needed */
  set_xflag(58, 0x20000);

#ifdef TARGET_SUPPORTS_QUADFP
  if (flg.qp) {
    /* set -y 57 0x4, enable quad-precision real */
    set_yflag(57, 0x4);
    /* set -y 57 0x8, enable quad-precision complex */
    set_yflag(57, 0x8);
  }
#endif

  if (XBIT(25, 0xf0)) {
    fprintf(stderr, "%s-I-Beta Release Optimizations Activated\n", version.lang);
  }

  if (flg.x[27] == -1)
    flg.x[27] = 4; /* default/max overlap of 4 */

  if (flg.es && !flg.p)
    flg.x[123] |= 0x100;

  empty_cl:
  if (gbl.src_file == NULL) {
    if (flg.ipa & 0x0a) {
      /* for IPA propagation or when generating static$init, no sourcefile */
      gbl.src_file = "pghpf.prelink.f";
      sourcefile = strdup(gbl.src_file);
      gbl.srcfil = NULL;
      copy_curr_file = 0;
    } else {
      gbl.src_file = "STDIN.f";
      sourcefile= strdup(gbl.src_file);
      gbl.srcfil = stdin;
    }
    goto do_curr_file;
  }

  if (errflg)
    finish();

  if (ipa_import_mode) {
    if (gbl.src_file) {
      sourcefile = strdup(gbl.src_file);
      basenam(gbl.src_file, "", sourcefile);
    } else {
      gbl.src_file = "STDIN.f";
      sourcefile = strdup(gbl.src_file);
    }
    file_suffix = "";
    for (char *s = sourcefile; *s; ++s) {
      if (*s == '.')
        file_suffix = s;
    }
  } else {
    if (!nosuffixcheck) {
      /* open sourcefile */
      for (i = 0; suffixes[i].nm; ++i) {
        int lsuf, lsrc;
        lsuf = strlen(suffixes[i].nm);
        lsrc = strlen(gbl.src_file);
        if (lsuf >= lsrc)
          continue;
        if (strcmp(gbl.src_file + (lsrc - lsuf), suffixes[i].nm))
          continue;
        if ((gbl.srcfil = fopen(gbl.src_file, "r")) != NULL) {
          /* fill in flg.freeform, file_suffix, fpp_, gbl.src_file */
          if (flg.freeform == -1)
            flg.freeform = suffixes[i].form;
          file_suffix = suffixes[i].nm;
          if (suffixes[i].fpp) {
            if (preproc == -1 || preproc == 1)
              fpp_ = TRUE;
            /* -nopreproc overrides use of .F extension */
          }
          /* strip pathname, if any */
          sourcefile = (char *)malloc(strlen(gbl.src_file) + 1);
          /* base name strips the pathname */
          basenam(gbl.src_file, "", sourcefile);
          goto is_open;
        }
        /* ** else will be reported_as_an error(2...) below ** */
      }
    }
    if ((gbl.srcfil = fopen(gbl.src_file, "r")) != NULL) {
      /* fill in sourcefile, file_suffix */
      sourcefile = (char *)malloc(strlen(gbl.src_file) + 1);
      basenam(gbl.src_file, "", sourcefile);
      file_suffix = "";
      for (char *s = sourcefile; *s; ++s) {
        if (*s == '.')
          file_suffix = s;
      }
      goto is_open;
    }
    /* not found */
    error(2, 4, 0, gbl.src_file, CNULL);
    is_open:
    if (preproc == 1)
      fpp_ = TRUE; /* -preproc forces preprocessing */
  }

  do_curr_file:

  if (gbl.file_name == NULL)
    gbl.file_name = gbl.src_file;
  if (sourcefile != NULL)
    gbl.module = mkfname(sourcefile, file_suffix, "");
  if (copy_curr_file)
    gbl.curr_file = gbl.src_file;

  /* process  object file: */

  gbl.objfil = NULL;

  if (sourcefile == NULL) {
    if (!flg.output || outfile_name == NULL) {
      gbl.outfil = stdout;
    } else {
      if ((gbl.outfil = fopen(outfile_name, "w")) == NULL)
        error(4, 0, 0, "Unable to open output file", outfile_name);
    }
    if (OUTPUT_DWARF && (dbg_file == NULL) && outfile_name != NULL) {
      /* make dwarf debug info file from the source file */
      mkDwfInfoFilename();
      if ((dbg_file = fopen(dbg_file_name, "wb")) == NULL)
        errfatal(9);
      dwarf_set_fn();
    } else {
      flg.debug = 0;
    }
    if (stboutfile != NULL) {
      if ((gbl.stbfil = fopen(stboutfile, "w")) == NULL)
        errfatal(9);
    } else {
      gbl.stbfil = 0;
    }
  } else {
    /* process listing file */
    if (flg.code || flg.list || flg.xref) {
      if (listfile == NULL) {
        /* make listing filename from sourcefile name */
        listfile = mkfname(sourcefile, file_suffix, LISTFILE);
      }
      if ((fd = fopen(listfile, "w")) == NULL)
        errfatal(3);
      list_init(fd);
    }
    if (OUTPUT_DWARF && (dbg_file == NULL)) {
      /* make dwarf debug info file from the source file */
      if (outfile_name != NULL) {
        mkDwfInfoFilename();
      } else {
        dbg_file_name = mkfname(sourcefile, file_suffix, ".dbg");
      }
      if ((dbg_file = fopen(dbg_file_name, "wb")) == NULL)
        errfatal(9);
    }
    if (stboutfile) {
      if ((gbl.stbfil = fopen(stboutfile, "w")) == NULL)
        errfatal(9);
    } else {
      gbl.stbfil = NULL;
    }

    /* process assembly output file */
    if (flg.asmcode) {
      if (asmfile == NULL) {
        /* make assembly filename from sourcefile name */
        asmfile = mkfname(sourcefile, file_suffix, ASMFILE);
      }
      if ((gbl.asmfil = fopen(asmfile, "w")) == NULL)
        errfatal(9);
    } else /* do this for compilers which write asm code to stdout */
      gbl.asmfil = stdout;
    /* process source output file */
    if (flg.output && !flg.es) {
      /* make output filename from sourcefile name */
      if (outfile_name == NULL) {
        outfile_name = mkfname(sourcefile, file_suffix, ".ilm");
      }
      if ((gbl.outfil = fopen(outfile_name, "w")) == NULL)
        error(4, 0, 0, "Unable to open output file", outfile_name);
    } else
      gbl.outfil = stdout;


    if (flg.doprelink && (flg.ipa & 0x03) == 0 && gbl.ipaname == NULL) {
      gbl.ipaname = mkfname(sourcefile, file_suffix, ".d");
      gbl.gblfil = NULL;
      unlink(gbl.ipaname);
    }

    /* create temporary file for preprocessor output & preprocess */
    if (!ipa_import_mode) {
      if (fpp_) {
        if (flg.es) {
          if (cppfile == NULL)
            gbl.cppfil = stdout;
          else if ((gbl.cppfil = fopen(cppfile, "w")) == NULL)
            errfatal(5);
        } else {
          if ((gbl.cppfil = tmpfile()) == NULL)
            errfatal(5);
        }
        fpp();
        if (flg.es || gbl.maxsev >= 3)
          finish();
        if (flg.list)
          list_page();
        scan_init(gbl.cppfil);
      } else
        scan_init(gbl.srcfil);
    }
#if DEBUG
    assert(flg.es == 0, "init:flg.esA", 0, 0);
#endif
    assemble_init();
    if (OUTPUT_DWARF && dbg_file != NULL) {
      dwarf_set_fn();
    }
  }
  free(sourcefile);
  gbl.func_count = 0;

  if (XBIT(125, 0x8))
    gbl.ftn_true = 1;
  else
    gbl.ftn_true = -1;

  /*
   * direct_init() must be called at a point where we are sure that
   * the values of flg members, especially xflags, can be propagated
   * to the global, routine, etc. directive data structures. For example,
   * direct_init() can only be called after the code above which can
   * disable the cuda/accel features in the code by clearing their
   * respective xflags.
   */
  direct_init();

  /* set mach, currently need for -mp=align optimization on sandybridge */
  set_mach(&mach, flg.tpvalue[0]);

  return;
}

/* *************************************************************** */

moddir_list *module_directory_list = NULL;

#if DEBUG

static void
do_debug(const char *phase)
{
  if (debugfunconly > 0 && gbl.func_count != debugfunconly) {
    /* only for some functions */
    return;
  }
  if (dodebug)
    fprintf(gbl.dbgfil, "{%s after %s\n", feature, phase);

  current_phase = phase;
  execute_actions_for_keyword(phase_dump_map, phase);
} /* do_debug */

#endif /* if DEBUG */

/* call this routine to clean up data structures if not compiling all the
 * way to the end */
static void
cleanup(void)
{
  direct_rou_end();
  dinit_end();
  df_dinit_end();
  freearea(15);
  postprocessing = FALSE;
} /* cleanup */

static void
reptime(void)
{
  char buf[80];
  int i;
  INT total;
  int prct;
  int tmp;

  total = 0;
  for (i = 0; i < _N_WHO; i++)
    total += xtimes[i];

  if (!DBGBIT(0, 8) || DBGBIT(14, 3))
    goto xbitcheck;

  if (flg.code || flg.list || flg.xref) {
    list_line("");
    list_line("  Timing stats:");
  } else if (gbl.dbgfil)
    fprintf(gbl.dbgfil, "  Timing stats:\n");
  for (i = 0; i < _N_WHO; i++) {
    if (xtimes[i]) {
      tmp = 100 * xtimes[i];
      prct = tmp / total;
      sprintf(buf, "    %-10.10s %15d millisecs %5d%%", who[i], xtimes[i],
              prct);
      if (flg.code || flg.list || flg.xref)
        list_line(buf);
      else if (gbl.dbgfil)
        fprintf(gbl.dbgfil, "%s\n", buf);
    }
  }

  sprintf(buf, "    Total time %15d millisecs", total);
  if (flg.code || flg.list || flg.xref) {
    list_line(buf);
  } else if (gbl.dbgfil)
    fprintf(gbl.dbgfil, "%s\n", buf);

  xbitcheck:
  if (!XBIT(0, 1))
    return;
  fprintf(stderr, "  Timing stats:\n");

  for (i = 0; i < _N_WHO; i++) {
    if (xtimes[i]) {
      tmp = 100 * xtimes[i];
      prct = tmp / total;
      sprintf(buf, "    %-10.10s %15d millisecs %5d%%", who[i], xtimes[i],
              prct);
      fprintf(stderr, "%s\n", buf);
    }
  }
  sprintf(buf, "    Total time %15d millisecs", total);
  fprintf(stderr, "%s\n", buf);
}

static void
datastructure_reinit(void)
{
  /* initialize global variables:  */
  gbl.currsub = 0;
  gbl.arets = FALSE;
  gbl.rutype = RU_PROG;
  gbl.cmblks = NOSYM;
  gbl.externs = NOSYM;
  gbl.consts = NOSYM;
  gbl.locals = NOSYM;
  gbl.statics = NOSYM;
  gbl.ent_select = 0;
  gbl.stfuncs = NOSYM;
  gbl.locaddr = 0;
  gbl.saddr = 0;
  set_bss_addr(0);
  gbl.autobj = NOSYM;
  gbl.asgnlbls = 0;
  gbl.exitstd = 0;
  gbl.tp_adjarr = NOSYM;
  gbl.p_adjarr = NOSYM;
  gbl.p_adjstr = NOSYM;
  gbl.denorm = FALSE;
  gbl.inomptarget = false;
  /* restore opt flag to its original value */
  flg.opt = saveoptflag;
  flg.vect = savevectflag;
  flg.x[8] = savex8flag;
  flg.recursive = saverecursive;

  sym_init();   /* initialize symbol table module */
  dinit_init(); /* initialize data init file module  */
  /* close data initialization files */
  dinit_end();
  if (astb.df != NULL)
    fclose(astb.df);
  astb.df = NULL;

  astout_init();
} /* datastructure_reinit */

/** \brief perform initializations for new user subprogram unit.
*/
void
reinit(void)
{
  scan_opt_restore(); /* if OPTIONS statement was seen in prev */

  datastructure_reinit();

  semant_init(ipa_export_file != 0 && ipa_import_mode);
  /* initialize semantic analyzer.
   * WARNING:  when compiling module subprograms,
   * it's assumed that the certain data structures
   * (asts, dtypes, etc.) of entities in the
   * module specification part will have the same
   * indices when imported into a CONTAINS'd
   * subprogram. All inits, on which importing
   * module information depends, must be peformed
   * before semant_init().
   */
  if (flg.xref)
    xrefinit();   /* initialize cross reference module */
  dpm_out_init(); /* initialize dp output module -- should
                   * be replaced with call to transform_init().
                   */

  queue_tbp(0, 0, 0, 0, TBP_CLEAR_STALE_RECORDS);
}

/* *************************************************************** */

static int exitcode;

/** \brief set exit code for compiler (see finish() function)
    \param ec - the exit code to set
*/
void
set_exitcode(int ec)
{
  exitcode = ec;
}

/** \brief Write summary line to terminal, and exit compiler.
*/
void
finish(void)
{
  int maxfilsev;
  static int called = 0;

  if (!ipa_import_mode)
    scan_fini();
  if (IPA_INHERIT_ENABLED && (flg.opt >= 2 || IPA_COLLECTION_ENABLED)) {
    ipa_fini();
  }
  ipasave_fini();
  DUMP("fini");
  symtab_fini();
  fih_fini();
  ast_fini();
  direct_fini();
  sem_fini();
  mod_fini();
  if (XBIT(123, 0x30000)) {
    import_module_print();
  }

  called++;
  if (gbl.maxsev < 3 && called == 1 && (XBIT(123, 2) || XBIT(123, 8))) {
    FILE *fp;
    char *dependfile;

    if (XBIT(123, 8)) {
      /* -MD option:  Print list of include files to file <program>.d */
      dependfile = mkfname(gbl.module, "", ".d");
      if ((fp = fopen(dependfile, "w")) == NULL)
        errfatal(5);
    } else {
      /* -M option:  Print list of include files to stdout */
      fp = stdout;
    }
    if (gbl.moddependfil) {
      rewind(gbl.moddependfil);
      while (1) {
        int c;
        c = fgetc(gbl.moddependfil);
        if (c == EOF)
          break;
        fputc(c, fp);
      }
    }
    if (!XBIT(123, 0x40000)) {
      fprintf(fp, "%s%s : ", gbl.module, OBJFILE);
      fprintf(fp, "%s ", gbl.src_file);
    } else {
      fprintf(fp, "\"%s%s\" : ", gbl.module, OBJFILE);
      fprintf(fp, "\"%s\" ", gbl.src_file);
    }
    if (gbl.dependfil) {
      rewind(gbl.dependfil);
      while (1) {
        int c;
        c = fgetc(gbl.dependfil);
        if (c == EOF)
          break;
        fputc(c, fp);
      }
    }
    fputc('\n', fp);
    if (XBIT(123, 8))
      fclose(fp);
  }

  if (!flg.es) {
    reptime();
    maxfilsev = summary(TRUE, FALSE);
  } else
    maxfilsev = gbl.maxsev;

  if (maxfilsev >= 3) {
    /* remove objectfile if there were severe errors */
    if (flg.object && gbl.objfil)
      if (!DBGBIT(0, 16))
        unlink(objectfile);
  } else {
    if (gbl.objfil != NULL)
      fclose(gbl.objfil);
    if (IPA_ENABLED || IPA_INHERIT_ENABLED)
      ipasave_closefile();
    if (IPA_INHERIT_ENABLED)
      ipa_closefile();
    if (!flg.es) {
      fini();
    }
  }
  if (gbl.asmfil != NULL && gbl.asmfil != stdout)
    fclose(gbl.asmfil);
  if (gbl.outfil != NULL && gbl.outfil != stdout)
    fclose(gbl.outfil);
  if (IPA_ENABLED && ipa_export_file && !ipa_import_mode) {
    /* export the program unit for IPA recompilation */
    ipa_export_close();
  }

  freearea(8);      /* temporary filenames and pathnames space  */
  free_getitem_p(); /* getitem_p tbl contains area 8 pointers */
  destroy_action_map(&phase_dump_map);
  /*free( gbl.src_file );*/
  gbl.src_file = NULL;
  if (maxfilsev >= 3) {
    if (!XBIT(123, 0x40000) || exitcode == 0)
      exit(1);
    else
      exit(exitcode);
  } else
    exit(0);
}

/* ******************************************************************* */

/* dummies for dwarf */
FILE *dbg_file = NULL;
char *dbg_file_name = NULL;
void dwarf_set_fn(void) {}
void setrefsymbol(int symbol) {}
void scan_for_dwarf_module(void) {}

static void
do_set_tp(const char *tp)
{
  if (tp) {
    if (strcmp(tp, "x64") == 0) {
      set_tp("k8-64");
      set_tp("p7-64");
    } else {
      set_tp(tp);
    }
  }
}

/** \brief This function creates a dwarf debug info filename from source file.
*/
static void
mkDwfInfoFilename(void)
{
  int i;
  /* first, find the file suffix of the output file (created by the driver) */
  for (i = strlen(outfile_name) - 1; i > 0; i--)
    if (outfile_name[i] == '.')
      break;
  if (i == 0)
    i = strlen(outfile_name) - 1; /* punt if no suffix */
  dbg_file_name = mkfname(outfile_name, &outfile_name[i], ".dbg");
}

/** \brief called at end of processing contains subprograms */
void
end_contained(void)
{
  lower_end_contains();
  if (ipa_export_file && !ipa_import_mode) {
    ipa_export_endcontained();
  }
}

static void
fini()
{
  assemble_end();
}

/* dummies required to link when we don't have IPA */

int IPA_Vestigial = 0;

void ipa_closefile() {}
void ipa_export() {}
void ipa_export_close() {}
void ipa_export_endcontained() {}
void ipa_export_endmodule() {}
void ipa_export_highpoint() {}
void ipa_export_open(char *export_filename) {}
void ipa_header1(int currfunc) {}
void ipa_header2(int currfunc) {}
void ipa_import_highpoint(void) {}
void ipa_import_open(char *import_file, BIGUINT offset) {}
void ipa_import(void) {}
void ipa_init() {}
void ipa_report() {}
void ipasave_closefile() {}
void ipasave_compname(char *name, int argc, char *argv[]) {}
void ipasave_endfunc() {}
void ipasave_fini(void) {}
void ipasave(void) {}
void ipa_startfunc(int currfunc) {}
void ipa_fini() {}
void fill_ipasym() {}
void ipa() {}
void ipa_set_vestigial_host() {}
int IPA_isnoconflict(int sptr) { return 0; }
