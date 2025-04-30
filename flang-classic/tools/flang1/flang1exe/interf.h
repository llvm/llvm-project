/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
  *
  * \brief Various definitions and prototypes for importing/exporting modules
  * and IPA information.
  */

typedef struct {
  /* 'high water' marks of certain structures necessary for exporting data
   * during the creation and updating of the module file.
   * Upon the creation (export_all()) of the module file, the values are the
   * start values of the indices corresponding to the data structures.  After
   * structures are created from the data read in from a module file, these
   * values are updated in preparation for subsequent exporting of relevant
   * data structures associated with the 'header' of a module subroutine or
   * function.
   */
  struct {
    int dt;
    int ast;
    int maxast;  /* max ast of asts to be exported */
    int maxsptr; /* max sptr of syms to be exported */
  } hmark;
  LOGICAL hpf_library;             /* host scope uses hpf_library */
  LOGICAL hpf_local_library;       /* host scope uses hpf_local_library */
  LOGICAL iso_c_library;           /* host scope uses iso_c_bindings */
  LOGICAL iso_fortran_env_library; /* host scope uses iso_c_bindings */
  LOGICAL ieee_arith_library;      /* host scope uses ieee_arithmetic */
} EXPORTB;

typedef struct moddir_list {
  char *module_directory;
  struct moddir_list *next;
} moddir_list;
extern moddir_list *module_directory_list;

extern EXPORTB exportb;

typedef struct {
  char *modulename;
  int modulesym;
} IMPORT_LIST;

extern struct imported_modules_struct {
  IMPORT_LIST *list;
  int avail;
  int size;
  int module_avail;
  int host_avail;
} imported_modules;

int symbol_is_visible(int sptralias);
void adjust_symbol_accessibility(int);
void update_use_tree_exceptions(void);
void export_module(FILE *, char *, int, int);
void export_append_sym(int);
void export_append_host_sym(int);
void export_host_subprogram(FILE *, int, int, int, int);
void export_module_subprogram(FILE *, int, int, int, int);
int get_module_file_name(char *modulename, char *filename, int len);

/*  getitem area for USE statement temp storage; pick an area not used by
 *  semant.
 */
#define USE_AREA 11
#define USE_TREE_AREA 14

/** 
   When importing a module, to mark whether to import private members
   or not.
 */
typedef enum { INCLUDE_PRIVATES, EXCLUDE_PRIVATES } WantPrivates;

void import_init(void);
int import_inline(FILE *, char *);
int import_interproc(FILE *, char *, char *, char *);
int import_static(FILE *, char *);
SPTR import_module(FILE *, char *, SPTR, WantPrivates, int);
void import_host(FILE *, const char *, int, int, int, int, int, int, int);
void import_module_end(void);
int imported_directly(char *name, int except);
void init_use_tree(void);
void remove_from_use_tree(char *module);
int aliased_sym_visible(int sptralias);
void interf_init(void);
void ipa_import_open(char *import_file, BIGUINT offset);
void ipa_export_open(char *export_filename); /* exterf.c */
void ipa_export(void);                       /* exterf.c */
void ipa_export_close(void);                 /* exterf.c */

#ifdef INSIDE_INTERF

/* 0x1 deprecated */
/* 0x2 deprecated */
#define MOD_ANY 0x4  /* 64-bit target mod file */
#define MOD_I8  0x8  /* -i8 */
#define MOD_R8  0x10 /* -r8 */
#define MOD_LA  0x20 /* -Mlarge_arrays */
#define MOD_PG  0x40 /* compilers' own module files */

#undef IVSN
#define IVSN 35
#undef IVSN_24
#define IVSN_24 24
#undef IVSN_27
#define IVSN_27 27
/*
 * WARNING -- changing IVSN means that old .mod files can't be read.
 *   IF the version number must be changed, presumably we will add the
 *   ability to read the older version ...
 *
 * HISTORY:
 *  <= 24 - 10.6 & before
 *     25 - add DT_DEFERCHAR and DT_DEFERNCHAR,  32 more SYM flags, and
 *          another SYM field --- briefly used in DEV only, NOT released,
 *          compatibility FAILS
 *     26 - add DT_DEFERCHAR and DT_DEFERNCHAR,  32 more SYM flags, and
 *          3 more SYM fields.  Prefix the set of new flags & fields with
 *          ' A' to help with compatiblity with IVSN 24
 *     27 - add platform(x86 and x86-64 after version number.  This is to
 *          to prevent 32-bit module being used by 64-bit compiler and
 *          vice versa.
 *     28 - 32 more SYM flags, and 3 more SYM fields.  Prefix the set of
 *          `new flags & fields with
 *          ' B' to help with compatiblity with IVSN 24
 *     29 - Embed descriptor for scalar "non-polymorphic" pointer members in
 *          the derived type object. We do this for consistency with other
 *          members including polymorphic scalars.
 *     30 - Add more 1 AST field, w18; used to record the 'end label' of
 *          various MP ASTs such as PARALLEL/PDO/TASK/SECTIONS.
 *     31 - Add MP_ATOMICxxx for atomic operations
 *     32 - add compiler own module files flag into platform flag. 
 *          It is set if it is compiler module file.
 *     33 - Add MP_TASKLOOP[REG] for taskloop
 *     34 - half precision and half-complex datatypes
 *     35 - Add flag for null subc in typedef initializer
 */

/*
 *  Interface file format:
 *
 *  line 1  -  Vversionnumber module-name
 *  line 2  -  source-file-name-len  source-file-name Sstb.firstosym
 *  line 3  -  time-date-stamp
 *
 *  for module files:
 *  zero or more 'use' lines:
 *          -  use module-name [scope]
 * where 'module-name' is the textual module name, and 'scope' is an integer
 * greater than zero if the module is used at the start of the module
 * and should be transitively used in the using program.
 *
 *  Remaining lines - records begin with a 'letter':'
 *
 *  -  -  comment line
 *  A  -  ast definition line
 *  B  -  based symbol definition line
 *  C  -  reserved for constructors
 *  D  -  data dtype definition line
 *  E  -  equivalence line
 *  F  -  formal arguments
 *  G  -  word of an align descriptor
 *  H  -  shadow information for an array
 *  I  -  data initialization
 *  L  -  storage overLap list
 *  M  -  mangled derived symbol descriptor
 *  N  -  namelist members
 *  O  -  overloaded (generic) descriptor
 *  P  -  module predeclared (e.g. ST_HL, ST_HLL) symbol becomes predeclared
 *  Q  -  module procedure descriptor
 *  R  -  Rename line for variables in used modules
 *  S  -  symbol definition line
 *  T  -  shape descriptor
 *  U  -  element of a distribute record
 *  V  -  STDs
 *  W  -  ast arg table
 *  X  -  asd (subscript) table
 *  Y  -  ast list items
 *  Z  -  end of file
 *
 * ******
 *  For static initialization files, there is another section,
 *  following the Z line, containing initialization info:
 *  A  -  ast pointer to value reference in data statement
 *  D  -  DO start in data statement
 *  E  -  DO end in data statement
 *  I  -  variable initialization in type statement
 *  L  -  literal integer
 *  J  -  start of data statement
 *  O  -  implied do initializer
 *  S  -  typedef initializer (subc used)
 *  R  -  aRray initializer (subc used)
 *  V  -  Variable being initialized in data statement
 *  W  -  subtype pointer to variable being initialized
 *  X  -  eXpression initializer
 */

extern EXPORTB exportb;
#endif
