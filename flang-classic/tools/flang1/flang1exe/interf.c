/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file interf.c
   \brief Routines for importing symbols from .mod files and from IPA.
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
#include "state.h"
#include "lz.h"
#include "rtlRtns.h"

#define INSIDE_INTERF
#define INTERFH_RCSID
#include "interf.h"

#define TRACEFLAG 48
#define TRACEBIT 1
#define TRACESTRING "interf-"
#include "trace.h"
#include "fih.h"
#include "symutl.h"
#include "lower.h"
#include "extern.h"

/* true, if reading in a module file for a 'contained' subprogram. */
static LOGICAL inmodulecontains = FALSE;

static int for_module = 0;
static LOGICAL for_inliner = FALSE;
static LOGICAL for_interproc = FALSE;
static LOGICAL for_static = FALSE;
static LOGICAL for_host = FALSE;
static LOGICAL old_astversion = FALSE;
static int top_scope_level = 0;

static int import_errno = 0;
static int import_osym = 0;

static void put_dinit_record(int, INT);
static void put_data_statement(int, int, int, lzhandle *, const char *, int);
static int import_mk_newsym(char *name, int stype);

static int BASEsym, BASEast, BASEdty, BASEmod, ADJmod;
static int HOST_OLDSCOPE = 0, HOST_NEWSCOPE = 0;

static char **modinclist = NULL;
static int modinclistsize = 0, modinclistavl = 0;

#define MAX_FNAME_LEN 2050
#define MOD_SUFFIX ".mod"

/** \brief 'interface' initialization, called once per compilation
  * (source file).
  */
void
interf_init()
{
/* Disable checking SYM size in Windows.
 * https://github.com/flang-compiler/flang/issues/1043
 */
#if DEBUG && !defined(_WIN64)
  assert(sizeof(SYM) / sizeof(INT) == 46, "bad SYM size",
         sizeof(SYM) / sizeof(INT), 4);
  assert(sizeof(AST) / sizeof(int) == 19, "interf_init:inconsistent AST size",
         sizeof(AST) / sizeof(int), 2);
#endif
}

/* ------------------------------------------------------------------ */
/* ----------------------- Import Utilities ------------------------- */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*   Read symbols from export file  */
/* This is used for:
 *   module interface files
 *   interprocedural analysis
 *   procedure inlining
 *   static variable initialization
 */

/*  getitem area for module temp storage; pick an area not used by
 *  the caller of export/import.
 */
#define MOD_AREA 18
#define PERM_AREA 8
#define MOD_USE_AREA 4

/* ------------------------------------------------------------------ */
/* ----------------------- Import Utilities ------------------------- */
/* ------------------------------------------------------------------ */

/* ----------------------------------------------------------- */

typedef struct {          /* info on data type read from encoded file */
  int id;                 /* number of this dtype when mod file created */
  int ty;                 /* type of dtype, TY_PTR, etc.  */
  int new_id;             /* dtype number for this compilation */
  LOGICAL dtypeinstalled; /* set if dtype complete */
  int hashnext;           /* in hash table linked list */
} DITEM;

typedef struct symitem {/* info on symbol read from encoded mod file */
  int sptr;             /* symbol table pointer when mod file created */
  int stype;            /* STYPE(sptr) */
  int sc;
  int dtype; /* (old) pointer to dtype */
  int ty;    /* TY_ value of dtype (constants only) */
  /* used to stash namelist pointer also */
  int symlk;
  int flags1, flags2, flags3, flags4;
  int new_sptr; /* symbol table pointer for this compilation */
  SYM sym;
  char name[MAXIDLEN + 1]; /* symbol name (only certain stypes) */
  char *strptr;            /* pointer to char string (constant) */
  struct symitem *next;
  struct symitem *hashnext; /* next item from hash table */
  int socptr;               /* overlap region pointer */
  int shadowptr;            /* new shadow region pointer */
} SYMITEM;

typedef struct alnitem {/* info on align descriptor from encoded file */
  int aln;              /* pointer when descriptor exported */
  int new_aln;          /* pointer after import */
  int r_target;
  int r_alignee;
  LOGICAL aligninstalled; /* if set, descriptor has been installed */
  struct alnitem *next;
} ALNITEM;

typedef struct dstitem {/* info on distribute descr from encoded file */
  int dst;              /* pointer when descriptor exported */
  int new_dst;          /* pointer after import */
  int rank;
  LOGICAL distinstalled; /* if set, descriptor has been installed */
  struct dstitem *next;
} DSTITEM;

typedef struct {/* info on ast read from mod file */
  int type;     /* A_TYPE(ast) */
  AST a;        /* AST data */
  int new_ast, old_ast;
  int link; /* link to next ast in hash table */
  int list, flags, shape;
} ASTITEM;

typedef struct {/* info on STDs read from file */
  int old;      /* old STD index */
  int ast;
  int label;
  int lineno;
  int findex;
  int flags;
  int new; /* new STD index */
} STDITEM;

typedef struct {/* info on a shd item read from file */
  int old;      /* old shd index */
  int new;      /* new shd index */
  int ndim;     /* number of dimensions */
  struct {      /* for each dimension lwb:upb:stride */
    int lwb;
    int upb;
    int stride;
  } shp[7];
} SHDITEM;

typedef struct {     /* info on argt item read from file */
  int old;           /* old argt index */
  int callfg;        /* 1 if any args has call flag set */
  int new;           /* new argt index */
  LOGICAL installed; /* this entry has been processed */
} ARGTITEM;

typedef struct {     /* info on ASD item read from file */
  int old;           /* old asd index */
  LOGICAL installed; /* this entry has been processed */
  int ndim;          /* number of dimensions */
  int subs[7];       /* subscripts */
} ASDITEM;

typedef struct {     /* info on astli list read file */
  int new;           /* new astli index */
  LOGICAL installed; /* this entry has been processed */
} ASTLIITEM;

typedef struct {/* module procedure record */
  int modp;     /* old sym pointer of module procedure */
  int syml;     /* symitem list of generics/operators */
} MODPITEM;

static struct {/* table of dtypes read from mod file   */
  DITEM *base;
  int avl;
  int sz;
} dtz;

static struct {/* table of formal arguments */
  int *base;
  int avl;
  int sz;
} flz;

static struct {/* table of overloaded functions */
  int *base;
  int avl;
  int sz;
} ovz;

static struct {/* table of derived mangled symbols */
  int *base;
  int avl;
  int sz;
} mdz;
#define MN_NENTRIES 2

static struct {  /* table of asts read from mod file   */
  ASTITEM *base; /* NOTE: index of ASTITEM + firstuast == ast index */
  int avl;
  int sz;
} astz;

/* hash size is 1024 entries = 2^10, so HASHMASK has 10 lower-order bits on */
#define ASTZHASHSIZE 1024
#define ASTZHASHMASK 0x03ff
static int astzhash[ASTZHASHSIZE];

static struct {/* table of stds read from file */
  STDITEM *base;
  int avl;
  int sz;
} stdz;

static struct {/* table of shds read from file */
  SHDITEM *base;
  int avl;
  int sz;
} shdz;

static struct {/* table of argts read from file */
  ARGTITEM *base;
  int avl;
  int sz;
} argtz;

static struct {/* table of asds read from file */
  ASDITEM *base;
  int avl;
  int sz;
} asdz;

static struct {/* table of astlis read from file */
  ASTLIITEM *base;
  int avl;
  int sz;
} astliz;

static struct {/* table of module procedure records read */
  MODPITEM *base;
  int avl;
  int sz;
} modpz;

static SYMITEM *symbol_list; /* list of symbols read from mod file  */
static ALNITEM *align_list;  /* list of align descrs read from mod file */
static DSTITEM *dist_list;   /* list of dist descrs read from mod file */

#define SYMHASHSIZE 521
static SYMITEM *symhash[SYMHASHSIZE];
#define DTHASHSIZE 521
static int dthash[DTHASHSIZE];

#define BUFF_LEN 4096
static char *buff = NULL;
static int buff_sz;
static char *currp;
static char import_name[MAXIDLEN + 1];

static int curr_platform = MOD_ANY;

static char *import_sourcename = NULL;
static int import_sourcename_len = 0;
static LOGICAL ignore_private = FALSE;
static int curr_import_findex = 0;

static char *read_line(FILE *);
static ISZ_T get_num(int);
static void get_string(char *);
static void get_nstring(char *, int);

static void new_dtypes(void);
static int dtype_ivsn = 0;
static int new_dtype(int);

static void new_asts(void);
static int new_ast(int);
static void new_stds(void);
static int new_std(int);
static int new_argt(int);
static int new_asd(int);
static int new_astli(int, int);
static int new_shape(int);

static int new_symbol(int);
static int new_symbol_if_module(int old_sptr);
static void new_symbol_and_link(int, int *, SYMITEM **);
static void fill_links_symbol(SYMITEM *, WantPrivates);
static int can_find_symbol(int);
static int can_find_dtype(int);
#ifdef FLANG_INTERF_UNUSED
static SYMITEM *find_symbol(int);
static int common_conflict(void);
static int install_common(SYMITEM *, int);
static LOGICAL common_mem_eq(int, int);
#endif
static int new_installed_dtype(int old_dt);
static DITEM * finddthash(int old_dt);

static const char *import_file_name;
static void import_constant(SYMITEM *ps);
static void import_symbol(SYMITEM *ps);
static void import_ptr_constant(SYMITEM *ps);
static void import(lzhandle *fdlz, WantPrivates, int ivsn);
static int import_skip_use_stmts(lzhandle *fdlz);
static void import_done(lzhandle *, int nested);
static lzhandle *import_header_only(FILE *fd, const char *file_name,
                                    int import_which, int* ivsn_return);
static void get_component_init(lzhandle *, const char *, char *, int);

struct imported_modules_struct imported_modules = {NULL, 0, 0, 0, 0};

/** \brief Initialize import of module symbols, etc. */
void
import_init(void)
{
  imported_modules.avail = 0;
  if (imported_modules.size == 0) {
    imported_modules.size = 10;
    NEW(imported_modules.list, IMPORT_LIST, imported_modules.size);
  }
  exterf_init();
} /* import_init */

/** \brief Wrap-up module symbol import phase */
void
import_fini(void)
{
  FREE(imported_modules.list);
  imported_modules.avail = 0;
  imported_modules.size = 0;
} /* import_fini */

static void
add_imported(int modulesym)
{
  int il;
  char *modulename;

  modulename = SYMNAME(modulesym);
  Trace(("add %s to imported module list", modulename));
  il = imported_modules.avail++;
  NEED(il + 1, imported_modules.list, IMPORT_LIST, imported_modules.size,
       imported_modules.size + 10);
  imported_modules.list[il].modulesym = modulesym;
  imported_modules.list[il].modulename =
      (char *)getitem(PERM_AREA, strlen(modulename) + 1);
  strcpy(imported_modules.list[il].modulename, modulename);
} /* add_imported */

#undef READ_LINE
#define READ_LINE p = read_line(fd)
#define READ_LZLINE currp = p = ulz(fdlz)

static const char *import_corrupt_msg;
static const char *import_oldfile_msg;
static const char *import_incompatible_msg;

#define IMPORT_WHICH_PRELINK -1
#define IMPORT_WHICH_IPA -2
#define IMPORT_WHICH_INLINE -3
#define IMPORT_WHICH_HOST -4
#define IMPORT_WHICH_NESTED -5

static void
set_message(int import_which, const char *file_name)
{
  import_file_name = file_name;
  switch (import_which) {
  case IMPORT_WHICH_PRELINK:
    for_static = TRUE;
    import_corrupt_msg = "Corrupt Prelink file";
    import_oldfile_msg = "Corrupt Prelink file";
    import_incompatible_msg = "Corrupt Prelink file";
    break;
  case IMPORT_WHICH_IPA:
    for_interproc = TRUE;
    import_corrupt_msg = "Corrupt or Old IPA file";
    import_oldfile_msg = "Old IPA file";
    import_incompatible_msg = "Incompatible or Old IPA file";
    break;
  case IMPORT_WHICH_INLINE:
    for_inliner = TRUE;
    import_corrupt_msg = "Corrupt or Old Inline file";
    import_oldfile_msg = "Old Inline file";
    import_incompatible_msg = "Incompatible or Old Inline file";
    break;
  case IMPORT_WHICH_HOST:
    for_host = 1;
    import_corrupt_msg = "Corrupt";
    import_oldfile_msg = "Corrupt";
    import_incompatible_msg = "Corrupt";
    break;
  default: /* must be for a module */
    for_module = import_which;
    import_corrupt_msg = "Corrupt or Old Module file";
    import_oldfile_msg = "Old Module file";
    import_incompatible_msg = "Incompatible Module file";
  }
} /* set_message */

/**
 * The USE processing builds a 'use-graph'; each module that is directly or
 * indirectly used is a node in the graph, with an edge to all the modules used
 * by that node.  If a module is used publicly and privately, or used with
 * different 'except' lists, each is a different node in the graph.  The USE
 * graph must be acyclic (this is checked).  Each module file is read in once
 * for each node in the graph (this could be optimized with some work to be once
 * for each module, but we expect the effect to be small).  A depth-first
 * traversal of the graph gives a topological order of the nodes, giving a legal
 * order in which to import the USES_LIST is the list of modules used by this
 * module
 */
typedef struct uses_list {
  struct uses_list *next;
  struct to_be_imported *use_module;
  int directlyused;
} USES_LIST;

typedef struct to_be_imported {
  struct to_be_imported *prev; /* used only by the to_be_used_list_head list */
  struct to_be_imported *next; /* used only by the to_be_used_list_head list */
  struct to_be_imported
      *order; /* used only by the to_be_used_list_order_head list */
  USES_LIST *uses;
  LOGICAL public;
  int exceptlist;
  char *modulename;
  char *modulefilename;
  char *fullfilename;
  LOGICAL visited;
  int sl;
} TOBE_IMPORTED_LIST;

static TOBE_IMPORTED_LIST *to_be_used_list_head, *to_be_used_list_tail;
static TOBE_IMPORTED_LIST *to_be_used_list_order_head,
    *to_be_used_list_order_tail;
static USES_LIST *use_tree = NULL, /* this list is the root of the use_tree */
    *use_tree_end = NULL;

static int import_use_stmts(lzhandle *fdlz, TOBE_IMPORTED_LIST *, const char *, int,
                            int);
static int get_module_file_name_from_user(TOBE_IMPORTED_LIST *il,
                                          const char *from_file_name);

static void
dump_list_node(USES_LIST *node, int indent)
{
  int i;
  USES_LIST *n;

  for (n = node; n; n = n->next) {
    if (!n->directlyused) {
      continue;
    }
    for (i = indent; i; i--)
      printf("  ");
    printf("module %s[%p]: exceptlist %d, sl %d", n->use_module->modulename,
           n->use_module, n->use_module->exceptlist, n->use_module->sl);
    if (n->use_module->public && n->use_module->uses) {
      printf(", uses:\n");
      dump_list_node(n->use_module->uses, indent + 1);
    } else {
      printf(":\n");
    }
  }
}

void
dump_use_tree(void)
{
  if (use_tree) {
    printf("USE TREE:\n");
    dump_list_node(use_tree, 1);
  }
}

void
init_use_tree(void)
{
  use_tree = use_tree_end = 0;
}

void remove_from_use_tree(char *module)
{
  USES_LIST *prev = use_tree;
  for (USES_LIST *n = use_tree; n; n = n->next) {
    if (strcmp(n->use_module->modulename, module) == 0) {
      if (n == use_tree) {
        use_tree = use_tree->next;
      } else {
        prev->next = n->next;
      }
    }
    prev = n;
  }
}

static TOBE_IMPORTED_LIST *
already_to_be_used(char *modulename, int public, int except)
{
  TOBE_IMPORTED_LIST *il;
  for (il = to_be_used_list_head; il; il = il->next) {
    if (il->public == public && strcmp(modulename, il->modulename) == 0) {
      if (except == 0 && il->exceptlist == 0) {
        return il;
      } else if (except != 0 && il->exceptlist != 0) {
        if (same_sym_list(except, il->exceptlist)) {
          return il;
        }
      }
    }
  }
  return NULL;
} /* already_to_be_used */

/** \brief Clear visited nodes in the module use_tree.
  *
  * Intended to operate on the use_tree only.  Does not follow indirect links
  * nor does it recurse through private modules uses.
  */
static void
clear_list_nodes_visited(USES_LIST *list)
{
  USES_LIST *l;
  for (l = list; l; l = l->next) {
    if (!l->directlyused) {
      continue;
    }
    l->use_module->visited = 0;
    if (l->use_module->public && l->use_module->uses) {
      clear_list_nodes_visited(l->use_module->uses);
    }
  }
}

static USES_LIST *
find_next_modname_in_list(char *name, USES_LIST *list)
{
  USES_LIST *l;

  for (l = list; l; l = l->next) {
    if (strcmp(l->use_module->modulename, name) == 0) {
      return l;
    }
  }
  return NULL;
}

/** \brief  Search list on name (only)
  *
  * Use ONLY if it is guaranteed that the module can be on the list only once or
  * the first item on the list is the desired item.
  */
static TOBE_IMPORTED_LIST *
find_modname_in_list(char *name, USES_LIST *list)
{
  USES_LIST *l;

  for (l = list; l; l = l->next) {
    if (l->use_module->public && strcmp(l->use_module->modulename, name) == 0) {
      return l->use_module;
    }
  }
  return NULL;
}

/** \brief Search list for use node using name and exception list */
static TOBE_IMPORTED_LIST *
find_use_node_in_list(char *name, int exception, USES_LIST *ul)
{
  USES_LIST *mun;

  for (mun = ul; mun; mun = mun->next) {
    if (mun->use_module->public &&
        strcmp(mun->use_module->modulename, name) == 0 &&
        (mun->use_module->exceptlist == exception ||
         same_sym_list(mun->use_module->exceptlist, exception))) {
      break;
    }
  }
  return (mun ? mun->use_module : NULL);
}

LOGICAL
imported_directly(char *name, int except)
{

  return (find_use_node_in_list(name, except, use_tree) != NULL);
}

/** \brief Add a TOBE_IMPORTED_LIST item to the end of the use_tree list */
static void
add_to_use_tree(TOBE_IMPORTED_LIST *um)
{
  USES_LIST *ul = (USES_LIST *)getitem(MOD_USE_AREA, sizeof(USES_LIST));
  ul->use_module = um;
  ul->directlyused = 1;
  ul->next = NULL;

  if (!use_tree) {
    use_tree = ul;
  } else {
    use_tree_end->next = ul;
  }
  use_tree_end = ul;
}

/** \brief Create and insert a TOBE_IMPORTED_LIST item into the beginning of
 *  the use_tree
 */
static TOBE_IMPORTED_LIST *
insert_node_into_use_tree(char *modulename)
{
  TOBE_IMPORTED_LIST *il;
  USES_LIST *ul;

#if DEBUG
  if (DBGBIT(0, 0x10000))
    fprintf(gbl.dbgfil, "add_node_to_use_tree( %s )\n", modulename);
#endif

  ul = (USES_LIST *)getitem(MOD_USE_AREA, sizeof(USES_LIST));
  if (!use_tree) {
    use_tree = ul;
    use_tree->next = NULL;
    use_tree_end = ul;
  } else {
    ul->next = use_tree;
  }
  use_tree = ul;
  ul->directlyused = 1;

  il = (TOBE_IMPORTED_LIST *)getitem(MOD_USE_AREA, sizeof(TOBE_IMPORTED_LIST));
  ul->use_module = il;

  il->prev = NULL;
  il->next = NULL;
  il->uses = NULL;
  il->order = NULL;
  il->public = 1;
  il->exceptlist = 0;  /* will be set after "apply_use" rename processing */
  il->visited = FALSE; /* initialize for depth-first search */
  il->modulename = (char *)getitem(MOD_USE_AREA, strlen(modulename) + 1);
  strcpy(il->modulename, modulename);
  il->modulefilename = NULL;
  il->fullfilename = NULL;
  il->sl = 0;

  return il;
}

/** \brief Add il into curr_use_list after *curr_use_list item */
static USES_LIST **
add_use_tree_uses(USES_LIST **curr_use_list, TOBE_IMPORTED_LIST *il)
{
  USES_LIST *ul = (USES_LIST *)getitem(MOD_USE_AREA, sizeof(USES_LIST));
  USES_LIST *tmp;

  ul->directlyused = 1;
  if (!*curr_use_list) {
    *curr_use_list = ul;
    ul->next = NULL;
  } else {
    tmp = (*curr_use_list)->next;
    ul->next = tmp;
    (*curr_use_list)->next = ul;
  }
  ul->use_module = il;
  return &(ul->next);
}

static TOBE_IMPORTED_LIST *
add_to_be_used_list(char *modulename, int public, int except,
                    TOBE_IMPORTED_LIST *ilfrom, const char *from_file_name)
{
  TOBE_IMPORTED_LIST *il;
  il = already_to_be_used(modulename, public, except);
  if (il)
    return il;
#if DEBUG
  if (DBGBIT(0, 0x10000))
    fprintf(gbl.dbgfil, "add_to_be_used_list( %s, %d )\n", modulename, public);
#endif
  il = (TOBE_IMPORTED_LIST *)getitem(MOD_USE_AREA, sizeof(TOBE_IMPORTED_LIST));
  il->prev = to_be_used_list_tail;
  il->next = NULL;
  il->uses = NULL;
  il->order = NULL;
  il->public = public; /* save 'public' bit here */
  il->exceptlist = except;
  il->visited = FALSE;
  il->sl = 0;
  il->modulename = (char *)getitem(MOD_USE_AREA, strlen(modulename) + 1);
  strcpy(il->modulename, modulename);
  if (from_file_name) {
    il->modulefilename = (char *)getitem(
        MOD_USE_AREA, strlen(il->modulename) + strlen(MOD_SUFFIX) + 1);
    il->fullfilename = (char *)getitem(MOD_USE_AREA, MAX_FNAME_LEN + 1);
    strcpy(il->modulefilename, il->modulename);
    convert_2dollar_signs_to_hyphen(il->modulefilename);
    strcat(il->modulefilename, MOD_SUFFIX);
    if (!get_module_file_name_from_user(il, from_file_name)) {  
      if (!get_module_file_name(il->modulefilename, il->fullfilename,
                                MAX_FNAME_LEN)) {
        error(4, 0, gbl.lineno, "Unable to open MODULE file",
              il->modulefilename);
        return NULL;
      }
    }
  }
  if (to_be_used_list_tail == NULL) {
    to_be_used_list_head = il;
  } else {
    to_be_used_list_tail->next = il;
  }
  to_be_used_list_tail = il;

  if (public && strcmp(modulename, "iso_c_binding") == 0) {
    add_isoc_intrinsics();
  }

  return il;
} /* add_to_be_used_list */

static int
alreadyused(char *modulename)
{
  int il, sptr;
  for (il = 0; il < imported_modules.avail; ++il) {
    if (strcmp(modulename, imported_modules.list[il].modulename) == 0) {
      return imported_modules.list[il].modulesym;
    }
  }
  /* if we are compiling a routine contained in a module,
   * see if the module name is the same as the module we are compiling */
  for (sptr = gbl.currsub; sptr; sptr = SCOPEG(sptr)) {
    if (STYPEG(sptr) == ST_MODULE) {
      if (strcmp(modulename, SYMNAME(sptr)) == 0) {
        return sptr;
      }
    }
    if (SCOPEG(sptr) == sptr)
      break; /* prevent infinite loop */
  }
  return 0;
} /* alreadyused */

static void
add_use_edge(TOBE_IMPORTED_LIST *ilfrom, TOBE_IMPORTED_LIST *il,
             int directlyused)
{
  USES_LIST *ul;
  if (ilfrom == NULL)
    return;
  for (ul = ilfrom->uses; ul; ul = ul->next) {
    if (ul->use_module == il)
      return;
  }
  ul = (USES_LIST *)getitem(MOD_USE_AREA, sizeof(USES_LIST));
  ul->next = ilfrom->uses;
  ul->use_module = il;
  ul->directlyused = directlyused;
  ilfrom->uses = ul;
} /* add_use_edge */

static void
update_list_exceptions(USES_LIST *list)
{
  USES_LIST *l;

  for (l = list; l; l = l->next) {
    if (!l->directlyused || !l->use_module->public) {
      continue;
    }
    if (l->use_module->sl >= sem.scope_size) {
      /* FS#22824: should not be indexing above sem.scope_level.
         When it's >= sem.scope_size it would access random memory.
      interr("bad saved scope level", l->use_module->sl, ERR_Severe);
      */
    } else {
      l->use_module->exceptlist = get_scope(l->use_module->sl)->except;
    }
    update_list_exceptions(l->use_module->uses);
  }
}

void
update_use_tree_exceptions(void)
{
  update_list_exceptions(use_tree);
}

static int
get_module_file_name_from_user(TOBE_IMPORTED_LIST *il, const char *from_file_name)
{
  char *chfrom, *chfromdup, *slash, saveslash;
  chfromdup = strdup(from_file_name);

  /* try the directory from from_file_name */
  slash = NULL;
  for (chfrom = chfromdup; *chfrom; ++chfrom) {
    if (*chfrom == '/')
      slash = chfrom;
#ifdef HOST_WIN
    if (*chfrom == '\\')
      slash = chfrom;
#endif
  }
  if (slash) {
    saveslash = *slash;
    *slash = '\0'; /* have a directory, terminate the string */
    if (fndpath(il->modulefilename, il->fullfilename, MAX_FNAME_LEN,
                chfromdup) == 0) {
      *slash = saveslash;
      FREE(chfromdup);
      return 1;
    }
    *slash = saveslash;
  }
  FREE(chfromdup);
  return 0;
} /* get_module_file_name_from_user */

int
get_module_file_name(char *module_file_name, char *full_file_name, int len)
{
  if (module_directory_list) {
    moddir_list *ml;
    for (ml = module_directory_list; ml; ml = ml->next) {
      if (fndpath(module_file_name, full_file_name, len,
                  ml->module_directory) == 0) {
        return 1;
      }
    }
  } else {
    /* look in current directory before include directories */
    if (fndpath(module_file_name, full_file_name, len, ".") == 0) {
      return 1;
    }
  }
  if (flg.idir) {
    int i;
    char *chp;
    for (i = 0; (chp = flg.idir[i]); ++i) {
      if (fndpath(module_file_name, full_file_name, len, chp) == 0) {
        return 1;
      }
    }
  }
  if (fndpath(module_file_name, full_file_name, len, DIRWORK) == 0) {
    return 1;
  }
  if (flg.stdinc == 0) {
    if (fndpath(module_file_name, full_file_name, len, DIRSINCS) == 0) {
      return 1;
    }
  } else if (flg.stdinc != (char *)1) {
    if (fndpath(module_file_name, full_file_name, len, flg.stdinc) == 0) {
      return 1;
    }
  }
  return 0;
} /* get_module_file_name */

static void
topsort(TOBE_IMPORTED_LIST *il)
{
  USES_LIST *ul;
  TOBE_IMPORTED_LIST *ulil;

  il->visited = TRUE;
  for (ul = il->uses; ul; ul = ul->next) {
    ulil = ul->use_module;
    if (ulil->visited == FALSE) {
      topsort(ulil);
    }
  }
  if (to_be_used_list_order_tail == NULL) {
    to_be_used_list_order_head = il;
  } else {
    to_be_used_list_order_tail->order = il;
  }
  to_be_used_list_order_tail = il;
} /* topsort */

static int
sym_visible_in_scope(USES_LIST *list, int sptrsym, char *symscopenm)
{
  USES_LIST *l;
  int scopesptr = 0;
  int sl;
  SCOPESTACK *scope;

  /* breath first search of USE tree */
  l = list;
  while ((l = find_next_modname_in_list(symscopenm, l))) {
    TOBE_IMPORTED_LIST *n = l->use_module;
    if (n->visited) {
      l = l->next;
      continue;
    }

    sl = n->sl;
    if (!sl || sl >= sem.scope_level) {
      l = l->next;
      continue;
    }

    /*
     * This use_tree node ("l") is an instance of the the sptrsym's defining
     * module and
     * nothing above it (no USEing scope) has aliased (renamed) it. If this
     * scope is public
     * and symsptr is not on the exception list or if this scope is private and
     * symsptr in on
     * the only list, then this module/scope provides a visible instance of
     * sptrsym.
     */
    scope = get_scope(sl);
    if (!is_except_in_scope(scope, sptrsym) &&
        !is_private_in_scope(scope, sptrsym)) {
      return scope->sptr;
    }
    l = l->next;
  }

  /* not found, recurse through the rest of the use_tree looking for aliases of
   * sptrsym */
  for (l = list; l; l = l->next) {
    int symavl, sptrloop;
    LOGICAL hidden;
    if (!l->directlyused) {
      continue;
    }
    sl = l->use_module->sl;
    if (l->use_module->visited || !sl || sl >= sem.scope_level) {
      continue;
    }
    l->use_module->visited = 1;
    hidden = FALSE;
    symavl =
        sl == sem.scope_level - 1 ? stb.stg_avail : sem.scope_stack[sl + 1].symavl;

    for (sptrloop = sem.scope_stack[sl].symavl; sptrloop < symavl; sptrloop++) {
      /* if an alias of sptrsym is found in the the current (sl) scope,
       * then the symbol is hidden */
      if (STYPEG(sptrloop) == ST_ALIAS && SYMLKG(sptrloop) == sptrsym &&
          SCOPEG(sptrloop) == sem.scope_stack[sl].sptr) {
        hidden = TRUE;
        break; /* out of the alias search loop */
      }
    }
    if (hidden)
      continue; /* with the next item on list */

    if (l->use_module->public &&
        (scopesptr =
             sym_visible_in_scope(l->use_module->uses, sptrsym, symscopenm))) {
      break;
    }
  }

  return scopesptr;
}

static LOGICAL
scope_in_scope_stack(int sptr)
{
  SCOPESTACK *iface_scope =
      sem.interface ? next_scope_kind(curr_scope(), SCOPE_INTERFACE) : 0;
  SCOPESTACK *sptr_scope = next_scope_sptr(curr_scope(), sptr);
  return sptr_scope > iface_scope;
}

/** \brief Determine if there is a path through the USEs that makes
 * the symbol aliased by sptralias visible.
 *
 * If so, return the sptr of the module containing the symbol definition or, in
 * some cases, the sptr of the module containing the exposing alias.
 * Otherwise return 0.
 */
int
aliased_sym_visible(int sptralias)
{
  int sptrsym = SYMLKG(sptralias);
  int scopesym = 0;
  int sptr;

  if (STYPEG(sptrsym) != ST_PROC) {
    scopesym = SCOPEG(sptrsym);
    sptr = sptrsym;
  } else {
    scopesym = ENCLFUNCG(sptrsym);
    sptr = sptralias;
  }

  if (!scope_in_scope_stack(scopesym)) {
    if (!PRIVATEG(sptralias)) {
      /* happens when the the original symbol has been indirectly use associate
       * through
       * a module the exposes the symbol with a "USE ..., ONLY:" clause.
       */
      return SCOPEG(sptralias);
    } else {
      return 0;
    }
  }

  clear_list_nodes_visited(use_tree);
  return sym_visible_in_scope(use_tree, sptr, SYMNAME(scopesym));
}

static LOGICAL
alias_may_need_adjustment(int sptralias, int currmod)
{
  int sptrsym = SYMLKG(sptralias);
  int scopesym = 0;
  int scopealias = SCOPEG(sptralias);

  if (STYPEG(sptrsym) != ST_PROC) {
    scopesym = SCOPEG(sptrsym);
  } else {
    scopesym = ENCLFUNCG(sptrsym);
  }

  /*
     don't check normal contain'd subroutine/function aliases unless it is
     the module being processed
   */
  if (scopealias != currmod &&
      (STYPEG(sptrsym) == ST_PROC || STYPEG(sptrsym) == ST_ENTRY) &&
      scopesym == scopealias)
    return FALSE;

  /*
     If this func/subr alias is private and from a different scope than the
     func/subr (see
     previous if stmt), then this symbol is private in the alias scope.  Don't
     check.
  */
  if ((STYPEG(sptrsym) == ST_PROC || STYPEG(sptrsym) == ST_ENTRY) &&
      PRIVATEG(sptralias) && scopesym == scopealias)
    return FALSE;

  /* don't check compiler generated symbols */
  if (CCSYMG(sptrsym) || HCCSYMG(sptrsym) || CFUNCG(sptralias))
    return FALSE;

  /* don't check symbols not defined in a module */
  if (!scopesym || STYPEG(scopesym) != ST_MODULE)
    return FALSE;

  /* don't check aliases in the current module */
  if (scopealias == gbl.currmod)
    return FALSE;

  /* don't check private symbols from an included module */
  if (PRIVATEG(sptralias) && PRIVATEG(sptrsym) && scopesym == scopealias)
    return FALSE;

  /* don't check if in an interface block and the alias is PRIVATE */
  if (sem.interface && PRIVATEG(sptralias))
    return FALSE;

  return TRUE;
}

void
adjust_symbol_accessibility(int currmod)
{
  int sptr;
  int sptrmodscope;

  if (!use_tree) {
    return;
  }

  sptr = gbl.internal > 1 ? stb.firstosym : stb.firstusym;
  for (; sptr < stb.stg_avail; sptr++) {
    if (STYPEG(sptr) == ST_ALIAS && alias_may_need_adjustment(sptr, currmod)) {
      sptrmodscope = aliased_sym_visible(sptr);
      if (sptrmodscope) {
        HIDDENP(SYMLKG(sptr), 0);
        if (STYPEG(SYMLKG(sptr)) == ST_PROC) {
          PRIVATEP(SCOPEG(SYMLKG(sptr)), 0); /* fix-up proc's original alias */
        }
      } else {
        HIDDENP(SYMLKG(sptr), 1);
        if (STYPEG(SYMLKG(sptr)) == ST_PROC &&
            ENCLFUNCG(SYMLKG(sptr)) != currmod) {
          PRIVATEP(SCOPEG(SYMLKG(sptr)), 1); /* fix-up proc's original alias */
        }
      }
    }
  }
}

static void
do_nested_uses(WantPrivates wantPrivates)
{
  TOBE_IMPORTED_LIST *il;
  LOGICAL nested_in_host = FALSE;
  int saveBASEsym, saveBASEast, saveBASEdty;
  int sl, ivsn;
  int save_import_osym;

  if (for_module || for_host) {
    nested_in_host = TRUE;
    saveBASEsym = BASEsym;
    saveBASEast = BASEast;
    saveBASEdty = BASEdty;
    BASEsym = stb.firstusym;
    BASEast = astb.firstuast;
    BASEdty = DT_MAX;
  }

#if DEBUG
  if (DBGBIT(0, 0x10000))
    fprintf(gbl.dbgfil, "Enter do_nested_uses\n");
#endif

  if (to_be_used_list_head == NULL)
    return;

  for (il = to_be_used_list_head; il; il = il->next) {
    FILE *fd;
    lzhandle *fdlz;
    int i;
    /* open this file */
    if (il->modulefilename == NULL) {
      il->modulefilename = (char *)getitem(
          MOD_USE_AREA, strlen(il->modulename) + strlen(MOD_SUFFIX) + 1);
      il->fullfilename = (char *)getitem(MOD_USE_AREA, MAX_FNAME_LEN + 1);
      strcpy(il->modulefilename, il->modulename);
      strcat(il->modulefilename, MOD_SUFFIX);

      if (!get_module_file_name(il->modulefilename, il->fullfilename,
                                MAX_FNAME_LEN)) {
        error(4, 0, gbl.lineno, "Unable to open MODULE file",
              il->modulefilename);
        continue;
      }
    }

#if DEBUG
    if (DBGBIT(0, 0x10000))
      fprintf(gbl.dbgfil, "Open nested module file: %s\n", il->fullfilename);
#endif
    fd = fopen(il->fullfilename, "r");
    if (fd == NULL) {
      error(4, 0, gbl.lineno, "Unable to open MODULE file", il->modulefilename);
      continue;
    }
    save_import_osym = import_osym;
    fdlz = import_header_only(fd, il->modulefilename, 999, NULL);
    if (fdlz) {
      i = import_use_stmts(fdlz, il, il->fullfilename, IMPORT_WHICH_NESTED,
                           il->public);
    }
    import_osym = save_import_osym;
    import_done(fdlz, 1);
    fclose(fd);
  }

  /* create a topological sort of the modules */
  to_be_used_list_order_head = to_be_used_list_order_tail = NULL;
  for (il = to_be_used_list_head; il; il = il->next) {
    if (il->visited == FALSE) {
      topsort(il);
    }
  }

  for (il = to_be_used_list_order_head; il; il = il->order) {
    int modulesym;
    modulesym = alreadyused(il->modulename);
    if (modulesym) {
      if (il->public && nested_in_host) {
        sl = have_use_scope(modulesym);
        if (il->exceptlist != 0 || sl < top_scope_level) {
          int ex, base, s;
          ;
          /* public USE, no uses near enough open the module scope */
          save_scope_level();
          push_scope_level(modulesym, SCOPE_USE);
          il->sl = sem.scope_level;
          il->visited = FALSE;
          /* fill in the except list */
          curr_scope()->except = il->exceptlist;
          s = modulesym;
          if (STYPEG(s) == ST_ALIAS)
            s = SYMLKG(s);
          base = CMEMFG(s);
          for (ex = il->exceptlist; ex; ex = SYMI_NEXT(ex))
            SYMI_SPTR(ex) += base;
          restore_scope_level();
        } else if (sl >= top_scope_level) {
          il->sl = sl;
        }
      }
    } else {
      /* do the 'USE' */
      FILE *fd;
      lzhandle *fdlz;
      int module_sym, savescope;

#if DEBUG
      if (DBGBIT(0, 0x10000))
        fprintf(gbl.dbgfil, "Do nested use: %s\n", il->fullfilename);
#endif
      fd = fopen(il->fullfilename, "r");
      if (fd == NULL)
        continue;
      module_sym = import_mk_newsym(il->modulename, ST_MODULE);

      savescope = stb.curr_scope;
      if (nested_in_host) {
        save_scope_level();
        push_scope_level(module_sym, SCOPE_USE);
        curr_scope()->except = il->exceptlist;
        il->sl = sem.scope_level;
        il->visited = FALSE;
        restore_scope_level();
      }
      stb.curr_scope = module_sym;

      save_import_osym = import_osym;
      /* import_header_only function sets the original fortran source code */
      fdlz = import_header_only(fd, il->modulefilename, module_sym, &ivsn);
      if (fdlz) {
        if (import_skip_use_stmts(fdlz) == 0) {
          import(fdlz, wantPrivates, ivsn);
        }
      }
      import_osym = save_import_osym;
      import_done(fdlz, 1);
      fclose(fd);
      if (nested_in_host) {
        if (!(il->public) && save_import_osym != ANCESTORG(savescope)) {
          /* pop off the 'use' from the scope list */
          save_scope_level();
          pop_scope_level(SCOPE_USE);
          il->sl = 0;
          restore_scope_level();
        } else {
          int ex, base;
          int s = module_sym;
          if (STYPEG(s) == ST_ALIAS)
            s = SYMLKG(s);
          base = CMEMFG(s);
          for (ex = il->exceptlist; ex; ex = SYMI_NEXT(ex))
            SYMI_SPTR(ex) += base;
        }
      }
      /* restore curr_scope symbol */
      stb.curr_scope = savescope;
      add_imported(module_sym);
    }
  }
#if DEBUG
  if (DBGBIT(0, 0x10000))
    fprintf(gbl.dbgfil, "Exit do_nested_uses\n");
#endif
  if (for_module || for_host) {
    BASEsym = saveBASEsym;
    BASEast = saveBASEast;
    BASEdty = saveBASEdty;
  }
} /* do_nested_uses */

static lzhandle *
import_header_only(FILE *fd, const char *file_name, int import_which, int* ivsn_return)
{
  int j, compress;
  char *p;
  lzhandle *fdlz;
  int ivsn;
  int in_platform = 0;

#if DEBUG
  if (DBGBIT(0, 0x10000))
    fprintf(gbl.dbgfil, "import_header_only(%s)\n", file_name);
#endif
  import_errno = 0;
  if (buff == NULL) {
    buff_sz = BUFF_LEN;
    NEW(buff, char, buff_sz);
  }

  set_message(import_which, file_name);

  /* read the first line */
  READ_LINE; /* IVSN name-of-module */
  if (for_static && p == NULL) {
    import_errno = -1;
    FREE(buff);
    buff = 0;
    return NULL;
  }
  if (p == NULL) {
    error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    return NULL;
  }

  if (*currp != 'V') {
    error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    return NULL;
  }
  ++currp;
  /* get info from the first line */
  ivsn = get_num(10);
  if (ivsn < IVSN_24) {
    error(4, 3, gbl.lineno, import_oldfile_msg, import_file_name);
    error(4, 0, gbl.lineno, "Recompile source file", import_sourcename);
    return NULL;
  }
  if (ivsn_return)
    *ivsn_return = ivsn;
  if (ivsn == IVSN_24) {
    if (BASEdty == DT_MAX) {
      /* ivsn == 24 => before adding DT_DEFER[N]CHAR and increasing
       * the SYM flags & fields
       */
      BASEdty = DT_MAX_43; /* DT_MAX before adding DT_DEFER[N]CHAR */
    }
  }
  /*
   * Three cases:
   * o V25 was never released.
   * o V24 & V26 can procede wihout any additional checks.
   * o >= V27 needs further checks.
   */
  if (ivsn >= IVSN_27) {
    while (*currp == ' ')
      ++currp;
    if (ivsn > IVSN_27) {
      if (ivsn > IVSN) {
        /*
         *  old compiler reading new .mod file ??
         */
        error(4, 3, gbl.lineno, import_oldfile_msg, import_file_name);
        error(4, 0, gbl.lineno, "Recompile source file", import_sourcename);
        return NULL;
      }
      if (*currp != ':') {
        error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
        return NULL;
      }
      ++currp;
    }
    if (*currp != '0') {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      return NULL;
    }
    ++currp;
    if (*currp != 'x') {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      return NULL;
    }
    ++currp;
    if (XBIT(124, 0x10)) {
      curr_platform = curr_platform | MOD_I8;
    }
    if (XBIT(124, 0x8)) {
      curr_platform = curr_platform | MOD_R8;
    }
    if (XBIT(68, 0x1)) {
      curr_platform = curr_platform | MOD_LA;
    }

    in_platform = get_num(16);
#if DO_MODULE_OPTION_CHECK
    if (ivsn >= IVSN && curr_platform != in_platform) {
      if (!(in_platform & MOD_PG)) {
        if ((curr_platform | MOD_I8 | MOD_R8 | MOD_PG) != 
            (in_platform | MOD_I8 | MOD_R8 | MOD_PG)) {
          error(4, 3, gbl.lineno, import_incompatible_msg, import_file_name);
          error(4, 0, gbl.lineno,
                "Compile source file with the same compiler options",
                import_sourcename);
          return NULL;
        }
      }
    }
#endif
  }
  get_string(import_name);

  READ_LINE; /* source-filename-len source-filename firstosym compress */
  if (p == NULL) {
    error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    return NULL;
  }

  j = get_num(10); /* length of file name */
  if (import_sourcename == NULL || import_sourcename_len < j + 1) {
    import_sourcename =
        getitem(8, j + 1); /*area 8 is freed at the end of main()*/
  }
  if (j == 0) {
    import_sourcename[0] = '\0';
  } else {
    get_nstring(import_sourcename, j);
    /* put the file names in the fihb */
  }

  if (*currp == ' ' && *(currp + 1) == 'S') {
    ++currp;
    ++currp;
    import_osym = get_num(10);
    if (!can_map_initsym(import_osym)) {
      error(4, 3, gbl.lineno, import_incompatible_msg, import_file_name);
      error(4, 0, gbl.lineno, "Compile source file with the same compiler",
            import_sourcename);
      return NULL;
    }
  } else {
    error(4, 3, gbl.lineno, import_incompatible_msg, import_file_name);
    error(4, 0, gbl.lineno, "Compile source file with the same compiler",
          import_sourcename);
    return NULL;
  }
  compress = 0;
  if (*currp == ' ') {
    compress = get_num(10);
  }
  fdlz = ulzinit(fd, 0 /*compress*/);
  READ_LZLINE; /* time-date stamp */
  if (p == NULL) {
    error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    ulzfini(fdlz);
    return NULL;
  }
  return fdlz;
} /* import_header_only */

/* nested_public is usually one; it is zero if this is a nested use statement
 * where the module in which the use appears is itself private */
static int
import_use_stmts(lzhandle *fdlz, TOBE_IMPORTED_LIST *ilfrom,
                 const char *from_file_name, int import_which, int nested_public)
{
  char *p;
  TOBE_IMPORTED_LIST *il;
  TOBE_IMPORTED_LIST *n;
  USES_LIST **curr_use_list;

#if DEBUG
  if (DBGBIT(0, 0x10000))
    fprintf(gbl.dbgfil, "LOOK FOR USE STATEMENTS\n");
#endif
  if (import_which > 0) {
    n = insert_node_into_use_tree(SYMNAME(
        import_which)); /* add module from USE statement to the use tree root
                           list */
    curr_use_list = &(n->uses);
  }

  /* look for any 'use' statements */
  while (1) {
    char modulename[MAXIDLEN + 1], private[10], direct[10];
    int publicflag = 0;
    int directflag = 0;
    int ex = 0;
    int n;

    READ_LZLINE; /* "use module" or "enduse" */
    if (p == NULL) {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      return 1;
    }
    if (*currp == 'e') /* "enduse" */
      break;
    currp += 4; /* past 'use ' */
    get_string(modulename);
    get_string(private);
    publicflag = !strcmp(private, "public");
    /* get the 'except' list, if any */
    n = get_num(10);
    while (n--) {
      int s;
      s = get_num(10);
      ex = add_symitem(s, ex);
    }
    get_string(direct);
    directflag = !strcmp(direct, "direct");
    if (sem.scope_stack != NULL) {
      SCOPESTACK *scope = next_scope_kind_symname(0, SCOPE_MODULE, modulename);
      if (scope != 0) {
        error(500, 3, gbl.lineno, modulename, SYMNAME(import_which));
        if (import_which && directflag) {
          il = already_to_be_used(modulename, publicflag, scope->except);
          if (publicflag) {
            curr_use_list = add_use_tree_uses(curr_use_list, il);
            il->sl = get_scope_level(scope);
          }
        }
        continue; /* already processed this 'use' */
      }
    }
    if (!publicflag || !nested_public) {
      il = add_to_be_used_list(modulename, 0, 0, ilfrom, from_file_name);
    } else {
      il = add_to_be_used_list(modulename, 1, ex, ilfrom, from_file_name);
    }
    if (il) {
      add_use_edge(ilfrom, il, directflag);
      if (import_which > 0 && directflag && publicflag) {
        curr_use_list = add_use_tree_uses(curr_use_list, il);
      }
    }
    if (strcmp(modulename, "ieee_features") == 0) {
      sem.ieee_features = TRUE;
    }
  }

#if DEBUG
  if (DBGBIT(0, 0x10000))
    fprintf(gbl.dbgfil, "DONE ADDING USE STMTS\n");
#endif
  return 0;
} /* import_use_stmts */

/** \brief Look for any 'use' statements and skip over them */
static int
import_skip_use_stmts(lzhandle *fdlz)
{
  char *p;

  while (1) {
    READ_LZLINE; /* "use module" or "enduse" */
    if (p == NULL) {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      return 1;
    }
    if (*currp == 'e') /* "enduse" */
      break;
  }
  return 0;
} /* import_skip_use_stmts */

/** \brief Add .mod file to list of .mod files used */
static void
addmodfile(const char *filename)
{
  int m;
  for (m = 0; m < modinclistavl; ++m) {
    if (strcmp(filename, modinclist[m]) == 0)
      return; /* got it */
  }
  NEED(modinclistavl + 1, modinclist, char *, modinclistsize,
       modinclistsize + 40);
  /* allocate space to keep the name */
  modinclist[modinclistavl] = (char *)sccalloc(strlen(filename) + 1);
  strcpy(modinclist[modinclistavl], filename);
  ++modinclistavl;
} /* addmodfile */

static lzhandle *
import_header(FILE *fd, const char *file_name, int import_which, int* ivsn_return)
{
  lzhandle *fdlz;
  int i;
  int cur_findex_backup = 0;

  fdlz = import_header_only(fd, file_name, import_which, ivsn_return);
  if (fdlz == NULL)
    return fdlz;
  to_be_used_list_head = to_be_used_list_tail = NULL;
  i = import_use_stmts(fdlz, NULL, file_name, import_which, 1);
  if (i != 0) {
    ulzfini(fdlz);
    return NULL;
  }
  /* save curr_import_findex which will be likely changed in
   * function do_nested_uses. */
  cur_findex_backup = curr_import_findex;
  if (to_be_used_list_head != NULL) {
    /* do USEs of modules */
    do_nested_uses(INCLUDE_PRIVATES);
  }
  /* restore file index after do_nested_uses */
  curr_import_findex = cur_findex_backup;
  if (import_which > 0 && XBIT(123, 0x30000)) {
    TOBE_IMPORTED_LIST *il;
    if (modinclist == NULL) {
      modinclistsize = 40;
      modinclistavl = 0;
      NEW(modinclist, char *, modinclistsize);
    }
    /* do each of the nested uses */
    for (il = to_be_used_list_head; il; il = il->next) {
      addmodfile(il->fullfilename);
    }
    /* do this file */
    addmodfile(file_name);
  }
  /* have to do this again, in case any modules were imported between
   * then and now */
  set_message(import_which, file_name);
  return fdlz;
} /* import_header */

static void
import_done(lzhandle *fdlz, int nested)
{
  if (fdlz)
    ulzfini(fdlz);
  freearea(MOD_AREA);
  if (!nested) {
    if (buff) {
      FREE(buff);
      buff = 0;
    }
    freearea(MOD_USE_AREA);
    init_use_tree();
  }
  for_static = FALSE;
  for_interproc = FALSE;
  for_inliner = FALSE;
  for_module = 0;
  inmodulecontains = FALSE;
  import_corrupt_msg = NULL;
  import_oldfile_msg = NULL;
  import_incompatible_msg = NULL;
  import_file_name = NULL;
} /* import_done */

/** \brief Find a NMPTR that shares NMPTR for different symbols with the same
  * name.
  *
  * Note that putsname always inserts a new name into the name table.
  */
static int
find_nmptr(char *symname)
{
  int hash, hptr, len;
  len = strlen(symname);
  HASH_ID(hash, symname, len);
  for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
    if (strcmp(SYMNAME(hptr), symname) == 0) {
      return NMPTRG(hptr);
    }
  }
  return putsname(symname, len);
} /* find_nmptr */

#ifdef FLANG_INTERF_UNUSED
/** \brief Find a nmptr index for this name, then link this symbol into
 * the stb.hashtb hash links.
 */
static void
hash_link_name(int sptr, char *symname)
{
  int hash, hptr, len, nmptr;
  len = strlen(symname);
  HASH_ID(hash, symname, len);
  nmptr = 0;
  for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
    if (strcmp(SYMNAME(hptr), symname) == 0) {
      nmptr = NMPTRG(hptr);
      break;
    }
  }
  if (!nmptr)
    nmptr = putsname(symname, len);
  NMPTRP(sptr, nmptr);
  HASHLKP(sptr, stb.hashtb[hash]);
  stb.hashtb[hash] = sptr;
} /* hash_link_name */
#endif

static int
find_member_name(char *symname, int stype, int scopesym, int offset)
{
  int sptr, base;
  int hash, len;
  int dtype = 0;

  base = CMEMFG(scopesym);

  if (STYPEG(scopesym) == ST_TYPEDEF) {
    int scope;
    dtype = DTYPEG(scopesym);
    for (scope = SCOPEG(scopesym); scope; scope = SCOPEG(scope)) {
      scopesym = scope;
    }
  }

  if (base == 0 || scopesym == gbl.currmod || offset < 0) {
    /* check hash table */
    len = strlen(symname);
    HASH_ID(hash, symname, len);
    for (sptr = stb.hashtb[hash]; sptr; sptr = HASHLKG(sptr)) {
      if (STYPEG(sptr) == stype && strcmp(SYMNAME(sptr), symname) == 0) {
        int scope;
        for (scope = SCOPEG(sptr); scope; scope = SCOPEG(scope)) {
          if (dtype && (!CLASSG(sptr) && !VTABLEG(sptr))) {
            if (scope == scopesym && dtype == ENCLDTYPEG(sptr))
              return sptr;
          } else {
            if (scope == scopesym)
              return sptr;
          }
          if (SCOPEG(scope) == scope)
            break;
        }
        if (stype == ST_PROC && ENCLFUNCG(sptr) == scopesym) {
          /* when module A uses module B, the ST_PROC symbol for a function
           * Bfunc has its ENCLFUNC set to A.  Check the SCOPE of the ST_ALIAS
           */
          if (STYPEG(ENCLFUNCG(sptr)) == ST_MODULE) {
            if (STYPEG(SCOPEG(sptr)) == ST_ALIAS &&
                SCOPEG(SCOPEG(sptr)) != scope) {
              continue;
            }
          }
          return sptr;
        }
      }
    }
    return 0;
  }
  /* check first at 'base+offset' */
  sptr = base + offset;
  if (sptr >= stb.stg_avail)
    return 0;
  if (STYPEG(sptr) == stype && strcmp(SYMNAME(sptr), symname) == 0) {
    int scope;
    for (scope = SCOPEG(sptr); scope; scope = SCOPEG(scope)) {
      if (scope == scopesym)
        return sptr;
      if (SCOPEG(scope) == scope)
        break;
    }
  }
  for (sptr = base + offset - 10;
       sptr <= stb.stg_avail && sptr <= base + offset + 10; ++sptr) {
    if (STYPEG(sptr) == stype && strcmp(SYMNAME(sptr), symname) == 0) {
      int scope;
      for (scope = SCOPEG(sptr); scope; scope = SCOPEG(scope)) {
        if (scope == scopesym)
          return sptr;
        if (SCOPEG(scope) == scope)
          break;
      }
      if (stype == ST_PROC && ENCLFUNCG(sptr) == scopesym)
        return sptr;
    }
  }
  /* check hash table */
  len = strlen(symname);
  HASH_ID(hash, symname, len);
  for (sptr = stb.hashtb[hash]; sptr; sptr = HASHLKG(sptr)) {
    if (STYPEG(sptr) == stype && strcmp(SYMNAME(sptr), symname) == 0) {
      int scope;
      for (scope = SCOPEG(sptr); scope; scope = SCOPEG(scope)) {
        if (scope == scopesym)
          return sptr;
        if (SCOPEG(scope) == scope)
          break;
      }
    }
  }
  return 0;
} /* find_member_name */

static int module_base = 0; /* base symbol for modules */

/* The following routines manage the symbol_list hash table.
 * Note: Searching down the symbol_list is way way too expensive
 */
static void
inithash(void)
{
  int i;
  for (i = 0; i < SYMHASHSIZE; ++i) {
    symhash[i] = NULL;
  }
  for (i = 0; i < DTHASHSIZE; ++i) {
    dthash[i] = 0;
  }
} /* inithash */

static void
inserthash(int sptr, SYMITEM *ps)
{
  int h;
  h = sptr % SYMHASHSIZE;
  ps->hashnext = symhash[h];
  symhash[h] = ps;
} /* inserthash */

static SYMITEM *
findhash(int sptr)
{
  int h;
  SYMITEM *ps;
  h = sptr % SYMHASHSIZE;
  for (ps = symhash[h]; ps; ps = ps->hashnext) {
    if (ps->sptr == sptr)
      return ps;
  }
  return NULL;
} /* findhash */

static void
insertdthash(int old_dt, int d)
{
  int h;
  h = old_dt % DTHASHSIZE;
  dtz.base[d].hashnext = dthash[h];
  dthash[h] = d + 1; /* offset hash links by one, since zero is legal */
} /* insertdthash */

static DITEM *
finddthash(int old_dt)
{
  int h;
  int d;
  h = old_dt % SYMHASHSIZE;
  for (d = dthash[h]; d; d = dtz.base[d - 1].hashnext) {
    DITEM *pd;
    pd = dtz.base + (d - 1);
    if (pd->id == old_dt)
      return pd;
  }
  return NULL;
} /* finddthash */

/*
 * \brief Adjust type code for IVSN < 34
 * had inserted TY_HALF and TY_HCMPLX
 */
static int
adjust_pre34_dty(int dty)
{
  /* pre-half precision, adjust datatypes */
  if (dty < TY_HALF) {
    /* no changes */
  } else if (dty < TY_HCMPLX - 1) {
    /* TY_REAL to TY_QUAD increment by 1 to add TY_HALF */
    dty += 1;
  } else {
    /* increment by 2 to add TY_HALF and TY_HCMPLX */
    dty += 2;
  }
  return dty;
} /* adjust_pre34_dty */

static int
adjust_pre34_dtype(int dtype)
{
  /* pre-half precision, adjust datatypes */
  if (dtype < DT_REAL2) {
    /* no changes */
  } else if (dtype < DT_CMPLX4 - 1) {
    /* DT_REAL4 to DT_QUAD increment by 1 to add DT_REAL2 */
    dtype += 1;
  } else {
    /* increment by 2 to add DT_REAL2 and DT_CMPLX4 */
    dtype += 2;
  }
  return dtype;
} /* adjust_pre34_dtype */

static int original_symavl = 0;
static unsigned A_IDSTR_mask = (1 << 5); /* A_IDSTR is AST bit flag f5 */
static LOGICAL any_ptr_constant = FALSE;

/** \brief This is the main module import function.
  *
  * Below is the file format and order in which fields are read in.
<pre>
    version	Vnn modulename
    file	len filename firstosymbol compress
    date	mm/dd/yyyy  hh:mm:ss
    uses*	use modulename public/private exceptcount except*
  direct/indirect
    enduse
    astline	A astndx type flags shape hashlk w3 w4 w5 w6 w7 w8 w9 w10 hw21
  hw22 w12 opt1 opt2 repl visit w18
    container  C hostsptr hoststype
    datatype   D datatype ty_val [type-specific information]
    dtyper	d datatype sptrstype symboloffset modulename symbolname
    dtyper	e datatype sptrstype scopestype scopename symbolname
    equiv	E lineno sptr first substring subscript* 0
    formal	F functionsptr numargs arg*
    align	G alignptr target alignee aligntype alignsym ...
    shadow	H sptr dims
    overlap	L sptr overlapsptr
    mangled	M sptr nmangled [mangledsptr mangledmember]*
    overload	O sptr noverloaded overloaded*
    predecl	P sptr
    generic	Q sptr modproc* 0
    renamesym	R sptr stype modoffset modname symname
    sym	S sptr stype sc b3 b4 dtype symlk scope nmptr flags1 flags2 ...
    shape	T count [lwb upb stride]*
    distrib	U dist rank targettype isstar inherit formattype proc dynamic
  orig...
    std	V stdidx astidx label lineno flags
    argt	W nargs [arg]*
    asd	X ndim [subscript]*
    astli	Y [sptr triplet]* -1
    endsection	Z
 </pre>
 */
static void
import(lzhandle *fdlz, WantPrivates wantPrivates, int ivsn)
{
  char *p;
  int i, j;
  int sptr, nmlsptr, nmlline, nml, prevnml, ovlp;
  DITEM *pd;
  SYMITEM *ps, *qs, *previoussymbol;
  SYMITEM *last_symitem;
  ASTITEM *pa;
  int *pf;
  int evp, last_evp, first_evp;
  int dscptr;
  STDITEM *p_std;
  SHDITEM *p_shd;
  ARGTITEM *p_argt;
  ASDITEM *p_asd;
  ASTLIITEM *p_astli;
  int stringlen;
  int new_id, subtype, ndims, stype;
  int paramct, save_dtype_ivsn;
  char module_name[MAXIDLEN + 1], rename_name[MAXIDLEN + 1],
      idname[MAXIDLEN + 1], scope_name[MAXIDLEN + 1];
  int module_sym, scope_sym, rename_sym, offset, scope_stype;
  int hash;
  int first_ast;

  save_dtype_ivsn = dtype_ivsn;
  dtype_ivsn = ivsn;
  module_base = 0;
  original_symavl = stb.stg_avail;

  dtz.sz = 64;
  NEW(dtz.base, DITEM, dtz.sz);
  dtz.avl = 0;

  flz.sz = 64;
  NEW(flz.base, int, flz.sz);
  flz.avl = 0;

  ovz.sz = 64;
  NEW(ovz.base, int, ovz.sz);
  ovz.avl = 0;

  mdz.sz = 64;
  NEW(mdz.base, int, mdz.sz);
  mdz.avl = 0;

  astz.sz = 64;
  NEW(astz.base, ASTITEM, astz.sz);
  astz.avl = 0;
  BZERO(astzhash, int, ASTZHASHSIZE);

  stdz.sz = 64;
  NEW(stdz.base, STDITEM, stdz.sz);
  stdz.avl = 0;

  shdz.sz = 64;
  NEW(shdz.base, SHDITEM, shdz.sz);
  shdz.avl = 1;

  argtz.sz = 64;
  NEW(argtz.base, ARGTITEM, argtz.sz);
  BZERO(argtz.base + 0, ARGTITEM, 1); /* entry 0 is the empty arg table */
  argtz.base[0].installed = TRUE;
  argtz.avl = 1;

  asdz.sz = 64;
  NEW(asdz.base, ASDITEM, asdz.sz);
  asdz.avl = 0;

  astliz.sz = 64;
  NEW(astliz.base, ASTLIITEM, astliz.sz);
  astliz.avl = 0;

  modpz.sz = 64;
  NEW(modpz.base, MODPITEM, modpz.sz);
  modpz.avl = 0;

  /* add symbols to the end of the symbol_list so they get
   * added to the symbol table in the correct order.
   * allocate an item whose next field will become the head of the list.
   */
  symbol_list = last_symitem = (SYMITEM *)getitem(MOD_AREA, sizeof(SYMITEM));
  BZERO(symbol_list, SYMITEM, 1);
  last_symitem->next = NULL;
  inithash();

  align_list = NULL;
  dist_list = NULL;

  first_evp = last_evp = 0;

  previoussymbol = NULL;

  first_ast = 0;
  old_astversion = FALSE;

  while (1) { /*  read remainder of lines in file:  */
    READ_LZLINE;
#if DEBUG
    assert(p, "import: can't read line", 0, 4);
#else
    if (p == NULL)
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
#endif
    currp++;
    switch (p[0]) {
    case '-': /* info only line */
      break;

    case 'A': /* ast definition line */
      astz.avl++;
      NEED(astz.avl, astz.base, ASTITEM, astz.sz, astz.sz + 64);
      pa = astz.base + (astz.avl - 1);
      BZERO(pa, ASTITEM, 1);
      pa->old_ast = get_num(10);
      pa->type = get_num(10);
      pa->flags = get_num(16);
      pa->a.shape = get_num(10);
      pa->a.hshlk = get_num(10);
      pa->a.w3 = get_num(10);
      pa->a.w4 = get_num(10);
      pa->a.w5 = get_num(10);
      pa->a.w6 = get_num(10);
      pa->a.w7 = get_num(10);
      pa->a.w8 = get_num(10);
      pa->a.w9 = get_num(10);
      pa->a.w10 = get_num(10);
      pa->a.hw21 = get_num(10);
      pa->a.hw22 = get_num(10);
      pa->a.w12 = get_num(10);
      pa->a.opt1 = get_num(10);
      pa->a.opt2 = get_num(10);
      pa->a.repl = get_num(10);
      pa->a.visit = get_num(10);
      pa->a.w18 = get_num(10); /* IVSN 30 */
      pa->a.w19 = get_num(10);
      if (pa->flags & A_IDSTR_mask) {
        get_string(idname);
        sptr = getsymbol(idname);
        pa->a.w4 = sptr;
      }
      hash = pa->old_ast & ASTZHASHMASK;
      pa->link = astzhash[hash];
      astzhash[hash] = astz.avl;
      if (!first_ast) {
        if (astb.firstuast == 12 && pa->old_ast < 12) {
          /* older versions of the compiler reserved ASTs numbered
           * 1-9, inclusive; 10 was the first avail for user ASTs).
           */
          old_astversion = TRUE;
        }
        first_ast = pa->old_ast;
      }
      break;

    case 'B': /* iso_c intrinsic function ST_ISOC */
      sptr = get_num(10);
      assert(sptr < stb.firstusym, "Invalid sptr in mod file B record", 0,
             ERR_Fatal);
      get_string(module_name);
      if (strcmp(module_name, "iso_c_binding") == 0) {
        if (strcmp(SYMNAME(sptr), "c_sizeof") == 0) {
          STYPEP(sptr, ST_PD);
        } else {
          STYPEP(sptr, ST_INTRIN);
        }
      } else if (strcmp(module_name, "ieee_arithmetic") == 0) {
        STYPEP(sptr, ST_PD);
      } else if (strcmp(module_name, "iso_fortran_env") == 0) {
        STYPEP(sptr, ST_PD);
      }
      break;

    case 'C': /* containing subprogram symbol */
      sptr = get_num(10);
      stype = get_num(10);
      get_string(rename_name);
      /* look for this symbol */
      if (STYPEG(sptr) != stype || strcmp(rename_name, SYMNAME(sptr))) {
        interrf(ERR_Severe, "import: host program symbol %s (%d) not found!",
                rename_name, sptr);
        continue;
      }
      ps = (SYMITEM *)getitem(MOD_AREA, sizeof(SYMITEM));
      BZERO(ps, SYMITEM, 1);
      Trace(("Add host symbol %d to symbol list", sptr));
      last_symitem->next = ps;
      last_symitem = ps;
      ps->next = NULL;
      ps->sptr = sptr;
      ps->new_sptr = sptr;
      ps->sc = -1; /* don't change this symbol, already done */
      inserthash(sptr, ps);
      break;

    case 'D': /*  data type defn line  */
      dtz.avl++;
      NEED(dtz.avl, dtz.base, DITEM, dtz.sz, dtz.sz + 64);
      pd = dtz.base + (dtz.avl - 1);
      pd->dtypeinstalled = FALSE;

      pd->id = get_num(10);
      pd->ty = get_num(10);
      if (ivsn < 34)
        pd->ty = adjust_pre34_dty(pd->ty);
      insertdthash(pd->id, dtz.avl - 1);
      switch (pd->ty) {
      case TY_PTR:
        subtype = get_num(10);
        new_id = get_type(2, TY_PTR, subtype);
        break;
      case TY_ARRAY:
        subtype = get_num(10); /* array of dty */
        ndims = get_num(10);   /* ndims */
        if (ndims == 0) {
          new_id = get_type(3, TY_ARRAY, subtype);
          DTY(new_id + 2) = 0;
        } else {
          new_id = get_array_dtype(ndims, subtype);
          ADD_ZBASE(new_id) = get_num(10); /* zbase */
          ADD_NUMELM(new_id) = get_num(10);
          ADD_ASSUMSHP(new_id) = get_num(10);
          ADD_DEFER(new_id) = get_num(10);
          ADD_ADJARR(new_id) = get_num(10);
          ADD_ASSUMSZ(new_id) = get_num(10);
          ADD_NOBOUNDS(new_id) = get_num(10);
          for (i = 0; i < ndims; ++i) {
            READ_LZLINE;
#if DEBUG
            assert(p, "import: can't read arr line", 0, 4);
#else
            if (p == NULL)
              error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
#endif
            ADD_LWBD(new_id, i) = get_num(10);
            ADD_UPBD(new_id, i) = get_num(10);
            ADD_MLPYR(new_id, i) = get_num(10);
            ADD_LWAST(new_id, i) = get_num(10);
            ADD_UPAST(new_id, i) = get_num(10);
            ADD_EXTNTAST(new_id, i) = get_num(10);
          }
        }
        break;
      case TY_UNION:
      case TY_STRUCT:
      case TY_DERIVED:
        new_id = get_type(6, pd->ty, 0);
        DTY(new_id + 1) = get_num(10); /* (old) first member */
        DTY(new_id + 2) = get_num(10); /* size */
        DTY(new_id + 3) = get_num(10); /* (old) tag */
        DTY(new_id + 4) = get_num(10); /* align */
        DTY(new_id + 5) = 0;           /* ICT */
        break;
      case TY_CHAR:
      case TY_NCHAR:
        stringlen = get_num(10);
        new_id = get_type(2, TY_NONE, stringlen);
        /* use TY_NONE to avoid 'sharing' character data types */
        DTY(new_id) = pd->ty;
        break;
      case TY_PROC:
        subtype = get_num(10);
        new_id = get_type(6, TY_PROC, subtype);
        DTY(new_id + 2) = get_num(10); /* (old) interface */
        paramct = get_num(10);
        DTY(new_id + 3) = paramct;
        dscptr = ++aux.dpdsc_avl; /* one more for implicit arg */
        DTY(new_id + 4) = dscptr; /* dpdsc */
        aux.dpdsc_avl += paramct;
        NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
             aux.dpdsc_avl + 100);
        for (i = 0; i < paramct; i++) {
          aux.dpdsc_base[dscptr + i] = get_num(10); /* (old) arg */
        }
        DTY(new_id + 5) = get_num(10); /* (old) fval  */
        break;
      }
      /* new_id is already filled in! */
      pd->new_id = new_id;
      break;

    case 'd': /*  data type rename line, for 'used' derived types  */
      dtz.avl++;
      NEED(dtz.avl, dtz.base, DITEM, dtz.sz, dtz.sz + 64);
      pd = dtz.base + (dtz.avl - 1);
      pd->dtypeinstalled = FALSE;

      pd->id = get_num(10);
      insertdthash(pd->id, dtz.avl - 1);
      stype = get_num(10);
      offset = get_num(10);
      get_string(module_name);
      get_string(rename_name);
      /* look for a symbol with name 'rename' in a module with
       * name 'module_name' and STYPE of 'stype' */
      module_sym = findByNameStypeScope(module_name, ST_MODULE, 0);
      if (module_sym == 0) {
        interrf(ERR_Severe, "import: module %s: not found!", module_name);
        continue;
      }
      rename_sym = find_member_name(rename_name, stype, module_sym, offset);
      if (rename_sym == 0) {
        interrf(ERR_Severe,
          "import: module %s (%d,base=%d) member symbol %s (offset=%d): not found!",
          module_name, module_sym, CMEMFG(module_sym), rename_name, offset);
        continue;
      }
      new_id = DTYPEG(rename_sym);
      pd->ty = 0;
      pd->new_id = new_id;
      pd->dtypeinstalled = TRUE;
      break;

    case 'e': /*  data type rename line, for 'used' derived types  */
      dtz.avl++;
      NEED(dtz.avl, dtz.base, DITEM, dtz.sz, dtz.sz + 64);
      pd = dtz.base + (dtz.avl - 1);
      pd->dtypeinstalled = FALSE;

      pd->id = get_num(10);
      insertdthash(pd->id, dtz.avl - 1);
      stype = get_num(10);
      scope_stype = get_num(10);
      get_string(module_name);
      get_string(rename_name);
      /* look for a symbol with name 'rename' in a module with
       * name 'module_name' and STYPE of 'stype' */
      scope_sym = findByNameStypeScope(module_name, scope_stype, 0);
      if (scope_sym == 0) {
        interrf(ERR_Severe, "import: subprogram %s: not found!", module_name);
        continue;
      }
      rename_sym = findByNameStypeScope(rename_name, stype, scope_sym);
      if (rename_sym == 0) {
        interrf(ERR_Severe, "import: subprogram %s (%d) symbol %s: not found!",
                module_name, scope_sym, rename_name);
        continue;
      }
      new_id = DTYPEG(rename_sym);
      pd->ty = 0;
      pd->new_id = new_id;
      pd->dtypeinstalled = TRUE;
      break;

    case 'E': /* equivalence line */
      /* E private lineno sptr substring [subscripts] -1 */
      if (sem.interface == 0) {
        j = get_num(10); /* is it private */
        if (!ignore_private || wantPrivates == INCLUDE_PRIVATES || j == 0) {
          int ss, numss, ess;
          evp = sem.eqv_avail;
          ++sem.eqv_avail;
          NEED(sem.eqv_avail, sem.eqv_base, EQVV, sem.eqv_size,
               sem.eqv_size + 20);

          if (last_evp) {
            EQV(last_evp).next = evp;
          } else {
            first_evp = evp;
          }
          EQV(evp).next = 0;
          EQV(evp).ps = 0;
          EQV(evp).lineno = get_num(10);
          EQV(evp).sptr = get_num(10);
          /* set negative to avoid redoing it: */
          EQV(evp).is_first = -get_num(10);
          EQV(evp).byte_offset = 0;
          EQV(evp).substring = get_num(10);
          EQV(evp).subscripts = 0;
          numss = 0;
          do {
            ss = get_num(10);
            if (ss >= 0) {
              Trace(("Equivalence subscript for sym %d at ast %d",
                     EQV(evp).sptr, ss));
              if (numss == 0) {
                ess = sem.eqv_ss_avail;
                sem.eqv_ss_avail += 2;
                NEED(sem.eqv_ss_avail, sem.eqv_ss_base, int, sem.eqv_ss_size,
                     sem.eqv_ss_size + 50);
                EQV(evp).subscripts = ess;
                numss = 1;
              } else {
                ++sem.eqv_ss_avail;
                NEED(sem.eqv_ss_avail, sem.eqv_ss_base, int, sem.eqv_ss_size,
                     sem.eqv_ss_size + 50);
                ++numss;
              }
              EQV_NUMSS(ess) = numss;
              EQV_SS(ess, numss - 1) = ss;
            }
          } while (ss > 0);
          last_evp = evp;
        }
      }
      break;

    case 'F':             /* formal arguments: subprogram cnt arg ... */
      sptr = get_num(10); /* function sptr */
      i = get_num(10);    /* number of arguments */
      j = flz.avl;
      flz.avl += (i + 2);
      NEED(flz.avl, flz.base, int, flz.sz, flz.avl + 32);
      pf = flz.base + j;
      *pf++ = sptr;
      *pf++ = i;
      while (i--)
        *pf++ = get_num(10);
      break;

    case 'L': /* storage overlap list */
      sptr = get_num(10);
      if (previoussymbol && previoussymbol->sptr == sptr) {
        ps = previoussymbol; /* should be here */
      } else {
        ps = findhash(sptr);
      }
      /* overlap list line must follow symbol line */
      if (!ps)
        break;
      ps->socptr = soc.avail;
      ovlp = get_num(10);
      while (ovlp > 0) {
        /* link the list forwards */
        NEED(soc.avail + 1, soc.base, SOC_ITEM, soc.size, soc.size + 1000);
        SOC_SPTR(soc.avail) = ovlp;
        SOC_NEXT(soc.avail) = soc.avail + 1;
        ++soc.avail;
        ovlp = get_num(10);
      }
      /* unlink the last one entered */
      SOC_NEXT(soc.avail - 1) = 0;
      break;

    case 'm': /* mr for ACCROUT info, mt for DEVTYPE info */
      break;

    case 'M':             /* mangled derived symbols: derived cnt symbols ... */
      sptr = get_num(10); /* derived sptr */
                          /*
                           * get the number of mangled symbols - careful, the number of
                           * symbols present is actually three times this number.
                           */
      i = get_num(10);
      j = mdz.avl;
      mdz.avl += (MN_NENTRIES * i + 2);
      NEED(mdz.avl, mdz.base, int, mdz.sz, mdz.avl + 32);
      pf = mdz.base + j;
      *pf++ = sptr;
      *pf++ = i;
      while (i--) {
        READ_LZLINE;
        *pf++ = get_num(10); /* mangled - MN_SPTR */
        *pf++ = get_num(10); /* member  - MN_MEM */
      }
      break;

    case 'O': /* overloaded functions: generic/operator cnt function ... */
      sptr = get_num(10); /* generic sptr */
      i = get_num(10);    /* number of overloaded functions */
      j = ovz.avl;
      ovz.avl += (i + 2);
      NEED(ovz.avl, ovz.base, int, ovz.sz, ovz.avl + 32);
      pf = ovz.base + j;
      *pf++ = sptr;
      *pf++ = i;
      while (i--)
        *pf++ = get_num(10);
      break;

    case 'P': /* module predeclared (ST_HL, ST_HLL) is predeclared (ST_PD) */
      sptr = get_num(10);
      STYPEP(sptr, ST_PD);
      break;

    case 'Q':             /* module procedures: generic/operator ... */
      sptr = get_num(10); /* module procedure sptr */
      j = modpz.avl++;
      NEED(modpz.avl, modpz.base, MODPITEM, modpz.sz, modpz.avl + 32);
      modpz.base[j].modp = sptr;
      modpz.base[j].syml = 0;
      while (TRUE) {
        int s;
        s = get_num(10);
        if (s == 0)
          break;
        modpz.base[j].syml = add_symitem(s, modpz.base[j].syml);
      }
      break;

    case 'R': /*  symbol rename line, for 'used' modules */
      sptr = get_num(10);
      stype = get_num(10);
      offset = get_num(10) - 1;
      get_string(module_name);
      get_string(rename_name);
      get_string(scope_name);
      /* look for a symbol with name 'rename' in a module with
       * name 'module_name' and STYPE of 'stype' */
      module_sym = findByNameStypeScope(module_name, ST_MODULE, 0);
      if (module_sym == 0) {
        interrf(ERR_Severe, "import: module %s: not found!", module_name);
        continue;
      }
      if (offset < 0 && strlen(scope_name) != 0 &&
          strcmp(scope_name, ".") != 0) {
        int tsym = find_member_name(scope_name, ST_TYPEDEF, module_sym, -1);
        if (tsym) {
          module_sym = tsym;
        }
      }
      rename_sym = find_member_name(rename_name, stype, module_sym, offset);
      if (rename_sym == 0) {
        interrf(ERR_Severe,
          "import: module %s (%d,base=%d) member symbol %s (offset=%d): not found!",
          module_name, module_sym, CMEMFG(module_sym), rename_name, offset);
        continue;
      }
      ps = (SYMITEM *)getitem(MOD_AREA, sizeof(SYMITEM));
      BZERO(ps, SYMITEM, 1);
      Trace(("Add renamed symbol %d to symbol list", sptr));
      last_symitem->next = ps;
      last_symitem = ps;
      ps->next = NULL;
      ps->sptr = sptr;
      ps->new_sptr = rename_sym;
      ps->sc = -1; /* don't change this symbol, already done */
      inserthash(sptr, ps);
      break;

    case 'S': /*  symbol definition line */
      sptr = get_num(10);
      qs = findhash(sptr);
      if (qs) {
        Trace(("Symbol %d(%s) already in symbol list as %s\n", sptr, qs->name,
               stb.stypes[qs->stype]));
        if (qs->stype != ST_UNKNOWN)
          goto skip_sym;
        ps = qs;
      } else {
        ps = (SYMITEM *)getitem(MOD_AREA, sizeof(SYMITEM));
        BZERO(ps, SYMITEM, 1);
        Trace(("Add symbol %d to symbol list", sptr));
        last_symitem->next = ps;
        last_symitem = ps;
        ps->next = NULL;
        inserthash(sptr, ps);
      }
      previoussymbol = ps;
      ps->name[0] = '\0'; /* no name */
      ps->sptr = sptr;
      ps->stype = get_num(10);
      ps->sc = get_num(10);
      ps->sym.b3 = get_num(10);
      ps->sym.b4 = get_num(10);
      ps->dtype = get_num(10);
      ps->symlk = get_num(10);
      ps->sym.scope = get_num(10);
      ps->sym.nmptr = get_num(10);
      ps->sym.palign = get_num(10);
      ps->flags1 = get_num(16);
      ps->flags2 = get_num(16);

#undef GETFIELD
#define GETFIELD(f) ps->sym.f = get_num(10)
      if (currp[1] == 'A') {
        /*
         * New flags & fields were added for IVSN 26.  exterf prefixed
         * the new set of flags & fields with ' A'. So if ' A' is
         * present, read the new fields; otherwise, an old version of
         * the .mod file is being read.
         *
         * IVSN 26 flags & fields:
         */
        currp += 2; /* skip passed ' A' */
        ps->flags3 = get_num(16);
        GETFIELD(w34);
        GETFIELD(w35);
        GETFIELD(w36);
      }
      if (currp[1] == 'B') {
        /*
         * New flags & fields were added for IVSN 28.  exterf prefixed
         * the new set of flags & fields with ' B'. So if ' B' is
         * present, read the new fields; otherwise, an old version of
         * the .mod file is being read.
         *
         * IVSN 28 flags & fields:
         */
        currp += 2; /* skip passed ' B' */
        ps->flags4 = get_num(16);
        GETFIELD(lineno);
        GETFIELD(w39);
        GETFIELD(w40);
      }
      GETFIELD(w9);
      GETFIELD(w10);
      GETFIELD(w11);
      GETFIELD(w12);
      GETFIELD(w13);
      GETFIELD(w14);
      GETFIELD(w15);
      GETFIELD(w16);
      GETFIELD(w17);
      GETFIELD(w18);
      GETFIELD(w19);
      GETFIELD(w20);
      GETFIELD(w21);
      GETFIELD(w22);
      GETFIELD(w23);
      GETFIELD(w24);
      GETFIELD(w25);
      GETFIELD(w26);
      GETFIELD(w27);
      GETFIELD(w28);
      GETFIELD(uname);
      GETFIELD(w30);
      GETFIELD(w31);
      GETFIELD(w32);
#undef GETFIELD
      ps->new_sptr = 0;
      ps->strptr = NULL;

      Trace(("Importing symbol %d with stype %d", ps->sptr, ps->stype));

      /*  read additional tokens from line depending on symbol type: */

      switch (ps->stype) {
      case ST_CONST:
        ps->ty = get_num(10);
        if (ivsn < 34)
          ps->ty = adjust_pre34_dty(ps->ty);
        switch (ps->ty) {
        case TY_BINT:
        case TY_SINT:
        case TY_INT:
        case TY_INT8:
        case TY_BLOG:
        case TY_SLOG:
        case TY_LOG:
        case TY_LOG8:
        case TY_REAL:
        case TY_CMPLX:
        case TY_DBLE:
        case TY_QUAD:
        case TY_NCHAR:
        case TY_DCMPLX:
        case TY_QCMPLX:
          if (ps->sym.nmptr) {
            get_string(ps->name);
          }
          break;
        case TY_CHAR:
          stringlen = get_num(10);
          ps->strptr = (char *)getitem(MOD_AREA, stringlen + 1);
          for (i = 0; i < stringlen; i++)
            ps->strptr[i] = get_num(16);
          ps->strptr[i] = '\0';
          break;
        case TY_NONE:
          break;
        }
        break;

      case ST_UNKNOWN:
      case ST_IDENT:
      case ST_PARAM:
      case ST_PROC:
      case ST_MEMBER:
      case ST_STRUCT:
      case ST_VAR:
      case ST_ARRAY:
      case ST_DESCRIPTOR:
      case ST_CMBLK:
      case ST_ENTRY:
      case ST_ALIAS:
      case ST_ARRDSC:
      case ST_USERGENERIC:
      case ST_OPERATOR:
      case ST_TYPEDEF:
      case ST_STAG:
      case ST_MODULE:
      case ST_MODPROC:
      case ST_PLIST:
      case ST_LABEL:
      case ST_CONSTRUCT:
      case ST_BLOCK:
        get_string(ps->name);
        break;

      case ST_NML:
        get_string(ps->name);
        prevnml = 0;
        while (1) {
          READ_LZLINE;
          if (p == NULL || p[0] != 'N') {
            error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
          }
          currp++;
          nmlsptr = get_num(10);
          nmlline = get_num(10);
          if (nmlsptr < 0)
            break;
          nml = aux.nml_avl++;
          NEED(aux.nml_avl, aux.nml_base, NMLDSC, aux.nml_size,
               aux.nml_size + 100);
          NML_SPTR(nml) = nmlsptr;
          NML_NEXT(nml) = 0;
          NML_LINENO(nml) = nmlline;
          if (prevnml) {
            NML_NEXT(prevnml) = nml;
          } else {
            /* stash it */
            ps->ty = nml;
          }
          prevnml = nml;
        }
        break;

      default:
        interr("import:unrec stype", ps->stype, 3);
        i = 0;
      }
      if (exportb.hmark.maxsptr < ps->sptr)
        exportb.hmark.maxsptr = ps->sptr; /* max (old) sptr read */
    skip_sym:
      break;

    case 'T': /* shape record: cnt, <lwb : upb : stride> ... */
      i = shdz.avl++;
      NEED(shdz.avl, shdz.base, SHDITEM, shdz.sz, shdz.sz + 64);
      pa->shape = i;
      p_shd = shdz.base + i;
      j = get_num(10);
      p_shd->ndim = j;
      for (i = 0; i < j; i++) {
        p_shd->shp[i].lwb = get_num(10);
        p_shd->shp[i].upb = get_num(10);
        p_shd->shp[i].stride = get_num(10);
      }
      p_shd->new = 0;
      break;

    case 'V': /* std record:  std, ast, label, lineno, flags */
      i = stdz.avl++;
      NEED(stdz.avl, stdz.base, STDITEM, stdz.sz, stdz.sz + 64);
      p_std = stdz.base + i;
      p_std->old = get_num(10);
      p_std->ast = get_num(10);
      p_std->label = get_num(10);
      p_std->lineno = get_num(10);
      p_std->flags = get_num(16);
      p_std->new = i;
      break;

    case 'W': /* argt record:  cnt, args ... */
      i = argtz.avl++;
      NEED(argtz.avl, argtz.base, ARGTITEM, argtz.sz, argtz.sz + 64);
      /* ARGT record follows an AST record */
      pa->list = i;
      p_argt = argtz.base + i;
      p_argt->callfg = 0;
      j = get_num(10);
      /*
       * allocate the space for the argt; copy in the 'old' values
       * and we'll fix up when references occur.
       */
      p_argt->new = mk_argt(j);
      for (i = 0; i < j; i++)
        ARGT_ARG(p_argt->new, i) = get_num(10);
      p_argt->installed = FALSE;
      break;

    case 'X': /* asd record:  ndim, subs ... */
      i = asdz.avl++;
      NEED(asdz.avl, asdz.base, ASDITEM, asdz.sz, asdz.sz + 64);
      p_asd = asdz.base + i;
      p_asd->ndim = get_num(10);
      /* ASD record follows an AST record */
      pa->list = i;
      /*
       * stash the 'old' values of the subscripts and we'll fix up
       * when references occur.
       */
      for (i = 0; i < p_asd->ndim; i++)
        p_asd->subs[i] = get_num(10);
      p_asd->installed = FALSE;
      break;

    case 'Y': /* astli  record:
               *     <sptr, triple>... -1
               */
      i = astliz.avl++;
      NEED(astliz.avl, astliz.base, ASTLIITEM, astliz.sz, astliz.sz + 64);
      p_astli = astliz.base + i;
      /* ASTLI record follows an AST record */
      pa->list = i;
      /*
       * allocate the space for the astli; copy in the 'old' values
       * and we'll fix up when references occur.
       */
      start_astli();
      while (TRUE) {
        j = get_num(10);
        if (j < 0)
          break;
        i = add_astli();
        ASTLI_SPTR(i) = j; /* already got the damn thing */
        ASTLI_TRIPLE(i) = get_num(10);
        ASTLI_FLAGS(i) = 0;
      }
      p_astli->new = ASTLI_HEAD;
      p_astli->installed = FALSE;
      break;

    case 'Z': /*  EOF  */
      goto exit_loop;

    default:
      Trace(("unrecognized line in file %s: %s", import_file_name, p));
      interr("import: unrec line", p[0], 4);
    }
  }

exit_loop:
  symbol_list = symbol_list->next;

  any_ptr_constant = FALSE;
  if (for_module) {
    /* install all simple constant symbols first */
    for (ps = symbol_list; ps != NULL; ps = ps->next) {
      if (ps->stype == ST_CONST)
        import_constant(ps);
    }
    module_base = stb.stg_avail;
  }
  /* install all symbols */
  for (ps = symbol_list; ps != NULL; ps = ps->next) {
    if (!for_module && ps->stype == ST_CONST) {
      import_constant(ps);
    } else if (ps->new_sptr == 0 && ps->sc >= 0) {
      import_symbol(ps);
    }
    if (for_module && sem.scope_stack) {
      switch (ps->stype) {
      case ST_DESCRIPTOR:
        if (stb.curr_scope != curr_scope()->sptr &&
            DLLG(ps->new_sptr) == DLL_EXPORT && CLASSG(ps->new_sptr) &&
            SCG(ps->new_sptr) == SC_EXTERN) {
          /* import type descriptor */
          DLLP(ps->new_sptr, DLL_IMPORT);
        }
        break;
      case ST_ENTRY:
      case ST_PROC:
        if (stb.curr_scope != curr_scope()->sptr &&
            SCG(ps->new_sptr) == SC_EXTERN &&
            DLLG(ps->new_sptr) == DLL_EXPORT) {
          DLLP(ps->new_sptr, DLL_IMPORT);
        }
        FLANG_FALLTHROUGH;
      case ST_CMBLK:
        if (stb.curr_scope != curr_scope()->sptr &&
            DLLG(ps->new_sptr) == DLL_EXPORT) {
          DLLP(ps->new_sptr, DLL_IMPORT);
        }
        break;
      case ST_IDENT:
      case ST_VAR:
      case ST_ARRAY:
      case ST_STRUCT:
        if (stb.curr_scope != curr_scope()->sptr &&
            SCG(ps->new_sptr) == SC_CMBLK && DLLG(ps->new_sptr) == DLL_EXPORT) {
          DLLP(ps->new_sptr, DLL_IMPORT);
        }
        break;
      }
    }
  }
  if (for_module) {
    if (any_ptr_constant) {
      for (ps = symbol_list; ps != NULL; ps = ps->next) {
        if (ps->stype == ST_CONST)
          import_ptr_constant(ps);
      }
    }
    CMEMFP(for_module, module_base);
  }
  BZERO(stb.stg_base, SYM, 1);
  /* postprocess imported symbols */
  for (ps = symbol_list; ps != NULL; ps = ps->next) {
    if (ps->sc >= 0) {
      fill_links_symbol(ps, wantPrivates);
    }
  }

  new_dtypes();

  new_asts();
  old_astversion = FALSE;

  if (for_static) {
    /* read the data initialization info, change the symbol pointers,
     * and write it to the real data initialization file */
    while (1) {
      int ptype, lineno, anyivl, anyict;
      INT pcon;
      READ_LZLINE;
      if (p == NULL) {
        error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      }
      ++currp;
      if (p[0] == 'Z')
        break;
      switch (p[0]) {
      case 'I': /* initialization info */
        ptype = get_num(10);
        pcon = get_num(16);
        put_dinit_record(ptype, pcon);
        break;
      case 'J': /* data initialization file record */
        lineno = get_num(10);
        anyivl = get_num(10);
        anyict = get_num(10);
        put_data_statement(lineno, anyivl, anyict, fdlz, import_file_name, 0);
        break;
      case 'V': /* varref record */
      case 'W': /* varref subtype record */
      case 'D': /* dostart record */
      case 'E': /* doend record */
      case 'A': /* ict ast record */
      case 'S': /* ict subtype record */
      default:
        /* bad file */
        error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
        break;
      } /* switch p[0] */
    }   /* while */
  } else if (for_module) {
    /* read the data initialization info, change the symbol pointers,
     * and rebuild the IVL/ACL initialization structs  */
    while (1) {
      int ptype, lineno, anyivl, anyict;
      INT pcon;
      READ_LZLINE;
      if (p == NULL) {
        error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      }
      ++currp;
      if (p[0] == 'Z')
        break;
      switch (p[0]) {
      case 'I': /* initialization info */
                /* ignore the I entries, just consume them */
        ptype = get_num(10);
        pcon = get_num(16);
        break;
      case 'J': /* data initialization file record */
        lineno = get_num(10);
        anyivl = get_num(10);
        anyict = get_num(10);
        put_data_statement(lineno, anyivl, anyict, fdlz, import_file_name, 0);
        break;
      case 'T': /* derived type component init */
        get_component_init(fdlz, import_file_name, p, 0);
        break;
      case 'V': /* varref record */
      case 'W': /* varref subtype record */
      case 'D': /* dostart record */
      case 'E': /* doend record */
      case 'A': /* ict ast record */
      case 'S': /* ict subtype record */
      default:
        /* bad file */
        error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
        break;
      } /* switch p[0] */
    }   /* while */
  }

  /* fill in asts & symbols for equivalences; also, add equivalences to
   * semant's equivalence list.
   */
  for (evp = first_evp; evp != 0; evp = EQV(evp).next) {
    int ss, numss, j;
    EQV(evp).sptr = new_symbol(EQV(evp).sptr);
    if (EQV(evp).substring) {
      EQV(evp).substring = new_ast(EQV(evp).substring);
      EQV(evp)
          .byte_offset = get_int_cval(A_SPTRG(A_ALIASG(EQV(evp).substring)));
    }
    ss = EQV(evp).subscripts;
    if (ss > 0) {
      numss = EQV_NUMSS(ss);
      for (j = 0; j < numss; ++j) {
        EQV_SS(ss, j) = new_ast(EQV_SS(ss, j));
        Trace(("equivalence subscript for new symbol %d at new ast %d",
               EQV(evp).sptr, EQV_SS(ss, j)));
        /* subscripts must be a constant, or would have gotten an error */
        if (A_TYPEG(EQV_SS(ss, j)) != A_CNST) {
          Trace(("UNKNOWN EQUIVALENCE IMPORTED"));
        }
      }
    }
  }
  /* link in list of equivalences */
  if (first_evp) {
    EQV(last_evp).next = sem.eqvlist;
    sem.eqvlist = first_evp;
  }

  /* Fix up formal argument (parameter) descriptors.  Mimic the semantic
   * actions needed to create a subprogram/function ST_PROC within an
   * interface block; this implies that the arguments are created in another
   * scope.
   */
  pf = flz.base;
  for (j = 0; j < flz.avl;) {
    sptr = *pf++; /* function/subroutine (old) sptr */
    sptr = new_symbol(sptr);
    dscptr = ++aux.dpdsc_avl; /* one more for implicit argument */
    DPDSCP(sptr, dscptr);
    i = *pf++; /* # of arguments */
    PARAMCTP(sptr, i);
    aux.dpdsc_avl += i;
    NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
         aux.dpdsc_avl + 100);
    aux.dpdsc_base[dscptr - 1] = 0;
    j += (i + 2);
    while (i--) {
      int s;
      s = *pf++; /* argument (old) sptr */
      if (s) {
        s = new_symbol(s);
        {
          HIDDENP(s, 1);
          IGNOREP(s, 1);
        }
      }
      aux.dpdsc_base[dscptr++] = s;
    }
  }

  /* Fix up overloaded function (generic) descriptors.  */

  pf = ovz.base;
  for (j = 0; j < ovz.avl;) {
    int s;
    int oldcnt;
    sptr = *pf++; /* generic/operator (old) sptr */
    sptr = new_symbol(sptr);

    i = *pf++; /* # of functions */
    oldcnt = GNCNTG(sptr);
    GNCNTP(sptr, oldcnt + i);
    j += (i + 2);
    dscptr = GNDSCG(sptr);
    while (i--) {
      s = *pf++; /* function (old) sptr */
      s = new_symbol(s);
      if (STYPEG(s) == ST_MODPROC && SYMLKG(s))
        s = SYMLKG(s);
      if (STYPEG(s) == ST_ALIAS)
        s = SYMLKG(s);
      dscptr = add_symitem(s, dscptr);
    }
    GNDSCP(sptr, dscptr);
    if (STYPEG(sptr) == ST_USERGENERIC) {
      if (GSAMEG(sptr)) {
        s = new_symbol(GSAMEG(sptr));
        if (STYPEG(s) == ST_MODPROC && SYMLKG(s) > NOSYM)
          s = SYMLKG(s);
        if (STYPEG(s) == ST_ALIAS)
          s = SYMLKG(s);
        GSAMEP(sptr, s);
        GSAMEP(s, sptr);
      }
      if (GTYPEG(sptr)) {
        s = new_symbol(GTYPEG(sptr));
        GTYPEP(sptr, s);
      }
    }
  }

  /* Fix up module procedure descriptors.  */

  for (j = 0; j < modpz.avl; j++) {
    for (i = modpz.base[j].syml; i; i = SYMI_NEXT(i)) {
      int s;
      s = SYMI_SPTR(i);
      SYMI_SPTR(i) = new_symbol(s);
    }
    sptr = new_symbol(modpz.base[j].modp);
    SYMIP(sptr, modpz.base[j].syml);
  }

  /* fix up statements (stds) */

  if (!for_module && !for_host)
    new_stds();

  FREE(dtz.base);
  FREE(flz.base);
  FREE(ovz.base);
  FREE(mdz.base);
  FREE(astz.base);
  FREE(stdz.base);
  FREE(shdz.base);
  FREE(argtz.base);
  FREE(asdz.base);
  FREE(astliz.base);
  FREE(modpz.base);
  original_symavl = 0;

#if DEBUG
  if (DBGBIT(5, 0x800)) {
    fprintf(gbl.dbgfil, "****** After Importing %s ******\n", import_file_name);
    symdmp(gbl.dbgfil, 0);
  }
#endif
  dtype_ivsn = save_dtype_ivsn;
}

static char *
read_line(FILE *fd)
{
  int i;
  int ch;

  i = 0;
  while (TRUE) {
    ch = getc(fd);
    if (ch == EOF)
      return NULL;
    if (i + 1 >= buff_sz) {
      buff_sz += BUFF_LEN;
      buff = sccrelal(buff, buff_sz);
    }
    buff[i++] = ch;
    if (ch == '\n')
      break;
  }
  buff[i] = '\0';
  currp = buff;
  return buff;
}

int
import_inline(FILE *fd, char *file_name)
{
  int saveSym, saveAst, saveDty, saveCmblk, ivsn;
  lzhandle *fdlz;
  ADJmod = 0;
  BASEmod = 0;
  BASEsym = stb.firstusym;
  BASEast = astb.firstuast;
  BASEdty = DT_MAX;
  saveSym = stb.stg_avail;
  saveAst = astb.stg_avail;
  saveDty = stb.dt.stg_avail;
  saveCmblk = gbl.cmblks;
  fdlz = import_header(fd, file_name, IMPORT_WHICH_INLINE, &ivsn);
  if (fdlz) {
    import(fdlz, INCLUDE_PRIVATES, ivsn);
  }
  import_done(fdlz, 0);
  fclose(fd);
  if (import_errno) {
    stb.stg_avail = saveSym;
    astb.stg_avail = saveAst;
    stb.dt.stg_avail = saveDty;
    gbl.cmblks = saveCmblk;
  }
#if DEBUG
  if (DBGBIT(4, 16384) || DBGBIT(5, 16384)) {
    fprintf(gbl.dbgfil, "\n>>>>>> import_line begin\n");
    if (DBGBIT(4, 16384))
      dump_ast();
    if (DBGBIT(5, 16384)) {
      symdmp(gbl.dbgfil, DBGBIT(5, 8));
      dmp_dtype();
    }
    fprintf(gbl.dbgfil, "\n>>>>>> import_line end\n");
  }
#endif
  return import_errno;
}

/** \brief import a routine for static analysis */
int
import_static(FILE *fd, char *file_name)
{
  lzhandle *fdlz;
  int ivsn;
  ADJmod = 0;
  BASEmod = 0;
  BASEsym = stb.firstusym;
  BASEast = astb.firstuast;
  BASEdty = DT_MAX;
  fdlz = import_header(fd, file_name, IMPORT_WHICH_PRELINK, &ivsn);
  if (fdlz) {
    import(fdlz, INCLUDE_PRIVATES, ivsn);
  }
  import_done(fdlz, 0);
  return import_errno;
} /* import_static */

static int IPARECOMPILE = FALSE;

/** \brief This is called either for a USE statement, or to import the
  * specification part of a module for the contained subprograms.
  */
SPTR
import_module(FILE *fd, char *file_name, SPTR modsym, WantPrivates wantPrivates,
              int scope_level)
{
  SPTR modulesym = SPTR_NULL;
  lzhandle *fdlz;
  int savescope = stb.curr_scope, ivsn;
  ADJmod = 0;
  BASEmod = 0;
  BASEsym = stb.firstusym;
  BASEast = astb.firstuast;
  BASEdty = DT_MAX;
  top_scope_level = scope_level;
  /* for a USE statement, push the module scope between
   * the outer scope and its outer scope */
  /* We can't optimize away the 'import_header' even if the
    * module is already used; it may have been used with ONLY
    * or rename clauses, and the renaming clauses can give any
    * name in that module or in modules indirectly used.  The only
    * way we have to find which names can be used is to put the
    * directly and indirectly used modules on the scope stack */
  fdlz = import_header(fd, file_name, modsym, &ivsn);
  if (fdlz) {
    TOBE_IMPORTED_LIST *l;
    modulesym = 0;
    if (!IPARECOMPILE) {
      save_scope_level();
      modulesym = alreadyused(SYMNAME(modsym));
      if (modulesym != 0) {
        push_scope_level(modulesym, SCOPE_USE);
        l = find_modname_in_list(SYMNAME(modulesym), use_tree);
      } else {
        modsym = import_mk_newsym(SYMNAME(modsym), ST_MODULE);
        push_scope_level(modsym, SCOPE_USE);
        l = find_modname_in_list(SYMNAME(modsym), use_tree);
      }
      l->sl = sem.scope_level;
      restore_scope_level();
    }
    if (modulesym == 0) {
      /* set 'curr_scope' for symbols created when the module is imported */
      modulesym = modsym;
      stb.curr_scope = modulesym;
      import(fdlz, wantPrivates, ivsn);
      add_imported(modulesym);
      stb.curr_scope = savescope;
    }
  }
  import_done(fdlz, 1);

  top_scope_level = 0; /* restore to zero */
  return modulesym;
}

void
import_module_end(void)
{
}

void
import_module_print(void)
{
  if (modinclistavl > 0) {
    int m, c;
    char *fname;
    FILE *fp;
    if (!XBIT(123, 0x20000)) {
      fp = stdout;
    } else {
      char *f, *g;
      /* leave enough room for '.m' */
      fname = (char *)sccalloc(strlen(gbl.src_file) + 3);
      basenam(gbl.src_file, "", fname);
      g = NULL;
      for (f = fname; *f; ++f) {
        if (*f == '.')
          g = f;
      }
      if (!g)
        g = f; /* last char in string */
      *g++ = '.';
      *g++ = 'm';
      *g = '\0';
      fp = fopen(fname, "w");
      if (fp == NULL) {
        error(213, 4, 0, fname, CNULL);
      }
    }
    c = fprintf(fp, "%s :", gbl.src_file);
    for (m = 0; m < modinclistavl; ++m) {
      if (c + strlen(modinclist[m]) >= 80) {
        fprintf(fp, " \\\n   ");
        c = 3;
      }
      c += fprintf(fp, " %s", modinclist[m]);
    }
    fprintf(fp, "\n");
  }
}

/** \brief This is called to restore symbol table, etc., data for contained
 * subprograms.
 */
void
import_host(FILE *fd, const char *file_name, int oldsymavl, int oldastavl,
            int olddtyavl, int modbase, int moddiff, int oldscope, int newscope)
{
  lzhandle *fdlz;
  int ivsn;
  for_host = TRUE;
  /* push the a 'NORMAL' outer scope */
  ADJmod = moddiff;
  BASEmod = modbase;
  BASEsym = oldsymavl;
  BASEast = oldastavl;
  BASEdty = olddtyavl;
  HOST_OLDSCOPE = oldscope;
  HOST_NEWSCOPE = newscope;
  fdlz = import_header(fd, file_name, IMPORT_WHICH_HOST, &ivsn);
  if (fdlz) {
    import(fdlz, INCLUDE_PRIVATES, ivsn);
  }
  import_done(fdlz, 1);
  for_host = FALSE;
  HOST_OLDSCOPE = 0;
  HOST_NEWSCOPE = 0;
}

extern void export_fix_host_append_list(int (*)(int));

/** \brief This is called to restore symbol table, etc., data for contained
 * subprograms.
 */
void
import_host_subprogram(FILE *fd, const char *file_name, int oldsymavl, int oldastavl,
                       int olddtyavl, int modbase, int moddiff)
{
  lzhandle *fdlz;
  int ivsn;
  for_host = TRUE;
  /* push the a 'NORMAL' outer scope */
  ADJmod = moddiff;
  BASEmod = modbase;
  BASEsym = oldsymavl;
  BASEast = oldastavl;
  BASEdty = olddtyavl;
  fdlz = import_header(fd, file_name, IMPORT_WHICH_HOST, &ivsn);
  if (fdlz) {
    import(fdlz, INCLUDE_PRIVATES, ivsn);
  }
  export_fix_host_append_list(new_symbol);
  import_done(fdlz, 1);
  for_host = FALSE;
} /* import_host_subprogram */

static ISZ_T
get_num(int radix)
{
  char *chp;
  ISZ_T val = 0;
  INT num[2];

  while (*currp == ' ')
    currp++;
  if (*currp == '\n')
    return 0;
  chp = currp;
  while (*currp != ' ' && *currp != '\n' && *currp != '\0' && *currp != ':')
    currp++;
  /*
   * atoxi64  will 'fail' if it doesn't find a number in which case
   * num is not set; need to ensure that val remains 0.
   */
  if (atoxi64(chp, num, (int)(currp - chp), radix) >= 0) {
    INT64_2_ISZ(num, val);
  }
  return val;
}

static void
get_string(char *dest)
{
  int i;
  char ch;

  while (*currp == ' ')
    currp++;
  i = 0;
  while ((ch = *currp) != ' ' && ch != '\n' && ch != '\0') {
    dest[i++] = ch;
    currp++;
  }
  dest[i] = '\0';
}

/** \brief read 'len' characters */
static void
get_nstring(char *dest, int len)
{
  int i;
  char ch;

  while (*currp == ' ')
    currp++;
  i = 0;
  while (len-- && (ch = *currp) != '\n' && ch != '\0') {
    dest[i++] = ch;
    currp++;
  }
  dest[i] = '\0';
}

#ifdef FLANG_INTERF_UNUSED
static char *
getlstring(int area)
{
  char *p;
  int len;
  char *s;
  len = get_num(10);
  p = currp;
  if (*p != ':') {
    return NULL;
  }
  ++p;
  if (len == 0) {
    currp = p;
    return NULL;
  }
  s = getitem(area, len + 1);
  strncpy(s, p, len);
  s[len] = '\0';
  p += len;
  currp = p;
  return s;
} /* getlstring */
#endif

static int ipa_ast(int a);
static int dindex(int dtype);
static int get_symbolxref(int sptr);

/** \brief Change symbol number, if necessary, and write record to data init
  * file
  */
static void
put_dinit_record(int ptype, INT pcon)
{
  INT sptr;
  switch (ptype) {
  case DINIT_FMT: /* should not happen */
    break;

  case DINIT_END:      /* write this unchanged */
  case DINIT_ENDTYPE:  /* write this unchanged */
  case DINIT_STARTARY: /* write this unchanged */
  case DINIT_ENDARY:   /* write this unchanged */
  case 0:              /* write this unchanged */
  case DINIT_ZEROES:   /* write this unchanged */
  case DINIT_OFFSET:   /* write this unchanged */
  case DINIT_REPEAT:   /* write this unchanged */
    dinit_put(ptype, pcon);
    break;
  case DINIT_STR:     /* change symbol number */
  case DINIT_NML:     /* change symbol number */
  case DINIT_LABEL:   /* change symbol number */
  case DINIT_TYPEDEF: /* change symbol number */
  case DINIT_LOC:     /* change symbol number */
    sptr = new_symbol((int)pcon);
    dinit_put(ptype, sptr);
    break;
  default:
    switch (DTY(ptype)) {
    case TY_DBLE:
    case TY_CMPLX:
    case TY_DCMPLX:
    case TY_QUAD:
    case TY_QCMPLX:
    case TY_INT8:
    case TY_LOG8:
    case TY_CHAR:
    case TY_NCHAR:
      /* update sptr */
      sptr = new_symbol((int)pcon);
      dinit_put(ptype, sptr);
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
      /* write out unchanged */
      dinit_put(ptype, pcon);
      break;
    } /* switch */
    break;
  } /* switch */
} /* put_dinit_record */

static VAR *
getivl(lzhandle *fdlz, const char *file_name, int ipa)
{
  char *p;
  VAR *first = NULL;
  VAR *prev = NULL;
  VAR *thisone;
  VAR *lastone;
  int more;
  static int doendmore;

  do {
    int ast, dtype, id;
    int astvar, astlowbd, astupbd, aststep;
    VAR *subone;
    READ_LZLINE;
    if (p == NULL) {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    }
    ++currp;
    if (p[0] == 'Z') {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      break;
    }
    switch (p[0]) {
    case 'V': /* varref: V ast.varref.ptr dtype more */
      ast = get_num(10);
      dtype = get_num(10);
      id = get_num(10);
      more = get_num(10);
      thisone = (VAR *)getitem(PERM_AREA, sizeof(VAR));
      memset(thisone, 0, sizeof(VAR));
      thisone->id = Varref;
      thisone->next = NULL;
      thisone->u.varref.id = id;
      thisone->u.varref.subt = NULL;
      if (ipa == 1) {
        thisone->u.varref.ptr = ast;
        thisone->u.varref.dtype = dtype;
      } else if (ipa == 2) {
        thisone->u.varref.ptr = ipa_ast(ast);
        thisone->u.varref.dtype = dindex(dtype);
      } else {
        thisone->u.varref.ptr = new_ast(ast);
        thisone->u.varref.dtype = new_dtype(dtype);
      }
      lastone = thisone;
      break;
    case 'W': /* subtype: W dtype more */
      dtype = get_num(10);
      more = get_num(10);
      subone = getivl(fdlz, file_name, ipa);
      thisone = (VAR *)getitem(PERM_AREA, sizeof(VAR));
      memset(thisone, 0, sizeof(VAR));
      thisone->next = NULL;
      thisone->id = Varref;
      thisone->u.varref.subt = subone;
      if (ipa == 1) {
        thisone->u.varref.dtype = dtype;
      } else if (ipa == 2) {
        thisone->u.varref.dtype = dindex(dtype);
      } else {
        thisone->u.varref.dtype = new_dtype(dtype);
      }
      lastone = thisone;
      break;
    case 'D': /* do: D ast.indvar ast.lowbd ast.upbd ast.step more */
      astvar = get_num(10);
      astlowbd = get_num(10);
      astupbd = get_num(10);
      aststep = get_num(10);
      more = get_num(10);
      subone = getivl(fdlz, file_name, ipa);
      more = doendmore;
      thisone = (VAR *)getitem(PERM_AREA, sizeof(VAR));
      memset(thisone, 0, sizeof(VAR));
      thisone->id = Dostart;
      if (ipa == 1) {
        thisone->u.dostart.indvar = astvar;
        thisone->u.dostart.lowbd = astlowbd;
        thisone->u.dostart.upbd = astupbd;
        thisone->u.dostart.step = aststep;
      } else if (ipa == 2) {
        thisone->u.dostart.indvar = ipa_ast(astvar);
        thisone->u.dostart.lowbd = ipa_ast(astlowbd);
        thisone->u.dostart.upbd = ipa_ast(astupbd);
        thisone->u.dostart.step = ipa_ast(aststep);
      } else {
        thisone->u.dostart.indvar = new_ast(astvar);
        thisone->u.dostart.lowbd = new_ast(astlowbd);
        thisone->u.dostart.upbd = new_ast(astupbd);
        thisone->u.dostart.step = new_ast(aststep);
      }
      thisone->next = subone;
      for (lastone = subone; lastone->next; lastone = lastone->next)
        ;
      lastone->next = (VAR *)getitem(PERM_AREA, sizeof(VAR));
      lastone = lastone->next;
      lastone->id = Doend;
      lastone->u.doend.dostart = thisone;
      lastone->next = NULL;
      break;
    case 'E': /* doend: E more */
      doendmore = get_num(10);
      return first;
    default:
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    }
    if (prev) {
      prev->next = thisone;
    } else {
      first = thisone;
    }
    prev = lastone;
  } while (more);
  return first;
} /* getivl */

static ACL *
getict(lzhandle *fdlz, const char *file_name, int ipa)
{
  char *p;
  ACL *first = NULL;
  ACL *thisone;
  ACL *prev = NULL;
  int more;
  int i;

  do {
    int op;
    int init_ast, limit_ast, step_ast;
    int sptr, dtype, ptrdtype, repeatc, is_const, ast, nosubc;
    ACL *subone;
    READ_LZLINE;
    if (p == NULL) {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    }
    ++currp;
    if (p[0] == 'Z') {
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
      break;
    }
    switch (p[0]) {
    case 'A': /* ast initializer: A sptr dtype ptrdtype repeat is_const value
                 more */
      sptr = get_num(10);
      dtype = get_num(10);
      ptrdtype = get_num(10);
      repeatc = get_num(10);
      is_const = get_num(10);
      ast = get_num(10);
      more = get_num(10);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_AST;
      thisone->subc = NULL;
      thisone->next = NULL;
      if (ipa == 1) {
        thisone->sptr = sptr;
        thisone->dtype = dtype;
        thisone->ptrdtype = ptrdtype;
        thisone->repeatc = repeatc;
        thisone->is_const = is_const;
        thisone->u1.ast = ast;
      } else if (ipa == 2) {
        thisone->sptr = get_symbolxref(sptr);
        thisone->dtype = dindex(dtype);
        thisone->ptrdtype = dindex(ptrdtype);
        thisone->repeatc = ipa_ast(repeatc);
        thisone->is_const = is_const;
        thisone->u1.ast = ipa_ast(ast);
      } else {
        thisone->sptr = new_symbol(sptr);
        thisone->dtype = new_dtype(dtype);
        thisone->ptrdtype = new_dtype(ptrdtype);
        thisone->repeatc = new_ast(repeatc);
        thisone->is_const = is_const;
        thisone->u1.ast = new_ast(ast);
      }
      Trace(("reading ict at %x with ast=%d,repeat=%d", thisone, ast, repeatc));
      break;
    case 'I': /* ident initializer: I sptr dtype ptrdtype repeat value more */
      sptr = get_num(10);
      dtype = get_num(10);
      ptrdtype = get_num(10);
      repeatc = get_num(10);
      ast = get_num(10);
      more = get_num(10);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_AST;
      thisone->subc = NULL;
      thisone->next = NULL;
      if (ipa == 1) {
        thisone->sptr = sptr;
        thisone->dtype = dtype;
        thisone->ptrdtype = ptrdtype;
        thisone->repeatc = repeatc;
        thisone->u1.ast = ast;
      } else if (ipa == 2) {
        thisone->sptr = get_symbolxref(sptr);
        thisone->dtype = dindex(dtype);
        thisone->ptrdtype = dindex(ptrdtype);
        thisone->repeatc = ipa_ast(repeatc);
        thisone->u1.ast = ipa_ast(ast);
      } else {
        thisone->sptr = new_symbol(sptr);
        thisone->dtype = new_dtype(dtype);
        thisone->ptrdtype = new_dtype(ptrdtype);
        thisone->repeatc = new_ast(repeatc);
        thisone->u1.ast = new_ast(ast);
      }
      Trace(("reading ict at %x with ast=%d,repeat=%d", thisone, ast, repeatc));
      break;
    case 'L': /* literal integer */
      i = get_num(10);
      more = get_num(10);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_ICONST;
      thisone->u1.i = i;
      break;
    case 'N': /* NULL ROP */
      if (first == NULL)
        return NULL;
      break;
    case 'O': /* implied do initializer */
      sptr = get_num(10);
      init_ast = get_num(10);
      limit_ast = get_num(10);
      step_ast = get_num(10);
      more = get_num(10);
      subone = getict(fdlz, file_name, ipa);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_IDO;
      thisone->subc = subone;
      thisone->next = NULL;
      /* alloc do struct */
      thisone->u1.doinfo = (DOINFO *)getitem(PERM_AREA, sizeof(DOINFO));
      memset(thisone->u1.doinfo, 0, sizeof(DOINFO));
      if (ipa == 1) {
        thisone->u1.doinfo->index_var = sptr;
        thisone->u1.doinfo->init_expr = init_ast;
        thisone->u1.doinfo->limit_expr = limit_ast;
        thisone->u1.doinfo->step_expr = step_ast;
      } else if (ipa == 2) {
        thisone->u1.doinfo->index_var = get_symbolxref(sptr);
        thisone->u1.doinfo->init_expr = ipa_ast(init_ast);
        thisone->u1.doinfo->limit_expr = ipa_ast(limit_ast);
        thisone->u1.doinfo->step_expr = ipa_ast(step_ast);
      } else {
        thisone->u1.doinfo->index_var = new_symbol(sptr);
        thisone->u1.doinfo->init_expr = new_ast(init_ast);
        thisone->u1.doinfo->limit_expr = new_ast(limit_ast);
        thisone->u1.doinfo->step_expr = new_ast(step_ast);
      }
      break;
    case 'P': /* repeat initializer: P sptr dtype ptrdtype value more */
      sptr = get_num(10);
      dtype = get_num(10);
      ptrdtype = get_num(10);
      ast = get_num(10);
      more = get_num(10);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_AST;
      thisone->subc = NULL;
      thisone->next = NULL;
      if (ipa == 1) {
        thisone->sptr = sptr;
        thisone->dtype = dtype;
        thisone->ptrdtype = ptrdtype;
        thisone->u1.ast = ast;
      } else if (ipa == 2) {
        thisone->sptr = get_symbolxref(sptr);
        thisone->dtype = dindex(dtype);
        thisone->ptrdtype = dindex(ptrdtype);
        thisone->u1.ast = ipa_ast(ast);
      } else {
        thisone->sptr = new_symbol(sptr);
        thisone->dtype = new_dtype(dtype);
        thisone->ptrdtype = new_dtype(ptrdtype);
        thisone->u1.ast = new_ast(ast);
      }
      Trace(("reading ict at %x with ast=%d", thisone, ast));
      break;
    case 'S': /* struct (typedef) initializer sptr dtype ptrdtype repeat more */
      sptr = get_num(10);
      dtype = get_num(10);
      ptrdtype = get_num(10);
      repeatc = get_num(10);
      more = get_num(10);
      nosubc = get_num(10);
      subone = nosubc ? NULL : getict(fdlz, file_name, ipa);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_SCONST;
      thisone->subc = subone;
      thisone->next = NULL;
      if (ipa == 1) {
        thisone->sptr = sptr;
        thisone->dtype = dtype;
        thisone->ptrdtype = ptrdtype;
        thisone->repeatc = repeatc;
      } else if (ipa == 2) {
        thisone->sptr = get_symbolxref(sptr);
        thisone->dtype = dindex(dtype);
        thisone->ptrdtype = dindex(ptrdtype);
        thisone->repeatc = ipa_ast(repeatc);
      } else {
        thisone->sptr = new_symbol(sptr);
        thisone->dtype = new_dtype(dtype);
        thisone->ptrdtype = new_dtype(ptrdtype);
        thisone->repeatc = new_ast(repeatc);
      }

      Trace(("reading Struct ict at %x with sub-ict at %x", thisone, subone));
      break;
    case 'R': /* Array initializer sptr dtype ptrdtype more */
      sptr = get_num(10);
      dtype = get_num(10);
      ptrdtype = get_num(10);
      more = get_num(10);
      subone = getict(fdlz, file_name, ipa);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_ACONST;
      thisone->subc = subone;
      thisone->next = NULL;
      if (ipa == 1) {
        thisone->sptr = sptr;
        thisone->dtype = dtype;
        thisone->ptrdtype = ptrdtype;
      } else if (ipa == 2) {
        thisone->sptr = get_symbolxref(sptr);
        thisone->dtype = dindex(dtype);
        thisone->ptrdtype = dindex(ptrdtype);
      } else {
        thisone->sptr = new_symbol(sptr);
        thisone->dtype = new_dtype(dtype);
        thisone->ptrdtype = new_dtype(ptrdtype);
      }
      Trace(("reading Array ict at %x with sub-ict at %x", thisone, subone));
      break;
    case 'U': /* union/struct initializer: U sptr dtype ptrdtype repeat value
                 more */
    case 'V':
    case 'T':
      sptr = get_num(10);
      dtype = get_num(10);
      ptrdtype = get_num(10);
      repeatc = get_num(10);
      ast = get_num(10);
      more = get_num(10);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      switch (p[0]) {
      case 'U':
        thisone->id = AC_VMSUNION;
        break;
      case 'V':
        thisone->id = AC_VMSSTRUCT;
        break;
      case 'T':
        thisone->id = AC_TYPEINIT;
        break;
      }
      subone = getict(fdlz, file_name, ipa);
      thisone->subc = subone;
      thisone->next = NULL;
      if (ipa == 1) {
        thisone->sptr = sptr;
        thisone->dtype = dtype;
        thisone->ptrdtype = ptrdtype;
        thisone->repeatc = repeatc;
        thisone->u1.ast = ast;
      } else if (ipa == 2) {
        thisone->sptr = get_symbolxref(sptr);
        thisone->dtype = dindex(dtype);
        thisone->ptrdtype = dindex(ptrdtype);
        thisone->repeatc = ipa_ast(repeatc);
        thisone->u1.ast = ipa_ast(ast);
      } else {
        thisone->sptr = new_symbol(sptr);
        thisone->dtype = new_dtype(dtype);
        thisone->ptrdtype = new_dtype(ptrdtype);
        thisone->repeatc = new_ast(repeatc);
        thisone->u1.ast = new_ast(ast);
      }
      Trace(("reading ict at %x with ast=%d,repeat=%d", thisone, ast, repeatc));
      break;
    case 'X': /* expression initializer expr->op sptr dtype ptrdtype more*/
      op = get_num(10);
      sptr = get_num(10);
      dtype = get_num(10);
      ptrdtype = get_num(10);
      repeatc = get_num(10);
      thisone = GET_ACL(PERM_AREA);
      memset(thisone, 0, sizeof(ACL));
      thisone->id = AC_IEXPR;
      thisone->subc = NULL;
      thisone->next = NULL;
      thisone->u1.expr = (AEXPR *)getitem(PERM_AREA, sizeof(AEXPR));
      memset(thisone->u1.expr, 0, sizeof(AEXPR));
      thisone->u1.expr->op = op;
      if (ipa == 1) {
        thisone->sptr = sptr;
        thisone->dtype = dtype;
        thisone->ptrdtype = ptrdtype;
        thisone->repeatc = repeatc;
      } else if (ipa == 2) {
        thisone->sptr = get_symbolxref(sptr);
        thisone->dtype = dindex(dtype);
        thisone->ptrdtype = dindex(ptrdtype);
        thisone->repeatc = ipa_ast(repeatc);
      } else {
        thisone->sptr = new_symbol(sptr);
        thisone->dtype = new_dtype(dtype);
        thisone->ptrdtype = new_dtype(ptrdtype);
        thisone->repeatc = new_ast(repeatc);
      }
      more = get_num(10);
      thisone->u1.expr->lop = getict(fdlz, file_name, ipa);
      if (BINOP(thisone->u1.expr)) {
        thisone->u1.expr->rop = getict(fdlz, file_name, ipa);
      } else {
        thisone->u1.expr->rop = NULL;
      }
      break;
    default:
      error(4, 0, gbl.lineno, import_corrupt_msg, import_file_name);
    }
    if (prev) {
      prev->next = thisone;
    } else {
      first = thisone;
    }
    if (thisone->sptr)
      save_struct_init(thisone);
    prev = thisone;
  } while (more);
  Trace(("getict returning ict %x", first));
  return first;
} /* getict */

static void
put_data_statement(int lineno, int anyivl, int anyict, lzhandle *fdlz,
                   const char *file_name, int ipa)
{
  int nw;
  char *ptr;
  VAR *ivl;
  ACL *ict;

  if (astb.df == NULL) {
    astb.df = tmpfile();
    if (astb.df == NULL)
      errfatal(5);
  }
  if (anyivl) {
    ivl = getivl(fdlz, file_name, ipa);
  } else {
    ivl = NULL;
  }
  if (anyict) {
    ict = getict(fdlz, file_name, ipa);
  } else {
    ict = NULL;
  }

  /* For modules, insure that only one ICT/IVL list is put out for each named
   * constant
   * by building the initialization lists only if we are importing the module
   * that
   * defines this named constant. */
  if (for_module && SCOPEG(sym_of_ast(ivl->u.varref.ptr)) != stb.curr_scope) {
    return;
  }
  if (ivl && ict) {
    if (ivl->id == Varref && ivl->u.varref.ptr) {
      int sptr = A_SPTRG(ivl->u.varref.ptr);
      if (PARAMG(sptr)) {
        if (STYPEG(sptr) != ST_PARAM) {
          sptr = NMCNSTG(sptr);
        }
        CONVAL2P(sptr, put_getitem_p(ict));
      }
    }
  }
  Trace(("Writing ICL record at line %d, ivl %x, ict %x", lineno, ivl, ict));
  nw = fwrite(&lineno, sizeof(lineno), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);
  nw = fwrite(&gbl.findex, sizeof(gbl.findex), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);
  ptr = (char *)ivl;
  nw = fwrite(&ptr, sizeof(ptr), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);
  ptr = (char *)ict;
  nw = fwrite(&ptr, sizeof(ptr), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);

  if (!for_interproc) {
    sem.dinit_nbr_inits++;
  }
} /* put_data_statement */

static void
get_component_init(lzhandle *fdlz, const char *file_name, char *p, int ipa)
{
  ACL *ict;
  int sptr;
  int dtype;
  int ptrdtype;
  int tag;
  int repeatc;
  int ast;

#if DEBUG
  assert(p[0] == 'T', "get_component_init: invalid input record (%c)\n", p[0],
         2);
#endif

  sptr = get_num(10);
  dtype = get_num(10);
  ptrdtype = get_num(10);
  repeatc = get_num(10);
  ast = get_num(10);
  if (ipa == 1) {
  } else if (ipa == 2) {
    sptr = get_symbolxref(sptr);
    dtype = dindex(dtype);
    if (ptrdtype)
      ptrdtype = dindex(ptrdtype);
    ast = ipa_ast(ast);
  } else {
    sptr = new_symbol(sptr);
    dtype = new_dtype(dtype);
    ast = new_ast(ast);
    if (ptrdtype)
      ptrdtype = new_dtype(ptrdtype);
  }

  tag = DTY(dtype + 3);
  if (DTYPEG(tag) != dtype) {
    DTY(dtype + 5) = DTY(DTYPEG(tag) + 5);
    getict(fdlz, file_name, ipa); /* consume it */
  } else {
    ict = GET_ACL(PERM_AREA);
    ict->id = AC_TYPEINIT;
    ict->sptr = sptr;
    ict->dtype = dtype;
    ict->ptrdtype = ptrdtype;
    ict->repeatc = repeatc;
    ict->u1.ast = ast;
    ict->subc = getict(fdlz, file_name, ipa);
    DTY(ict->dtype + 5) = put_getitem_p(ict);
  }
} /* get_component_init */

static int
install_dtype(DITEM *pd)
{
  int dtype, nd, na, ns;
  int paramct, dpdsc;
  int i;

  dtype = pd->new_id;
  if (pd->dtypeinstalled)
    return dtype;
  pd->dtypeinstalled = TRUE; /* set flag now just in case recursion occurs */
  switch (pd->ty) {
  case TY_PTR:
    nd = new_dtype(DTY(dtype + 1));
    DTY(dtype + 1) = nd;
    break;
  case TY_ARRAY:
    nd = new_dtype(DTY(dtype + 1));
    DTY(dtype + 1) = nd;
    if (DTY(dtype + 2) == 0)
      break;
    na = new_ast(ADD_ZBASE(dtype));
    ADD_ZBASE(dtype) = na;
    /*  fill in array dtypes with the new asts */
    for (i = 0; i < ADD_NUMDIM(dtype); i++) {
      na = new_ast(ADD_LWBD(dtype, i));
      ADD_LWBD(dtype, i) = na;
      na = new_ast(ADD_UPBD(dtype, i));
      ADD_UPBD(dtype, i) = na;
      na = new_ast(ADD_MLPYR(dtype, i));
      ADD_MLPYR(dtype, i) = na;
      na = new_ast(ADD_LWAST(dtype, i));
      ADD_LWAST(dtype, i) = na;
      na = new_ast(ADD_UPAST(dtype, i));
      ADD_UPAST(dtype, i) = na;
      na = new_ast(ADD_EXTNTAST(dtype, i));
      ADD_EXTNTAST(dtype, i) = na;
    }
    na = new_ast(ADD_NUMELM(dtype));
    ADD_NUMELM(dtype) = na;
    break;
  case TY_UNION:
  case TY_STRUCT:
  case TY_DERIVED:
    /* because we dump all dtypes, we may get some dtype records that
       aren't needed.  If we can't find the members, we'll assume that
       this dtype wasn't really needed */
    if (can_find_symbol(DTY(dtype + 1))) {
      ns = new_symbol(DTY(dtype + 1)); /* first member */
      DTY(dtype + 1) = ns;
      /* the tag is updated later, in 'new_dtypes()' */
    } else {
      /* kill the tag field, too. */
      DTY(dtype + 1) = NOSYM;
    }
    break;
  case TY_CHAR:
  case TY_NCHAR:
    na = new_ast(DTY(dtype + 1));
    DTY(dtype + 1) = na;
    break;
  case TY_PROC:
    nd = new_dtype(DTY(dtype + 1));
    DTY(dtype + 1) = nd;
    ns = DTY(dtype + 2); /* interface */
    if (ns) {
      ns = new_symbol(ns);
      DTY(dtype + 2) = ns;
    }
    paramct = DTY(dtype + 3);
    dpdsc = DTY(dtype + 4);
    if (paramct) {
      for (i = 0; i < paramct; i++) {
        ns = new_symbol(aux.dpdsc_base[dpdsc + i]);
        aux.dpdsc_base[dpdsc + i] = ns;
      }
    }
    ns = DTY(dtype + 5); /* fval */
    if (ns) {
      ns = new_symbol(ns);
      DTY(dtype + 5) = ns;
      aux.dpdsc_base[dpdsc - 1] = ns;
    }
    break;
  default:
    interr("module:new_dtype, illegal type", pd->ty, 0);
    return DT_INT;
  }
  return dtype;
} /* install_dtype */

static int
new_dtype(int old_dt)
{
  DITEM *pd;

  pd = finddthash(old_dt);
  if (pd == NULL) {
    if (dtype_ivsn < 34)
      old_dt = adjust_pre34_dtype(old_dt);
    if (old_dt < BASEdty)
      return old_dt;
    interr("module:new_dtype, dt nfd", old_dt, 0);
    return DT_INT;
  }
  return pd->new_id;
} /* new_dtype */

static int
new_installed_dtype(int old_dt)
{
  DITEM *pd;
  int dtype;

  pd = finddthash(old_dt);
  if (pd == NULL) {
    if (dtype_ivsn < 34)
      old_dt = adjust_pre34_dtype(old_dt);
    if (old_dt < BASEdty)
      return old_dt;
    interr("module:new_installed_dtype, dt nfd", old_dt, 0);
    return DT_INT;
  }
  if (pd->dtypeinstalled) {
    dtype = pd->new_id;
  } else {
    dtype = install_dtype(pd);
  }
  return dtype;
} /* new_installed_dtype */

static void
new_dtypes(void)
{
  DITEM *pd;
  int j, dtype;

  for (j = 0; j < dtz.avl; j++) {
    pd = dtz.base + j;
    if (pd->dtypeinstalled) {
      dtype = pd->new_id;
    } else {
      dtype = install_dtype(pd);
    }
    switch (DTY(dtype)) {
    case TY_UNION:
    case TY_STRUCT:
    case TY_DERIVED:
      if (pd->ty != 0 && DTY(dtype + 3)) {
        int ns;
        ns = new_symbol(DTY(dtype + 3));
        DTY(dtype + 3) = ns;
      }
    }
  }
  if (inmodulecontains)
    exportb.hmark.dt = stb.dt.stg_avail; /* for subsequent 'export_dtypes()' */
}

static int
fill_ast(ASTITEM *pa)
{
  int type;
  int alias;
  int ast;
  int sptr, osptr;
  int lop, rop, left, right;
  int stride;
  int optype;
  int dtype = 0;
  int count;
  int argt;
  int shape;
  int asd;
  int astli;
  int std;
  int l1, l2, l3, l4;
  int i, j;
  SYMITEM *ps;
  /* WARNING, recursive calls (possibly thru other procedures) may
   * clobber this area.  Grab what you need first.
   */
  BZERO(astb.stg_base, AST, 1);
  astb.stg_base[0].type = pa->type;

#define GETFIELD(f) astb.stg_base[0].f = pa->a.f
  GETFIELD(f2);
  GETFIELD(shape);
  GETFIELD(w3);
  GETFIELD(w4);
  GETFIELD(w5);
  GETFIELD(w6);
  GETFIELD(w7);
  GETFIELD(w8);
  GETFIELD(w9);
  GETFIELD(w10);
  GETFIELD(hw21);
  GETFIELD(hw22);
  GETFIELD(w12);
  GETFIELD(opt1);
  GETFIELD(opt2);
  GETFIELD(repl);
  GETFIELD(visit);
  GETFIELD(w18); /* IVSN 30 */
  GETFIELD(w19);
#undef GETFIELD

  switch (type = A_TYPEG(0)) {
  case A_ID:
    ps = NULL;
    alias = A_ALIASG(0);
    osptr = A_SPTRG(0);
    if (pa->flags & A_IDSTR_mask) {
      sptr = osptr; /* This is one when we import already */
    } else {
      new_symbol_and_link(osptr, &sptr, &ps);
    }
    /* not all symbols were installed, so dtype may be wrong */
    if (ps && ps->sc >= 0) {
      dtype = new_installed_dtype(ps->dtype);
      DTYPEP(sptr, dtype);
    }
    ast = mk_id(sptr);
    if (STYPEG(sptr) == ST_PARAM && alias) {
      alias = new_ast(alias);
      A_ALIASP(ast, alias);
    }
    break;
  case A_CNST:
    dtype = A_DTYPEG(0);
    osptr = A_SPTRG(0);
    new_symbol_and_link(osptr, &sptr, &ps);
    /* not all symbols were installed, so dtype may be wrong */
    if (ps) {
      dtype = new_dtype(ps->dtype);
      DTYPEP(sptr, dtype);
    }
    ast = mk_cnst(sptr);
    break;
  case A_LABEL:
    sptr = new_symbol((int)A_SPTRG(0));
    ast = mk_label(sptr);
    break;
  case A_BINOP:
    lop = A_LOPG(0);
    rop = A_ROPG(0);
    optype = A_OPTYPEG(0);
    dtype = new_dtype((int)A_DTYPEG(0));
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_binop(optype, lop, rop, dtype);
    break;
  case A_UNOP:
    lop = A_LOPG(0);
    optype = A_OPTYPEG(0);
    dtype = new_dtype((int)A_DTYPEG(0));
    lop = new_ast(lop);
    ast = mk_unop(optype, lop, dtype);
    break;
  case A_CMPLXC:
    lop = A_LOPG(0);
    rop = A_ROPG(0);
    dtype = new_dtype((int)A_DTYPEG(0));
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_cmplxc(lop, rop, dtype);
    break;
  case A_PAREN:
    lop = A_LOPG(0);
    dtype = new_dtype((int)A_DTYPEG(0));
    lop = new_ast(lop);
    ast = mk_paren(lop, dtype);
    break;
  case A_CONV:
    lop = A_LOPG(0);
    dtype = new_dtype((int)A_DTYPEG(0));
    lop = new_ast(lop);
    ast = mk_convert(lop, dtype);
    break;
  case A_MEM:
    lop = A_PARENTG(0);
    rop = A_MEMG(0);
    dtype = new_installed_dtype((int)A_DTYPEG(0));
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_member(lop, rop, dtype);
    break;
  case A_SUBSTR:
    lop = A_LOPG(0);
    left = A_LEFTG(0);
    right = A_RIGHTG(0);
    alias = A_ALIASG(0);
    dtype = new_dtype((int)A_DTYPEG(0));
    lop = new_ast(lop);
    left = new_ast(left);
    right = new_ast(right);
    if (alias)
      alias = new_ast(alias);
    ast = mk_substr(lop, left, right, dtype);
    break;
  case A_INIT:
    sptr = A_SPTRG(0);
    left = A_LEFTG(0);
    right = A_RIGHTG(0);
    dtype = new_dtype((int)A_DTYPEG(0));
    if (sptr)
      sptr = new_symbol(sptr);
    left = new_ast(left);
    if (right)
      right = new_ast(right);
    ast = mk_init(left, dtype);
    A_RIGHTP(ast, right);
    A_SPTRP(ast, sptr);
    break;
  case A_SUBSCR:
    lop = A_LOPG(0);
    asd = A_ASDG(0);
    dtype = A_DTYPEG(0);
    lop = new_ast(lop);
    i = new_asd(pa->list);
    dtype = new_dtype(dtype);
    ast = mk_subscr(lop, asdz.base[i].subs, asdz.base[i].ndim, dtype);
    break;
  case A_TRIPLE:
    lop = A_LBDG(0);
    rop = A_UPBDG(0);
    stride = A_STRIDEG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    stride = new_ast(stride);
    ast = mk_triple(lop, rop, stride);
    break;
  case A_FUNC:
  case A_CALL:
    lop = A_LOPG(0);
    count = A_ARGCNTG(0);
    argt = 0;
    if (type == A_FUNC) {
      dtype = A_DTYPEG(0);
      shape = A_SHAPEG(0);
      dtype = new_dtype(dtype);
      shape = new_shape(pa->shape);
    }
    lop = new_ast(lop);
    argt = new_argt(pa->list);
    /* 'simulate' everything which begin_call does */
    ast = new_node(type);
    A_LOPP(ast, lop);
    A_ARGCNTP(ast, count);
    A_ARGSP(ast, argt);
    A_DTYPEP(ast, dtype);
    if (type == A_FUNC) {
      A_CALLFGP(ast, argtz.base[pa->list].callfg);
      A_SHAPEP(ast, shape);
    }
    if (for_inliner) {
      /* simulate what we would have done in semfin had we seen this
       * function call before the inliner.
       * put the procedure on the aux.list[ST_PROC] list.
       * expose the FVAL in the argument list for array valued functions */
      if (A_TYPEG(lop) == A_ID) {
        int fval, dpdsc;
        sptr = A_SPTRG(lop);
        if (STYPEG(sptr) == ST_PROC && SLNKG(sptr) == 0) {
          SLNKP(sptr, aux.list[ST_PROC]);
          aux.list[ST_PROC] = sptr;
        }
        dtype = DTYPEG(sptr);
        dpdsc = DPDSCG(sptr);
        fval = FVALG(sptr);
        if (DTY(dtype) == TY_ARRAY && dpdsc && fval) {
          if (aux.dpdsc_base[dpdsc] != fval && aux.dpdsc_base[dpdsc - 1] == 0) {
            aux.dpdsc_base[dpdsc - 1] = fval;
            DPDSCP(sptr, dpdsc - 1);
            PARAMCTP(sptr, PARAMCTG(sptr) + 1);
            FUNCP(sptr, 0);
          }
        }
      }
    }
    break;
  case A_INTR:
  case A_ICALL:
    optype = A_OPTYPEG(0);
    lop = A_LOPG(0);
    count = A_ARGCNTG(0);
    if (type == A_INTR) {
      dtype = A_DTYPEG(0);
      shape = A_SHAPEG(0);
      dtype = new_dtype(dtype);
      shape = new_shape(pa->shape);
    }
    lop = new_ast(lop);
    argt = new_argt(pa->list);
    /* 'simulate' everything which begin_call does */
    ast = new_node(type);
    A_LOPP(ast, lop);
    A_ARGCNTP(ast, count);
    A_ARGSP(ast, argt);
    A_DTYPEP(ast, dtype);
    A_OPTYPEP(ast, optype);
    if (type == A_INTR) {
      A_CALLFGP(ast, argtz.base[pa->list].callfg);
      A_SHAPEP(ast, shape);
      /* make sure the runtime library functions are declared */
      switch (optype) {
      case I_SIZE:
        (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_size), stb.user.dt_int);
        break;
      case I_LBOUND:
        (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_lb), stb.user.dt_int);
        break;
      case I_UBOUND:
        (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_ub), stb.user.dt_int);
        break;
      }
    }
    break;
  case A_ASN:
    lop = A_DESTG(0);
    rop = A_SRCG(0);
    dtype = A_DTYPEG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    dtype = new_dtype(dtype);
    ast = mk_assn_stmt(lop, rop, dtype);
    break;
  case A_IF:
    lop = A_IFEXPRG(0);
    rop = A_IFSTMTG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_stmt(A_IF, 0);
    A_IFEXPRP(ast, lop);
    A_IFSTMTP(ast, rop);
    break;
  case A_IFTHEN:
  case A_ELSEIF:
    lop = A_IFEXPRG(0);
    lop = new_ast(lop);
    ast = mk_stmt(type, 0);
    A_IFEXPRP(ast, lop);
    break;
  case A_AIF:
    lop = A_IFEXPRG(0);
    l1 = A_L1G(0);
    l2 = A_L2G(0);
    l3 = A_L3G(0);
    lop = new_ast(lop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    l3 = new_ast(l3);
    ast = mk_stmt(A_AIF, 0);
    A_IFEXPRP(ast, lop);
    A_L1P(ast, l1);
    A_L2P(ast, l2);
    A_L3P(ast, l3);
    break;
  case A_GOTO:
    l1 = A_L1G(0);
    l1 = new_ast(l1);
    ast = mk_stmt(A_GOTO, 0);
    A_L1P(ast, l1);
    break;
  case A_CGOTO:
  case A_AGOTO:
    lop = A_LOPG(0);
    lop = new_ast(lop);
    astli = new_astli(pa->list, pa->type);
    ast = mk_stmt(type, 0);
    A_LISTP(ast, astli);
    A_LOPP(ast, lop);
    break;
  case A_ASNGOTO:
    Trace(("assigned goto at ast %d cannot be imported from file %s",
           pa->old_ast, import_file_name));
    interr("new_ast:ast type not supported", type, 3);
    ast = 0;
    break;
  case A_DO:
    lop = A_DOLABG(0);
    rop = A_DOVARG(0);
    l1 = A_M1G(0);
    l2 = A_M2G(0);
    l3 = A_M3G(0);
    l4 = A_M4G(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    l3 = new_ast(l3);
    l4 = new_ast(l4);
    ast = mk_stmt(A_DO, 0);
    A_DOLABP(ast, lop);
    A_DOVARP(ast, rop);
    A_M1P(ast, l1);
    A_M2P(ast, l2);
    A_M3P(ast, l3);
    A_M4P(ast, l4);
    break;
  case A_DOWHILE:
    lop = A_DOLABG(0);
    rop = A_IFEXPRG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_stmt(A_DOWHILE, 0);
    A_DOLABP(ast, lop);
    A_IFEXPRP(ast, rop);
    break;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    lop = A_LOPG(0);
    lop = new_ast(lop);
    ast = mk_stmt(type, 0);
    A_LOPP(ast, lop);
    break;
  case A_ALLOC:
    optype = A_TKNG(0);
    lop = A_LOPG(0);
    rop = A_SRCG(0);
    l1 = A_DESTG(0);
    l2 = A_STARTG(0);
    l3 = A_M3G(0);
    l4 = A_DTYPEG(0);
    i = A_FIRSTALLOCG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    l3 = new_ast(l3);
    l4 = new_installed_dtype(l4);
    ast = mk_stmt(A_ALLOC, 0);
    A_SRCP(ast, rop);
    A_LOPP(ast, lop);
    A_DESTP(ast, l1);
    A_STARTP(ast, l2);
    A_M3P(ast, l3);
    A_TKNP(ast, optype);
    A_DTYPEP(ast, l4);
    A_FIRSTALLOCP(ast, i);
    break;
  case A_WHERE:
    lop = A_IFEXPRG(0);
    rop = A_IFSTMTG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_stmt(A_WHERE, 0);
    A_IFEXPRP(ast, lop);
    A_IFSTMTP(ast, rop);
    break;
  case A_FORALL:
    lop = A_IFEXPRG(0);
    rop = A_IFSTMTG(0);
    std = A_SRCG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    astli = new_astli(pa->list, pa->type);
    std = new_std(std);
    ast = mk_stmt(A_FORALL, 0);
    A_LISTP(ast, astli);
    A_IFEXPRP(ast, lop);
    A_IFSTMTP(ast, rop);
    A_SRCP(ast, std);
    break;
  case A_REDIM:
    lop = A_SRCG(0);
    lop = new_ast(lop);
    ast = mk_stmt(A_REDIM, 0);
    A_SRCP(ast, lop);
    break;
  case A_ENTRY:
    sptr = A_SPTRG(0);
    sptr = new_symbol(sptr);
    ast = mk_stmt(type, 0);
    A_SPTRP(ast, sptr);
    break;
  case A_COMSTR:
    Trace(("comment at ast %d cannot be imported from file %s", pa->old_ast,
           import_file_name));
    interr("new_ast:ast type not supported", type, 3);
    ast = 0;
    break;
  case A_COMMENT:
  case A_ELSE:
  case A_ENDIF:
  case A_ELSEFORALL:
  case A_ELSEWHERE:
  case A_ENDWHERE:
  case A_ENDFORALL:
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
    ast = mk_stmt(type, 0);
    break;
  case A_HLOCALIZEBNDS:
    lop = A_LOPG(0);
    rop = A_ITRIPLEG(0);
    l1 = A_OTRIPLEG(0);
    l2 = A_DIMG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    ast = mk_stmt(A_HLOCALIZEBNDS, 0);
    A_LOPP(ast, lop);
    A_ITRIPLEP(ast, rop);
    A_OTRIPLEP(ast, l1);
    A_DIMP(ast, l2);
    break;
  case A_HALLOBNDS:
    lop = A_LOPG(0);
    lop = new_ast(lop);
    ast = mk_stmt(A_HALLOBNDS, 0);
    A_LOPP(ast, lop);
    break;
  case A_HCYCLICLP:
    lop = A_LOPG(0);
    rop = A_ITRIPLEG(0);
    l1 = A_OTRIPLEG(0);
    l2 = A_OTRIPLE1G(0);
    l3 = A_DIMG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    l3 = new_ast(l3);
    ast = mk_stmt(A_HCYCLICLP, 0);
    A_LOPP(ast, lop);
    A_ITRIPLEP(ast, rop);
    A_OTRIPLEP(ast, l1);
    A_OTRIPLE1P(ast, l2);
    A_DIMP(ast, l3);
    break;
  case A_HOFFSET:
    l1 = A_DESTG(0);
    lop = A_LOPG(0);
    rop = A_ROPG(0);
    l1 = new_ast(l1);
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_stmt(A_HOFFSET, 0);
    A_DESTP(ast, l1);
    A_LOPP(ast, lop);
    A_ROPP(ast, rop);
    break;
  case A_HSECT:
    lop = A_LOPG(0);
    rop = A_BVECTG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = new_node(type);
    A_DTYPEP(ast, DT_INT);
    A_LOPP(ast, lop);
    A_BVECTP(ast, rop);
    break;
  case A_HCOPYSECT:
    lop = A_DESTG(0);
    rop = A_SRCG(0);
    l1 = A_DDESCG(0);
    l2 = A_SDESCG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    ast = new_node(type);
    A_DTYPEP(ast, DT_INT);
    A_DESTP(ast, lop);
    A_SRCP(ast, rop);
    A_DDESCP(ast, l1);
    A_SDESCP(ast, l2);
    break;
  case A_HPERMUTESECT:
    lop = A_DESTG(0);
    rop = A_SRCG(0);
    l1 = A_DDESCG(0);
    l2 = A_SDESCG(0);
    l3 = A_BVECTG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    l3 = new_ast(l3);
    ast = new_node(type);
    A_DTYPEP(ast, DT_INT);
    A_DESTP(ast, lop);
    A_SRCP(ast, rop);
    A_DDESCP(ast, l1);
    A_SDESCP(ast, l2);
    A_BVECTP(ast, l3);
    break;
  case A_HOVLPSHIFT:
    rop = A_SRCG(0);
    l2 = A_SDESCG(0);
    rop = new_ast(rop);
    l2 = new_ast(l2);
    ast = new_node(type);
    A_DTYPEP(ast, DT_INT);
    A_SRCP(ast, rop);
    A_SDESCP(ast, l2);
    break;
  case A_HGETSCLR:
    lop = A_DESTG(0);
    rop = A_SRCG(0);
    lop = new_ast(lop);
    rop = new_ast(rop);
    ast = mk_stmt(type, 0);
    A_DESTP(ast, lop);
    A_SRCP(ast, rop);
    break;
  case A_HGATHER:
  case A_HSCATTER:
    i = A_VSUBG(0);
    lop = A_DESTG(0);
    rop = A_SRCG(0);
    l1 = A_DDESCG(0);
    l2 = A_SDESCG(0);
    l3 = A_MDESCG(0);
    j = A_BVECTG(0);
    i = new_ast(i);
    lop = new_ast(lop);
    rop = new_ast(rop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    l3 = new_ast(l3);
    j = new_ast(j);
    ast = new_node(type);
    A_DTYPEP(ast, DT_INT);
    A_VSUBP(ast, i);
    A_DESTP(ast, lop);
    A_SRCP(ast, rop);
    A_DDESCP(ast, l1);
    A_SDESCP(ast, l2);
    A_MDESCP(ast, l3);
    A_BVECTP(ast, j);
    break;
  case A_HCSTART:
    lop = A_LOPG(0);
    l1 = A_DESTG(0);
    l2 = A_SRCG(0);
    lop = new_ast(lop);
    l1 = new_ast(l1);
    l2 = new_ast(l2);
    ast = new_node(type);
    A_DTYPEP(ast, DT_INT);
    A_LOPP(ast, lop);
    A_DESTP(ast, l1);
    A_SRCP(ast, l2);
    break;
  case A_HCFINISH:
  case A_HCFREE:
  case A_HOWNERPROC:
  case A_HLOCALOFFSET:
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
    lop = A_LOPG(0);
    if (lop)
      lop = new_ast(lop);
    ast = mk_stmt(type, 0);
    A_LOPP(ast, lop);
    break;
  case A_MP_BMPSCOPE:
  case A_MP_PARALLEL:
  case A_MP_ENDPARALLEL:
  case A_CRITICAL:
  case A_MASTER:
  case A_ENDATOMIC:
  case A_BARRIER:
  case A_NOBARRIER:
    ast = mk_stmt(type, 0);
    break;
  case A_ENDCRITICAL:
    lop = A_LOPG(0);
    lop = new_ast(lop); /* corresponding critical */
    ast = mk_stmt(type, 0);
    A_LOPP(ast, lop);
    A_LOPP(lop, ast);
    break;
  case A_ENDMASTER:
    lop = A_LOPG(0);
    count = A_ARGCNTG(0);
    argt = new_argt(pa->list);
    lop = new_ast(lop); /* corresponding master */
    ast = mk_stmt(type, 0);
    A_ARGCNTP(ast, count);
    A_ARGSP(ast, argt);
    A_LOPP(ast, lop);
    A_LOPP(lop, ast);
    break;
  default:
    Trace(("unknown ast type %d at ast %d from file %s", type, pa->old_ast,
           import_file_name));
    interr("new_ast:unexpected ast type", type, 3);
    ast = 0;
    break;
  }
#if DEBUG
  if (DBGBIT(3, 64))
    fprintf(gbl.dbgfil, "old ast %d new ast %d\n", pa->old_ast, ast);
#endif

  pa->new_ast = ast;
  return ast;
} /* fill_ast */

static void
new_asts(void)
{
  int i;

  for (i = 0; i < astz.avl; i++) {
    (void)fill_ast(astz.base + i);
  }
  BZERO(astb.stg_base + 0, AST, 1); /* reinitialize AST #0 */
}

static int
new_ast(int old_ast)
{
  ASTITEM *pa;
  int hash, s;

  hash = old_ast & ASTZHASHMASK;
  for (s = astzhash[hash]; s; s = pa->link) {
    pa = astz.base + (s - 1);
    if (pa->old_ast == old_ast)
      break;
  }
  if (!s) {
    if (old_ast < BASEast) {
      return old_ast;
    }
    Trace(("cannot find ast %d in file %s", old_ast, import_file_name));
    interr("incomplete interface file, missing AST", old_ast, 3);
    error(4, 0, gbl.lineno, "incomplete IPA file, missing AST ", "");
  }
  if (pa->new_ast)
    return pa->new_ast;
  return fill_ast(pa);
} /* new_ast */

static void
new_stds(void)
{
  STDITEM *p_std;
  int std;
  int ast;
  int lab;
  int i;

  for (i = 0; i < stdz.avl; i++) {
    int flags, bit;
    p_std = stdz.base + i;
    ast = new_ast(p_std->ast);
    lab = new_symbol(p_std->label);
    std = add_stmt(ast);
    STD_LABEL(std) = lab;
    STD_LINENO(std) = p_std->lineno;
#define GETBIT(f)                                          \
  astb.std.stg_base[std].flags.bits.f = (flags & bit) ? 1 : 0; \
  bit <<= 1;
    flags = p_std->flags;
    bit = 1;
    GETBIT(ex);
    GETBIT(st);
    GETBIT(br);
    GETBIT(delete);
    GETBIT(ignore);
    GETBIT(split);
    GETBIT(minfo);
    GETBIT(local);
    GETBIT(pure);
    GETBIT(par);
    GETBIT(cs);
    GETBIT(parsect);
    GETBIT(orig);
#undef GETBIT
  }
}

static int
new_std(int old_std)
{
  STDITEM *p_std;
  int i;

  if (old_std)
    for (i = 0; i < stdz.avl; i++) {
      p_std = stdz.base + i;
      if (p_std->old == old_std)
        return p_std->new;
    }
  return 0;
}

/** \brief Look up an (old) argt.
  *
  * If this is the first lookup, process the argument
  * asts from the old argt. The space for the new argt has already been
  * allocated and the 'arg' fields in the argt are overwritten by new asts.
  *
  * \return the index of the ASDITEM (NOT the new argt).
  */
static int
new_argt(int offset)
{
  int j;
  int cnt;
  int argt;
  int ast;

  if (offset == 0)
    return 0;
  argt = argtz.base[offset].new;
  if (!argtz.base[offset].installed) {
    cnt = ARGT_CNT(argt);
    /* set flag early */
    argtz.base[offset].installed = TRUE;
    for (j = 0; j < cnt; j++) {
      ast = ARGT_ARG(argt, j);
      ast = new_ast(ast);
      ARGT_ARG(argt, j) = ast;
      if (A_CALLFGG(ast))
        argtz.base[offset].callfg = 1;
    }
  }
  return argt;
}

/** \brief Look up an (old) asd.
  *
  * If this is the first lookup, process the subscript
  * ASTs; return the index of the ASDITEM item.  NOTE: this function doesn't
  * create a NEW asd; it will provide the necessary information for an ensuing
  * call to mk_subscr().
  */
static int
new_asd(int offset)
{
  int j;
  int cnt;
  int ast;

  if (!asdz.base[offset].installed) {
    cnt = asdz.base[offset].ndim;
    asdz.base[offset].installed = TRUE;
    for (j = 0; j < cnt; ++j) {
      ast = asdz.base[offset].subs[j];
      asdz.base[offset].subs[j] = new_ast(ast);
    }
  }
  return offset;
}

/** \brief Look up an (old) ast list.
  *
  * If this is the first lookup, process the
  * information of each item in the list.  The space for the new astli has
  * already been allocated and the fields in each astli item are overwritten.
  *
  * \return the index to the head of the 'new' ast list
  */
static int
new_astli(int offset, int atype)
{
  int astli;
  int ast;
  int sptr;

  if (!astliz.base[offset].installed) {
    astliz.base[offset].installed = TRUE;
    switch (atype) {
    case A_CGOTO:
    case A_AGOTO:
      for (astli = astliz.base[offset].new; astli; astli = ASTLI_NEXT(astli)) {
        ast = ASTLI_AST(astli);
        ast = new_ast(ast);
        ASTLI_AST(astli) = ast;
      }
      break;
    case A_FORALL:
      for (astli = astliz.base[offset].new; astli; astli = ASTLI_NEXT(astli)) {
        sptr = ASTLI_SPTR(astli);
        sptr = new_symbol(sptr);
        ast = ASTLI_TRIPLE(astli);
        ast = new_ast(ast);
        ASTLI_SPTR(astli) = sptr;
        ASTLI_TRIPLE(astli) = ast;
      }
      break;
    default:
      interr("new_astli: unsupport ast type", atype, 0);
    }
  }
  return astliz.base[offset].new;
}

/** \brief Look up an (old) shd.
  *
  * If this is the first lookup, process the specifiers
  * (asts) for each dimension.  After all of the specifiers have been processed,
  * create a new shape descriptor, stashing its index in the SHDITEM and return
  * the index to the new shape descriptor.
  */
static int
new_shape(int offset)
{
  SHDITEM *p_shd;
  int j;
  int cnt;
  int ast;

  if (offset == 0)
    return 0;

  p_shd = shdz.base + offset;
  if (p_shd->new == 0) {
    cnt = p_shd->ndim;
    for (j = 0; j < cnt; ++j) {
      ast = p_shd->shp[j].lwb;
      p_shd->shp[j].lwb = new_ast(ast);
      ast = p_shd->shp[j].upb;
      p_shd->shp[j].upb = new_ast(ast);
      ast = p_shd->shp[j].stride;
      p_shd->shp[j].stride = new_ast(ast);
    }
    add_shape_rank(cnt);
    for (j = 0; j < cnt; ++j)
      add_shape_spec(p_shd->shp[j].lwb, p_shd->shp[j].upb,
                     p_shd->shp[j].stride);
    p_shd->new = mk_shape();
  }
  return p_shd->new;
}

static void
fill_ST_MODULE(SYMITEM *ps, int sptr)
{
  int flags, bit;
  SYM save_sym0;

  save_sym0 = stb.stg_base[0];
#define GETBIT(f)                          \
  stb.stg_base[0].f = (flags & bit) ? 1 : 0; \
  bit <<= 1;
  flags = ps->flags1;
  bit = 1;
  GETBIT(f1);
  GETBIT(f2);
  GETBIT(f3);
  GETBIT(f4);
  GETBIT(f5);
  GETBIT(f6);
  GETBIT(f7);
  GETBIT(f8);
  GETBIT(f9);
  GETBIT(f10);
  GETBIT(f11);
  GETBIT(f12);
  GETBIT(f13);
  GETBIT(f14);
  GETBIT(f15);
  GETBIT(f16);
  GETBIT(f17);
  GETBIT(f18);
  GETBIT(f19);
  GETBIT(f20);
  GETBIT(f21);
  GETBIT(f22);
  GETBIT(f23);
  GETBIT(f24);
  GETBIT(f25);
  GETBIT(f26);
  GETBIT(f27);
  GETBIT(f28);
  GETBIT(f29);
  GETBIT(f30);
  GETBIT(f31);
  GETBIT(f32);
  flags = ps->flags2;
  bit = 1;
  GETBIT(f33);
  GETBIT(f34);
  GETBIT(f35);
  GETBIT(f36);
  GETBIT(f37);
  GETBIT(f38);
  GETBIT(f39);
  GETBIT(f40);
  GETBIT(f41);
  GETBIT(f42);
  GETBIT(f43);
  GETBIT(f44);
  GETBIT(f45);
  GETBIT(f46);
  GETBIT(f47);
  GETBIT(f48);
  GETBIT(f49);
  GETBIT(f50);
  GETBIT(f51);
  GETBIT(f52);
  GETBIT(f53);
  GETBIT(f54);
  GETBIT(f55);
  GETBIT(f56);
  GETBIT(f57);
  GETBIT(f58);
  GETBIT(f59);
  GETBIT(f60);
  GETBIT(f61);
  GETBIT(f62);
  GETBIT(f63);
  GETBIT(f64);
  flags = ps->flags3;
  bit = 1;
  GETBIT(f65);
  GETBIT(f66);
  GETBIT(f67);
  GETBIT(f68);
  GETBIT(f69);
  GETBIT(f70);
  GETBIT(f71);
  GETBIT(f72);
  GETBIT(f73);
  GETBIT(f74);
  GETBIT(f75);
  GETBIT(f76);
  GETBIT(f77);
  GETBIT(f78);
  GETBIT(f79);
  GETBIT(f80);
  GETBIT(f81);
  GETBIT(f82);
  GETBIT(f83);
  GETBIT(f84);
  GETBIT(f85);
  GETBIT(f86);
  GETBIT(f87);
  GETBIT(f88);
  GETBIT(f89);
  GETBIT(f90);
  GETBIT(f91);
  GETBIT(f92);
  GETBIT(f93);
  GETBIT(f94);
  GETBIT(f95);
  GETBIT(f96);
  flags = ps->flags4;
  bit = 1;
  GETBIT(f97);
  GETBIT(f98);
  GETBIT(f99);
  GETBIT(f100);
  GETBIT(f101);
  GETBIT(f102);
  GETBIT(f103);
  GETBIT(f104);
  GETBIT(f105);
  GETBIT(f106);
  GETBIT(f107);
  GETBIT(f108);
  GETBIT(f109);
  GETBIT(f110);
  GETBIT(f111);
  GETBIT(f112);
  GETBIT(f113);
  GETBIT(f114);
  GETBIT(f115);
  GETBIT(f116);
  GETBIT(f117);
  GETBIT(f118);
  GETBIT(f119);
  GETBIT(f120);
  GETBIT(f121);
  GETBIT(f122);
  GETBIT(f123);
  GETBIT(f124);
  GETBIT(f125);
  GETBIT(f126);
  GETBIT(f127);
  GETBIT(f128);
#undef GETBIT

  NEEDMODP(sptr, NEEDMODG(0));
  TYPDP(sptr, TYPDG(0));
  PRIVATEP(sptr, PRIVATEG(0));
  HAS_TBP_BOUND_TO_SMPP(sptr, HAS_TBP_BOUND_TO_SMPG(0));
  HAS_SMP_DECP(sptr, HAS_SMP_DECG(0)); 
  stb.stg_base[0] = save_sym0;

} /* fill_ST_MODULE */

static void
fill_sym(SYMITEM *ps, int sptr)
{
  int flags, bit;
  stb.stg_base[sptr].stype = ps->stype;
  stb.stg_base[sptr].sc = ps->sc;
  stb.stg_base[sptr].dtype = ps->dtype;
  switch (ps->stype) {
  case ST_ALIAS:
  case ST_MODPROC:
    stb.stg_base[sptr].symlk = 0;
    break;
  default:
    stb.stg_base[sptr].symlk = ps->symlk;
  }
#define GETBIT(f)                             \
  stb.stg_base[sptr].f = (flags & bit) ? 1 : 0; \
  bit <<= 1;
  flags = ps->flags1;
  bit = 1;
  GETBIT(f1);
  GETBIT(f2);
  GETBIT(f3);
  GETBIT(f4);
  GETBIT(f5);
  GETBIT(f6);
  GETBIT(f7);
  GETBIT(f8);
  GETBIT(f9);
  GETBIT(f10);
  GETBIT(f11);
  GETBIT(f12);
  GETBIT(f13);
  GETBIT(f14);
  GETBIT(f15);
  GETBIT(f16);
  GETBIT(f17);
  GETBIT(f18);
  GETBIT(f19);
  GETBIT(f20);
  GETBIT(f21);
  GETBIT(f22);
  GETBIT(f23);
  GETBIT(f24);
  GETBIT(f25);
  GETBIT(f26);
  GETBIT(f27);
  GETBIT(f28);
  GETBIT(f29);
  GETBIT(f30);
  GETBIT(f31);
  GETBIT(f32);
  flags = ps->flags2;
  bit = 1;
  GETBIT(f33);
  GETBIT(f34);
  GETBIT(f35);
  GETBIT(f36);
  GETBIT(f37);
  GETBIT(f38);
  GETBIT(f39);
  GETBIT(f40);
  GETBIT(f41);
  GETBIT(f42);
  GETBIT(f43);
  GETBIT(f44);
  GETBIT(f45);
  GETBIT(f46);
  GETBIT(f47);
  GETBIT(f48);
  GETBIT(f49);
  GETBIT(f50);
  GETBIT(f51);
  GETBIT(f52);
  GETBIT(f53);
  GETBIT(f54);
  GETBIT(f55);
  GETBIT(f56);
  GETBIT(f57);
  GETBIT(f58);
  GETBIT(f59);
  GETBIT(f60);
  GETBIT(f61);
  GETBIT(f62);
  GETBIT(f63);
  GETBIT(f64);
  flags = ps->flags3;
  bit = 1;
  GETBIT(f65);
  GETBIT(f66);
  GETBIT(f67);
  GETBIT(f68);
  GETBIT(f69);
  GETBIT(f70);
  GETBIT(f71);
  GETBIT(f72);
  GETBIT(f73);
  GETBIT(f74);
  GETBIT(f75);
  GETBIT(f76);
  GETBIT(f77);
  GETBIT(f78);
  GETBIT(f79);
  GETBIT(f80);
  GETBIT(f81);
  GETBIT(f82);
  GETBIT(f83);
  GETBIT(f84);
  GETBIT(f85);
  GETBIT(f86);
  GETBIT(f87);
  GETBIT(f88);
  GETBIT(f89);
  GETBIT(f90);
  GETBIT(f91);
  GETBIT(f92);
  GETBIT(f93);
  GETBIT(f94);
  GETBIT(f95);
  GETBIT(f96);
  flags = ps->flags4;
  bit = 1;
  GETBIT(f97);
  GETBIT(f98);
  GETBIT(f99);
  GETBIT(f100);
  GETBIT(f101);
  GETBIT(f102);
  GETBIT(f103);
  GETBIT(f104);
  GETBIT(f105);
  GETBIT(f106);
  GETBIT(f107);
  GETBIT(f108);
  GETBIT(f109);
  GETBIT(f110);
  GETBIT(f111);
  GETBIT(f112);
  GETBIT(f113);
  GETBIT(f114);
  GETBIT(f115);
  GETBIT(f116);
  GETBIT(f117);
  GETBIT(f118);
  GETBIT(f119);
  GETBIT(f120);
  GETBIT(f121);
  GETBIT(f122);
  GETBIT(f123);
  GETBIT(f124);
  GETBIT(f125);
  GETBIT(f126);
  GETBIT(f127);
  GETBIT(f128);
#undef GETBIT

#undef GETFIELD
#define GETFIELD(f) stb.stg_base[sptr].f = ps->sym.f
  GETFIELD(b3);
  GETFIELD(b4);
  GETFIELD(scope);
  /*GETFIELD(nmptr); don't use*/
  GETFIELD(w9);
  GETFIELD(w10);
  GETFIELD(w11);
  GETFIELD(w12);
  GETFIELD(w13);
  GETFIELD(w14);
  GETFIELD(w15);
  GETFIELD(w16);
  GETFIELD(w17);
  GETFIELD(w18);
  GETFIELD(w19);
  GETFIELD(w20);
  GETFIELD(w21);
  GETFIELD(w22);
  GETFIELD(w23);
  GETFIELD(w24);
  GETFIELD(w25);
  GETFIELD(w26);
  GETFIELD(w27);
  GETFIELD(w28);
  GETFIELD(uname);
  GETFIELD(w30);
  GETFIELD(w31);
  GETFIELD(w32);
  GETFIELD(w34);
  GETFIELD(w35);
  GETFIELD(w36);
  GETFIELD(lineno);
  GETFIELD(w39); 
  GETFIELD(w40); 
  GETFIELD(palign);
#undef GETFIELD
  stb.stg_base[sptr].uname = 0;
} /* fill_sym */

static void
import_constant(SYMITEM *ps)
{
  INT val[4];
  int sptr = 0;
  int dtype;

  Trace(("import_constant(%d)", ps->sptr));

  dtype = new_dtype(ps->dtype);
  /* just move the CONVAL fields into sym[0] */
  BCOPY(stb.stg_base, &ps->sym, SYM, 1);
  STYPEP(0, ST_CONST);
  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_WORD:
  case TY_REAL:
  case TY_DWORD:
  case TY_DBLE:
  case TY_CMPLX:
  case TY_INT8:
  case TY_LOG8:
    val[0] = CONVAL1G(0);
    val[1] = CONVAL2G(0);
    sptr = getcon(val, dtype);
    break;
  case TY_QUAD:
    val[0] = CONVAL1G(0);
    val[1] = CONVAL2G(0);
    val[2] = CONVAL3G(0);
    val[3] = CONVAL4G(0);
    sptr = getcon(val, dtype);
    break;
  case TY_CHAR:
    /* need to check dtype if length is 0 or 1, achar(0) has length of 1 */
    if (strlen(ps->strptr) == 0) {
      if (DTY(dtype + 1) == astb.k1)
        sptr = getstring(ps->strptr, 1);
      else
          if (DTY(dtype + 1) == astb.i1)
        sptr = getstring(ps->strptr, 1);
      else
        sptr = getstring(ps->strptr, strlen(ps->strptr));
    } else {
      sptr = getstring(ps->strptr, strlen(ps->strptr));
    }
    break;
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_HOLL:
  case TY_NCHAR:
    import_symbol(ps);
    return;
  case TY_PTR:
    if (CONVAL1G(0)) {
      /*
       * defer these for the import of pointer constants -- cannot
       * call import_symbol() because the symbol in the CONVAL1
       * may not yet been processed yet; therefore, wait until 'all'
       * of the symbols in the 'symbol_list' have been processed.
       */
      any_ptr_constant = TRUE;
      return;
    }
    break;
  default:
    interr("import_const: unknown constant datatype", dtype, 4);
    return;
  }
  ps->new_sptr = sptr;
  NMPTRP(sptr, 0);
  if (ps->sym.nmptr) {
    /* import the name also */
    NMPTRP(sptr, find_nmptr(ps->name));
  }
  Trace(("import_constant(%d) returning %d", ps->sptr, sptr));
} /* import_constant */

static void
import_symbol(SYMITEM *ps)
{
  SYMTYPE stype;
  SC_KIND sc;
  INT val[4];
  int sptr, s1, s2;
  LOGICAL set_dcld;

  stype = ps->stype;
  Trace(("import_symbol(%s=%d) with stype=%d", ps->name, ps->sptr, stype));
  set_dcld = FALSE;

  sc = ps->sc;

  if (stype == ST_CONST) {
    int dtype;
    /* just move the CONVAL fields into sym[0] */
    BCOPY(stb.stg_base, &ps->sym, SYM, 1);
    /* handle the rest of the constant cases */
    dtype = new_dtype(ps->dtype);
    switch (DTY(dtype)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
    case TY_WORD:
    case TY_REAL:
    case TY_DWORD:
    case TY_DBLE:
    case TY_CMPLX:
    case TY_INT8:
    case TY_LOG8:
    case TY_QUAD:
    case TY_CHAR:
      /* already handled */
      return;
    case TY_DCMPLX:
    case TY_QCMPLX:
      s1 = CONVAL1G(0);
      s2 = CONVAL2G(0);
      val[0] = new_symbol(s1);
      val[1] = new_symbol(s2);
      sptr = getcon(val, dtype);
      break;
    case TY_HOLL:
    case TY_NCHAR:
      s1 = CONVAL1G(0);
      val[0] = new_symbol(s1);
      val[1] = CONVAL2G(0);
      sptr = getcon(val, dtype);
      break;
    default:
      /* error message already issued */
      return;
    }
    ps->new_sptr = sptr;
    return;
  }

  /*
   * Same processing of labels for all types of import. Replace the label
   * with a compiler-created label for which astout will substitute with
   * a fortran label.
   */
  if (stype == ST_LABEL) {
    sptr = getlab();
    fill_sym(ps, sptr); /* copy flags, RFCNT, etc. */
    CCSYMP(sptr, 1);    /* reset fields defined by getlab() */
    SYMLKP(sptr, 0);
    ps->new_sptr = sptr;
    return;
  }

  sptr = getsymbol(ps->name);
  if (stype == ST_MODULE) {
    /* see if there is a module symbol somewhere in the hash link list */
    int s2;
    for (s2 = sptr; s2 > NOSYM; s2 = HASHLKG(s2)) {
      if (NMPTRG(s2) == NMPTRG(sptr) && STYPEG(s2) == ST_MODULE &&
          SCOPEG(s2) == SCOPEG(sptr)) {
        /* found it */
        sptr = s2;
        ps->new_sptr = sptr;
        fill_ST_MODULE(ps, sptr);
        return;
      }
    }
  } else if (module_base == 0 || sptr < module_base) {
    sptr = getocsym(sptr, stb.ovclass[stype], FALSE);
  }

  if (for_module == sptr && stype == ST_MODULE) {
    /* re-use the 'module' symbol, if there is one, when importing
     * modules */
  } else if (STYPEG(sptr) == ST_UNKNOWN && SCOPEG(sptr) == stb.curr_scope) {
    int s2;
    if (sptr < module_base) {
      /* don't reuse the 'unknown' symbol */
      SCOPEP(sptr, 0);
      IGNOREP(sptr, 0);
      /* this occurs when 'sptr' is from the 'rename' clause */
      s2 = sptr;
      sptr = insert_sym(sptr);
      if (flg.debug && s2)
        set_modusename(s2, sptr);
    }
    /* else reuse the 'unknown' symbol */
  } else if (for_host && sptr < BASEsym && SCOPEG(sptr) == stb.curr_scope &&
             ps->sym.scope == stb.curr_scope && STYPEG(sptr) == stype) {
    /* redefinition of the same symbol */
  } else if (stype == ST_MODULE && STYPEG(sptr) == ST_MODULE) {
    /* re-use 'module' symbol */
    ps->new_sptr = sptr;
    return;
  } else if (((stype == ST_ENTRY && !for_inliner) || stype == ST_PROC)) {
    /* find the appropriate procedure to use, if it's available */
    /* at this point, generics have been resolved in caller and callee */
    for (s2 = sptr; s2 > NOSYM; s2 = HASHLKG(s2)) {
      if (s2 < original_symavl && NMPTRG(s2) == NMPTRG(sptr) &&
          (STYPEG(s2) == ST_ENTRY || STYPEG(s2) == ST_PROC)) {
        /* name matches, right stype, check if it should be or is
         * a module procedure */
        if ((SCOPEG(s2) == 0 && ps->sym.scope == 0) ||
            (SCOPEG(s2) != 0 && ps->sym.scope != 0 &&
             new_symbol(ps->sym.scope) == SCOPEG(s2))) {
          if ((ENCLFUNCG(s2) == 0 && ps->sym.w28 == 0) ||
              (ENCLFUNCG(s2) != 0 && ps->sym.w28 != 0 &&
               new_symbol(ps->sym.w28) == ENCLFUNCG(s2))) {
            break;
          }
        }
      }
    }
    if (s2 > NOSYM) {
      sptr = s2;
    } else {
      sptr = insert_sym(sptr);
    }
  } else {
    int sptr1 = NOSYM;
    if (stype == ST_MODPROC && STYPEG(sptr) == ST_USERGENERIC) {
      /* Looking for  a MODPROC, found a USERGENERIC.  See if there is a
       * MODPROC with the same name
       */
      for (sptr1 = first_hash(sptr); sptr1 > NOSYM; sptr1 = HASHLKG(sptr1)) {
        if (NMPTRG(sptr) == NMPTRG(sptr1) && STYPEG(sptr1) == ST_MODPROC) {
          sptr = sptr1;
          break;
        }
      }
    }
    if (sptr1 <= NOSYM) {
      sptr = insert_sym(sptr);
    }
  }
  if (for_inliner && SCOPEG(sptr) == stb.curr_scope) {
    fill_sym(ps, sptr);
    if (!XBIT(126, 1) && ST_ISVAR(STYPEG(sptr)))
      SCOPEP(sptr, stb.curr_scope);
  } else {
    fill_sym(ps, sptr);
  }
  ps->new_sptr = sptr;
} /* import_symbol */

static void
import_ptr_constant(SYMITEM *ps)
{
  int stype;
  stype = ps->stype;
  Trace(
      ("import_ptr_constant(%s=%d) with stype=%d", ps->name, ps->sptr, stype));

  if (stype == ST_CONST) {
    int dtype;
    INT val[2];
    int sptr, s1;
    /* just move the CONVAL fields into sym[0] */
    BCOPY(stb.stg_base, &ps->sym, SYM, 1);
    /* handle the rest of the constant cases */
    dtype = new_dtype(ps->dtype);
    switch (DTY(dtype)) {
    case TY_PTR:
      s1 = CONVAL1G(0);
      if (s1) {
        val[0] = new_symbol(s1);
        val[1] = CONVAL2G(0);
        sptr = getcon(val, dtype);
        ps->new_sptr = sptr;
      }
      break;
    default:
      /* already handled */
      break;
    }
  }
} /* import_ptr_constant */

/** \brief Fill DTYPE, and symbol links to other symbol links */
static void
fill_links_symbol(SYMITEM *ps, WantPrivates wantPrivates)
{
  int ast, alias;
  int first, last;
  int mem, nml;
  int old_mem;
  int old_sptr, sptr, stype;
  DTYPE dtype; 

  old_sptr = ps->sptr;
  sptr = ps->new_sptr;
  stype = STYPEG(sptr);

  if (ps->dtype == 0 && CVLENG(sptr)
      /* make sure we do not clear CVLEN for CLASS since it may be
       * a type bound procedure that overloads CVLEN with VTOFF or
       * VTABLE. TBD: may need to revisit with unlimited polymorphic
       * types.
       */
      && (!CLASSG(sptr) || (stype != ST_MEMBER && stype != ST_PROC &&
                            stype != ST_USERGENERIC && stype != ST_OPERATOR))) {
    /* A function return value  or a subprog argument that is a
     * fixed length string.  Need to regenerate the dtype because
     * the dtype length is an ast that may not have been exported
     */
    int clen = CVLENG(sptr);
    int dty;
    /* HACK clen < 0 ==> TY_NCHAR */
    if (clen < 0) {
      clen = -clen;
      dty = TY_NCHAR;
    } else {
      dty = TY_CHAR;
    }
    dtype = get_type(2, dty, mk_cval(clen, DT_INT4));
    CVLENP(sptr, 0);
  } else {
    dtype = new_dtype(ps->dtype);
  }
  DTYPEP(sptr, dtype);

  switch (stype) {
  case ST_CONST:
    /*SLNKP(sptr, 0);??*/
    SYMLKP(sptr, NOSYM);
    break;
  case ST_PARAM:
    /*
     * TBD - for named array constants, there are two symbols with the
     * same name, an ST_PARAM and an ST_ARRAY.  The ST_PARAM's SYMLK
     * field locates the ST_ARRAY.  In a non-module context, the
     * semantic analyzer creates both of these symbols.  The second symbol
     * is not hashed into the symbol table; subsequent references of the
     * name constant during parsing always locate the ST_PARAM symbol.
     * When these symbols are read in from the module file, the
     * symbols are hashed, with the ST_ARRAY in front of the ST_PARAM.
     * This doesn't cause a problem except in the context where a
     * 'constant' is required.
     *
     * To fix, the ST_ARRAY needs to be 'unhashed',
     * but first, the data inits for the constant must find their way into
     * the module file --- TBD.
     */
    /*SLNKP(sptr, 0);???*/
    if (DTY(dtype) != TY_ARRAY) {
      ast = mk_id(sptr);
      if (!TY_ISWORD(DTY(dtype))) {
        CONVAL1P(sptr, new_symbol((int)CONVAL1G(sptr)));
      }
      if (DTY(dtype) < TY_PTR && A_ALIASG(ast) == 0) {
        alias = mk_cval1(CONVAL1G(sptr), DTYPEG(sptr));
        A_ALIASP(ast, alias);
      }
      /*
       * For DERIVED parameters, the CONVAL2 field was only meaningful
       * in its module file (and it's not an ast!!).  So, do not try
       * to create an ast; in fact, clear it.
       */
      ast = 0;
      if (DTY(dtype) != TY_DERIVED) {
        ast = new_ast((int)CONVAL2G(sptr)); /* ast of expression */
      }
      CONVAL2P(sptr, ast);
      if ((!IGNOREG(sptr)) && (!ignore_private || !PRIVATEG(sptr))) {
        if (!PRIVATEG(sptr) || wantPrivates == INCLUDE_PRIVATES) {
          add_param(sptr); /* add_param() sets SYMLK */
          end_param();
        }
      }
    } else
      CONVAL1P(sptr, new_symbol((int)CONVAL1G(sptr)));
    DCLDP(sptr, 1);
    break;
  case ST_UNKNOWN:
    ENCLFUNCP(sptr, 0);
    break;
  case ST_IDENT:
  case ST_VAR:
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_STRUCT:
  case ST_UNION:
    REFP(sptr, 0);
    if (for_interproc) {
      /* initialization information is not imported here */
      DINITP(sptr, 0);
    }
    if (for_inliner) {
      AUTOBJP(sptr, 0);
    }
    if (SLNKG(sptr) > NOSYM && can_find_symbol(SLNKG(sptr))) {
      SLNKP(sptr, new_symbol(SLNKG(sptr)));
    }
#if DEBUG
    /* aux.list[] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "fill_links_symbol: corrupted aux.list[]", sptr, 3);
#endif
    if (ps->socptr) {
      int sp;
      SOCPTRP(sptr, ps->socptr);
      for (sp = ps->socptr; sp; sp = SOC_NEXT(sp)) {
        SOC_SPTR(sp) = new_symbol(SOC_SPTR(sp));
      }
    }

    if (ps->sc == SC_CMBLK || CMBLKG(sptr)) {
      /* don't clear the SYMLK field for common block members */
      /* storage class for common members at this point could
       * be based, for instance, so don't just check ps->sc */
    } else if (ps->sc == SC_BASED || POINTERG(sptr)) {
      SYMLKP(sptr, NOSYM);
    } else if (!for_inliner && !for_static && !CFUNCG(sptr)) {
      SYMLKP(sptr, NOSYM);
      HIDDENP(sptr, 1);
    }
    if (ps->sc == SC_LOCAL)
      ADDRESSP(sptr, 0);
    DCLDP(sptr, 1);
    if (ps->sc == SC_DUMMY && IGNOREG(sptr)) {
      MIDNUMP(sptr, 0);
      DESCRP(sptr, 0);
      PTROFFP(sptr, 0);
      if (SDSCG(sptr)) {
        if (can_find_symbol(SDSCG(sptr)))
          SDSCP(sptr, new_symbol(SDSCG(sptr)));
        if (IGNOREG(SDSCG(sptr))) {
          SDSCP(sptr, 0);
        }
      }
    } else {
      if (MIDNUMG(sptr))
        MIDNUMP(sptr, new_symbol(MIDNUMG(sptr)));
      if (DESCRG(sptr))
        DESCRP(sptr, new_symbol(DESCRG(sptr)));
      if (PTROFFG(sptr))
        PTROFFP(sptr, new_symbol(PTROFFG(sptr)));
      if (SDSCG(sptr))
        SDSCP(sptr, new_symbol(SDSCG(sptr)));
    }

    if (ADJARRG(sptr) && SYMLKG(sptr) != NOSYM) {
      SYMLKP(sptr, new_symbol(SYMLKG(sptr)));
    }

    if (ADJLENG(sptr) && ADJSTRLKG(sptr) && ADJSTRLKG(sptr) != NOSYM) {
      ADJSTRLKP(sptr, new_symbol(ADJSTRLKG(sptr)));
    }
    if (PARAMVALG(sptr))
      PARAMVALP(sptr, new_ast(PARAMVALG(sptr)));
    if (NMCNSTG(sptr))
      NMCNSTP(sptr, new_symbol(NMCNSTG(sptr)));
    if (CVLENG(sptr))
      CVLENP(sptr, new_symbol(CVLENG(sptr)));
    if (CFUNCG(sptr) && ALTNAMEG(sptr)) {
      ALTNAMEP(sptr, new_symbol(ALTNAMEG(sptr)));
    }
    if (stype == ST_DESCRIPTOR && PARENTG(sptr) && CLASSG(sptr) &&
        can_find_dtype(PARENTG(sptr))) {
      PARENTP(sptr, new_dtype(PARENTG(sptr)));
    } else if (stype == ST_DESCRIPTOR && PARENTG(sptr) && CLASSG(sptr)) {
      PARENTP(sptr, 0);
    }
#ifdef DSCASTG
    if (stype != ST_DESCRIPTOR && DSCASTG(sptr))
      DSCASTP(sptr, new_ast(DSCASTG(sptr)));
#endif

    break;
  case ST_PLIST:
    /* SYMLK may need to be updated if it appears in a common block */
    if (for_interproc) {
      /* initialization information is not imported here */
      DINITP(sptr, 0);
    }
    break;
  case ST_CMBLK:
    DINITP(sptr, 0);
    /* process all elements of the common block */
    SYMLKP(sptr, gbl.cmblks);
    gbl.cmblks = sptr;

    first = last = 0;
    for (old_mem = CMEMFG(sptr); old_mem > NOSYM; old_mem = SYMLKG(mem)) {
      mem = new_symbol(old_mem);
      SCP(mem, SC_CMBLK);
      CMBLKP(mem, sptr);
      if (last)
        SYMLKP(last, mem);
      else
        first = mem;
      last = mem;
    }
    SYMLKP(last, NOSYM);
    CMEMFP(sptr, first);
    CMEMLP(sptr, last);
    CMBLKP(sptr, 0);
    if (ALTNAMEG(sptr))
      ALTNAMEP(sptr, new_symbol(ALTNAMEG(sptr)));
    break;
  case ST_PROC:
    if (ASSOC_PTRG(sptr)) {
      ASSOC_PTRP(sptr, new_symbol(ASSOC_PTRG(sptr)));
    }
    if (PTR_TARGETG(sptr)) {
      PTR_TARGETP(sptr, new_symbol(PTR_TARGETG(sptr)));
    }
    if (IS_PROC_DUMMYG(sptr) && SDSCG(sptr)) {
      SDSCP(sptr, new_symbol(SDSCG(sptr)));
    }
    if (FVALG(sptr) && can_find_symbol(FVALG(sptr))) {
      int fval;
      fval = new_symbol(FVALG(sptr));
      FVALP(sptr, fval);
      pop_sym(fval); /* never need to hash to return value name */
    }
    PARAMCTP(sptr, 0); /* TBD: fill in args */
    DPDSCP(sptr, 0);
    SYMLKP(sptr, NOSYM);
    if (FUNCG(sptr))
      DCLDP(sptr, 1); /* ensure functions are type declared */
    if (SLNKG(sptr) > NOSYM && can_find_symbol(SLNKG(sptr))) {
      SLNKP(sptr, new_symbol(SLNKG(sptr)));
    } 
#if DEBUG
    /* aux.list[ST_PROC] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "fill_links_symbol: corrupted aux.list[ST_PROC]", sptr, 3);
#endif
    if (GSAMEG(sptr))
      GSAMEP(sptr, new_symbol(GSAMEG(sptr)));
    if (for_interproc || for_static) {
      HIDDENP(sptr, 1);
    }
    if (ALTNAMEG(sptr))
      ALTNAMEP(sptr, new_symbol(ALTNAMEG(sptr)));
    if (SCOPEG(sptr) && can_find_symbol(SCOPEG(sptr)))
      SCOPEP(sptr, new_symbol(SCOPEG(sptr)));
    if (CLASSG(sptr) && TBPLNKG(sptr) && can_find_dtype(TBPLNKG(sptr))) {
      TBPLNKP(sptr, new_dtype(TBPLNKG(sptr)));
    }
    break;
  case ST_ENTRY:
    if (FVALG(sptr)) {
      int fval;
      fval = new_symbol(FVALG(sptr));
      FVALP(sptr, fval);
      if (NMPTRG(fval) == NMPTRG(sptr))
        pop_sym(fval);
    }
    PARAMCTP(sptr, 0); /* TBD: fill in args */
    DPDSCP(sptr, 0);
    SYMLKP(sptr, NOSYM);
    if (for_interproc || for_static) {
      HIDDENP(sptr, 1);
    }
    if (ALTNAMEG(sptr))
      ALTNAMEP(sptr, new_symbol(ALTNAMEG(sptr)));
    if (SCOPEG(sptr))
      SCOPEP(sptr, new_symbol(SCOPEG(sptr)));
    break;
  case ST_NML:
    /* link into the list of namelists */
    SYMLKP(sptr, sem.nml);
    sem.nml = sptr;
    /* the first namelist entry was stashed in ps->ty */
    CMEMFP(sptr, ps->ty);
    /* get new symbol numbers for each of the namelist members */
    for (nml = CMEMFG(sptr); nml; nml = NML_NEXT(nml)) {
      NML_SPTR(nml) = new_symbol(NML_SPTR(nml));
    }
    if (ADDRESSG(sptr)) {
      /* the PLIST for the namelist is stored here */
      ADDRESSP(sptr, new_symbol(ADDRESSG(sptr)));
    }
    break;
  case ST_USERGENERIC:
    /* these field are not valid, and we need them to be inited
       to zero to handle multiple generic interfaces with same
       name. */
    if (GTYPEG(sptr) && !GNCNTG(sptr)) {
      /* Remap overloaded type */
      GTYPEP(sptr, new_symbol(GTYPEG(sptr)));
    }
    GNDSCP(sptr, 0);
    GNCNTP(sptr, 0);
    if (CLASSG(sptr) && TBPLNKG(sptr) && can_find_dtype(TBPLNKG(sptr))) {
      TBPLNKP(sptr, new_dtype(TBPLNKG(sptr)));
    }
    FLANG_FALLTHROUGH;
  case ST_STFUNC:
  case ST_PD:
  case ST_ISOC:
    SYMLKP(sptr, NOSYM);
    break;
  case ST_INTRIN:
    switch (DTY(dtype)) {
    case TY_DCMPLX:
      GDCMPLXP(GNRINTRG(sptr), sptr);
      break;
    case TY_CMPLX:
      GCMPLXP(GNRINTRG(sptr), sptr);
      break;
    }
    break;

  case ST_LABEL:
    if (!CCSYMG(sptr))
      SYMLKP(sptr, NOSYM);
    break;
  case ST_TYPEDEF:
  case ST_STAG:
    SYMLKP(sptr, NOSYM);
    if (BASETYPEG(sptr))
      BASETYPEP(sptr, new_dtype(BASETYPEG(sptr)));
    PARENTP(sptr, new_symbol(PARENTG(sptr)));
    SDSCP(sptr, new_symbol(SDSCG(sptr)));
    if (TYPDEF_INITG(sptr) > NOSYM)
      TYPDEF_INITP(sptr, new_symbol(TYPDEF_INITG(sptr)));
    break;
  case ST_MEMBER:
    if (SYMLKG(sptr) == NOSYM) {
    } else if (SYMLKG(sptr) == old_sptr) {
      SYMLKP(sptr, sptr);
    } else {
      SYMLKP(sptr, new_symbol(SYMLKG(sptr)));
    }
    if (PSMEMG(sptr)) {
      if (PSMEMG(sptr) == old_sptr) {
        PSMEMP(sptr, sptr);
      } else {
        PSMEMP(sptr, new_symbol(PSMEMG(sptr)));
      }
    }
    if (VARIANTG(sptr) && VARIANTG(sptr) != NOSYM) {
      /* don't reinsert the parent member; parent must have been
       * inserted already */
      VARIANTP(sptr, new_symbol(VARIANTG(sptr)));
    }
    if (MIDNUMG(sptr))
      MIDNUMP(sptr, new_symbol(MIDNUMG(sptr)));
    if (DESCRG(sptr))
      DESCRP(sptr, new_symbol(DESCRG(sptr)));
    if (PTROFFG(sptr))
      PTROFFP(sptr, new_symbol(PTROFFG(sptr)));
    if (SDSCG(sptr))
      SDSCP(sptr, new_symbol(SDSCG(sptr)));
    if (ENCLDTYPEG(sptr))
      ENCLDTYPEP(sptr, new_dtype(ENCLDTYPEG(sptr)));
    if (PASSG(sptr)) {
      PASSP(sptr, new_symbol(PASSG(sptr)));
    }
    if (PARENTG(sptr)) {
      PARENTP(sptr, new_symbol(PARENTG(sptr)));
    }
    if (VTABLEG(sptr)) {
      VTABLEP(sptr, new_symbol(VTABLEG(sptr)));
    }
    if (IFACEG(sptr)) {
      IFACEP(sptr, new_symbol(IFACEG(sptr)));
    }
    if (BINDG(sptr)) {
      BINDP(sptr, new_symbol(BINDG(sptr)));
    }
    if (LENG(sptr) && LENPARMG(sptr)) {
      LENP(sptr, new_ast(LENG(sptr)));
    }
    if (INITKINDG(sptr) && PARMINITG(sptr)) {
      PARMINITP(sptr, new_ast(PARMINITG(sptr)));
    }
    if (KINDASTG(sptr)) {
      KINDASTP(sptr, new_ast(KINDASTG(sptr)));
    }

    break;
  case ST_OPERATOR:
    if (INKINDG(sptr))
      bind_intrinsic_opr(PDNUMG(sptr), sptr);
    /* these field are not valid, and we need them to be inited
       to zero to handle multiple generic interfaces with same
       name. */
    GNDSCP(sptr, 0);
    GNCNTP(sptr, 0);
    SYMLKP(sptr, NOSYM);
    if (CLASSG(sptr) && TBPLNKG(sptr) && can_find_dtype(TBPLNKG(sptr))) {
      TBPLNKP(sptr, new_dtype(TBPLNKG(sptr)));
    }
    break;
  case ST_ARRDSC:
    SECDSCP(sptr, new_symbol(SECDSCG(sptr)));
    if (ARRAYG(sptr))
      ARRAYP(sptr, new_symbol(ARRAYG(sptr)));
    SYMLKP(sptr, NOSYM);
    break;
  case ST_ALIAS:
    SYMLKP(sptr, new_symbol(ps->symlk));
    if (PRIVATEG(sptr)) {
    }
    if (GSAMEG(sptr))
      GSAMEP(sptr, new_symbol(GSAMEG(sptr)));
    break;
  case ST_MODULE:
    break;
  case ST_MODPROC:
    if (ps->symlk)
      SYMLKP(sptr, new_symbol(ps->symlk));
    if (GSAMEG(sptr))
      GSAMEP(sptr, new_symbol(GSAMEG(sptr)));
    if (SYMLKG(sptr) && GSAMEG(sptr)) {
      /* if for modules, don't do this twice for module symbols */
      if (!for_module || SYMLKG(sptr) < module_base)
        GSAMEP(SYMLKG(sptr), GSAMEG(sptr)); /* ST_ENTRY -> generic */
      /* only do this for modules, and don't do twice for module symbols */
      if (for_module && GSAMEG(sptr) < module_base)
        GSAMEP(GSAMEG(sptr), SYMLKG(sptr)); /* generic -> ST_ENTRY */
      /* this line was removed because processing generics
       * also sets GSAMEP, and we can't do this twice */
    }
    break;

  case ST_BLOCK:
    if (STARTLABG(sptr))
      STARTLABP(sptr, new_symbol(STARTLABG(sptr)));
    if (ENDLABG(sptr))
      ENDLABP(sptr, new_symbol(ENDLABG(sptr)));
    break;

  default:
    interr("new_symbol:unexp stype", ps->stype, 3);
    break;
  }
  if (ENCLFUNCG(sptr) && can_find_symbol(ENCLFUNCG(sptr)))
    ENCLFUNCP(sptr, new_symbol(ENCLFUNCG(sptr)));

  if (ps->sym.scope) {
    if (for_module || for_host) {
      SCOPEP(sptr, new_symbol(ps->sym.scope));
    } else if (for_static) {
      SCOPEP(sptr, new_symbol_if_module(ps->sym.scope));
    }
  }
} /* fill_links_symbol */

static int
new_symbol(int old_sptr)
{
  SYMITEM *ps;
  int sptr;

  sptr = map_initsym(old_sptr, import_osym);
  if (sptr)
    return sptr;
  ps = findhash(old_sptr);
  if (ps)
    return ps->new_sptr;
  if (old_sptr == HOST_OLDSCOPE)
    return HOST_NEWSCOPE;
  if (old_sptr < BASEmod)
    return old_sptr;
  if (old_sptr < BASEsym)
    return old_sptr + ADJmod;
  interr("interf:new_symbol, symbol not found", old_sptr, 4);
  return 0;
} /* new_symbol */

static int
new_symbol_if_module(int old_sptr)
{
  SYMITEM *ps;
  int sptr, newsptr;

  sptr = map_initsym(old_sptr, import_osym);
  if (sptr)
    return 0;

  newsptr = 0;
  ps = findhash(old_sptr);
  if (ps) {
    if (ps->stype == ST_MODULE) {
      return ps->new_sptr;
    } else {
      return 0;
    }
  }
  if (old_sptr < BASEmod) {
    if (STYPEG(old_sptr) == ST_MODULE) {
      return old_sptr;
    }
  } else if (old_sptr < BASEsym) {
    if (STYPEG(old_sptr + ADJmod) == ST_MODULE) {
      return old_sptr + ADJmod;
    }
  }
  return 0;
} /* new_symbol_if_module */

static void
new_symbol_and_link(int old_sptr, int *pnew, SYMITEM **pps)
{
  SYMITEM *ps;
  int sptr;

  sptr = map_initsym(old_sptr, import_osym);
  if (sptr) {
    if (pnew)
      *pnew = sptr;
    if (pps)
      *pps = NULL;
    return;
  }

  ps = findhash(old_sptr);
  if (ps) {
    if (pnew)
      *pnew = ps->new_sptr;
    if (pps)
      *pps = ps;
    return;
  }
  if (old_sptr < BASEmod) {
    if (pnew)
      *pnew = old_sptr;
    if (pps)
      *pps = NULL;
    return;
  }
  if (old_sptr < BASEsym) {
    if (pnew)
      *pnew = old_sptr + ADJmod;
    if (pps)
      *pps = NULL;
    return;
  }
  interr("interf:new_symbol_and_link, symbol not found", old_sptr, 4);
} /* new_symbol_and_link */

#ifdef FLANG_INTERF_UNUSED
static SYMITEM *
find_symbol(int old_sptr)
{
  SYMITEM *ps;

  ps = findhash(old_sptr);
  if (ps)
    return ps;
#if DEBUG
  Trace(("cannot find old symbol %d in file %s", old_sptr, import_file_name));
  for (ps = symbol_list; ps != NULL; ps = ps->next) {
    Trace(("symbol list is %8lx = %4d (%4d) %s", ps, ps->sptr, ps->new_sptr,
           ps->name));
  }
  interr("module:find_symbol,stnfd", old_sptr, 0);
#endif
  return symbol_list;
}
#endif

static int
can_find_symbol(int old_sptr)
{
  SYMITEM *ps;

  ps = findhash(old_sptr);
  if (ps)
    return 1;
  return 0;
}

static int
can_find_dtype(int old_dt)
{
  DITEM *pd;
  pd = finddthash(old_dt);
  if (pd)
    return 1;
  return 0;
}

#ifdef FLANG_INTERF_UNUSED
/** \brief Ensure that the common blocks from the interface file do not already
  * exist in the subprogram or if they do, their elements match.
  *
  * \return 0 if there aren't any conflicts; return the sptr to the common
  * block which conflicts.
  */
static int
common_conflict(void)
{
  int cmblk, diff, prevcmblk, nextcmblk;
  SYMITEM *ps;

  for (ps = symbol_list; ps; ps = ps->next) {
    if (ps->stype == ST_CMBLK) {
      for (cmblk = gbl.cmblks; cmblk != NOSYM; cmblk = SYMLKG(cmblk)) {
        if (!IGNOREG(cmblk) && strcmp(ps->name, SYMNAME(cmblk)) == 0) {
          Trace(("COMMON/%s/ already declared at symbol %d", ps->name, cmblk));
          diff = install_common(ps, cmblk);
          if (diff)
            return diff;
        }
      }
    }
  }
  prevcmblk = 0;
  for (cmblk = gbl.cmblks; cmblk > NOSYM; cmblk = nextcmblk) {
    nextcmblk = SYMLKG(cmblk);
    if (!IGNOREG(cmblk)) {
      /* keep cmblk on gbl.cmblks list */
      prevcmblk = cmblk;
    } else {
      /* remove cmblk from gbl.cmblks list */
      if (prevcmblk) {
        SYMLKP(prevcmblk, nextcmblk);
      } else {
        gbl.cmblks = nextcmblk;
      }
      SYMLKP(cmblk, NOSYM);
    }
  }
  return 0;
} /* common_conflict */
#endif

#ifdef FLANG_INTERF_UNUSED
/** \brief Compare the existing common block with the common block from the
  * interface file. Install the members while they match.
  *
  * NOTE: use sym entry 0 to hold the contents of the symbol from the
  * interface file so that the symtab macros can be used.
  */
static int
install_common(SYMITEM *pscmblk, int cmblk)
{
  SYMITEM *ps, *psfirst;
  int sptr;
  BCOPY(stb.stg_base, &pscmblk->sym, SYM, 1);
  ps = psfirst = find_symbol(CMEMFG(0));
  sptr = CMEMFG(cmblk);
  for (ps = psfirst; TRUE; ps = find_symbol(ps->symlk)) {
    if (!common_mem_eq(DTYPEG(sptr), new_dtype(ps->dtype)))
      goto common_diff;
    /* if end of inlined cblock reached, then ok: */
    if (ps->symlk == NOSYM)
      break;

    /*  check for inlined common block longer than pre-existing: */
    sptr = SYMLKG(sptr);
    if (sptr == NOSYM)
      goto common_diff;
  }
  /* the same */
  if (pscmblk->new_sptr) {
    IGNOREP(pscmblk->new_sptr, 1);
    HIDDENP(pscmblk->new_sptr, 1);
  }
  pscmblk->new_sptr = cmblk;
  pscmblk->sc = -1;
  sptr = CMEMFG(cmblk);
  for (ps = psfirst; TRUE; ps = find_symbol(ps->symlk)) {
    if (ps->new_sptr) {
      IGNOREP(ps->new_sptr, 1);
      HIDDENP(ps->new_sptr, 1);
    }
    ps->new_sptr = sptr;
    ps->sc = -1;
    /* if end of inlined cblock reached, then ok: */
    if (ps->symlk == NOSYM)
      break;
    sptr = SYMLKG(sptr);
  }

  BZERO(stb.stg_base, SYM, 1);
  return 0;

common_diff:
  BZERO(stb.stg_base, SYM, 1);
  return cmblk;
} /* install_common */
#endif

#ifdef FLANG_INTERF_UNUSED
/** \brief return TRUE if two data types are equal.
  *
  * This function only needs to handle dtype situations resulting
  * from commonblock elements.
  */
static LOGICAL
common_mem_eq(int d1, int d2)
{
  ADSC *ad1, *ad2;
  int n;

  if (d1 == d2)
    return TRUE;

  if (DTY(d1) != TY_ARRAY && DTY(d2) != TY_ARRAY)
    return FALSE;

  if (DTY(d1 + 1) != DTY(d2 + 1))
    return FALSE; /* element types not the same */

  ad1 = AD_DPTR(d1);
  ad2 = AD_DPTR(d2);
  n = AD_NUMDIM(ad1);
  if (n != AD_NUMDIM(ad2))
    return FALSE;

  while (--n >= 0) {
    if (AD_UPAST(ad1, n) != AD_UPAST(ad2, n) ||
        AD_LWAST(ad1, n) != AD_LWAST(ad2, n))
      return FALSE; /* dimensions don't match */
  }

  return TRUE;
}
#endif

static int
import_mk_newsym(char *name, int stype)
{
  int sptr;

  sptr = getsymbol(name);
  /* if this is ST_UNKNOWN, or is a MODULE and we want a MODULE, use it.
   * otherwise, insert a new symbol */
  if (STYPEG(sptr) != ST_UNKNOWN &&
      (STYPEG(sptr) != ST_MODULE || stype != ST_MODULE))
    sptr = insert_sym(sptr);
  STYPEP(sptr, stype);
  SCOPEP(sptr, 0);

  return sptr;
}

void rw_import_state(RW_ROUTINE, RW_FILE)
{
  int nw;
  int i;
  int nodecnt;
  USES_LIST *usenode, *prevnode;
  TOBE_IMPORTED_LIST *um;

  RW_SCALAR(imported_modules.avail);
  /* since the imported_modules.list is never actually freed or
   * shrunk, it should already be of the proper size */
  RW_FD(imported_modules.list, IMPORT_LIST, imported_modules.avail);

  /* save/restore the use_tree root (root is actually a list).  If this is
   * a read, then
   * freed so the USES_LIST items still exist and can be re-used.
   */
  if (!ISREAD()) {
    for (nodecnt = 0, usenode = use_tree; usenode;
         usenode = usenode->next, nodecnt++)
      ;
  }

  RW_SCALAR(nodecnt);
  prevnode = NULL;
  usenode = use_tree;
  if (ISREAD())
    init_use_tree();

  for (i = 0; i < nodecnt; i++) {
    /* since the MOD_USE_AREA has not been deallocated, the TOBE_IMPORTED_LIST
     * nodes
     * still exist.  Just save and restore ptrs to them.
     */
    if (!ISREAD()) {
      um = usenode->use_module;
      prevnode = usenode;
      usenode = usenode->next;
    }
    RW_SCALAR(um);
    if (ISREAD()) {
      add_to_use_tree(um);
    }
  }

} /* rw_import_state */

/* ----------------------------------------------------------------- */

static int
ipa_ast(int a)
{
  return new_ast(a);
}

static int
dindex(int dtype)
{
  return new_dtype(dtype);
}

static int
get_symbolxref(int sptr)
{
  return new_symbol(sptr);
}

