/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Main module to generate LLVM debug informations using metadata
 */

#include "lldebug.h"
#include "dtypeutl.h"
#include "global.h"
#include "symtab.h"
#include "ll_structure.h"
#include "ll_builder.h"
#include "dwarf2.h"
#include "error.h"
#include "version.h"
#include "fih.h"
#include "llassem.h"
#include "cgllvm.h"
#include "cgmain.h"
#include "flang/ADT/hash.h"
#include "symfun.h"
#define RTE_C
#include "rte.h"
#undef RTE_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef HOST_WIN
#include <unistd.h>
#endif

#if defined(_WIN64) && !defined(PATH_MAX)
#define PATH_MAX 260
#endif
#ifdef _WIN64
#include <direct.h>
#endif
#include "upper.h"

#ifdef __cplusplus
/* clang-format off */
inline SPTR GetParamSptr(int dpdsc, int i) {
  return static_cast<SPTR>(aux.dpdsc_base[dpdsc + i]);
}
/* clang-format on */
#else
#define GetParamSptr(dpdsc, i) (aux.dpdsc_base[dpdsc + i])
#endif

#if !defined(DECLLINEG)
#define DECLLINEG(sptr) 0
#endif

#ifndef DW_TAG_auto_variable
#define DW_TAG_auto_variable 0x100
#endif

#ifndef DW_TAG_arg_variable
#define DW_TAG_arg_variable 0x101
#endif

#ifndef DW_TAG_return_variable
#define DW_TAG_return_variable 0x102
#endif

#ifndef DW_TAG_vector_type
#define DW_TAG_vector_type 0x103
#endif

const int DIFLAG_ARTIFICIAL = 1 << 6;
const int DIFLAG_ISMAINPGM = 1 << 21; // removed in release_90
static int DIFLAG_PURE;      // removed in release_80
static int DIFLAG_ELEMENTAL; // removed in release_80
static int DIFLAG_RECURSIVE; // removed in release_80

const int DISPFLAG_LOCALTOUNIT = 1 << 2; // added in release_80
const int DISPFLAG_DEFINITION = 1 << 3;  // added in release_80
const int DISPFLAG_OPTIMIZED = 1 << 4;   // added in release_80
const int DISPFLAG_PURE = 1 << 5;        // added in release_80
const int DISPFLAG_ELEMENTAL = 1 << 6;   // added in release_80
const int DISPFLAG_RECURSIVE = 1 << 7;   // added in release_80
const int DISPFLAG_MAINSUBPROGRAM = 1 << 8; // added in release_90

typedef struct {
  LL_MDRef mdnode; /**< mdnode for block */
  int sptr;        /**< block sptr */
  int startline;
  int endline;
  int keep;
  LL_MDRef *line_mdnodes; /**< mdnodes for block lines */
  LL_MDRef null_loc;
} BLKINFO;

typedef struct {
  LL_MDRef mdnode;
  INSTR_LIST *instr;
  int sptr;
} PARAMINFO;

struct sptr_to_mdnode_map {
  int sptr;
  LL_MDRef mdnode;
  struct sptr_to_mdnode_map *next;
};

typedef struct import_entity {
  SPTR entity;                 /**< sptr of pending import entity */
  IMPORT_TYPE entity_type;     /**< 0 for DECLARATION; 1 for MODULE; 2 for UNIT */
  SPTR func;                   /**< sptr of function import to */
  struct import_entity *next;  /**< pointer to the next node */
  struct import_entity *child; /**< pointer to the child node */
} import_entity;

#define BLK_STACK_SIZE 1024
#define PARAM_STACK_SIZE 1024

struct LL_DebugInfo {
  LL_Module *module;           /**< Pointer to the containing LL_Module */
  LL_MDRef llvm_dbg_sp;        /**< List of subprogram mdnodes */
  LL_MDRef llvm_dbg_gv;        /**< List of global variables mdnodes */
  LL_MDRef llvm_dbg_retained;  /**< List of retained type mdnodes */
  LL_MDRef llvm_dbg_enum;      /**< List of enum mdnodes */
  LL_MDRef llvm_dbg_imported;  /**< List of imported entity mdnodes */
  LL_MDRef *llvm_dbg_lv_array; /**< List of formal parameters to routine */
  char producer[1024];
  LL_MDRef comp_unit_mdnode;
  LL_MDRef *file_array;
  int file_array_sz;
  LL_MDRef cur_subprogram_mdnode;
  unsigned cur_subprogram_func_ptr_offset;
  LL_MDRef cur_parameters_mdnode;
  LL_MDRef cur_module_mdnode;
  LL_MDRef cur_cmnblk_mdnode;
  int cur_subprogram_lineno;
  LL_MDRef cur_subprogram_line_mdnode;
  LL_MDRef cur_subprogram_null_loc;
  LL_MDRef cur_line_mdnode;
  PARAMINFO param_stack[PARAM_STACK_SIZE];
  LL_MDRef *dtype_array;
  int dtype_array_sz;
  LL_MDRef texture_type_mdnode;

  BLKINFO cur_blk;
  BLKINFO *blk_tab;
  int blk_tab_size;
  int blk_idx;
  char *cur_module_name;

  int param_idx;
  int routine_count;
  int routine_idx;

  struct sptr_to_mdnode_map *sptrs_to_mdnodes;
  hashmap_t subroutine_mdnodes;
  hashset_t entity_func_added;
  import_entity *import_entity_list; /**< list of entities to be imported to func */
  bool need_dup_composite_type; /**< indicator of duplicate composite type when needed */
  SPTR gbl_var_sptr; /**< current global variable symbol */
  LL_MDRef gbl_obj_mdnode; /**< mdnode reference to the global object in STATICS */
  LL_MDRef gbl_obj_exp_mdnode; /**< the expression mdnode of the above global object */

  unsigned scope_is_global : 1;
};

static LL_MDRef lldbg_emit_modified_type(LL_DebugInfo *, DTYPE, SPTR, int);
#ifdef FLANG_DEBUGINFO_UNUSED
static LL_MDRef lldbg_create_module_flag_mdnode(LL_DebugInfo *db, int severity,
                                                const char *name, int value);
#endif
static LL_MDRef lldbg_create_outlined_parameters_node(LL_DebugInfo *db);
static LL_MDRef lldbg_create_file_mdnode(LL_DebugInfo *db, const char *filename,
                                         char *sourcedir, LL_MDRef context,
                                         int index);
static LL_MDRef lldbg_emit_type(LL_DebugInfo *db, DTYPE dtype, SPTR sptr,
                                int findex, bool is_reference,
                                bool skip_first_dim,
                                bool skipDataDependentTypes,
                                SPTR data_sptr = SPTR_NULL);
static LL_MDRef lldbg_fwd_local_variable(LL_DebugInfo *db, int sptr, int findex,
                                         int emit_dummy_as_local);
static LL_MDRef lldbg_create_imported_entity(LL_DebugInfo *db, SPTR entity_sptr,
                                             SPTR func_sptr,
                                             IMPORT_TYPE entity_type,
                                             LL_MDRef elements_mdnode,
                                             LL_MDRef imported_list);
static void lldbg_emit_imported_entity(LL_DebugInfo *db, SPTR entity_sptr,
                                       SPTR func_sptr, IMPORT_TYPE entity_type,
                                       LL_MDRef elements, LL_MDRef imported_list);
static LL_MDRef lldbg_create_subrange_mdnode(LL_DebugInfo *db, LL_MDRef count,
                                             LL_MDRef lb, LL_MDRef ub,
                                             LL_MDRef st);
static LL_MDRef lldbg_create_generic_subrange_mdnode(LL_DebugInfo *db,
                                                     LL_MDRef lb, LL_MDRef ub,
                                                     LL_MDRef st);
static LL_MDRef lldbg_create_subrange_via_sdsc(LL_DebugInfo *db, int findex,
                                               SPTR sptr, int rank);
static void lldbg_get_bounds_for_sdsc(LL_DebugInfo *db, int findex, SPTR sptr,
                                      int rank, LL_MDRef *count_expr_mdnode,
                                      LL_MDRef *lbnd_expr_mdnode,
                                      LL_MDRef *ubnd_expr_mdnode,
                                      LL_MDRef *stride_expr_mdnode);

static void lldbg_get_bounds_for_assumed_rank_sdsc(
    LL_DebugInfo *db, SPTR sptr, LL_MDRef *lbnd_expr_mdnode,
    LL_MDRef *ubnd_expr_mdnode, LL_MDRef *stride_expr_mdnode);
static void lldbg_register_param_mdnode(LL_DebugInfo *db, LL_MDRef mdnode,
                                        int sptr);
INLINE static int set_dilocalvariable_flags(int sptr);
/* ---------------------------------------------------------------------- */

void
InitializeDIFlags(const LL_IRFeatures *feature)
{
#ifdef FLANG_LLVM_EXTENSIONS
  if (ll_feature_debug_info_ver70(feature)) {
    DIFLAG_PURE = 1 << 27;
    DIFLAG_ELEMENTAL = 1 << 28;
    DIFLAG_RECURSIVE = 1 << 29;
  } else {
    DIFLAG_PURE = 1 << 22;
    DIFLAG_ELEMENTAL = 1 << 23;
    DIFLAG_RECURSIVE = 1 << 24;
  }
#else
  // do nothing
#endif
}

char *
lldbg_alloc(INT size)
{
  char *p = (char *)getitem(LLVM_LONGTERM_AREA, size);
  assert(p, "lldbg_alloc(), out of memory", 0, ERR_Fatal);
  memset(p, 0, size);
  return p;
}

int
lldbg_get_di_routine_idx(const LL_DebugInfo *db)
{
  return db->routine_idx;
}

static ISZ_T
lldbg_get_sizeof(int element)
{
  ISZ_T sz;
#if defined(FLDSZG)
  if (FIELDG(element) && FLDSZG(element))
    return FLDSZG(element);
#endif
  if (!element || !DTYPEG(element))
    return 0;
  sz = zsize_of(DTYPEG(element)) * 8;
  if (sz < 0)
    sz = 0;
  return sz;
}

/**
   \brief Make an i32 operand with a DWARF tag and possibly a version number.

   LLVM 3.6 onwards do not encode the debug info version in the tag field.
 */
static int
make_dwtag(LL_DebugInfo *db, int tag)
{
  if (ll_feature_versioned_dw_tag(&db->module->ir))
    tag |= db->module->ir.debug_info_version << 16;
  return tag;
}

static LL_MDRef
lldbg_create_module_flag_mdnode(LL_DebugInfo *db, int severity,
                                const char *name, int value)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_add_i32(mdb, severity);
  llmd_add_string(mdb, name);
  llmd_add_i32(mdb, value);

  return llmd_finish(mdb);
}

static LL_MDRef
get_file_mdnode(LL_DebugInfo *db, int index)
{
  if (index * 2 < db->file_array_sz)
    return db->file_array[index * 2];
  return ll_get_md_null();
}

static LL_MDRef
get_filedesc_mdnode(LL_DebugInfo *db, int index)
{
  if ((index * 2 + 1) < db->file_array_sz)
    return db->file_array[index * 2 + 1];
  return ll_get_md_null();
}

static LL_MDRef
lldbg_create_compile_unit_mdnode(LL_DebugInfo *db, int lang_tag,
                                 const char *filename, char *sourcedir,
                                 char *producer, int main, bool optimized,
                                 const char *compflags, int vruntime,
                                 LL_MDRef *enum_types_list,
                                 LL_MDRef *retained_types_list,
                                 LL_MDRef *subprograms_list, LL_MDRef *gv_list,
                                 LL_MDRef *imported_entity_list)
{
  LLMD_Builder mdb = llmd_init(db->module);
  LL_MDRef cur_mdnode;

  llmd_set_class(mdb, LL_DICompileUnit);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_compile_unit));

  if (ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_i32(mdb, 0); /* Unused field. */
    llmd_add_i32(mdb, lang_tag);
    llmd_add_string(mdb, filename);
    llmd_add_string(mdb, sourcedir);
  } else {
    LL_MDRef file_mdnode = get_filedesc_mdnode(db, 1);
    if (LL_MDREF_IS_NULL(file_mdnode))
      file_mdnode = lldbg_create_file_mdnode(db, filename, sourcedir,
                                             ll_get_md_null(), 1);
    llmd_add_md(mdb, file_mdnode);
    llmd_add_i32(mdb, lang_tag);
  }

  llmd_add_string(mdb, producer);
  if (ll_feature_debug_info_pre34(&db->module->ir))
    llmd_add_i1(mdb, main);
  llmd_add_i1(mdb, optimized);
  llmd_add_string(mdb, compflags);
  llmd_add_i32(mdb, vruntime);

  *enum_types_list = ll_create_flexible_md_node(db->module);
  *retained_types_list = ll_create_flexible_md_node(db->module);
  *subprograms_list = ll_create_flexible_md_node(db->module);
  *gv_list = ll_create_flexible_md_node(db->module);
  *imported_entity_list = ll_create_flexible_md_node(db->module);

  if (ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_md(mdb,
                ll_get_md_node(db->module, LL_PlainMDNode, enum_types_list, 1));
    llmd_add_md(mdb, ll_get_md_node(db->module, LL_PlainMDNode,
                                    retained_types_list, 1));
    llmd_add_md(
        mdb, ll_get_md_node(db->module, LL_PlainMDNode, subprograms_list, 1));
    llmd_add_md(mdb, ll_get_md_node(db->module, LL_PlainMDNode, gv_list, 1));
  } else {
    llmd_add_md(mdb, *enum_types_list);
    llmd_add_md(mdb, *retained_types_list);
    if (!ll_feature_subprogram_not_in_cu(&db->module->ir))
      llmd_add_md(mdb, *subprograms_list);
    llmd_add_md(mdb, *gv_list);
    if (ll_feature_subprogram_not_in_cu(&db->module->ir))
      llmd_add_i32(mdb, 1); /* emissionMode: FullDebug */
    llmd_add_md(mdb, *imported_entity_list);
    llmd_add_string(mdb, "");
    if (!XBIT(120, 0x40000000))
      llmd_add_i32(mdb, 2); /* nameTableKind: None */
  }

  llmd_set_distinct(mdb);
  cur_mdnode = llmd_finish(mdb);
  ll_extend_named_md_node(db->module, MD_llvm_dbg_cu, cur_mdnode);

  return cur_mdnode;
}

static LL_MDRef
lldbg_create_module_mdnode(LL_DebugInfo *db, LL_MDRef _, char *name,
                           LL_MDRef scope, LL_MDRef file, int lineno)
{
  LLMD_Builder mdb;
  char *module_name, *pname, *pmname;
  unsigned tag = ll_feature_debug_info_pre34(&db->module->ir) ? DW_TAG_namespace
                                                              : DW_TAG_module;

  if (name && db->cur_module_name && !strcmp(name, db->cur_module_name))
    return db->cur_module_mdnode;

  mdb = llmd_init(db->module);
  module_name = (char *)lldbg_alloc(strlen(name) + 1);
  pname = name;
  pmname = module_name;
  while (*pname != '\0') {
    *pmname = tolower(*pname);
    pname++;
    pmname++;
  }
  *pmname = '\0'; /* append null char to end of string */

  if (ll_feature_no_file_in_namespace(&db->module->ir)) {
    // Use the DIModule template
    llmd_set_class(mdb, LL_DIModule);
    llmd_add_i32(mdb, make_dwtag(db, DW_TAG_module)); // tag
    llmd_add_md(mdb, scope);                          // scope
    llmd_add_string(mdb, module_name);                // name
    if (ll_feature_debug_info_ver11(&db->module->ir)) {
      llmd_add_md(mdb, file);                         // file
      llmd_add_i32(mdb, lineno);                      // lineno
    }
  } else {
    llmd_set_class(mdb, LL_DINamespace);
    llmd_add_i32(mdb, make_dwtag(db, tag));
    if (!ll_feature_debug_info_pre34(&db->module->ir))
      llmd_add_md(mdb, scope);
    llmd_add_null(mdb);
    llmd_add_string(mdb, module_name);
    if (ll_feature_debug_info_pre34(&db->module->ir))
      llmd_add_md(mdb, scope);
    llmd_add_i32(mdb, lineno);
  }
  db->cur_module_name = module_name;
  db->cur_module_mdnode = llmd_finish(mdb);
  if (flg.debug && ll_feature_no_file_in_namespace(&db->module->ir))
    ll_add_module_debug(cpu_llvm_module->module_debug_map, module_name,
                        db->cur_module_mdnode);
  return db->cur_module_mdnode;
}

static LL_MDRef
lldbg_create_file_mdnode(LL_DebugInfo *db, const char *filename,
                         char *sourcedir, LL_MDRef context, int index)
{
  LLMD_Builder mdb = llmd_init(db->module);
  LL_MDRef cur_mdnode;

  llmd_set_class(mdb, LL_DIFile);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_file_type));

  if (!ll_feature_debug_info_pre34(&db->module->ir)) {
    LLMD_Builder pairmd = llmd_init(db->module);
    llmd_set_class(pairmd, LL_DIFile);
    llmd_add_string(pairmd, filename);
    llmd_add_string(pairmd, sourcedir);
    cur_mdnode = llmd_finish(pairmd);

    llmd_add_md(mdb, cur_mdnode);
    NEEDB(fihb.stg_avail * 2, db->file_array, LL_MDRef, db->file_array_sz,
          fihb.stg_avail * 2);
    db->file_array[2 * index] = llmd_finish(mdb);
    db->file_array[2 * index + 1] = cur_mdnode;
  } else {
    llmd_add_string(mdb, filename);
    llmd_add_string(mdb, sourcedir);
    llmd_add_md(mdb, context);

    cur_mdnode = llmd_finish(mdb);
    NEEDB(fihb.stg_avail * 2, db->file_array, LL_MDRef, db->file_array_sz,
          fihb.stg_avail * 2);
    db->file_array[2 * index] = cur_mdnode;
    db->file_array[2 * index + 1] = ll_get_md_null();
  }
  return cur_mdnode;
}

/**
 * \brief Create a sub-program mdnode and store it in db->cur_subprogram_mdnode
 *
 * Don't set the function pointer field, but remember where it goes in
 * cur_subprogram_func_ptr_offset.
 */
static void
lldbg_create_subprogram_mdnode(
    LL_DebugInfo *db, LL_MDRef context, const char *routine,
    const char *mips_linkage_name, LL_MDRef def_context, int line,
    LL_MDRef type_mdnode, int is_local, int is_definition, int virtuality,
    int vindex, int spFlags, int flags, bool is_optimized,
    LL_MDRef template_param_mdnode, LL_MDRef decl_desc_mdnode,
    LL_MDRef lv_list_mdnode, int scope)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DISubprogram);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_subprogram));

  if (ll_feature_debug_info_pre34(&db->module->ir))
    llmd_add_i32(mdb, 0);
  else
    llmd_add_md(mdb, def_context);
  llmd_add_md(mdb, context);
  llmd_add_string(mdb, routine);
  llmd_add_string(mdb, routine);
  llmd_add_string(mdb, mips_linkage_name);
  if (ll_feature_debug_info_pre34(&db->module->ir))
    llmd_add_md(mdb, def_context);
  llmd_add_i32(mdb, line);
  llmd_add_md(mdb, type_mdnode);
  if (!ll_feature_debug_info_ver90(&db->module->ir)) {
    llmd_add_i1(mdb, is_local); // removed in release_90
    llmd_add_i1(mdb, is_definition);
  }
  llmd_add_i32(mdb, virtuality);
  llmd_add_i32(mdb, vindex);
  llmd_add_null(mdb);
  llmd_add_i32(mdb, flags);
  if (ll_feature_debug_info_ver90(&db->module->ir)) {
    llmd_add_i32(mdb, spFlags);
  } else {
    llmd_add_i1(mdb, is_optimized);
  }

  /* The actual function pointer is inserted here later by
   * lldbg_set_func_ptr(). */
  db->cur_subprogram_func_ptr_offset = llmd_get_nelems(mdb);
  llmd_add_null(mdb);

  llmd_add_md(mdb, template_param_mdnode);
  llmd_add_md(mdb, decl_desc_mdnode);
  if (ll_feature_subprogram_not_in_cu(&db->module->ir))
    llmd_add_md(mdb, db->comp_unit_mdnode);

  /* Add extra layer of indirection before 3.4. */
  if (!ll_feature_debug_info_ver70(&db->module->ir)) {
    if (ll_feature_debug_info_pre34(&db->module->ir))
      llmd_add_md(mdb,
          ll_get_md_node(db->module, LL_PlainMDNode, &lv_list_mdnode, 1));
    else
      llmd_add_md(mdb, lv_list_mdnode);
  } else if (ll_feature_debug_info_ver13(&db->module->ir))
    llmd_add_md(mdb, lv_list_mdnode);

  llmd_add_i32(mdb, scope);

  /* Request a distinct mdnode so that it can be updated with a function pointer
   * later. */
  llmd_set_distinct(mdb);
  db->cur_subprogram_mdnode = llmd_finish(mdb);
  ll_extend_md_node(db->module, db->llvm_dbg_sp, db->cur_subprogram_mdnode);
}

void
lldbg_set_func_ptr(LL_DebugInfo *db, LL_Value *func_ptr)
{
  LL_MDRef mdref = ll_get_md_value(db->module, func_ptr);
  ll_update_md_node(db->module, db->cur_subprogram_mdnode,
                    db->cur_subprogram_func_ptr_offset, mdref);
}

LL_MDRef
lldbg_subprogram(LL_DebugInfo *db)
{
  LL_MDRef rv = db->cur_subprogram_mdnode;
  db->cur_subprogram_mdnode = (LL_MDRef)0;
  return rv;
}

void
lldbg_reset_module(LL_DebugInfo *db)
{
  db->cur_module_name = NULL;
  db->cur_module_mdnode = ll_get_md_null();
}

static LL_MDRef
lldbg_create_global_variable_mdnode(LL_DebugInfo *db, LL_MDRef context,
                                    const char *display_name, const char *name,
                                    const char *mips_linkage_name,
                                    LL_MDRef def_context, int line,
                                    LL_MDRef type_mdnode, int is_local,
                                    int is_definition, LL_Value *var_ptr,
                                    int addrspace, int flags, ISZ_T off,
                                    SPTR sptr, LL_MDRef fwd)
{
  LLMD_Builder mdb = llmd_init(db->module);
  LL_MDRef cur_mdnode;

  llmd_set_class(mdb, LL_DIGlobalVariable);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_variable));
  llmd_add_i32(mdb, 0);
  llmd_add_md(mdb, context);
#ifdef FLANG_LLVM_EXTENSIONS
  if (ll_feature_debug_info_ver70(&db->module->ir) && 
      !XBIT(183, 0x40000000) &&
      (flags & DIFLAG_ARTIFICIAL))
    display_name = ""; // Do not expose the name of compiler created variable.
#endif
  llmd_add_string(mdb, display_name);
  llmd_add_string(mdb, name);
  llmd_add_string(mdb, mips_linkage_name);
  llmd_add_md(mdb, def_context);
  llmd_add_i32(mdb, line);
  llmd_add_md(mdb, type_mdnode);
  llmd_add_i32(mdb, is_local);
  llmd_add_i32(mdb, is_definition);
  if (!ll_feature_from_global_to_md(&db->module->ir))
    llmd_add_md(mdb, ll_get_md_value(db->module, var_ptr));
#ifdef FLANG_LLVM_EXTENSIONS
  if (ll_feature_debug_info_ver70(&db->module->ir))
    llmd_add_i32(mdb, flags);
#endif
  if (addrspace >= 0)
    llmd_add_i32(mdb, addrspace);

  if (ll_feature_from_global_to_md(&db->module->ir))
    llmd_set_distinct(mdb);
  cur_mdnode = ll_finish_variable(mdb, fwd);

  if (ll_feature_from_global_to_md(&db->module->ir)) {
    LL_MDRef expr_mdnode;
    const ISZ_T off0 = ((off >> 27) == 0) ? off : 0;
    const unsigned v = lldbg_encode_expression_arg(LL_DW_OP_int, off0);
    const unsigned cnt = (off0 > 0) ? 2 : 0;
    LLMD_Builder mdb2 = llmd_init(db->module);
    llmd_set_class(mdb2, LL_DIGlobalVariableExpression);
    llmd_add_md(mdb2, cur_mdnode);
    if (!ll_feature_debug_info_ver90(&cpu_llvm_module->ir) &&
        ftn_array_need_debug_info(sptr)) {
      /* Handle the Fortran allocatable array cases. Emit expression mdnode with
       * a sigle argument of DW_OP_deref because of using sptr array$p instead
       * of sptr array for debugging purpose.
       */
      const unsigned deref = lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
      expr_mdnode = lldbg_emit_expression_mdnode(db, 1, deref);
    } else
    if (ll_feature_use_5_diexpression(&db->module->ir)) {
      const unsigned add = lldbg_encode_expression_arg(LL_DW_OP_plus_uconst, 0);
      expr_mdnode = lldbg_emit_expression_mdnode(db, cnt, add, v);
    } else {
      const unsigned add = lldbg_encode_expression_arg(LL_DW_OP_plus, 0);
      expr_mdnode = lldbg_emit_expression_mdnode(db, cnt, add, v);
    }
    llmd_add_md(mdb2, expr_mdnode);
    cur_mdnode = llmd_finish(mdb2);
  }

  ll_extend_md_node(db->module, db->llvm_dbg_gv, cur_mdnode);
  return cur_mdnode;
}

static LL_MDRef
lldbg_create_block_mdnode(LL_DebugInfo *db, LL_MDRef routine_context, int line,
                          int column, int findex, int ID)
{
  LLMD_Builder mdb = llmd_init(db->module);

  if (!line)
    line = 1;

  llmd_set_class(mdb, LL_DILexicalBlock);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_lexical_block));
  if (!ll_feature_debug_info_pre34(&db->module->ir))
    llmd_add_md(mdb, get_filedesc_mdnode(db, findex));
  llmd_add_md(mdb, routine_context);
  llmd_add_i32(mdb, line);
  llmd_add_i32(mdb, column);
  if (ll_feature_debug_info_pre34(&db->module->ir))
    llmd_add_md(mdb, get_file_mdnode(db, findex));
  llmd_add_i32(mdb, ID);

  return llmd_finish(mdb);
}

INLINE static LL_MDRef
lldbg_create_string_type_mdnode(LL_DebugInfo *db, ISZ_T sz, DBLINT64 alignment,
                                const char *name, int encoding)
{
  LLMD_Builder mdb = llmd_init(db->module);

  if (ll_feature_has_diextensions(&db->module->ir)) {
    llmd_set_class(mdb, LL_DIStringType);
  } else {
    llmd_set_class(mdb, LL_DIBasicType_string);
  }
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_string_type));
  llmd_add_string(mdb, name);
  llmd_add_i64(mdb, sz);
  llmd_add_INT64(mdb, alignment);
  if (!ll_feature_has_diextensions(&db->module->ir)) {
    llmd_add_i32(mdb, encoding);
  }
  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_basic_type_mdnode(LL_DebugInfo *db, LL_MDRef context,
                               const char *name, LL_MDRef fileref, int line,
                               ISZ_T sz, DBLINT64 alignment, DBLINT64 offset,
                               int flags, int dwarf_encoding)
{
  DBLINT64 size;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(sz, size);
  llmd_set_class(mdb, LL_DIBasicType);

  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_base_type));
  if (ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_md(mdb, context);
    llmd_add_string(mdb, name);
    llmd_add_md(mdb, fileref);
  } else {
    llmd_add_null(mdb);
    llmd_add_null(mdb);
    llmd_add_string(mdb, name);
  }
  llmd_add_i32(mdb, line);
  llmd_add_INT64(mdb, size);
  llmd_add_INT64(mdb, alignment);
  llmd_add_INT64(mdb, offset);
  llmd_add_i32(mdb, flags);
  llmd_add_i32(mdb, dwarf_encoding);

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_pointer_type_mdnode(LL_DebugInfo *db, LL_MDRef context,
                                 const char *name, LL_MDRef fileref, int line,
                                 ISZ_T sz, DBLINT64 alignment, DBLINT64 offset,
                                 int flags, LL_MDRef pts_to)
{
  DBLINT64 size;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(sz, size);
  llmd_set_class(mdb, LL_DIDerivedType);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_pointer_type));

  if (!ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_null(mdb);
    llmd_add_null(mdb);
    llmd_add_string(mdb, name);
  } else {
    llmd_add_md(mdb, context);
    llmd_add_string(mdb, name);
    llmd_add_md(mdb, fileref);
  }
  llmd_add_i32(mdb, line);
  llmd_add_INT64(mdb, size);
  llmd_add_INT64(mdb, alignment);
  llmd_add_INT64(mdb, offset);
  llmd_add_i32(mdb, flags);
  llmd_add_md(mdb, pts_to);

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_ftn_array_type_mdnode(LL_DebugInfo *db, LL_MDRef context, int line,
                                   ISZ_T sz, DBLINT64 alignment, LL_MDRef eleTy,
                                   LL_MDRef subscripts)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DIFortranArrayType);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_array_type));
  llmd_add_md(mdb, context);
  llmd_add_i32(mdb, line);
  llmd_add_i64(mdb, sz);
  llmd_add_INT64(mdb, alignment);
  llmd_add_md(mdb, eleTy);
  llmd_add_md(mdb, subscripts);

  return llmd_finish(mdb);
}

/**
   \brief Create an array type, \c DW_TAG_array_type
   \param db
   \param context
   \param line       line number
   \param sz       size of array, must be in bits
   \param alignment  alignment of array
   \param pts_to
   \param subscripts
   \param data_location
   \param associated
   \param allocated
   \param rank
 */
static LL_MDRef
lldbg_create_array_type_mdnode(LL_DebugInfo *db, LL_MDRef context, int line,
                               ISZ_T sz, DBLINT64 alignment, LL_MDRef pts_to,
                               LL_MDRef subscripts, LL_MDRef data_location,
                               LL_MDRef associated, LL_MDRef allocated,
                               LL_MDRef rank)
{
  DBLINT64 size;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(sz, size);
  llmd_set_class(mdb, LL_DICompositeType);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_array_type));

  if (!ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_null(mdb);
    llmd_add_null(mdb);
    llmd_add_string(mdb, "");
  } else {
    llmd_add_md(mdb, context);
    llmd_add_string(mdb, "");
    llmd_add_md(mdb, context);
  }
  llmd_add_i32(mdb, line);
  llmd_add_INT64(mdb, size);
  llmd_add_INT64(mdb, alignment);
  llmd_add_i32(mdb, 0);
  llmd_add_i32(mdb, 0);
  llmd_add_md(mdb, pts_to);
  llmd_add_md(mdb, subscripts);
  llmd_add_i32(mdb, 0);
  if (ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_i32(mdb, 0);
  } else {
    llmd_add_null(mdb);
    llmd_add_null(mdb);
  }
  if (ll_feature_debug_info_ver90(&db->module->ir)) {
    if (!LL_MDREF_IS_NULL(data_location)) {
      llmd_add_null(mdb);
      llmd_add_md(mdb, data_location);
    } else {
      llmd_add_null(mdb);
      llmd_add_null(mdb);
    }
    if (!LL_MDREF_IS_NULL(associated))
      llmd_add_md(mdb, associated);
    else
      llmd_add_null(mdb);
    if (!LL_MDREF_IS_NULL(allocated))
      llmd_add_md(mdb, allocated);
    else
      llmd_add_null(mdb);
    if (!LL_MDREF_IS_NULL(rank))
      llmd_add_md(mdb, rank);
    else
      llmd_add_null(mdb);
  }

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_aggregate_type_mdnode(LL_DebugInfo *db, int dw_tag,
                                   LL_MDRef context, const char *name,
                                   LL_MDRef fileref, int line, ISZ_T sz,
                                   DBLINT64 alignment, int flags, LL_MDRef members,
                                   int runtime)
{
  DBLINT64 size;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(sz, size);
  llmd_set_class(mdb, LL_DICompositeType);
  llmd_add_i32(mdb, make_dwtag(db, dw_tag));

  if (!ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_md(mdb, fileref);
    llmd_add_null(mdb); /* Here should be non compile unit scope */
    llmd_add_string(mdb, name);
    llmd_add_i32(mdb, line);
    llmd_add_INT64(mdb, size);
    llmd_add_INT64(mdb, alignment);
    llmd_add_i32(mdb, 0);
    llmd_add_i32(mdb, flags);
    llmd_add_null(mdb); /* Derived from ? */
    llmd_add_md(mdb, members);
    llmd_add_i32(mdb, runtime);
    llmd_add_null(mdb); /* Virtual table holder ? */
    llmd_add_null(mdb);
    llmd_add_null(mdb); /* Unique identifier ? */
  } else {
    llmd_add_md(mdb, context);
    llmd_add_string(mdb, name);
    llmd_add_md(mdb, fileref);
    llmd_add_i32(mdb, line);
    llmd_add_INT64(mdb, size);
    llmd_add_INT64(mdb, alignment);
    llmd_add_i32(mdb, 0);
    llmd_add_i32(mdb, flags);
    llmd_add_i32(mdb, 0);
    llmd_add_md(mdb, members);
    llmd_add_i32(mdb, runtime);
    llmd_add_i32(mdb, 0);
  }

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_structure_type_mdnode(LL_DebugInfo *db, LL_MDRef context,
                                   const char *name, LL_MDRef fileref, int line,
                                   ISZ_T sz, DBLINT64 alignment, int flags,
                                   LL_MDRef members, int runtime)
{
  return lldbg_create_aggregate_type_mdnode(db, DW_TAG_structure_type, context,
                                            name, fileref, line, sz, alignment,
                                            flags, members, runtime);
}

static LL_MDRef
lldbg_create_union_type_mdnode(LL_DebugInfo *db, LL_MDRef context,
                               const char *name, LL_MDRef fileref, int line,
                               ISZ_T sz, DBLINT64 alignment, int flags,
                               LL_MDRef members, int runtime)
{
  return lldbg_create_aggregate_type_mdnode(db, DW_TAG_union_type, context,
                                            name, fileref, line, sz, alignment,
                                            flags, members, runtime);
}

static LL_MDRef
lldbg_create_member_mdnode(LL_DebugInfo *db, LL_MDRef fileref,
                           LL_MDRef parent_mdnode, const char *name, int line,
                           ISZ_T sz, DBLINT64 alignment, DBLINT64 offset, int flags,
                           LL_MDRef type_mdnode, LL_MDRef fwd)
{
  DBLINT64 size;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(sz, size);
  llmd_set_class(mdb, LL_DIDerivedType);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_member));
  llmd_add_md(mdb, fileref);
  if (ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_string(mdb, name);
    llmd_add_md(mdb, fileref);
  } else {
    llmd_add_md(mdb, parent_mdnode);
    llmd_add_string(mdb, name);
  }
  llmd_add_i32(mdb, line);
  llmd_add_INT64(mdb, size);
  llmd_add_INT64(mdb, alignment);
  llmd_add_INT64(mdb, offset);
  llmd_add_i32(mdb, flags);
  llmd_add_md(mdb, type_mdnode);

  return ll_finish_variable(mdb, fwd);
}

static void
lldbg_create_aggregate_members_type(LL_DebugInfo *db, SPTR first, int findex,
                                    LL_MDRef file_mdnode,
                                    LL_MDRef members_mdnode,
                                    LL_MDRef parent_mdnode)
{
  LL_MDRef member_mdnode, member_type_mdnode, fwd;
  ISZ_T sz;
  DBLINT64 align, offset;
  SPTR element, member;
  DTYPE elem_dtype;
  hash_data_t val;
  bool is_desc_member = false;
  bool contains_allocatable = false;
  SPTR base_sptr = SPTR_NULL;

  if (!ll_feature_debug_info_pre34(&db->module->ir))
    file_mdnode = get_filedesc_mdnode(db, findex);
  for (element = first; element > NOSYM; element = SYMLKG(element)) {
    if (CCSYMG(element))
      continue;
    if (SCG(element) == SC_BASED && POINTERG(element)) {
      /* member syms linked list could contain descriptor(s):
       * case #1: array type sym "foo" followed by "foo$p", "foo$o", "foo$sd";
       * case #2: structure type sym "zar" followed by "zar$p", "zar$td";
       * case #3: pointer type sym "bar" followed by "bar$p".
       */
      base_sptr = element;
      if (ALLOCATTRG(element) && SDSCG(element)) {
        if (!ll_feature_debug_info_ver90(&db->module->ir) && db->gbl_var_sptr) {
          contains_allocatable = true;
          db->need_dup_composite_type |= true;
        }
      } else {
        if (!SDSCG(element))
          element = SYMLKG(element);
        assert(element > NOSYM,
               "lldbg_create_aggregate_members_type: element not exists",
               element, ERR_Fatal);
        is_desc_member = true;
        if (!ll_feature_debug_info_ver90(&db->module->ir)) {
          db->need_dup_composite_type = false;
        }
      }
    }
    elem_dtype = DTYPEG(element);
    sz = lldbg_get_sizeof(element);
    align[1] = ((alignment(elem_dtype) + 1) * 8);
    align[0] = 0;
    offset[1] = ((ADDRESSG(element)) * 8);
    offset[0] = 0;
    member_type_mdnode =
        lldbg_emit_type(db, elem_dtype, element, findex, false, false, false);
    if (hashmap_lookup(db->module->mdnodes_fwdvars, INT2HKEY(element), &val)) {
      fwd = (LL_MDRef)(unsigned long)val;
      hashmap_erase(db->module->mdnodes_fwdvars, INT2HKEY(element), NULL);
    } else {
      fwd = ll_get_md_null();
    }
    member = element;
    if (is_desc_member) {
      member = base_sptr;
      is_desc_member = false;
    }
    if (!ll_feature_debug_info_ver90(&db->module->ir) && contains_allocatable) {
      db->need_dup_composite_type |= true;
    }
    if (base_sptr && SDSCG(element)) {
      /* for case #1 and case #2, sdsc always appears as the last descriptor.
       * skip all of the subsequent descriptors.
       */
      element = SDSCG(element);
      base_sptr = SPTR_NULL;
    }
    member_mdnode = lldbg_create_member_mdnode(
        db, file_mdnode, parent_mdnode, CCSYMG(member) ? "" : SYMNAME(member),
        0, sz, align, offset, 0, member_type_mdnode, fwd);
    ll_extend_md_node(db->module, members_mdnode, member_mdnode);
  }
}

#ifdef FLANG_DEBUGINFO_UNUSED
static bool
map_sptr_to_mdnode(LL_MDRef *mdnode, LL_DebugInfo *db, int sptr)
{
  struct sptr_to_mdnode_map *map = db->sptrs_to_mdnodes;
  if (sptr > NOSYM) {
    for (; map != NULL; map = map->next) {
      if (map->sptr == sptr) {
        if (mdnode != NULL)
          *mdnode = map->mdnode;
        return true;
      }
    }
  }
  return false;
}
#endif

/**
   \brief Fill in extra data about a symbol

   Probes any debug symbol information that may have been saved by the front-end
   in order to generate a good display name and proper namespace or class scope
   for a symbol.  Falls back to the symbol table name and the compile unit's
   outermost scope if better symbolic information cannot be found.
 */
static void
get_extra_info_for_sptr(const char **display_name, LL_MDRef *scope_mdnode,
                        LL_MDRef *type_mdnode, LL_DebugInfo *db, int sptr)
{
  *display_name = SYMNAME(sptr);
  if (scope_mdnode != NULL) {
    if (db->cur_cmnblk_mdnode != ll_get_md_null())
      *scope_mdnode = db->cur_cmnblk_mdnode;
    else
    if (db->cur_subprogram_mdnode != ll_get_md_null())
      *scope_mdnode = db->cur_subprogram_mdnode;
    else if (db->cur_module_mdnode != ll_get_md_null() &&
             ((STYPEG(sptr) != ST_ENTRY) ||
              (STYPEG(sptr) == ST_ENTRY && INMODULEG(sptr))))
      *scope_mdnode = db->cur_module_mdnode;
    else
      *scope_mdnode = lldbg_emit_compile_unit(db);
  }

}

#ifdef FLANG_DEBUGINFO_UNUSED
static LL_MDRef
lldbg_create_enumeration_type_mdnode(LL_DebugInfo *db, LL_MDRef context,
                                     char *name, LL_MDRef fileref, int line,
                                     ISZ_T sz, DBLINT64 alignment,
                                     LL_MDRef elements)
{
  return lldbg_create_aggregate_type_mdnode(
      db, DW_TAG_enumeration_type, context, name, fileref, line, sz, alignment,
      /*flags=*/0, elements, /*runtime=*/0);
}

static LL_MDRef
lldbg_create_enumerator_mdnode(LL_DebugInfo *db, int sptr, DBLINT64 value)
{
  LLMD_Builder mdb = llmd_init(db->module);
  const char *name;

  llmd_set_class(mdb, LL_DIEnumerator);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_enumerator));
  /* TODO(pmk): this fails to find pretty names for enumeration members */
  get_extra_info_for_sptr(&name, NULL /* scope */, NULL /* type */, db, sptr);
  llmd_add_string(mdb, name);
  llmd_add_INT64(mdb, value);

  return llmd_finish(mdb);
}

/**
   Create an mdnode that is a list of enumerators, starting from element in the
   symbol table.
 */
static LL_MDRef
lldbg_create_enumerator_list(LL_DebugInfo *db, int element)
{
  LLMD_Builder mdb = llmd_init(db->module);

  /* empty enum: required for version  3.7, works for all */
  if (element <= NOSYM) {
    llmd_add_md(mdb, ll_get_md_null());
  } else
    while (element > NOSYM) {
      DBLINT64 value;
      value[0] = CONVAL1G(element);
      value[1] = CONVAL2G(element);
      llmd_add_md(mdb, lldbg_create_enumerator_mdnode(db, element, value));
      element = SYMLKG(element);
    }

  /* The symbol table linked list has the enumerators in backwards order. */
  llmd_reverse(mdb);

  return llmd_finish(mdb);
}
#endif

static LL_MDRef
lldbg_create_vector_type_mdnode(LL_DebugInfo *db, LL_MDRef context, ISZ_T sz,
                                DBLINT64 alignment, LL_MDRef type,
                                LL_MDRef subscripts)
{
  DBLINT64 size;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(sz, size);
  llmd_set_class(mdb, LL_DICompositeType);
  /* vector types are marked as arrays in LLVM debug information. */
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_array_type));
  llmd_add_null(mdb);
  llmd_add_null(mdb);
  llmd_add_string(mdb, "");
  llmd_add_i32(mdb, 0);
  llmd_add_INT64(mdb, size);
  llmd_add_INT64(mdb, alignment);
  llmd_add_i32(mdb, 0);
  llmd_add_i32(mdb, 0);
  llmd_add_md(mdb, type);
  llmd_add_md(mdb, subscripts);
  llmd_add_i32(mdb, 0);
  llmd_add_null(mdb);
  llmd_add_null(mdb);

  return llmd_finish(mdb);
}

#ifdef FLANG_DEBUGINFO_UNUSED
static LL_MDRef
lldbg_create_derived_type_mdnode(LL_DebugInfo *db, int dw_tag, LL_MDRef context,
                                 char *name, LL_MDRef fileref, int line,
                                 ISZ_T sz, DBLINT64 alignment, DBLINT64 offset,
                                 int flags, LL_MDRef derived)
{
  DBLINT64 size;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(sz, size);
  llmd_set_class(mdb, LL_DIDerivedType);
  llmd_add_i32(mdb, make_dwtag(db, dw_tag));
  llmd_add_md(mdb, context);
  if (ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_string(mdb, name);
    llmd_add_md(mdb, fileref);
  } else {
    llmd_add_md(mdb, fileref);
    llmd_add_string(mdb, name);
  }
  llmd_add_i32(mdb, line);
  llmd_add_INT64(mdb, size);
  llmd_add_INT64(mdb, alignment);
  llmd_add_INT64(mdb, offset);
  llmd_add_i32(mdb, flags);
  llmd_add_md(mdb, derived);

  return llmd_finish(mdb);
}
#endif

static LL_MDRef
lldbg_create_subroutine_type_mdnode(LL_DebugInfo *db, LL_MDRef context,
                                    LL_MDRef fileref, LL_MDRef params,
                                    const int cc)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DISubroutineType);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_subroutine_type));
  llmd_add_i32(mdb, 0);
  llmd_add_null(mdb);
  llmd_add_string(mdb, "");
  llmd_add_i32(mdb, 0);
  llmd_add_i64(mdb, 0);
  llmd_add_i64(mdb, 0);
  llmd_add_i64(mdb, 0);
  llmd_add_i32(mdb, 0);
  llmd_add_null(mdb);
  llmd_add_md(mdb, params);
  llmd_add_i32(mdb, 0);
  llmd_add_null(mdb);
  llmd_add_null(mdb);
  if (ll_feature_subprogram_not_in_cu(&db->module->ir))
    llmd_add_i32(mdb, cc);

  return llmd_finish(mdb);
}

static LL_MDRef
emit_deref_expression_mdnode(LL_DebugInfo *db)
{
  const unsigned deref = lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
  return lldbg_emit_expression_mdnode(db, 1, deref);
}

static LL_MDRef
lldbg_create_ftn_subrange_mdnode(LL_DebugInfo *db, ISZ_T clb, LL_MDRef lbv,
                                 ISZ_T cub, LL_MDRef ubv)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DIFortranSubrange);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_subrange_type));

  if (LL_MDREF_IS_NULL(lbv) && LL_MDREF_IS_NULL(ubv)) {
    llmd_add_i64(mdb, clb);
    llmd_add_i64(mdb, cub);
    llmd_add_null(mdb);
    llmd_add_null(mdb);
    llmd_add_null(mdb);
    llmd_add_null(mdb);
  } else if (LL_MDREF_IS_NULL(lbv)) {
    llmd_add_i64(mdb, clb);
    llmd_add_i64(mdb, 0);
    llmd_add_null(mdb);
    llmd_add_null(mdb);
    llmd_add_md(mdb, ubv);
    llmd_add_md(mdb, emit_deref_expression_mdnode(db));
  } else if (LL_MDREF_IS_NULL(ubv)) {
    llmd_add_i64(mdb, 0);
    llmd_add_i64(mdb, cub);
    llmd_add_md(mdb, lbv);
    llmd_add_md(mdb, emit_deref_expression_mdnode(db));
    llmd_add_null(mdb);
    llmd_add_null(mdb);
  } else {
    llmd_add_i64(mdb, 0);
    llmd_add_i64(mdb, 0);
    llmd_add_md(mdb, lbv);
    llmd_add_md(mdb, emit_deref_expression_mdnode(db));
    llmd_add_md(mdb, ubv);
    llmd_add_md(mdb, emit_deref_expression_mdnode(db));
  }
  return llmd_finish(mdb);
}

#define F90_DESC_SIZE 10 /* num of fields of Struct F90_Desc */
#define F90_DESCDIM_SIZE 6 /* num of fields of Struct F90_DescDim */
/* Create subrange mdnode based on array descriptor */
static LL_MDRef
lldbg_create_ftn_subrange_via_sdsc(LL_DebugInfo *db, int findex, SPTR sptr,
                                   int rank)
{
  LL_MDRef array_desc_mdnode, lbnd_expr_mdnode, ubnd_expr_mdnode;

  const int gbl_offset = (db->gbl_var_sptr && db->need_dup_composite_type &&
                         ADDRESSG(db->gbl_var_sptr) > 0 ) ?
                         ADDRESSG(db->gbl_var_sptr) : 0;
  /* array descrpitor object's offset within an aggregate object needs to be
   * counted in, if exists. */
  const int orig_offset = (SCG(SDSCG(sptr)) == SC_CMBLK || 
                           STYPEG(SDSCG(sptr)) == ST_MEMBER) ?
                           ADDRESSG(SDSCG(sptr)) : 0;
  const int lbnd_offset = gbl_offset + orig_offset +
                          8 * (F90_DESC_SIZE + rank * F90_DESCDIM_SIZE);
  const int extent_offset = lbnd_offset + 8;
  const unsigned v1 = lldbg_encode_expression_arg(LL_DW_OP_int, lbnd_offset);
  const unsigned v2 = lldbg_encode_expression_arg(LL_DW_OP_int, extent_offset);
  const unsigned one = lldbg_encode_expression_arg(LL_DW_OP_int, 1);
  const unsigned add = lldbg_encode_expression_arg(LL_DW_OP_plus_uconst, 0);
  const unsigned dup = lldbg_encode_expression_arg(LL_DW_OP_dup, 0);
  const unsigned deref = lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
  const unsigned swap = lldbg_encode_expression_arg(LL_DW_OP_swap, 0);
  const unsigned plus = lldbg_encode_expression_arg(LL_DW_OP_plus, 0);
  const unsigned minus = lldbg_encode_expression_arg(LL_DW_OP_minus, 0);
  const unsigned constu = lldbg_encode_expression_arg(LL_DW_OP_constu, 0);

  array_desc_mdnode = (db->need_dup_composite_type) ?
                      db->gbl_obj_mdnode :
                      ll_get_global_debug(db->module, SDSCG(sptr));
  if (LL_MDREF_IS_NULL(array_desc_mdnode)) {
    array_desc_mdnode = lldbg_fwd_local_variable(db, SDSCG(sptr), findex, false);
    if (db->need_dup_composite_type) {
      db->gbl_obj_mdnode = array_desc_mdnode;
      hashmap_erase(db->module->mdnodes_fwdvars, INT2HKEY(SDSCG(sptr)), NULL);
    }
  }
  lbnd_expr_mdnode = lldbg_emit_expression_mdnode(db, 3, add, v1, deref);
  ubnd_expr_mdnode =
      lldbg_emit_expression_mdnode(db, 12, dup, add, v1, deref, swap, add, v2,
                                   deref, plus, constu, one, minus);

  LLMD_Builder mdb = llmd_init(db->module);
  llmd_set_class(mdb, LL_DIFortranSubrange);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_subrange_type));
  llmd_add_i64(mdb, 0);
  llmd_add_i64(mdb, 0);
  llmd_add_md(mdb, array_desc_mdnode);
  llmd_add_md(mdb, lbnd_expr_mdnode);
  llmd_add_md(mdb, array_desc_mdnode);
  llmd_add_md(mdb, ubnd_expr_mdnode);
  return llmd_finish(mdb);
}

static void
lldbg_get_bounds_for_sdsc(LL_DebugInfo *db, int findex, SPTR sptr,
                          int rank, LL_MDRef *count_expr_mdnode,
                          LL_MDRef *lbnd_expr_mdnode,
                          LL_MDRef *ubnd_expr_mdnode,
                          LL_MDRef *stride_expr_mdnode)
{

  /*
   * Please consider below derived type, which has allocatable array as member.
   * type dt
   *  integer :: var1
   *  integer :: var2
   *  integer, allocatable :: arr (:,:)
   * end type dt
   * Below should be its datalayout,
   *o0=0      o1       o2                   o3
   * |--------|--------|--------------------|---------------------|
   * |<-var1->|<-var2->|<--address of arr-->|<-Descriptor of arr->|
   *
   * for allocatable array 'arr' DW_OP_push_object_address produces 'o2' to get
   * to the descriptor start we need 'o3-o2' which we are calling here
   * 'descr_offset_wrt_array'.
   */
  const int descr_offset_wrt_array =
      (SCG(SDSCG(sptr)) == SC_CMBLK || STYPEG(SDSCG(sptr)) == ST_MEMBER)
          ? ADDRESSG(SDSCG(sptr)) - ADDRESSG(sptr)
          : 0;

  const int F90_Desc_byte_len = DESC_HDR_INT_LEN * (DESC_HDR_BYTE_LEN - DESC_HDR_TAG);
  const int F90_DescDim_size = 8 * DESC_DIM_LEN;    /* sizeof(F90_DescDim)*/
  const int F90_Desc_dim_offset = 8 * DESC_HDR_LEN; /* offsetof(F90_Desc, dim)*/
  const int count_offset_wrt_lbound =
      8 * (DESC_DIM_EXTENT - DESC_DIM_LOWER); /* offsetof(F90_DescDim, extent)*/
  const int ubound_offset_wrt_lbound =
      8 * (DESC_DIM_UPPER - DESC_DIM_LOWER); /* offsetof(F90_DescDim, ubound)*/
  const int lstride_offset_wrt_lbound =
      8 *
      (DESC_DIM_LMULT - DESC_DIM_LOWER); /* offsetof(F90_DescDim, lstride)*/

  const int target_size_offset = descr_offset_wrt_array + F90_Desc_byte_len;
  const int lower_offset =
      descr_offset_wrt_array + F90_Desc_dim_offset + (rank * F90_DescDim_size);
  const int count_offset = lower_offset + count_offset_wrt_lbound;
  const int upper_offset = lower_offset + ubound_offset_wrt_lbound;
  const int stride_offset = lower_offset + lstride_offset_wrt_lbound;

  const unsigned v0 = lldbg_encode_expression_arg(LL_DW_OP_int, count_offset);
  const unsigned v1 = lldbg_encode_expression_arg(LL_DW_OP_int, lower_offset);
  const unsigned v2 = lldbg_encode_expression_arg(LL_DW_OP_int, upper_offset);
  const unsigned v3 = lldbg_encode_expression_arg(LL_DW_OP_int, stride_offset);
  const unsigned v4 =
      lldbg_encode_expression_arg(LL_DW_OP_int, target_size_offset);

  const unsigned add = lldbg_encode_expression_arg(LL_DW_OP_plus_uconst, 0);
  const unsigned mul = lldbg_encode_expression_arg(LL_DW_OP_mul, 0);
  const unsigned deref = lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
  const unsigned pushobj =
      lldbg_encode_expression_arg(LL_DW_OP_push_object_address, 0);

  if (count_expr_mdnode)
    *count_expr_mdnode =
        lldbg_emit_expression_mdnode(db, 4, pushobj, add, v0, deref);
  if (lbnd_expr_mdnode)
    *lbnd_expr_mdnode =
      lldbg_emit_expression_mdnode(db, 4, pushobj, add, v1, deref);
  if (ubnd_expr_mdnode)
    *ubnd_expr_mdnode =
      lldbg_emit_expression_mdnode(db, 4, pushobj, add, v2, deref);
  if (stride_expr_mdnode) {
    if (zsize_of(DTYPEG(sptr)) > 0)
      *stride_expr_mdnode = lldbg_emit_expression_mdnode(
          db, 9, pushobj, add, v3, deref, pushobj, add, v4, deref, mul);
    else
      *stride_expr_mdnode = ll_get_md_null();
  }
}

static LL_MDRef
lldbg_create_subrange_via_sdsc(LL_DebugInfo *db, int findex, SPTR sptr,
                               int rank)
{
  LL_MDRef lbnd_expr_mdnode, ubnd_expr_mdnode, stride_expr_mdnode;
  lldbg_get_bounds_for_sdsc(db, findex, sptr, rank, NULL, &lbnd_expr_mdnode,
                            &ubnd_expr_mdnode, &stride_expr_mdnode);

  return lldbg_create_subrange_mdnode(db, ll_get_md_null(), lbnd_expr_mdnode,
                                      ubnd_expr_mdnode, stride_expr_mdnode);
}

static void
lldbg_get_bounds_for_assumed_rank_sdsc(LL_DebugInfo *db, SPTR sptr,
                                       LL_MDRef *lbnd_expr_mdnode,
                                       LL_MDRef *ubnd_expr_mdnode,
                                       LL_MDRef *stride_expr_mdnode)
{
  const int F90_Desc_byte_len = DESC_HDR_INT_LEN * (DESC_HDR_BYTE_LEN - DESC_HDR_TAG);
  const int F90_DescDim_size = 8 * DESC_DIM_LEN;    /* sizeof(F90_DescDim)*/
  const int F90_Desc_dim_offset = 8 * DESC_HDR_LEN; /* offsetof(F90_Desc, dim)*/
  const int ubound_offset_wrt_lbound =
      8 * (DESC_DIM_UPPER - DESC_DIM_LOWER); /* offsetof(F90_DescDim, ubound)*/
  const int lstride_offset_wrt_lbound =
      8 * (DESC_DIM_LMULT - DESC_DIM_LOWER); /* offsetof(F90_DescDim, lstride)*/

  const int target_size_offset = F90_Desc_byte_len;
  const int lower_offset = F90_Desc_dim_offset;
  const int upper_offset = lower_offset + ubound_offset_wrt_lbound;
  const int stride_offset = lower_offset + lstride_offset_wrt_lbound;

  const unsigned v0 =
      lldbg_encode_expression_arg(LL_DW_OP_int, F90_DescDim_size);
  const unsigned v1 = lldbg_encode_expression_arg(LL_DW_OP_int, lower_offset);
  const unsigned v2 = lldbg_encode_expression_arg(LL_DW_OP_int, upper_offset);
  const unsigned v3 = lldbg_encode_expression_arg(LL_DW_OP_int, stride_offset);
  const unsigned v4 =
      lldbg_encode_expression_arg(LL_DW_OP_int, target_size_offset);

  const unsigned add = lldbg_encode_expression_arg(LL_DW_OP_plus_uconst, 0);
  const unsigned mul = lldbg_encode_expression_arg(LL_DW_OP_mul, 0);
  const unsigned plus = lldbg_encode_expression_arg(LL_DW_OP_plus, 0);
  const unsigned constu = lldbg_encode_expression_arg(LL_DW_OP_constu, 0);
  const unsigned deref = lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
  const unsigned pushobj =
      lldbg_encode_expression_arg(LL_DW_OP_push_object_address, 0);
  const unsigned over = lldbg_encode_expression_arg(LL_DW_OP_over, 0);

  if (lbnd_expr_mdnode)
    *lbnd_expr_mdnode = lldbg_emit_expression_mdnode(
        db, 9, pushobj, over, constu, v0, mul, add, v1, plus, deref);
  if (ubnd_expr_mdnode)
    *ubnd_expr_mdnode = lldbg_emit_expression_mdnode(
        db, 9, pushobj, over, constu, v0, mul, add, v2, plus, deref);
  if (stride_expr_mdnode) {
    if (zsize_of(DTYPEG(sptr)) > 0)
      *stride_expr_mdnode = lldbg_emit_expression_mdnode(
          db, 14, pushobj, over, constu, v0, mul, add, v3, plus, deref, pushobj,
          add, v4, deref, mul);
    else
      *stride_expr_mdnode = ll_get_md_null();
  }
}

static LL_MDRef
lldbg_create_subrange_mdnode_pre11(LL_DebugInfo *db, ISZ_T lb, ISZ_T ub)
{
  DBLINT64 count, low, high;
  DBLINT64 one;
  LLMD_Builder mdb = llmd_init(db->module);

  ISZ_2_INT64(lb, low);
  ISZ_2_INT64(ub, high);
  llmd_set_class(mdb, LL_DISubRange);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_subrange_type));
  llmd_add_INT64(mdb, low);
  if (ll_feature_debug_info_pre34(&db->module->ir)) {
    llmd_add_INT64(mdb, high);
  } else {
    /* Count for LLVM 3.4+ */
    ISZ_2_INT64(1, one);
    sub64(high, low, count);
    /* In 3.7 syntax empty subrange is denoted with count: -1 */
    if (ll_feature_debug_info_subrange_needs_count(&db->module->ir)) {
      if (!count[0] && !count[1])
        ISZ_2_INT64(-1, count);
    }
    add64(one, count, count);
    llmd_add_INT64(mdb, count);
  }

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_subrange_mdnode(LL_DebugInfo *db, LL_MDRef count,
                                             LL_MDRef lb, LL_MDRef ub,
                                             LL_MDRef st)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DISubRange);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_subrange_type));
  if (count != ll_get_md_null())
    llmd_add_md(mdb, count);
  else
    llmd_add_null(mdb);
  llmd_add_md(mdb, lb);
  if (ub != ll_get_md_null())
    llmd_add_md(mdb, ub);
  else
    llmd_add_null(mdb);
  if (st != ll_get_md_null())
    llmd_add_md(mdb, st);
  else
    llmd_add_null(mdb);

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_generic_subrange_mdnode(LL_DebugInfo *db, LL_MDRef lb, LL_MDRef ub,
                                     LL_MDRef st)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DIGenericSubRange);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_subrange_type));
  llmd_add_md(mdb, lb);
  if (ub != ll_get_md_null())
    llmd_add_md(mdb, ub);
  else
    llmd_add_null(mdb);
  if (st != ll_get_md_null())
    llmd_add_md(mdb, st);
  else
    llmd_add_null(mdb);

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_unspecified_mdnode(LL_DebugInfo *db, int dw_tag)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_add_i32(mdb, make_dwtag(db, dw_tag));

  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_unspecified_type_mdnode(LL_DebugInfo *db)
{
  return ll_get_md_null();
}

static LL_MDRef
lldbg_create_unspecified_parameters_mdnode(LL_DebugInfo *db)
{
  return lldbg_create_unspecified_mdnode(db, DW_TAG_unspecified_parameters);
}

LL_MDRef
lldbg_emit_empty_expression_mdnode(LL_DebugInfo *db)
{
  LLMD_Builder mdb = llmd_init(db->module);
  llmd_set_class(mdb, LL_DIExpression);
  return llmd_finish(mdb);
}

int
lldbg_encode_expression_arg(LL_DW_OP_t op, int value)
{
  DEBUG_ASSERT(ll_dw_op_ok(op), "invalid op");
  return (op == LL_DW_OP_int) ? (value << 1) : ((op << 1) | 1);
}

LL_MDRef
lldbg_emit_expression_mdnode(LL_DebugInfo *db, unsigned cnt, ...)
{
  unsigned i;
  va_list ap;
  LLMD_Builder mdb = llmd_init(db->module);
  // enforce an arbitrary limit for now to catch bugs
  DEBUG_ASSERT(cnt < 20, "DIExpression unsupported");
  llmd_set_class(mdb, LL_DIExpression);
  va_start(ap, cnt);
  for (i = 0; i != cnt; ++i) {
    int arg = va_arg(ap, int);
    llmd_add_i32(mdb, arg);
  }
  va_end(ap);
  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_local_variable_mdnode(LL_DebugInfo *db, int dw_tag,
                                   LL_MDRef context, char *name,
                                   LL_MDRef fileref, int line, int argnum,
                                   LL_MDRef type_mdnode, int flags,
                                   LL_MDRef fwd, int sptr = 0)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DILocalVariable);
  if (!ll_feature_debug_info_ver38(&db->module->ir))
    llmd_add_i32(mdb, make_dwtag(db, dw_tag));
  llmd_add_md(mdb, context);
  if (flags & DIFLAG_ARTIFICIAL)
    llmd_add_string(mdb, ""); // Do not expose the name of compiler created variable.
  else
    llmd_add_string(mdb, name);
  if (!ll_feature_dbg_local_variable_embeds_argnum(&db->module->ir))
    llmd_add_i32(mdb, argnum);
  llmd_add_md(mdb, fileref);
  if (ll_feature_dbg_local_variable_embeds_argnum(&db->module->ir))
    llmd_add_i32(mdb, line | (argnum << 24));
  else
    llmd_add_i32(mdb, line);
  llmd_add_md(mdb, type_mdnode);
  llmd_add_i32(mdb, flags);
  llmd_add_i32(mdb, 0);

  if (sptr && (flags & DIFLAG_ARTIFICIAL)) {
    llmd_set_distinct(mdb);
  }

  if (fwd)
    return ll_finish_variable(mdb, fwd);
  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_create_location_mdnode(LL_DebugInfo *db, int line, int column,
                             LL_MDRef scope)
{
  LLMD_Builder mdb = llmd_init(db->module);

  llmd_set_class(mdb, LL_DILocation);
  llmd_add_i32(mdb, line);
  llmd_add_i32(mdb, column);
  llmd_add_md(mdb, scope);
  llmd_add_null(mdb); /* InlinedAt */
  return llmd_finish(mdb);
}

void
lldbg_reset_dtype_array(LL_DebugInfo *db, const int off)
{
  BZERO(db->dtype_array + off, LL_MDRef, db->dtype_array_sz - off);
}

void
lldbg_update_arrays(LL_DebugInfo *db, int lastDType, int newSz)
{
  const int fromSz = db->dtype_array_sz;
  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  NEED(newSz, db->dtype_array, LL_MDRef, db->dtype_array_sz, newSz);
  if (newSz > fromSz) {
    BZERO(db->dtype_array + lastDType, LL_MDRef, newSz - lastDType);
  }
}

void
lldbg_init_arrays(LL_DebugInfo *db)
{
  const int new_size = stb.dt.stg_avail + 2000;
  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  NEED(new_size, db->dtype_array, LL_MDRef, db->dtype_array_sz, new_size);
  if (db->module->ir.is_nvvm && (db->dtype_array_sz > DT_MAX)) {
    BZERO(&db->dtype_array[DT_MAX], LL_MDRef, db->dtype_array_sz - DT_MAX);
  }
}

void
lldbg_init(LL_Module *module)
{
  LL_DebugInfo *db;
  int sptr;

  if (module->debug_info) {
    const int newSz = stb.dt.stg_avail + 2000;
    db = module->debug_info;
    NEEDB(newSz, db->dtype_array, LL_MDRef, db->dtype_array_sz, newSz);
    return;
  }

  db = (LL_DebugInfo *)calloc(1, sizeof(LL_DebugInfo));
  module->debug_info = db;

  /* calloc initializes most struct members to the right initial value. */
  db->module = module;
  db->blk_idx = -1;
  db->cur_module_name = NULL;
  db->import_entity_list = NULL;
  db->need_dup_composite_type = false;
  db->gbl_var_sptr = SPTR_NULL;
  db->gbl_obj_mdnode = ll_get_md_null();
  db->gbl_obj_exp_mdnode = ll_get_md_null();

  if (!ll_feature_debug_info_pre34(&db->module->ir)) {
    const int mdVers = ll_feature_versioned_dw_tag(&module->ir)
                           ? 1
                           : module->ir.debug_info_version;
    const unsigned dwarfVers = ll_feature_dwarf_version(&module->ir);
    if (!module->named_mdnodes[MD_llvm_module_flags]) {
      ll_extend_named_md_node(
          module, MD_llvm_module_flags,
          lldbg_create_module_flag_mdnode(db, 2, "Dwarf Version", dwarfVers));
      ll_extend_named_md_node(
          module, MD_llvm_module_flags,
          lldbg_create_module_flag_mdnode(db, 2, "Debug Info Version", mdVers));
    }
  }

  sprintf(db->producer, "%s %s %s%s", version.product, version.lang,
          version.vsn, version.bld);
  NEW(db->file_array, LL_MDRef, fihb.stg_avail * 2);
  BZERO(db->file_array, LL_MDRef, fihb.stg_avail * 2);
  db->file_array_sz = fihb.stg_avail * 2;
  NEW(db->dtype_array, LL_MDRef, stb.dt.stg_avail + 2000);
  BZERO(db->dtype_array, LL_MDRef, stb.dt.stg_avail + 2000);
  db->dtype_array_sz = stb.dt.stg_avail + 2000;
  for (sptr = gbl.entries; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    db->routine_count++;
  }
  if (db->routine_count)
    db->llvm_dbg_lv_array =
        (LL_MDRef *)calloc(db->routine_count, sizeof(LL_MDRef));
  else
    db->llvm_dbg_lv_array = NULL;
}

void lldbg_free_import_entity_list(import_entity *import_entity_list) {
  while (import_entity_list != NULL) {
    import_entity *node = import_entity_list;
    if (node->child)
      lldbg_free_import_entity_list(node->child);
    node->child = NULL;
    import_entity_list = node->next;
    free(node);
  }
}

void
lldbg_free(LL_DebugInfo *db)
{
  if (!db)
    return;
  FREE(db->file_array);
  FREE(db->dtype_array);
  db->file_array_sz = 0;
  db->dtype_array_sz = 0;
  while (db->sptrs_to_mdnodes != NULL) {
    struct sptr_to_mdnode_map *nsp = db->sptrs_to_mdnodes;
    db->sptrs_to_mdnodes = nsp->next;
    free(nsp);
  }
  lldbg_free_import_entity_list(db->import_entity_list);
  db->import_entity_list = NULL;
  free(db);
}

/**
   \brief Double any backslash characters in \p name
   \param name   A filename

   Precondition: \p name must be allocated on the heap, as it may be
   deallocated.
 */
static char *
double_backslash(char *name)
{
  int len;
  char *new_name, *psrc, *pdst;
  int bs_count;

  psrc = name;
  bs_count = 0;
  len = 1;
  while (*psrc != '\0') {
    if (*psrc == '\\')
      bs_count++;
    len++;
    psrc++;
  }
  if (!bs_count)
    return name;
  NEW(new_name, char, len + bs_count);
  psrc = name;
  pdst = new_name;
  while (*psrc != '\0') {
    if (*psrc == '\\') {
      *pdst++ = '\\';
    }
    *pdst++ = *psrc++;
  }
  *pdst = '\0';
  FREE(name);

  return new_name;
}

/**
   \brief Get the filename associated with file index, \p findex.
   \param findex  The file's index
   \return a heap allocated string; caller must deallocate.
 */
static char *
get_filename(int findex)
{
  char *fullpath;
  const char *dirname = FIH_DIRNAME(findex);
  const char *filename = FIH_FILENAME(findex);
  const int dirnameLen = dirname ? strlen(dirname) : 0;
  const int filenameLen = filename ? strlen(filename) : 0;
  const char *dir = dirname ? dirname : "";
  const char *file = filename ? filename : "";
#if defined(TARGET_WIN)
  const char *sep = "\\";
#else
  const char *sep = "/";
#endif
  const bool addSep = (dirnameLen > 0) && (dirname[dirnameLen - 1] != *sep);
  const int allocLen = dirnameLen + filenameLen + 1 + (addSep ? 1 : 0);

  NEW(fullpath, char, allocLen);
  snprintf(fullpath, allocLen, "%s%s%s", dir, (addSep ? sep : ""), file);
  return double_backslash(fullpath);
}

#define MAX_FNAME 1024

static char *
get_currentdir(void)
{
  char *cwd = (char *)malloc(PATH_MAX);
  getcwd(cwd, PATH_MAX);
  return cwd;
}

LL_MDRef
lldbg_emit_compile_unit(LL_DebugInfo *db)
{
  int lang_tag;
  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  if (LL_MDREF_IS_NULL(db->comp_unit_mdnode)) {
    lang_tag = DW_LANG_Fortran90;
    db->comp_unit_mdnode = lldbg_create_compile_unit_mdnode(
        db, lang_tag, get_filename(1), get_currentdir(), db->producer, 1,
        flg.opt >= 1/*isOptimized Flag*/, flg.cmdline,
        0, &db->llvm_dbg_enum, &db->llvm_dbg_retained, &db->llvm_dbg_sp,
        &db->llvm_dbg_gv, &db->llvm_dbg_imported);
  }
  return db->comp_unit_mdnode;
}

static LL_MDRef
lldbg_emit_file(LL_DebugInfo *db, int findex)
{
  LL_MDRef cu_mnode;

  cu_mnode = lldbg_emit_compile_unit(db);
  if (LL_MDREF_IS_NULL(get_file_mdnode(db, findex))) {
    cu_mnode = ll_get_md_null();
    lldbg_create_file_mdnode(db, get_filename(findex), get_currentdir(),
                             cu_mnode, findex);
  }
  return ll_feature_debug_info_need_file_descriptions(&db->module->ir)
	  ? get_filedesc_mdnode(db, 1) : get_file_mdnode(db, findex);
}

static LOGICAL
is_procedure_dtype(DTYPE dtype) {
  return dtype > DT_NONE && DTY(dtype) == TY_PROC;
}

static LOGICAL
is_procedure_ptr_dtype(DTYPE dtype) {
  return ((dtype > DT_NONE) && (DTY(dtype) == TY_PTR) &&
          is_procedure_dtype(DTySeqTyElement(dtype)));
}

LOGICAL
is_procedure_ptr(SPTR sptr) {
  if (sptr > NOSYM && (POINTERG(sptr))) {
    switch (STYPEG(sptr)) {
    case ST_PROC:
    case ST_ENTRY:
      /* subprograms aren't considered to be procedure pointers */
      break;
    default:
      return is_procedure_ptr_dtype(DTYPEG(sptr));
    }
  }
  return FALSE;
}

static LL_MDRef
lldbg_emit_parameter_list(LL_DebugInfo *db, DTYPE dtype, DTYPE ret_dtype,
                          SPTR sptr, int findex)
{
  LLMD_Builder mdb = llmd_init(db->module);
  LL_MDRef parameter_mdnode, retval_mdnode;
  int num_args;
  DTYPE call_dtype = dtype;
  int dpdsc, paramct, i;
  SPTR fval;
  int is_reference;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);

  while (DTY(call_dtype) == TY_ARRAY || DTY(call_dtype) == TY_PTR)
    call_dtype = DTySeqTyElement(call_dtype);

  dpdsc = DPDSCG(sptr);
  paramct = PARAMCTG(sptr);
  fval = FVALG(sptr);

  if (ret_dtype) {
#ifdef CUDA_DEVICE
    if (CUDAG(sptr) & CUDA_DEVICE)
      retval_mdnode =
          lldbg_emit_modified_type(db, ret_dtype, SPTR_NULL, findex);
    else
#endif
      /* Reference is used in case of non basic type */
      retval_mdnode =
          lldbg_emit_type(db, ret_dtype, SPTR_NULL, findex,
                          !DT_ISBASIC(ret_dtype), false, true);
  } else {
    if (ll_feature_debug_info_pre34(&db->module->ir))
      retval_mdnode = ll_get_md_null();
    else
      retval_mdnode = lldbg_create_unspecified_type_mdnode(db);
  }

  llmd_add_md(mdb, retval_mdnode);

  /* do the argument list;
   * kernel arguments are either typed value arguments, or global pointer
   * arguments */

  /* do the return value, if it appears in the argument list */
  num_args = 0;
  if (fval && SCG(fval) == SC_DUMMY) {
    is_reference =
        ((SCG(fval) == SC_DUMMY) && HOMEDG(fval) && !PASSBYVALG(fval));
    parameter_mdnode = lldbg_emit_type(db, DTYPEG(fval), fval, findex,
                                       is_reference, false, true);
    llmd_add_md(mdb, parameter_mdnode);
    ++num_args;
  }
  for (i = 0; i < paramct; i++) {
    SPTR param_sptr = GetParamSptr(dpdsc, i);
    if (param_sptr == fval)
      continue;
    is_reference = ((SCG(param_sptr) == SC_DUMMY) && HOMEDG(param_sptr) &&
                    !PASSBYVALG(param_sptr));
    parameter_mdnode = lldbg_emit_type(db, DTYPEG(param_sptr), param_sptr,
                                       findex, is_reference, false, true);
    llmd_add_md(mdb, parameter_mdnode);
    ++num_args;
  }

  if (num_args == 0 && ll_feature_debug_info_pre34(&db->module->ir)) {
    parameter_mdnode = lldbg_create_unspecified_parameters_mdnode(db);
    llmd_add_md(mdb, parameter_mdnode);
  }
  return llmd_finish(mdb);
}

INLINE static LL_MDRef
lldbg_emit_subroutine_type(LL_DebugInfo *db, SPTR sptr, DTYPE ret_dtype,
                           int findex, LL_MDRef file_mdnode)
{
  LL_MDRef subroutine_type_mdnode;
  int cc = 0;
  DTYPE dtype = DTYPEG(sptr);
  LL_MDRef parameters_mdnode =
      lldbg_emit_parameter_list(db, dtype, ret_dtype, sptr, findex);
  cc = (gbl.rutype == RU_PROG) ? 2 : 0;
  subroutine_type_mdnode = lldbg_create_subroutine_type_mdnode(
      db, ll_get_md_null(), file_mdnode, parameters_mdnode, cc);
  return subroutine_type_mdnode;
}

static LL_MDRef
lldbg_emit_outlined_subroutine(LL_DebugInfo *db, int sptr, int ret_dtype,
                               int findex, LL_MDRef file_mdnode)
{
  LL_MDRef subroutine_type_mdnode;
  int cc = 0;
  LL_MDRef parameters_mdnode = lldbg_create_outlined_parameters_node(db);
  cc = (gbl.rutype == RU_PROG) ? 2 : 0;
  subroutine_type_mdnode = lldbg_create_subroutine_type_mdnode(
      db, ll_get_md_null(), file_mdnode, parameters_mdnode, cc);
  return subroutine_type_mdnode;
}

static void
lldbg_reserve_lexical_block(LL_DebugInfo *db, int sptr, int lineno, int findex)
{
  int endline, startline;

  endline = ENDLINEG(sptr);
  startline = lineno;
  if (endline < startline)
    endline = startline;
  NEEDB((db->blk_idx + 1), db->blk_tab, BLKINFO, db->blk_tab_size,
        (db->blk_tab_size + 64));
  db->blk_tab[db->blk_idx].mdnode = ll_get_md_null();
  db->blk_tab[db->blk_idx].sptr = sptr;
  db->blk_tab[db->blk_idx].startline = startline;
  db->blk_tab[db->blk_idx].endline = endline;
  db->blk_tab[db->blk_idx].line_mdnodes = NULL;
  db->blk_idx++;
}

static BLKINFO *
lldbg_limit_lexical_blocks(LL_DebugInfo *db, int startline)
{
  int i;
  BLKINFO *parent_blk;

  for (i = 0; i < db->blk_idx; i++)
    db->blk_tab[i].keep = 0;
  parent_blk = NULL;
  for (i = 0; i < db->blk_idx; i++) {
    if (db->blk_tab[i].startline <= startline &&
        db->blk_tab[i].endline > startline) {
      /* We have found a candidate, is it the best ? */
      if (parent_blk == NULL) {
        parent_blk = &db->blk_tab[i];
      } else if (parent_blk->startline <= db->blk_tab[i].startline ||
                 parent_blk->endline >= db->blk_tab[i].endline) {
        parent_blk->keep = 0;
        parent_blk = &db->blk_tab[i];
      }
      parent_blk->keep = 1;
    }
  }
  if (parent_blk == NULL) {
    for (i = 0; i < db->blk_idx; i++)
      db->blk_tab[i].keep = 1;
    return NULL;
  }

  for (i = 0; i < db->blk_idx; i++) {
    if (!db->blk_tab[i].keep &&
        db->blk_tab[i].startline >= parent_blk->startline &&
        db->blk_tab[i].endline <= parent_blk->endline)
      db->blk_tab[i].keep = 1;
  }
  return parent_blk;
}

static void
lldbg_assign_lexical_block(LL_DebugInfo *db, int idx, int findex,
                           bool targetNVVM)
{
  static int ID = 0;
  int endline, startline;
  BLKINFO *parent_blk;
  LL_MDRef parent_blk_mdnode;
  int i;
  int lineno = 0, column = 0;

  startline = db->blk_tab[idx].startline;
  endline = db->blk_tab[idx].endline;
  parent_blk = NULL;
  for (i = 0; i < db->blk_idx; i++) {
    if (i != idx) {
      if (db->blk_tab[i].keep && db->blk_tab[i].startline <= startline &&
          db->blk_tab[i].endline >= endline && db->blk_tab[i].mdnode) {
        /* We have found a candidate, is it the best ? */
        if (parent_blk == NULL)
          parent_blk = &db->blk_tab[i];
        else if ((parent_blk->startline <= db->blk_tab[i].startline) ||
                 (parent_blk->endline >= db->blk_tab[i].endline))
          parent_blk = &db->blk_tab[i];
      }
    }
  }
  if (parent_blk != NULL) {
    parent_blk_mdnode = parent_blk->mdnode;
    assert(parent_blk_mdnode, "Parent of a DILexicalBlock must exist",
           parent_blk_mdnode, ERR_Severe);
  } else
    parent_blk_mdnode = db->cur_subprogram_mdnode;
  db->blk_tab[idx].mdnode = lldbg_create_block_mdnode(
      db, parent_blk_mdnode, startline, 1, findex, ID++);
  db->blk_tab[idx].line_mdnodes =
      (LL_MDRef *)calloc((endline - startline + 1), sizeof(LL_MDRef));
  db->blk_tab[idx].null_loc =
      lldbg_create_location_mdnode(db, lineno, column, db->blk_tab[idx].mdnode);
}

static void
lldbg_emit_lexical_block(LL_DebugInfo *db, int sptr, int lineno, int findex,
                         bool targetNVVM)
{
  static int ID = 0;
  LL_MDRef null_loc_mdnode;
  int endline, startline;
  BLKINFO *parent_blk;
  LL_MDRef parent_blk_mdnode;
  int i;

  endline = ENDLINEG(sptr);
  startline = lineno;
  if (endline < startline)
    endline = startline;
  parent_blk = NULL;
  for (i = 0; i < db->blk_idx; i++) {
    if (db->blk_tab[i].startline <= startline &&
        db->blk_tab[i].endline >= endline) {
      /* We have found a candidate, is it the best ? */
      if (parent_blk == NULL)
        parent_blk = &db->blk_tab[i];
      else if (parent_blk->startline <= db->blk_tab[i].startline ||
               parent_blk->endline >= db->blk_tab[i].endline)
        parent_blk = &db->blk_tab[i];
    }
  }
  if (parent_blk != NULL)
    parent_blk_mdnode = parent_blk->mdnode;
  else
    parent_blk_mdnode = db->cur_subprogram_mdnode;
  NEEDB((db->blk_idx + 1), db->blk_tab, BLKINFO, db->blk_tab_size,
        (db->blk_tab_size + 64));
  db->blk_tab[db->blk_idx].mdnode =
      STYPEG(sptr) == ST_BLOCK ? lldbg_create_block_mdnode(db,
      parent_blk_mdnode, lineno, 1, findex, ID++)
      : parent_blk_mdnode;
  db->blk_tab[db->blk_idx].sptr = sptr;
  db->blk_tab[db->blk_idx].startline = startline;
  db->blk_tab[db->blk_idx].endline = endline;
  db->blk_tab[db->blk_idx].keep = 1;
  db->blk_tab[db->blk_idx].line_mdnodes =
      (LL_MDRef *)calloc((endline - startline + 1), sizeof(LL_MDRef));
    null_loc_mdnode =
        lldbg_create_location_mdnode(db, 0, 0, db->blk_tab[db->blk_idx].mdnode);
  db->blk_tab[db->blk_idx].null_loc = null_loc_mdnode;
  db->blk_idx++;
}

static void
lldbg_emit_lexical_blocks(LL_DebugInfo *db, int sptr, int findex,
                          bool targetNVVM)
{
  int encl_block;
  int blk_sptr;
  int idx;
  int lineno;

  for (idx = 0; idx < db->blk_idx; idx++) {
    if (db->blk_tab[idx].line_mdnodes) {
      free(db->blk_tab[idx].line_mdnodes);
      db->blk_tab[idx].line_mdnodes = NULL;
    }
  }
  db->blk_idx = 0;
  lineno = FUNCLINEG(sptr);
  lldbg_emit_lexical_block(db, sptr, lineno, findex, targetNVVM);
  db->cur_blk = db->blk_tab[0];

  /* An extraordinary amount of the total compilation time (19% on 510.parest
   * with the DEV compiler on a Tuleta) is spent for C++ compilations in the
   * following loop nest, which passes over most of the symbol table in search
   * of ST_BLOCK entries whose enclosing entries match any of the ones named by
   * db->blk_tab[].sptr so that they can be passed in order, without duplicates,
   * to lldbg_emit_lexical_block().
   *
   * So we use the original algorithm only for Fortran and C, which don't
   * exhibit the problem and whose symbol table re-use on a subprogram basis
   * wouldn't allow the code below to work anyway.
   */
  blk_sptr = sptr + 1;
  for (; blk_sptr < stb.stg_avail; blk_sptr++) {
    if (STYPEG(blk_sptr) == ST_BLOCK) {
      /*
       * check to see if it is enclosed by one of the prior blocks
       * entered
       */
      encl_block = ENCLFUNCG(blk_sptr);
      while (true) {
        /*
         * It may be the case that the owner of the block is
         * a fake block.  For example, a parallel region creates
         * a block just to ensure that the scoping is correct;
         * any lexical block in the parallel region is owned
         * by the fake block.
         *
         * Just check the owner -- if it's fake, look at its
         * owner, etc.
         */
        if (encl_block == 0 || STYPEG(encl_block) != ST_BLOCK ||
            ENDLABG(encl_block))
          break;
        encl_block = ENCLFUNCG(encl_block);
      }
      for (idx = 0; idx < db->blk_idx; idx++) {
        if (db->blk_tab[idx].sptr == encl_block) {
          lldbg_emit_lexical_block(db, blk_sptr, STARTLINEG(blk_sptr), findex,
                                   targetNVVM);
          break;
        }
      }
    }
  }
}

static void
lldbg_reserve_lexical_blocks(LL_DebugInfo *db, int sptr, int findex)
{
  int encl_block;
  int blk_sptr;
  int idx;

  for (idx = 0; idx < db->blk_idx; idx++) {
    if (db->blk_tab[idx].line_mdnodes) {
      free(db->blk_tab[idx].line_mdnodes);
      db->blk_tab[idx].line_mdnodes = NULL;
    }
  }
  db->blk_idx = 0;
  lldbg_reserve_lexical_block(db, sptr, FUNCLINEG(sptr), findex);
  db->cur_blk = db->blk_tab[0];
  blk_sptr = sptr + 1;
  for (; blk_sptr < stb.stg_avail; blk_sptr++) {
    if (STYPEG(blk_sptr) == ST_BLOCK) {
      /*
       * check to see if it is enclosed by one of the prior blocks
       * entered
       */
      encl_block = ENCLFUNCG(blk_sptr);
      while (true) {
        /*
         * It may be the case that the owner of the block is
         * a fake block.  For example, a parallel region creates
         * a block just to ensure that the scoping is correct;
         * any lexical block in the parallel region is owned
         * by the fake block.
         *
         * Just check the owner -- if it's fake, look at its
         * owner, etc.
         */
        if (encl_block == 0 || STYPEG(encl_block) != ST_BLOCK ||
            ENDLABG(encl_block))
          break;
        encl_block = ENCLFUNCG(encl_block);
      }
      for (idx = 0; idx < db->blk_idx; idx++) {
        if (db->blk_tab[idx].sptr == encl_block) {
          lldbg_reserve_lexical_block(db, blk_sptr, STARTLINEG(blk_sptr),
                                      findex);
          break;
        }
      }
    }
  }
}

static void
lldbg_assign_lexical_blocks(LL_DebugInfo *db, int findex, BLKINFO *parent_blk,
                            bool targetNVVM)
{
  int idx;
  if (parent_blk == NULL) {
    for (idx = 0; idx < db->blk_idx; idx++)
      lldbg_assign_lexical_block(db, idx, findex, targetNVVM);
  } else {
    for (idx = 0; idx < db->blk_idx; idx++)
      if (db->blk_tab[idx].keep)
        lldbg_assign_lexical_block(db, idx, findex, targetNVVM);
    for (idx = 0; idx < db->blk_idx; idx++)
      if (!db->blk_tab[idx].keep)
        db->blk_tab[idx].mdnode = parent_blk->mdnode;
  }
}

/**
   \brief Construct the flag set that corresponds with LLVM metadata
 */
INLINE static int
set_disubprogram_flags(LL_DebugInfo *db, int sptr)
{
  int flags = 0;
  if (CCSYMG(sptr))
    flags |= DIFLAG_ARTIFICIAL;
  if (!ll_feature_debug_info_ver90(&db->module->ir))
    if (gbl.rutype == RU_PROG)
      flags |= DIFLAG_ISMAINPGM;
  if (!ll_feature_debug_info_ver80(&db->module->ir))
    if (PUREG(sptr))
      flags |= DIFLAG_PURE;
  return flags;
}

INLINE static int
set_disubprogram_spflags(LL_DebugInfo *db, int sptr, bool is_def, bool is_local,
                         bool is_optimized)
{
  int spFlags = 0;
  if (is_local)
    spFlags |= DISPFLAG_LOCALTOUNIT;       /* 1 << 2 */
  if (is_def)
    spFlags |= DISPFLAG_DEFINITION;        /* 1 << 3 */
  if (is_optimized)
    spFlags |= DISPFLAG_OPTIMIZED;         /* 1 << 4 */
  if (PUREG(sptr))
    spFlags |= DISPFLAG_PURE;              /* 1 << 5 */
  if (ELEMENTALG(sptr))
    spFlags |= DISPFLAG_ELEMENTAL;         /* 1 << 6 */
  if (RECURG(sptr))
    spFlags |= DISPFLAG_RECURSIVE;         /* 1 << 7 */
  if (ll_feature_debug_info_ver90(&db->module->ir))
    if (gbl.rutype == RU_PROG)
      spFlags |= DISPFLAG_MAINSUBPROGRAM;  /* 1 << 8 */
  return spFlags;
}

void
lldbg_emit_outlined_subprogram(LL_DebugInfo *db, int sptr, int findex,
                               const char *func_name, int startlineno,
                               bool targetNVVM)
{
  LL_MDRef file_mdnode;
  LL_MDRef type_mdnode;
  LL_MDRef lv_list_mdnode;
  const char *mips_linkage_name = "";
  int virtuality = 0;
  int vindex = 0;
  int spFlags = 0;
  int flags = 0;
  int is_optimized = 0;
  int sc = SCG(sptr);
  int is_def;
  int is_local = (sc == SC_STATIC);
  int lineno;
  BLKINFO *parent_blk;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  file_mdnode = lldbg_emit_file(db, findex);
  type_mdnode = lldbg_emit_outlined_subroutine(
      db, sptr, DTyReturnType(DTYPEG(sptr)), findex, file_mdnode);
  if (ll_feature_has_diextensions(&db->module->ir))
    flags = set_disubprogram_flags(db, sptr);
  db->cur_line_mdnode = ll_get_md_null();
  lv_list_mdnode = ll_create_flexible_md_node(db->module);
  if (db->routine_idx >= db->routine_count)
    db->routine_count = db->routine_idx + 1;
  db->llvm_dbg_lv_array = (LL_MDRef *)realloc(
      db->llvm_dbg_lv_array, sizeof(LL_MDRef) * db->routine_count);
  db->llvm_dbg_lv_array[db->routine_idx++] = lv_list_mdnode;

  lldbg_reserve_lexical_blocks(db, sptr, findex);
  parent_blk = lldbg_limit_lexical_blocks(db, startlineno);

  if (parent_blk == NULL)
    lineno = startlineno;
  else
    lineno = parent_blk->startline;
  mips_linkage_name = func_name;
  is_def = DEFDG(sptr);
  is_def |= (STYPEG(sptr) == ST_ENTRY);
  spFlags = set_disubprogram_spflags(db, sptr, is_def, is_local, is_optimized);
  if (ll_feature_debug_info_pre34(&db->module->ir))
    lldbg_create_subprogram_mdnode(
        db, file_mdnode, func_name, mips_linkage_name, file_mdnode, lineno,
        type_mdnode, is_local, is_def, virtuality, vindex, spFlags,
        flags, is_optimized, ll_get_md_null(), ll_get_md_null(), lv_list_mdnode,
        lineno);
  else if (ll_feature_debug_info_ver38(&(db)->module->ir))
    lldbg_create_subprogram_mdnode(
        db, lldbg_emit_compile_unit(db), func_name, mips_linkage_name,
        get_filedesc_mdnode(db, findex), lineno, type_mdnode, is_local,
        is_def, virtuality, vindex, spFlags, flags, is_optimized,
        ll_get_md_null(), ll_get_md_null(), lv_list_mdnode, lineno);
  else
    lldbg_create_subprogram_mdnode(
        db, file_mdnode, func_name, mips_linkage_name,
        get_filedesc_mdnode(db, findex), lineno, type_mdnode, is_local,
        is_def, virtuality, vindex, spFlags, flags, is_optimized,
        ll_get_md_null(), ll_get_md_null(), lv_list_mdnode, lineno);
  db->cur_subprogram_null_loc =
      lldbg_create_location_mdnode(db, lineno, 1, db->cur_subprogram_mdnode);
  db->cur_subprogram_lineno = lineno;
  db->param_idx = 0;
  memset(db->param_stack, 0, sizeof(PARAMINFO) * PARAM_STACK_SIZE);
  lldbg_assign_lexical_blocks(db, findex, parent_blk, targetNVVM);
}

LL_MDRef
lldbg_emit_module_mdnode(LL_DebugInfo *db, int sptr)
{
  LL_MDRef module_mdnode;

  LL_MDRef file_mdnode = lldbg_emit_file(db, 1);
  module_mdnode =
      ll_get_module_debug(db->module->module_debug_map, SYMNAME(sptr));
  if (!LL_MDREF_IS_NULL(module_mdnode))
    return module_mdnode;

  return lldbg_create_module_mdnode(db, ll_get_md_null(), SYMNAME(sptr),
                                    lldbg_emit_compile_unit(db),
                                    file_mdnode, FUNCLINEG(sptr));
}

void
lldbg_emit_subprogram(LL_DebugInfo *db, SPTR sptr, DTYPE ret_dtype, int findex,
                      bool targetNVVM)
{
  LL_MDRef file_mdnode;
  LL_MDRef type_mdnode;
  LL_MDRef lv_list_mdnode;
  LL_MDRef context_mdnode;
  LL_MDRef scope;
  LL_MDRef imported_list;
  const char *mips_linkage_name = "";
  const char *func_name;
  int virtuality = 0;
  int vindex = 0;
  int spFlags = 0;
  int flags = 0;
  bool is_optimized = flg.opt >= 1;
  int sc = SCG(sptr);
  int is_def;
  int is_local = (sc == SC_STATIC);
  int lineno;
  hash_data_t scopeData;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  file_mdnode = lldbg_emit_file(db, findex);
  /* For `DI' syntax use file desctipion */
  if (ll_feature_debug_info_need_file_descriptions(&db->module->ir))
    file_mdnode = get_filedesc_mdnode(db, findex);
  type_mdnode =
      lldbg_emit_subroutine_type(db, sptr, ret_dtype, findex, file_mdnode);
  db->cur_line_mdnode = ll_get_md_null();
  lv_list_mdnode = ll_create_flexible_md_node(db->module);
  if (db->routine_idx >= db->routine_count)
    db->routine_count = db->routine_idx + 1;
  db->llvm_dbg_lv_array = (LL_MDRef *)realloc(
      db->llvm_dbg_lv_array, sizeof(LL_MDRef) * db->routine_count);
  db->llvm_dbg_lv_array[db->routine_idx++] = lv_list_mdnode;

  lineno = FUNCLINEG(sptr);
  if (ll_feature_has_diextensions(&db->module->ir))
    flags = set_disubprogram_flags(db, sptr);
  get_extra_info_for_sptr(&func_name, &context_mdnode,
                          NULL /* pmk: &type_mdnode */, db, sptr);
  is_def = DEFDG(sptr);
  is_def |= (STYPEG(sptr) == ST_ENTRY);
  if (INMODULEG(sptr) && ll_feature_create_dimodule(&db->module->ir)) {
    context_mdnode = lldbg_emit_module_mdnode(db, INMODULEG(sptr));
  }
  scope = ll_feature_debug_info_pre34(&db->module->ir)
              ? file_mdnode
              : get_filedesc_mdnode(db, findex);
  if (CONTAINEDG(sptr) && db->subroutine_mdnodes &&
      hashmap_lookup(db->subroutine_mdnodes, INT2HKEY(gbl.outersub),
                     &scopeData)) {
    context_mdnode = (LL_MDRef)(unsigned long)scopeData;
  }
  spFlags = set_disubprogram_spflags(db, sptr, is_def, is_local, is_optimized);
  lldbg_create_subprogram_mdnode(db, context_mdnode, func_name,
                                 mips_linkage_name, scope, lineno, type_mdnode,
                                 is_local, is_def, virtuality, vindex,
                                 spFlags, flags, is_optimized, ll_get_md_null(),
                                 ll_get_md_null(), lv_list_mdnode, lineno);
  if (!db->subroutine_mdnodes)
    db->subroutine_mdnodes = hashmap_alloc(hash_functions_direct);
  imported_list = ll_feature_debug_info_ver17(&db->module->ir) ?
                 lv_list_mdnode : db->llvm_dbg_imported;
  scopeData = (hash_data_t)(unsigned long)db->cur_subprogram_mdnode;
  hashmap_replace(db->subroutine_mdnodes, INT2HKEY(sptr), &scopeData);
  while (db->import_entity_list) {
    import_entity *child = db->import_entity_list->child;
    LL_MDRef elements_mdnode =
        (child ? ll_create_flexible_md_node(db->module) : (LL_MDRef) NULL);
    /* There are pending entities to be imported into this func */
    lldbg_emit_imported_entity(db, db->import_entity_list->entity, sptr,
                               db->import_entity_list->entity_type,
                               elements_mdnode, imported_list);
    while (child) {
      LL_MDRef element = lldbg_create_imported_entity(db, child->entity, sptr,
                                                      child->entity_type,
                                                      (LL_MDRef) NULL,
                                                      (LL_MDRef) NULL);
      ll_extend_md_node(db->module, elements_mdnode, element);
      child = child->next;
    }
    db->import_entity_list = db->import_entity_list->next;
  }
  db->cur_subprogram_null_loc =
      lldbg_create_location_mdnode(db, 0, 0, db->cur_subprogram_mdnode);
  db->cur_subprogram_lineno = lineno;
  db->cur_subprogram_line_mdnode = ll_get_md_null();
  db->param_idx = 0;
  memset(db->param_stack, 0, sizeof(PARAMINFO) * PARAM_STACK_SIZE);
  lldbg_emit_lexical_blocks(db, sptr, findex, targetNVVM);
}

/**
   \brief Is \p fn a function, but not the current function?
   \param fn  The symbol to test

   Helper to hide the differences between compilers.
 */
INLINE static bool
getLexBlock_notCurrentFunc(const int fn)
{
  return (STYPEG(fn) == ST_ENTRY) && (fn != gbl.currsub);
}

LL_MDRef
lldbg_get_var_line(LL_DebugInfo *db, int sptr)
{
  int idx;
  const int blk_sptr = ENCLFUNCG(sptr);

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  if (blk_sptr == 0) {
    assert(db->blk_idx > 0,
           "get_var_line(): empty blk_tab when "
           "processing sptr",
           sptr, ERR_Fatal);
    return db->blk_tab[0].null_loc;
  }
  switch (STYPEG(blk_sptr)) {
  case ST_BLOCK:
    if (flg.smp) {
      const int fn = ENCLFUNCG(blk_sptr);
      if (fn && getLexBlock_notCurrentFunc(fn))
        return db->blk_tab[0].null_loc;
    }
    for (idx = db->blk_idx - 1; idx >= 0; --idx) {
      if (db->blk_tab[idx].sptr == blk_sptr)
        return db->blk_tab[idx].null_loc;
    }
    break;
  case ST_PROC:
    return db->cur_subprogram_null_loc;
  case ST_ENTRY:
    return db->cur_subprogram_null_loc;
  default:
    assert(false, "get_var_line(): line not found for sptr", sptr, ERR_Fatal);
    break;
  }
  return ll_get_md_null();
}

static BLKINFO *
get_lexical_block_info(LL_DebugInfo *db, int sptr, bool unchecked)
{
  int idx;
  const int blk_sptr = ENCLFUNCG(sptr);

  if (SCG(sptr) == SC_DUMMY) {
    /* Assume the initial block of the function if this is a dummy */
    return &db->blk_tab[0];
  } else if (blk_sptr == 0) {
    assert(db->blk_idx > 0,
           "get_lexical_block_info(): empty blk_tab when "
           "processing sptr",
           sptr, ERR_Fatal);
    return &db->blk_tab[0];
  } else if (flg.smp && STYPEG(blk_sptr) == ST_BLOCK) {
    const int fn = ENCLFUNCG(blk_sptr);
    if (fn && getLexBlock_notCurrentFunc(fn))
      return &db->blk_tab[0];
  }

  for (idx = db->blk_idx - 1; idx >= 0; --idx) {
    if (db->blk_tab[idx].sptr == blk_sptr)
      return &db->blk_tab[idx];
  }

  if (unchecked)
    return &db->cur_blk;

  assert(false, "get_lexical_block_info(): block not found for sptr", sptr,
         ERR_Fatal);
  return NULL;
}

void
lldbg_emit_line(LL_DebugInfo *db, int lineno)
{
  static int last_line = 0;
  static LL_MDRef last_subprogram_mdnode = 0U;
  int idx = 0;
  int startline, endline;
  int i, j;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  if (db->blk_idx < 0) {
    /* lldbg_emit_subprogram has not been called for this function
     * Don't do anything for NOW, might need to be revisited later
     */
    db->cur_line_mdnode = ll_get_md_null();
    return;
  }
  if ((last_line != lineno) ||
      ((!LL_MDREF_IS_NULL(db->cur_subprogram_mdnode)) &&
       (db->cur_subprogram_mdnode != last_subprogram_mdnode))) {
    j = db->blk_idx - 1;
    while (j >= 0) {
      if (db->blk_tab[j].keep) {
        idx = j;
        startline = db->blk_tab[idx].startline;
        endline = db->blk_tab[idx].endline;

        if (lineno >= startline && lineno <= endline)
          break;
      }
      j--;
    }
    if (!db->blk_tab[idx].keep) {
      db->cur_line_mdnode = ll_get_md_null();
      return;
    }
    startline = db->blk_tab[idx].startline;
    endline = db->blk_tab[idx].endline;
    if (lineno >= startline) {
      if (lineno > endline) {
        db->blk_tab[idx].line_mdnodes =
            (LL_MDRef *)realloc(db->blk_tab[idx].line_mdnodes,
                                sizeof(LL_MDRef) * (lineno - startline + 1));
        for (i = lineno; i > endline; i--)
          db->blk_tab[idx].line_mdnodes[i - startline] = ll_get_md_null();
        endline = lineno;
        db->blk_tab[idx].endline = endline;
      }
      db->cur_line_mdnode = db->blk_tab[idx].line_mdnodes[lineno - startline];
      if (LL_MDREF_IS_NULL(db->cur_line_mdnode)) {
        db->cur_line_mdnode = lldbg_create_location_mdnode(
            db, lineno, 1, db->blk_tab[idx].mdnode);
        db->blk_tab[idx].line_mdnodes[lineno - startline] = db->cur_line_mdnode;
      }
    } else {
      db->cur_line_mdnode =
          lldbg_create_location_mdnode(db, lineno, 1, db->blk_tab[idx].mdnode);
    }
    // it is not yet column aware so comparing only line
    if (lineno == db->cur_subprogram_lineno)
      db->cur_subprogram_line_mdnode = db->cur_line_mdnode;

    last_line = lineno;
    last_subprogram_mdnode = db->cur_subprogram_mdnode;
  }
}

LL_MDRef
lldbg_get_line(LL_DebugInfo *db)
{
  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  return db->cur_line_mdnode;
}

LL_MDRef
lldbg_cons_line(LL_DebugInfo *db)
{
  if (LL_MDREF_IS_NULL(db->cur_line_mdnode))
    return db->cur_subprogram_null_loc;
  return lldbg_get_line(db);
}

static int
dwarf_encoding(DTYPE dtype)
{
  TY_KIND ty = DTY(dtype);
  switch (ty) {
  case TY_PTR:
    return DW_ATE_address;
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_INT8:
    return DW_ATE_signed;
  case TY_WORD:
  case TY_DWORD:
    return DW_ATE_unsigned;
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
    return DW_ATE_boolean;
  case TY_REAL:
  case TY_DBLE:
  case TY_QUAD:
    return DW_ATE_float;
  case TY_CMPLX:
  case TY_DCMPLX:
    return DW_ATE_complex_float;
  default:
    break;
  }
  return (TY_ISUNSIGNED(ty)) ? DW_ATE_unsigned : DW_ATE_signed;
}

static LL_MDRef
lldbg_create_outlined_parameters_node(LL_DebugInfo *db)
{
  db->cur_parameters_mdnode = ll_create_flexible_md_node(db->module);
  return db->cur_parameters_mdnode;
}

void
lldbg_emit_outlined_parameter_list(LL_DebugInfo *db, int findex,
                                   DTYPE *param_dtypes, int num_args)
{
  LL_MDRef parameters_mdnode, parameter_mdnode;
  int i;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  parameters_mdnode = db->cur_parameters_mdnode;

  if (ll_feature_debug_info_pre34(&db->module->ir))
    ll_extend_md_node(db->module, parameters_mdnode, ll_get_md_null());
  else if (!ll_feature_debug_info_ver38(&(db)->module->ir))
    ll_extend_md_node(db->module, parameters_mdnode,
                      lldbg_create_unspecified_type_mdnode(db));

  for (i = 0; i < num_args; i++) {
    if (param_dtypes[i]) {
      parameter_mdnode = lldbg_emit_type(db, param_dtypes[i], SPTR_NULL, findex,
                                         false, true, true);
      ll_extend_md_node(db->module, parameters_mdnode, parameter_mdnode);
    }
  }

  if (num_args == 0) {
    parameter_mdnode = lldbg_create_unspecified_parameters_mdnode(db);
    ll_extend_md_node(db->module, parameters_mdnode, parameter_mdnode);
  }
}

static LL_MDRef
lldbg_emit_modified_type(LL_DebugInfo *db, DTYPE dtype, SPTR sptr, int findex)
{
  const TY_KIND dty = DTY(DTYPEG(sptr));
  return lldbg_emit_type(db, (dty == TY_ARRAY ? DT_CPTR : dtype), sptr, findex,
                         false, false, false);
}

#ifdef FLANG_DEBUGINFO_UNUSED
static LL_MDRef
lldbg_emit_accel_cmblk_type(LL_DebugInfo *db, int cmblk, int findex)
{
  LL_MDRef cu_mdnode, file_mdnode, type_mdnode;
  LL_MDRef members_mdnode;
  ISZ_T sz;
  DBLINT64 align;

  cu_mdnode = lldbg_emit_compile_unit(db);
  file_mdnode = lldbg_emit_file(db, findex);
  members_mdnode = ll_create_flexible_md_node(db->module);
  sz = (SIZEG(cmblk) * 8);
  align[1] = 64;
  align[0] = 0;
  type_mdnode = lldbg_create_aggregate_type_mdnode(
      db, DW_TAG_class_type, cu_mdnode, "", file_mdnode, 0, sz, align, 0,
      members_mdnode, 0);
  lldbg_create_aggregate_members_type(db, CMEMFG(cmblk), findex, file_mdnode,
                                      members_mdnode, type_mdnode);
  return type_mdnode;
}

static LL_MDRef
lldbg_emit_accel_function_static_type(LL_DebugInfo *db, SPTR first, int findex)
{
  LL_MDRef cu_mdnode, file_mdnode, type_mdnode;
  LL_MDRef members_mdnode;
  DBLINT64 align;
  int sptr;
  ISZ_T total_size = 0;

  cu_mdnode = lldbg_emit_compile_unit(db);
  file_mdnode = lldbg_emit_file(db, findex);
  members_mdnode = ll_create_flexible_md_node(db->module);
  sptr = first;
  while (sptr > NOSYM) {
    total_size += ZSIZEOF(DTYPEG(sptr));
    sptr = SYMLKG(sptr);
  }
  total_size *= 8;
  align[1] = 64;
  align[0] = 0;
  type_mdnode = lldbg_create_aggregate_type_mdnode(
      db, DW_TAG_class_type, cu_mdnode, "", file_mdnode, 0, total_size, align,
      0, members_mdnode, 0);
  lldbg_create_aggregate_members_type(db, first, findex, file_mdnode,
                                      members_mdnode, type_mdnode);
  return type_mdnode;
}
#endif

INLINE static void
dtype_array_check_set(LL_DebugInfo *db, DTYPE at, LL_MDRef md)
{
  if (at >= 0 && at < db->dtype_array_sz)
    db->dtype_array[at] = md;
}

INLINE static bool
is_assumed_char(DTYPE dtype)
{
  return (DTY(dtype) == TY_CHAR) && (dtype == DT_ASSCHAR);
}

INLINE static bool
is_deferred_char(DTYPE dtype)
{
  return (DTY(dtype) == TY_CHAR) && (dtype == DT_DEFERCHAR);
}

INLINE static char *
next_assumed_len_character_name(void)
{
  static unsigned counter;
  static char name[32];
  snprintf(name, 32, "character(*)!%u", ++counter);
  return name;
}

INLINE static LL_MDRef
lldbg_create_assumed_len_string_type_mdnode(LL_DebugInfo *db, SPTR sptr,
                                            int findex)
{
  SPTR lenArg;
  int paramPos;
  LL_MDRef mdLen;
  LL_MDRef mdLenExp;
  LLMD_Builder mdb = llmd_init(db->module);
  const char *name = next_assumed_len_character_name();
  const long long size = 32;
  const long long alignment = 0;
  const int encoding = 0;

  cg_fetch_clen_parampos(&lenArg, &paramPos, sptr);
  if ((paramPos != -1) && db->cur_subprogram_mdnode) {
    mdLen = lldbg_emit_param_variable(db, lenArg, findex, paramPos, true);
    mdLenExp = lldbg_emit_empty_expression_mdnode(db);
  } else {
    mdLen = mdLenExp = ll_get_md_null();
  }
  llmd_set_class(mdb, LL_DIStringType);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_string_type));
  llmd_add_string(mdb, name);
  llmd_add_i64(mdb, size);
  llmd_add_i64(mdb, alignment);
  llmd_add_i32(mdb, encoding);
  llmd_add_md(mdb, mdLen);
  llmd_add_md(mdb, mdLenExp);
  return llmd_finish(mdb);
}

INLINE static LL_MDRef
lldbg_create_deferred_len_string_type_mdnode(LL_DebugInfo *db, SPTR sptr,
                                            int findex)
{
  LL_MDRef mdLen = ll_get_md_null();
  LL_MDRef mdLenExp = ll_get_md_null();
  LLMD_Builder mdb = llmd_init(db->module);
  const char *name = "character(*)";
  char gbl_arr_len[MAXIDLEN] = {0};
  const long long size = 32;
  const long long alignment = 0;
  const int encoding = 0;

  if (SDSCG(REVMIDLNKG(sptr)) && (DTY(DTYPEG(sptr)) == TY_PTR)) {
    /* get the array descriptor */
    SPTR sdscsptr = SDSCG(REVMIDLNKG(sptr));
    BLKINFO *blk_info = get_lexical_block_info(db, sdscsptr, true);
    LL_MDRef file_mdnode;
    if (ll_feature_debug_info_need_file_descriptions(&db->module->ir))
      file_mdnode = get_filedesc_mdnode(db, findex);
    else
      file_mdnode = lldbg_emit_file(db, findex);

    LL_MDRef type_mdnode =
             lldbg_emit_type(db, DT_INT, sdscsptr, findex, false, false, false);

    if (SCG(sptr) == SC_CMBLK) {
      /* common blocks are global symbols */
      /* create a new global variable that can be referred by string length AT
       * of deferred length array.
       * Length of an array is stored in array descriptor
       * (SDSCG(REVMIDLNKG(sptr))) for local scope. A Local deferred array
       * variable has two parts one each for array and its descriptor.

       * %mdi_options$p_352 = alloca [1 x i8]*, align 8	==> array
       * %mdi_options$sd_351 = alloca [16 x i64], align 8 ==> descriptor

       * But module variables (considered as global variables) are
       * in a single common block variable.

       * %struct_modmdi_0_ = type < { [144 x i8]  } > ==> type of common block
       * @_modmdi_0_=common global %struct_modmdi_0_  zeroinitializer,align 64,
                               ==> global variable that holds array and descr

       * Here, we find the offset of the descriptor(sdscsptr) in a common block
       * structure and add it with the offset of len member of the
       * descriptor's structure to get the array's length.
       */
      strcpy(gbl_arr_len, SYMNAME(sptr));
      strcat(gbl_arr_len, "_len");
      SPTR lensptr = getsymbol(gbl_arr_len);
      /* byte at which length is stored in fortan array  descriptor */
      const int F90_Desc_byte_len = DESC_HDR_INT_LEN
                     * (DESC_HDR_BYTE_LEN - DESC_HDR_TAG);

      /* find the offset of the descriptor in common block struct and
         add it with len_off */
      int len_offset = ADDRESSG(sdscsptr) + F90_Desc_byte_len;

      mdLen = lldbg_create_global_variable_mdnode
                 (db, db->cur_module_mdnode, SYMNAME(lensptr), NULL, NULL,
                  ll_get_md_null(), 0, type_mdnode, 0, 1, NULL, -1,
                  DIFLAG_ARTIFICIAL, len_offset, lensptr, ll_get_md_null());

      /* The created global variable should be linked with the same common
       * block variable as descriptor. So that location metadata  will be
       * generated and use the same base addr as of descriptor
       */
      /* MIDNUMG: cmn_blk variable of the sptr
       * CMEMLG: last element of cmn blk members linked list */
      SYMLKP(CMEMLG(MIDNUMG(sdscsptr)), lensptr);
      CMEMLP(MIDNUMG(sdscsptr), lensptr); //make it as last member

      /* add it to global debug so that it will be added to debug reference of
       * common block variable in "lldbg_create_cmblk_mem_mdnode_list"
       */
      ll_add_global_debug(db->module, lensptr, mdLen);

      /* string length reference should be DIGlobalvariable, get it from
       * DIGlobalExpression
       */
      LL_MDNode *node = db->module->mdnodes[LL_MDREF_value(mdLen) - 1];
      mdLen = node->elem[0];
    } else {
      /* create a local variable to hold the string length */
      mdLen = lldbg_create_local_variable_mdnode(
                        db, DW_TAG_auto_variable, blk_info->mdnode, NULL,
                        file_mdnode, 0, 0, type_mdnode, DIFLAG_ARTIFICIAL,
                        ll_get_md_null(), 1 /*distinct*/);

      /* string length is preserved in DESC_HDR_BYTE_LEN or
       * len field of Fortran descriptor. i.e. offsetof(F90_Desc, len),
       * extract it using !DIExpression
       */
      const int F90_Desc_byte_len =
                   DESC_HDR_INT_LEN * (DESC_HDR_BYTE_LEN - DESC_HDR_TAG);
      const int target_size_offset = F90_Desc_byte_len;
      const unsigned v1 =
       lldbg_encode_expression_arg(LL_DW_OP_int, target_size_offset);
      const unsigned add =
       lldbg_encode_expression_arg(LL_DW_OP_plus_uconst, 0);

      /* emit an @llvm.dbg.declare with required !DIExpression */
      LL_MDRef expr_mdnode = lldbg_emit_expression_mdnode(db, 2, add, v1);
      insert_llvm_dbg_declare(mdLen, sdscsptr, (LL_Type *) NULL,
                            make_mdref_op(expr_mdnode), OPF_NONE);

      mdLenExp = lldbg_emit_empty_expression_mdnode(db);
    }
  }

  llmd_set_class(mdb, LL_DIStringType);
  llmd_add_i32(mdb, make_dwtag(db, DW_TAG_string_type));
  llmd_add_string(mdb, name);
  llmd_add_i64(mdb, size);
  llmd_add_i64(mdb, alignment);
  llmd_add_i32(mdb, encoding);
  llmd_add_md(mdb, mdLen);
  llmd_add_md(mdb, mdLenExp);
  // String length metadata is different for every new deferred array variable
  llmd_set_distinct(mdb);
  return llmd_finish(mdb);
}

static LL_MDRef
lldbg_fwd_local_variable(LL_DebugInfo *db, int sptr, int findex,
                         int emit_dummy_as_local)
{
  LL_MDRef mdnode;
  hash_data_t data;
  if (hashmap_lookup(db->module->mdnodes_fwdvars, INT2HKEY(sptr), &data)) {
    mdnode = (LL_MDRef)(unsigned long)data;
  } else {
    const unsigned id = ll_reserve_md_node(db->module);
    mdnode = LL_MDREF_INITIALIZER(MDRef_Node, id);
    hashmap_insert(db->module->mdnodes_fwdvars, INT2HKEY(sptr),
                   (hash_data_t)INT2HKEY(mdnode));
  }
  return mdnode;
}

/**
   \brief Initialize the subrange bound variables
   \param cb      The value of a constant bound (or 0) [output]
   \param bound_sptr A forward metadata reference for \p sptr [output]
   \param sptr    The symbol corresponding to the bound
   \param defVal  The default value for \p cb, the constant bound
   \param findex  Pass through argument for creating forward reference
   \param db      The debug info
 */
INLINE static void
init_subrange_bound_pre11(LL_DebugInfo *db, ISZ_T *cb, LL_MDRef *bound_sptr,
                          SPTR sptr, ISZ_T defVal, int findex)
{
  if (sptr) {
    switch (STYPEG(sptr)) {
    case ST_CONST:
      *bound_sptr = ll_get_md_null();
      *cb = ad_val_of(sptr);
      return;
    case ST_VAR:
      if (!db->scope_is_global) {
        *bound_sptr = lldbg_fwd_local_variable(db, sptr, findex, false);
        *cb = 0;
        return;
      }
      break;
    default:
      break;
    }
  }
  *bound_sptr = ll_get_md_null();
  *cb = defVal;
}

INLINE static void
init_subrange_bound(LL_DebugInfo *db, LL_MDRef *bound_sptr, SPTR sptr,
                    ISZ_T defVal, int findex)
{
  if (sptr) {
    switch (STYPEG(sptr)) {
    case ST_CONST:
      *bound_sptr = ll_get_md_i64(db->module, ad_val_of(sptr));
      return;
    case ST_VAR:
      if (!db->scope_is_global) {
        *bound_sptr = lldbg_fwd_local_variable(db, sptr, findex, false);
        return;
      }
      break;
    default:
      break;
    }
  }
  *bound_sptr = ll_get_md_i64(db->module, defVal);
}

static LL_MDRef
lldbg_emit_type(LL_DebugInfo *db, DTYPE dtype, SPTR sptr, int findex,
                bool is_reference, bool skip_first_dim,
                bool skipDataDependentTypes, SPTR data_sptr)
{
  LL_MDRef cu_mdnode, file_mdnode, type_mdnode;
  LL_MDRef subscripts_mdnode, subscript_mdnode;
  LL_MDRef elem_type_mdnode;
  LL_MDRef members_mdnode;
  LL_MDRef parameters_mdnode;
  DBLINT64 align, offset;
  ISZ_T sz, lb, ub, dim_ele;
  SPTR element;
  DTYPE elem_dtype;

  dim_ele = 0;
  if (((DTY(dtype) == TY_ARRAY) && skip_first_dim) ||
      (dtype >= db->dtype_array_sz))
    type_mdnode = ll_get_md_null();
  else
    if (db->need_dup_composite_type)
      type_mdnode = ll_get_md_null();
    else
      type_mdnode = db->dtype_array[dtype];
  if (LL_MDREF_IS_NULL(type_mdnode)) {
    if (is_assumed_char(dtype)) {
      /* For assumed length string type, emit !DIStringType metadata node
       * if LLVM version is 11 and above. Here compiler created
       * local variable holds the string length.
       */
      if (ll_feature_debug_info_ver11(&db->module->ir))
        type_mdnode =
            lldbg_create_assumed_len_string_type_mdnode(db, sptr, findex);
      else {
        type_mdnode =
            lldbg_emit_type(db, DT_CPTR, sptr, findex, false, false, false);
#if defined(FLANG_LLVM_EXTENSIONS)
        if (!skipDataDependentTypes) {
#endif
          dtype_array_check_set(db, dtype, type_mdnode);
#if defined(FLANG_LLVM_EXTENSIONS)
        }
#endif
      }
    } else if (is_deferred_char(dtype)) {
        /* For deferred length string type, emit !DIStringType metadata node
         * if LLVM version is 11 and above. Here Fortran descriptor contains
         * the string length.
         */
        if (ll_feature_debug_info_ver11(&db->module->ir))
          type_mdnode =
              lldbg_create_deferred_len_string_type_mdnode(db, sptr, findex);
    } else
        if (DT_ISBASIC(dtype) && (DTY(dtype) != TY_PTR)) {

      cu_mdnode = lldbg_emit_compile_unit(db);
      sz = zsize_of(dtype) * 8;
      if (sz < 0)
        sz = 0; /* do not emit negative sizes */
      align[1] = ((alignment(dtype) + 1) * 8);
      align[0] = 0;
      offset[0] = 0;
      offset[1] = 0;
      cu_mdnode = ll_get_md_null();
#if defined(FLANG_LLVM_EXTENSIONS)
      if (ll_feature_from_global_to_md(&db->module->ir) &&
          (DTY(dtype) == TY_CHAR))
        type_mdnode = lldbg_create_string_type_mdnode(
            db, sz, align, stb.tynames[DTY(dtype)], dwarf_encoding(dtype));
      else
#endif
        type_mdnode = lldbg_create_basic_type_mdnode(
            db, cu_mdnode, stb.tynames[DTY(dtype)], ll_get_md_null(), 0, sz,
            align, offset, 0, dwarf_encoding(dtype));
      dtype_array_check_set(db, dtype, type_mdnode);
    } else if (DTY(dtype) == TY_ANY) {
      cu_mdnode = lldbg_emit_compile_unit(db);
      sz = 0;
      align[1] = 0;
      align[0] = 0;
      offset[0] = 0;
      offset[1] = 0;
      cu_mdnode = ll_get_md_null();
      type_mdnode = lldbg_create_basic_type_mdnode(
          db, cu_mdnode, stb.tynames[DTY(dtype)], ll_get_md_null(), 0, sz,
          align, offset, 0, dwarf_encoding(dtype));
      dtype_array_check_set(db, dtype, type_mdnode);
    } else {
      /* lldbg_emit_compile_unit() must be called at least once before any call
       * to get_filedesc_mdnode() because it establishes db->file_array[]
       * as a hidden side effect.
       */
      cu_mdnode = lldbg_emit_compile_unit(db);

      /* Set file node upfront. `DI' syntax wants a link to a file description,
       * rather than a reference to it */
      if (ll_feature_debug_info_need_file_descriptions(&db->module->ir))
        file_mdnode = get_filedesc_mdnode(db, findex);
      else
        file_mdnode = lldbg_emit_file(db, findex);

      if ((dtype == DT_NONE) && (STYPEG(sptr) == ST_PROC)) {
        type_mdnode = ll_get_md_null();
        dtype_array_check_set(db, dtype, type_mdnode);
        return type_mdnode;
      }
      switch (DTY(dtype)) {
      case TY_PTR: {
        /* Fortran arrays with SDSC and MIDNUM attributes have the type of either
         * Pointer to FortranArrayType or FortranArrayType.
         */
        if (!ll_feature_debug_info_ver90(&cpu_llvm_module->ir)) {
          if (ftn_array_need_debug_info(sptr)) {
            SPTR array_sptr = (SPTR)REVMIDLNKG(sptr);
            type_mdnode = lldbg_emit_type(db, DTYPEG(array_sptr), array_sptr,
                                          findex, false, false, false);
            /* Emit FortranArrayType instead of pointer to FortranArrayType
             * to workaround a known gdb bug not able to debug array bounds.
             * i.e.
             * 1) On POWER, gdb 7.x fails to read array bounds either w/ or
             * w/o the pointer type layer; gdb 8.x only works w/o the pointer
             * type layer.
             * 2) On X86, gdb 7.x works either w/ or w/o the pointer type layer,
             * however, gdb 8.x only works w/o the pointer type layer.
             */
            return type_mdnode;
          }
        }
        type_mdnode = lldbg_emit_type(db, DTySeqTyElement(dtype),
                                      is_procedure_ptr(sptr)
                                          ? DTyInterface(DTySeqTyElement(dtype))
                                          : sptr,
                                      findex, false, false, false);
        sz = (ZSIZEOF(dtype) * 8);
        align[1] = ((alignment(dtype) + 1) * 8);
        align[0] = 0;
        offset[0] = 0;
        offset[1] = 0;
        cu_mdnode = ll_get_md_null();
          type_mdnode = lldbg_create_pointer_type_mdnode(
              db, cu_mdnode, "", ll_get_md_null(), 0, sz, align, offset, 0,
              type_mdnode);
          dtype_array_check_set(db, dtype, type_mdnode);
        break;
      }

      case TY_PFUNC:
      case TY_PROC:
        parameters_mdnode = lldbg_emit_parameter_list(
            db, dtype, DTyReturnType(dtype), sptr, findex);
        type_mdnode = lldbg_create_subroutine_type_mdnode(
            db, cu_mdnode, file_mdnode, parameters_mdnode, 0);
        dtype_array_check_set(db, dtype, type_mdnode);
        break;

      case TY_ARRAY: {
        LLMD_Builder mdb = llmd_init(db->module);
        LL_MDRef dataloc = ll_get_md_null();
        LL_MDRef is_live = ll_get_md_null();
        LL_MDRef associated = ll_get_md_null();
        LL_MDRef allocated = ll_get_md_null();
        LL_MDRef rank = ll_get_md_null();
        ADSC *ad;
        int i, numdim;
        elem_dtype = DTySeqTyElement(dtype);
        sz = zsize_of(dtype) * 8;
        if (sz < 0)
          sz = 0; /* don't emit debug with negative sizes */
        align[0] = 0;
        align[1] = (alignment(dtype) + 1) * 8;
        ad = AD_DPTR(dtype);
        numdim = AD_NUMDIM(ad);
        if ((!ll_feature_debug_info_ver90(&db->module->ir) ||
             db->module->ir.dwarf_version < LL_DWARF_Version_5) &&
            data_sptr && ASSUMRANKG(data_sptr)) {
          // Set dimension of array to maximum for DWARF version lower than5
          numdim = 7;
        }
        if (numdim >= 1 && numdim <= 7) {
          // Generate dataLocation field DW_TAG_array_type for assumed shape
          // arrays, pointers and allocatables. For pointers and allocatables
          // generate allocated / associated.
          if (ll_feature_debug_info_ver90(&db->module->ir)) {
            if ((SCG(sptr) == SC_DUMMY) && data_sptr &&
                db->cur_subprogram_mdnode) {
              // Assumed shape array
              LL_Type *dataloctype = LLTYPE(data_sptr);
              /* make_lltype_from_sptr() should have added a pointer to
               * the type of this local variable. Remove it */
              if (!dataloctype)
                dataloctype = make_lltype_from_sptr(data_sptr);
              if (dataloctype->data_type == LL_PTR)
                dataloctype = dataloctype->sub_types[0];
              if (SCG(data_sptr) == SC_DUMMY) {
                LL_MDRef type_mdnode = lldbg_emit_type(
                    db, __POINT_T, data_sptr, findex, false, false, false);
                int parnum_lldbg = 0;
                if (has_multiple_entries(gbl.currsub))
                  parnum_lldbg = get_entry_parnum(data_sptr);
                else
                  parnum_lldbg = get_parnum(data_sptr);
                dataloc = lldbg_create_local_variable_mdnode(
                    db, DW_TAG_arg_variable, db->cur_subprogram_mdnode, NULL,
                    file_mdnode, db->cur_subprogram_lineno,
                    parnum_lldbg, type_mdnode,
                    set_dilocalvariable_flags(data_sptr), ll_get_md_null());
                lldbg_register_param_mdnode(db, dataloc, data_sptr);

              } else
                dataloc =
                    lldbg_emit_local_variable(db, data_sptr, findex, true);

              OPERAND *ld = make_operand();
              ld->ot_type = OT_MDNODE;
              ld->val.sptr = data_sptr;

              /* lets generate llvm.dbg.value intrinsic for it.*/
              insert_llvm_dbg_value(ld, dataloc, data_sptr, dataloctype);
            } else if (ALLOCATTRG(sptr) || POINTERG(sptr)) {
              // Variables with allocatable/pointer attribute.
              if (SCG(SDSCG(sptr)) == SC_CMBLK ||
                  STYPEG(SDSCG(sptr)) == ST_MEMBER) {
                const unsigned deref =
                    lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
                const unsigned pushobj = lldbg_encode_expression_arg(
                    LL_DW_OP_push_object_address, 0);
                dataloc = lldbg_emit_expression_mdnode(db, 2, pushobj, deref);
                if (ll_feature_debug_info_ver90(&db->module->ir)) {
                  is_live = lldbg_emit_expression_mdnode(db, 2, pushobj, deref);
                  if (ALLOCATTRG(sptr))
                    allocated = is_live;
                  else
                    associated = is_live;
                }
              } else {
                SPTR datasptr = MIDNUMG(sptr);
                if (datasptr == NOSYM)
                  datasptr = SYMLKG(sptr);
                if ((SCG(sptr) == SC_DUMMY) || ((SCG(datasptr) == SC_DUMMY) &&
                                                !db->cur_subprogram_mdnode)) {
                  const unsigned zero =
                      lldbg_encode_expression_arg(LL_DW_OP_int, 0);
                  const unsigned constu =
                      lldbg_encode_expression_arg(LL_DW_OP_constu, 0);
                  dataloc = lldbg_emit_expression_mdnode(db, 2, constu, zero);
                  if (ll_feature_debug_info_ver90(&db->module->ir)) {
                    is_live = lldbg_emit_expression_mdnode(db, 2, constu, zero);
                    if (ALLOCATTRG(sptr))
                      allocated = is_live;
                    else
                      associated = is_live;
                  }
                  // If cur_subprogram_md is not yet ready, we are interested
                  // only in type. datalocation is about value than type. So
                } else {
                  LL_Type *dataloctype = LLTYPE(datasptr);
                  /* make_lltype_from_sptr() should have added a pointer to
                   * the type of this local variable. Remove it */
                  if (!dataloctype)
                    dataloctype = make_lltype_from_sptr(datasptr);
                  if (dataloctype->data_type == LL_PTR)
                    dataloctype = dataloctype->sub_types[0];
                  if (SCG(datasptr) == SC_DUMMY) {
                    LL_MDRef type_mdnode = lldbg_emit_type(
                        db, __POINT_T, datasptr, findex, false, false, false);
                    int parnum_lldbg = 0;
                    if (has_multiple_entries(gbl.currsub))
                      parnum_lldbg = get_entry_parnum(data_sptr);
                    else
                      parnum_lldbg = get_parnum(data_sptr);
                    dataloc = lldbg_create_local_variable_mdnode(
                        db, DW_TAG_arg_variable, db->cur_subprogram_mdnode,
                        NULL, file_mdnode, db->cur_subprogram_lineno,
                        parnum_lldbg, type_mdnode,
                        set_dilocalvariable_flags(datasptr), ll_get_md_null());
                    lldbg_register_param_mdnode(db, dataloc, datasptr);
                  } else
                    dataloc =
                        lldbg_emit_local_variable(db, datasptr, findex, true);
                  insert_llvm_dbg_declare(dataloc, datasptr, dataloctype, NULL,
                                          OPF_NONE);
                  if (ll_feature_debug_info_ver90(&db->module->ir)) {
                    BLKINFO *blk_info = get_lexical_block_info(db, sptr, true);
                    LL_MDRef type_mdnode = lldbg_emit_type(
                        db, DT_LOG, sptr, findex, false, false, false);
                    is_live = lldbg_create_local_variable_mdnode(
                        db, DW_TAG_auto_variable, blk_info->mdnode, NULL,
                        file_mdnode, 0, 0, type_mdnode, DIFLAG_ARTIFICIAL,
                        ll_get_md_null(), 1 /*distinct*/);

                    if (ALLOCATTRG(sptr))
                      allocated = is_live;
                    else
                      associated = is_live;

                    insert_llvm_dbg_declare(is_live, datasptr, dataloctype,
                                            NULL, OPF_NONE);
                  }
                }
              }
            }
          }
          // For DWARF version 5 and greater make use of DW_OP_rank and
          // DW_TAG_generic_subrange for assumed rank array.
          if (ll_feature_debug_info_ver90(&db->module->ir) &&
              db->module->ir.dwarf_version >= LL_DWARF_Version_5 && data_sptr &&
              ASSUMRANKG(data_sptr)) {
            LL_MDRef lbnd_expr_mdnode, ubnd_expr_mdnode, stride_expr_mdnode;

            const unsigned pushobj =
                lldbg_encode_expression_arg(LL_DW_OP_push_object_address, 0);
            const unsigned v1 = lldbg_encode_expression_arg(LL_DW_OP_int, 8);
            const unsigned v2 =
                lldbg_encode_expression_arg(LL_DW_OP_int, 7);
            const unsigned add =
                lldbg_encode_expression_arg(LL_DW_OP_plus_uconst, 0);
            const unsigned deref =
                lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
            const unsigned constu =
                lldbg_encode_expression_arg(LL_DW_OP_constu, 0);
            const unsigned op_and =
                lldbg_encode_expression_arg(LL_DW_OP_and, 0);
            // Get rank of assumed rank array from descriptor
            rank = lldbg_emit_expression_mdnode(db, 7, pushobj, add, v1, deref,
                                                constu, v2, op_and);
            // Generate generic subrange
            lldbg_get_bounds_for_assumed_rank_sdsc(
                db, data_sptr, &lbnd_expr_mdnode, &ubnd_expr_mdnode,
                &stride_expr_mdnode);
            subscript_mdnode = lldbg_create_generic_subrange_mdnode(
                db, lbnd_expr_mdnode, ubnd_expr_mdnode, stride_expr_mdnode);
            llmd_add_md(mdb, subscript_mdnode);
          } else {

            for (i = 0; i < numdim; ++i) {
              SPTR lower_bnd = AD_LWBD(ad, i);
              SPTR upper_bnd = AD_UPBD(ad, i);
              if (ll_feature_debug_info_ver90(&db->module->ir)) {
                LL_MDRef lbv = ll_get_md_null();
                LL_MDRef ubv = ll_get_md_null();
                LL_MDRef st = ll_get_md_null();
                if ((ll_feature_debug_info_ver90(&db->module->ir) &&
                     db->module->ir.dwarf_version < LL_DWARF_Version_5 &&
                     data_sptr && ASSUMRANKG(data_sptr)) ||
                    ALLOCATTRG(sptr) || POINTERG(sptr)) {
                  /* Create subrange mdnode based on array descriptor */
                  subscript_mdnode = lldbg_create_subrange_via_sdsc(
                      db, findex,
                      (data_sptr && ASSUMRANKG(data_sptr)) ? data_sptr : sptr,
                      i);
                } else if ((SCG(sptr) == SC_DUMMY) && data_sptr &&
                           db->cur_subprogram_mdnode) {
                  // assumed shape array
                  LL_MDRef count, s_bnd;
                  init_subrange_bound(db, &lbv, lower_bnd, 1, findex);
                  init_subrange_bound(db, &ubv, upper_bnd, 0, findex);
                  lldbg_get_bounds_for_sdsc(db, findex, data_sptr, i, &count,
                                            NULL, NULL, &s_bnd);

                  if (ll_feature_debug_info_ver11(&db->module->ir))
                    subscript_mdnode = lldbg_create_subrange_mdnode(
                        db, count, lbv, ll_get_md_null(), s_bnd);
                  else
                    subscript_mdnode = lldbg_create_subrange_mdnode(
                        db, ll_get_md_null(), lbv, ubv, s_bnd);
                } else {
                  // explicit shape array, assumed size array
                  init_subrange_bound(db, &lbv, lower_bnd, 1, findex);
                  if (!ll_feature_debug_info_ver90(&db->module->ir) ||
                      (upper_bnd != SPTR_NULL)) // assumed size
                    init_subrange_bound(db, &ubv, upper_bnd, 0, findex);

                  subscript_mdnode =
                      lldbg_create_subrange_mdnode(db, ll_get_md_null(), lbv, ubv, st);
                }
                llmd_add_md(mdb, subscript_mdnode);
              } else if (ll_feature_has_diextensions(&db->module->ir)) {
                // use PGI metadata extensions
                LL_MDRef lbv;
                LL_MDRef ubv;
                if (SDSCG(sptr) && MIDNUMG(sptr) &&
                    !(lower_bnd && STYPEG(lower_bnd) == ST_CONST && upper_bnd &&
                      STYPEG(upper_bnd) == ST_CONST)) {
                  /* Create subrange mdnode based on array descriptor */
                  subscript_mdnode =
                      lldbg_create_ftn_subrange_via_sdsc(db, findex, sptr, i);
                } else {
                  const ISZ_T M = 1ul << ((sizeof(ISZ_T) * 8) - 1);
                  init_subrange_bound_pre11(db, &lb, &lbv, lower_bnd, 1,
                                            findex);
                  init_subrange_bound_pre11(db, &ub, &ubv, upper_bnd, M,
                                            findex);
                  subscript_mdnode =
                      lldbg_create_ftn_subrange_mdnode(db, lb, lbv, ub, ubv);
                }
                llmd_add_md(mdb, subscript_mdnode);
              } else {
                // cons the old debug metadata
                if (lower_bnd && STYPEG(lower_bnd) == ST_CONST && upper_bnd &&
                    STYPEG(upper_bnd) == ST_CONST) {
                  lb = ad_val_of(lower_bnd); /* or get_isz_cval() */
                  if (upper_bnd)
                    ub = ad_val_of(upper_bnd); /* or get_isz_cval() */
                  else
                    ub = 0; /* error or zero-size */
                  subscript_mdnode =
                      lldbg_create_subrange_mdnode_pre11(db, lb, ub);
                  llmd_add_md(mdb, subscript_mdnode);
                } else {
                  subscript_mdnode =
                      lldbg_create_subrange_mdnode_pre11(db, 1, 1);
                  llmd_add_md(mdb, subscript_mdnode);
                }
              }
            }
          }
        }
        elem_type_mdnode =
            lldbg_emit_type(db, elem_dtype, sptr, findex, false, false, false);
        cu_mdnode = ll_get_md_null();
        subscripts_mdnode = llmd_finish(mdb);
        if (ll_feature_debug_info_ver90(&db->module->ir)) {
          type_mdnode = lldbg_create_array_type_mdnode(
              db, cu_mdnode, 0, sz, align, elem_type_mdnode, subscripts_mdnode,
              dataloc, associated, allocated, rank);
        } else if (ll_feature_has_diextensions(&db->module->ir)) {
          type_mdnode = lldbg_create_ftn_array_type_mdnode(
              db, cu_mdnode, 0, sz, align, elem_type_mdnode, subscripts_mdnode);
        } else
          type_mdnode = lldbg_create_array_type_mdnode(
              db, cu_mdnode, 0, sz, align, elem_type_mdnode, subscripts_mdnode,
              dataloc, associated, allocated, rank);
        dtype_array_check_set(db, dtype, type_mdnode);
        break;
      }
      case TY_UNION:
        members_mdnode = ll_create_flexible_md_node(db->module);
        sz = (ZSIZEOF(dtype) * 8);
        align[1] = ((alignment(dtype) + 1) * 8);
        align[0] = 0;
        type_mdnode = lldbg_create_union_type_mdnode(
            db, cu_mdnode,
            DTyAlgTyTag(dtype) ? SYMNAME(DTyAlgTyTag(dtype)) : "", file_mdnode,
            0, sz, align, 0, members_mdnode, 0);
        FLANG_FALLTHROUGH;
      case TY_STRUCT:
        if (LL_MDREF_IS_NULL(type_mdnode)) {
          members_mdnode = ll_create_flexible_md_node(db->module);
          sz = (ZSIZEOF(dtype) * 8);
          align[1] = ((alignment(dtype) + 1) * 8);
          align[0] = 0;
          if (!ll_feature_debug_info_pre34(&db->module->ir))
            file_mdnode = get_filedesc_mdnode(db, findex);
          type_mdnode = lldbg_create_structure_type_mdnode(
              db, cu_mdnode,
              DTyAlgTyTag(dtype) ? SYMNAME(DTyAlgTyTag(dtype)) : "",
              file_mdnode, 0, sz, align, 0, members_mdnode, 0);
        }
        dtype_array_check_set(db, dtype, type_mdnode);
        element = DTyAlgTyMember(dtype);
        lldbg_create_aggregate_members_type(db, element, findex, file_mdnode,
                                            members_mdnode, type_mdnode);
        break;
      case TY_VECT: {
        LLMD_Builder ssmdb = llmd_init(db->module);
        /* TODO: Check that vector datatype is what's expected by LLVM */
        lb = 0;
        ub = DTyVecLength(dtype) - 1;
        if (ll_feature_debug_info_ver90(&db->module->ir))
          subscript_mdnode = lldbg_create_subrange_mdnode(
              db, ll_get_md_null(), ll_get_md_i64(db->module, lb),
              ll_get_md_i64(db->module, ub), ll_get_md_null());
        else
          subscript_mdnode =
              lldbg_create_subrange_mdnode_pre11(db, lb, DTyVecLength(dtype));

        llmd_add_md(ssmdb, subscript_mdnode);
        subscripts_mdnode = llmd_finish(ssmdb);
        sz = (ZSIZEOF(dtype) * 8);
        align[1] = ((alignment(dtype) + 1) * 8);
        align[0] = 0;
        type_mdnode = lldbg_emit_type(db, DTySeqTyElement(dtype), SPTR_NULL,
                                      findex, false, false, false);
        type_mdnode = lldbg_create_vector_type_mdnode(
            db, cu_mdnode, sz, align, type_mdnode, subscripts_mdnode);
        dtype_array_check_set(db, dtype, type_mdnode);
        break;
      }
      default:
        assert(0, "dtype not yet handled ", DTY(dtype), ERR_Fatal);
      }
    }
  }

  if (is_reference) {
    if (DT_ISBASIC(dtype) && DTY(dtype) != TY_PTR) {
      cu_mdnode = lldbg_emit_compile_unit(db);
      sz = (ZSIZEOF(DT_CPTR) * 8);
      align[1] = ((alignment(DT_CPTR) + 1) * 8);
      align[0] = 0;
      offset[0] = 0;
      offset[1] = 0;
      cu_mdnode = ll_get_md_null();
      type_mdnode = lldbg_create_pointer_type_mdnode(
          db, cu_mdnode, "", ll_get_md_null(), 0, sz, align, offset, 0,
          type_mdnode);
    } else if (DTY(dtype) == TY_ARRAY) {
      cu_mdnode = lldbg_emit_compile_unit(db);
      sz = (ZSIZEOF(DT_CPTR) * 8);
      align[1] = ((alignment(DT_CPTR) + 1) * 8);
      align[0] = 0;
      offset[0] = 0;
      offset[1] = 0;
      cu_mdnode = ll_get_md_null();
      type_mdnode = lldbg_create_pointer_type_mdnode(
          db, cu_mdnode, "", ll_get_md_null(), 0, sz, align, offset, 0,
          type_mdnode);
    }
  }
  return type_mdnode;
}

static void
llObjtodbgAddUnique(LL_ObjToDbgList *odl, LL_MDRef mdadd)
{
  LL_ObjToDbgListIter i;
  for (llObjtodbgFirst(odl, &i); !llObjtodbgAtEnd(&i); llObjtodbgNext(&i)) {
    LL_MDRef md = llObjtodbgGet(&i);
    if (md == mdadd)
      return;
  }
  llObjtodbgPush(odl, mdadd);
}

void
lldbg_emit_global_variable(LL_DebugInfo *db, SPTR sptr, ISZ_T off, int findex,
                           LL_Value *value)
{
  LL_MDRef scope_mdnode, file_mdnode, type_mdnode, mdref, fwd;
  int sc, decl_line, is_local, flags;
  const char *display_name;
  bool savedScopeIsGlobal;
  hash_data_t val;

  // Dont emit if it is uplevel variable
  if (sptr && UPLEVELG(sptr))
    return;

  /* A deferred array variable inside module will create a new
     Global variable mdnode for array length and add it to end of
     common block members while processing itself.
     If it comes here as part of common block variables from
     "add_debug_cmnblk_variables", exit
  */

  if (!LL_MDREF_IS_NULL(ll_get_global_debug(cpu_llvm_module, sptr)))
    return;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  if ((!sptr) || (!DTYPEG(sptr)))
    return;
  savedScopeIsGlobal = db->scope_is_global;
  db->scope_is_global = true;
  db->gbl_var_sptr = sptr;
  SPTR new_sptr = (SPTR)REVMIDLNKG(sptr);
  get_extra_info_for_sptr(&display_name, &scope_mdnode, &type_mdnode, db, sptr);
  if (ll_feature_debug_info_ver90(&cpu_llvm_module->ir) && CCSYMG(sptr) &&
      new_sptr &&
      (is_procedure_ptr(new_sptr) ||
       ((STYPEG(new_sptr) == ST_ARRAY) &&
        (POINTERG(new_sptr) || ALLOCATTRG(new_sptr)) && SDSCG(new_sptr)))) {
    type_mdnode = lldbg_emit_type(db, DTYPEG(new_sptr), new_sptr, findex, false,
                                  false, false);
    display_name = SYMNAME(new_sptr);
    flags = CCSYMG(new_sptr) ? DIFLAG_ARTIFICIAL : 0;
  } else {
    type_mdnode =
        lldbg_emit_type(db, DTYPEG(sptr), sptr, findex, false, false, false);
    flags = CCSYMG(sptr) ? DIFLAG_ARTIFICIAL : 0;
  }
  file_mdnode = ll_feature_debug_info_need_file_descriptions(&db->module->ir)
                    ? get_filedesc_mdnode(db, findex)
                    : lldbg_emit_file(db, findex);
  sc = SCG(sptr);
  decl_line = DECLLINEG(sptr);
  if (!decl_line)
    decl_line = FUNCLINEG(sptr);
  is_local = (sc == SC_STATIC);
  if (hashmap_lookup(db->module->mdnodes_fwdvars, INT2HKEY(sptr), &val)) {
    fwd = (LL_MDRef)(unsigned long)val;
    hashmap_erase(db->module->mdnodes_fwdvars, INT2HKEY(sptr), NULL);
  } else {
    fwd = ll_get_md_null();
  }
  if (!ll_feature_debug_info_ver90(&db->module->ir)
        || pointer_scalar_need_debug_info(sptr)) {
    if (ftn_array_need_debug_info(sptr)) {
      SPTR array_sptr = (SPTR)REVMIDLNKG(sptr);
      /* Overwrite the display_name and flags to represent the user defined
       * array instead of a compiler generated symbol of array pointer.
       */
      display_name = SYMNAME(array_sptr);
      flags = 0;
    }
  }
  mdref = lldbg_create_global_variable_mdnode(
      db, scope_mdnode, display_name, SYMNAME(sptr), "", file_mdnode, decl_line,
      type_mdnode, is_local, DEFDG(sptr) || (sc != SC_EXTERN), value, -1, flags,
      off, sptr, fwd);

  if (!ll_feature_debug_info_ver90(&db->module->ir)) {
    if (!LL_MDREF_IS_NULL(db->gbl_obj_mdnode)) {
      if (LL_MDREF_IS_NULL(db->gbl_obj_exp_mdnode)) {
        /* Create a dummy global var expression mdnode to be associated to
         * the global static object.
         */
        db->gbl_obj_exp_mdnode = lldbg_create_global_variable_mdnode(
            db, scope_mdnode, "", "", "", file_mdnode, 0, type_mdnode, 0, 0,
            NULL, -1, DIFLAG_ARTIFICIAL, 0, SPTR_NULL, db->gbl_obj_mdnode);
      }
      if (db->need_dup_composite_type) {
        /* erase dtype record to allow duplication for allocatable array type
         * within derived type.
         */
        dtype_array_check_set(db, DTYPEG(sptr), ll_get_md_null());
        db->need_dup_composite_type = false;
      }
    }
  }
  db->gbl_var_sptr = SPTR_NULL;

  if (!LL_MDREF_IS_NULL(mdref)) {
    LL_ObjToDbgList **listp = llassem_get_objtodbg_list(sptr);
    if (listp) {
      if (!*listp)
        *listp = llObjtodbgCreate();
      llObjtodbgAddUnique(*listp, mdref);
      // Associate a dummy global var mdnode to the global static object.
      if (!LL_MDREF_IS_NULL(db->gbl_obj_exp_mdnode))
        llObjtodbgAddUnique(*listp, db->gbl_obj_exp_mdnode);
    }
    ll_add_global_debug(db->module, sptr, mdref);
    // `RU_SUBR/RU_PROG/RU_FUNC` is set for modules imported from different
    // CompileUnits. For same compilation unit, 'RU_BDATA' is set.
    if (sc == SC_CMBLK) {
      const char *modvar_name;
      if (CCSYMG(MIDNUMG(sptr))) {
        modvar_name = new_debug_name(SYMNAME(ENCLFUNCG(sptr)),
                                     SYMNAME(sptr), NULL);
      } else {
        modvar_name = new_debug_name(SYMNAME(ENCLFUNCG(sptr)),
                                     SYMNAME(MIDNUMG(sptr)), SYMNAME(sptr));
      }
      if (ll_feature_from_global_to_md(&db->module->ir)) {
        /* Lookup the !DIGlobalVariable node instead of the
         * !DIGlobalVariableExpression node. This step is needed
         * by the lldbg_create_imported_entity() */
        LL_MDNode *node = db->module->mdnodes[LL_MDREF_value(mdref) - 1];
        mdref = node->elem[0];
      }
      ll_add_module_debug(db->module->modvar_debug_map, modvar_name, mdref);
    }
  }
  db->scope_is_global = savedScopeIsGlobal;
}

#ifdef FLANG_ACCEL
static const char *
lldbg_get_addrspace_suffix(int addrspace)
{
  switch (addrspace) {
  case 1:
    return "_gbl";
  case 4:
    return "_cst";
  default:
    return "_gen";
  }
}
#endif

static void
lldbg_cancel_value_call(LL_DebugInfo *db, SPTR sptr)
{
  int i;
  for (i = 0; i < db->param_idx; i++)
    if (db->param_stack[i].sptr == sptr)
      db->param_stack[i].instr->flags |= CANCEL_CALL_DBG_VALUE;
}

void
lldbg_register_value_call(LL_DebugInfo *db, INSTR_LIST *instr, int sptr)
{
  int i;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);

  for (i = 0; i < db->param_idx; i++) {
    if (db->param_stack[i].sptr == sptr) {
      db->param_stack[i].instr = instr;
      return;
    }
  }
  assert(db->param_idx < PARAM_STACK_SIZE,
         "lldbg_register_value_call(), param stack is full", sptr, ERR_Fatal);
  db->param_stack[db->param_idx].instr = instr;
  db->param_stack[db->param_idx].mdnode = ll_get_md_null();
  db->param_stack[db->param_idx].sptr = sptr;
  db->param_idx++;
}

LL_MDRef
get_param_mdnode(LL_DebugInfo *db, int sptr)
{
  int i;
  for (i = 0; i < db->param_idx; i++)
    if (db->param_stack[i].sptr == sptr)
      return db->param_stack[i].mdnode;
  return ll_get_md_null();
}

static void
lldbg_register_param_mdnode(LL_DebugInfo *db, LL_MDRef mdnode, int sptr)
{
  int i;
  for (i = 0; i < db->param_idx; i++) {
    if (db->param_stack[i].sptr == sptr) {
      db->param_stack[i].mdnode = mdnode;
      return;
    }
  }
  assert(db->param_idx < PARAM_STACK_SIZE,
         "lldbg_register_param_mdnode(),"
         " param stack is full",
         sptr, ERR_Fatal);
  db->param_stack[db->param_idx].instr = NULL;
  db->param_stack[db->param_idx].mdnode = mdnode;
  db->param_stack[db->param_idx].sptr = sptr;
  db->param_idx++;
}

void
lldbg_emit_lv_list(LL_DebugInfo *db)
{
  int i;
  LL_MDRef lv_list_mdnode;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  assert(db->routine_idx > 0 && db->routine_idx <= db->routine_count,
         "lldbg_emit_lv_list(), invalid routine_idx", 0, ERR_Fatal);
  lv_list_mdnode = db->llvm_dbg_lv_array[db->routine_idx - 1];
  for (i = 0; i < db->param_idx; i++) {
    if (!LL_MDREF_IS_NULL(db->param_stack[i].mdnode) &&
        ((db->param_stack[i].instr == NULL) ||
         !(db->param_stack[i].instr->flags & CANCEL_CALL_DBG_VALUE)))
      ll_extend_md_node(db->module, lv_list_mdnode, db->param_stack[i].mdnode);
  }
}

/**
   \brief Construct the flag set that corresponds with LLVM metadata
 */
INLINE static int
set_dilocalvariable_flags(int sptr)
{

#ifdef THISG
  if (ENCLFUNCG(sptr) && THISG(ENCLFUNCG(sptr)) == sptr) {
    return DIFLAG_ARTIFICIAL;
  }
#endif
  /* Mark the variable as artificial if (Compiler Created Symbol)
   * flag is set except in the case of function result variable.
   * This is done because Function result variable is available
   * in source. So user expect it to be visible in debugger.
   */
  if (CCSYMG(sptr) && (FVALG(GBL_CURRFUNC) != sptr)) {
    return DIFLAG_ARTIFICIAL;
  } else {
    return 0;
  }
}

LL_MDRef
lldbg_emit_local_variable(LL_DebugInfo *db, SPTR sptr, int findex,
                          int emit_dummy_as_local)
{
  LL_MDRef file_mdnode, type_mdnode, var_mdnode;
  char *symname;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  if (ll_feature_debug_info_need_file_descriptions(&db->module->ir))
    file_mdnode = get_filedesc_mdnode(db, findex);
  else
    file_mdnode = lldbg_emit_file(db, findex);

  SPTR new_sptr = (SPTR)REVMIDLNKG(sptr);
  /* If it's an associate statement, associating another variable
   * take the pointer to type of associated variable.*/
  if (new_sptr && CCSYMG(sptr) && !SDSCG(new_sptr) &&
      (ADDRTKNG(new_sptr) || is_procedure_ptr(new_sptr))) {
    if (is_procedure_ptr(new_sptr))
      type_mdnode =
          lldbg_emit_type(db, DTySeqTyElement(DTYPEG(new_sptr)),
                          DTyInterface(DTySeqTyElement(DTYPEG(new_sptr))),
                          findex, false, false, false);
    else
      type_mdnode = lldbg_emit_type(db, DTYPEG(new_sptr), sptr, findex, false,
                                    false, false);
    DBLINT64 align = {0};
    DBLINT64 offset = {0};
    offset[0] = offset[1] = 0;
    align[1] = ((alignment(DT_CPTR) + 1) * 8);
    type_mdnode = lldbg_create_pointer_type_mdnode(
        db, lldbg_emit_compile_unit(db), "", ll_get_md_null(), 0,
        (ZSIZEOF(DT_CPTR) * 8), align, offset, 0, type_mdnode);

  } else if (ll_feature_debug_info_ver90(&db->module->ir) &&
      (ASSUMRANKG(sptr) || ASSUMSHPG(sptr)) && SDSCG(sptr))
    type_mdnode =
        lldbg_emit_type(db, __POINT_T, sptr, findex, false, false, false);
  else
    type_mdnode =
        lldbg_emit_type(db, DTYPEG(sptr), sptr, findex, false, false, false);
#ifdef THISG
  if (ENCLFUNCG(sptr) && THISG(ENCLFUNCG(sptr)) == sptr) {
    symname = "this";
  } else
#endif
  {
    symname = (char *)lldbg_alloc(strlen(SYMNAME(sptr)) + 1);
    strcpy(symname, SYMNAME(sptr));
  }
  if (SCG(sptr) == SC_DUMMY && !emit_dummy_as_local) {
    lldbg_cancel_value_call(db, sptr);
    var_mdnode = get_param_mdnode(db, sptr);
    assert(!LL_MDREF_IS_NULL(var_mdnode),
           "lldbg_emit_local_variable: parameter not in param stack for sptr ",
           sptr, ERR_Fatal);
  } else {
    int flags = set_dilocalvariable_flags(sptr);

    // This is base address of Assumed shape array, need to be used as
    // dataLocation field of DW_TAG_array_type. Make it artificial.
    if (ll_feature_debug_info_ver90(&db->module->ir) && ASSUMSHPG(sptr) &&
        SDSCG(sptr))
      flags = DIFLAG_ARTIFICIAL;

    BLKINFO *blk_info = get_lexical_block_info(db, sptr, true);
    LL_MDRef fwd;
    hash_data_t val;
    if (hashmap_lookup(db->module->mdnodes_fwdvars, INT2HKEY(sptr), &val)) {
      fwd = (LL_MDRef)(unsigned long)val;
      hashmap_erase(db->module->mdnodes_fwdvars, INT2HKEY(sptr), NULL);
    } else {
      fwd = ll_get_md_null();
    }
    if (ll_feature_debug_info_ver90(&db->module->ir) &&
        !pointer_scalar_need_debug_info(sptr)) {
      if (SDSCG(sptr))
        sptr = SDSCG(sptr);
    } else if (ftn_array_need_debug_info(sptr) ||
               pointer_scalar_need_debug_info(sptr)) {
      SPTR array_sptr =(SPTR)REVMIDLNKG(sptr);
      /* Overwrite the symname and flags to represent the user defined array
       * instead of a compiler generated symbol of array or scalar pointer.
       */
      symname = (char *)lldbg_alloc(strlen(SYMNAME(array_sptr)) + 1);
      strcpy(symname, SYMNAME(array_sptr));
      flags = 0;
    }
    var_mdnode = lldbg_create_local_variable_mdnode(
        db, DW_TAG_auto_variable, blk_info->mdnode,
        PASSBYVALG(sptr) ? SYMNAME(MIDNUMG(sptr)) : symname,
        file_mdnode, 0, 0, type_mdnode, flags, fwd, sptr);
  }
  return var_mdnode;
}

typedef struct {
  LL_DebugInfo *db;
  int findex;
} CleanupBounds_t;

static void
cleanup_bounds(hash_key_t ksptr, hash_data_t dmdnode, void *ctxt)
{
  CleanupBounds_t *s = (CleanupBounds_t *)ctxt;
  LL_DebugInfo *db = s->db;
  const int findex = s->findex;
  const SPTR sptr = (SPTR)HKEY2INT(ksptr);
  lldbg_emit_local_variable(db, sptr, findex, true);
}

void
lldbg_cleanup_missing_bounds(LL_DebugInfo *db, int findex)
{
  if (hashmap_size(db->module->mdnodes_fwdvars)) {
    CleanupBounds_t s = {db, findex};
    hashmap_iterate(db->module->mdnodes_fwdvars, cleanup_bounds, &s);
    hashmap_clear(db->module->mdnodes_fwdvars);
  }
}

LL_MDRef
lldbg_emit_param_variable(LL_DebugInfo *db, SPTR sptr, int findex, int parnum,
                          bool unnamed)
{
  LL_MDRef file_mdnode, type_mdnode, var_mdnode = 0;
  char *symname;
  bool is_reference;
  DTYPE dtype;
  int flags;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  // FIXME: Initialize var_mdnode.
  if (LL_MDREF_IS_NULL(db->cur_subprogram_mdnode))
    return var_mdnode;
  if (ll_feature_debug_info_need_file_descriptions(&db->module->ir))
    file_mdnode = get_filedesc_mdnode(db, findex);
  else
    file_mdnode = lldbg_emit_file(db, findex);
  is_reference = ((SCG(sptr) == SC_DUMMY) && HOMEDG(sptr) && !PASSBYVALG(sptr));
  dtype = DTYPEG(sptr) ? DTYPEG(sptr) : DT_ADDR;
  if (ll_feature_debug_info_ver90(&db->module->ir)) {
    if ((ASSUMRANKG(sptr) || ASSUMSHPG(sptr)) && SDSCG(sptr)) {
      type_mdnode = lldbg_emit_type(db, dtype, SDSCG(sptr), findex,
                                    is_reference, true, false, sptr);
      if (has_multiple_entries(gbl.currsub))
        parnum = get_entry_parnum(SDSCG(sptr));
      else
        parnum = get_parnum(SDSCG(sptr));
    } else if (STYPEG(sptr) == ST_ARRAY &&
               (ALLOCATTRG(sptr) || POINTERG(sptr)) && SDSCG(sptr)) {
      type_mdnode = lldbg_emit_type(db, dtype, sptr, findex, is_reference, true,
                                    false, MIDNUMG(sptr));
      if (has_multiple_entries(gbl.currsub))
        parnum = get_entry_parnum(SDSCG(sptr));
      else
        parnum = get_parnum(SDSCG(sptr));
    } else {
      type_mdnode =
          lldbg_emit_type(db, dtype, sptr, findex, is_reference, true, false);
    }
  } else {
    type_mdnode =
        lldbg_emit_type(db, dtype, sptr, findex, is_reference, true, false);
  }
  if (unnamed) {
    symname = NULL;
#ifdef THISG
  } else if (ENCLFUNCG(sptr) && THISG(ENCLFUNCG(sptr)) == sptr) {
    symname = "this";
#endif
  } else {
    symname = (char *)lldbg_alloc(sizeof(char) * (strlen(SYMNAME(sptr)) + 1));
    /* In pass by value case flang creates a dummy variable with name
     * prefixed with "_V_". For debug info creation we are using the
     * absolute name. */
    if (PASSBYVALG(sptr))
      strcpy(symname, SYMNAME(MIDNUMG(sptr)));
    else
      strcpy(symname, SYMNAME(sptr));
  }
  flags = set_dilocalvariable_flags(sptr);
  var_mdnode = lldbg_create_local_variable_mdnode(
      db, DW_TAG_arg_variable, db->cur_subprogram_mdnode, symname, file_mdnode,
      db->cur_subprogram_lineno, parnum, type_mdnode, flags, ll_get_md_null());
  if (ll_feature_debug_info_ver90(&db->module->ir) &&
      ((STYPEG(sptr) == ST_ARRAY && (ALLOCATTRG(sptr) || POINTERG(sptr))) ||
       ASSUMRANKG(sptr) || ASSUMSHPG(sptr)) &&
      SDSCG(sptr)) {
    lldbg_register_param_mdnode(db, var_mdnode, SDSCG(sptr));
  } else
    lldbg_register_param_mdnode(db, var_mdnode, sptr);
  return var_mdnode;
}

LL_MDRef
lldbg_emit_ptr_param_variable(LL_DebugInfo *db, SPTR sptr, int findex,
                              int parnum)
{
  LL_MDRef file_mdnode, type_mdnode, var_mdnode, cu_mdnode;
  char *symname;
  ISZ_T sz;
  DBLINT64 align = {0};
  DBLINT64 offset = {0};
  int is_reference = 0;
  int flags;

  assert(db, "Debug info not enabled", 0, ERR_Fatal);
  if (ll_feature_debug_info_need_file_descriptions(&db->module->ir))
    file_mdnode = get_filedesc_mdnode(db, findex);
  else
    file_mdnode = lldbg_emit_file(db, findex);
  is_reference = ((SCG(sptr) == SC_DUMMY) && HOMEDG(sptr) && !PASSBYVALG(sptr));
  type_mdnode = lldbg_emit_type(db, DTYPEG(sptr), sptr, findex, is_reference,
                                false, false);
  cu_mdnode = lldbg_emit_compile_unit(db);
  sz = (ZSIZEOF(DT_CPTR) * 8);
  align[1] = ((alignment(DT_CPTR) + 1) * 8);
  cu_mdnode = ll_get_md_null();
  type_mdnode =
      lldbg_create_pointer_type_mdnode(db, cu_mdnode, "", ll_get_md_null(), 0,
                                       sz, align, offset, 0, type_mdnode);

#ifdef THISG
  if (ENCLFUNCG(sptr) && THISG(ENCLFUNCG(sptr)) == sptr) {
    symname = "this";
  } else
#endif
  {
    symname = (char *)lldbg_alloc(sizeof(char) * (strlen(SYMNAME(sptr)) + 1));
    strcpy(symname, SYMNAME(sptr));
  }
  flags = set_dilocalvariable_flags(sptr);
  var_mdnode = lldbg_create_local_variable_mdnode(
      db, DW_TAG_arg_variable, db->cur_subprogram_mdnode, symname, file_mdnode,
      db->cur_subprogram_lineno, parnum, type_mdnode, flags, ll_get_md_null());
  lldbg_register_param_mdnode(db, var_mdnode, sptr);
  return var_mdnode;
}

void
lldbg_function_end(LL_DebugInfo *db, int func)
{
  LL_Type *type;
  LL_Value *value;
  SPTR i;

  if (!(flg.debug && db))
    return;

  for (i = stb.firstusym; i != stb.stg_avail; ++i) {
    const int st = STYPEG(i);
    const int sc = SCG(i);

    if ((st == ST_ENTRY) || (st == ST_PROC) || (st == ST_CONST) ||
        (st == ST_PARAM) || (!DCLDG(i)) || CCSYMG(i) ||
        ((sc != SC_EXTERN) && (sc != SC_STATIC)))
      continue;

    if (!REFG(i)) {
      // generate unreferenced variables
      // add these to DWARF output as <optimized out> variables
      LL_Type *cache = LLTYPE(i);
      const DTYPE dtype = DTYPEG(i);
      process_dtype_struct(dtype); // make sure type is emitted
      type = make_lltype_from_dtype(dtype);
      value = ll_create_value_from_type(db->module, type, "undef");
      lldbg_emit_global_variable(db, i, 0, 1, value);
      LLTYPE(i) = cache;
    } else if ((!SNAME(i)) && REFG(i)) {
      // add referenced variables not discovered as yet
      const char *name;
      char *buff;
      LL_Type *cache = LLTYPE(i);
      const DTYPE dtype = DTYPEG(i);
      process_dtype_struct(dtype); // make sure type is emitted
      type = ll_get_pointer_type(make_lltype_from_dtype(dtype));
      name = get_llvm_name(i);
      // Hack: splice in the LLVM user-defined IR type name
      buff = (char *)getitem(LLVM_LONGTERM_AREA, strlen(name) + 6);
      sprintf(buff, "ptr @%s", name);
      value = ll_create_value_from_type(db->module, type, (const char *)buff);
      lldbg_emit_global_variable(db, i, 0, 1, value);
      LLTYPE(i) = cache;
    }
  }
}

static LL_MDRef
lldbg_create_imported_entity(LL_DebugInfo *db, SPTR entity_sptr, SPTR func_sptr,
                             IMPORT_TYPE entity_type, LL_MDRef elements_mdnode,
                             LL_MDRef imported_list)
{
  LLMD_Builder mdb;
  LL_MDRef entity_mdnode, scope_mdnode = 0, file_mdnode, cur_mdnode;
  unsigned tag;
  bool has_name = false;

  switch (entity_type) {
  case IMPORTED_DECLARATION: {
    const char *modvar_name;
    tag = DW_TAG_imported_declaration;
    if (CCSYMG(MIDNUMG(entity_sptr))) {
      modvar_name = new_debug_name(SYMNAME(ENCLFUNCG(entity_sptr)),
                                   SYMNAME(entity_sptr), NULL);
    } else {
      modvar_name = new_debug_name(SYMNAME(ENCLFUNCG(entity_sptr)),
                                   SYMNAME(MIDNUMG(entity_sptr)), SYMNAME(entity_sptr));
    }
    entity_mdnode = ll_get_module_debug(db->module->modvar_debug_map, modvar_name);
    break;
  }
  case IMPORTED_MODULE: {
    tag = DW_TAG_imported_module;
    entity_mdnode = ll_get_module_debug(db->module->module_debug_map, SYMNAME(entity_sptr));
    break;
  }
  case IMPORTED_UNIT: /* TODO: not implemented yet */
    return ll_get_md_null();
  }
  mdb = llmd_init(db->module);
  // FIXME: Initialize scope_mdnode properly.
  scope_mdnode = (func_sptr == gbl.currsub) ? db->cur_subprogram_mdnode : scope_mdnode;
  if (!entity_mdnode || !scope_mdnode)
    return ll_get_md_null();

  file_mdnode = ll_feature_debug_info_need_file_descriptions(&db->module->ir)
                    ? get_filedesc_mdnode(db, 1)
                    : lldbg_emit_file(db, 1);

  llmd_set_class(mdb, LL_DIImportedEntity);
  llmd_add_i32(mdb, make_dwtag(db, tag));  // tag
  llmd_add_md(mdb, entity_mdnode);         // entity
  llmd_add_md(mdb, scope_mdnode);          // scope
  llmd_add_md(mdb, file_mdnode);           // file
  llmd_add_i32(mdb, FUNCLINEG(func_sptr)); // line? no accurate line number yet
  if (entity_type == IMPORTED_DECLARATION) {
    const char *alias_name = lookup_modvar_alias(entity_sptr);
    if (alias_name && strcmp(alias_name, SYMNAME(entity_sptr))) {
      llmd_add_string(mdb, alias_name);    // variable renamed
      has_name = true;
    }
  }
  if (!has_name)
    llmd_add_string(mdb, NULL);
  llmd_add_md(mdb, elements_mdnode); // elements

  cur_mdnode = llmd_finish(mdb);
  if (imported_list)
    ll_extend_md_node(db->module, imported_list, cur_mdnode);
  return cur_mdnode;
}

static void
lldbg_emit_imported_entity(LL_DebugInfo *db, SPTR entity_sptr,
                           SPTR func_sptr, IMPORT_TYPE entity_type,
                           LL_MDRef elements_mdnode,
                           LL_MDRef imported_list)
{
  static hashset_t entity_func_added;
  const char *entity_func;

  if (INMODULEG(func_sptr) == entity_sptr)
    return;
  if (!entity_func_added)
    entity_func_added = hashset_alloc(hash_functions_strings);
  entity_func = new_debug_name(SYMNAME(entity_sptr), SYMNAME(func_sptr), NULL);
  if (hashset_lookup(entity_func_added, entity_func))
    return;
  hashset_insert(entity_func_added, entity_func);
  lldbg_create_imported_entity(db, entity_sptr, func_sptr, entity_type,
                               elements_mdnode, imported_list);
}

void
lldbg_create_cmblk_mem_mdnode_list(SPTR sptr, SPTR gblsym)
{
  SPTR var;
  LL_MDRef mdref;
  LL_ObjToDbgList **listp = &AG_OBJTODBGLIST(gblsym);
  if (!*listp)
    *listp = llObjtodbgCreate();
  for (var = CMEMFG(sptr); var > NOSYM; var = SYMLKG(var)) {
      mdref = ll_get_global_debug(cpu_llvm_module, var);
      if (!LL_MDREF_IS_NULL(mdref))
        llObjtodbgAddUnique(*listp, mdref);
  }
  /* add processing for COMMON */
  mdref = ll_get_global_debug(cpu_llvm_module, sptr);
  if (!LL_MDREF_IS_NULL(mdref))
    llObjtodbgAddUnique(*listp, mdref);
}

static LL_MDRef
lldbg_create_common_block_mdnode(LL_DebugInfo *db, LL_MDRef scope,
                                 char *name)
{
  LLMD_Builder mdb;
  char *common_block_name, *pname, *pmname;

  mdb = llmd_init(db->module);
  llmd_set_distinct(mdb);
  common_block_name = (char *)lldbg_alloc(strlen(name) + 1);
  pname = name;
  pmname = common_block_name;
  while (*pname != '\0') {
    *pmname = tolower(*pname);
    pname++;
    pmname++;
  }
  *pmname = '\0'; /* append null char to end of string */

  // Use the DICommonBlock template
  llmd_set_class(mdb, LL_DICommonBlock);
  llmd_add_md(mdb, scope);                 // scope
  llmd_add_md(mdb, ll_get_md_null());      // declaration
  llmd_add_string(mdb, common_block_name); // name
  return llmd_finish(mdb);
}

LL_MDRef
lldbg_emit_common_block_mdnode(LL_DebugInfo *db, SPTR sptr)
{
  LL_MDRef scope_modnode, cmnblk_mdnode;
  SPTR scope = SCOPEG(sptr);
  const char *cmnblk_name = new_debug_name(SYMNAME(scope), SYMNAME(sptr), NULL);

  cmnblk_mdnode = ll_get_module_debug(db->module->common_debug_map, cmnblk_name);
  if (!LL_MDREF_IS_NULL(cmnblk_mdnode))
    return cmnblk_mdnode;
  scope_modnode = db->cur_subprogram_mdnode
                      ? db->cur_subprogram_mdnode
                      : lldbg_emit_module_mdnode(db, scope);
  cmnblk_mdnode = lldbg_create_common_block_mdnode(
      db, scope_modnode, SYMNAME(sptr));
  db->cur_cmnblk_mdnode = cmnblk_mdnode;
  ll_add_module_debug(db->module->common_debug_map, cmnblk_name, cmnblk_mdnode);  
  if (db->cur_subprogram_mdnode)
    add_debug_cmnblk_variables(db, sptr);
  db->cur_cmnblk_mdnode = (LL_MDRef)0;
  return cmnblk_mdnode;
}

void
lldbg_add_pending_import_entity_to_child(LL_DebugInfo *db, SPTR entity,
                                         IMPORT_TYPE entity_type)
{
  import_entity *new_node;

  new_node = (import_entity *)lldbg_alloc(sizeof *new_node);
  new_node->entity = entity;
  new_node->entity_type = entity_type;
  new_node->func = gbl.currsub;
  new_node->next = db->import_entity_list->child;
  new_node->child = NULL;
  db->import_entity_list->child = new_node;
}

void
lldbg_add_pending_import_entity(LL_DebugInfo *db, SPTR entity,
                                IMPORT_TYPE entity_type)
{
  import_entity *node_ptr, *new_node;

  if (db->import_entity_list) {
    if (db->import_entity_list->func != gbl.currsub) {
      db->import_entity_list = NULL;
    } else {
      node_ptr = db->import_entity_list;
      while (node_ptr) {
        if (node_ptr->entity == entity)
          return;
        node_ptr = node_ptr->next;
      }
    }
  }
  new_node = (import_entity *)lldbg_alloc(sizeof *new_node);
  new_node->entity = entity;
  new_node->entity_type = entity_type;
  new_node->func = gbl.currsub;
  new_node->next = db->import_entity_list;
  new_node->child = NULL;
  db->import_entity_list = new_node;
}

const char*
new_debug_name(const char *str1, const char *str2, const char *str3)
{
  int size;
  char *new_name;
  const char *sep = "/";

  if (!str1 || !str2)
    return NULL;
  if (str3) {
    size = strlen(str1) + strlen(str2) + strlen(str3) + 3;
  } else {
    size = strlen(str1) + strlen(str2) + 2;
  }
  new_name = (char *)lldbg_alloc(size);
  strcpy(new_name, str1);
  strcat(new_name, sep);
  strcat(new_name, str2);
  if (str3) {
    strcat(new_name, sep);
    strcat(new_name, str3);
  }
  return (const char *)new_name;
}

LL_MDRef
lldbg_get_subprogram_line(LL_DebugInfo *db)
{
  return db->cur_subprogram_line_mdnode;
}
