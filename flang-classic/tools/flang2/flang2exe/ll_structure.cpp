/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief LLVM bridge representation
 */

#include "gbldefs.h"
#include "error.h"
#include "ll_builder.h"
#include "ll_structure.h"
#include "lldebug.h"
#include "global.h"
#include "go.h"
#include "llassem.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/**
   \brief Get the LLVM version being used
   The -x 249 flag is set by the -Mllvm= driver option.
 */
LL_IRVersion
get_llvm_version(void)
{
  return flg.x[249] ? ((LL_IRVersion)flg.x[249]) : LL_Version_3_2;
}

static void *
ll_manage_mem(LLVMModuleRef module, void *space)
{
  struct LL_ManagedMallocs_ *mem =
      (struct LL_ManagedMallocs_ *)malloc(sizeof(struct LL_ManagedMallocs_));
  mem->storage = space;
  mem->next = module->first_malloc;
  module->first_malloc = mem;
  return space;
}

static char *
ll_manage_strdup(LLVMModuleRef module, const char *str)
{
  return (char *)ll_manage_mem(module, strdup(str));
}

static void *
ll_manage_calloc(LLVMModuleRef module, size_t members, size_t member_size)
{
  void *space = calloc(members, member_size);
  return ll_manage_mem(module, space);
}

static void *
ll_manage_malloc(LLVMModuleRef module, size_t malloc_size)
{
  void *space = malloc(malloc_size);
  return ll_manage_mem(module, space);
}

static void
ll_destroy_instruction(struct LL_Instruction_ *instruction)
{
  if (instruction->comment) {
    free(instruction->comment);
  }

  if (instruction->operands) {
    free(instruction->operands);
  }

  free(instruction);
}

static void
ll_destroy_basic_block(struct LL_BasicBlock_ *basic_block)
{
  struct LL_Instruction_ *current;
  struct LL_Instruction_ *next;

  current = basic_block->first;
  while (current != NULL) {
    next = current->next;
    ll_destroy_instruction(current);
    current = next;
  }

  free(basic_block->name);
  free(basic_block);
}

static void
free_iterator(hash_key_t key, void *context)
{
  (void)context;
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  free((void *)key);
#pragma GCC diagnostic pop
}

void
ll_destroy_function(struct LL_Function_ *function)
{
  struct LL_BasicBlock_ *current;
  struct LL_BasicBlock_ *next;

  current = function->first;
  while (current != NULL) {
    next = current->next;
    ll_destroy_basic_block(current);
    current = next;
  }

  /* function->name is allocated by ll_manage_strdup(). */
  if (function->arguments)
    free(function->arguments);

  if (function->used_local_names) {
    /* Local names were allocated by strdupo in unique_name(). */
    hashset_iterate(function->used_local_names, free_iterator, NULL);
    hashset_free(function->used_local_names);
  }

  free(function);
}

void
ll_destroy_mem(struct LL_ManagedMallocs_ *current)
{
  free(current->storage);
  free(current);
}

/*
 * Types are uniqued within a module to save memory and to allow type
 * equivalence to be tested by pointer comparison.
 *
 * Named structs are identified by name. All other types are uniqued by their
 * structure which is simple to do since cyclic types can only be created by
 * using named structs.
 *
 * The functions types_equal and types_hash below don't expect to be called
 * with a named struct, they will treat it like an anonynous struct.
 */

/* How many elements in the sub_types array? */
static BIGUINT64
get_num_subtypes(const struct LL_Type_ *type)
{
  switch (type->data_type) {
  case LL_PTR:
  case LL_ARRAY:
  case LL_VECTOR:
    return 1;
  default:
    return type->sub_elements;
  }
}

static hash_value_t
types_hash(hash_key_t key)
{
  const struct LL_Type_ *t = (const struct LL_Type_ *)key;
  hash_accu_t hacc = HASH_ACCU_INIT;
  unsigned i, nsubtypes;

  HASH_ACCU_ADD(hacc, t->data_type);
  HASH_ACCU_ADD(hacc, t->flags);
  HASH_ACCU_ADD(hacc, t->addrspace);
  HASH_ACCU_ADD(hacc, (unsigned long)t->sub_elements);

  /* Subtypes have already been uniqued, so just hash pointer values. */
  nsubtypes = get_num_subtypes(t);
  for (i = 0; i != nsubtypes; i++)
    HASH_ACCU_ADD(hacc, (unsigned long)t->sub_types[i]);

  HASH_ACCU_FINISH(hacc);
  return HASH_ACCU_VALUE(hacc);
}

static int
types_equal(hash_key_t key_a, hash_key_t key_b)
{
  const struct LL_Type_ *a = (const struct LL_Type_ *)key_a;
  const struct LL_Type_ *b = (const struct LL_Type_ *)key_b;
  unsigned i, n;

  if (a->data_type != b->data_type)
    return false;
  if (a->flags != b->flags)
    return false;
  if (a->addrspace != b->addrspace)
    return false;
  if (a->sub_elements != b->sub_elements)
    return false;

  n = get_num_subtypes(a);
  for (i = 0; i != n; i++)
    if (a->sub_types[i] != b->sub_types[i])
      return false;

  /* FIXME: We really shouldn't be adding named structs to the uniquing
   * tables, but at the moment we may be creating multiple versions of a
   * named struct for different address spaces.
   *
   * The correct way of handling this is to store the address space in
   * pointer types only, just like LLVM does it.
   */
  if (a->data_type == LL_STRUCT && (a->flags & LL_TYPE_IS_NAMED_STRUCT))
    return a->str == b->str;

  return true;
}

static const hash_functions_t types_hash_functions = {types_hash, types_equal};

/*
 * Create a unique name based on a printf-like template.
 *
 * 1. Pick a name based on format + ap that isn't already in 'names'.
 * 2. Copy the name into malloc'ed memory.
 * 3. Insert new pointer into names.
 * 4. Return malloced pointer.
 */
static char *
unique_name(hashset_t names, char prefix, const char *format, va_list ap)
{
  char buffer[256] = {prefix, 0};
  size_t prefix_length;
  unsigned count = 0;
  int reseeded = 0;
  char *unique_name;

  /* The return value from vsnprintf() is useless because Microsoft Visual
   * Studio doesn't follow the standard. */
  vsnprintf(buffer + 1, sizeof(buffer) - 1, format, ap);
  buffer[sizeof(buffer) - 1] = '\0';
  prefix_length = strlen(buffer);

  /* Make room for a ".%u" suffix. */
  if (prefix_length > sizeof(buffer) - 12)
    prefix_length = sizeof(buffer) - 12;

  /* Search for a not previously used name. */
  while (hashset_lookup(names, buffer)) {
    /* Try a pretty .1, .2, ... .9 suffix sequence at first, but then
     * switch to a scheme that isn't quadratic time. */
    if (++count == 10 && !reseeded) {
      count = 10 * hashset_size(names);
      reseeded = 1;
    }
    sprintf(buffer + prefix_length, ".%u", count);
  }

  unique_name = strdup(buffer);
  hashset_insert(names, unique_name);

  return unique_name;
}

/*
 * Hash functions for interned constants.
 *
 * LLVM constants are identified by their uniqued type pointer and textual
 * representation.
 */

static hash_value_t
constants_hash(hash_key_t key)
{
  const LL_Value *t = (const LL_Value *)key;
  const unsigned char *p = (const unsigned char *)t->data;
  hash_accu_t hacc = HASH_ACCU_INIT;

  HASH_ACCU_ADD(hacc, (unsigned long)t->type_struct);
  for (; *p; p++)
    HASH_ACCU_ADD(hacc, *p);

  HASH_ACCU_FINISH(hacc);
  return HASH_ACCU_VALUE(hacc);
}

static int
constants_equal(hash_key_t key_a, hash_key_t key_b)
{
  const LL_Value *a = (const LL_Value *)key_a;
  const LL_Value *b = (const LL_Value *)key_b;

  return a->type_struct == b->type_struct && strcmp(a->data, b->data) == 0;
}

static const hash_functions_t constants_hash_functions = {constants_hash,
                                                          constants_equal};

/*
 * Metadata nodes.
 *
 * A metadata node is a tuple of MDRefs, structurally uniqued by a hashmap. The
 * node header and element array is allocated in contiguous memory.
 */

static hash_value_t
mdnode_hash(hash_key_t key)
{
  const LL_MDNode *t = (const LL_MDNode *)key;
  unsigned i;
  hash_accu_t hacc = HASH_ACCU_INIT;

  HASH_ACCU_ADD(hacc, t->num_elems);
  for (i = 0; i < t->num_elems; i++) {
    HASH_ACCU_ADD(hacc, LL_MDREF_kind(t->elem[i]));
    HASH_ACCU_ADD(hacc, LL_MDREF_value(t->elem[i]));
  }

  HASH_ACCU_FINISH(hacc);
  return HASH_ACCU_VALUE(hacc);
}

static int
mdnode_equal(hash_key_t key_a, hash_key_t key_b)
{
  const LL_MDNode *a = (const LL_MDNode *)key_a;
  const LL_MDNode *b = (const LL_MDNode *)key_b;
  unsigned i;

  if (a->num_elems != b->num_elems)
    return false;

  for (i = 0; i < a->num_elems; i++)
    if (a->elem[i] != b->elem[i])
      return false;

  return true;
}

static const hash_functions_t mdnode_hash_functions = {mdnode_hash,
                                                       mdnode_equal};

void
llObjtodbgPush(LL_ObjToDbgList *odl, LL_MDRef md)
{
  LL_ObjToDbgList *p = odl;
  while (p->next)
    p = p->next;
  if (p->used == LL_ObjToDbgBucketSize) {
    p->next = llObjtodbgCreate();
    p = p->next;
  }
  p->refs[p->used++] = md;
}

void
llObjtodbgFree(LL_ObjToDbgList *ods)
{
  LL_ObjToDbgList *n;
  for (; ods; ods = n) {
    n = ods->next;
    free(ods);
  }
}

/**
   \brief Reset the id map for named struct types.

   This means that ll_get_struct_type() will not return any of the struct types
   created so far, and previously used type ids can be reused.

   Reset the `id -> type` mapping, but deliberately keep the \c used_type_names
   set. We still want to guarantee globally unique type names in the module.
 */
void
ll_reset_module_types(LLVMModuleRef module)
{
  hashmap_clear(module->user_structs_byid);
}

/**
   \brief Compute the set of IR features in module when generating IR for the
   specified LLVM version.
 */
static void
compute_ir_feature_vector(LLVMModuleRef module, enum LL_IRVersion vers)
{
  if (strncmp(module->target_triple, "nvptx", 5) == 0)
    module->ir.is_nvvm = 1;
  if (strncmp(module->target_triple, "spir", 4) == 0)
    module->ir.is_spir = 1;

  module->ir.version = vers;
  InitializeDIFlags(&module->ir);
  if (XBIT(120, 0x200)) {
    module->ir.dwarf_version = LL_DWARF_Version_2;
  } else if (XBIT(120, 0x4000)) {
    module->ir.dwarf_version = LL_DWARF_Version_3;
  } else if (XBIT(120, 0x1000000)) {
    module->ir.dwarf_version = LL_DWARF_Version_4;
  } else if (XBIT(120, 0x2000000)) {
    module->ir.dwarf_version = LL_DWARF_Version_5;
  } else { // DWARF 4 is the default
    module->ir.dwarf_version = LL_DWARF_Version_4;
  }

  if (ll_feature_versioned_dw_tag(&module->ir)) {
    /* LLVMDebugVersion 12 was used by LLVM versions 3.1 through 3.5, and we
     * don't support LLVM versions older than 3.1, so the version as always 12.
     */
    module->ir.debug_info_version = 12;
  } else {
    /* LLVM 3.6 onwards encodes the debug info version in a module flag
     * metadata node, and the numbering sequence started over.
     *
     * LLVM 3.6 used v2 which had the stringified header fields in metadata
     * nodes. This format was abandoned in 3.7, and we don't support it.
     */
    if (vers >= LL_Version_3_7)
      module->ir.debug_info_version = 3;
    else
      module->ir.debug_info_version = 1;
  }
}

/**
   \brief Convert the cached DWARF version to an unsigned

   The DWARF version is either the default or specified from the command-line.
 */
unsigned
ll_feature_dwarf_version(const LL_IRFeatures *feature)
{
  switch (feature->dwarf_version) {
  case LL_DWARF_Version_2:
    return 2;
  case LL_DWARF_Version_3:
    return 3;
  case LL_DWARF_Version_4:
    return 4;
  case LL_DWARF_Version_5:
    return 5;
  }
}

struct triple_info {
  const char *prefix;
  const char *datalayout;
};

static const struct triple_info known_triples[] = {
    /* These prefixes are tried in order against target_triple. */
    {"nvptx64-", "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                 "-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:"
                 "128-n16:32:64"},
    {"spir64-", "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64"
                "-v64:64:64-v96:128:128-v128:128:128-v192:256:256"
                "-v256:256:256-v512:512:512-v1024:1024:1024"},
    {"i386", "e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"},
    {"x86_64-apple", "e-m:o-i64:64-f80:128-n8:16:32:64-S128"},
    {"x86_64-", "e-p:64:64-i64:64-f80:128-n8:16:32:64-S128"},
    {"armv7-", "e-p:32:32-i64:64-v128:64:128-n32-S64"},
    {"aarch64-", "e-m:e-i64:64-i128:128-n32:64-S128"},
    {"powerpc64le", "e-p:64:64-i64:64-n32:64"},
    {"", ""}};

/* Compute the data layout for the requested target triple. */
static void
compute_datalayout(LLVMModuleRef module)
{
  const struct triple_info *triple = known_triples;

  while (strncmp(module->target_triple, triple->prefix, strlen(triple->prefix)))
    triple++;
  module->datalayout_string = triple->datalayout;
}

void
ll_destroy_module(LLVMModuleRef module)
{
  struct LL_Function_ *current;
  struct LL_Function_ *next;
  struct LL_ManagedMallocs_ *cur_m;
  struct LL_ManagedMallocs_ *next_m;
  unsigned i;

  current = module->first;
  while (current != NULL) {
    next = current->next;
    ll_destroy_function(current);
    current = next;
  }

  cur_m = module->first_malloc;
  while (cur_m != NULL) {
    next_m = cur_m->next;
    ll_destroy_mem(cur_m);
    cur_m = next_m;
  }
  free(module->module_vars.values);
  free(module->user_structs.values);
  hashmap_free(module->user_structs_byid);
  hashset_free(module->used_type_names);
  hashset_free(module->used_global_names);
  hashset_free(module->anon_types);

  free(module->constants);
  hashmap_free(module->constants_map);

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  for (i = 0; i < module->mdstrings_count; i++)
    free((char *)module->mdstrings[i]);
#pragma GCC diagnostic pop
  free(module->mdstrings);
  hashmap_free(module->mdstrings_map);

  for (i = 0; i < module->mdnodes_count; i++)
    free(module->mdnodes[i]);
  free(module->mdnodes);
  hashmap_free(module->mdnodes_map);

  hashmap_free(module->global_debug_map);
  hashmap_free(module->module_debug_map);
  hashmap_free(module->common_debug_map);
  hashmap_free(module->modvar_debug_map);

  for (i = 0; i < MD_NUM_NAMES; i++)
    free(module->named_mdnodes[i]);

  lldbg_free(module->debug_info);
  free(module);
}

LLVMModuleRef
ll_create_module(const char *module_name, const char *target_triple,
                 enum LL_IRVersion llvm_ir_version)
{
  LLVMModuleRef new_module = (LLVMModuleRef)calloc(1, sizeof(LL_Module));
  new_module->first_malloc = NULL;
  new_module->module_name = ll_manage_strdup(new_module, module_name);
  new_module->target_triple = ll_manage_strdup(new_module, target_triple);
  new_module->first = new_module->last = NULL;
  new_module->module_vars.values = (LL_Value **)calloc(16, sizeof(LL_Value *));
  new_module->module_vars.num_values = 16;
  new_module->num_module_vars = 0;
  new_module->user_structs.values = (LL_Value **)calloc(16, sizeof(LL_Value *));
  new_module->user_structs.num_values = 16;
  new_module->num_user_structs = 0;
  new_module->written_user_structs = 0;
  new_module->user_structs_byid = hashmap_alloc(hash_functions_direct);
  new_module->used_type_names = hashset_alloc(hash_functions_strings);
  new_module->used_global_names = hashset_alloc(hash_functions_strings);
  new_module->anon_types = hashset_alloc(types_hash_functions);
  new_module->num_refs = 0;
  new_module->extern_func_refs = NULL;

  new_module->constants = (LL_Value **)calloc(16, sizeof(LL_Value *));
  new_module->constants_alloc = 16;
  new_module->constants_map = hashmap_alloc(constants_hash_functions);

  new_module->mdstrings = (const char **)calloc(16, sizeof(char *));
  new_module->mdstrings_alloc = 16;
  new_module->mdstrings_map = hashmap_alloc(hash_functions_strings);

  new_module->mdnodes = (LL_MDNode **)calloc(16, sizeof(LL_MDNode *));
  new_module->mdnodes_alloc = 16;
  new_module->mdnodes_map = hashmap_alloc(mdnode_hash_functions);
  new_module->mdnodes_fwdvars = hashmap_alloc(hash_functions_direct);

  new_module->global_debug_map = hashmap_alloc(hash_functions_direct);

  new_module->module_debug_map = hashmap_alloc(hash_functions_strings);
  new_module->common_debug_map = hashmap_alloc(hash_functions_strings);
  new_module->modvar_debug_map = hashmap_alloc(hash_functions_strings);

  compute_ir_feature_vector(new_module, llvm_ir_version);
  compute_datalayout(new_module);
   
#ifdef _WIN64
  if (flg.linker_directives) {
    add_linker_directives(new_module);
  }
#endif

  return new_module;
}

void
add_linker_directives(LLVMModuleRef module) {
  if (get_llvm_version() < LL_Version_5_0) {
    LLMD_Builder mdb = llmd_init(module);
    char* linker_directive;
    for (int i = 0; (linker_directive = flg.linker_directives[i]); ++i) {
      LLMD_Builder submdb = llmd_init(module);

      llmd_add_string(submdb, linker_directive);
      LL_MDRef submd = llmd_finish(submdb);

      llmd_add_md(mdb, submd);
    }
    LL_MDRef md = llmd_finish(mdb);

    LLMD_Builder boilerplate_mdb = llmd_init(module);

    llmd_add_i32(boilerplate_mdb, 6);
    llmd_add_string(boilerplate_mdb, "Linker Options");
    llmd_add_md(boilerplate_mdb, md);

    LL_MDRef boilerplate_md = llmd_finish(boilerplate_mdb);
    ll_extend_named_md_node(module, MD_llvm_module_flags, boilerplate_md);

    LLMD_Builder debug_mdb = llmd_init(module);

    const int mdVers = ll_feature_versioned_dw_tag(&module->ir) ? 1 :
      module->ir.debug_info_version;

    llmd_add_i32(debug_mdb, 1);
    llmd_add_string(debug_mdb, "Debug Info Version");
    llmd_add_i32(debug_mdb, mdVers);

    LL_MDRef debug_md = llmd_finish(debug_mdb);

    ll_extend_named_md_node(module, MD_llvm_module_flags, debug_md);
  } else {
    int i;
    char *linker_directive;
    LLMD_Builder mdb = llmd_init(module);
    for (i = 0; (linker_directive = flg.linker_directives[i]); ++i) {
      llmd_add_string(mdb, linker_directive);
    }
    LL_MDRef linker_md = llmd_finish(mdb);
    ll_extend_named_md_node(module, MD_llvm_linker_options, linker_md);
  }
}

struct LL_Function_ *
ll_create_function(LLVMModuleRef module, const char *name, LL_Type *return_type,
                   int is_kernel, int launch_bounds, int launch_bounds_minctasm,
                   const char *calling_convention, enum LL_LinkageType linkage)
{
  LL_Function *new_function = (LL_Function *)calloc(1, sizeof(LL_Function));
  new_function->name = ll_manage_strdup(module, name);
  new_function->return_type = return_type;
  new_function->first = NULL;
  new_function->last = NULL;
  new_function->next = NULL;
  new_function->arguments = NULL;
  new_function->num_args = 0;
  new_function->num_locals = 0;
  new_function->is_kernel = is_kernel;
  new_function->launch_bounds = launch_bounds;
  new_function->launch_bounds_minctasm = launch_bounds_minctasm;
  new_function->calling_convention =
      ll_manage_strdup(module, calling_convention);
  new_function->linkage = linkage;

  if (module->last == NULL) {
    module->first = new_function;
  } else {
    module->last->next = new_function;
  }

  module->last = new_function;

  new_function->local_vars.values =
      (LL_Value **)ll_manage_calloc(module, 16, sizeof(LL_Value));
  new_function->local_vars.num_values = 16;

  return new_function;
}

/**
   \brief Create an LL_Function

   Must be given its name and full \c LL_FUNCTION type.  Note: This does not add
   the new function to the module's list of functions.
 */
LL_Function *
ll_create_function_from_type(LL_Type *func_type, const char *name)
{
  LLVMModuleRef module = func_type->module;
  LL_Function *new_function = (LL_Function *)calloc(1, sizeof(LL_Function));

  CHECK(func_type->data_type == LL_FUNCTION);
  CHECK(func_type->sub_elements > 0);

  new_function->name = ll_manage_strdup(module, name);
  new_function->return_type = func_type->sub_types[0];
  ll_set_function_num_arguments(new_function, func_type->sub_elements - 1);

  return new_function;
}


LL_Function *
ll_create_device_function_from_type(LLVMModuleRef module, LL_Type *func_type,
                                    const char *name, int is_kernel,
                                    int launch_bounds,
                                    const char *calling_convention,
                                    enum LL_LinkageType linkage)
{
  LL_Function *new_function = (LL_Function *)calloc(1, sizeof(LL_Function));

  CHECK(func_type->data_type == LL_FUNCTION);
  CHECK(func_type->sub_elements > 0);

  new_function->name = ll_manage_strdup(module, name);
  new_function->return_type = func_type->sub_types[0];
  ll_set_function_num_arguments(new_function, func_type->sub_elements - 1);

  new_function->is_kernel = is_kernel;
  new_function->launch_bounds = launch_bounds;
  new_function->calling_convention =
      ll_manage_strdup(module, calling_convention);
  new_function->linkage = linkage;

  if (module->last == NULL) {
    module->first = new_function;
  } else {
    module->last->next = new_function;
  }

  module->last = new_function;

  new_function->local_vars.values =
      (LL_Value **)ll_manage_calloc(module, 16, sizeof(LL_Value));
  new_function->local_vars.num_values = 16;

  return new_function;
}

void
ll_create_sym(struct LL_Symbols_ *symbol_table, unsigned index, LL_Value *new_value)
{
  int new_size;

  if (index >= symbol_table->num_values) {
    new_size = (3 * (index + 1)) / 2;
    symbol_table->values = (LL_Value **)realloc(symbol_table->values,
                                                new_size * sizeof(LL_Value *));
    memset(&(symbol_table->values[symbol_table->num_values]), 0,
           (new_size - symbol_table->num_values) * sizeof(LL_Value *));
    symbol_table->num_values = new_size;
  }
  symbol_table->values[index] = new_value;
}

void
ll_set_function_num_arguments(struct LL_Function_ *function, int num_args)
{
  function->arguments = (LL_Value **)calloc(num_args, sizeof(LL_Value *));
  function->num_args = num_args;
}

void
ll_set_function_argument(struct LL_Function_ *function, int index,
                         LL_Value *argument)
{
  function->arguments[index] = argument;
}

const char *
ll_get_str_type_for_basic_type(enum LL_BaseDataType type)
{
  switch (type) {
  case LL_NOTYPE:
    return "";
  case LL_LABEL:
    return "label";
  case LL_METADATA:
    return "metadata";
  case LL_VOID:
    return "void";
  case LL_I1:
    return "i1";
  case LL_I8:
    return "i8";
  case LL_I16:
    return "i16";
  case LL_I24:
    return "i24";
  case LL_I32:
    return "i32";
  case LL_I40:
    return "i40";
  case LL_I48:
    return "i48";
  case LL_I56:
    return "i56";
  case LL_I64:
    return "i64";
  case LL_I128:
    return "i128";
  case LL_I256:
    return "i256";
  case LL_HALF:
    return "half";
  case LL_FLOAT:
    return "float";
  case LL_DOUBLE:
    return "double";
  case LL_FP128:
    return "fp128";
  case LL_X86_FP80:
    return "x86_fp80";
  case LL_PPC_FP128:
    return "ppc_fp128";
  case LL_X86_MMX:
    return "x86_mmx";
  default:
    return "ERR";
  }
  return "ERR";
}

/**
   \brief Get the size of a type in bytes without checks
   \param type   The type to be examined
   \param noSize Set if not null and \p type isn't concrete [output]
   \return size of an object of type \p type or 0
 */
static ISZ_T
LLTypeGetBytesUnchecked(LL_Type *type, int *noSize)
{
  if (noSize)
    *noSize = 0;
  switch (type->data_type) {
  case LL_I1:
  case LL_I8:
    return 1;
  case LL_I16:
    return 2;
  case LL_I24:
    return 3;
  case LL_I32:
    return 4;
  case LL_I40:
    return 5;
  case LL_I48:
    return 6;
  case LL_I56:
    return 7;
  case LL_I64:
    return 8;
  case LL_I128:
    return 16;
  case LL_I256:
    return 32;
  case LL_HALF:
    return 2;
  case LL_FLOAT:
    return 4;
  case LL_DOUBLE:
    return 8;
  case LL_X86_FP80:
  case LL_FP128:
  case LL_PPC_FP128:
    return 16;
  case LL_PTR:
    /* FIXME: Use the data layout so we can cross compile. */
    return sizeof(void *); // 8 * TARGET_PTRSIZE
  case LL_ARRAY:
  case LL_VECTOR:
    return type->sub_elements * ll_type_bytes(type->sub_types[0]);
  case LL_STRUCT: {
    unsigned i;
    ISZ_T sum;
    if (type->sub_offsets) {
      /* This should be the size in memory sans assumptions */
      return type->sub_offsets[type->sub_elements];
    }
    sum = 0;
    /* Note: We're assuming the struct has no implicit padding. */
    for (i = 0; i < type->sub_elements; i++)
      sum += ll_type_bytes(type->sub_types[i]);
    return sum;
  }
  default:
    if (noSize)
      *noSize = 1;
    break;
  }
  return 0;
}

/**
   \brief Get the size of a type in bytes
   \param type   The type to be examined
   \return size of an object of type \p type or 0
 */
ISZ_T
ll_type_bytes(LL_Type *type)
{
  int notConcrete;
  ISZ_T size = LLTypeGetBytesUnchecked(type, &notConcrete);
  if (notConcrete) {
    interr("ll_type_bytes: Not a concrete type", type->data_type, ERR_Fatal);
  }
  return size;
}

/**
   \brief Get the size of a type in bytes without checks
   \param type   The type to be examined
   \return size of object of type \p type or 0
 */
ISZ_T
ll_type_bytes_unchecked(LL_Type *type)
{
  return LLTypeGetBytesUnchecked(type, NULL);
}

/**
   \brief Get the number of bits in the given integer type
   \param type  The \ref LL_Type to inspect
   \return 0 iff \p type is not an integer type
 */
unsigned
ll_type_int_bits(LL_Type *type)
{
  switch (type->data_type) {
  case LL_I1:
    return 1;
  case LL_I8:
    return 8;
  case LL_I16:
    return 16;
  case LL_I24:
    return 24;
  case LL_I32:
    return 32;
  case LL_I40:
    return 40;
  case LL_I48:
    return 48;
  case LL_I56:
    return 56;
  case LL_I64:
    return 64;
  case LL_I128:
    return 128;
  case LL_I256:
    return 256;
  default:
    break;
  }
  return 0;
}

int
ll_type_is_pointer_to_function(LL_Type *ty)
{
  return (ty->data_type == LL_PTR) &&
         (ty->sub_types[0]->data_type == LL_FUNCTION);
}

LL_Type *
ll_type_array_elety(LL_Type *ty)
{
  return (ty->data_type == LL_ARRAY) ? ty->sub_types[0] : (LL_Type *)0;
}

int
ll_type_is_fp(LL_Type *ty)
{
  switch (ty->data_type) {
  case LL_HALF:
  case LL_FLOAT:
  case LL_DOUBLE:
  case LL_FP128:
  case LL_X86_FP80:
  case LL_PPC_FP128:
    return 1;
  default:
    break;
  }
  return 0;
}

int
ll_type_is_mem_seq(LL_Type *ty)
{
  switch (ty->data_type) {
  case LL_PTR:
  case LL_ARRAY:
    return 1;
  default:
    break;
  }
  return 0;
}

/**
   \brief Create a "blank" value
 */
static LL_Value *
ll_create_blank_value(LLVMModuleRef module, const char *data)
{
  LL_Value *ret_value = (LL_Value *)ll_manage_malloc(module, sizeof(LL_Value));
  ret_value->data = (data ? ll_manage_strdup(module, data) : NULL);
  ret_value->linkage = LL_NO_LINKAGE;
  ret_value->mvtype = LL_DEFAULT;
  ret_value->type_struct = NULL;
  ret_value->align_bytes = 0;
  ret_value->flags = 0;
  ret_value->storage = NULL;
  ret_value->dbg_mdnode = ll_get_md_null();
  ret_value->dbg_sptr = 0;
  return ret_value;
}

LL_Value *
ll_create_pointer_value(LLVMModuleRef module, enum LL_BaseDataType type,
                        const char *data, int addrspace)
{
  LL_Value *ret_value = ll_create_blank_value(module, data);
  LL_Type *base_type;

  base_type = ll_create_basic_type(module, type, addrspace);

  ret_value->type_struct = ll_get_pointer_type(base_type);

  return ret_value;
}

LL_Value *
ll_create_value_from_type(LLVMModuleRef module, LL_Type *type, const char *data)
{
  LL_Value *ret_value = ll_create_blank_value(module, data);
  ret_value->type_struct = type;
  return ret_value;
}

LL_Value *
ll_create_array_value_from_type(LLVMModuleRef module, LL_Type *type,
                                const char *data, int addrspace)
{
  LL_Value *ret_value = ll_create_blank_value(module, data);
  ret_value->type_struct = ll_get_array_type(type, 0, addrspace);
  return ret_value;
}

LL_Value *
ll_create_pointer_value_from_type(LLVMModuleRef module, LL_Type *type,
                                  const char *data, int addrspace)
{
  LL_Value *ret_value = ll_create_blank_value(module, data);
  ret_value->type_struct =
      ll_get_pointer_type(ll_get_addrspace_type(type, addrspace));
  return ret_value;
}

/**
   \brief get pointer's address space
   \param ptr  A pointer type

   Given a pointer type, return the address space that is being pointed into.
 */
int
ll_get_pointer_addrspace(LL_Type *ptr)
{
  assert(ptr->data_type == LL_PTR, "ll_get_pointer_addrspace: Pointer required",
         ptr->data_type, ERR_Fatal);
  return ptr->sub_types[0]->addrspace;
}

/*
 * Get a uniqued copy of type.
 *
 * If we already have a type identical to t, return that. Otherwise create a
 * copy of t and remember that.
 *
 * The copy will be shallow - the str and sub_types pointers will be copied
 * directly from type.
 */
static struct LL_Type_ *
unique_type(LLVMModuleRef module, const struct LL_Type_ *type)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  hash_key_t existing = hashset_lookup(module->anon_types, type);
  if (existing)
    return (struct LL_Type_ *)existing;
#pragma GCC diagnostic pop

  /* No such type exists. Save a copy. */
  struct LL_Type_ *copy =
      (struct LL_Type_ *)ll_manage_malloc(module, sizeof(struct LL_Type_));
  memcpy(copy, type, sizeof(*copy));
  copy->module = module;
  hashset_insert(module->anon_types, copy);
  return copy;
}

LL_Type *
ll_create_basic_type(LLVMModuleRef module, enum LL_BaseDataType type,
                     int addrspace)
{
  struct LL_Type_ new_type;
  struct LL_Type_ *ret_type;

  assert(type <= LL_X86_MMX, "Basic LLVM base data type required", type,
         ERR_Fatal);

  new_type.str = NULL;
  new_type.data_type = type;
  new_type.flags = 0;
  new_type.sub_types = NULL;
  new_type.sub_offsets = NULL;
  new_type.sub_elements = 0;
  new_type.sub_padding = 0;
  new_type.addrspace = addrspace;

  ret_type = unique_type(module, &new_type);

  /* LLVM basic types don't have address spaces (only pointer types do), so
   * don't include it in the representation string. */
  if (!ret_type->str) {
    ret_type->str = ll_get_str_type_for_basic_type(type);
  }

  return ret_type;
}

/**
   \brief Create an integer type with \p bits bits.
   \param module    The LLVM module
   \param bits      The number of bits used in the representation
   \param addrspace The address space where type will be in
   \return  A uniqued integral \ref LL_Type

   See \ref LL_BaseDataType for all the integer bitwidths supported.
 */
LL_Type *
ll_create_int_type_with_addrspace(LLVMModuleRef module, unsigned bits,
                                  int addrspace)
{
  enum LL_BaseDataType bdt = LL_NOTYPE;
  switch (bits) {
  case 1:
    bdt = LL_I1;
    break;
  case 8:
    bdt = LL_I8;
    break;
  case 16:
    bdt = LL_I16;
    break;
  case 24:
    bdt = LL_I24;
    break;
  case 32:
    bdt = LL_I32;
    break;
  case 40:
    bdt = LL_I40;
    break;
  case 48:
    bdt = LL_I48;
    break;
  case 56:
    bdt = LL_I56;
    break;
  case 64:
    bdt = LL_I64;
    break;
  case 128:
    bdt = LL_I128;
    break;
  case 256:
    bdt = LL_I256;
    break;
  default:
    interr("Unsupport integer bitwidth", bits, ERR_Fatal);
  }
  return ll_create_basic_type(module, bdt, addrspace);
}

/**
   \brief Create an integer type with \p bits bits.
   \param module   The LLVM module
   \param bits     The number of bits used in the representation
   \return  A uniqued integral \ref LL_Type

   See \ref LL_BaseDataType for all the integer bitwidths supported.
 */
LL_Type *
ll_create_int_type(LLVMModuleRef module, unsigned bits)
{
  return ll_create_int_type_with_addrspace(module, bits, LL_AddrSp_Default);
}

/**
   \brief Get a version of type in different address space.

   Note: In LLVM IR only pointer types have an address space.
 */
LL_Type *
ll_get_addrspace_type(LL_Type *type, int addrspace)
{
  struct LL_Type_ new_type;

  if (addrspace == type->addrspace)
    return type;

  memcpy(&new_type, type, sizeof(new_type));
  new_type.addrspace = addrspace;

  return unique_type(type->module, &new_type);
}

/**
   \brief Create a pointer type pointing to the given pointee type.
 */
LL_Type *
ll_get_pointer_type(LL_Type *type)
{
  struct LL_Type_ new_type;
  struct LL_Type_ *ret_type;
  LLVMModuleRef module = type->module;

  new_type.str = NULL;
  new_type.data_type = LL_PTR;
  new_type.flags = 0;
  new_type.sub_types = &type;
  new_type.sub_offsets = NULL;
  new_type.sub_elements = 1;
  new_type.sub_padding = NULL;
  new_type.addrspace = 0;//type->addrspace;

  ret_type = unique_type(module, &new_type);

  /* We need to assign a representation string and allocate a proper array
   * for sub_types if unique_type() actually allocated a new type. */
  if (!ret_type->str) {
    char ptrstr[32] = "ptr";
    char *new_str;
    int size;

    if (type->addrspace) {
      snprintf(ptrstr, sizeof ptrstr, "ptr addrspace(%d)", type->addrspace);
    }
    size = strlen(ptrstr) + 1;
    new_str = (char *)ll_manage_malloc(module, size);
    snprintf(new_str, sizeof new_str, "%s", ptrstr);

    ret_type->str = new_str;
    ret_type->sub_types =
        (LL_Type **)ll_manage_malloc(module, sizeof(LL_Type *));
    ret_type->sub_types[0] = type;
  }

  return ret_type;
}

LL_Type *
ll_get_array_type(LL_Type *type, BIGUINT64 num_elements, int addrspace)
{
  LLVMModuleRef module = type->module;
  struct LL_Type_ new_type;
  struct LL_Type_ *ret_type;

  new_type.str = NULL;
  new_type.data_type = LL_ARRAY;
  new_type.flags = 0;
  new_type.sub_types = &type;
  new_type.sub_offsets = NULL;
  new_type.sub_elements = num_elements;
  new_type.sub_padding = NULL;
  new_type.addrspace = addrspace;

  ret_type = unique_type(module, &new_type);

  if (!ret_type->str) {
    const char *suffix = "]";
    char prefix[32];
    char *new_str;

    sprintf(prefix, "[%" BIGIPFSZ "u x ", num_elements);
    new_str = (char *)ll_manage_malloc(
        module, strlen(prefix) + strlen(type->str) + strlen(suffix) + 1);
    sprintf(new_str, "%s%s%s", prefix, type->str, suffix);

    ret_type->str = new_str;
    ret_type->sub_types =
        (LL_Type **)ll_manage_malloc(module, sizeof(LL_Type *));
    ret_type->sub_types[0] = type;
  }

  return ret_type;
}

/**
  \brief Get a vector type <tt> \<num x elem\> </tt>
 */
LL_Type *
ll_get_vector_type(LL_Type *type, unsigned num_elements)
{
  LLVMModuleRef module = type->module;
  struct LL_Type_ new_type;
  struct LL_Type_ *ret_type;

  new_type.str = NULL;
  new_type.data_type = LL_VECTOR;
  new_type.flags = 0;
  new_type.sub_types = &type;
  new_type.sub_offsets = NULL;
  new_type.sub_elements = num_elements;
  new_type.sub_padding = NULL;
  new_type.addrspace = 0;

  ret_type = unique_type(module, &new_type);

  if (!ret_type->str) {
    char prefix[32];
    char *new_str;

    sprintf(prefix, "<%u x ", num_elements);

    new_str = (char *)ll_manage_malloc(module,
                                       strlen(prefix) + strlen(type->str) + 2);
    sprintf(new_str, "%s%s>", prefix, type->str);

    ret_type->str = new_str;
    ret_type->sub_types =
        (LL_Type **)ll_manage_malloc(module, sizeof(LL_Type *));
    ret_type->sub_types[0] = type;
  }

  return ret_type;
}

/**
 * \brief Replace the entire body of a named struct type. This may be used to
 * change the number of elements in an existing named struct.
 *
 * It is not possible to change the body of an anonymous struct type.
 *
 * The struct type is identified by the LL_Value returned from
 * ll_create_named_struct_type().
 */
void
ll_set_struct_body(LL_Type *ctype, LL_Type *const *elements,
                   unsigned *const offsets, char *const pads,
                   unsigned num_elements, int is_packed)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  struct LL_Type_ *type = (struct LL_Type_ *)ctype; /* cast away const */
#pragma GCC diagnostic pop
  assert(type->data_type == LL_STRUCT &&
             (type->flags & LL_TYPE_IS_NAMED_STRUCT),
         "Can only set the body on a named struct type", 0, ERR_Fatal);
  if (type->sub_types)
    free(type->sub_types);
  type->sub_elements = num_elements;
  type->sub_types = NULL;
  type->sub_offsets = NULL;
  type->sub_padding = NULL;
  if (num_elements > 0) {
    type->sub_types = (LL_Type **)calloc(num_elements, sizeof(LL_Type *));
    if (elements)
      memcpy(type->sub_types, elements, num_elements * sizeof(LL_Type *));
    if (offsets) {
      type->sub_offsets =
          (unsigned *)calloc(num_elements + 1, sizeof(unsigned));
      memcpy(type->sub_offsets, offsets, (num_elements + 1) * sizeof(unsigned));
    }
    if (pads) {
      type->sub_padding = (char *)calloc(num_elements, 1);
      memcpy(type->sub_padding, pads, num_elements);
    }
  }
  if (is_packed)
    type->flags |= LL_TYPE_IS_PACKED_STRUCT;
}

LL_Value *
ll_named_struct_type_exists(LLVMModuleRef module, int id, const char *format,
                            ...)
{
  va_list ap;
  char buffer[256];
  LL_Value *struct_value;

  buffer[0] = '%';
  buffer[1] = '\0';
  va_start(ap, format);
  vsnprintf(buffer + 1, sizeof(buffer) - 1, format, ap);
  va_end(ap);
  buffer[sizeof(buffer) - 1] = '\0';
  if (hashset_lookup(module->used_type_names, buffer)) {
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
    if (hashmap_lookup(module->user_structs_byid, INT2HKEY(id),
                       (hash_data_t *)&struct_value)) {
      return struct_value;
    }
#pragma GCC diagnostic pop
  }
  return NULL;
}

/**
   \brief Create a new named struct type in the module.

   Type can be uniquely named if parameter unique is set to true

   If id is positive, the new struct type is associated with the id so it can be
   retrieved with ll_get_struct_type(). Usually, the TY_STRUCT dtype is used as
   an id.

   It is an error to create multiple types with the same positive id, unless
   ll_reset_module_types() is called in between.

   A zero or negative id is ignored.

   The name of the new struct type will be based on the printf-style format and
   additional arguments. A '%' character will be prepended, don't include one in
   the format string.

   The new struct name may not be exactly as requested, so don't depend on it.
   The type name may be modified to ensure its uniqueness.
 */
LL_Type *
ll_create_named_struct_type(LLVMModuleRef module, int id, bool unique,
                            const char *format, ...)
{
  va_list ap;
  struct LL_Type_ *new_type;
  LL_Value *struct_value;

  if (!unique) {
    va_start(ap, format);
    struct_value = ll_named_struct_type_exists(module, id, format, ap);
    va_end(ap);
    if (struct_value)
      return struct_value->type_struct;
    ll_remove_struct_type(module, id);
  }
  va_start(ap, format);
  new_type = (struct LL_Type_ *)calloc(1, sizeof(struct LL_Type_));
  new_type->str = unique_name(module->used_type_names, '%', format, ap);
  va_end(ap);

  new_type->module = module;
  new_type->data_type = LL_STRUCT;
  new_type->flags = LL_TYPE_IS_NAMED_STRUCT;

  /* Maintain a list of types in the module so we can print them later. */
  struct_value = ll_create_value_from_type(module, new_type, new_type->str);
  ll_create_sym(&(module->user_structs), module->num_user_structs,
                struct_value);
  module->num_user_structs++;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (id > 0) {
    hash_key_t old_id = hashmap_replace(module->user_structs_byid, INT2HKEY(id),
                                        (hash_data_t *)&struct_value);
    assert(!old_id, "Duplicate structs created for id.", id, ERR_Fatal);
  }
#pragma GCC diagnostic pop

  return new_type;
}

/**
   \brief Remove struct type from hashmap

   This is required whenever a struct has a same dtype but in different modules
   defined in same file
 */
void
ll_remove_struct_type(LLVMModuleRef module, int struct_id)
{
  hashmap_erase(module->user_structs_byid, INT2HKEY(struct_id), NULL);
}

/**
   \brief Get an existing named struct type by id.

   The id must be positive.

   If no named struct type has been created with that id, NULL is returned. If
   required is true, an internal compiler error is signaled instead of returning
   NULL.
 */
LL_Type *
ll_get_struct_type(LLVMModuleRef module, int struct_id, int required)
{
  LL_Value *struct_value = NULL;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (hashmap_lookup(module->user_structs_byid, INT2HKEY(struct_id),
                     (hash_data_t *)&struct_value))
    return struct_value->type_struct;
#pragma GCC diagnostic pop

  assert(!required, "Can't find user defined struct.", struct_id, ERR_Fatal);
  return 0;
}

/**
   \brief Get an anonymous struct type.

   Anonymous structs are identified by their contents, they don't have a name.
   This will either return an existing anonymous struct type with the required
   elements, or create a new type with a copy of the elements array.

   A packed struct has alignment 1 and no padding between members.
 */
LL_Type *
ll_create_anon_struct_type(LLVMModuleRef module, LL_Type *elements[],
                           unsigned num_elements, bool is_packed, int addrspace)
{
  struct LL_Type_ new_type;
  struct LL_Type_ *ret_type;

  new_type.str = NULL;
  new_type.data_type = LL_STRUCT;
  new_type.flags = is_packed ? LL_TYPE_IS_PACKED_STRUCT : 0;
  new_type.sub_types = elements;
  new_type.sub_offsets = (unsigned *)calloc(num_elements + 1, sizeof(unsigned));
  new_type.sub_elements = num_elements;
  new_type.sub_padding = NULL;
  new_type.addrspace = 0; //addrspace;

  unsigned i, offset;
  for (i = 0, offset = 0; i < num_elements; ++i) {
    new_type.sub_offsets[i] = offset;
    offset += ll_type_bytes(new_type.sub_types[i]);
  }
  new_type.sub_offsets[num_elements] = offset;

  ret_type = unique_type(module, &new_type);

  if (!ret_type->str) {
    if (num_elements == 0) {
      ret_type->str = is_packed ? "<{}>" : "{}";
      ret_type->sub_types = NULL;
    } else {
      char *new_str;
      unsigned i, len, pos;

      /* Build new_str = "{i32, i8}". First compute length to allocate. */
      len = 3;
      for (i = 0; i < num_elements; i++)
        len += strlen(elements[i]->str) + 2;

      new_str = (char *)ll_manage_malloc(module, len);
      sprintf(new_str, is_packed ? "<{%s" : "{%s", elements[0]->str);
      pos = strlen(new_str);
      for (i = 1; i < num_elements; i++) {
        sprintf(new_str + pos, ", %s", elements[i]->str);
        pos += strlen(new_str + pos);
      }
      strcat(new_str + pos, is_packed ? "}>" : "}");
      ret_type->str = new_str;
      ret_type->sub_types = (LL_Type **)ll_manage_malloc(
          module, num_elements * sizeof(struct LL_Type_ *));
      memcpy(ret_type->sub_types, elements,
             num_elements * sizeof(struct LL_Type_ *));
    }
  }

  return ret_type;
}

/**
   \brief Get a function type.

   The return type is passed as \c args[0], so the \c args[] array should have a
   total of <tt> num_args+1 </tt> elements.
 */
LL_Type *
ll_create_function_type(LLVMModuleRef module, LL_Type *args[],
                        unsigned num_args, int is_varargs)
{
  struct LL_Type_ new_type;
  struct LL_Type_ *ret_type;

  new_type.str = NULL;
  new_type.data_type = LL_FUNCTION;
  new_type.flags = is_varargs ? LL_TYPE_IS_VARARGS_FUNC : 0;
  new_type.sub_types = args;
  new_type.sub_offsets = NULL;
  new_type.sub_elements = num_args + 1;
  new_type.sub_padding = NULL;
  new_type.addrspace = 0;

  ret_type = unique_type(module, &new_type);

  if (!ret_type->str) {
    char *new_str;
    unsigned i, len, pos;

    /* Build new_str = "i32 (i8, i16)". First compute length to allocate. */
    len = 7;
    for (i = 0; i <= num_args; i++)
      len += strlen(args[i]->str) + 2;

    new_str = (char *)ll_manage_malloc(module, len);
    /* Warning: MSVC's version of sprintf does not support %n by default. */
    sprintf(new_str, "%s (", args[0]->str);
    pos = strlen(new_str);

    for (i = 1; i <= num_args; i++) {
      sprintf(new_str + pos, i > 1 ? ", %s" : "%s", args[i]->str);
      pos += strlen(new_str + pos);
    }

    if (!is_varargs)
      strcat(new_str + pos, ")");
    else if (num_args == 0)
      strcat(new_str + pos, "...)");
    else
      strcat(new_str + pos, ", ...)");

    ret_type->str = new_str;
    ret_type->sub_types = (LL_Type **)ll_manage_malloc(
        module, (1 + num_args) * sizeof(struct LL_Type_ *));
    memcpy(ret_type->sub_types, args,
           (1 + num_args) * sizeof(struct LL_Type_ *));
  }
  return ret_type;
}

/** \brief Get the textual representation of a calling convention. */
const char *
ll_get_calling_conv_str(enum LL_CallConv cc)
{
  switch (cc) {
  case LL_CallConv_C:
    return "ccc";
  case LL_CallConv_Fast:
    return "fastcc";
  case LL_CallConv_Cold:
    return "coldcc";
  case LL_CallConv_X86_StdCall:
    return "x86_stdcallcc";
  case LL_CallConv_X86_FastCall:
    return "x86_fastcallcc";
  case LL_CallConv_X86_ThisCall:
    return "x86_thiscallcc";
  case LL_CallConv_X86_VectorCall:
    return "x86_vectorcallcc";
  case LL_CallConv_APCS:
    return "arm_apcscc";
  case LL_CallConv_AAPCS:
    return "arm_aapcscc";
  case LL_CallConv_AAPCS_VFP:
    return "arm_aapcs_vfpcc";
  case LL_CallConv_PTX_Kernel:
    return "ptx_kernel";
  case LL_CallConv_PTX_Device:
    return "ptx_device";
  case LL_CallConv_SPIR_FUNC:
    return "spir_func";
  case LL_CallConv_SPIR_KERNEL:
    return "spir_kernel";
  }
  interr("Unknown LLVM calling convention", cc, ERR_Fatal);
  return "unknown cc";
}

/*
 * Interned constants.
 *
 * Constants are uniqued by their type_struct pointer (which is already
 * uniqued) and the text of their data.
 */

/* Find or create an LL_Value of the given type and textual representation.
 * Like ll_create_value_from_type() with uniquing.
 * Return an index into module->constants */
static unsigned
intern_constant(LLVMModuleRef module, LL_Type *type, const char *data)
{
  LL_Value temp;
  hash_data_t oldval;
  LL_Value *newval;
  unsigned slot;

  memset(&temp, 0, sizeof(temp));
  temp.data = data;
  temp.type_struct = type;

  /* Was this constant seen before? */
  if (hashmap_lookup(module->constants_map, &temp, &oldval))
    return HKEY2INT(oldval);

  /* First time we see this constant. */
  newval = ll_create_value_from_type(module, type, data);
  slot = module->constants_count;
  if (++module->constants_count > module->constants_alloc) {
    module->constants_alloc *= 2;
    module->constants = (LL_Value **)realloc(module->constants,
                                             module->constants_alloc *
                                                 sizeof(module->constants[0]));
  }
  module->constants[slot] = newval;
  hashmap_insert(module->constants_map, newval, INT2HKEY(slot));

  return slot;
}

/* Get the slot number if an interned constant int. */
static unsigned
intern_const_int(LLVMModuleRef module, unsigned bits, long long value)
{
  LL_Type *type = ll_create_int_type(module, bits);
  char buf[32];

  sprintf(buf, "%lld", value);
  return intern_constant(module, type, buf);
}

/**
   \brief Get a shared LL_Value representing a constant integer of up to 64
   bits.
 */
LL_Value *
ll_get_const_int(LLVMModuleRef module, unsigned bits, long long value)
{
  unsigned slot = intern_const_int(module, bits, value);
  return module->constants[slot];
}

/**
   \brief Get a pointer to an LLVM global given its name and type.

   This will prepend \c \@ to the name and add one level of indirection to the
   type.
 */
LL_Value *
ll_get_global_pointer(const char *name, LL_Type *type)
{
  char *llvmname = (char *)malloc(strlen(name) + 2);
  unsigned slot;

  sprintf(llvmname, "@%s", name);
  type = ll_get_pointer_type(type);
  slot = intern_constant(type->module, type, llvmname);
  free(llvmname);

  return type->module->constants[slot];
}

/**
   \brief Get the constant function pointer value representing a function

   Note that the type of this function pointer depends on the added function
   arguments.
 */
LL_Value *
ll_get_function_pointer(LLVMModuleRef module, LL_Function *function)
{
  LL_Type *func_type;
  unsigned i;
  LL_Type **args =
      (LL_Type **)malloc((1 + function->num_args) * sizeof(LL_Type *));
  args[0] = function->return_type;
  for (i = 0; i < function->num_args; ++i)
    args[i + 1] = function->arguments[i]->type_struct;

  /* FIXME: LL_Function needs a is_varargs flag. */
  func_type = ll_create_function_type(module, args, function->num_args, false);
  free(args);

  return ll_get_global_pointer(function->name, func_type);
}

/* Return the type corresponding to applying one gep index */
static LL_Type *
apply_gep_index(LL_Type *type, unsigned idx)
{
  switch (type->data_type) {
  case LL_PTR:
  case LL_VECTOR:
  case LL_ARRAY:
    return type->sub_types[0];
  case LL_STRUCT:
    /* When statics and common blocks create real struct types, we can:
       assert(idx < type->sub_elements,
            "apply_gep_index: GEP index outside struct", idx, 4);
     */
    if (idx < type->sub_elements)
      return type->sub_types[idx];
    else
      return type;
  default:
    interr("apply_gep_index: Invalid data type for GEP", type->data_type,
           ERR_Fatal);
  }
  return NULL;
}

/**
   \brief Get a shared \ref LL_Value representing a constant getelementptr of
   the provided pointer value.

   This creates a gep-expression that may be used to initialize globals and
   metadata. It does not create an LL_GEP instruction.

   The num_idx argument is the number of index operands on the gep
   expression. It should be at least 1.

   The gep indices following num_idx should be ints.

   Note: element type is the exected element type, this does not get elements of
   any different type.
 */
LL_Value *
ll_get_const_gep(LLVMModuleRef module, LL_Value *ptr, unsigned num_idx, ...)
{
  va_list ap;
  unsigned slot;
  char *pointee;
  /* Space for getelementptr(<ptr>, i32 <idx0>, i32 <idx1>, ...) */
  char *name = (char *)malloc(19 + strlen(ptr->type_struct->str) +
                              2 * strlen(ptr->data) + 16 * num_idx);
  char *p = name;
  LL_Type *type = ptr->type_struct;

  /*** getelementpointer can only be used on pointers ***/
  assert(num_idx >= 1, "ll_get_const_gep: Need at least one index.", num_idx,
         ERR_Fatal);
  assert(type->data_type == LL_PTR,
         "ll_get_const_gep: "
         "Expected a pointer type.",
         type->data_type, ERR_Fatal);

  /*** Compose pointee type string ****/
  pointee = (char *)malloc(3 + strlen(ptr->type_struct->sub_types[0]->str));
  pointee[0] = '\0';

  /* Not every version of LLVM requires pointee type for GEP */
  if (ll_feature_explicit_gep_load_type(&module->ir))
    sprintf(pointee, "%s, ", ptr->type_struct->sub_types[0]->str);

  /*** Put everything together ***/
  sprintf(p, "getelementptr(%s%s %s", pointee, ptr->type_struct->str,
          ptr->data);
  p += strlen(p);

  va_start(ap, num_idx);
  while (num_idx--) {
    int idx = va_arg(ap, int);
    sprintf(p, ", i32 %d", idx);
    p += strlen(p);
    type = apply_gep_index(type, idx);
  }
  va_end(ap);
  sprintf(p, ")");

  /* getelementptr produces a pointer in the same address space as
   * the original pointer. */
  type =
      ll_get_addrspace_type(type, ll_get_pointer_addrspace(ptr->type_struct));
  type = ll_get_pointer_type(type);

  slot = intern_constant(module, type, name);

  free(name);
  free(pointee);
  return module->constants[slot];
}

/**
   \brief Get a shared \ref LL_Value representing a constant bitcast of value to
   type.

   This creates a bitcast expression that may be used to initialize globals and
   metadata. It does not create an \c LL_BITCAST instruction.
 */
LL_Value *
ll_get_const_bitcast(LLVMModuleRef module, LL_Value *value, LL_Type *type)
{
  char *name;
  unsigned slot;

  if (value->type_struct == type)
    return value;

  /* Space for bitcast(<value> to <type>). */
  name = (char *)malloc(15 + strlen(value->type_struct->str) +
                        strlen(value->data) + strlen(type->str));
  sprintf(name, "bitcast(%s %s to %s)", value->type_struct->str, value->data,
          type->str);

  slot = intern_constant(module, type, name);

  free(name);
  return module->constants[slot];
}

/**
   \brief Get a shared \ref LL_Value representing an addrspacecast of the
   pointer value to type.

   When targeting LLVM version that don't have the addrspacecast instruction,
   this will use ptrtoint/inttoptr instead.
 */
LL_Value *
ll_get_const_addrspacecast(LLVMModuleRef module, LL_Value *value, LL_Type *type)
{
  char *name;
  unsigned slot;
  unsigned fromaddr = ll_get_pointer_addrspace(value->type_struct);
  unsigned destaddr = ll_get_pointer_addrspace(type);

  assert(fromaddr != destaddr,
         "ll_get_const_addrspacecast: "
         "Address spaces must differ",
         0, ERR_Fatal);

  /* Space for
   *   addrspacecast(<value> to <type>) or
   *   inttoptr(i64 ptrtoint(<value> to i64) to <type>)
   */
  name = (char *)malloc(48 + strlen(value->type_struct->str) +
                        strlen(value->data) + strlen(type->str));

  if (ll_feature_use_addrspacecast(&module->ir)) {
    sprintf(name, "addrspacecast(%s %s to %s)", value->type_struct->str,
            value->data, type->str);
  } else {
    const int ptrbits = 8 * ll_type_bytes(type);
    sprintf(name, "inttoptr(i%d ptrtoint(%s %s to i%d) to %s)", ptrbits,
            value->type_struct->str, value->data, ptrbits, type->str);
  }

  slot = intern_constant(module, type, name);

  free(name);
  return module->constants[slot];
}

/*
 * Metadata representation.
 */

/** \brief Get an LL_MDRef representing an i1 boolean value. */
LL_MDRef
ll_get_md_i1(int value)
{
  LL_MDRef mdref = LL_MDREF_INITIALIZER(MDRef_SmallInt1, value);
  assert(value == 0 || value == 1, "ll_get_md_i1: Invalid i1 value", value,
         ERR_Fatal);
  return mdref;
}

/** \brief Get an LL_MDRef representing an i32 integer value. */
LL_MDRef
ll_get_md_i32(LLVMModuleRef module, int value)
{
  /* Will value fit in the 29 bits available for MDRef_SmallInt32? */
  if ((value >> 29) == 0) {
    return LL_MDREF_ctor(MDRef_SmallInt32, value);
  }
  /* Create an interned constant instead. */
  return LL_MDREF_ctor(MDRef_Constant, intern_const_int(module, 32, value));
}

/** \brief Get an LL_MDRef representing an i64 integer value. */
LL_MDRef
ll_get_md_i64(LLVMModuleRef module, long long value)
{
  /* Will value fit in the 29 bits available for MDRef_SmallInt64? */
  if ((value >> 29) == 0) {
    LL_MDRef mdref = LL_MDREF_INITIALIZER(MDRef_SmallInt64, (unsigned)value);
    return mdref;
  }
  /* Create an interned constant instead. */
  return LL_MDREF_ctor(MDRef_Constant, intern_const_int(module, 64, value));
}

#define NEEDS_ESC(C) ((C) < 32 || (C) >= 127 || (C) == '\\' || (C) == '"')

/**
   \brief Get an LL_MDRef representing a raw binary string.

   The string can contain arbitrary bytes, including nuls; appropriate escaping
   will be added.
 */
LL_MDRef
ll_get_md_rawstring(LLVMModuleRef module, const void *rawstr, size_t length)
{
  const unsigned char *ustr = (const unsigned char *)rawstr;
  unsigned num_escapes = 0;
  unsigned slot;
  size_t i;
  char *str, *p;
  hash_data_t oldval;
  LL_MDRef mdref = LL_MDREF_INITIALIZER(MDRef_String, 0);

  /* Count the number of bytes that need escaping. */
  for (i = 0; i < length; i++)
    if (NEEDS_ESC(ustr[i]))
      ++num_escapes;

  /* Make a copy with escaped bytes, a !" prefix and a "\0 suffix. */
  p = str = (char *)malloc(length + 3 * num_escapes + 4);
  *p++ = '!';
  *p++ = '"';
  for (i = 0; i < length; i++) {
    if (NEEDS_ESC(ustr[i])) {
      sprintf(p, "\\%02x", ustr[i]);
      p += 3;
    } else {
      *p++ = ustr[i];
    }
  }
  *p++ = '"';
  *p++ = 0;

  /* Is this a known string? */
  if (hashmap_lookup(module->mdstrings_map, str, &oldval)) {
    mdref = LL_MDREF_ctor(mdref, HKEY2INT(oldval));
    free(str);
    return mdref;
  }

  /* This string hasn't been seen before. Allocate a new slot. */
  slot = module->mdstrings_count;
  if (++module->mdstrings_count > module->mdstrings_alloc) {
    module->mdstrings_alloc *= 2;
    module->mdstrings = (const char **)realloc(
        module->mdstrings,
        module->mdstrings_alloc * sizeof(module->mdstrings[0]));
  }
  module->mdstrings[slot] = str;
  hashmap_insert(module->mdstrings_map, str, INT2HKEY(slot));

  mdref = LL_MDREF_ctor(mdref, slot);
  assert(LL_MDREF_value(mdref) == slot, "Metadata string table overflow", 0,
         ERR_Fatal);
  return mdref;
}

/**
   \brief Get an LL_MDRef representing a string.

   The string can contain arbitrary bytes, except for nul; appropriate escaping
   will be added.
 */
LL_MDRef
ll_get_md_string(LLVMModuleRef module, const char *str)
{
  return ll_get_md_rawstring(module, str, strlen(str));
}

/**
   \brief Get an LL_MDRef representing an arbitrary LLVM constant value
 */
LL_MDRef
ll_get_md_value(LLVMModuleRef module, LL_Value *value)
{
  return LL_MDREF_ctor(
      MDRef_Constant, intern_constant(module, value->type_struct, value->data));
}

/* Allocate a mdnode an initialize its content array. */
static LL_MDNode *
alloc_mdnode(LLVMModuleRef module, enum LL_MDClass mdclass,
             const LL_MDRef *elems, unsigned nelems, int is_distinct)
{
  LL_MDNode *node =
      (LL_MDNode *)malloc(sizeof(LL_MDNode) + nelems * sizeof(LL_MDRef));

  node->num_elems = nelems;
  node->mdclass = mdclass;
  node->is_distinct = is_distinct;
  node->is_flexible = false;
  memcpy(node->elem, elems, nelems * sizeof(LL_MDRef));

  /* Check for bitfield overflow. */
  assert(node->num_elems == nelems, "MDNode overflow", nelems, ERR_Fatal);
  assert(node->mdclass == mdclass, "invalid MDNode class", mdclass, ERR_Fatal);
  return node;
}

#define MIN_FLEX_SIZE 4

/* Allocate a flexible mdnode and initialize its contents. */
static LL_MDNode *
alloc_flexible_mdnode(LLVMModuleRef module, const LL_MDRef *elems,
                      unsigned nelems)
{
  LL_MDNode *node;
  unsigned size;

  /* Flexible nodes always have an allocated capacity that is a power of two
   * and at least 4. */
  if (nelems <= MIN_FLEX_SIZE) {
    size = MIN_FLEX_SIZE;
  } else {
    /* Size is the smallest power of two >= nelems. */
    size = nelems - 1;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size += 1;
  }

  node = (LL_MDNode *)malloc(sizeof(LL_MDNode) + size * sizeof(LL_MDRef));
  node->num_elems = nelems;
  node->mdclass = LL_PlainMDNode;
  node->is_distinct = false;
  node->is_flexible = true;
  if (nelems)
    memcpy(node->elem, elems, nelems * sizeof(LL_MDRef));

  assert(node->num_elems == nelems, "MDNode overflow", nelems, ERR_Fatal);

  return node;
}

/**
   \brief Append element to flexible mdnode
 */
static void
mdnode_append(LLVMModuleRef module, LL_MDNode **pnode, LL_MDRef elem)
{
  unsigned nelems = (*pnode)->num_elems;

  assert((*pnode)->is_flexible, "Not a flexible metadata node", 0, ERR_Fatal);

  /* The allocated capacity will be a power of two >= 4.
   * Check if the new element will fit. */
  if (nelems >= MIN_FLEX_SIZE && (nelems & (nelems - 1)) == 0) {
    /* Node is full. Reallocate to the next power of two. */
    *pnode = (LL_MDNode *)realloc(*pnode, sizeof(LL_MDNode) +
                                              2 * nelems * sizeof(LL_MDRef));
  }
  (*pnode)->elem[(*pnode)->num_elems++] = elem;
}

/**
   \brief Reserve a slot for an LL_MDNode
   \param module  the LL_Module
   \return the reserved slot plus 1
 */
unsigned
ll_reserve_md_node(LLVMModuleRef module)
{
  const unsigned slot = module->mdnodes_count;

  if (++module->mdnodes_count > module->mdnodes_alloc) {
    module->mdnodes_alloc *= 2;
    module->mdnodes = (LL_MDNode **)realloc(
        module->mdnodes, module->mdnodes_alloc * sizeof(module->mdnodes[0]));
  }
  module->mdnodes[slot] = NULL;
  return slot + 1;
}

/**
   \brief Insert an allocated mdnode into the module's list
   \return the new node number (the position in the list)

   The node number is one higher than the index into mdnodes, so node !10 lives
   in slot 9.
 */
INLINE static unsigned
insert_mdnode(LLVMModuleRef module, LL_MDNode *node)
{
  const unsigned mdNum = ll_reserve_md_node(module);
  module->mdnodes[mdNum - 1] = node;
  return mdNum;
}

void
ll_set_md_node(LLVMModuleRef module, unsigned mdNum, LL_MDNode *node)
{
  assert((mdNum > 0) && (mdNum <= module->mdnodes_alloc), "slot out of bounds",
         mdNum, ERR_Fatal);
  assert(!module->mdnodes[mdNum - 1], "slot already set", mdNum, ERR_Fatal);
  module->mdnodes[mdNum - 1] = node;
}

/**
   \brief Get an LL_MDRef representing a numbered metadata node
 */
LL_MDRef
ll_get_md_node(LLVMModuleRef module, enum LL_MDClass mdclass,
               const LL_MDRef *elems, unsigned nelems)
{
  LL_MDNode *node = alloc_mdnode(module, mdclass, elems, nelems, false);
  LL_MDRef mdref = LL_MDREF_INITIALIZER(MDRef_Node, 0);
  hash_data_t oldval;
  unsigned mdnum;

  if (hashmap_lookup(module->mdnodes_map, node, &oldval)) {
    /* This is a duplicate, free the one we just allocated. */
    free(node);
    mdnum = HKEY2INT(oldval);
  } else {
    mdnum = insert_mdnode(module, node);
    hashmap_insert(module->mdnodes_map, node, INT2HKEY(mdnum));
  }

  mdref = LL_MDREF_ctor(mdref, mdnum);
  return mdref;
}

/**
   \brief Add \p sptr &rarr; \p mdnode to global debug map
   \param module  The module containing the map
   \param sptr    The key to be added
   \param mdnode  The value to be added

   If the key, \p sptr, is already in the map, the map is unaltered.
 */
void
ll_add_global_debug(LLVMModuleRef module, int sptr, LL_MDRef mdnode)
{
  hash_data_t oldval;
  const hash_key_t key = INT2HKEY(sptr);
  const hash_data_t value = INT2HKEY(mdnode);

  if (!hashmap_lookup(module->global_debug_map, key, &oldval))
    hashmap_insert(module->global_debug_map, key, value);
}

LL_MDRef
ll_get_global_debug(LLVMModuleRef module, int sptr)
{
  hash_data_t oldval;
  const hash_key_t key = INT2HKEY(sptr);

  if (hashmap_lookup(module->global_debug_map, key, &oldval))
    return HKEY2INT(oldval);
  return LL_MDREF_ctor(0, 0);
}

/**
   \brief Create a distinct metadata node which will not be uniqued with other
   identical metadata nodes.
 */
LL_MDRef
ll_create_distinct_md_node(LLVMModuleRef module, enum LL_MDClass mdclass,
                           const LL_MDRef *elems, unsigned nelems)
{
  LL_MDNode *node = alloc_mdnode(module, mdclass, elems, nelems, true);
  LL_MDRef md = LL_MDREF_INITIALIZER(MDRef_Node, insert_mdnode(module, node));
  return md;
}

/**
   \brief Create a distinct metadata node that can have elements appended
 */
LL_MDRef
ll_create_flexible_md_node(LLVMModuleRef module)
{
  LL_MDNode *node = alloc_flexible_mdnode(module, NULL, 0);
  LL_MDRef md = LL_MDREF_INITIALIZER(MDRef_Node, insert_mdnode(module, node));
  return md;
}

/**
   \brief Append an element to end of a flexible metadata node

   The node must have been created by ll_create_flexible_md_node()
 */
void
ll_extend_md_node(LLVMModuleRef module, LL_MDRef flexnode, LL_MDRef elem)
{
  unsigned slot = LL_MDREF_value(flexnode) - 1;
  assert(LL_MDREF_kind(flexnode) == MDRef_Node && slot < module->mdnodes_count,
         "Bad flexnode reference", 0, ERR_Fatal);
  mdnode_append(module, &module->mdnodes[slot], elem);
}

/**
   \brief Update one of the elements in a metadata node

   The node being updated must be either distinct or flexible, it is not allowed
   to update nodes that may be shared.
 */
void
ll_update_md_node(LLVMModuleRef module, LL_MDRef node_to_update,
                  unsigned elem_index, LL_MDRef elem)
{
  unsigned slot = LL_MDREF_value(node_to_update) - 1;
  LL_MDNode *node;
  assert(LL_MDREF_kind(node_to_update) == MDRef_Node &&
             slot < module->mdnodes_count,
         "ll_update_md_node: Bad metadata node reference", 0, ERR_Fatal);
  node = module->mdnodes[slot];
  assert(node->is_distinct || node->is_flexible,
         "ll_update_md_node: Cannot update potentially shared node", 0,
         ERR_Fatal);
  assert(elem_index < node->num_elems,
         "ll_update_md_node: Element index out of range", elem_index,
         ERR_Fatal);
  node->elem[elem_index] = elem;
}

/**
   \brief Add a named metadata node to the module
 */
void
ll_set_named_md_node(LLVMModuleRef module, enum LL_MDName name,
                     const LL_MDRef *elems, unsigned nelems)
{
  assert(name < MD_NUM_NAMES, "Invalid metadata name", name, ERR_Fatal);
  free(module->named_mdnodes[name]);
  module->named_mdnodes[name] = alloc_flexible_mdnode(module, elems, nelems);
}

/**
   \brief Append an element to a named metadata node

   Creates the named metadata node if needed.
 */
void
ll_extend_named_md_node(LLVMModuleRef module, enum LL_MDName name,
                        LL_MDRef elem)
{
  assert(name < MD_NUM_NAMES, "Invalid metadata name", name, ERR_Fatal);
  if (module->named_mdnodes[name])
    mdnode_append(module, &module->named_mdnodes[name], elem);
  else
    module->named_mdnodes[name] = alloc_flexible_mdnode(module, &elem, 1);
}

/**
   \brief Append a pointer to the \c \@llvm.used global

   Pointer bitcasts and/or addrspacecasts will be added as needed.
 */
void
ll_append_llvm_used(LLVMModuleRef module, LL_Value *ptr)
{
  unsigned addrspace = ll_get_pointer_addrspace(ptr->type_struct);
  LL_Type *i8ptr = ll_get_pointer_type(ll_create_int_type(module, 8));

  if (addrspace) {
    /* Use an addrspacecast to get an addrspace 0 pointer. */
    ptr = ll_get_const_addrspacecast(module, ptr, i8ptr);
  } else if (ptr->type_struct != i8ptr) {
    /* Use a bitcast to convert pointer types withon addrspace 0. */
    ptr = ll_get_const_bitcast(module, ptr, i8ptr);
  }

  /* Weed out duplicate entries, but don't bother with a full linear search.
   * Just catch consecutive duplicates. */
  if (module->num_llvm_used > 0 &&
      module->llvm_used.values[module->num_llvm_used - 1] == ptr)
    return;

  ll_create_sym(&module->llvm_used, module->num_llvm_used++, ptr);
}

/*
 * Global objects.
 *
 * The LL_Module maintains a linked list of LL_Objects with global scope. These
 * objects can be global variables, global constants, and aliases.
 */

static void
append_global(LLVMModuleRef module, LL_Object *object)
{
  object->next = NULL;
  if (module->last_global)
    module->last_global->next = object;
  else
    module->first_global = object;
  module->last_global = object;
}

static LL_Object *
create_global(LLVMModuleRef module, enum LL_ObjectKind kind, LL_Type *type,
              int addrspace, const char *format, va_list args)
{
  LL_Object *object =
      (LL_Object *)ll_manage_calloc(module, 1, sizeof(LL_Object));

  object->kind = kind;
  object->type = type;
  object->linkage = LL_EXTERNAL_LINKAGE;

  object->address.type_struct =
      ll_get_pointer_type(ll_get_addrspace_type(type, addrspace));
  object->address.data =
      unique_name(module->used_global_names, '@', format, args);

  append_global(module, object);
  return object;
}

/**
   \brief Create a new global alias.

   The name of the alias object is produced by the sprintf-link format string
   and arguments. If the generated name clashes with an existing global object
   in the module, a unique name will be generated.

   The aliasee must be the address of another global object or a constant
   expression computing an address inside a global object.
 */
LL_Object *
ll_create_global_alias(LL_Value *aliasee_ptr, const char *format, ...)
{
  int addrspace = ll_get_pointer_addrspace(aliasee_ptr->type_struct);
  LLVMModuleRef module = aliasee_ptr->type_struct->module;
  va_list ap;
  LL_Object *object;

  va_start(ap, format);
  object =
      create_global(module, LLObj_Alias, aliasee_ptr->type_struct->sub_types[0],
                    addrspace, format, ap);
  va_end(ap);

  object->init_style = LLInit_ConstExpr;
  object->init_data.const_expr = aliasee_ptr;

  return object;
}

/**
   \brief Create a new unique name for a local value in function.

   The name is based on the provided printf-style format string and arguments.
   The leading \c \% character should not be part for the format string.

   Returns a pointer to a unique local name in function, including the leading
   \c \%. The memory for the returned name is managed by the function.
 */
const char *
ll_create_local_name(LL_Function *function, const char *format, ...)
{
  va_list ap;
  char *name;

  if (!function->used_local_names)
    function->used_local_names = hashset_alloc(hash_functions_strings);

  va_start(ap, format);
  name = unique_name(function->used_local_names, '%', format, ap);
  va_end(ap);

  return name;
}

/**
   \brief Create a new local object in function.

   The object will be allocated by an LLVM alloca instruction in the entry
   block. It will not be initialized.

   The name of the local object is produced by the sprintf-link format string
   and arguments. If the generated name clashes with an existing local value in
   the function, a unique name will be generated.
 */
LL_Object *
ll_create_local_object(LL_Function *function, LL_Type *type,
                       unsigned align_bytes, const char *format, ...)
{
  LL_Object *object =
      (LL_Object *)ll_manage_calloc(type->module, 1, sizeof(LL_Object));
  va_list ap;

  if (!function->used_local_names)
    function->used_local_names = hashset_alloc(hash_functions_strings);

  object->kind = LLObj_Local;
  object->type = type;
  object->linkage = LL_INTERNAL_LINKAGE;
  object->align_bytes = align_bytes;
  object->address.type_struct = ll_get_pointer_type(type);
  va_start(ap, format);
  object->address.data =
      unique_name(function->used_local_names, '%', format, ap);
  va_end(ap);

  /* Append object to the linked list of locals in function. */
  if (function->last_local) {
    function->last_local->next = object;
    function->last_local = object;
  } else {
    function->first_local = function->last_local = object;
  }

  return object;
}

/* Global (covers all modules) */
static hashmap_t _ll_proto_map;

/* Global (head of proto list):
 *
 * We maintain a list to ease iteration and to also produce
 * a consistent debugging output.  A hash table might reshuffle the contents
 * and make debugging more difficult.
 */
static LL_FnProto *_ll_proto_head;

/**
   \brief Find the function (interface) name to be used as a key
   \param func_sptr  An sptr to a function symbol

   Given a function \p sptr return the function name (or interface name) to be
   used as a key for the rest of the \c fnname arguments for the
   <tt>ll_proto_*</tt> API.
 */
const char *
ll_proto_key(SPTR func_sptr)
{
  /* This is disabled for now, we plan on enabling this soon and cleaning up the
   * macros below.
   */
#if defined(TARGET_LLVM) && !defined(MATTD)
  return get_llvm_name(func_sptr);
#endif /* TARGET_LLVM && !MATTD */

#ifdef MATTD
  /* Fortran must check for interface names, C/C++ is straight forward) */
  const char *nm = NULL;
  const char *ifacenm = get_llvm_ifacenm(func_sptr);
  if (find_ag(ifacenm))
    nm = ifacenm;
  else
    nm = get_llvm_name(func_sptr);

  assert(nm, "ll_proto_key: No function name discovered", func_sptr, 4);
  return nm;
#endif /* MATTD */
  return NULL;
}

/**
   \brief Initialization

   Should only ever be called once for the entire compilation
 */
void
ll_proto_init(void)
{
/* TODO: Remove language guard once cg_llvm_init and end are untangled. */
  if (_ll_proto_map)
    return;
  if (!(_ll_proto_map = hashmap_alloc(hash_functions_strings)))
    interr("ll_proto_init: Could not allocate hashmap", 0, ERR_Fatal);
}

/* Set the name of the function. */
static void
ll_proto_update_name(LL_FnProto *proto, const char *fnname)
{
  free(proto->fn_name);
  proto->fn_name = strdup(fnname);
}

/**
   \brief Uniquely instantiate a new proto to the global map
 */
LL_FnProto *
ll_proto_add(const char *fnname, struct LL_ABI_Info_ *abi)
{
  LL_FnProto *proto;
  const char *key;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto))
    return proto;
#pragma GCC diagnostic pop

  proto = (LL_FnProto *)calloc(1, sizeof(LL_FnProto));
  if (!proto)
    interr("ll_proto_add: Could not allocate proto instance", 0, ERR_Fatal);

  proto->abi = abi;

  /* Keys are stored, we must remember the name string since
   * fortran will reset the name per module compilation.
   * Do not deallocate the keys unless the hashmap is deallocated.
   */
  key = strdup(fnname);
  proto->fn_name = strdup(fnname);
  hashmap_insert(_ll_proto_map, key, (hash_data_t)proto);

  if (!_ll_proto_head) {
    _ll_proto_head = proto;
  } else {
    proto->next = _ll_proto_head;
    _ll_proto_head = proto;
  }

  return proto;
}

/**
   \brief Convienience routine to ll_proto_add

   Also sets the proper name to use when defining/declaring the function.
 */
LL_FnProto *
ll_proto_add_sptr(SPTR func_sptr, struct LL_ABI_Info_ *abi)
{
  LL_FnProto *proto = ll_proto_add(ll_proto_key(func_sptr), abi);
  ll_proto_update_name(proto, get_llvm_name(func_sptr));
  return proto;
}

void
ll_proto_set_abi(const char *fnname, struct LL_ABI_Info_ *abi)
{
  LL_FnProto *proto = NULL;
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  /* Fortran might not yet have added this to the hash */
  if (hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto)) {
    proto->abi = abi;
  }
#pragma GCC diagnostic pop
}

struct LL_ABI_Info_ *
ll_proto_get_abi(const char *fnname)
{
  LL_FnProto *proto = NULL;
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto);
#pragma GCC diagnostic pop
  return proto ? proto->abi : NULL;
}

/**
   \brief Set the flag "function has body"
   \param fnname       a key (function name)
   \param has_defined  true iff \p fnname is a definition

   Sets the bit representing the information that the body of the function
   represented by \p fnname is defined.
 */
void
ll_proto_set_defined_body(const char *fnname, bool has_defined)
{
  LL_FnProto *proto = NULL;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (!hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto))
    interr("ll_proto_set_defined_body: Entry not found", 0, ERR_Fatal);
#pragma GCC diagnostic pop

  proto->has_defined_body = has_defined;
}

bool
ll_proto_has_defined_body(const char *fnname)
{
  LL_FnProto *proto = NULL;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (!hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto))
    interr("ll_proto_has_defined_body: Entry not found", 0, ERR_Fatal);
#pragma GCC diagnostic pop

  return proto->has_defined_body;
}

/**
   \brief Set the flag "function is weak"
   \param fnname       a key (function name)
   \param is_weak      true iff \p fnname is weak

   Sets the bit representing the information that the function
   represented by \p fnname is weak.
 */
void
ll_proto_set_weak(const char *fnname, bool is_weak)
{
  LL_FnProto *proto = NULL;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (!hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto))
    interr("ll_proto_set_weak: Entry not found", 0, ERR_Fatal);
#pragma GCC diagnostic pop

  proto->is_weak = is_weak;
}

bool
ll_proto_is_weak(const char *fnname)
{
  LL_FnProto *proto = NULL;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (!hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto))
    interr("ll_proto_is_weak: Entry not found", 0, ERR_Fatal);
#pragma GCC diagnostic pop

  return proto->is_weak;
}

/**
   \brief Set compiler generated intrinsic string
   \param fnname              The key to lookup the prototype
   \param intrinsic_decl_str  The LLVM declare statement text

   The prototype of \p fnname must already have been added to the map.
 */
void
ll_proto_set_intrinsic(const char *fnname, const char *intrinsic_decl_str)
{
  LL_FnProto *proto = NULL;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  if (!hashmap_lookup(_ll_proto_map, fnname, (hash_data_t *)&proto))
    interr("ll_proto_set_intrinsic: Entry not found", 0, ERR_Fatal);
#pragma GCC diagnostic pop

  proto->intrinsic_decl_str = strdup(intrinsic_decl_str);
}

void
ll_proto_iterate(LL_FnProto_Handler callback)
{
  LL_FnProto *proto;
  for (proto = _ll_proto_head; proto; proto = proto->next)
    callback(proto);
}

static void
dump_proto(int counter, const LL_FnProto *proto)
{
  FILE *fil = gbl.dbgfil ? gbl.dbgfil : stdout;
  fprintf(fil,
          "%d) %s: abi(%p) has_defined_body(%d) is_weak(%d) "
          "intrinsic(%s)\n",
          counter, proto->fn_name, proto->abi, proto->has_defined_body,
          proto->is_weak, proto->intrinsic_decl_str);
}

/**
   \brief Debugging: Dump the contents of the map
 */
void
ll_proto_dump(void)
{
  LL_FnProto *proto;
  int counter = 0;
  FILE *fil = gbl.dbgfil ? gbl.dbgfil : stdout;

  fprintf(fil, "** Function Name - to - Prototype Map **\n");
  for (proto = _ll_proto_head; proto; proto = proto->next)
    dump_proto(++counter, proto);
  fprintf(fil, "****************************************\n");
}

/**
   \brief Add \p module_name &rarr; \p mdnode to module debug map
   \param module       The module containing the map
   \param module_name  The key to be added
   \param mdnode       The value to be added

   If the key, \p module_name, is already in the map, the map is unaltered.
 */
void
ll_add_module_debug(hashmap_t map, const char *module_name, LL_MDRef mdnode)
{
  hash_data_t oldval;
  if (!hashmap_lookup(map, module_name, &oldval))
    hashmap_insert(map, module_name, INT2HKEY(mdnode));
}

LL_MDRef
ll_get_module_debug(hashmap_t map, const char *module_name)
{
  hash_data_t oldval;
  if (hashmap_lookup(map, module_name, &oldval))
    return HKEY2INT(oldval);
  return LL_MDREF_ctor(0, 0);
}

#ifdef OMP_OFFLOAD_LLVM
void
ll_set_device_function_arguments(LLVMModuleRef module,
                                 struct LL_Function_ *function,
                                 struct LL_ABI_Info_ *abi)
{
  int i;
  for (i = 1; i <= abi->nargs; i++) {
    LL_ABI_ArgInfo *arg = &abi->arg[i];
    LL_Value *argument = (LL_Value *)malloc(sizeof(LL_Value));
    argument->type_struct = arg->type;
    ll_set_function_argument(function, (i - 1), argument);
  }
}
#endif
