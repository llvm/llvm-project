/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Debug info generation for LLVM IR.
 */

#ifndef LLDEBUG_H_
#define LLDEBUG_H_

#include "gbldefs.h"
#include "ll_structure.h"
#include "llutil.h"

/**
   \brief Create a metadata node for the current subprogram
   \param db
   \param sptr
   \param ret_dtype
   \param findex

   Side-effect: stores the metadata node in the LL_DebugInfo struct.

   A function pointer to the corresponding LLVM function must be set later by
   lldbg_set_func_ptr().
 */
void lldbg_emit_subprogram(LL_DebugInfo *db, SPTR sptr, DTYPE ret_dtype,
                           int findex, bool targetNVVM);

/**
   \brief Create a metadata node for the outlined subprogram \p sptr
   \param db
   \param sptr
   \param findex
   \param func_name
   \param startlineno

   Side-effect: store the metadata in the LL_DebugInfo struct.


   A function pointer to the corresponding LLVM function must be set later by
   lldbg_set_func_ptr().
 */
void lldbg_emit_outlined_subprogram(LL_DebugInfo *db, int sptr, int findex,
                                    const char *func_name, int startlineno,
                                    bool targetNVVM);

void lldbg_emit_cmblk_variables(LL_DebugInfo *, int, int, char *, int);

struct INSTR_TAG;

/// \brief Write out metadata definitions to the current LLVM file
void write_metadata_defs(LL_DebugInfo *db);

/**
   \brief ...
 */
char *lldbg_alloc(INT size);

/// \brief Return debuginfo routine index
int lldbg_get_di_routine_idx(const LL_DebugInfo *db);

/// \brief Encode an argument to lldbg_emit_expression_mdnode()
int lldbg_encode_expression_arg(LL_DW_OP_t op, int value);

/**
   \brief Always produce \c !dbg metadata for current location

   This produces location info even when none exists.
 */
LL_MDRef lldbg_cons_line(LL_DebugInfo *db);

/**
   \brief Create a metadata node for the current compile unit
   \param db

   This function is idempotent.
 */
LL_MDRef lldbg_emit_compile_unit(LL_DebugInfo *db);

/**
   \brief Emit empty expression mdnode

   Metadata for \code{!DIExpression()}.
 */
LL_MDRef lldbg_emit_empty_expression_mdnode(LL_DebugInfo *db);

/**
   \brief Emit expression mdnode
   \param db   pointer to the debug info
   \param cnt  the number of arguments to the expression

   Each argument to the DIExpression needs to be encoded using the function
   lldbg_encode_expression_arg().

   Metadata for \code{!DIExpression(} \e op [, \e op ]* \code{)}
 */
LL_MDRef lldbg_emit_expression_mdnode(LL_DebugInfo *db, unsigned cnt, ...);

/**
   \brief Emit a metadata node for a local variable in the current function
   \return a reference to the variable

   The returned reference can be used as the last argument to \c
   llvm.dbg.declare or \c llvm.dbg.value.
 */
LL_MDRef lldbg_emit_local_variable(LL_DebugInfo *db, SPTR sptr, int findex,
                                   int emit_dummy_as_local);

/**
   \brief ...
 */
LL_MDRef lldbg_emit_module_mdnode(LL_DebugInfo *db, int sptr);

/**
   \brief Emit DILocalVariable for \p sptr parameter

   Emits a metadata node for a formal parameter to the current function.  The
   returned reference can be used as the last argument to \c llvm.dbg.declare
   or \c llvm.dbg.value.
 */
LL_MDRef lldbg_emit_param_variable(LL_DebugInfo *db, SPTR sptr, int findex,
                                   int parnum, bool unnamed);

/**
   \brief ...
 */
LL_MDRef lldbg_emit_ptr_param_variable(LL_DebugInfo *db, SPTR sptr, int findex,
                                       int parnum);

/// \brief Get metadata node representing the current line for \c !dbg
LL_MDRef lldbg_get_line(LL_DebugInfo *db);

/**
   \brief Get the metadata node representing the line for a var definition
   \param sptr  The variable to lookup
 */
LL_MDRef lldbg_get_var_line(LL_DebugInfo *db, int sptr);

/**
   \brief Get the \c DISubprogram for the current procedure
   \param db  the debug info object

   Note this has a side-effect: it clears the cached metadata.  This is to
   prevent the next function from re-using this one's DISubprogram.
 */
LL_MDRef lldbg_subprogram(LL_DebugInfo *db);

/**
   \brief The types of entity for debug metadata !DIImportedEntity
 */
typedef enum import_entity_type {
  IMPORTED_DECLARATION = 0,
  IMPORTED_MODULE = 1,
  IMPORTED_UNIT = 2
} IMPORT_TYPE;

/**
   \brief Emit DICommonBlock mdnode
 */
LL_MDRef lldbg_emit_common_block_mdnode(LL_DebugInfo *db, SPTR sptr);

/**
   \brief ...
 */
void lldbg_create_cmblk_mem_mdnode_list(SPTR sptr, SPTR gblsym);

/**
   \brief Add symbol to the CHILD field of pending import entity.
   \param db      the debug info object
   \param entity  the symbol of module or variable to be imported
   \param entity_type  0 indicates DECLARATION, 1 indicates MODULE, 
   2 indicates UNIT.
 */
void lldbg_add_pending_import_entity_to_child(LL_DebugInfo *db, SPTR entity,
                                             IMPORT_TYPE entity_type);

/**
   \brief Add one symbol to the list of pending import entities.
   \param db      the debug info object
   \param entity  the symbol of module or variable to be imported
   \param entity_type  0 indicates DECLARATION, 1 indicates MODULE, 
   2 indicates UNIT.
 */
void lldbg_add_pending_import_entity(LL_DebugInfo *db, SPTR entity,
                                     IMPORT_TYPE entity_type);

/**
   \brief Create a temporary debug name.
 */
const char *new_debug_name(const char *str1, const char *str2, const char *str3);

/**
   \brief ...
 */
void lldbg_cleanup_missing_bounds(LL_DebugInfo *db, int findex);

/**
   \brief ...
 */
void lldbg_emit_accel_global_variable(LL_DebugInfo *db, SPTR sptr, int findex,
                                      LL_Value *var_ptr, int addrspace,
                                      int is_local);

/**
   \brief Emit a metadata node for a global variable.

   Note that all LLVM globals are referenced as pointers, so \p value should
   have a pointer type.
 */
void lldbg_emit_global_variable(LL_DebugInfo *db, SPTR sptr, BIGINT off,
                                int findex, LL_Value *value);

/**
   \brief ...
 */
void lldbg_emit_line(LL_DebugInfo *db, int lineno);

/**
   \brief ...
 */
void lldbg_emit_lv_list(LL_DebugInfo *db);

/**
   \brief ...
 */
void lldbg_emit_outlined_parameter_list(LL_DebugInfo *db, int findex,
                                        DTYPE *param_dtypes, int num_args);

/**
   \brief Free all memory used by \p db
   \param db

   Don't call this directly, it is called from ll_destroy_module.
 */
void lldbg_free(LL_DebugInfo *db);

/**
   \brief Construct debug information at end of routine
   \param db    debug info instance
   \param func  current function symbol
 */
void lldbg_function_end(LL_DebugInfo *db, int func);

/**
   \brief Initialize dtype arrays
   \param db
 */
void lldbg_init_arrays(LL_DebugInfo *db);

/**
   \brief Allocate and initialize debug info generation for module
   \param module
 */
void lldbg_init(LL_Module *module);

/**
   \brief ...
 */
void lldbg_register_value_call(LL_DebugInfo *db, INSTR_LIST *instr, int sptr);

/**
   \brief ...
 */
void lldbg_reset_dtype_array(LL_DebugInfo *db, const int off);

/// \brief Provide a function pointer to the curent subprogram
void lldbg_set_func_ptr(LL_DebugInfo *db, LL_Value *func_ptr);

/**
   \brief Make room for new dtypes
   \param db         The debug info
   \param lastDType  dtype from which to bzero when extended
   \param newSz      the new size of dtype_array
 */
void lldbg_update_arrays(LL_DebugInfo *db, int lastDType, int newSz);

/// \brief Initialize the DIFLAG values
/// The values may vary depending on the LLVM branch being used
void InitializeDIFlags(const LL_IRFeatures *feature);

void lldbg_reset_module(LL_DebugInfo *db);

/// \brief Get the debug location mdnode of the current procedure.
LL_MDRef lldbg_get_subprogram_line(LL_DebugInfo *db);

/// \brief Return TRUE if SPTR is pointer to procedure.
LOGICAL is_procedure_ptr(SPTR sptr);

/// \brief Get parameter mdnode for SPTR
LL_MDRef get_param_mdnode(LL_DebugInfo *db, int sptr);
#endif /* LLDEBUG_H_ */
