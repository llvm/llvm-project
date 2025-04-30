/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LL_BUILDER_H_
#define LL_BUILDER_H_

/**
   \file
   \brief Convenience functions for constructing LLVM IR.
 */

#include "ll_structure.h"
#include "llutil.h"

/*
 * LLMD_Builder -- Builder for metadata nodes.
 *
 * AN LLMD_Builder can be used to construct a single LLVM metadata node by
 * adding one element at a time. A set of llmd_add_* functions are provided for
 * adding different types of elements to the node.
 *
 * The LLMD_Builder struct is opaque. Allocate one with llmd_init() and destroy
 * it with llmd_finish().
 */
typedef struct LLMD_Builder_ *LLMD_Builder;

#define LL_MD_IS_NULL(MD) (!(MD))

/* Create a builder for constructing a metadata node in module. The allocated
 * memory must be freed by passing the returned builder to llmd_finish(). */
LLMD_Builder llmd_init(LL_Module *module);

void llmd_add_null(LLMD_Builder);
void llmd_add_i1(LLMD_Builder, int value);
void llmd_add_i32(LLMD_Builder, int value);
void llmd_add_i64(LLMD_Builder, long long value);

/* Add an i64 represented by two 32-bit numbers containing the least
 * significant bits and the most significant bits. */
void llmd_add_i64_lsb_msb(LLMD_Builder, unsigned lsb, unsigned msb);

/* Add an i64 represented as an DBLINT64 array */
void llmd_add_INT64(LLMD_Builder, DBLINT64 value);

/* Add a nul-terminated metadata string. The string may contain any non-nul
 * characters, escapes will be added. */
void llmd_add_string(LLMD_Builder, const char *value);

/* Add a reference to another metadata item. */
void llmd_add_md(LLMD_Builder, LL_MDRef value);

/* Add a general LL_Value reference. Note that the value must not be
 * function-local. */
void llmd_add_value(LLMD_Builder, LL_Value *value);

/* Reverse the list of node elements added so far. */
void llmd_reverse(LLMD_Builder);

/* Get the number of elements added so far. */
unsigned llmd_get_nelems(LLMD_Builder);

/* Request that a distinct metadata node is created so that it can be updated
 * later with ll_update_md_node(). */
void llmd_set_distinct(LLMD_Builder);

/* Set a metadata node class for the node being built. */
void llmd_set_class(LLMD_Builder, enum LL_MDClass);

/* Return the built metadata node after freeing any resources used by the
 * builder. */
LL_MDRef llmd_finish(LLMD_Builder);

LL_MDRef ll_finish_variable(LLMD_Builder, LL_MDRef);

/**
   \brief ...
 */
LLMD_Builder llmd_init(LL_Module *module);

/**
   \brief ...
 */
LL_MDRef ll_finish_variable(LLMD_Builder mdb, LL_MDRef fwd);

/**
   \brief ...
 */
LL_MDRef llmd_finish(LLMD_Builder mdb);

/**
   \brief ...
 */
unsigned llmd_get_nelems(LLMD_Builder mdb);

/**
   \brief ...
 */
void llmd_add_i1(LLMD_Builder mdb, int value);

/**
   \brief ...
 */
void llmd_add_i32(LLMD_Builder mdb, int value);

/**
   \brief ...
 */
void llmd_add_i64(LLMD_Builder mdb, long long value);

/**
   \brief ...
 */
void llmd_add_i64_lsb_msb(LLMD_Builder mdb, unsigned lsb, unsigned msb);

/**
   \brief ...
 */
void llmd_add_INT64(LLMD_Builder mdb, DBLINT64 value);

/**
   \brief ...
 */
void llmd_add_md(LLMD_Builder mdb, LL_MDRef mdnode);

/**
   \brief ...
 */
void llmd_add_null(LLMD_Builder mdb);

/**
   \brief ...
 */
void llmd_add_string(LLMD_Builder mdb, const char *value);

/**
   \brief ...
 */
void llmd_add_value(LLMD_Builder mdb, LL_Value *value);

/**
   \brief ...
 */
void llmd_reverse(LLMD_Builder mdb);

/**
   \brief ...
 */
void llmd_set_class(LLMD_Builder mdb, enum LL_MDClass mdclass);

/**
   \brief ...
 */
void llmd_set_distinct(LLMD_Builder mdb);


#endif /* LL_BUILDER_H_ */
