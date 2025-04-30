/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LLMPUTIL_H_
#define LLMPUTIL_H_

/** \file
 *  \brief OpenMP utility routines for LLVM compilers
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"

/** Uplevel data structure containing a list of shared variables for the region
 * nest that this uplevel belongs to.  The shared variables in this structure
 * are represented as a list of unique sptrs.
 */
typedef struct {
  int *vals;      /* Array containing shared var sptrs */
  int vals_size;  /* Total allocated slots in vals */
  int vals_count; /* Traditionally "available" or vals_avl */
  DTYPE dtype;    ///< The true dtype containing fields and their offsets
  SPTR parent;    /* sptr of its parent */
  /* TODO: Consider using a hashset to speed-unique lookup */
} LLUplevel;

/** First private variable:
 * This information is necessary to generate code that allows a particular task
 * to know which variables are stored in its allocated memory (task memory).
 * First privates for the task have to be stored in that memory.
 */
typedef struct _llprivate_t {
  int shared_sptr;  /**< Represents the caller's copy */
  int private_sptr; /**< Represents the callee's copy (local) */
} LLFirstPrivate;

/// Task data structure containing a list of private variables for the task.
typedef struct {
  int scope_sptr;        /**< Outlined task's scope sptr (BMPSCOPE ST_BLOCK) */
  int task_sptr;         /**< Outlined function representing the task */
  LLFirstPrivate *privs; /**< Array of private sptrs for this task */
  int privs_count;
  int privs_size;
  int actual_size; /**< Bytes in task: First priv size + base task size */
} LLTask;

/// Data attributes for each data reference used in an OpenMP target region.
typedef enum {
  // No flags
    OMP_TGT_MAPTYPE_NONE            = 0x000,
  // copy data from host to device
    OMP_TGT_MAPTYPE_TO              = 0x001,
  // copy data from device to host
    OMP_TGT_MAPTYPE_FROM            = 0x002,
  // copy regardless of the reference count
    OMP_TGT_MAPTYPE_ALWAYS          = 0x004,
  // force unmapping of data
    OMP_TGT_MAPTYPE_DELETE          = 0x008,
  // map the pointer as well as the pointee
    OMP_TGT_MAPTYPE_PTR_AND_OBJ     = 0x010,
  // pass device base address to kernel
    OMP_TGT_MAPTYPE_TARGET_PARAM    = 0x020,
  // return base device address of mapped data
    OMP_TGT_MAPTYPE_RETURN_PARAM    = 0x040,
  // private variable - not mapped
    OMP_TGT_MAPTYPE_PRIVATE         = 0x080,
  // copy by value - not mapped
    OMP_TGT_MAPTYPE_LITERAL         = 0x100,
  // mapping is implicit
    OMP_TGT_MAPTYPE_IMPLICIT        = 0x200,
  // member of struct, member given by 4 MSBs - 1
    OMP_TGT_MAPTYPE_MEMBER_OF       = 0xffff000000000000
} tgt_map_type;


/* The modes of target related regions. */
typedef enum {
  mode_none_target,
  mode_target,
  mode_target_teams,
  mode_target_teams_distribute,
  mode_target_teams_distribute_simd,
  mode_target_teams_distribute_parallel_for,
  mode_target_teams_distribute_parallel_for_simd,
  mode_target_parallel,
  mode_target_parallel_for,
  mode_target_parallel_for_simd,
  mode_target_simd,
  mode_target_data_enter_region,
  mode_target_data_exit_region,
  mode_target_data_region,
  mode_outlinedfunc_teams,
  mode_outlinedfunc_parallel,
  mode_targetupdate_begin,
  mode_targetupdate_end,
} OMP_TARGET_MODE;

bool is_omp_mode_target(OMP_TARGET_MODE mode);

/* The name of the modes of target related regions. */
static const char *omp_target_mode_names[] = {
                                    "None target",
                                    "target",
                                    "target teams",
                                    "target teams distribute",
                                    "target teams distribute simd",
                                    "target teams distribute parallel for",
                                    "target teams distribute parallel for simd",
                                    "target parallel",
                                    "target parallel for",
                                    "target parallel for simd",
                                    "target simd",
                                    "target data enter",
                                    "target data exit",
                                    "target data",
                                    "outlined teams region",
                                    "outlined parallel region",
                                    "target update begin",
                                    "target update end" };


/* Obtain a previously created task object, where scope_sptr is the BMPSCOPE
 * scope sptr containing the task.
 */
extern LLTask *llmp_get_task(int scope_sptr);

/* Return the task base size without any private values being stored. */
extern int llmp_task_get_base_task_size(void);

/* Return the task's total size including task metadata and priv vars */
extern int llmp_task_get_size(LLTask *task);

/* Set the task function sptr */
extern void llmp_task_set_fnsptr(LLTask *task, int task_sptr);

/* Return a task a object associated to 'task_sptr' */
extern LLTask *llmp_task_get_by_fnsptr(int task_sptr);

/* Returns the sptr of the 'private' (local to the callee) copy of the
 * private variable represented by 'sptr'.
 */
extern int llmp_task_get_private(const LLTask *task, int sptr, int incl);

/// \brief Uniquely add a shared variable
int llmp_add_shared_var(LLUplevel *up, int shared_sptr);

/// \brief Return a new key (index) into our table of all uplevels
int llmp_get_next_key(void);

/**
   \brief ...
 */
int llmp_task_add_loopvar(LLTask *task, int num, DTYPE dtype);

/**
   \brief Add a private sptr to the task object.
   priv:   sptr to the private copy of the private variable.
           ADDRESSP is called to set the offset to the kmpc task
           object where this private data will live during program
           execution.
 /
 */
int llmp_task_add_private(LLTask *task, int shared_sptr, SPTR private_sptr);

/**
   \brief ...
 */
int llmp_task_get_base_task_size(void);

/**
   \brief ...
 */
int llmp_task_get_private(const LLTask *task, int sptr, int encl);

/**
   \brief ...
 */
INT llmp_task_get_privoff(int sptr, const LLTask *task);

/**
   \brief ...
 */
int llmp_task_get_size(LLTask *task);

/**
   \brief ...
 */
int llmp_uplevel_has_parent(int uplevel);

/**
   \brief Create task object that can be searched for later using \p scope_sptr
   \param scope_sptr ...
 */
LLTask *llmp_create_task(int scope_sptr);

/**
   \brief ...
 */
LLTask *llmp_get_task(int scope_sptr);

/**
   \brief ...
 */
LLTask *llmp_task_get_by_fnsptr(int task_sptr);

/// \brief Retrieve an LLUplevel instance by key
LLUplevel *llmp_create_uplevel_bykey(int key);

/**
   \brief Create an LLUplevel instance
   \param stblock_sptr Block where this region nest begins.
   This is used as a key into the global list of all uplevels.
 */
LLUplevel *llmp_create_uplevel(int uplevel_sptr);

/// \brief Obtain a previously created uplevel
LLUplevel *llmp_get_uplevel(int uplevel_sptr);

/** Return an uplevel pointer if it has an entry in uplevel table
    or NULL if there is no entry.
 */
LLUplevel *llmp_has_uplevel(int uplevel_sptr);

/**
   \brief ...
 */
void dump_all_uplevel(void);

/**
   \brief ...
 */
void dump_uplevel(LLUplevel *up);

/**
   \brief ...
 */
void llmp_add_shared_var_charlen(LLUplevel *up, int shared_sptr);

/**
   \brief ...
 */
void llmp_append_uplevel(int from_sptr, int to_sptr);

/**
   \brief ...
 */
void llmp_concur_add_shared_var(int uplevel_sptr, int shared_sptr);

/**
   \brief ...
 */
void llmp_reset_uplevel(void);

/**
   \brief Return symbol pointer of its parent.
 */
SPTR llmp_get_parent_sptr(SPTR);

/**
   \brief Create a task object if it does not already exist for \p scope_sptr
   Add a private sptr to the task object.  shared, priv: See
   llmp_task_add_private
 */
void llmp_task_add(int scope_sptr, int shared_sptr, SPTR private_sptr);

/**
   \brief ...
 */
void llmp_task_set_fnsptr(LLTask *task, int task_sptr);

/**
   \brief Set the dtype (actual struct of member pointers)
 */
void llmp_uplevel_set_dtype(LLUplevel *up, DTYPE dtype);

/**
   \brief Set uplevel parent field.
 */
void llmp_uplevel_set_parent(SPTR uplevel_sptr, SPTR parent_sptr);

/**
   \brief Return outermost uplevel of current region.
 */
LLUplevel *llmp_outermost_uplevel(SPTR child);

/**
   \brief Return uplevel pointer of current uplevel's parent
 */
LLUplevel *llmp_parent_uplevel(SPTR child);

#endif /* LLMPUTIL_H_ */
