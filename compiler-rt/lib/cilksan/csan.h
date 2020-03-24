// -*- C++ -*-
#ifndef __CSAN_H__
#define __CSAN_H__

#include <csi/csi.h>

#include <cstring>

typedef struct {
  csi_id_t num_func;
  csi_id_t num_func_exit;
  csi_id_t num_loop;
  csi_id_t num_loop_exit;
  csi_id_t num_bb;
  csi_id_t num_call;
  csi_id_t num_load;
  csi_id_t num_store;
  csi_id_t num_detach;
  csi_id_t num_task;
  csi_id_t num_task_exit;
  csi_id_t num_detach_continue;
  csi_id_t num_sync;
  csi_id_t num_alloca;
  csi_id_t num_allocfn;
  csi_id_t num_free;
} csan_instrumentation_counts_t;

struct csan_source_loc_t {
  char *name;
  int32_t line_number;
  int32_t column_number;
  char *filename;
};

struct obj_source_loc_t {
  char *name;
  int32_t line_number;
  char *filename;
};

EXTERN_C

__attribute__((const))
const csan_source_loc_t *__csan_get_func_source_loc(const csi_id_t func_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_func_exit_source_loc(const csi_id_t func_exit_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_loop_source_loc(const csi_id_t loop_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_loop_exit_source_loc(const csi_id_t loop_exit_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_bb_source_loc(const csi_id_t bb_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_call_source_loc(const csi_id_t call_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_load_source_loc(const csi_id_t load_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_store_source_loc(const csi_id_t store_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_detach_source_loc(const csi_id_t detach_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_task_source_loc(const csi_id_t task_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_task_exit_source_loc(const csi_id_t task_exit_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_detach_continue_source_loc(const csi_id_t detach_continue_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_sync_source_loc(const csi_id_t sync_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_alloca_source_loc(const csi_id_t alloca_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_allocfn_source_loc(const csi_id_t allocfn_id);
__attribute__((const))
const csan_source_loc_t *__csan_get_free_source_loc(const csi_id_t free_id);

__attribute__((const))
const obj_source_loc_t *__csan_get_load_obj_source_loc(const csi_id_t load_id);
__attribute__((const))
const obj_source_loc_t *__csan_get_store_obj_source_loc(const csi_id_t store_id);
__attribute__((const))
const obj_source_loc_t *__csan_get_alloca_obj_source_loc(const csi_id_t alloca_id);
__attribute__((const))
const obj_source_loc_t *__csan_get_allocfn_obj_source_loc(const csi_id_t allocfn_id);

__attribute__((const))
const char *__csan_get_allocfn_str(const allocfn_prop_t prop);
__attribute__((const))
const char *__csan_get_free_str(const free_prop_t prop);

EXTERN_C_END

#endif // __CSAN_H__
