/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Various definitions for the libomptarget runtime
 */

#ifndef TGT_RUNTIME_H__
#define TGT_RUNTIME_H__

#include "ompaccel.h"
#include "gbldefs.h"

#define OMPACCEL_DEFAULT_DEVICEID -1

/* TGT API macros and structs */
enum {
  TGT_API_BAD,
  TGT_API_REGISTER_LIB,
  TGT_API_UNREGISTER_LIB,
  TGT_API_TARGET,
  TGT_API_TARGET_NOWAIT,
  TGT_API_TARGET_TEAMS,
  TGT_API_TARGET_TEAMS_NOWAIT,
  TGT_API_TARGET_TEAMS_PARALLEL,
  TGT_API_TARGET_TEAMS_PARALLEL_NOWAIT,
  TGT_API_TARGET_DATA_BEGIN,
  TGT_API_TARGET_DATA_BEGIN_DEPEND,
  TGT_API_TARGET_DATA_BEGIN_NOWAIT,
  TGT_API_TARGET_DATA_BEGIN_NOWAIT_DEPEND,
  TGT_API_TARGET_DATA_END,
  TGT_API_TARGET_DATA_END_DEPEND,
  TGT_API_TARGET_DATA_END_NOWAIT,
  TGT_API_TARGET_DATA_END_NOWAIT_DEPEND,
  TGT_API_TARGETUPDATE,
  TGT_API_N_ENTRIES /* <-- Always last */
};

typedef struct any_tgt_struct {
  const char *name;
  DTYPE dtype;
  int byval;
  int psptr;
} TGT_ST_TYPE;

struct tgt_api_entry_t {
  const char *name;     /* TGT API function name                    */
  const int ret_iliopc; /* TGT API function return value ili opcode */
  const DTYPE ret_dtype;  /* TGT API function return value type       */
};

/**
   \brief Register the file and load cubin file
 */
int ll_make_tgt_register_lib(void);

/**
   \brief Register the file and load cubin file
 */
int ll_make_tgt_register_lib2(void);
/**
   \brief Unregister the file
 */
int ll_make_tgt_unregister_lib(void);

/**
   \brief Start offload for target region
 */
int ll_make_tgt_target(SPTR, int64_t, SPTR);

/**
   \brief Start offload for target teams region
 */
int ll_make_tgt_target_teams(SPTR, int64_t, SPTR, int32_t, int32_t);

/**
   \brief Start offload for target teams region
 */
int ll_make_tgt_target_teams_parallel(SPTR, int64_t, SPTR, int32_t, int32_t, int32_t, int32_t);

/**
   \brief Start target data begin.
 */
int ll_make_tgt_target_data_begin(int, OMPACCEL_TINFO *);

/**
   \brief Finish target data begin.
 */
int ll_make_tgt_target_data_end(int, OMPACCEL_TINFO *);

/**
   \brief Finish target update begin.
 */
int ll_make_tgt_targetupdate_end(int, OMPACCEL_TINFO *);
/**
   \brief create tgt_offload_entry dtype
 */
DTYPE ll_make_tgt_offload_entry(char *);

void init_tgtutil();

#endif /* __TGT_RUNTIME_H__ */
