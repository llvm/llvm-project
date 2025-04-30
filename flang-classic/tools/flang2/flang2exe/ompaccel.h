/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief ompaccel.c - OpenMP GPU Offload for NVVM Targets. It uses
 * libomptarget
 */

#ifndef OMPACCEL_H_
#define OMPACCEL_H_

#include "llmputil.h"
#include "expand.h"

/* Find if the func_sptr whether it is a kernel or not. */
#define IS_OMP_DEVICE_KERNEL(func_sptr) (OMPACCFUNCKERNELG(func_sptr))
/* Find if the func_sptr whether device function or not. */
#define IS_OMP_DEVICE_FUNC(func_sptr) (OMPACCFUNCDEVG(func_sptr))
/* Find whether we build ILI for OpenMP target or not.*/
#define IS_OMP_DEVICE_CG                     \
  (flg.omptarget && gbl.ompaccel_isdevice && \
   (IS_OMP_DEVICE_FUNC(GBL_CURRFUNC) | IS_OMP_DEVICE_KERNEL(GBL_CURRFUNC)))

#define OMPACCEL_DATA_FUNCTION NOSYM
#define OMPACCEL_DATA_MAX_SYM 50

typedef struct {
  SPTR shared_sym;
  SPTR private_sym;
  int redop;
} OMPACCEL_RED_SYM;

typedef struct {
  SPTR shuffleFn;
  SPTR interWarpCopy;
  SPTR copyToScratchPad;
  SPTR reduceScratchPad;
} OMPACCEL_RED_FUNCS;

typedef struct {
  SPTR host_sym;    /* host symbol */
  SPTR device_sym;  /* device symbol */
  int map_type;     /* map type */
  bool in_map;          /* set if it occurs in map */
  int ili_base;         /* symbol base */
  int ili_lowerbound;   /* lower bound */
  int ili_length;       /* length */
} OMPACCEL_SYM;

/* Target Info is the main struct which keeps all the information about target
 * or target data region. Each outlined function must have target info, if there
 * several nested outlining their information are kept hierarchically. Also,
 * each target data construct creates a target info. */
typedef struct _OMPACCEL_TARGET OMPACCEL_TINFO;

struct _OMPACCEL_TARGET{
  SPTR func_sptr;                         /*  Kernel or device function sptr          */
  OMPACCEL_SYM *symbols;                  /*  Keeps host and device symbols along with map-type */
  int n_symbols;                          /*  Number of parameters         */
  int sz_symbols;                         /*  Size of symbols array */
  OMPACCEL_SYM *quiet_symbols;            /*  Keeps sc_based symbols. They don't be passed to the device */
  int n_quiet_symbols;                    /*  Number of quiet_symbols */
  int sz_quiet_symbols;                   /*  Size of quite_symbols */
  OMP_TARGET_MODE mode;                   /*  Combined construct mode */
  OMPACCEL_TINFO* parent_tinfo;           /*  Parent tinfo is used for nested outlining in device. */
  bool nowait;                            /*  async      */
  int n_reduction_symbols;                /*  Number of reduction symbols */
  OMPACCEL_RED_SYM *reduction_symbols;    /*  Reduction symbols along with the reduction operator */
  OMPACCEL_RED_FUNCS reduction_funcs;     /*  Auxiliary functions for reduction */
};

static bool isOmpaccelRegistered = false;

extern OMPACCEL_TINFO **tinfos;

#define NVVM_WARPSIZE 32

typedef enum NVVM_SREG_ENUM {
  threadIdX,
  threadIdY,
  threadIdZ,
  blockIdX,
  blockIdY,
  blockIdZ,
  blockDimX,
  blockDimY,
  blockDimZ,
  gridDimX,
  gridDimY,
  gridDimZ,
  warpSize
} nvvm_sregs;

static const char *NVVM_SREG[] = {
    "llvm.nvvm.read.ptx.sreg.tid.x",    "llvm.nvvm.read.ptx.sreg.tid.y",
    "llvm.nvvm.read.ptx.sreg.tid.z",    "llvm.nvvm.read.ptx.sreg.ctaid.x",
    "llvm.nvvm.read.ptx.sreg.ctaid.y",  "llvm.nvvm.read.ptx.sreg.ctaid.z",
    "llvm.nvvm.read.ptx.sreg.ntid.x",   "llvm.nvvm.read.ptx.sreg.ntid.y",
    "llvm.nvvm.read.ptx.sreg.ntid.z",   "llvm.nvvm.read.ptx.sreg.nctaid.x",
    "llvm.nvvm.read.ptx.sreg.nctaid.y", "llvm.nvvm.read.ptx.sreg.nctaid.z",
    "llvm.nvvm.read.ptx.sreg.warpsize"};

typedef enum NVVM_INTRINSICS_ENUM { barrier0, barrier } nvvm_intrinsics;

static const char *NVVM_INTRINSICS[] = {"llvm.nvvm.barrier0",
                                        "llvm.nvvm.barrier"};

typedef enum NVVM_BARRIERS { CTA_BARRIER, PARTIAL_BARRIER } nvvm_barriers;

/* keeps beginning of the nvvm special register symbols */
static SPTR init_nvvm_syms = NOSYM;
static SPTR init_nvvm_intrinsics = NOSYM;

/* ################################################ */
/* OpenMP ACCEL - Utils                             */
/* ################################################ */
/**
   \brief Set LLVM's target-triple which is passed with
   -fopenmp-target=<target-triple>
 */
void ompaccel_set_targetriple(const char *);

/**
   \brief Return fopenmp-targets triple
 */
const char *ompaccel_get_targetriple(void);

/**
   \brief return whether tgt runtime is registered or not
 */
bool ompaccel_is_tgt_registered(void);

/**
   \brief Set tgt runtime as registered
 */
void ompaccel_register_tgt(void);

/**
   \brief Emit a ctor function which register tgt runtime
 */
void ompaccel_emit_tgt_register(void);

#ifdef DEBUG
#define ompaccelInternalFail(message)                                   \
  do {                                                                  \
    char buffer[10000];                                                 \
    sprintf(buffer, "File:%s Line %d, Function:%s", __FILE__, __LINE__, \
            __FUNCTION__);                                              \
    error((error_code_t)1204, ERR_Fatal, 0, message, buffer);           \
  } while (0)
#define ompaccelInternalFailure(format, ...)                                \
  do {                                                                      \
  char buffer[10000];                                                       \
  sprintf(buffer,                                                           \
          "[OpenMP-Offload] Error: [%s] at [File:%s Line %d, Function:%s]", \
          format, __FILE__, __LINE__, __FUNCTION__);                        \
  interrf(ERR_Fatal, buffer, ##__VA_ARGS__);                                \
  } while (0)

#else
#define ompaccelInternalFail(message)
#define ompaccelInternalFailures(format, ...)
#endif

/* ################################################ */
/* OpenMP ACCEL - NVVM Helpers                      */
/* ################################################ */
/**
   \brief Create NVVM special symbols and intrinsics
 */
void ompaccel_initsyms(void);

/**
   \brief Get special register. (nvvm device only)
 */
int ompaccel_nvvm_get(nvvm_sregs sreg);

/**
   \brief Get global thread id. It does not take into account master-warp. (nvvm
   device only)
 */
int ompaccel_nvvm_get_gbl_tid(void);

/**
   \brief Emit shuffle reduce for reduction. (nvvm device only)
 */
SPTR ompaccel_nvvm_emit_shuffle_reduce(OMPACCEL_RED_SYM *, int, SPTR);

/**
   \brief Emit reduce for reduction. (nvvm device only)
 */
SPTR ompaccel_nvvm_emit_reduce(OMPACCEL_RED_SYM *, int);

/**
   \brief Emit inter warp copy for reduction. (nvvm device only)
 */
SPTR ompaccel_nvvm_emit_inter_warp_copy(OMPACCEL_RED_SYM *, int);

/* ################################################ */
/* OpenMP ACCEL - Target Information data structure */
/* ################################################ */
/**
   \brief Initialize ompaccel, which keeps target region and data information
   structures
 */
void ompaccel_init(void);

/**
   \brief Create target info with data. Target info is designed to keep all the
   symbols which occur in the OpenMP construct's region along with their
   map-types. It also keeps the hierarchy.
 */
OMPACCEL_TINFO *ompaccel_tinfo_create(SPTR, int);

/**
   \brief Get target and data info of function
 */
OMPACCEL_TINFO *ompaccel_tinfo_get(int);
/**
   \brief Return whether parameter function sptr has target info or not.
 */
bool ompaccel_tinfo_has(int);
/**
   \brief Return current target info.
 */
OMPACCEL_TINFO *ompaccel_tinfo_current_get(void);
/**
   \brief Return current target data info.
 */
OMPACCEL_TINFO *ompaccel_tinfo_current_get_targetdata(void);
/**
   \brief Return current target region mode.
 */
OMP_TARGET_MODE ompaccel_tinfo_current_target_mode(void);
/**
   \brief Set the target region mode if it is combined construct.
 */
void ompaccel_tinfo_current_set_mode(OMP_TARGET_MODE);
/**
   \brief Set the target region mode if it is combined construct.
 */
void ompaccel_tinfo_set_mode_next_target(OMP_TARGET_MODE);
/**
   \brief Add a host symbol to the current target info.
 */
void ompaccel_tinfo_current_add_sym(SPTR, SPTR, int);
/**
   \brief Update map-type of the host symbol of the current target info.
 */
void ompaccel_tinfo_current_addupdate_mapitem(SPTR, int);
/**
   \brief Add reduction symbols to the current target info.
 */
void ompaccel_tinfo_current_add_reductionitem(SPTR, SPTR, int);

/**
   \brief Return whether is the the symbol is current tinfo or not.
 */
bool ompaccel_tinfo_current_is_registered(SPTR);

/**
   \brief Return device symbol of passed host symbol of current target info. It
   is designed to replace host symbols of outlined function code with device
   symbols.
 */
SPTR ompaccel_tinfo_current_get_devsptr(SPTR);
/**
   \brief Return device symbol's datatype of the passed host symbol's datatype
   of current target info.
 */
DTYPE ompaccel_tinfo_current_get_dev_dtype(DTYPE);
/**
   \brief Return device symbol of parent target info of the passed host symbol.
   It is used when there nested outlining in the device code.
 */
SPTR ompaccel_tinfo_parent_get_devsptr(SPTR);

/**
   \brief Create device symbol from the host symbol.
   Parameter count can be anything.
 */
SPTR
ompaccel_create_device_symbol(SPTR sptr, int count);

/* OpenMP ACCEL - Target Information data structure */

/* ################################################ */
/* OpenMP ACCEL - Dump routines                     */
/* ################################################ */
/**
   \brief Dump single target region.
 */
void dumpomptarget(OMPACCEL_TINFO *);

/**
   \brief Dump all target regions.
 */
void dumpomptargets(void);

/* ################################################ */
/* OpenMP ACCEL - Error messages                    */
/* ################################################ */
#define OMPACCELMESSAGE "OpenMP Accelerator Model:"
void ompaccel_msg_interr(const char *, const char *);
void ompaccel_msg_err(const char *, const char *);
void ompaccel_msg_warn(const char *, const char *);
void ompaccel_msg_info(const char *, const char *);

/* ################################################ */
/* OpenMP ACCEL - Expander                          */
/* ################################################ */
/**
   \brief Expand ILM and emit code for mploop
 */
void exp_ompaccel_mploop(ILM *ilmp, int);
/**
   \brief Expand ILM and emit code for mploopfini
 */
void exp_ompaccel_mploopfini(ILM *ilmp, int, int);
/**
   \brief Expand ILM and emit code for bpar
 */
void exp_ompaccel_bpar(ILM *ilmp, int, SPTR, SPTR, int(incrOutlinedCnt()));
/**
   \brief Expand ILM and emit code for epar
 */
void exp_ompaccel_epar(ILM *, int, int, int(decrOutlinedCnt()));
/**
   \brief Expand ILM and emit code for bteams
 */
void exp_ompaccel_bteams(ILM *ilmp, int, int, SPTR, SPTR,
                         int(incrOutlinedCnt()));
/**
   \brief Expand ILM and emit code for eteams
 */
void exp_ompaccel_eteams(ILM *ilmp, int, int, int(decrOutlinedCnt()));
/**
   \brief Expand ILM and emit code for btarget
 */
void exp_ompaccel_btarget(ILM *, int, SPTR, SPTR, int(incrOutlinedCnt()),
                          SPTR *, int *);
/**
   \brief Expand ILM and emit code for etarget
 */
void exp_ompaccel_etarget(ILM *, int, SPTR, int, SPTR, int(decrOutlinedCnt()));
/**
   \brief Expand ILM and emit code for reduction
 */
void exp_ompaccel_reduction(ILM *, int);
/**
   \brief Expand ILM and emit code for map
 */
void exp_ompaccel_map(ILM *, int, int);
/**
   \brief Expand ILM and emit code for emap
 */
void exp_ompaccel_emap(ILM *, int);
/**
   \brief Expand ILM and emit code for looptripcount
 */
void exp_ompaccel_looptripcount(ILM *, int);
/**
   \brief Expand ILM and emit code for reductionitem
 */
void exp_ompaccel_reductionitem(ILM *, int);
/**
   \brief Expand ILM and emit code for targetdata
 */
void exp_ompaccel_targetdata(ILM *, int, ILM_OP);
/**
   \brief Expand ILM and emit code for etargetdata
 */
void exp_ompaccel_etargetdata(ILM *, int);

int mk_ompaccel_store(int ili_value, DTYPE dtype, int nme, int ili_address);
int mk_ompaccel_mul(int ili1, DTYPE dtype1, int ili2, DTYPE dtype2);
int mk_ompaccel_add(int ili1, DTYPE dtype1, int ili2, DTYPE dtype2);
int mk_ompaccel_ldsptr(SPTR sptr);
void init_test();
#endif
