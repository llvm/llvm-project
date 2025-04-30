/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief  semantic analyzer routines which process SMP statements.
 */

#include "gbldefs.h"
#include "global.h"
#include "gramsm.h"
#include "gramtk.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "semstk.h"
#include "ast.h"
#include "fdirect.h"
#include "pragma.h"
#include "mach.h"
#include "rte.h" /* for get_static_descriptor */

#include "llmputil.h"
#include "mp.h"
#include "atomic_common.h"

/* contents of this file:  */

static void add_clause(int, LOGICAL);
static bool clause_errchk(BIGINT64, const char *);
static void accel_pragmagen(int, int, int);

static int sched_type(char *);
static void set_iftype(int, char *, char *, char *);
#ifdef FLANG_SEMSMP_UNUSED
static void validate_if(int, char *);
#endif
static int cancel_type(char *);
static int emit_bpar(void);
static int emit_btarget(int);
static void do_schedule(int);
static void do_private(void);
static void do_firstprivate(int);
static void do_lastprivate(void);
static void do_reduction(void);
static void do_copyin(void);
static void do_copyprivate(void);
static int size_of_allocatable(int);
static void do_default_clause(int);
static void begin_parallel_clause(int);
static void end_reduction(REDUC *, int);
static void end_lastprivate(int);
static void end_workshare(int s_std, int e_std);
static void deallocate_privates(int);
static void add_assignment(int, SST *);
static void add_assignment_before(int, SST *, int);
static void add_ptr_assignment(int, SST *);
static void assign_cval(int, int, int);
static int enter_dir(int, LOGICAL, LOGICAL, BITMASK64);
static int leave_dir(int, LOGICAL, LOGICAL);
static const char *name_of_dir(int);
static int find_reduc_intrinsic(int);
static int get_csect_sym(const char *);
static int get_csect_pfxlen(void);
static void check_barrier(void);
static void check_crit(char *);
static int check_cancel(int);
static void check_targetdata(int, const char *);
static void check_valid_data_sharing(int);
#ifdef FLANG_SEMSMP_UNUSED
static LOGICAL check_map_data_sharing(int);
#endif
static void cray_pointer_check(ITEM *, int);
static void other_firstlast_check(ITEM *, int);
static void non_private_check(int, const char *);
static void private_check();
static void deallocate_no_scope_sptr();
static int get_stblk_uplevel_sptr();
static int add_firstprivate_assn(int, int, int);
static void begin_combine_constructs(BIGINT64);
static void end_targteams();
static LOGICAL is_last_private(int);
static void mp_add_shared_var(int, int);
static void mk_reduction_list(void);
static void mk_shared_list(void);
static void mk_lastprivate_list(void);
static void save_private_list(void);
static void save_firstprivate_list(void);
static void save_shared_list(void);
static void restore_clauses(void);
static void do_bdistribute(int, LOGICAL);
static void do_bteams(int);
static int get_mp_bind_type(char *);
static LOGICAL is_valid_atomic_read(int, int);
static LOGICAL is_valid_atomic_write(int, int);
static LOGICAL is_valid_atomic_capture(int, int);
static LOGICAL is_valid_atomic_update(int, int);
static int mk_atomic_update_binop(int, int);
static int mk_atomic_update_intr(int, int);
static void do_map();
static LOGICAL use_atomic_for_reduction(int);

#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
static char *map_type;
bool isalways = false;
static int get_omp_combined_mode(BIGINT64 type);
static void mp_handle_map_clause(SST *, int, char *, int, int, bool);
static void mp_check_maptype(const char *maptype);
static LOGICAL is_in_omptarget(int d);
static LOGICAL is_in_omptarget_data(int d);
#endif
#ifdef OMP_OFFLOAD_LLVM
static void gen_reduction_ompaccel(REDUC *reducp, REDUC_SYM *reduc_symp,
                                   LOGICAL rmme, LOGICAL in_parallel);
#endif

/*-------- define data structures and macros local to this file: --------*/

/* define macros used to access table the clause table, "ct".
 * Their values range from 0 .. CL_MAXV.
 *
 * N O T E:  The static array of struct, cl, is initialized the names of
 *           these clauses.  If changes are made to the CL_ macros,
 *           R E M E M B E R  to change clname.
 */
#define CL_DEFAULT 0
#define CL_PRIVATE 1
#define CL_SHARED 2
#define CL_FIRSTPRIVATE 3
#define CL_LASTPRIVATE 4
#define CL_SCHEDULE 5
#define CL_ORDERED 6
#define CL_REDUCTION 7
#define CL_IF 8
#define CL_COPYIN 9
#define CL_COPYPRIVATE 10
#define CL_MP_SCHEDTYPE 11
#define CL_CHUNK 12
#define CL_NOWAIT 13
#define CL_NUM_THREADS 14
#define CL_COLLAPSE 15
#define CL_UNTIED 16
#define CL_COPYOUT 17
#define CL_LOCAL 18
#define CL_CACHE 19
#define CL_SHORTLOOP 20
#define CL_VECTOR 21
#define CL_PARALLEL 22
#define CL_SEQ 23
#define CL_HOST 24
#define CL_UNROLL 25
#define CL_KERNEL 26
#define CL_COPY 27
#define CL_MIRROR 28
#define CL_REFLECTED 29
#define CL_UPDATEHOST 30
#define CL_UPDATESELF 31
#define CL_UPDATEDEV 32
#define CL_INDEPENDENT 33
#define CL_WAIT 34
#define CL_CUFTILE 35
#define CL_KERNEL_GRID 36
#define CL_KERNEL_BLOCK 37
#define CL_SEQUNROLL 38
#define CL_PARUNROLL 39
#define CL_VECUNROLL 40
#define CL_CREATE 41
#define CL_ACCPRESENT 42
#define CL_ACCPCOPY 43
#define CL_ACCPCOPYIN 44
#define CL_ACCPCOPYOUT 45
#define CL_ACCPCREATE 46
#define CL_ACCPNOT 47
#define CL_ASYNC 48
#define CL_STREAM 49
#define CL_DEVICE 50
#define CL_WORKER 51
#define CL_GANG 52
#define CL_NUM_WORKERS 53
#define CL_NUM_GANGS 54
#define CL_VECTOR_LENGTH 55
#define CL_USE_DEVICE 56
#define CL_DEVICEPTR 57
#define CL_DEVICE_RESIDENT 58
#define CL_FINAL 59
#define CL_MERGEABLE 60
#define CL_DEVICEID 61
#define CL_ACCDELETE 62
#define CL_ACCPDELETE 63
#define CL_ACCLINK 64
#define CL_DEVICE_TYPE 65
#define CL_AUTO 66
#define CL_TILE 67
#define CL_GANGCHUNK 68
#define CL_DEFNONE 69
#define CL_NUM_GANGS2 70
#define CL_NUM_GANGS3 71
#define CL_GANGDIM 72
#define CL_DEFPRESENT 73
#define CL_FORCECOLLAPSE 74
#define CL_FINALIZE 75
#define CL_IFPRESENT 76
#define CL_SAFELEN 77
#define CL_SIMDLEN 78
#define CL_LINEAR 79
#define CL_ALIGNED 80
#define CL_USE_DEVICE_PTR 81
#define CL_DEPEND 82
#define CL_INBRANCH 83
#define CL_NOTINBRANCH 84
#define CL_UNIFORM 85
#define CL_GRAINSIZE 86
#define CL_NUM_TASKS 87
#define CL_NOGROUP 88
#define CL_OMPDEVICE 89
#define CL_MAP 90
#define CL_DEFAULTMAP 91
#define CL_TO 92
#define CL_LINK 93
#define CL_FROM 94
#define CL_NUM_TEAMS 95
#define CL_THREAD_LIMIT 96
#define CL_DIST_SCHEDULE 97
#define CL_PRIORITY 98
#define CL_IS_DEVICE_PTR 99
#define CL_SIMD 100
#define CL_THREADS 101
#define CL_DEVICE_NUM 102
#define CL_DEFAULT_ASYNC 103
#define CL_ACCDECL 104
#define CL_PROC_BIND 105
#define CL_ACCNO_CREATE 106
#define CL_ACCATTACH 107
#define CL_ACCDETACH 108
#define CL_ACCCOMPARE 109
#define CL_PGICOMPARE 110
#define CL_MAXV 111 /* This must be the last clause */
/*
 * define bit flag for each statement which may have clauses.  Used for
 * checking for illegal clauses.
 */
#define BT_PAR 0x001
#define BT_SINGLE 0x002
#define BT_PDO 0x004
#define BT_PARDO 0x008
#define BT_DOACROSS 0x010
#define BT_SECTS 0x020
#define BT_PARSECTS 0x040
#define BT_PARWORKS 0x080
#define BT_TASK 0x100
#define BT_ACCREG 0x200
#define BT_ACCKERNELS 0x400
#define BT_ACCPARALLEL 0x800
#define BT_ACCKDO 0x1000
#define BT_ACCPDO 0x2000
#define BT_ACCKLOOP 0x4000
#define BT_ACCPLOOP 0x8000
#define BT_ACCDATAREG 0x10000
#define BT_ACCDECL 0x20000
#define BT_ACCUPDATE 0x40000
#define BT_ACCENDREG 0x80000
#define BT_ACCSCALARREG 0x100000
#define BT_CUFKERNEL 0x200000
#define BT_ACCHOSTDATA 0x400000
#define BT_ACCENTERDATA 0x800000
#define BT_ACCEXITDATA 0x1000000
#define BT_SIMD 0x2000000
#define BT_TASKGROUP 0x4000000
#define BT_TASKLOOP 0x8000000
#define BT_TARGET 0x10000000
#define BT_DISTRIBUTE 0x20000000
#define BT_TEAMS 0x400000000
#define BT_DECLTARGET 0x800000000
#define BT_DECLSIMD 0x1000000000
#define BT_ACCINITSHUTDOWN 0x2000000000
#define BT_ACCSET 0x4000000000
#define BT_ACCSERIAL 0x8000000000
#define BT_ACCSLOOP 0x10000000000

static struct cl_tag { /* clause table */
  int present;
  BIGINT64 val;
  void *first;
  void *last;
  const char *name;
  BIGINT64 stmt; /* stmts which may use the clause */
} cl[CL_MAXV] = {
    {0, 0, NULL, NULL, "DEFAULT",
     BT_PAR | BT_PARDO | BT_PARSECTS | BT_PARWORKS | BT_TASK | BT_TEAMS |
         BT_TASKLOOP},
    {0, 0, NULL, NULL, "PRIVATE",
     BT_PAR | BT_PDO | BT_PARDO | BT_DOACROSS | BT_SECTS | BT_PARSECTS |
         BT_SINGLE | BT_PARWORKS | BT_TASK | BT_ACCPARALLEL | BT_ACCKDO |
         BT_ACCPDO | BT_ACCKLOOP | BT_ACCPLOOP | BT_SIMD | BT_TARGET |
         BT_TASKLOOP | BT_TEAMS | BT_DISTRIBUTE | BT_ACCSERIAL | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "SHARED",
     BT_PAR | BT_PARDO | BT_DOACROSS | BT_PARSECTS | BT_PARWORKS | BT_TASK |
         BT_TASKLOOP | BT_TEAMS},
    {0, 0, NULL, NULL, "FIRSTPRIVATE",
     BT_PAR | BT_PDO | BT_PARDO | BT_SECTS | BT_PARSECTS | BT_SINGLE |
         BT_PARWORKS | BT_TASK | BT_ACCPARALLEL | BT_TARGET | BT_TEAMS |
         BT_TASKLOOP | BT_DISTRIBUTE | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "LASTPRIVATE",
     BT_PDO | BT_PARDO | BT_DOACROSS | BT_SECTS | BT_PARSECTS | BT_SIMD |
         BT_TASKLOOP | BT_DISTRIBUTE},
    {0, 0, NULL, NULL, "SCHEDULE", BT_PDO | BT_PARDO},
    {0, 0, NULL, NULL, "ORDERED", BT_PDO | BT_PARDO},
    {0, 0, NULL, NULL, "REDUCTION",
     BT_PAR | BT_PDO | BT_PARDO | BT_DOACROSS | BT_SECTS | BT_PARSECTS |
         BT_PARWORKS | BT_ACCPARALLEL | BT_ACCKDO | BT_ACCPDO | BT_ACCKLOOP |
         BT_ACCPLOOP | BT_SIMD | BT_TEAMS | BT_ACCSERIAL | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "IF",
     BT_PAR | BT_PARDO | BT_PARSECTS | BT_PARWORKS | BT_TASK | BT_ACCREG |
         BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG | BT_ACCSCALARREG |
         BT_ACCUPDATE | BT_ACCENTERDATA | BT_ACCEXITDATA | BT_TARGET |
         BT_TASKLOOP | BT_ACCSERIAL | BT_ACCHOSTDATA},
    {0, 0, NULL, NULL, "COPYIN",
     BT_PAR | BT_PARDO | BT_PARSECTS | BT_PARWORKS | BT_ACCREG | BT_ACCKERNELS |
         BT_ACCPARALLEL | BT_ACCDATAREG | BT_ACCSCALARREG | BT_ACCDECL |
         BT_ACCENTERDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "COPYPRIVATE", BT_SINGLE},
    {0, 0, NULL, NULL, "MP_SCHEDTYPE", BT_DOACROSS},
    {0, 0, NULL, NULL, "CHUNK", BT_DOACROSS},
    {0, 0, NULL, NULL, "NOWAIT",
     BT_SINGLE | BT_SECTS | BT_PDO | BT_ACCREG | BT_ACCKERNELS |
         BT_ACCPARALLEL | BT_ACCSCALARREG | BT_ACCENDREG | BT_CUFKERNEL |
         BT_TARGET},
    {0, 0, NULL, NULL, "NUM_THREADS",
     BT_PAR | BT_PARDO | BT_PARSECTS | BT_PARWORKS},
    {0, 0, NULL, NULL, "COLLAPSE",
     BT_PDO | BT_PARDO | BT_ACCKDO | BT_ACCPDO | BT_ACCKLOOP | BT_ACCPLOOP |
         BT_SIMD | BT_TASKLOOP | BT_DISTRIBUTE | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "UNTIED", BT_TASK | BT_TASKLOOP},
    {0, 0, NULL, NULL, "COPYOUT",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCEXITDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "LOCAL",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCENTERDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "CACHE", BT_ACCKDO},
    {0, 0, NULL, NULL, "SHORTLOOP",
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "VECTOR",
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "PARALLEL",
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP},
    {0, 0, NULL, NULL, "SEQ",
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "HOST", BT_ACCKDO | BT_ACCKLOOP},
    {0, 0, NULL, NULL, "UNROLL",
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "KERNEL", BT_ACCKDO | BT_ACCKLOOP},
    {0, 0, NULL, NULL, "COPY",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "MIRROR", BT_ACCDATAREG | BT_ACCDECL},
    {0, 0, NULL, NULL, "REFLECTED", BT_ACCDECL},
    {0, 0, NULL, NULL, "UPDATE HOST",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "UPDATE SELF",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "UPDATE DEVICE",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "INDEPENDENT",
     BT_ACCKDO | BT_ACCPDO | BT_ACCKLOOP | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "WAIT",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCSCALARREG |
         BT_ACCENDREG | BT_CUFKERNEL | BT_ACCDATAREG | BT_ACCUPDATE |
         BT_ACCENTERDATA | BT_ACCEXITDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "TILE", BT_CUFKERNEL},
    {0, 0, NULL, NULL, "KERNEL_GRID", BT_CUFKERNEL},
    {0, 0, NULL, NULL, "KERNEL_BLOCK", BT_CUFKERNEL},
    {0, 0, NULL, NULL, "UNROLL", /* for sequential loops */
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "UNROLL", /* for parallel loops */
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP},
    {0, 0, NULL, NULL, "UNROLL", /* for vector loops */
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "CREATE",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCENTERDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "PRESENT",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "PRESENT_OR_COPY",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "PRESENT_OR_COPYIN",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCENTERDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "PRESENT_OR_COPYOUT",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCEXITDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "PRESENT_OR_CREATE",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCENTERDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "PRESENT_OR_NOT",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCENTERDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "ASYNC",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCUPDATE | BT_ACCENTERDATA | BT_ACCEXITDATA |
         BT_ACCSERIAL},
    {0, 0, NULL, NULL, "STREAM", BT_CUFKERNEL},
    {0, 0, NULL, NULL, "DEVICE", BT_CUFKERNEL},
    {0, 0, NULL, NULL, "WORKER",
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "GANG",
     BT_ACCKDO | BT_ACCKLOOP | BT_ACCPDO | BT_ACCPLOOP | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "NUM_WORKERS",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "NUM_GANGS",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "VECTOR_LENGTH",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "USE_DEVICE", BT_ACCHOSTDATA},
    {0, 0, NULL, NULL, "DEVICEPTR",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG | BT_ACCSERIAL | BT_ACCDECL},
    {0, 0, NULL, NULL, "DEVICE_RESIDENT", BT_ACCDECL},
    {0, 0, NULL, NULL, "FINAL", BT_TASK | BT_TASKLOOP},
    {0, 0, NULL, NULL, "MERGEABLE", BT_TASK | BT_TASKLOOP},
    {0, 0, NULL, NULL, "DEVICEID",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCUPDATE | BT_ACCHOSTDATA | BT_ACCENTERDATA |
         BT_ACCEXITDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "DELETE", BT_ACCEXITDATA},
    {0, 0, NULL, NULL, "PDELETE", BT_ACCEXITDATA},
    {0, 0, NULL, NULL, "LINK", BT_ACCDECL},
    {0, 0, NULL, NULL, "DEVICE_TYPE",
     BT_ACCKLOOP | BT_ACCPLOOP | BT_ACCKDO | BT_ACCPDO | BT_ACCKERNELS |
         BT_ACCPARALLEL | BT_ACCINITSHUTDOWN | BT_ACCSET | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "AUTO",
     BT_ACCKLOOP | BT_ACCPLOOP | BT_ACCKDO | BT_ACCPDO | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "TILE",
     BT_ACCKLOOP | BT_ACCPLOOP | BT_ACCKDO | BT_ACCPDO | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "GANG(STATIC:)",
     BT_ACCKLOOP | BT_ACCPLOOP | BT_ACCKDO | BT_ACCPDO | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "DEFAULT(NONE)",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCREG | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "NUM_GANGS(dim:2)",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "NUM_GANGS(dim:3)",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "GANG(DIM:)", BT_ACCPLOOP | BT_ACCPDO | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "DEFAULT(PRESENT)",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCREG | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "COLLAPSE(FORCE)",
     BT_ACCKLOOP | BT_ACCPLOOP | BT_ACCKDO | BT_ACCPDO | BT_ACCSLOOP},
    {0, 0, NULL, NULL, "FINALIZE", BT_ACCEXITDATA},
    {0, 0, NULL, NULL, "IF_PRESENT", BT_ACCUPDATE | BT_ACCHOSTDATA},
    {0, 0, NULL, NULL, "SAFELEN", BT_SIMD | BT_PDO | BT_PARDO},
    {0, 0, NULL, NULL, "SIMDLEN", BT_SIMD | BT_PDO | BT_PARDO | BT_DECLSIMD},
    {0, 0, NULL, NULL, "LINEAR", BT_SIMD | BT_PDO | BT_PARDO | BT_DECLSIMD},
    {0, 0, NULL, NULL, "ALIGNED", BT_SIMD | BT_PDO | BT_PARDO | BT_DECLSIMD},
    {0, 0, NULL, NULL, "USE_DEVICE_PTR", BT_TARGET},
    {0, 0, NULL, NULL, "DEPEND", BT_TASK | BT_TARGET},
    {0, 0, NULL, NULL, "INBRANCH", BT_DECLSIMD},
    {0, 0, NULL, NULL, "NOTINBRANCH", BT_DECLSIMD},
    {0, 0, NULL, NULL, "UNIFORM", BT_DECLSIMD},
    {0, 0, NULL, NULL, "GRAINSIZE", BT_TASKLOOP},
    {0, 0, NULL, NULL, "NUM_TASKS", BT_TASKLOOP},
    {0, 0, NULL, NULL, "NOGROUP", BT_TASKLOOP},
    {0, 0, NULL, NULL, "OMPDEVICE", BT_TARGET},
    {0, 0, NULL, NULL, "MAP", BT_TARGET},
    {0, 0, NULL, NULL, "DEFAULTMAP", BT_TARGET},
    {0, 0, NULL, NULL, "TO", BT_TARGET},
    {0, 0, NULL, NULL, "LINK", BT_TARGET},
    {0, 0, NULL, NULL, "FROM", BT_TARGET},
    {0, 0, NULL, NULL, "NUM_TEAMS", BT_TEAMS},
    {0, 0, NULL, NULL, "THREAD_LIMIT", BT_TEAMS},
    {0, 0, NULL, NULL, "DIST_SCHEDULE", BT_DISTRIBUTE},
    {0, 0, NULL, NULL, "PRIORITY", BT_TASKLOOP},
    {0, 0, NULL, NULL, "IS_DEVICE_PTR", BT_TARGET},
    {0, 0, NULL, NULL, "SIMD", BT_PDO | BT_PARDO | BT_SIMD},
    {0, 0, NULL, NULL, "THREADS", BT_TARGET},
    {0, 0, NULL, NULL, "DEVICE_NUM", BT_ACCINITSHUTDOWN | BT_ACCSET},
    {0, 0, NULL, NULL, "DEFAULT_ASYNC", BT_ACCSET},
    {0, 0, NULL, NULL, "DECLARE", BT_ACCDECL},
    {0, 0, NULL, NULL, "PROC_BIND", BT_PAR | BT_PARDO},
    {0, 0, NULL, NULL, "NO_CREATE",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCDECL | BT_ACCENTERDATA | BT_ACCSERIAL},
    {0, 0, NULL, NULL, "ATTACH",
     BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG | BT_ACCENTERDATA |
         BT_ACCSERIAL},
    {0, 0, NULL, NULL, "DETACH", BT_ACCEXITDATA},
    {0, 0, NULL, NULL, "COMPARE",
     BT_ACCREG | BT_ACCKERNELS | BT_ACCPARALLEL | BT_ACCDATAREG |
         BT_ACCSCALARREG | BT_ACCSERIAL},
};

#define CL_PRESENT(d) cl[d].present
#define CL_VAL(d) cl[d].val
#define CL_NAME(d) cl[d].name
#define CL_STMT(d) cl[d].stmt
#define CL_FIRST(d) cl[d].first
#define CL_LAST(d) cl[d].last

struct savcl_tag { /* save clause table for combined construct */
  int present;
  BIGINT64 val;
  void *first;
  void *last;
};

static struct savcl_tag sav_cl[CL_MAXV];

/* used for combined constructs of:
 * target teams distribute parallel for simd where clauses
 * can be applied to applicable construct.
 */
#define SAVCL_PRESENT(d) sav_cl[d].present
#define SAVCL_VAL(d) sav_cl[d].val /* sptr in some cases */
#define SAVCL_FIRST(d) sav_cl[d].first
#define SAVCL_LAST(d) sav_cl[d].last

/* combined target/data constructs and also use fo if(xxx:) clause */
#define OMP_DEFAULT 0x0
#define OMP_TARGET 0x1
#define OMP_TARGETDATA 0x2
#define OMP_TARGETENTERDATA 0x4
#define OMP_TARGETEXITDATA 0x8
#define OMP_TARGETUPDATE 0x10
#define OMP_PARALLEL 0x20
#define OMP_TASK 0x40
#define OMP_TASKLOOP 0x80

static int recent_loop_clause = 0;

static int chunk;
static int distchunk;
static int mp_iftype;
static ISZ_T kernel_do_nest;
static LOGICAL has_team = FALSE;


static LOGICAL any_pflsr_private = FALSE;

#ifdef FLANG_SEMSMP_UNUSED
static void add_pragmasyms(int pragmatype, int pragmascope, ITEM *itemp, int);
#endif
static void add_pragma(int pragmatype, int pragmascope, int pragmaarg);

#define OPT_OMP_ATOMIC !XBIT(69,0x1000)

static int kernel_argnum;

/**
   \brief Semantic analysis for SMP statements.
   \param rednum   reduction number
   \param top      top of stack after reduction
 */
void
semsmp(int rednum, SST *top)
{
  int sptr, sptr1;
  int dtype;
  ITEM *itemp; /* Pointers to items */
  int doif;
  int prev_doif;
  int ast, arg, std;
  int opc;
  int clause;
  INT rhstop;
  int op, d, ctype, bind_type;
  int ditype, ditype2, ditype3, pr1, pr2;
  BIGINT64 bttype;
  BITMASK64 dimask, dinestmask;
  LOGICAL dignorenested;
  const char *dirname;
  const char *nmptr;
  REDUC *reducp;
  REDUC_SYM *reduc_symp;
  REDUC_SYM *reduc_symp_last;
  REDUC_SYM *reduc_symp_curr;
  SST *e1;

  switch (rednum) {
  /* ------------------------------------------------------------------ */
  /*
   *	<declare simd> ::= <declare simd begin> <opt par list>
   */
  case DECLARE_SIMD1:
    apply_nodepchk(gbl.lineno, 2);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<declare simd name> ::= |
   */
  case DECLARE_SIMD_NAME1:
    break;
  /*
   *	<declare simd name> ::= ( <id> )
   */
  case DECLARE_SIMD_NAME2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<declare target> ::= ( <ident list> ) |
   */
  case DECLARE_TARGET1:
    break;
  /*
   *	<declare target> ::= <par list>
   */
  case DECLARE_TARGET2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<smp stmt> ::= <mp begin> <mp stmt>
   */
  case SMP_STMT1:
    SST_ASTP(LHS, SST_ASTG(RHS(2)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<mp begin> ::=
   */
  case MP_BEGIN1:
    parstuff_init();
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<mp stmt> ::=	<par begin> <opt par list>  |
   */
  case MP_STMT1:
    clause_errchk(BT_PAR, "OMP PARALLEL");
    mp_create_bscope(0);
    DI_BPAR(sem.doif_depth) = emit_bpar();
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endparallel>            |
   */
  case MP_STMT2:
    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_PAR, TRUE, 0);
    --sem.parallel;
    par_pop_scope();
    ast = emit_epar();
    mp_create_escope();
    if (doif) {
      A_LOPP(DI_BPAR(doif), ast);
      A_LOPP(ast, DI_BPAR(doif));
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp critical> <opt csident> |
   */
  case MP_STMT3:
    ast = 0;
    doif = enter_dir(DI_CRITICAL, FALSE, 0, DI_B(DI_ATOMIC_CAPTURE));
    sptr = 0;
    if (SST_IDG(RHS(2))) {
      check_crit(scn.id.name + SST_SYMG(RHS(2)));
      sptr = get_csect_sym(scn.id.name + SST_SYMG(RHS(2)));
      DI_CRITSYM(sem.doif_depth) = sptr;
    } else {
      check_crit(NULL);
      DI_CRITSYM(sem.doif_depth) = 0;
      sptr = get_csect_sym("unspc");
    }
    if (doif && sptr) {
      /*can't call emit_bcs_ecs - it checks for nested critical sections*/
      ast = mk_stmt(A_MP_CRITICAL, 0);
      DI_BEGINP(doif) = ast;
      if (!XBIT(69, 0x100))
        A_MEMP(ast, CMEMFG(sptr));
      else if (CMEMFG(sptr) != CMEMLG(sptr))
        A_MEMP(ast, SYMLKG(CMEMFG(sptr)));
      else
        A_MEMP(ast, CMEMFG(sptr));
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endcritical> <opt csident> |
   */
  case MP_STMT4:
    ast = 0;
    prev_doif = sem.doif_depth;
    doif = leave_dir(DI_CRITICAL, FALSE, 0);
    if (SST_IDG(RHS(2)))
      nmptr = scn.id.name + SST_SYMG(RHS(2));
    else
      nmptr = NULL;
    sptr = 0;
    if (DI_ID(prev_doif) == DI_CRITICAL) {
      sptr = DI_CRITSYM(prev_doif);
      if (sptr) {
        if (nmptr == NULL)
          error(155, 3, gbl.lineno,
                "CRITICAL is named, matching ENDCRITICAL is not -",
                SYMNAME(sptr) + get_csect_pfxlen());
        else if (strcmp(nmptr, SYMNAME(sptr) + get_csect_pfxlen()) != 0)
          error(155, 3, gbl.lineno,
                "CRITICAL and ENDCRITICAL names must be the same -", nmptr);
      } else if (nmptr != NULL)
        error(155, 3, gbl.lineno,
              "ENDCRITICAL is named, matching CRITICAL is not -", nmptr);
      else
        sptr = get_csect_sym("unspc");
    }
    if (doif && sptr) {
      ast = mk_stmt(A_MP_ENDCRITICAL, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
      if (!XBIT(69, 0x100))
        A_MEMP(ast, CMEMFG(sptr));
      else if (CMEMFG(sptr) != CMEMLG(sptr))
        A_MEMP(ast, SYMLKG(CMEMFG(sptr)));
      else
        A_MEMP(ast, CMEMFG(sptr));
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <single begin> <opt par list>  |
   */
  case MP_STMT5:
    ast = 0;
    clause_errchk(BT_SINGLE, "OMP SINGLE");
    doif = SST_CVALG(RHS(1));
    if (doif) {
      ast = mk_stmt(A_MP_SINGLE, 0);
      DI_BEGINP(doif) = ast;
      (void)add_stmt(ast);
      ast = 0;
    }
    par_push_scope(TRUE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, ast);
    break;
  /*
   *      <mp stmt> ::= <mp endsingle> <opt endsingle list> |
   */
  case MP_STMT6:
    ast = 0;
    end_parallel_clause(sem.doif_depth);
    doif = leave_dir(DI_SINGLE, TRUE, 2);
    if (doif) {
      ast = mk_stmt(A_MP_ENDSINGLE, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
      if (CL_PRESENT(CL_NOWAIT) && CL_PRESENT(CL_COPYPRIVATE)) {
        error(155, 3, gbl.lineno,
              "NOWAIT and COPYPRIVATE are mutually exclusive", NULL);
      }

      if (CL_PRESENT(CL_COPYPRIVATE)) {
        for (itemp = (ITEM *)CL_FIRST(CL_COPYPRIVATE); itemp != ITEM_END;
             itemp = itemp->next) {
          sptr = itemp->t.sptr;
          if (sptr == 0)
            continue;
          if (STYPEG(sptr) == ST_CMBLK) {
            if (CMEMFG(sptr) == 0) {
              error(38, 3, gbl.lineno, SYMNAME(sptr), NULL);
            }
          } else if (!DCLDG(sptr)) {
            error(38, 3, gbl.lineno, SYMNAME(sptr), NULL);
          }

        }
      }

      /* Handle if no wait is not set, which means we wait... barrier */
      if (!CL_PRESENT(CL_NOWAIT)) {
        (void)add_stmt(ast);
        if (CL_PRESENT(CL_COPYPRIVATE)) /* kmpc copypriv will barrier for us */
          ast = 0;
        else
          ast = mk_stmt(A_MP_BARRIER, 0);
      }

      do_copyprivate();
    }
    par_pop_scope();
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <pdo begin> <opt par list>  |
   */
  case MP_STMT7:
    clause_errchk(BT_PDO, "OMP DO");
    do_schedule(SST_CVALG(RHS(1)));
    sem.expect_do = TRUE;
    get_stblk_uplevel_sptr();
    par_push_scope(TRUE);
    get_stblk_uplevel_sptr();
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endpdo> <opt nowait>    |
   */
  case MP_STMT8:
    ast = 0;
    doif = leave_dir(DI_PDO, FALSE, 0);
    if (doif) {
      if (!CL_PRESENT(CL_NOWAIT)) {
        ast = mk_stmt(A_MP_BARRIER, 0);
      } else {
        /* check if cancel construct is present */
        ast = DI_BEGINP(doif);
        if (A_ENDLABG(ast)) {
          error(155, 3, gbl.lineno,
                "OMP DO that is canceled must not have an nowait clause", NULL);
        }
      }
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp barrier>                |
   */
  case MP_STMT9:
    ast = 0;
    check_barrier();
    ast = mk_stmt(A_MP_BARRIER, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp master>	            |
   */
  case MP_STMT10:
    ast = 0;
    doif = enter_dir(DI_MASTER, TRUE, 1,
                     DI_B(DI_PDO) | DI_B(DI_PARDO) | DI_B(DI_DOACROSS) |
                         DI_B(DI_PARSECTS) | DI_B(DI_SECTS) | DI_B(DI_SINGLE) |
                         DI_B(DI_TASK) | DI_B(DI_ATOMIC_CAPTURE) |
                         DI_B((DI_SIMD | DI_PDO)) | DI_B((DI_PARDO | DI_SIMD)));
    if (doif) {
      ast = mk_stmt(A_MP_MASTER, 0);
      DI_BEGINP(doif) = ast;
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endmaster>              |
   */
  case MP_STMT11:
    ast = 0;
    doif = leave_dir(DI_MASTER, TRUE, 1);
    if (doif) {
      ast = mk_stmt(A_MP_ENDMASTER, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp atomic begin> <opt atomic type>
   */
  case MP_STMT12:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <doacross begin> <opt par list>   |
   */
  case MP_STMT13:
    clause_errchk(BT_DOACROSS, "SMP DOACROSS");
    do_schedule(SST_CVALG(RHS(1)));
    sem.expect_do = TRUE;
    mp_create_bscope(0);
    DI_BPAR(sem.doif_depth) = emit_bpar();
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <paralleldo begin> <opt par list> |
   */
  case MP_STMT14:
    clause_errchk(BT_PARDO, "OMP PARALLEL DO");
    do_schedule(SST_CVALG(RHS(1)));
    sem.expect_do = TRUE;
    mp_create_bscope(0);
    DI_BPAR(sem.doif_depth) = emit_bpar();
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endpardo> |
   */
  case MP_STMT15:
    (void)leave_dir(DI_PARDO, FALSE, 0);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <parallelsections begin> <opt par list> |
   */
  case MP_STMT16:
    ast = 0;
    clause_errchk(BT_PARSECTS, "OMP PARALLEL SECTIONS");
    doif = SST_CVALG(RHS(1));
    mp_create_bscope(0);
    DI_BPAR(sem.doif_depth) = emit_bpar();
    par_push_scope(FALSE);
    if (doif && sem.parallel <= 1) {
      /* only distribute the work if in the outermost
       * parallel region or not in a parallel region.
       */
      DI_SECT_CNT(doif) = 0;
      ast = mk_stmt(A_MP_SECTIONS, 0);
      A_ENDLABP(ast, 0);
      DI_BEGINP(doif) = ast;
      (void)add_stmt(ast);
      begin_parallel_clause(sem.doif_depth);
      if (DI_LASTPRIVATE(doif)) {
        sptr = get_itemp(DT_INT4);
        ENCLFUNCP(sptr, BLK_SYM(sem.scope_level));
        DI_SECT_VAR(doif) = sptr;
        assign_cval(sptr, -1, DT_INT4);
      }

      /* implied section - empty if there is no code */
      ast = mk_stmt(A_MP_SECTION, 0);
      (void)add_stmt(ast);
      if (DI_LASTPRIVATE(doif)) {
        sptr = get_itemp(DT_INT4);
        ENCLFUNCP(sptr, BLK_SYM(sem.scope_level));
        DI_SECT_VAR(doif) = sptr;
        assign_cval(sptr, DI_SECT_CNT(doif), DT_INT4);
      }

      ast = 0;
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endparsections> |
   */
  case MP_STMT17:
    prev_doif = sem.doif_depth;
    doif = leave_dir(DI_PARSECTS, TRUE, 0);
    if (doif && sem.parallel <= 1) {
      /* create fake section */
      ast = mk_stmt(A_MP_LSECTION, 0);
      (void)add_stmt(ast);
      end_parallel_clause(prev_doif);
      /* only distribute the work if in the outermost
       * parallel region or not in a parallel region.
       */

      ast = mk_stmt(A_MP_ENDSECTIONS, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
      (void)add_stmt(ast);
    }
    --sem.parallel;
    par_pop_scope();
    ast = emit_epar();
    mp_create_escope();
    A_LOPP(DI_BPAR(prev_doif), ast);
    A_LOPP(ast, DI_BPAR(prev_doif));
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <sections begin> <opt par list> |
   */
  case MP_STMT18:
    ast = 0;
    clause_errchk(BT_SECTS, "OMP SECTION");
    doif = SST_CVALG(RHS(1));
    par_push_scope(TRUE);
    if (doif && sem.parallel <= 1) {
      /* only distribute the work if in the outermost
       * parallel region or not in a parallel region.
       */
      DI_SECT_CNT(doif) = 0;
      ast = mk_stmt(A_MP_SECTIONS, 0);
      A_ENDLABP(ast, 0);
      DI_BEGINP(doif) = ast;
      (void)add_stmt(ast);
      begin_parallel_clause(sem.doif_depth);
      ast = 0;

      if (DI_LASTPRIVATE(doif)) {
        sptr = get_itemp(DT_INT4);
        ENCLFUNCP(sptr, BLK_SYM(sem.scope_level));
        DI_SECT_VAR(doif) = sptr;
        assign_cval(sptr, -1, DT_INT4);
      }
      DI_SECT_CNT(sem.doif_depth)++;
      ast = mk_stmt(A_MP_SECTION, 0);
      (void)add_stmt(ast);
      if (DI_LASTPRIVATE(doif)) {
        assign_cval(sptr, DI_SECT_CNT(doif), DT_INT4);
      }
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp section> |
   */
  case MP_STMT19:
    ast = 0;
    if (DI_ID(sem.doif_depth) != DI_SECTS &&
        DI_ID(sem.doif_depth) != DI_PARSECTS) {
      error(155, 3, gbl.lineno, "Illegal context for SECTION", NULL);
      SST_ASTP(LHS, 0);
      break;
    }
    if (sem.parallel <= 1) {
      /* only distribute the work if in the outermost
       * parallel region or not in a parallel region.
       */
      DI_SECT_CNT(sem.doif_depth)++;
      ast = mk_stmt(A_MP_SECTION, 0);
      (void)add_stmt(ast);
      if (DI_LASTPRIVATE(sem.doif_depth)) {
        assign_cval(DI_SECT_VAR(sem.doif_depth), DI_SECT_CNT(sem.doif_depth),
                    DT_INT);
      }
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endsections> <opt nowait> |
   */
  case MP_STMT20:
    ast = 0;
    prev_doif = sem.doif_depth;
    doif = leave_dir(DI_SECTS, FALSE, 0);
    if (doif && sem.parallel <= 1) {
      /* create fake section */
      ast = mk_stmt(A_MP_LSECTION, 0);
      (void)add_stmt(ast);
      end_parallel_clause(prev_doif);
      /* only distribute the work if in the outermost
       * parallel region or not in a parallel region.
       */
      ast = mk_stmt(A_MP_ENDSECTIONS, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
      (void)add_stmt(ast);
    }
    if (doif && sem.parallel <= 1) {
      /* only distribute the work if in the outermost
       * parallel region or not in a parallel region.
       */
      if (!CL_PRESENT(CL_NOWAIT)) {
        ast = mk_stmt(A_MP_BARRIER, 0);
        (void)add_stmt(ast);
      } else {
        /* check if cancel construct is present */
        ast = DI_BEGINP(doif);
        if (A_ENDLABG(ast)) {
          error(155, 3, gbl.lineno,
                "SECTIONS construct that is canceled must "
                "not have an nowait clause",
                NULL);
        }
      }
    }
    par_pop_scope();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp flush> |
   */
  case MP_STMT21:
    ast = mk_stmt(A_MP_FLUSH, 0);
    (void)add_stmt(ast);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp flush> ( <ident list> ) |
   */
  case MP_STMT22:
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      sptr = refsym(itemp->t.sptr, OC_OTHER);
      VOLP(sptr, 1);
    }
    ast = mk_stmt(A_MP_FLUSH, 0);
    (void)add_stmt(ast);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp ordered> <opt ordered list> |
   */
  case MP_STMT23:
    ast = 0;
    doif = enter_dir(DI_ORDERED, TRUE, 3,
                     DI_B(DI_DOACROSS) | DI_B(DI_PARSECTS) | DI_B(DI_SECTS) |
                         DI_B(DI_SINGLE) | DI_B(DI_CRITICAL) | DI_B(DI_MASTER) |
                         DI_B(DI_TASK) | DI_B(DI_ATOMIC_CAPTURE));
    if (doif) {
      nmptr = "_mp_orders_begx";
      while (--doif) {
        if (DI_ID(doif) == DI_PDO || DI_ID(doif) == DI_PARDO) {
          if (DI_IS_ORDERED(doif)) {
            sptr = DI_DOINFO(doif + 1)->index_var;
            if (size_of(DTYPEG(sptr)) < 8)
              nmptr = "_mp_orders_beg";
            else
              nmptr = "_mp_orders_beg8";
          } else
            error(155, 3, DI_LINENO(doif),
                  "DO must have the ORDERED clause specified", NULL);
          break;
        }
      }
      ast = mk_stmt(A_MP_BORDERED, 0);
      (void)add_stmt(ast);
      ast = 0;

      if (sem.parallel && doif == 0) {
        /* DO directive not present */
        error(155, 3, gbl.lineno, "Illegal context for", "ORDERED");
      }
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endordered>
   */
  case MP_STMT24:
    ast = 0;
    doif = leave_dir(DI_ORDERED, TRUE, 3);
    if (doif) {
      nmptr = "_mp_orders_endx";
      while (--doif) {
        if (DI_ID(doif) == DI_PDO || DI_ID(doif) == DI_PARDO) {
          if (DI_IS_ORDERED(doif)) {
            sptr = DI_DOINFO(doif + 1)->index_var;
            if (size_of(DTYPEG(sptr)) < 8)
              nmptr = "_mp_orders_end";
            else
              nmptr = "_mp_orders_end8";
          }
          break;
        }
      }
      ast = mk_stmt(A_MP_EORDERED, 0);
      (void)add_stmt(ast);
      ast = 0;
    }
    SST_ASTP(LHS, ast);
    break;

  /*
   *      <mp stmt> ::= <mp workshare> |
   */
  case MP_STMT25:
    ast = 0;
    mp_create_bscope(0);
    doif = enter_dir(DI_WORKSHARE, FALSE, 1,
                     DI_B(DI_PDO) | DI_B(DI_PARDO) | DI_B(DI_DOACROSS) |
                         DI_B(DI_SECTS) | DI_B(DI_SINGLE) | DI_B(DI_MASTER) |
                         DI_B(DI_ORDERED) | DI_B(DI_ATOMIC_CAPTURE) |
                         DI_B((DI_SIMD | DI_PDO)));
    SST_CVALP(LHS, doif);
    par_push_scope(TRUE);
    if (doif) {
      DI_SECT_CNT(doif) = 0;
    }
    ast = mk_stmt(A_MP_WORKSHARE, 0);
    DI_BEGINP(doif) = ast;
    SST_ASTP(LHS, ast);

    break;
  /*
   *      <mp stmt> ::= <mp endworkshare> <opt nowait> |
   */
  case MP_STMT26:
    ast = 0;
    prev_doif = sem.doif_depth;
    doif = leave_dir(DI_WORKSHARE, FALSE, 1);
    if (doif) {
      ast = mk_stmt(A_MP_ENDWORKSHARE, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
      (void)add_stmt(ast);
      end_workshare(A_STDG(DI_BEGINP(doif)), A_STDG(ast));
    }
    if (doif && sem.parallel <= 1) {
      /* only distribute the work if in the outermost
       * parallel region or not in a parallel region.
       */
      if (CL_PRESENT(CL_NOWAIT)) {
        ast = mk_stmt(A_MP_BARRIER, 0);
        (void)add_stmt(ast);
      }
    }
    par_pop_scope();
    mp_create_escope();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <parworkshare begin> <opt par list> |
   */
  case MP_STMT27:
    ast = 0;
    clause_errchk(BT_PARWORKS, "OMP WORKSHARE");
    mp_create_bscope(0);
    DI_BPAR(sem.doif_depth) = emit_bpar();
    par_push_scope(FALSE);
    begin_parallel_clause(doif = sem.doif_depth);
    if (doif) {
      DI_SECT_CNT(doif) = 0;
    }
    ast = mk_stmt(A_MP_WORKSHARE, 0);
    DI_BEGINP(doif) = ast;
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endparworkshare> |
   */
  case MP_STMT28:
    doif = sem.doif_depth;
    if (doif) {
      ast = mk_stmt(A_MP_ENDWORKSHARE, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
      (void)add_stmt(ast);
      end_workshare(A_STDG(DI_BEGINP(doif)), A_STDG(ast));
    }
    end_parallel_clause(doif);
    (void)leave_dir(DI_PARWORKS, TRUE, 0);
    --sem.parallel;
    par_pop_scope();
    ast = emit_epar();
    mp_create_escope();
    A_LOPP(DI_BPAR(doif), ast);
    A_LOPP(ast, DI_BPAR(doif));
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <task begin> <opt par list> |
   */
  case MP_STMT29:
    ast = 0;
    clause_errchk(BT_TASK, "OMP TASK");
    doif = SST_CVALG(RHS(1));
    mp_create_bscope(0);
    if (doif) {
      ast = mk_stmt(A_MP_TASK, 0);
      A_ENDLABP(ast, 0);
      DI_BEGINP(doif) = ast;
      if (CL_PRESENT(CL_UNTIED)) {
        A_UNTIEDP(ast, 1);
      }
      if (CL_PRESENT(CL_IF)) {
        if (mp_iftype != IF_DEFAULT && mp_iftype != IF_TASK)
          error(155, 3, gbl.lineno,
                "IF (task:) or IF is expected in TASK construct ", NULL);
        else
          A_IFPARP(ast, CL_VAL(CL_IF));
      }
      if (CL_PRESENT(CL_FINAL)) {
        A_FINALPARP(ast, CL_VAL(CL_FINAL));
      }
      if (CL_PRESENT(CL_MERGEABLE)) {
        A_MERGEABLEP(ast, 1);
      }
      if (sem.parallel) {
        /*
         * Task is within a parallel region.
         */
        if (CL_PRESENT(CL_DEFAULT) && CL_VAL(CL_DEFAULT) == PAR_SCOPE_SHARED) {
          A_EXEIMMP(ast, 1);
        } else if (CL_PRESENT(CL_SHARED)) {
          /*
          if (any_pflsr_private)
              A_EXEIMMP(ast, 1);
          */
          /* Any SHARED privates?? */
          for (itemp = CL_FIRST(CL_SHARED); itemp != ITEM_END;
               itemp = itemp->next) {
            sptr = itemp->t.sptr;
            if (STYPEG(sptr) != SC_CMBLK && SCG(sptr) == SC_PRIVATE) {
              A_EXEIMMP(ast, 1);
              break;
            }
          }
        }
      }
      (void)add_stmt(ast);
      sem.task++;
    }
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
    if (doif) {
      ast = mk_stmt(A_MP_TASKREG, 0);
      (void)add_stmt(ast);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endtask> |
   */
  case MP_STMT30:
    end_parallel_clause(sem.doif_depth);
    doif = leave_dir(DI_TASK, FALSE, 1);
    if (doif) {
      sem.task--;
      par_pop_scope();
      ast = mk_stmt(A_MP_ENDTASK, 0);
      A_LOPP(DI_BEGINP(doif), ast);
      A_LOPP(ast, DI_BEGINP(doif));
      (void)add_stmt(ast);
      mp_create_escope();
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp taskwait>
   */
  case MP_STMT31:
    ast = mk_stmt(A_MP_TASKWAIT, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp taskyield>
   */
  case MP_STMT32:
    ast = mk_stmt(A_MP_TASKYIELD, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endatomic> |
   */
  case MP_STMT33:
    if (sem.mpaccatomic.action_type == ATOMIC_CAPTURE) {
      int ecs;
      if (use_opt_atomic(sem.doif_depth)) {
        ecs = mk_stmt(A_MP_ENDATOMIC, 0);
        std = add_stmt(ecs);
      } else {
        ecs = emit_bcs_ecs(A_MP_ENDCRITICAL);
        A_LOPP(ecs, sem.mpaccatomic.ast);
        A_LOPP(sem.mpaccatomic.ast, ecs);
      }
      sem.mpaccatomic.ast = 0;
      leave_dir(DI_ATOMIC_CAPTURE, FALSE, 1);
    } else if (sem.mpaccatomic.accassignc > 1) {
      error(155, 3, gbl.lineno, "Too many statements in ATOMIC CONSTRUCT",
            NULL);
    }
    sem.mpaccatomic.accassignc = 0;
    sem.mpaccatomic.seen = FALSE;
    sem.mpaccatomic.apply = FALSE;
    sem.mpaccatomic.pending = FALSE;
    sem.mpaccatomic.action_type = ATOMIC_UNDEF;
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <taskloop begin> <opt par list> |
   */
  case MP_STMT34:
  share_taskloop:
    ast = 0;
    clause_errchk(BT_TASKLOOP, "OMP TASKLOOP");
    doif = SST_CVALG(RHS(1));
    mp_create_bscope(0);
    if (doif) {
      do_schedule(doif);
      ast = mk_stmt(A_MP_TASKLOOP, 0);
      A_ENDLABP(ast, 0);
      DI_BEGINP(doif) = ast;
      if (CL_PRESENT(CL_UNTIED)) {
        A_UNTIEDP(ast, 1);
      }
      if (CL_PRESENT(CL_IF)) {
        if (mp_iftype != IF_DEFAULT && mp_iftype != IF_TASK)
          error(155, 3, gbl.lineno,
                "IF (task:) or IF is expected in TASKLOOP construct ", NULL);
        else
          A_IFPARP(ast, CL_VAL(CL_IF));
      }
      if (CL_PRESENT(CL_FINAL)) {
        A_FINALPARP(ast, CL_VAL(CL_FINAL));
      }
      if (CL_PRESENT(CL_MERGEABLE)) {
        A_MERGEABLEP(ast, 1);
      }
      if (CL_PRESENT(CL_NOGROUP)) {
        A_NOGROUPP(ast, 1);
      }
      if (CL_PRESENT(CL_NUM_TASKS)) {
        A_NUM_TASKSP(ast, CL_VAL(CL_NUM_TASKS));
      } else if (CL_PRESENT(CL_GRAINSIZE)) {
        A_GRAINSIZEP(ast, CL_VAL(CL_GRAINSIZE));
      }
      if (CL_PRESENT(CL_PRIORITY)) {
        A_PRIORITYP(ast, CL_VAL(CL_PRIORITY));
      }
      if (sem.parallel) {
        /*
         * Task is within a parallel region.
         */
        if (CL_PRESENT(CL_DEFAULT) && CL_VAL(CL_DEFAULT) == PAR_SCOPE_SHARED) {
          ;
        } else if (CL_PRESENT(CL_SHARED)) {
          /* Any SHARED privates?? */
          for (itemp = CL_FIRST(CL_SHARED); itemp != ITEM_END;
               itemp = itemp->next) {
            sptr = itemp->t.sptr;
            if (STYPEG(sptr) != SC_CMBLK && SCG(sptr) == SC_PRIVATE) {
              A_EXEIMMP(ast, 1);
              break;
            }
          }
        }
      }
      (void)add_stmt(ast);
      sem.task++;
    }
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);
    sem.expect_do = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endtaskloop> |
   */
  case MP_STMT35:
    doif = leave_dir(DI_TASKLOOP, FALSE, 1);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <taskloopsimd begin> <opt par list> |
   */
  case MP_STMT36:
    apply_nodepchk(gbl.lineno, 1);
    goto share_taskloop;
    break;
  /*
   *	<mp stmt> ::= <mp endtaskloopsimd> |
   */
  case MP_STMT37:
    doif = leave_dir(DI_TASKLOOP, FALSE, 1);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp cancel> <id name> <opt par ifclause> |
   */
  case MP_STMT38:
    ctype = (cancel_type(scn.id.name + SST_CVALG(RHS(2))));
    d = check_cancel(ctype);
    if (d > 0) {
      ast = mk_stmt(A_MP_CANCEL, 0);
      add_stmt(ast);
      if (CL_PRESENT(CL_IF))
        A_IFPARP(ast, CL_VAL(CL_IF));
      A_LOPP(ast, d);
      if (A_ENDLABG(d)) {
        A_ENDLABP(ast, A_ENDLABG(d));
      } else {
        int lab = getlab();
        int astlab = mk_label(lab);
        A_ENDLABP(d, astlab);
        A_ENDLABP(ast, astlab);
      }
      A_CANCELKINDP(ast, ctype);
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <dosimd begin> <opt par list>  |
   */
  case MP_STMT39:
    clause_errchk((BT_SIMD | BT_PDO), "OMP DOSIMD");
    do_schedule(SST_CVALG(RHS(1)));
    sem.expect_do = TRUE;
    get_stblk_uplevel_sptr();
    par_push_scope(TRUE);
    get_stblk_uplevel_sptr();
    begin_parallel_clause(sem.doif_depth);
    DI_ISSIMD(sem.doif_depth) = TRUE;
    apply_nodepchk(gbl.lineno, 1);

    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp enddosimd> <opt nowait> |
   */
  case MP_STMT40:
    ast = 0;
    doif = leave_dir(DI_PDO, FALSE, 0);
    if (doif) {
      if (!CL_PRESENT(CL_NOWAIT)) {
        ast = mk_stmt(A_MP_BARRIER, 0);
      }
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <simd begin> <opt par list> |
   */
  case MP_STMT41:
    clause_errchk(BT_SIMD, "OMP SIMD");
    if (CL_PRESENT(CL_SAFELEN) || CL_PRESENT(CL_LINEAR) ||
        CL_PRESENT(CL_ALIGNED) || CL_PRESENT(CL_PRIVATE) ||
        CL_PRESENT(CL_LASTPRIVATE) || CL_PRESENT(CL_REDUCTION) ||
        CL_PRESENT(CL_COLLAPSE)) {
      errwarn((error_code_t)604);
      sem.expect_simd_do = FALSE;
      par_push_scope(TRUE);
      SST_ASTP(LHS, 0);
      break;
    }
    sem.expect_simd_do = TRUE;
    par_push_scope(TRUE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);
    apply_nodepchk(gbl.lineno, 1);
    break;
  /*
   *	<mp stmt> ::= <mp endsimd> |
   */
  case MP_STMT42:
    ast = 0;
    end_parallel_clause(doif = sem.doif_depth);
    doif = leave_dir(DI_SIMD, FALSE, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <targetdata begin> <opt par list>  |
   */
  case MP_STMT43: {
    check_targetdata(OMP_TARGETDATA, "OMP TARGET DATA");
    doif = SST_CVALG(RHS(1));
    ast = mk_stmt(A_MP_TARGETDATA, 0);
    if (CL_PRESENT(CL_IF)) {
      if (mp_iftype != OMP_DEFAULT && mp_iftype != OMP_TARGETDATA)
        error(155, 3, gbl.lineno,
              "IF (target data:) or IF is expected in TARGET DATA construct ",
              NULL);
      else
        A_IFPARP(ast, CL_VAL(CL_IF));
      mp_iftype = IF_DEFAULT;
    }
    if (doif) {
      DI_BTARGET(doif) = ast;
    }
    add_stmt(ast);
  }
    SST_ASTP(LHS, 0);
    do_map();
    break;
  /*
   *	<mp stmt> ::= <mp endtargetdata> |
   */
  case MP_STMT44: {
    doif = leave_dir(DI_TARGETDATA, TRUE, 0);
    ast = mk_stmt(A_MP_ENDTARGETDATA, 0);
    if (CL_PRESENT(CL_IF)) {
      A_IFEXPRP(ast, CL_VAL(CL_IF));
    }
    if (doif) {
      A_LOPP(DI_BTARGET(doif), ast);
      A_LOPP(ast, DI_BTARGET(doif));
    }
    add_stmt(ast);
  }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targetenterdata begin> <opt par list>  |
   */
  case MP_STMT45: {
    check_targetdata(OMP_TARGETENTERDATA, "OMP TARGET ENTER DATA");
    ast = mk_stmt(A_MP_TARGETENTERDATA, 0);
    if (CL_PRESENT(CL_IF)) {
      if (mp_iftype != OMP_DEFAULT && mp_iftype != OMP_TARGETENTERDATA)
        error(155, 3, gbl.lineno,
              "IF (target enter data:) or IF is expected "
              "in TARGET ENTER DATA construct ",
              NULL);
      else
        A_IFPARP(ast, CL_VAL(CL_IF));
      mp_iftype = IF_DEFAULT;
    }
    if (CL_PRESENT(CL_DEPEND)) {
    }
    if (CL_PRESENT(CL_NOWAIT)) {
    }
    add_stmt(ast);
    (void)leave_dir(DI_TARGETENTERDATA, TRUE, 0);
  }
    SST_ASTP(LHS, 0);
    do_map();
    break;
  /*
   *	<mp stmt> ::= <targetexitdata begin> <opt par list>  |
   */
  case MP_STMT46: {
    check_targetdata(OMP_TARGETEXITDATA, "OMP TARGET EXIT DATA");
    ast = mk_stmt(A_MP_TARGETEXITDATA, 0);
    if (CL_PRESENT(CL_IF)) {
      if (mp_iftype != IF_DEFAULT && mp_iftype != IF_TARGETEXITDATA)
        error(155, 3, gbl.lineno,
              "IF (target exit data:) or IF is expected in "
              "TARGET EXIT DATA construct ",
              NULL);
      else
        A_IFPARP(ast, CL_VAL(CL_IF));
      mp_iftype = IF_DEFAULT;
    }
    if (CL_PRESENT(CL_DEPEND)) {
    }
    if (CL_PRESENT(CL_NOWAIT)) {
    }
    add_stmt(ast);
    (void)leave_dir(DI_TARGETEXITDATA, TRUE, 0);
  }
    SST_ASTP(LHS, 0);
    do_map();
    break;
  /*
   *	<mp stmt> ::= <targetupdate begin> <opt par list> |
   */
  case MP_STMT47: {
    check_targetdata(OMP_TARGETUPDATE, "OMP TARGET UPDATE");
    ast = mk_stmt(A_MP_TARGETUPDATE, 0);
    if (CL_PRESENT(CL_IF)) {
      if (mp_iftype != IF_DEFAULT && mp_iftype != IF_TARGETUPDATE)
        error(
            155, 3, gbl.lineno,
            "IF (target update:) or IF is expected in TARGET UPDATE construct ",
            NULL);
      else
        A_IFPARP(ast, CL_VAL(CL_IF));
      mp_iftype = IF_DEFAULT;
    }
    if (CL_PRESENT(CL_DEPEND)) {
    }
    if (CL_PRESENT(CL_NOWAIT)) {
    }
    add_stmt(ast);
    (void)leave_dir(DI_TARGETUPDATE, TRUE, 0);
  }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <target begin> <opt par list> |
   */
  case MP_STMT48:
    clause_errchk(BT_TARGET, "OMP TARGET");
    mp_create_bscope(0);
    DI_BTARGET(sem.doif_depth) = emit_btarget(A_MP_TARGET);
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if(flg.omptarget)
      A_COMBINEDTYPEP(DI_BTARGET(sem.doif_depth),
                    get_omp_combined_mode(BT_TARGET));
#endif
    par_push_scope(TRUE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endtarget> |
   */
  case MP_STMT49:
    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_TARGET, TRUE, 0);
    sem.target--;
    par_pop_scope();
    ast = emit_etarget();
    mp_create_escope();
    if (doif) {
      A_LOPP(DI_BTARGET(doif), ast);
      A_LOPP(ast, DI_BTARGET(doif));
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <teams begin> <opt par list> |
   */
  case MP_STMT50:
    ast = 0;
    clause_errchk(BT_TEAMS, "OMP_TEAMS");
    doif = SST_CVALG(RHS(1));
    do_bteams(doif);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endteams> |
   */
  case MP_STMT51:
    ast = 0;
    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_TEAMS, TRUE, 0);
    --sem.teams;
    par_pop_scope();
    mp_create_escope();
    if (doif) {
      ast = mk_stmt(A_MP_ENDTEAMS, 0);
      A_LOPP(DI_BTEAMS(doif), ast);
      A_LOPP(ast, DI_BTEAMS(doif));
      add_stmt(ast);
      ast = 0;
    }
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <distribute begin> <opt par list> |
   */
  case MP_STMT52:
    clause_errchk(BT_DISTRIBUTE, "OMP DISTRIBUTE");
    doif = SST_CVALG(RHS(1));
    sem.expect_do = TRUE;
    do_bdistribute(doif, TRUE);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp enddistribute> |
   */
  case MP_STMT53:
    doif = leave_dir(DI_DISTRIBUTE, TRUE, 1);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <distsimd begin> <opt par list> |
   */
  case MP_STMT54:
    clause_errchk((BT_DISTRIBUTE | BT_SIMD), "OMP DISTRIBUTE SIMD");
    doif = SST_CVALG(RHS(1));
    sem.expect_do = TRUE;
    do_bdistribute(doif, TRUE);
    apply_nodepchk(gbl.lineno, 1);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp enddistsimd> |
   */
  case MP_STMT55:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <distpardo begin> <opt par list> |
   */
  case MP_STMT56:
    ast = 0;
    clause_errchk((BT_DISTRIBUTE | BT_PARDO), "OMP DISTRIBUTE PARALLE DO");
    begin_combine_constructs((BT_DISTRIBUTE | BT_PARDO));
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp enddistpardo> |
   */
  case MP_STMT57:
    doif = leave_dir(DI_DISTPARDO, TRUE, 1);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <distpardosimd begin> <opt par list> |
   */
  case MP_STMT58:
    ast = 0;
    clause_errchk((BT_DISTRIBUTE | BT_PARDO | BT_SIMD),
                  "OMP DISTRIBUTE PARALLE DO SIMD");
    begin_combine_constructs((BT_DISTRIBUTE | BT_PARDO | BT_SIMD));
    DI_ISSIMD(sem.doif_depth) = TRUE;
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp enddistpardosimd> |
   */
  case MP_STMT59:
    doif = leave_dir(DI_DISTPARDO, TRUE, 1);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <pardosimd begin> <opt par list> |
   */
  case MP_STMT60:
    clause_errchk((BT_PARDO | BT_SIMD), "OMP PARALLEL DO SIMD");
    do_schedule(SST_CVALG(RHS(1)));
    sem.expect_do = TRUE;
    mp_create_bscope(0);
    DI_BPAR(sem.doif_depth) = emit_bpar();
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
    DI_ISSIMD(sem.doif_depth) = TRUE;
    SST_ASTP(LHS, 0);
    apply_nodepchk(gbl.lineno, 1);
    break;
  /*
   *	<mp stmt> ::= <mp endpardosimd> |
   */
  case MP_STMT61:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targpar begin> <opt par list> |
   */
  case MP_STMT62:
    ast = 0;
    clause_errchk((BT_TARGET | BT_PAR), "OMP TARGET PARALLEL");
    begin_combine_constructs((BT_TARGET | BT_PAR));
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endtargpar> |
   */
  case MP_STMT63:
    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_PAR, TRUE, 0);
    --sem.parallel;
    par_pop_scope();
    ast = emit_epar();
    mp_create_escope();
    if (doif) {
      A_LOPP(DI_BPAR(doif), ast);
      A_LOPP(ast, DI_BPAR(doif));
    }

    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_TARGET, TRUE, 0);
    sem.target--;
    par_pop_scope();
    if (doif) {
      ast = emit_etarget();
      mp_create_escope();
      A_LOPP(DI_BTARGET(doif), ast);
      A_LOPP(ast, DI_BTARGET(doif));
    }

    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targpardo begin> <opt par list> |
   */
  case MP_STMT64:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TARGET | BT_PARDO), "OMP TARGET PARALLEL DO");
    begin_combine_constructs((BT_TARGET | BT_PARDO));
    sem.expect_do = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endtargpardo> |
   */
  case MP_STMT65:
    /* end target, parallel do is handled in do_end */
    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_TARGET, TRUE, 0);
    sem.target--;
    par_pop_scope();
    if (doif) {
      ast = emit_etarget();
      mp_create_escope();
      A_LOPP(DI_BTARGET(doif), ast);
      A_LOPP(ast, DI_BTARGET(doif));
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targparsimd begin> <opt par list> |
   */
  case MP_STMT66:
    SST_ASTP(LHS, 0);
    /* Don't think this construct exists */
    clause_errchk((BT_TARGET | BT_PAR | BT_SIMD), "OMP TARGET PARALLEL SIMD");
    break;
  /*
   *	<mp stmt> ::= <mp endtargparsimd> |
   */
  case MP_STMT67:
    /* this construct does not exist - remove it */
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targpardosimd begin> <opt par list> |
   */
  case MP_STMT68:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TARGET | BT_PARDO | BT_SIMD),
                  "OMP TARGET PARALLEL DO SIMD");
    begin_combine_constructs((BT_TARGET | BT_PARDO | BT_SIMD));
    sem.expect_do = TRUE;
    DI_ISSIMD(sem.doif_depth) = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endtargpardosimd> |
   */
  case MP_STMT69:
    /* end target, parallel do is handled in do_end */
    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_TARGET, TRUE, 0);
    sem.target--;
    par_pop_scope();
    if (doif) {
      ast = emit_etarget();
      mp_create_escope();
      A_LOPP(DI_BTARGET(doif), ast);
      A_LOPP(ast, DI_BTARGET(doif));
    }
    SST_ASTP(LHS, 0);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targsimd begin> <opt par list> |
   */
  case MP_STMT70:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TARGET | BT_SIMD), "OMP TARGET SIMD");
    mp_create_bscope(0);
    DI_BTARGET(sem.doif_depth) = emit_btarget(A_MP_TARGET);
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if(flg.omptarget)
      A_COMBINEDTYPEP(DI_BTARGET(sem.doif_depth),
                      get_omp_combined_mode(BT_TARGET | BT_SIMD));
#endif
    par_push_scope(TRUE);
    begin_parallel_clause(sem.doif_depth);
    SST_ASTP(LHS, 0);

    if (CL_PRESENT(CL_COLLAPSE)) {
      sem.collapse = CL_VAL(CL_COLLAPSE);
    }
    sem.expect_simd_do = TRUE;
    par_push_scope(TRUE);
    begin_parallel_clause(sem.doif_depth);
    apply_nodepchk(gbl.lineno, 1);
    SST_ASTP(LHS, 0);

    break;
  /*
   *	<mp stmt> ::= <mp endtargsimd> |
   */
  case MP_STMT71:
    end_parallel_clause(doif = sem.doif_depth);
    (void)leave_dir(DI_TARGET, TRUE, 0);
    sem.target--;
    par_pop_scope();
    if (doif) {
      ast = emit_etarget();
      mp_create_escope();
      A_LOPP(DI_BTARGET(doif), ast);
      A_LOPP(ast, DI_BTARGET(doif));
    }
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targteams begin> <opt par list> |
   */
  case MP_STMT72:
    clause_errchk((BT_TARGET | BT_TEAMS), "OMP TARGET TEAMS");
    begin_combine_constructs((BT_TARGET | BT_TEAMS));
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp endtargteams> |
   */
  case MP_STMT73:
    end_targteams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <teamsdist begin> <opt par list> |
   */
  case MP_STMT74:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TEAMS | BT_DISTRIBUTE), "OMP TEAMS DISTRIBUTE");
    begin_combine_constructs((BT_TEAMS | BT_DISTRIBUTE));
    sem.expect_do = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endteamsdist> |
   */
  case MP_STMT75:
    doif = leave_dir(DI_TEAMSDIST, TRUE, 1);
    end_teams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <teamsdistsimd begin> <opt par list> |
   */
  case MP_STMT76:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TEAMS | BT_DISTRIBUTE | BT_SIMD),
                  "OMP TEAMS DISTRIBUTE SIMD");
    begin_combine_constructs((BT_TEAMS | BT_DISTRIBUTE | BT_SIMD));
    sem.expect_do = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endteamsdistsimd> |
   */
  case MP_STMT77:
    doif = leave_dir(DI_TEAMSDIST, TRUE, 1);
    end_teams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targteamsdist begin> <opt par list> |
   */
  case MP_STMT78:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TARGET | BT_TEAMS | BT_DISTRIBUTE),
                  "OMP TARGET TEAMS DISTRIBUTE");
    begin_combine_constructs((BT_TARGET | BT_TEAMS | BT_DISTRIBUTE));
    sem.expect_do = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endtargteamsdist> |
   */
  case MP_STMT79:
    doif = leave_dir(DI_TARGTEAMSDIST, TRUE, 1);
    end_targteams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targteamsdistsimd begin> <opt par list> |
   */
  case MP_STMT80:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_SIMD),
                  "OMP TARGET TEAMS DISTRIBUTE SIMD");
    begin_combine_constructs((BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_SIMD));
    sem.expect_do = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endtargteamsdistsimd> |
   */
  case MP_STMT81:
    doif = leave_dir(DI_TARGTEAMSDIST, TRUE, 1);
    end_targteams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <teamsdistpardo begin> <opt par list> |
   */
  case MP_STMT82:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TEAMS | BT_DISTRIBUTE | BT_PARDO),
                  "OMP TEAMS DISTRIBUTE PARALLEL Do");
    begin_combine_constructs((BT_TEAMS | BT_DISTRIBUTE | BT_PARDO));
    break;
  /*
   *	<mp stmt> ::= <mp endteamsdistpardo> |
   */
  case MP_STMT83:
    doif = leave_dir(DI_TEAMSDISTPARDO, TRUE, 1);
    end_teams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targteamsdistpardo begin> <opt par list> |
   */
  case MP_STMT84:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_PARDO),
                  "OMP TARGET TEAMS DISTRIBUTE PARDO");
    begin_combine_constructs((BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_PARDO));
    break;
  /*
   *	<mp stmt> ::= <mp endtargteamsdistpardo> |
   */
  case MP_STMT85:
    doif = leave_dir(DI_TARGTEAMSDISTPARDO, TRUE, 1);
    end_targteams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <teamsdistpardosimd begin> <opt par list> |
   */
  case MP_STMT86:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TEAMS | BT_DISTRIBUTE | BT_PARDO | BT_SIMD),
                  "OMP TEAMS DISTRIBUTE PARALLEL DO SIMD");
    begin_combine_constructs((BT_TEAMS | BT_DISTRIBUTE | BT_PARDO | BT_SIMD));
    DI_ISSIMD(sem.doif_depth) = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endteamsdistpardosimd> |
   */
  case MP_STMT87:
    doif = leave_dir(DI_TEAMSDISTPARDO, TRUE, 1);
    end_teams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <targteamsdistpardosimd begin> <opt par list> |
   */
  case MP_STMT88:
    SST_ASTP(LHS, 0);
    clause_errchk((BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_PARDO | BT_SIMD),
                  "OMP TARGET TEAMS DISTRIBUTE PARDO SIMD");
    begin_combine_constructs(
        (BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_PARDO | BT_SIMD));
    doif = SST_CVALG(RHS(1));
    DI_ISSIMD(doif) = TRUE;
    break;
  /*
   *	<mp stmt> ::= <mp endtargteamsdistpardosimd> |
   */
  case MP_STMT89:
    doif = leave_dir(DI_TARGTEAMSDISTPARDO, TRUE, 1);
    end_targteams();
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<mp stmt> ::= <mp taskgroup> |
   */
  case MP_STMT90:
    ast = mk_stmt(A_MP_TASKGROUP, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp endtaskgroup> |
   */
  case MP_STMT91:
    ast = mk_stmt(A_MP_ETASKGROUP, 0);
    SST_ASTP(LHS, ast);
    break;
  /*
   *	<mp stmt> ::= <mp cancellationpoint> <id name>
   */
  case MP_STMT92:
    ctype = cancel_type(scn.id.name + SST_CVALG(RHS(2)));
    d = check_cancel(ctype);
    if (d > 0) {
      ast = mk_stmt(A_MP_CANCELLATIONPOINT, 0);
      add_stmt(ast);
      A_LOPP(ast, d);
      if (A_ENDLABG(d)) {
        A_ENDLABP(ast, A_ENDLABG(d));
      } else {
        int lab = getlab();
        int astlab = mk_label(lab);
        A_ENDLABP(d, astlab);
        A_ENDLABP(ast, astlab);
      }
      A_CANCELKINDP(ast, ctype);
    }
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt csident> ::= |
   */
  case OPT_CSIDENT1:
    SST_IDP(LHS, 0);
    break;
  /*
   *	<opt csident> ::= ( <id name> )
   */
  case OPT_CSIDENT2:
    SST_IDP(LHS, 1);
    SST_SYMP(LHS, SST_SYMG(RHS(2)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt nowait> ::=  |
   */
  case OPT_NOWAIT1:
    break;
  /*
   *	<opt nowait> ::= <opt comma> <nowait>
   */
  case OPT_NOWAIT2:
    break;

  /*
   *      <nowait> ::= NOWAIT
   */
  case NOWAIT1:
    add_clause(CL_NOWAIT, TRUE);
    break;

  /*
   *      <opt endsingle list> ::= |
   */
  case OPT_ENDSINGLE_LIST1:
    break;
  /*
   *	<opt endsingle list> ::= <opt comma> <endsingle list>
   */
  case OPT_ENDSINGLE_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<endsingle list> ::= <endsingle list> <opt comma> <endsingle item> |
   */
  case ENDSINGLE_LIST1:
    break;
  /*
   *	<endsingle list> ::= <endsingle item>
   */
  case ENDSINGLE_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <endsingle item> ::= <nowait> |
   */
  case ENDSINGLE_ITEM1:
    break;

  /*
   *      <endsingle item> ::= COPYPRIVATE ( <tp list> )
   */
  case ENDSINGLE_ITEM2:
    add_clause(CL_COPYPRIVATE, FALSE);
    if (CL_FIRST(CL_COPYPRIVATE) == NULL)
      CL_FIRST(CL_COPYPRIVATE) = SST_BEGG(RHS(3));
    else
      ((ITEM *)CL_LAST(CL_COPYPRIVATE))->next = SST_BEGG(RHS(3));
    CL_LAST(CL_COPYPRIVATE) = SST_ENDG(RHS(3));

    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<par begin> ::= <mp parallel>
   */
  case PAR_BEGIN1:
    doif = enter_dir(DI_PAR, FALSE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt par list> ::= |
   */
  case OPT_PAR_LIST1:
    break;
  /*
   *	<opt par list> ::= <opt comma> <par list>
   */
  case OPT_PAR_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<par list> ::= <par list> <opt comma> <par attr> |
   */
  case PAR_LIST1:
    break;
  /*
   *	<par list> ::= <par attr>
   */
  case PAR_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<par attr> ::= DEFAULT ( <id name> ) |
   */
  case PAR_ATTR1:
    add_clause(CL_DEFAULT, TRUE);
    nmptr = scn.id.name + SST_SYMG(RHS(3));
    if (strcmp(nmptr, "none") == 0)
      CL_VAL(CL_DEFAULT) = PAR_SCOPE_NONE;
    else if (strcmp(nmptr, "private") == 0)
      CL_VAL(CL_DEFAULT) = PAR_SCOPE_PRIVATE;
    else if (strcmp(nmptr, "shared") == 0)
      CL_VAL(CL_DEFAULT) = PAR_SCOPE_SHARED;
    else if (strcmp(nmptr, "firstprivate") == 0)
      CL_VAL(CL_DEFAULT) = PAR_SCOPE_FIRSTPRIVATE;
    else {
      error(34, 3, gbl.lineno, nmptr, CNULL);
      CL_VAL(CL_DEFAULT) = PAR_SCOPE_SHARED;
    }
    break;
  /*
   *	<par attr> ::= <private list>      |
   */
  case PAR_ATTR2:
    break;
  /*
   *	<par attr> ::= SHARED  ( <pflsr list> ) |
   */
  case PAR_ATTR3:
    add_clause(CL_SHARED, FALSE);
    cray_pointer_check(SST_BEGG(RHS(3)), CL_SHARED);
    if (CL_FIRST(CL_SHARED) == NULL)
      CL_FIRST(CL_SHARED) = SST_BEGG(RHS(3));
    else
      ((ITEM *)CL_LAST(CL_SHARED))->next = SST_BEGG(RHS(3));
    CL_LAST(CL_SHARED) = SST_ENDG(RHS(3));

    /* Check data sharing sanity */
    for (itemp = CL_FIRST(CL_SHARED); itemp != ITEM_END; itemp = itemp->next) {
      check_valid_data_sharing(itemp->t.sptr);
    }
    break;
  /*
   *	<par attr> ::= <firstprivate> |
   */
  case PAR_ATTR4:
    break;
  /*
   *	<par attr> ::= <lastprivate> |
   */
  case PAR_ATTR5:
    break;
  /*
   *	<par attr> ::= <schedule> |
   */
  case PAR_ATTR6:
    break;
  /*
   *	<par attr> ::= ORDERED <opt expression> |
   */
  case PAR_ATTR7:
    break;
  /*
   *	<par attr> ::= REDUCTION ( <reduction> ) |
   */
  case PAR_ATTR8:
    break;
  /*
   *	<par attr> ::= <par ifclause> |
   */
  case PAR_ATTR9:
    break;
  /*
   *	<par attr> ::= COPYIN ( <cmn ident list> ) |
   */
  case PAR_ATTR10:
    add_clause(CL_COPYIN, FALSE);
    if (CL_FIRST(CL_COPYIN) == NULL)
      CL_FIRST(CL_COPYIN) = SST_BEGG(RHS(3));
    else
      ((ITEM *)CL_LAST(CL_COPYIN))->next = SST_BEGG(RHS(3));
    CL_LAST(CL_COPYIN) = SST_ENDG(RHS(3));
    break;
  /*
   *	<par attr> ::= NUM_THREADS ( <expression> ) |
   */
  case PAR_ATTR11:
    add_clause(CL_NUM_THREADS, TRUE);
    chk_scalartyp(RHS((3)), DT_INT4, FALSE);
    CL_VAL(CL_NUM_THREADS) = SST_ASTG(RHS(3));
    break;
  /*
   *	<par attr> ::= COLLAPSE ( <expression> ) |
   */
  case PAR_ATTR12:
    if (SST_IDG(RHS(3)) == S_CONST) {
      add_clause(CL_COLLAPSE, TRUE);
      CL_VAL(CL_COLLAPSE) = chkcon_to_isz(RHS(3), TRUE);
      if (CL_VAL(CL_COLLAPSE) < 0) {
        CL_VAL(CL_COLLAPSE) = 0;
        error(155, 3, gbl.lineno,
              "The COLLAPSE expression must be a positive integer constant",
              CNULL);
      }
    } else {
      CL_VAL(CL_COLLAPSE) = 0;
      error(155, 3, gbl.lineno,
            "The COLLAPSE expression must be a positive integer constant",
            CNULL);
    }
    break;
  /*
   *	<par attr> ::= UNTIED |
   */
  case PAR_ATTR13:
    add_clause(CL_UNTIED, TRUE);
    break;

  /*
   *	<par attr> ::= FINAL
   */
  case PAR_ATTR14:
    add_clause(CL_FINAL, TRUE);
    chk_scalartyp(RHS((3)), DT_LOG4, FALSE);
    CL_VAL(CL_FINAL) = SST_ASTG(RHS(3));
    break;

  /*
   *	<par attr> ::= MERGEABLE |
   */
  case PAR_ATTR15:
    add_clause(CL_MERGEABLE, TRUE);
    break;
  /*
   *	<par attr> ::= PROC_BIND ( <id name> ) |
   */
  case PAR_ATTR16:
    bind_type = get_mp_bind_type(scn.id.name + SST_CVALG(RHS(3)));
    if (bind_type) {
      add_clause(CL_PROC_BIND, TRUE);
      CL_VAL(CL_PROC_BIND) = bind_type;
    }
    break;
  /*
   *	<par attr> ::= SAFELEN ( <expression> ) |
   */
  case PAR_ATTR17:
    error(547, ERR_Warning, gbl.lineno, "SAFELEN", CNULL);
    break;
  /*
   *	<par attr> ::= <linear clause> |
   */
  case PAR_ATTR18:
    error(547, ERR_Warning, gbl.lineno, "LINEAR", CNULL);
    break;
  /*
   *	<par attr> ::= <aligned clause> |
   */
  case PAR_ATTR19:
    error(547, ERR_Warning, gbl.lineno, "ALIGNED", CNULL);
    break;
  /*
   *	<par attr> ::= SIMDLEN ( <expression> ) |
   */
  case PAR_ATTR20:
    if (SST_IDG(RHS(3)) == S_CONST) {
      ISZ_T simdlen = chkcon_to_isz(RHS(3), TRUE);
      if (simdlen >= 0) {
        apply_simdlen(gbl.lineno, 1, simdlen);
        break;
      }
    }
    error(155, 3, gbl.lineno,
          "The SIMDLEN expression must be a positive integer constant",
          CNULL);
    break;
  /*
   *	<par attr> ::= <uniform clause> |
   */
  case PAR_ATTR21:
    error(547, ERR_Warning, gbl.lineno, "UNIFORM", CNULL);
    break;
  /*
   *	<par attr> ::= INBRANCH |
   */
  case PAR_ATTR22:
    error(547, ERR_Warning, gbl.lineno, "INBRANCH", CNULL);
    break;
  /*
   *	<par attr> ::= NOTINBRANCH |
   */
  case PAR_ATTR23:
    error(547, ERR_Warning, gbl.lineno, "NOINBRANCH", CNULL);
    break;
  /*
   *	<par attr> ::= LINK ( <ident list> ) |
   */
  case PAR_ATTR24:
    error(547, ERR_Warning, gbl.lineno, "LINK", CNULL);
    break;
  /*
   *	<par attr> ::= DEVICE ( <expression> ) |
   */
  case PAR_ATTR25:
    error(547, ERR_Warning, gbl.lineno, "DEVICE", CNULL);
    break;
    /*
     *	<par attr> ::= <map clause> |
     */
  case PAR_ATTR26:
    break;
    /*
     *	<par attr> ::= <depend clause> |
     */
  case PAR_ATTR27:
    error(547, ERR_Warning, gbl.lineno, "DEPEND", CNULL);
    break;
  /*
   *	<par attr> ::= IS_DEVICE_PTR ( <ident list> ) |
   */
  case PAR_ATTR28:
    error(547, ERR_Warning, gbl.lineno, "IS_DEVICE_PTR", CNULL);
    break;
  /*
   *	<par attr> ::= DEFAULTMAP ( <id name> : <id name> ) |
   */
  case PAR_ATTR29:
    error(547, ERR_Warning, gbl.lineno, "DEFAULTMAP", CNULL);
    break;
  /*
   *	<par attr> ::= <motion clause> |
   */
  case PAR_ATTR30:
    break;
  /*
   *	<par attr> ::= DIST_SCHEDULE ( <id name> <opt distchunk> ) |
   */
  case PAR_ATTR31:
    if (sched_type(scn.id.name + SST_CVALG(RHS(3))) != DI_SCH_STATIC) {
      error(155, 3, gbl.lineno,
            "Static scheduling is expected in dist_schedule", NULL);
    }
    add_clause(CL_DIST_SCHEDULE, TRUE);
    break;
  /*
   *	<par attr> ::= GRAINSIZE ( <expression> ) |
   */
  case PAR_ATTR32:
    if (CL_PRESENT(CL_NUM_TASKS)) {
      error(155, 3, gbl.lineno,
            "Grainsize and num_tasks cannot be present in same taskloop", NULL);
      break;
    } else if (CL_PRESENT(CL_GRAINSIZE)) {
      error(155, 3, gbl.lineno,
            "At most one grainsize can be present in taskloop", NULL);
      break;
    }
    add_clause(CL_GRAINSIZE, TRUE);
    chk_scalartyp(RHS((3)), DT_INT4, FALSE);
    CL_VAL(CL_GRAINSIZE) = SST_ASTG(RHS(3));
    break;
  /*
   *	<par attr> ::= NUM_TASKS ( <expression> ) |
   */
  case PAR_ATTR33:
    if (CL_PRESENT(CL_NUM_TASKS)) {
      error(155, 3, gbl.lineno,
            "At most one grainsize can be present in taskloop", NULL);
      break;
    } else if (CL_PRESENT(CL_GRAINSIZE)) {
      error(155, 3, gbl.lineno,
            "Grainsize and num_tasks cannot be present in same taskloop", NULL);
      break;
    }
    add_clause(CL_NUM_TASKS, TRUE);
    chk_scalartyp(RHS((3)), DT_INT4, FALSE);
    CL_VAL(CL_NUM_TASKS) = SST_ASTG(RHS(3));
    break;
  /*
   *	<par attr> ::= PRIORITY ( <expression> ) |
   */
  case PAR_ATTR34:
    if (CL_PRESENT(CL_PRIORITY)) {
      error(155, 3, gbl.lineno,
            "At most one priority can be present in taskloop", NULL);
    }
    add_clause(CL_PRIORITY, TRUE);
    chk_scalartyp(RHS((3)), DT_INT4, FALSE);
    CL_VAL(CL_PRIORITY) = SST_ASTG(RHS(3));
    break;
  /*
   *	<par attr> ::= NUM_TEAMS ( <expression> ) |
   */
  case PAR_ATTR35:
    add_clause(CL_NUM_TEAMS, TRUE);
    chk_scalartyp(RHS((3)), DT_INT4, FALSE);
    CL_VAL(CL_NUM_TEAMS) = SST_ASTG(RHS(3));
    break;
  /*
   *	<par attr> ::= THREAD_LIMIT( <expression> ) |
   */
  case PAR_ATTR36:
    add_clause(CL_THREAD_LIMIT, TRUE);
    chk_scalartyp(RHS((3)), DT_INT4, FALSE);
    CL_VAL(CL_THREAD_LIMIT) = SST_ASTG(RHS(3));
    break;
  /*
   *	<par attr> ::= NOGROUP
   */
  case PAR_ATTR37:
    add_clause(CL_NOGROUP, TRUE);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *    <opt expression> ::= |
   */
  case OPT_EXPRESSION1:
    add_clause(CL_ORDERED, TRUE);
    break;
  /*
   *    <opt expression> ::= ( <expression> )
   */
  case OPT_EXPRESSION2:
    add_clause(CL_ORDERED, TRUE);
    error(547, ERR_Warning, gbl.lineno, "ORDERED(n)", CNULL);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *    <opt ordered list> ::= |
   */
  case OPT_ORDERED_LIST1:
    break;
  /*
   *    <opt ordered list> ::= <ordered list>
   */
  case OPT_ORDERED_LIST2:
    break;
  /* ------------------------------------------------------------------ */
  /*
   *    <ordered list> ::= <ordered list> <opt comma> <ordered attr> |
   */
  case ORDERED_LIST1:
    break;
  /*
   *    <ordered list> ::= <ordered attr>
   */
  case ORDERED_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <ordered attr> ::=  SIMD |
   */
  case ORDERED_ATTR1:
    error(547, ERR_Warning, gbl.lineno, "SIMD", CNULL);
    break;
  /*
   *    <ordered attr> ::= THREADS |
   */
  case ORDERED_ATTR2:
    error(547, ERR_Warning, gbl.lineno, "THREAD", CNULL);
    break;
  /*
   *    <ordered attr> ::= DEPEND <depend attr>
   */
  case ORDERED_ATTR3:
    error(547, ERR_Warning, gbl.lineno, "DEPEND", CNULL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pflsr list> ::= <pflsr list> , <cmn ident> |
   */
  case PFLSR_LIST1:
    rhstop = 3;
    goto add_pflsr_to_list;
  /*
   *	<pflsr list> ::= <cmn ident>
   */
  case PFLSR_LIST2:
    rhstop = 1;
  add_pflsr_to_list:
    sptr = SST_SYMG(RHS(rhstop));
    if (STYPEG(sptr) != ST_CMBLK) {
      sptr = find_outer_sym(sptr);
      if (SCG(sptr) == SC_CMBLK && THREADG(CMBLKG(sptr)))
        error(155, 3, gbl.lineno,
              "A THREADPRIVATE common block member may "
              "only appear in the COPYIN clause -",
              SYMNAME(sptr));
      itemp = (ITEM *)getitem(0, sizeof(ITEM));
      itemp->next = ITEM_END;
      itemp->t.sptr = sptr;
      if (rhstop == 1)
        /* adding first item to list */
        SST_BEGP(LHS, itemp);
      else
        /* adding subsequent items to list */
        (SST_ENDG(RHS(1)))->next = itemp;
      SST_ENDP(LHS, itemp);
      if (SCG(sptr) == SC_PRIVATE)
        any_pflsr_private = TRUE;
    } else {
      ITEM *fitemp, *litemp;
      if (THREADG(sptr))
        error(155, 3, gbl.lineno,
              "A THREADPRIVATE common block may only "
              "appear in the COPYIN clause -",
              SYMNAME(sptr));
      fitemp = NULL;
      /*
       * Add all of the common block members to the 'item' list.
       * TBD - need to add any variables which are equivalenced
       *       to the members!!!
       */
      for (sptr1 = CMEMFG(sptr); sptr1 > NOSYM; sptr1 = SYMLKG(sptr1)) {
        itemp = (ITEM *)getitem(0, sizeof(ITEM));
        itemp->next = ITEM_END;
        itemp->t.sptr = find_outer_sym(sptr1);
        if (fitemp == NULL)
          fitemp = itemp;
        else
          litemp->next = itemp;
        litemp = itemp;
      }
      if (fitemp == NULL) {
        /* The common block is empty (error was reported by <cmn ident>.
         * If this is the first in the list, need to recover by creating
         * a symbol.
         */
        if (rhstop != 1)
          break;
        itemp = (ITEM *)getitem(0, sizeof(ITEM));
        itemp->next = ITEM_END;
        itemp->t.sptr = find_outer_sym(ref_ident(sptr));
        fitemp = itemp;
        litemp = itemp;
      }
      if (rhstop == 1)
        /* adding first item to list */
        SST_BEGP(LHS, fitemp);
      else
        /* adding subsequent items to list */
        (SST_ENDG(RHS(1)))->next = fitemp;
      SST_ENDP(LHS, litemp);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<private list> ::= PRIVATE ( <pflsr list> )
   */
  case PRIVATE_LIST1:
    clause = CL_PRIVATE;
    goto prepare_private_shared;

  /* ------------------------------------------------------------------ */
  /*
   *	<firstprivate> ::= FIRSTPRIVATE ( <pflsr list> )
   */
  case FIRSTPRIVATE1:
    clause = CL_FIRSTPRIVATE;
    other_firstlast_check(SST_BEGG(RHS(3)), clause);
  prepare_private_shared:
    add_clause(clause, FALSE);
    cray_pointer_check(SST_BEGG(RHS(3)), clause);
    if (CL_FIRST(clause) == NULL)
      CL_FIRST(clause) = SST_BEGG(RHS(3));
    else
      ((ITEM *)CL_LAST(clause))->next = SST_BEGG(RHS(3));
    CL_LAST(clause) = SST_ENDG(RHS(3));

    /* Check data sharing sanity */
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      check_valid_data_sharing(itemp->t.sptr);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<lastprivate> ::= LASTPRIVATE ( <pflsr list> )
   */
  case LASTPRIVATE1:
    add_clause(CL_LASTPRIVATE, FALSE);
    other_firstlast_check(SST_BEGG(RHS(3)), CL_LASTPRIVATE);
    cray_pointer_check(SST_BEGG(RHS(3)), CL_LASTPRIVATE);
    /*
     * create a fake REDUC_SYM item (from area 0 freed during the end of
     * statement processing.
     */
    reduc_symp = reduc_symp_last = (REDUC_SYM *)getitem(0, sizeof(REDUC_SYM));
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      REDUC_SYM *rsp;
      /*
       * Need to keep the REDUC_SYM items around until the end of the
       * parallel do, so allocate them in area 1.
       */
      rsp = (REDUC_SYM *)getitem(1, sizeof(REDUC_SYM));
      rsp->Private = 0;
      rsp->shared = itemp->t.sptr;
      rsp->next = NULL;
      reduc_symp_last->next = rsp;
      reduc_symp_last = rsp;
    }
    /* skip past the fake REDUC_SYM item */
    reduc_symp = reduc_symp->next;
    if (CL_FIRST(CL_LASTPRIVATE) == NULL)
      CL_FIRST(CL_LASTPRIVATE) = reduc_symp;
    else
      ((REDUC_SYM *)CL_LAST(CL_LASTPRIVATE))->next = reduc_symp;
    CL_LAST(CL_LASTPRIVATE) = reduc_symp;

    /* Check data sharing sanity */
    for (itemp = SST_BEGG(RHS(3)); itemp != ITEM_END; itemp = itemp->next) {
      check_valid_data_sharing(itemp->t.sptr);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<schedule> ::= SCHEDULE <sched type> |
   */
  case SCHEDULE1:
    add_clause(CL_SCHEDULE, TRUE);
    CL_VAL(CL_SCHEDULE) = SST_IDG(RHS(2));
    break;
  /*
   *	<schedule> ::= MP_SCHEDTYPE = <id name> |
   */
  case SCHEDULE2:
    add_clause(CL_MP_SCHEDTYPE, TRUE);
    CL_VAL(CL_SCHEDULE) = sched_type(scn.id.name + SST_CVALG(RHS(3)));
    break;
  /*
   *	<schedule> ::= CHUNK = <expression>
   */
  case SCHEDULE3:
    add_clause(CL_CHUNK, TRUE);
    chk_scalartyp(RHS(3), DT_INT, FALSE);
    chunk = SST_ASTG(RHS(3));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<sched type> ::= |
   */
  case SCHED_TYPE1:
    SST_IDP(LHS, DI_SCH_STATIC);
    break;
  /*
   *	<sched type> ::= ( <id name> <opt chunk> )
   */
  case SCHED_TYPE2:
    SST_IDP(LHS, sched_type(scn.id.name + SST_CVALG(RHS(2))));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt chunk> ::= |
   */
  case OPT_CHUNK1:
    break;
  /*
   *	<opt chunk> ::= , <expression>
   */
  case OPT_CHUNK2:
    chk_scalartyp(RHS(2), DT_INT, FALSE);
    chunk = SST_ASTG(RHS(2));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <opt distchunk> ::= |
   */
  case OPT_DISTCHUNK1:
    break;
  /*
   *    <opt distchunk> ::= , <expression>
   */
  case OPT_DISTCHUNK2:
    chk_scalartyp(RHS(2), DT_INT, FALSE);
    distchunk = SST_ASTG(RHS(2));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<reduction> ::= <reduc op> : <pflsr list> |
   */
  case REDUCTION1:
    if (SST_IDG(RHS(1)) == 1 && SST_SYMG(RHS(1)) == 0)
      /* error occurred, so just ignore it */
      break;
    add_clause(CL_REDUCTION, FALSE);
    /*
     * Need to keep the REDUC items around until the end of the
     * parallel do, so allocate them in area 1.
     */
    reducp = (REDUC *)getitem(1, sizeof(REDUC));
    reducp->next = NULL;
    if (SST_IDG(RHS(1)) == 0) {
      reducp->opr = SST_OPTYPEG(RHS(1));
      if (reducp->opr == OP_LOG)
        reducp->intrin = SST_OPCG(RHS(1));
    } else {
      reducp->opr = 0;
      reducp->intrin = SST_SYMG(RHS(1));
    }
    rhstop = 3;
    goto reduction_shared;
  /*
   *	<reduction> ::= <pflsr list>
   */
  case REDUCTION2:
    add_clause(CL_REDUCTION, FALSE);
    /*
     * Need to keep the REDUC items around until the end of the
     * parallel do, so allocate them in area 1.
     */
    reducp = (REDUC *)getitem(1, sizeof(REDUC));
    reducp->next = NULL;
    reducp->opr = OP_ADD;
    rhstop = 1;
  reduction_shared:
    if (CL_FIRST(CL_REDUCTION) == NULL)
      CL_FIRST(CL_REDUCTION) = reducp;
    else
      ((REDUC *)CL_LAST(CL_REDUCTION))->next = reducp;
    CL_LAST(CL_REDUCTION) = reducp;
    /*
     * create a fake REDUC_SYM item (from area 0 freed during the end of
     * statement processing.
     */
    reducp->list = reduc_symp_last = (REDUC_SYM *)getitem(0, sizeof(REDUC_SYM));
    reducp->list->next = NULL;
    for (itemp = SST_BEGG(RHS(rhstop)); itemp != ITEM_END;
         itemp = itemp->next) {
      /*
       * Need to keep the REDUC_SYM items around until the end of the
       * parallel do, so allocate them in area 1.
       */
      reduc_symp = (REDUC_SYM *)getitem(1, sizeof(REDUC_SYM));
      reduc_symp->Private = 0;
      reduc_symp->shared = itemp->t.sptr;
      reduc_symp->next = NULL;
      for (reduc_symp_curr = reducp->list->next; reduc_symp_curr;
           reduc_symp_curr = reduc_symp_curr->next) {
        if (reduc_symp_curr->shared == reduc_symp->shared) {
          error(155, 2, gbl.lineno, "Duplicate name in reduction clause -",
                SYMNAME(reduc_symp->shared));
          break;
        }
      }

      reduc_symp_last->next = reduc_symp;
      reduc_symp_last = reduc_symp;
      if (STYPEG(reduc_symp->shared) != ST_VAR &&
          STYPEG(reduc_symp->shared) != ST_ARRAY) {
        error(155, 3, gbl.lineno,
              "Reduction variable must be a scalar or array variable -",
              SYMNAME(reduc_symp->shared));
        /*
         * pass up 0 so that do_reduction() & end_reduction()
         * will ignore this item.
         */
        reduc_symp->shared = 0;
      } else {
        dtype = DTYPEG(reduc_symp->shared);
        dtype = DDTG(dtype);
        if (!DT_ISBASIC(dtype)) {
          error(155, 3, gbl.lineno,
                "Reduction variable must be of intrinsic type -",
                SYMNAME(reduc_symp->shared));
          reduc_symp->shared = 0;
        }
      }
    }
    /* skip past the fake REDUC_SYM item */
    reducp->list = reducp->list->next;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<reduc op> ::= <addop> |
   */
  case REDUC_OP1:
    SST_IDP(LHS, 0);
    break;
  /*
   *	<reduc op> ::= *       |
   */
  case REDUC_OP2:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_MUL);
    break;
  /*
   *	<reduc op> ::= .AND.   |
   */
  case REDUC_OP3:
    opc = OP_LAND;
    goto reduc_logop;
  /*
   *	<reduc op> ::= .OR.    |
   */
  case REDUC_OP4:
    opc = OP_LOR;
    goto reduc_logop;
  /*
   *	<reduc op> ::= .EQV.   |
   */
  case REDUC_OP5:
    opc = OP_LEQV;
    goto reduc_logop;
  /*
   *	<reduc op> ::= .NEQV.  |
   */
  case REDUC_OP6:
    opc = OP_LNEQV;
  reduc_logop:
    SST_IDP(LHS, 0);
    SST_OPTYPEP(LHS, OP_LOG);
    SST_OPCP(LHS, opc);
    break;
  /*
   *	<reduc op> ::= <ident>
   */
  case REDUC_OP7:
    sptr = find_reduc_intrinsic(SST_SYMG(RHS(1)));
    SST_SYMP(LHS, sptr);
    SST_IDP(LHS, 1);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *    <par ifclause> ::= IF ( <expression> ) |
   */
  case PAR_IFCLAUSE1:
    rhstop = 3;
    if (CL_PRESENT(CL_IF)) {
      if (mp_iftype != IF_DEFAULT)
        error(155, ERR_Severe, gbl.lineno,
              "All IF must have directive-name-modifier", NULL);
    }
    goto share_if_expr;
    break;
  /*
   *    <par ifclause> ::= IF ( <id name> : <expression> ) |
   */
  case PAR_IFCLAUSE2:
    set_iftype(1, scn.id.name + SST_CVALG(RHS(3)), NULL, NULL);
    rhstop = 5;
    goto share_if_expr;
    break;
  /*
   *	<par ifclause> ::= IF ( <id name> <id name> : <expression> ) |
   */
  case PAR_IFCLAUSE3:
    set_iftype(2, scn.id.name + SST_CVALG(RHS(3)),
               scn.id.name + SST_CVALG(RHS(4)), NULL);
    rhstop = 6;
    goto share_if_expr;
    break;
  /*
   *	<par ifclause> ::= IF ( <id name> <id name> <id name> : <expression> )
   */
  case PAR_IFCLAUSE4:
    set_iftype(3, scn.id.name + SST_CVALG(RHS(3)),
               scn.id.name + SST_CVALG(RHS(4)),
               scn.id.name + SST_CVALG(RHS(5)));
    rhstop = 7;
    goto share_if_expr;
    break;

  share_if_expr:
    add_clause(CL_IF, TRUE);
    chk_scalartyp(RHS((rhstop)), DT_LOG4, FALSE);
    CL_VAL(CL_IF) = SST_ASTG(RHS(rhstop));
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<opt par ifclause> ::= |
   */
  case OPT_PAR_IFCLAUSE1:
    break;
  /*
   *	<opt par ifclause> ::= , <par ifclause>
   */
  case OPT_PAR_IFCLAUSE2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <linear clause> ::= LINEAR ( <linear expr> )
   */
  case LINEAR_CLAUSE1:
    error(547, ERR_Warning, gbl.lineno, "LINEAR", CNULL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <linear expr> ::= <linear modifier> <linear opt step>
   */
  case LINEAR_EXPR1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <linear modifier> ::= <pflsr list> |
   */
  case LINEAR_MODIFIER1:
    break;
  /*
   *    <linear modifier> ::= <id name> ( <pflsr list> )
   */
  case LINEAR_MODIFIER2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <linear opt step> ::= |
   */
  case LINEAR_OPT_STEP1:
    break;
  /*
   *    <linear opt step> ::= : <expression>
   */
  case LINEAR_OPT_STEP2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<aligned clause> ::= ALIGNED ( <aligned> )
   */
  case ALIGNED_CLAUSE1:
    error(547, ERR_Warning, gbl.lineno, "ALIGNED", CNULL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<aligned> ::= <pflsr list> |
   */
  case ALIGNED1:
    break;
  /*
   *	<aligned> ::= <pflsr list> : <expression>
   */
  case ALIGNED2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<uniform clause> ::= UNIFORM ( <pflsr list> )
   */
  case UNIFORM_CLAUSE1:
    error(547, ERR_Warning, gbl.lineno, "UNIFORM", CNULL);
    break;

    /* ------------------------------------------------------------------ */
    /*
     *	<map clause> ::= MAP ( <map item> )
     */
  case MAP_CLAUSE1:
    break;

    /* ------------------------------------------------------------------ */
    /*
     *	<map item> ::= <accel data list> |
     */
  case MAP_ITEM1:
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if (flg.omptarget) {
      mp_handle_map_clause(top, CL_MAP, "tofrom", 1, DI_ID(sem.doif_depth),
                           isalways);
    }
#endif
    break;
    /*
     *	<map item> ::= <map type> : <accel data list>
     */
  case MAP_ITEM2:
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if (flg.omptarget) {
      if (strlen(map_type) == 0)
        error(1205, ERR_Severe, gbl.lineno, scn.id.name + SST_CVALG(RHS(1)), 0);

      nmptr = SYMNAME(SST_SYMG(RHS(1)));
      mp_handle_map_clause(top, CL_MAP, map_type, 3, DI_ID(sem.doif_depth),
                           isalways);
    }
#endif
    break;

    /* ------------------------------------------------------------------ */
    /*
     *	<map type> ::= <id name> |
     */
  case MAP_TYPE1:
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if (flg.omptarget) {
      mp_check_maptype(scn.id.name + SST_CVALG(RHS(1)));
      map_type = scn.id.name + SST_CVALG(RHS(1));
    }
#endif
    break;
    /*
     *	<map type> ::= ALWAYS <opt comma> <id name>
     */
  case MAP_TYPE2:
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if (flg.omptarget) {
      mp_check_maptype(scn.id.name + SST_CVALG(RHS(1)));
      map_type = scn.id.name + SST_CVALG(RHS(1));
      isalways = true;
      break;
    }
#endif
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <depend clause> ::= DEPEND ( <depend attr> )
   */
  case DEPEND_CLAUSE1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <depend attr> ::=  <id name> |
   */
  case DEPEND_ATTR1:
    /* expect SOURCE keyword */
    break;
  /*
   *    <depend attr> ::= <id name> : <depend data list>
   */
  case DEPEND_ATTR2:
    /* expect sink or in/out/inout here in id name */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<motion clause> ::= TO ( <var ref list> ) |
   */
  case MOTION_CLAUSE1:
    break;
  /*
   *	<motion clause> ::= FROM ( <var ref list> )
   */
  case MOTION_CLAUSE2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <depend data list> ::= <var ref list> |
   */
  case DEPEND_DATA_LIST1:
    break;
  /*
   *    <depend data list> ::= <depend data>
   */
  case DEPEND_DATA_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <depend data> ::= <ident> <addop> <constant>
   */
  case DEPEND_DATA1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<single begin> ::= <mp single>
   */
  case SINGLE_BEGIN1:
    parstuff_init();
    doif =
        enter_dir(DI_SINGLE, TRUE, 2,
                  DI_B(DI_PDO) | DI_B(DI_PARDO) | DI_B(DI_DOACROSS) |
                      DI_B(DI_PARSECTS) | DI_B(DI_SECTS) | DI_B(DI_SINGLE) |
                      DI_B(DI_CRITICAL) | DI_B(DI_MASTER) | DI_B(DI_ORDERED) |
                      DI_B(DI_TASK) | DI_B(DI_ATOMIC_CAPTURE) |
                      DI_B((DI_PDO | DI_SIMD)) | DI_B((DI_PARDO | DI_SIMD)));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pdo begin> ::= <mp pdo>
   */
  case PDO_BEGIN1:
    parstuff_init();
    doif =
        enter_dir(DI_PDO, FALSE, 0,
                  DI_B(DI_PDO) | DI_B(DI_PARDO) | DI_B(DI_DOACROSS) |
                      DI_B(DI_PARSECTS) | DI_B(DI_SECTS) | DI_B(DI_SINGLE) |
                      DI_B(DI_CRITICAL) | DI_B(DI_MASTER) | DI_B(DI_ORDERED) |
                      DI_B(DI_TASK) | DI_B(DI_ATOMIC_CAPTURE) |
                      DI_B((DI_PDO | DI_SIMD)) | DI_B((DI_PARDO | DI_SIMD)));
    SST_CVALP(LHS, sem.doif_depth); /* always pass up do's DOIF index */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<mp atomic begin> ::= <mp atomic>
   */
  case MP_ATOMIC_BEGIN1:
    sem.mpaccatomic.is_acc = FALSE;
    sem.mpaccatomic.accassignc = 0;
    sem.mpaccatomic.is_acc = 0;
    sem.mpaccatomic.pending = FALSE;
    sem.mpaccatomic.apply = FALSE;
    sem.mpaccatomic.action_type = ATOMIC_UNDEF;
    sem.mpaccatomic.mem_order = MO_UNDEF;
    sem.mpaccatomic.ast = 0;
    sem.mpaccatomic.seen = TRUE;

    if (use_opt_atomic(sem.doif_depth)) {
      sem.mpaccatomic.ast = mk_stmt(A_MP_ATOMIC, 0);
      (void)add_stmt(sem.mpaccatomic.ast);
    } else {
      sem.mpaccatomic.ast = emit_bcs_ecs(A_MP_CRITICAL);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<doacross begin> ::= <mp doacross>
   */
  case DOACROSS_BEGIN1:
    parstuff_init();
    doif =
        enter_dir(DI_DOACROSS, FALSE, 0,
                  DI_B(DI_PDO) | DI_B(DI_PARDO) | DI_B(DI_DOACROSS) |
                      DI_B(DI_PARSECTS) | DI_B(DI_SECTS) | DI_B(DI_SINGLE) |
                      DI_B(DI_CRITICAL) | DI_B(DI_MASTER) | DI_B(DI_ORDERED) |
                      DI_B(DI_TASK) | DI_B(DI_ATOMIC_CAPTURE) |
                      DI_B((DI_PDO | DI_SIMD)) | DI_B((DI_PARDO | DI_SIMD)));
    SST_CVALP(LHS, sem.doif_depth); /* always pass up do's DOIF index */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<paralleldo begin> ::= <mp pardo>
   */
  case PARALLELDO_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_PARDO, FALSE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, sem.doif_depth); /* always pass up do's DOIF index */
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<parallelsections begin> ::= <mp parsections>
   */
  case PARALLELSECTIONS_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_PARSECTS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<sections begin> ::= <mp sections>
   */
  case SECTIONS_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_SECTS, FALSE, 0,
                     DI_B(DI_PDO) | DI_B(DI_PARDO) | DI_B(DI_DOACROSS) |
                         DI_B(DI_SECTS) | DI_B(DI_SINGLE) | DI_B(DI_CRITICAL) |
                         DI_B(DI_MASTER) | DI_B(DI_ORDERED) | DI_B(DI_TASK) |
                         DI_B(DI_ATOMIC_CAPTURE) | DI_B((DI_PDO | DI_SIMD)) |
                         DI_B((DI_PARDO | DI_SIMD)));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<parworkshare begin> ::= <mp parworkshare>
   */
  case PARWORKSHARE_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_PARWORKS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<task begin> ::= <mp task>
   */
  case TASK_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TASK, FALSE, 2, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<dosimd begin> ::= <mp dosimd>
   */
  case DOSIMD_BEGIN1:
    parstuff_init();
    doif =
        enter_dir(DI_PDO, FALSE, 0,
                  DI_B(DI_PDO) | DI_B(DI_PARDO) | DI_B(DI_DOACROSS) |
                      DI_B(DI_PARSECTS) | DI_B(DI_SECTS) | DI_B(DI_SINGLE) |
                      DI_B(DI_CRITICAL) | DI_B(DI_MASTER) | DI_B(DI_ORDERED) |
                      DI_B(DI_TASK) | DI_B(DI_ATOMIC_CAPTURE) |
                      DI_B((DI_PDO | DI_SIMD)) | DI_B((DI_PARDO | DI_SIMD)));
    SST_CVALP(LHS, sem.doif_depth); /* always pass up do's DOIF index */
    if (doif)
      DI_ISSIMD(doif) = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<simd begin> ::= <mp simd>
   */
  case SIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_SIMD, FALSE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, sem.doif_depth); /* always pass up do's DOIF index */

    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targetdata begin> ::= <mp targetdata>
   */
  case TARGETDATA_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGETDATA, TRUE, 0,
                     DI_B(DI_ATOMIC_CAPTURE) | DI_B(DI_TARGET) |
                         DI_B(DI_TARGETENTERDATA) | DI_B(DI_TARGETEXITDATA) |
                         DI_B(DI_TARGETUPDATE) | DI_B(DI_TARGETDATA));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targetenterdata begin> ::= <mp targetenterdata>
   */
  case TARGETENTERDATA_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGETENTERDATA, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targetexitdata begin> ::= <mp targetexitdata>
   */
  case TARGETEXITDATA_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGETEXITDATA, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<target begin> ::= <mp target>
   */
  case TARGET_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targetupdate begin> ::= <mp targetupdate>
   */
  case TARGETUPDATE_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGETUPDATE, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<teams begin> ::= <mp teams>
   */
  case TEAMS_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TEAMS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<distribute begin> ::= <mp distribute>
   */
  case DISTRIBUTE_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_DISTRIBUTE, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<distsimd begin> ::= <mp distsimd>
   */
  case DISTSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_DISTRIBUTE, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<distpardo begin> ::= <mp distpardo>
   */
  case DISTPARDO_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_DISTPARDO, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<distpardosimd begin> ::= <mp distpardosimd>
   */
  case DISTPARDOSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_DISTPARDO, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<pardosimd begin> ::= <mp pardosimd>
   */
  case PARDOSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_PARDO, FALSE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, sem.doif_depth); /* always pass up do's DOIF index */
    if (doif)
      DI_ISSIMD(doif) = TRUE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targpar begin> ::= <mp targpar>
   */
  case TARGPAR_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targpardo begin> ::= <mp targpardo>
   */
  case TARGPARDO_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targparsimd begin> ::= <mp targparsimd>
   */
  case TARGPARSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targpardosimd begin> ::= <mp targpardosimd>
   */
  case TARGPARDOSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targsimd begin> ::= <mp targsimd>
   */
  case TARGSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targteams begin> ::= <mp targteams>
   */
  case TARGTEAMS_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<teamsdist begin> ::= <mp teamsdist>
   */
  case TEAMSDIST_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TEAMS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<teamsdistsimd begin> ::= <mp teamsdistsimd>
   */
  case TEAMSDISTSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TEAMS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targteamsdist begin> ::= <mp targteamsdist>
   */
  case TARGTEAMSDIST_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targteamsdistsimd begin> ::= <mp targteamsdistsimd>
   */
  case TARGTEAMSDISTSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<teamsdistpardo begin> ::= <mp teamsdistpardo>
   */
  case TEAMSDISTPARDO_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TEAMS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targteamsdistpardo begin> ::= <mp targteamsdistpardo>
   */
  case TARGTEAMSDISTPARDO_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<teamsdistpardosimd begin> ::= <mp teamsdistpardosimd>
   */
  case TEAMSDISTPARDOSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TEAMS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<targteamsdistpardosimd begin> ::= <mp targteamsdistpardosimd>
   */
  case TARGTEAMSDISTPARDOSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TARGET, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<taskloop begin> ::= <mp taskloop>
   */
  case TASKLOOP_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TASKLOOP, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<taskloopsimd begin> ::= <mp taskloopsimd>
   */
  case TASKLOOPSIMD_BEGIN1:
    parstuff_init();
    doif = enter_dir(DI_TASKLOOP, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    SST_CVALP(LHS, doif);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel stmt> ::= <accel begin> ACCREGION <opt accel list>  |
   */
  case ACCEL_STMT1:
    ditype = DI_ACCREG;
    dimask = 0;
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCREG;
    dirname = "ACC REGION";
    pr1 = PR_ACCEL;
    pr2 = 0;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCKERNELS <opt accel list>  |
   */
  case ACCEL_STMT2:
    ditype = DI_ACCKERNELS;
    dimask = 0;
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCKERNELS;
    dirname = "ACC KERNELS";
    pr1 = PR_ACCKERNELS;
    pr2 = 0;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> PARALLEL <opt accel list>  |
   */
  case ACCEL_STMT3:
    ditype = DI_ACCPARALLEL;
    dimask = 0;
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCPARALLEL;
    dirname = "ACC PARALLEL";
    pr1 = PR_ACCPARCONSTRUCT;
    pr2 = 0;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCDATA <opt accel list>  |
   */
  case ACCEL_STMT4:
    ditype = DI_ACCDATAREG;
    dimask = 0;
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCDATAREG;
    dirname = "ACC DATA";
    pr1 = PR_ACCDATAREG;
    pr2 = 0;
    dignorenested = FALSE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCDATAREGION <opt accel list>  |
   */
  case ACCEL_STMT5:
    ditype = DI_ACCDATAREG;
    dimask = 0;
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCDATAREG;
    dirname = "ACC DATA REGION";
    pr1 = PR_ACCDATAREG;
    pr2 = 0;
    dignorenested = FALSE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCDO <opt accel list>  |
   */
  case ACCEL_STMT6:
    ditype = DI_ACCDO;
    dimask = 0;
    dinestmask = 0;
    if (DI_IN_NEST(sem.doif_depth, DI_ACCREG)) {
      bttype = BT_ACCKDO;
      pr1 = PR_ACCELLP;
      dirname = "ACC DO";
    } else if (DI_IN_NEST(sem.doif_depth, DI_ACCKERNELS)) {
      bttype = BT_ACCKDO;
      pr1 = PR_ACCKLOOP;
      dirname = "ACC LOOP";
    } else {
      bttype = BT_ACCPDO;
      pr1 = PR_ACCPLOOP;
      dirname = "ACC LOOP";
    }
    pr2 = 0;
    dignorenested = FALSE;
    goto ACCEL_ENTER_REGION;

  /*
   *	<accel stmt> ::= <accel begin> ACCLOOP <opt accel list>  |
   */
  case ACCEL_STMT7:
    ditype = DI_ACCLOOP;
    dimask = 0;
    dinestmask = 0;
    if (DI_IN_NEST(sem.doif_depth, DI_ACCREG)) {
      bttype = BT_ACCKLOOP;
      pr1 = PR_ACCELLP;
      dirname = "ACC DO";
    } else if (DI_IN_NEST(sem.doif_depth, DI_ACCKERNELS)) {
      bttype = BT_ACCKLOOP;
      pr1 = PR_ACCKLOOP;
      dirname = "ACC LOOP";
    } else if (DI_IN_NEST(sem.doif_depth, DI_ACCSERIAL)) {
      bttype = BT_ACCSLOOP;
      pr1 = PR_ACCSLOOP;
      dirname = "ACC LOOP";
    } else {
      bttype = BT_ACCPLOOP;
      pr1 = PR_ACCPLOOP;
      dirname = "ACC LOOP";
    }
    pr2 = 0;
    dignorenested = FALSE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCREGIONDO <opt accel list>  |
   */
  case ACCEL_STMT8:
    dimask = DI_B(DI_ACCREG) | DI_B(DI_ACCDO);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCREG | BT_ACCKDO;
    dirname = "ACC REGION DO";
    pr1 = PR_ACCEL;
    pr2 = PR_ACCELLP;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCREGIONLOOP <opt accel list>  |
   */
  case ACCEL_STMT9:
    ditype = DI_ACCREGLOOP;
    dimask = DI_B(DI_ACCREG) | DI_B(DI_ACCLOOP);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCREG | BT_ACCKLOOP;
    dirname = "ACC REGION LOOP";
    pr1 = PR_ACCEL;
    pr2 = PR_ACCELLP;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCKERNELSDO <opt accel list>  |
   */
  case ACCEL_STMT10:
    ditype = DI_ACCKERNELSDO;
    dimask = DI_B(DI_ACCKERNELS) | DI_B(DI_ACCDO);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCKERNELS | BT_ACCKDO;
    dirname = "ACC KERNELS DO";
    pr1 = PR_ACCKERNELS;
    pr2 = PR_ACCTKLOOP;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCKERNELSLOOP <opt accel list>  |
   */
  case ACCEL_STMT11:
    dimask = DI_B(DI_ACCKERNELS) | DI_B(DI_ACCLOOP);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCKERNELS | BT_ACCKLOOP;
    dirname = "ACC KERNELS LOOP";
    pr1 = PR_ACCKERNELS;
    pr2 = PR_ACCTKLOOP;
    dignorenested = TRUE;
  ACCEL_ENTER_REGION:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCPARALLELDO <opt accel list>  |
   */
  case ACCEL_STMT12:
    ditype = DI_ACCPARALLELDO;
    dimask = DI_B(DI_ACCPARALLEL) | DI_B(DI_ACCDO);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCPARALLEL | BT_ACCPDO;
    dirname = "ACC PARALLEL DO";
    pr1 = PR_ACCPARCONSTRUCT;
    pr2 = PR_ACCTPLOOP;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCPARALLELLOOP <opt accel list>  |
   */
  case ACCEL_STMT13:
    ditype = DI_ACCPARALLELLOOP;
    dimask = DI_B(DI_ACCPARALLEL) | DI_B(DI_ACCLOOP);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCPARALLEL | BT_ACCPLOOP;
    dirname = "ACC PARALLEL LOOP";
    pr1 = PR_ACCPARCONSTRUCT;
    pr2 = PR_ACCTPLOOP;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> <accel update dir> |
   */
  case ACCEL_STMT14:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDREGION <opt end accel list> |
   */
  case ACCEL_STMT15:
    ditype = DI_ACCREG;
    ditype2 = DI_ACCREGDO;
    ditype3 = DI_ACCREGLOOP;
    bttype = BT_ACCENDREG;
    dirname = "ACC END REGION";
    pr1 = PR_ENDACCEL;
  ACCEL_END_REGION:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDKERNELS |
   */
  case ACCEL_STMT16:
    ditype = DI_ACCKERNELS;
    ditype2 = DI_ACCKERNELSDO;
    ditype3 = DI_ACCKERNELSLOOP;
    bttype = 0;
    dirname = "ACC END KERNELS";
    pr1 = PR_ACCENDKERNELS;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDKERNDO |
   */
  case ACCEL_STMT17:
    ditype = DI_ACCKERNELS;
    ditype2 = DI_ACCKERNELSDO;
    ditype3 = DI_ACCKERNELSLOOP;
    bttype = 0;
    dirname = "ACC END KERNELS DO";
    pr1 = PR_ACCENDKERNELS;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDKERNLOOP |
   */
  case ACCEL_STMT18:
    ditype = DI_ACCKERNELS;
    ditype2 = DI_ACCKERNELSDO;
    ditype3 = DI_ACCKERNELSLOOP;
    bttype = 0;
    dirname = "ACC END KERNELS LOOP";
    pr1 = PR_ACCENDKERNELS;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDPARALLEL |
   */
  case ACCEL_STMT19:
    ditype = DI_ACCPARALLEL;
    ditype2 = DI_ACCPARALLELDO;
    ditype3 = DI_ACCPARALLELLOOP;
    bttype = 0;
    dirname = "ACC END PARALLEL";
    pr1 = PR_ACCENDPARCONSTRUCT;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDPARDO |
   */
  case ACCEL_STMT20:
    ditype = DI_ACCPARALLEL;
    ditype2 = DI_ACCPARALLELDO;
    ditype3 = DI_ACCPARALLELLOOP;
    bttype = 0;
    dirname = "ACC END PARALLEL DO";
    pr1 = PR_ACCENDPARCONSTRUCT;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDPARLOOP |
   */
  case ACCEL_STMT21:
    ditype = DI_ACCPARALLEL;
    ditype2 = DI_ACCPARALLELDO;
    ditype3 = DI_ACCPARALLELLOOP;
    bttype = 0;
    dirname = "ACC END PARALLEL LOOP";
    pr1 = PR_ACCENDPARCONSTRUCT;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= ACCENDDATAREGION |
   */
  case ACCEL_STMT22:
    ditype = DI_ACCDATAREG;
    ditype2 = 0;
    ditype3 = 0;
    bttype = 0;
    dirname = "ACC END DATA REGION";
    pr1 = PR_ACCENDDATAREG;
    goto ACCEL_END_REGION;

  /*
   *	<accel stmt> ::= ACCENDDATA
   */
  case ACCEL_STMT23:
    ditype = DI_ACCDATAREG;
    ditype2 = 0;
    ditype3 = 0;
    bttype = 0;
    dirname = "ACC END DATA";
    pr1 = PR_ACCENDDATAREG;
    goto ACCEL_END_REGION;

  /*
   *	<accel stmt> ::= <accel begin> ACCSCALARREGION <opt accel list>  |
   */
  case ACCEL_STMT24:
    ditype = DI_ACCREG;
    dimask = DI_B(DI_ACCREG);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    ditype2 = -1;
    ditype3 = -1;
    bttype = BT_ACCSCALARREG;
    dirname = "ACC SCALAR REGION";
    pr1 = PR_ACCSCALARREG;
    pr2 = 0;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDSCALARREGION |
   */
  case ACCEL_STMT25:
    ditype = DI_ACCREG;
    ditype2 = 0;
    ditype3 = 0;
    bttype = BT_ACCENDREG;
    dirname = "ACC END SCALAR REGION";
    pr1 = PR_ENDACCEL;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCSCALAR ACCREGION <opt accel list>  |
   */
  case ACCEL_STMT26:
    /* error case, should not occur */
    interr("semsmp: bad accelerator directive", rednum, 3);
    break;
  /*
   *	<accel stmt> ::= ACCENDSCALAR
   */
  case ACCEL_STMT27:
    /* error case, should not occur */
    interr("semsmp: bad accelerator directive", rednum, 3);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCWAIT <opt wait list> |
   */
  case ACCEL_STMT28:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> CACHE ( <accel data list> ) |
   */
  case ACCEL_STMT29:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCHOSTDATA <opt accel list> |
   */
  case ACCEL_STMT30:
    ditype = DI_ACCHOSTDATA;
    dimask = 0;
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL);
    bttype = BT_ACCHOSTDATA;
    dirname = "ACC HOST_DATA";
    pr1 = PR_ACCHOSTDATA;
    pr2 = 0;
    dignorenested = FALSE;
    goto ACCEL_ENTER_REGION;
  /*
   *	<accel stmt> ::= ACCENDHOSTDATA |
   */
  case ACCEL_STMT31:
    ditype = DI_ACCHOSTDATA;
    ditype2 = 0;
    ditype3 = 0;
    bttype = 0;
    dirname = "ACC END HOSTDATA";
    pr1 = PR_ACCENDHOSTDATA;
    goto ACCEL_END_REGION;
  /*
   *	<accel stmt> ::= <accel begin> ACCENTER ACCDATA <opt accel list>  |
   */
  case ACCEL_STMT32:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCEXIT ACCDATA <opt accel list>
   */
  case ACCEL_STMT33:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDLOOP |
   */
  case ACCEL_STMT34:
    /* ignore the endloop */
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCENDDO
   */
  case ACCEL_STMT35:
    /* ignore the endloop */
    SST_ASTP(LHS, 0);
    break;

  /*
   *	<accel stmt> ::= <accel begin> ACCATOMIC |
   *	                 <accel begin> ACCATOMICUPDATE |
   *	                 <accel begin> ACCATOMICREAD |
   *	                 <accel begin> ACCATOMICWRITE |
   *	                 <accel begin> ACCATOMICCAPTURE
   */
  case ACCEL_STMT36:
  case ACCEL_STMT37:
  case ACCEL_STMT38:
  case ACCEL_STMT39:
  case ACCEL_STMT40: {
    int atomic_action;
    if (rednum == ACCEL_STMT36 || rednum == ACCEL_STMT37) {
      atomic_action = A_ATOMIC;
      sem.mpaccatomic.action_type = ATOMIC_UPDATE;
    } else if (rednum == ACCEL_STMT38) {
      atomic_action = A_ATOMICREAD;
      sem.mpaccatomic.action_type = ATOMIC_READ;
    } else if (rednum == ACCEL_STMT39) {
      atomic_action = A_ATOMICWRITE;
      sem.mpaccatomic.action_type = ATOMIC_WRITE;
    } else if (rednum == ACCEL_STMT40) {
      atomic_action = A_ATOMICCAPTURE;
      sem.mpaccatomic.action_type = ATOMIC_CAPTURE;
    }

    sem.mpaccatomic.is_acc = TRUE;
    sem.mpaccatomic.accassignc = 0;
    if (sem.mpaccatomic.pending) {
      sem.mpaccatomic.pending = FALSE;
      error(155, 3, gbl.lineno,
            "Statement after ATOMIC is not an assignment (no nesting)", CNULL);
    } else {
      int ast_atomic;
      sem.mpaccatomic.seen = TRUE;
      ast_atomic = mk_stmt(atomic_action, 0);
      add_stmt(ast_atomic);
      sem.mpaccatomic.ast = ast_atomic;
    }
  }
    SST_ASTP(LHS, 0);
    break;

  /*
   *	<accel begin> ::= ACCENDATOMIC
   */
  case ACCEL_STMT41:
    if (sem.mpaccatomic.is_acc == FALSE) {
      error(155, 3, gbl.lineno, "Unmatched atomic end", CNULL);
    } else {
      /* Nothing to do, yet... */
      if (sem.mpaccatomic.action_type == ATOMIC_CAPTURE) {
        int end_atomic;
        end_atomic = mk_stmt(A_ENDATOMIC, 0);
        add_stmt(end_atomic);
        sem.mpaccatomic.ast = end_atomic;
      }
      /* reset the sem.mpaccatomic.is_acc if the current processing statement
         is leaving the atomic region. If we do not reset this variable, the
         following statement can be compiled even it is incorrect syntax.

         !$acc atomic update
           assignment stmt
         !$acc end atomic
         !$acc end atomic

         The second end atomic directive is unmatched one and the compiler
         should issue an error.

         Furthur more, this tag can be used to detect the illegal staements
         in the atomic region. For example, this flag is used in semant3.c
         to detect the mulitple assignment stmts which are illegal in the
         atomic region.

         by daniel tian
       */
      if ((sem.mpaccatomic.is_acc &&
           sem.mpaccatomic.action_type == ATOMIC_CAPTURE &&
           sem.mpaccatomic.accassignc > 2) ||
          (sem.mpaccatomic.action_type != ATOMIC_CAPTURE &&
           sem.mpaccatomic.accassignc > 1)) {
        error(
            155, 3, gbl.lineno,
            "Multiple Assignment Statements were illegal in the Atomic Region",
            NULL);
      }
      sem.mpaccatomic.is_acc = FALSE;
    }
    sem.mpaccatomic.seen = FALSE;
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCINIT <opt accel init list> |
   */
  case ACCEL_STMT42:
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCSHUTDOWN <opt accel shutdown list> |
   */
  case ACCEL_STMT43:
    break;
  /*
   *	<accel stmt> ::= <accel begin> <accel setdev dir>
   */
  case ACCEL_STMT44:
    break;
  /*
   *	<accel stmt> ::= <accel begin> CACHE ( <ident> : <accel sdata list> )
   */
  case ACCEL_STMT45:
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel stmt> ::= <accel begin> ACCSERIAL <opt accel list>  |
   */
  case ACCEL_STMT46:
    ditype = DI_ACCSERIAL;
    dimask = DI_B(DI_ACCSERIAL);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL) |
                 DI_B(DI_ACCSERIAL);
    ditype2 = -1;
    ditype3 = -1;
    bttype = BT_ACCSERIAL;
    dirname = "ACC SERIAL";
    pr1 = PR_ACCSERIAL;
    pr2 = 0;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
    /*
     *	<accel stmt> ::= <accel begin> ACCENDSERIAL |
     */
  case ACCEL_STMT47:
    ditype = DI_ACCSERIAL;
    ditype2 = 0;
    ditype3 = 0;
    bttype = 0;
    dirname = "ACC END SERIAL";
    pr1 = PR_ENDACCEL;
    goto ACCEL_END_REGION;
    /*
     *	<accel stmt> ::= <accel begin> ACCSERIALLOOP <opt accel list>  |
     */
  case ACCEL_STMT48:
    ditype = DI_ACCSERIALLOOP;
    dimask = DI_B(DI_ACCSERIAL) | DI_B(DI_ACCLOOP);
    dinestmask = DI_B(DI_ACCREG) | DI_B(DI_ACCKERNELS) | DI_B(DI_ACCPARALLEL) |
                 DI_B(DI_ACCSERIAL);
    bttype = BT_ACCSERIAL | BT_ACCSLOOP;
    dirname = "ACC SERIAL LOOP";
    pr1 = PR_ACCSERIAL;
    pr2 = PR_ACCTSLOOP;
    dignorenested = TRUE;
    goto ACCEL_ENTER_REGION;
    /*
     *	<accel stmt> ::= <accel begin> ACCENDSERIALLOOP
     */
  case ACCEL_STMT49:
    ditype = DI_ACCSERIAL;
    ditype2 = 0;
    ditype3 = DI_ACCSERIALLOOP;
    bttype = 0;
    dirname = "ACC END SERIAL LOOP";
    pr1 = PR_ACCENDSERIAL;
    goto ACCEL_END_REGION;

  /*
   * <accel stmt> ::= <accel begin> <accel compare dir> |
   */
  case ACCEL_STMT50:
    break;
  /*
   * <accel stmt> ::= <pgi begin> <pgi compare dir>
   */
  case ACCEL_STMT51:
    accel_pragmagen(PR_PCASTCOMPARE, 0, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <accel begin> ::=
   */
  case ACCEL_BEGIN1:
    parstuff_init();
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <pgi begin> ::=
   */
  case PGI_BEGIN1:
    parstuff_init();
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt accel list> ::= |
   */
  case OPT_ACCEL_LIST1:
    break;
  /*
   *	<opt accel list> ::= <opt comma> <accel list>
   */
  case OPT_ACCEL_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel list> ::= <accel list> <opt comma> <accel attr> |
   */
  case ACCEL_LIST1:
    break;
  /*
   *	<accel list> ::= <accel attr>
   */
  case ACCEL_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel attr> ::= COPYIN ( <accel data list> ) |
   */
  case ACCEL_ATTR1:
    break;
  /*
   *	<accel attr> ::= COPYOUT ( <accel data list> ) |
   */
  case ACCEL_ATTR2:
    break;
  /*
   *	<accel attr> ::= LOCAL ( <accel data list> ) |
   */
  case ACCEL_ATTR3:
    break;
  /*
   *	<accel attr> ::= CREATE ( <accel data list> ) |
   */
  case ACCEL_ATTR4:
    break;
  /*
   *	<accel attr> ::= PRESENT ( <accel data list> ) |
   */
  case ACCEL_ATTR5:
    break;
  /*
   *	<accel attr> ::= PCOPY ( <accel data list> ) |
   */
  case ACCEL_ATTR6:
    break;
  /*
   *	<accel attr> ::= PCOPYIN ( <accel data list> ) |
   */
  case ACCEL_ATTR7:
    break;
  /*
   *	<accel attr> ::= PCOPYOUT ( <accel data list> ) |
   */
  case ACCEL_ATTR8:
    break;
  /*
   *	<accel attr> ::= PLOCAL ( <accel data list> ) |
   */
  case ACCEL_ATTR9:
    break;
  /*
   *	<accel attr> ::= PCREATE ( <accel data list> ) |
   */
  case ACCEL_ATTR10:
    break;
  /*
   *	<accel attr> ::= DEVICEPTR ( <accel data list> ) |
   */
  case ACCEL_ATTR11:
    break;
  /*
   *	<accel attr> ::= PRIVATE ( <accel data list> ) |
   */
  case ACCEL_ATTR12:
    break;
  /*
   *	<accel attr> ::= FIRSTPRIVATE ( <accel data list> ) |
   */
  case ACCEL_ATTR13:
    break;
  /*
   *	<accel attr> ::= CACHE ( <accel data list> ) |
   */
  case ACCEL_ATTR14:
    break;
  /*
   *	<accel attr> ::= SHORTLOOP |
   */
  case ACCEL_ATTR15:
    break;
  /*
   *	<accel attr> ::= VECTOR ( <ident> : <expression> ) |
   */
  case ACCEL_ATTR16:
    break;
  /*
   *	<accel attr> ::= VECTOR ( <expression> ) |
   */
  case ACCEL_ATTR17:
    clause = CL_VECTOR;
    recent_loop_clause = clause;
    arg = 3;
    goto acc_sched_shared;
  /*
   *	<accel attr> ::= VECTOR |
   */
  case ACCEL_ATTR18:
    clause = CL_VECTOR;
    recent_loop_clause = clause;
    goto acc_nowidth_shared;
  /*
   *	<accel attr> ::= PARALLEL ( <expression> ) |
   */
  case ACCEL_ATTR19:
    clause = CL_PARALLEL;
    recent_loop_clause = clause;
    arg = 3;
    goto acc_sched_shared;
  /*
   *	<accel attr> ::= PARALLEL |
   */
  case ACCEL_ATTR20:
    clause = CL_PARALLEL;
    recent_loop_clause = clause;
    goto acc_nowidth_shared;
  /*
   *	<accel attr> ::= SEQ ( <expression> ) |
   */
  case ACCEL_ATTR21:
    if (ACCSTRICT || ACCVERYSTRICT)
      error(531, ACCVERYSTRICT ? 3 : 2, gbl.lineno,
            "seq clause with (<expression>)", "");
    clause = CL_SEQ;
    recent_loop_clause = clause;
    arg = 3;
    goto acc_sched_shared;
  /*
   *	<accel attr> ::= SEQ |
   */
  case ACCEL_ATTR22:
    clause = CL_SEQ;
    recent_loop_clause = clause;
    goto acc_nowidth_shared;
  /*
   *	<accel attr> ::= HOST ( <expression> ) |
   */
  case ACCEL_ATTR23:
    clause = CL_HOST;
    recent_loop_clause = clause;
    arg = 3;

  acc_sched_shared:
    break;
  /*
   *	<accel attr> ::= HOST |
   */
  case ACCEL_ATTR24:
    clause = CL_HOST;
    recent_loop_clause = clause;

  acc_nowidth_shared:
    break;
  /*
   *	<accel attr> ::= IF ( <expression> ) |
   */
  case ACCEL_ATTR25:
    break;
  /*
   *	<accel attr> ::= UNROLL ( <expression> )
   */
  case ACCEL_ATTR26:
    switch (recent_loop_clause) {
    case CL_SEQ:
      clause = CL_SEQUNROLL;
      break;
    case CL_PARALLEL:
    case CL_GANG:
      clause = CL_PARUNROLL;
      break;
    case CL_VECTOR:
    case CL_WORKER:
      clause = CL_VECUNROLL;
      break;
    default:
      clause = CL_UNROLL;
      break;
    }
    arg = 3;
    goto acc_sched_shared;
  /*
   *	<accel attr> ::= INDEPENDENT |
   */
  case ACCEL_ATTR27:
    break;
  /*
   *	<accel attr> ::= KERNEL
   */
  case ACCEL_ATTR28:
    break;
  /*
   *	<accel attr> ::= COPY ( <accel data list> )
   */
  case ACCEL_ATTR29:
    break;
  /*
   *	<accel attr> ::= MIRROR ( <accel data list> )
   */
  case ACCEL_ATTR30:
    break;
  /*
   *	<accel attr> ::= ACCUPDATE HOST ( <accel data list> ) |
   */
  case ACCEL_ATTR31:
    clause = CL_UPDATEHOST;
    op = 4;
  acc_update_clause_shared:
    break;
  /*
   *	<accel attr> ::= ACCUPDATE SELF ( <accel data list> ) |
   */
  case ACCEL_ATTR32:
    clause = CL_UPDATESELF;
    op = 4;
    goto acc_update_clause_shared;
  /*
   *	<accel attr> ::= ACCUPDATE DEVICE ( <accel data list> ) |
   */
  case ACCEL_ATTR33:
    clause = CL_UPDATEDEV;
    op = 4;
    goto acc_update_clause_shared;
  /*
   *	<accel attr> ::= <accel short update> |
   */
  case ACCEL_ATTR34:
    break;
  /*
   *	<accel attr> ::= ACCUPDATE ACCIN ( <accel data list> ) |
   */
  case ACCEL_ATTR35:
    clause = CL_UPDATEDEV;
    op = 4;
    goto acc_update_clause_shared;
    break;
  /*
   *	<accel attr> ::= ACCUPDATE ACCOUT ( <accel data list> )
   */
  case ACCEL_ATTR36:
    clause = CL_UPDATEHOST;
    op = 4;
    goto acc_update_clause_shared;
    break;
  /*
   *	<accel attr> ::= ACCWAIT |
   */
  case ACCEL_ATTR37:
    break;
  /*
   *	<accel attr> ::= NOWAIT
   */
  case ACCEL_ATTR38:
    break;
  /*
   *	<accel attr> ::= WORKER ( <ident> : <expression> ) |
   */
  case ACCEL_ATTR39:
    break;
  /*
   *	<accel attr> ::= WORKER ( <expression> ) |
   */
  case ACCEL_ATTR40:
    break;
  /*
   *	<accel attr> ::= WORKER |
   */
  case ACCEL_ATTR41:
    break;
  /*
   *	<accel attr> ::= GANG ( <acc gang args> ) |
   */
  case ACCEL_ATTR42:
    break;
  /*
   *	<accel attr> ::= GANG |
   */
  case ACCEL_ATTR43:
    break;
  /*
   *	<accel attr> ::= COLLAPSE ( <expression> ) |
   */
  case ACCEL_ATTR44:
    break;
  /*
   *	<accel attr> ::= ASYNC |
   */
  case ACCEL_ATTR45:
    break;
  /*
   *	<accel attr> ::= ASYNC ( <expression> ) |
   */
  case ACCEL_ATTR46:
    break;
  /*
   *	<accel attr> ::= REDUCTION ( <reduction> ) |
   */
  case ACCEL_ATTR47:
    break;
  /*
   *	<accel attr> ::= NUM_WORKERS ( <expression> ) |
   */
  case ACCEL_ATTR48:
    break;
  /*
   *	<accel attr> ::= NUM_GANGS ( <gangsizes> ) |
   */
  case ACCEL_ATTR49:
    break;
  /*
   *	<accel attr> ::= VECTOR_LENGTH ( <expression> ) |
   */
  case ACCEL_ATTR50:
    break;
  /*
   *	<accel attr> ::= USE_DEVICE ( <accel data list> )
   */
  case ACCEL_ATTR51:
    break;
  /*
   *	<accel attr> ::= DEVICEID ( <expression> )
   */
  case ACCEL_ATTR52:
    break;
    /*
     *	<accel attr> ::= DELETE ( <accel data list> ) |
     */
  case ACCEL_ATTR53:
    break;
  /*
   *	<accel attr> ::= PDELETE ( <accel data list> )
   */
  case ACCEL_ATTR54:
    break;
  /*
   *	<accel attr> ::= ACCWAIT ( <accel wait list> )
   */
  case ACCEL_ATTR55:
    break;
  /*
   *	<accel attr> ::= DEVICE_TYPE ( <devtype list> ) |
   */
  case ACCEL_ATTR56:
    clause = CL_DEVICE_TYPE;
    add_clause(clause, FALSE);
    CL_VAL(CL_DEVICE_TYPE) = SST_DEVICEG(RHS(3));
    break;
  /*
   *	<accel attr> ::= AUTO |
   */
  case ACCEL_ATTR57:
    break;
  /*
   *	<accel attr> ::= ACCTILE ( <accsizelist> ) |
   */
  case ACCEL_ATTR58:
    break;
  /*
   *	<accel attr> ::= DEFAULT ( <ident> ) |
   */
  case ACCEL_ATTR59:
    break;
  /*
   *	<accel attr> ::= PNOT ( <accel data list> )
   */
  case ACCEL_ATTR60:
    break;
  /*
   *	<accel attr> ::= COLLAPSE ( <ident> : <expression> )
   */
  case ACCEL_ATTR61:
    break;
  /*
   *	<accel attr> ::= ACCFINALIZE |
   */
  case ACCEL_ATTR62:
    break;
  /*
   *	<accel attr> ::= ACCIFPRESENT |
   */
  case ACCEL_ATTR63:
    break;
  /*
   *	<accel attr> ::= ACCATTACH ( <accel data list> ) |
   */
  case ACCEL_ATTR64:
    break;
  /*
   *	<accel attr> ::= ACCDETACH ( <accel data list> ) |
   */
  case ACCEL_ATTR65:
    break;
  /*
   *	<accel attr> ::= NO_CREATE ( <accel data list> )
   */
  case ACCEL_ATTR66:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<acc gang args> ::= <acc gang arg> |
   */
  case ACC_GANG_ARGS1:
    break;
  /*
   *	<acc gang args> ::= <acc gang args> , <acc gang arg>
   */
  case ACC_GANG_ARGS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<acc gang arg> ::= <expression> |
   */
  case ACC_GANG_ARG1:
    break;
  /*
   *	<acc gang arg> ::= <ident> : <accsize> |
   */
  case ACC_GANG_ARG2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<gangsizes> ::= <expression> |
   */
  case GANGSIZES1:
    break;
  /*
   *	<gangsizes> ::= <expression> , <gangsize2>
   */
  case GANGSIZES2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<gangsize2> ::= <expression> |
   */
  case GANGSIZE21:
    break;
  /*
   *	<gangsize2> ::= <expression> , <gangsize3>
   */
  case GANGSIZE22:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<gangsize3> ::= <expression>
   */
  case GANGSIZE31:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accsizelist> ::= <accsize> |
   */
  case ACCSIZELIST1:
    break;
  /*
   *	<accsizelist> ::= <accsizelist> , <accsize>
   */
  case ACCSIZELIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accsize> ::= <expression> |
   */
  case ACCSIZE1:
    chktyp(RHS(1), DT_INT, FALSE);
    SST_ASTP(LHS, SST_ASTG(RHS(1)));
    break;
  /*
   *	<accsize> ::= *
   */
  case ACCSIZE2:
    SST_ASTP(LHS, new_node(A_NULL));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt end accel list> ::= |
   */
  case OPT_END_ACCEL_LIST1:
    break;
  /*
   *	<opt end accel list> ::= <end accel list>
   */
  case OPT_END_ACCEL_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<end accel list> ::= <end accel list> <opt comma> <end accel attr> |
   */
  case END_ACCEL_LIST1:
    break;
  /*
   *	<end accel list> ::= <end accel attr>
   */
  case END_ACCEL_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<end accel attr> ::= ACCWAIT |
   */
  case END_ACCEL_ATTR1:
    break;
  /*
   *	<end accel attr> ::= NOWAIT
   */
  case END_ACCEL_ATTR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel data list> ::= <accel data list> , <accel data> |
   */
  case ACCEL_DATA_LIST1:
  accel_data_list1:
    if (SST_ASTG(RHS(3))) {
      itemp = (ITEM *)getitem(0, sizeof(ITEM));
      itemp->next = ITEM_END;
      itemp->ast = SST_ASTG(RHS(3));
      if (SST_ENDG(RHS(1)) != ITEM_END) {
        (SST_ENDG(RHS(1)))->next = itemp;
        SST_ENDP(LHS, itemp);
      } else {
        SST_BEGP(LHS, itemp);
        SST_ENDP(LHS, itemp);
      }
    }
    break;
  /*
   *	<accel data list> ::= <accel data>
   */
  case ACCEL_DATA_LIST2:
  accel_data_list2:
    if (SST_ASTG(RHS(1))) {
      itemp = (ITEM *)getitem(0, sizeof(ITEM));
      itemp->next = ITEM_END;
      itemp->ast = SST_ASTG(RHS(1));
      SST_BEGP(LHS, itemp);
      SST_ENDP(LHS, itemp);
    } else {
      SST_BEGP(LHS, ITEM_END);
      SST_ENDP(LHS, ITEM_END);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel data> ::= <accel data name> ( <accel sub list> ) |
   */
  case ACCEL_DATA1:
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
    if(is_in_omptarget(sem.doif_depth) || is_in_omptarget_data(sem.doif_depth)) {
      //todo support array section in the map clause for openmp
      if (SST_IDG(RHS(1)) == S_IDENT || SST_IDG(RHS(1)) == S_DERIVED) {
        sptr = SST_SYMG(RHS(1));
      } else {
        sptr = SST_LSYMG(RHS(1));
      }
      error(1206, ERR_Warning, gbl.lineno, sptr ? SYMNAME(sptr) : CNULL, CNULL);
      goto accel_data2;
      break;
    }
#endif
  accel_data1:
    if (SST_IDG(RHS(1)) == S_IDENT || SST_IDG(RHS(1)) == S_DERIVED) {
      sptr = SST_SYMG(RHS(1));
    } else {
      sptr = SST_LSYMG(RHS(1));
    }
    switch (STYPEG(sptr)) {
    case ST_MEMBER:
    case ST_ARRAY:
      itemp = SST_BEGG(RHS(3));
      (void)mkvarref(RHS(1), itemp);
      SST_PARENP(LHS, 0); /* ? */
      break;
    case ST_STFUNC:
      error(155, 3, gbl.lineno, "Illegal use of statement function -",
            SYMNAME(sptr));
      SST_ASTP(top, 0);
      break;
    default:
      error(155, 3, gbl.lineno, "Unknown symbol used in data clause -",
            SYMNAME(sptr));
      SST_ASTP(top, 0);
      break;
    }
    break;
  /*
   *	<accel data> ::= <accel data name> |
   */
  case ACCEL_DATA2:
  accel_data2:
    if (SST_IDG(RHS(1)) == S_IDENT || SST_IDG(RHS(1)) == S_DERIVED) {
      sptr = SST_SYMG(RHS(1));
    } else {
      sptr = SST_LSYMG(RHS(1));
    }
    if (STYPEG(sptr) == ST_PARAM || STYPEG(sptr) == ST_CONST) {
      error(155, 2, gbl.lineno, "Constant or Parameter used in data clause -",
            SYMNAME(sptr));
      SST_ASTP(LHS, 0);
    } else if (STYPEG(sptr) == ST_STFUNC) {
      error(155, 3, gbl.lineno, "Illegal use of statement function -",
            SYMNAME(sptr));
      SST_ASTP(LHS, 0);
    } else {
      if (SST_IDG(RHS(1)) == S_IDENT) {
        SST_ASTP(LHS, mk_id(sptr));
      } else {
        SST_ASTP(LHS, SST_ASTG(RHS(1)));
      }
    }
    break;
  /*
   *	<accel data> ::= <constant>
   */
  case ACCEL_DATA3:
    /* ignore constants; these sometimes come from preprocessor names getting
     * into data clauses */
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<accel data> ::= <common>
   */
  case ACCEL_DATA4:
    sptr = SST_SYMG(RHS(1));
    SST_SYMP(LHS, sptr);
    SST_DTYPEP(LHS, 0);
    SST_ASTP(LHS, mk_id(sptr));
    break;
  /*
   *	<accel data> ::= <accel data name> '<' <ident> '>' ( <accel sub list> )
   *|
   */
  case ACCEL_DATA5:
    break;
  /*
   *	<accel data> ::= <accel data name> '<' <ident> '>'
   */
  case ACCEL_DATA6:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdata list> ::= <accel mdata list> , <accel mdata> |
   */
  case ACCEL_MDATA_LIST1:
    goto accel_data_list1;
  /*
   *	<accel mdata list> ::= <accel mdata>
   */
  case ACCEL_MDATA_LIST2:
    goto accel_data_list2;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdata> ::= <accel mdata name> ( <accel sub list> ) |
   */
  case ACCEL_MDATA1:
    goto accel_data1;
  /*
   *	<accel mdata> ::= <accel mdata name> |
   */
  case ACCEL_MDATA2:
    goto accel_data2;
  /*
   *	<accel mdata> ::= <constant>
   */
  case ACCEL_MDATA3:
    SST_ASTP(LHS, 0);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<accel sdata list> ::= <accel sdata list> , <accel sdata> |
   */
  case ACCEL_SDATA_LIST1:
    goto accel_data_list1;
  /*
   *	<accel sdata list> ::= <accel sdata>
   */
  case ACCEL_SDATA_LIST2:
    goto accel_data_list2;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel sdata> ::= <accel sdata name> |
   */
  case ACCEL_SDATA1:
    goto accel_data2;
  /*
   *	<accel sdata> ::= <constant>
   */
  case ACCEL_SDATA2:
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel sub list> ::= <accel sub list> , <accel sub> |
   */
  case ACCEL_SUB_LIST1:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.stkp = SST_E1G(RHS(3));
    (SST_ENDG(RHS(1)))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;
  /*
   *	<accel sub list> ::= <accel sub>
   */
  case ACCEL_SUB_LIST2:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->t.stkp = SST_E1G(RHS(1));
    SST_BEGP(LHS, itemp);
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel sub> ::= <opt sub> : <opt sub> |
   */
  case ACCEL_SUB1:
    e1 = (SST *)getitem(sem.ssa_area, sizeof(SST));
    SST_IDP(e1, S_TRIPLE);
    SST_E1P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E1G(e1)) = *RHS(1);
    SST_E2P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E2G(e1)) = *RHS(3);
    SST_E3P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    SST_IDP(SST_E3G(e1), S_NULL);
    SST_E1P(LHS, e1);
    SST_E2P(LHS, 0);
    break;
  /*
   *	<accel sub> ::= <expression>
   */
  case ACCEL_SUB2:
    e1 = (SST *)getitem(sem.ssa_area, sizeof(SST));
    *e1 = *RHS(1);
    SST_E1P(LHS, e1);
    SST_E2P(LHS, 0);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<accel update dir> ::= ACCUPDATE <accel update list> |
   */
  case ACCEL_UPDATE_DIR1:
    break;
  /*
   *	<accel update dir> ::= ACCUPDATEHOST ( <accel data list> ) <opt update
   *list> |
   */
  case ACCEL_UPDATE_DIR2:
    clause = CL_UPDATEHOST;
    op = 3;
    goto acc_update_clause_shared;
    break;
  /*
   *	<accel update dir> ::= ACCUPDATESELF ( <accel data list> ) <opt update
   *list> |
   */
  case ACCEL_UPDATE_DIR3:
    clause = CL_UPDATESELF;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel update dir> ::= ACCUPDATEDEV ( <accel data list> ) <opt update
   *list> |
   */
  case ACCEL_UPDATE_DIR4:
    clause = CL_UPDATEDEV;
    op = 3;
    goto acc_update_clause_shared;
    break;
  /*
   *	<accel update dir> ::= ACCUPDATEIN ( <accel data list> ) <opt update
   *list> |
   */
  case ACCEL_UPDATE_DIR5:
    clause = CL_UPDATEDEV;
    op = 3;
    goto acc_update_clause_shared;
    break;
  /*
   *	<accel update dir> ::= ACCUPDATEOUT ( <accel data list> ) <opt update
   *list>
   */
  case ACCEL_UPDATE_DIR6:
    clause = CL_UPDATEHOST;
    op = 3;
    goto acc_update_clause_shared;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt update list> ::= |
   */
  case OPT_UPDATE_LIST1:
    break;
  /*
   *	<opt update list> ::= <accel update list>
   */
  case OPT_UPDATE_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt atomic type> ::= |
   */
  case OPT_ATOMIC_TYPE1:
    sem.mpaccatomic.action_type = ATOMIC_UPDATE;
    sem.mpaccatomic.mem_order = MO_UNDEF;
    break;

  /*
   *	<opt atomic type> ::= <pre seq_cst> UPDATE <post seq_cst> |
   */
  case OPT_ATOMIC_TYPE2:
    sem.mpaccatomic.action_type = ATOMIC_UPDATE;
    break;

  /*
   *	<opt atomic type> ::= <pre seq_cst> READ <post seq_cst> |
   */
  case OPT_ATOMIC_TYPE3:
    sem.mpaccatomic.action_type = ATOMIC_READ;
    break;

  /*
   *	<opt atomic type> ::= <pre seq_cst> WRITE <post seq_cst>  |
   */
  case OPT_ATOMIC_TYPE4:
    sem.mpaccatomic.action_type = ATOMIC_WRITE;
    break;

  /*
   *	<opt atomic type> ::= <pre seq_cst> CAPTURE <post seq_cst> |
   */
  case OPT_ATOMIC_TYPE5:
    sem.mpaccatomic.action_type = ATOMIC_CAPTURE;
    (void)enter_dir(DI_ATOMIC_CAPTURE, FALSE, 0, 0);
    break;
  /*
   *    <opt atomic type> ::= <seq cst>
   */
  case OPT_ATOMIC_TYPE6:
    sem.mpaccatomic.action_type = ATOMIC_UPDATE;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <pre seq_cst> ::= |
   */
  case PRE_SEQ_CST1:
    break;
  /*
   *    <pre seq_cst> ::= <seq cst> <opt comma>
   */
  case PRE_SEQ_CST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <post seq_cst> ::= |
   */
  case POST_SEQ_CST1:
    break;
  /*
   *    <post seq_cst> ::= <opt comma> <seq cst>
   */
  case POST_SEQ_CST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *    <seq cst> ::= SEQ_CST
   */
  case SEQ_CST1:
    sem.mpaccatomic.mem_order = MO_SEQ_CST;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel update list> ::= <accel update attr> |
   */
  case ACCEL_UPDATE_LIST1:
    break;
  /*
   *	<accel update list> ::= <accel update list> <opt comma> <accel update
   *attr>
   */
  case ACCEL_UPDATE_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel update attr> ::=  HOST ( <accel data list> ) |
   */
  case ACCEL_UPDATE_ATTR1:
    clause = CL_UPDATEHOST;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel update attr> ::= SELF ( <accel data list> ) |
   */
  case ACCEL_UPDATE_ATTR2:
    clause = CL_UPDATESELF;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel update attr> ::= DEVICE ( <accel data list> )
   */
  case ACCEL_UPDATE_ATTR3:
    clause = CL_UPDATEDEV;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel update attr> ::= ACCIN ( <accel data list> ) |
   */
  case ACCEL_UPDATE_ATTR4:
    clause = CL_UPDATEDEV;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel update attr> ::= ACCOUT ( <accel data list> )
   */
  case ACCEL_UPDATE_ATTR5:
    clause = CL_UPDATEHOST;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel update attr> ::= IF ( <expression> ) |
   */
  case ACCEL_UPDATE_ATTR6:
    break;
  /*
   *	<accel update attr> ::= ASYNC |
   */
  case ACCEL_UPDATE_ATTR7:
    break;
  /*
   *	<accel update attr> ::= ASYNC ( <expression> )
   */
  case ACCEL_UPDATE_ATTR8:
    break;
  /*
   *	<accel update attr> ::= DEVICEID ( <expression> )
   */
  case ACCEL_UPDATE_ATTR9:
    break;
  /*
   *	<accel update attr> ::= ACCWAIT |
   */
  case ACCEL_UPDATE_ATTR10:
    break;
  /*
   *	<accel update attr> ::= ACCWAIT ( <accel wait list> )
   */
  case ACCEL_UPDATE_ATTR11:
    break;
  /*
   *	<accel update attr> ::= ACCIFPRESENT
   */
  case ACCEL_UPDATE_ATTR12:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel short update> ::= ACCUPDATEHOST ( <accel data list> )  |
   */
  case ACCEL_SHORT_UPDATE1:
    clause = CL_UPDATEHOST;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel short update> ::= ACCUPDATESELF ( <accel data list> )  |
   */
  case ACCEL_SHORT_UPDATE2:
    clause = CL_UPDATESELF;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel short update> ::= ACCUPDATEDEV ( <accel data list> )  |
   */
  case ACCEL_SHORT_UPDATE3:
    clause = CL_UPDATEDEV;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel short update> ::= ACCUPDATEIN ( <accel data list> )  |
   */
  case ACCEL_SHORT_UPDATE4:
    clause = CL_UPDATEDEV;
    op = 3;
    goto acc_update_clause_shared;
  /*
   *	<accel short update> ::= ACCUPDATEOUT ( <accel data list> )
   */
  case ACCEL_SHORT_UPDATE5:
    clause = CL_UPDATEHOST;
    op = 3;
    goto acc_update_clause_shared;
  /* ------------------------------------------------------------------ */
  /*
   *	<opt wait list> ::= |
   */
  case OPT_WAIT_LIST1:
    /* begin the pragma before the first argument */
    add_pragma(PR_ACCBEGINDIR, PR_NOSCOPE, 0);
    break;
  /*
   *	<opt wait list> ::= ( <accel wait list> ) |
   */
  case OPT_WAIT_LIST2:
    break;
  /*
   *	<opt wait list> ::= <opt wait list> <wait item>
   */
  case OPT_WAIT_LIST3:
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<wait item> ::= IF ( <expression> )
   */
  case WAIT_ITEM1:
    break;
  /*
   *	<wait item> ::= DEVICEID ( <expression> )
   */
  case WAIT_ITEM2:
    break;
  /*
   *	<wait item> ::= ASYNC |
   */
  case WAIT_ITEM3:
    break;
  /*
   *	<wait item> ::= ASYNC ( <expression> )
   */
  case WAIT_ITEM4:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel wait list> ::= <expression> |
   */
  case ACCEL_WAIT_LIST1:
    break;
  /*
   *	<accel wait list> ::= <accel wait list> , <expression>
   */
  case ACCEL_WAIT_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel stmt> ::= <kernel begin> KERNEL DO <kernel do list>
   */
  case KERNEL_STMT1:
    SST_ASTP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel begin> ::=
   */
  case KERNEL_BEGIN1:
    parstuff_init();
    kernel_do_nest = 1;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel do list> ::= <kernel do nest> <kernel do shape> <kernel do args>
   */
  case KERNEL_DO_LIST1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel do nest> ::= |
   */
  case KERNEL_DO_NEST1:
    kernel_do_nest = 1;
    break;
  /*
   *	<kernel do nest> ::= ( ) |
   */
  case KERNEL_DO_NEST2:
    kernel_do_nest = 1;
    break;
  /*
   *	<kernel do nest> ::= ( <expression> )
   */
  case KERNEL_DO_NEST3:
    kernel_do_nest = chkcon_to_isz(RHS(2), TRUE);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel do shape> ::= |
   */
  case KERNEL_DO_SHAPE1:
    CL_FIRST(CL_KERNEL_GRID) = NULL;
    CL_FIRST(CL_KERNEL_BLOCK) = NULL;
    break;
  /*
   *	<kernel do shape> ::= '<<<' '>>>' |
   */
  case KERNEL_DO_SHAPE2:
    CL_FIRST(CL_KERNEL_GRID) = NULL;
    CL_FIRST(CL_KERNEL_BLOCK) = NULL;
    break;
  /*
   *	<kernel do shape> ::= '<<<' <kernel do grid shape> , <kernel do block
   *shape> <kernel do args> '>>>'
   */
  case KERNEL_DO_SHAPE3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel do grid shape> ::= |
   */
  case KERNEL_DO_GRID_SHAPE1:
    CL_FIRST(CL_KERNEL_GRID) = NULL;
    break;
  /*
   *	<kernel do grid shape> ::= * |
   */
  case KERNEL_DO_GRID_SHAPE2:
    CL_FIRST(CL_KERNEL_GRID) = NULL;
    break;
  /*
   *	<kernel do grid shape> ::= <expression> |
   */
  case KERNEL_DO_GRID_SHAPE3:
    chk_scalartyp(RHS(1), DT_INT4, FALSE);
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = SST_ASTG(RHS(1));
    CL_FIRST(CL_KERNEL_GRID) = itemp;
    CL_LAST(CL_KERNEL_GRID) = itemp;
    CL_PRESENT(CL_KERNEL_GRID) = 1;
    break;
  /*
   *	<kernel do grid shape> ::= <elp> ) |
   */
  case KERNEL_DO_GRID_SHAPE4:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = mk_cval(1, DT_INT);
    CL_FIRST(CL_KERNEL_GRID) = itemp;
    CL_LAST(CL_KERNEL_GRID) = itemp;
    CL_PRESENT(CL_KERNEL_GRID) = 1;
    break;
  /*
   *	<kernel do grid shape> ::= <elp> * ) |
   */
  case KERNEL_DO_GRID_SHAPE5:
    CL_FIRST(CL_KERNEL_GRID) = NULL;
    break;
  /*
   *	<kernel do grid shape> ::= <elp> <kernel shape list> )
   */
  case KERNEL_DO_GRID_SHAPE6:
    CL_FIRST(CL_KERNEL_GRID) = SST_BEGG(RHS(2));
    CL_LAST(CL_KERNEL_GRID) = SST_ENDG(RHS(2));
    CL_PRESENT(CL_KERNEL_GRID) = 1;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel do block shape> ::= |
   */
  case KERNEL_DO_BLOCK_SHAPE1:
    CL_FIRST(CL_KERNEL_BLOCK) = NULL;
    break;
  /*
   *	<kernel do block shape> ::= * |
   */
  case KERNEL_DO_BLOCK_SHAPE2:
    CL_FIRST(CL_KERNEL_BLOCK) = NULL;
    break;
  /*
   *	<kernel do block shape> ::= <expression> |
   */
  case KERNEL_DO_BLOCK_SHAPE3:
    chk_scalartyp(RHS(1), DT_INT4, FALSE);
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = SST_ASTG(RHS(1));
    CL_FIRST(CL_KERNEL_BLOCK) = itemp;
    CL_LAST(CL_KERNEL_BLOCK) = itemp;
    CL_PRESENT(CL_KERNEL_BLOCK) = 1;
    break;
  /*
   *	<kernel do block shape> ::= <elp> ) |
   */
  case KERNEL_DO_BLOCK_SHAPE4:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = mk_cval(1, DT_INT);
    CL_FIRST(CL_KERNEL_BLOCK) = itemp;
    CL_LAST(CL_KERNEL_BLOCK) = itemp;
    CL_PRESENT(CL_KERNEL_BLOCK) = 1;
    break;
  /*
   *	<kernel do block shape> ::= <elp> * ) |
   */
  case KERNEL_DO_BLOCK_SHAPE5:
    CL_FIRST(CL_KERNEL_BLOCK) = NULL;
    break;
  /*
   *	<kernel do block shape> ::= <elp> <kernel shape list> )
   */
  case KERNEL_DO_BLOCK_SHAPE6:
    CL_FIRST(CL_KERNEL_BLOCK) = SST_BEGG(RHS(2));
    CL_LAST(CL_KERNEL_BLOCK) = SST_ENDG(RHS(2));
    CL_PRESENT(CL_KERNEL_BLOCK) = 1;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel shape list> ::= <kernel shape expr> , <kernel shape expr> |
   */
  case KERNEL_SHAPE_LIST1:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next->next = ITEM_END;
    itemp->ast = SST_ASTG(RHS(1));
    itemp->next->ast = SST_ASTG(RHS(3));
    SST_BEGP(LHS, itemp);
    SST_ENDP(LHS, itemp->next);
    break;
  /*
   *	<kernel shape list> ::= <kernel shape list> , <kernel shape expr>
   */
  case KERNEL_SHAPE_LIST2:
    itemp = (ITEM *)getitem(0, sizeof(ITEM));
    itemp->next = ITEM_END;
    itemp->ast = SST_ASTG(RHS(3));
    (SST_ENDG(RHS(1)))->next = itemp;
    SST_ENDP(LHS, itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel shape expr> ::= |
   */
  case KERNEL_SHAPE_EXPR1:
    SST_ASTP(LHS, mk_cval(1, DT_INT));
    break;
  /*
   *	<kernel shape expr> ::= * |
   */
  case KERNEL_SHAPE_EXPR2:
    SST_ASTP(LHS, new_node(A_NULL));
    break;
  /*
   *	<kernel shape expr> ::= <expression>
   */
  case KERNEL_SHAPE_EXPR3:
    chk_scalartyp(RHS(1), DT_INT, TRUE);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel do args> ::= |
   */
  case KERNEL_DO_ARGS1:
    kernel_argnum = 0;
    break;
  /*
   *	<kernel do args> ::= <kernel do args> , <kernel do arg>
   */
  case KERNEL_DO_ARGS2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<kernel do arg> ::= |
   */
  case KERNEL_DO_ARG1:
    if (kernel_argnum >= 0)
      ++kernel_argnum;
    break;
  /*
   *	<kernel do arg> ::= <expression> |
   */
  case KERNEL_DO_ARG2:
    if (kernel_argnum < 0) {
      error(155, 3, gbl.lineno,
            "Non-keyword CUF KERNEL DO argument may not "
            "follow a keyword argument",
            NULL);
    } else {
      ++kernel_argnum;
      /* first argument is shared memory, ignore it (should be zero)
       * second argument is stream argument */
      if (kernel_argnum == 1) {
        int astx, sptr;
        /* must be zero */
        astx = SST_ASTG(RHS(1));
        if (astx && A_TYPEG(astx) != A_CNST) {
          error(155, 3, gbl.lineno,
                "Shared memory value for CUF KERNEL DO must be zero", NULL);
        } else if (astx) {
          sptr = A_SPTRG(astx);
          if (STYPEG(sptr) != ST_CONST) {
            error(155, 3, gbl.lineno,
                  "Shared memory value for CUF KERNEL DO must be zero", NULL);
          } else if (!DT_ISINT(DTYPEG(sptr))) {
            error(155, 3, gbl.lineno,
                  "Shared memory value for CUF KERNEL DO must be zero", NULL);
          } else if (CONVAL1G(sptr) != 0 || CONVAL2G(sptr) != 0) {
            error(155, 3, gbl.lineno,
                  "Shared memory value for CUF KERNEL DO must be zero", NULL);
          }
        }
      } else if (kernel_argnum == 2) {
        chk_scalartyp(RHS(1), DT_INT8, FALSE);
        itemp = (ITEM *)getitem(0, sizeof(ITEM));
        itemp->next = ITEM_END;
        itemp->ast = SST_ASTG(RHS(1));
        CL_FIRST(CL_STREAM) = itemp;
        CL_LAST(CL_STREAM) = itemp;
        CL_PRESENT(CL_STREAM) = 1;
      } else {
        error(155, 3, gbl.lineno, "Too many arguments in CUF KERNEL DO <<< >>>",
              NULL);
      }
    }
    break;
  /*
   *	<kernel do arg> ::= <id name> = <expression>
   */
  case KERNEL_DO_ARG3:
    kernel_argnum = -1;
    if (strcmp(scn.id.name + SST_CVALG(RHS(1)), "stream") == 0) {
      if (CL_PRESENT(CL_STREAM)) {
        error(155, 3, gbl.lineno,
              "Two STREAM arguments to CUF KERNEL directive", "");
      } else {
        chk_scalartyp(RHS(3), DT_INT8, FALSE);
        itemp = (ITEM *)getitem(0, sizeof(ITEM));
        itemp->next = ITEM_END;
        itemp->ast = SST_ASTG(RHS(3));
        CL_FIRST(CL_STREAM) = itemp;
        CL_LAST(CL_STREAM) = itemp;
        CL_PRESENT(CL_STREAM) = 1;
      }
    } else if (strcmp(scn.id.name + SST_CVALG(RHS(1)), "device") == 0) {
      if (CL_PRESENT(CL_DEVICE)) {
        error(155, 3, gbl.lineno,
              "Two DEVICE arguments to CUF KERNEL directive", "");
      } else {
        chk_scalartyp(RHS(3), DT_INT8, FALSE);
        itemp = (ITEM *)getitem(0, sizeof(ITEM));
        itemp->next = ITEM_END;
        itemp->ast = SST_ASTG(RHS(3));
        CL_FIRST(CL_DEVICE) = itemp;
        CL_LAST(CL_DEVICE) = itemp;
        CL_PRESENT(CL_DEVICE) = 1;
      }
    } else {
      error(155, 3, gbl.lineno, "Unknown keyword to CUF KERNEL directive -",
            scn.id.name + SST_CVALG(RHS(1)));
    }
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<opt accel init list> ::= |
   */
  case OPT_ACCEL_INIT_LIST1:
    break;
  /*
   *	<opt accel init list> ::= <opt accel init list> <opt comma> <acc init
   *attr> |
   */
  case OPT_ACCEL_INIT_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<acc init attr> ::= DEVICE_NUM ( <expression> )
   */
  case ACC_INIT_ATTR1:
    break;
  /*
   *	<acc init attr> ::= DEVICE_TYPE ( <devtype list> ) |
   */
  case ACC_INIT_ATTR2:
    break;
  /* ------------------------------------------------------------------ */
  /*
   *	<accel setdev dir> ::= ACCSET <accel setdev list>
   */
  case ACCEL_SETDEV_DIR1:
    break;
  /*
   *	<accel setdev list> ::= <accel setdev attr> |
   */
  case ACCEL_SETDEV_LIST1:
    break;
  /*
   *	<accel setdev list> ::= <accel setdev list> <opt comma> <accel setdev
   *attr>
   */
  case ACCEL_SETDEV_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel setdev attr> ::= DEVICE_TYPE ( <ident> ) |
   */
  case ACCEL_SETDEV_ATTR1:
    break;
  /*
   *	<accel setdev attr> ::= DEVICE_NUM ( <expression> ) |
   */
  case ACCEL_SETDEV_ATTR2:
    break;
  /*
   *	<accel setdev attr> ::= DEFAULT_ASYNC ( <expression> )
   */
  case ACCEL_SETDEV_ATTR3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<opt accel shutdown list> ::= |
   */
  case OPT_ACCEL_SHUTDOWN_LIST1:
    break;
  /*
   *	<opt accel shutdown list> ::= <opt accel shutdown list> <opt comma> <acc
   *shutdown attr> |
   */
  case OPT_ACCEL_SHUTDOWN_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<acc shutdown attr> ::= DEVICE_NUM ( <expression> ) |
   */
  case ACC_SHUTDOWN_ATTR1:
    break;
  /*
   *	<acc shutdown attr> ::= DEVICE_TYPE ( <devtype list> )
   */
  case ACC_SHUTDOWN_ATTR2:
    break;

  /*
   * <accel compare dir> ::= COMPARE ( <accel data list> )
   */
  case ACCEL_COMPARE_DIR1:
    clause = CL_ACCCOMPARE;
    op = 3;
    add_clause(clause, FALSE);
    if (CL_FIRST(clause) == NULL)
      CL_FIRST(clause) = SST_BEGG(RHS(op));
    else
      ((ITEM *)CL_LAST(clause))->next = SST_BEGG(RHS(op));
    CL_LAST(clause) = SST_ENDG(RHS(op));
    break;

  /*
   * <pgi compare dir> ::= PGICOMPARE ( <accel data list> )
   */
  case PGI_COMPARE_DIR1:
    clause = CL_PGICOMPARE;
    op = 3;
    add_clause(clause, FALSE);
    if (CL_FIRST(clause) == NULL)
      CL_FIRST(clause) = SST_BEGG(RHS(op));
    else
      ((ITEM *)CL_LAST(clause))->next = SST_BEGG(RHS(op));
    CL_LAST(clause) = SST_ENDG(RHS(op));
    break;
  /* ------------------------------------------------------------------ */
  default:
    interr("semsmp: bad rednum ", rednum, 3);
    break;
  }
}

void
parstuff_init(void)
{
  int i;

  chunk = 0;
  distchunk = 0;
  for (i = 0; i < CL_MAXV; i++) {
    CL_PRESENT(i) = 0;
    CL_VAL(i) = 0;
    CL_FIRST(i) = NULL;
    CL_LAST(i) = NULL;
  }
  recent_loop_clause = 0;
  any_pflsr_private = FALSE;
  mp_iftype = IF_DEFAULT;
}

static void
add_clause(int clause, LOGICAL one_only)
{
  if (CL_PRESENT(clause)) {
    if (one_only)
      error(155, 3, gbl.lineno, "Repeated clause -", CL_NAME(clause));
  } else
    CL_PRESENT(clause) = 1;
}

static bool
clause_errchk(BIGINT64 bt, const char *dirname)
{
  int i;
  bool any = false;

  for (i = 0; i < CL_MAXV; i++)
    if (CL_PRESENT(i)) {
      any = true;
      if (!(CL_STMT(i) & bt))
        error(533, 3, gbl.lineno, CL_NAME(i), dirname);
    }
  return any;
}

static void
add_pragma(int pragmatype, int pragmascope, int pragmaarg)
{
  int ast;

  ast = mk_stmt(A_PRAGMA, 0);
  A_PRAGMATYPEP(ast, pragmatype);
  A_PRAGMASCOPEP(ast, pragmascope);
  A_LOPP(ast, pragmaarg);
  (void)add_stmt(ast);
}

#ifdef FLANG_SEMSMP_UNUSED
static void
add_pragma2(int pragmatype, int pragmascope, int pragmaarg, int pragmaarg2)
{
  int ast;

  ast = mk_stmt(A_PRAGMA, 0);
  A_PRAGMATYPEP(ast, pragmatype);
  A_PRAGMASCOPEP(ast, pragmascope);
  A_LOPP(ast, pragmaarg);
  A_ROPP(ast, pragmaarg2);
  (void)add_stmt(ast);
}
#endif

#ifdef FLANG_SEMSMP_UNUSED
static void
add_pragma3(int pragmatype, int pragmascope, int pragmaarg, int pragmaarg2,
            int pragmaarg3)
{
  int ast;

  ast = mk_stmt(A_PRAGMA, 0);
  A_PRAGMATYPEP(ast, pragmatype);
  A_PRAGMASCOPEP(ast, pragmascope);
  A_LOPP(ast, pragmaarg);
  A_ROPP(ast, pragmaarg2);
  A_PRAGMAARGP(ast, pragmaarg3);
  (void)add_stmt(ast);
}
#endif

#ifdef FLANG_SEMSMP_UNUSED
static void
add_pragmasyms(int pragmatype, int pragmascope, ITEM *itemp, int docopy)
{
  int prtype = pragmatype;
  for (; itemp != ITEM_END; itemp = itemp->next) {
    int sptr, ast_devcopy = 0;
    sptr = memsym_of_ast(itemp->ast);
    if (docopy)
      prtype = itemp->t.cltype;
#ifdef DEVCOPYG
    if (DEVCOPYG(sptr)) {
      if ((sem.parallel || sem.task || sem.target || sem.teams)) {
        int stblk;
        stblk = BLK_UPLEVEL_SPTR(sem.scope_level);
        if (!stblk)
          stblk = get_stblk_uplevel_sptr();
        mp_add_shared_var(DEVCOPYG(sptr), stblk);
      }
      ast_devcopy = mk_id(DEVCOPYG(sptr));
    }
#endif
      add_pragma2(prtype, pragmascope, itemp->ast, ast_devcopy);
  }
}
#endif

#ifdef FLANG_SEMSMP_UNUSED
static void
add_reduction_pragmas(void)
{
  REDUC *reducp;
  REDUC_SYM *reduc_symp;
  int accreduct_op, ast;
  char *name;

  for (reducp = CL_FIRST(CL_REDUCTION); reducp; reducp = reducp->next) {
    switch (reducp->opr) {
    case 0: /* intrinsic */
      name = SYMNAME(reducp->intrin);
      if (strcmp(name, "max") == 0) {
        accreduct_op = PR_ACCREDUCT_OP_MAX;
        break;
      }
      if (strcmp(name, "min") == 0) {
        accreduct_op = PR_ACCREDUCT_OP_MIN;
        break;
      }
      if (strcmp(name, "iand") == 0) {
        accreduct_op = PR_ACCREDUCT_OP_BITAND;
        break;
      }
      if (strcmp(name, "ior") == 0) {
        accreduct_op = PR_ACCREDUCT_OP_BITIOR;
        break;
      }
      if (strcmp(name, "ieor") == 0) {
        accreduct_op = PR_ACCREDUCT_OP_BITEOR;
        break;
      }
      interr("add_reduction_pragmas - illegal intrinsic", reducp->intrin, 3);
      break;
    case OP_SUB:
    /* OP_SUB reduction operator: flag as PGI extension? */
    case OP_ADD:
      accreduct_op = PR_ACCREDUCT_OP_ADD;
      break;
    case OP_MUL:
      accreduct_op = PR_ACCREDUCT_OP_MUL;
      break;
    case OP_LOG:
      switch (reducp->intrin) {
      case OP_LAND:
        accreduct_op = PR_ACCREDUCT_OP_LOGAND;
        break;
      case OP_LEQV:
        accreduct_op = PR_ACCREDUCT_OP_EQV;
        break;
      case OP_LOR:
        accreduct_op = PR_ACCREDUCT_OP_LOGOR;
        break;
      case OP_LNEQV:
        accreduct_op = PR_ACCREDUCT_OP_NEQV;
        break;
      default:
        interr("add_reduction_pragmas - illegal log operator", reducp->intrin,
               3);
      }
      break;
    default:
      interr("add_reduction_pragmas - illegal operator", reducp->opr, 3);
      break;
    }
    ast = mk_stmt(A_PRAGMA, 0);
    A_PRAGMATYPEP(ast, PR_ACCREDUCTOP);
    A_PRAGMASCOPEP(ast, PR_NOSCOPE);
    A_PRAGMAVALP(ast, accreduct_op);
    (void)add_stmt(ast);
    for (reduc_symp = reducp->list; reduc_symp; reduc_symp = reduc_symp->next) {
      if (reduc_symp->shared == 0)
        /* error - illegal reduction variable */
        continue;
      add_pragma(PR_ACCREDUCTION, PR_NOSCOPE, mk_id(reduc_symp->shared));
    }
  }
}
#endif

#ifdef FLANG_SEMSMP_UNUSED
static void
add_wait_pragmas(ITEM *itemp)
{
  if (!itemp) {
    add_pragma(PR_ACCWAIT, PR_NOSCOPE, 0);
  } else {
    for (; itemp; itemp = itemp->next) {
      add_pragma(PR_ACCWAITARG, PR_NOSCOPE, itemp->ast);
    }
  }
  CL_PRESENT(CL_WAIT) = 0;
} /* add_wait_pragmas */
#endif

static void
accel_pragmagen(int pragma, int pragma1, int pragma2)
{
}

static int
sched_type(char *nm)
{
  /* allow both the openmp & SGI schedule spellings */
  if (sem_strcmp(nm, "static") == 0 || sem_strcmp(nm, "simple") == 0 ||
      sem_strcmp(nm, "interleave") == 0 || sem_strcmp(nm, "interleaved") == 0)
    return DI_SCH_STATIC;

  if (sem_strcmp(nm, "dynamic") == 0)
    return DI_SCH_DYNAMIC;

  if (sem_strcmp(nm, "guided") == 0 || sem_strcmp(nm, "gss") == 0)
    return DI_SCH_GUIDED;

  if (sem_strcmp(nm, "runtime") == 0)
    return DI_SCH_RUNTIME;

  if (sem_strcmp(nm, "auto") == 0)
    return DI_SCH_AUTO;

  error(34, 3, gbl.lineno, nm, CNULL);
  return DI_SCH_STATIC;
}

/* return 1: parallel
          2: do
          3: taskgroup
          4: sections
          0: unknown
*/
static int
cancel_type(char *nm)
{
  if (sem_strcmp(nm, "parallel") == 0)
    return 1;
  else if (sem_strcmp(nm, "do") == 0)
    return 2;
  else if (sem_strcmp(nm, "sections") == 0)
    return 3;
  else if (sem_strcmp(nm, "taskgroup") == 0)
    return 4;
  else
    error(155, 3, gbl.lineno,
          "Unknown construct-type-clause in CANCEL construct", NULL);
  return 0;
}

static int
get_iftype(int argcnt, char *nm1, char *nm2, char *nm3)
{
  switch (argcnt) {
  case 3:
    if (strcmp(nm1, "target") == 0) {
      if (strcmp(nm2, "exit") == 0) {
        if (strcmp(nm3, "data") == 0)
          return IF_TARGETEXITDATA;
        else
          error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm3);
      } else if (strcmp(nm2, "enter") == 0) {
        if (strcmp(nm3, "data") == 0)
          return IF_TARGETENTERDATA;
        else
          error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm3);
      } else {
        error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm2);
      }
    } else {
      error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm1);
    }
    break;
  case 2:
    if (strcmp(nm1, "target") == 0) {
      if (strcmp(nm2, "update") == 0)
        return IF_TARGETUPDATE;
      else if (strcmp(nm2, "data") == 0)
        return IF_TARGETDATA;
      else
        error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm2);
    } else {
      error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm1);
    }
    break;
  case 1:
  default:
    if (strcmp(nm1, "parallel") == 0)
      return IF_PARALLEL;
    else if (strcmp(nm1, "target") == 0)
      return IF_TARGET;

    else if (strcmp(nm1, "task") == 0)
      return IF_TASK;
    else if (strcmp(nm1, "taskloop") == 0)
      return IF_TASKLOOP;
    else
      error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm1);
  }
  error(155, ERR_Severe, gbl.lineno, "Unknown directive modifier", nm1);
  return 0;
}

static void
set_iftype(int argcnt, char *nm, char *nm2, char *nm3)
{
  /* now check nm against current region */

  int prev;
  int type;
  prev = sem.doif_depth;

  type = get_iftype(argcnt, nm, nm2, nm3);

  if (CL_PRESENT(CL_IF)) {
    if (mp_iftype == IF_DEFAULT)
      error(155, ERR_Severe, gbl.lineno,
            "At most one IF without directive-name-modifier can be present",
            NULL);
    else if (type == IF_DEFAULT && mp_iftype != IF_DEFAULT)
      error(155, ERR_Severe, gbl.lineno,
            "All IF must have directive-name-modifier", NULL);
    else if ((mp_iftype & type) == type)
      error(155, ERR_Severe, gbl.lineno,
            "At most one IF with same directive-name-modifier can be present",
            NULL);
  }
  mp_iftype = mp_iftype | type;
}

#ifdef FLANG_SEMSMP_UNUSED
static void
validate_if(int type, char *nm)
{
  if (!CL_PRESENT(CL_IF)) {
    return;
  }
  switch (type) {
  case IF_TARGET:
    if ((mp_iftype != IF_DEFAULT) && (mp_iftype & type) != IF_TARGET) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            NULL);
    }
    break;
  case IF_TARGETUPDATE:
    if ((mp_iftype != IF_DEFAULT) && mp_iftype != IF_TARGETUPDATE) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            NULL);
    }
    break;
  case IF_TARGETDATA:
    if ((mp_iftype != IF_DEFAULT) && mp_iftype != IF_TARGETDATA) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            NULL);
    }
    break;
  case IF_TARGETENTERDATA:
    if ((mp_iftype != IF_DEFAULT) && mp_iftype != IF_TARGETENTERDATA) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            NULL);
    }
    break;
  case IF_TARGETEXITDATA:
    if ((mp_iftype != IF_DEFAULT) && mp_iftype != IF_TARGETEXITDATA) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            CNULL);
    }
    break;
  case IF_PARALLEL:
    if ((mp_iftype != IF_DEFAULT) && (mp_iftype & type) != IF_PARALLEL) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            NULL);
    }
    break;
  case IF_TASK:
    if ((mp_iftype != IF_DEFAULT) && (mp_iftype & type) != IF_TASK) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            NULL);
    }
    break;
  case IF_TASKLOOP:
    if ((mp_iftype != IF_DEFAULT) && (mp_iftype & type) != IF_TASKLOOP) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
            NULL);
    }
    break;
  default:
    if (type == (IF_TARGET | IF_PARALLEL)) {
      if ((mp_iftype != IF_DEFAULT) &&
          ((mp_iftype & type) != (IF_TARGET | IF_PARALLEL))) {
        error(155, ERR_Severe, gbl.lineno, "Unexpected directive in if clause",
              NULL);
      }
    }
  }
}
#endif

static int
get_stblk_uplevel_sptr()
{
  int i;
  int sptr = BLK_UPLEVEL_SPTR(sem.scope_level);
  int scope = BLK_SCOPE_SPTR(sem.scope_level);

  if (sptr == 0) {
    i = sem.scope_level;
    if (i > 0) {
      while (sptr == 0 && i) {
        sptr = BLK_UPLEVEL_SPTR(i);
        scope = BLK_SCOPE_SPTR(i);
        --i;
      }
      if (sptr == 0) {
        /* If we get here, this should really be an error */
        sptr = getccsym('b', sem.blksymnum++, ST_BLOCK);
        PARSYMSP(sptr, 0);
        llmp_create_uplevel(sptr);
        BLK_UPLEVEL_SPTR(sem.scope_level) = sptr;
        if (scope) {
          BLK_SCOPE_SPTR(sem.scope_level) = scope;
          PARUPLEVELP(scope, sptr);
        }
      } else {
        BLK_UPLEVEL_SPTR(sem.scope_level) = sptr;
        BLK_SCOPE_SPTR(sem.scope_level) = scope;
      }
    }
  }
  return sptr;
}

static int
emit_btarget(int atype)
{
  int ast;

  ast = mk_stmt(atype, 0);
  sem.target++;
  if (CL_PRESENT(CL_IF)) {
    if (mp_iftype != OMP_DEFAULT && (mp_iftype & OMP_TARGET) != OMP_TARGET)
      error(155, 3, gbl.lineno,
            "IF (target:) or IF is expected in TARGET or "
            "combined TARGET construct ",
            NULL);
    else
      A_IFPARP(ast, CL_VAL(CL_IF));
  }
  if (CL_PRESENT(CL_DEPEND)) {
  }
  if (CL_PRESENT(CL_NOWAIT)) {
  }
  (void)add_stmt(ast);
  return ast;
}

int
emit_etarget()
{
  int ast;
  ast = mk_stmt(A_MP_ENDTARGET, 0);

  if (sem.target < 0)
    sem.target = 0;
  (void)add_stmt(ast);
  return ast;
}

static int
emit_bpar(void)
{
  int opc;
  int ast;

  if (sem.parallel++ == 0) {
    /* outermost parallel */
    opc = A_MP_PARALLEL;
  } else
    /* nested parallel */
    opc = A_MP_PARALLEL;

  ast = mk_stmt(A_MP_PARALLEL, 0);
  A_ENDLABP(ast, 0);
  if (CL_PRESENT(CL_IF)) {
    if (mp_iftype != OMP_DEFAULT && (mp_iftype & OMP_PARALLEL) != OMP_PARALLEL)
      error(155, 3, gbl.lineno,
            "IF (parallel:) or IF is expected in PARALLEL "
            "or combined PARALLEL construct ",
            NULL);
    else
      A_IFPARP(ast, CL_VAL(CL_IF));
  }
  if (CL_PRESENT(CL_NUM_THREADS))
    A_NPARP(ast, CL_VAL(CL_NUM_THREADS));

  /* PROC_BIND ast should be constant value */
  if (CL_PRESENT(CL_PROC_BIND)) {
    A_PROCBINDP(ast, CL_VAL(CL_PROC_BIND));
  }

  (void)add_stmt(ast);
  return ast;
}

int
emit_epar(void)
{
  int opc;
  int ast;

  if (sem.parallel == 0)
    /* outermost parallel region */
    opc = A_MP_ENDPARALLEL;
  else {
    /* nested parallel */
    opc = A_MP_ENDPARALLEL;
    if (sem.parallel < 0)
      sem.parallel = 0;
  }
  ast = mk_stmt(A_MP_ENDPARALLEL, 0);
  (void)add_stmt(ast);
  return ast;
}

int
emit_bcs_ecs(int opc)
{
  int ast;
#if DEBUG
  assert(opc == A_MP_CRITICAL || opc == A_MP_ENDCRITICAL,
         "emit_bcs_ecs - illegal opc", opc, 3);
#endif
  ast = 0;
  /* If already in a critical section, don't create another one */
  if (DI_IN_NEST(sem.doif_depth, DI_CRITICAL) == 0) {
    ast = mk_stmt(opc, 0);
    (void)add_stmt(ast);
  }
  return ast;
}

static void
do_schedule(int doif)
{
  if (doif == 0)
    return;

  DI_DISTCHUNK(doif) = 0;
  if (CL_PRESENT(CL_SCHEDULE) || CL_PRESENT(CL_MP_SCHEDTYPE)) {
    DI_SCHED_TYPE(doif) = CL_VAL(CL_SCHEDULE);
    if (chunk) {
      if (DI_SCHED_TYPE(doif) == DI_SCH_RUNTIME ||
          DI_SCHED_TYPE(doif) == DI_SCH_AUTO) {
        error(155, 3, gbl.lineno,
              "chunk size not allowed with SCHEDULE AUTO or RUNTIME", NULL);
        DI_CHUNK(doif) = 0;
      } else if (A_ALIASG(chunk))
        DI_CHUNK(doif) = chunk;
      else {
        int sptr;
        int ast;
        sptr = get_itemp(DT_INT);
        ENCLFUNCP(sptr, BLK_SYM(sem.scope_level));
        set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));

        ast = mk_assn_stmt(mk_id(sptr), chunk, DT_INT);
        (void)add_stmt(ast);
        DI_CHUNK(doif) = A_DESTG(ast);
      }
    } else
      DI_CHUNK(doif) = 0;
  } else {
    /* default schedule */
    DI_SCHED_TYPE(doif) = DI_SCH_STATIC;
    if (XBIT(69, 0x08))
      DI_SCHED_TYPE(doif) = DI_SCH_DYNAMIC;
    else if (XBIT(69, 0x10))
      DI_SCHED_TYPE(doif) = DI_SCH_GUIDED;
    else if (XBIT(69, 0x20))
      DI_SCHED_TYPE(doif) = DI_SCH_RUNTIME;
    DI_CHUNK(doif) = 0;
  }
  DI_IS_ORDERED(doif) = CL_PRESENT(CL_ORDERED);
  DI_ISSIMD(doif) = 0;
  sem.collapse = 0;
  if (CL_PRESENT(CL_COLLAPSE)) {
    sem.collapse = CL_VAL(CL_COLLAPSE);
  }
}

static void
do_dist_schedule(int doif, LOGICAL chk_collapse)
{
  int ast;
  if (doif == 0)
    return;
  if (CL_PRESENT(CL_DIST_SCHEDULE)) {
    if (distchunk) {
      if (A_ALIASG(distchunk))
        DI_DISTCHUNK(doif) = distchunk;
      else {
        int sptr;
        sptr = get_itemp(DT_INT);
        ENCLFUNCP(sptr, BLK_SYM(sem.scope_level));
        set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));

        ast = mk_assn_stmt(mk_id(sptr), distchunk, DT_INT);
        (void)add_stmt(ast);
        DI_DISTCHUNK(doif) = A_DESTG(ast);
      }
    } else
      DI_DISTCHUNK(doif) = 0;
  } else {
    DI_DISTCHUNK(doif) = 0;
  }
  DI_CHUNK(doif) = DI_DISTCHUNK(doif);
  DI_SCHED_TYPE(doif) = DI_SCH_DIST_STATIC;
  DI_IS_ORDERED(doif) = CL_PRESENT(CL_ORDERED);
  DI_ISSIMD(doif) = 0;
  sem.collapse = 0;
  if (chk_collapse) {
    if (CL_PRESENT(CL_COLLAPSE))
      sem.collapse = CL_VAL(CL_COLLAPSE);
  }
}

/* Handle distribute parallel do construct
 * We are making 2 loops:
 *   1) distribute loop
 *   2) parallel do loop
 */
int
do_distbegin(DOINFO *doinfo, int do_label, int construct_name)
{
  int iv, doif, sptr, initvar, limitvar, stepvar;
  int dast, dovar, step_expr;
  int past, aast;

  iv = doinfo->index_var;
  if (!DT_ISINT(DTYPEG(iv))) {
    error(155, ERR_Severe, gbl.lineno,
          "The index variable of a distribute DO must be integer -",
          SYMNAME(iv));
    sem.expect_dist_do = FALSE;
    return do_begin(doinfo);
  } else if (sem.teams <= 0) {
    error(155, ERR_Severe, gbl.lineno,
          "DISTRIBUTE construct must be nested in TEAMS construct", NULL);
    sem.expect_dist_do = FALSE;
    return do_begin(doinfo);
  }
  doinfo->prev_dovar = DOVARG(iv);
  doinfo->distloop = LP_DISTPARDO;
  DOCHK(iv);
  DOVARP(iv, 1);
  step_expr = doinfo->step_expr;

  /* Distribute loop */
  dast = mk_stmt(A_MP_PDO, 0 /* SST_ASTG(RHS(1)) BLOCKDO */);
  dovar = mk_id(iv);

  /* need to store to bounds/stride so that we can put it in
   * uplevel struct for reference in parallel do
   */
  initvar = getccssym_sc("i", sem.itemps++, ST_VAR, SC_PRIVATE);
  DTYPEP(initvar, DTYPEG(iv));
  aast = mk_assn_stmt(mk_id(initvar), doinfo->init_expr, DTYPEG(initvar));
  add_stmt(aast);

  limitvar = getccssym_sc("i", sem.itemps++, ST_VAR, SC_PRIVATE);
  DTYPEP(limitvar, DTYPEG(iv));
  aast = mk_assn_stmt(mk_id(limitvar), doinfo->limit_expr, DTYPEG(initvar));
  add_stmt(aast);

  stepvar = getccssym_sc("i", sem.itemps++, ST_VAR, SC_PRIVATE);
  DTYPEP(stepvar, DTYPEG(iv));
  aast = mk_assn_stmt(mk_id(stepvar), doinfo->step_expr, DTYPEG(initvar));
  add_stmt(aast);

  ENCLFUNCP(initvar, ENCLFUNCG(iv));
  ENCLFUNCP(limitvar, ENCLFUNCG(iv));
  ENCLFUNCP(stepvar, ENCLFUNCG(iv));
  DTYPEP(initvar, DTYPEG(iv));
  DTYPEP(limitvar, DTYPEG(iv));
  DTYPEP(stepvar, DTYPEG(iv));
  ASSNP(limitvar, 1);
  ASSNP(initvar, 1);
  ASSNP(stepvar, 1);

  A_DOVARP(dast, dovar);
  A_M1P(dast, mk_id(initvar));
  A_M2P(dast, mk_id(limitvar));

  if (A_TYPEG(step_expr) == A_CNST) {
    A_M3P(dast, step_expr);
  } else {
    A_M3P(dast, mk_id(stepvar));
  }
  A_CHUNKP(dast, DI_DISTCHUNK(sem.doif_depth));
  A_DISTCHUNKP(dast, DI_DISTCHUNK(sem.doif_depth));
  A_SCHED_TYPEP(dast, DI_SCHED_TYPE(sem.doif_depth));
  A_ORDEREDP(dast, DI_IS_ORDERED(sem.doif_depth));
  if (doinfo->lastval_var) {
    int lv_ast = mk_id(doinfo->lastval_var);
    A_LASTVALP(dast, lv_ast);
  } else {
    A_LASTVALP(dast, 0);
  }
  A_ENDLABP(dast, 0);
  A_DISTPARDOP(dast, 0);
  A_DISTRIBUTEP(dast, 1);

  NEED_DOIF(doif, DI_DO);
  DI_DO_LABEL(doif) = do_label;
  DI_DO_AST(doif) = dast;
  DI_DOINFO(doif) = doinfo;
  DI_NAME(doif) = construct_name;
  direct_loop_enter();
  (void)add_stmt(dast);

  /* simulate enter_dir(DI_PARDO...)  */
  {
    int cur, prev;
    prev = sem.doif_depth;
    NEED_DOIF(cur, DI_PARDO);
    DI_REDUC(cur) = NULL;
    DI_LASTPRIVATE(cur) = NULL;
    DI_REGIONVARS(cur) = NULL;
    DI_ALLOCATED(cur) = NULL;
    DI_SECT_VAR(cur) = 0;
  }
  restore_clauses();

  doif = sem.doif_depth;
  if (CL_PRESENT(CL_COLLAPSE))
    sem.collapse = CL_VAL(CL_COLLAPSE);
  sem.collapse_depth = sem.collapse;
  do_schedule(doif);
  sem.expect_do = TRUE;
  mk_lastprivate_list();
  if (has_team) { /* in same construct as teams */
    mk_reduction_list();
    mk_shared_list();
    has_team = FALSE;
  }
  mp_create_bscope(0);
  DI_BPAR(sem.doif_depth) = emit_bpar();
  par_push_scope(FALSE);
  begin_parallel_clause(sem.doif_depth);

  set_parref_flag(initvar, initvar, BLK_UPLEVEL_SPTR(sem.scope_level));
  set_parref_flag(limitvar, limitvar, BLK_UPLEVEL_SPTR(sem.scope_level));
  ref_object(initvar);
  ref_object(limitvar);
  sptr = decl_private_sym(iv);
  DTYPEP(sptr, DTYPEG(iv));

  doinfo = get_doinfo(1);
  doinfo->index_var = sptr;
  doinfo->init_expr = mk_id(initvar);
  doinfo->limit_expr = mk_id(limitvar);
  doinfo->distloop = LP_PARDO_OTHER;
  ADDRTKNP(initvar, 1);
  ADDRTKNP(limitvar, 1);
  ADDRTKNP(iv, 1);
  if (A_TYPEG(step_expr) == A_CNST) {
    doinfo->step_expr = step_expr;
  } else {
    ref_object(stepvar);
    doinfo->step_expr = mk_id(stepvar);
    ADDRTKNP(stepvar, 1);
  }
  sem.expect_do = FALSE;
  do_lastval(doinfo);
  if (sem.collapse_depth < 2) {
    sem.collapse_depth = 0;
    past = do_parbegin(doinfo);
    (void)add_stmt(past);
    dast = 0;
  } else {
    doinfo->collapse = sem.collapse_depth;
    past = collapse_begin(doinfo);
    dast = past;
  }

  NEED_DOIF(doif, DI_DO);
  DI_DO_LABEL(doif) = 0;
  DI_DO_AST(doif) = past;
  DI_DOINFO(doif) = doinfo;
  DI_DO_LABEL(doif) = do_label;
  DI_NAME(doif) = construct_name;
  direct_loop_enter(); /* Check if we need this */
  A_DISTPARDOP(past, 1);
  A_ENDLABP(past, 0);

  return dast;
}

static void
do_private(void)
{
  ITEM *itemp;

  if (CL_PRESENT(CL_PRIVATE))
    for (itemp = CL_FIRST(CL_PRIVATE); itemp != ITEM_END; itemp = itemp->next) {
      non_private_check(itemp->t.sptr, "PRIVATE");
      (void)decl_private_sym(itemp->t.sptr);
    }
}


static void
mk_firstprivate(int sptr1, int taskdupstd)
{
  int savepar, savetask, saveteams, savetarget, sptr, std;
  SST tmpsst;
  if(sptr1 == SPTR_NULL)
      return;
  set_parref_flag(sptr1, sptr1, BLK_UPLEVEL_SPTR(sem.scope_level));
  non_private_check(sptr1, "FIRSTPRIVATE");
  (void)mk_storage(sptr1, &tmpsst);
  sptr = decl_private_sym(sptr1);
  {
    savepar = sem.parallel;
    savetask = sem.task;
    savetarget = sem.target;
    saveteams = sem.teams;
    sem.parallel = 0;
    sem.task = 0;
    sem.target = 0;
    sem.teams = 0;
    /* TODO: Task is done in above?
     *       should not do for task here?
     */

    std = 0;
    if (!POINTERG(sptr)) {
      if (!XBIT(54, 0x1) && ALLOCATTRG(sptr)) {
        std = sem.scope_stack[sem.scope_level].end_prologue;
        if (std == 0) {
          std = STD_PREV(0);
        }
        add_assignment_before(sptr, &tmpsst, std);
      } else {
        add_assignment(sptr, &tmpsst);
      }
    } else {
      add_ptr_assignment(sptr, &tmpsst);
    }
    sem.task = savetask;
    sem.teams = saveteams;
    saveteams = sem.teams;
    sem.target = savetarget;
    if (SC_BASED == SCG(sptr)) {
      add_firstprivate_assn(sptr, sptr1, taskdupstd);
    } else if (sem.task && TASKG(sptr)) {
      int ast = mk_stmt(A_MP_TASKFIRSTPRIV, 0);
      int sptr1_ast = mk_id(sptr1);
      int sptr_ast = mk_id(sptr);
      A_LOPP(ast, sptr1_ast);
      A_ROPP(ast, sptr_ast);
      add_stmt_after(ast, taskdupstd);
    }
    sem.parallel = savepar;
  }
}

static void
do_firstprivate(int istask) 
{
  ITEM *itemp;
  int ast, taskdupstd, cntfp = 0, maxfp = 50, sptr, i;
  int *fpsptr;
  LOGICAL isnew;
  taskdupstd = 0;
  NEW(fpsptr, int, maxfp);
  LLUplevel *uplevel;
  if (istask && sem.parallel) {
    uplevel = llmp_has_uplevel(get_stblk_uplevel_sptr());
    if (uplevel != NULL)
      uplevel = llmp_has_uplevel(uplevel->parent);
    if (uplevel != NULL) {
      if (maxfp < uplevel->vals_count) {
        NEED(maxfp, fpsptr, int, maxfp, uplevel->vals_count);
      }
      maxfp = uplevel->vals_count;
      for (i = 0; i < uplevel->vals_count; ++i) {
        sptr = uplevel->vals[i];
        if (sptr == SPTR_NULL)
          continue;
        if ((SCG(sptr) == SC_PRIVATE) ||
            (SDSCG(sptr) != 0 && SCG(SDSCG(sptr)) == SC_PRIVATE))
          fpsptr[cntfp++] = sptr;
      }
    }
  }
  if (CL_PRESENT(CL_FIRSTPRIVATE)) {
    for (itemp = CL_FIRST(CL_FIRSTPRIVATE); itemp != ITEM_END;
         itemp = itemp->next) {
      isnew = TRUE;
      sptr = itemp->t.sptr;
      for (i = 0; i < cntfp; ++i)
        if (fpsptr[i] == sptr) {
            isnew = FALSE;
            break;
        }
      if (isnew) {
        if (cntfp > maxfp) {
            NEED(maxfp, fpsptr, int, maxfp, maxfp + 100);
            maxfp = maxfp + 100;
        }
        fpsptr[cntfp++] = sptr;
      }
    }
  }

  if(cntfp) {
    if (istask) {
      ast = mk_stmt(A_MP_TASKDUP, 0);
      taskdupstd = add_stmt(ast);
    }
    for(i=0;i<cntfp;++i)
      mk_firstprivate(fpsptr[i], taskdupstd);
    if (istask) {
      ast = mk_stmt(A_MP_ETASKDUP, 0);
      add_stmt(ast);
    }
  }
  FREE(fpsptr);
}

static void
do_lastprivate(void)
{
  int sptr;
  REDUC_SYM *reduc_symp;
  SCOPESTACK *scope;

  if (!CL_PRESENT(CL_LASTPRIVATE))
    return;
  for (reduc_symp = CL_FIRST(CL_LASTPRIVATE); reduc_symp;
       reduc_symp = reduc_symp->next) {
    sptr = reduc_symp->shared;
    if (sem.doif_depth > 1 && DI_ID(sem.doif_depth - 1) == DI_PAR &&
        SCG(sptr) == SC_PRIVATE && is_last_private(sptr) &&
        (DI_ID(sem.doif_depth) == DI_PARDO || DI_ID(sem.doif_depth) == DI_PDO ||
         DI_ID(sem.doif_depth) == DI_SINGLE ||
         DI_ID(sem.doif_depth) == DI_PARSECTS ||
         DI_ID(sem.doif_depth) == DI_PARWORKS ||
         DI_ID(sem.doif_depth) == DI_SECTS)) {

      scope = curr_scope();
      while ((scope = next_scope(scope)) != 0) {
        if (scope->di_par == sem.doif_depth - 1) {
          if (SCOPEG(sptr) == scope->sptr)
            error(155, ERR_Severe, gbl.lineno, SYMNAME(sptr),
                  "private variable may not appear in worksharing construct");
        }
      }
    }

    set_parref_flag(reduc_symp->shared, reduc_symp->shared,
                    BLK_UPLEVEL_SPTR(sem.scope_level));
    non_private_check(reduc_symp->shared, "LASTPRIVATE");
    reduc_symp->Private = decl_private_sym(reduc_symp->shared);
  }
  DI_LASTPRIVATE(sem.doif_depth) = CL_FIRST(CL_LASTPRIVATE);
}

static void
mk_lastprivate_list(void)
{
  REDUC_SYM *reduc_symp;
  REDUC_SYM *first, *last, *tmp;
  first = last = NULL;

  if (CL_PRESENT(CL_LASTPRIVATE)) {
    first = last = (REDUC_SYM *)getitem(0, sizeof(REDUC_SYM));
    for (reduc_symp = CL_FIRST(CL_LASTPRIVATE); reduc_symp;
         reduc_symp = reduc_symp->next) {

      tmp = (REDUC_SYM *)getitem(1, sizeof(REDUC_SYM));
      tmp->shared = reduc_symp->Private;
      last->next = tmp;
      last = last->next;
    }
    last->next = NULL;
    first = first->next;
  }
  CL_FIRST(CL_LASTPRIVATE) = first;
  CL_LAST(CL_LASTPRIVATE) = last;
}

static LOGICAL
validate_atomic_expr(int lop, int rop, int read)
{
  int sptr;
  DTYPE dtype1, dtype2;
  if (sem.mpaccatomic.accassignc > 2) {
    error(155, ERR_Severe, gbl.lineno,
          "Too many statements in ATOMIC CONSTRUCT", CNULL);
    return FALSE;
  } else if (sem.mpaccatomic.accassignc > 1 &&
             sem.mpaccatomic.action_type != ATOMIC_CAPTURE) {
    error(155, ERR_Severe, gbl.lineno,
          "Too many statements in ATOMIC CONSTRUCT", CNULL);
    return FALSE;
  }
  {
    sptr = memsym_of_ast(lop);
    if (sptr && ALLOCATTRG(sptr)) {
      if (A_TYPEG(lop) != A_SUBSCR) {
        error(155, ERR_Severe, gbl.lineno,
              "Alloctable is not allowed on lhs in ATOMIC", CNULL);
        return FALSE;
      }
    }
  }

  dtype1 = A_DTYPEG(lop);
  dtype2 = A_DTYPEG(rop);

  if (!DT_ISSCALAR(dtype1) && !DT_ISSCALAR(dtype2)) {
    error(155, ERR_Severe, gbl.lineno,
          "Scalar intrinsic type is expected in ATOMIC", CNULL);
    return FALSE;
  }

  if ((DTY(dtype1) == TY_DERIVED) || (DTY(dtype2) == TY_DERIVED)) {
    error(155, ERR_Severe, gbl.lineno,
          "Scalar intrinsic type is expected in ATOMIC", CNULL);
    return FALSE;
  }
  if (lop == rop) {
    error(155, ERR_Severe, gbl.lineno,
          "lhs and rhs must be distinctive locations in ATOMIC", CNULL);
    return FALSE;
  }
  return TRUE;
}

static int
mk_atomic_read(int lop, int src)
{
  int ast = 0;
  if (is_valid_atomic_read(lop, src)) {
    ast = mk_atomic(A_MP_ATOMICREAD, 0, src, A_DTYPEG(src));
    A_MEM_ORDERP(ast, sem.mpaccatomic.mem_order);
  }
  return ast;
}

static int
mk_atomic_write(int lop, int rop)
{
  int ast = 0;

  if (is_valid_atomic_write(lop, rop)) {
    ast = mk_stmt(A_MP_ATOMICWRITE, 0);
    A_LOPP(ast, lop);
    A_ROPP(ast, rop);
    A_MEM_ORDERP(ast, sem.mpaccatomic.mem_order);
  }
  return ast;
}

static void
_is_atomic_update_binop(int rop, int *arg)
{
  int lhs, rhs, cnt;
  int lop = arg[0];
  sem.mpaccatomic.rmw_op = AOP_UNDEF;
  cnt = 0;
  {
    if (A_TYPEG(rop) == A_BINOP) {
      switch (A_OPTYPEG(rop)) {
      case OP_ADD:
        sem.mpaccatomic.rmw_op = AOP_ADD;
        break;
      case OP_SUB:
        sem.mpaccatomic.rmw_op = AOP_SUB;
        break;
      case OP_MUL:
        sem.mpaccatomic.rmw_op = AOP_MUL;
        break;
      case OP_DIV:
        sem.mpaccatomic.rmw_op = AOP_DIV;
        break;
      case OP_LOR:
        sem.mpaccatomic.rmw_op = AOP_OR;
        break;
      case OP_LAND:
        sem.mpaccatomic.rmw_op = AOP_OR;
        break;
      case OP_LEQV:
        sem.mpaccatomic.rmw_op = AOP_EQV;
        break;
      case OP_LNEQV:
        sem.mpaccatomic.rmw_op = AOP_NEQV;
        break;
      default:
        return;
      }
      lhs = A_LOPG(rop);
      rhs = A_ROPG(rop);
      if (lop == lhs) {
        ++cnt;
      }
      if (lop == rhs) {
        ++cnt;
      }
    } else
      return;
  }
  arg[1] = arg[1] + cnt;
}

static LOGICAL
is_atomic_update_binop(int lop, int rop)
{
  int arg[2];
  ast_visit(1, 1);
  arg[0] = lop;
  arg[1] = 0;
  ast_traverse(rop, NULL, _is_atomic_update_binop, arg);
  ast_unvisit();
  if (arg[1] == 1) {
    return TRUE;
  } else if (arg[1] > 1) {
    sem.mpaccatomic.rmw_op = AOP_UNDEF;
    return TRUE;
  } else {
    return FALSE;
  }
}

static LOGICAL
is_atomic_update_intr(int lop, int rop)
{
  int cnt;
  int argcnt, argt, i;
  ATOMIC_RMW_OP aop_op = sem.mpaccatomic.rmw_op;

  switch (aop_op) {
  case AOP_AND:
  case AOP_OR:
  case AOP_XOR:
  case AOP_MIN:
  case AOP_MAX:
  case AOP_EQV:
  case AOP_NEQV:
    break;
  default:
    error(155, ERR_Severe, gbl.lineno, "Unexpected ATOMIC UPDATE intrinsic",
          CNULL);
    sem.mpaccatomic.rmw_op = AOP_UNDEF;
    return FALSE;
  }
  argcnt = A_ARGCNTG(rop);
  argt = A_ARGSG(rop);
  cnt = 0;
  for (i = 0; i < argcnt; ++i) {
    if (lop == ARGT_ARG(argt, i))
      cnt++;
  }
  if (cnt == 0)
    return FALSE;
  else if (cnt > 1) {
    sem.mpaccatomic.rmw_op = AOP_UNDEF;
    return TRUE;
  } else
    return TRUE;
}

static int
mk_atomic_update_binop(int lop, int rop)
{
  int ast;

  if (is_atomic_update_binop(lop, rop)) {
    if (sem.mpaccatomic.rmw_op == AOP_UNDEF) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected ATOMIC UPDATE statement",
            CNULL);
      return 0;
    }
  }
  ast = mk_stmt(A_MP_ATOMICUPDATE, 0);
  A_LOPP(ast, lop);
  A_ROPP(ast, rop);

  A_OPTYPEP(ast, sem.mpaccatomic.rmw_op); /* AOP_ADD/SUB/... */
  A_MEM_ORDERP(ast, sem.mpaccatomic.mem_order);
  return ast;
}

static int
mk_atomic_update_intr(int lop, int rop)
{
  int ast;
  ATOMIC_RMW_OP aop_op;
  MEMORY_ORDER mem_order = sem.mpaccatomic.mem_order;

  if (is_atomic_update_intr(lop, rop)) {
    if (sem.mpaccatomic.rmw_op == AOP_UNDEF) {
      error(155, ERR_Severe, gbl.lineno, "Unexpected ATOMIC UPDATE statement ",
            CNULL);
      return 0;
    }
  }
  ast = mk_stmt(A_MP_ATOMICUPDATE, 0);
  A_LOPP(ast, lop);
  A_ROPP(ast, rop);

  aop_op = sem.mpaccatomic.rmw_op;
  A_OPTYPEP(ast, aop_op); /* AOP_ADD/SUB/... */
  A_MEM_ORDERP(ast, mem_order);

  return ast;
}

static int
mk_atomic_capture(int lop, int rop)

{
  int ast = 0;
  ATOMIC_RMW_OP aop_op;
  MEMORY_ORDER mem_order = sem.mpaccatomic.mem_order;

  if (is_valid_atomic_capture(lop, rop)) {
    aop_op = sem.mpaccatomic.rmw_op;
    ast = mk_stmt(A_MP_ATOMICCAPTURE, 0);
    A_LOPP(ast, lop);
    A_ROPP(ast, rop);
    A_OPTYPEP(ast, aop_op); /* AOP_ADD/SUB/... */
    A_MEM_ORDERP(ast, mem_order);
  }
  return ast;
}

int
do_openmp_atomics(SST *l_stktop, SST *r_stktop)
{
  int ast, lop, rop, shape;
  int action_type = sem.mpaccatomic.action_type;
  LOGICAL atomic_ok = FALSE;
  DTYPE dtype;
  sem.mpaccatomic.apply = TRUE;

  if (mklvalue(l_stktop, 1) == 0) {
    /* Avoid assignment ILM's if lvalue is illegal */
    error(155, 3, gbl.lineno, "Expect lvalue on lhs in ATOMIC CONSTRUCT",
          CNULL);
    return 0;
  }
  dtype = SST_DTYPEG(l_stktop);
  shape = SST_SHAPEG(l_stktop);

  if (shape) {
    error(155, 3, gbl.lineno, "Expect scalar type in ATOMIC CONSTRUCT", CNULL);
    return 0;
  } else if (DTYG(dtype) == TY_STRUCT || DTYG(dtype) == TY_DERIVED ||
             DTY(dtype) == TY_ARRAY) {
    error(155, 3, gbl.lineno, "Expect scalar type in ATOMIC CONSTRUCT", CNULL);
    return 0;
  }
  ast = 0;
  lop = SST_ASTG(l_stktop);

  switch (action_type) {
  case ATOMIC_UPDATE:
    mkexpr1(r_stktop);
    rop = SST_ASTG(r_stktop);
    atomic_ok = validate_atomic_expr(lop, rop, 0);
    if (atomic_ok) {
      if (A_TYPEG(rop) == A_BINOP)
        ast = mk_atomic_update_binop(lop, rop);
      else if (A_TYPEG(rop) == A_INTR)
        ast = mk_atomic_update_intr(lop, rop);
      else
        error(155, 3, gbl.lineno, "Invalid ATOMIC UPDATE statement", CNULL);
    }
    if (ast)
      (void)add_stmt(ast);
    sem.mpaccatomic.mem_order = MO_UNDEF;
    sem.mpaccatomic.rmw_op = AOP_UNDEF;
    sem.mpaccatomic.seen = FALSE;
    return 0;
  case ATOMIC_READ:
    if (mklvalue(r_stktop, 1) == 0) {
      error(155, 3, gbl.lineno, "Invalid ATOMIC READ", CNULL);
    }
    rop = SST_ASTG(r_stktop);
    ast = mk_atomic_read(lop, rop);
    mkexpr1(r_stktop);
    if (ast) {
      SST_ASTP(r_stktop, ast);
    }
    sem.mpaccatomic.mem_order = MO_UNDEF;
    sem.mpaccatomic.seen = FALSE;
    return ast;

  case ATOMIC_WRITE:
    mkexpr1(r_stktop);
    rop = SST_ASTG(r_stktop);
    ast = mk_atomic_write(lop, rop);
    if (ast)
      (void)add_stmt(ast);
    sem.mpaccatomic.mem_order = MO_UNDEF;
    sem.mpaccatomic.seen = FALSE;
    return 0;
  case ATOMIC_CAPTURE:
    mkexpr1(r_stktop);
    rop = SST_ASTG(r_stktop);
    ast = mk_atomic_capture(lop, rop);
    if (ast)
      (void)add_stmt(ast);
    sem.mpaccatomic.rmw_op = AOP_UNDEF;
    return 0;
  default:
    break;
  }
  return ast;
}

static LOGICAL
is_valid_atomic_update(int lop, int rop)
{
  LOGICAL isvalid = TRUE;

  if (!lop || !rop) {
    isvalid = FALSE;
    goto end_valid;
  }
  isvalid = validate_atomic_expr(lop, rop, 0);
  if (isvalid) {
    if (A_TYPEG(rop) == A_BINOP) {
      if (is_atomic_update_binop(lop, rop)) {
        if (sem.mpaccatomic.rmw_op == AOP_UNDEF) {
          isvalid = FALSE;
          goto end_valid;
        }
      }
    } else if (A_TYPEG(rop) == A_INTR) {
      if (is_atomic_update_intr(lop, rop)) {
        if (sem.mpaccatomic.rmw_op == AOP_UNDEF) {
          isvalid = FALSE;
          goto end_valid;
        }
      }
    } else {
      isvalid = FALSE;
      goto end_valid;
    }
  } else {
    return isvalid;
  }
end_valid:
  if (!isvalid)
    error(155, 3, gbl.lineno, "Invalid ATOMIC UPDATE statement", CNULL);
  return isvalid;
}

static LOGICAL
is_valid_atomic_read(int lop, int rop)
{
  LOGICAL isvalid = TRUE;

  if (!lop || !rop) {
    isvalid = FALSE;
    goto end_valid;
  }
  isvalid = validate_atomic_expr(lop, rop, 1);

end_valid:
  return isvalid;
}

static LOGICAL
is_valid_atomic_write(int lop, int rop)
{
  LOGICAL isvalid = TRUE;

  if (!lop || !rop) {
    isvalid = FALSE;
    goto end_valid;
  }
  isvalid = validate_atomic_expr(lop, rop, 0);

end_valid:
  return isvalid;
}

static LOGICAL
is_valid_atomic_capture(int lop, int rop)
{
  LOGICAL isvalid = TRUE;
  LOGICAL isupdate = FALSE;

  if (!lop || !rop) {
    isvalid = FALSE;
    goto end_valid;
  }
  isvalid = validate_atomic_expr(lop, rop, 0);
  if (!isvalid)
    return isvalid;

  if (A_TYPEG(rop) == A_BINOP) {
    isupdate = is_atomic_update_binop(lop, rop);
    if (isupdate && sem.mpaccatomic.rmw_op == AOP_UNDEF) {
      isvalid = FALSE;
      goto end_valid;
    }
  } else if (A_TYPEG(rop) == A_INTR) {
    isupdate = is_atomic_update_intr(lop, rop);
    if (isupdate && sem.mpaccatomic.rmw_op == AOP_UNDEF) {
      isvalid = FALSE;
      goto end_valid;
    }
  }

  /* This could be just atomic write, make sure lhs is not in rhs */
  if (!isupdate) {
    if (contains_ast(rop, lop)) {
      isvalid = FALSE;
      goto end_valid;
    }
  }

end_valid:
  if (!isvalid)
    error(155, ERR_Severe, gbl.lineno, "Invalid ATOMIC CAPTURE statement ",
          CNULL);
  return isvalid;
}

LOGICAL
validate_omp_atomic(SST *l_stktop, SST *r_stktop)
{
  SST lstk, rstk;
  int action_type = sem.mpaccatomic.action_type;
  lstk = *l_stktop;
  rstk = *r_stktop;

  if (mklvalue(&lstk, 1) == 0) {
    error(155, 3, gbl.lineno, "Invalid ATOMIC statement: lhs", CNULL);
    return FALSE;
  }
  switch (action_type) {
  case ATOMIC_UPDATE:
    mkexpr1(&rstk);
    return is_valid_atomic_update(SST_ASTG(&lstk), SST_ASTG(&rstk));
  case ATOMIC_READ:
    return is_valid_atomic_read(SST_ASTG(&lstk), SST_ASTG(&rstk));
  case ATOMIC_WRITE:
    mkexpr1(&rstk);
    return is_valid_atomic_write(SST_ASTG(&lstk), SST_ASTG(&rstk));
  case ATOMIC_CAPTURE:
    mkexpr1(&rstk);
    return is_valid_atomic_capture(SST_ASTG(&lstk), SST_ASTG(&rstk));
  default:
    break;
  }
  return FALSE;
}

static void
do_reduction(void)
{
  REDUC *reducp;
  REDUC_SYM *reduc_symp;

  if (!CL_PRESENT(CL_REDUCTION))
    return;

  for (reducp = CL_FIRST(CL_REDUCTION); reducp; reducp = reducp->next) {
    for (reduc_symp = reducp->list; reduc_symp; reduc_symp = reduc_symp->next) {
      int dtype, ast;
      INT val[2];
      INT conval = 0;
      SST cnst;
      SST lhs;
      char *nm;

      if (reduc_symp->shared == 0)
        /* error - illegal reduction variable */
        continue;
      reduc_symp->Private = decl_private_sym(reduc_symp->shared);
      set_parref_flag(reduc_symp->shared, reduc_symp->shared,
                      BLK_UPLEVEL_SPTR(sem.scope_level));
      (void)mk_storage(reduc_symp->Private, &lhs);
      /*
       * emit the initialization of the private copy
       */
      dtype = DT_INT; /* assume the init constant is integer */
      switch (reducp->opr) {
      case 0: /* intrinsic */
        nm = SYMNAME(reducp->intrin);
        if (strcmp(nm, "max") == 0) {
          dtype = DTYPEG(reduc_symp->shared);
          dtype = DDTG(dtype);
          if (DT_ISINT(dtype)) {
            if (size_of(dtype) <= 4) {
              conval = 0x80000000;
              dtype = DT_INT;
            } else {
              val[0] = 0x80000000;
              val[1] = 0x00000000;
              conval = getcon(val, dtype);
            }
          } else if (dtype == DT_REAL)
            /* -3.402823466E+38 */
            conval = 0xff7fffff;
          else {
            /* -1.79769313486231571E+308 */
            val[0] = 0xffefffff;
            val[1] = 0xffffffff;
            conval = getcon(val, DT_DBLE);
          }
          break;
        }
        if (strcmp(nm, "min") == 0) {
          dtype = DTYPEG(reduc_symp->shared);
          dtype = DDTG(dtype);
          if (DT_ISINT(dtype)) {
            if (size_of(dtype) <= 4) {
              conval = 0x7fffffff;
              dtype = DT_INT;
            } else {
              val[0] = 0x7fffffff;
              val[1] = 0xffffffff;
              conval = getcon(val, dtype);
            }
          } else if (dtype == DT_REAL)
            /* 3.402823466E+38 */
            conval = 0x7f7fffff;
          else {
            /* 1.79769313486231571E+308 */
            val[0] = 0x7fefffff;
            val[1] = 0xffffffff;
            conval = getcon(val, DT_DBLE);
          }
          break;
        }
        if (strcmp(nm, "iand") == 0) {
          dtype = DTYPEG(reduc_symp->shared);
          dtype = DDTG(dtype);
          if (size_of(dtype) <= 4) {
            conval = 0xffffffff;
            dtype = DT_INT;
          } else {
            val[0] = 0xffffffff;
            val[1] = 0xffffffff;
            conval = getcon(val, dtype);
          }
          break;
        }
        if (strcmp(nm, "ior") == 0) {
          conval = 0;
          break;
        }
        if (strcmp(nm, "ieor") == 0) {
          conval = 0;
          break;
        }
        interr("do_reduction - illegal intrinsic", reducp->intrin, 0);
        break;
      case OP_ADD:
      case OP_SUB:
        conval = 0;
        break;
      case OP_MUL:
        conval = 1;
        break;
      case OP_LOG:
        dtype = DT_LOG;
        switch (reducp->intrin) {
        case OP_LAND:
        case OP_LEQV:
          conval = SCFTN_TRUE;
          break;
        case OP_LOR:
        case OP_LNEQV:
          conval = SCFTN_FALSE;
          break;
        default:
          interr("do_reduction - illegal log operator", reducp->intrin, 0);
        }
        break;
      default:
        interr("do_reduction - illegal operator", reducp->opr, 0);
        break;
      }
      SST_IDP(&cnst, S_CONST);
      SST_DTYPEP(&cnst, dtype);
      SST_CVALP(&cnst, conval);
      ast = mk_cval1(conval, dtype);
      SST_ASTP(&cnst, ast);
      (void)add_stmt(assign(&lhs, &cnst));
    }
  }
  DI_REDUC(sem.doif_depth) = CL_FIRST(CL_REDUCTION);
}

static void
save_private_list(void)
{
  ITEM *tmp, *first, *last, *itemp;
  SAVCL_FIRST(CL_PRIVATE) = NULL;
  SAVCL_LAST(CL_PRIVATE) = NULL;

  /* save private in a more permanent area for other construct */
  if (CL_PRESENT(CL_PRIVATE)) {
    last = first = (ITEM *)getitem(0, sizeof(ITEM));
    for (itemp = CL_FIRST(CL_PRIVATE); itemp != ITEM_END; itemp = itemp->next) {
      tmp = (ITEM *)getitem(1, sizeof(ITEM));
      tmp->t.sptr = itemp->t.sptr;
      last->next = tmp;
      last = last->next;
    }
    first = first->next;
    last->next = ITEM_END;

    SAVCL_FIRST(CL_PRIVATE) = first;
    SAVCL_LAST(CL_PRIVATE) = last;
  }
}

static void
save_firstprivate_list(void)
{
  ITEM *tmp, *first, *last, *itemp;
  SAVCL_FIRST(CL_FIRSTPRIVATE) = NULL;
  SAVCL_LAST(CL_FIRSTPRIVATE) = NULL;

  if (CL_PRESENT(CL_FIRSTPRIVATE)) {
    last = first = (ITEM *)getitem(0, sizeof(ITEM));
    for (itemp = CL_FIRST(CL_FIRSTPRIVATE); itemp != ITEM_END;
         itemp = itemp->next) {

      tmp = (ITEM *)getitem(1, sizeof(ITEM));
      tmp->t.sptr = itemp->t.sptr;
      last->next = tmp;
      last = last->next;
    }
    first = first->next;
    last->next = ITEM_END;

    SAVCL_FIRST(CL_FIRSTPRIVATE) = first;
    SAVCL_LAST(CL_FIRSTPRIVATE) = last;
  }
}

static void
save_shared_list(void)
{
  ITEM *tmp, *first, *last, *itemp;
  SAVCL_FIRST(CL_SHARED) = NULL;
  SAVCL_LAST(CL_SHARED) = NULL;
  /* save shared in a more permanent area for other construct */
  if (CL_PRESENT(CL_SHARED)) {
    last = first = (ITEM *)getitem(0, sizeof(ITEM));
    for (itemp = CL_FIRST(CL_SHARED); itemp != ITEM_END; itemp = itemp->next) {

      tmp = (ITEM *)getitem(1, sizeof(ITEM));
      tmp->t.sptr = itemp->t.sptr;
      last->next = tmp;
      last = last->next;
    }
    first = first->next;
    last->next = ITEM_END;

    SAVCL_FIRST(CL_SHARED) = first;
    SAVCL_LAST(CL_SHARED) = last;
  }
}

static void
mk_shared_list(void)
{
  ITEM *first, *last, *itemp;

  first = last = NULL;

  if (CL_PRESENT(CL_SHARED)) {
    last = first = (ITEM *)getitem(0, sizeof(ITEM));
    for (itemp = CL_FIRST(CL_SHARED); itemp != ITEM_END; itemp = itemp->next) {
      /* get the current scope symbol */
      itemp->t.sptr = getsymbol(SYMNAME(itemp->t.sptr));
      last->next = itemp;
      last = last->next;
    }
    first = first->next;
    last->next = ITEM_END;
  }
  CL_FIRST(CL_SHARED) = first;
  CL_LAST(CL_SHARED) = last;
}

static void
mk_reduction_list(void)
{
  REDUC *reducp;
  REDUC_SYM *reduc_symp;
  REDUC *first, *last, *curr = NULL;
  REDUC_SYM *symp;
  REDUC_SYM *symp_last;

  first = last = NULL;
  if (CL_PRESENT(CL_REDUCTION)) {
    first = last = (REDUC *)getitem(0, sizeof(REDUC));
    last->next = NULL;
    for (reducp = CL_FIRST(CL_REDUCTION); reducp; reducp = reducp->next) {
      curr = (REDUC *)getitem(1, sizeof(REDUC));
      curr->opr = reducp->opr;
      curr->intrin = reducp->intrin;
      last->next = curr;
      last = last->next;

      curr->list = symp_last = (REDUC_SYM *)getitem(0, sizeof(REDUC_SYM));
      for (reduc_symp = reducp->list; reduc_symp;
           reduc_symp = reduc_symp->next) {

        if (reduc_symp->shared == 0)
          continue;

        symp = (REDUC_SYM *)getitem(1, sizeof(REDUC_SYM));
        symp->shared = reduc_symp->Private;
        symp_last->next = symp;
        symp_last = symp_last->next;
      }
      symp_last->next = NULL;
      curr->list = curr->list->next;
    }
    last->next = NULL;
    first = first->next;
  }
  CL_FIRST(CL_REDUCTION) = first;
  CL_LAST(CL_REDUCTION) = last;
}

static void
do_copyin(void)
{
  ITEM *itemp;
  int sptr;
  int ast;
  int stblk;

  if (CL_PRESENT(CL_COPYIN)) {
    ast = mk_stmt(A_MP_BCOPYIN, 0);
    (void)add_stmt(ast);
    for (itemp = CL_FIRST(CL_COPYIN); itemp != ITEM_END; itemp = itemp->next) {
      sptr = itemp->t.sptr;
      if (STYPEG(sptr) == ST_CMBLK) {
        if (!THREADG(sptr)) {
          error(155, 3, gbl.lineno, SYMNAME(sptr),
                "is not a THREADPRIVATE common block");
          continue;
        }
      } else if (SCG(sptr) == SC_CMBLK && !HCCSYMG(CMBLKG(sptr))) {
        sptr = refsym(sptr, OC_OTHER);
        if (!THREADG(CMBLKG(sptr))) {
          error(155, 3, gbl.lineno, SYMNAME(sptr),
                "is not a member of a THREADPRIVATE common block");
          continue;
        }
      } else if (!THREADG(sptr)) {
        error(155, 3, gbl.lineno, SYMNAME(sptr), "is not THREADPRIVATE");
        continue;
      }
      ast = mk_stmt(A_MP_COPYIN, 0);
      A_SPTRP(ast, sptr);
      if (!ALLOCATTRG(sptr))
        A_ROPP(ast, astb.i0);
      else
        A_ROPP(ast, size_of_allocatable(sptr));
      (void)add_stmt(ast);
      if ((sem.parallel || sem.task || sem.target || sem.teams)) {
        stblk = BLK_UPLEVEL_SPTR(sem.scope_level);
        if (!stblk)
          stblk = get_stblk_uplevel_sptr();
        mp_add_shared_var(sptr, stblk);
        /* add first element of common block to uplevel */
        if (CMEMFG(sptr)) {
          mp_add_shared_var(CMEMFG(sptr), stblk);
        }
      }
    }
    ast = mk_stmt(A_MP_ECOPYIN, 0);
    (void)add_stmt(ast);
  }
}

static void
do_copyprivate()
{
  ITEM *itemp;
  int sptr;
  int ast;

  if (CL_PRESENT(CL_COPYPRIVATE)) {
    ast = mk_stmt(A_MP_BCOPYPRIVATE, 0);
    (void)add_stmt(ast);
    for (itemp = CL_FIRST(CL_COPYPRIVATE); itemp != ITEM_END;
         itemp = itemp->next) {
      sptr = itemp->t.sptr;
      if (STYPEG(sptr) == ST_CMBLK) {
        if (!THREADG(sptr)) {
          error(155, 3, gbl.lineno, SYMNAME(sptr),
                "is not a THREADPRIVATE common block");
          continue;
        }
      } else if (SCG(sptr) == SC_CMBLK && !HCCSYMG(CMBLKG(sptr))) {
        sptr = refsym(sptr, OC_OTHER);
        if (!THREADG(CMBLKG(sptr))) {
          error(155, 3, gbl.lineno, SYMNAME(sptr),
                "is not a member of a THREADPRIVATE common block");
          continue;
        }
      }
      ast = mk_stmt(A_MP_COPYPRIVATE, 0);
      A_SPTRP(ast, sptr);
      if (!ALLOCATTRG(sptr))
        A_ROPP(ast, astb.i0);
      else
        A_ROPP(ast, size_of_allocatable(sptr));
      (void)add_stmt(ast);
    }
    ast = mk_stmt(A_MP_ECOPYPRIVATE, 0);
    (void)add_stmt(ast);
  }
}

static void
do_map()
{
  if (!flg.omptarget)
    return;

  ITEM *item;
  int ast;
  if (CL_PRESENT(CL_MAP)) {
    for (item = (ITEM *)CL_FIRST(CL_MAP); item != ITEM_END; item = item->next) {
      ast = mk_stmt(A_MP_MAP, 0);
      (void)add_stmt(ast);
      A_LOPP(ast, item->ast);
      A_PRAGMATYPEP(ast, item->t.cltype);
      // TODO ompaccel do later lower/upper bounds
    }
  }
  ast = mk_stmt(A_MP_EMAP, 0);
  (void)add_stmt(ast);
}

static int
size_of_allocatable(int sptr)
{
  int nelems, dtype, dtyper, eltype;
  int ast;

  ast = mk_id(sptr);
  dtype = DTYPEG(sptr);
  nelems = 0;
  if (size_of(DT_PTR) == 8) {
    dtyper = DT_INT8; /* 64-bit */
  } else {
    dtyper = DT_INT4; /* 32-bit */
  }
  if (DTY(dtype) == TY_ARRAY) {
    int argt, func_ast;
    argt = mk_argt(2);
    ARGT_ARG(argt, 0) = ast;
    ARGT_ARG(argt, 1) = astb.ptr0;
    func_ast = mk_id(intast_sym[I_SIZE]);
    nelems = mk_func_node(A_INTR, func_ast, 2, argt);
    A_DTYPEP(nelems, dtyper);
    A_DTYPEP(func_ast, dtyper);
    A_OPTYPEP(nelems, I_SIZE);
    eltype = DTY(dtype + 1);
  } else
    eltype = dtype;
  /*  multiply by element type  */
  if (eltype == DT_ASSCHAR || eltype == DT_ASSNCHAR || eltype == DT_DEFERCHAR ||
      eltype == DT_DEFERNCHAR) {
    ast = ast_intr(I_LEN, dtyper, 1, ast);
  } else
    ast = size_ast_of(ast, eltype);
  if (nelems)
    ast = mk_binop(OP_MUL, ast, nelems, dtyper);

  return ast;
}

/*
 * Process the DEFAULT clause -- this can only be done after all of the
 * clauses which may declare private variables have been processed.
 */
static void
do_default_clause(int doif)
{
  ITEM *itemp;
  int sptr;
  SCOPE_SYM *symlast, *symp;

  /*
   * The DEFAULT scope is 'PRIVATE' or 'NONE'.  Save away the default
   * scope value and process the symbols which appeared in all of the
   * SHARED clauses.  The basic idea, if the default scope is not 'SHARED',
   * is to first look for symbols which were explicitly declared in the
   * current scope.  Private variables, appearing in various clauses of the
   * current directive, have already been explicitly declared and have their
   * SCOPE fields set to the current scope.  The function which will check
   * the scope of variables is sem_check_scope(); various semant functions,
   * such as ref_object() and ref_entry(), will call sem_check_scope().
   *
   * Variables appearing in the SHARED clause need to have their SCOPE
   * fields set to the current scope.  But, when it's time to leave the
   * current scope, these symbols cannot be removed from the symbol table's
   * hash lists.  Consequently, these variables need to have the scope
   * fields restored to their outer/previous values.  These variables and
   * there previous scope values will be saved in a list which will be
   * processsed by semsym.c:sem_pop_scope().
   */
  if (CL_PRESENT(CL_DEFAULT) && CL_VAL(CL_DEFAULT) != PAR_SCOPE_SHARED) {
    sem.scope_stack[sem.scope_level].par_scope = CL_VAL(CL_DEFAULT);
    sem.scope_stack[sem.scope_level].end_prologue = STD_PREV(0);
  } else if (!CL_PRESENT(CL_DEFAULT) && DI_ID(doif) == DI_TASK) {
    /* TASK without a DEFAULT clause.  Could have used
     * PAR_SCOPE_FIRSTPRIVATE, but decided to distinguish between  the
     * presence of DEFAULT(FIRST_PRIVATE) and firstprivate implied by
     * TASK.
     */
    sem.scope_stack[sem.scope_level].par_scope = PAR_SCOPE_TASKNODEFAULT;
    sem.scope_stack[sem.scope_level].end_prologue = STD_PREV(0);
  }

  if (!CL_PRESENT(CL_SHARED)) {
    return;
  }
  /*
   * create a fake SCOPE_SYM item (from area 0 freed during the end of
   * statement processing.
   */
  sem.scope_stack[sem.scope_level].shared_list = symlast =
      (SCOPE_SYM *)getitem(0, sizeof(SCOPE_SYM));

  symlast->sptr = 0;
  for (itemp = CL_FIRST(CL_SHARED); itemp != ITEM_END; itemp = itemp->next) {
    sptr = itemp->t.sptr;
    /*
     * Need to keep the SCOPE_SYM items around until the end of the
     * parallel directive, so allocate them in area 1.
     */
    symlast->next = symp = (SCOPE_SYM *)getitem(1, sizeof(SCOPE_SYM));
    symp->sptr = sptr;
    /* save the scope of the variable */
    symp->scope = SCOPEG(sptr);
    symp->next = NULL;
    symlast = symp;
    /* set the scope to the current scope level */
    SCOPEP(sptr, sem.scope_stack[sem.scope_level].sptr);
  }
  /* skip past the fake SCOPE_SYM item */
  sem.scope_stack[sem.scope_level].shared_list =
      sem.scope_stack[sem.scope_level].shared_list->next;
}

int
is_sptr_in_shared_list(int sptr)
{
  int region_level, current_level;
  SCOPE_SYM *list;

  /* sem.scope_level may not be the same as SCOPEG
   * of the sptr, for example,
   * !$omp parallel shared(sptr)  sem.scope_level
   * !$omp do                     sem.scope_level+1
   * if sptr is reference in do, we will miss it
   */

  region_level = sem.scope_stack[sem.scope_level].rgn_scope;
  current_level = sem.scope_level;
  if (sem.scope_stack[current_level].kind == SCOPE_PAR) {
    while (current_level > 0 && current_level >= region_level) {
      list = sem.scope_stack[current_level].shared_list;
      for (; list; list = list->next) {
        if (list->sptr == sptr)
          return 1;
      }
      current_level--;
    }
  }

  return 0;
}

static void
begin_parallel_clause(int doif)
{
  {
    switch (DI_ID(doif)) {
      int ast;
    default:
      break;
    case DI_PARDO:
      break;
    case DI_PDO:
      ast = mk_stmt(A_MP_BPDO, 0);
      (void)add_stmt(ast);
      break;
    }
  }

  switch (DI_ID(doif)) {
  case DI_PAR:
  case DI_PARDO:
  case DI_PDO:
  case DI_DOACROSS:
  case DI_SECTS:
  case DI_PARSECTS:
  case DI_SINGLE:
  case DI_PARWORKS:
  case DI_TASK:
  case DI_TASKLOOP:
  case DI_SIMD:
  case DI_TARGET:
  case DI_TEAMS:
  case DI_DISTRIBUTE:
  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTPARDO:
  case DI_TEAMSDISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
  case DI_TARGPARDO:
    do_private();
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  switch (DI_ID(doif)) {
  case DI_SINGLE:
  case DI_SECTS:
  case DI_PDO:
  case DI_SIMD:
    private_check();
    FLANG_FALLTHROUGH;
  case DI_PAR:
  case DI_PARDO:
  case DI_PARSECTS:
  case DI_PARWORKS:
  case DI_TASK:
  case DI_TASKLOOP:
  case DI_TARGET:
  case DI_TEAMS:
  case DI_DISTRIBUTE:
  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
  case DI_TEAMSDISTPARDO:
  case DI_TARGPARDO:
    do_firstprivate((DI_ID(doif) == DI_TASK || DI_ID(doif) == DI_TASKLOOP));
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  switch (DI_ID(doif)) {
  case DI_PARDO:
  case DI_PDO:
  case DI_DOACROSS:
  case DI_SECTS:
  case DI_PARSECTS:
  case DI_SIMD:
  case DI_DISTRIBUTE:
  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTPARDO:
  case DI_TEAMSDISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
  case DI_TARGPARDO:
  case DI_TASKLOOP:
    do_lastprivate();
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  switch (DI_ID(doif)) {
  case DI_PAR:
  case DI_PARDO:
  case DI_TARGPARDO:
  case DI_PDO:
  case DI_DOACROSS:
  case DI_SECTS:
  case DI_PARSECTS:
  case DI_PARWORKS:
  case DI_SIMD:
  case DI_TEAMS:
    do_reduction();
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  switch (DI_ID(doif)) {
  case DI_TARGET:
    do_map();
    break;
  default:
    break;
  }

  switch (DI_ID(doif)) {
  case DI_PAR:
  case DI_PARDO:
  case DI_TARGPARDO:
  case DI_PARSECTS:
  case DI_PARWORKS:
    do_copyin();
    FLANG_FALLTHROUGH;
  case DI_TASK:
  case DI_TASKLOOP:
  case DI_TEAMS:
    do_default_clause(doif);
    FLANG_FALLTHROUGH;
  default:
    break;
  }
}

void
end_parallel_clause(int doif)
{
  /* combine reduction variables */
  switch (DI_ID(doif)) {
  case DI_PAR:
  case DI_PARDO:
  case DI_TARGPARDO:
  case DI_PDO:
  case DI_DOACROSS:
  case DI_PARSECTS:
  case DI_SECTS:
  case DI_PARWORKS:
  case DI_SIMD:
  case DI_TEAMS:
    end_reduction(DI_REDUC(doif), doif);
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  /* last privates */
  switch (DI_ID(doif)) {
  case DI_PARDO:
  case DI_TARGPARDO:
  case DI_PDO:
  case DI_DOACROSS:
  case DI_PARSECTS:
  case DI_SECTS:
  case DI_SIMD:
  case DI_DISTRIBUTE:
  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTPARDO:
  case DI_TEAMSDISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
  case DI_TASKLOOP:
    end_lastprivate(doif);
    break;
  default:
    break;
  }

  /* deallocate any allocated privates */
  switch (DI_ID(doif)) {
  case DI_SINGLE:
  case DI_PAR:
  case DI_PARDO:
  case DI_TARGPARDO:
  case DI_PDO:
  case DI_DOACROSS:
  case DI_PARSECTS:
  case DI_SECTS:
  case DI_PARWORKS:
  case DI_TASK:
  case DI_TASKLOOP:
  case DI_TARGET:
  case DI_TEAMS:
  case DI_DISTRIBUTE:
  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTPARDO:
  case DI_TEAMSDISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
  case DI_SIMD:
    deallocate_privates(doif);
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  {
    switch (DI_ID(doif)) {
      int ast;
    default:
      break;
    case DI_PARDO:
    case DI_TARGPARDO:
      break;
    case DI_PDO:
      ast = mk_stmt(A_MP_EPDO, 0);
      (void)add_stmt(ast);
      break;
    }
  }
}

static ATOMIC_RMW_OP
get_atomic_rmw_op(int op)
{
  switch (op) {
  case OP_ADD:
    return AOP_ADD;
  case OP_SUB:
    return AOP_SUB;
  case OP_MUL:
    return AOP_MUL;
  case OP_DIV:
    return AOP_DIV;
  case OP_LOR:
    return AOP_OR;
  case OP_LAND:
    return AOP_AND;
  case OP_LEQV:
    return AOP_EQV;
  case OP_LNEQV:
    return AOP_NEQV;
  default:
    return AOP_UNDEF;
  }
}

static void
gen_reduction(REDUC *reducp, REDUC_SYM *reduc_symp, LOGICAL rmme,
              LOGICAL in_parallel)
{
  int ast;
  LOGICAL nobar = FALSE;
  SST lhs;
  SST op1, opr, op2;
  SST intrin;
  ITEM *arg1, *arg2;
  int opc, sptr, encl, scope;
  LOGICAL noatomic = FALSE;
  int ast_crit = 0;
  int ast_endcrit = 0;
  ATOMIC_RMW_OP save_aop = sem.mpaccatomic.rmw_op;

  if (rmme) {
    sptr = reduc_symp->shared;
    if (SCG(sptr) == SC_LOCAL && !in_parallel) {
      nobar = TRUE;
    } else if (SCG(sptr) == SC_PRIVATE) {
      encl = ENCLFUNCG(sptr);
      scope = BLK_SYM(sem.scope_level);
      if (encl == scope) {
        nobar = TRUE;
      } else {
        return;
      }
    } else {
      return;
    }
  }
  if (use_atomic_for_reduction(sem.doif_depth))
    add_stmt(mk_stmt(A_MP_ATOMIC, 0));

  (void)mk_storage(reduc_symp->shared, &lhs);
  (void)mk_storage(reduc_symp->shared, &op1);
  (void)mk_storage(reduc_symp->Private, &op2);
  switch (opc = reducp->opr) {
  case 0: /* intrinsic - always 2 arguments */
    SST_IDP(&intrin, S_IDENT);
    SST_SYMP(&intrin, reducp->intrin);
    arg1 = (ITEM *)getitem(0, sizeof(ITEM));
    arg1->t.stkp = &op1;
    arg2 = (ITEM *)getitem(0, sizeof(ITEM));
    arg1->next = arg2;
    arg2->t.stkp = &op2;
    arg2->next = ITEM_END;
    /*
     * Generate:
     *    shared  <-- intrin(shared, private)
     */
    (void)ref_intrin(&intrin, arg1);
    if (use_atomic_for_reduction(sem.doif_depth) &&
        sem.mpaccatomic.rmw_op != AOP_UNDEF) {
      MEMORY_ORDER save_mem_order = sem.mpaccatomic.mem_order;
      sem.mpaccatomic.mem_order = MO_SEQ_CST;
      mklvalue(&lhs, 1);
      ast = SST_ASTG(&intrin);
      ast = mk_atomic_update_intr(SST_ASTG(&lhs), ast);
      (void)add_stmt(ast);

      sem.mpaccatomic.rmw_op = save_aop;
      sem.mpaccatomic.mem_order = save_mem_order;
      add_stmt(mk_stmt(A_MP_ENDATOMIC, 0));
      goto end_reduction;
    } else {
      add_stmt(mk_stmt(A_MP_ENDATOMIC, 0));
      noatomic = TRUE;
      ast_crit = emit_bcs_ecs(A_MP_CRITICAL);
    }

    (void)add_stmt(assign(&lhs, &intrin));
    goto end_reduction;
  case OP_SUB:
    opc = OP_ADD;
    FLANG_FALLTHROUGH;
  case OP_ADD:
  case OP_MUL:
    SST_OPTYPEP(&opr, opc);
    goto do_binop;
  case OP_LOG:
    SST_OPTYPEP(&opr, opc);
    opc = reducp->intrin;
    SST_OPCP(&opr, opc);
  /*
   * Generate:
   *    shared  <--  shared <op> private
   */
  do_binop:

    binop(&op1, &op1, &opr, &op2);
    if (SST_IDG(&op1) == S_CONST) {
      ast = mk_cval1(SST_CVALG(&op1), (int)SST_DTYPEG(&op1));
    } else {
      ast = mk_binop(opc, SST_ASTG(&op1), SST_ASTG(&op2), SST_DTYPEG(&op1));
    }
    SST_ASTP(&op1, ast);
    SST_SHAPEP(&op1, A_SHAPEG(ast));

    if (use_atomic_for_reduction(sem.doif_depth)&& get_atomic_rmw_op(opc) != AOP_UNDEF) {
      MEMORY_ORDER save_mem_order = sem.mpaccatomic.mem_order;

      sem.mpaccatomic.rmw_op = get_atomic_rmw_op(opc);
      sem.mpaccatomic.mem_order = MO_SEQ_CST;
      mklvalue(&lhs, 1);
      ast = mk_atomic_update_binop(SST_ASTG(&lhs), ast);
      (void)add_stmt(ast);

      sem.mpaccatomic.rmw_op = save_aop;
      sem.mpaccatomic.mem_order = save_mem_order;
      add_stmt(mk_stmt(A_MP_ENDATOMIC, 0));
      goto end_reduction;
    } else {
      add_stmt(mk_stmt(A_MP_ENDATOMIC, 0));
      ast_crit = emit_bcs_ecs(A_MP_CRITICAL);
      noatomic = TRUE;
    }

    (void)add_stmt(assign(&lhs, &op1));

    goto end_reduction;
  default:
    interr("end_reduction - illegal operator", reducp->opr, 0);
    goto end_reduction;
  }
end_reduction:
  if (noatomic) {
    ast_endcrit = emit_bcs_ecs(A_MP_ENDCRITICAL);
    A_LOPP(ast_crit, ast_endcrit);
    A_LOPP(ast_endcrit, ast_crit);
  }
  if (nobar) {
    reduc_symp->shared = 0;
  }
}

static void
end_reduction(REDUC *red, int doif)
{
  REDUC *reducp;
  REDUC_SYM *reduc_symp;
  int ast_crit, ast_endcrit;
#ifdef OMP_OFFLOAD_LLVM
  int ast_red;
#endif
  int save_par, save_target, save_teams;
  LOGICAL done = FALSE;
  LOGICAL in_parallel = FALSE;

  if (red == NULL)
    return;

  sem.ignore_default_none = TRUE;
  /*
   * Do not want ref_object() -> sem_check_scope() to apply any default
   * scoping rules of the variables referenced in the updates.
   * Could really use sem.ignore_default_scope.
   */
  save_par = sem.parallel;
  sem.parallel = 0;
  save_target = sem.target;
  sem.target = 0;
  save_teams = sem.teams;
  sem.teams = 0;
  in_parallel = (save_par || save_target || save_teams);

  if (DI_ID(doif) == DI_SIMD) {
    for (reducp = red; reducp; reducp = reducp->next) {
      for (reduc_symp = reducp->list; reduc_symp;
           reduc_symp = reduc_symp->next) {
        if (reduc_symp->shared == 0)
          /* error - illegal reduction variable */
          continue;
        if (!use_atomic_for_reduction(sem.doif_depth) && !done) {
          ast_crit = emit_bcs_ecs(A_MP_CRITICAL);
          done = TRUE;
        }
        gen_reduction(reducp, reduc_symp, TRUE, in_parallel);
      }
    }
  }

  for (reducp = red; reducp; reducp = reducp->next) {
    for (reduc_symp = reducp->list; reduc_symp; reduc_symp = reduc_symp->next) {
      if (reduc_symp->shared == 0)
        /* error - illegal reduction variable or set by loop above */
        continue;
      if (!use_atomic_for_reduction(sem.doif_depth) && !done) {
#ifdef OMP_OFFLOAD_LLVM
        ast_red = mk_stmt(A_MP_BREDUCTION, 0);
        (void) add_stmt(ast_red);
#endif
        ast_crit = emit_bcs_ecs(A_MP_CRITICAL);
#ifdef OMP_OFFLOAD_LLVM
        if (!use_atomic_for_reduction(sem.doif_depth)) {
          A_ISOMPREDUCTIONP(ast_crit, 1);
          gen_reduction_ompaccel(reducp, reduc_symp, FALSE, in_parallel);
        }
#endif
        done = TRUE;
      }
      gen_reduction(reducp, reduc_symp, FALSE, in_parallel);
    }
  }

  sem.ignore_default_none = FALSE;
  sem.parallel = save_par;
  sem.target = save_target;
  sem.teams = save_teams;
  if (!use_atomic_for_reduction(sem.doif_depth)) {
    ast_endcrit = emit_bcs_ecs(A_MP_ENDCRITICAL);
    A_LOPP(ast_crit, ast_endcrit);
    A_LOPP(ast_endcrit, ast_crit);
#ifdef OMP_OFFLOAD_LLVM
    A_ISOMPREDUCTIONP(ast_endcrit, 1);
#endif
#ifdef OMP_OFFLOAD_LLVM
    ast_red = mk_stmt(A_MP_EREDUCTION, 0);
    (void)add_stmt(ast_red);
#endif
  }
}

static void
end_lastprivate(int doif)
{
  REDUC_SYM *reduc_symp;
  int i1, i2;
  int lab;
  int sptr;
  int save_par, save_target, save_teams;

  if (DI_LASTPRIVATE(doif) == NULL)
    return;
  lab = 0;
  switch (DI_ID(doif)) {
  default:
    break;
  case DI_SIMD:
    sptr = DI_DOINFO(doif + 1)->index_var;
    i1 = mk_id(sptr);
    sptr = DI_DOINFO(doif + 1)->lastval_var;
    i2 = mk_id(sptr);
    i1 = mk_binop(OP_EQ, i1, i2, DT_LOG4);
    i2 = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(i2, i1);
    (void)add_stmt(i2);
    lab = 1;
    break;
  case DI_DISTRIBUTE:
  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTPARDO:
  case DI_TEAMSDISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
  case DI_PARDO:
  case DI_TARGPARDO:
  case DI_PDO:
  case DI_DOACROSS:
  case DI_TASKLOOP:
    i1 = astb.k0;
    sptr = DI_DOINFO(doif + 1)->lastval_var;
    i2 = mk_id(sptr);
    i1 = mk_binop(OP_NE, i1, i2, DT_LOG4);
    i2 = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(i2, i1);
    (void)add_stmt(i2);
    lab = 1;
    break;
  case DI_PARSECTS:
  case DI_SECTS:
    /* Todo: use p_last to determine which one is the last iteration */
    if ((sptr = DI_SECT_VAR(doif))) {
      i1 = mk_id(sptr);
      i2 = mk_cval1(DI_SECT_CNT(doif), DT_INT4);
      i1 = mk_binop(OP_EQ, i1, i2, DT_LOG4);
      i2 = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(i2, i1);
      (void)add_stmt(i2);
      lab = 1;
    }
    break;
  }

  sem.ignore_default_none = TRUE;
  /*
   * Do not want ref_object() -> sem_check_scope() to apply any default
   * scoping rules of the variables referenced in the updates.
   * Could really use sem.ignore_default_scope.
   */
  save_par = sem.parallel;
  sem.parallel = 0;
  save_target = sem.target;
  sem.target = 0;
  save_teams = sem.teams;
  sem.teams = 0;
  for (reduc_symp = DI_LASTPRIVATE(doif); reduc_symp;
       reduc_symp = reduc_symp->next) {
    SST tmpsst;
    (void)mk_storage(reduc_symp->Private, &tmpsst);
    if (!POINTERG(reduc_symp->shared))
      add_assignment(reduc_symp->shared, &tmpsst);
    else {
      add_ptr_assignment(reduc_symp->shared, &tmpsst);
    }
  }
  sem.ignore_default_none = FALSE;
  sem.parallel = save_par;
  sem.target = save_target;
  sem.teams = save_teams;

  if (lab) {
    i2 = mk_stmt(A_ENDIF, 0);
    (void)add_stmt(i2);
  }
}

#define IN_WRKSHR 0
#define IN_PARALLEL 1
static void
end_workshare(int s_std, int e_std)
{
  int std;
  int ast;
  int state = IN_WRKSHR;
  int parallellevel = 0;
  int lasterror = 0;

  for (std = STD_NEXT(s_std); std && std != e_std; std = STD_NEXT(std)) {
    ast = STD_AST(std);
    switch (state) {
    case IN_WRKSHR:
      switch (A_TYPEG(ast)) {
      case A_FORALL:
      case A_ENDFORALL:
      case A_ASN:
      case A_WHERE:
      case A_ELSEWHERE:
      case A_ENDWHERE:
      case A_MP_CRITICAL:
      case A_MP_ENDCRITICAL:
      case A_MP_ATOMIC:
      case A_MP_ENDATOMIC:
      case A_MP_ATOMICREAD:
      case A_MP_ATOMICWRITE:
      case A_MP_ATOMICUPDATE:
      case A_MP_ATOMICCAPTURE:
      case A_MP_BMPSCOPE:
      case A_MP_EMPSCOPE:
        break;
      case A_MP_PARALLEL:
        parallellevel++;
        state = IN_PARALLEL;
        break;
      default:
        if (lasterror != STD_LINENO(std)) {
          error(155, 3, STD_LINENO(std),
                "Statement not allowed in WORKSHARE construct", NULL);
          lasterror = STD_LINENO(std);
        }
        break;
      }
      break;
    case IN_PARALLEL:
      switch (A_TYPEG(ast)) {
      case A_MP_PARALLEL:
        parallellevel++;
        break;
      case A_MP_ENDPARALLEL:
        if (--parallellevel == 0) {
          state = IN_WRKSHR;
        }
        break;
      }
      break;
    }
  }
}

static void
do_btarget(int doif)
{
  mp_create_bscope(0);
  DI_BTARGET(doif) = emit_btarget(A_MP_TARGET);
  par_push_scope(TRUE);
  begin_parallel_clause(doif);
}

static void
do_bteams(int doif)
{
  int ast, num_teams, thread_limit;
  mp_create_bscope(0);

  num_teams = 0;
  thread_limit = 0;
  ast = mk_stmt(A_MP_TEAMS, 0);
  DI_BTEAMS(doif) = ast;

  if (CL_PRESENT(CL_NUM_TEAMS)) {
    num_teams = CL_VAL(CL_NUM_TEAMS);
  }
  if (CL_PRESENT(CL_THREAD_LIMIT)) {
    thread_limit = CL_VAL(CL_THREAD_LIMIT);
  }
  A_NTEAMSP(ast, num_teams);
  A_THRLIMITP(ast, thread_limit);
  add_stmt(ast);

  sem.teams++;
  par_push_scope(FALSE);
  begin_parallel_clause(doif);
}

static void
do_bdistribute(int doif, LOGICAL chk_collapse)
{
  int ast;

  do_dist_schedule(doif, chk_collapse);
  ast = mk_stmt(A_MP_DISTRIBUTE, 0);
  DI_BDISTRIBUTE(doif) = ast;
  add_stmt(ast);

  get_stblk_uplevel_sptr();
  par_push_scope(TRUE);
  get_stblk_uplevel_sptr();
  begin_parallel_clause(doif);
}

static struct {
  int chunk;
  int distchunk;
  int mp_iftype;
} sav_chk;

static void
save_clauses()
{
  int i, ast, sptr;
  for (i = 0; i < CL_MAXV; i++) {
    SAVCL_PRESENT(i) = CL_PRESENT(i);
    SAVCL_VAL(i) = CL_VAL(i);
    SAVCL_FIRST(i) = CL_FIRST(i);
    SAVCL_LAST(i) = CL_LAST(i);
  }
  sav_chk.chunk = chunk;
  sav_chk.distchunk = distchunk;
  sav_chk.mp_iftype = mp_iftype;

  save_private_list();
  save_firstprivate_list();
  save_shared_list();

  /* create tmp and store the value from ast to tmp */
  if (CL_PRESENT(CL_NUM_THREADS)) {
    sptr = get_itemp(DT_INT);
    ast = CL_VAL(CL_NUM_THREADS);
    ast = mk_assn_stmt(mk_id(sptr), ast, DT_INT);
    (void)add_stmt(ast);
    SAVCL_VAL(CL_NUM_THREADS) = sptr;
  }
  if (CL_PRESENT(CL_NUM_TEAMS)) {
    sptr = get_itemp(DT_INT);
    ast = CL_VAL(CL_NUM_TEAMS);
    ast = mk_assn_stmt(mk_id(sptr), ast, DT_INT);
    (void)add_stmt(ast);
    SAVCL_VAL(CL_NUM_TEAMS) = sptr;
  }
  if (CL_PRESENT(CL_THREAD_LIMIT)) {
    sptr = get_itemp(DT_INT);
    ast = CL_VAL(CL_THREAD_LIMIT);
    ast = mk_assn_stmt(mk_id(sptr), ast, DT_INT);
    (void)add_stmt(ast);
    SAVCL_VAL(CL_THREAD_LIMIT) = sptr;
  }
  if (CL_PRESENT(CL_IF)) {
    sptr = get_itemp(DT_INT);
    ast = CL_VAL(CL_IF);
    ast = mk_assn_stmt(mk_id(sptr), ast, DT_INT);
    (void)add_stmt(ast);
    SAVCL_VAL(CL_IF) = sptr;
  }
  if (CL_PRESENT(CL_SCHEDULE)) {
    if (chunk) {
      sptr = get_itemp(DT_INT);
      ast = chunk;
      ast = mk_assn_stmt(mk_id(sptr), ast, DT_INT);
      (void)add_stmt(ast);
      sav_chk.chunk = sptr;
    }
  }
  if (CL_PRESENT(CL_DIST_SCHEDULE)) {
    if (distchunk) {
      sptr = get_itemp(DT_INT);
      ast = distchunk;
      ast = mk_assn_stmt(mk_id(sptr), ast, DT_INT);
      (void)add_stmt(ast);
      sav_chk.distchunk = sptr;
    }
  }

  /* todo: ordered(n) */
}

static void
restore_clauses(void)
{
  int i, sptr;
  for (i = 0; i < CL_MAXV; i++) {
    CL_PRESENT(i) = SAVCL_PRESENT(i);
    CL_VAL(i) = SAVCL_VAL(i);
    CL_FIRST(i) = SAVCL_FIRST(i);
    CL_LAST(i) = SAVCL_LAST(i);
  }
  chunk = sav_chk.chunk;
  distchunk = sav_chk.distchunk;
  mp_iftype = sav_chk.mp_iftype;
  switch (DI_ID(sem.doif_depth)) {
  default:
    break;
  case DI_TARGET:
    if (CL_PRESENT(CL_IF) &&
        (mp_iftype == IF_DEFAULT || mp_iftype == IF_TARGET)) {
      sptr = CL_VAL(CL_IF);
      CL_VAL(CL_IF) = mk_id(sptr);
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
    break;
  case DI_TEAMS:
    if (CL_PRESENT(CL_NUM_TEAMS)) {
      sptr = CL_VAL(CL_NUM_TEAMS);
      CL_VAL(CL_NUM_TEAMS) = mk_id(sptr);
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
    if (CL_PRESENT(CL_THREAD_LIMIT)) {
      sptr = CL_VAL(CL_THREAD_LIMIT);
      CL_VAL(CL_THREAD_LIMIT) = mk_id(sptr);
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
    break;
  case DI_PARDO:
  case DI_TARGPARDO:
    if (CL_PRESENT(CL_NUM_THREADS)) {
      sptr = CL_VAL(CL_NUM_THREADS);
      CL_VAL(CL_NUM_THREADS) = mk_id(sptr);
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
    if (CL_PRESENT(CL_IF) &&
        (mp_iftype == IF_DEFAULT || mp_iftype == IF_PARALLEL)) {
      sptr = CL_VAL(CL_IF);
      CL_VAL(CL_IF) = mk_id(sptr);
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
    if (CL_PRESENT(CL_SCHEDULE)) {
      if (chunk) {
        sptr = sav_chk.chunk;
        chunk = mk_id(sptr);
        set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
      }
    }
    break;
  case DI_PAR:
    if (CL_PRESENT(CL_NUM_THREADS)) {
      sptr = CL_VAL(CL_NUM_THREADS);
      CL_VAL(CL_NUM_THREADS) = mk_id(sptr);
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
    if (CL_PRESENT(CL_IF) &&
        (mp_iftype == IF_DEFAULT || mp_iftype == IF_PARALLEL)) {
      sptr = CL_VAL(CL_IF);
      CL_VAL(CL_IF) = mk_id(sptr);
      set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
    }
    break;
  case DI_DISTRIBUTE:
  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTPARDO:
  case DI_TEAMSDISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
    if (CL_PRESENT(CL_DIST_SCHEDULE)) {
      if (distchunk) {
        sptr = sav_chk.distchunk;
        distchunk = mk_id(sptr);
        set_parref_flag(sptr, sptr, BLK_UPLEVEL_SPTR(sem.scope_level));
      }
    }
    break;
  }
}

/* handle begin combine constructs for target/teams/distribute/parallel/do
 */
static void
begin_combine_constructs(BIGINT64 construct)
{
  int doif = sem.doif_depth;
  int ast;
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
  int combinedMode;
#endif
  LOGICAL do_enter = FALSE;

  has_team = FALSE;
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
  combinedMode = get_omp_combined_mode(construct);
  if (flg.omptarget) {
    if (!CL_PRESENT(CL_SCHEDULE)) {
      if (combinedMode == mode_target_teams_distribute_parallel_for_simd ||
          combinedMode == mode_target_teams_distribute_parallel_for)
        add_clause(CL_SCHEDULE, TRUE);
      CL_VAL(CL_SCHEDULE) = DI_SCH_STATIC;
      chunk = 3;
    }
  }
#endif
  save_clauses();

  if (BT_SIMD & construct) {
    apply_nodepchk(gbl.lineno, 1);
  }

  if (BT_TARGET & construct) {
    do_btarget(sem.doif_depth);
    ast = DI_BTARGET(sem.doif_depth);
#ifdef OMP_OFFLOAD_LLVM
    if (flg.omptarget) {
      if (combinedMode == mode_target_teams_distribute_parallel_for_simd) {
        errwarn(1203);
        combinedMode = mode_target_teams_distribute_parallel_for;
      } else if (combinedMode == mode_target_parallel_for_simd) {
        errwarn(1203);
        combinedMode = mode_target_parallel_for;
      } else if (combinedMode == mode_target_teams_distribute) {
        error(1202, ERR_Severe, gbl.lineno, "target teams distribute",
              "parallel do");
      } else if (combinedMode == mode_target_teams) {
        error(1202, ERR_Severe, gbl.lineno, "target teams",
             "distribute parallel do");
      }
      A_COMBINEDTYPEP(ast, combinedMode);
    }
#endif
    do_enter = TRUE;
  }
  if (BT_TEAMS & construct) {
    if (do_enter) {
      parstuff_init();
      doif = enter_dir(DI_TEAMS, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
    }
    restore_clauses();
    do_bteams(sem.doif_depth);
    do_enter = TRUE;
    has_team = TRUE;
  }
  if (BT_DISTRIBUTE & construct) {
    if (do_enter) {
      parstuff_init();
      if ((BT_PARDO & construct)) {
        if (BT_TARGET & construct)
          doif = enter_dir(DI_TARGTEAMSDISTPARDO, TRUE, 0,
                           DI_B(DI_ATOMIC_CAPTURE));
        else if (BT_TEAMS & construct)
          doif = enter_dir(DI_TEAMSDISTPARDO, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
        else
          doif = enter_dir(DI_DISTPARDO, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
      } else {
        if (BT_TARGET & construct)
          doif = enter_dir(DI_TARGTEAMSDIST, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
        else if (BT_TEAMS & construct)
          doif = enter_dir(DI_TEAMSDIST, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
        else
          doif = enter_dir(DI_DISTRIBUTE, TRUE, 0, DI_B(DI_ATOMIC_CAPTURE));
      }
    }
    restore_clauses();
    if ((BT_PARDO & construct))
      sem.expect_dist_do = TRUE;
    do_bdistribute(sem.doif_depth, !(BT_PARDO & construct));

    /* need to push scope so that dovar is not the same as
     * lastprivate(dovar) for distributed parallel do loop
     */
    if ((BT_PARDO & construct)) {
      par_push_scope(TRUE);
    }
    return;
  }
  if ((BT_PARDO & construct)) {
    if (do_enter) {
      parstuff_init();
      doif = enter_dir(DI_PARDO, FALSE, 0, DI_B(DI_ATOMIC_CAPTURE));
    }
    do_schedule(doif);
    sem.expect_do = TRUE;
    restore_clauses();
    mp_create_bscope(0);
    DI_BPAR(doif) = emit_bpar();
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
  }
  if (BT_PAR & construct) {
    if (do_enter) {
      parstuff_init();
      doif = enter_dir(DI_PAR, FALSE, 0, DI_B(DI_ATOMIC_CAPTURE));
    }
    mp_create_bscope(0);
    restore_clauses();
    DI_BPAR(doif) = emit_bpar();
    par_push_scope(FALSE);
    begin_parallel_clause(sem.doif_depth);
    return;
  }
}

void
end_teams()
{
  int doif, ast;
  end_parallel_clause(doif = sem.doif_depth);
  (void)leave_dir(DI_TEAMS, TRUE, 0);
  --sem.teams;
  par_pop_scope();
  if (doif) {
    ast = mk_stmt(A_MP_ENDTEAMS, 0);
    add_stmt(ast);
    A_LOPP(DI_BTEAMS(doif), ast);
    A_LOPP(ast, DI_BTEAMS(doif));
    mp_create_escope();
  }
}

void
end_target()
{
  int doif, ast;
  end_parallel_clause(doif = sem.doif_depth);
  (void)leave_dir(DI_TARGET, TRUE, 0);
  sem.target--;
  par_pop_scope();
  if (doif) {
    ast = emit_etarget();
    mp_create_escope();
    A_LOPP(DI_BTARGET(doif), ast);
    A_LOPP(ast, DI_BTARGET(doif));
  }
}

void
end_targteams()
{
  end_teams();
  end_target();
}

static void
deallocate_privates(int doif)
{
  ITEM *itemp;

  for (itemp = DI_ALLOCATED(doif); itemp != NULL; itemp = itemp->next) {
    gen_conditional_dealloc(ALLOCATTRG(itemp->t.sptr), mk_id(itemp->t.sptr),
                            STD_PREV(0));
  }
}

/**
 * This function includes the actions performed by the production
 * `<var ref> ::= <ident>` in the context of starting a parallel region
 * or in the context of a parallel region.  In either case, the symbol
 * returned is the 'outer' symbol upon which the private symbol is
 * based.
 * In addition:
 * 1.  if the symbol is a function entry, refer to its local variable,
 * 2.  if the symbol is based, its associated pointer variable is
 *     created.
 * 3.  if the symbol is not yet a variable, it will be classified as if
 *     it was declared in the outer/nonparallel region.
 */
int
find_outer_sym(int sym)
{
  int sptr;

  sptr = refsym(sym, OC_OTHER);
  if (STYPEG(sptr) != ST_PARAM) {
    if (!IS_INTRINSIC(STYPEG(sptr)) && STYPEG(sptr) != ST_PROC) {
      DCLCHK(sptr);
    }
    /* Pick up the data type from the symbol table entry which was
     * either: 1) explicitly set by the user, or 2) has the current
     * default value.
     */
    if (DTYPEG(sptr) == DT_NONE) {
      /* This is only okay if identifier is an intrinsic,
       * generic, or predeclared.  This means the function was
       * used as an identifier without parenthesized arguments.
       */
      if (IS_INTRINSIC(STYPEG(sptr)))
        setimplicit(sptr);
    } else if (STYPEG(sptr) == ST_ENTRY && gbl.rutype == RU_FUNC)
      sptr = ref_entry(sptr);
  }
  if (IS_INTRINSIC(STYPEG(sptr))) {
    int sptr1;
    sptr1 = newsym(sptr);
    if (sptr1 != 0)
      sptr = sptr1;
    else {
      sptr = insert_sym(sptr);
      SCOPEP(sptr, stb.curr_scope);
    }
  }
  if (SCG(sptr) == SC_NONE)
    sem_set_storage_class(sptr);
  if (SCG(sptr) == SC_BASED)
    (void)ref_based_object(sptr);

  switch (STYPEG(sptr)) {
  case ST_UNKNOWN:
  case ST_IDENT:
    STYPEP(sptr, ST_VAR);
    break;
  default:
    break;
  }

  return sptr;
}

int
mk_storage(int sptr, SST *stkp)
{
#if DEBUG
  switch (STYPEG(sptr)) {
  case ST_UNKNOWN:
  case ST_IDENT:
    interr("mk_storage: stype should have been set", sptr, 2);
    STYPEP(sptr, ST_VAR);
    break;
  default:
    break;
  }
#endif
  SST_IDP(stkp, S_IDENT);
  SST_DTYPEP(stkp, DTYPEG(sptr));
  SST_SYMP(stkp, sptr);
  SST_CVLENP(stkp, 0);
  SST_SHAPEP(stkp, 0);

  return SST_SYMG(stkp);
}

/*  x <== y, x is x's symbol table entry; y_stkp is a pointer to y's SST */
static void
add_assignment_before(int x, SST *y_stkp, int std)
{
  SST x_sst;

  SST_IDP(&x_sst, S_IDENT);
  SST_DTYPEP(&x_sst, DTYPEG(x));
  SST_SYMP(&x_sst, x);
  SST_ASTP(&x_sst, 0);
  (void)add_stmt_before(assign(&x_sst, y_stkp), std);
}

/*  x <== y, x is x's symbol table entry; y_stkp is a pointer to y's SST */
static void
add_assignment(int x, SST *y_stkp)
{
  SST x_sst;

  SST_IDP(&x_sst, S_IDENT);
  SST_DTYPEP(&x_sst, DTYPEG(x));
  SST_SYMP(&x_sst, x);
  SST_ASTP(&x_sst, 0);
  (void)add_stmt(assign(&x_sst, y_stkp));
}

/*  x => y, x is x's symbol table entry; y_stkp is a pointer to y's SST */
static void
add_ptr_assignment(int x, SST *y_stkp)
{
  SST x_sst;

  SST_IDP(&x_sst, S_IDENT);
  SST_DTYPEP(&x_sst, DTYPEG(x));
  SST_SYMP(&x_sst, x);
  SST_ASTP(y_stkp, mk_id(SST_SYMG(y_stkp)));
  (void)add_stmt(assign_pointer(&x_sst, y_stkp));
}

/** \brief Add an assignment to the parallel prologue.
    \param dstsym The symbol assigned to.
    \param srcsym The symbol assigned from.
 */
void
add_assign_firstprivate(int dstsym, int srcsym)
{
  SST srcsst, dstsst;
  int where, savepar, savetask, savetarget, ast;
  int dupwhere;

  dupwhere = where = sem.scope_stack[sem.scope_level].end_prologue;
  if (where == 0) {
    interr("add_assign_firstprivate - can't find prologue", 0, 3);
    return;
  }
  (void)mk_storage(srcsym, &srcsst);
  SST_IDP(&dstsst, S_IDENT);
  SST_DTYPEP(&dstsst, DTYPEG(dstsym));
  SST_SYMP(&dstsst, dstsym);
  SST_ASTP(&dstsst, 0);
  /* assign() calls ref_object() calls sem_check_scope() which can call
   * back here.  Avoid infinite recursion by setting sem.parallel to zero
   * for the duration of assign(), preventing sem_check_scope() from
   * calling here again.
   */
  savepar = sem.parallel;
  savetask = sem.task;
  savetarget = sem.target;
  sem.parallel = 0;
  if (sem.task && TASKG(dstsym)) {
    ast = mk_stmt(A_MP_TASKFIRSTPRIV, 0);
    int src_ast = mk_id(srcsym);
    int dst_ast = mk_id(dstsym);
    A_LOPP(ast, src_ast);
    A_ROPP(ast, dst_ast);
    where = add_stmt_after(ast, where);
  }
  set_parref_flag(srcsym, srcsym, BLK_UPLEVEL_SPTR(sem.scope_level));
  sem.task = 0;
  sem.target = 0;
  if (!POINTERG(srcsym))
    where = add_stmt_after(assign(&dstsst, &srcsst), where);
  else {
    SST_ASTP(&srcsst, mk_id(SST_SYMG(&srcsst)));
    where = add_stmt_after(assign_pointer(&dstsst, &srcsst), where);
  }
  sem.parallel = savepar;
  sem.task = savetask;
  sem.target = savetarget;
  sem.scope_stack[sem.scope_level].end_prologue = where;
  if (sem.task && TASKG(dstsym)) {
    ast = mk_stmt(A_MP_TASKDUP, 0);
    add_stmt_after(ast, dupwhere);
    ast = mk_stmt(A_MP_ETASKDUP, 0);
    add_stmt_after(ast, where);
  }
}

static void
assign_cval(int sptr, int v, int d)
{
  SST tmpsst;
  int ast;

  SST_IDP(&tmpsst, S_CONST);
  SST_DTYPEP(&tmpsst, d);
  SST_CVALP(&tmpsst, v);
  ast = mk_cval1(v, d);
  SST_ASTP(&tmpsst, ast);
  add_assignment(sptr, &tmpsst);
}

static int
enter_dir(int typ,               /* begin what structured directive */
          LOGICAL ignore_nested, /* ignore directive if nested within itself */
          LOGICAL ignore_sev,    /* error severity if nested directive ignored;
                                  * 0 => don't issue error message.
                                  */
          BITMASK64 illegal_region /* bit vector - which directives cannot
                                    * contain this directive.
                                    */
)
{
  int prev;
  int cur;
  char bf[128];
  LOGICAL ignore_it;

  prev = sem.doif_depth;
  NEED_DOIF(cur, typ);
  DI_REDUC(cur) = NULL;
  DI_LASTPRIVATE(cur) = NULL;
  DI_REGIONVARS(cur) = NULL;
  DI_ALLOCATED(cur) = NULL;
  DI_SECT_VAR(cur) = 0;
  ignore_it = FALSE;
  if (ignore_nested && (DI_NEST(prev) & DI_B(typ))) {
    /* nested directive */
    if (ignore_sev) {
      sprintf(bf, "Nested directive %s ignored", name_of_dir(typ));
      error(155, ignore_sev, gbl.lineno, bf, NULL);
    }
    ignore_it = TRUE;
  }
  if (DI_NEST(prev) & illegal_region) {
    switch (typ) {
    /*
     * These are legally nested as long as they bind to different PARALLEL
     * directives.
     */
    case DI_PDO:
    case DI_SINGLE:
    case DI_SECTS:
      while (prev) {
        int di;
        di = DI_ID(prev);
        switch (di) {
        case DI_PDO:
        case DI_SINGLE:
        case DI_SECTS:
          goto nest_err; /* bind to the same parallel directive */
        case DI_PAR:
          goto return_it; /* bind to a different parallel directive */
        default:
          if (DI_NEST(prev) & illegal_region) {
            if (DI_B(di) & ~illegal_region) {
              /*
               * Need to skip the immediate enclosing construct
               * if it's actually legal, such as a DI_DO (see
               * f21436)
               */
              break; /* go to the next level */
            }
            /*
             * This one is in the set of regions that cannot
             * enclose the one being entered.
             */
            goto nest_err;
          }
          break;
        }
        prev--;
      }
      /* Not lexically bound to the same parallel directive */
      break;
    default:
      break;
    }
  nest_err:
    error(155, 3, gbl.lineno, "Illegal context for", name_of_dir(typ));
    ignore_it = TRUE;
  }
return_it:
  if (ignore_it)
    return 0;
  return cur;
}

static int
leave_dir(int typ,               /* end of which structured directive */
          LOGICAL ignore_nested, /* ignore directive if nested within itself */
          LOGICAL ignore_sev /* error severity if nested directive ignored */
)
{
  int prev;
  int cur;
  char bf[128];

  deallocate_no_scope_sptr();
  cur = sem.doif_depth;
  if (DI_ID(cur) == typ) {
    sem.doif_depth--;
    prev = sem.doif_depth;
    if (ignore_nested && (DI_NEST(prev) & DI_B(typ))) {
      /* nested directive */
      if (ignore_sev) {
        sprintf(bf, "Nested directive END %s ignored", name_of_dir(typ));
        error(155, ignore_sev, gbl.lineno, bf, NULL);
      }
      return 0;
    }
    return cur;
  }
  if (typ == DI_PARDO) {
    if (DI_ISSIMD(cur))
      error(155, 3, gbl.lineno,
            "ENDPARALLELDOSIMD must immediately follow a DO loop", NULL);
    else
      error(155, 3, gbl.lineno,
            "ENDPARALLELDO must immediately follow a DO loop", NULL);
  }
  if (typ == DI_PDO || typ == DI_SIMD)
    error(155, 3, gbl.lineno, "ENDDO must immediately follow a DO loop", NULL);
  else
    error(155, 3, gbl.lineno, "Unmatched directive END", name_of_dir(typ));
  return 0;
}

static const char *
name_of_dir(int typ)
{
  switch (typ) {
  case DI_PAR:
    return "PARALLEL";
  case DI_PARDO:
    return "PARALLEL DO";
  case DI_TARGPARDO:
    return "TARGET PARALLEL DO";
  case DI_PDO:
    return "DO";
  case DI_SIMD:
    return "SIMD";
  case DI_DOACROSS:
    return "DOACROSS";
  case DI_PARSECTS:
    return "PARALLEL SECTIONS";
  case DI_SECTS:
    return "SECTIONS";
  case DI_SINGLE:
    return "SINGLE";
  case DI_CRITICAL:
    return "CRITICAL";
  case DI_TASK:
    return "TASK";
  case DI_ATOMIC_CAPTURE:
    return "ATOMIC CAPTURE";
  case DI_MASTER:
    return "MASTER";
  case DI_ORDERED:
    return "ORDERED";
  case DI_PARWORKS:
    return "PARALLEL WORKSHARE";
  case DI_ACCREG:
    return "REGION";
  case DI_ACCKERNELS:
    return "KERNELS";
  case DI_ACCPARALLEL:
    return "PARALLEL";
  case DI_ACCSERIAL:
    return "SERIAL";
  case DI_ACCDO:
    return "DO";
  case DI_ACCLOOP:
    return "LOOP";
  case DI_ACCREGDO:
    return "REGION DO";
  case DI_ACCREGLOOP:
    return "REGION LOOP";
  case DI_ACCKERNELSDO:
    return "KERNELS DO";
  case DI_ACCKERNELSLOOP:
    return "KERNELS LOOP";
  case DI_ACCPARALLELDO:
    return "PARALLEL DO";
  case DI_ACCPARALLELLOOP:
    return "PARALLEL LOOP";
  case DI_ACCSERIALLOOP:
    return "SERIAL LOOP";
  case DI_ACCDATAREG:
    return "DATA";
  case DI_ACCHOSTDATA:
    return "HOST_DATA";
  case DI_TARGET:
    return "TARGET";
  case DI_TEAMS:
    return "TEAMS";
  case DI_DISTRIBUTE:
    return "DISTRIBUTE";
  case DI_TEAMSDIST:
    return "TEAMS DISTRIBUTE";
  case DI_TARGTEAMSDIST:
    return "TARGET TEAMS DISTRIBUTE";
  case DI_DISTPARDO:
    return "DISTRIBUTE PARALLEL DO";
  case DI_TEAMSDISTPARDO:
    return "TEAMS DISTRIBUTE PARALLEL DO";
  case DI_TARGTEAMSDISTPARDO:
    return "TARGET TEAMS DISTRIBUTE PARALLEL DO";
  case DI_TASKLOOP:
    return "TASKLOOP";
  }
  return "NEED NAME";
}

static int
find_reduc_intrinsic(int ident)
{
  char *nm;
  int sptr;

  nm = SYMNAME(ident);
  if (strcmp(nm, "max") != 0 && strcmp(nm, "min") != 0 &&
      strcmp(nm, "iand") != 0 && strcmp(nm, "ior") != 0 &&
      strcmp(nm, "ieor") != 0) {
    error(155, 3, gbl.lineno,
          "The reduction intrinsic must be MAX, MIN, IAND, IOR, or IEOR, not",
          nm);
    return 0;
  }
  sptr = ident;
  do {
    if (STYPEG(sptr) == ST_GENERIC)
      return sptr;
    sptr = HASHLKG(sptr);
  } while (sptr && NMPTRG(sptr) == NMPTRG(ident));
  interr("find_reduc_intrinsic: generic not found ", ident, 3);
  return 0;
}

static int
get_csect_sym(const char *nm)
{
#undef CSECT_PFX
#define CSECT_PFX "__cs_"
  int sptr = getsymf(CSECT_PFX "%s", nm);

  sptr = refsym_inscope(sptr, OC_CMBLK);
  if (STYPEG(sptr) == ST_UNKNOWN) {
    STYPEP(sptr, ST_CMBLK);
    SYMLKP(sptr, gbl.cmblks); /* link into list */
    gbl.cmblks = sptr;
    if (!XBIT(69, 0x100)) {
      int sptr1;
      ADSC *ad;
      int dtype;

      /*
       * kmpc requires a semaphore variable to be 32 bytes and
       * 8-byte aligned
       */
      sptr1 = get_next_sym(SYMNAME(sptr), "sem");
      dtype = get_array_dtype(1, DT_INT8);
      DTYPEP(sptr1, dtype);
      STYPEP(sptr1, ST_ARRAY);
      SCP(sptr1, SC_CMBLK);
      ad = AD_DPTR(dtype);
      AD_LWAST(ad, 0) = AD_LWBD(ad, 0) = 0;
      AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = mk_isz_cval(4, astb.bnd.dtype);
      AD_EXTNTAST(ad, 0) = mk_isz_cval(8, astb.bnd.dtype);
      CMEMFP(sptr, sptr1);
      CMEMLP(sptr, sptr1);
      CMBLKP(sptr1, sptr);
      SCP(sptr1, SC_CMBLK);
      SYMLKP(sptr1, NOSYM);
    } else {
      int sptr1, sptr2;
      ADSC *ad;
      int dtype;

      sptr1 = get_next_sym(SYMNAME(sptr), "sel");
      dtype = get_array_dtype(1, DT_INT4);
      DTYPEP(sptr1, dtype);
      STYPEP(sptr1, ST_ARRAY);
      SCP(sptr1, SC_CMBLK);
      ad = AD_DPTR(dtype);
      AD_LWAST(ad, 0) = AD_LWBD(ad, 0) = 0;
      AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = mk_isz_cval(16, astb.bnd.dtype);
      AD_EXTNTAST(ad, 0) = mk_isz_cval(16, astb.bnd.dtype);

      sptr2 = get_next_sym(SYMNAME(sptr), "sem");
      DTYPEP(sptr2, dtype);
      STYPEP(sptr2, ST_ARRAY);
      SCP(sptr2, SC_CMBLK);

      CMEMFP(sptr, sptr1);
      SYMLKP(sptr1, sptr2);
      SYMLKP(sptr2, NOSYM);
      CMEMLP(sptr, sptr2);
      CMBLKP(sptr1, sptr);
      CMBLKP(sptr2, sptr);
    }
  }
  return sptr;
}

static int
get_csect_pfxlen(void)
{
  return strlen(CSECT_PFX);
}

static void
check_barrier(void)
{
  int prev;
  prev = sem.doif_depth;
  while (prev > 0) {
    switch (DI_ID(prev)) {
    case DI_PDO:
    case DI_PARDO:
    case DI_TARGPARDO:
    case DI_SECTS:
    case DI_PARSECTS:
    case DI_SINGLE:
    case DI_CRITICAL:
    case DI_MASTER:
    case DI_ORDERED:
    case DI_TASK:
    case DI_TASKLOOP:
      error(155, 3, gbl.lineno, "Illegal context for barrier", NULL);
      return;
    case DI_PAR: /* reached the barrier's binding thread */
      return;
    default:
      break;
    }
    prev--;
  }
}

static void
check_crit(char *nm)
{
  int sptr;
  int prev;

  prev = sem.doif_depth;
  while (--prev) {
    if (DI_ID(prev) == DI_CRITICAL) {
      sptr = DI_CRITSYM(prev);
      if (sptr && nm != NULL) {
        if (strcmp(nm, SYMNAME(sptr) + get_csect_pfxlen()) == 0) {
          error(155, 3, gbl.lineno,
                "CRITICAL sections with the same name may not be nested -", nm);
          break;
        }
      } else if (sptr == 0 && nm == NULL) {
        error(155, 3, gbl.lineno, "Unnamed CRITICAL sections may not be nested",
              NULL);
        break;
      }
    }
  }
}

static void
check_targetdata(int type, const char *nm)
{
  int i;
  if (type == OMP_TARGET) {
    clause_errchk(BT_TARGET, nm);
  }
  for (i = 0; i < CL_MAXV; i++) {
    if (CL_PRESENT(i)) {
      switch (i) {
      case CL_OMPDEVICE:
      case CL_IF:
        break;
      case CL_MAP:
        if (type == OMP_TARGETUPDATE)
          error(533, 3, gbl.lineno, CL_NAME(i), nm);
        break;
      case CL_DEPEND:
        if (type != OMP_TARGETENTERDATA && type != OMP_TARGETEXITDATA &&
            type != OMP_TARGETUPDATE)
          error(533, 3, gbl.lineno, CL_NAME(i), nm);
        break;
      case CL_TO:
      case CL_FROM:
        if (type != OMP_TARGETUPDATE)
          error(533, 3, gbl.lineno, CL_NAME(i), nm);
        break;
      case CL_NOWAIT:
        if (type != OMP_TARGETENTERDATA && type != OMP_TARGETEXITDATA)
          error(533, 3, gbl.lineno, CL_NAME(i), nm);
        break;
      case CL_USE_DEVICE_PTR:
        if (type != OMP_TARGETDATA)
          error(533, 3, gbl.lineno, CL_NAME(i), nm);
        break;
      default:
        error(533, 3, gbl.lineno, CL_NAME(i), nm);
      }
    }
  }
}

/* from cancel_type 1: parallel
                    2: do
                    3: sections
                    4: taskgroup
*/
static int
check_cancel(int cancel_type)
{
  int prev;
  int res;
  prev = sem.doif_depth;
  while (prev > 0) {
    switch (DI_ID(prev)) {
    case DI_PAR:
      if (cancel_type == 1) {
        res = DI_BPAR(prev);
        return res;
      } else {
        error(155, 3, gbl.lineno,
              "Expect PARALLEL as construct-type-clause in "
              "CANCEL/CANCELLATION POINT",
              NULL);
        return 0;
      }
      break;
    case DI_DO:
      if (cancel_type == 2) {
        if ((prev - 1) > 0 && DI_ID(prev - 1) != DI_PDO)
          break;
        res = DI_DO_AST(prev); /* This is a do ast */
        if (A_ORDEREDG(res)) {
          error(155, 3, gbl.lineno,
                "A loop construct that is canceled must "
                "not have an ordered clause",
                NULL);
          return 0;
        }
        return res;
      }
      break;
    case DI_SECTS:
    case DI_PARSECTS:
      if (cancel_type == 3) {
        res = DI_BEGINP(prev);
        return res;
      } else {
        error(155, 3, gbl.lineno,
              "Expect SECTIONS as construct-type-clause in "
              "CANCEL/CANCELLATION POINT",
              NULL);
        return 0;
      }
      break;
    case DI_TASK:
      if (cancel_type == 4) {
        res = DI_BEGINP(prev);
        return res;
      } else {
        error(155, 3, gbl.lineno,
              "Expect TASKGROUP as construct-type-clause "
              "in CANCEL/CANCELLATION POINT",
              NULL);
        return 0;
      }
      break;
    default:
      break;
    }
    --prev;
  }
  if (prev <= 0) {
    error(155, 3, gbl.lineno,
          "CANCEL/CANCELLATION POINT expects enclosing region to be  PARALLEL,"
          " DO, SECTIONS, or TASK",
          NULL);
  }
  return 0;
}

static int
get_mp_bind_type(char *nm)
{
  INT val[2];
  int cnst_sptr;
  val[0] = 0;

  if (strcmp(nm, "master") == 0) {
    /* MP_PROC_BIND_MASTER */
    val[1] = 2;
  } else if (strcmp(nm, "close") == 0) {
    /* MP_PROC_BIND_CLOSE */
    val[1] = 3;
  } else if (strcmp(nm, "spread") == 0) {
    /* MP_PROC_BIND_SPREAD */
    val[1] = 4;
  } else {
    /* MP_PROC_BIND_FALSE */
    error(155, 3, gbl.lineno, "Unknown PROC_BIND type", CNULL);
    return 0;
  }

  cnst_sptr = getcon(val, DT_INT);
  return mk_cnst(cnst_sptr);
}

static void
cray_pointer_check(ITEM *itp, int clause)
{
  ITEM *itemp;
  int sptr;
  char bf[128];

  sprintf(bf, "A Cray pointer may not appear in the %s clause -",
          CL_NAME(clause));
  for (itemp = itp; itemp != ITEM_END; itemp = itemp->next) {
    sptr = itemp->t.sptr;
    if (SCG(sptr) == SC_BASED && MIDNUMG(sptr) && !CCSYMG(MIDNUMG(sptr)) &&
        !HCCSYMG(MIDNUMG(sptr)))
      error(155, 3, gbl.lineno, bf, SYMNAME(sptr));
  }
}

static void
other_firstlast_check(ITEM *itp, int clause)
{
  ITEM *itemp;
  int sptr;

  for (itemp = itp; itemp != ITEM_END; itemp = itemp->next) {
    sptr = itemp->t.sptr;
    if ((SCG(sptr) == SC_BASED || SCG(sptr) == SC_DUMMY) && MIDNUMG(sptr)) {
    }
  }
}
static void
private_check()
{
  ITEM *itemp;
  int sptr1;
  char bf[128];

  if (CL_PRESENT(CL_FIRSTPRIVATE)) {
    for (itemp = CL_FIRST(CL_FIRSTPRIVATE); itemp != ITEM_END;
         itemp = itemp->next) {
      sptr1 = itemp->t.sptr;
      if (SCG(sptr1) == SC_PRIVATE &&
          sem.scope_stack[sem.scope_level].par_scope) {
        if (SCOPEG(sptr1) == sem.scope_stack[sem.scope_level - 1].sptr) {
          sprintf(
              bf,
              "private variable may not appear in the FIRSTPRIVATE clause ");
          error(155, 3, gbl.lineno, bf, SYMNAME(sptr1));
        }
      }
    }
  }
}

void
add_non_private(int sptr)
{
  int i;
  i = sem.non_private_avail;
  ++sem.non_private_avail;
  NEED(sem.non_private_avail, sem.non_private_base, int, sem.non_private_size,
       sem.non_private_size + 20);
  sem.non_private_base[i] = sptr;
}

static void
non_private_check(int sptr, const char *cl)
{
  int i;
  for (i = 0; i < sem.non_private_avail; i++) {
    if (sem.non_private_base[i] == sptr) {
      char bf[128];
      sprintf(bf, "may not appear in a %s clause", cl);
      error(155, 3, gbl.lineno, SYMNAME(sptr), bf);
      break;
    }
  }
}

void
add_no_scope_sptr(int oldsptr, int newsptr, int lineno)
{
  int i;
  if (sem.doif_depth == 0)
    return;
  i = DI_NOSCOPE_AVL(sem.doif_depth);
  ++DI_NOSCOPE_AVL(sem.doif_depth);
  NEED(DI_NOSCOPE_AVL(sem.doif_depth), DI_NOSCOPE_BASE(sem.doif_depth),
       NOSCOPE_SYM, DI_NOSCOPE_SIZE(sem.doif_depth),
       DI_NOSCOPE_SIZE(sem.doif_depth) + 20);
  BZERO(DI_NOSCOPE_BASE(sem.doif_depth) + i, NOSCOPE_SYM,
        DI_NOSCOPE_SIZE(sem.doif_depth) - i);

  (DI_NOSCOPE_BASE(sem.doif_depth))[i].oldsptr = oldsptr;
  (DI_NOSCOPE_BASE(sem.doif_depth))[i].newsptr = newsptr;
  (DI_NOSCOPE_BASE(sem.doif_depth))[i].lineno = gbl.lineno;
  (DI_NOSCOPE_BASE(sem.doif_depth))[i].is_dovar = 0;
}

static void
deallocate_no_scope_sptr()
{
  if (sem.doif_depth == 0)
    return;
  FREE((DI_NOSCOPE_BASE(sem.doif_depth)));
  DI_NOSCOPE_AVL(sem.doif_depth) = 0;
  DI_NOSCOPE_SIZE(sem.doif_depth) = 0;
  DI_NOSCOPE_BASE(sem.doif_depth) = NULL;
}

void
clear_no_scope_sptr()
{
  int i, newsptr;

  for (i = 0; i < DI_NOSCOPE_AVL(sem.doif_depth); i++) {
    newsptr = (DI_NOSCOPE_BASE(sem.doif_depth))[i].newsptr;
    if (newsptr) {
      if ((DI_NOSCOPE_BASE(sem.doif_depth))[i].is_dovar) {
        if (SCG(newsptr) == SC_PRIVATE)
          pop_sym(newsptr);
      }
    }
  }
  DI_NOSCOPE_AVL(sem.doif_depth) = 0;
}

void
check_no_scope_sptr()
{
  int i, in_forall;

  if (sem.doif_depth == 0)
    return;
  for (i = 0; i < DI_NOSCOPE_AVL(sem.doif_depth); i++) {
    if ((DI_NOSCOPE_BASE(sem.doif_depth))[i].newsptr) {
      if (!(DI_NOSCOPE_BASE(sem.doif_depth))[i].is_dovar) {
        error(155, 3, gbl.lineno,
              SYMNAME((DI_NOSCOPE_BASE(sem.doif_depth))[i].oldsptr),
              "must appear in a SHARED or PRIVATE clause");
        break;
      }
    }
  }
  in_forall = DI_NOSCOPE_FORALL(sem.doif_depth);

  if (in_forall)
    return;

  clear_no_scope_sptr();
}

void
is_dovar_sptr(int sptr)
{
  int i;
  if (sem.doif_depth == 0)
    return;
  for (i = 0; i < DI_NOSCOPE_AVL(sem.doif_depth); i++) {
    if ((DI_NOSCOPE_BASE(sem.doif_depth))[i].newsptr == sptr) {
      (DI_NOSCOPE_BASE(sem.doif_depth))[i].is_dovar = 1;
      break;
    }
  }
}

void
par_add_stblk_shvar()
{
}

static LLUplevel *
findUplevelForSharedVar(int sptr, int stblk)
{
  LLUplevel *up, *curr_up;
  if (SCG(sptr) == SC_PRIVATE) {
    SPTR paruplevel;
    SPTR encl = ENCLFUNCG(sptr);
    /* find variable scope which contains uplevel struct */
    paruplevel = PARUPLEVELG(encl);
    while (!paruplevel && encl) {
      encl = ENCLFUNCG(encl);
      paruplevel = PARUPLEVELG(encl);
    }
    up = NULL;
    if (paruplevel) {
      up = llmp_get_uplevel(paruplevel);
#if DEBUG
      assert(up, "uplevel does not exist", paruplevel, 3);
#endif
      /* find the paruplevel where up is its parent */
      while (stblk) {
        curr_up = llmp_parent_uplevel(stblk);
        if (up == curr_up) {
          return llmp_get_uplevel(stblk);
        }
        stblk = llmp_get_parent_sptr(stblk);
      }
    }
    return NULL;
  } else {
    up = llmp_outermost_uplevel(stblk);
    return up;
  }
}

static bool
needCharLen(int sptr)
{
  DTYPE dtype = DTYPEG(sptr);
  TY_KIND dty = DTYG(dtype);
  switch (dty) {
  case TY_CHAR:
  case TY_NCHAR:
    return true;
  case TY_PTR:
    if (DTYG(DTYG(dtype)) == TY_CHAR) {
      return true;
    } else if (DTYG(DTYG(dtype)) == TY_NCHAR) {
      return true;
    }
    FLANG_FALLTHROUGH;
  default:
    return false;
  }
  return false;
}

static void
mp_add_shared_var(int sptr, int stblk)
{
  int dolen = 0;

  if (stblk) {
    LLUplevel *up;
    up = findUplevelForSharedVar(sptr, stblk);
    if (!up) {
      return;
    }
    if (needCharLen(sptr) || DTY(DTYPEG(sptr)) == TY_CHAR) {
      /* how we load and search uplevel struct
       * put cvlen field first if being referenced.
       */
      if (CVLENG(sptr)) {
        llmp_add_shared_var(up, CVLENG(sptr));
        PARREFP(CVLENG(sptr), 1);
      } else if (ADJLENG(sptr)) {
        int cvlen = CVLENG(sptr);
        if (cvlen == 0) {
          cvlen = sym_get_scalar(SYMNAME(sptr), "len", DT_INT);
          CVLENP(sptr, cvlen);
          if (SCG(sptr) == SC_DUMMY)
            CCSYMP(cvlen, 1);
        }
        llmp_add_shared_var(up, CVLENG(sptr));
      }
    }
    dolen = llmp_add_shared_var(up, sptr);
    PARREFP(sptr, 1);
    if (dolen && needCharLen(sptr)) {
      llmp_add_shared_var_charlen(up, sptr);
    }
    return;
  }
}

static void
parref_bnd(int ast, int stblk)
{
  if (ast && A_TYPEG(ast) == A_ID) {
    int sptr;
    sptr = A_SPTRG(ast);
    mp_add_shared_var(sptr, stblk);
  }
}

void
set_parref_flag(int sptr, int psptr, int stblk)
{
  if (!SCG(sptr))
    return;
  if (STYPEG(sptr) == ST_MEMBER)
    return;
  /* For OpenMP target offload, we put every symbols into the uplevel struct.
   * Because every symbols must be sent to the target device, and are loaded from the uplevel struct.*/
  if (!flg.omptarget && (SCG(sptr) == SC_CMBLK || SCG(sptr) == SC_STATIC))
    return;
  if (SCG(sptr) == SC_EXTERN && ST_ISVAR(sptr)) /* No global vars in uplevel */
    return;
  if (DINITG(sptr) || SAVEG(sptr)) {
    if (SCG(sptr) != SC_LOCAL) {
      if (SCG(sptr) == SC_BASED) {
        int sym = MIDNUMG(sptr);
        if (SCG(sym) != SC_LOCAL)
          return;
      }
    }
  }
  if (!stblk)
    stblk = get_stblk_uplevel_sptr();

  mp_add_shared_var(sptr, stblk);
  if (psptr)
    PARREFP(psptr, 1);
  if (DTY(DTYPEG(sptr)) == TY_ARRAY || POINTERG(sptr) || ALLOCATTRG(sptr)) {
    int descr, sdsc, midnum;
    descr = DESCRG(sptr);
    sdsc = SDSCG(sptr);
    midnum = MIDNUMG(sptr);
    if (descr) {
      mp_add_shared_var(descr, stblk);
    }
    if (sdsc) {
      mp_add_shared_var(sdsc, stblk);
    }
    if (midnum) {
      mp_add_shared_var(midnum, stblk);
    }
  } else if (STYPEG(sptr) == ST_PROC && IS_PROC_DUMMYG(sptr)) {
    int sdsc = SDSCG(sptr);
    if (sdsc == 0) {
      get_static_descriptor(sptr);
      sdsc = SDSCG(sptr);
    }
    mp_add_shared_var(sdsc, stblk);
  }
  if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
    ADSC *ad;
    ad = AD_DPTR(DTYPEG(sptr));
    if (AD_ADJARR(ad) || ALLOCATTRG(sptr) || ASSUMSHPG(sptr)) {
      int i, ndim;
      ndim = AD_NUMDIM(ad);
      for (i = 0; i < ndim; i++) {
        parref_bnd(AD_LWAST(ad, i), stblk);
        parref_bnd(AD_UPAST(ad, i), stblk);
        parref_bnd(AD_MLPYR(ad, i), stblk);
        parref_bnd(AD_EXTNTAST(ad, i), stblk);
      }
      parref_bnd(AD_NUMELM(ad), stblk);
      parref_bnd(AD_ZBASE(ad), stblk);
    }
  }
}

/**
   \brief set_parref_flag set PARREF flag after semant phase
   \param sptr            shared symbol - set PARREF field and put in uplevel
   structure
   \param psptr           shared symbol - use it as key to search for uplevel
   \param std             statement where the reference of sptr occurs
 */
void
set_parref_flag2(int sptr, int psptr, int std)
{
  int i, stblk, parsyms, ast, key;
  LLUplevel *up;
  if (!SCG(sptr))
    return;
  if (STYPEG(sptr) == ST_MEMBER)
    return;
  if (SCG(sptr) == SC_CMBLK || SCG(sptr) == SC_STATIC)
    return;
  if (SCG(sptr) == SC_EXTERN && ST_ISVAR(sptr)) /* No global vars in uplevel */
    return;
  if (DINITG(sptr) || SAVEG(sptr)) {
    if (SCG(sptr) != SC_LOCAL) {
      if (SCG(sptr) == SC_BASED) {
        int sym = MIDNUMG(sptr);
        if (SCG(sym) != SC_LOCAL)
          return;
      }
    }
  }
  if (std) { /* use std to trace back to previous A_MP_BMPSCOPE */
    int nested = 0;
    std = STD_PREV(std);
    ast = STD_AST(std);
    while (std && ast) {
      if (A_TYPEG(ast) == A_MP_BMPSCOPE) {
        nested++;
        if (nested == 1)
          break;
      }
      if (A_TYPEG(ast) == A_MP_EMPSCOPE)
        nested--;
      std = STD_PREV(std);
      ast = STD_AST(std);
    }
    if (std && ast && A_TYPEG(ast) == A_MP_BMPSCOPE) {
      int paruplevel, astblk;
      astblk = A_STBLKG(ast);
      stblk = A_SPTRG(astblk);
      paruplevel = PARUPLEVELG(stblk);
      mp_add_shared_var(sptr, paruplevel);
    }
    return;
  }
  for (stblk = stb.firstusym; stblk < stb.stg_avail; ++stblk) {
    parsyms = PARSYMSG(stblk);
    if (STYPEG(stblk) == ST_BLOCK && parsyms) {
      /* do exhaustive search for each stblk because we don't know which stblck
       * psptr is in.
       * those MIDNUM/DESCRIPTOR are set very late so there is way to know when
       * we check
       * scope that it needs temp/midnum/etc.  Very inefficient.
       */
      up = llmp_get_uplevel(stblk);
      if (up) {
        if (psptr)
          key = psptr;
        else
          key = sptr;
        for (i = 0; i < up->vals_count; ++i) {
          if (up->vals[i] == key) {
            if (psptr)
              mp_add_shared_var(sptr, stblk);
            else
              set_parref_flag(sptr, sptr, stblk);
          }
        }
      }
    }
  }
}

static void
set_private_bnd_encl(int ast, int scope, int encl)
{
  if (ast && A_TYPEG(ast) == A_ID) {
    int sptr;
    sptr = A_SPTRG(ast);
    SCOPEP(sptr, scope);
    ENCLFUNCP(sptr, encl);
  }
}

void
set_private_encl(int old, int new)
{
  /* make sure its midnum and bound has has scope and encl set - backend relies
   * on it */
  int scope, encl, midnum, sdsc, descr;

  if ((ALLOCG(old) || POINTERG(old)) && new) {

    scope = SCOPEG(new);
    encl = ENCLFUNCG(new);

    midnum = MIDNUMG(new);
    if (midnum) {
      SCOPEP(midnum, scope);
      ENCLFUNCP(midnum, encl);
    }
    sdsc = SDSCG(new);
    if (sdsc) {
      SCOPEP(sdsc, scope);
      ENCLFUNCP(sdsc, encl);
    }
    descr = DESCRG(new);
    if (descr) {
      SCOPEP(descr, scope);
      ENCLFUNCP(descr, encl);
    }
    if (DTY(DTYPEG(new)) == TY_ARRAY) {
      ADSC *ad;
      ad = AD_DPTR(DTYPEG(new));
      if (AD_ADJARR(ad) || ALLOCATTRG(new) || ASSUMSHPG(new)) {
        int i, ndim;
        ndim = AD_NUMDIM(ad);
        for (i = 0; i < ndim; i++) {
          set_private_bnd_encl(AD_LWAST(ad, i), scope, encl);
          set_private_bnd_encl(AD_UPAST(ad, i), scope, encl);
          set_private_bnd_encl(AD_MLPYR(ad, i), scope, encl);
          set_private_bnd_encl(AD_EXTNTAST(ad, i), scope, encl);
        }
        set_private_bnd_encl(AD_NUMELM(ad), scope, encl);
        set_private_bnd_encl(AD_ZBASE(ad), scope, encl);
      }
    }
  }
}

void
set_private_taskflag(int sptr)
{
  /* make sure its midnum and bound has has scope and encl set - backend relies
   * on it */
  int midnum, sdsc;

  if (!sem.task)
    return;

  if (ALLOCG(sptr) || POINTERG(sptr)) {

    midnum = MIDNUMG(sptr);
    if (midnum) {
      TASKP(midnum, 1);
    }
    sdsc = SDSCG(sptr);
    if (sdsc) {
      TASKP(sdsc, 1);
    }
  } else if (ADJARRG(sptr) || RUNTIMEG(sptr)) {
    midnum = MIDNUMG(sptr);
    if (midnum && SCG(midnum) == SC_PRIVATE)
      TASKP(midnum, 1);
  }
}

static int
add_firstprivate_assn(int sptr, int sptr1, int std)
{
  int add = 0;
  if (!sem.task)
    return 0;

  if (std == 0)
    std = STD_PREV(0);
  if (ALLOCG(sptr) || POINTERG(sptr) || ADJARRG(sptr)) {
    int midnum = MIDNUMG(sptr);
    int midnum1 = MIDNUMG(sptr1);
    int sdsc, sdsc1;

    if (midnum && TASKG(midnum)) {
      int midnum1_ast;
      int ast = mk_stmt(A_MP_TASKFIRSTPRIV, 0);
      int midnum_ast = mk_id(midnum);
      if (midnum1) {
        midnum1_ast = mk_id(midnum1);
      } else {
        midnum1_ast = astb.i0;
      }
      A_LOPP(ast, midnum1_ast);
      A_ROPP(ast, midnum_ast);
      add_stmt_after(ast, std);
      add = 1;
    }
    sdsc = SDSCG(sptr);
    sdsc1 = SDSCG(sptr1);
    if (sdsc && TASKG(sdsc)) {
      int sdsc1_ast;
      int ast = mk_stmt(A_MP_TASKFIRSTPRIV, 0);
      int sdsc_ast = mk_id(sdsc);
      if (sdsc1)
        sdsc1_ast = mk_id(sdsc1);
      else
        sdsc1_ast = astb.i0;
      A_LOPP(ast, sdsc1_ast);
      A_ROPP(ast, sdsc_ast);
      add_stmt_after(ast, std);
      add = 1;
    }
  }
  return add;
}

/* Return 'TRUE' if sptr is the shared sptr for a last private value */
static LOGICAL
is_last_private(int sptr)
{
  const REDUC_SYM *sym;

  for (sym = CL_FIRST(CL_LASTPRIVATE); sym; sym = sym->next)
    if (sptr == sym->shared || sptr == sym->Private)
      return TRUE;

  return FALSE;
}

/* Return 'TRUE' if sptr is in the specified clause list */
static LOGICAL
is_in_list(int clause, int sptr)
{
  const ITEM *item;

  for (item = CL_FIRST(clause); item && item != ITEM_END; item = item->next) {
    const int sym = item->t.sptr;
    if (sptr == sym)
      return TRUE;
  }

  return FALSE;
}
#ifdef OMP_OFFLOAD_LLVM

static void
gen_reduction_ompaccel(REDUC *reducp, REDUC_SYM *reduc_symp, LOGICAL rmme,
                       LOGICAL in_parallel)
{
  int ast_reditem;
  REDUC *current_red = reducp;
  REDUC_SYM *current_redsym;
  while (true) {
    if (current_red == NULL)
      break;
    current_redsym = current_red->list;
    while (true) {
      if (current_redsym == NULL)
        break;

      ast_reditem = mk_stmt(A_MP_REDUCTIONITEM, 0);
      A_SHSYMP(ast_reditem, current_redsym->shared);
      A_PRVSYMP(ast_reditem, current_redsym->Private);
      if (current_red->opr == 0)
        A_REDOPRP(ast_reditem, current_red->intrin);
      else
        A_REDOPRP(ast_reditem, current_red->opr);
      add_stmt(ast_reditem);

      current_redsym = current_redsym->next;
    }
    current_red = current_red->next;
  }
}
#endif /* OMP_OFFLOAD_LLVM */

#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
static void
mp_check_maptype(const char *maptype)
{
  if (strcmp(maptype, "tofrom") && strcmp(maptype, "from") &&
      strcmp(maptype, "to") && strcmp(maptype, "alloc") &&
      strcmp(maptype, "release") && strcmp(maptype, "delete"))
    error(1205, ERR_Severe, gbl.lineno, maptype, 0);
}

static void
mp_handle_map_clause(SST *top, int clause, char *maptype, int op, int construct,
                     bool isalways)
{
  ITEM *itemp, *itembeg, *itemend;
  int type = 0;
  type |= OMP_TGT_MAPTYPE_TARGET_PARAM;
  if (isalways)
    type |= OMP_TGT_MAPTYPE_ALWAYS;

  if (!strcmp(maptype, "tofrom"))
    type |= OMP_TGT_MAPTYPE_FROM | OMP_TGT_MAPTYPE_TO;
  else if (!strcmp(maptype, "from"))
    type |= OMP_TGT_MAPTYPE_FROM;
  else if (!strcmp(maptype, "to"))
    type |= OMP_TGT_MAPTYPE_TO;
  else if (!strcmp(maptype, "alloc"))
    type |= OMP_TGT_MAPTYPE_NONE; // todo opmaccel dunno what to pass
  else if (!strcmp(maptype, "delete"))
    type |= OMP_TGT_MAPTYPE_DELETE;
  else if (!strcmp(maptype, "release"))
    type |= OMP_TGT_MAPTYPE_NONE; // todo opmaccel dunno what to pass

  if (construct == DI_TARGETENTERDATA) {
    if (strcmp(maptype, "to") && strcmp(maptype, "alloc")) {
      error(1205, ERR_Severe, gbl.lineno, maptype, 0);
    }
  }
  if (construct == DI_TARGETEXITDATA) {
    if (strcmp(maptype, "from") && strcmp(maptype, "delete") &&
        strcmp(maptype, "release")) {
      error(1203, ERR_Severe, gbl.lineno, maptype, 0);
    }
  }

  itembeg = SST_BEGG(RHS(op));
  itemend = SST_ENDG(RHS(op));
  if (itembeg == ITEM_END)
    return;
  for (itemp = itembeg; itemp != ITEM_END; itemp = itemp->next) {
    itemp->t.cltype = type;
  }
  add_clause(clause, FALSE);
  if (CL_FIRST(clause) == NULL)
    CL_FIRST(clause) = itembeg;
  else
    ((ITEM *)CL_LAST(clause))->next = itembeg;
  CL_LAST(clause) = itemend;
}

static int
get_omp_combined_mode(BIGINT64 type)
{
  BIGINT64 combined_type;
  combined_type = BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_PARDO | BT_SIMD;
  if ((type & combined_type) == combined_type)
    return mode_target_teams_distribute_parallel_for_simd;
  combined_type = BT_TARGET | BT_TEAMS | BT_DISTRIBUTE | BT_PARDO;
  if ((type & combined_type) == combined_type)
    return mode_target_teams_distribute_parallel_for;
  combined_type = BT_TARGET | BT_TEAMS | BT_DISTRIBUTE;
  if ((type & combined_type) == combined_type)
    return mode_target_teams_distribute;
  combined_type = BT_TARGET | BT_TEAMS;
  if ((type & combined_type) == combined_type)
    return mode_target_teams;
  combined_type = BT_TARGET | BT_PARDO;
  if ((type & combined_type) == combined_type)
    return mode_target_parallel_for;
  combined_type = BT_TARGET | BT_PAR;
  if ((type & combined_type) == combined_type)
    return mode_target_parallel;
  combined_type = BT_TARGET | BT_PARDO | BT_SIMD;
  if ((type & combined_type) == combined_type)
    return mode_target_parallel_for_simd;
  combined_type = BT_TARGET | BT_SIMD;
  if ((type & combined_type) == combined_type)
    return mode_target_simd;
  if ((type & BT_TARGET))
    return mode_target;
  return mode_none_target;
  return -1;
}
#endif
/* Return FALSE if the sptr is presented in multiple
 * data sharing clauses: (e.g., shared(x) private(x)),
 * which is illegal.
 *
 * See OpenMP 4.5 specification, page 188, lines 16-17.
 */
static void
check_valid_data_sharing(int sptr)
{
  int count = 0;

  /* In shared list? */
  if (is_in_list(CL_SHARED, sptr))
    ++count;

  /* In private list? */
  if (is_in_list(CL_PRIVATE, sptr)) {
    if (count) {
      error(155, ERR_Severe, gbl.lineno, SYMNAME(sptr),
            "is used in multiple data sharing clauses");
      return;
    } else {
      ++count;
    }
  }

  /* In lastprivate or firstprivate or both? */
  if (is_last_private(sptr) || is_in_list(CL_FIRSTPRIVATE, sptr)) {
    if (count) {
      error(155, ERR_Severe, gbl.lineno, SYMNAME(sptr),
            "is used in multiple data sharing clauses");
    }
  }
}

#ifdef FLANG_SEMSMP_UNUSED
static LOGICAL
check_map_data_sharing(int sptr)
{
  int count = 0;

  /* In shared list? */
  if (is_in_list(CL_SHARED, sptr))
    ++count;

  /* In private list? */
  if (is_in_list(CL_PRIVATE, sptr)) {
    if (count)
      return FALSE;
    else
      ++count;
  }

  if (is_in_list(CL_FIRSTPRIVATE, sptr)) {
    if (count)
      return FALSE;
    else
      ++count;
  }

  if (is_in_list(CL_LASTPRIVATE, sptr)) {
    if (count)
      return FALSE;
    else
      ++count;
  }

  return TRUE;
}
#endif

#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
static LOGICAL is_in_omptarget_data(int d)
{
  if(flg.omptarget && (DI_IN_NEST(d, DI_TARGETENTERDATA) ||
     DI_IN_NEST(d, DI_TARGETEXITDATA) || 
     DI_IN_NEST(d, DI_TARGETDATA)))
    return TRUE;
  return FALSE;
}
#endif

#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
static LOGICAL is_in_omptarget(int d)
{
  if(flg.omptarget && (DI_IN_NEST(d, DI_TARGET) ||
      DI_IN_NEST(d, DI_TARGTEAMSDISTPARDO) ||
      DI_IN_NEST(d, DI_TARGPARDO) ||
      DI_IN_NEST(d, DI_TARGETSIMD) ||
      DI_IN_NEST(d, DI_TARGTEAMSDIST) ||
      DI_IN_NEST(d, DI_TARGETENTERDATA)))
    return TRUE;
  return FALSE;
}
#endif

/**
 * \brief Decide to use optimized atomic usage.
 */
LOGICAL use_opt_atomic(int d)
{
#ifdef OMP_OFFLOAD_LLVM
  return is_in_omptarget(d);
#endif
  return OPT_OMP_ATOMIC;
}

/**
   \brief Decide whether to use llvm atomic for reduction or not.
   Atomic is used only for teams reduction.
 */
static LOGICAL use_atomic_for_reduction(int d)
{
#ifdef OMP_OFFLOAD_LLVM
  if(flg.omptarget && DI_IN_NEST(d, DI_TARGET) ) {
    if(DI_IN_NEST(d, DI_PARDO) ||
        DI_IN_NEST(d, DI_TARGTEAMSDISTPARDO))
      return OPT_OMP_ATOMIC;
    else
      return TRUE;
  }
#endif
  return OPT_OMP_ATOMIC;
}
