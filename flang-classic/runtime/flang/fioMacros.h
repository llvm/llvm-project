/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/**
 * \file
 * \brief descriptor and other definitions
 */

/* TODO FOR FLANG: is this still needed ??? */

#ifndef _PGHPF_H_
#define _PGHPF_H_

#include <stdlib.h>
#include <stdarg.h>
#include "fortDt.h"
#include "fioStructs.h"
#include "FuncArgMacros.h"

/* special argument pointers */

#if defined(_WIN64)
WIN_API __INT_T ENTCOMN(0, 0)[4];
WIN_API __STR_T ENTCOMN(0C, 0c)[1];
#else
#if defined(DESC_I8)
extern __INT4_T ENTCOMN(0, 0)[];
#else
extern __INT_T ENTCOMN(0, 0)[];
#endif
extern __STR_T ENTCOMN(0C, 0c)[];
#endif

#define ABSENT (ENTCOMN(0, 0) + 2)
#define ABSENTC ENTCOMN(0C, 0c)

/* argument pointer tests */

#if defined(DESC_I8)
#define ISPRESENT(p)                                                           \
  ((p) &&                                                                      \
   ((__INT4_T *)(p) < ENTCOMN(0, 0) || (__INT4_T *)(p) > (ENTCOMN(0, 0) + 3)))
#else
#define ISPRESENT(p)                                                           \
  ((p) &&                                                                      \
   ((__INT_T *)(p) < ENTCOMN(0, 0) || (__INT_T *)(p) > (ENTCOMN(0, 0) + 3)))
#endif

#define ISPRESENTC(p) ((CADR(p)) && (CADR(p) != ABSENTC))

/* section descriptor pointer tests */

#define ISSEQUENCE(p) (F90_TAG_G(p) < 0 && F90_TAG_G(p) != -__DESC)

#define ISSCALAR(p) (F90_TAG_G(p) > 0 && F90_TAG_G(p) != __DESC)

#define TYPEKIND(p)                                                            \
  ((dtype)(F90_TAG_G(p) == __DESC                                              \
               ? (p)->kind                                                     \
               : (F90_TAG_G(p) < 0 ? -F90_TAG_G(p) : F90_TAG_G(p))))

/* local mode flag declaration and test macro */

#if defined(_WIN64)
WIN_API __INT_T ENTCOMN(LOCAL_MODE, local_mode)[1];
#else
#if defined(DESC_I8)
extern __INT4_T ENTCOMN(LOCAL_MODE, local_mode)[];
#else
extern __INT_T ENTCOMN(LOCAL_MODE, local_mode)[];
#endif
#endif

/* __gen_block implementation__
 * The following mask is used to check for a gen_block dimension against
 * the isstar argument in ENTFTN(template) and ENTFTN(qopy_in) located
 * in rdst.c
 *
 * We use 3 bit sets per dimension in the range of bits 7..27 of the
 * isstar argument in ENTFTN(template) and ENTFTN(qopy_in) ...
 *
 * isstar == 0   =>    block, block(k), cyclic, or cyclic(k)
 * isstar == 1   =>    gen_block
 * isstar == 2   =>    indirect (not yet supported)
 * isstar == 3   =>    reserved for future expansion
 * isstar == 4   =>    reserved for future expansion
 * isstar == 5   =>    reserved for future expansion
 * isstar == 6   =>    reserved for future expansion
 * isstar == 7   =>    reserved for future expansion
 */

#define EXTENSION_BLOCK_MASK 0xFFFFF80

/* address alignment macros */

#define ALIGNZ 16
#define ALIGNR(x) (((x) + ALIGNZ - 1) & ~(ALIGNZ - 1))

/*
 * message macros for stats/profiling/tracing
 * note: PROF is also referenced in entry.c
 */

void __fort_chn_prune(struct chdr *c);

void __fort_entry_arecv(int, long, int);
void __fort_entry_arecv_done(int);
void __fort_entry_asend(int, long, int);
void __fort_entry_asend_done(int);
void __fort_entry_await(int);
void __fort_entry_await_done(int);
void __fort_entry_copy(long);
void __fort_entry_copy_done(void);
void __fort_entry_init(void);
void __fort_entry_recv(int, long);
void __fort_entry_recv_done(int);
void __fort_entry_send(int, long);
void __fort_entry_send_done(int);
void __fort_entry_term(void);

void __fort_sethand(void);

int __fort_stat_init(void);
void __fort_stat_arecv(int cpu, long len, int reqn);
void __fort_stat_arecv_done(int cpu);
void __fort_stat_asend(int cpu, long len, int reqn);
void __fort_stat_asend_done(int cpu);
void __fort_stat_await(int reqn);
void __fort_stat_await_done(int reqn);
void __fort_stat_copy(long len);
void __fort_stat_copy_done(void);
void __fort_stat_function_entry(int line, int lines, int cline, char *func,
                               char *file, int funcl, int filel);
void __fort_stat_function_exit(void);
void __fort_stat_line_entry(int line);
void __fort_stat_recv(int cpu, long len);
void __fort_stat_recv_done(int cpu);
void __fort_stat_send(int cpu, long len);
void __fort_stat_send_done(int cpu);
void __fort_stat_term(void);

int __fort_prof_init(void);
void __fort_prof_arecv(int cpu, long len, int reqn);
void __fort_prof_arecv_done(int cpu);
void __fort_prof_asend(int cpu, long len, int reqn);
void __fort_prof_asend_done(int cpu);
void __fort_prof_await(int reqn);
void __fort_prof_await_done(int reqn);
void __fort_prof_copy(long len);
void __fort_prof_copy_done(void);
void __fort_prof_function_entry(int line, int lines, int cline, char *func,
                               char *file, int funcl, int filel);
void __fort_prof_function_exit(void);
void __fort_prof_line_entry(int line);
void __fort_prof_recv(int cpu, long len);
void __fort_prof_recv_done(int cpu);
void __fort_prof_send(int cpu, long len);
void __fort_prof_send_done(int cpu);
void __fort_prof_term(void);

void __fort_procargs(void);

int __fort_trac_init(void);
void __fort_trac_arecv(int cpu, long len, int reqn);
void __fort_trac_arecv_done(int cpu);
void __fort_trac_asend(int cpu, long len, int reqn);
void __fort_trac_asend_done(int cpu);
void __fort_trac_await(int reqn);
void __fort_trac_await_done(int reqn);
void __fort_trac_copy(long len);
void __fort_trac_copy_done(void);
void __fort_trac_function_entry(int line, int lines, int cline, DCHAR(func),
                               DCHAR(file) DCLEN64(funcl) DCLEN64(filel));
void __fort_trac_function_exit(void);
void __fort_trac_line_entry(int line);
void __fort_trac_recv(int cpu, long len);
void __fort_trac_recv_done(int cpu);
void __fort_trac_send(int cpu, long len);
void __fort_trac_send_done(int cpu);
void __fort_trac_term(void);

void __fort_traceback(void);

void __fort_begpar(int ncpus);
void __fort_endpar(void);
void __fort_abortx(void);
void __fort_setarg(void);

void __fort_set_second(double d);

#define __DIST_ENTRY_RECV(cpu, len)
#define __DIST_ENTRY_RECV_DONE(cpu)
#define __DIST_ENTRY_SEND(cpu, len)
#define __DIST_ENTRY_SEND_DONE(cpu)
#define __DIST_ENTRY_COPY(len)

#define __DIST_ENTRY_COPY_DONE()
#define __DIST_ENTRY_ARECV(cpu, len, reqn)
#define __DIST_ENTRY_ARECV_DONE(cpu)
#define __DIST_ENTRY_ASEND(cpu, len, reqn)
#define __DIST_ENTRY_ASEND_DONE(cpu)
#define __DIST_ENTRY_AWAIT(reqn)
#define __DIST_ENTRY_AWAIT_DONE(reqn)

#if defined(_WIN64)
/* prototypes for HPF */

/* The PG compilers cannot yet create dlls that share data. For
 * data shared in this way, a function interface is provided for
 * the windows environment.
 */

extern __LOG_T __get_fort_mask_log(void);
extern __LOG1_T __get_fort_mask_log1(void);
extern __LOG2_T __get_fort_mask_log2(void);
extern __LOG4_T __get_fort_mask_log4(void);
extern __LOG8_T __get_fort_mask_log8(void);
extern __INT1_T __get_fort_mask_int1(void);
extern __INT2_T __get_fort_mask_int2(void);
extern __INT4_T __get_fort_mask_int4(void);
extern __INT8_T __get_fort_mask_int8(void);
extern __LOG_T __get_fort_true_log(void);
extern __LOG_T *__get_fort_true_log_addr(void);
extern __LOG1_T __get_fort_true_log1(void);
extern __LOG2_T __get_fort_true_log2(void);
extern __LOG4_T __get_fort_true_log4(void);
extern __LOG8_T __get_fort_true_log8(void);

extern void __set_fort_mask_log(__LOG_T);
extern void __set_fort_mask_log1(__LOG1_T);
extern void __set_fort_mask_log2(__LOG2_T);
extern void __set_fort_mask_log4(__LOG4_T);
extern void __set_fort_mask_log8(__LOG8_T);
extern void __set_fort_mask_int1(__INT1_T);
extern void __set_fort_mask_int2(__INT2_T);
extern void __set_fort_mask_int4(__INT4_T);
extern void __set_fort_mask_int8(__INT8_T);
extern void __set_fort_true_log(__LOG_T);
extern void __set_fort_true_log1(__LOG1_T);
extern void __set_fort_true_log2(__LOG2_T);
extern void __set_fort_true_log4(__LOG4_T);
extern void __set_fort_true_log8(__LOG8_T);

#define GET_DIST_MASK_LOG __get_fort_mask_log()
#define GET_DIST_MASK_LOG1 __get_fort_mask_log1()
#define GET_DIST_MASK_LOG2 __get_fort_mask_log2()
#define GET_DIST_MASK_LOG4 __get_fort_mask_log4()
#define GET_DIST_MASK_LOG8 __get_fort_mask_log8()
#define GET_DIST_MASK_INT1 __get_fort_mask_int1()
#define GET_DIST_MASK_INT2 __get_fort_mask_int2()
#define GET_DIST_MASK_INT4 __get_fort_mask_int4()
#define GET_DIST_MASK_INT8 __get_fort_mask_int8()
#define GET_DIST_TRUE_LOG __get_fort_true_log()
#define GET_DIST_TRUE_LOG_ADDR __get_fort_true_log_addr()
#define GET_DIST_TRUE_LOG1 __get_fort_true_log1()
#define GET_DIST_TRUE_LOG2 __get_fort_true_log2()
#define GET_DIST_TRUE_LOG4 __get_fort_true_log4()
#define GET_DIST_TRUE_LOG8 __get_fort_true_log8()

#define SET_DIST_MASK_LOG(n) __set_fort_mask_log(n)
#define SET_DIST_MASK_LOG1(n) __set_fort_mask_log1(n)
#define SET_DIST_MASK_LOG2(n) __set_fort_mask_log2(n)
#define SET_DIST_MASK_LOG4(n) __set_fort_mask_log4(n)
#define SET_DIST_MASK_LOG8(n) __set_fort_mask_log8(n)
#define SET_DIST_MASK_INT1(n) __set_fort_mask_int1(n)
#define SET_DIST_MASK_INT2(n) __set_fort_mask_int2(n)
#define SET_DIST_MASK_INT4(n) __set_fort_mask_int4(n)
#define SET_DIST_MASK_INT8(n) __set_fort_mask_int8(n)
#define SET_DIST_TRUE_LOG(n) __set_fort_true_log(n)
#define SET_DIST_TRUE_LOG1(n) __set_fort_true_log1(n)
#define SET_DIST_TRUE_LOG2(n) __set_fort_true_log2(n)
#define SET_DIST_TRUE_LOG4(n) __set_fort_true_log4(n)
#define SET_DIST_TRUE_LOG8(n) __set_fort_true_log8(n)

extern void *__get_fort_maxs(int);
extern void *__get_fort_mins(int);
extern int __get_fort_shifts(int);
extern int __get_fort_size_of(int);
extern void *__get_fort_trues(int);
extern const char *__get_fort_typenames(int);
extern void *__get_fort_units(int);

extern void __set_fort_maxs(int, void *);
extern void __set_fort_mins(int, void *);
extern void __set_fort_shifts(int, int);
extern void __set_fort_size_of(int, int);
extern void __set_fort_trues(int, void *);
extern void __set_fort_typenames(int, const char *);
extern void __set_fort_units(int, void *);

#define GET_DIST_MAXS(idx) __get_fort_maxs(idx)
#define GET_DIST_MINS(idx) __get_fort_mins(idx)
#define GET_DIST_SHIFTS(idx) __get_fort_shifts(idx)
#define GET_DIST_SIZE_OF(idx) __get_fort_size_of(idx)
#define GET_DIST_TRUES(idx) __get_fort_trues(idx)
#define GET_DIST_TYPENAMES(idx) __get_fort_typenames(idx)
#define GET_DIST_UNITS(idx) __get_fort_units(idx)

#define SET_DIST_MAXS(idx, val) __set_fort_maxs(idx, val)
#define SET_DIST_MINS(idx, val) __set_fort_mins(idx, val)
#define SET_DIST_SHIFTS(idx, val) __set_fort_shifts(idx, val)
#define SET_DIST_SIZE_OF(idx, val) __set_fort_size_of(idx, val)
#define SET_DIST_TRUES(idx, val) __set_fort_trues(idx, val)
#define SET_DIST_TYPENAMES(idx, val) __set_fort_typenames(idx, val)
#define SET_DIST_UNITS(idx, val) __set_fort_units(idx, val)

extern long long int *__get_fort_one(void);
extern long long int *__get_fort_zed(void);

#define GET_DIST_ONE __get_fort_one()
#define GET_DIST_ZED __get_fort_zed()

extern int __get_fort_debug(void);
extern int __get_fort_debugn(void);
extern long __get_fort_heapz(void);
extern int __get_fort_ioproc(void);
extern int __get_fort_lcpu(void);
extern int *__get_fort_lcpu_addr(void);
extern int __get_fort_pario(void);
extern int __get_fort_quiet(void);
extern int __get_fort_tcpus(void);
extern int *__get_fort_tcpus_addr(void);
extern int *__get_fort_tids(void);
extern int __get_fort_tids_elem(int);
extern const char *__get_fort_transnam(void);

extern void __set_fort_debug(int);
extern void __set_fort_debugn(int);
extern void __set_fort_heapz(long heapz);
extern void __set_fort_ioproc(int);
extern void __set_fort_lcpu(int);
extern void __set_fort_pario(int);
extern void __set_fort_quiet(int);
extern void __set_fort_tcpus(int);
extern void __set_fort_tids(int *);
extern void __set_fort_tids_elem(int, int);

#define GET_DIST_IOPROC 0
#define GET_DIST_TRANSNAM "rpm1"
#define GET_DIST_HEAPZ 0
#define GET_DIST_LCPU 0
#define GET_DIST_TCPUS 1
#define GET_DIST_DEBUG __get_fort_debug()
#define GET_DIST_DEBUGN __get_fort_debugn()
#define GET_DIST_LCPU_ADDR __get_fort_lcpu_addr()
#define GET_DIST_PARIO __get_fort_pario()
#define GET_DIST_QUIET __get_fort_quiet()
#define GET_DIST_TCPUS_ADDR __get_fort_tcpus_addr()
#define GET_DIST_TIDS __get_fort_tids()
#define GET_DIST_TIDS_ELEM(idx) __get_fort_tids_elem(idx)

#define SET_DIST_DEBUG(n) __set_fort_debug(n)
#define SET_DIST_DEBUGN(n) __set_fort_debugn(n)
#define SET_DIST_HEAPZ(n) __set_fort_heapz(n)
#define SET_DIST_IOPROC(n) __set_fort_ioproc(n)
#define SET_DIST_LCPU(n) __set_fort_lcpu(n)
#define SET_DIST_PARIO(n) __set_fort_pario(n)
#define SET_DIST_QUIET(n) __set_fort_quiet(n)
#define SET_DIST_TCPUS(n) __set_fort_tcpus(n)
#define SET_DIST_TIDS(n) __set_fort_tids(n)
#define SET_DIST_TIDS_ELEM(idx, val) __set_fort_tids_elem(idx, val)

#else /* linux */

extern __LOG_T __fort_mask_log;
extern __LOG1_T __fort_mask_log1;
extern __LOG2_T __fort_mask_log2;
extern __LOG4_T __fort_mask_log4;
extern __LOG8_T __fort_mask_log8;
extern __INT1_T __fort_mask_int1;
extern __INT2_T __fort_mask_int2;
extern __INT4_T __fort_mask_int4;
extern __INT8_T __fort_mask_int8;
extern __LOG_T __fort_true_log;
extern __LOG1_T __fort_true_log1;
extern __LOG2_T __fort_true_log2;
extern __LOG4_T __fort_true_log4;
extern __LOG8_T __fort_true_log8;

#define GET_DIST_MASK_LOG __fort_mask_log
#define GET_DIST_MASK_LOG1 __fort_mask_log1
#define GET_DIST_MASK_LOG2 __fort_mask_log2
#define GET_DIST_MASK_LOG4 __fort_mask_log4
#define GET_DIST_MASK_LOG8 __fort_mask_log8
#define GET_DIST_MASK_INT1 __fort_mask_int1
#define GET_DIST_MASK_INT2 __fort_mask_int2
#define GET_DIST_MASK_INT4 __fort_mask_int4
#define GET_DIST_MASK_INT8 __fort_mask_int8
#define GET_DIST_TRUE_LOG __fort_true_log
#define GET_DIST_TRUE_LOG_ADDR &__fort_true_log
#define GET_DIST_TRUE_LOG1 __fort_true_log1
#define GET_DIST_TRUE_LOG2 __fort_true_log2
#define GET_DIST_TRUE_LOG4 __fort_true_log4
#define GET_DIST_TRUE_LOG8 __fort_true_log8

#define SET_DIST_MASK_LOG(n) __fort_mask_log = n
#define SET_DIST_MASK_LOG1(n) __fort_mask_log1 = n
#define SET_DIST_MASK_LOG2(n) __fort_mask_log2 = n
#define SET_DIST_MASK_LOG4(n) __fort_mask_log4 = n
#define SET_DIST_MASK_LOG8(n) __fort_mask_log8 = n
#define SET_DIST_MASK_INT1(n) __fort_mask_int1 = n
#define SET_DIST_MASK_INT2(n) __fort_mask_int2 = n
#define SET_DIST_MASK_INT4(n) __fort_mask_int4 = n
#define SET_DIST_MASK_INT8(n) __fort_mask_int8 = n
#define SET_DIST_TRUE_LOG(n) __fort_true_log = n
#define SET_DIST_TRUE_LOG1(n) __fort_true_log1 = n
#define SET_DIST_TRUE_LOG2(n) __fort_true_log2 = n
#define SET_DIST_TRUE_LOG4(n) __fort_true_log4 = n
#define SET_DIST_TRUE_LOG8(n) __fort_true_log8 = n

extern void *__fort_maxs[__NTYPES];
extern void *__fort_mins[__NTYPES];
extern int __fort_shifts[__NTYPES];
extern void *__fort_trues[__NTYPES];
extern const char *__fort_typenames[__NTYPES];
extern void *__fort_units[__NTYPES];

#define GET_DIST_MAXS(idx) __fort_maxs[idx]
#define GET_DIST_MINS(idx) __fort_mins[idx]
#define GET_DIST_SHIFTS(idx) __fort_shifts[idx]
#define GET_DIST_SIZE_OF(idx) __fort_size_of[idx]
#define GET_DIST_TRUES(idx) __fort_trues[idx]
#define GET_DIST_TYPENAMES(idx) __fort_typenames[idx]
#define GET_DIST_UNITS(idx) __fort_units[idx]

#define SET_DIST_MAXS(idx, val) __fort_maxs[idx] = val
#define SET_DIST_MINS(idx, val) __fort_mins[idx] = val
#define SET_DIST_SHIFTS(idx, val) __fort_shifts[idx] = val
#define SET_DIST_SIZE_OF(idx, val) __fort_size_of[idx] = val
#define SET_DIST_TRUES(idx, val) __fort_trues[idx] = val
#define SET_DIST_TYPENAMES(idx, val) __fort_typenames[idx] = val
#define SET_DIST_UNITS(idx, val) __fort_units[idx] = val

extern long long int __fort_one[4];
extern long long int __fort_zed[4];

#define GET_DIST_ONE __fort_one
#define GET_DIST_ZED __fort_zed

#include "fort_vars.h"
extern const char *__fort_transnam;

#define GET_DIST_HEAPZ 0
#define GET_DIST_IOPROC 0
#define GET_DIST_LCPU 0
#define GET_DIST_TCPUS 1
#define GET_DIST_TRANSNAM "rpm1"
#define GET_DIST_DEBUG __fort_debug
#define GET_DIST_DEBUGN __fort_debugn
#define GET_DIST_LCPU_ADDR &__fort_lcpu
#define GET_DIST_PARIO __fort_pario
#define GET_DIST_QUIET __fort_quiet
#define GET_DIST_TCPUS_ADDR &__fort_tcpus
#define GET_DIST_TIDS __fort_tids
#define GET_DIST_TIDS_ELEM(idx) __fort_tids[idx]

#define SET_DIST_DEBUG(n) __fort_debug = n
#define SET_DIST_DEBUGN(n) __fort_debugn = n
#define SET_DIST_HEAPZ(n) __fort_heapz = n
#define SET_DIST_IOPROC(n) __fort_ioproc = n
#define SET_DIST_LCPU(n) __fort_lcpu = n
#define SET_DIST_PARIO(n) __fort_pario = n
#define SET_DIST_QUIET(n) __fort_quiet = n
#define SET_DIST_TCPUS(n) __fort_tcpus = n
#define SET_DIST_TIDS(n) __fort_tids = n
#define SET_DIST_TIDS_ELEM(idx, val) __fort_tids[idx] = val

#endif /* windows, linux */

extern int __fort_size_of[__NTYPES];

extern int __fort_entry_mflag;

/* values for __fort_quiet */
/* stats */
#define Q_CPU 0x01             /* cpu */
#define Q_CPUS 0x02            /* cpus */
#define Q_MSG 0x04             /* message */
#define Q_MSGS 0x08            /* messages */
#define Q_MEM 0x10             /* memory */
#define Q_MEMS 0x20            /* memories */
#define Q_PROF 0x40            /* profile output (not used) */
#define Q_TRAC 0x80            /* trace output (not used) */
                               /* profiling */
#define Q_PROF_AVG 0x00400000  /* output min/avg/max for profile */
#define Q_PROF_NONE 0x00800000 /* disable profiling */

/* maximum number of fortran array dimensions */

#define MAXDIMS 7

/* generic all-dimensions bit mask */

#define ALLDIMS (~(-1 << MAXDIMS))

/* __fort_test debug flags */

#define DEBUG_COPYIN_GLOBAL 0x00001
#define DEBUG_COPYIN_LOCAL 0x00002
#define DEBUG_EXCH 0x00004
#define DEBUG_GRAD 0x00008
#define DEBUG_MMUL 0x00010
#define DEBUG_OLAP 0x00020
#define DEBUG_RDST 0x00040
#define DEBUG_REDU 0x00080
#define DEBUG_SCAL 0x00100
#define DEBUG_SCAN 0x00200
#define DEBUG_SCAT 0x00400
#define DEBUG_LVAL 0x00800
#define DEBUG_HFIO 0x01000
#define DEBUG_ALLO 0x02000
#define DEBUG_TIME 0x04000
#define DEBUG_CSHF 0x08000
#define DEBUG_EOSH 0x10000
#define DEBUG_CHAN 0x20000
#define DEBUG_COPY 0x40000
#define DEBUG_DIST 0x80000
#define DEBUG_CHECK 0x100000

#define __CORMEM_M0 666
#define __CORMEM_M1 0x58ad5e3
#define __CORMEM_M2 0x61f072b

#if defined(_WIN64)
WIN_API __INT_T *CORMEM;
#else
#if defined(DESC_I8)
extern __INT4_T CORMEM[];
#else
extern __INT_T CORMEM[];
#endif
#endif /* C90 */

#if !defined(NPLIMIT)
#define NPLIMIT 0
#endif /* NPLIMIT */

#if defined(_WIN64)
void __CORMEM_SCAN(void);
#else
#define __CORMEM_SCAN()                                                        \
  {                                                                            \
    __INT_T n;                                                                 \
    n = ((__CORMEM_M1 ^ CORMEM[2]) - CORMEM[0]) - __CORMEM_M0;                 \
    if (n != ((__CORMEM_M2 ^ CORMEM[7]) + CORMEM[1]) - __CORMEM_M0)            \
      __fort_abort("Memory corrupted, aborting\n");                             \
    if (NPLIMIT != 0 && (n == 0 || n > NPLIMIT))                               \
      n = NPLIMIT;                                                             \
    if (n != 0 && GET_DIST_TCPUS > n)                                         \
      __fort_abort("Number of processors exceeds license\n");                   \
  }
#endif

/* arithmetic macros */

#define Sign(y) ((y) < 0 ? -1 : 1)
#define Abs(y) ((y) < 0 ? -(y) : (y))
#define Ceil(x, y)                                                             \
  (((x) ^ (y)) > 0 ? (Abs(x) + Abs(y) - 1) / Abs(y) : (x) / (y))
#define Floor(x, y)                                                            \
  (((x) ^ (y)) < 0 ? -(Abs(x) + Abs(y) - 1) / Abs(y) : (x) / (y))
#define Min(x, y) ((x) < (y) ? (x) : (y))
#define Max(x, y) ((x) > (y) ? (x) : (y))

/* Multiplication by a reciprocal replaces division.  The low order
   binary fraction part (mantissa) of the reciprocal is stored in the
   descriptor as an unsigned integer. */

#define RECIP_FRACBITS 32
#define RECIP(y) (1UL + ((unsigned long)0xFFFFFFFF) / ((unsigned long)(y)))

#define RECIP_DIV(q, x, y)                                                     \
  {                                                                            \
    *(q) = ((y) == 1 ? (x) : (x) / (y));                                       \
  }
#define RECIP_MOD(r, x, y)                                                     \
  {                                                                            \
    *(r) = ((y) == 1 ? 0 : (x) % (y));                                         \
  }
#define RECIP_DIVMOD(q, r, x, y)                                               \
  {                                                                            \
    if ((y) == 1)                                                              \
      *(q) = (x), *(r) = 0;                                                    \
    else {                                                                     \
      register long _p_ = (x) / (y);                                           \
      register long _m_ = (x)-_p_ * (y);                                       \
      *(q) = _p_, *(r) = _m_;                                                  \
    }                                                                          \
  }

/* descriptor flags */

/** \def __ASSUMED_SIZE
 *  \brief This descriptor flag is used to indicate that the array is declared
 *         assumed size.
 */
#define __ASSUMED_SIZE 0x00000001
#define __SEQUENCE 0x00000002
/** \def __ASSUMED_SHAPE
 *  \brief This descriptor flag is used to indicate that the array is declared
 *         assumed shape.
 */
#define __ASSUMED_SHAPE 0x00000004
/** \def __SAVE
 *  \brief descriptor flag (reserved)
 */
#define __SAVE 0x00000008
/** \def __INHERIT
 *  \brief descriptor flag (reserved)
 */
#define __INHERIT 0x00000010
/** \def __NO_OVERLAPS
 *  \brief descriptor flag (reserved)
 */
#define __NO_OVERLAPS 0x00000020

typedef enum { __INOUT = 0, __IN = 1, __OUT = 2 } _io_intent;

#define __INTENT_MASK 0x3
#define __INTENT_SHIFT 6
#define __INTENT_INOUT (__INOUT << __INTENT_SHIFT)
#define __INTENT_IN (__IN << __INTENT_SHIFT)
#define __INTENT_OUT (__OUT << __INTENT_SHIFT)

typedef enum {
  __OMITTED = 0,
  __PRESCRIPTIVE = 1,
  __DESCRIPTIVE = 2,
  __TRANSCRIPTIVE = 3
} _io_spec;

#define __DIST_TARGET_MASK 0x3
#define __DIST_TARGET_SHIFT 8
#define __PRESCRIPTIVE_DIST_TARGET (__PRESCRIPTIVE << __DIST_TARGET_SHIFT)
#define __DESCRIPTIVE_DIST_TARGET (__DESCRIPTIVE << __DIST_TARGET_SHIFT)
#define __TRANSCRIPTIVE_DIST_TARGET (__TRANSCRIPTIVE << __DIST_TARGET_SHIFT)

#define __DIST_FORMAT_MASK 0x3
#define __DIST_FORMAT_SHIFT 10
#define __PRESCRIPTIVE_DIST_FORMAT (__PRESCRIPTIVE << __DIST_FORMAT_SHIFT)
#define __DESCRIPTIVE_DIST_FORMAT (__DESCRIPTIVE << __DIST_FORMAT_SHIFT)
#define __TRANSCRIPTIVE_DIST_FORMAT (__TRANSCRIPTIVE << __DIST_FORMAT_SHIFT)

#define __ALIGN_TARGET_MASK 0x3
#define __ALIGN_TARGET_SHIFT 12
#define __PRESCRIPTIVE_ALIGN_TARGET (__PRESCRIPTIVE << __ALIGN_TARGET_SHIFT)
#define __DESCRIPTIVE_ALIGN_TARGET (__DESCRIPTIVE << __ALIGN_TARGET_SHIFT)

/** \def __IDENTITY_MAP
 *  \brief descriptor flag (reserved)
 */
#define __IDENTITY_MAP 0x00004000
/** \def __DYNAMIC
 *  \brief descriptor flag (reserved)
 */
#define __DYNAMIC 0x00008000
/** \def __TEMPLATE
 *  \brief descriptor flag (reserved)
 */
#define __TEMPLATE 0x00010000
/** \def __LOCAL
 *  \brief descriptor flag (reserved)
 */
#define __LOCAL 0x00020000
/** \def __F77_LOCAL_DUMMY
 *  \brief descriptor flag (reserved)
 */
#define __F77_LOCAL_DUMMY 0x00040000

/** \def __OFF_TEMPLATE
 *  \brief descriptor flag (reserved)
 */
#define __OFF_TEMPLATE 0x00080000
/** \def __DIST_TARGET_AXIS
 *  \brief descriptor flag (reserved)
 */
#define __DIST_TARGET_AXIS 0x00100000
/** \def __ASSUMED_OVERLAPS
 *  \brief descriptor flag (reserved)
 */
#define __ASSUMED_OVERLAPS 0x00200000
/** \def __SECTZBASE
 *  \brief When creating a section of an array, set up each dimension and 
 *  compute GSIZE (see ENTF90(SECT, sect) in dist.c).
 */
#define __SECTZBASE 0x00400000
/** \def __BOGUSBOUNDS
 *  \brief When creating an array section, defer set up of the bounds in the
 *         descriptor after a copy.
 */
#define __BOGUSBOUNDS 0x00800000
/** \def __NOT_COPIED
 *  \brief descriptor flag (reserved)
 */
#define __NOT_COPIED 0x01000000
/** \def __NOREINDEX 
 *  \brief When creating an array section, use the existing bounds.
 *         Do not reset the lower bound to 1 and the upper bound to 
 *         the extent.
 */
#define __NOREINDEX 0x02000000
/** \def __ASSUMED_GB_EXTENT
 *  \brief descriptor flag (reserved)
 */
#define __ASSUMED_GB_EXTENT 0x08000000
/** \def __DUMMY_COLLAPSE_PAXIS
 *  \brief descriptor flag (reserved)
 */
#define __DUMMY_COLLAPSE_PAXIS 0x10000000

/** \def __SEQUENTIAL_SECTION
 *
 * Used to determine if an array section passed as a parameter to an
 * F77 subroutine needs to be copied or whether it can be passed as is.
 * Set in ptr_assign and tested by the inline code.
 */
#define __SEQUENTIAL_SECTION 0x20000000

/* processors descriptor */

typedef struct procdim procdim;
typedef struct proc proc;

#define PROC_HDR_INT_LEN 5
#define PROC_DIM_INT_LEN 5

struct procdim {
  __INT_T shape;       /* (1) extent of processor dimension */
  __INT_T shape_shift; /* (2) shape div shift amount */
  __INT_T shape_recip; /* (3) 1/shape mantissa */
  __INT_T coord;       /* (4) this processor's coordinate */
  __INT_T stride;      /* (5) coordinate multiplier */
};

struct proc {
  __INT_T tag;   /* (1) structure type tag == __PROC */
  __INT_T rank;  /* (2) processor arrangement rank */
  __INT_T flags; /* (3) descriptor flags */
  __INT_T base;  /* (4) base processor number */
  __INT_T size;  /* (5) size of processor arrangement */

  /* the following array must be the last member of this struct.  its
     length is adjusted to equal rank when dynamically allocated. */

  procdim dim[MAXDIMS]; /* per dimension data */
};

/* * * * * NEW DESCRIPTOR STRUCTURE * * * * */

/* The compiler declares these descriptors as arrays of integers which
   may be read or written directly in the generated code.  The
   descriptor header contains pointers so it must be properly aligned
   and the size must reflect the possible difference in length between
   an integer and a pointer.  These declarations must correspond to
   the interface defined in frontend rte.h file */

#define F90_DESC_HDR_INT_LEN 8
#define F90_DESC_HDR_PTR_LEN 2
#define F90_DESC_DIM_INT_LEN 6
#define F90_DESC_DIM_PTR_LEN 0

#define DIST_DESC_HDR_INT_LEN 16
#define DIST_DESC_HDR_PTR_LEN 4
#define DIST_DESC_DIM_INT_LEN 34
#define DIST_DESC_DIM_PTR_LEN 1

typedef struct F90_Desc F90_Desc;
typedef struct F90_DescDim F90_DescDim;
typedef struct DIST_Desc DIST_Desc;
typedef struct DIST_DescDim DIST_DescDim;

/** \brief Fortran descriptor dimension info 
 *
 * Each F90_Desc structure below has up to \ref MAXDIMS number of F90_DescDim
 * structures that correspond to each dimension of an array. Each F90_DescDim 
 * has 6 fields: \ref lbound, \ref extent, \ref sstride, \ref soffset, 
 * \ref lstride, and \ref ubound.
 *
 * The \ref lbound field is the lowerbound of the dimension.
 *
 * The \ref extent field is the extent of the dimension (e.g., 
 * extent = max((ubound - lbound) + 1), 0).
 *
 * Fields \ref sstride (i.e., section index stride on array) and \ref soffset
 * (i.e., section offset onto array) are not needed in the Fortran runtime.
 * They were needed in languages like HPF. For Fortran, their corresponding
 * macros, \ref F90_DPTR_SSTRIDE_G and \ref F90_DPTR_SOFFSET_G are set to 
 * constants (see below). However, we still need to preserve the space in 
 * F90_DescDim for backward compatibility.
 *
 * The field \ref lstride is the "section index multiplier" for the dimension. 
 * It is used in the mapping of an array section dimension to its original 
 * array. See the __fort_finish_descriptor() runtime routine in dist.c for an 
 * example of how to compute this field. See the print_loop() runtime routine
 * in dbug.c for an example of how to use this field.
 *
 * The \ref ubound field is the upperbound of the dimension.
 */
struct F90_DescDim {/* descriptor dimension info */

  __INT_T lbound;  /**< (1) lower bound */
  __INT_T extent;  /**< (2) array extent */
  __INT_T sstride; /**< (3) reserved */
  __INT_T soffset; /**< (4) reserved */
  __INT_T lstride; /**< (5) section index multiplier */
  __INT_T ubound;  /**< (6) upper bound */
};

/* type descriptor forward reference. Declared in type.h */
typedef struct type_desc TYPE_DESC;

/** \brief Fortran descriptor header 
 *
 * The fields minus F90_DescDim below should remain consistent
 * with the object_desc and proc_desc structures of type.h in terms of length 
 * and type. These fields are also mirrored in the Fortran Front-end's
 * rte.h header file.
 *
 * The \ref tag field is used to identify the type of the descriptor. This is
 * typically \ref __DESC, which identifies the descriptor as a regular array
 * section descriptor. If the tag is a basic type, such as an \ref __INT4, 
 * \ref__REAL4, etc. then it is a 1 word pseudo descriptor. The pseudo
 * descriptor is used as a place holder when we want to pass a scalar into a 
 * runtime routine that also requires a descriptor argument.  When tag is
 * \ref __POLY, then we have an object_desc (see its definition in type.h). 
 * When tag is \ref __PROCPTR, we have a proc_desc (see its definition in 
 * type.h). 
 *
 * The \ref rank field equals the total number of dimensions of the associated
 * array. If rank is 0, then this descriptor may be associated with a
 * derived type object, a pointer to a derived type object, or an
 * allocatable scalar.
 * 
 * The \ref kind field holds the base type of the associated array. It is one
 * of the basic types defined in \ref _DIST_TYPE.
 *
 * The \ref flags field holds various descriptor flags defined above. Most of
 * the flags defined above, denoted as reserved,  are not used for Fortran. 
 * The flags that are typically used for Fortran are \ref __ASSUMED_SIZE and 
 * \ref __ASSUMED_SHAPE.
 *
 * The \ref len field holds the byte length of the associated array's base type
 * (see also kind field).
 *
 * The \ref lsize field holds the total number of elements in the associated 
 * array section.
 *
 * In distributed memory languages, such as HPF, \ref gsize represents the total
 * number of elements that are distributed across multiple processors. In 
 * Fortran, the \ref lsize and \ref gsize fields are usually the same, however 
 * this is a case in the reshape intrinsic where \ref gsize != \ref lsize. There
 * may just be an incidental difference during the execution of reshape. There 
 * may also be others in the Fortran runtime where \ref gsize != \ref lsize, 
 * however, they too may just be incidental differences. Therefore, use 
 * \ref lsize instead of \ref gsize when querying the total number of elements
 * in a Fortran array.
 *
 * The \ref lbase field is the index offset section adjustment. It is used in 
 * the mapping of an array section to its original array.  
 * See the \ref __DIST_SET_SECTIONXX and \ref __DIST_SET_SECTIONX macros below  
 * for examples of how to compute this field. See the __fort_print_local() 
 * runtime routine in dbug.c for an example of how to use this field.
 *
 * The \ref gbase field historically was used in distributed memory languages 
 * like HPF. Therefore, \ref gbase is usually 0 and may always be 0 in the  
 * Fortran runtime (needs more investigation to confirm if it's always 0 in 
 * this case).
 *
 * When set, the \ref dist_desc field holds a pointer to the type descriptor of
 * the associated object (see also the \ref TYPE_DESC definition in type.h). 
 *
 * The \ref dim fields hold up to \ref number of F90_DescDim structures. It's
 * also possible that \ref dim is empty when this descriptor is associated with
 * a derived type, a pointer to a derived type, or an allocatable scalar.
 *
 * The number in paranthesis for each field below corresponds with the
 * subscript index in the descriptor. The first 9 values are denoted. 
 * The first index is 1 because we assume Fortran style arrays when we
 * reference these fields in the Fortran front-end. When generating assembly,
 * the first index will be 0. After the gbase field,
 * the subscript value depends on three conditions: Whether the target's
 * pointers are 64-bit, whether the target's native integers are 64-bit, 
 * and whether large arrays are enabled. See the \ref DESC_HDR_LEN macro
 * in the Fortran front-end's rte.h file for more information. The first
 * 9 subscript values are also mirrored in the following macros in
 * rte.h: \ref DESC_HDR_TAG, \ref DESC_HDR_RANK, \ref DESC_HDR_KIND, 
 * \ref DESC_HDR_BYTE_LEN, \ref DESC_HDR_FLAGS, \ref DESC_HDR_LSIZE, 
 * \ref DESC_HDR_GSIZE, \ref DESC_HDR_LBASE, and \ref DESC_HDR_GBASE.
 *
 */
struct F90_Desc {

  __INT_T tag;                 /**< (1) tag field; usually \ref __DESC 
                                        (see also _DIST_TYPE) */
  __INT_T rank;                /**< (2) array section rank */
  __INT_T kind;                /**< (3) array base type */
  __INT_T len;                 /**< (4) byte length of base type */
  __INT_T flags;               /**< (5) descriptor flags */
  __INT_T lsize;               /**< (6) local array section size */
  __INT_T gsize;               /**< (7) global array section size 
                                        (usually same as \ref lsize) */
  __INT_T lbase;               /**< (8) index offset section adjustment */
  POINT(__INT_T, gbase);       /**< (9) global offset of first element of 
                                        section (usually 0) */
  POINT(TYPE_DESC, dist_desc); /**<     When set, this is a pointer to the 
                                        object's type descriptor */
  F90_DescDim dim[MAXDIMS];    /**<     F90 dimensions (Note: We append
                                        \ref rank number of F90_DescDim 
                                        structures to an F90_Desc structure) */
};

struct DIST_DescDim {/* DIST dim info */

  __INT_T lab;               /* (1) local array lower bound */
  __INT_T uab;               /* (2) local array upper bound */
  __INT_T loffset;           /* (3) local index offset */
  __INT_T cofstr;            /* (4) cyclic offset stride */
  __INT_T no;                /* (5) negative overlap allowance */
  __INT_T po;                /* (6) positive overlap allowance */
  __INT_T olb;               /* (7) global owned lower bounds */
  __INT_T oub;               /* (8) global owned upper bounds */
  __INT_T clb;               /* (9) template cycle lower bound */
  __INT_T cno;               /* (10) template cycle count */
  __INT_T taxis;             /* (11) corresponding target dim */
  __INT_T tstride;           /* (12) section index template stride */
  __INT_T toffset;           /* (13) section index templat offset */
  __INT_T tlb;               /* (14) template lower bound */
  __INT_T tub;               /* (15) template upper bound */
  __INT_T paxis;             /* (16) correspding processor dim */
  __INT_T block;             /* (17) template block size */
  __INT_T block_shift;       /* (18) block div shift amount */
  __INT_T block_recip;       /* (19) block reciprocal mantissa */
  __INT_T cycle;             /* (20) template cycle period */
  __INT_T cycle_shift;       /* (21) cycle div shift amount */
  __INT_T cycle_recip;       /* (22) cycle reciprocal mantissa */
  __INT_T pshape;            /* (23) extent of processor dim */
  __INT_T pshape_shift;      /* (24) pshape div shift amount */
  __INT_T pshape_recip;      /* (25) 1/pshape mantissa */
  __INT_T pcoord;            /* (26) this processor's coordinate */
  __INT_T pstride;           /* (27) owning processor multiplier */
  __INT_T astride;           /* (28) array index stride onto temp */
  __INT_T aoffset;           /* (29) array index offset onto temp */
  __INT_T cl;                /* (30) cyclic loop lower bound */
  __INT_T cn;                /* (31) cycle loop trip count */
  __INT_T cs;                /* (32) cyclic loop stride */
  __INT_T clof;              /* (33) cyclic loop local index offset*/
  __INT_T clos;              /* (34) cyclic loop index offset str */
  POINT(__INT_T, gen_block); /* (35) gen_block array */
};

struct DIST_Desc {                /* DIST header */
  __INT_T scoff;                 /* (1) scalar subscript offset */
  __INT_T heapb;                 /* (2) global heap block multiplier */
  __INT_T pbase;                 /* (3) owning processor base offset */
  __INT_T mapped;                /* (4) bitmap (by section dim) */
  __INT_T dfmt;                  /* (5) dist format fields by dim */
  __INT_T cached;                /* (6) bitmap-cyclic loop bounds flag*/
  __INT_T single;                /* (7) bitmap 0=norm 1=single aligned*/
  __INT_T replicated;            /* (8) bitmap 0=norm 1=repl */
  __INT_T nonsequence;           /* (9) bitmap by section dim */
  __INT_T info[MAXDIMS];         /* (10) align-target dim information */
  POINT(proc, dist_target);      /* (17) target proc arrangement */
  POINT(F90_Desc, align_target); /* (18) ultimate align-target */
  POINT(F90_Desc, next_alignee); /* (19) next alignee list */
  POINT(F90_Desc, actual_arg);   /* (20) arg-assoc global array */
  DIST_DescDim dim[MAXDIMS];      /*      first DIST dimension desc */
};

/*
 * Macros for defining and  accessing the desciptors and their fields
 */

#define F90_DPTR_LBOUND_G(p) (p##_f90_dim->lbound)
#define F90_DPTR_EXTENT_G(p) (p##_f90_dim->extent)
/* no longer need section stride/section offset */
#define F90_DPTR_SSTRIDE_G(p) 1
#define F90_DPTR_SOFFSET_G(p) 0
#define F90_DPTR_LSTRIDE_G(p) (p##_f90_dim->lstride)
#define F90_DPTR_UBOUND_G(p) (p##_f90_dim->extent + p##_f90_dim->lbound - 1)

#define F90_DPTR_LBOUND_P(p, v) (p##_f90_dim->lbound = v)
#define F90_DPTR_EXTENT_P(p, v) (p##_f90_dim->extent = v)
#define F90_DPTR_SSTRIDE_P(p, v) (p##_f90_dim->sstride = v)
#define F90_DPTR_SOFFSET_P(p, v) (p##_f90_dim->soffset = v)
#define F90_DPTR_LSTRIDE_P(p, v) (p##_f90_dim->lstride = v)

#define F90_DIM_LBOUND_G(p, i) (p->dim[i].lbound)
#define F90_DIM_EXTENT_G(p, i) (p->dim[i].extent)
/* no longer need section stride/section offset */
#define F90_DIM_SSTRIDE_G(p, i) 1
#define F90_DIM_SOFFSET_G(p, i) 0
#define F90_DIM_LSTRIDE_G(p, i) (p->dim[i].lstride)
#define F90_DIM_UBOUND_G(p, i) (p->dim[i].extent + p->dim[i].lbound - 1)

#define F90_DIM_LBOUND_P(p, i, v) (p->dim[i].lbound = v)
#define F90_DIM_EXTENT_P(p, i, v) (p->dim[i].extent = v)
#define F90_DIM_SSTRIDE_P(p, i, v) (p->dim[i].sstride = v)
#define F90_DIM_SOFFSET_P(p, i, v) (p->dim[i].soffset = v)
#define F90_DIM_LSTRIDE_P(p, i, v) (p->dim[i].lstride = v)

#define DPTR_UBOUND_G(p) (p##_f90_dim->extent + p##_f90_dim->lbound - 1)
#define DPTR_UBOUND_P(p, v)                                                    \
  (p##_f90_dim->extent = (v)-p##_f90_dim->lbound + 1);                         \
  (p##_f90_dim->ubound = v)

#define DIM_UBOUND_G(p, i) (p->dim[i].extent + p->dim[i].lbound - 1)
#define DIM_UBOUND_P(p, i, v)                                                  \
  (p->dim[i].extent = (v)-p->dim[i].lbound + 1; p->dim[i].ubound = v;)

#define F90_TAG_G(p) (*(int *)&(p->tag))
#define F90_RANK_G(p) ((p)->rank)
#define F90_KIND_G(p) ((p)->kind)
#define F90_LEN_G(p) ((p)->len)
#define F90_FLAGS_G(p) ((p)->flags)
#define F90_LSIZE_G(p) ((p)->lsize)
#define F90_GSIZE_G(p) ((p)->gsize)
#define F90_LBASE_G(p) ((p)->lbase)
#define F90_GBASE_G(p) ((p)->gbase)
#define F90_DIST_DESC_G(p) ((p)->dist_desc)

#define F90_TAG_P(p, v) ((p)->tag = (v))
#define F90_RANK_P(p, v) ((p)->rank = (v))
#define F90_KIND_P(p, v) ((p)->kind = (v))
/* TODO: p->len should be size_t, but it is not. This needs to be updated. This
 * casting will work as long as we don't have a character type longer then 32
 * bits. We need to cast to the type of p->len to be safe for now. If we update
 * the descriptor to have the len field be size_t, then we won't be backwards
 * compatible with existing object files. */
#define F90_LEN_P(p, v) ((p)->len = ((__INT_T)v)) 
#define F90_FLAGS_P(p, v) ((p)->flags = (v))
#define F90_LSIZE_P(p, v) ((p)->lsize = (v))
#define F90_GSIZE_P(p, v) ((p)->gsize = (v))
#define F90_LBASE_P(p, v) ((p)->lbase = (v))
#define F90_GBASE_P(p, v) ((p)->gbase = (v))
#define F90_DIST_DESC_P(p, v) ((p)->dist_desc = (v))


extern __INT_T f90DummyGenBlock;
extern __INT_T *f90DummyGenBlockPtr;
/*
 * Macros used by Fortran runtime to access the (uninstantiated) dist specific parts
 * of the descriptor.
 */
#define SET_F90_VAR_DIST_DESC_PTR(f, h) /*(f.dist_desc = NULL)*/

#define SET_F90_DIST_DESC_PTR(p, r) /*(p->dist_desc = NULL)*/

#define SIZE_OF_RANK_n_ARRAY_DESC(n)                                           \
  (sizeof(F90_Desc) - (MAXDIMS - n) * sizeof(F90_DescDim))

#define DIST_DPTR_LAB_G(p) F90_DPTR_LBOUND_G(p)
#define DIST_DPTR_UAB_G(p) F90_DPTR_UBOUND_G(p)
#define DIST_DPTR_LOFFSET_G(p) (-(F90_DPTR_LSTRIDE_G(p) * DIST_DPTR_LAB_G(p)))
#define DIST_DPTR_COFSTR_G(p) 0
#define DIST_DPTR_NO_G(p) 0
#define DIST_DPTR_PO_G(p) 0
#define DIST_DPTR_OLB_G(p) F90_DPTR_LBOUND_G(p)
#define DIST_DPTR_OUB_G(p) F90_DPTR_UBOUND_G(p)
#define DIST_DPTR_CLB_G(p) F90_DPTR_LBOUND_G(p)
#define DIST_DPTR_CNO_G(p) 1
#define DIST_DPTR_TAXIS_G(p) 0
#define DIST_DPTR_TSTRIDE_G(p) F90_DPTR_SSTRIDE_G(p)
#define DIST_DPTR_TOFFSET_G(p) F90_DPTR_SOFFSET_G(p)
#define DIST_DPTR_TLB_G(p) F90_DPTR_LBOUND_G(p)
#define DIST_DPTR_TUB_G(p) F90_DPTR_UBOUND_G(p)
#define DIST_DPTR_PAXIS_G(p) 0
#define DIST_DPTR_BLOCK_G(p) 1
#define DIST_DPTR_BLOCK_SHIFT_G(p) 0
#define DIST_DPTR_BLOCK_RECIP_G(p) 0
#define DIST_DPTR_CYCLE_G(p) 1
#define DIST_DPTR_CYCLE_SHIFT_G(p) 0
#define DIST_DPTR_CYCLE_RECIP_G(p) 0
#define DIST_DPTR_PSHAPE_G(p) 1
#define DIST_DPTR_PSHAPE_SHIFT_G(p) 0
#define DIST_DPTR_PSHAPE_RECIP_G(p) 0
#define DIST_DPTR_PCOORD_G(p) 0
#define DIST_DPTR_PSTRIDE_G(p) F90_DPTR_SSTRIDE_G(p)
#define DIST_DPTR_ASTRIDE_G(p) F90_DPTR_SSTRIDE_G(p)
#define DIST_DPTR_AOFFSET_G(p) 0
#define DIST_DPTR_CL_G(p) 0
#define DIST_DPTR_CN_G(p) 1
#define DIST_DPTR_CS_G(p) 0
#define DIST_DPTR_CLOF_G(p) 0
#define DIST_DPTR_CLOS_G(p) 0
#define DIST_DPTR_GEN_BLOCK_G(p) (f90DummyGenBlockPtr)

#define DIST_DPTR_LAB_P(p, v)
#define DIST_DPTR_UAB_P(p, v)
#define DIST_DPTR_LOFFSET_P(p, v)
#define DIST_DPTR_COFSTR_P(p, v)
#define DIST_DPTR_NO_P(p, v)
#define DIST_DPTR_PO_P(p, v)
#define DIST_DPTR_OLB_P(p, v)
#define DIST_DPTR_OUB_P(p, v)
#define DIST_DPTR_CLB_P(p, v)
#define DIST_DPTR_CNO_P(p, v)
#define DIST_DPTR_TAXIS_P(p, v)
#define DIST_DPTR_TSTRIDE_P(p, v)
#define DIST_DPTR_TOFFSET_P(p, v)
#define DIST_DPTR_TLB_P(p, v)
#define DIST_DPTR_TUB_P(p, v)
#define DIST_DPTR_PAXIS_P(p, v)
#define DIST_DPTR_BLOCK_P(p, v)
#define DIST_DPTR_BLOCK_SHIFT_P(p, v)
#define DIST_DPTR_BLOCK_RECIP_P(p, v)
#define DIST_DPTR_CYCLE_P(p, v)
#define DIST_DPTR_CYCLE_SHIFT_P(p, v)
#define DIST_DPTR_CYCLE_RECIP_P(p, v)
#define DIST_DPTR_PSHAPE_P(p, v)
#define DIST_DPTR_PSHAPE_SHIFT_P(p, v)
#define DIST_DPTR_PSHAPE_RECIP_P(p, v)
#define DIST_DPTR_PCOORD_P(p, v)
#define DIST_DPTR_PSTRIDE_P(p, v)
#define DIST_DPTR_ASTRIDE_P(p, v)
#define DIST_DPTR_AOFFSET_P(p, v)
#define DIST_DPTR_CL_P(p, v)
#define DIST_DPTR_CN_P(p, v)
#define DIST_DPTR_CS_P(p, v)
#define DIST_DPTR_CLOF_P(p, v)
#define DIST_DPTR_CLOS_P(p, v)
#define DIST_DPTR_GEN_BLOCK_P(p, v)

#define DIST_DIM_LAB_G(p, i) F90_DIM_LBOUND_G(p, i)
#define DIST_DIM_UAB_G(p, i) F90_DIM_UBOUND_G(p, i)
#define DIST_DIM_LOFFSET_G(p, i) (-(F90_DIM_LSTRIDE_G(p, i)))
#define DIST_DIM_COFSTR_G(p, i) 0
#define DIST_DIM_NO_G(p, i) 0
#define DIST_DIM_PO_G(p, i) 0
#define DIST_DIM_OLB_G(p, i) F90_DIM_LBOUND_G(p, i)
#define DIST_DIM_OUB_G(p, i) F90_DIM_UBOUND_G(p, i)
#define DIST_DIM_CLB_G(p, i) F90_DIM_LBOUND_G(p, i)
#define DIST_DIM_CNO_G(p, i) 1
#define DIST_DIM_TAXIS_G(p, i) 0
#define DIST_DIM_TSTRIDE_G(p, i) F90_DIM_LSTRIDE_G(p, i)
#define DIST_DIM_TOFFSET_G(p, i) F90_DIM_SOFFSET_G(p, i)
#define DIST_DIM_TLB_G(p, i) F90_DIM_LBOUND_G(p, i)
#define DIST_DIM_TUB_G(p, i) F90_DIM_UBOUND_G(p, i)
#define DIST_DIM_PAXIS_G(p, i) 0
#define DIST_DIM_BLOCK_G(p, i) 0
#define DIST_DIM_BLOCK_SHIFT_G(p, i) 0
#define DIST_DIM_BLOCK_RECIP_G(p, i) 0
#define DIST_DIM_CYCLE_G(p, i) 0
#define DIST_DIM_CYCLE_SHIFT_G(p, i) 0
#define DIST_DIM_CYCLE_RECIP_G(p, i) 0
#define DIST_DIM_PSHAPE_G(p, i) 1
#define DIST_DIM_PSHAPE_SHIFT_G(p, i) 0
#define DIST_DIM_PSHAPE_recip_G(p, i) 0
#define DIST_DIM_PCOORD_G(p, i) 0
#define DIST_DIM_PSTRIDE_G(p, i) F90_DIM_SSTRIDE_G(p, i)
#define DIST_DIM_ASTRIDE_G(p, i) F90_DIM_SSTRIDE_G(p, i)
#define DIST_DIM_AOFFSET_G(p, i) 0
#define DIST_DIM_CL_G(p, i) 0
#define DIST_DIM_CN_G(p, i) 1
#define DIST_DIM_CS_G(p, i) 0
#define DIST_DIM_CLOF_G(p, i) 0
#define DIST_DIM_CLOS_G(p, i) 0
#define DIST_DIM_GEN_BLOCK_G(p, i) (f90DummyGenBlockPtr)

#define DIST_DIM_LAB_P(p, i, v)
#define DIST_DIM_UAB_P(p, i, v)
#define DIST_DIM_LOFFSET_P(p, i)
#define DIST_DIM_COFSTR_P(p, i, v)
#define DIST_DIM_NO_P(p, i, v)
#define DIST_DIM_PO_P(p, i, v)
#define DIST_DIM_OLB_P(p, i, v)
#define DIST_DIM_OUB_P(p, i, v)
#define DIST_DIM_CLB_P(p, i, v)
#define DIST_DIM_CNO_P(p, i, v)
#define DIST_DIM_TAXIS_P(p, i, v)
#define DIST_DIM_TSTRIDE_P(p, i, v)
#define DIST_DIM_TOFFSET_P(p, i, v)
#define DIST_DIM_TLB_P(p, i, v)
#define DIST_DIM_TUB_P(p, i, v)
#define DIST_DIM_PAXIS_P(p, i, v)
#define DIST_DIM_BLOCK_P(p, i, v)
#define DIST_DIM_BLOCK_SHIFT_P(p, i, v)
#define DIST_DIM_BLOCK_RECIP_P(p, i, v)
#define DIST_DIM_CYCLE_P(p, i, v)
#define DIST_DIM_CYCLE_SHIFT_P(p, i, v)
#define DIST_DIM_CYCLE_RECIP_P(p, i, v)
#define DIST_DIM_PSHAPE_P(p, i, v)
#define DIST_DIM_PSHAPE_SHIFT_P(p, i, v)
#define DIST_DIM_PSHAPE_recip_P(p, i, v)
#define DIST_DIM_PCOORD_P(p, i, v)
#define DIST_DIM_PSTRIDE_P(p, i, v)
#define DIST_DIM_ASTRIDE_P(p, i, v)
#define DIST_DIM_AOFFSET_P(p, i, v)
#define DIST_DIM_CL_P(p, i, v)
#define DIST_DIM_CN_P(p, i, v)
#define DIST_DIM_CS_P(p, i, v)
#define DIST_DIM_CLOF_P(p, i, v)
#define DIST_DIM_CLOS_P(p, i, v)
#define DIST_DIM_GEN_BLOCK_P(p, i, v)

#define DIST_SCOFF_G(p) 0
#define DIST_HEAPB_G(p) 0
#define DIST_PBASE_G(p) 0
#define DIST_MAPPED_G(p) 0
#define DIST_DFMT_G(p) 0
#define DIST_CACHED_G(p) 0
#define DIST_SINGLE_G(p) 0
#define DIST_REPLICATED_G(p) 0
#define DIST_NONSEQUENCE_G(p) 0
#define DIST_DIST_TARGET_G(p) NULL
#define DIST_ALIGN_TARGET_G(p) (p)
#define DIST_NEXT_ALIGNEE_G(p) (p)
#define DIST_ACTUAL_ARG_G(p) NULL
#define DIST_INFO_G(p, i) 0

#define DIST_SCOFF_P(p, v)
#define DIST_HEAPB_P(p, v)
#define DIST_PBASE_P(p, v)
#define DIST_MAPPED_P(p, v)
#define DIST_DFMT_P(p, v)
#define DIST_CACHED_P(p, v)
#define DIST_SINGLE_P(p, v)
#define DIST_REPLICATED_P(p, v)
#define DIST_NONSEQUENCE_P(p, v)
#define DIST_DIST_TARGET_P(p, v)
#define DIST_ALIGN_TARGET_P(p, v)
#define DIST_NEXT_ALIGNEE_P(p, v)
#define DIST_ACTUAL_ARG_P(p, v)
#define DIST_INFO_P(p, i, v)

#define F90_DIM_NAME(p) p##_f90_dim
#define DIST_DIM_NAME(p) p##_fort_dim
#define DESC_VAR_NM(p) p##_desc_var_
#define DECL_HDR_PTRS(p) F90_Desc *p
#define DECL_HDR_VARS(v)                                                       \
  F90_Desc DESC_VAR_NM(v);                                                     \
  F90_Desc *v = &DESC_VAR_NM(v)
#define DECL_F90_DIM_PTR(p) F90_DescDim *p##_f90_dim
#define DECL_DIST_DIM_PTR(p) DIST_DescDim *p##_fort_dim
#define DECL_DIM_PTRS(p) DECL_F90_DIM_PTR(p)

/* DANGER, DANGER this cause problems if "i" is pre|post incr|decr */
#define SET_DIM_PTRS(n, p, i) n##_f90_dim = &((p)->dim[i]);


#define NONSEQ_OVERLAP (1 << MAXDIMS)
#define NONSEQ_SECTION (1)

/* DFMT_COLLAPSED must be zero */
#define DFMT_COLLAPSED 0
#define DFMT_BLOCK 1
#define DFMT_BLOCK_K 2
#define DFMT_CYCLIC 3
#define DFMT_CYCLIC_K 4
#define DFMT_GEN_BLOCK 5
#define DFMT_INDIRECT 6
#define DFMT__MASK 0xf
#define DFMT__WIDTH 4
#define DFMT(D, DIM) (DIST_DFMT_G(D) >> DFMT__WIDTH * ((DIM)-1) & DFMT__MASK)

/* replication descriptor */

typedef struct repl_t repl_t;
struct repl_t {
  int ncopies;       /* number of identical copies */
  int ndim;          /* number of replicated dims */
  int ngrp;          /* number of replication groups */
  int grpi;          /* my repl. group index */
  int plow;          /* my repl. group lowest proc number */
  int pcnt[MAXDIMS]; /* processor counts */
  int pstr[MAXDIMS]; /* processor strides */
  int gstr[MAXDIMS]; /* replication group index strides */
};

/* overloaded schedule start/free function types */

typedef void (*sked_start_fn)(void *schedule, char *rb, char *sb, F90_Desc *rd,
                              F90_Desc *sd);
typedef void (*sked_free_fn)(void *schedule);

/* communication schedule */

typedef struct sked sked;
struct sked {
  dtype tag;           /* structure type tag == __SKED */
  void *arg;           /* unspecified pointer argument */
  void (*start)(); /* function called by ENTFTN(comm_start) */
  void (*free)();  /* function called by ENTFTN(comm_free) */
};

/* overloaded reduction function types; keep in sync with red.h */

typedef void (*local_reduc_fn)(void *, __INT_T, void *, __INT_T, __LOG_T *,
                               __INT_T, __INT_T *, __INT_T, __INT_T, __INT_T);

typedef void (*local_reduc_back_fn)(void *, __INT_T, void *, __INT_T, __LOG_T *,
                                    __INT_T, __INT_T *, __INT_T, __INT_T,
                                    __INT_T, __LOG_T);

typedef void (*global_reduc_fn)(__INT_T, void *, void *, void *, void *, __INT_T);


#if defined(DEBUG)
/***** The following are internal to the HPF runtime library *****/
/***** and are not defined as compiler interfaces.           *****/
void I8(__fort_print_local)(void *b, F90_Desc *d);
void I8(__fort_print_vector)(char *msg, void *adr, __INT_T str, __INT_T cnt,
                            dtype kind);
void I8(__fort_show_index)(__INT_T rank, __INT_T *index);
void I8(__fort_show_section)(F90_Desc *d);
void I8(__fort_describe)(char *b, F90_Desc *d);
void __fort_print_scalar(void *adr, dtype kind);
void __fort_show_flags(__INT_T flags);
void __fort_show_scalar(void *adr, dtype kind);
#endif /* DEBUG */

/*
 * routines that are inlined for F90
 */


#define __DIST_SET_ALLOCATION(d, dim, no, po)                                   \
  F90_FLAGS_P((d), F90_FLAGS_G((d)) & ~__TEMPLATE);

#define __DIST_SET_DISTRIBUTION(d, dim, lbound, ubound, paxis, block)           \
  {                                                                            \
    DECL_DIM_PTRS(_dd);                                                        \
    SET_DIM_PTRS(_dd, (d), (dim)-1);                                           \
    F90_DPTR_LBOUND_P(_dd, lbound);                                            \
    DPTR_UBOUND_P(_dd, ubound);                                                \
    F90_DPTR_SSTRIDE_P(_dd, 1);                                                \
    F90_DPTR_SOFFSET_P(_dd, 0);                                                \
    F90_DPTR_LSTRIDE_P(_dd, 0);                                                \
  }

#define __DIST_INIT_SECTION(d, r, a)                                            \
  F90_TAG_P((d), __DESC);                                                      \
  F90_RANK_P((d), r);                                                          \
  F90_KIND_P((d), F90_KIND_G(a));                                              \
  F90_LEN_P((d), F90_LEN_G(a));                                                \
  F90_FLAGS_P((d), F90_FLAGS_G(a));                                            \
  F90_GSIZE_P((d), F90_GSIZE_G(a));                                            \
  F90_LSIZE_P((d), F90_LSIZE_G(a));                                            \
  F90_GBASE_P((d), F90_GBASE_G(a));                                            \
  F90_LBASE_P((d), F90_LBASE_G(a));                                            \
  F90_DIST_DESC_P((d), F90_DIST_DESC_G(a)); /* TYPE_DESC pointer */

#define __DIST_INIT_DESCRIPTOR(d, rank, kind, len, flags, target)               \
  F90_TAG_P((d), __DESC);                                                      \
  F90_RANK_P((d), rank);                                                       \
  F90_KIND_P((d), kind);                                                       \
  F90_LEN_P((d), len);                                                         \
  F90_FLAGS_P((d), ((flags) | __TEMPLATE |                                     \
                    __SEQUENTIAL_SECTION & ~(__NOT_COPIED | __OFF_TEMPLATE))); \
  F90_GSIZE_P((d), 0);                                                         \
  F90_LSIZE_P((d), 0);                                                         \
  F90_GBASE_P((d), 0);                                                         \
  F90_DIST_DESC_P((d), 0);                                                      \
  F90_LBASE_P((d), 1);

/** \def __DIST_SET_SECTIONXX
 *  \brief This macro is used for creating an array section.
 *
 * Note: We no longer need section stride/section offset. 
 *
 * This macro is the same as __DIST_SET_SECTIONX,
 * except it collects the strides into the LSTRIDE;
 * SSTRIDE and SOFFSET are just copied (left at one, zero)
 */
#define __DIST_SET_SECTIONXX(d, ddim, a, adim, l, u, s, noreindex, gsize)       \
  {                                                                            \
    DECL_DIM_PTRS(__dd);                                                       \
    DECL_DIM_PTRS(__ad);                                                       \
    __INT_T __extent, __myoffset;                                              \
    SET_DIM_PTRS(__ad, a, adim - 1);                                           \
    SET_DIM_PTRS(__dd, d, ddim - 1);                                           \
    __extent = u - l + s; /* section extent */                                 \
    if (s != 1) {                                                              \
      if (s == -1)                                                             \
        __extent = -__extent;                                                  \
      else                                                                     \
        __extent /= s;                                                         \
    }                                                                          \
    if (__extent < 0)                                                          \
      __extent = 0;                                                            \
    if (noreindex && s == 1) {                                                 \
      F90_DPTR_LBOUND_P(__dd, l);                     /* lower bound */        \
      DPTR_UBOUND_P(__dd, __extent == 0 ? l - 1 : u); /* upper bound */        \
      __myoffset = 0;                                                          \
    } else {                                                                   \
      F90_DPTR_LBOUND_P(__dd, 1);    /* lower bound */                         \
      DPTR_UBOUND_P(__dd, __extent); /* upper bound */                         \
      __myoffset = l - s;                                                      \
    }                                                                          \
    F90_DPTR_SSTRIDE_P(__dd, 1);                                               \
    F90_DPTR_SOFFSET_P(__dd, 0);                                               \
    F90_DPTR_LSTRIDE_P(__dd, F90_DPTR_LSTRIDE_G(__ad) * s);                    \
    F90_LBASE_P(d, F90_LBASE_G(d) + __myoffset * F90_DPTR_LSTRIDE_G(__ad));    \
    if (F90_DPTR_LSTRIDE_G(__dd) != gsize)                                     \
      F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));                \
    gsize *= __extent;                                                         \
  }
/** \def __DIST_SET_SECTIONX
 *  \brief This macro is used for creating an array section.
 *
 * This macro is the same as __DIST_SET_SECTIONXX but without the gsize 
 * argument. 
 */ 
#define __DIST_SET_SECTIONX(d, ddim, a, adim, l, u, s, noreindex)               \
  {                                                                            \
    DECL_DIM_PTRS(__dd);                                                       \
    DECL_DIM_PTRS(__ad);                                                       \
    __INT_T __extent, __myoffset;                                              \
    SET_DIM_PTRS(__ad, a, adim - 1);                                           \
    SET_DIM_PTRS(__dd, d, ddim - 1);                                           \
    __extent = u - l + s; /* section extent */                                 \
    if (s != 1) {                                                              \
      if (s == -1)                                                             \
        __extent = -__extent;                                                  \
      else                                                                     \
        __extent /= s;                                                         \
    }                                                                          \
    if (__extent < 0)                                                          \
      __extent = 0;                                                            \
    if (noreindex && s == 1) {                                                 \
      F90_DPTR_LBOUND_P(__dd, l);                     /* lower bound */        \
      DPTR_UBOUND_P(__dd, __extent == 0 ? l - 1 : u); /* upper bound */        \
      __myoffset = 0;                                                          \
    } else {                                                                   \
      F90_DPTR_LBOUND_P(__dd, 1);    /* lower bound */                         \
      DPTR_UBOUND_P(__dd, __extent); /* upper bound */                         \
      __myoffset = l - s;                                                      \
    }                                                                          \
    F90_DPTR_SSTRIDE_P(__dd, 1);                                               \
    F90_DPTR_SOFFSET_P(__dd, 0);                                               \
    F90_DPTR_LSTRIDE_P(__dd, F90_DPTR_LSTRIDE_G(__ad) * s);                    \
    F90_LBASE_P(d, F90_LBASE_G(d) + __myoffset * F90_DPTR_LSTRIDE_G(__ad));    \
  }

__INT_T
I8(is_nonsequential_section)(F90_Desc *d, __INT_T dim);

void *I8(__fort_create_conforming_mask_array)(const char *what, char *ab,
                                              char *mb, F90_Desc *as,
                                              F90_Desc *ms, F90_Desc *new_ms);

void I8(__fort_reverse_array)(char *db, char *ab, F90_Desc *dd, F90_Desc *ad);

/* mleair - 12/03/1998 __gen_block implementation__
 * Added functions F90_Desc() and __fort_gen_block_bounds
 * for modularity ...
 */

__INT_T *I8(__fort_new_gen_block)(F90_Desc *d, int dim);

void I8(__fort_gen_block_bounds)(F90_Desc *d, int dim, __INT_T *the_olb,
                                __INT_T *the_oub, __INT_T pcoord);

int __fort_gcd(int, int);

int __fort_lcm(int, int);

void I8(__fort_init_descriptor)(F90_Desc *d, __INT_T rank, dtype kind,
                               __INT_T len, __INT_T flags, void *target);

void I8(__fort_set_distribution)(F90_Desc *d, __INT_T dim, __INT_T lbound,
                                __INT_T ubound, __INT_T paxis, __INT_T *block);

/* __gen_block implementation__
 * Added variable arguments to set_alignment so we can pass in gbCopy array
 * from ENTFTN(template)
 */

void I8(__fort_set_alignment)(F90_Desc *d, __INT_T dim, __INT_T lbound,
                             __INT_T ubound, __INT_T taxis, __INT_T tstride,
                             __INT_T toffset, ...);

void I8(__fort_set_allocation)(F90_Desc *d, __INT_T dim, __INT_T no, __INT_T po);

void I8(__fort_use_allocation)(F90_Desc *d, __INT_T dim, __INT_T no, __INT_T po,
                              F90_Desc *a);

typedef enum { __SINGLE, __SCALAR } _set_single_enum;

void I8(__fort_set_single)(F90_Desc *d, F90_Desc *a, __INT_T dim, __INT_T i,
                          _set_single_enum what);

void I8(__fort_finish_descriptor)(F90_Desc *d);

void I8(__fort_init_section)(F90_Desc *d, __INT_T rank, F90_Desc *a);

void I8(__fort_set_section)(F90_Desc *d, __INT_T ddim, F90_Desc *a, __INT_T adim,
                           __INT_T l, __INT_T u, __INT_T s);

void I8(__fort_finish_section)(F90_Desc *d);

int compute_lstride(F90_Desc *d, int dim);

void I8(__fort_copy_descriptor)(F90_Desc *d, F90_Desc *d0);

F90_Desc *I8(__fort_inherit_template)(F90_Desc *d, __INT_T rank,
                                     F90_Desc *target);

proc *__fort_defaultproc(int rank);

proc *__fort_localproc(void);

int __fort_myprocnum(void);

int __fort_is_ioproc(void);

int I8(__fort_owner)(F90_Desc *d, __INT_T *gidx);

void I8(__fort_localize)(F90_Desc *d, __INT_T *idxv, int *cpu, __INT_T *off);

void I8(__fort_describe_replication)(F90_Desc *d, repl_t *r);

int I8(__fort_next_owner)(F90_Desc *d, repl_t *r, int *pc, int owner);

int I8(__fort_islocal)(F90_Desc *d, __INT_T *gidx);

__INT_T
I8(__fort_local_offset)(F90_Desc *d, __INT_T *gidx);

void *I8(__fort_local_address)(void *base, F90_Desc *d, __INT_T *gidx);

void I8(__fort_cycle_bounds)(F90_Desc *d);

__INT_T
I8(__fort_block_bounds)(F90_Desc *d, __INT_T dim, __INT_T ci, 
                       __INT_T *bl, __INT_T *bu);

__INT_T
I8(__fort_cyclic_loop)(F90_Desc *d, __INT_T dim, __INT_T l, __INT_T u, 
                      __INT_T s, __INT_T *cl, __INT_T *cu, __INT_T *cs, 
                      __INT_T *clof, __INT_T *clos);

int I8(__fort_block_loop)(F90_Desc *d, int dim, __INT_T l, __INT_T u, int s,
                         __INT_T ci, __INT_T *bl, __INT_T *bu);

int I8(__fort_stored_alike)(F90_Desc *dd, F90_Desc *sd);

int I8(__fort_conform)(F90_Desc *s, __INT_T *smap, F90_Desc *t, __INT_T *tmap);

int I8(__fort_covers_procs)(F90_Desc *s, F90_Desc *t);

int I8(__fort_aligned)(F90_Desc *t, __INT_T *tmap, F90_Desc *u, __INT_T *umap);

int I8(__fort_aligned_axes)(F90_Desc *t, int tx, F90_Desc *u, int ux);

void __fort_abort(const char *msg);

void __fort_abortp(const char *s);

void __fort_bcopy(char *to, char *fr, size_t n);

void __fort_bcopysl(char *to, char *fr, size_t cnt, size_t tostr, size_t frstr,
                   size_t size);

void I8(__fort_fills)(char *ab, F90_Desc *ad, void *fill);

chdr *I8(__fort_copy)(void *db, void *sb, F90_Desc *dd, F90_Desc *sd, int *smap);

void I8(__fort_copy_out)(void *db, void *sb, F90_Desc *dd, F90_Desc *sd,
                        __INT_T flags);

int __fort_exchange_counts(int *counts);

void I8(__fort_get_scalar)(void *temp, void *b, F90_Desc *d, __INT_T *gidx);

void I8(__fort_reduce_section)(void *vec1, dtype typ1, int siz1, void *vec2,
                               dtype typ2, int siz2, int cnt,
                               global_reduc_fn fn_g, int dim, F90_Desc *d);

void I8(__fort_replicate_result)(void *vec1, dtype typ1, int siz1, void *vec2,
                                dtype typ2, int siz2, int cnt, F90_Desc *d);

sked *I8(__fort_comm_sked)(chdr *ch, char *rb, char *sb, dtype kind, int len);

int I8(__fort_ptr_aligned)(char *p1, dtype kind, int len, char *p2);

char *I8(__fort_ptr_offset)(char **pointer, __POINT_T *offset, char *base,
                           dtype kind, __CLEN_T len, char *area);

char *I8(__fort_alloc)(__INT_T nelem, dtype kind, size_t len, __STAT_T *stat,
                      char **pointer, __POINT_T *offset, char *base, int check,
                      void *(*mallocfn)(size_t));

char *I8(__fort_allocate)(int nelem, dtype kind, size_t len, char *base,
                         char **pointer, __POINT_T *offset);

int I8(__fort_allocated)(char *area);

char *I8(__fort_local_allocate)(int nelem, dtype kind, size_t len, char *base,
                               char **pointer, __POINT_T *offset);

char *I8(__fort_dealloc)(char *area, __STAT_T *stat, void (*freefn)(void *));

void I8(__fort_deallocate)(char *area);

void I8(__fort_local_deallocate)(char *area);

void *__fort_malloc_without_abort(size_t n);
void *__fort_calloc_without_abort(size_t n);

void *__fort_malloc(size_t n);

void *__fort_realloc(void *ptr, size_t n);

void *__fort_calloc(size_t n, size_t s);

void __fort_free(void *ptr);

void *__fort_gmalloc_without_abort(size_t n);
void *__fort_gcalloc_without_abort(size_t n);

void *__fort_gmalloc(size_t n);

void *__fort_grealloc(void *ptr, size_t n);

void *__fort_gcalloc(size_t n, size_t s);

void __fort_gfree(void *ptr);

void *__fort_gsbrk(int n);

/* group */

struct cgrp {
  int ncpus;   /* number of cpus */
  int cpus[1]; /* actually ncpus entries */
};

chdr *__fort_chn_1to1(chdr *fir, int dnd, int dlow, int *dcnt, int *dstr,
                     int snd, int slow, int *scnt, int *sstr);

chdr *__fort_chn_1toN(chdr *fir, int dnd, int dlow, int *dcnt, int *dstr,
                     int snd, int slow, int *scnt, int *sstr);

void __fort_sendl(chdr *c, int indx, void *adr, long cnt, long str, int typ,
                 long ilen);

void __fort_recvl(chdr *c, int indx, void *adr, long cnt, long str, int typ,
                 long ilen);

chdr *__fort_chain_em_up(chdr *list, chdr *c);

void __fort_setbase(chdr *c, char *bases, char *baser, int typ, long ilen);

void __fort_adjbase(chdr *c, char *bases, char *baser, int typ, long ilen);

void __fort_doit(chdr *c);

void __fort_frechn(chdr *c);

void __fort_rsendl(int cpu, void *adr, long cnt, long str, int typ, long ilen);

void __fort_rrecvl(int cpu, void *adr, long cnt, long str, int typ, long ilen);

void __fort_rsend(int cpu, void *adr, long cnt, long str, int typ);

void __fort_rrecv(int cpu, void *adr, long cnt, long str, int typ);

void __fort_rbcstl(int src, void *adr, long cnt, long str, int typ, long ilen);

void __fort_rbcst(int src, void *adr, long cnt, long str, int typ);

void __fort_bcstchn(struct chdr *c,int scpu, int ncpus, int *cpus);

void __fort_exit(int s);

/* tracing functions -- entry.c */

void __fort_tracecall(const char *msg);

/* utility functions -- util.c */

int I8(__fort_varying_int)(void *b, __INT_T *size);

int I8(__fort_varying_log)(void *b, __INT_T *size);

int I8(__fort_fetch_int)(void *b, F90_Desc *d);

void I8(__fort_store_int)(void *b, F90_Desc *d, int val);

int I8(__fort_fetch_log)(void *b, F90_Desc *d);

void I8(__fort_store_log)(void *b, F90_Desc *d, int val);

int I8(__fort_fetch_int_element)(void *b, F90_Desc *d, int i);

void I8(__fort_store_int_element)(void *b, F90_Desc *d, int i, int val);

void I8(__fort_fetch_int_vector)(void *b, F90_Desc *d, int *vec, int veclen);

void I8(__fort_store_int_vector)(void *b, F90_Desc *d, int *vec, int veclen);

void __fort_ftnstrcpy(char *dst,  /*  destination string, blank-filled */
                     int len,    /*  length of destination space */
                     char *src); /*  null terminated source string  */

const char *__fort_getopt(const char *opt);

long __fort_getoptn(const char *opt, long def);

int __fort_atol(char *p);

long __fort_strtol(const char *str, char **ptr, int base);

void __fort_initndx(int nd, int *cnts, int *ncnts, int *strs, int *nstrs,
                    int *mults);

int __fort_findndx(int cpu, int nd, int low, int *nstrs, int *mults);

void __fort_barrier(void);

void __fort_par_unlink(char *fn);

void __fort_zopen(char *path);

void __fort_erecv(int cpu, struct ents *e);
void __fort_esend(int cpu, struct ents *e);
void __fort_rstchn(struct chdr *c);

void ENTFTN(INSTANCE, instance)(F90_Desc *dd, F90_Desc *td, __INT_T *p_kind, __INT_T *p_len, __INT_T *p_collapse, ...);
void ENTFTN(TEMPLATE, template)(F90_Desc *dd, __INT_T *p_rank, __INT_T *p_flags, ...);
void ENTFTN(SECT, sect)(F90_Desc *d, F90_Desc *a, ...);
__LOG_T ENTFTN(ASSOCIATED, associated) (char *pb, F90_Desc *pd, char *tb, F90_Desc *td);

/* FIXME should ENTF90 prototypes live here? */
void ENTF90(TEMPLATE, template)(F90_Desc *dd, __INT_T *p_rank, __INT_T *p_flags, __INT_T *p_kind, __INT_T *p_len, ...);
void ENTF90(TEMPLATE1, template1)(F90_Desc *dd, __INT_T *p_flags, __INT_T *p_kind, __INT_T *p_len, __INT_T *p_l1, __INT_T *p_u1);
void ENTF90(TEMPLATE2, template2)(F90_Desc *dd, __INT_T *p_flags, __INT_T *p_kind, __INT_T *p_len, __INT_T *p_l1, __INT_T *p_u1, __INT_T *p_l2, __INT_T *p_u2);
void ENTF90(TEMPLATE3, template3)(F90_Desc *dd, __INT_T *p_flags, __INT_T *p_kind, __INT_T *p_len, __INT_T *p_l1, __INT_T *p_u1, __INT_T *p_l2, __INT_T *p_u2, __INT_T *p_l3, __INT_T *p_u3);

#endif /*_PGHPF_H_*/
