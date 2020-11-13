/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTTEST_H_INCLUDED
#define ABTTEST_H_INCLUDED

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "abt.h"

/* We always have to use assert in our test suite. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

/** @defgroup TESTUTIL Test utility
 * This group is for test utility routines.
 */

/**
 * @ingroup TESTUTIL
 * @brief   Initialize Argobots and test environment.
 *
 * ATS_init() internally invokes ABT_init(). Therefore, the test code
 * does not need to call ABT_init().
 *
 * Environment variables:
 * - ATS_VERBOSE: numeric value. It sets the level of verbose output.
 *   This is used by ATS_error(). If not set, ATS_VERBOSE is the
 *   same as 0.
 *
 * @param[in] argc         the number of arguments
 * @param[in] argv         argument vector
 * @param[in] num_xstreams the number of xstreams used in a prgoram
 */
void ATS_init(int argc, char **argv, int num_xstreams);

/**
 * @ingroup TESTUTIL
 * @brief   Finailize Argobots and test environment.
 *
 * ATS_finalize() internally invokes ABT_finalize(). Therefore, the test
 * code does not need to call ABT_finalize().
 * If err is not zero, or errors have been catched by ATS_error(), this
 * routine returns EXIT_FAILURE. Otherwise, it returns EXIT_SUCCESS;
 *
 * @param[in] err  user error code
 * @return Status
 * @retval EXIT_SUCCESS on no error
 * @retval EXIT_FAILURE on error
 */
int ATS_finalize(int err);

/**
 * @ingroup TESTUTIL
 * @brief   Print the formatted string according to verbose level.
 *
 * ATS_printf() behaves like printf(), but it prints out the string
 * only when level is equal to or greater than the value of ATS_VERBOSE.
 *
 * @param[in] level  verbose level
 * @param[in] format format string
 */
void ATS_printf(int level, const char *format, ...);

/**
 * @ingroup TESTUTIL
 * @brief   Check the error code.
 *
 * ATS_error() checks the error code and outputs the string of error code
 * if the error code is not ABT_SUCCESS. Currently, if the error code is not
 * ABT_SUCCESS, this routine calles exit() to terminate the test code.
 *
 * @param[in] err   error code
 * @param[in] msg   user message
 * @param[in] file  file name
 * @param[in] line  line number
 */
void ATS_error(int err, const char *msg, const char *file, int line);

typedef enum {
    ATS_ARG_N_ES = 0,   /* # of ESs */
    ATS_ARG_N_ULT = 1,  /* # of ULTs */
    ATS_ARG_N_TASK = 2, /* # of tasklets */
    ATS_ARG_N_ITER = 3  /* # of iterations */
} ATS_arg;

/**
 * @ingroup TESTUTIL
 * @brief   Read the argument vector.
 *
 * \c ATS_read_args reads the argument vector \c argv and save valid
 * arguments internally.  \c ATS_get_arg_val() is used to get the saved
 * argument value.
 *
 * @param[in] argc  the number of arguments
 * @param[in] argv  argument vector
 */
void ATS_read_args(int argc, char **argv);

/**
 * @ingroup TESTUTIL
 * @brief   Get the argument value.
 *
 * \c ATS_get_arg_val returns the argument value corresponding to \c arg.
 *
 * @param[in] arg  argument kind
 * @return Argument value
 */
int ATS_get_arg_val(ATS_arg arg);

/**
 * @ingroup TESTUTIL
 * @brief   Print a line.
 *
 * \c ATS_print_line prints out a line, which consists of \c len characters
 * of \c c, to a file pointer \c fp.
 *
 * @param[in] fp   file pointer
 * @param[in] c    character as a line element
 * @param[in] len  length of line
 */
void ATS_print_line(FILE *fp, char c, int len);

#define ATS_ERROR_IF(cond) ATS_error_if(cond, #cond, __FILE__, __LINE__)
#define ATS_ERROR(e, m) ATS_error(e, m, __FILE__, __LINE__)
#define ATS_UNUSED(a) (void)(a)

#if defined(__x86_64__)
static inline uint64_t ATS_get_cycles()
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    uint64_t cycle = ((uint64_t)lo) | (((int64_t)hi) << 32);
    return cycle;
}
#elif defined(__aarch64__)
static inline uint64_t ATS_get_cycles()
{
    register uint64_t cycle;
    __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(cycle));
    return cycle;
}
#else
static inline uint64_t ATS_get_cycles()
{
    /* Return a nanosecond as the best effort. */
    return (uint64_t)(ABT_get_wtime() * 1.0e9);
}
#endif

#if defined(__PGIC__) || defined(__ibmxl__)

/* Those two compilers have implementation issues in __atomic built-ins. */
static inline int ATS_atomic_load(volatile int *p_val)
{
    __sync_synchronize();
    int val = *p_val;
    __sync_synchronize();
    return val;
}
static inline void ATS_atomic_store(volatile int *p_val, int val)
{
    __sync_synchronize();
    *p_val = val;
    __sync_synchronize();
}
static inline int ATS_atomic_fetch_add(volatile int *p_val, int val)
{
    return __sync_fetch_and_add(p_val, val);
}

#else

static inline int ATS_atomic_load(volatile int *p_val)
{
    return __atomic_load_n(p_val, __ATOMIC_ACQUIRE);
}
static inline void ATS_atomic_store(volatile int *p_val, int val)
{
    __atomic_store_n(p_val, val, __ATOMIC_RELEASE);
}
static inline int ATS_atomic_fetch_add(volatile int *p_val, int val)
{
    return __atomic_fetch_add(p_val, val, __ATOMIC_ACQ_REL);
}

#endif

#endif /* ABTTEST_H_INCLUDED */
