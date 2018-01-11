/* cilkview.h                  -*-C++-*-
 *
 *************************************************************************
 *
 * Copyright (C) 2010-2016, Intel Corporation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * *********************************************************************
 * 
 * PLEASE NOTE: This file is a downstream copy of a file mainitained in
 * a repository at cilkplus.org. Changes made to this file that are not
 * submitted through the contribution process detailed at
 * http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 * time that a new version is released. Changes only submitted to the
 * GNU compiler collection or posted to the git repository at
 * https://bitbucket.org/intelcilkruntime/itnel-cilk-runtime.git are
 * not tracked.
 * 
 * We welcome your contributions to this open source project. Thank you
 * for your assistance in helping us improve Cilk Plus.
 *
 **************************************************************************/

#ifndef INCLUDED_CILKVIEW_H
#define INCLUDED_CILKVIEW_H

#include <cilk/cilk_api.h>

#ifdef _WIN32
#   ifndef _WINBASE_
__CILKRTS_BEGIN_EXTERN_C
unsigned long __stdcall GetTickCount();
__CILKRTS_END_EXTERN_C
#   endif
#endif  // _WIN32

#if defined __unix__ || defined __APPLE__ || defined __VXWORKS__
#   include <sys/time.h>
#endif  // defined __unix__ || defined __APPLE__

/// @brief Return the system clock with millisecond resolution
///
/// This function returns a long integer representing the number of
/// milliseconds since an arbitrary starting point, e.g., since the system was
/// started or since the Unix Epoch.  The result is meaningless by itself, but
/// the difference between two sequential calls to __cilkview_getticks()
/// represents the time interval that elapsed between them (in ms).
static inline unsigned long long __cilkview_getticks()
{
#if __INTEL_COMPILER > 1200
    // When inlined, prevent code motion around this call
    __notify_zc_intrinsic((void*) "test_getticks_start", 0);
#endif

#ifdef _WIN32
    // Return milliseconds elapsed since the system started
    return GetTickCount();
#elif defined(__unix__) || defined(__APPLE__) || defined __VXWORKS__
    // Return milliseconds elapsed since the Unix Epoch
    // (1-Jan-1970 00:00:00.000 UTC)
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec * 1000ULL + t.tv_usec / 1000;
#else
#   error test_getticks() not implemented for this OS
#endif

#if __INTEL_COMPILER > 1200
    // When inlined, prevent code motion around this call
    __notify_zc_intrinsic((void*) "test_getticks_end", 0);
#endif
}

typedef struct
{
    unsigned int        size;           // Size of structure in bytes
    unsigned int        status;         // 1 = success, 0 = failure
    unsigned long long  time;           // Time in milliseconds
    unsigned long long  work;
    unsigned long long  span;
    unsigned long long  burdened_span;
    unsigned long long  spawns;
    unsigned long long  syncs;
    unsigned long long  strands;
    unsigned long long  atomic_ins;
    unsigned long long  frames;
} cilkview_data_t;

typedef struct
{
    cilkview_data_t *start;     // Values at start of interval
    cilkview_data_t *end;       // Values at end of interval
    const char *label;          // Name for this interval
    unsigned int flags;         // What to do - see flags below
} cilkview_report_t;

// What __cilkview_report should do.  The flags can be ORed together
enum
{
    CV_REPORT_WRITE_TO_LOG = 1,     // Write parallelism report to the log (xml or text)
    CV_REPORT_WRITE_TO_RESULTS = 2  // Write parallelism data to results file
};

#ifndef CILKVIEW_NO_REPORT
static void __cilkview_do_report(cilkview_data_t *start,
                          cilkview_data_t *end,
                          const char *label,
                          unsigned int flags);
#endif /* CILKVIEW_NO_REPORT */

/*
 * Metacall data
 *
 * A metacall is a way to pass data to a function implemented by a tool.
 * Metacalls are always instrumented when the tool is loaded.
 */

// Tool code for Cilkview
#define METACALL_TOOL_CILKVIEW 2

// Metacall codes implemented by Cilkview
enum
{
    CV_METACALL_PUTS,
    CV_METACALL_QUERY,
    CV_METACALL_START,
    CV_METACALL_STOP,
    CV_METACALL_RESET,
    CV_METACALL_USE_DEFAULT_GRAIN,
    CV_METACALL_CONNECTED,
    CV_METACALL_SUSPEND,
    CV_METACALL_RESUME,
    CV_METACALL_REPORT
};

#if ! defined(CILK_STUB) && defined(__INTEL_COMPILER)
#  define __cilkview_metacall(code,data) \
    __cilkrts_metacall(METACALL_TOOL_CILKVIEW, code, data)
#else
#  define __cilkview_metacall(annotation,expr) (annotation, (void) (expr))
#endif

// Write arbitrary string to the log
#define __cilkview_puts(arg) \
    __cilkview_metacall(CV_METACALL_PUTS, arg)

// Retrieve the Cilkview performance counters.  The parameter must be a 
// cilkview_data_t
#define __cilkview_query(d)                             \
    do {                                                \
        d.size = sizeof(d);                             \
        d.status = 0;                                   \
        __cilkview_metacall(CV_METACALL_QUERY, &d);     \
        if (0 == d.status)                              \
            d.time = __cilkview_getticks();             \
    } while (0)

// Write report to log or results file. If end is NULL, Cilkview will
// use the current values.
#define __cilkview_report(start, end, label, flags) \
    __cilkview_do_report(start, end, label, flags)

// Control the workspan performance counters for the final report
#define __cilkview_workspan_start() \
    __cilkview_metacall(CV_METACALL_START, 0)
#define __cilkview_workspan_stop() \
    __cilkview_metacall(CV_METACALL_STOP, 0)
#define __cilkview_workspan_reset() \
    __cilkview_metacall(CV_METACALL_RESET, 0)
#define __cilkview_workspan_suspend() \
    __cilkview_metacall(CV_METACALL_SUSPEND, 0)
#define __cilkview_workspan_resume() \
    __cilkview_metacall(CV_METACALL_RESUME, 0)

#define __cilkview_use_default_grain_size() \
    __cilkview_metacall(CV_METACALL_USE_DEFAULT, 0)

// Sets the int is_connected to 1 if Cilkview is active
#define __cilkview_connected(is_connected) \
    __cilkview_metacall(CV_METACALL_CONNECTED, &is_connected)


#ifndef CILKVIEW_NO_REPORT

// Stop Microsoft include files from complaining about getenv and fopen
#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable: 1786)	// Suppress warnings that getenv, fopen are deprecated
#endif

static void __cilkview_do_report(cilkview_data_t *start,
                                 cilkview_data_t *end,
                                 const char *label,
                                 unsigned int flags)
{
    int under_cilkview = 0;
    unsigned long long elapsed_ms;
    int worker_count = 0;
    char *nworkers;
    char *outfile;
    FILE *f;

    // Check whether we're running under Cilkview
    __cilkview_connected(under_cilkview);

    // If we're running under Cilkview, let it do those things that need
    // to be done
    if (under_cilkview)
    {
        cilkview_report_t d = {start, end, label, flags};
        __cilkview_metacall(CV_METACALL_REPORT, &d);
        return;
    }

    // We're not running under Cilkview.
    //
    // If we weren't asked to write to the results file, we're done.
    if (0 == (flags & CV_REPORT_WRITE_TO_RESULTS))
        return;

    // Calculate the elapse milliseconds
    if (NULL == end)
        elapsed_ms = __cilkview_getticks() - start->time;
    else
        elapsed_ms = end->time - start->time;

    // Determine how many workers we're using for this trial run
    nworkers = getenv("CILK_NWORKERS");
    if (NULL != nworkers)
        worker_count = atoi(nworkers);
    if (0 == worker_count)
        worker_count = 16;

    // Open the output file and write the trial data to it
    outfile = getenv("CILKVIEW_OUTFILE");
    if (NULL == outfile)
        outfile = (char *)"cilkview.out";

    f = fopen(outfile, "a");
    if (NULL == f)
        fprintf(stderr, "__cilkview_do_report: unable to append to file %s\n", outfile);
    else
    {
        fprintf(f, "%s trial %d %f\n", label,
                worker_count,
                ((float)elapsed_ms) / 1000.0f);
        fclose(f);
    }
}
#ifdef _WIN32
#pragma warning(pop)
#endif

#endif  // CILKVIEW_NO_REPORT


#endif /* ! defined(INCLUDED_CILKVIEW_H) */
