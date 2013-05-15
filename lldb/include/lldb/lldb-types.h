//===-- lldb-types.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_types_h_
#define LLDB_lldb_types_h_

#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

//----------------------------------------------------------------------
// All host systems must define:
//  lldb::condition_t       The native condition type (or a substitute class) for conditions on the host system.
//  lldb::mutex_t           The native mutex type for mutex objects on the host system.
//  lldb::thread_t          The native thread type for spawned threads on the system
//  lldb::thread_arg_t      The type of the one any only thread creation argument for the host system
//  lldb::thread_result_t   The return type that gets returned when a thread finishes.
//  lldb::thread_func_t     The function prototype used to spawn a thread on the host system.
//  #define LLDB_INVALID_PROCESS_ID ...
//  #define LLDB_INVALID_THREAD_ID ...
//  #define LLDB_INVALID_HOST_THREAD ...
//  #define IS_VALID_LLDB_HOST_THREAD ...
//----------------------------------------------------------------------

// TODO: Add a bunch of ifdefs to determine the host system and what
// things should be defined. Currently MacOSX is being assumed by default
// since that is what lldb was first developed for.

namespace lldb {
        //----------------------------------------------------------------------
        // MacOSX Types
        //----------------------------------------------------------------------
        typedef ::pthread_mutex_t   mutex_t;
        typedef pthread_cond_t      condition_t;
        typedef pthread_t           thread_t;                   // Host thread type
        typedef void *              thread_arg_t;               // Host thread argument type
        typedef void *              thread_result_t;            // Host thread result type
        typedef void *              (*thread_func_t)(void *);   // Host thread function type
        typedef void                (*LogOutputCallback) (const char *, void *baton);
        typedef bool                (*CommandOverrideCallback)(void *baton, const char **argv);
} // namespace lldb

#if defined(__MINGW32__)

const lldb::thread_t lldb_invalid_host_thread_const = { NULL, 0 } ;
#define LLDB_INVALID_HOST_THREAD         (lldb_invalid_host_thread_const)
#define IS_VALID_LLDB_HOST_THREAD(t)     (!(NULL == (t).p && 0 == (t).x))

#else

#define LLDB_INVALID_HOST_THREAD         ((lldb::thread_t)NULL)
#define IS_VALID_LLDB_HOST_THREAD(t)     ((t) != LLDB_INVALID_HOST_THREAD)

#endif

#define LLDB_INVALID_HOST_TIME           { 0, 0 }

namespace lldb 
{
    typedef uint64_t    addr_t;
    typedef uint64_t    user_id_t;
    typedef uint64_t    pid_t;
    typedef uint64_t    tid_t;
    typedef uint64_t    offset_t;
    typedef int32_t     break_id_t;
    typedef int32_t     watch_id_t;
    typedef void *      clang_type_t;
}


#endif  // LLDB_lldb_types_h_
