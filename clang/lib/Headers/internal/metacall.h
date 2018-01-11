// -*- C++ -*-

/*
 *  Copyright (C) 2009-2016, Intel Corporation
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *    * Neither the name of Intel Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 *  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 *  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *  
 *  *********************************************************************
 *  
 *  PLEASE NOTE: This file is a downstream copy of a file mainitained in
 *  a repository at cilkplus.org. Changes made to this file that are not
 *  submitted through the contribution process detailed at
 *  http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 *  time that a new version is released. Changes only submitted to the
 *  GNU compiler collection or posted to the git repository at
 *  https://bitbucket.org/intelcilkruntime/itnel-cilk-runtime.git are
 *  not tracked.
 *  
 *  We welcome your contributions to this open source project. Thank you
 *  for your assistance in helping us improve Cilk Plus.
 *
 ******************************************************************************
 *
 * metacall.h
 *
 * This is an internal header file defining part of the metacall
 * interface used by Cilkscreen.  It is not a stable API and is
 * subject to change without notice.
 */

// Provides the enum of metacall kinds.  This is used by Cilkscreen and the
// runtime, and will probably be used by any future ptools.

#pragma once

///////////////////////////////////////////////////////////////////////////////

enum
{
    // Notify Cilkscreen to stop/start instrumenting code
    HYPER_DISABLE_INSTRUMENTATION = 0,
    HYPER_ENABLE_INSTRUMENTATION = 1,

    // Write 0 in *(char *)arg if the p-tool is sequential.  The Cilk runtime
    // system invokes this metacall to know whether to spawn worker threads.
    HYPER_ZERO_IF_SEQUENTIAL_PTOOL = 2,

    // Write 0 in *(char *)arg if the runtime must force reducers to
    // call the reduce() method even if no actual stealing occurs.
    HYPER_ZERO_IF_FORCE_REDUCE = 3,

    // Inform cilkscreen about the current stack pointer.
    HYPER_ESTABLISH_C_STACK = 4,

    // Inform Cilkscreen about the current worker
    HYPER_ESTABLISH_WORKER = 5,

    // Tell tools to ignore a block of memory.  Parameter is a 2 element
    // array: void *block[2] = {_begin, _end};  _end is 1 beyond the end
    // of the block to be ignored.  Essentially, if p is a pointer to an
    // array, _begin = &p[0], _end = &p[max]
    HYPER_IGNORE_MEMORY_BLOCK = 6

    // If you add metacalls here, remember to update BOTH workspan.cpp AND
    // cilkscreen-common.cpp!
};

typedef struct
{
    unsigned int tool;  // Specifies tool metacall is for
                        // (eg. system=0, cilkscreen=1, cilkview=2).
                        // All tools should understand system codes.
                        // Tools should ignore all other codes, except
                        // their own.

    unsigned int code;  // Tool-specific code specifies what to do and how to
                        // interpret data

    void        *data;
} metacall_data_t;

#define METACALL_TOOL_SYSTEM 0

///////////////////////////////////////////////////////////////////////////////
