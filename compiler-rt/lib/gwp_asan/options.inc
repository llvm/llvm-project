//===-- options.inc ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_OPTION
#error "Define GWP_ASAN_OPTION prior to including this file!"
#endif

GWP_ASAN_OPTION(bool, Enabled, true, "Is GWP-ASan enabled? Defaults to true.")

GWP_ASAN_OPTION(
    bool, PerfectlyRightAlign, false,
    "When allocations are right-aligned, should we perfectly align them up to "
    "the page boundary? By default (false), we round up allocation size to the "
    "nearest power of two (1, 2, 4, 8, 16) up to a maximum of 16-byte "
    "alignment for performance reasons. For Bionic, we use 8-byte alignment by "
    "default. Setting this to true can find single byte buffer-overflows for "
    "multibyte allocations at the cost of performance, and may be incompatible "
    "with some architectures.")

GWP_ASAN_OPTION(int, MaxSimultaneousAllocations, 16,
                "Number of simultaneously-guarded allocations available in the "
                "pool. Defaults to 16.")

GWP_ASAN_OPTION(int, SampleRate, 5000,
                "The probability (1 / SampleRate) that an allocation is "
                "selected for GWP-ASan sampling. Default is 5000. Sample rates "
                "up to (2^30 - 1) are supported.")

// Developer note - This option is not actually processed by GWP-ASan itself. It
// is included here so that a user can specify whether they want signal handlers
// or not. The supporting allocator should inspect this value to see whether
// signal handlers need to be installed, and then use
// crash_handler::installSignalHandlers() in order to install the handlers. Note
// that in order to support signal handlers, you will need to link against the
// optional crash_handler component.
GWP_ASAN_OPTION(
    bool, InstallSignalHandlers, true,
    "Install GWP-ASan signal handlers for SIGSEGV during dynamic loading. This "
    "allows better error reports by providing stack traces for allocation and "
    "deallocation when reporting a memory error. GWP-ASan's signal handler "
    "will forward the signal to any previously-installed handler, and user "
    "programs that install further signal handlers should make sure they do "
    "the same. Note, if the previously installed SIGSEGV handler is SIG_IGN, "
    "we terminate the process after dumping the error report.")

GWP_ASAN_OPTION(bool, InstallForkHandlers, true,
                "Install GWP-ASan atfork handlers to acquire internal locks "
                "before fork and release them after.")
