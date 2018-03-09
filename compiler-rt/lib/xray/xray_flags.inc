//===-- xray_flags.inc ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// XRay runtime flags.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_FLAG
#error "Define XRAY_FLAG prior to including this file!"
#endif

XRAY_FLAG(bool, patch_premain, false,
          "Whether to patch instrumentation points before main.")
XRAY_FLAG(const char *, xray_logfile_base, "xray-log.",
          "Filename base for the xray logfile.")
XRAY_FLAG(const char *, xray_mode, "", "Mode to install by default.")
XRAY_FLAG(uptr, xray_page_size_override, 0,
          "Override the default page size for the system, in bytes. The size "
          "should be a power-of-two.")

// Basic (Naive) Mode logging options.
XRAY_FLAG(bool, xray_naive_log, false,
          "DEPRECATED: Use xray_mode=xray-basic instead.")
XRAY_FLAG(int, xray_naive_log_func_duration_threshold_us, 5,
          "Naive logging will try to skip functions that execute for fewer "
          "microseconds than this threshold.")
XRAY_FLAG(int, xray_naive_log_max_stack_depth, 64,
          "Naive logging will keep track of at most this deep a call stack, "
          "any more and the recordings will be droppped.")
XRAY_FLAG(int, xray_naive_log_thread_buffer_size, 1024,
          "The number of entries to keep on a per-thread buffer.")

// FDR (Flight Data Recorder) Mode logging options.
XRAY_FLAG(bool, xray_fdr_log, false,
          "DEPRECATED: Use xray_mode=xray-fdr instead.")
XRAY_FLAG(int, xray_fdr_log_func_duration_threshold_us, 5,
          "FDR logging will try to skip functions that execute for fewer "
          "microseconds than this threshold.")
XRAY_FLAG(int, xray_fdr_log_grace_period_us, 0,
          "DEPRECATED: use xray_fdr_log_grace_period_ms instead.")
XRAY_FLAG(int, xray_fdr_log_grace_period_ms, 100,
          "FDR logging will wait this much time in microseconds before "
          "actually flushing the log; this gives a chance for threads to "
          "notice that the log has been finalized and clean up.")
