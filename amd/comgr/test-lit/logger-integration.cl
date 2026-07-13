// COM: Exercise AMD_COMGR_REDIRECT_LOGS and AMD_COMGR_LOG_LEVEL end-to-end.
// COM: Redirection must COPY logs to the named file without moving them away
// COM: from the caller's returned comgr.log, and AMD_COMGR_LOG_LEVEL controls
// COM: verbosity in both. The debug header ("amd_comgr_do_action:") appears
// COM: only at level 4; the #warning below is a compiler diagnostic returned
// COM: to the caller and copied to the redirect sink at any level.

// COM: Debug level (4): the debug header must appear in BOTH the redirect file
// COM: and the caller's returned log (copy, not move).
// RUN: AMD_COMGR_LOG_LEVEL=4 AMD_COMGR_REDIRECT_LOGS=%t.debug.log \
// RUN:   logger-redirect %s 1.2 %t.debug.returned.log
// RUN: FileCheck %s < %t.debug.log
// RUN: FileCheck %s < %t.debug.returned.log
// CHECK: amd_comgr_do_action:

// COM: Non-debug levels info (3), warning (2), and error (1): the debug header
// COM: is suppressed from both destinations, but the compiler diagnostic still
// COM: reaches both (proving the sink is live below level 4, not merely empty).
// RUN: AMD_COMGR_LOG_LEVEL=3 AMD_COMGR_REDIRECT_LOGS=%t.info.log \
// RUN:   logger-redirect %s 1.2 %t.info.returned.log
// RUN: FileCheck --check-prefix=SUPPRESSED --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.info.log
// RUN: FileCheck --check-prefix=SUPPRESSED --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.info.returned.log
// RUN: AMD_COMGR_LOG_LEVEL=2 AMD_COMGR_REDIRECT_LOGS=%t.warning.log \
// RUN:   logger-redirect %s 1.2 %t.warning.returned.log
// RUN: FileCheck --check-prefix=SUPPRESSED --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.warning.log
// RUN: FileCheck --check-prefix=SUPPRESSED --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.warning.returned.log
// RUN: AMD_COMGR_LOG_LEVEL=1 AMD_COMGR_REDIRECT_LOGS=%t.error.log \
// RUN:   logger-redirect %s 1.2 %t.error.returned.log
// RUN: FileCheck --check-prefix=SUPPRESSED --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.error.log
// RUN: FileCheck --check-prefix=SUPPRESSED --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.error.returned.log
// SUPPRESSED: comgr-logger-integration-marker

// COM: Redirect open failure: when AMD_COMGR_REDIRECT_LOGS names an unopenable
// COM: destination, the diagnostic must reach the caller's returned comgr.log
// COM: unconditionally, even at level 0 where ordinary logs are suppressed.
// RUN: rm -rf %t.nodir
// RUN: AMD_COMGR_LOG_LEVEL=0 AMD_COMGR_REDIRECT_LOGS=%t.nodir/redirect.log \
// RUN:   logger-redirect %s 1.2 %t.openfail.returned.log
// RUN: FileCheck --check-prefix=OPENFAIL %s < %t.openfail.returned.log
// OPENFAIL: unable to redirect log to file

#warning comgr-logger-integration-marker
void kernel add(__global float *A, __global float *B, __global float *C) {
    *C = *A + *B;
}
