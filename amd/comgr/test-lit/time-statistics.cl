// COM: Check for any runtime errors with the Comgr Profilier
// RUN: AMD_COMGR_TIME_STATISTICS=1 compile-minimal-test %s %t.bin
// RUN: test -f PerfStatsLog.txt

void kernel add(__global float *A, __global float *B, __global float *C) {
    *C = *A + *B;
}
