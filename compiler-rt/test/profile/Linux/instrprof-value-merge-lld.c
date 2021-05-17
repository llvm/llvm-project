// REQUIRES: lld-available
/// Test ld with GC.

// RUN: %clang_pgogen -o %t -O3 %S/Inputs/instrprof-value-merge.c -fuse-ld=lld -ffunction-sections -fdata-sections -Wl,--gc-sections -z start-stop-gc
// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: llvm-profdata show -counts -function=main -ic-targets -memop-sizes %t.profdir/default_*.profraw | FileCheck %S/Inputs/instrprof-value-merge.c
