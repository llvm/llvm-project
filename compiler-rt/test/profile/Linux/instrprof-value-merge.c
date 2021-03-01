// RUN: %clang_pgogen -o %t -O3 %S/Inputs/instrprof-value-merge.c
// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: llvm-profdata show -counts -function=main -ic-targets -memop-sizes %t.profdir/default_*.profraw | FileCheck %S/Inputs/instrprof-value-merge.c

/// -z start-stop-gc requires binutils 2.37.
// RUN: %clang_pgogen -o %t -O3 %S/Inputs/instrprof-value-merge.c -fuse-ld=bfd -ffunction-sections -fdata-sections -Wl,--gc-sections
// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: llvm-profdata show -counts -function=main -ic-targets -memop-sizes %t.profdir/default_*.profraw | FileCheck %S/Inputs/instrprof-value-merge.c

// RUN: %clang_pgogen -o %t -O3 %S/Inputs/instrprof-value-merge.c -fuse-ld=gold -ffunction-sections -fdata-sections -Wl,--gc-sections
// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t 1
// RUN: llvm-profdata show -counts -function=main -ic-targets -memop-sizes %t.profdir/default_*.profraw | FileCheck %S/Inputs/instrprof-value-merge.c
