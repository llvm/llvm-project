// Value profiling is currently not supported in lightweight mode.
// RUN: %clang_pgogen -o %t -g -mllvm --profile-correlate=debug-info -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t.dSYM %t.proflite

// RUN: %clang_pgogen -o %t.normal -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// RUN: llvm-profdata show --all-functions --counts %t.normal.profdata > %t.normal.functions
// RUN: llvm-profdata show --all-functions --counts %t.profdata > %t.functions
// RUN: diff %t.normal.functions %t.functions

// RUN: %clang_pgogen -o %t.cov -g -mllvm --profile-correlate=debug-info -mllvm -pgo-function-entry-coverage -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.cov.proflite %run %t.cov
// RUN: llvm-profdata merge -o %t.cov.profdata --debug-info=%t.cov.dSYM %t.cov.proflite

// RUN: %clang_pgogen -o %t.cov.normal -mllvm --pgo-function-entry-coverage -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.cov.profraw %run %t.cov.normal
// RUN: llvm-profdata merge -o %t.cov.normal.profdata %t.cov.profraw

// RUN: llvm-profdata show --all-functions --counts %t.cov.normal.profdata > %t.cov.normal.functions
// RUN: llvm-profdata show --all-functions --counts %t.cov.profdata > %t.cov.functions
// RUN: diff %t.cov.normal.functions %t.cov.functions

// Test debug info correlate with online merging.

// RUN: env LLVM_PROFILE_FILE=%t-1.profraw %run %t.normal
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t-1.profraw %t-2.profraw

// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t.dSYM %t.profdir/

// RUN: llvm-profdata show --all-functions --counts %t.normal.profdata > %t.normal.functions
// RUN: llvm-profdata show --all-functions --counts %t.profdata > %t.functions
// RUN: diff %t.normal.functions %t.functions

// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.cov.proflite %run %t.cov
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.cov.proflite %run %t.cov
// RUN: llvm-profdata merge -o %t.cov.profdata --debug-info=%t.cov.dSYM %t.profdir/

// RUN: llvm-profdata show --all-functions --counts %t.cov.normal.profdata > %t.cov.normal.functions
// RUN: llvm-profdata show --all-functions --counts %t.cov.profdata > %t.cov.functions
// RUN: diff %t.cov.normal.functions %t.cov.functions
