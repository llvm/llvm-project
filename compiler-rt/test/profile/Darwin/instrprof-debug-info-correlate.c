// Value profiling is currently not supported in lightweight mode.
// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t.dSYM %t.proflite

// RUN: %clang_pgogen -o %t.normal -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// RUN: diff %t.normal.profdata %t.profdata

// RUN: %clang_pgogen -o %t.cov -g -mllvm --debug-info-correlate -mllvm -pgo-function-entry-coverage -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.cov.proflite %run %t.cov
// RUN: llvm-profdata merge -o %t.cov.profdata --debug-info=%t.cov.dSYM %t.cov.proflite

// RUN: %clang_pgogen -o %t.cov.normal -mllvm --pgo-function-entry-coverage -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.cov.profraw %run %t.cov.normal
// RUN: llvm-profdata merge -o %t.cov.normal.profdata %t.cov.profraw

// RUN: diff %t.cov.normal.profdata %t.cov.profdata
