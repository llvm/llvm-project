// Value profiling is currently not supported in lightweight mode.
// RUN: %clang_pgogen -o %t.normal -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// RUN: %clang_pgogen -o %t.d4 -g -gdwarf-4 -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.d4.proflite %run %t.d4
// RUN: llvm-profdata merge -o %t.d4.profdata --debug-info=%t.d4 %t.d4.proflite

// RUN: diff %t.normal.profdata %t.d4.profdata

// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t %t.proflite

// RUN: diff %t.normal.profdata %t.profdata
