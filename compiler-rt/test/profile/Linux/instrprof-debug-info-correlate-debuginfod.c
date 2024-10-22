// REQUIRES: curl
// RUN: rm -rf %t

// Default instrumented build with no profile correlation.
// RUN: %clang_pgogen -o %t.default -Wl,--build-id=0x12345678 -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.default
// RUN: llvm-profdata merge -o %t.default.profdata %t.profraw

// Build with profile debuginfo correlation.
// RUN: %clang_pgogen -o %t.correlate.exe -Wl,--build-id=0x12345678 -g -gdwarf-4 -mllvm --debug-info-correlate -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.debug-info-correlate.proflite %run %t.correlate.exe

// Test llvm-profdata merge profile correlation with --debuginfod option.
// RUN: mkdir -p %t/buildid/12345678
// RUN: cp %t.correlate.exe %t/buildid/12345678/debuginfo
// RUN: mkdir -p %t/debuginfod-cache
// RUN: env DEBUGINFOD_CACHE_PATH=%t/debuginfod-cache DEBUGINFOD_URLS=file://%t llvm-profdata merge -o %t.correlate-debuginfod.profdata --debuginfod --correlate=debug-info %t.debug-info-correlate.proflite
// RUN: diff <(llvm-profdata show --all-functions --counts %t.default.profdata) <(llvm-profdata show --all-functions --counts %t.correlate-debuginfod.profdata)

// Test llvm-profdata merge profile correlation with --debug-file-directory option.
// RUN: mkdir -p %t/.build-id/12
// RUN: cp %t.correlate.exe %t/.build-id/12/345678.debug
// RUN: llvm-profdata merge -o %t.correlate-debug-file-dir.profdata --debug-file-directory %t --correlate=debug-info %t.debug-info-correlate.proflite
// RUN: diff <(llvm-profdata show --all-functions --counts %t.default.profdata) <(llvm-profdata show --all-functions --counts %t.correlate-debug-file-dir.profdata)
