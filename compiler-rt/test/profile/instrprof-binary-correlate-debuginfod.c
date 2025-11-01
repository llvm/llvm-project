// REQUIRES: linux || windows
// REQUIRES: curl

// Default instrumented build with no profile correlation.
// RUN: %clang_profgen -o %t.default.exe -Wl,--build-id=0x12345678 -fprofile-instr-generate -fcoverage-mapping %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.default.profraw %run %t.default.exe
// RUN: llvm-profdata merge -o %t.default.profdata %t.default.profraw
// RUN: llvm-profdata show --all-functions --counts %t.default.profdata > %t.default.profdata.show

// Build with profile binary correlation.
// RUN: %clang_profgen -o %t.correlate.exe -Wl,--build-id=0x12345678 -fprofile-instr-generate -fcoverage-mapping -mllvm -profile-correlate=binary %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// Strip above binary and run
// RUN: llvm-strip %t.correlate.exe -o %t.stripped.exe
// RUN: env LLVM_PROFILE_FILE=%t.correlate.profraw %run %t.stripped.exe

// Test llvm-profdata merge profile correlation with --debuginfod option.
// RUN: mkdir -p %t/buildid/12345678
// RUN: cp %t.correlate.exe %t/buildid/12345678/debuginfo
// RUN: mkdir -p %t/debuginfod-cache
// RUN: env DEBUGINFOD_CACHE_PATH=%t/debuginfod-cache DEBUGINFOD_URLS=file://%t llvm-profdata merge -o %t.correlate-debuginfod.profdata --debuginfod --correlate=binary %t.correlate.profraw
// RUN: llvm-profdata show --all-functions --counts %t.correlate-debuginfod.profdata > %t.correlate-debuginfod.profdata.show
// RUN: diff %t.default.profdata.show %t.correlate-debuginfod.profdata.show

// Test llvm-profdata merge profile correlation with --debug-file-directory option.
// RUN: mkdir -p %t/.build-id/12
// RUN: cp %t.correlate.exe %t/.build-id/12/345678.debug
// RUN: llvm-profdata merge -o %t.correlate-debug-file-dir.profdata --debug-file-directory %t --correlate=binary %t.correlate.profraw
// RUN: llvm-profdata show --all-functions --counts %t.correlate-debug-file-dir.profdata > %t.correlate-debug-file-dir.profdata.show
// RUN: diff %t.default.profdata.show %t.correlate-debug-file-dir.profdata.show

// Test error for llvm-profdata merge profile correlation with only --correlate=binary option.
// RUN: not llvm-profdata merge -o %t.correlate-error.profdata --correlate=binary %t.correlate.profraw 2>&1 | FileCheck %s --check-prefix=MISSING-FLAG
// MISSING-FLAG: error: Expected --debuginfod or --debug-file-directory when --correlate is provided

// Test error for llvm-profdata merge profile correlation when a proper --correlate option is not provided.
// RUN: not llvm-profdata merge -o %t.correlate-error.profdata --debug-file-directory %t --correlate="" %t.correlate.profraw 2>&1 | FileCheck %s --check-prefix=MISSING-CORRELATION-KIND
// MISSING-CORRELATION-KIND: error: Expected --correlate when --debug-file-directory is provided

// Test error for llvm-profdata merge profile correlation with mixing correlation options.
// RUN: not llvm-profdata merge -o %t.error.profdata --binary-file=%t.correlate.exe --debug-file-directory %t --correlate=binary %t.correlate.profraw  2>&1 | FileCheck %s --check-prefix=MIXING-FLAGS
// MIXING-FLAGS: error: Expected only one of -binary-file, -debuginfod or -debug-file-directory
