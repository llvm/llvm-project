// REQUIRES: linux || windows
// Default
// RUN: %clang -o %t.normal -fprofile-instr-generate -fcoverage-mapping %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw
// RUN: llvm-cov report --instr-profile=%t.normal.profdata %t.normal > %t.normal.report
// RUN: llvm-cov show --instr-profile=%t.normal.profdata %t.normal > %t.normal.show

// With -profile-correlate=binary flag
// RUN: %clang -o %t-1.exe -fprofile-instr-generate -fcoverage-mapping -mllvm -profile-correlate=binary %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t-1.profraw %run %t-1.exe
// RUN: llvm-profdata merge -o %t-1.profdata --binary-file=%t-1.exe %t-1.profraw
// RUN: llvm-cov report --instr-profile=%t-1.profdata %t-1.exe > %t-1.report
// RUN: llvm-cov show --instr-profile=%t-1.profdata %t-1.exe > %t-1.show
// RUN: llvm-profdata show --all-functions --counts %t.normal.profdata > %t.normal.profdata.show
// RUN: llvm-profdata show --all-functions --counts %t-1.profdata > %t-1.profdata.show
// RUN: diff %t.normal.profdata.show %t-1.profdata.show
// RUN: diff %t.normal.report %t-1.report
// RUN: diff %t.normal.show %t-1.show

// Strip above binary and run
// RUN: llvm-strip %t-1.exe -o %t-2.exe
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t-2.exe
// RUN: llvm-profdata merge -o %t-2.profdata --binary-file=%t-1.exe %t-2.profraw
// RUN: llvm-cov report --instr-profile=%t-2.profdata %t-1.exe > %t-2.report
// RUN: llvm-cov show --instr-profile=%t-2.profdata %t-1.exe > %t-2.show
// RUN: llvm-profdata show --all-functions --counts %t-2.profdata > %t-2.profdata.show
// RUN: diff %t.normal.profdata.show %t-2.profdata.show
// RUN: diff %t.normal.report %t-2.report
// RUN: diff %t.normal.show %t-2.show

// Online merging.
// RUN: env LLVM_PROFILE_FILE=%t-3.profraw %run %t.normal
// RUN: env LLVM_PROFILE_FILE=%t-4.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.merged.profdata %t-3.profraw %t-4.profraw
// RUN: llvm-cov report --instr-profile=%t.normal.merged.profdata %t.normal > %t.normal.merged.report
// RUN: llvm-cov show --instr-profile=%t.normal.merged.profdata %t.normal > %t.normal.merged.show

// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m-4.profraw %run %t-2.exe
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m-4.profraw %run %t-2.exe
// RUN: llvm-profdata merge -o %t-4.profdata --binary-file=%t-1.exe  %t.profdir
// RUN: llvm-cov report --instr-profile=%t-4.profdata %t-1.exe > %t-4.report
// RUN: llvm-cov show --instr-profile=%t-4.profdata %t-1.exe > %t-4.show
// RUN: llvm-profdata show --all-functions --counts %t.normal.merged.profdata > %t.normal.merged.profdata.show
// RUN: llvm-profdata show --all-functions --counts %t-4.profdata > %t-4.profdata.show
// RUN: diff %t.normal.merged.profdata.show %t-4.profdata.show
// RUN: diff %t.normal.merged.report %t-4.report
// RUN: diff %t.normal.merged.show %t-4.show

// TODO: After adding support for binary ID, test binaries with different binary IDs.
