// REQUIRES: linux || windows
// Default
// RUN: %clang -o %t.normal -fprofile-instr-generate -fcoverage-mapping -fuse-ld=lld %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// With -profile-correlate=binary flag
// RUN: %clang -o %t-1.exe -fprofile-instr-generate -fcoverage-mapping -mllvm -profile-correlate=binary -fuse-ld=lld %S/Inputs/instrprof-debug-info-correlate-main.cpp %S/Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t-1.profraw %run %t-1.exe
// RUN: llvm-profdata merge -o %t-1.profdata --binary-file=%t-1.exe %t-1.profraw
// RUN: diff %t.normal.profdata %t-1.profdata

// Strip above binary and run
// RUN: llvm-strip %t-1.exe -o %t-2.exe
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t-2.exe
// RUN: llvm-profdata merge -o %t-2.profdata --binary-file=%t-1.exe %t-2.profraw
// RUN: diff %t.normal.profdata %t-2.profdata

// Online merging.
// RUN: env LLVM_PROFILE_FILE=%t-3.profraw %run %t.normal
// RUN: env LLVM_PROFILE_FILE=%t-4.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.merged.profdata %t-3.profraw %t-4.profraw
// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m-4.profraw %run %t-2.exe
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m-4.profraw %run %t-2.exe
// RUN: llvm-profdata merge -o %t-4.profdata --binary-file=%t-1.exe  %t.profdir
// RUN: diff %t.normal.merged.profdata %t-4.profdata

// TODO: After adding support for binary ID, test binaries with different binary IDs.
