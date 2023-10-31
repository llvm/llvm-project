// Disable full debug info and verify that we get warnings during merging

// RUN: %clang_pgogen -o %t -gline-tables-only -mllvm --profile-correlate=debug-info -mllvm --disable-vp=true %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t %t.proflite --max-debug-info-correlation-warnings=2 2>&1 >/dev/null | FileCheck %s --check-prefixes=CHECK,LIMIT --implicit-check-not=warning
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t %t.proflite --max-debug-info-correlation-warnings=0 2>&1 >/dev/null | FileCheck %s --check-prefixes=CHECK,NOLIMIT --implicit-check-not=warning

// CHECK: warning: Could not find address of function
// CHECK: warning: Could not find address of function
// NOLIMIT: warning: Could not find address of function
// NOLIMIT: warning: Could not find address of function
// NOLIMIT: warning: Could not find address of function
// LIMIT: warning: Suppressed 3 additional warnings
