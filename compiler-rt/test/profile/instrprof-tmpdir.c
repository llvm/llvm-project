// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cd %t
// RUN: %clang_profgen -o %t/binary %s
//
// Check that a dir separator is appended after %t is subsituted.
// RUN: env TMPDIR="%t" LLVM_PROFILE_FILE="%%traw1.profraw" %run ./binary
// RUN: llvm-profdata show ./raw1.profraw | FileCheck %s -check-prefix TMPDIR
//
// Check that substitution works even if a redundant dir separator is added.
// RUN: env TMPDIR="%t" LLVM_PROFILE_FILE="%%t/raw2.profraw" %run ./binary
// RUN: llvm-profdata show ./raw2.profraw | FileCheck %s -check-prefix TMPDIR
//
// Check that we fall back to the default path if TMPDIR is missing.
// RUN: env -u TMPDIR LLVM_PROFILE_FILE="%%t/raw3.profraw" %run ./binary 2>&1 | FileCheck %s -check-prefix MISSING
// RUN: llvm-profdata show ./default.profraw | FileCheck %s -check-prefix TMPDIR

// TMPDIR: Maximum function count: 1

// MISSING: Unable to get the TMPDIR environment variable, referenced in {{.*}}raw3.profraw. Using the default path.

int main() { return 0; }
