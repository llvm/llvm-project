// Test a profile with only a header is generated when a src file is not in the
//    selected files list provided via -fprofile-list.

// RUN: mkdir -p %t.d
// RUN: echo "src:other.c" > %t-file.list
// RUN: %clang_profgen -fprofile-list=%t-file.list -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show %t.profraw | FileCheck %s --check-prefix=RAW-PROFILE-HEADER-ONLY

// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show %t.profdata | FileCheck %s --check-prefix=INDEXED-PROFILE-HEADER-ONLY

int main() { return 0; }

// RAW-PROFILE-HEADER-ONLY: Instrumentation level: Front-end
// RAW-PROFILE-HEADER-ONLY-NEXT: Total functions: 0
// RAW-PROFILE-HEADER-ONLY-NEXT: Maximum function count: 0
// RAW-PROFILE-HEADER-ONLY-NEXT: Maximum internal block count: 0

// INDEXED-PROFILE-HEADER-ONLY: Instrumentation level: Front-end
// INDEXED-PROFILE-HEADER-ONLY-NEXT: Total functions: 0
// INDEXED-PROFILE-HEADER-ONLY-NEXT: Maximum function count: 0
// INDEXED-PROFILE-HEADER-ONLY-NEXT: Maximum internal block count: 0
