// RUN: %clang -fprofile-generate -E -dM %s | FileCheck -match-full-lines -check-prefix=PROFGEN %s
// RUN: %clang -fprofile-instr-generate -E -dM %s | FileCheck -match-full-lines -check-prefix=PROFGEN %s
// RUN: %clang -fcs-profile-generate -E -dM %s | FileCheck -match-full-lines -check-prefix=PROFGEN %s
//
// PROFGEN:#define __LLVM_INSTR_PROFILE_GENERATE 1
