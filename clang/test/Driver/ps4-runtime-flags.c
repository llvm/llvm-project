// REQUIRES: x86-registered-target
//
// Test the profile runtime library to be linked for PS4 compiler.
// Check PS4 runtime flag --dependent-lib which does not append the default library search path.
//
// RUN: %clang -target x86_64-scei-ps4 --coverage %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fprofile-arcs %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fno-profile-arcs -fprofile-arcs %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fno-profile-arcs %s -### 2>&1 | FileCheck -check-prefix=CHECK-PS4-NO-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fprofile-arcs -fno-profile-arcs %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-NO-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fprofile-generate %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fno-profile-generate %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-NO-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fprofile-generate=dir %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fprofile-instr-generate %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fno-profile-instr-generate %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-NO-PROFILE %s
// RUN: %clang -target x86_64-scei-ps4 -fprofile-instr-generate=somefile.profraw %s -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-PROFILE %s
//
// CHECK-PS4-PROFILE: "--dependent-lib=libclang_rt.profile-x86_64.a"
// CHECK-PS4-NO-PROFILE-NOT: "--dependent-lib=libclang_rt.profile-x86_64.a"
