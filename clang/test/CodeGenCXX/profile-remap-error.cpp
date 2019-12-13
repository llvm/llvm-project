// REQUIRES: x86-registered-target

// RUN: not %clang_cc1 -triple x86_64-linux-gnu -fprofile-sample-use=%S/Inputs/profile-remap.samples -fprofile-remapping-file=%S/Inputs/profile-remap-error.map -fexperimental-new-pass-manager -O2 %s -emit-llvm -o - 2>&1 | FileCheck %s

// CHECK:      error: {{.*}}/profile-remap-error.map:1: Could not demangle 'unmangled' as a <name>; invalid mangling?
// CHECK-NEXT: error: {{.*}}/profile-remap-error.map: Could not create remapper: Malformed sample profile data
// CHECK-NEXT: error: {{.*}}/profile-remap.samples: Could not open profile: Malformed sample profile data
