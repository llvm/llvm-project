// This file tests that -fprofile-sample-use enables location tracking
// generation in the same way that -Rpass does. The sample profiler needs
// to associate line locations in the profile to the code, so it needs the
// frontend to emit source location annotations.

// RUN: %clang_cc1 %s -fprofile-sample-use=%S/Inputs/profile-sample-use-loc-tracking.prof -emit-llvm -o - 2>/dev/null | FileCheck %s

// -fprofile-sample-use should produce source location annotations, exclusively
// (just like -gmlt).
// CHECK: , !dbg !
// CHECK-NOT: DW_TAG_base_type

// But llvm.dbg.cu should be missing (to prevent writing debug info to
// the final output).
// CHECK-NOT: !llvm.dbg.cu = !{

int bar(int j) {
  return (j + j - 2) * (j - 2) * j;
}
