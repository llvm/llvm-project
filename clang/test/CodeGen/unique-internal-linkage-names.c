// RUN: %clang_cc1 %s -S -emit-llvm -funique-internal-linkage-names -o - | FileCheck %s

// Check that we do not crash when overloading extern functions.

inline void overloaded_external() {}
extern void overloaded_external();

// CHECK: define internal void @overloaded_internal() [[ATTR:#[0-9]+]] {
static void overloaded_internal() {}
extern void overloaded_internal();

void markUsed() {
  overloaded_external();
  overloaded_internal();
}

// CHECK: attributes [[ATTR]] =
// CHECK-SAME: "sample-profile-suffix-elision-policy"="selected"
