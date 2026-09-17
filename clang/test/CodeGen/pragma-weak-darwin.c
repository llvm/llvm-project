// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s | FileCheck %s

void strong_target(void) {}

#pragma weak weak_alias = strong_target

void use_alias(void) {
  weak_alias();
}

// CHECK-DAG: @weak_alias = weak alias void (), ptr @strong_target
// CHECK-LABEL: define{{.*}} void @strong_target()
// CHECK-LABEL: define{{.*}} void @use_alias()
// CHECK: call void @weak_alias()
