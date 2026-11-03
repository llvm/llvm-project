// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s | FileCheck %s

// Mach-O supports weak aliases: test that the alias is emitted with weak linkage,
// and that calls reference the alias, not the aliasee. Regression test for:
//
// - https://github.com/llvm/llvm-project/issues/71001
// - https://github.com/llvm/llvm-project/issues/111321

// CHECK-DAG: @pragma_weak_alias = weak alias void (), ptr @strong_target
#pragma weak pragma_weak_alias = strong_target

// CHECK-DAG: @attr_weak_alias = weak alias void (), ptr @strong_target
void attr_weak_alias(void) __attribute__((weak, alias("strong_target")));

// CHECK-LABEL: define{{.*}} void @strong_target()
void strong_target(void) {}

// CHECK-LABEL: define{{.*}} void @use_alias()
void use_alias(void) {
  // CHECK: call void @pragma_weak_alias()
  pragma_weak_alias();
}

// CHECK-LABEL: define{{.*}} void @use_attr_alias()
void use_attr_alias(void) {
  // CHECK: call void @attr_weak_alias()
  attr_weak_alias();
}
