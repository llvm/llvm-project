// RUN: %clang_cc1 -x c -debug-info-kind=line-tables-only -emit-llvm -fsanitize=returns-nonnull-attribute -o - %s | FileCheck %s
// The UBSAN function call in the epilogue needs to have a debug location.

__attribute__((returns_nonnull)) void *allocate() {}

// CHECK: define {{.*}}nonnull i8* @allocate(){{.*}} !dbg
// CHECK: call void @__ubsan_handle_nonnull_return_v1_abort
// CHECK-SAME:  !dbg ![[LOC:[0-9]+]]
// CHECK: ret i8*
// CHECK-SAME:  !dbg ![[LOC]]
