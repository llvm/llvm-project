// Test memory tagging extension intrinsics
// RUN: %clang_cc1 -triple aarch64-none-linux-eabi -target-feature +mte -O3 -S -emit-llvm -o - %s  | FileCheck %s
#include <stddef.h>
#include <arm_acle.h>

// CHECK-LABEL: define{{.*}} ptr @create_tag1
int *create_tag1(int *a, unsigned b) {
// CHECK: [[T1:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[T2:%[0-9]+]] = tail call ptr @llvm.aarch64.irg(ptr %a, i64 [[T1]])
        return __arm_mte_create_random_tag(a,b);
}

// CHECK-LABEL: define{{.*}} ptr @create_tag2
short *create_tag2(short *a, unsigned b) {
// CHECK: [[T1:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[T2:%[0-9]+]] = tail call ptr @llvm.aarch64.irg(ptr %a, i64 [[T1]])
        return __arm_mte_create_random_tag(a,b);
}

// CHECK-LABEL: define{{.*}} ptr @create_tag3
char *create_tag3(char *a, unsigned b) {
// CHECK: [[T1:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[T2:%[0-9]+]] = tail call ptr @llvm.aarch64.irg(ptr %a, i64 [[T1]])
// CHECK: ret ptr [[T2:%[0-9]+]]
        return __arm_mte_create_random_tag(a,b);
}

// CHECK-LABEL: define{{.*}} ptr @increment_tag1
char *increment_tag1(char *a) {
// CHECK: call ptr @llvm.aarch64.addg(ptr %a, i64 3)
        return __arm_mte_increment_tag(a,3);
}

// CHECK-LABEL: define{{.*}} ptr @increment_tag2
short *increment_tag2(short *a) {
// CHECK: [[T1:%[0-9]+]] = tail call ptr @llvm.aarch64.addg(ptr %a, i64 3)
        return __arm_mte_increment_tag(a,3);
}

// CHECK-LABEL: define{{.*}} i32 @exclude_tag
unsigned exclude_tag(int *a, unsigned m) {
// CHECK: [[T0:%[0-9]+]] = zext i32 %m to i64
// CHECK: [[T2:%[0-9]+]] = tail call i64 @llvm.aarch64.gmi(ptr %a, i64 [[T0]])
// CHECK: trunc i64 [[T2]] to i32
  return __arm_mte_exclude_tag(a, m);
}

// CHECK-LABEL: define{{.*}} ptr @get_tag1
int *get_tag1(int *a) {
// CHECK: [[T1:%[0-9]+]] = tail call ptr @llvm.aarch64.ldg(ptr %a, ptr %a)
   return __arm_mte_get_tag(a);
}

// CHECK-LABEL: define{{.*}} ptr @get_tag2
short *get_tag2(short *a) {
// CHECK: [[T1:%[0-9]+]] = tail call ptr @llvm.aarch64.ldg(ptr %a, ptr %a)
   return __arm_mte_get_tag(a);
}

// CHECK-LABEL: define{{.*}} void @set_tag1
void set_tag1(int *a) {
// CHECK: tail call void @llvm.aarch64.stg(ptr %a, ptr %a)
   __arm_mte_set_tag(a);
}

// CHECK-LABEL: define{{.*}} i64 @subtract_pointers
ptrdiff_t subtract_pointers(int *a, int *b) {
// CHECK: [[T2:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(ptr %a, ptr %b)
// CHECK: ret i64 [[T2]]
   return __arm_mte_ptrdiff(a, b);
}

// CHECK-LABEL: define{{.*}} i64 @subtract_pointers_null_1
ptrdiff_t subtract_pointers_null_1(int *a) {
// CHECK: [[T1:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(ptr %a, ptr null)
// CHECK: ret i64 [[T1]]
   return __arm_mte_ptrdiff(a, NULL);
}

// CHECK-LABEL: define{{.*}} i64 @subtract_pointers_null_2
ptrdiff_t subtract_pointers_null_2(int *a) {
// CHECK: [[T1:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(ptr null, ptr %a)
// CHECK: ret i64 [[T1]]
   return __arm_mte_ptrdiff(NULL, a);
}

// Check arithmetic promotion on return type
// CHECK-LABEL: define{{.*}} i32 @subtract_pointers4
int subtract_pointers4(void* a, void *b) {
// CHECK: [[T0:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(ptr %a, ptr %b)
// CHECK-NEXT: %cmp = icmp slt i64 [[T0]], 1
// CHECK-NEXT:  = zext i1 %cmp to i32
  return __arm_mte_ptrdiff(a,b) <= 0;
}
