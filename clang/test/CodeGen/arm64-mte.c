// Test memory tagging extension intrinsics
// RUN: %clang_cc1 -triple aarch64-none-linux-eabi -target-feature +mte -O3 -S -emit-llvm -o - %s  | FileCheck %s
#include <stddef.h>
#include <arm_acle.h>

// CHECK-LABEL: define i32* @create_tag1
int *create_tag1(int *a, unsigned b) {
// CHECK: [[T0:%[0-9]+]] = bitcast i32* %a to i8*
// CHECK: [[T1:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[T2:%[0-9]+]] = tail call i8* @llvm.aarch64.irg(i8* [[T0]], i64 [[T1]])
// CHECK: bitcast i8* [[T2]] to i32*
        return __arm_mte_create_random_tag(a,b);
}

// CHECK-LABEL: define i16* @create_tag2
short *create_tag2(short *a, unsigned b) {
// CHECK: [[T0:%[0-9]+]] = bitcast i16* %a to i8*
// CHECK: [[T1:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[T2:%[0-9]+]] = tail call i8* @llvm.aarch64.irg(i8* [[T0]], i64 [[T1]])
// CHECK: bitcast i8* [[T2]] to i16*
        return __arm_mte_create_random_tag(a,b);
}

// CHECK-LABEL: define i8* @create_tag3
char *create_tag3(char *a, unsigned b) {
// CHECK: [[T1:%[0-9]+]] = zext i32 %b to i64
// CHECK: [[T2:%[0-9]+]] = tail call i8* @llvm.aarch64.irg(i8* %a, i64 [[T1]])
// CHECK: ret i8* [[T2:%[0-9]+]]
        return __arm_mte_create_random_tag(a,b);
}

// CHECK-LABEL: define i8* @increment_tag1
char *increment_tag1(char *a) {
// CHECK: call i8* @llvm.aarch64.addg(i8* %a, i64 3)
        return __arm_mte_increment_tag(a,3);
}

// CHECK-LABEL: define i16* @increment_tag2
short *increment_tag2(short *a) {
// CHECK: [[T0:%[0-9]+]] = bitcast i16* %a to i8*
// CHECK: [[T1:%[0-9]+]] = tail call i8* @llvm.aarch64.addg(i8* [[T0]], i64 3)
// CHECK: [[T2:%[0-9]+]]  = bitcast i8* [[T1]] to i16*
        return __arm_mte_increment_tag(a,3);
}

// CHECK-LABEL: define i32 @exclude_tag
unsigned exclude_tag(int *a, unsigned m) {
// CHECK: [[T0:%[0-9]+]] = zext i32 %m to i64
// CHECK: [[T1:%[0-9]+]] = bitcast i32* %a to i8*
// CHECK: [[T2:%[0-9]+]] = tail call i64 @llvm.aarch64.gmi(i8* [[T1]], i64 [[T0]])
// CHECK: trunc i64 [[T2]] to i32
  return __arm_mte_exclude_tag(a, m);
}

// CHECK-LABEL: define i32* @get_tag1
int *get_tag1(int *a) {
// CHECK: [[T0:%[0-9]+]] = bitcast i32* %a to i8*
// CHECK: [[T1:%[0-9]+]] = tail call i8* @llvm.aarch64.ldg(i8* [[T0]], i8* [[T0]])
// CHECK: [[T2:%[0-9]+]]  = bitcast i8* [[T1]] to i32*
   return __arm_mte_get_tag(a);
}

// CHECK-LABEL: define i16* @get_tag2
short *get_tag2(short *a) {
// CHECK: [[T0:%[0-9]+]] = bitcast i16* %a to i8*
// CHECK: [[T1:%[0-9]+]] = tail call i8* @llvm.aarch64.ldg(i8* [[T0]], i8* [[T0]])
// CHECK: [[T2:%[0-9]+]]  = bitcast i8* [[T1]] to i16*
   return __arm_mte_get_tag(a);
}

// CHECK-LABEL: define void @set_tag1
void set_tag1(int *a) {
// CHECK: [[T0:%[0-9]+]] = bitcast i32* %a to i8*
// CHECK: tail call void @llvm.aarch64.stg(i8* [[T0]], i8* [[T0]])
   __arm_mte_set_tag(a);
}

// CHECK-LABEL: define i64 @subtract_pointers
ptrdiff_t subtract_pointers(int *a, int *b) {
// CHECK: [[T0:%[0-9]+]] = bitcast i32* %a to i8*
// CHECK: [[T1:%[0-9]+]] = bitcast i32* %b to i8*
// CHECK: [[T2:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(i8* [[T0]], i8* [[T1]])
// CHECK: ret i64 [[T2]]
   return __arm_mte_ptrdiff(a, b);
}

// CHECK-LABEL: define i64 @subtract_pointers_null_1
ptrdiff_t subtract_pointers_null_1(int *a) {
// CHECK: [[T0:%[0-9]+]] = bitcast i32* %a to i8*
// CHECK: [[T1:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(i8* [[T0]], i8* null)
// CHECK: ret i64 [[T1]]
   return __arm_mte_ptrdiff(a, NULL);
}

// CHECK-LABEL: define i64 @subtract_pointers_null_2
ptrdiff_t subtract_pointers_null_2(int *a) {
// CHECK: [[T0:%[0-9]+]] = bitcast i32* %a to i8*
// CHECK: [[T1:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(i8* null, i8* [[T0]])
// CHECK: ret i64 [[T1]]
   return __arm_mte_ptrdiff(NULL, a);
}

// Check arithmetic promotion on return type
// CHECK-LABEL: define i32 @subtract_pointers4
int subtract_pointers4(void* a, void *b) {
// CHECK: [[T0:%[0-9]+]] = tail call i64 @llvm.aarch64.subp(i8* %a, i8* %b)
// CHECK-NEXT: %cmp = icmp slt i64 [[T0]], 1
// CHECK-NEXT:  = zext i1 %cmp to i32
  return __arm_mte_ptrdiff(a,b) <= 0;
}
