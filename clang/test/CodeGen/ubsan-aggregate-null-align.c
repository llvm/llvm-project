// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=alignment,null \
// RUN:   -emit-llvm -std=c23 %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-UBSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -std=c23 %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-NO-UBSAN
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=alignment,null \
// RUN:   -emit-llvm -xc++ %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-UBSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -xc++ %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-NO-UBSAN

// Test that EmitAggregateCopy emits null and alignment checks when sanitizers
// are enabled for aggregate copy operations with pointers.

typedef struct AlignedStruct {
  alignas(16) int a;
  int b;
  int c;
  int d;
} AlignedStruct;

typedef struct NormalStruct {
  int x;
  int y;
  int z;
} NormalStruct;

#if __cplusplus
extern "C" {
#endif
// Stack-to-stack copies are optimized away (compiler knows they're valid)
// CHECK-LABEL: define {{.*}}void @test_aligned_struct()
void test_aligned_struct() {
  AlignedStruct src = {1, 2, 3, 4};
  AlignedStruct dest;

  // CHECK-UBSAN: call void @llvm.memcpy
  // CHECK-NO-UBSAN: call void @llvm.memcpy

  dest = src;
}

// CHECK-LABEL: define {{.*}}void @test_normal_struct()
void test_normal_struct() {
  NormalStruct src = {10, 20, 30};
  NormalStruct dest;

  // CHECK-UBSAN: call void @llvm.memcpy
  // CHECK-NO-UBSAN: call void @llvm.memcpy

  dest = src;
}

// This is the key test - copying through pointers requires runtime checks
// CHECK-LABEL: define {{.*}}void @test_pointer_to_ptr(
void test_pointer_to_ptr(AlignedStruct *src, AlignedStruct *dest) {
  // CHECK-UBSAN: %[[SRC_LOAD:.*]] = load ptr, ptr %src.addr
  // CHECK-UBSAN: %[[DEST_LOAD:.*]] = load ptr, ptr %dest.addr

  // Check source pointer is non-null and aligned
  // CHECK-UBSAN: %[[SRC_NONNULL:.*]] = icmp ne ptr %[[SRC_LOAD]], null
  // CHECK-UBSAN: %[[SRC_INT:.*]] = ptrtoint ptr %[[SRC_LOAD]] to i64
  // CHECK-UBSAN: %[[SRC_MASK:.*]] = and i64 %[[SRC_INT]], 15
  // CHECK-UBSAN: %[[SRC_ALIGNED:.*]] = icmp eq i64 %[[SRC_MASK]], 0
  // CHECK-UBSAN: %[[SRC_OK:.*]] = and i1 %[[SRC_NONNULL]], %[[SRC_ALIGNED]]
  // CHECK-UBSAN: br i1 %[[SRC_OK]], label %cont, label %handler.type_mismatch

  // CHECK-UBSAN: handler.type_mismatch:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort
  // CHECK-UBSAN: unreachable

  // CHECK-UBSAN: cont:
  // Check destination pointer is non-null and aligned
  // CHECK-UBSAN: %[[DEST_NONNULL:.*]] = icmp ne ptr %[[DEST_LOAD]], null
  // CHECK-UBSAN: %[[DEST_INT:.*]] = ptrtoint ptr %[[DEST_LOAD]] to i64
  // CHECK-UBSAN: %[[DEST_MASK:.*]] = and i64 %[[DEST_INT]], 15
  // CHECK-UBSAN: %[[DEST_ALIGNED:.*]] = icmp eq i64 %[[DEST_MASK]], 0
  // CHECK-UBSAN: %[[DEST_OK:.*]] = and i1 %[[DEST_NONNULL]], %[[DEST_ALIGNED]]
  // CHECK-UBSAN: br i1 %[[DEST_OK]], label %cont{{.*}}, label %handler.type_mismatch

  // CHECK-UBSAN: handler.type_mismatch{{.*}}:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort
  // CHECK-UBSAN: unreachable

  // CHECK-UBSAN: cont{{.*}}:
  // CHECK-UBSAN: call void @llvm.memcpy

  // Without sanitizers, no checks - just direct memcpy
  // CHECK-NO-UBSAN-NOT: @__ubsan_handle
  // CHECK-NO-UBSAN: call void @llvm.memcpy

  *dest = *src;
}

// Array copies also need checks for non-constant indices
// CHECK-LABEL: define {{.*}}void @test_array_copy()
void test_array_copy() {
  AlignedStruct src[3] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  AlignedStruct dest[3];

  // First element - no checks needed (compiler knows it's aligned)
  // CHECK-UBSAN: %arrayidx = getelementptr inbounds [3 x %struct.AlignedStruct]
  // CHECK-UBSAN: %arrayidx1 = getelementptr inbounds [3 x %struct.AlignedStruct]
  // CHECK-UBSAN: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %arrayidx1, ptr align 16 %arrayidx, i64 16, i1 false)
  dest[0] = src[0];

  // Second element - needs runtime checks
  // CHECK-UBSAN: %arrayidx{{.*}} = getelementptr inbounds [3 x %struct.AlignedStruct]
  // CHECK-UBSAN: %arrayidx{{.*}} = getelementptr inbounds [3 x %struct.AlignedStruct]
  // CHECK-UBSAN: icmp ne ptr %arrayidx{{.*}}, null
  // CHECK-UBSAN: ptrtoint ptr %arrayidx{{.*}} to i64
  // CHECK-UBSAN: and i64 %{{.*}}, 15
  // CHECK-UBSAN: icmp eq i64 %{{.*}}, 0
  // CHECK-UBSAN: br i1 %{{.*}}, label %cont, label %handler.type_mismatch
  // CHECK-UBSAN: handler.type_mismatch:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort
  dest[1] = src[1];

  // Third element - also needs checks
  // CHECK-UBSAN: icmp ne ptr %arrayidx{{.*}}, null
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort
  dest[2] = src[2];

  // Without sanitizers, no checks
  // CHECK-NO-UBSAN-NOT: @__ubsan_handle
}

// Test with normal struct through pointers
// CHECK-LABEL: define {{.*}}void @test_normal_struct_ptrs(
void test_normal_struct_ptrs(NormalStruct *src, NormalStruct *dest) {
  // Should still check for null even with normal alignment
  // CHECK-UBSAN: icmp ne ptr %{{.*}}, null
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort

  // CHECK-NO-UBSAN-NOT: @__ubsan_handle

  *dest = *src;
}
#if __cplusplus
}
#endif
