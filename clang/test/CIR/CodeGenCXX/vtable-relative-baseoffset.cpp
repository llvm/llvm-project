// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -fclangir -emit-cir %s -o - | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -fclangir -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -emit-llvm %s -o - | FileCheck --check-prefix=OGCG %s

// vbase-offset.cpp

struct V {
  int x;
};

struct A : virtual V {
};

struct B : A {
};
// CIR-LABEL: @_Z1fP1B(
// CIR:         [[P:%.*]] = cir.load align(8) {{%.*}} : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR-NEXT:    [[VPTR_PTR:%.*]] = cir.vtable.get_vptr [[P]] : !cir.ptr<!rec_B> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:    [[VTABLE:%.*]] = cir.load align(8) [[VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-NEXT:    [[VTABLE_BYTES:%.*]] = cir.cast bitcast [[VTABLE]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR-NEXT:    [[VBASE_OFFSET_SLOT_OFFSET:%.*]] = cir.const #cir.int<-12> : !s64i
// CIR-NEXT:    [[VBASE_OFFSET_SLOT:%.*]] = cir.ptr_stride [[VTABLE_BYTES]], [[VBASE_OFFSET_SLOT_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = cir.cast bitcast [[VBASE_OFFSET_SLOT]] : !cir.ptr<!u8i> -> !cir.ptr<!s32i>
// CIR-NEXT:    [[VBASE_OFFSET:%.*]] = cir.load align(4) [[VBASE_OFFSET_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    [[P_BYTES:%.*]] = cir.cast bitcast [[P]] : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR-NEXT:    [[VBASE_PTR_BYTES:%.*]] = cir.ptr_stride [[P_BYTES]], [[VBASE_OFFSET]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// CIR-NEXT:    [[VBASE_PTR_B:%.*]] = cir.cast bitcast [[VBASE_PTR_BYTES]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_B>
// CIR-NEXT:    [[VBASE_PTR:%.*]] = cir.cast bitcast [[VBASE_PTR_B]] : !cir.ptr<!rec_B> -> !cir.ptr<!rec_V>
// CIR-NEXT:    [[X_PTR:%.*]] = cir.get_member [[VBASE_PTR]][0] {name = "x"} : !cir.ptr<!rec_V> -> !cir.ptr<!s32i>
// CIR-NEXT:    [[X:%.*]] = cir.load align(4) [[X_PTR]] : !cir.ptr<!s32i>, !s32i
//
// LLVM-LABEL: @_Z1fP1B(
// LLVM:         [[P:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[P]], align 8
// LLVM-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -12
// LLVM-NEXT:    [[VBASE_OFFSET_I32:%.*]] = load i32, ptr [[VBASE_OFFSET_PTR]], align 4
// LLVM-NEXT:    [[VBASE_OFFSET:%.*]] = sext i32 [[VBASE_OFFSET_I32]] to i64
// LLVM-NEXT:    [[VBASE_PTR:%.*]] = getelementptr i8, ptr [[P]], i64 [[VBASE_OFFSET]]
// LLVM-NEXT:    [[X_PTR:%.*]] = getelementptr inbounds nuw [[STRUCT_V:%.*]], ptr [[VBASE_PTR]], i32 0, i32 0
// LLVM-NEXT:    [[X:%.*]] = load i32, ptr [[X_PTR]], align 4
// LLVM:         ret i32
//
// OGCG-LABEL: define dso_local noundef i32 @_Z1fP1B(
// OGCG-SAME: ptr noundef [[P:%.*]]) #[[ATTR0:[0-9]+]] {
// OGCG-NEXT:  [[ENTRY:.*]]:
// OGCG-NEXT:    [[P_ADDR:%.*]] = alloca ptr, align 8
// OGCG-NEXT:    store ptr [[P]], ptr [[P_ADDR]], align 8
// OGCG-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[P_ADDR]], align 8
// OGCG-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// OGCG-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// OGCG:       [[CAST_NOTNULL]]:
// OGCG-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -12
// OGCG-NEXT:    [[VBASE_OFFSET:%.*]] = load i32, ptr [[VBASE_OFFSET_PTR]], align 4
// OGCG-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i32 [[VBASE_OFFSET]]
// OGCG-NEXT:    br label %[[CAST_END]]
// OGCG:       [[CAST_END]]:
// OGCG-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// OGCG-NEXT:    [[X:%.*]] = getelementptr inbounds nuw [[STRUCT_V:%.*]], ptr [[CAST_RESULT]], i32 0, i32 0
// OGCG-NEXT:    [[TMP2:%.*]] = load i32, ptr [[X]], align 4
// OGCG-NEXT:    ret i32 [[TMP2]]
//
int f(B *p) {
  return static_cast<V *>(p)->x;
}
