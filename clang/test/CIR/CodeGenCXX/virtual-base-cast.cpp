// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix=OGCG %s
struct A { int a; virtual int aa(); };
struct B { int b; virtual int bb(); };
struct C : virtual A, virtual B { int c; virtual int aa(); virtual int bb(); };
struct AA { int a; virtual int aa(); };
struct BB { int b; virtual int bb(); };
struct CC : AA, BB { virtual int aa(); virtual int bb(); virtual int cc(); };
struct D : virtual C, virtual CC { int e; };

D* x;

// CIR-LABEL: @_Z1av()

// This uses the vtable to get the offset to the base object. The offset from
// the vptr to the base object offset in the vtable is a compile-time constant.
// CIR: %[[X_ADDR:.*]] = cir.get_global @x : !cir.ptr<!cir.ptr<!rec_D>>
// CIR: %[[X:.*]] = cir.load{{.*}} %[[X_ADDR]]
// CIR: %[[X_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[X]] : !cir.ptr<!rec_D> -> !cir.ptr<!cir.vptr>
// CIR: %[[X_VPTR_BASE:.*]] = cir.load{{.*}} %[[X_VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR: %[[X_BASE_I8PTR:.*]] = cir.cast bitcast %[[X_VPTR_BASE]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR:  %[[OFFSET_OFFSET:.*]] = cir.const #cir.int<-32> : !s64i
// CIR:  %[[OFFSET_PTR:.*]] = cir.ptr_stride %[[X_BASE_I8PTR]], %[[OFFSET_OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:  %[[OFFSET_PTR_CAST:.*]] = cir.cast bitcast %[[OFFSET_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR:  %[[OFFSET:.*]] = cir.load{{.*}} %[[OFFSET_PTR_CAST]] : !cir.ptr<!s64i>, !s64i
// CIR:  %[[VBASE_ADDR:.*]] = cir.ptr_stride {{.*}}, %[[OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:  cir.cast bitcast %[[VBASE_ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_D>

// LLVM-LABEL: @_Z1av(
// LLVM:       [[OBJ:%.*]] = load ptr, ptr @x
// LLVM-NEXT:  [[VTABLE:%.*]] = load ptr, ptr [[OBJ]]
// LLVM-NEXT:  [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -32
// LLVM-NEXT:  [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]]
// LLVM-NEXT:  [[ADD_PTR:%.*]] = getelementptr i8, ptr [[OBJ]], i64 [[VBASE_OFFSET]]
// LLVM:       ret ptr

// OGCG-LABEL:  @_Z1av(
// OGCG:       [[ENTRY:.*]]:
// OGCG-NEXT:    [[TMP0:%.*]] = load ptr, ptr @x, align 8
// OGCG-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// OGCG-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// OGCG:       [[CAST_NOTNULL]]:
// OGCG-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -32
// OGCG-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// OGCG-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[VBASE_OFFSET]]
// OGCG-NEXT:    br label %[[CAST_END]]
// OGCG:       [[CAST_END]]:
// OGCG-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// OGCG-NEXT:    ret ptr [[CAST_RESULT]]
A* a() { return x; }

// LLVM-LABEL: @_Z1bv(
// LLVM:       [[OBJ:%.*]] = load ptr, ptr @x
// LLVM-NEXT:  [[VTABLE:%.*]] = load ptr, ptr [[OBJ]]
// LLVM-NEXT:  [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -40
// LLVM-NEXT:  [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]]
// LLVM-NEXT:  [[ADD_PTR:%.*]] = getelementptr i8, ptr [[OBJ]], i64 [[VBASE_OFFSET]]
// LLVM:       ret ptr

// OGCG-LABEL:  @_Z1bv(
// OGCG:       [[ENTRY:.*]]:
// OGCG-NEXT:    [[TMP0:%.*]] = load ptr, ptr @x, align 8
// OGCG-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// OGCG-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// OGCG:       [[CAST_NOTNULL]]:
// OGCG-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -40
// OGCG-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// OGCG-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[VBASE_OFFSET]]
// OGCG-NEXT:    br label %[[CAST_END]]
// OGCG:       [[CAST_END]]:
// OGCG-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// OGCG-NEXT:    ret ptr [[CAST_RESULT]]
//
B* b() { return x; }

// LLVM-LABEL: @_Z1cv(
// LLVM:       [[OBJ:%.*]] = load ptr, ptr @x
// LLVM-NEXT:  [[VTABLE:%.*]] = load ptr, ptr [[OBJ]]
// LLVM-NEXT:  [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -48
// LLVM-NEXT:  [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]]
// LLVM-NEXT:  [[OFFSET:%.*]] = add i64 [[VBASE_OFFSET]], 16
// LLVM-NEXT:  [[ADD_PTR:%.*]] = getelementptr i8, ptr [[OBJ]], i64 [[OFFSET]]
// LLVM:       ret ptr

// OGCG-LABEL:  @_Z1cv(
// OGCG:       [[ENTRY:.*]]:
// OGCG-NEXT:    [[TMP0:%.*]] = load ptr, ptr @x, align 8
// OGCG-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// OGCG-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// OGCG:       [[CAST_NOTNULL]]:
// OGCG-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -48
// OGCG-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// OGCG-NEXT:    [[TMP2:%.*]] = add i64 [[VBASE_OFFSET]], 16
// OGCG-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP2]]
// OGCG-NEXT:    br label %[[CAST_END]]
// OGCG:       [[CAST_END]]:
// OGCG-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// OGCG-NEXT:    ret ptr [[CAST_RESULT]]
//
BB* c() { return x; }

struct E { int e; };
struct F : E, D { int f; };

F* y;

// CIR-LABEL: @_Z1dv
// CIR: %[[OFFSET:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: %[[ADJUST:.*]] = cir.const #cir.int<16> : !s64i
// CIR: cir.add %[[OFFSET]], %[[ADJUST]] : !s64i


// LLVM-LABEL: @_Z1dv(
// LLVM:       [[OBJ:%.*]] = load ptr, ptr @y
// LLVM-NEXT:  [[VTABLE:%.*]] = load ptr, ptr [[OBJ]]
// LLVM-NEXT:  [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -48
// LLVM-NEXT:  [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]]
// LLVM-NEXT:  [[OFFSET:%.*]] = add i64 [[VBASE_OFFSET]], 16
// LLVM-NEXT:  [[ADD_PTR:%.*]] = getelementptr i8, ptr [[OBJ]], i64 [[OFFSET]]
// LLVM:       ret ptr

// OGCG-LABEL:  @_Z1dv(
// OGCG:       [[ENTRY:.*]]:
// OGCG-NEXT:    [[TMP0:%.*]] = load ptr, ptr @y, align 8
// OGCG-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// OGCG-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// OGCG:       [[CAST_NOTNULL]]:
// OGCG-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// OGCG-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -48
// OGCG-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// OGCG-NEXT:    [[TMP2:%.*]] = add i64 [[VBASE_OFFSET]], 16
// OGCG-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP2]]
// OGCG-NEXT:    br label %[[CAST_END]]
// OGCG:       [[CAST_END]]:
// OGCG-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// OGCG-NEXT:    ret ptr [[CAST_RESULT]]
//
BB* d() { return y; }

