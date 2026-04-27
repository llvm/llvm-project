// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck --check-prefix=CLLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s
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

// CLLVM-LABEL:  @_Z1av(
// CLLVM:         [[TMP1:%.*]] = alloca ptr, i64 1, align 8
// CLLVM-NEXT:    [[TMP2:%.*]] = load ptr, ptr @x, align 8
// CLLVM-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8
// CLLVM-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP3]], i64 -32
// CLLVM-NEXT:    [[TMP5:%.*]] = load i64, ptr [[TMP4]], align 8
// CLLVM-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP2]], i64 [[TMP5]]
// CLLVM-NEXT:    store ptr [[TMP6]], ptr [[TMP1]], align 8
// CLLVM-NEXT:    [[TMP7:%.*]] = load ptr, ptr [[TMP1]], align 8
// CLLVM-NEXT:    ret ptr [[TMP7]]
//
// LLVM-LABEL:  @_Z1av(
// LLVM:       [[ENTRY:.*]]:
// LLVM-NEXT:    [[TMP0:%.*]] = load ptr, ptr @x, align 8
// LLVM-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// LLVM-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// LLVM:       [[CAST_NOTNULL]]:
// LLVM-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -32
// LLVM-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// LLVM-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[VBASE_OFFSET]]
// LLVM-NEXT:    br label %[[CAST_END]]
// LLVM:       [[CAST_END]]:
// LLVM-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// LLVM-NEXT:    ret ptr [[CAST_RESULT]]
//
A* a() { return x; }
// CHECK: @_Z1av() [[NUW:#[0-9]+]]
// CHECK: [[VBASEOFFSETPTRA:%[a-zA-Z0-9\.]+]] = getelementptr i8, ptr {{.*}}, i64 -16
// CHECK: load i32, ptr [[VBASEOFFSETPTRA]]
// CHECK: }

// CLLVM-LABEL:  @_Z1bv(
// CLLVM:         [[TMP1:%.*]] = alloca ptr, i64 1, align 8
// CLLVM-NEXT:    [[TMP2:%.*]] = load ptr, ptr @x, align 8
// CLLVM-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8
// CLLVM-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP3]], i64 -40
// CLLVM-NEXT:    [[TMP5:%.*]] = load i64, ptr [[TMP4]], align 8
// CLLVM-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[TMP2]], i64 [[TMP5]]
// CLLVM-NEXT:    store ptr [[TMP6]], ptr [[TMP1]], align 8
// CLLVM-NEXT:    [[TMP7:%.*]] = load ptr, ptr [[TMP1]], align 8
// CLLVM-NEXT:    ret ptr [[TMP7]]
//
// LLVM-LABEL:  @_Z1bv(
// LLVM:       [[ENTRY:.*]]:
// LLVM-NEXT:    [[TMP0:%.*]] = load ptr, ptr @x, align 8
// LLVM-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// LLVM-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// LLVM:       [[CAST_NOTNULL]]:
// LLVM-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -40
// LLVM-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// LLVM-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[VBASE_OFFSET]]
// LLVM-NEXT:    br label %[[CAST_END]]
// LLVM:       [[CAST_END]]:
// LLVM-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// LLVM-NEXT:    ret ptr [[CAST_RESULT]]
//
B* b() { return x; }

// CLLVM-LABEL:  @_Z1cv(
// CLLVM:         [[TMP1:%.*]] = alloca ptr, i64 1, align 8
// CLLVM-NEXT:    [[TMP2:%.*]] = load ptr, ptr @x, align 8
// CLLVM-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8
// CLLVM-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP3]], i64 -48
// CLLVM-NEXT:    [[TMP5:%.*]] = load i64, ptr [[TMP4]], align 8
// CLLVM-NEXT:    [[TMP6:%.*]] = add i64 [[TMP5]], 16
// CLLVM-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP2]], i64 [[TMP6]]
// CLLVM-NEXT:    store ptr [[TMP7]], ptr [[TMP1]], align 8
// CLLVM-NEXT:    [[TMP8:%.*]] = load ptr, ptr [[TMP1]], align 8
// CLLVM-NEXT:    ret ptr [[TMP8]]
//
// LLVM-LABEL:  @_Z1cv(
// LLVM:       [[ENTRY:.*]]:
// LLVM-NEXT:    [[TMP0:%.*]] = load ptr, ptr @x, align 8
// LLVM-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// LLVM-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// LLVM:       [[CAST_NOTNULL]]:
// LLVM-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -48
// LLVM-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// LLVM-NEXT:    [[TMP2:%.*]] = add i64 [[VBASE_OFFSET]], 16
// LLVM-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP2]]
// LLVM-NEXT:    br label %[[CAST_END]]
// LLVM:       [[CAST_END]]:
// LLVM-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// LLVM-NEXT:    ret ptr [[CAST_RESULT]]
//
BB* c() { return x; }

struct E { int e; };
struct F : E, D { int f; };

F* y;

// CIR-LABEL: @_Z1dv
// CIR: %[[OFFSET:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: %[[ADJUST:.*]] = cir.const #cir.int<16> : !s64i
// CIR: cir.add %[[OFFSET]], %[[ADJUST]] : !s64i

// CLLVM-LABEL:  @_Z1dv(
// CLLVM:         [[TMP1:%.*]] = alloca ptr, i64 1, align 8
// CLLVM-NEXT:    [[TMP2:%.*]] = load ptr, ptr @y, align 8
// CLLVM-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[TMP2]], align 8
// CLLVM-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[TMP3]], i64 -48
// CLLVM-NEXT:    [[TMP5:%.*]] = load i64, ptr [[TMP4]], align 8
// CLLVM-NEXT:    [[TMP6:%.*]] = add i64 [[TMP5]], 16
// CLLVM-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[TMP2]], i64 [[TMP6]]
// CLLVM-NEXT:    store ptr [[TMP7]], ptr [[TMP1]], align 8
// CLLVM-NEXT:    [[TMP8:%.*]] = load ptr, ptr [[TMP1]], align 8
// CLLVM-NEXT:    ret ptr [[TMP8]]
//
// LLVM-LABEL:  @_Z1dv(
// LLVM:       [[ENTRY:.*]]:
// LLVM-NEXT:    [[TMP0:%.*]] = load ptr, ptr @y, align 8
// LLVM-NEXT:    [[TMP1:%.*]] = icmp eq ptr [[TMP0]], null
// LLVM-NEXT:    br i1 [[TMP1]], label %[[CAST_END:.*]], label %[[CAST_NOTNULL:.*]]
// LLVM:       [[CAST_NOTNULL]]:
// LLVM-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[TMP0]], align 8
// LLVM-NEXT:    [[VBASE_OFFSET_PTR:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 -48
// LLVM-NEXT:    [[VBASE_OFFSET:%.*]] = load i64, ptr [[VBASE_OFFSET_PTR]], align 8
// LLVM-NEXT:    [[TMP2:%.*]] = add i64 [[VBASE_OFFSET]], 16
// LLVM-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP2]]
// LLVM-NEXT:    br label %[[CAST_END]]
// LLVM:       [[CAST_END]]:
// LLVM-NEXT:    [[CAST_RESULT:%.*]] = phi ptr [ [[ADD_PTR]], %[[CAST_NOTNULL]] ], [ null, %[[ENTRY]] ]
// LLVM-NEXT:    ret ptr [[CAST_RESULT]]
//
BB* d() { return y; }

