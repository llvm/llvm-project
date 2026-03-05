
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct A { int a; virtual int aa(); };
struct B { int b; virtual int bb(); };
struct C : virtual A, virtual B { int c; virtual int aa(); virtual int bb(); };
struct AA { int a; virtual int aa(); };
struct BB { int b; virtual int bb(); };
struct CC : AA, BB { virtual int aa(); virtual int bb(); virtual int cc(); };
struct D : virtual C, virtual CC { int e; };

D* x;

A* a() { return x; }
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

// FIXME: this version should include null check.
// LLVM-LABEL: @_Z1av()
// LLVM:  %[[OFFSET_OFFSET:.*]] = getelementptr i8, ptr {{.*}}, i64 -32
// LLVM:  %[[OFFSET_PTR:.*]] = load i64, ptr %[[OFFSET_OFFSET]], align 8
// LLVM:  %[[VBASE_ADDR:.*]] = getelementptr i8, ptr {{.*}}, i64 %[[OFFSET_PTR]]
// LLVM:  store ptr %[[VBASE_ADDR]], ptr {{.*}}, align 8

B* b() { return x; }
BB* c() { return x; }

// Put the vbptr at a non-zero offset inside a non-virtual base.
struct E { int e; };
struct F : E, D { int f; };

F* y;

BB* d() { return y; }
// CIR-LABEL: @_Z1dv
// CIR: %[[OFFSET:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: %[[ADJUST:.*]] = cir.const #cir.int<16> : !s64i
// CIR: cir.binop(add, %[[OFFSET]], %[[ADJUST]]) : !s64i

// LLVM-LABEL: @_Z1dv
// LLVM: %[[OFFSET_OFFSET:.*]] = getelementptr i8, ptr {{.*}}, i64 -48
// LLVM: %[[OFFSET_PTR:.*]] = load i64, ptr %[[OFFSET_OFFSET]], align 8
// LLVM: %[[ADJUST:.*]] = add i64 %[[OFFSET_PTR]], 16
// LLVM: %[[VBASE_ADDR:.*]] = getelementptr i8, ptr {{.*}}, i64 %[[ADJUST]]
// LLVM: store ptr %[[VBASE_ADDR]],
