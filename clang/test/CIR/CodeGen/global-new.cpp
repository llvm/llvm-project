// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR_BEFORE
// RUN: FileCheck %s -check-prefix=CIR_AFTER --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file=%t.ll

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fexceptions -fcxx-exceptions %s -o %t.eh.cir 2>&1
// RUN: FileCheck %s -check-prefix=CIR_EH --input-file=%t.eh.cir

struct e { e(int); };
e *g = new e(0);

// CIR_BEFORE: ![[ty:.*]] = !cir.struct<struct "e" {!cir.int<u, 8>}

// CIR_BEFORE: cir.global  external @g = ctor : !cir.ptr<![[ty]]> {
// CIR_BEFORE:     %[[GlobalAddr:.*]] = cir.get_global @g : !cir.ptr<!cir.ptr<![[ty]]>>
// CIR_BEFORE:     %[[Size:.*]] = cir.const #cir.int<1> : !u64i
// CIR_BEFORE:     %[[NewAlloc:.*]] = cir.call @_Znwm(%[[Size]]) : (!u64i) -> !cir.ptr<!void>
// CIR_BEFORE:     %[[NewCasted:.*]] = cir.cast(bitcast, %[[NewAlloc]] : !cir.ptr<!void>), !cir.ptr<![[ty]]>
// CIR_BEFORE:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR_BEFORE:     cir.call @_ZN1eC1Ei(%[[NewCasted]], %[[ZERO]]) : (!cir.ptr<![[ty]]>, !s32i) -> ()
// CIR_BEFORE:     cir.store %3, %[[GlobalAddr]] : !cir.ptr<![[ty]]>, !cir.ptr<!cir.ptr<![[ty]]>>
// CIR_BEFORE: }

// CIR_AFTER:  {{%.*}} = cir.const #cir.int<1> : !u64i
// CIR_AFTER:  {{%.*}} = cir.call @_Znwm(%1) : (!u64i) -> !cir.ptr<!void>

// CIR_EH: cir.try synthetic {
// CIR_EH:   cir.call exception @_ZN1eC1Ei
// CIR_EH:   cir.yield
// CIR_EH: } cleanup {
// CIR_EH:   cir.call @_ZdlPvm
// CIR_EH:   cir.yield
// CIR_EH: } catch [#cir.unwind {
// CIR_EH:   cir.resume
// CIR_EH: }]

// LLVM-DAG: @llvm.global_ctors = appending constant [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65536, ptr @__cxx_global_var_init, ptr null }, { i32, ptr, ptr } { i32 65536, ptr @__cxx_global_var_init.1, ptr null }]
// LLVM: define internal void @__cxx_global_var_init()
// LLVM: call ptr @_Znwm(i64 1)

// LLVM: define internal void @__cxx_global_var_init.1()
// LLVM:   call ptr @_Znwm(i64 1)

// LLVM: define void @_GLOBAL__sub_I_global_new.cpp()
// LLVM:   call void @__cxx_global_var_init()
// LLVM:   call void @__cxx_global_var_init.1()

struct PackedStruct {
};
PackedStruct*const packed_2 = new PackedStruct();