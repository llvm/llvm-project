// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR_BEFORE
// RUN: FileCheck %s -check-prefix=CIR_AFTER --input-file=%t.cir
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file=%t.ll
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-cir -fexceptions -fcxx-exceptions %s -o %t.eh.cir
// RUN: FileCheck %s -check-prefix=CIR_EH --input-file=%t.eh.cir
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-cir-flat -fexceptions -fcxx-exceptions %s -o %t.eh.flat.cir
// RUN: FileCheck %s -check-prefix=CIR_FLAT_EH --input-file=%t.eh.flat.cir
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -fexceptions -fcxx-exceptions %s -o %t.eh.ll
// RUN: FileCheck %s -check-prefix=LLVM_EH --input-file=%t.eh.ll

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

// CIR_EH: cir.try synthetic cleanup {
// CIR_EH:   cir.call exception @_ZN1eC1Ei{{.*}} cleanup {
// CIR_EH:     cir.call @_ZdlPvm
// CIR_EH:     cir.yield
// CIR_EH:   }
// CIR_EH:   cir.yield
// CIR_EH: } catch [#cir.unwind {
// CIR_EH:   cir.resume
// CIR_EH: }]

// CIR_FLAT_EH: cir.func internal private  @__cxx_global_var_init()
// CIR_FLAT_EH: ^bb3:
// CIR_FLAT_EH:   %exception_ptr, %type_id = cir.eh.inflight_exception
// CIR_FLAT_EH:   cir.call @_ZdlPvm({{.*}}) : (!cir.ptr<!void>, !u64i) -> ()
// CIR_FLAT_EH:   cir.br ^bb4(%exception_ptr, %type_id : !cir.ptr<!void>, !u32i)

// LLVM_EH: define internal void @__cxx_global_var_init() personality ptr @__gxx_personality_v0
// LLVM_EH:   call ptr @_Znwm(i64 1)
// LLVM_EH:   br label %[[L2:.*]],

// LLVM_EH: [[L2]]:
// LLVM_EH:   invoke void @_ZN1eC1Ei
// LLVM_EH:           to label %[[CONT:.*]] unwind label %[[PAD:.*]],

// LLVM_EH: [[CONT]]:
// LLVM_EH:   br label %[[END:.*]],

// LLVM_EH: [[PAD]]:
// LLVM_EH:   landingpad { ptr, i32 }
// LLVM_EH:      cleanup
// LLVM_EH:   call void @_ZdlPvm
// LLVM_EH:   br label %[[RESUME:.*]],

// LLVM_EH: [[RESUME]]:
// LLVM_EH:   resume { ptr, i32 }

// LLVM_EH: [[END]]:
// LLVM_EH:   store ptr {{.*}}, ptr @g, align 8
// LLVM_EH:   ret void
// LLVM_EH: }

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