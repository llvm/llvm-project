// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// This duplicates a test case in global-init.cpp, but having it by itself
// forces the __cxa_atexit function to be emitted for this case, which was
// broken in the original implementation.

struct ArrayDtor {
  ~ArrayDtor();
};

ArrayDtor arrDtor[16];

// CIR-BEFORE-LPP:      cir.global external @arrDtor = #cir.zero : !cir.array<!rec_ArrayDtor x 16>
// CIR-BEFORE-LPP-SAME:   dtor {
// CIR-BEFORE-LPP:          %[[THIS:.*]] = cir.get_global @arrDtor : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>>
// CIR-BEFORE-LPP:          cir.array.dtor %[[THIS]] : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>> {
// CIR-BEFORE-LPP:          ^bb0(%[[ELEM:.*]]: !cir.ptr<!rec_ArrayDtor>):
// CIR-BEFORE-LPP:            cir.call @_ZN9ArrayDtorD1Ev(%[[ELEM]]) nothrow : (!cir.ptr<!rec_ArrayDtor>) -> ()
// CIR-BEFORE-LPP:            cir.yield
// CIR-BEFORE-LPP:          }
// CIR-BEFORE-LPP:        }

// CIR: cir.global external @arrDtor = #cir.zero : !cir.array<!rec_ArrayDtor x 16> {alignment = 16 : i64}
// CIR: cir.func internal private @__cxx_global_array_dtor(%[[ARR_ARG:.*]]: !cir.ptr<!void> {{.*}}) {
// CIR:   %[[CONST15:.*]] = cir.const #cir.int<15> : !u64i
// CIR:   %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARR_ARG]] : !cir.ptr<!void> -> !cir.ptr<!rec_ArrayDtor>
// CIR:   %[[END:.*]] = cir.ptr_stride %[[BEGIN]], %[[CONST15]] : (!cir.ptr<!rec_ArrayDtor>, !u64i) -> !cir.ptr<!rec_ArrayDtor>
// CIR:   %[[CUR_ADDR:.*]] = cir.alloca !cir.ptr<!rec_ArrayDtor>, !cir.ptr<!cir.ptr<!rec_ArrayDtor>>, ["__array_idx"]
// CIR:   cir.store %[[END]], %[[CUR_ADDR]] : !cir.ptr<!rec_ArrayDtor>, !cir.ptr<!cir.ptr<!rec_ArrayDtor>>
// CIR:   cir.do {
// CIR:     %[[CUR:.*]] = cir.load %[[CUR_ADDR]] : !cir.ptr<!cir.ptr<!rec_ArrayDtor>>, !cir.ptr<!rec_ArrayDtor>
// CIR:     cir.call @_ZN9ArrayDtorD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_ArrayDtor>) -> ()
// CIR:     %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:     %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[NEG_ONE]] : (!cir.ptr<!rec_ArrayDtor>, !s64i) -> !cir.ptr<!rec_ArrayDtor>
// CIR:     cir.store %[[NEXT]], %[[CUR_ADDR]] : !cir.ptr<!rec_ArrayDtor>, !cir.ptr<!cir.ptr<!rec_ArrayDtor>>
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[CUR:.*]] = cir.load %[[CUR_ADDR]] : !cir.ptr<!cir.ptr<!rec_ArrayDtor>>, !cir.ptr<!rec_ArrayDtor>
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[CUR]], %[[BEGIN]]) : !cir.ptr<!rec_ArrayDtor>, !cir.bool
// CIR:     cir.condition(%[[CMP]])
// CIR:   }
// CIR:   cir.return
// CIR: }
//
// CIR: cir.func internal private @__cxx_global_var_init() {
// CIR:   %[[ARR:.*]] = cir.get_global @arrDtor : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>>
// CIR:   %[[DTOR:.*]] = cir.get_global @__cxx_global_array_dtor : !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[DTOR_CAST:.*]] = cir.cast bitcast %[[DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!void>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[ARR_CAST:.*]] = cir.cast bitcast %[[ARR]] : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>> -> !cir.ptr<!void>
// CIR:   %[[HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_CAST]], %[[ARR_CAST]], %[[HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()

// LLVM: define internal void @__cxx_global_array_dtor(ptr %[[ARR_ARG:.*]]) {
// LLVM:   %[[BEGIN:.*]] = getelementptr %struct.ArrayDtor, ptr %[[ARR_ARG]], i32 0
// LLVM:   %[[END:.*]] = getelementptr %struct.ArrayDtor, ptr %[[BEGIN]], i64 15
// LLVM:   %[[CUR_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[END]], ptr %[[CUR_ADDR]]
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_COND:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[CUR_ADDR]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[BEGIN]]
// LLVM:   br i1 %[[CMP]], label %[[LOOP_BODY]], label %[[LOOP_END:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[CUR_ADDR]]
// LLVM:   call void @_ZN9ArrayDtorD1Ev(ptr %[[CUR]]) #0
// LLVM:   %[[PREV:.*]] = getelementptr %struct.ArrayDtor, ptr %[[CUR]], i64 -1
// LLVM:   store ptr %[[PREV]], ptr %[[CUR_ADDR]]
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[LOOP_END]]:
// LLVM:   ret void
// LLVM: }
//
// LLVM: define internal void @__cxx_global_var_init() {
// LLVM:   call void @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr @arrDtor, ptr @__dso_handle)

// Note: OGCG defines these functions in reverse order of CIR->LLVM.
// Note also: OGCG doesn't pass the address of the array to the destructor function.
//            Instead, it uses the global directly in the helper function.

// OGCG: define internal void @__cxx_global_var_init() {{.*}} section ".text.startup" {
// OGCG:   call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr null, ptr @__dso_handle)

// OGCG: define internal void @__cxx_global_array_dtor(ptr noundef %[[ARG:.*]]) {{.*}} section ".text.startup" {
// OGCG: entry:
// OGCG:   %[[UNUSED_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG]], ptr %[[UNUSED_ADDR]]
// OGCG:   br label %[[LOOP_BODY:.*]]
// OGCG: [[LOOP_BODY]]:
// OGCG:   %[[PREV:.*]] = phi ptr [ getelementptr inbounds (%struct.ArrayDtor, ptr @arrDtor, i64 16), %entry ], [ %[[CUR:.*]], %[[LOOP_BODY]] ]
// OGCG:   %[[CUR]] = getelementptr inbounds %struct.ArrayDtor, ptr %[[PREV]], i64 -1
// OGCG:   call void @_ZN9ArrayDtorD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[CUR]])
// OGCG:   %[[DONE:.*]] = icmp eq ptr %[[CUR]], @arrDtor
// OGCG:   br i1 %[[DONE]], label %[[LOOP_END:.*]], label %[[LOOP_BODY]]
// OGCG: [[LOOP_END]]:
// OGCG:   ret void
// OGCG: }

// Common init function for all globals with default priority

// CIR: cir.func private @_GLOBAL__sub_I_[[FILENAME:.*]]() {
// CIR:   cir.call @__cxx_global_var_init() : () -> ()

// LLVM: define void @_GLOBAL__sub_I_[[FILENAME:.*]]()
// LLVM:   call void @__cxx_global_var_init()

// OGCG: define internal void @_GLOBAL__sub_I_[[FILENAME:.*]]() {{.*}} section ".text.startup" {
// OGCG:   call void @__cxx_global_var_init()
