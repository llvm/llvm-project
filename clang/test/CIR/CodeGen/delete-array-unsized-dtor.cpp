// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++14 -fno-sized-deallocation -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir -check-prefix=CIR-BEFORE %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++14 -fno-sized-deallocation -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++14 -fno-sized-deallocation -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

// When operator delete[] is unsized but the element type has a non-trivial
// destructor, a cookie is still needed to store the element count for the
// destructor loop.

struct Dtor {
  ~Dtor();
  int x;
};

void test(Dtor *ptr) {
  delete[] ptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z4testP4Dtor
// CIR-BEFORE:   cir.delete_array %{{.*}} : !cir.ptr<!rec_Dtor> {
// CIR-BEFORE-SAME: delete_fn = @_ZdaPv,
// CIR-BEFORE-SAME: delete_params = #cir.usual_delete_params<>,
// CIR-BEFORE-SAME: element_dtor = @_ZN4DtorD1Ev}

// CIR: cir.func {{.*}} @_Z4testP4Dtor
// CIR:   %[[PTR:.*]] = cir.load
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne
// CIR:   cir.if %[[NOT_NULL]] {
//
// Read the array cookie.
// CIR:     %[[BYTE_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_Dtor> -> !cir.ptr<!u8i>
// CIR:     %[[NEG_COOKIE:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:     %[[ALLOC_PTR:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[NEG_COOKIE]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:     %[[VOID_PTR:.*]] = cir.cast bitcast %[[ALLOC_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:     %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[ALLOC_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR:     %[[NUM_ELEM:.*]] = cir.load{{.*}} %[[COOKIE_PTR]] : !cir.ptr<!u64i>, !u64i
//
// Compute end pointer and check for empty array.
// CIR:     %[[END:.*]] = cir.ptr_stride %[[PTR]], %[[NUM_ELEM]] : (!cir.ptr<!rec_Dtor>, !u64i) -> !cir.ptr<!rec_Dtor>
// CIR:     %[[NOT_EMPTY:.*]] = cir.cmp ne %[[END]], %[[PTR]] : !cir.ptr<!rec_Dtor>
// CIR:     cir.if %[[NOT_EMPTY]] {
//
// Destruct elements in reverse order.
// CIR:       %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_Dtor>, !cir.ptr<!cir.ptr<!rec_Dtor>>, ["__array_idx"]
// CIR:       cir.store %[[END]], %[[ARR_IDX]]
// CIR:       cir.do {
// CIR:         %[[CUR:.*]] = cir.load %[[ARR_IDX]]
// CIR:         %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:         %[[PREV:.*]] = cir.ptr_stride %[[CUR]], %[[NEG_ONE]] : (!cir.ptr<!rec_Dtor>, !s64i) -> !cir.ptr<!rec_Dtor>
// CIR:         cir.store %[[PREV]], %[[ARR_IDX]]
// CIR:         cir.call @_ZN4DtorD1Ev(%[[PREV]]) nothrow : (!cir.ptr<!rec_Dtor>) -> ()
// CIR:         cir.yield
// CIR:       } while {
// CIR:         %[[CUR2:.*]] = cir.load %[[ARR_IDX]]
// CIR:         %[[DONE:.*]] = cir.cmp ne %[[CUR2]], %[[PTR]] : !cir.ptr<!rec_Dtor>
// CIR:         cir.condition(%[[DONE]])
// CIR:       }
// CIR:     }
//
// Call unsized operator delete[] with just the pointer.
// CIR:     cir.call @_ZdaPv(%[[VOID_PTR]]) nothrow : (!cir.ptr<!void>) -> ()
// CIR:   }

// LLVM: define {{.*}} void @_Z4testP4Dtor
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[PTR]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[DELETE_NOTNULL:.*]], label %[[DONE:.*]]
// LLVM: [[DELETE_NOTNULL]]:
// LLVM:   %[[ALLOC_PTR:.*]] = getelementptr i8, ptr %[[PTR]], i64 -8
// LLVM:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]], align 4
// LLVM:   %[[ARR_END:.*]] = getelementptr %struct.Dtor, ptr %[[PTR]], i64 %[[NUM_ELEM]]
// LLVM:   %[[NOT_EMPTY:.*]] = icmp ne ptr %[[ARR_END]], %[[PTR]]
// LLVM:   br i1 %[[NOT_EMPTY]], label %[[DESTROY:.*]], label %[[CALL_DELETE:.*]]
// LLVM: [[DESTROY]]:
// LLVM:   store ptr %[[ARR_END]], ptr %[[ARR_IDX:.*]]
// LLVM:   br label %[[DTOR_LOOP:.*]]
// LLVM: [[LOOP_COND:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[PTR]]
// LLVM:   br i1 %[[CMP]], label %[[DTOR_LOOP]], label %[[LOOP_END:.*]]
// LLVM: [[DTOR_LOOP]]:
// LLVM:   %[[ELEM:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[PREV:.*]] = getelementptr %struct.Dtor, ptr %[[ELEM]], i64 -1
// LLVM:   store ptr %[[PREV]], ptr %[[ARR_IDX]]
// LLVM:   call void @_ZN4DtorD1Ev(ptr %[[PREV]])
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[LOOP_END]]:
// LLVM:   br label %[[CALL_DELETE]]
// LLVM: [[CALL_DELETE]]:
// LLVM:   call void @_ZdaPv(ptr %[[ALLOC_PTR]])
// LLVM:   br label %[[DONE]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z4testP4Dtor
// OGCG:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[PTR]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[DELETE_END:.*]], label %[[DELETE_NOT_NULL:.*]]
// OGCG: [[DELETE_NOT_NULL]]:
// OGCG:   %[[ALLOC_PTR:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i64 -8
// OGCG:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]], align 4
// OGCG:   %[[ARR_END:.*]] = getelementptr inbounds %struct.Dtor, ptr %[[PTR]], i64 %[[NUM_ELEM]]
// OGCG:   %[[ARR_EMPTY:.*]] = icmp eq ptr %[[PTR]], %[[ARR_END]]
// OGCG:   br i1 %[[ARR_EMPTY]], label %[[DESTROY_DONE:.*]], label %[[DESTROY_BODY:.*]]
// OGCG: [[DESTROY_BODY]]:
// OGCG:   %[[DESTROY_PAST:.*]] = phi ptr [ %[[ARR_END]], %[[DELETE_NOT_NULL]] ], [ %[[DESTROY_ELEM:.*]], %[[DESTROY_BODY]] ]
// OGCG:   %[[DESTROY_ELEM]] = getelementptr inbounds %struct.Dtor, ptr %[[DESTROY_PAST]], i64 -1
// OGCG:   call void @_ZN4DtorD1Ev(ptr {{.*}} %[[DESTROY_ELEM]])
// OGCG:   %[[DESTROY_CMP:.*]] = icmp eq ptr %[[DESTROY_ELEM]], %[[PTR]]
// OGCG:   br i1 %[[DESTROY_CMP]], label %[[DESTROY_DONE]], label %[[DESTROY_BODY]]
// OGCG: [[DESTROY_DONE]]:
// OGCG:   call void @_ZdaPv(ptr {{.*}} %[[ALLOC_PTR]])
// OGCG:   br label %[[DELETE_END]]
// OGCG: [[DELETE_END]]:
// OGCG:   ret void
