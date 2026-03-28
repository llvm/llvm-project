// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir -check-prefix=CIR-BEFORE %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

void test_delete_array(int *ptr) {
  delete[] ptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z17test_delete_arrayPi
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR-BEFORE:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-BEFORE:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!s32i>
// CIR-BEFORE:   cir.if %[[NOT_NULL]] {
// CIR-BEFORE:     cir.delete_array %[[PTR]] : !cir.ptr<!s32i> {delete_fn = @_ZdaPv, delete_params = #cir.usual_delete_params<>
// CIR-BEFORE:   }

// CIR: cir.func {{.*}} @_Z17test_delete_arrayPi
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!s32i>
// CIR:   cir.if %[[NOT_NULL]] {
// CIR:     %[[VOID_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR:     cir.call @_ZdaPv(%[[VOID_PTR]])
// CIR:   }

// LLVM: define {{.*}} void @_Z17test_delete_arrayPi
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[PTR]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[DELETE_NOTNULL:.*]], label %[[DELETE_END:.*]]
// LLVM: [[DELETE_NOTNULL]]:
// LLVM:   call void @_ZdaPv(ptr %[[PTR]])
// LLVM:   br label %[[DELETE_END]]
// LLVM: [[DELETE_END]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z17test_delete_arrayPi
// OGCG:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[PTR]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[DELETE_END:.*]], label %[[DELETE_NOT_NULL:.*]]
// OGCG: [[DELETE_NOT_NULL]]:
// OGCG:   call void @_ZdaPv(ptr {{.*}} %[[PTR]])
// OGCG:   br label %[[DELETE_END]]
// OGCG: [[DELETE_END]]:
// OGCG:   ret void

struct SimpleArrDelete {
  void operator delete[](void *);
  int member;
};
void test_simple_delete_array(SimpleArrDelete *ptr) {
  delete[] ptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z24test_simple_delete_arrayP15SimpleArrDelete
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR-BEFORE:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_SimpleArrDelete>
// CIR-BEFORE:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!rec_SimpleArrDelete>
// CIR-BEFORE:   cir.if %[[NOT_NULL]] {
// CIR-BEFORE:     cir.delete_array %[[PTR]] : !cir.ptr<!rec_SimpleArrDelete> {delete_fn = @_ZN15SimpleArrDeletedaEPv, delete_params = #cir.usual_delete_params<>
// CIR-BEFORE:   }

// CIR: cir.func {{.*}} @_Z24test_simple_delete_arrayP15SimpleArrDelete
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_SimpleArrDelete>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!rec_SimpleArrDelete>
// CIR:   cir.if %[[NOT_NULL]] {
// CIR:     %[[VOID_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_SimpleArrDelete> -> !cir.ptr<!void>
// CIR:     cir.call @_ZN15SimpleArrDeletedaEPv(%[[VOID_PTR]])
// CIR:   }

// LLVM: define {{.*}} void @_Z24test_simple_delete_arrayP15SimpleArrDelete
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[PTR]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[DELETE_NOTNULL:.*]], label %[[DELETE_END:.*]]
// LLVM: [[DELETE_NOTNULL]]:
// LLVM:   call void @_ZN15SimpleArrDeletedaEPv(ptr %[[PTR]])
// LLVM:   br label %[[DELETE_END]]
// LLVM: [[DELETE_END]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z24test_simple_delete_arrayP15SimpleArrDelete
// OGCG:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[PTR]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[ARR_DELETE_END:.*]], label %[[ARR_DELETE_NOT_NULL:.*]]
// OGCG: [[ARR_DELETE_NOT_NULL]]:
// OGCG:   call void @_ZN15SimpleArrDeletedaEPv(ptr {{.*}} %[[PTR]])
// OGCG:   br label %[[ARR_DELETE_END]]
// OGCG: [[ARR_DELETE_END]]:

typedef __typeof(sizeof(int)) size_t;

struct SizedArrayDelete {
  void operator delete[](void *, size_t);
  int member;
};
void test_sized_array_delete(SizedArrayDelete *ptr) {
  delete[] ptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z23test_sized_array_deleteP16SizedArrayDelete
// CIR-BEFORE:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR-BEFORE:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_SizedArrayDelete>
// CIR-BEFORE:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!rec_SizedArrayDelete>
// CIR-BEFORE:   cir.if %[[NOT_NULL]] {
// CIR-BEFORE:     cir.delete_array %[[PTR]] : !cir.ptr<!rec_SizedArrayDelete> {delete_fn = @_ZN16SizedArrayDeletedaEPvm, delete_params = #cir.usual_delete_params<size = true>
// CIR-BEFORE:   }

// CIR: cir.func {{.*}} @_Z23test_sized_array_deleteP16SizedArrayDelete
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_SizedArrayDelete>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!rec_SizedArrayDelete>
// CIR:   cir.if %[[NOT_NULL]] {
// CIR:     %[[BYTE_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_SizedArrayDelete> -> !cir.ptr<!u8i>
// CIR:     %[[NEG_COOKIE:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:     %[[ALLOC_BYTE_PTR:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[NEG_COOKIE]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:     %[[VOID_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:     %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR:     %[[NUM_ELEM:.*]] = cir.load align(4) %[[COOKIE_PTR]] : !cir.ptr<!u64i>, !u64i
// CIR:     %[[ELEM_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CIR:     %[[ARRAY_SIZE:.*]] = cir.mul %[[ELEM_SIZE]], %[[NUM_ELEM]] : !u64i
// CIR:     %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:     %[[TOTAL_SIZE:.*]] = cir.add %[[ARRAY_SIZE]], %[[COOKIE_SIZE]] : !u64i
// CIR:     cir.call @_ZN16SizedArrayDeletedaEPvm(%[[VOID_PTR]], %[[TOTAL_SIZE]])
// CIR:   }

// LLVM: define {{.*}} void @_Z23test_sized_array_deleteP16SizedArrayDelete
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[PTR]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[DELETE_NOTNULL:.*]], label %[[DELETE_END:.*]]
// LLVM: [[DELETE_NOTNULL]]:
// LLVM:   %[[ALLOC_PTR:.*]] = getelementptr i8, ptr %[[PTR]], i64 -8
// LLVM:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]], align 4
// LLVM:   %[[ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// LLVM:   %[[TOTAL_SIZE:.*]] = add i64 %[[ARRAY_SIZE]], 8
// LLVM:   call void @_ZN16SizedArrayDeletedaEPvm(ptr %[[ALLOC_PTR]], i64 %[[TOTAL_SIZE]])
// LLVM:   br label %[[DELETE_END]]
// LLVM: [[DELETE_END]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z23test_sized_array_deleteP16SizedArrayDelete
// OGCG:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[PTR]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[DELETE_END:.*]], label %[[DELETE_NOT_NULL:.*]]
// OGCG: [[DELETE_NOT_NULL]]:
// OGCG:   %[[ALLOC_PTR:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i64 -8
// OGCG:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]], align 4
// OGCG:   %[[ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// OGCG:   %[[TOTAL_SIZE:.*]] = add i64 %[[ARRAY_SIZE]], 8
// OGCG:   call void @_ZN16SizedArrayDeletedaEPvm(ptr {{.*}} %[[ALLOC_PTR]], i64 {{.*}} %[[TOTAL_SIZE]])
// OGCG:   br label %[[DELETE_END]]
// OGCG: [[DELETE_END]]:

struct Destructed {
  ~Destructed();
  int x;
};
void test_delete_array_destructed(Destructed *ptr) {
  delete[] ptr;
}

// CIR-BEFORE: cir.func {{.*}} @_Z28test_delete_array_destructedP10Destructed
// CIR-BEFORE:   %[[PTR:.*]] = cir.load
// CIR-BEFORE:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_Destructed>
// CIR-BEFORE:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!rec_Destructed>
// CIR-BEFORE:   cir.if %[[NOT_NULL]] {
// CIR-BEFORE:     cir.delete_array %[[PTR]] : !cir.ptr<!rec_Destructed> {
// CIR-BEFORE-SAME:       delete_fn = @_ZdaPvm,
// CIR-BEFORE-SAME:       delete_params = #cir.usual_delete_params<size = true>,
// CIR-BEFORE-SAME:       element_dtor = @_ZN10DestructedD1Ev}
// CIR-BEFORE:   }

// CIR: cir.func {{.*}} @_Z28test_delete_array_destructedP10Destructed
// CIR:   %[[PTR:.*]] = cir.load
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_Destructed>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]] : !cir.ptr<!rec_Destructed>
// CIR:   cir.if %[[NOT_NULL]] {
//
// Read the array cookie.
// CIR:     %[[BYTE_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_Destructed> -> !cir.ptr<!u8i>
// CIR:     %[[NEG_COOKIE:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:     %[[ALLOC_BYTE_PTR:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[NEG_COOKIE]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:     %[[VOID_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:     %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR:     %[[NUM_ELEM:.*]] = cir.load{{.*}} %[[COOKIE_PTR]] : !cir.ptr<!u64i>, !u64i
//
// Destruct elements in reverse order.
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:     %[[NUM_ELEM_MINUS_ONE:.*]] = cir.sub %[[NUM_ELEM]], %[[ONE]] : !u64i
// CIR:     %[[END:.*]] = cir.ptr_stride %[[PTR]], %[[NUM_ELEM_MINUS_ONE]] : (!cir.ptr<!rec_Destructed>, !u64i) -> !cir.ptr<!rec_Destructed>
// CIR:     %[[NOT_EMPTY:.*]] = cir.cmp ne %[[END]], %[[PTR]] : !cir.ptr<!rec_Destructed>
// CIR:     cir.if %[[NOT_EMPTY]] {
// CIR:       %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_Destructed>, !cir.ptr<!cir.ptr<!rec_Destructed>>, ["__array_idx"] {alignment = 1 : i64}
// CIR:       cir.store %[[END]], %[[ARR_IDX]] : !cir.ptr<!rec_Destructed>, !cir.ptr<!cir.ptr<!rec_Destructed>>
// CIR:       cir.do {
// CIR:         %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_Destructed>>, !cir.ptr<!rec_Destructed>
// CIR:         cir.call @_ZN10DestructedD1Ev(%[[ARR_CUR]]) : (!cir.ptr<!rec_Destructed>) -> ()
// CIR:         %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:         %[[ARR_NEXT:.*]] = cir.ptr_stride %[[ARR_CUR]], %[[NEG_ONE]] : (!cir.ptr<!rec_Destructed>, !s64i) -> !cir.ptr<!rec_Destructed>
// CIR:         cir.store %[[ARR_NEXT]], %[[ARR_IDX]] : !cir.ptr<!rec_Destructed>, !cir.ptr<!cir.ptr<!rec_Destructed>>
// CIR:         cir.yield
// CIR:       } while {
// CIR:         %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_Destructed>>, !cir.ptr<!rec_Destructed>
// CIR:         %[[CMP:.*]] = cir.cmp ne %[[ARR_CUR]], %[[PTR]] : !cir.ptr<!rec_Destructed>
// CIR:         cir.condition(%[[CMP]])
// CIR:       }
// CIR:     }
//
// Compute total size and call delete function.
// CIR:     %[[ELEM_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CIR:     %[[ARRAY_SIZE:.*]] = cir.mul %[[ELEM_SIZE]], %[[NUM_ELEM]] : !u64i
// CIR:     %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:     %[[TOTAL_SIZE:.*]] = cir.add %[[ARRAY_SIZE]], %[[COOKIE_SIZE]] : !u64i
// CIR:     cir.call @_ZdaPvm(%[[VOID_PTR]], %[[TOTAL_SIZE]])
// CIR:   }

// LLVM: define {{.*}} void @_Z28test_delete_array_destructedP10Destructed
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[PTR]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[DELETE_NOTNULL:.*]], label %[[DONE:.*]]
// LLVM: [[DELETE_NOTNULL]]:
// LLVM:   %[[ALLOC_PTR:.*]] = getelementptr i8, ptr %[[PTR]], i64 -8
// LLVM:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]], align 4
// LLVM:   %[[NUM_ELEM_MINUS_ONE:.*]] = sub i64 %[[NUM_ELEM]], 1
// LLVM:   %[[ARR_END:.*]] = getelementptr %struct.Destructed, ptr %[[PTR]], i64 %[[NUM_ELEM_MINUS_ONE]]
// LLVM:   %[[NOT_EMPTY:.*]] = icmp ne ptr %[[ARR_END]], %[[PTR]]
// LLVM:   br i1 %[[NOT_EMPTY]], label %[[DESTROY_ELEMENTS:.*]], label %[[CALL_DELETE:.*]]
// LLVM: [[DESTROY_ELEMENTS:.*]]:
// LLVM:   store ptr %[[ARR_END]], ptr %[[ARR_IDX:.*]]
// LLVM:   br label %[[DELETE_ELEMENT:.*]]
// LLVM: [[LOOP_CONDITION:.*]]
// LLVM:   %[[ARR_CUR:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[ARR_CUR]], %[[PTR]]
// LLVM:   br i1 %[[CMP]], label %[[DELETE_ELEMENT:.*]], label %[[LOOP_END:.*]]
// LLVM: [[DELETE_ELEMENT]]:
// LLVM:   %[[ELEM:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   call void @_ZN10DestructedD1Ev(ptr %[[ELEM]])
// LLVM:   %[[NEXT:.*]] = getelementptr %struct.Destructed, ptr %[[ELEM]], i64 -1
// LLVM:   store ptr %[[NEXT]], ptr %[[ARR_IDX]]
// LLVM:   br label %[[LOOP_CONDITION]]
// LLVM: [[LOOP_END]]:
// LLVM:   br label %[[CALL_DELETE]]
// LLVM: [[CALL_DELETE]]:
// LLVM:   %[[ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// LLVM:   %[[TOTAL_SIZE:.*]] = add i64 %[[ARRAY_SIZE]], 8
// LLVM:   call void @_ZdaPvm(ptr %[[ALLOC_PTR]], i64 %[[TOTAL_SIZE]])
// LLVM:   br label %[[DONE]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z28test_delete_array_destructedP10Destructed
// OGCG:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[PTR]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[DELETE_END:.*]], label %[[DELETE_NOT_NULL:.*]]
// OGCG: [[DELETE_NOT_NULL]]:
// OGCG:   %[[ALLOC_PTR:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i64 -8
// OGCG:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]], align 4
// OGCG:   %[[ARR_END:.*]] = getelementptr inbounds %struct.Destructed, ptr %[[PTR]], i64 %[[NUM_ELEM]]
// OGCG:   %[[ARR_IS_EMPTY:.*]] = icmp eq ptr %[[PTR]], %[[ARR_END]]
// OGCG:   br i1 %[[ARR_IS_EMPTY]], label %[[ARRAY_DESTROY_DONE1:.*]], label %[[ARRAY_DESTROY_BODY:.*]]
// OGCG: [[ARRAY_DESTROY_BODY]]:
// OGCG:   %[[ARRAY_DESTROY_ELEMENT_PAST:.*]] = phi ptr [ %[[ARR_END]], %[[DELETE_NOT_NULL]] ], [ %[[ARRAY_DESTROY_ELEMENT:.*]], %[[ARRAY_DESTROY_BODY]] ]
// OGCG:   %[[ARRAY_DESTROY_ELEMENT]] = getelementptr inbounds %struct.Destructed, ptr %[[ARRAY_DESTROY_ELEMENT_PAST]], i64 -1
// OGCG:   call void @_ZN10DestructedD1Ev(ptr {{.*}} %[[ARRAY_DESTROY_ELEMENT]])
// OGCG:   %[[ARRAY_DESTROY_DONE:.*]] = icmp eq ptr %[[ARRAY_DESTROY_ELEMENT]], %[[PTR]]
// OGCG:   br i1 %[[ARRAY_DESTROY_DONE]], label %[[ARRAY_DESTROY_DONE1:.*]], label %[[ARRAY_DESTROY_BODY]]
// OGCG: [[ARRAY_DESTROY_DONE1]]:
// OGCG:   %[[ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// OGCG:   %[[TOTAL_SIZE:.*]] = add i64 %[[ARRAY_SIZE]], 8
// OGCG:   call void @_ZdaPvm(ptr {{.*}} %[[ALLOC_PTR]], i64 {{.*}} %[[TOTAL_SIZE]])
// OGCG:   br label %[[DELETE_END]]
// OGCG: [[DELETE_END]]:
// OGCG:   ret void
