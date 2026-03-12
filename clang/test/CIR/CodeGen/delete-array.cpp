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
// CIR-BEFORE:   cir.delete_array %[[PTR]] : !cir.ptr<!s32i> {delete_fn = @_ZdaPv, delete_params = #cir.usual_delete_params<>

// CIR: cir.func {{.*}} @_Z17test_delete_arrayPi
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[VOID_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR:   cir.call @_ZdaPv(%[[VOID_PTR]])

// LLVM: define {{.*}} void @_Z17test_delete_arrayPi
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   call void @_ZdaPv(ptr %[[PTR]])

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
// CIR-BEFORE:   cir.delete_array %[[PTR]] : !cir.ptr<!rec_SimpleArrDelete> {delete_fn = @_ZN15SimpleArrDeletedaEPv, delete_params = #cir.usual_delete_params<>

// CIR: cir.func {{.*}} @_Z24test_simple_delete_arrayP15SimpleArrDelete
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[VOID_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_SimpleArrDelete> -> !cir.ptr<!void>
// CIR:   cir.call @_ZN15SimpleArrDeletedaEPv(%[[VOID_PTR]])

// LLVM: define {{.*}} void @_Z24test_simple_delete_arrayP15SimpleArrDelete
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   call void @_ZN15SimpleArrDeletedaEPv(ptr %[[PTR]])

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
// CIR-BEFORE:   cir.delete_array %[[PTR]] : !cir.ptr<!rec_SizedArrayDelete> {delete_fn = @_ZN16SizedArrayDeletedaEPvm, delete_params = #cir.usual_delete_params<size = true>

// CIR: cir.func {{.*}} @_Z23test_sized_array_deleteP16SizedArrayDelete
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[BYTE_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_SizedArrayDelete> -> !cir.ptr<!u8i>
// CIR:   %[[NEG_COOKIE:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[ALLOC_BYTE_PTR:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[NEG_COOKIE]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[VOID_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:   %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR:   %[[NUM_ELEM:.*]] = cir.load align(4) %[[COOKIE_PTR]] : !cir.ptr<!u64i>, !u64i
// CIR:   %[[ELEM_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CIR:   %[[ARRAY_SIZE:.*]] = cir.mul %[[ELEM_SIZE]], %[[NUM_ELEM]] : !u64i
// CIR:   %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:   %[[TOTAL_SIZE:.*]] = cir.add %[[ARRAY_SIZE]], %[[COOKIE_SIZE]] : !u64i
// CIR:   cir.call @_ZN16SizedArrayDeletedaEPvm(%[[VOID_PTR]], %[[TOTAL_SIZE]])

// LLVM: define {{.*}} void @_Z23test_sized_array_deleteP16SizedArrayDelete
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[ALLOC_PTR:.*]] = getelementptr i8, ptr %[[PTR]], i64 -8
// LLVM:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]], align 4
// LLVM:   %[[ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// LLVM:   %[[TOTAL_SIZE:.*]] = add i64 %[[ARRAY_SIZE]], 8
// LLVM:   call void @_ZN16SizedArrayDeletedaEPvm(ptr %[[ALLOC_PTR]], i64 %[[TOTAL_SIZE]])

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
