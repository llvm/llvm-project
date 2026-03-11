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
