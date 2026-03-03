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
// CIR-BEFORE:   cir.delete_array %[[PTR]] : !cir.ptr<!s32i> {
// CIR-BEFORE:     %[[VOID_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-BEFORE:     cir.call @_ZdaPv(%[[VOID_PTR]]) {{.*}} : (!cir.ptr<!void>{{.*}}) -> ()
// CIR-BEFORE:     cir.yield
// CIR-BEFORE:   }

// CIR: cir.func {{.*}} @_Z17test_delete_arrayPi
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[VOID_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR:   cir.call @_ZdaPv(%[[VOID_PTR]])

// LLVM: define {{.*}} void @_Z17test_delete_arrayPi
// LLVM:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// LLVM:   call void @_ZdaPv(ptr {{.*}} %[[PTR]])

// OGCG: define {{.*}} void @_Z17test_delete_arrayPi
// OGCG:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[PTR]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[DELETE_END:.*]], label %[[DELETE_NOT_NULL:.*]]
// OGCG: [[DELETE_NOT_NULL]]:
// OGCG:   call void @_ZdaPv(ptr {{.*}} %[[PTR]])
// OGCG:   br label %[[DELETE_END]]
// OGCG: [[DELETE_END]]:
// OGCG:   ret void
