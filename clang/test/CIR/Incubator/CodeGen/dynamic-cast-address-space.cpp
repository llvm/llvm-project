// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OGCG

struct Base {
  virtual ~Base();
};

struct Derived : Base {};

// Test dynamic_cast to void* with address space attribute.
// The result pointer should preserve the address space of the source pointer.

// CIR-BEFORE: cir.func {{.*}} @_Z30ptr_cast_to_complete_addrspaceP4Base
// CIR-BEFORE:   %{{.+}} = cir.dyn_cast ptr %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!void>
// CIR-BEFORE: }

// CIR: cir.func {{.*}} @_Z30ptr_cast_to_complete_addrspaceP4Base
// CIR:   %[[#SRC:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base>>, !cir.ptr<!rec_Base>
// CIR:   %[[#SRC_IS_NOT_NULL:]] = cir.cast ptr_to_bool %[[#SRC]] : !cir.ptr<!rec_Base> -> !cir.bool
// CIR:   %{{.+}} = cir.ternary(%[[#SRC_IS_NOT_NULL]], true {
// CIR:     %[[#SRC_BYTES_PTR:]] = cir.cast bitcast %{{.+}} : !cir.ptr<!rec_Base> -> !cir.ptr<!u8i>
// CIR:     %[[#DST_BYTES_PTR:]] = cir.ptr_stride %[[#SRC_BYTES_PTR]], %{{.+}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:     %[[#CASTED_PTR:]] = cir.cast bitcast %[[#DST_BYTES_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:     cir.yield %[[#CASTED_PTR]] : !cir.ptr<!void>
// CIR:   }, false {
// CIR:     %[[#NULL_PTR:]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:     cir.yield %[[#NULL_PTR]] : !cir.ptr<!void>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!void>
// CIR: }

// LLVM: define dso_local ptr @_Z30ptr_cast_to_complete_addrspaceP4Base
// LLVM-SAME: (ptr %{{.+}})
// LLVM-DAG: alloca ptr, {{.*}}addrspace(5)
// LLVM-DAG: %[[#RETVAL_ALLOCA:]] = alloca ptr, {{.*}}addrspace(5)
// LLVM-DAG: %[[#RETVAL_ASCAST:]] = addrspacecast ptr addrspace(5) %[[#RETVAL_ALLOCA]] to ptr
// LLVM-DAG: %[[#PTR_ASCAST:]] = addrspacecast ptr addrspace(5) %{{.+}} to ptr
// LLVM:   store ptr %{{.+}}, ptr %[[#PTR_ASCAST]], align 8
// LLVM:   %[[#SRC:]] = load ptr, ptr %[[#PTR_ASCAST]], align 8
// LLVM:   %[[#SRC_IS_NOT_NULL:]] = icmp ne ptr %[[#SRC]], null
// LLVM:   br i1 %[[#SRC_IS_NOT_NULL]], label %[[#TRUE_BLOCK:]], label %[[#FALSE_BLOCK:]]
// LLVM: [[#TRUE_BLOCK]]:
// LLVM:   %[[#VTABLE:]] = load ptr, ptr %[[#SRC]], align 8
// LLVM:   %[[#OFFSET_PTR:]] = getelementptr i64, ptr %[[#VTABLE]], i64 -2
// LLVM:   %[[#OFFSET:]] = load i64, ptr %[[#OFFSET_PTR]], align 8
// LLVM:   %[[#RESULT:]] = getelementptr i8, ptr %[[#SRC]], i64 %[[#OFFSET]]
// LLVM:   br label %[[#MERGE:]]
// LLVM: [[#FALSE_BLOCK]]:
// LLVM:   br label %[[#MERGE]]
// LLVM: [[#MERGE]]:
// LLVM:   %[[#PHI:]] = phi ptr [ null, %[[#FALSE_BLOCK]] ], [ %[[#RESULT]], %[[#TRUE_BLOCK]] ]
// LLVM:   store ptr %[[#PHI]], ptr %[[#RETVAL_ASCAST]], align 8
// LLVM:   %[[#RET:]] = load ptr, ptr %[[#RETVAL_ASCAST]], align 8
// LLVM:   ret ptr %[[#RET]]
// LLVM: }

// OGCG: define dso_local noundef ptr @_Z30ptr_cast_to_complete_addrspaceP4Base
// OGCG-SAME: (ptr noundef %{{.+}})
// OGCG:   %[[RETVAL_ASCAST:[a-z0-9.]+]] = addrspacecast ptr addrspace(5) %{{.+}} to ptr
// OGCG:   %[[PTR_ASCAST:[a-z0-9.]+]] = addrspacecast ptr addrspace(5) %{{.+}} to ptr
// OGCG:   store ptr %{{.+}}, ptr %[[PTR_ASCAST]], align 8
// OGCG:   %[[SRC:[0-9]+]] = load ptr, ptr %[[PTR_ASCAST]], align 8
// OGCG:   icmp eq ptr %[[SRC]], null
// OGCG: dynamic_cast.notnull:
// OGCG:   %[[VTABLE:[a-z0-9]+]] = load ptr, ptr %[[SRC]], align 8
// OGCG:   getelementptr inbounds i64, ptr %[[VTABLE]], i64 -2
// OGCG:   %[[OFFSET:[a-z0-9.]+]] = load i64, ptr %{{.+}}, align 8
// OGCG:   %[[RESULT:[0-9]+]] = getelementptr inbounds i8, ptr %[[SRC]], i64 %[[OFFSET]]
// OGCG: dynamic_cast.end:
// OGCG:   %[[PHI:[0-9]+]] = phi ptr [ %[[RESULT]], %dynamic_cast.notnull ], [ null, %dynamic_cast.null ]
// OGCG:   ret ptr %[[PHI]]
// OGCG: }
void *ptr_cast_to_complete_addrspace(Base *ptr) {
  return dynamic_cast<void *>(ptr);
}