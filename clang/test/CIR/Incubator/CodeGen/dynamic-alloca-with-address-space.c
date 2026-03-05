// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -DOCL12 -x cl -std=cl1.2 \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR-CL12
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x cl -std=cl2.0 \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR-CL20
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -DOCL12 -x cl -std=cl1.2 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG-CL12
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x cl -std=cl2.0 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG-CL20


#if defined(OCL12)
#define CAST (char *)(unsigned long)
#else
#define CAST (char *)
#endif

void allocas(unsigned long n) {
    char *a = CAST __builtin_alloca(n);
    char *uninitialized_a = CAST __builtin_alloca_uninitialized(n);
}

// CIR-LABEL: cir.func {{.*}} @allocas
// CIR:         %[[ALLOCA1:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, {{.*}} ["bi_alloca"]
// CIR:         cir.cast address_space %[[ALLOCA1]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:         %[[ALLOCA2:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, {{.*}} ["bi_alloca"]
// CIR:         cir.cast address_space %[[ALLOCA2]] : !cir.ptr<!u8i> -> !cir.ptr<!void>

// LLVM-LABEL: define {{.*}} void @allocas(i64 %{{.*}})
// LLVM:         %[[BI_ALLOCA1:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// LLVM:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA1]] to ptr
// LLVM:         %[[BI_ALLOCA2:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// LLVM:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA2]] to ptr

// OGCG-LABEL: define {{.*}} void @allocas(i64 {{.*}} %n)
// OGCG:         %[[BI_ALLOCA1:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// OGCG:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA1]] to ptr
// OGCG:         %[[BI_ALLOCA2:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// OGCG:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA2]] to ptr

// CIR-CL12-NOT: addrspacecast
// CIR-CL20-NOT: addrspacecast
// OGCG-CL12-NOT: addrspacecast
// OGCG-CL20-NOT: addrspacecast