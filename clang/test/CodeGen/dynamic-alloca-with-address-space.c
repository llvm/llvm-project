// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -DOCL12 -x cl -std=cl1.2 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-CL12
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x cl -std=cl2.0 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-CL20

#if defined(OCL12)
    #define CAST (char *)(unsigned long)
#else
    #define CAST (char *)
#endif

void allocas(unsigned long n) {
    char *a = CAST __builtin_alloca(n);
    char *uninitialized_a = CAST __builtin_alloca_uninitialized(n);
    char *aligned_a = CAST __builtin_alloca_with_align(n, 8);
    char *aligned_uninitialized_a = CAST __builtin_alloca_with_align_uninitialized(n, 8);
}

// CHECK: @allocas(
// CHECK: store i64 %n, ptr %n.addr.ascast, align 8
// CHECK: %0 = load i64, ptr %n.addr.ascast, align 8
// CHECK: %1 = alloca i8, i64 %0, align 8, addrspace(5)
// CHECK: %2 = addrspacecast ptr addrspace(5) %1 to ptr
// CHECK: store ptr %2, ptr %a.ascast, align 8
// CHECK: %3 = load i64, ptr %n.addr.ascast, align 8
// CHECK: %4 = alloca i8, i64 %3, align 8, addrspace(5)
// CHECK: %5 = addrspacecast ptr addrspace(5) %4 to ptr
// CHECK: store ptr %5, ptr %uninitialized_a.ascast, align 8
// CHECK: %6 = load i64, ptr %n.addr.ascast, align 8
// CHECK: %7 = alloca i8, i64 %6, align 1, addrspace(5)
// CHECK: %8 = addrspacecast ptr addrspace(5) %7 to ptr
// CHECK: store ptr %8, ptr %aligned_a.ascast, align 8
// CHECK: %9 = load i64, ptr %n.addr.ascast, align 8
// CHECK: %10 = alloca i8, i64 %9, align 1, addrspace(5)
// CHECK: %11 = addrspacecast ptr addrspace(5) %10 to ptr
// CHECK: store ptr %11, ptr %aligned_uninitialized_a.ascast, align 8
// CHECK: ret void
// CHECK-CL12-NOT: addrspacecast
// CHECK-CL20-NOT: addrspacecast
