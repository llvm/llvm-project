// RUN: %clang_cc1 -triple spir64 -cl-std=CL2.0 -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64 -cl-std=CL2.0 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spir64 -cl-std=CL2.0 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Calls to non-kernel functions on SPIR use the spir_func calling convention.

int helper(global int *p) { return *p + 1; }
void store(global int *p, int v) { *p = v; }

kernel void k(global int *p) { store(p, helper(p)); }

// CIR: cir.func {{.*}}@helper({{.*}}) -> !s32i cc(spir_function)
// CIR: cir.func {{.*}}@store({{.*}}) cc(spir_function)
// CIR: cir.func {{.*}}@k({{.*}}) cc(spir_kernel)
// CIR:   %[[R:.*]] = cir.call @helper(%{{.*}}) cc(spir_function)
// CIR:   cir.call @store(%{{.*}}, %[[R]]) cc(spir_function)

// LLVM: define dso_local spir_func i32 @helper(
// LLVM: define dso_local spir_func void @store(
// LLVM: define dso_local spir_kernel void @k(

// TODO(cir): CIR does not yet emit the __clang_ocl_kern_imp_ kernel stub, so
// classic CodeGen calls helper/store from there.
// OGCG:   call spir_func void @__clang_ocl_kern_imp_k(
// OGCG: define dso_local spir_func void @__clang_ocl_kern_imp_k(

// LLVM:   %[[R:.*]] = call spir_func i32 @helper(ptr addrspace(1) noundef %{{.*}})
// LLVM:   call spir_func void @store(ptr addrspace(1) noundef %{{.*}}, i32 noundef %[[R]])
