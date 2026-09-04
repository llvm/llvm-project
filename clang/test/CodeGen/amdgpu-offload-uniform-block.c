// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,UNIFORM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fno-offload-uniform-block \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,REMAINDER

#ifdef __AMDGPU__
int foo(void) { return __builtin_amdgcn_workgroup_size_x(); }
#else
int foo(void) { return 0; }
#endif

// CHECK-LABEL: define{{.*}} i32 @foo(
// UNIFORM: getelementptr inbounds i8, ptr addrspace(4) {{.*}}, i64 12
// UNIFORM-NOT: select i1
// UNIFORM: "uniform-work-group-size"
// REMAINDER: select i1 {{.*}}, i32 12, i32 18
// REMAINDER-NOT: "uniform-work-group-size"
