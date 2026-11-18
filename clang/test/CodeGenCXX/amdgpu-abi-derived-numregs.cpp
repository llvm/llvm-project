// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -emit-llvm -o - %s | FileCheck %s --check-prefix=AMDGPU
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck %s --check-prefix=SPIRV

// Base class registers count towards the 16 register limit for direct
// passing, so a derived class behaves like the equivalent flat struct.

struct B { int big[64]; };
struct D : B { int x[3]; };
struct F { int big[64]; int x[3]; };

struct SB { int a[4]; };
struct SD : SB { int x[3]; };

// AMDGPU-LABEL: define {{.*}}void @_Z3retv(ptr addrspace(5) {{.*}}sret(%struct.D)
// SPIRV-LABEL: define {{.*}}void @_Z3retv(ptr {{.*}}sret(%struct.D)
D ret() { return D(); }

// AMDGPU-LABEL: define {{.*}}void @_Z8ret_flatv(ptr addrspace(5) {{.*}}sret(%struct.F)
// SPIRV-LABEL: define {{.*}}void @_Z8ret_flatv(ptr {{.*}}sret(%struct.F)
F ret_flat() { return F(); }

// AMDGPU-LABEL: define {{.*}}void @_Z3arg1DPS_(ptr addrspace(5) {{.*}}byref(%struct.D)
// SPIRV-LABEL: define {{.*}}void @_Z3arg1DPS_(ptr {{.*}}byref(%struct.D)
void arg(D d, D *p) { *p = d; }

// AMDGPU-LABEL: define {{.*}}void @_Z8arg_flat1FPS_(ptr addrspace(5) {{.*}}byref(%struct.F)
// SPIRV-LABEL: define {{.*}}void @_Z8arg_flat1FPS_(ptr {{.*}}byref(%struct.F)
void arg_flat(F f, F *p) { *p = f; }

// AMDGPU-LABEL: define {{.*}}%struct.SD @_Z9small_retv(
// SPIRV-LABEL: define {{.*}}%struct.SD @_Z9small_retv(
SD small_ret() { return SD(); }
