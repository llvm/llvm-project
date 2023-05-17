// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck --check-prefixes=COMMON,CHECK %s

// Derived from CodeGenCUDA/amdgpu-kernel-arg-pointer-type.cu by deleting references to HOST
// The original test passes the result through opt O2, but that seems to introduce invalid
// addrspace casts which are not being fixed as part of the present change.

// COMMON-LABEL: define{{.*}} amdgpu_kernel void @_Z7kernel1Pi(ptr {{.*}} %x)
// CHECK-NOT: ={{.*}} addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to ptr
__attribute__((amdgpu_kernel)) void kernel1(int *x) {
  x[0]++;
}

// COMMON-LABEL: define{{.*}} amdgpu_kernel void @_Z7kernel2Ri(ptr {{.*}} nonnull align 4 dereferenceable(4) %x)
// CHECK-NOT: ={{.*}} addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to ptr
__attribute__((amdgpu_kernel)) void kernel2(int &x) {
  x++;
}

// CHECK-LABEL: define{{.*}} amdgpu_kernel void  @_Z7kernel3PU3AS2iPU3AS1i(ptr addrspace(2){{.*}} %x, ptr addrspace(1){{.*}} %y)
// CHECK-NOT: ={{.*}} addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to ptr
__attribute__((amdgpu_kernel)) void kernel3(__attribute__((address_space(2))) int *x,
                                            __attribute__((address_space(1))) int *y) {
  y[0] = x[0];
}

// COMMON-LABEL: define{{.*}} void @_Z4funcPi(ptr{{.*}} %x)
// CHECK-NOT: ={{.*}} addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to ptr
__attribute__((amdgpu_kernel)) void func(int *x) {
  x[0]++;
}

struct S {
  int *x;
  float *y;
};
// `by-val` struct is passed by-indirect-alias (a mix of by-ref and indirect
// by-val). However, the enhanced address inferring pass should be able to
// assume they are global pointers.
//

// COMMON-LABEL: define{{.*}} amdgpu_kernel void @_Z7kernel41S(ptr addrspace(4){{.*}} byref(%struct.S) align 8 %0)
__attribute__((amdgpu_kernel)) void kernel4(struct S s) {
  s.x[0]++;
  s.y[0] += 1.f;
}

// COMMON-LABEL: define{{.*}} amdgpu_kernel void @_Z7kernel5P1S(ptr {{.*}} %s)
__attribute__((amdgpu_kernel)) void kernel5(struct S *s) {
  s->x[0]++;
  s->y[0] += 1.f;
}

struct T {
  float *x[2];
};
// `by-val` array is passed by-indirect-alias (a mix of by-ref and indirect
// by-val). However, the enhanced address inferring pass should be able to
// assume they are global pointers.
//
// COMMON-LABEL: define{{.*}} amdgpu_kernel void @_Z7kernel61T(ptr addrspace(4){{.*}} byref(%struct.T) align 8 %0)
__attribute__((amdgpu_kernel)) void kernel6(struct T t) {
  t.x[0][0] += 1.f;
  t.x[1][0] += 2.f;
}

// Check that coerced pointers retain the noalias attribute when qualified with __restrict.
// COMMON-LABEL: define{{.*}} amdgpu_kernel void @_Z7kernel7Pi(ptr noalias{{.*}} %x)
__attribute__((amdgpu_kernel)) void kernel7(int *__restrict x) {
  x[0]++;
}

// Single element struct.
struct SS {
  float *x;
};
// COMMON-LABEL: define{{.*}} amdgpu_kernel void @_Z7kernel82SS(ptr %a.coerce)
// CHECK-NOT: ={{.*}} addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to ptr
__attribute__((amdgpu_kernel)) void kernel8(struct SS a) {
  *a.x += 3.f;
}

