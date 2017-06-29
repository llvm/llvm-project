// RUN: %clang_cc1 %s -emit-llvm -o - -O0 -finclude-default-header -ffake-address-space-map -triple i686-pc-darwin | FileCheck -check-prefixes=COM,X86 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -O0 -finclude-default-header -triple amdgcn-amdhsa-amd-amdgizcl | FileCheck -check-prefixes=COM,AMD %s

typedef struct {
  int cells[9];
} Mat3X3;

typedef struct {
  int cells[16];
} Mat4X4;

struct StructOneMember {
  int2 x;
};

struct StructTwoMember {
  int2 x;
  int2 y;
};

// COM-LABEL: define void @foo
Mat4X4 __attribute__((noinline)) foo(Mat3X3 in) {
  Mat4X4 out;
  return out;
}

// COM-LABEL: define {{.*}} void @ker
// Expect two mem copies: one for the argument "in", and one for
// the return value.
// X86: call void @llvm.memcpy.p0i8.p1i8.i32(i8*
// X86: call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)*
// AMD: call void @llvm.memcpy.p5i8.p1i8.i64(i8 addrspace(5)*
// AMD: call void @llvm.memcpy.p1i8.p5i8.i64(i8 addrspace(1)*
kernel void ker(global Mat3X3 *in, global Mat4X4 *out) {
  out[0] = foo(in[1]);
}

// AMD-LABEL: define void @FuncOneMember(%struct.StructOneMember addrspace(5)* byval align 8 %u)
void FuncOneMember(struct StructOneMember u) {
  u.x = (int2)(0, 0);
}

// AMD-LABEL: define amdgpu_kernel void @KernelOneMember
// AMD-SAME:  (<2 x i32> %[[u_coerce:.*]])
// AMD:  %[[u:.*]] = alloca %struct.StructOneMember, align 8, addrspace(5)
// AMD:  %[[coerce_dive:.*]] = getelementptr inbounds %struct.StructOneMember, %struct.StructOneMember addrspace(5)* %[[u]], i32 0, i32 0
// AMD:  store <2 x i32> %[[u_coerce]], <2 x i32> addrspace(5)* %[[coerce_dive]]
// AMD:  call void @FuncOneMember(%struct.StructOneMember addrspace(5)* byval align 8 %[[u]])
kernel void KernelOneMember(struct StructOneMember u) {
  FuncOneMember(u);
}

// AMD-LABEL: define void @FuncTwoMember(%struct.StructTwoMember addrspace(5)* byval align 8 %u)
void FuncTwoMember(struct StructTwoMember u) {
  u.x = (int2)(0, 0);
}

// AMD-LABEL: define amdgpu_kernel void @KernelTwoMember
// AMD-SAME:  (%struct.StructTwoMember %[[u_coerce:.*]])
// AMD:  %[[u:.*]] = alloca %struct.StructTwoMember, align 8, addrspace(5)
// AMD:  store %struct.StructTwoMember %[[u_coerce]], %struct.StructTwoMember addrspace(5)* %[[u]]
// AMD:  call void @FuncTwoMember(%struct.StructTwoMember addrspace(5)* byval align 8 %[[u]])
kernel void KernelTwoMember(struct StructTwoMember u) {
  FuncTwoMember(u);
}
