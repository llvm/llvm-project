// RUN: %clang_cc1 %s -emit-llvm -o - -O0 -finclude-default-header -ffake-address-space-map -triple i686-pc-darwin | FileCheck -enable-var-scope -check-prefixes=COM,X86 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -O0 -finclude-default-header -triple amdgcn-amdhsa-amd | FileCheck -enable-var-scope -check-prefixes=COM,AMDGCN %s

typedef struct {
  int cells[9];
} Mat3X3;

typedef struct {
  int cells[16];
} Mat4X4;

typedef struct {
  int cells[1024];
} Mat32X32;

typedef struct {
  int cells[4096];
} Mat64X64;

struct StructOneMember {
  int2 x;
};

struct StructTwoMember {
  int2 x;
  int2 y;
};

struct LargeStructOneMember {
  int2 x[100];
};

struct LargeStructTwoMember {
  int2 x[40];
  int2 y[20];
};


// X86-LABEL: define void @foo(%struct.Mat4X4* noalias sret %agg.result, %struct.Mat3X3* byval align 4 %in)
// AMDGCN-LABEL: define %struct.Mat4X4 @foo([9 x i32] %in.coerce)
Mat4X4 __attribute__((noinline)) foo(Mat3X3 in) {
  Mat4X4 out;
  return out;
}

// COM-LABEL: define {{.*}} void @ker
// Expect two mem copies: one for the argument "in", and one for
// the return value.
// X86: call void @llvm.memcpy.p0i8.p1i8.i32(i8*
// X86: call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)*

// AMDGCN: load [9 x i32], [9 x i32] addrspace(1)*
// AMDGCN: call %struct.Mat4X4 @foo([9 x i32]
// AMDGCN: call void @llvm.memcpy.p1i8.p5i8.i64(i8 addrspace(1)*
kernel void ker(global Mat3X3 *in, global Mat4X4 *out) {
  out[0] = foo(in[1]);
}

// X86-LABEL: define void @foo_large(%struct.Mat64X64* noalias sret %agg.result, %struct.Mat32X32* byval align 4 %in)
// AMDGCN-LABEL: define void @foo_large(%struct.Mat64X64 addrspace(5)* noalias sret %agg.result, %struct.Mat32X32 addrspace(5)* byval align 4 %in)
Mat64X64 __attribute__((noinline)) foo_large(Mat32X32 in) {
  Mat64X64 out;
  return out;
}

// COM-LABEL: define {{.*}} void @ker_large
// Expect two mem copies: one for the argument "in", and one for
// the return value.
// X86: call void @llvm.memcpy.p0i8.p1i8.i32(i8*
// X86: call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)*
// AMDGCN: call void @llvm.memcpy.p5i8.p1i8.i64(i8 addrspace(5)*
// AMDGCN: call void @llvm.memcpy.p1i8.p5i8.i64(i8 addrspace(1)*
kernel void ker_large(global Mat32X32 *in, global Mat64X64 *out) {
  out[0] = foo_large(in[1]);
}

// AMDGCN-LABEL: define void @FuncOneMember(<2 x i32> %u.coerce)
void FuncOneMember(struct StructOneMember u) {
  u.x = (int2)(0, 0);
}

// AMDGCN-LABEL: define void @FuncOneLargeMember(%struct.LargeStructOneMember addrspace(5)* byval align 8 %u)
void FuncOneLargeMember(struct LargeStructOneMember u) {
  u.x[0] = (int2)(0, 0);
}

// AMDGCN-LABEL: define amdgpu_kernel void @KernelOneMember
// AMDGCN-SAME:  (<2 x i32> %[[u_coerce:.*]])
// AMDGCN:  %[[u:.*]] = alloca %struct.StructOneMember, align 8, addrspace(5)
// AMDGCN:  %[[coerce_dive:.*]] = getelementptr inbounds %struct.StructOneMember, %struct.StructOneMember addrspace(5)* %[[u]], i32 0, i32 0
// AMDGCN:  store <2 x i32> %[[u_coerce]], <2 x i32> addrspace(5)* %[[coerce_dive]]
// AMDGCN:  call void @FuncOneMember(<2 x i32>
kernel void KernelOneMember(struct StructOneMember u) {
  FuncOneMember(u);
}

// AMDGCN-LABEL: define amdgpu_kernel void @KernelLargeOneMember(
// AMDGCN:  %[[U:.*]] = alloca %struct.LargeStructOneMember, align 8, addrspace(5)
// AMDGCN:  store %struct.LargeStructOneMember %u.coerce, %struct.LargeStructOneMember addrspace(5)* %[[U]], align 8
// AMDGCN:  call void @FuncOneLargeMember(%struct.LargeStructOneMember addrspace(5)* byval align 8 %[[U]])
kernel void KernelLargeOneMember(struct LargeStructOneMember u) {
  FuncOneLargeMember(u);
}

// AMDGCN-LABEL: define void @FuncTwoMember(<2 x i32> %u.coerce0, <2 x i32> %u.coerce1)
void FuncTwoMember(struct StructTwoMember u) {
  u.y = (int2)(0, 0);
}

// AMDGCN-LABEL: define void @FuncLargeTwoMember(%struct.LargeStructTwoMember addrspace(5)* byval align 8 %u)
void FuncLargeTwoMember(struct LargeStructTwoMember u) {
  u.y[0] = (int2)(0, 0);
}


// AMDGCN-LABEL: define amdgpu_kernel void @KernelTwoMember
// AMDGCN-SAME:  (%struct.StructTwoMember %[[u_coerce:.*]])
// AMDGCN:  %[[u:.*]] = alloca %struct.StructTwoMember, align 8, addrspace(5)
// AMDGCN: %[[LD0:.*]] = load <2 x i32>, <2 x i32> addrspace(5)*
// AMDGCN: %[[LD1:.*]] = load <2 x i32>, <2 x i32> addrspace(5)*
// AMDGCN: call void @FuncTwoMember(<2 x i32> %[[LD0]], <2 x i32> %[[LD1]])
kernel void KernelTwoMember(struct StructTwoMember u) {
  FuncTwoMember(u);
}

// AMDGCN-LABEL: define amdgpu_kernel void @KernelLargeTwoMember
// AMDGCN-SAME:  (%struct.LargeStructTwoMember %[[u_coerce:.*]])
// AMDGCN:  %[[u:.*]] = alloca %struct.LargeStructTwoMember, align 8, addrspace(5)
// AMDGCN:  store %struct.LargeStructTwoMember %[[u_coerce]], %struct.LargeStructTwoMember addrspace(5)* %[[u]]
// AMDGCN:  call void @FuncLargeTwoMember(%struct.LargeStructTwoMember addrspace(5)* byval align 8 %[[u]])
kernel void KernelLargeTwoMember(struct LargeStructTwoMember u) {
  FuncLargeTwoMember(u);
}
