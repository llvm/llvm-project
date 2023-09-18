// RUN: %clang_cc1 %s -emit-llvm -o - -O0 -ffake-address-space-map -triple i686-pc-darwin | FileCheck -enable-var-scope -check-prefixes=ALL,X86 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -O0 -triple amdgcn | FileCheck -enable-var-scope -check-prefixes=ALL,AMDGCN %s
// RUN: %clang_cc1 %s -emit-llvm -o - -cl-std=CL2.0 -O0 -triple amdgcn | FileCheck -enable-var-scope -check-prefixes=ALL,AMDGCN,AMDGCN20 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -cl-std=CL1.2 -O0 -triple spir-unknown-unknown-unknown | FileCheck -enable-var-scope -check-prefixes=SPIR %s
// RUN: %clang_cc1 %s -emit-llvm -o - -cl-std=CL3.0 -O0 -triple amdgcn -cl-ext=+__opencl_c_program_scope_global_variables | FileCheck -enable-var-scope -check-prefixes=ALL,AMDGCN,AMDGCN20 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -cl-std=CL3.0 -O0 -triple amdgcn | FileCheck -enable-var-scope -check-prefixes=ALL,AMDGCN %s

typedef int int2 __attribute__((ext_vector_type(2)));

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

#if (__OPENCL_C_VERSION__ == 200) || (__OPENCL_C_VERSION__ >= 300 && defined(__opencl_c_program_scope_global_variables))
struct LargeStructOneMember g_s;
#endif

// X86-LABEL: define{{.*}} void @foo(ptr noalias sret(%struct.Mat4X4) align 4 %agg.result, ptr noundef byval(%struct.Mat3X3) align 4 %in)
// AMDGCN-LABEL: define{{.*}} %struct.Mat4X4 @foo([9 x i32] %in.coerce)
Mat4X4 __attribute__((noinline)) foo(Mat3X3 in) {
  Mat4X4 out;
  return out;
}

// ALL-LABEL: define {{.*}} void @ker
// Expect two mem copies: one for the argument "in", and one for
// the return value.
// X86: call void @llvm.memcpy.p0.p1.i32(ptr
// X86: call void @llvm.memcpy.p1.p0.i32(ptr addrspace(1)

// AMDGCN: load [9 x i32], ptr addrspace(1)
// AMDGCN: call %struct.Mat4X4 @foo([9 x i32]
// AMDGCN: call void @llvm.memcpy.p1.p5.i64(ptr addrspace(1)
kernel void ker(global Mat3X3 *in, global Mat4X4 *out) {
  out[0] = foo(in[1]);
}

// X86-LABEL: define{{.*}} void @foo_large(ptr noalias sret(%struct.Mat64X64) align 4 %agg.result, ptr noundef byval(%struct.Mat32X32) align 4 %in)
// AMDGCN-LABEL: define{{.*}} void @foo_large(ptr addrspace(5) noalias sret(%struct.Mat64X64) align 4 %agg.result, ptr addrspace(5) noundef byref(%struct.Mat32X32) align 4 %{{.*}}
// AMDGCN:       %in = alloca %struct.Mat32X32, align 4, addrspace(5)
// AMDGCN-NEXT:  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 4 %in, ptr addrspace(5) align 4 %{{.*}}, i64 4096, i1 false)
Mat64X64 __attribute__((noinline)) foo_large(Mat32X32 in) {
  Mat64X64 out;
  return out;
}

// ALL-LABEL: define {{.*}} void @ker_large
// Expect two mem copies: one for the argument "in", and one for
// the return value.
// X86: call void @llvm.memcpy.p0.p1.i32(ptr
// X86: call void @llvm.memcpy.p1.p0.i32(ptr addrspace(1)
// AMDGCN: call void @llvm.memcpy.p5.p1.i64(ptr addrspace(5)
// AMDGCN: call void @llvm.memcpy.p1.p5.i64(ptr addrspace(1)
kernel void ker_large(global Mat32X32 *in, global Mat64X64 *out) {
  out[0] = foo_large(in[1]);
}

// AMDGCN-LABEL: define{{.*}} void @FuncOneMember(<2 x i32> %u.coerce)
void FuncOneMember(struct StructOneMember u) {
  u.x = (int2)(0, 0);
}

// AMDGCN-LABEL: define{{.*}} void @FuncOneLargeMember(ptr addrspace(5) noundef byref(%struct.LargeStructOneMember) align 8 %{{.*}}
// AMDGCN:  %u = alloca %struct.LargeStructOneMember, align 8, addrspace(5)
// AMDGCN:  call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 8 %u, ptr addrspace(5) align 8 %{{.*}}, i64 800, i1 false)
// AMDGCN-NOT: addrspacecast
// AMDGCN:   store <2 x i32> %{{.*}}, ptr addrspace(5)
void FuncOneLargeMember(struct LargeStructOneMember u) {
  u.x[0] = (int2)(0, 0);
}

// AMDGCN20-LABEL: define{{.*}} void @test_indirect_arg_globl()
// AMDGCN20:  %[[byval_temp:.*]] = alloca %struct.LargeStructOneMember, align 8, addrspace(5)
// AMDGCN20:  call void @llvm.memcpy.p5.p1.i64(ptr addrspace(5) align 8 %[[byval_temp]], ptr addrspace(1) align 8 @g_s, i64 800, i1 false)
// AMDGCN20:  call void @FuncOneLargeMember(ptr addrspace(5) noundef byref(%struct.LargeStructOneMember) align 8 %[[byval_temp]])
#if (__OPENCL_C_VERSION__ == 200) || (__OPENCL_C_VERSION__ >= 300 && defined(__opencl_c_program_scope_global_variables))
void test_indirect_arg_globl(void) {
  FuncOneLargeMember(g_s);
}
#endif

// AMDGCN-LABEL: define{{.*}} amdgpu_kernel void @test_indirect_arg_local()
// AMDGCN: %[[byval_temp:.*]] = alloca %struct.LargeStructOneMember, align 8, addrspace(5)
// AMDGCN: call void @llvm.memcpy.p5.p3.i64(ptr addrspace(5) align 8 %[[byval_temp]], ptr addrspace(3) align 8 @test_indirect_arg_local.l_s, i64 800, i1 false)
// AMDGCN: call void @FuncOneLargeMember(ptr addrspace(5) noundef byref(%struct.LargeStructOneMember) align 8 %[[byval_temp]])
kernel void test_indirect_arg_local(void) {
  local struct LargeStructOneMember l_s;
  FuncOneLargeMember(l_s);
}

// AMDGCN-LABEL: define{{.*}} void @test_indirect_arg_private()
// AMDGCN: %[[p_s:.*]] = alloca %struct.LargeStructOneMember, align 8, addrspace(5)
// AMDGCN-NOT: @llvm.memcpy
// AMDGCN-NEXT: call void @FuncOneLargeMember(ptr addrspace(5) noundef byref(%struct.LargeStructOneMember) align 8 %[[p_s]])
void test_indirect_arg_private(void) {
  struct LargeStructOneMember p_s;
  FuncOneLargeMember(p_s);
}

// AMDGCN-LABEL: define{{.*}} amdgpu_kernel void @KernelOneMember
// AMDGCN-SAME:  (<2 x i32> %[[u_coerce:.*]])
// AMDGCN:  %[[u:.*]] = alloca %struct.StructOneMember, align 8, addrspace(5)
// AMDGCN:  %[[coerce_dive:.*]] = getelementptr inbounds %struct.StructOneMember, ptr addrspace(5) %[[u]], i32 0, i32 0
// AMDGCN:  store <2 x i32> %[[u_coerce]], ptr addrspace(5) %[[coerce_dive]]
// AMDGCN:  call void @FuncOneMember(<2 x i32>
kernel void KernelOneMember(struct StructOneMember u) {
  FuncOneMember(u);
}

// SPIR: call void @llvm.memcpy.p0.p1.i32
// SPIR-NOT: addrspacecast
kernel void KernelOneMemberSpir(global struct StructOneMember* u) {
  FuncOneMember(*u);
}

// AMDGCN-LABEL: define{{.*}} amdgpu_kernel void @KernelLargeOneMember(
// AMDGCN:  %[[U:.*]] = alloca %struct.LargeStructOneMember, align 8, addrspace(5)
// AMDGCN:  store %struct.LargeStructOneMember %u.coerce, ptr addrspace(5) %[[U]], align 8
// AMDGCN:  call void @FuncOneLargeMember(ptr addrspace(5) noundef byref(%struct.LargeStructOneMember) align 8 %[[U]])
kernel void KernelLargeOneMember(struct LargeStructOneMember u) {
  FuncOneLargeMember(u);
}

// AMDGCN-LABEL: define{{.*}} void @FuncTwoMember(<2 x i32> %u.coerce0, <2 x i32> %u.coerce1)
void FuncTwoMember(struct StructTwoMember u) {
  u.y = (int2)(0, 0);
}

// AMDGCN-LABEL: define dso_local void @FuncLargeTwoMember
// AMDGCN-SAME: (ptr addrspace(5) noundef byref([[STRUCT_LARGESTRUCTTWOMEMBER:%.*]]) align 8 [[TMP0:%.*]])
// AMDGCN: %[[U:.*]] = alloca %struct.LargeStructTwoMember, align 8, addrspace(5)
// AMDGCN: call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) align 8 %[[U]], ptr addrspace(5) align 8 [[TMP0]], i64 480, i1 false)
void FuncLargeTwoMember(struct LargeStructTwoMember u) {
  u.y[0] = (int2)(0, 0);
}

// AMDGCN-LABEL: define{{.*}} amdgpu_kernel void @KernelTwoMember
// AMDGCN-SAME:  (%struct.StructTwoMember %[[u_coerce:.*]])
// AMDGCN:  %[[u:.*]] = alloca %struct.StructTwoMember, align 8, addrspace(5)
// AMDGCN: %[[LD0:.*]] = load <2 x i32>, ptr addrspace(5)
// AMDGCN: %[[LD1:.*]] = load <2 x i32>, ptr addrspace(5)
// AMDGCN: call void @FuncTwoMember(<2 x i32> %[[LD0]], <2 x i32> %[[LD1]])
kernel void KernelTwoMember(struct StructTwoMember u) {
  FuncTwoMember(u);
}

// AMDGCN-LABEL: define{{.*}} amdgpu_kernel void @KernelLargeTwoMember
// AMDGCN-SAME:  (%struct.LargeStructTwoMember %[[u_coerce:.*]])
// AMDGCN:  %[[u:.*]] = alloca %struct.LargeStructTwoMember, align 8, addrspace(5)
// AMDGCN:  store %struct.LargeStructTwoMember %[[u_coerce]], ptr addrspace(5) %[[u]]
// AMDGCN:  call void @FuncLargeTwoMember(ptr addrspace(5) noundef byref(%struct.LargeStructTwoMember) align 8 %[[u]])
kernel void KernelLargeTwoMember(struct LargeStructTwoMember u) {
  FuncLargeTwoMember(u);
}
