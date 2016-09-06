// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -fcuda-is-device -S -emit-llvm -o - -x cuda %s | \
// RUN:   FileCheck -check-prefix=CHECK -check-prefix=LP32 %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -fcuda-is-device -S -emit-llvm -o - -x cuda %s | \
// RUN:   FileCheck -check-prefix=CHECK -check-prefix=LP64 %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

__device__ int read_tid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.w()

  int x = __nvvm_read_ptx_sreg_tid_x();
  int y = __nvvm_read_ptx_sreg_tid_y();
  int z = __nvvm_read_ptx_sreg_tid_z();
  int w = __nvvm_read_ptx_sreg_tid_w();

  return x + y + z + w;

}

__device__ int read_ntid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.w()

  int x = __nvvm_read_ptx_sreg_ntid_x();
  int y = __nvvm_read_ptx_sreg_ntid_y();
  int z = __nvvm_read_ptx_sreg_ntid_z();
  int w = __nvvm_read_ptx_sreg_ntid_w();

  return x + y + z + w;

}

__device__ int read_ctaid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.w()

  int x = __nvvm_read_ptx_sreg_ctaid_x();
  int y = __nvvm_read_ptx_sreg_ctaid_y();
  int z = __nvvm_read_ptx_sreg_ctaid_z();
  int w = __nvvm_read_ptx_sreg_ctaid_w();

  return x + y + z + w;

}

__device__ int read_nctaid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.w()

  int x = __nvvm_read_ptx_sreg_nctaid_x();
  int y = __nvvm_read_ptx_sreg_nctaid_y();
  int z = __nvvm_read_ptx_sreg_nctaid_z();
  int w = __nvvm_read_ptx_sreg_nctaid_w();

  return x + y + z + w;

}

__device__ int read_ids() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nwarpid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.smid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nsmid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.gridid()

  int a = __nvvm_read_ptx_sreg_laneid();
  int b = __nvvm_read_ptx_sreg_warpid();
  int c = __nvvm_read_ptx_sreg_nwarpid();
  int d = __nvvm_read_ptx_sreg_smid();
  int e = __nvvm_read_ptx_sreg_nsmid();
  int f = __nvvm_read_ptx_sreg_gridid();

  return a + b + c + d + e + f;

}

__device__ int read_lanemasks() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.eq()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.le()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.lt()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.ge()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.gt()

  int a = __nvvm_read_ptx_sreg_lanemask_eq();
  int b = __nvvm_read_ptx_sreg_lanemask_le();
  int c = __nvvm_read_ptx_sreg_lanemask_lt();
  int d = __nvvm_read_ptx_sreg_lanemask_ge();
  int e = __nvvm_read_ptx_sreg_lanemask_gt();

  return a + b + c + d + e;

}

__device__ long long read_clocks() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clock()
// CHECK: call i64 @llvm.nvvm.read.ptx.sreg.clock64()

  int a = __nvvm_read_ptx_sreg_clock();
  long long b = __nvvm_read_ptx_sreg_clock64();

  return a + b;
}

__device__ int read_pms() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm0()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm1()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm2()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm3()

  int a = __nvvm_read_ptx_sreg_pm0();
  int b = __nvvm_read_ptx_sreg_pm1();
  int c = __nvvm_read_ptx_sreg_pm2();
  int d = __nvvm_read_ptx_sreg_pm3();

  return a + b + c + d;

}

__device__ void sync() {

// CHECK: call void @llvm.nvvm.bar.sync(i32 0)

  __nvvm_bar_sync(0);

}


// NVVM intrinsics

// The idea is not to test all intrinsics, just that Clang is recognizing the
// builtins defined in BuiltinsNVPTX.def
__device__ void nvvm_math(float f1, float f2, double d1, double d2) {
// CHECK: call float @llvm.nvvm.fmax.f
  float t1 = __nvvm_fmax_f(f1, f2);
// CHECK: call float @llvm.nvvm.fmin.f
  float t2 = __nvvm_fmin_f(f1, f2);
// CHECK: call float @llvm.nvvm.sqrt.rn.f
  float t3 = __nvvm_sqrt_rn_f(f1);
// CHECK: call float @llvm.nvvm.rcp.rn.f
  float t4 = __nvvm_rcp_rn_f(f2);
// CHECK: call float @llvm.nvvm.add.rn.f
  float t5 = __nvvm_add_rn_f(f1, f2);

// CHECK: call double @llvm.nvvm.fmax.d
  double td1 = __nvvm_fmax_d(d1, d2);
// CHECK: call double @llvm.nvvm.fmin.d
  double td2 = __nvvm_fmin_d(d1, d2);
// CHECK: call double @llvm.nvvm.sqrt.rn.d
  double td3 = __nvvm_sqrt_rn_d(d1);
// CHECK: call double @llvm.nvvm.rcp.rn.d
  double td4 = __nvvm_rcp_rn_d(d2);

// CHECK: call void @llvm.nvvm.membar.cta()
  __nvvm_membar_cta();
// CHECK: call void @llvm.nvvm.membar.gl()
  __nvvm_membar_gl();
// CHECK: call void @llvm.nvvm.membar.sys()
  __nvvm_membar_sys();
// CHECK: call void @llvm.nvvm.barrier0()
  __syncthreads();
}

__device__ int di;
__shared__ int si;
__device__ long dl;
__shared__ long sl;
__device__ long long dll;
__shared__ long long sll;

// Check for atomic intrinsics
// CHECK-LABEL: nvvm_atom
__device__ void nvvm_atom(float *fp, float f, int *ip, int i, unsigned int *uip, unsigned ui, long *lp, long l,
                          long long *llp, long long ll) {
  // CHECK: atomicrmw add
  __nvvm_atom_add_gen_i(ip, i);
  // CHECK: atomicrmw add
  __nvvm_atom_add_gen_l(&dl, l);
  // CHECK: atomicrmw add
  __nvvm_atom_add_gen_ll(&sll, ll);

  // CHECK: atomicrmw sub
  __nvvm_atom_sub_gen_i(ip, i);
  // CHECK: atomicrmw sub
  __nvvm_atom_sub_gen_l(&dl, l);
  // CHECK: atomicrmw sub
  __nvvm_atom_sub_gen_ll(&sll, ll);

  // CHECK: atomicrmw and
  __nvvm_atom_and_gen_i(ip, i);
  // CHECK: atomicrmw and
  __nvvm_atom_and_gen_l(&dl, l);
  // CHECK: atomicrmw and
  __nvvm_atom_and_gen_ll(&sll, ll);

  // CHECK: atomicrmw or
  __nvvm_atom_or_gen_i(ip, i);
  // CHECK: atomicrmw or
  __nvvm_atom_or_gen_l(&dl, l);
  // CHECK: atomicrmw or
  __nvvm_atom_or_gen_ll(&sll, ll);

  // CHECK: atomicrmw xor
  __nvvm_atom_xor_gen_i(ip, i);
  // CHECK: atomicrmw xor
  __nvvm_atom_xor_gen_l(&dl, l);
  // CHECK: atomicrmw xor
  __nvvm_atom_xor_gen_ll(&sll, ll);

  // CHECK: atomicrmw xchg
  __nvvm_atom_xchg_gen_i(ip, i);
  // CHECK: atomicrmw xchg
  __nvvm_atom_xchg_gen_l(&dl, l);
  // CHECK: atomicrmw xchg
  __nvvm_atom_xchg_gen_ll(&sll, ll);

  // CHECK: atomicrmw max i32*
  __nvvm_atom_max_gen_i(ip, i);
  // CHECK: atomicrmw umax i32*
  __nvvm_atom_max_gen_ui((unsigned int *)ip, i);
  // CHECK: atomicrmw max
  __nvvm_atom_max_gen_l(&dl, l);
  // CHECK: atomicrmw umax
  __nvvm_atom_max_gen_ul((unsigned long *)&dl, l);
  // CHECK: atomicrmw max i64*
  __nvvm_atom_max_gen_ll(&sll, ll);
  // CHECK: atomicrmw umax i64*
  __nvvm_atom_max_gen_ull((unsigned long long *)&sll, ll);

  // CHECK: atomicrmw min i32*
  __nvvm_atom_min_gen_i(ip, i);
  // CHECK: atomicrmw umin i32*
  __nvvm_atom_min_gen_ui((unsigned int *)ip, i);
  // CHECK: atomicrmw min
  __nvvm_atom_min_gen_l(&dl, l);
  // CHECK: atomicrmw umin
  __nvvm_atom_min_gen_ul((unsigned long *)&dl, l);
  // CHECK: atomicrmw min i64*
  __nvvm_atom_min_gen_ll(&sll, ll);
  // CHECK: atomicrmw umin i64*
  __nvvm_atom_min_gen_ull((unsigned long long *)&sll, ll);

  // CHECK: cmpxchg
  // CHECK-NEXT: extractvalue { i32, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_i(ip, 0, i);
  // CHECK: cmpxchg
  // CHECK-NEXT: extractvalue { {{i32|i64}}, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_l(&dl, 0, l);
  // CHECK: cmpxchg
  // CHECK-NEXT: extractvalue { i64, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_ll(&sll, 0, ll);

  // CHECK: call float @llvm.nvvm.atomic.load.add.f32.p0f32
  __nvvm_atom_add_gen_f(fp, f);

  // CHECK: call i32 @llvm.nvvm.atomic.load.inc.32.p0i32
  __nvvm_atom_inc_gen_ui(uip, ui);

  // CHECK: call i32 @llvm.nvvm.atomic.load.dec.32.p0i32
  __nvvm_atom_dec_gen_ui(uip, ui);

  // CHECK: ret
}

// CHECK-LABEL: nvvm_ldg
__device__ void nvvm_ldg(const void *p) {
  // CHECK: call i8 @llvm.nvvm.ldg.global.i.i8.p0i8(i8* {{%[0-9]+}}, i32 1)
  // CHECK: call i8 @llvm.nvvm.ldg.global.i.i8.p0i8(i8* {{%[0-9]+}}, i32 1)
  __nvvm_ldg_c((const char *)p);
  __nvvm_ldg_uc((const unsigned char *)p);

  // CHECK: call i16 @llvm.nvvm.ldg.global.i.i16.p0i16(i16* {{%[0-9]+}}, i32 2)
  // CHECK: call i16 @llvm.nvvm.ldg.global.i.i16.p0i16(i16* {{%[0-9]+}}, i32 2)
  __nvvm_ldg_s((const short *)p);
  __nvvm_ldg_us((const unsigned short *)p);

  // CHECK: call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* {{%[0-9]+}}, i32 4)
  // CHECK: call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* {{%[0-9]+}}, i32 4)
  __nvvm_ldg_i((const int *)p);
  __nvvm_ldg_ui((const unsigned int *)p);

  // LP32: call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* {{%[0-9]+}}, i32 4)
  // LP32: call i32 @llvm.nvvm.ldg.global.i.i32.p0i32(i32* {{%[0-9]+}}, i32 4)
  // LP64: call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* {{%[0-9]+}}, i32 8)
  // LP64: call i64 @llvm.nvvm.ldg.global.i.i64.p0i64(i64* {{%[0-9]+}}, i32 8)
  __nvvm_ldg_l((const long *)p);
  __nvvm_ldg_ul((const unsigned long *)p);

  // CHECK: call float @llvm.nvvm.ldg.global.f.f32.p0f32(float* {{%[0-9]+}}, i32 4)
  __nvvm_ldg_f((const float *)p);
  // CHECK: call double @llvm.nvvm.ldg.global.f.f64.p0f64(double* {{%[0-9]+}}, i32 8)
  __nvvm_ldg_d((const double *)p);

  // In practice, the pointers we pass to __ldg will be aligned as appropriate
  // for the CUDA <type>N vector types (e.g. short4), which are not the same as
  // the LLVM vector types.  However, each LLVM vector type has an alignment
  // less than or equal to its corresponding CUDA type, so we're OK.
  //
  // PTX Interoperability section 2.2: "For a vector with an even number of
  // elements, its alignment is set to number of elements times the alignment of
  // its member: n*alignof(t)."

  // CHECK: call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0v2i8(<2 x i8>* {{%[0-9]+}}, i32 2)
  // CHECK: call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0v2i8(<2 x i8>* {{%[0-9]+}}, i32 2)
  typedef char char2 __attribute__((ext_vector_type(2)));
  typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_c2((const char2 *)p);
  __nvvm_ldg_uc2((const uchar2 *)p);

  // CHECK: call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0v4i8(<4 x i8>* {{%[0-9]+}}, i32 4)
  // CHECK: call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0v4i8(<4 x i8>* {{%[0-9]+}}, i32 4)
  typedef char char4 __attribute__((ext_vector_type(4)));
  typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_c4((const char4 *)p);
  __nvvm_ldg_uc4((const uchar4 *)p);

  // CHECK: call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0v2i16(<2 x i16>* {{%[0-9]+}}, i32 4)
  // CHECK: call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0v2i16(<2 x i16>* {{%[0-9]+}}, i32 4)
  typedef short short2 __attribute__((ext_vector_type(2)));
  typedef unsigned short ushort2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_s2((const short2 *)p);
  __nvvm_ldg_us2((const ushort2 *)p);

  // CHECK: call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0v4i16(<4 x i16>* {{%[0-9]+}}, i32 8)
  // CHECK: call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0v4i16(<4 x i16>* {{%[0-9]+}}, i32 8)
  typedef short short4 __attribute__((ext_vector_type(4)));
  typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_s4((const short4 *)p);
  __nvvm_ldg_us4((const ushort4 *)p);

  // CHECK: call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0v2i32(<2 x i32>* {{%[0-9]+}}, i32 8)
  // CHECK: call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0v2i32(<2 x i32>* {{%[0-9]+}}, i32 8)
  typedef int int2 __attribute__((ext_vector_type(2)));
  typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_i2((const int2 *)p);
  __nvvm_ldg_ui2((const uint2 *)p);

  // CHECK: call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0v4i32(<4 x i32>* {{%[0-9]+}}, i32 16)
  // CHECK: call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0v4i32(<4 x i32>* {{%[0-9]+}}, i32 16)
  typedef int int4 __attribute__((ext_vector_type(4)));
  typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_i4((const int4 *)p);
  __nvvm_ldg_ui4((const uint4 *)p);

  // CHECK: call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0v2i64(<2 x i64>* {{%[0-9]+}}, i32 16)
  // CHECK: call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0v2i64(<2 x i64>* {{%[0-9]+}}, i32 16)
  typedef long long longlong2 __attribute__((ext_vector_type(2)));
  typedef unsigned long long ulonglong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_ll2((const longlong2 *)p);
  __nvvm_ldg_ull2((const ulonglong2 *)p);

  // CHECK: call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0v2f32(<2 x float>* {{%[0-9]+}}, i32 8)
  typedef float float2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_f2((const float2 *)p);

  // CHECK: call <4 x float> @llvm.nvvm.ldg.global.f.v4f32.p0v4f32(<4 x float>* {{%[0-9]+}}, i32 16)
  typedef float float4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_f4((const float4 *)p);

  // CHECK: call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0v2f64(<2 x double>* {{%[0-9]+}}, i32 16)
  typedef double double2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_d2((const double2 *)p);
}
