// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -fcuda-is-device -S -emit-llvm -o - -x cuda %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -fcuda-is-device -S -emit-llvm -o - -x cuda %s | FileCheck %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

__device__ int read_tid() {

// CHECK: call i32 @llvm.ptx.read.tid.x()
// CHECK: call i32 @llvm.ptx.read.tid.y()
// CHECK: call i32 @llvm.ptx.read.tid.z()
// CHECK: call i32 @llvm.ptx.read.tid.w()

  int x = __builtin_ptx_read_tid_x();
  int y = __builtin_ptx_read_tid_y();
  int z = __builtin_ptx_read_tid_z();
  int w = __builtin_ptx_read_tid_w();

  return x + y + z + w;

}

__device__ int read_ntid() {

// CHECK: call i32 @llvm.ptx.read.ntid.x()
// CHECK: call i32 @llvm.ptx.read.ntid.y()
// CHECK: call i32 @llvm.ptx.read.ntid.z()
// CHECK: call i32 @llvm.ptx.read.ntid.w()

  int x = __builtin_ptx_read_ntid_x();
  int y = __builtin_ptx_read_ntid_y();
  int z = __builtin_ptx_read_ntid_z();
  int w = __builtin_ptx_read_ntid_w();

  return x + y + z + w;

}

__device__ int read_ctaid() {

// CHECK: call i32 @llvm.ptx.read.ctaid.x()
// CHECK: call i32 @llvm.ptx.read.ctaid.y()
// CHECK: call i32 @llvm.ptx.read.ctaid.z()
// CHECK: call i32 @llvm.ptx.read.ctaid.w()

  int x = __builtin_ptx_read_ctaid_x();
  int y = __builtin_ptx_read_ctaid_y();
  int z = __builtin_ptx_read_ctaid_z();
  int w = __builtin_ptx_read_ctaid_w();

  return x + y + z + w;

}

__device__ int read_nctaid() {

// CHECK: call i32 @llvm.ptx.read.nctaid.x()
// CHECK: call i32 @llvm.ptx.read.nctaid.y()
// CHECK: call i32 @llvm.ptx.read.nctaid.z()
// CHECK: call i32 @llvm.ptx.read.nctaid.w()

  int x = __builtin_ptx_read_nctaid_x();
  int y = __builtin_ptx_read_nctaid_y();
  int z = __builtin_ptx_read_nctaid_z();
  int w = __builtin_ptx_read_nctaid_w();

  return x + y + z + w;

}

__device__ int read_ids() {

// CHECK: call i32 @llvm.ptx.read.laneid()
// CHECK: call i32 @llvm.ptx.read.warpid()
// CHECK: call i32 @llvm.ptx.read.nwarpid()
// CHECK: call i32 @llvm.ptx.read.smid()
// CHECK: call i32 @llvm.ptx.read.nsmid()
// CHECK: call i32 @llvm.ptx.read.gridid()

  int a = __builtin_ptx_read_laneid();
  int b = __builtin_ptx_read_warpid();
  int c = __builtin_ptx_read_nwarpid();
  int d = __builtin_ptx_read_smid();
  int e = __builtin_ptx_read_nsmid();
  int f = __builtin_ptx_read_gridid();

  return a + b + c + d + e + f;

}

__device__ int read_lanemasks() {

// CHECK: call i32 @llvm.ptx.read.lanemask.eq()
// CHECK: call i32 @llvm.ptx.read.lanemask.le()
// CHECK: call i32 @llvm.ptx.read.lanemask.lt()
// CHECK: call i32 @llvm.ptx.read.lanemask.ge()
// CHECK: call i32 @llvm.ptx.read.lanemask.gt()

  int a = __builtin_ptx_read_lanemask_eq();
  int b = __builtin_ptx_read_lanemask_le();
  int c = __builtin_ptx_read_lanemask_lt();
  int d = __builtin_ptx_read_lanemask_ge();
  int e = __builtin_ptx_read_lanemask_gt();

  return a + b + c + d + e;

}

__device__ long long read_clocks() {

// CHECK: call i32 @llvm.ptx.read.clock()
// CHECK: call i64 @llvm.ptx.read.clock64()

  int a = __builtin_ptx_read_clock();
  long long b = __builtin_ptx_read_clock64();

  return a + b;
}

__device__ int read_pms() {

// CHECK: call i32 @llvm.ptx.read.pm0()
// CHECK: call i32 @llvm.ptx.read.pm1()
// CHECK: call i32 @llvm.ptx.read.pm2()
// CHECK: call i32 @llvm.ptx.read.pm3()

  int a = __builtin_ptx_read_pm0();
  int b = __builtin_ptx_read_pm1();
  int c = __builtin_ptx_read_pm2();
  int d = __builtin_ptx_read_pm3();

  return a + b + c + d;

}

__device__ void sync() {

// CHECK: call void @llvm.ptx.bar.sync(i32 0)

  __builtin_ptx_bar_sync(0);

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
  __nvvm_bar0();
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
