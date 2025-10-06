// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_70 -target-feature +ptx63 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX63_SM70 -check-prefix=LP64 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu sm_80 -target-feature +ptx70 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX70_SM80 -check-prefix=LP32 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_80 -target-feature +ptx70 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX70_SM80 -check-prefix=LP64 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu sm_60 -target-feature +ptx62 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=LP32 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_60 -target-feature +ptx62 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=LP64 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_61 -target-feature +ptx62 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=LP64 %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_53 -target-feature +ptx62 \
// RUN:   -DERROR_CHECK -fcuda-is-device -S -o /dev/null -x cuda -verify %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu sm_86 -target-feature +ptx72 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX72_SM86 -check-prefix=LP32 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_86 -target-feature +ptx72 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX72_SM86 -check-prefix=LP64 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_89 -target-feature +ptx81 -DPTX=81\
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX81_SM89 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_90 -target-feature +ptx78 -DPTX=78 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX78_SM90 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_100 -target-feature +ptx86 -DPTX=86 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX86_SM100 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_100a -target-feature +ptx86 -DPTX=86 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX86_SM100a %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_101a -target-feature +ptx86 -DPTX=86 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX86_SM101a %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_120a -target-feature +ptx86 -DPTX=86 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX86_SM120a %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_103a -target-feature +ptx87 -DPTX=87 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX87_SM103a %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_100a -target-feature +ptx87 -DPTX=87 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX87_SM100a %s
// ###  The last run to check with the highest SM and PTX version available
// ###  to make sure target builtins are still accepted.
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_120a -target-feature +ptx87 -DPTX=87 \
// RUN:            -disable-llvm-optzns -fcuda-is-device -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX86_SM120a %s

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

__device__ bool reflect() {

// CHECK: call i32 @llvm.nvvm.reflect(ptr {{.*}})

  unsigned x = __nvvm_reflect("__CUDA_ARCH");
  return x >= 700;

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
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpsize()

  int a = __nvvm_read_ptx_sreg_laneid();
  int b = __nvvm_read_ptx_sreg_warpid();
  int c = __nvvm_read_ptx_sreg_nwarpid();
  int d = __nvvm_read_ptx_sreg_smid();
  int e = __nvvm_read_ptx_sreg_nsmid();
  int f = __nvvm_read_ptx_sreg_gridid();
  int g = __nvvm_read_ptx_sreg_warpsize();

  return a + b + c + d + e + f + g;

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
// CHECK: call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()

  int a = __nvvm_read_ptx_sreg_clock();
  long long b = __nvvm_read_ptx_sreg_clock64();
  long long c = __nvvm_read_ptx_sreg_globaltimer();

  return a + b + c;
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

// CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)

  __nvvm_bar_sync(0);

}

__device__ void activemask() {

// CHECK: call i32 @llvm.nvvm.activemask()

  __nvvm_activemask();

}

__device__ void exit() {

// CHECK: call void @llvm.nvvm.exit()

  __nvvm_exit();

}

// NVVM intrinsics

// The idea is not to test all intrinsics, just that Clang is recognizing the
// builtins defined in BuiltinsNVPTX.td
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

// CHECK: call float @llvm.nvvm.fabs.f32
  float t6 = __nvvm_fabs_f(f1);
// CHECK: call float @llvm.nvvm.fabs.ftz.f32
  float t7 = __nvvm_fabs_ftz_f(f2);

// CHECK: call double @llvm.fabs.f64
  double td5 = __nvvm_fabs_d(d1);

// CHECK: call void @llvm.nvvm.membar.cta()
  __nvvm_membar_cta();
// CHECK: call void @llvm.nvvm.membar.gl()
  __nvvm_membar_gl();
// CHECK: call void @llvm.nvvm.membar.sys()
  __nvvm_membar_sys();
// CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
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
__device__ void nvvm_atom(float *fp, float f, double *dfp, double df,
                          unsigned short *usp, unsigned short us, int *ip,
                          int i, unsigned int *uip, unsigned ui, long *lp,
                          long l, long long *llp, long long ll) {
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 4
  __nvvm_atom_add_gen_i(ip, i);
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_add_gen_l(&dl, l);
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 8
  __nvvm_atom_add_gen_ll(&sll, ll);

  // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 4
  __nvvm_atom_sub_gen_i(ip, i);
  // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_sub_gen_l(&dl, l);
  // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 8
  __nvvm_atom_sub_gen_ll(&sll, ll);

  // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 4
  __nvvm_atom_and_gen_i(ip, i);
  // CHECK: atomicrmw and ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_and_gen_l(&dl, l);
  // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 8
  __nvvm_atom_and_gen_ll(&sll, ll);

  // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 4
  __nvvm_atom_or_gen_i(ip, i);
  // CHECK: atomicrmw or ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_or_gen_l(&dl, l);
  // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 8
  __nvvm_atom_or_gen_ll(&sll, ll);

  // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 4
  __nvvm_atom_xor_gen_i(ip, i);
  // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_xor_gen_l(&dl, l);
  // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 8
  __nvvm_atom_xor_gen_ll(&sll, ll);

  // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 4
  __nvvm_atom_xchg_gen_i(ip, i);
  // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_xchg_gen_l(&dl, l);
  // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 8
  __nvvm_atom_xchg_gen_ll(&sll, ll);

  // CHECK: atomicrmw max ptr {{.*}} seq_cst, align 4
  __nvvm_atom_max_gen_i(ip, i);
  // CHECK: atomicrmw umax ptr {{.*}} seq_cst, align 4
  __nvvm_atom_max_gen_ui((unsigned int *)ip, i);
  // CHECK: atomicrmw max ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_max_gen_l(&dl, l);
  // CHECK: atomicrmw umax ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_max_gen_ul((unsigned long *)&dl, l);
  // CHECK: atomicrmw max ptr {{.*}} seq_cst, align 8
  __nvvm_atom_max_gen_ll(&sll, ll);
  // CHECK: atomicrmw umax ptr {{.*}} seq_cst, align 8
  __nvvm_atom_max_gen_ull((unsigned long long *)&sll, ll);

  // CHECK: atomicrmw min ptr {{.*}} seq_cst, align 4
  __nvvm_atom_min_gen_i(ip, i);
  // CHECK: atomicrmw umin ptr {{.*}} seq_cst, align 4
  __nvvm_atom_min_gen_ui((unsigned int *)ip, i);
  // CHECK: atomicrmw min ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_min_gen_l(&dl, l);
  // CHECK: atomicrmw umin ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_min_gen_ul((unsigned long *)&dl, l);
  // CHECK: atomicrmw min ptr {{.*}} seq_cst, align 8
  __nvvm_atom_min_gen_ll(&sll, ll);
  // CHECK: atomicrmw umin ptr {{.*}} seq_cst, align 8
  __nvvm_atom_min_gen_ull((unsigned long long *)&sll, ll);

  // CHECK: cmpxchg ptr {{.*}} seq_cst seq_cst, align 4
  // CHECK-NEXT: extractvalue { i32, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_i(ip, 0, i);
  // CHECK: cmpxchg ptr {{.*}} seq_cst seq_cst, align {{4|8}}
  // CHECK-NEXT: extractvalue { {{i32|i64}}, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_l(&dl, 0, l);
  // CHECK: cmpxchg ptr {{.*}} seq_cst seq_cst, align 8
  // CHECK-NEXT: extractvalue { i64, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_ll(&sll, 0, ll);

  // CHECK: atomicrmw fadd ptr {{.*}} seq_cst, align 4
  __nvvm_atom_add_gen_f(fp, f);

  // CHECK: atomicrmw uinc_wrap ptr {{.*}} seq_cst, align 4
  __nvvm_atom_inc_gen_ui(uip, ui);

  // CHECK: atomicrmw udec_wrap ptr {{.*}} seq_cst, align 4
  __nvvm_atom_dec_gen_ui(uip, ui);


  //////////////////////////////////////////////////////////////////
  // Atomics with scope (only supported on sm_60+).

#if ERROR_CHECK || __CUDA_ARCH__ >= 600

  // CHECK: call i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_ll(&sll, ll);
  // CHECK: call i32 @llvm.nvvm.atomic.add.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.add.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.add.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.add.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_ll(&sll, ll);

  // CHECK: call float @llvm.nvvm.atomic.add.gen.f.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_f' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_f(fp, f);
  // CHECK: call double @llvm.nvvm.atomic.add.gen.f.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_d' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_d(dfp, df);
  // CHECK: call float @llvm.nvvm.atomic.add.gen.f.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_f' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_f(fp, f);
  // CHECK: call double @llvm.nvvm.atomic.add.gen.f.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_d' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_d(dfp, df);

  // CHECK: call i32 @llvm.nvvm.atomic.exch.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.exch.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.exch.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.exch.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.exch.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.exch.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ul' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ull' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ul' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ull' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ul' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ull' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ul' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ull' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.inc.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_inc_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_inc_gen_ui((unsigned int *)ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.inc.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_inc_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_inc_gen_ui((unsigned int *)ip, i);

  // CHECK: call i32 @llvm.nvvm.atomic.dec.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_dec_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_dec_gen_ui((unsigned int *)ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.dec.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_dec_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_dec_gen_ui((unsigned int *)ip, i);

  // CHECK: call i32 @llvm.nvvm.atomic.and.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_and_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_and_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.and.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.and.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_and_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_and_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.and.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_and_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_and_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.and.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_and_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_and_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.and.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.and.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_and_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_and_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.and.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_and_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_and_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.or.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_or_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_or_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.or.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.or.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_or_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_or_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.or.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_or_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_or_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.or.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_or_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_or_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.or.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.or.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_or_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_or_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.or.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_or_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_or_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.xor.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xor_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_xor_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.xor.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xor_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_xor_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.xor.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xor_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_xor_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.xor.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xor_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_xor_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.xor.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xor_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_xor_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.xor.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xor_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_xor_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_cas_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_cas_gen_i(ip, i, 0);
  // LP32: call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_cas_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_cas_gen_l(&dl, l, 0);
  // CHECK: call i64 @llvm.nvvm.atomic.cas.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_cas_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_cas_gen_ll(&sll, ll, 0);

  // CHECK: call i32 @llvm.nvvm.atomic.cas.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_cas_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_cas_gen_i(ip, i, 0);
  // LP32: call i32 @llvm.nvvm.atomic.cas.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_cas_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_cas_gen_l(&dl, l, 0);
  // CHECK: call i64 @llvm.nvvm.atomic.cas.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_cas_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_cas_gen_ll(&sll, ll, 0);
#endif

#if __CUDA_ARCH__ >= 700
  // CHECK_PTX63_SM70: cmpxchg ptr {{.*}} seq_cst seq_cst, align 2
  // CHECK_PTX63_SM70-NEXT: extractvalue { i16, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_us(usp, 0, us);
  // CHECK_PTX63_SM70: call i16 @llvm.nvvm.atomic.cas.gen.i.cta.i16.p0
  __nvvm_atom_cta_cas_gen_us(usp, 0, us);
  // CHECK_PTX63_SM70: call i16 @llvm.nvvm.atomic.cas.gen.i.sys.i16.p0
  __nvvm_atom_sys_cas_gen_us(usp, 0, us);
#endif

  // CHECK: ret
}

// CHECK-LABEL: nvvm_ldg
__device__ void nvvm_ldg(const void *p) {
  // CHECK: load i8, ptr addrspace(1) {{%[0-9]+}}, align 1, !invariant.load
  // CHECK: load i8, ptr addrspace(1) {{%[0-9]+}}, align 1, !invariant.load
  // CHECK: load i8, ptr addrspace(1) {{%[0-9]+}}, align 1, !invariant.load
  __nvvm_ldg_c((const char *)p);
  __nvvm_ldg_uc((const unsigned char *)p);
  __nvvm_ldg_sc((const signed char *)p);

  // CHECK: load i16, ptr addrspace(1) {{%[0-9]+}}, align 2, !invariant.load
  // CHECK: load i16, ptr addrspace(1) {{%[0-9]+}}, align 2, !invariant.load
  __nvvm_ldg_s((const short *)p);
  __nvvm_ldg_us((const unsigned short *)p);

  // CHECK: load i32, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  // CHECK: load i32, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  __nvvm_ldg_i((const int *)p);
  __nvvm_ldg_ui((const unsigned int *)p);

  // LP32: load i32, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  // LP32: load i32, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  // LP64: load i64, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  // LP64: load i64, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  __nvvm_ldg_l((const long *)p);
  __nvvm_ldg_ul((const unsigned long *)p);

  // CHECK: load float, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  __nvvm_ldg_f((const float *)p);
  // CHECK: load double, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  __nvvm_ldg_d((const double *)p);

  // In practice, the pointers we pass to __ldg will be aligned as appropriate
  // for the CUDA <type>N vector types (e.g. short4), which are not the same as
  // the LLVM vector types.  However, each LLVM vector type has an alignment
  // less than or equal to its corresponding CUDA type, so we're OK.
  //
  // PTX Interoperability section 2.2: "For a vector with an even number of
  // elements, its alignment is set to number of elements times the alignment of
  // its member: n*alignof(t)."

  // CHECK: load <2 x i8>, ptr addrspace(1) {{%[0-9]+}}, align 2, !invariant.load
  // CHECK: load <2 x i8>, ptr addrspace(1) {{%[0-9]+}}, align 2, !invariant.load
  // CHECK: load <2 x i8>, ptr addrspace(1) {{%[0-9]+}}, align 2, !invariant.load
  typedef char char2 __attribute__((ext_vector_type(2)));
  typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));
  typedef signed char schar2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_c2((const char2 *)p);
  __nvvm_ldg_uc2((const uchar2 *)p);
  __nvvm_ldg_sc2((const schar2 *)p);

  // CHECK: load <4 x i8>, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  // CHECK: load <4 x i8>, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  // CHECK: load <4 x i8>, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  typedef char char4 __attribute__((ext_vector_type(4)));
  typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));
  typedef signed char schar4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_c4((const char4 *)p);
  __nvvm_ldg_uc4((const uchar4 *)p);
  __nvvm_ldg_sc4((const schar4 *)p);

  // CHECK: load <2 x i16>, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  // CHECK: load <2 x i16>, ptr addrspace(1) {{%[0-9]+}}, align 4, !invariant.load
  typedef short short2 __attribute__((ext_vector_type(2)));
  typedef unsigned short ushort2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_s2((const short2 *)p);
  __nvvm_ldg_us2((const ushort2 *)p);

  // CHECK: load <4 x i16>, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  // CHECK: load <4 x i16>, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  typedef short short4 __attribute__((ext_vector_type(4)));
  typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_s4((const short4 *)p);
  __nvvm_ldg_us4((const ushort4 *)p);

  // CHECK: load <2 x i32>, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  // CHECK: load <2 x i32>, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  typedef int int2 __attribute__((ext_vector_type(2)));
  typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_i2((const int2 *)p);
  __nvvm_ldg_ui2((const uint2 *)p);

  // CHECK: load <4 x i32>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  // CHECK: load <4 x i32>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  typedef int int4 __attribute__((ext_vector_type(4)));
  typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_i4((const int4 *)p);
  __nvvm_ldg_ui4((const uint4 *)p);

  // LP32: load <2 x i32>, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  // LP32: load <2 x i32>, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  // LP64: load <2 x i64>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  // LP64: load <2 x i64>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  typedef long long2 __attribute__((ext_vector_type(2)));
  typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_l2((const long2 *)p);
  __nvvm_ldg_ul2((const ulong2 *)p);

  // CHECK: load <2 x i64>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  // CHECK: load <2 x i64>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  typedef long long longlong2 __attribute__((ext_vector_type(2)));
  typedef unsigned long long ulonglong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_ll2((const longlong2 *)p);
  __nvvm_ldg_ull2((const ulonglong2 *)p);

  // CHECK: load <2 x float>, ptr addrspace(1) {{%[0-9]+}}, align 8, !invariant.load
  typedef float float2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_f2((const float2 *)p);

  // CHECK: load <4 x float>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  typedef float float4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_f4((const float4 *)p);

  // CHECK: load <2 x double>, ptr addrspace(1) {{%[0-9]+}}, align 16, !invariant.load
  typedef double double2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_d2((const double2 *)p);
}

// CHECK-LABEL: nvvm_ldu
__device__ void nvvm_ldu(const void *p) {
  // CHECK: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // CHECK: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // CHECK: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  __nvvm_ldu_c((const char *)p);
  __nvvm_ldu_uc((const unsigned char *)p);
  __nvvm_ldu_sc((const signed char *)p);

  // CHECK: call i16 @llvm.nvvm.ldu.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call i16 @llvm.nvvm.ldu.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  __nvvm_ldu_s((const short *)p);
  __nvvm_ldu_us((const unsigned short *)p);

  // CHECK: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_i((const int *)p);
  __nvvm_ldu_ui((const unsigned int *)p);

  // LP32: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // LP32: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // LP64: call i64 @llvm.nvvm.ldu.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  // LP64: call i64 @llvm.nvvm.ldu.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_l((const long *)p);
  __nvvm_ldu_ul((const unsigned long *)p);

  // CHECK: call float @llvm.nvvm.ldu.global.f.f32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_f((const float *)p);
  // CHECK: call double @llvm.nvvm.ldu.global.f.f64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_d((const double *)p);

  // CHECK: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  typedef char char2 __attribute__((ext_vector_type(2)));
  typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));
  typedef signed char schar2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_c2((const char2 *)p);
  __nvvm_ldu_uc2((const uchar2 *)p);
  __nvvm_ldu_sc2((const schar2 *)p);

  // CHECK: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  typedef char char4 __attribute__((ext_vector_type(4)));
  typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));
  typedef signed char schar4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_c4((const char4 *)p);
  __nvvm_ldu_uc4((const uchar4 *)p);
  __nvvm_ldu_sc4((const schar4 *)p);

  // CHECK: call <2 x i16> @llvm.nvvm.ldu.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <2 x i16> @llvm.nvvm.ldu.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  typedef short short2 __attribute__((ext_vector_type(2)));
  typedef unsigned short ushort2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_s2((const short2 *)p);
  __nvvm_ldu_us2((const ushort2 *)p);

  // CHECK: call <4 x i16> @llvm.nvvm.ldu.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  // CHECK: call <4 x i16> @llvm.nvvm.ldu.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  typedef short short4 __attribute__((ext_vector_type(4)));
  typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_s4((const short4 *)p);
  __nvvm_ldu_us4((const ushort4 *)p);

  // CHECK: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // CHECK: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  typedef int int2 __attribute__((ext_vector_type(2)));
  typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_i2((const int2 *)p);
  __nvvm_ldu_ui2((const uint2 *)p);

  // CHECK: call <4 x i32> @llvm.nvvm.ldu.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  // CHECK: call <4 x i32> @llvm.nvvm.ldu.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  typedef int int4 __attribute__((ext_vector_type(4)));
  typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_i4((const int4 *)p);
  __nvvm_ldu_ui4((const uint4 *)p);

  // LP32: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // LP32: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // LP64: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // LP64: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef long long2 __attribute__((ext_vector_type(2)));
  typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_l2((const long2 *)p);
  __nvvm_ldu_ul2((const ulong2 *)p);

  // CHECK: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // CHECK: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef long long longlong2 __attribute__((ext_vector_type(2)));
  typedef unsigned long long ulonglong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_ll2((const longlong2 *)p);
  __nvvm_ldu_ull2((const ulonglong2 *)p);

  // CHECK: call <2 x float> @llvm.nvvm.ldu.global.f.v2f32.p0(ptr {{%[0-9]+}}, i32 8)
  typedef float float2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_f2((const float2 *)p);

  // CHECK: call <4 x float> @llvm.nvvm.ldu.global.f.v4f32.p0(ptr {{%[0-9]+}}, i32 16)
  typedef float float4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_f4((const float4 *)p);

  // CHECK: call <2 x double> @llvm.nvvm.ldu.global.f.v2f64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef double double2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_d2((const double2 *)p);
}

// CHECK-LABEL: nvvm_shfl
__device__ void nvvm_shfl(int i, float f, int a, int b) {
  // CHECK: call i32 @llvm.nvvm.shfl.down.i32(i32
  __nvvm_shfl_down_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.down.f32(float
  __nvvm_shfl_down_f32(f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.up.i32(i32
  __nvvm_shfl_up_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.up.f32(float
  __nvvm_shfl_up_f32(f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.bfly.i32(i32
  __nvvm_shfl_bfly_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.bfly.f32(float
  __nvvm_shfl_bfly_f32(f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.idx.i32(i32
  __nvvm_shfl_idx_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.idx.f32(float
  __nvvm_shfl_idx_f32(f, a, b);
  // CHECK: ret void
}

__device__ void nvvm_vote(int pred) {
  // CHECK: call i1 @llvm.nvvm.vote.all(i1
  __nvvm_vote_all(pred);
  // CHECK: call i1 @llvm.nvvm.vote.any(i1
  __nvvm_vote_any(pred);
  // CHECK: call i1 @llvm.nvvm.vote.uni(i1
  __nvvm_vote_uni(pred);
  // CHECK: call i32 @llvm.nvvm.vote.ballot(i1
  __nvvm_vote_ballot(pred);
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_pm_event_mask
__device__ void nvvm_pm_event_mask() {
  // CHECK: call void @llvm.nvvm.pm.event.mask(i16 255)
  __nvvm_pm_event_mask(255);
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_nanosleep
__device__ void nvvm_nanosleep(int d) {
#if __CUDA_ARCH__ >= 700
  // CHECK_PTX70_SM80: call void @llvm.nvvm.nanosleep
  __nvvm_nanosleep(d);

  // CHECK_PTX70_SM80: call void @llvm.nvvm.nanosleep
  __nvvm_nanosleep(1);
#endif
}

// CHECK-LABEL: nvvm_mbarrier
__device__ void nvvm_mbarrier(long long* addr, __attribute__((address_space(3))) long long* sharedAddr, int count, long long state) {
  #if __CUDA_ARCH__ >= 800
  __nvvm_mbarrier_init(addr, count);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.init
  __nvvm_mbarrier_init_shared(sharedAddr, count);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.init.shared

  __nvvm_mbarrier_inval(addr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.inval
  __nvvm_mbarrier_inval_shared(sharedAddr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.inval.shared

  __nvvm_mbarrier_arrive(addr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive
  __nvvm_mbarrier_arrive_shared(sharedAddr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.shared
  __nvvm_mbarrier_arrive_noComplete(addr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.noComplete
  __nvvm_mbarrier_arrive_noComplete_shared(sharedAddr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared

  __nvvm_mbarrier_arrive_drop(addr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop
  __nvvm_mbarrier_arrive_drop_shared(sharedAddr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop.shared
  __nvvm_mbarrier_arrive_drop_noComplete(addr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete
  __nvvm_mbarrier_arrive_drop_noComplete_shared(sharedAddr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete.shared

  __nvvm_mbarrier_test_wait(addr, state);
  // CHECK_PTX70_SM80: call i1 @llvm.nvvm.mbarrier.test.wait
  __nvvm_mbarrier_test_wait_shared(sharedAddr, state);
  // CHECK_PTX70_SM80: call i1 @llvm.nvvm.mbarrier.test.wait.shared

  __nvvm_mbarrier_pending_count(state);
  // CHECK_PTX70_SM80: call i32 @llvm.nvvm.mbarrier.pending.count
  #endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_async_copy
__device__ void nvvm_async_copy(__attribute__((address_space(3))) void* dst, __attribute__((address_space(1))) const void* src, long long* addr, __attribute__((address_space(3))) long long* sharedAddr) {
  #if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive
  __nvvm_cp_async_mbarrier_arrive(addr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive.shared
  __nvvm_cp_async_mbarrier_arrive_shared(sharedAddr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc
  __nvvm_cp_async_mbarrier_arrive_noinc(addr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared
  __nvvm_cp_async_mbarrier_arrive_noinc_shared(sharedAddr);

  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.4(
  __nvvm_cp_async_ca_shared_global_4(dst, src);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.8(
  __nvvm_cp_async_ca_shared_global_8(dst, src);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.16(
  __nvvm_cp_async_ca_shared_global_16(dst, src);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.cg.shared.global.16(
  __nvvm_cp_async_cg_shared_global_16(dst, src);

  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.4.s({{.*}}, i32 2)
  __nvvm_cp_async_ca_shared_global_4(dst, src, 2);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.8.s({{.*}}, i32 2)
  __nvvm_cp_async_ca_shared_global_8(dst, src, 2);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.16.s({{.*}}, i32 2)
  __nvvm_cp_async_ca_shared_global_16(dst, src, 2);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.cg.shared.global.16.s({{.*}}, i32 2)
  __nvvm_cp_async_cg_shared_global_16(dst, src, 2);
  
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.commit.group
  __nvvm_cp_async_commit_group();
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.group(i32 0)
  __nvvm_cp_async_wait_group(0);
    // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.group(i32 8)
  __nvvm_cp_async_wait_group(8);
    // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.group(i32 16)
  __nvvm_cp_async_wait_group(16);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.all
  __nvvm_cp_async_wait_all();
  #endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_cvt_sm80
__device__ void nvvm_cvt_sm80() {
#if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rn(1, 1);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rn_relu(1, 1);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rz(1, 1);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rz_relu(1, 1);

  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rn(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rn(1, 1);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rn.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rn_relu(1, 1);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rz(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rz(1, 1);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rz.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rz_relu(1, 1);

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rn(float 1.000000e+00)
  __nvvm_f2bf16_rn(1);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rn.relu(float 1.000000e+00)
  __nvvm_f2bf16_rn_relu(1);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rz(float 1.000000e+00)
  __nvvm_f2bf16_rz(1);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rz.relu(float 1.000000e+00)
  __nvvm_f2bf16_rz_relu(1);

  // CHECK_PTX70_SM80: call i32 @llvm.nvvm.f2tf32.rna(float 1.000000e+00)
  __nvvm_f2tf32_rna(1);
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_cvt_sm89
__device__ void nvvm_cvt_sm89() {
#if (PTX >= 81) && (__CUDA_ARCH__ >= 890)
  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.ff.to.e4m3x2.rn(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e4m3x2_rn(1.0f, 1.0f);
  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.ff.to.e4m3x2.rn.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e4m3x2_rn_relu(1.0f, 1.0f);
  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.ff.to.e5m2x2.rn(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e5m2x2_rn(1.0f, 1.0f);
  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.ff.to.e5m2x2.rn.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e5m2x2_rn_relu(1.0f, 1.0f);

  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.f16x2.to.e4m3x2.rn(<2 x half> splat (half 0xH3C00))
  __nvvm_f16x2_to_e4m3x2_rn({1.0f16, 1.0f16});
  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.f16x2.to.e4m3x2.rn.relu(<2 x half> splat (half 0xH3C00))
  __nvvm_f16x2_to_e4m3x2_rn_relu({1.0f16, 1.0f16});
  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.f16x2.to.e5m2x2.rn(<2 x half> splat (half 0xH3C00))
  __nvvm_f16x2_to_e5m2x2_rn({1.0f16, 1.0f16});
  // CHECK_PTX81_SM89: call i16 @llvm.nvvm.f16x2.to.e5m2x2.rn.relu(<2 x half> splat (half 0xH3C00))
  __nvvm_f16x2_to_e5m2x2_rn_relu({1.0f16, 1.0f16});

  // CHECK_PTX81_SM89: call <2 x half> @llvm.nvvm.e4m3x2.to.f16x2.rn(i16 18504)
  __nvvm_e4m3x2_to_f16x2_rn(0x4848);
  // CHECK_PTX81_SM89: call <2 x half> @llvm.nvvm.e4m3x2.to.f16x2.rn.relu(i16 18504)
  __nvvm_e4m3x2_to_f16x2_rn_relu(0x4848);
  // CHECK_PTX81_SM89: call <2 x half> @llvm.nvvm.e5m2x2.to.f16x2.rn(i16 19532)
  __nvvm_e5m2x2_to_f16x2_rn(0x4c4c);
  // CHECK_PTX81_SM89: call <2 x half> @llvm.nvvm.e5m2x2.to.f16x2.rn.relu(i16 19532)
  __nvvm_e5m2x2_to_f16x2_rn_relu(0x4c4c);

  // CHECK_PTX81_SM89: call i32 @llvm.nvvm.f2tf32.rna.satfinite(float 1.000000e+00)
  __nvvm_f2tf32_rna_satfinite(1.0f);
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_cvt_sm90
__device__ void nvvm_cvt_sm90() {
#if (PTX >= 78) && (__CUDA_ARCH__ >= 900)
  // CHECK_PTX78_SM90: call i32 @llvm.nvvm.f2tf32.rn(float 1.000000e+00)
  __nvvm_f2tf32_rn(1.0f); 
  // CHECK_PTX78_SM90: call i32 @llvm.nvvm.f2tf32.rn.relu(float 1.000000e+00)
  __nvvm_f2tf32_rn_relu(1.0f); 
  // CHECK_PTX78_SM90: call i32 @llvm.nvvm.f2tf32.rz(float 1.000000e+00)
  __nvvm_f2tf32_rz(1.0f); 
  // CHECK_PTX78_SM90: call i32 @llvm.nvvm.f2tf32.rz.relu(float 1.000000e+00)
  __nvvm_f2tf32_rz_relu(1.0f); 
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_cvt_sm100
__device__ void nvvm_cvt_sm100() {
#if (PTX >= 86) && (__CUDA_ARCH__ >= 1000)
  // CHECK_PTX86_SM100: call i32 @llvm.nvvm.f2tf32.rn.satfinite(float 1.000000e+00)
  __nvvm_f2tf32_rn_satfinite(1.0f); 
  // CHECK_PTX86_SM100: call i32 @llvm.nvvm.f2tf32.rn.relu.satfinite(float 1.000000e+00)
  __nvvm_f2tf32_rn_relu_satfinite(1.0f); 
  // CHECK_PTX86_SM100: call i32 @llvm.nvvm.f2tf32.rz.satfinite(float 1.000000e+00)
  __nvvm_f2tf32_rz_satfinite(1.0f); 
  // CHECK_PTX86_SM100: call i32 @llvm.nvvm.f2tf32.rz.relu.satfinite(float 1.000000e+00)
  __nvvm_f2tf32_rz_relu_satfinite(1.0f); 
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_cvt_sm100a_sm101a_sm120a
__device__ void nvvm_cvt_sm100a_sm101a_sm120a() {
#if (PTX >= 86) && \
    (__CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL || \
     __CUDA_ARCH_FEAT_SM120_ALL)
  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.e2m3x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.e2m3x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.e2m3x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e2m3x2_rn_satfinite(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.e2m3x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.e2m3x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.e2m3x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e2m3x2_rn_relu_satfinite(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.e3m2x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.e3m2x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.e3m2x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e3m2x2_rn_satfinite(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.e3m2x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.e3m2x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.e3m2x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e3m2x2_rn_relu_satfinite(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn(i16 19532)
  // CHECK_PTX86_SM101a: call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn(i16 19532)
  // CHECK_PTX86_SM120a: call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn(i16 19532)
  __nvvm_e2m3x2_to_f16x2_rn(0x4C4C);

  // CHECK_PTX86_SM100a: call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn.relu(i16 18504)
  // CHECK_PTX86_SM101a: call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn.relu(i16 18504)
  // CHECK_PTX86_SM120a: call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn.relu(i16 18504)
  __nvvm_e2m3x2_to_f16x2_rn_relu(0x4848);

  // CHECK_PTX86_SM100a: call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn(i16 18504)
  // CHECK_PTX86_SM101a: call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn(i16 18504)
  // CHECK_PTX86_SM120a: call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn(i16 18504)
  __nvvm_e3m2x2_to_f16x2_rn(0x4848);

  // CHECK_PTX86_SM100a: call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn.relu(i16 19532)
  // CHECK_PTX86_SM101a: call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn.relu(i16 19532)
  // CHECK_PTX86_SM120a: call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn.relu(i16 19532)
  __nvvm_e3m2x2_to_f16x2_rn_relu(0x4C4C);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.e2m1x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.e2m1x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.e2m1x2.rn.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e2m1x2_rn_satfinite(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.e2m1x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.e2m1x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.e2m1x2.rn.relu.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_e2m1x2_rn_relu_satfinite(1.0f, 1.0f);
  
  // CHECK_PTX86_SM100a: call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn(i16 76)
  // CHECK_PTX86_SM101a: call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn(i16 76)
  // CHECK_PTX86_SM120a: call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn(i16 76)
  __nvvm_e2m1x2_to_f16x2_rn(0x004C);
  
  // CHECK_PTX86_SM100a: call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn.relu(i16 76)
  // CHECK_PTX86_SM101a: call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn.relu(i16 76)
  // CHECK_PTX86_SM120a: call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn.relu(i16 76)
  __nvvm_e2m1x2_to_f16x2_rn_relu(0x004C);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rz(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rz(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rz(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_ue8m0x2_rz(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rz.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rz.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rz.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_ue8m0x2_rz_satfinite(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rp(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rp(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rp(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_ue8m0x2_rp(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rp.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rp.satfinite(float 1.000000e+00, float 1.000000e+00)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.ff.to.ue8m0x2.rp.satfinite(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff_to_ue8m0x2_rp_satfinite(1.0f, 1.0f);

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz(<2 x bfloat> splat (bfloat 0xR3DCD)
  __nvvm_bf16x2_to_ue8m0x2_rz({(__bf16)0.1f, (__bf16)0.1f});

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz.satfinite(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz.satfinite(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz.satfinite(<2 x bfloat> splat (bfloat 0xR3DCD)
  __nvvm_bf16x2_to_ue8m0x2_rz_satfinite({(__bf16)0.1f, (__bf16)0.1f});

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp(<2 x bfloat> splat (bfloat 0xR3DCD)
  __nvvm_bf16x2_to_ue8m0x2_rp({(__bf16)0.1f, (__bf16)0.1f});

  // CHECK_PTX86_SM100a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp.satfinite(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM101a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp.satfinite(<2 x bfloat> splat (bfloat 0xR3DCD)
  // CHECK_PTX86_SM120a: call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp.satfinite(<2 x bfloat> splat (bfloat 0xR3DCD)
  __nvvm_bf16x2_to_ue8m0x2_rp_satfinite({(__bf16)0.1f, (__bf16)0.1f});

  // CHECK_PTX86_SM100a: call <2 x bfloat> @llvm.nvvm.ue8m0x2.to.bf16x2(i16 19532)
  // CHECK_PTX86_SM101a: call <2 x bfloat> @llvm.nvvm.ue8m0x2.to.bf16x2(i16 19532)
  // CHECK_PTX86_SM120a: call <2 x bfloat> @llvm.nvvm.ue8m0x2.to.bf16x2(i16 19532)
  __nvvm_ue8m0x2_to_bf16x2(0x4C4C);
  
#endif
  // CHECK: ret void
}

__device__ void nvvm_cvt_sm100a_sm103a() {
#if (PTX >= 87) && (__CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM103_ALL)
  
  typedef __fp16 f16x2 __attribute__((ext_vector_type(2)));
  typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
  typedef char uint8x4 __attribute__((ext_vector_type(4)));

// CHECK_PTX87_SM100a: %[[R1:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x half> %[[R1]], ptr %r1
// CHECK_PTX87_SM103a: %[[R1:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x half> %[[R1]], ptr %r1
  f16x2 r1 =  __nvvm_ff2f16x2_rs(1.0f, 1.0f, 0);
  
// CHECK_PTX87_SM100a: %[[R2:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x half> %[[R2]], ptr %r2
// CHECK_PTX87_SM103a: %[[R2:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x half> %[[R2]], ptr %r2
  f16x2 r2 =  __nvvm_ff2f16x2_rs_relu(1.0f, 1.0f, 0);
  
// CHECK_PTX87_SM100a: %[[R3:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x half> %[[R3]], ptr %r3
// CHECK_PTX87_SM103a: %[[R3:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x half> %[[R3]], ptr %r3
  f16x2 r3 =  __nvvm_ff2f16x2_rs_satfinite(1.0f, 1.0f, 0);

// CHECK_PTX87_SM100a: %[[R4:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x half> %[[R4]], ptr %r4
// CHECK_PTX87_SM103a: %[[R4:.*]] = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x half> %[[R4]], ptr %r4
  f16x2 r4 =  __nvvm_ff2f16x2_rs_relu_satfinite(1.0f, 1.0f, 0);

// CHECK_PTX87_SM100a: %[[R5:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x bfloat> %[[R5]], ptr %r5
// CHECK_PTX87_SM103a: %[[R5:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x bfloat> %[[R5]], ptr %r5
  bf16x2 r5 =  __nvvm_ff2bf16x2_rs(1.0f, 1.0f, 0);

// CHECK_PTX87_SM100a: %[[R6:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x bfloat> %[[R6]], ptr %r6
// CHECK_PTX87_SM103a: %[[R6:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x bfloat> %[[R6]], ptr %r6
  bf16x2 r6 =  __nvvm_ff2bf16x2_rs_relu(1.0f, 1.0f, 0);

// CHECK_PTX87_SM100a: %[[R7:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x bfloat> %[[R7]], ptr %r7
// CHECK_PTX87_SM103a: %[[R7:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x bfloat> %[[R7]], ptr %r7
  bf16x2 r7 =  __nvvm_ff2bf16x2_rs_satfinite(1.0f, 1.0f, 0);

// CHECK_PTX87_SM100a: %[[R8:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM100a: store <2 x bfloat> %[[R8]], ptr %r8
// CHECK_PTX87_SM103a: %[[R8:.*]] = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu.satfinite(float 1.000000e+00, float 1.000000e+00, i32 0)
// CHECK_PTX87_SM103a: store <2 x bfloat> %[[R8]], ptr %r8
  bf16x2 r8 =  __nvvm_ff2bf16x2_rs_relu_satfinite(1.0f, 1.0f, 0);

// CHECK_PTX87_SM100a: %[[R9:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e4m3x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R9]], ptr %r9
// CHECK_PTX87_SM103a: %[[R9:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e4m3x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R9]], ptr %r9
  uint8x4 r9 =  __nvvm_f32x4_to_e4m3x4_rs_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R10:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e4m3x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R10]], ptr %r10
// CHECK_PTX87_SM103a: %[[R10:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e4m3x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R10]], ptr %r10
  uint8x4 r10 =  __nvvm_f32x4_to_e4m3x4_rs_relu_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R11:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e5m2x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R11]], ptr %r11
// CHECK_PTX87_SM103a: %[[R11:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e5m2x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R11]], ptr %r11
  uint8x4 r11 =  __nvvm_f32x4_to_e5m2x4_rs_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R12:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e5m2x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R12]], ptr %r12
// CHECK_PTX87_SM103a: %[[R12:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e5m2x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R12]], ptr %r12
  uint8x4 r12 =  __nvvm_f32x4_to_e5m2x4_rs_relu_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R13:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e2m3x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R13]], ptr %r13
// CHECK_PTX87_SM103a: %[[R13:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e2m3x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R13]], ptr %r13
  uint8x4 r13 = __nvvm_f32x4_to_e2m3x4_rs_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);  

// CHECK_PTX87_SM100a: %[[R14:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e2m3x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R14]], ptr %r14
// CHECK_PTX87_SM103a: %[[R14:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e2m3x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R14]], ptr %r14
  uint8x4 r14 = __nvvm_f32x4_to_e2m3x4_rs_relu_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R15:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e3m2x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R15]], ptr %r15
// CHECK_PTX87_SM103a: %[[R15:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e3m2x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R15]], ptr %r15
  uint8x4 r15 = __nvvm_f32x4_to_e3m2x4_rs_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R16:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e3m2x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store <4 x i8> %[[R16]], ptr %r16
// CHECK_PTX87_SM103a: %[[R16:.*]] = call <4 x i8> @llvm.nvvm.f32x4.to.e3m2x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store <4 x i8> %[[R16]], ptr %r16
  uint8x4 r16 = __nvvm_f32x4_to_e3m2x4_rs_relu_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R17:.*]] = call i16 @llvm.nvvm.f32x4.to.e2m1x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store i16 %[[R17]], ptr %r17
// CHECK_PTX87_SM103a: %[[R17:.*]] = call i16 @llvm.nvvm.f32x4.to.e2m1x4.rs.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store i16 %[[R17]], ptr %r17
  short r17 = __nvvm_f32x4_to_e2m1x4_rs_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);

// CHECK_PTX87_SM100a: %[[R18:.*]] = call i16 @llvm.nvvm.f32x4.to.e2m1x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM100a: store i16 %[[R18]], ptr %r18
// CHECK_PTX87_SM103a: %[[R18:.*]] = call i16 @llvm.nvvm.f32x4.to.e2m1x4.rs.relu.satfinite(<4 x float> splat (float 1.000000e+00), i32 0)
// CHECK_PTX87_SM103a: store i16 %[[R18]], ptr %r18
  short r18 = __nvvm_f32x4_to_e2m1x4_rs_relu_satfinite({1.0f, 1.0f, 1.0f, 1.0f}, 0);
#endif
}

#define NAN32 0x7FBFFFFF
#define NAN16 (__bf16)0x7FBF
#define BF16 (__bf16)0.1f
#define BF16_2 (__bf16)0.2f
#define NANBF16 (__bf16)0xFFC1
#define BF16X2 {(__bf16)0.1f, (__bf16)0.1f}
#define BF16X2_2 {(__bf16)0.2f, (__bf16)0.2f}
#define NANBF16X2 {NANBF16, NANBF16}

// CHECK-LABEL: nvvm_abs_neg_bf16_bf16x2_sm80
__device__ void nvvm_abs_neg_bf16_bf16x2_sm80() {
#if __CUDA_ARCH__ >= 800

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fabs.bf16(bfloat 0xR3DCD)
  __nvvm_abs_bf16(BF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fabs.v2bf16(<2 x bfloat> splat (bfloat 0xR3DCD))
  __nvvm_abs_bf16x2(BF16X2);

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.neg.bf16(bfloat 0xR3DCD)
  __nvvm_neg_bf16(BF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.neg.bf16x2(<2 x bfloat> splat (bfloat 0xR3DCD))
  __nvvm_neg_bf16x2(BF16X2);
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_min_max_sm80
__device__ void nvvm_min_max_sm80() {
#if __CUDA_ARCH__ >= 800

  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmin.nan.f
  __nvvm_fmin_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmin.ftz.nan.f
  __nvvm_fmin_ftz_nan_f(0.1f, (float)NAN32);

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.bf16
  __nvvm_fmin_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.ftz.bf16
  __nvvm_fmin_ftz_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.nan.bf16
  __nvvm_fmin_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.ftz.nan.bf16
  __nvvm_fmin_ftz_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.bf16x2
  __nvvm_fmin_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.ftz.bf16x2
  __nvvm_fmin_ftz_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.nan.bf16x2
  __nvvm_fmin_nan_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.ftz.nan.bf16x2
  __nvvm_fmin_ftz_nan_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.nan.f
  __nvvm_fmax_nan_f(0.1f, 0.11f);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.ftz.nan.f
  __nvvm_fmax_ftz_nan_f(0.1f, (float)NAN32);

  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.nan.f
  __nvvm_fmax_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.ftz.nan.f
  __nvvm_fmax_ftz_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.bf16
  __nvvm_fmax_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.ftz.bf16
  __nvvm_fmax_ftz_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.nan.bf16
  __nvvm_fmax_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.ftz.nan.bf16
  __nvvm_fmax_ftz_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.bf16x2
  __nvvm_fmax_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.ftz.bf16x2
  __nvvm_fmax_ftz_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.nan.bf16x2
  __nvvm_fmax_nan_bf16x2(NANBF16X2, BF16X2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.ftz.nan.bf16x2
  __nvvm_fmax_ftz_nan_bf16x2(NANBF16X2, BF16X2);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.nan.f
  __nvvm_fmax_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.ftz.nan.f
  __nvvm_fmax_ftz_nan_f(0.1f, (float)NAN32);

#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_fma_bf16_bf16x2_sm80
__device__ void nvvm_fma_bf16_bf16x2_sm80() {
#if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fma.rn.bf16
  __nvvm_fma_rn_bf16(BF16, BF16_2, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fma.rn.relu.bf16
  __nvvm_fma_rn_relu_bf16(BF16, BF16_2, BF16_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fma.rn.bf16x2
  __nvvm_fma_rn_bf16x2(BF16X2, BF16X2_2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fma.rn.relu.bf16x2
  __nvvm_fma_rn_relu_bf16x2(BF16X2, BF16X2_2, BF16X2_2);
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_min_max_sm86
__device__ void nvvm_min_max_sm86() {
#if __CUDA_ARCH__ >= 860

  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmin.xorsign.abs.bf16
  __nvvm_fmin_xorsign_abs_bf16(BF16, BF16_2);
  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmin.nan.xorsign.abs.bf16
  __nvvm_fmin_nan_xorsign_abs_bf16(BF16, NANBF16);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmin.xorsign.abs.bf16x2
  __nvvm_fmin_xorsign_abs_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmin.nan.xorsign.abs.bf16x2
  __nvvm_fmin_nan_xorsign_abs_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.xorsign.abs.f
  __nvvm_fmin_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.ftz.xorsign.abs.f
  __nvvm_fmin_ftz_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.nan.xorsign.abs.f
  __nvvm_fmin_nan_xorsign_abs_f(-0.1f, (float)NAN32);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f
  __nvvm_fmin_ftz_nan_xorsign_abs_f(-0.1f, (float)NAN32);

  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmax.xorsign.abs.bf16
  __nvvm_fmax_xorsign_abs_bf16(BF16, BF16_2);
  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmax.nan.xorsign.abs.bf16
  __nvvm_fmax_nan_xorsign_abs_bf16(BF16, NANBF16);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmax.xorsign.abs.bf16x2
  __nvvm_fmax_xorsign_abs_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmax.nan.xorsign.abs.bf16x2
  __nvvm_fmax_nan_xorsign_abs_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.xorsign.abs.f
  __nvvm_fmax_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.ftz.xorsign.abs.f
  __nvvm_fmax_ftz_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.nan.xorsign.abs.f
  __nvvm_fmax_nan_xorsign_abs_f(-0.1f, (float)NAN32);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f
  __nvvm_fmax_ftz_nan_xorsign_abs_f(-0.1f, (float)NAN32);
#endif
  // CHECK: ret void
}
