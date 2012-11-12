// REQUIRES: nvptx-registered-target
// REQUIRES: nvptx64-registered-target
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

int read_tid() {

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

int read_ntid() {

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

int read_ctaid() {

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

int read_nctaid() {

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

int read_ids() {

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

int read_lanemasks() {

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


long read_clocks() {

// CHECK: call i32 @llvm.ptx.read.clock()
// CHECK: call i64 @llvm.ptx.read.clock64()

  int a = __builtin_ptx_read_clock();
  long b = __builtin_ptx_read_clock64();

  return (long)a + b;

}

int read_pms() {

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

void sync() {

// CHECK: call void @llvm.ptx.bar.sync(i32 0)

  __builtin_ptx_bar_sync(0);

}


// NVVM intrinsics

// The idea is not to test all intrinsics, just that Clang is recognizing the
// builtins defined in BuiltinsNVPTX.def
void nvvm_math(float f1, float f2, double d1, double d2) {
// CHECK: call float @llvm.nvvm.fmax.f
  float t1 = __nvvm_fmax_f(f1, f2);
// CHECK: call float @llvm.nvvm.fmin.f
  float t2 = __nvvm_fmin_f(f1, f2);
// CHECK: call float @llvm.nvvm.sqrt.rn.f
  float t3 = __nvvm_sqrt_rn_f(f1);
// CHECK: call float @llvm.nvvm.rcp.rn.f
  float t4 = __nvvm_rcp_rn_f(f2);

// CHECK: call double @llvm.nvvm.fmax.d
  double td1 = __nvvm_fmax_d(d1, d2);
// CHECK: call double @llvm.nvvm.fmin.d
  double td2 = __nvvm_fmin_d(d1, d2);
// CHECK: call double @llvm.nvvm.sqrt.rn.d
  double td3 = __nvvm_sqrt_rn_d(d1);
// CHECK: call double @llvm.nvvm.rcp.rn.d
  double td4 = __nvvm_rcp_rn_d(d2);
}
