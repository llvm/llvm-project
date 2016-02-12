// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Make sure we handle target overloads correctly.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:     -fcuda-target-overloads -emit-llvm -o - %s \
// RUN:     | FileCheck -check-prefix=CHECK-BOTH -check-prefix=CHECK-HOST %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:     -fcuda-target-overloads -emit-llvm -o - %s \
// RUN:     | FileCheck -check-prefix=CHECK-BOTH -check-prefix=CHECK-DEVICE \
// RUN:       -check-prefix=CHECK-DEVICE-STRICT %s

// Check target overloads handling with disabled call target checks.
// RUN: %clang_cc1 -DNOCHECKS -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:    -fcuda-disable-target-call-checks -fcuda-target-overloads -o - %s \
// RUN:     | FileCheck -check-prefix=CHECK-BOTH -check-prefix=CHECK-HOST \
// RUN:    -check-prefix=CHECK-BOTH-NC -check-prefix=CHECK-HOST-NC %s
// RUN: %clang_cc1 -DNOCHECKS -triple nvptx64-nvidia-cuda -emit-llvm \
// RUN:    -fcuda-disable-target-call-checks -fcuda-target-overloads \
// RUN:    -fcuda-is-device -o - %s \
// RUN:     | FileCheck -check-prefix=CHECK-BOTH -check-prefix=CHECK-DEVICE \
// RUN:    -check-prefix=CHECK-BOTH-NC -check-prefix=CHECK-DEVICE-NC %s

#include "Inputs/cuda.h"

typedef int (*fp_t)(void);
typedef void (*gp_t)(void);

// CHECK-HOST: @hp = global i32 ()* @_Z1hv
// CHECK-HOST: @chp = global i32 ()* @ch
// CHECK-HOST: @dhp = global i32 ()* @_Z2dhv
// CHECK-HOST: @cdhp = global i32 ()* @cdh
// CHECK-HOST: @gp = global void ()* @_Z1gv

// CHECK-BOTH-LABEL: define i32 @_Z2dhv()
__device__ int dh(void) { return 1; }
// CHECK-DEVICE: ret i32 1
__host__ int dh(void) { return 2; }
// CHECK-HOST:   ret i32 2

// CHECK-BOTH-LABEL: define i32 @_Z2hdv()
__host__ __device__ int hd(void) { return 3; }
// CHECK-BOTH:   ret i32 3

// CHECK-DEVICE-LABEL: define i32 @_Z1dv()
__device__ int d(void) { return 8; }
// CHECK-DEVICE:   ret i32 8

// CHECK-HOST-LABEL: define i32 @_Z1hv()
__host__ int h(void) { return 9; }
// CHECK-HOST:   ret i32 9

// CHECK-BOTH-LABEL: define void @_Z1gv()
__global__ void g(void) {}
// CHECK-BOTH:   ret void

// mangled names of extern "C" __host__ __device__ functions clash
// with those of their __host__/__device__ counterparts, so
// overloading of extern "C" functions can only happen for __host__
// and __device__ functions -- we never codegen them in the same
// compilation and therefore mangled name conflict is not a problem.

// CHECK-BOTH-LABEL: define i32 @cdh()
extern "C" __device__ int cdh(void) {return 10;}
// CHECK-DEVICE:   ret i32 10
extern "C" __host__ int cdh(void) {return 11;}
// CHECK-HOST:     ret i32 11

// CHECK-DEVICE-LABEL: define i32 @cd()
extern "C" __device__ int cd(void) {return 12;}
// CHECK-DEVICE:   ret i32 12

// CHECK-HOST-LABEL: define i32 @ch()
extern "C" __host__ int ch(void) {return 13;}
// CHECK-HOST:     ret i32 13

// CHECK-BOTH-LABEL: define i32 @chd()
extern "C" __host__ __device__ int chd(void) {return 14;}
// CHECK-BOTH:     ret i32 14

// HD functions are sometimes allowed to call H or D functions -- this
// is an artifact of the source-to-source splitting performed by nvcc
// that we need to mimic. During device mode compilation in nvcc, host
// functions aren't present at all, so don't participate in
// overloading. But in clang, H and D functions are present in both
// compilation modes. Clang normally uses the target attribute as a
// tiebreaker between overloads with otherwise identical priority, but
// in order to match nvcc's behavior, we sometimes need to wholly
// discard overloads that would not be present during compilation
// under nvcc.

template <typename T> T template_vs_function(T arg) { return 15; }
__device__ float template_vs_function(float arg) { return 16; }

// Here we expect to call the templated function during host
// compilation, even if -fcuda-disable-target-call-checks is passed,
// and even though C++ overload rules prefer the non-templated
// function.
// CHECK-BOTH-LABEL: define void @_Z5hd_tfv()
__host__ __device__ void hd_tf(void) {
  template_vs_function(1.0f);
  // CHECK-HOST: call float @_Z20template_vs_functionIfET_S0_(float
  // CHECK-DEVICE: call float @_Z20template_vs_functionf(float
  template_vs_function(2.0);
  // CHECK-HOST: call double @_Z20template_vs_functionIdET_S0_(double
  // CHECK-DEVICE: call float @_Z20template_vs_functionf(float
}

// Calls from __host__ and __device__ functions should always call the
// overloaded function that matches their mode.
// CHECK-HOST-LABEL: define void @_Z4h_tfv()
__host__ void h_tf() {
  template_vs_function(1.0f);
  // CHECK-HOST: call float @_Z20template_vs_functionIfET_S0_(float
  template_vs_function(2.0);
  // CHECK-HOST: call double @_Z20template_vs_functionIdET_S0_(double
}

// CHECK-DEVICE-LABEL: define void @_Z4d_tfv()
__device__ void d_tf() {
  template_vs_function(1.0f);
  // CHECK-DEVICE: call float @_Z20template_vs_functionf(float
  template_vs_function(2.0);
  // CHECK-DEVICE: call float @_Z20template_vs_functionf(float
}

// In case we have a mix of HD and H-only or D-only candidates in the
// overload set, normal C++ overload resolution rules apply first.
template <typename T> T template_vs_hd_function(T arg) { return 15; }
__host__ __device__ float template_vs_hd_function(float arg) { return 16; }

// CHECK-BOTH-LABEL: define void @_Z7hd_thdfv()
__host__ __device__ void hd_thdf() {
  template_vs_hd_function(1.0f);
  // CHECK-HOST: call float @_Z23template_vs_hd_functionf(float
  // CHECK-DEVICE: call float @_Z23template_vs_hd_functionf(float
  template_vs_hd_function(1);
  // CHECK-HOST: call i32 @_Z23template_vs_hd_functionIiET_S0_(i32
  // CHECK-DEVICE-STRICT: call float @_Z23template_vs_hd_functionf(float
  // CHECK-DEVICE-NC: call i32 @_Z23template_vs_hd_functionIiET_S0_(i32
}

// CHECK-HOST-LABEL: define void @_Z6h_thdfv()
__host__ void h_thdf() {
  template_vs_hd_function(1.0f);
  // CHECK-HOST: call float @_Z23template_vs_hd_functionf(float
  template_vs_hd_function(1);
  // CHECK-HOST: call i32 @_Z23template_vs_hd_functionIiET_S0_(i32
}

// CHECK-DEVICE-LABEL: define void @_Z6d_thdfv()
__device__ void d_thdf() {
  template_vs_hd_function(1.0f);
  // CHECK-DEVICE: call float @_Z23template_vs_hd_functionf(float
  template_vs_hd_function(1);
  // Host-only function template is not callable with strict call checks,
  // so for device side HD function will be the only choice.
  // CHECK-DEVICE: call float @_Z23template_vs_hd_functionf(float
}

// Check that overloads still work the same way on both host and
// device side when the overload set contains only functions from one
// side of compilation.
__device__ float device_only_function(int arg) { return 17; }
__device__ float device_only_function(float arg) { return 18; }

__host__ float host_only_function(int arg) { return 19; }
__host__ float host_only_function(float arg) { return 20; }

// CHECK-BOTH-LABEL: define void @_Z6hd_dofv()
__host__ __device__ void hd_dof() {
#ifdef NOCHECKS
  device_only_function(1.0f);
  // CHECK-BOTH-NC: call float @_Z20device_only_functionf(float
  device_only_function(1);
  // CHECK-BOTH-NC: call float @_Z20device_only_functioni(i32
  host_only_function(1.0f);
  // CHECK-BOTH-NC: call float @_Z18host_only_functionf(float
  host_only_function(1);
  // CHECK-BOTH-NC: call float @_Z18host_only_functioni(i32
#endif
}


// CHECK-HOST-LABEL: define void @_Z5hostfv()
__host__ void hostf(void) {
  fp_t hp = h; // CHECK-HOST: store {{.*}} @_Z1hv, {{.*}} %hp,
  fp_t chp = ch; // CHECK-HOST: store {{.*}} @ch, {{.*}} %chp,
  fp_t dhp = dh; // CHECK-HOST: store {{.*}} @_Z2dhv, {{.*}} %dhp,
  fp_t cdhp = cdh; // CHECK-HOST: store {{.*}} @cdh, {{.*}} %cdhp,
  fp_t hdp = hd; // CHECK-HOST: store {{.*}} @_Z2hdv, {{.*}} %hdp,
  fp_t chdp = chd; // CHECK-HOST: store {{.*}} @chd, {{.*}} %chdp,
  gp_t gp = g; // CHECK-HOST: store {{.*}} @_Z1gv, {{.*}} %gp,

  h();     // CHECK-HOST: call i32 @_Z1hv()
  ch();    // CHECK-HOST: call i32 @ch()
  dh();    // CHECK-HOST: call i32 @_Z2dhv()
  cdh();   // CHECK-HOST: call i32 @cdh()
  g<<<0,0>>>();  // CHECK-HOST: call void @_Z1gv()
}

// CHECK-DEVICE-LABEL: define void @_Z7devicefv()
__device__ void devicef(void) {
  fp_t dp = d;   // CHECK-DEVICE: store {{.*}} @_Z1dv, {{.*}} %dp,
  fp_t cdp = cd; // CHECK-DEVICE: store {{.*}} @cd, {{.*}} %cdp,
  fp_t dhp = dh; // CHECK-DEVICE: store {{.*}} @_Z2dhv, {{.*}} %dhp,
  fp_t cdhp = cdh; // CHECK-DEVICE: store {{.*}} @cdh, {{.*}} %cdhp,
  fp_t hdp = hd; // CHECK-DEVICE: store {{.*}} @_Z2hdv, {{.*}} %hdp,
  fp_t chdp = chd; // CHECK-DEVICE: store {{.*}} @chd, {{.*}} %chdp,

  d();     // CHECK-DEVICE: call i32 @_Z1dv()
  cd();    // CHECK-DEVICE: call i32 @cd()
  dh();    // CHECK-DEVICE: call i32 @_Z2dhv()
  cdh();   // CHECK-DEVICE: call i32 @cdh()
}

// CHECK-BOTH-LABEL: define void @_Z11hostdevicefv()
__host__ __device__ void hostdevicef(void) {
#if defined (NOCHECKS)
  fp_t dp = d;   // CHECK-BOTH-NC: store {{.*}} @_Z1dv, {{.*}} %dp,
  fp_t cdp = cd; // CHECK-BOTH-NC: store {{.*}} @cd, {{.*}} %cdp,
  fp_t hp = h; // CHECK-BOTH-NC: store {{.*}} @_Z1hv, {{.*}} %hp,
  fp_t chp = ch; // CHECK-BOTH-NC: store {{.*}} @ch, {{.*}} %chp,
#endif
  fp_t dhp = dh; // CHECK-BOTH: store {{.*}} @_Z2dhv, {{.*}} %dhp,
  fp_t cdhp = cdh; // CHECK-BOTH: store {{.*}} @cdh, {{.*}} %cdhp,
  fp_t hdp = hd; // CHECK-BOTH: store {{.*}} @_Z2hdv, {{.*}} %hdp,
  fp_t chdp = chd; // CHECK-BOTH: store {{.*}} @chd, {{.*}} %chdp,
#if defined (NOCHECKS) && !defined(__CUDA_ARCH__)
  gp_t gp = g; // CHECK-HOST-NC: store {{.*}} @_Z1gv, {{.*}} %gp,
#endif

#if defined (NOCHECKS)
  d();     // CHECK-BOTH-NC: call i32 @_Z1dv()
  cd();    // CHECK-BOTH-NC: call i32 @cd()
  h();     // CHECK-BOTH-NC: call i32 @_Z1hv()
  ch();    // CHECK-BOTH-NC: call i32 @ch()
#endif
  dh();    // CHECK-BOTH: call i32 @_Z2dhv()
  cdh();   // CHECK-BOTH: call i32 @cdh()
#if defined (NOCHECKS) && !defined(__CUDA_ARCH__)
  g<<<0,0>>>();  // CHECK-HOST-NC: call void @_Z1gv()
#endif
}

// Test for address of overloaded function resolution in the global context.
fp_t hp = h;
fp_t chp = ch;
fp_t dhp = dh;
fp_t cdhp = cdh;
gp_t gp = g;

int x;
// Check constructors/destructors for D/H functions
struct s_cd_dh {
  __host__ s_cd_dh() { x = 11; }
  __device__ s_cd_dh() { x = 12; }
  __host__ ~s_cd_dh() { x = 21; }
  __device__ ~s_cd_dh() { x = 22; }
};

struct s_cd_hd {
  __host__ __device__ s_cd_hd() { x = 31; }
  __host__ __device__ ~s_cd_hd() { x = 32; }
};

// CHECK-BOTH: define void @_Z7wrapperv
#if defined(__CUDA_ARCH__)
__device__
#else
__host__
#endif
void wrapper() {
  s_cd_dh scddh;
  // CHECK-BOTH: call void @_ZN7s_cd_dhC1Ev(
  s_cd_hd scdhd;
  // CHECK-BOTH: call void @_ZN7s_cd_hdC1Ev

  // CHECK-BOTH: call void @_ZN7s_cd_hdD1Ev(
  // CHECK-BOTH: call void @_ZN7s_cd_dhD1Ev(
}
// CHECK-BOTH: ret void

// Now it's time to check what's been generated for the methods we used.

// CHECK-BOTH: define linkonce_odr void @_ZN7s_cd_dhC2Ev(
// CHECK-HOST:   store i32 11,
// CHECK-DEVICE: store i32 12,
// CHECK-BOTH: ret void

// CHECK-BOTH: define linkonce_odr void @_ZN7s_cd_hdC2Ev(
// CHECK-BOTH:   store i32 31,
// CHECK-BOTH: ret void

// CHECK-BOTH: define linkonce_odr void @_ZN7s_cd_hdD2Ev(
// CHECK-BOTH: store i32 32,
// CHECK-BOTH: ret void

// CHECK-BOTH: define linkonce_odr void @_ZN7s_cd_dhD2Ev(
// CHECK-HOST:   store i32 21,
// CHECK-DEVICE: store i32 22,
// CHECK-BOTH: ret void

