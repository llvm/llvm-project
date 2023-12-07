// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:   %s -S -emit-llvm -o - | FileCheck %s -check-prefix=CHECK

// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:   -S -o - %s | FileCheck -check-prefix=GFX90A %s

// REQUIRES: amdgpu-registered-target

typedef half __attribute__((ext_vector_type(2))) half2;

// CHECK-LABEL: test_global_add_f64
// CHECK: call double @llvm.amdgcn.global.atomic.fadd.f64.p1.f64(ptr addrspace(1) %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_global_add_f64$local:
// GFX90A:  global_atomic_add_f64
void test_global_add_f64(__global double *addr, double x) {
  double *rtn;
  *rtn = __builtin_amdgcn_global_atomic_fadd_f64(addr, x);
}

// CHECK-LABEL: test_global_add_half2
// CHECK: call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) %{{.*}}, <2 x half> %{{.*}})
// GFX90A-LABEL:  test_global_add_half2
// GFX90A:  global_atomic_pk_add_f16 v2, v[0:1], v2, off glc
void test_global_add_half2(__global half2 *addr, half2 x) {
  half2 *rtn;
  *rtn = __builtin_amdgcn_global_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_global_global_min_f64
// CHECK: call double @llvm.amdgcn.global.atomic.fmin.f64.p1.f64(ptr addrspace(1) %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_global_global_min_f64$local
// GFX90A:  global_atomic_min_f64
void test_global_global_min_f64(__global double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_global_atomic_fmin_f64(addr, x);
}

// CHECK-LABEL: test_global_max_f64
// CHECK: call double @llvm.amdgcn.global.atomic.fmax.f64.p1.f64(ptr addrspace(1) %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_global_max_f64$local
// GFX90A:  global_atomic_max_f64
void test_global_max_f64(__global double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_global_atomic_fmax_f64(addr, x);
}

// CHECK-LABEL: test_flat_add_local_f64
// CHECK: call double @llvm.amdgcn.flat.atomic.fadd.f64.p3.f64(ptr addrspace(3) %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_flat_add_local_f64$local
// GFX90A:  ds_add_rtn_f64
void test_flat_add_local_f64(__local double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_flat_atomic_fadd_f64(addr, x);
}

// CHECK-LABEL: test_flat_global_add_f64
// CHECK: call double @llvm.amdgcn.flat.atomic.fadd.f64.p1.f64(ptr addrspace(1) %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_flat_global_add_f64$local
// GFX90A:  global_atomic_add_f64
void test_flat_global_add_f64(__global double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_flat_atomic_fadd_f64(addr, x);
}

// CHECK-LABEL: test_flat_min_flat_f64
// CHECK: call double @llvm.amdgcn.flat.atomic.fmin.f64.p0.f64(ptr %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_flat_min_flat_f64$local
// GFX90A:  flat_atomic_min_f64
void test_flat_min_flat_f64(__generic double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_flat_atomic_fmin_f64(addr, x);
}

// CHECK-LABEL: test_flat_global_min_f64
// CHECK: call double @llvm.amdgcn.flat.atomic.fmin.f64.p1.f64(ptr addrspace(1) %{{.*}}, double %{{.*}})
// GFX90A:  test_flat_global_min_f64$local
// GFX90A:  global_atomic_min_f64
void test_flat_global_min_f64(__global double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_flat_atomic_fmin_f64(addr, x);
}

// CHECK-LABEL: test_flat_max_flat_f64
// CHECK: call double @llvm.amdgcn.flat.atomic.fmax.f64.p0.f64(ptr %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_flat_max_flat_f64$local
// GFX90A:  flat_atomic_max_f64
void test_flat_max_flat_f64(__generic double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_flat_atomic_fmax_f64(addr, x);
}

// CHECK-LABEL: test_flat_global_max_f64
// CHECK: call double @llvm.amdgcn.flat.atomic.fmax.f64.p1.f64(ptr addrspace(1) %{{.*}}, double %{{.*}})
// GFX90A-LABEL:  test_flat_global_max_f64$local
// GFX90A:  global_atomic_max_f64
void test_flat_global_max_f64(__global double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_flat_atomic_fmax_f64(addr, x);
}

// CHECK-LABEL: test_ds_add_local_f64
// CHECK: call double @llvm.amdgcn.ds.fadd.f64(ptr addrspace(3) %{{.*}}, double %{{.*}},
// GFX90A:  test_ds_add_local_f64$local
// GFX90A:  ds_add_rtn_f64
void test_ds_add_local_f64(__local double *addr, double x){
  double *rtn;
  *rtn = __builtin_amdgcn_ds_atomic_fadd_f64(addr, x);
}

// CHECK-LABEL: test_ds_addf_local_f32
// CHECK: call float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %{{.*}}, float %{{.*}},
// GFX90A-LABEL:  test_ds_addf_local_f32$local
// GFX90A:  ds_add_rtn_f32
void test_ds_addf_local_f32(__local float *addr, float x){
  float *rtn;
  *rtn = __builtin_amdgcn_ds_atomic_fadd_f32(addr, x);
}

// CHECK-LABEL: @test_global_add_f32
// CHECK: call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) %{{.*}}, float %{{.*}})
void test_global_add_f32(float *rtn, global float *addr, float x) {
  *rtn = __builtin_amdgcn_global_atomic_fadd_f32(addr, x);
}
