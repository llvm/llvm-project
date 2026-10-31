// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_60 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DFEATURES -verify=feature %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx42 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DFEATURES -verify=feature %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DNEGATIVE_LAYOUT -verify=layout %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DLARGE_LAYOUT -verify=layout %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DCONSTANTS -verify=constant %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device \
// RUN:   -fsyntax-only -DCONSTANTS -verify=constant %s

#define DEVICE extern "C" __attribute__((device))

#ifdef FEATURES
DEVICE void features(int *i, float *f, unsigned stride) {
  // feature-error-re@+1 {{'__hmma_m16n16k16_ld_a' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_ld_a(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_ld_b' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_ld_b(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_ld_c_f16' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_ld_c_f16(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_ld_c_f32' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_ld_c_f32(f, f, stride, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_st_c_f16' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_st_c_f16(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_st_c_f32' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_st_c_f32(f, f, stride, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_mma_f16f16' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_mma_f16f16(i, i, i, i, 0, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_mma_f32f16' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_mma_f32f16(f, i, i, i, 0, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_mma_f16f32' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_mma_f16f32(i, i, i, f, 0, 0);
  // feature-error-re@+1 {{'__hmma_m16n16k16_mma_f32f32' needs target feature {{.*}}sm_70{{.*}}ptx60}}
  __hmma_m16n16k16_mma_f32f32(f, i, i, f, 0, 0);
}
#endif

// Reject out-of-range compute layouts without indexing past the layout table.
#ifdef NEGATIVE_LAYOUT
DEVICE void negative_layout(int *p) {
  // layout-error@+1 {{unimplemented builtin call: __hmma_m16n16k16_mma_f16f16}}
  __hmma_m16n16k16_mma_f16f16(p, p, p, p, -1, 0);
}
#endif
#ifdef LARGE_LAYOUT
DEVICE void large_layout(int *p) {
  // layout-error@+1 {{unimplemented builtin call: __hmma_m16n16k16_mma_f16f16}}
  __hmma_m16n16k16_mma_f16f16(p, p, p, p, 4, 0);
}
#endif

#ifdef CONSTANTS
DEVICE void constant_arguments(int *p, int layout, int satfinite) {
  // constant-error@+1 {{argument to '__hmma_m16n16k16_ld_a' must be a constant integer}}
  __hmma_m16n16k16_ld_a(p, p, 16, layout);
  // constant-error@+1 {{argument to '__hmma_m16n16k16_st_c_f16' must be a constant integer}}
  __hmma_m16n16k16_st_c_f16(p, p, 16, layout);
  // constant-error@+1 {{argument to '__hmma_m16n16k16_mma_f16f16' must be a constant integer}}
  __hmma_m16n16k16_mma_f16f16(p, p, p, p, layout, 0);
  // constant-error@+1 {{argument to '__hmma_m16n16k16_mma_f16f16' must be a constant integer}}
  __hmma_m16n16k16_mma_f16f16(p, p, p, p, 0, satfinite);
}
#endif

