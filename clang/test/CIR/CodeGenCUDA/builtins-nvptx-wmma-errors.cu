// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_60 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DFEATURES -verify=feature %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx42 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DFEATURES -verify=feature %s -o /dev/null
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
}
#endif

#ifdef CONSTANTS
DEVICE void constant_arguments(int *p, int layout) {
  // constant-error@+1 {{argument to '__hmma_m16n16k16_ld_a' must be a constant integer}}
  __hmma_m16n16k16_ld_a(p, p, 16, layout);
}
#endif
