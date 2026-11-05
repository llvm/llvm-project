// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_60 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DFEATURES -ferror-limit 0 -verify=feature %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx60 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DFEATURES -ferror-limit 0 -verify=feature %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DRESTRICTED_LAYOUTS -DBAD_LAYOUT=0 -verify=layout %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DRESTRICTED_LAYOUTS -DBAD_LAYOUT=1 -verify=layout %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DRESTRICTED_LAYOUTS -DBAD_LAYOUT=2 -verify=layout %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DRESTRICTED_LAYOUTS -DBAD_LAYOUT=3 -verify=layout %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DRESTRICTED_LAYOUTS -DBAD_LAYOUT=4 -verify=layout %s -o /dev/null
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 \
// RUN:   -target-feature +ptx70 -x cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir -DRESTRICTED_LAYOUTS -DBAD_LAYOUT=5 -verify=layout %s -o /dev/null

#define DEVICE extern "C" __attribute__((device))

#ifdef FEATURES
DEVICE void target_features(int *i, float *f, double *d, unsigned stride) {
  // feature-error-re@+1 {{'__hmma_m32n8k16_ld_a' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m32n8k16_ld_a(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m32n8k16_ld_b' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m32n8k16_ld_b(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m32n8k16_ld_c_f16' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m32n8k16_ld_c_f16(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m32n8k16_ld_c_f32' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m32n8k16_ld_c_f32(f, f, stride, 0);
  // feature-error-re@+1 {{'__hmma_m8n32k16_ld_a' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m8n32k16_ld_a(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m8n32k16_ld_b' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m8n32k16_ld_b(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m8n32k16_ld_c_f16' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m8n32k16_ld_c_f16(i, i, stride, 0);
  // feature-error-re@+1 {{'__hmma_m8n32k16_ld_c_f32' needs target feature {{.*}}sm_70{{.*}}ptx61}}
  __hmma_m8n32k16_ld_c_f32(f, f, stride, 0);
  // feature-error-re@+1 {{'__imma_m16n16k16_ld_a_s8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m16n16k16_ld_a_s8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m16n16k16_ld_b_s8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m16n16k16_ld_b_s8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m16n16k16_ld_a_u8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m16n16k16_ld_a_u8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m16n16k16_ld_b_u8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m16n16k16_ld_b_u8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m16n16k16_ld_c' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m16n16k16_ld_c(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m32n8k16_ld_a_s8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m32n8k16_ld_a_s8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m32n8k16_ld_b_s8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m32n8k16_ld_b_s8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m32n8k16_ld_a_u8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m32n8k16_ld_a_u8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m32n8k16_ld_b_u8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m32n8k16_ld_b_u8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m32n8k16_ld_c' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m32n8k16_ld_c(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n32k16_ld_a_s8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m8n32k16_ld_a_s8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n32k16_ld_b_s8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m8n32k16_ld_b_s8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n32k16_ld_a_u8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m8n32k16_ld_a_u8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n32k16_ld_b_u8' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m8n32k16_ld_b_u8(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n32k16_ld_c' needs target feature {{.*}}sm_72{{.*}}ptx63}}
  __imma_m8n32k16_ld_c(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n8k32_ld_a_s4' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __imma_m8n8k32_ld_a_s4(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n8k32_ld_b_s4' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __imma_m8n8k32_ld_b_s4(i, i, stride, 1);
  // feature-error-re@+1 {{'__imma_m8n8k32_ld_a_u4' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __imma_m8n8k32_ld_a_u4(i, i, stride, 0);
  // feature-error-re@+1 {{'__imma_m8n8k32_ld_b_u4' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __imma_m8n8k32_ld_b_u4(i, i, stride, 1);
  // feature-error-re@+1 {{'__imma_m8n8k32_ld_c' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __imma_m8n8k32_ld_c(i, i, stride, 0);
  // feature-error-re@+1 {{'__bmma_m8n8k128_ld_a_b1' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __bmma_m8n8k128_ld_a_b1(i, i, stride, 0);
  // feature-error-re@+1 {{'__bmma_m8n8k128_ld_b_b1' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __bmma_m8n8k128_ld_b_b1(i, i, stride, 1);
  // feature-error-re@+1 {{'__bmma_m8n8k128_ld_c' needs target feature {{.*}}sm_75{{.*}}ptx63}}
  __bmma_m8n8k128_ld_c(i, i, stride, 0);
  // feature-error-re@+1 {{'__dmma_m8n8k4_ld_a' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __dmma_m8n8k4_ld_a(d, d, stride, 0);
  // feature-error-re@+1 {{'__dmma_m8n8k4_ld_b' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __dmma_m8n8k4_ld_b(d, d, stride, 0);
  // feature-error-re@+1 {{'__dmma_m8n8k4_ld_c' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __dmma_m8n8k4_ld_c(d, d, stride, 0);
  // feature-error-re@+1 {{'__mma_bf16_m16n16k16_ld_a' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_bf16_m16n16k16_ld_a(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_bf16_m16n16k16_ld_b' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_bf16_m16n16k16_ld_b(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_bf16_m32n8k16_ld_a' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_bf16_m32n8k16_ld_a(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_bf16_m32n8k16_ld_b' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_bf16_m32n8k16_ld_b(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_bf16_m8n32k16_ld_a' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_bf16_m8n32k16_ld_a(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_bf16_m8n32k16_ld_b' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_bf16_m8n32k16_ld_b(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_tf32_m16n16k8_ld_a' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_tf32_m16n16k8_ld_a(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_tf32_m16n16k8_ld_b' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_tf32_m16n16k8_ld_b(i, i, stride, 0);
  // feature-error-re@+1 {{'__mma_tf32_m16n16k8_ld_c' needs target feature {{.*}}sm_80{{.*}}ptx70}}
  __mma_tf32_m16n16k8_ld_c(f, f, stride, 0);
}
#endif

#ifdef RESTRICTED_LAYOUTS
DEVICE void restricted_layout(int *dst, const int *src, unsigned stride) {
#if BAD_LAYOUT == 0
  // layout-error@+1 {{unimplemented builtin call: __imma_m8n8k32_ld_a_s4}}
  __imma_m8n8k32_ld_a_s4(dst, src, stride, 1);
#elif BAD_LAYOUT == 1
  // layout-error@+1 {{unimplemented builtin call: __imma_m8n8k32_ld_b_s4}}
  __imma_m8n8k32_ld_b_s4(dst, src, stride, 0);
#elif BAD_LAYOUT == 2
  // layout-error@+1 {{unimplemented builtin call: __imma_m8n8k32_ld_a_u4}}
  __imma_m8n8k32_ld_a_u4(dst, src, stride, 1);
#elif BAD_LAYOUT == 3
  // layout-error@+1 {{unimplemented builtin call: __imma_m8n8k32_ld_b_u4}}
  __imma_m8n8k32_ld_b_u4(dst, src, stride, 0);
#elif BAD_LAYOUT == 4
  // layout-error@+1 {{unimplemented builtin call: __bmma_m8n8k128_ld_a_b1}}
  __bmma_m8n8k128_ld_a_b1(dst, src, stride, 1);
#elif BAD_LAYOUT == 5
  // layout-error@+1 {{unimplemented builtin call: __bmma_m8n8k128_ld_b_b1}}
  __bmma_m8n8k128_ld_b_b1(dst, src, stride, 0);
#endif
}
#endif
