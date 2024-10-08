#ifndef _ARM64_
#define _ARM64_
#endif
#include <processthreadsapi.h>
#include <stdint.h>

void __init_cpu_features_resolver(unsigned long hwcap,
                                  const __ifunc_arg_t *arg) {}

void  CONSTRUCTOR_ATTRIBUTE __init_cpu_features(void) {
  if (__atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED))
    return;

  #define setCPUFeature(F) features |= 1ULL << F

  uint64_t features = 0;

  setCPUFeature(FEAT_INIT);
  setCPUFeature(FEAT_FP);

  // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
  if (IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE))
    setCPUFeature(FEAT_CRC);
  if (IsProcessorFeaturePresent(PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE))
    setCPUFeature(FEAT_LSE);
  if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE))
    setCPUFeature(FEAT_DOTPROD);

  if (IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE)) {
    setCPUFeature(FEAT_AES);
    setCPUFeature(FEAT_SHA2);
    setCPUFeature(FEAT_PMULL);
  }
  if (IsProcessorFeaturePresent(PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE))
    setCPUFeature(FEAT_JSCVT);

  if (IsProcessorFeaturePresent(PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE))
    setCPUFeature(FEAT_RCPC);

  __atomic_store(&__aarch64_cpu_features.features, &features,
                  __ATOMIC_RELAXED);
}
