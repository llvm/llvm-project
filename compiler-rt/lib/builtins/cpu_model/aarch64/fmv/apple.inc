#include <TargetConditionals.h>
#if TARGET_OS_OSX || TARGET_OS_IPHONE
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>

static bool isKnownAndSupported(const char *name) {
  int32_t val = 0;
  size_t size = sizeof(val);
  if (sysctlbyname(name, &val, &size, NULL, 0))
    return false;
  return val;
}

void __init_cpu_features_resolver(void) {
  // On Darwin platforms, this may be called concurrently by multiple threads
  // because the resolvers that use it are called lazily at runtime (unlike on
  // ELF platforms, where IFuncs are resolved serially at load time). This
  // function's effect on __aarch64_cpu_features must be idempotent.

  if (__atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED))
    return;

  uint64_t features = 0;

  // https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics
  static const struct {
    const char *sysctl_name;
    enum CPUFeatures feature;
  } feature_checks[] = {
      {"hw.optional.arm.FEAT_FlagM", FEAT_FLAGM},
      {"hw.optional.arm.FEAT_FlagM2", FEAT_FLAGM2},
      {"hw.optional.arm.FEAT_FHM", FEAT_FP16FML},
      {"hw.optional.arm.FEAT_DotProd", FEAT_DOTPROD},
      {"hw.optional.arm.FEAT_RDM", FEAT_RDM},
      {"hw.optional.arm.FEAT_LSE", FEAT_LSE},
      {"hw.optional.floatingpoint", FEAT_FP},
      {"hw.optional.AdvSIMD", FEAT_SIMD},
      {"hw.optional.armv8_crc32", FEAT_CRC},
      {"hw.optional.arm.FEAT_SHA1", FEAT_SHA1},
      {"hw.optional.arm.FEAT_SHA256", FEAT_SHA2},
      {"hw.optional.arm.FEAT_SHA3", FEAT_SHA3},
      {"hw.optional.arm.FEAT_AES", FEAT_AES},
      {"hw.optional.arm.FEAT_PMULL", FEAT_PMULL},
      {"hw.optional.arm.FEAT_FP16", FEAT_FP16},
      {"hw.optional.arm.FEAT_DIT", FEAT_DIT},
      {"hw.optional.arm.FEAT_DPB", FEAT_DPB},
      {"hw.optional.arm.FEAT_DPB2", FEAT_DPB2},
      {"hw.optional.arm.FEAT_JSCVT", FEAT_JSCVT},
      {"hw.optional.arm.FEAT_FCMA", FEAT_FCMA},
      {"hw.optional.arm.FEAT_LRCPC", FEAT_RCPC},
      {"hw.optional.arm.FEAT_LRCPC2", FEAT_RCPC2},
      {"hw.optional.arm.FEAT_FRINTTS", FEAT_FRINTTS},
      {"hw.optional.arm.FEAT_I8MM", FEAT_I8MM},
      {"hw.optional.arm.FEAT_BF16", FEAT_BF16},
      {"hw.optional.arm.FEAT_SB", FEAT_SB},
      {"hw.optional.arm.FEAT_SPECRES", FEAT_PREDRES},
      {"hw.optional.arm.FEAT_SSBS", FEAT_SSBS2},
      {"hw.optional.arm.FEAT_BTI", FEAT_BTI},
  };

  for (size_t I = 0, E = sizeof(feature_checks) / sizeof(feature_checks[0]);
        I != E; ++I)
    if (isKnownAndSupported(feature_checks[I].sysctl_name))
      features |= (1ULL << feature_checks[I].feature);

  features |= (1ULL << FEAT_INIT);

  __atomic_store(&__aarch64_cpu_features.features, &features,
                  __ATOMIC_RELAXED);
}

#endif // TARGET_OS_OSX || TARGET_OS_IPHONE
