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
  // ELF platforms, where IFuncs are resolved serially at load time).  This
  // function's effect on __aarch64_cpu_features should be idempotent, but even
  // so we need dispatch_once to resolve the race condition.  Dispatch is
  // available through libSystem, which we need anyway for the sysctl, so this
  // does not add a new dependency.

  static dispatch_once_t onceToken = 0;
  dispatch_once(&onceToken, ^{
    // https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics
    static const struct {
      const char *sysctl_name;
      enum CPUFeatures feature;
    } features[] = {
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

    for (size_t I = 0, E = sizeof(features) / sizeof(features[0]); I != E; ++I)
      if (isKnownAndSupported(features[I].sysctl_name))
        __aarch64_cpu_features.features |= (1ULL << features[I].feature);

    __aarch64_cpu_features.features |= (1ULL << FEAT_INIT);
  });
}

#endif // TARGET_OS_OSX || TARGET_OS_IPHONE
