// REQUIRES: aarch64-target-arch
// REQUIRES: librt_has_aarch64
// REQUIRES: native-run
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include "cpu_model/AArch64CPUFeatures.inc"
#include <stdio.h>

extern struct {
  unsigned long long features;
} __aarch64_cpu_features;

// For these targets __aarch64_cpu_features initialization
// needs to be manually triggered.
#if defined(__APPLE__) || defined(__Fuchsia__)
void __init_cpu_features_resolver(void);
#endif

static int check_feature(const char *name, enum CPUFeatures feature,
                         int supported) {
  // A set feature bit must include all dependencies checked by the builtin.
  int detected = (__aarch64_cpu_features.features >> feature) & 1;
  supported = supported != 0;
  if (supported == detected)
    return 0;

  printf("%s: feature detected = %d, builtin support = %d\n", name, detected,
         supported);
  return 1;
}

int main(void) {
#if defined(__APPLE__) || defined(__Fuchsia__)
  __init_cpu_features_resolver();
#endif

  int failed = 0;
#define CHECK_FEATURE(name, feature)                                           \
  failed |= check_feature(name, feature, __builtin_cpu_supports(name))

  // Check runtime feature bit and use the builtin to verify that transitive
  // dependencies for features are present on the host.
  CHECK_FEATURE("rng", FEAT_RNG);
  CHECK_FEATURE("flagm", FEAT_FLAGM);
  CHECK_FEATURE("flagm2", FEAT_FLAGM2);
  CHECK_FEATURE("fp16fml", FEAT_FP16FML);
  CHECK_FEATURE("dotprod", FEAT_DOTPROD);
  CHECK_FEATURE("sm4", FEAT_SM4);
  CHECK_FEATURE("rdm", FEAT_RDM);
  CHECK_FEATURE("lse", FEAT_LSE);
  CHECK_FEATURE("fp", FEAT_FP);
  CHECK_FEATURE("simd", FEAT_SIMD);
  CHECK_FEATURE("crc", FEAT_CRC);
  CHECK_FEATURE("cssc", FEAT_CSSC);
  CHECK_FEATURE("sha2", FEAT_SHA2);
  CHECK_FEATURE("sha3", FEAT_SHA3);
  CHECK_FEATURE("aes", FEAT_PMULL);
  CHECK_FEATURE("fp16", FEAT_FP16);
  CHECK_FEATURE("dit", FEAT_DIT);
  CHECK_FEATURE("dpb", FEAT_DPB);
  CHECK_FEATURE("dpb2", FEAT_DPB2);
  CHECK_FEATURE("jscvt", FEAT_JSCVT);
  CHECK_FEATURE("fcma", FEAT_FCMA);
  CHECK_FEATURE("rcpc", FEAT_RCPC);
  CHECK_FEATURE("rcpc2", FEAT_RCPC2);
  CHECK_FEATURE("frintts", FEAT_FRINTTS);
  CHECK_FEATURE("i8mm", FEAT_I8MM);
  CHECK_FEATURE("bf16", FEAT_BF16);
  CHECK_FEATURE("sve", FEAT_SVE);
  CHECK_FEATURE("f32mm", FEAT_SVE_F32MM);
  CHECK_FEATURE("f64mm", FEAT_SVE_F64MM);
  CHECK_FEATURE("sve2", FEAT_SVE2);
  CHECK_FEATURE("sve2-aes", FEAT_SVE_PMULL128);
  CHECK_FEATURE("sve2-bitperm", FEAT_SVE_BITPERM);
  CHECK_FEATURE("sve2-sha3", FEAT_SVE_SHA3);
  CHECK_FEATURE("sve2-sm4", FEAT_SVE_SM4);
  CHECK_FEATURE("sme", FEAT_SME);
  CHECK_FEATURE("memtag", FEAT_MEMTAG2);
  CHECK_FEATURE("sb", FEAT_SB);
  CHECK_FEATURE("ssbs", FEAT_SSBS2);
  CHECK_FEATURE("bti", FEAT_BTI);
  CHECK_FEATURE("wfxt", FEAT_WFXT);
  CHECK_FEATURE("sme-f64f64", FEAT_SME_F64);
  CHECK_FEATURE("sme-i16i64", FEAT_SME_I64);
  CHECK_FEATURE("sme2", FEAT_SME2);
  CHECK_FEATURE("rcpc3", FEAT_RCPC3);
  CHECK_FEATURE("mops", FEAT_MOPS);

#undef CHECK_FEATURE
  return failed;
}
