#if __has_include(<sys/auxv.h>)
#include <sys/auxv.h>
#define HAVE_SYS_AUXV_H
#endif

static void __init_cpu_features_constructor(unsigned long hwcap,
                                            const __ifunc_arg_t *arg) {
  unsigned long long feat[2] = {0, 0};
#define setCPUFeature(F) feat[(F) / 64] |= 1ULL << ((F) % 64)
  unsigned long hwcap2 = 0;
  if (hwcap & _IFUNC_ARG_HWCAP)
    hwcap2 = arg->_hwcap2;
  if (hwcap & HWCAP_CRC32)
    setCPUFeature(FEAT_CRC);
  if (hwcap & HWCAP_PMULL)
    setCPUFeature(FEAT_PMULL);
  if (hwcap & HWCAP_FLAGM)
    setCPUFeature(FEAT_FLAGM);
  if (hwcap2 & HWCAP2_FLAGM2)
    setCPUFeature(FEAT_FLAGM2);
  if (hwcap & HWCAP_SM4)
    setCPUFeature(FEAT_SM4);
  if (hwcap & HWCAP_ASIMDDP)
    setCPUFeature(FEAT_DOTPROD);
  if (hwcap & HWCAP_ASIMDFHM)
    setCPUFeature(FEAT_FP16FML);
  if (hwcap & HWCAP_FPHP)
    setCPUFeature(FEAT_FP16);
  if (hwcap & HWCAP_DIT)
    setCPUFeature(FEAT_DIT);
  if (hwcap & HWCAP_ASIMDRDM)
    setCPUFeature(FEAT_RDM);
  if (hwcap & HWCAP_SHA2)
    setCPUFeature(FEAT_SHA2);
  if (hwcap & HWCAP_JSCVT)
    setCPUFeature(FEAT_JSCVT);
  if (hwcap & HWCAP_FCMA)
    setCPUFeature(FEAT_FCMA);
  if (hwcap & HWCAP_SB)
    setCPUFeature(FEAT_SB);
  if (hwcap & HWCAP_SSBS)
    setCPUFeature(FEAT_SSBS2);
  if (hwcap2 & HWCAP2_MTE)
    setCPUFeature(FEAT_MEMTAG2);
  if (hwcap2 & HWCAP2_SVEPMULL)
    setCPUFeature(FEAT_SVE_PMULL128);
  if (hwcap2 & HWCAP2_SVEBITPERM)
    setCPUFeature(FEAT_SVE_BITPERM);
  if (hwcap2 & HWCAP2_SVESHA3)
    setCPUFeature(FEAT_SVE_SHA3);
  if (hwcap2 & HWCAP2_SVESM4)
    setCPUFeature(FEAT_SVE_SM4);
  if (hwcap2 & HWCAP2_DCPODP)
    setCPUFeature(FEAT_DPB2);
  if (hwcap & HWCAP_ATOMICS)
    setCPUFeature(FEAT_LSE);
  if (hwcap2 & HWCAP2_RNG)
    setCPUFeature(FEAT_RNG);
  if (hwcap2 & HWCAP2_I8MM)
    setCPUFeature(FEAT_I8MM);
  if (hwcap2 & HWCAP2_FRINT)
    setCPUFeature(FEAT_FRINTTS);
  if (hwcap2 & HWCAP2_SVEF32MM)
    setCPUFeature(FEAT_SVE_F32MM);
  if (hwcap2 & HWCAP2_SVEF64MM)
    setCPUFeature(FEAT_SVE_F64MM);
  if (hwcap2 & HWCAP2_BTI)
    setCPUFeature(FEAT_BTI);
  if (hwcap2 & HWCAP2_WFXT)
    setCPUFeature(FEAT_WFXT);
  if (hwcap2 & HWCAP2_SME)
    setCPUFeature(FEAT_SME);
  if (hwcap2 & HWCAP2_SME2)
    setCPUFeature(FEAT_SME2);
  if (hwcap2 & HWCAP2_SME_I16I64)
    setCPUFeature(FEAT_SME_I64);
  if (hwcap2 & HWCAP2_SME_F64F64)
    setCPUFeature(FEAT_SME_F64);
  if (hwcap2 & HWCAP2_MOPS)
    setCPUFeature(FEAT_MOPS);
  if (hwcap2 & HWCAP2_CSSC)
    setCPUFeature(FEAT_CSSC);
  if ((hwcap2 & HWCAP2_F8CVT) && (hwcap2 & HWCAP2_F8E4M3) &&
      (hwcap2 & HWCAP2_F8E5M2))
    setCPUFeature(FEAT_FP8);
  if (hwcap & HWCAP_F8MM8)
    setCPUFeature(FEAT_F8F32MM);
  if (hwcap2 & HWCAP2_F8DP4)
    setCPUFeature(FEAT_FP8DOT4);
  if (hwcap2 & HWCAP2_F8DP2)
    setCPUFeature(FEAT_FP8DOT2);
  if (hwcap2 & HWCAP2_F8FMA)
    setCPUFeature(FEAT_FP8FMA);
  if (hwcap2 & HWCAP2_SME_F8F32)
    setCPUFeature(FEAT_SME_F8F32);
  if (hwcap2 & HWCAP2_SME_SF8DP4)
    setCPUFeature(FEAT_SSVE_FP8DOT4);
  if (hwcap2 & HWCAP2_SME_SF8FMA)
    setCPUFeature(FEAT_SSVE_FP8FMA);
  if (hwcap & HWCAP_SME_SBITPERM)
    setCPUFeature(FEAT_SSVE_BITPERM);
  if (hwcap2 & HWCAP2_SME_SF8DP2)
    setCPUFeature(FEAT_SSVE_FP8DOT2);
  if (hwcap & HWCAP_SME_AES)
    setCPUFeature(FEAT_SSVE_AES);
  if (hwcap & HWCAP_SME_SFEXPA)
    setCPUFeature(FEAT_SSVE_FEXPA);
  if (hwcap2 & HWCAP2_LUT)
    setCPUFeature(FEAT_LUT);
  if (hwcap2 & HWCAP2_FAMINMAX)
    setCPUFeature(FEAT_FAMINMAX);
  if (hwcap2 & HWCAP2_SME_LUTV2)
    setCPUFeature(FEAT_SME_LUTV2);
  if (hwcap2 & HWCAP2_SME2P1)
    setCPUFeature(FEAT_SME2P1);
  if (hwcap & HWCAP_SME2P2)
    setCPUFeature(FEAT_SME2P2);
  if (hwcap2 & HWCAP2_SVE2P1)
    setCPUFeature(FEAT_SVE2P1);
  if (hwcap & HWCAP_SVE2P2)
    setCPUFeature(FEAT_SVE2P2);
  if (hwcap2 & HWCAP2_SME_F16F16)
    setCPUFeature(FEAT_SME_F16F16);
  if (hwcap & HWCAP_GCS)
    setCPUFeature(FEAT_GCS);
  if (hwcap2 & HWCAP2_SME_F8F16)
    setCPUFeature(FEAT_SME_F8F16);
  if (hwcap & HWCAP_F8MM4)
    setCPUFeature(FEAT_F8F16MM);
  if (hwcap & HWCAP_SVE_AES2)
    setCPUFeature(FEAT_SVE_AES2);
  if (hwcap & HWCAP_SVE_BFSCALE)
    setCPUFeature(FEAT_SVE_BFSCALE);
  if (hwcap & HWCAP_SVE_F16MM)
    setCPUFeature(FEAT_SVE_F16F32MM);
  if (hwcap & HWCAP_SME_SMOP4)
    setCPUFeature(FEAT_SME_MOP4);
  if (hwcap & HWCAP_FP) {
    setCPUFeature(FEAT_FP);
    // FP and AdvSIMD fields have the same value
    setCPUFeature(FEAT_SIMD);
  }
  if (hwcap & HWCAP_DCPOP)
    setCPUFeature(FEAT_DPB);
  if (hwcap & HWCAP_LRCPC)
    setCPUFeature(FEAT_RCPC);
  if (hwcap & HWCAP_ILRCPC)
    setCPUFeature(FEAT_RCPC2);
  if (hwcap2 & HWCAP2_LRCPC3)
    setCPUFeature(FEAT_RCPC3);
  if (hwcap2 & HWCAP2_BF16)
    setCPUFeature(FEAT_BF16);
  if (hwcap & HWCAP_SVE)
    setCPUFeature(FEAT_SVE);
  if (hwcap2 & HWCAP2_SVE2)
    setCPUFeature(FEAT_SVE2);
  if (hwcap & HWCAP_SHA3)
    setCPUFeature(FEAT_SHA3);
  setCPUFeature(FEAT_EXT);
  setCPUFeature(FEAT_INIT);

  __atomic_store_n(&__aarch64_cpu_features.features2, feat[1],
                   __ATOMIC_RELAXED);
  __atomic_store_n(&__aarch64_cpu_features.features, feat[0],
                   __ATOMIC_RELEASE);
}
