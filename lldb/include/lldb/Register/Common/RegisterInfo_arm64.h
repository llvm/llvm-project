//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_ARM64_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_ARM64_H
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Flags.h"
namespace lldb_private {
struct Arm64RegisterFeatures {
  bool fp16 = false;
  bool dit = false;
  bool ssbs = false;
  bool gcs = false;
  bool bti = false;
  bool mte = false;
  bool afp = false;
  bool sme = false;
  bool ebf16 = false;
  bool fpmr = false;
  bool poe = false;
  bool pan = false;
  bool uao = false;
  bool nmi = false;
};

enum class SVEState : uint8_t {
  // We have yet to look what features there are.
  Unknown,
  // We know that there is no SVE or streaming SVE (SME).
  Disabled,
  // We are in non-streaming mode but SVE is not active.
  FPSIMD,
  // We are in non-streaming mode and SVE is active.
  Full,
  // We are in streaming mode using streaming SVE.
  Streaming,
  // We are in non-streaming mode, and only have SVE while in streaming mode.
  StreamingFPSIMD
};

class RegisterInfo_arm64 {
public:
  enum { GPRegSet = 0, FPRegSet };
  // AArch64 register set mask value
  enum {
    eRegsetMaskDefault = 0,
    eRegsetMaskSVE = 1,
    eRegsetMaskSSVE = 2,
    eRegsetMaskPAuth = 4,
    eRegsetMaskMTE = 8,
    eRegsetMaskTLS = 16,
    eRegsetMaskZA = 32,
    eRegsetMaskZT = 64,
    eRegsetMaskFPMR = 128,
    eRegsetMaskGCS = 256,
    eRegsetMaskPOE = 512,
    eRegsetMaskDynamic = ~1,
  };

  // AArch64 Register set FP/SIMD feature configuration
  enum {
    eVectorQuadwordAArch64,
    eVectorQuadwordAArch64SVE,
    eVectorQuadwordAArch64SVEMax = 256
  };

  static std::shared_ptr<const RegisterDescription>
  CreateDescription(
      Flags features, uint32_t sve_vq = 0, uint32_t za_vq = 0,
      const Arm64RegisterFeatures *type_features = nullptr);
  static bool VectorSizeIsValid(uint32_t vq) {
    return vq <= eVectorQuadwordAArch64SVEMax;
  }
  static bool IsGPR(const RegisterDescription &description,
                    unsigned reg);
  static bool IsFPR(const RegisterDescription &description,
                    unsigned reg);
  static bool IsSVEReg(const RegisterDescription &description,
                       unsigned reg);
  static bool IsSVEZReg(unsigned reg);
  static bool IsSVEPReg(unsigned reg);
  static bool IsSVERegVG(unsigned reg);
  static bool IsSVERegFFR(unsigned reg);
  static bool IsPAuthReg(const RegisterDescription &description,
                         unsigned reg);
  static bool IsMTEReg(const RegisterDescription &description,
                       unsigned reg);
  static bool IsTLSReg(const RegisterDescription &description,
                       unsigned reg);
  static bool IsSMEReg(const RegisterDescription &description,
                       unsigned reg);
  static bool IsSMERegZA(const RegisterDescription &description,
                         unsigned reg);
  static bool IsSMERegZT(const RegisterDescription &description,
                         unsigned reg);
  static bool IsFPMRReg(const RegisterDescription &description,
                        unsigned reg);
  static bool IsGCSReg(const RegisterDescription &description,
                       unsigned reg);
  static bool IsPOEReg(const RegisterDescription &description,
                       unsigned reg);

  static uint32_t GetRegNumSVEZ0();
  static uint32_t GetRegNumSVEFFR();
  static uint32_t GetRegNumFPCR();
  static uint32_t GetRegNumFPSR();
  static uint32_t GetRegNumFPV0();
  static uint32_t GetRegNumSVEVG();
  static uint32_t
  GetRegNumSMESVG(const RegisterDescription &description);

private:
  static bool InGroup(const RegisterDescription &description,
                      unsigned number, llvm::StringRef group);
  static uint32_t Number(const RegisterDescription &description,
                         llvm::StringRef name);
};
} // namespace lldb_private

#endif
