//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_ARM64_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_ARM64_H

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Flags.h"
#include <utility>

namespace lldb_private {

class RegisterDescription_arm64 : public RegisterDescription {
public:
  struct TypeFeatures {
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

  /// Architectural inputs for one immutable register description. Saved-state
  /// acquisition and operating-system layouts are handled by the adapters.
  struct Config {
    Flags features;
    uint32_t sve_vq = 0;
    uint32_t za_vq = 0;
    const TypeFeatures *type_features = nullptr;

    static constexpr uint32_t InitialSVEVectorQuadwords = 1;
    static constexpr uint32_t MaxVectorQuadwords = 256;

    static bool VectorSizeIsValid(uint32_t vq) {
      return vq <= MaxVectorQuadwords;
    }
  };

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

  explicit RegisterDescription_arm64(const Config &config);
};

} // namespace lldb_private

#endif
