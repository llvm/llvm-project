//===- AARch64.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <cstdint>
#include "llvm/Support/AArch64AttributeParser.h"
#include "llvm/Support/AArch64BuildAttributes.h"

struct AArch64BuildAttrSubsections {
  struct PauthSubSection {
    uint64_t tagPlatform = 0;
    uint64_t tagSchema = 0;
  } pauth;
  uint32_t andFeatures = 0;
};

AArch64BuildAttrSubsections
extractBuildAttributesSubsections(const llvm::AArch64AttributeParser&);
