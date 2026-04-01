//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains helpers for OpenACC emission that don't need to be in
// CIRGenModule, but can't live in a single .cpp file.
//
//===----------------------------------------------------------------------===//
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/DeclOpenACC.h"

namespace clang::CIRGen {
inline aiir::acc::DataClauseModifier
convertOpenACCModifiers(OpenACCModifierKind modifiers) {
  using namespace aiir::acc;
  static_assert(static_cast<int>(OpenACCModifierKind::Zero) ==
                    static_cast<int>(DataClauseModifier::zero) &&
                static_cast<int>(OpenACCModifierKind::Readonly) ==
                    static_cast<int>(DataClauseModifier::readonly) &&
                static_cast<int>(OpenACCModifierKind::AlwaysIn) ==
                    static_cast<int>(DataClauseModifier::alwaysin) &&
                static_cast<int>(OpenACCModifierKind::AlwaysOut) ==
                    static_cast<int>(DataClauseModifier::alwaysout) &&
                static_cast<int>(OpenACCModifierKind::Capture) ==
                    static_cast<int>(DataClauseModifier::capture));

  DataClauseModifier aiirModifiers{};

  // The AIIR representation of this represents `always` as `alwaysin` +
  // `alwaysout`.  So do a small fixup here.
  if (isOpenACCModifierBitSet(modifiers, OpenACCModifierKind::Always)) {
    aiirModifiers = aiirModifiers | DataClauseModifier::always;
    modifiers &= ~OpenACCModifierKind::Always;
  }

  aiirModifiers = aiirModifiers | static_cast<DataClauseModifier>(modifiers);
  return aiirModifiers;
}

inline aiir::acc::DeviceType decodeDeviceType(const IdentifierInfo *ii) {
  // '*' case leaves no identifier-info, just a nullptr.
  if (!ii)
    return aiir::acc::DeviceType::Star;
  return llvm::StringSwitch<aiir::acc::DeviceType>(ii->getName())
      .CaseLower("default", aiir::acc::DeviceType::Default)
      .CaseLower("host", aiir::acc::DeviceType::Host)
      .CaseLower("multicore", aiir::acc::DeviceType::Multicore)
      .CasesLower({"nvidia", "acc_device_nvidia"},
                  aiir::acc::DeviceType::Nvidia)
      .CaseLower("radeon", aiir::acc::DeviceType::Radeon);
}
} // namespace clang::CIRGen
