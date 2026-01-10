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
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/DeclOpenACC.h"

namespace clang::CIRGen {
inline mlir::acc::DataClauseModifier
convertOpenACCModifiers(OpenACCModifierKind modifiers) {
  using namespace mlir::acc;
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

  DataClauseModifier mlirModifiers{};

  // The MLIR representation of this represents `always` as `alwaysin` +
  // `alwaysout`.  So do a small fixup here.
  if (isOpenACCModifierBitSet(modifiers, OpenACCModifierKind::Always)) {
    mlirModifiers = mlirModifiers | DataClauseModifier::always;
    modifiers &= ~OpenACCModifierKind::Always;
  }

  mlirModifiers = mlirModifiers | static_cast<DataClauseModifier>(modifiers);
  return mlirModifiers;
}

inline mlir::acc::DeviceType decodeDeviceType(const IdentifierInfo *ii) {
  // '*' case leaves no identifier-info, just a nullptr.
  if (!ii)
    return mlir::acc::DeviceType::Star;
  return llvm::StringSwitch<mlir::acc::DeviceType>(ii->getName())
      .CaseLower("default", mlir::acc::DeviceType::Default)
      .CaseLower("host", mlir::acc::DeviceType::Host)
      .CaseLower("multicore", mlir::acc::DeviceType::Multicore)
      .CasesLower({"nvidia", "acc_device_nvidia"},
                  mlir::acc::DeviceType::Nvidia)
      .CaseLower("radeon", mlir::acc::DeviceType::Radeon);
}
} // namespace clang::CIRGen
