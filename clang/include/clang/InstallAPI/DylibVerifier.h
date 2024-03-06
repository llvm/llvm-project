//===- InstallAPI/DylibVerifier.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_DYLIBVERIFIER_H
#define LLVM_CLANG_INSTALLAPI_DYLIBVERIFIER_H

#include "llvm/TextAPI/Target.h"

namespace clang {
namespace installapi {

/// A list of InstallAPI verification modes.
enum class VerificationMode {
  Invalid,
  ErrorsOnly,
  ErrorsAndWarnings,
  Pedantic,
};

} // namespace installapi
} // namespace clang
#endif // LLVM_CLANG_INSTALLAPI_DYLIBVERIFIER_H
