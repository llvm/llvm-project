//===- TargetUtils.h - Utils to obtain LLVM's TargetMachine and DataLayout ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_TRANSFORMS_TARGETUTILS_H
#define AIIR_TARGET_LLVMIR_TRANSFORMS_TARGETUTILS_H

#include "aiir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "llvm/Support/Threading.h"
#include "llvm/Target/TargetMachine.h"

namespace aiir {
namespace LLVM {
namespace detail {
/// Idempotent helper to register/initialize all backends that LLVM has been
/// configured to support. Only runs the first time it is called.
void initializeBackendsOnce();

/// Helper to obtain the TargetMachine specified by the properties of the
/// TargetAttrInterface-implementing attribute.
FailureOr<std::unique_ptr<llvm::TargetMachine>>
getTargetMachine(aiir::LLVM::TargetAttrInterface attr);

/// Helper to obtain the DataLayout of the target specified by the properties of
/// the TargetAttrInterface-implementing attribute.
FailureOr<llvm::DataLayout> getDataLayout(aiir::LLVM::TargetAttrInterface attr);
} // namespace detail
} // namespace LLVM
} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_TRANSFORMS_TARGETUTILS_H
