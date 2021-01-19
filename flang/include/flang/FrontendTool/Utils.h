//===--- Utils.h - Misc utilities for the flang front-end --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for various front-end actions
//  which were split from Frontend to minimise Frontend's dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_FRONTENDTOOL_UTILS_H
#define LLVM_FLANG_FRONTENDTOOL_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace Fortran::frontend {

class CompilerInstance;
class FrontendAction;

/// Construct the FrontendAction of a compiler invocation based on the
/// options specified for the compiler invocation.
///
/// \return - The created FrontendAction object
std::unique_ptr<FrontendAction> CreateFrontendAction(CompilerInstance &ci);

/// ExecuteCompilerInvocation - Execute the given actions described by the
/// compiler invocation object in the given compiler instance.
///
/// \return - True on success.
bool ExecuteCompilerInvocation(CompilerInstance *flang);

/// \param suffix The file extension
/// \return True if the file extension should be processed as fixed form
bool isFixedFormSuffix(llvm::StringRef suffix);

/// \param suffix The file extension
/// \return True if the file extension should be processed as free form
bool isFreeFormSuffix(llvm::StringRef suffix);

} // end namespace Fortran::frontend

#endif // LLVM_FLANG_FRONTENDTOOL_UTILS_H
