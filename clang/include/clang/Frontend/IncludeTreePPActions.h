//===- IncludeTreePPActions.h - PP actions using include-tree ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Uses the info from an include-tree to drive the preprocessor via
// \p PPCachedActions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_INCLUDETREEPPACTIONS_H
#define LLVM_CLANG_FRONTEND_INCLUDETREEPPACTIONS_H

#include "clang/Basic/LLVM.h"

namespace clang {

class PPCachedActions;

namespace cas {
class IncludeTreeRoot;
}

Expected<std::unique_ptr<PPCachedActions>>
createPPActionsFromIncludeTree(cas::IncludeTreeRoot &Root);

} // namespace clang

#endif
