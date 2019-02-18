//===----- LayoutFieldRandomizer.h - Entry Point for Randstruct --*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file provides the entry point for the Randstruct structure
// layout randomization code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_AST_LAYOUTFIELDRANDOMIZER_H
#define LLVM_CLANG_LIB_AST_LAYOUTFIELDRANDOMIZER_H

#include "clang/AST/AST.h"

namespace clang {
/// Rearranges the order of the supplied fields. Will make best effort to fit
// members into a cache line.
SmallVector<Decl *, 64> rearrange(const ASTContext &ctx,
                                  SmallVector<Decl *, 64> fields);
} // namespace clang

#endif
