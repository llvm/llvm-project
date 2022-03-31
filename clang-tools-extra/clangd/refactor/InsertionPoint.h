//===--- InsertionPoint.h - Where should we add new code? --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Tooling/Core/Replacement.h"

namespace clang {
namespace clangd {

// An anchor describes where to insert code into a decl sequence.
//
// It allows inserting above or below a block of decls matching some criterion.
// For example, "insert after existing constructors".
struct Anchor {
  // A predicate describing which decls are considered part of a block.
  // Match need not handle TemplateDecls, which are unwrapped before matching.
  std::function<bool(const Decl *)> Match;
  // Whether the insertion point should be before or after the matching block.
  enum Dir { Above, Below } Direction = Below;
};

// Returns the point to insert a declaration according to Anchors.
// Anchors are tried in order. For each, the first matching location is chosen.
SourceLocation insertionPoint(const DeclContext &Ctx,
                              llvm::ArrayRef<Anchor> Anchors);

// Returns an edit inserting Code inside Ctx.
// Location is chosen according to Anchors, falling back to the end of Ctx.
// Fails if the chosen insertion point is in a different file than Ctx itself.
llvm::Expected<tooling::Replacement> insertDecl(llvm::StringRef Code,
                                                const DeclContext &Ctx,
                                                llvm::ArrayRef<Anchor> Anchors);

// Variant for C++ classes that ensures the right access control.
SourceLocation insertionPoint(const CXXRecordDecl &InClass,
                              std::vector<Anchor> Anchors,
                              AccessSpecifier Protection);

// Variant for C++ classes that ensures the right access control.
// May insert a new access specifier if needed.
llvm::Expected<tooling::Replacement> insertDecl(llvm::StringRef Code,
                                                const CXXRecordDecl &InClass,
                                                std::vector<Anchor> Anchors,
                                                AccessSpecifier Protection);

} // namespace clangd
} // namespace clang
