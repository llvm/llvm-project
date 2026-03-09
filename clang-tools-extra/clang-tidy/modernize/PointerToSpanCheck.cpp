//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointerToSpanCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

/// Return true if \p T is an unsigned integer type commonly used for sizes
/// (size_t, unsigned, unsigned long, etc.).
static bool isSizeType(QualType T) {
  T = T.getCanonicalType();
  if (const auto *BT = T->getAs<BuiltinType>()) {
    switch (BT->getKind()) {
    case BuiltinType::UInt:
    case BuiltinType::ULong:
    case BuiltinType::ULongLong:
    case BuiltinType::Int:
    case BuiltinType::Long:
    case BuiltinType::LongLong:
      return true;
    default:
      return false;
    }
  }
  return false;
}

/// Return true if the parameter name suggests it represents a size or count.
static bool isSizeName(StringRef Name) {
  static const llvm::DenseSet<StringRef> ExactSizeNames = {
      "size",   "len",   "length",       "count",     "n",  "num",
      "nelems", "nelem", "num_elements", "num_elems", "sz", "cnt",
  };

  static constexpr llvm::StringLiteral SizeSuffixes[] = {
      "_size", "_len", "_length", "_count", "size", "len", "count", "num",
  };

  const std::string LowerStorage = Name.lower();
  const StringRef Lower(LowerStorage);
  return ExactSizeNames.contains(Lower) ||
         llvm::any_of(SizeSuffixes, [Lower](StringRef Suffix) {
           return Lower.ends_with(Suffix);
         });
}

void PointerToSpanCheck::registerMatchers(MatchFinder *Finder) {
  // Match function declarations (not just definitions) with at least 2 params.
  Finder->addMatcher(
      functionDecl(
          parameterCountIs(2),
          hasParameter(0, parmVarDecl(hasType(pointerType())).bind("ptr")),
          hasParameter(1, parmVarDecl().bind("size")), unless(isImplicit()),
          unless(isDeleted()))
          .bind("func"),
      this);

  // Also match functions with more params -- look for consecutive ptr+size.
  Finder->addMatcher(functionDecl(unless(parameterCountIs(2)),
                                  unless(isImplicit()), unless(isDeleted()))
                         .bind("funcN"),
                     this);
}

void PointerToSpanCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &SM = *Result.SourceManager;

  // Handle the exact 2-parameter case.
  if (const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func")) {
    const auto *PtrParam = Result.Nodes.getNodeAs<ParmVarDecl>("ptr");
    const auto *SizeParam = Result.Nodes.getNodeAs<ParmVarDecl>("size");

    if (!PtrParam || !SizeParam)
      return;
    if (SM.isInSystemHeader(Func->getLocation()))
      return;

    // Skip virtual methods.
    if (const auto *MD = dyn_cast<CXXMethodDecl>(Func))
      if (MD->isVirtual())
        return;

    // Skip main.
    if (Func->isMain())
      return;

    // Pointer must not be void* or function pointer.
    const auto *PT = PtrParam->getType()->getAs<PointerType>();
    if (!PT || PT->getPointeeType()->isVoidType() ||
        PT->getPointeeType()->isFunctionType())
      return;

    // Size param must be an integer type.
    if (!isSizeType(SizeParam->getType()))
      return;

    // Heuristic: size param name should suggest a size.
    // If unnamed, skip -- cannot verify intent.
    if (!SizeParam->getIdentifier() || !isSizeName(SizeParam->getName()))
      return;

    diag(PtrParam->getLocation(),
         "pointer and size parameters can be replaced with 'std::span'")
        << PtrParam->getSourceRange();
    diag(SizeParam->getLocation(), "size parameter declared here",
         DiagnosticIDs::Note)
        << SizeParam->getSourceRange();
    return;
  }

  // Handle N-parameter functions: scan for consecutive (ptr, size) pairs.
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("funcN");
  if (!Func)
    return;
  if (SM.isInSystemHeader(Func->getLocation()))
    return;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(Func))
    if (MD->isVirtual())
      return;
  if (Func->isMain())
    return;

  for (unsigned I = 0; I + 1 < Func->getNumParams(); ++I) {
    const ParmVarDecl *PtrParam = Func->getParamDecl(I);
    const ParmVarDecl *SizeParam = Func->getParamDecl(I + 1);

    const auto *PT = PtrParam->getType()->getAs<PointerType>();
    if (!PT || PT->getPointeeType()->isVoidType() ||
        PT->getPointeeType()->isFunctionType())
      continue;

    if (!isSizeType(SizeParam->getType()))
      continue;

    if (!SizeParam->getIdentifier() || !isSizeName(SizeParam->getName()))
      continue;

    diag(PtrParam->getLocation(),
         "pointer and size parameters can be replaced with 'std::span'")
        << PtrParam->getSourceRange();
    diag(SizeParam->getLocation(), "size parameter declared here",
         DiagnosticIDs::Note)
        << SizeParam->getSourceRange();
  }
}

} // namespace clang::tidy::modernize
