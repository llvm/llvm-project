//===--- ExpandAutoType.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "refactor/Tweak.h"

#include "Logger.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include <climits>
#include <memory>
#include <string>
#include <AST.h>
#include "XRefs.h"
#include "llvm/ADT/StringExtras.h"

namespace clang {
namespace clangd {
namespace {

/// Expand the "auto" type to the derived type
/// Before:
///    auto x = Something();
///    ^^^^
/// After:
///    MyClass x = Something();
///    ^^^^^^^
/// FIXME: Handle decltype as well
class ExpandAutoType : public Tweak {
public:
  const char *id() const final;
  Intent intent() const override { return Intent::Refactor;}
  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override;

private:
  /// Cache the AutoTypeLoc, so that we do not need to search twice.
  llvm::Optional<clang::AutoTypeLoc> CachedLocation;

  /// Create an error message with filename and line number in it
  llvm::Error createErrorMessage(const std::string& Message,
                                 const Selection &Inputs);

};

REGISTER_TWEAK(ExpandAutoType)

std::string ExpandAutoType::title() const { return "Expand auto type"; }

bool ExpandAutoType::prepare(const Selection& Inputs) {
  CachedLocation = llvm::None;
  if (auto *Node = Inputs.ASTSelection.commonAncestor()) {
    if (auto *TypeNode = Node->ASTNode.get<TypeLoc>()) {
      if (const AutoTypeLoc Result = TypeNode->getAs<AutoTypeLoc>()) {
        // Code in apply() does handle 'decltype(auto)' yet.
        if (!Result.getTypePtr()->isDecltypeAuto())
          CachedLocation = Result;
      }
    }
  }
  return (bool) CachedLocation;
}

Expected<Tweak::Effect> ExpandAutoType::apply(const Selection& Inputs) {
  auto& SrcMgr = Inputs.AST.getASTContext().getSourceManager();

  llvm::Optional<clang::QualType> DeducedType =
      getDeducedType(Inputs.AST.getASTContext(), CachedLocation->getBeginLoc());

  // if we can't resolve the type, return an error message
  if (DeducedType == llvm::None)
    return createErrorMessage("Could not deduce type for 'auto' type", Inputs);

  // if it's a lambda expression, return an error message
  if (isa<RecordType>(*DeducedType) &&
      dyn_cast<RecordType>(*DeducedType)->getDecl()->isLambda()) {
    return createErrorMessage("Could not expand type of lambda expression",
                              Inputs);
  }

  // if it's a function expression, return an error message
  // naively replacing 'auto' with the type will break declarations.
  // FIXME: there are other types that have similar problems
  if (DeducedType->getTypePtr()->isFunctionPointerType()) {
    return createErrorMessage("Could not expand type of function pointer",
                              Inputs);
  }

  std::string PrettyTypeName = printType(*DeducedType,
      Inputs.ASTSelection.commonAncestor()->getDeclContext());

  tooling::Replacement
      Expansion(SrcMgr, CharSourceRange(CachedLocation->getSourceRange(), true),
                PrettyTypeName);

  return Effect::mainFileEdit(SrcMgr, tooling::Replacements(Expansion));
}

llvm::Error ExpandAutoType::createErrorMessage(const std::string& Message,
                                               const Selection& Inputs) {
  auto& SrcMgr = Inputs.AST.getASTContext().getSourceManager();
  std::string ErrorMessage =
      Message + ": " +
          SrcMgr.getFilename(Inputs.Cursor).str() + " Line " +
          std::to_string(SrcMgr.getExpansionLineNumber(Inputs.Cursor));

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 ErrorMessage.c_str());
}

} // namespace
} // namespace clangd
} // namespace clang
