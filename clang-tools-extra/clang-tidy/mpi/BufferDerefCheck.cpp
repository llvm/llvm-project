//===--- BufferDerefCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BufferDerefCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace mpi {

void BufferDerefCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(callExpr().bind("CE"), this);
}

void BufferDerefCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CE = Result.Nodes.getNodeAs<CallExpr>("CE");
  if (!CE->getDirectCallee())
    return;

  if (!FuncClassifier)
    FuncClassifier.emplace(*Result.Context);

  const IdentifierInfo *Identifier = CE->getDirectCallee()->getIdentifier();
  if (!Identifier || !FuncClassifier->isMPIType(Identifier))
    return;

  // These containers are used, to capture the type and expression of a buffer.
  SmallVector<const Type *, 1> BufferTypes;
  SmallVector<const Expr *, 1> BufferExprs;

  // Adds the type and expression of a buffer that is used in the MPI call
  // expression to the captured containers.
  auto AddBuffer = [&CE, &Result, &BufferTypes,
                    &BufferExprs](const size_t BufferIdx) {
    // Skip null pointer constants and in place 'operators'.
    if (CE->getArg(BufferIdx)->isNullPointerConstant(
            *Result.Context, Expr::NPC_ValueDependentIsNull) ||
        tooling::fixit::getText(*CE->getArg(BufferIdx), *Result.Context) ==
            "MPI_IN_PLACE")
      return;

    const Expr *ArgExpr = CE->getArg(BufferIdx);
    if (!ArgExpr)
      return;
    const Type *ArgType = ArgExpr->IgnoreImpCasts()->getType().getTypePtr();
    if (!ArgType)
      return;
    BufferExprs.push_back(ArgExpr);
    BufferTypes.push_back(ArgType);
  };

  // Collect buffer types and argument expressions for all buffers used in the
  // MPI call expression. The number passed to the lambda corresponds to the
  // argument index of the currently verified MPI function call.
  if (FuncClassifier->isPointToPointType(Identifier)) {
    AddBuffer(0);
  } else if (FuncClassifier->isCollectiveType(Identifier)) {
    if (FuncClassifier->isReduceType(Identifier)) {
      AddBuffer(0);
      AddBuffer(1);
    } else if (FuncClassifier->isScatterType(Identifier) ||
               FuncClassifier->isGatherType(Identifier) ||
               FuncClassifier->isAlltoallType(Identifier)) {
      AddBuffer(0);
      AddBuffer(3);
    } else if (FuncClassifier->isBcastType(Identifier)) {
      AddBuffer(0);
    }
  }

  checkBuffers(BufferTypes, BufferExprs);
}

void BufferDerefCheck::checkBuffers(ArrayRef<const Type *> BufferTypes,
                                    ArrayRef<const Expr *> BufferExprs) {
  for (size_t I = 0; I < BufferTypes.size(); ++I) {
    unsigned IndirectionCount = 0;
    const Type *BufferType = BufferTypes[I];
    llvm::SmallVector<IndirectionType, 1> Indirections;

    // Capture the depth and types of indirections for the passed buffer.
    while (true) {
      if (BufferType->isPointerType()) {
        BufferType = BufferType->getPointeeType().getTypePtr();
        Indirections.push_back(IndirectionType::Pointer);
      } else if (BufferType->isArrayType()) {
        BufferType = BufferType->getArrayElementTypeNoTypeQual();
        Indirections.push_back(IndirectionType::Array);
      } else {
        break;
      }
      ++IndirectionCount;
    }

    if (IndirectionCount > 1) {
      // Referencing an array with '&' is valid, as this also points to the
      // beginning of the array.
      if (IndirectionCount == 2 &&
          Indirections[0] == IndirectionType::Pointer &&
          Indirections[1] == IndirectionType::Array)
        return;

      // Build the indirection description in reverse order of discovery.
      std::string IndirectionDesc;
      for (auto It = Indirections.rbegin(); It != Indirections.rend(); ++It) {
        if (!IndirectionDesc.empty())
          IndirectionDesc += "->";
        if (*It == IndirectionType::Pointer) {
          IndirectionDesc += "pointer";
        } else {
          IndirectionDesc += "array";
        }
      }

      const auto Loc = BufferExprs[I]->getSourceRange().getBegin();
      diag(Loc, "buffer is insufficiently dereferenced: %0") << IndirectionDesc;
    }
  }
}

void BufferDerefCheck::onEndOfTranslationUnit() { FuncClassifier.reset(); }
} // namespace mpi
} // namespace tidy
} // namespace clang
