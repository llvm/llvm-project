//===--- BufferDerefCheck.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BufferDerefCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/StaticAnalyzer/Checkers/MPIFunctionClassifier.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace mpi {

void BufferDerefCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(callExpr().bind("CE"), this);
}

void BufferDerefCheck::check(const MatchFinder::MatchResult &Result) {
  static ento::mpi::MPIFunctionClassifier FuncClassifier(*Result.Context);
  const auto *CE = Result.Nodes.getNodeAs<CallExpr>("CE");
  if (!CE->getDirectCallee())
    return;

  const IdentifierInfo *Identifier = CE->getDirectCallee()->getIdentifier();
  if (!Identifier || !FuncClassifier.isMPIType(Identifier))
    return;

  // These containers are used, to capture the type and expression of a buffer.
  SmallVector<const Type *, 1> BufferTypes;
  SmallVector<const Expr *, 1> BufferExprs;

  // Adds the type and expression of a buffer that is used in the MPI call
  // expression to the captured containers.
  auto addBuffer = [&CE, &Result, &BufferTypes,
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
  if (FuncClassifier.isPointToPointType(Identifier)) {
    addBuffer(0);
  } else if (FuncClassifier.isCollectiveType(Identifier)) {
    if (FuncClassifier.isReduceType(Identifier)) {
      addBuffer(0);
      addBuffer(1);
    } else if (FuncClassifier.isScatterType(Identifier) ||
               FuncClassifier.isGatherType(Identifier) ||
               FuncClassifier.isAlltoallType(Identifier)) {
      addBuffer(0);
      addBuffer(3);
    } else if (FuncClassifier.isBcastType(Identifier)) {
      addBuffer(0);
    }
  }

  checkBuffers(BufferTypes, BufferExprs);
}

void BufferDerefCheck::checkBuffers(ArrayRef<const Type *> BufferTypes,
                                    ArrayRef<const Expr *> BufferExprs) {
  for (size_t i = 0; i < BufferTypes.size(); ++i) {
    unsigned IndirectionCount = 0;
    const Type *BufferType = BufferTypes[i];
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

      const auto Loc = BufferExprs[i]->getSourceRange().getBegin();
      diag(Loc, "buffer is insufficiently dereferenced: %0") << IndirectionDesc;
    }
  }
}

} // namespace mpi
} // namespace tidy
} // namespace clang
