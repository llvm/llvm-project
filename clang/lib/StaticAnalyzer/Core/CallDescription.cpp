//===- CallDescription.cpp - function/method call matching     --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file defines a generic mechanism for matching for function and
/// method calls of C, C++, and Objective-C languages. Instances of these
/// classes are frequently used together with the CallEvent classes.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/AST/Decl.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include <iterator>

using namespace llvm;
using namespace clang;

using MaybeCount = Optional<unsigned>;

// A constructor helper.
static MaybeCount readRequiredParams(MaybeCount RequiredArgs,
                                     MaybeCount RequiredParams) {
  if (RequiredParams)
    return RequiredParams;
  if (RequiredArgs)
    return RequiredArgs;
  return None;
}

ento::CallDescription::CallDescription(CallDescriptionFlags Flags,
                                       ArrayRef<const char *> QualifiedName,
                                       MaybeCount RequiredArgs /*= None*/,
                                       MaybeCount RequiredParams /*= None*/)
    : RequiredArgs(RequiredArgs),
      RequiredParams(readRequiredParams(RequiredArgs, RequiredParams)),
      Flags(Flags) {
  assert(!QualifiedName.empty());
  this->QualifiedName.reserve(QualifiedName.size());
  llvm::copy(QualifiedName, std::back_inserter(this->QualifiedName));
}

/// Construct a CallDescription with default flags.
ento::CallDescription::CallDescription(ArrayRef<const char *> QualifiedName,
                                       MaybeCount RequiredArgs /*= None*/,
                                       MaybeCount RequiredParams /*= None*/)
    : CallDescription(CDF_None, QualifiedName, RequiredArgs, RequiredParams) {}

bool ento::CallDescription::matches(const CallEvent &Call) const {
  // FIXME: Add ObjC Message support.
  if (Call.getKind() == CE_ObjCMessage)
    return false;

  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return false;

  return matchesImpl(FD, Call.getNumArgs(), Call.parameters().size());
}

bool ento::CallDescription::matchesAsWritten(const CallExpr &CE) const {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(CE.getCalleeDecl());
  if (!FD)
    return false;

  return matchesImpl(FD, CE.getNumArgs(), FD->param_size());
}

bool ento::CallDescription::matchesImpl(const FunctionDecl *Callee,
                                        size_t ArgCount,
                                        size_t ParamCount) const {
  const auto *FD = Callee;
  if (!FD)
    return false;

  if (Flags & CDF_MaybeBuiltin) {
    return CheckerContext::isCLibraryFunction(FD, getFunctionName()) &&
           (!RequiredArgs || *RequiredArgs <= ArgCount) &&
           (!RequiredParams || *RequiredParams <= ParamCount);
  }

  if (!II) {
    II = &FD->getASTContext().Idents.get(getFunctionName());
  }

  const auto MatchNameOnly = [](const CallDescription &CD,
                                const NamedDecl *ND) -> bool {
    DeclarationName Name = ND->getDeclName();
    if (const auto *II = Name.getAsIdentifierInfo())
      return II == CD.II.getValue(); // Fast case.

    // Fallback to the slow stringification and comparison for:
    // C++ overloaded operators, constructors, destructors, etc.
    // FIXME This comparison is way SLOWER than comparing pointers.
    // At some point in the future, we should compare FunctionDecl pointers.
    return Name.getAsString() == CD.getFunctionName();
  };

  const auto ExactMatchArgAndParamCounts =
      [](size_t ArgCount, size_t ParamCount,
         const CallDescription &CD) -> bool {
    const bool ArgsMatch = !CD.RequiredArgs || *CD.RequiredArgs == ArgCount;
    const bool ParamsMatch =
        !CD.RequiredParams || *CD.RequiredParams == ParamCount;
    return ArgsMatch && ParamsMatch;
  };

  const auto MatchQualifiedNameParts = [](const CallDescription &CD,
                                          const Decl *D) -> bool {
    const auto FindNextNamespaceOrRecord =
        [](const DeclContext *Ctx) -> const DeclContext * {
      while (Ctx && !isa<NamespaceDecl, RecordDecl>(Ctx))
        Ctx = Ctx->getParent();
      return Ctx;
    };

    auto QualifierPartsIt = CD.begin_qualified_name_parts();
    const auto QualifierPartsEndIt = CD.end_qualified_name_parts();

    // Match namespace and record names. Skip unrelated names if they don't
    // match.
    const DeclContext *Ctx = FindNextNamespaceOrRecord(D->getDeclContext());
    for (; Ctx && QualifierPartsIt != QualifierPartsEndIt;
         Ctx = FindNextNamespaceOrRecord(Ctx->getParent())) {
      // If not matched just continue and try matching for the next one.
      if (cast<NamedDecl>(Ctx)->getName() != *QualifierPartsIt)
        continue;
      ++QualifierPartsIt;
    }

    // We matched if we consumed all expected qualifier segments.
    return QualifierPartsIt == QualifierPartsEndIt;
  };

  // Let's start matching...
  if (!ExactMatchArgAndParamCounts(ArgCount, ParamCount, *this))
    return false;

  if (!MatchNameOnly(*this, FD))
    return false;

  if (!hasQualifiedNameParts())
    return true;

  return MatchQualifiedNameParts(*this, FD);
}

ento::CallDescriptionSet::CallDescriptionSet(
    std::initializer_list<CallDescription> &&List) {
  Impl.LinearMap.reserve(List.size());
  for (const CallDescription &CD : List)
    Impl.LinearMap.push_back({CD, /*unused*/ true});
}

bool ento::CallDescriptionSet::contains(const CallEvent &Call) const {
  return static_cast<bool>(Impl.lookup(Call));
}

bool ento::CallDescriptionSet::containsAsWritten(const CallExpr &CE) const {
  return static_cast<bool>(Impl.lookupAsWritten(CE));
}
