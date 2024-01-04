//===- StdAnyChecker.cpp -------------------------------------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "llvm/Support/ErrorHandling.h"

#include "TaggedUnionModeling.h"

using namespace clang;
using namespace ento;
using namespace tagged_union_modeling;

REGISTER_MAP_WITH_PROGRAMSTATE(AnyHeldTypeMap, const MemRegion *, QualType)

class StdAnyChecker : public Checker<eval::Call, check::RegionChanges> {
  CallDescription AnyConstructor{{"std", "any", "any"}};
  CallDescription AnyAsOp{{"std", "any", "operator="}};
  CallDescription AnyReset{{"std", "any", "reset"}, 0, 0};
  CallDescription AnyCast{{"std", "any_cast"}, 1, 1};

  const BugType BadAnyType{this, "Bad std::any type access.", "BadAnyType"};
  const BugType NullAnyType{this, "std::any has no value", "NullAnyType"};

public:
  ProgramStateRef checkRegionChanges(ProgramStateRef State,
                                     const InvalidatedSymbols *,
                                     ArrayRef<const MemRegion *>,
                                     ArrayRef<const MemRegion *> Regions,
                                     const LocationContext *,
                                     const CallEvent *Call) const {
    if (!Call)
      return State;

    return removeInformationStoredForDeadInstances<AnyHeldTypeMap>(*Call, State,
                                                                   Regions);
  }

  bool evalCall(const CallEvent &Call, CheckerContext &C) const {
    // Do not take implementation details into consideration
    if (Call.isCalledFromSystemHeader())
      return false;

    if (AnyCast.matches(Call))
      return handleAnyCastCall(Call, C);

    if (AnyReset.matches(Call)) {
      const auto *AsMemberCall = dyn_cast<CXXMemberCall>(&Call);
      if (!AsMemberCall)
        return false;

      const auto *ThisMemRegion = AsMemberCall->getCXXThisVal().getAsRegion();
      if (!ThisMemRegion)
        return false;

      C.addTransition(setNullTypeAny(ThisMemRegion, C));
      return true;
    }

    bool IsAnyConstructor =
        isa<CXXConstructorCall>(Call) && AnyConstructor.matches(Call);
    bool IsAnyAssignmentOperatorCall =
        isa<CXXMemberOperatorCall>(Call) && AnyAsOp.matches(Call);

    if (IsAnyConstructor || IsAnyAssignmentOperatorCall) {
      auto State = C.getState();
      SVal ThisSVal = [&]() {
        if (IsAnyConstructor) {
          const auto *AsConstructorCall = dyn_cast<CXXConstructorCall>(&Call);
          return AsConstructorCall->getCXXThisVal();
        }
        if (IsAnyAssignmentOperatorCall) {
          const auto *AsMemberOpCall = dyn_cast<CXXMemberOperatorCall>(&Call);
          return AsMemberOpCall->getCXXThisVal();
        }
        llvm_unreachable("We must have an assignment operator or constructor");
      }();

      // Default constructor call.
      // In this case the any holds a null type.
      if (Call.getNumArgs() == 0) {
        const auto *ThisMemRegion = ThisSVal.getAsRegion();
        C.addTransition(setNullTypeAny(ThisMemRegion, C));
        return true;
      }

      if (Call.getNumArgs() != 1)
        return false;

      handleConstructorAndAssignment<AnyHeldTypeMap>(Call, C, ThisSVal);
      return true;
    }
    return false;
  }

private:
  // When a std::any is reset or default constructed it has a null type.
  // We represent it by storing an empty QualType.
  ProgramStateRef setNullTypeAny(const MemRegion *Mem,
                                 CheckerContext &C) const {
    auto State = C.getState();
    return State->set<AnyHeldTypeMap>(Mem, QualType{});
  }

  bool handleAnyCastCall(const CallEvent &Call, CheckerContext &C) const {
    auto State = C.getState();

    if (Call.getNumArgs() != 1)
      return false;

    auto ArgSVal = Call.getArgSVal(0);

    // The argument is either a const reference or a right value reference.
    // We need the type referred.
    const auto *ArgType = ArgSVal.getType(C.getASTContext())
                              .getTypePtr()
                              ->getPointeeType()
                              .getTypePtr();
    if (!isStdAny(ArgType))
      return false;

    const auto *AnyMemRegion = ArgSVal.getAsRegion();

    if (!State->contains<AnyHeldTypeMap>(AnyMemRegion))
      return false;

    // Get the type we are trying to retrieve from any.
    const CallExpr *CE = cast<CallExpr>(Call.getOriginExpr());
    const FunctionDecl *FD = CE->getDirectCallee();
    if (FD->getTemplateSpecializationArgs()->size() != 1)
      return false;

    const auto &FirstTemplateArgument =
        FD->getTemplateSpecializationArgs()->asArray()[0];
    if (FirstTemplateArgument.getKind() != TemplateArgument::ArgKind::Type)
      return false;

    auto TypeOut = FirstTemplateArgument.getAsType();
    const auto *TypeStored = State->get<AnyHeldTypeMap>(AnyMemRegion);

    // Report when we try to use std::any_cast on a std::any that held a null
    // type.
    if (TypeStored->isNull()) {
      ExplodedNode *ErrNode = C.generateErrorNode();
      if (!ErrNode)
        return false;
      llvm::SmallString<128> Str;
      llvm::raw_svector_ostream OS(Str);
      OS << "std::any " << AnyMemRegion->getDescriptiveName() << " is empty.";
      auto R = std::make_unique<PathSensitiveBugReport>(NullAnyType, OS.str(),
                                                        ErrNode);
      C.emitReport(std::move(R));
      return true;
    }

    // Check if the right type is being accessed.
    QualType RetrievedCanonicalType = TypeOut.getCanonicalType();
    QualType StoredCanonicalType = TypeStored->getCanonicalType();
    if (RetrievedCanonicalType == StoredCanonicalType)
      return true;

    // Report when the type we want to get out of std::any is wrong.
    ExplodedNode *ErrNode = C.generateNonFatalErrorNode();
    if (!ErrNode)
      return false;
    llvm::SmallString<128> Str;
    llvm::raw_svector_ostream OS(Str);
    std::string StoredTypeName = StoredCanonicalType.getAsString();
    std::string RetrievedTypeName = RetrievedCanonicalType.getAsString();
    OS << "std::any " << AnyMemRegion->getDescriptiveName() << " held "
       << indefiniteArticleBasedOnVowel(StoredTypeName[0]) << " \'"
       << StoredTypeName << "\', not "
       << indefiniteArticleBasedOnVowel(RetrievedTypeName[0]) << " \'"
       << RetrievedTypeName << "\'";
    auto R =
        std::make_unique<PathSensitiveBugReport>(BadAnyType, OS.str(), ErrNode);
    C.emitReport(std::move(R));
    return true;
  }
};

bool clang::ento::shouldRegisterStdAnyChecker(
    clang::ento::CheckerManager const &mgr) {
  return true;
}

void clang::ento::registerStdAnyChecker(clang::ento::CheckerManager &mgr) {
  mgr.registerChecker<StdAnyChecker>();
}