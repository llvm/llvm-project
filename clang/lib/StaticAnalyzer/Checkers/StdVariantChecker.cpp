//===- StdVariantChecker.cpp -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>

#include "TaggedUnionModeling.h"

using namespace clang;
using namespace ento;
using namespace tagged_union_modeling;

REGISTER_MAP_WITH_PROGRAMSTATE(VariantHeldMap, const MemRegion *, SVal)

namespace clang::ento::tagged_union_modeling {

const CXXConstructorDecl *
getConstructorDeclarationForCall(const CallEvent &Call) {
  const auto *ConstructorCall = dyn_cast<CXXConstructorCall>(&Call);
  if (!ConstructorCall)
    return nullptr;

  return ConstructorCall->getDecl();
}

bool isCopyConstructorCall(const CallEvent &Call) {
  if (const CXXConstructorDecl *ConstructorDecl =
          getConstructorDeclarationForCall(Call))
    return ConstructorDecl->isCopyConstructor();
  return false;
}

bool isCopyAssignmentCall(const CallEvent &Call) {
  const Decl *CopyAssignmentDecl = Call.getDecl();

  if (const auto *AsMethodDecl =
          dyn_cast_or_null<CXXMethodDecl>(CopyAssignmentDecl))
    return AsMethodDecl->isCopyAssignmentOperator();
  return false;
}

bool isMoveConstructorCall(const CallEvent &Call) {
  const CXXConstructorDecl *ConstructorDecl =
      getConstructorDeclarationForCall(Call);
  if (!ConstructorDecl)
    return false;

  return ConstructorDecl->isMoveConstructor();
}

bool isMoveAssignmentCall(const CallEvent &Call) {
  const Decl *CopyAssignmentDecl = Call.getDecl();

  const auto *AsMethodDecl =
      dyn_cast_or_null<CXXMethodDecl>(CopyAssignmentDecl);
  if (!AsMethodDecl)
    return false;

  return AsMethodDecl->isMoveAssignmentOperator();
}

bool isStdType(const Type *Type, llvm::StringRef TypeName) {
  auto *Decl = Type->getAsRecordDecl();
  if (!Decl)
    return false;
  return (Decl->getName() == TypeName) && Decl->isInStdNamespace();
}

bool isStdVariant(const Type *Type) {
  return isStdType(Type, llvm::StringLiteral("variant"));
}

} // end of namespace clang::ento::tagged_union_modeling

static std::optional<ArrayRef<TemplateArgument>>
getTemplateArgsFromVariant(const Type *VariantType) {
  const auto *TempSpecType = VariantType->getAs<TemplateSpecializationType>();
  if (!TempSpecType)
    return {};

  return TempSpecType->template_arguments();
}

static std::optional<QualType>
getNthTemplateTypeArgFromVariant(const Type *varType, unsigned i) {
  std::optional<ArrayRef<TemplateArgument>> VariantTemplates =
      getTemplateArgsFromVariant(varType);
  if (!VariantTemplates)
    return {};

  return (*VariantTemplates)[i].getAsType();
}

static bool isVowel(char a) {
  switch (a) {
  case 'a':
  case 'e':
  case 'i':
  case 'o':
  case 'u':
    return true;
  default:
    return false;
  }
}

static llvm::StringRef indefiniteArticleBasedOnVowel(char a) {
  if (isVowel(a))
    return "an";
  return "a";
}

class StdVariantChecker : public Checker<eval::Call, check::RegionChanges> {
  // Call descriptors to find relevant calls
  CallDescription VariantConstructor{{"std", "variant", "variant"}};
  CallDescription VariantAssignmentOperator{{"std", "variant", "operator="}};
  CallDescription StdGet{{"std", "get"}, 1, 1};
  CallDescription StdSwap{{"std", "swap"}, 2};
  CallDescription StdEmplace{{"std", "variant", "emplace"}};

  BugType BadVariantType{
      this, "The active type of std::variant differs from the accessed."};

public:
  ProgramStateRef checkRegionChanges(ProgramStateRef State,
                                     const InvalidatedSymbols *,
                                     ArrayRef<const MemRegion *>,
                                     ArrayRef<const MemRegion *> Regions,
                                     const LocationContext *,
                                     const CallEvent *Call) const {
    if (!Call)
      return State;

    return removeInformationStoredForDeadInstances<VariantHeldMap>(*Call, State,
                                                                   Regions);
  }

  bool evalCall(const CallEvent &Call, CheckerContext &C) const {
    // Check if the call was not made from a system header. If it was then
    // we do an early return because it is part of the implementation.
    if (Call.isCalledFromSystemHeader())
      return false;

    if (StdGet.matches(Call))
      return handleStdGetCall(Call, C);

    if (StdSwap.matches(Call))
      return handleStdSwapCall<VariantHeldMap>(Call, C);

    // TODO Implement the modeling of std::variants emplace method.
    if (StdEmplace.matches(Call))
      return handleStdVariantEmplaceCall(Call, C);

    // First check if a constructor call is happening. If it is a
    // constructor call, check if it is an std::variant constructor call.
    bool IsVariantConstructor =
        isa<CXXConstructorCall>(Call) && VariantConstructor.matches(Call);
    bool IsVariantAssignmentOperatorCall =
        isa<CXXMemberOperatorCall>(Call) &&
        VariantAssignmentOperator.matches(Call);

    if (IsVariantConstructor || IsVariantAssignmentOperatorCall) {
      if (Call.getNumArgs() == 0 && IsVariantConstructor) {
        handleDefaultConstructor(cast<CXXConstructorCall>(&Call), C);
        return true;
      }

      // FIXME Later this checker should be extended to handle constructors
      // with multiple arguments.
      if (Call.getNumArgs() != 1)
        return false;

      SVal ThisSVal;
      if (IsVariantConstructor) {
        const auto &AsConstructorCall = cast<CXXConstructorCall>(Call);
        ThisSVal = AsConstructorCall.getCXXThisVal();
      } else if (IsVariantAssignmentOperatorCall) {
        const auto &AsMemberOpCall = cast<CXXMemberOperatorCall>(Call);
        ThisSVal = AsMemberOpCall.getCXXThisVal();
      } else {
        return false;
      }

      return handleConstructorAndAssignment<VariantHeldMap>(Call, C, ThisSVal);
    }
    return false;
  }

private:
  // The default constructed std::variant must be handled separately.
  // When an std::variant instance is default constructed it holds
  // a value-initialized value of the first type alternative.
  void handleDefaultConstructor(const CXXConstructorCall *ConstructorCall,
                                CheckerContext &C) const {
    SVal ThisSVal = ConstructorCall->getCXXThisVal();

    const auto *const ThisMemRegion = ThisSVal.getAsRegion();
    if (!ThisMemRegion)
      return;

    // Get the first type alternative of the std::variant instance.
    assert((ThisSVal.getType(C.getASTContext())->isPointerType() ||
            ThisSVal.getType(C.getASTContext())->isReferenceType()) &&
           "The This SVal must be a pointer!");

    std::optional<QualType> DefaultType = getNthTemplateTypeArgFromVariant(
        ThisSVal.getType(C.getASTContext())->getPointeeType().getTypePtr(), 0);
    if (!DefaultType)
      return;

    // We conjure a symbol that represents the value-initialized value held by
    // the default constructed std::variant. This could be made more precise
    // if we would actually simulate the value-initialization of the value.
    //
    // We are storing pointer/reference typed SVals because when an
    // std::variant is constructed with a value constructor a reference is
    // received. The SVal representing this parameter will also have reference
    // type. We use this SVal to store information about the value held is an
    // std::variant instance. Here we are conforming to this and also use
    // reference type. Also if we would not use reference typed SVals
    // the analyzer would crash when handling the cast expression with the
    // reason that the SVal is a NonLoc SVal.
    SVal DefaultConstructedHeldValue = C.getSValBuilder().conjureSymbolVal(
        ConstructorCall->getOriginExpr(), C.getLocationContext(),
        C.getASTContext().getLValueReferenceType(*DefaultType), C.blockCount());

    ProgramStateRef State = ConstructorCall->getState();
    State =
        State->set<VariantHeldMap>(ThisMemRegion, DefaultConstructedHeldValue);
    C.addTransition(State);
  }

  bool handleStdGetCall(const CallEvent &Call, CheckerContext &C) const {
    ProgramStateRef State = Call.getState();

    const auto &ArgType = Call.getArgExpr(0)->getType().getTypePtr();
    // We have to make sure that the argument is an std::variant.
    // There is another std::get with std::pair argument
    if (!isStdVariant(ArgType))
      return false;

    // Get the mem region of the argument std::variant and look up the type
    // information that we know about it.
    const MemRegion *ArgMemRegion = Call.getArgSVal(0).getAsRegion();
    const SVal *StoredSVal = State->get<VariantHeldMap>(ArgMemRegion);
    if (!StoredSVal)
      return false;

    QualType RefStoredType = StoredSVal->getType(C.getASTContext());
    llvm::errs() << "Just dump the stored SVal\n";
    StoredSVal->dump();
    llvm::errs() <<"\n The type:\n";

    if (RefStoredType->getPointeeType().isNull())
      return false;
    QualType StoredType = RefStoredType->getPointeeType();
    StoredType->dump();

    const CallExpr *CE = cast<CallExpr>(Call.getOriginExpr());
    const FunctionDecl *FD = CE->getDirectCallee();
    if (FD->getTemplateSpecializationArgs()->size() < 1)
      return false;

    const auto &TypeOut = FD->getTemplateSpecializationArgs()->get(0);
    // std::get's first template parameter can be the type we want to get
    // out of the std::variant or a natural number which is the position of
    // the requested type in the argument type list of the std::variant's
    // argument.
    QualType RetrievedType;
    switch (TypeOut.getKind()) {
    case TemplateArgument::ArgKind::Type:
      RetrievedType = TypeOut.getAsType();
      break;
    case TemplateArgument::ArgKind::Integral:
      // In the natural number case we look up which type corresponds to the
      // number.
      if (std::optional<QualType> NthTemplate =
              getNthTemplateTypeArgFromVariant(
                  ArgType, TypeOut.getAsIntegral().getSExtValue())) {
        RetrievedType = *NthTemplate;
        break;
      }
      [[fallthrough]];
    default:
      return false;
    }

    QualType RetrievedCanonicalType = RetrievedType.getCanonicalType();
    QualType StoredCanonicalType = StoredType.getCanonicalType();
    if (RetrievedCanonicalType.isNull() || StoredType.isNull())
      return false;

    if (RetrievedCanonicalType == StoredCanonicalType) {
      llvm::errs() << "Variant Checker stroed sval bind dump:\n";
      StoredSVal->dump();
      llvm::errs() << "\n";
      State = State->BindExpr(CE, C.getLocationContext(), *StoredSVal);
      C.addTransition(State);
      return true;
    }

    ExplodedNode *ErrNode = C.generateErrorNode();
    if (!ErrNode)
      return false;
    llvm::SmallString<128> Str;
    llvm::raw_svector_ostream OS(Str);
    std::string StoredTypeName = StoredType.getAsString();
    std::string RetrievedTypeName = RetrievedType.getAsString();
    OS << "std::variant " << ArgMemRegion->getDescriptiveName() << " held "
       << indefiniteArticleBasedOnVowel(StoredTypeName[0]) << " \'"
       << StoredTypeName << "\', not "
       << indefiniteArticleBasedOnVowel(RetrievedTypeName[0]) << " \'"
       << RetrievedTypeName << "\'";
    auto R = std::make_unique<PathSensitiveBugReport>(BadVariantType, OS.str(),
                                                      ErrNode);
    C.emitReport(std::move(R));
    return true;
  }

  // TODO Implement modeling of std::variant's emplace method.
  // Currently when this method call is encountered we just
  // stop the modeling of that std::variant instance.
  bool handleStdVariantEmplaceCall(const CallEvent &Call,
                                   CheckerContext &C) const {
    const auto *AsMemberCall = dyn_cast_or_null<CXXMemberCall>(&Call);
    if (!AsMemberCall)
      return false;
    const MemRegion *ThisRegion = AsMemberCall->getCXXThisVal().getAsRegion();
    if (!ThisRegion)
      return false;
    C.addTransition(C.getState()->remove<VariantHeldMap>(ThisRegion));
    return true;
  }
};

bool clang::ento::shouldRegisterStdVariantChecker(
    clang::ento::CheckerManager const &mgr) {
  return true;
}

void clang::ento::registerStdVariantChecker(clang::ento::CheckerManager &mgr) {
  mgr.registerChecker<StdVariantChecker>();
}