//===--- VirtualNearMissCheck.cpp - clang-tidy-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "VirtualNearMissCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

/// Finds out if the given method overrides some method.
static bool isOverrideMethod(const CXXMethodDecl *MD) {
  return MD->size_overridden_methods() > 0 || MD->hasAttr<OverrideAttr>();
}

/// Checks whether the return types are covariant, according to
/// C++[class.virtual]p7.
///
/// Similar with clang::Sema::CheckOverridingFunctionReturnType.
/// \returns true if the return types of BaseMD and DerivedMD are covariant.
static bool checkOverridingFunctionReturnType(const ASTContext *Context,
                                              const CXXMethodDecl *BaseMD,
                                              const CXXMethodDecl *DerivedMD) {
  QualType BaseReturnTy =
      BaseMD->getType()->getAs<FunctionType>()->getReturnType();
  QualType DerivedReturnTy =
      DerivedMD->getType()->getAs<FunctionType>()->getReturnType();

  if (DerivedReturnTy->isDependentType() || BaseReturnTy->isDependentType())
    return false;

  // Check if return types are identical.
  if (Context->hasSameType(DerivedReturnTy, BaseReturnTy))
    return true;

  /// Check if the return types are covariant.
  /// BTy is the class type in return type of BaseMD. For example,
  ///    B* Base::md()
  /// While BRD is the declaration of B.
  QualType BTy, DTy;
  const CXXRecordDecl *BRD, *DRD;

  // Both types must be pointers or references to classes.
  if (const auto *DerivedPT = DerivedReturnTy->getAs<PointerType>()) {
    if (const auto *BasePT = BaseReturnTy->getAs<PointerType>()) {
      DTy = DerivedPT->getPointeeType();
      BTy = BasePT->getPointeeType();
    }
  } else if (const auto *DerivedRT = DerivedReturnTy->getAs<ReferenceType>()) {
    if (const auto *BaseRT = BaseReturnTy->getAs<ReferenceType>()) {
      DTy = DerivedRT->getPointeeType();
      BTy = BaseRT->getPointeeType();
    }
  }

  // The return types aren't either both pointers or references to a class type.
  if (DTy.isNull())
    return false;

  DRD = DTy->getAsCXXRecordDecl();
  BRD = BTy->getAsCXXRecordDecl();
  if (DRD == nullptr || BRD == nullptr)
    return false;

  if (DRD == BRD)
    return true;

  if (!Context->hasSameUnqualifiedType(DTy, BTy)) {
    // Begin checking whether the conversion from D to B is valid.
    CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                       /*DetectVirtual=*/false);

    // Check whether D is derived from B, and fill in a CXXBasePaths object.
    if (!DRD->isDerivedFrom(BRD, Paths))
      return false;

    // Check ambiguity.
    if (Paths.isAmbiguous(Context->getCanonicalType(BTy).getUnqualifiedType()))
      return false;

    // Check accessibility.
    // FIXME: We currently only support checking if B is accessible base class
    // of D, or D is the same class which DerivedMD is in.
    bool IsItself = DRD == DerivedMD->getParent();
    bool HasPublicAccess = false;
    for (const auto &Path : Paths) {
      if (Path.Access == AS_public)
        HasPublicAccess = true;
    }
    if (!HasPublicAccess && !IsItself)
      return false;
    // End checking conversion from D to B.
  }

  // Both pointers or references should have the same cv-qualification.
  if (DerivedReturnTy.getLocalCVRQualifiers() !=
      BaseReturnTy.getLocalCVRQualifiers())
    return false;

  // The class type D should have the same cv-qualification as or less
  // cv-qualification than the class type B.
  if (DTy.isMoreQualifiedThan(BTy))
    return false;

  return true;
}

/// \returns true if the param types are the same.
static bool checkParamTypes(const CXXMethodDecl *BaseMD,
                            const CXXMethodDecl *DerivedMD) {
  unsigned NumParamA = BaseMD->getNumParams();
  unsigned NumParamB = DerivedMD->getNumParams();
  if (NumParamA != NumParamB)
    return false;

  for (unsigned I = 0; I < NumParamA; I++) {
    if (BaseMD->getParamDecl(I)->getType() !=
        DerivedMD->getParamDecl(I)->getType())
      return false;
  }
  return true;
}

/// \returns true if derived method can override base method except for the
/// name.
static bool checkOverrideWithoutName(const ASTContext *Context,
                                     const CXXMethodDecl *BaseMD,
                                     const CXXMethodDecl *DerivedMD) {
  if (BaseMD->getTypeQualifiers() != DerivedMD->getTypeQualifiers())
    return false;

  if (BaseMD->isStatic() != DerivedMD->isStatic())
    return false;

  if (BaseMD->getAccess() != DerivedMD->getAccess())
    return false;

  if (BaseMD->getType() == DerivedMD->getType())
    return true;

  // Now the function types are not identical. Then check if the return types
  // are covariant and if the param types are the same.
  if (!checkOverridingFunctionReturnType(Context, BaseMD, DerivedMD))
    return false;
  return checkParamTypes(BaseMD, DerivedMD);
}

/// Check whether BaseMD overrides DerivedMD.
///
/// Prerequisite: the class which BaseMD is in should be a base class of that
/// DerivedMD is in.
static bool checkOverrideByDerivedMethod(const CXXMethodDecl *BaseMD,
                                         const CXXMethodDecl *DerivedMD) {
  if (BaseMD->getNameAsString() != DerivedMD->getNameAsString())
    return false;

  if (!checkParamTypes(BaseMD, DerivedMD))
    return false;

  return true;
}

/// Generate unique ID for given MethodDecl.
///
/// The Id is used as key for 'PossibleMap'.
/// Typical Id: "Base::func void (void)"
static std::string generateMethodId(const CXXMethodDecl *MD) {
  return MD->getQualifiedNameAsString() + " " + MD->getType().getAsString();
}

bool VirtualNearMissCheck::isPossibleToBeOverridden(
    const CXXMethodDecl *BaseMD) {
  std::string Id = generateMethodId(BaseMD);
  auto Iter = PossibleMap.find(Id);
  if (Iter != PossibleMap.end())
    return Iter->second;

  bool IsPossible = !BaseMD->isImplicit() && !isa<CXXConstructorDecl>(BaseMD) &&
                    BaseMD->isVirtual();
  PossibleMap[Id] = IsPossible;
  return IsPossible;
}

bool VirtualNearMissCheck::isOverriddenByDerivedClass(
    const CXXMethodDecl *BaseMD, const CXXRecordDecl *DerivedRD) {
  auto Key = std::make_pair(generateMethodId(BaseMD),
                            DerivedRD->getQualifiedNameAsString());
  auto Iter = OverriddenMap.find(Key);
  if (Iter != OverriddenMap.end())
    return Iter->second;

  bool IsOverridden = false;
  for (const CXXMethodDecl *DerivedMD : DerivedRD->methods()) {
    if (!isOverrideMethod(DerivedMD))
      continue;

    if (checkOverrideByDerivedMethod(BaseMD, DerivedMD)) {
      IsOverridden = true;
      break;
    }
  }
  OverriddenMap[Key] = IsOverridden;
  return IsOverridden;
}

void VirtualNearMissCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(cxxMethodDecl(unless(anyOf(isOverride(), isImplicit(),
                                                cxxConstructorDecl())))
                         .bind("method"),
                     this);
}

void VirtualNearMissCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *DerivedMD = Result.Nodes.getNodeAs<CXXMethodDecl>("method");
  assert(DerivedMD != nullptr);

  if (DerivedMD->isStatic())
    return;

  const ASTContext *Context = Result.Context;

  const auto *DerivedRD = DerivedMD->getParent();

  for (const auto &BaseSpec : DerivedRD->bases()) {
    if (const auto *BaseRD = BaseSpec.getType()->getAsCXXRecordDecl()) {
      for (const auto *BaseMD : BaseRD->methods()) {
        if (!isPossibleToBeOverridden(BaseMD))
          continue;

        if (isOverriddenByDerivedClass(BaseMD, DerivedRD))
          continue;

        unsigned EditDistance =
            BaseMD->getName().edit_distance(DerivedMD->getName());
        if (EditDistance > 0 && EditDistance <= EditDistanceThreshold) {
          if (checkOverrideWithoutName(Context, BaseMD, DerivedMD)) {
            // A "virtual near miss" is found.
            diag(DerivedMD->getLocStart(),
                 "method '%0' has a similar name and the same signature as "
                 "virtual method '%1'; did you mean to override it?")
                << DerivedMD->getQualifiedNameAsString()
                << BaseMD->getQualifiedNameAsString();
          }
        }
      }
    }
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
