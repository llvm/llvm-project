//===- LifetimeAnnotations.cpp -  -*--------------- C++------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "llvm/ADT/StringSet.h"

namespace clang::lifetimes {

const FunctionDecl *
getDeclWithMergedLifetimeBoundAttrs(const FunctionDecl *FD) {
  return FD != nullptr ? FD->getMostRecentDecl() : nullptr;
}

const CXXMethodDecl *
getDeclWithMergedLifetimeBoundAttrs(const CXXMethodDecl *CMD) {
  const FunctionDecl *FD = CMD;
  return cast_if_present<CXXMethodDecl>(
      getDeclWithMergedLifetimeBoundAttrs(FD));
}

bool isNormalAssignmentOperator(const FunctionDecl *FD) {
  OverloadedOperatorKind OO = FD->getDeclName().getCXXOverloadedOperator();
  bool IsAssignment = OO == OO_Equal || isCompoundAssignmentOperator(OO);
  if (!IsAssignment)
    return false;
  QualType RetT = FD->getReturnType();
  if (!RetT->isLValueReferenceType())
    return false;
  ASTContext &Ctx = FD->getASTContext();
  QualType LHST;
  auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && MD->isCXXInstanceMember())
    LHST = Ctx.getLValueReferenceType(MD->getFunctionObjectParameterType());
  else
    LHST = FD->getParamDecl(0)->getType();
  return Ctx.hasSameType(RetT, LHST);
}

bool isAssignmentOperatorLifetimeBound(const CXXMethodDecl *CMD) {
  CMD = getDeclWithMergedLifetimeBoundAttrs(CMD);
  return CMD && isNormalAssignmentOperator(CMD) && CMD->param_size() == 1 &&
         CMD->getParamDecl(0)->hasAttr<clang::LifetimeBoundAttr>();
}

/// Check if a function has a lifetimebound attribute on its function type
/// (which represents the implicit 'this' parameter for methods).
/// Returns the attribute if found, nullptr otherwise.
static const LifetimeBoundAttr *
getLifetimeBoundAttrFromFunctionType(const TypeSourceInfo &TSI) {
  // Walk through the type layers looking for a lifetimebound attribute.
  TypeLoc TL = TSI.getTypeLoc();
  while (true) {
    auto ATL = TL.getAsAdjusted<AttributedTypeLoc>();
    if (!ATL)
      break;
    if (auto *LBAttr = ATL.getAttrAs<LifetimeBoundAttr>())
      return LBAttr;
    TL = ATL.getModifiedLoc();
  }
  return nullptr;
}

bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD) {
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  // Attribute merging doesn't work well with attributes on function types (like
  // 'this' param). We need to check all redeclarations.
  auto CheckRedecls = [](const FunctionDecl *F) {
    return llvm::any_of(F->redecls(), [](const FunctionDecl *Redecl) {
      const TypeSourceInfo *TSI = Redecl->getTypeSourceInfo();
      return TSI && getLifetimeBoundAttrFromFunctionType(*TSI);
    });
  };

  if (CheckRedecls(FD))
    return true;
  if (const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern();
      Pattern && CheckRedecls(Pattern))
    return true;
  return isNormalAssignmentOperator(FD);
}

bool isInStlNamespace(const Decl *D) {
  const DeclContext *DC = D->getDeclContext();
  if (!DC)
    return false;
  if (const auto *ND = dyn_cast<NamespaceDecl>(DC))
    if (const IdentifierInfo *II = ND->getIdentifier()) {
      StringRef Name = II->getName();
      if (Name.size() >= 2 && Name.front() == '_' &&
          (Name[1] == '_' || isUppercase(Name[1])))
        return true;
    }
  return DC->isStdNamespace();
}

bool isPointerLikeType(QualType QT) {
  return isGslPointerType(QT) || QT->isPointerType() || QT->isNullPtrType();
}

static bool isReferenceOrPointerLikeType(QualType QT) {
  return QT->isReferenceType() || isPointerLikeType(QT);
}

bool shouldTrackImplicitObjectArg(const CXXMethodDecl *Callee,
                                  bool RunningUnderLifetimeSafety) {
  if (!Callee)
    return false;
  if (auto *Conv = dyn_cast<CXXConversionDecl>(Callee))
    if (isGslPointerType(Conv->getConversionType()) &&
        Callee->getParent()->hasAttr<OwnerAttr>())
      return true;
  if (!isGslPointerType(Callee->getFunctionObjectParameterType()) &&
      !isGslOwnerType(Callee->getFunctionObjectParameterType()))
    return false;

  // Begin and end iterators.
  static const llvm::StringSet<> IteratorMembers = {
      "begin", "end", "rbegin", "rend", "cbegin", "cend", "crbegin", "crend"};
  static const llvm::StringSet<> InnerPointerGetters = {
      // Inner pointer getters.
      "c_str", "data", "get"};
  static const llvm::StringSet<> ContainerFindFns = {
      // Map and set types.
      "find", "equal_range", "lower_bound", "upper_bound"};
  // Track dereference operator and transparent functions like begin(), get(),
  // etc. for all GSL pointers. Only do so for lifetime safety analysis and not
  // for Sema's statement-local analysis as it starts to have false-positives.
  if (RunningUnderLifetimeSafety &&
      isGslPointerType(Callee->getFunctionObjectParameterType()) &&
      isReferenceOrPointerLikeType(Callee->getReturnType())) {
    if (Callee->getOverloadedOperator() == OverloadedOperatorKind::OO_Star ||
        Callee->getOverloadedOperator() == OverloadedOperatorKind::OO_Arrow)
      return true;
    if (Callee->getIdentifier() &&
        (IteratorMembers.contains(Callee->getName()) ||
         InnerPointerGetters.contains(Callee->getName())))
      return true;
  }

  if (!isInStlNamespace(Callee->getParent()))
    return false;

  if (isPointerLikeType(Callee->getReturnType())) {
    if (!Callee->getIdentifier())
      // e.g., std::optional<T>::operator->() returns T*.
      return RunningUnderLifetimeSafety
                 ? Callee->getParent()->hasAttr<OwnerAttr>() &&
                       Callee->getOverloadedOperator() ==
                           OverloadedOperatorKind::OO_Arrow
                 : false;
    return IteratorMembers.contains(Callee->getName()) ||
           InnerPointerGetters.contains(Callee->getName()) ||
           ContainerFindFns.contains(Callee->getName());
  }
  if (Callee->getReturnType()->isReferenceType()) {
    if (!Callee->getIdentifier()) {
      auto OO = Callee->getOverloadedOperator();
      if (!Callee->getParent()->hasAttr<OwnerAttr>())
        return false;
      return OO == OverloadedOperatorKind::OO_Subscript ||
             OO == OverloadedOperatorKind::OO_Star;
    }
    return llvm::StringSwitch<bool>(Callee->getName())
        .Cases({"front", "back", "at", "top", "value"}, true)
        .Default(false);
  }
  return false;
}

bool shouldTrackFirstArgument(const FunctionDecl *FD) {
  if (!FD->getIdentifier() || FD->getNumParams() < 1)
    return false;
  if (!FD->isInStdNamespace())
    return false;
  // Track std:: algorithm functions that return an iterator whose lifetime is
  // bound to the first argument.
  if (FD->getNumParams() >= 2 && FD->isInStdNamespace() &&
      isGslPointerType(FD->getReturnType())) {
    if (llvm::StringSwitch<bool>(FD->getName())
            .Cases(
                {
                    "find",
                    "find_if",
                    "find_if_not",
                    "find_first_of",
                    "adjacent_find",
                    "search",
                    "find_end",
                    "lower_bound",
                    "upper_bound",
                    "partition_point",
                },
                true)
            .Default(false))
      return true;
  }
  const auto *RD = FD->getParamDecl(0)->getType()->getPointeeCXXRecordDecl();
  if (!RD || !RD->isInStdNamespace())
    return false;
  if (!RD->hasAttr<PointerAttr>() && !RD->hasAttr<OwnerAttr>())
    return false;

  if (FD->getNumParams() != 1)
    return false;

  if (FD->getReturnType()->isPointerType() ||
      isGslPointerType(FD->getReturnType())) {
    return llvm::StringSwitch<bool>(FD->getName())
        .Cases({"begin", "rbegin", "cbegin", "crbegin"}, true)
        .Cases({"end", "rend", "cend", "crend"}, true)
        .Case("data", true)
        .Default(false);
  }
  if (FD->getReturnType()->isReferenceType()) {
    return llvm::StringSwitch<bool>(FD->getName())
        .Cases({"get", "any_cast"}, true)
        .Default(false);
  }
  return false;
}

template <typename T> static bool isRecordWithAttr(QualType Type) {
  auto *RD = Type->getAsCXXRecordDecl();
  if (!RD)
    return false;
  // Generally, if a primary template class declaration is annotated with an
  // attribute, all its specializations generated from template instantiations
  // should inherit the attribute.
  //
  // However, since lifetime analysis occurs during parsing, we may encounter
  // cases where a full definition of the specialization is not required. In
  // such cases, the specialization declaration remains incomplete and lacks the
  // attribute. Therefore, we fall back to checking the primary template class.
  //
  // Note: it is possible for a specialization declaration to have an attribute
  // even if the primary template does not.
  //
  // FIXME: What if the primary template and explicit specialization
  // declarations have conflicting attributes? We should consider diagnosing
  // this scenario.
  bool Result = RD->hasAttr<T>();

  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
    Result |= CTSD->getSpecializedTemplate()->getTemplatedDecl()->hasAttr<T>();

  return Result;
}

bool isGslPointerType(QualType QT) { return isRecordWithAttr<PointerAttr>(QT); }
bool isGslOwnerType(QualType QT) { return isRecordWithAttr<OwnerAttr>(QT); }

static StringRef getName(const CXXRecordDecl &RD) {
  if (const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(&RD))
    return CTSD->getSpecializedTemplate()->getName();
  if (RD.getIdentifier())
    return RD.getName();
  return "";
}

static bool isStdUniquePtr(const CXXRecordDecl &RD) {
  return RD.isInStdNamespace() && getName(RD) == "unique_ptr";
}

bool isUniquePtrRelease(const CXXMethodDecl &MD) {
  return MD.getIdentifier() && MD.getName() == "release" &&
         MD.getNumParams() == 0 && isStdUniquePtr(*MD.getParent());
}

static bool hasInvalidatesAttr(const TypeSourceInfo &TSI) {
  // Walk through the type layers looking for an annotate attribute.
  TypeLoc TL = TSI.getTypeLoc();
  while (true) {
    auto ATL = TL.getAsAdjusted<AttributedTypeLoc>();
    if (!ATL)
      break;
    if (auto *AnnAttr = ATL.getAttrAs<AnnotateTypeAttr>()) {
      if (AnnAttr->getAnnotation() == "experimental_invalidates")
        return true;
    }
    TL = ATL.getModifiedLoc();
  }
  return false;
}

bool isContainerInvalidationMethod(const CXXMethodDecl &MD) {
  auto CheckRedecls = [](const FunctionDecl *F) {
    return llvm::any_of(F->redecls(), [](const FunctionDecl *Redecl) {
      const TypeSourceInfo *TSI = Redecl->getTypeSourceInfo();
      return TSI && hasInvalidatesAttr(*TSI);
    });
  };

  if (CheckRedecls(&MD))
    return true;
  if (const FunctionDecl *Pattern = MD.getTemplateInstantiationPattern();
      Pattern && CheckRedecls(Pattern))
    return true;

  const CXXRecordDecl *RD = MD.getParent();
  if (!isInStlNamespace(RD))
    return false;

  // `pop_back` is excluded: it only invalidates references to the removed
  // element, not to other elements.
  static const llvm::StringSet<> Vector = {// Insertion
                                           "insert", "emplace", "emplace_back",
                                           "push_back", "insert_range",
                                           "append_range",
                                           // Removal
                                           "erase", "clear",
                                           // Memory management
                                           "reserve", "resize", "shrink_to_fit",
                                           // Assignment
                                           "assign", "assign_range"};

  // `pop_*` methods are excluded: they only invalidate references to the
  // removed element, not to other elements.
  static const llvm::StringSet<> Deque = {// Insertion
                                          "insert", "emplace", "insert_range",
                                          // Removal
                                          "erase", "clear",
                                          // Memory management
                                          "resize", "shrink_to_fit",
                                          // Assignment
                                          "assign", "assign_range"};

  static const llvm::StringSet<> String = {
      // Insertion
      "insert", "push_back", "append", "replace", "replace_with_range",
      "insert_range", "append_range",
      // Removal
      "pop_back", "erase", "clear",
      // Memory management
      "reserve", "resize", "resize_and_overwrite", "shrink_to_fit",
      // Assignment
      "swap", "assign", "assign_range"};

  // FIXME: Add queue and stack and check for underlying container
  // (e.g. no invalidation for std::list).
  static const llvm::StringSet<> PriorityQueue = {// Insertion
                                                  "push", "emplace",
                                                  "push_range",
                                                  // Removal
                                                  "pop"};

  // `erase` and `extract` are excluded: they only affect the removed element,
  // not to other elements.
  static const llvm::StringSet<> NodeBased = {// Removal
                                              "clear"};

  // For `flat_*` container adaptors, `try_emplace` and `insert_or_assign`
  // only exist on `flat_map`. Listing them here is harmless since the methods
  // won't be found on other types.
  static const llvm::StringSet<> Flat = {// Insertion
                                         "insert", "emplace", "emplace_hint",
                                         "try_emplace", "insert_or_assign",
                                         "insert_range", "merge",
                                         // Removal
                                         "extract", "erase", "clear",
                                         // Assignment
                                         "replace"};

  const StringRef ContainerName = getName(*RD);
  // TODO: Consider caching this lookup by CXXMethodDecl pointer if this
  // StringSwitch becomes a performance bottleneck.
  const llvm::StringSet<> *InvalidatingMethods =
      llvm::StringSwitch<const llvm::StringSet<> *>(ContainerName)
          .Case("vector", &Vector)
          .Case("basic_string", &String)
          .Case("deque", &Deque)
          .Case("priority_queue", &PriorityQueue)
          .Cases({"set", "multiset", "map", "multimap", "unordered_set",
                  "unordered_multiset", "unordered_map", "unordered_multimap"},
                 &NodeBased)
          .Cases({"flat_map", "flat_set", "flat_multimap", "flat_multiset"},
                 &Flat)
          .Default(nullptr);

  if (!InvalidatingMethods)
    return false;

  // Handle Operators via OverloadedOperatorKind
  OverloadedOperatorKind OO = MD.getOverloadedOperator();
  if (OO != OO_None) {
    switch (OO) {
    case OO_Equal:     // operator= : Always invalidates (Assignment)
    case OO_PlusEqual: // operator+= : Append (String/Vector)
      return true;
    case OO_Subscript: // operator[] : Invalidation only for
                       // `flat_map` (Insert-or-access).
                       // `map` and `unordered_map` are excluded.
      return ContainerName == "flat_map";
    default:
      return false;
    }
  }

  if (!MD.getIdentifier())
    return false;

  return InvalidatingMethods->contains(MD.getName());
}
} // namespace clang::lifetimes
