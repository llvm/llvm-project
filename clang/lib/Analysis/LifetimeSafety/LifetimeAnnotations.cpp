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
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Analysis/AnyCall.h"
#include "clang/Basic/OperatorKinds.h"
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

const LifetimeBoundAttr *
getDirectImplicitObjectLifetimeBoundAttr(const FunctionDecl *FD) {
  if (const TypeSourceInfo *TSI = FD->getTypeSourceInfo())
    if (const auto *Attr = getLifetimeBoundAttrFromFunctionType(*TSI))
      return Attr;
  return nullptr;
}

bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD) {
  if (getImplicitObjectParamLifetimeBoundAttr(FD))
    return true;
  return isNormalAssignmentOperator(FD);
}

FunctionCallInfo::FunctionCallInfo(const Expr *Call) {
  if (!Call)
    return;

  std::optional<AnyCall> AC = AnyCall::forExpr(Call->IgnoreParenImpCasts());
  if (!AC)
    return;

  Args = AC->arguments();
  FD = dyn_cast_or_null<FunctionDecl>(AC->getDecl());
}

std::optional<LifetimeBoundParamInfo>
getTrackedArgInfo(const FunctionDecl *FD, llvm::ArrayRef<const Expr *> Args,
                  unsigned I) {
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  if (!FD || I >= Args.size())
    return std::nullopt;

  const ParmVarDecl *PVD = nullptr;

  if (const auto *Method = dyn_cast<CXXMethodDecl>(FD);
      Method && Method->isInstance() && !isa<CXXConstructorDecl>(FD)) {
    if (I == 0) {
      // For the 'this' argument, the attribute is on the method itself.
      if (implicitObjectParamIsLifetimeBound(Method) ||
          shouldTrackImplicitObjectArg(*Args[0], Method,
                                       /*RunningUnderLifetimeSafety=*/true))
        return LifetimeBoundParamInfo(Method);
      return std::nullopt;
    }
    if ((I - 1) < Method->getNumParams())
      // For explicit arguments, find the corresponding parameter declaration.
      PVD = Method->getParamDecl(I - 1);
  } else if (I == 0 && shouldTrackFirstArgument(FD)) {
    return LifetimeBoundParamInfo(FD->getParamDecl(I));
  } else if (I == 1 && shouldTrackSecondArgument(FD)) {
    return LifetimeBoundParamInfo(FD->getParamDecl(I));
  } else if (I < FD->getNumParams()) {
    // For free functions or static methods.
    PVD = FD->getParamDecl(I);
  }

  if (PVD && PVD->hasAttr<clang::LifetimeBoundAttr>())
    return LifetimeBoundParamInfo(PVD);

  return std::nullopt;
}

/// Returns true if `QT` is (a pointer or reference to) a `gsl::Owner`.
static bool isOwnerOrPointerToOwner(QualType QT) {
  if (QT->isPointerOrReferenceType())
    QT = QT->getPointeeType();
  return isGslOwnerType(QT);
}

template <typename AttrT> static PlaceRef getPlaceRefImpl(const AttrT *A) {
  PlaceRef Result;
  if (!A || A->places_size() == 0)
    return Result;
  if (A->places_size() == 1 && (*A->places_begin())->isStr("interior")) {
    Result.K = PlaceRef::Interior;
    return Result;
  }
  Result.K = PlaceRef::Places;
  Result.Names.assign(A->places_begin(), A->places_end());
  return Result;
}

PlaceRef getPlaceRef(const LifetimeBoundAttr *A) { return getPlaceRefImpl(A); }
PlaceRef getPlaceRef(const LifetimeInvalidatesAttr *A) {
  return getPlaceRefImpl(A);
}

/// Returns the implicit-object attribute of type `AttrT` from the function
/// type of any redeclaration of FD (or its template pattern).
template <typename AttrT>
static const AttrT *getImplicitObjectTypeAttr(const FunctionDecl *FD) {
  auto FromTSI = [](const FunctionDecl *F) -> const AttrT * {
    const TypeSourceInfo *TSI = F->getTypeSourceInfo();
    if (!TSI)
      return nullptr;
    TypeLoc TL = TSI->getTypeLoc();
    while (auto ATL = TL.getAsAdjusted<AttributedTypeLoc>()) {
      if (const auto *A = ATL.template getAttrAs<AttrT>())
        return A;
      TL = ATL.getModifiedLoc();
    }
    return nullptr;
  };
  auto CheckRedecls = [&](const FunctionDecl *F) -> const AttrT * {
    for (const FunctionDecl *Redecl : F->redecls())
      if (const AttrT *A = FromTSI(Redecl))
        return A;
    return nullptr;
  };
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  if (const AttrT *A = CheckRedecls(FD))
    return A;
  if (const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern())
    return CheckRedecls(Pattern);
  return nullptr;
}

const LifetimeInvalidatesAttr *
getImplicitObjectParamInvalidatesAttr(const FunctionDecl *FD) {
  return getImplicitObjectTypeAttr<LifetimeInvalidatesAttr>(FD);
}

const LifetimeBoundAttr *
getImplicitObjectParamLifetimeBoundAttr(const FunctionDecl *FD) {
  // Attribute merging doesn't work well with attributes on function types (like
  // 'this' param). We need to check all redeclarations.
  return getImplicitObjectTypeAttr<LifetimeBoundAttr>(FD);
}

PlaceRef getTrackedArgBorrow(const FunctionDecl *FD,
                             llvm::ArrayRef<const Expr *> Args, unsigned I) {
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  if (!FD || I >= Args.size())
    return PlaceRef::getObject();

  if (const auto *Method = dyn_cast<CXXMethodDecl>(FD);
      Method && Method->isInstance() && !isa<CXXConstructorDecl>(FD)) {
    if (I != 0)
      return getPlaceRef(
          I - 1 < Method->getNumParams()
              ? Method->getParamDecl(I - 1)->getAttr<LifetimeBoundAttr>()
              : nullptr);
    // The built-in table for standard owners decides first (`begin()`,
    // `data()`, `operator[]`, `get()`, conversion to a view, ...). Everything
    // it tracks on an owner hands out the owner's interior; what it tracks on
    // a `gsl::Pointer` (iterator arithmetic, `operator*`) merely propagates
    // the view's own borrow.
    if (shouldTrackImplicitObjectArg(*Args[0], Method,
                                     /*RunningUnderLifetimeSafety=*/true) &&
        (isGslOwnerType(Method->getFunctionObjectParameterType()) ||
         isGslOwnerType(Args[0]->getBestDynamicClassType())))
      return PlaceRef::getInterior();
    return getPlaceRef(getImplicitObjectParamLifetimeBoundAttr(Method));
  }
  // `std::begin(v)`, `std::data(v)`, `std::get(tuple)`, ...: interior of an
  // owner. Algorithms over iterators (`std::find(first, last, ..)`) propagate
  // the iterator's borrow.
  if (I == 0 && shouldTrackFirstArgument(FD) &&
      isOwnerOrPointerToOwner(FD->getParamDecl(0)->getType()))
    return PlaceRef::getInterior();
  if (I < FD->getNumParams())
    return getPlaceRef(FD->getParamDecl(I)->getAttr<LifetimeBoundAttr>());
  return PlaceRef::getObject();
}

std::optional<PlaceRef> getArgInvalidation(const FunctionDecl *FD,
                                           llvm::ArrayRef<const Expr *> Args,
                                           unsigned I) {
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  if (!FD || I >= Args.size())
    return std::nullopt;
  const auto *Method = dyn_cast<CXXMethodDecl>(FD);
  bool HasImplicitObject =
      Method && Method->isInstance() && !isa<CXXConstructorDecl>(FD);
  if (HasImplicitObject && I == 0) {
    // Standard containers: the built-in table says which methods invalidate
    // what the container owns.
    if (isInvalidationMethod(*Method))
      return PlaceRef::getInterior();
    if (const auto *A = getImplicitObjectParamInvalidatesAttr(Method))
      return getPlaceRef(A);
    return std::nullopt;
  }
  unsigned ParamIdx = HasImplicitObject ? I - 1 : I;
  if (ParamIdx >= FD->getNumParams())
    return std::nullopt;
  if (const auto *A =
          FD->getParamDecl(ParamIdx)->getAttr<LifetimeInvalidatesAttr>())
    return getPlaceRef(A);
  return std::nullopt;
}

std::optional<LifetimeBoundParamInfo>
getTrackingInfoForCallArg(const Expr *Call, const Expr *Source) {
  if (!Call || !Source)
    return std::nullopt;

  auto [FD, Args] = FunctionCallInfo(Call);
  if (!FD)
    return std::nullopt;

  for (unsigned I = 0; I < Args.size(); ++I)
    if (Args[I]->IgnoreParenImpCasts() == Source->IgnoreParenImpCasts())
      if (std::optional<LifetimeBoundParamInfo> ParamInfo =
              getTrackedArgInfo(FD, Args, I))
        return ParamInfo;

  return std::nullopt;
}

bool isInStlNamespace(const Decl *D) {
  for (const DeclContext *DC = D->getDeclContext(); DC; DC = DC->getParent()) {
    if (DC->isStdNamespace())
      return true;
    if (const auto *ND = dyn_cast<NamespaceDecl>(DC))
      if (const IdentifierInfo *II = ND->getIdentifier()) {
        StringRef Name = II->getName();
        if (Name.size() >= 2 && Name.front() == '_' &&
            (Name[1] == '_' || isUppercase(Name[1])))
          return true;
      }
  }
  return false;
}

bool isPointerLikeType(QualType QT) {
  return isGslPointerType(QT) || QT->isPointerType() || QT->isNullPtrType();
}

static bool isReferenceOrPointerLikeType(QualType QT) {
  return QT->isReferenceType() || isPointerLikeType(QT);
}

bool shouldTrackImplicitObjectArg(const Expr &ImplicitObjectArgument,
                                  const CXXMethodDecl *Callee,
                                  bool RunningUnderLifetimeSafety) {
  if (!Callee)
    return false;
  // Check both the declaring class and the call-site object: a gsl::Owner
  // may inherit its accessors from a non-Owner base (e.g. libc++ optional).
  const bool IsGslOwnerImplicitObject =
      isGslOwnerType(Callee->getFunctionObjectParameterType()) ||
      (RunningUnderLifetimeSafety &&
       isGslOwnerType(ImplicitObjectArgument.getBestDynamicClassType()));
  if (auto *Conv = dyn_cast<CXXConversionDecl>(Callee))
    if (isGslPointerType(Conv->getConversionType()) && IsGslOwnerImplicitObject)
      return true;
  if (!isGslPointerType(Callee->getFunctionObjectParameterType()) &&
      !IsGslOwnerImplicitObject)
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
    // Propagate origins through GSL pointer arithmetic and dereference
    // operators.
    switch (Callee->getOverloadedOperator()) {
    case OO_Arrow:
    case OO_Star:
    case OO_Plus:
    case OO_Minus:
    case OO_PlusPlus:
    case OO_MinusMinus:
      return true;
    default:
      break;
    }
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
                 ? IsGslOwnerImplicitObject &&
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
      if (!IsGslOwnerImplicitObject)
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

bool shouldTrackSecondArgument(const FunctionDecl *FD) {
  if (FD->getNumParams() < 2)
    return false;
  const auto *RD = FD->getParamDecl(1)->getType()->getAsCXXRecordDecl();
  if (!RD)
    return false;
  // For free-standing `+`/`-` operators annotated with `gsl::Pointer`, track
  // the second parameter when its type matches the return type.
  return RD->hasAttr<PointerAttr>() &&
         (FD->getOverloadedOperator() == OO_Plus ||
          FD->getOverloadedOperator() == OO_Minus) &&
         ASTContext::hasSameUnqualifiedType(FD->getParamDecl(1)->getType(),
                                            FD->getReturnType()) &&
         !isa<CXXMethodDecl>(FD);
}

template <typename T> static bool isRecordWithAttr(const CXXRecordDecl *RD) {
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

template <typename T> static bool isRecordWithAttr(QualType Type) {
  return isRecordWithAttr<T>(Type->getAsCXXRecordDecl());
}

bool isGslPointerType(QualType QT) { return isRecordWithAttr<PointerAttr>(QT); }
bool isGslOwnerType(QualType QT) { return isRecordWithAttr<OwnerAttr>(QT); }
bool isGslOwnerType(const CXXRecordDecl *RD) {
  return isRecordWithAttr<OwnerAttr>(RD);
}

bool isOwnerPtrCtor(const CXXConstructorDecl *Ctor, const ParmVarDecl *PVD) {
  return Ctor && PVD->getType()->isPointerType() &&
         isGslOwnerType(Ctor->getParent());
}

static StringRef getName(const CXXRecordDecl &RD) {
  if (const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(&RD))
    return CTSD->getSpecializedTemplate()->getName();
  if (RD.getIdentifier())
    return RD.getName();
  return "";
}

static StringRef getName(const FunctionDecl &FD) {
  if (FD.getIdentifier())
    return FD.getName();
  return "";
}

static bool isStdUniquePtr(const CXXRecordDecl &RD) {
  return RD.isInStdNamespace() && getName(RD) == "unique_ptr";
}

bool isUniquePtrRelease(const CXXMethodDecl &MD) {
  return MD.getIdentifier() && MD.getName() == "release" &&
         MD.getNumParams() == 0 && isStdUniquePtr(*MD.getParent());
}

bool isInvalidationMethod(const CXXMethodDecl &MD) {
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

  static const llvm::StringSet<> UniquePtr = {// Reallocation
                                              "reset"};

  const StringRef RecordName = getName(*RD);
  // TODO: Consider caching this lookup by CXXMethodDecl pointer if this
  // StringSwitch becomes a performance bottleneck.
  const llvm::StringSet<> *InvalidatingMethods =
      llvm::StringSwitch<const llvm::StringSet<> *>(RecordName)
          .Case("vector", &Vector)
          .Case("basic_string", &String)
          .Case("deque", &Deque)
          .Case("priority_queue", &PriorityQueue)
          .Cases({"set", "multiset", "map", "multimap", "unordered_set",
                  "unordered_multiset", "unordered_map", "unordered_multimap"},
                 &NodeBased)
          .Cases({"flat_map", "flat_set", "flat_multimap", "flat_multiset"},
                 &Flat)
          .Case("unique_ptr", &UniquePtr)
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
      return RecordName == "flat_map";
    default:
      return false;
    }
  }

  if (!MD.getIdentifier())
    return false;

  return InvalidatingMethods->contains(MD.getName());
}

bool destructsFirstArg(const FunctionDecl &FD) {
  if (isa<CXXDestructorDecl>(FD))
    return true;
  return isInStlNamespace(&FD) && getName(FD) == "destroy_at";
}

bool isStdCallableWrapperType(const CXXRecordDecl *RD) {
  if (!RD || !isInStlNamespace(RD))
    return false;
  StringRef Name = getName(*RD);
  return Name == "function" || Name == "move_only_function";
}

bool isStdReferenceCast(const FunctionDecl *FD) {
  if (!FD)
    return false;
  switch (FD->getBuiltinID()) {
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIforward:
  case Builtin::BIforward_like:
  case Builtin::BIas_const:
    return true;
  default:
    return false;
  }
}

} // namespace clang::lifetimes
