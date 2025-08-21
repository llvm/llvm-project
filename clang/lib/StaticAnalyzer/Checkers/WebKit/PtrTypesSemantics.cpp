//=======- PtrTypesSemantics.cpp ---------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PtrTypesSemantics.h"
#include "ASTUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include <optional>

using namespace clang;

namespace {

bool hasPublicMethodInBaseClass(const CXXRecordDecl *R, StringRef NameToMatch) {
  assert(R);
  assert(R->hasDefinition());

  for (const CXXMethodDecl *MD : R->methods()) {
    const auto MethodName = safeGetName(MD);
    if (MethodName == NameToMatch && MD->getAccess() == AS_public)
      return true;
  }
  return false;
}

} // namespace

namespace clang {

std::optional<const clang::CXXRecordDecl *>
hasPublicMethodInBase(const CXXBaseSpecifier *Base, StringRef NameToMatch) {
  assert(Base);

  const Type *T = Base->getType().getTypePtrOrNull();
  if (!T)
    return std::nullopt;

  const CXXRecordDecl *R = T->getAsCXXRecordDecl();
  if (!R) {
    auto CT = Base->getType().getCanonicalType();
    if (auto *TST = dyn_cast<TemplateSpecializationType>(CT)) {
      auto TmplName = TST->getTemplateName();
      if (!TmplName.isNull()) {
        if (auto *TD = TmplName.getAsTemplateDecl())
          R = dyn_cast_or_null<CXXRecordDecl>(TD->getTemplatedDecl());
      }
    }
    if (!R)
      return std::nullopt;
  }
  if (!R->hasDefinition())
    return std::nullopt;

  return hasPublicMethodInBaseClass(R, NameToMatch) ? R : nullptr;
}

std::optional<bool> isSmartPtrCompatible(const CXXRecordDecl *R,
                                         StringRef IncMethodName,
                                         StringRef DecMethodName) {
  assert(R);

  R = R->getDefinition();
  if (!R)
    return std::nullopt;

  bool hasRef = hasPublicMethodInBaseClass(R, IncMethodName);
  bool hasDeref = hasPublicMethodInBaseClass(R, DecMethodName);
  if (hasRef && hasDeref)
    return true;

  CXXBasePaths Paths;
  Paths.setOrigin(const_cast<CXXRecordDecl *>(R));

  bool AnyInconclusiveBase = false;
  const auto hasPublicRefInBase = [&](const CXXBaseSpecifier *Base,
                                      CXXBasePath &) {
    auto hasRefInBase = clang::hasPublicMethodInBase(Base, IncMethodName);
    if (!hasRefInBase) {
      AnyInconclusiveBase = true;
      return false;
    }
    return (*hasRefInBase) != nullptr;
  };

  hasRef = hasRef || R->lookupInBases(hasPublicRefInBase, Paths,
                                      /*LookupInDependent =*/true);
  if (AnyInconclusiveBase)
    return std::nullopt;

  Paths.clear();
  const auto hasPublicDerefInBase = [&](const CXXBaseSpecifier *Base,
                                        CXXBasePath &) {
    auto hasDerefInBase = clang::hasPublicMethodInBase(Base, DecMethodName);
    if (!hasDerefInBase) {
      AnyInconclusiveBase = true;
      return false;
    }
    return (*hasDerefInBase) != nullptr;
  };
  hasDeref = hasDeref || R->lookupInBases(hasPublicDerefInBase, Paths,
                                          /*LookupInDependent =*/true);
  if (AnyInconclusiveBase)
    return std::nullopt;

  return hasRef && hasDeref;
}

std::optional<bool> isRefCountable(const clang::CXXRecordDecl *R) {
  return isSmartPtrCompatible(R, "ref", "deref");
}

std::optional<bool> isCheckedPtrCapable(const clang::CXXRecordDecl *R) {
  return isSmartPtrCompatible(R, "incrementCheckedPtrCount",
                              "decrementCheckedPtrCount");
}

bool isRefType(const std::string &Name) {
  return Name == "Ref" || Name == "RefAllowingPartiallyDestroyed" ||
         Name == "RefPtr" || Name == "RefPtrAllowingPartiallyDestroyed";
}

bool isRetainPtr(const std::string &Name) {
  return Name == "RetainPtr" || Name == "RetainPtrArc";
}

bool isCheckedPtr(const std::string &Name) {
  return Name == "CheckedPtr" || Name == "CheckedRef";
}

bool isSmartPtrClass(const std::string &Name) {
  return isRefType(Name) || isCheckedPtr(Name) || isRetainPtr(Name) ||
         Name == "WeakPtr" || Name == "WeakPtrFactory" ||
         Name == "WeakPtrFactoryWithBitField" || Name == "WeakPtrImplBase" ||
         Name == "WeakPtrImplBaseSingleThread" || Name == "ThreadSafeWeakPtr" ||
         Name == "ThreadSafeWeakOrStrongPtr" ||
         Name == "ThreadSafeWeakPtrControlBlock" ||
         Name == "ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr";
}

bool isCtorOfRefCounted(const clang::FunctionDecl *F) {
  assert(F);
  const std::string &FunctionName = safeGetName(F);

  return isRefType(FunctionName) || FunctionName == "adoptRef" ||
         FunctionName == "UniqueRef" || FunctionName == "makeUniqueRef" ||
         FunctionName == "makeUniqueRefWithoutFastMallocCheck"

         || FunctionName == "String" || FunctionName == "AtomString" ||
         FunctionName == "UniqueString"
         // FIXME: Implement as attribute.
         || FunctionName == "Identifier";
}

bool isCtorOfCheckedPtr(const clang::FunctionDecl *F) {
  assert(F);
  return isCheckedPtr(safeGetName(F));
}

bool isCtorOfRetainPtr(const clang::FunctionDecl *F) {
  const std::string &FunctionName = safeGetName(F);
  return FunctionName == "RetainPtr" || FunctionName == "adoptNS" ||
         FunctionName == "adoptCF" || FunctionName == "retainPtr" ||
         FunctionName == "RetainPtrArc" || FunctionName == "adoptNSArc";
}

bool isCtorOfSafePtr(const clang::FunctionDecl *F) {
  return isCtorOfRefCounted(F) || isCtorOfCheckedPtr(F) || isCtorOfRetainPtr(F);
}

template <typename Predicate>
static bool isPtrOfType(const clang::QualType T, Predicate Pred) {
  QualType type = T;
  while (!type.isNull()) {
    if (auto *SpecialT = type->getAs<TemplateSpecializationType>()) {
      auto *Decl = SpecialT->getTemplateName().getAsTemplateDecl();
      return Decl && Pred(Decl->getNameAsString());
    } else if (auto *DTS = type->getAs<DeducedTemplateSpecializationType>()) {
      auto *Decl = DTS->getTemplateName().getAsTemplateDecl();
      return Decl && Pred(Decl->getNameAsString());
    } else
      break;
  }
  return false;
}

bool isRefOrCheckedPtrType(const clang::QualType T) {
  return isPtrOfType(
      T, [](auto Name) { return isRefType(Name) || isCheckedPtr(Name); });
}

bool isRetainPtrType(const clang::QualType T) {
  return isPtrOfType(T, [](auto Name) { return isRetainPtr(Name); });
}

bool isOwnerPtrType(const clang::QualType T) {
  return isPtrOfType(T, [](auto Name) {
    return isRefType(Name) || isCheckedPtr(Name) || Name == "unique_ptr" ||
           Name == "UniqueRef" || Name == "LazyUniqueRef";
  });
}

std::optional<bool> isUncounted(const QualType T) {
  if (auto *Subst = dyn_cast<SubstTemplateTypeParmType>(T)) {
    if (auto *Decl = Subst->getAssociatedDecl()) {
      if (isRefType(safeGetName(Decl)))
        return false;
    }
  }
  return isUncounted(T->getAsCXXRecordDecl());
}

std::optional<bool> isUnchecked(const QualType T) {
  if (auto *Subst = dyn_cast<SubstTemplateTypeParmType>(T)) {
    if (auto *Decl = Subst->getAssociatedDecl()) {
      if (isCheckedPtr(safeGetName(Decl)))
        return false;
    }
  }
  return isUnchecked(T->getAsCXXRecordDecl());
}

void RetainTypeChecker::visitTranslationUnitDecl(
    const TranslationUnitDecl *TUD) {
  IsARCEnabled = TUD->getLangOpts().ObjCAutoRefCount;
  DefaultSynthProperties = TUD->getLangOpts().ObjCDefaultSynthProperties;
}

void RetainTypeChecker::visitTypedef(const TypedefDecl *TD) {
  auto QT = TD->getUnderlyingType();
  if (!QT->isPointerType())
    return;

  auto PointeeQT = QT->getPointeeType();
  const RecordType *RT = PointeeQT->getAs<RecordType>();
  if (!RT) {
    if (TD->hasAttr<ObjCBridgeAttr>() || TD->hasAttr<ObjCBridgeMutableAttr>()) {
      RecordlessTypes.insert(TD->getASTContext()
                                 .getTypedefType(ElaboratedTypeKeyword::None,
                                                 /*Qualifier=*/std::nullopt, TD)
                                 .getTypePtr());
    }
    return;
  }

  for (auto *Redecl : RT->getOriginalDecl()->getMostRecentDecl()->redecls()) {
    if (Redecl->getAttr<ObjCBridgeAttr>() ||
        Redecl->getAttr<ObjCBridgeMutableAttr>()) {
      CFPointees.insert(RT);
      return;
    }
  }
}

bool RetainTypeChecker::isUnretained(const QualType QT, bool ignoreARC) {
  if (ento::cocoa::isCocoaObjectRef(QT) && (!IsARCEnabled || ignoreARC))
    return true;
  if (auto *RT = dyn_cast_or_null<RecordType>(
          QT.getCanonicalType()->getPointeeType().getTypePtrOrNull()))
    return CFPointees.contains(RT);
  return RecordlessTypes.contains(QT.getTypePtr());
}

std::optional<bool> isUnretained(const QualType T, bool IsARCEnabled) {
  if (auto *Subst = dyn_cast<SubstTemplateTypeParmType>(T)) {
    if (auto *Decl = Subst->getAssociatedDecl()) {
      if (isRetainPtr(safeGetName(Decl)))
        return false;
    }
  }
  if ((ento::cocoa::isCocoaObjectRef(T) && !IsARCEnabled) ||
      ento::coreFoundation::isCFObjectRef(T))
    return true;

  // RetainPtr strips typedef for CF*Ref. Manually check for struct __CF* types.
  auto CanonicalType = T.getCanonicalType();
  auto *Type = CanonicalType.getTypePtrOrNull();
  if (!Type)
    return false;
  auto Pointee = Type->getPointeeType();
  auto *PointeeType = Pointee.getTypePtrOrNull();
  if (!PointeeType)
    return false;
  auto *Record = PointeeType->getAsStructureType();
  if (!Record)
    return false;
  auto *Decl = Record->getOriginalDecl();
  if (!Decl)
    return false;
  auto TypeName = Decl->getName();
  return TypeName.starts_with("__CF") || TypeName.starts_with("__CG") ||
         TypeName.starts_with("__CM");
}

std::optional<bool> isUncounted(const CXXRecordDecl* Class)
{
  // Keep isRefCounted first as it's cheaper.
  if (!Class || isRefCounted(Class))
    return false;

  std::optional<bool> IsRefCountable = isRefCountable(Class);
  if (!IsRefCountable)
    return std::nullopt;

  return (*IsRefCountable);
}

std::optional<bool> isUnchecked(const CXXRecordDecl *Class) {
  if (!Class || isCheckedPtr(Class))
    return false; // Cheaper than below
  return isCheckedPtrCapable(Class);
}

std::optional<bool> isUncountedPtr(const QualType T) {
  if (T->isPointerType() || T->isReferenceType()) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl())
      return isUncounted(CXXRD);
  }
  return false;
}

std::optional<bool> isUncheckedPtr(const QualType T) {
  if (T->isPointerType() || T->isReferenceType()) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl())
      return isUnchecked(CXXRD);
  }
  return false;
}

std::optional<bool> isUnsafePtr(const QualType T, bool IsArcEnabled) {
  if (T->isPointerType() || T->isReferenceType()) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl()) {
      auto isUncountedPtr = isUncounted(CXXRD);
      auto isUncheckedPtr = isUnchecked(CXXRD);
      auto isUnretainedPtr = isUnretained(T, IsArcEnabled);
      std::optional<bool> result;
      if (isUncountedPtr)
        result = *isUncountedPtr;
      if (isUncheckedPtr)
        result = result ? *result || *isUncheckedPtr : *isUncheckedPtr;
      if (isUnretainedPtr)
        result = result ? *result || *isUnretainedPtr : *isUnretainedPtr;
      return result;
    }
  }
  return false;
}

std::optional<bool> isGetterOfSafePtr(const CXXMethodDecl *M) {
  assert(M);

  if (isa<CXXMethodDecl>(M)) {
    const CXXRecordDecl *calleeMethodsClass = M->getParent();
    auto className = safeGetName(calleeMethodsClass);
    auto method = safeGetName(M);

    if (isCheckedPtr(className) && (method == "get" || method == "ptr"))
      return true;

    if ((isRefType(className) && (method == "get" || method == "ptr")) ||
        ((className == "String" || className == "AtomString" ||
          className == "AtomStringImpl" || className == "UniqueString" ||
          className == "UniqueStringImpl" || className == "Identifier") &&
         method == "impl"))
      return true;

    if (isRetainPtr(className) && method == "get")
      return true;

    // Ref<T> -> T conversion
    // FIXME: Currently allowing any Ref<T> -> whatever cast.
    if (isRefType(className)) {
      if (auto *maybeRefToRawOperator = dyn_cast<CXXConversionDecl>(M)) {
        auto QT = maybeRefToRawOperator->getConversionType();
        auto *T = QT.getTypePtrOrNull();
        return T && (T->isPointerType() || T->isReferenceType());
      }
    }

    if (isCheckedPtr(className)) {
      if (auto *maybeRefToRawOperator = dyn_cast<CXXConversionDecl>(M)) {
        auto QT = maybeRefToRawOperator->getConversionType();
        auto *T = QT.getTypePtrOrNull();
        return T && (T->isPointerType() || T->isReferenceType());
      }
    }

    if (isRetainPtr(className)) {
      if (auto *maybeRefToRawOperator = dyn_cast<CXXConversionDecl>(M)) {
        auto QT = maybeRefToRawOperator->getConversionType();
        auto *T = QT.getTypePtrOrNull();
        return T && (T->isPointerType() || T->isReferenceType() ||
                     T->isObjCObjectPointerType());
      }
    }
  }
  return false;
}

bool isRefCounted(const CXXRecordDecl *R) {
  assert(R);
  if (auto *TmplR = R->getTemplateInstantiationPattern()) {
    // FIXME: String/AtomString/UniqueString
    const auto &ClassName = safeGetName(TmplR);
    return isRefType(ClassName);
  }
  return false;
}

bool isCheckedPtr(const CXXRecordDecl *R) {
  assert(R);
  if (auto *TmplR = R->getTemplateInstantiationPattern()) {
    const auto &ClassName = safeGetName(TmplR);
    return isCheckedPtr(ClassName);
  }
  return false;
}

bool isRetainPtr(const CXXRecordDecl *R) {
  assert(R);
  if (auto *TmplR = R->getTemplateInstantiationPattern())
    return isRetainPtr(safeGetName(TmplR));
  return false;
}

bool isSmartPtr(const CXXRecordDecl *R) {
  assert(R);
  if (auto *TmplR = R->getTemplateInstantiationPattern())
    return isSmartPtrClass(safeGetName(TmplR));
  return false;
}

bool isPtrConversion(const FunctionDecl *F) {
  assert(F);
  if (isCtorOfRefCounted(F))
    return true;

  // FIXME: check # of params == 1
  const auto FunctionName = safeGetName(F);
  if (FunctionName == "getPtr" || FunctionName == "WeakPtr" ||
      FunctionName == "dynamicDowncast" || FunctionName == "downcast" ||
      FunctionName == "checkedDowncast" || FunctionName == "bit_cast" ||
      FunctionName == "uncheckedDowncast" || FunctionName == "bitwise_cast" ||
      FunctionName == "bridge_cast" || FunctionName == "bridge_id_cast" ||
      FunctionName == "dynamic_cf_cast" || FunctionName == "checked_cf_cast" ||
      FunctionName == "dynamic_objc_cast" ||
      FunctionName == "checked_objc_cast")
    return true;

  auto ReturnType = F->getReturnType();
  if (auto *Type = ReturnType.getTypePtrOrNull()) {
    if (auto *AttrType = dyn_cast<AttributedType>(Type)) {
      if (auto *Attr = AttrType->getAttr()) {
        if (auto *AnnotateType = dyn_cast<AnnotateTypeAttr>(Attr)) {
          if (AnnotateType->getAnnotation() == "webkit.pointerconversion")
            return true;
        }
      }
    }
  }

  return false;
}

bool isTrivialBuiltinFunction(const FunctionDecl *F) {
  if (!F || !F->getDeclName().isIdentifier())
    return false;
  auto Name = F->getName();
  return Name.starts_with("__builtin") || Name == "__libcpp_verbose_abort" ||
         Name.starts_with("os_log") || Name.starts_with("_os_log");
}

bool isSingleton(const FunctionDecl *F) {
  assert(F);
  // FIXME: check # of params == 1
  if (auto *MethodDecl = dyn_cast<CXXMethodDecl>(F)) {
    if (!MethodDecl->isStatic())
      return false;
  }
  const auto &NameStr = safeGetName(F);
  StringRef Name = NameStr; // FIXME: Make safeGetName return StringRef.
  return Name == "singleton" || Name.ends_with("Singleton");
}

// We only care about statements so let's use the simple
// (non-recursive) visitor.
class TrivialFunctionAnalysisVisitor
    : public ConstStmtVisitor<TrivialFunctionAnalysisVisitor, bool> {

  // Returns false if at least one child is non-trivial.
  bool VisitChildren(const Stmt *S) {
    for (const Stmt *Child : S->children()) {
      if (Child && !Visit(Child))
        return false;
    }

    return true;
  }

  template <typename StmtOrDecl, typename CheckFunction>
  bool WithCachedResult(const StmtOrDecl *S, CheckFunction Function) {
    auto CacheIt = Cache.find(S);
    if (CacheIt != Cache.end())
      return CacheIt->second;

    // Treat a recursive statement to be trivial until proven otherwise.
    auto [RecursiveIt, IsNew] = RecursiveFn.insert(std::make_pair(S, true));
    if (!IsNew)
      return RecursiveIt->second;

    bool Result = Function();

    if (!Result) {
      for (auto &It : RecursiveFn)
        It.second = false;
    }
    RecursiveIt = RecursiveFn.find(S);
    assert(RecursiveIt != RecursiveFn.end());
    Result = RecursiveIt->second;
    RecursiveFn.erase(RecursiveIt);
    Cache[S] = Result;

    return Result;
  }

public:
  using CacheTy = TrivialFunctionAnalysis::CacheTy;

  TrivialFunctionAnalysisVisitor(CacheTy &Cache) : Cache(Cache) {}

  bool IsFunctionTrivial(const Decl *D) {
    if (auto *FnDecl = dyn_cast<FunctionDecl>(D)) {
      if (FnDecl->isVirtualAsWritten())
        return false;
    }
    return WithCachedResult(D, [&]() {
      if (auto *CtorDecl = dyn_cast<CXXConstructorDecl>(D)) {
        for (auto *CtorInit : CtorDecl->inits()) {
          if (!Visit(CtorInit->getInit()))
            return false;
        }
      }
      const Stmt *Body = D->getBody();
      if (!Body)
        return false;
      return Visit(Body);
    });
  }

  bool VisitStmt(const Stmt *S) {
    // All statements are non-trivial unless overriden later.
    // Don't even recurse into children by default.
    return false;
  }

  bool VisitAttributedStmt(const AttributedStmt *AS) {
    // Ignore attributes.
    return Visit(AS->getSubStmt());
  }

  bool VisitCompoundStmt(const CompoundStmt *CS) {
    // A compound statement is allowed as long each individual sub-statement
    // is trivial.
    return WithCachedResult(CS, [&]() { return VisitChildren(CS); });
  }

  bool VisitReturnStmt(const ReturnStmt *RS) {
    // A return statement is allowed as long as the return value is trivial.
    if (auto *RV = RS->getRetValue())
      return Visit(RV);
    return true;
  }

  bool VisitDeclStmt(const DeclStmt *DS) { return VisitChildren(DS); }
  bool VisitDoStmt(const DoStmt *DS) { return VisitChildren(DS); }
  bool VisitIfStmt(const IfStmt *IS) {
    return WithCachedResult(IS, [&]() { return VisitChildren(IS); });
  }
  bool VisitForStmt(const ForStmt *FS) {
    return WithCachedResult(FS, [&]() { return VisitChildren(FS); });
  }
  bool VisitCXXForRangeStmt(const CXXForRangeStmt *FS) {
    return WithCachedResult(FS, [&]() { return VisitChildren(FS); });
  }
  bool VisitWhileStmt(const WhileStmt *WS) {
    return WithCachedResult(WS, [&]() { return VisitChildren(WS); });
  }
  bool VisitSwitchStmt(const SwitchStmt *SS) { return VisitChildren(SS); }
  bool VisitCaseStmt(const CaseStmt *CS) { return VisitChildren(CS); }
  bool VisitDefaultStmt(const DefaultStmt *DS) { return VisitChildren(DS); }

  // break, continue, goto, and label statements are always trivial.
  bool VisitBreakStmt(const BreakStmt *) { return true; }
  bool VisitContinueStmt(const ContinueStmt *) { return true; }
  bool VisitGotoStmt(const GotoStmt *) { return true; }
  bool VisitLabelStmt(const LabelStmt *) { return true; }

  bool VisitUnaryOperator(const UnaryOperator *UO) {
    // Unary operators are trivial if its operand is trivial except co_await.
    return UO->getOpcode() != UO_Coawait && Visit(UO->getSubExpr());
  }

  bool VisitBinaryOperator(const BinaryOperator *BO) {
    // Binary operators are trivial if their operands are trivial.
    return Visit(BO->getLHS()) && Visit(BO->getRHS());
  }

  bool VisitCompoundAssignOperator(const CompoundAssignOperator *CAO) {
    // Compound assignment operator such as |= is trivial if its
    // subexpresssions are trivial.
    return VisitChildren(CAO);
  }

  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *ASE) {
    return VisitChildren(ASE);
  }

  bool VisitConditionalOperator(const ConditionalOperator *CO) {
    // Ternary operators are trivial if their conditions & values are trivial.
    return VisitChildren(CO);
  }

  bool VisitAtomicExpr(const AtomicExpr *E) { return VisitChildren(E); }

  bool VisitStaticAssertDecl(const StaticAssertDecl *SAD) {
    // Any static_assert is considered trivial.
    return true;
  }

  bool VisitCallExpr(const CallExpr *CE) {
    if (!checkArguments(CE))
      return false;

    auto *Callee = CE->getDirectCallee();
    if (!Callee)
      return false;

    if (isPtrConversion(Callee))
      return true;

    const auto &Name = safeGetName(Callee);

    if (Callee->isInStdNamespace() &&
        (Name == "addressof" || Name == "forward" || Name == "move"))
      return true;

    if (Name == "WTFCrashWithInfo" || Name == "WTFBreakpointTrap" ||
        Name == "WTFReportBacktrace" ||
        Name == "WTFCrashWithSecurityImplication" || Name == "WTFCrash" ||
        Name == "WTFReportAssertionFailure" || Name == "isMainThread" ||
        Name == "isMainThreadOrGCThread" || Name == "isMainRunLoop" ||
        Name == "isWebThread" || Name == "isUIThread" ||
        Name == "mayBeGCThread" || Name == "compilerFenceForCrash" ||
        isTrivialBuiltinFunction(Callee))
      return true;

    return IsFunctionTrivial(Callee);
  }

  bool
  VisitSubstNonTypeTemplateParmExpr(const SubstNonTypeTemplateParmExpr *E) {
    // Non-type template paramter is compile time constant and trivial.
    return true;
  }

  bool VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E) {
    return VisitChildren(E);
  }

  bool VisitPredefinedExpr(const PredefinedExpr *E) {
    // A predefined identifier such as "func" is considered trivial.
    return true;
  }

  bool VisitOffsetOfExpr(const OffsetOfExpr *OE) {
    // offsetof(T, D) is considered trivial.
    return true;
  }

  bool VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE) {
    if (!checkArguments(MCE))
      return false;

    bool TrivialThis = Visit(MCE->getImplicitObjectArgument());
    if (!TrivialThis)
      return false;

    auto *Callee = MCE->getMethodDecl();
    if (!Callee)
      return false;

    auto Name = safeGetName(Callee);
    if (Name == "ref" || Name == "incrementCheckedPtrCount")
      return true;

    std::optional<bool> IsGetterOfRefCounted = isGetterOfSafePtr(Callee);
    if (IsGetterOfRefCounted && *IsGetterOfRefCounted)
      return true;

    // Recursively descend into the callee to confirm that it's trivial as well.
    return IsFunctionTrivial(Callee);
  }

  bool VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) {
    if (!checkArguments(OCE))
      return false;
    auto *Callee = OCE->getCalleeDecl();
    if (!Callee)
      return false;
    // Recursively descend into the callee to confirm that it's trivial as well.
    return IsFunctionTrivial(Callee);
  }

  bool VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E) {
    if (auto *Expr = E->getExpr()) {
      if (!Visit(Expr))
        return false;
    }
    return true;
  }

  bool checkArguments(const CallExpr *CE) {
    for (const Expr *Arg : CE->arguments()) {
      if (Arg && !Visit(Arg))
        return false;
    }
    return true;
  }

  bool VisitCXXConstructExpr(const CXXConstructExpr *CE) {
    for (const Expr *Arg : CE->arguments()) {
      if (Arg && !Visit(Arg))
        return false;
    }

    // Recursively descend into the callee to confirm that it's trivial.
    return IsFunctionTrivial(CE->getConstructor());
  }

  bool VisitCXXInheritedCtorInitExpr(const CXXInheritedCtorInitExpr *E) {
    return IsFunctionTrivial(E->getConstructor());
  }

  bool VisitCXXNewExpr(const CXXNewExpr *NE) { return VisitChildren(NE); }

  bool VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
    return Visit(ICE->getSubExpr());
  }

  bool VisitExplicitCastExpr(const ExplicitCastExpr *ECE) {
    return Visit(ECE->getSubExpr());
  }

  bool VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *VMT) {
    return Visit(VMT->getSubExpr());
  }

  bool VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *BTE) {
    if (auto *Temp = BTE->getTemporary()) {
      if (!TrivialFunctionAnalysis::isTrivialImpl(Temp->getDestructor(), Cache))
        return false;
    }
    return Visit(BTE->getSubExpr());
  }

  bool VisitArrayInitLoopExpr(const ArrayInitLoopExpr *AILE) {
    return Visit(AILE->getCommonExpr()) && Visit(AILE->getSubExpr());
  }

  bool VisitArrayInitIndexExpr(const ArrayInitIndexExpr *AIIE) {
    return true; // The current array index in VisitArrayInitLoopExpr is always
                 // trivial.
  }

  bool VisitOpaqueValueExpr(const OpaqueValueExpr *OVE) {
    return Visit(OVE->getSourceExpr());
  }

  bool VisitExprWithCleanups(const ExprWithCleanups *EWC) {
    return Visit(EWC->getSubExpr());
  }

  bool VisitParenExpr(const ParenExpr *PE) { return Visit(PE->getSubExpr()); }

  bool VisitInitListExpr(const InitListExpr *ILE) {
    for (const Expr *Child : ILE->inits()) {
      if (Child && !Visit(Child))
        return false;
    }
    return true;
  }

  bool VisitMemberExpr(const MemberExpr *ME) {
    // Field access is allowed but the base pointer may itself be non-trivial.
    return Visit(ME->getBase());
  }

  bool VisitCXXThisExpr(const CXXThisExpr *CTE) {
    // The expression 'this' is always trivial, be it explicit or implicit.
    return true;
  }

  bool VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E) {
    // nullptr is trivial.
    return true;
  }

  bool VisitDeclRefExpr(const DeclRefExpr *DRE) {
    // The use of a variable is trivial.
    return true;
  }

  // Constant literal expressions are always trivial
  bool VisitIntegerLiteral(const IntegerLiteral *E) { return true; }
  bool VisitFloatingLiteral(const FloatingLiteral *E) { return true; }
  bool VisitFixedPointLiteral(const FixedPointLiteral *E) { return true; }
  bool VisitCharacterLiteral(const CharacterLiteral *E) { return true; }
  bool VisitStringLiteral(const StringLiteral *E) { return true; }
  bool VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) { return true; }

  bool VisitConstantExpr(const ConstantExpr *CE) {
    // Constant expressions are trivial.
    return true;
  }

  bool VisitImplicitValueInitExpr(const ImplicitValueInitExpr *IVIE) {
    // An implicit value initialization is trvial.
    return true;
  }

private:
  CacheTy &Cache;
  CacheTy RecursiveFn;
};

bool TrivialFunctionAnalysis::isTrivialImpl(
    const Decl *D, TrivialFunctionAnalysis::CacheTy &Cache) {
  TrivialFunctionAnalysisVisitor V(Cache);
  return V.IsFunctionTrivial(D);
}

bool TrivialFunctionAnalysis::isTrivialImpl(
    const Stmt *S, TrivialFunctionAnalysis::CacheTy &Cache) {
  TrivialFunctionAnalysisVisitor V(Cache);
  bool Result = V.Visit(S);
  assert(Cache.contains(S) && "Top-level statement not properly cached!");
  return Result;
}

} // namespace clang
