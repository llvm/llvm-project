//===- SemaTemplateDeduction.cpp - Template Argument Deduction ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements C++ template argument deduction.
//
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "TypeLocBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TemplateKinds.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include <algorithm>
#include <cassert>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace clang {

  /// Various flags that control template argument deduction.
  ///
  /// These flags can be bitwise-OR'd together.
  enum TemplateDeductionFlags {
    /// No template argument deduction flags, which indicates the
    /// strictest results for template argument deduction (as used for, e.g.,
    /// matching class template partial specializations).
    TDF_None = 0,

    /// Within template argument deduction from a function call, we are
    /// matching with a parameter type for which the original parameter was
    /// a reference.
    TDF_ParamWithReferenceType = 0x1,

    /// Within template argument deduction from a function call, we
    /// are matching in a case where we ignore cv-qualifiers.
    TDF_IgnoreQualifiers = 0x02,

    /// Within template argument deduction from a function call,
    /// we are matching in a case where we can perform template argument
    /// deduction from a template-id of a derived class of the argument type.
    TDF_DerivedClass = 0x04,

    /// Allow non-dependent types to differ, e.g., when performing
    /// template argument deduction from a function call where conversions
    /// may apply.
    TDF_SkipNonDependent = 0x08,

    /// Whether we are performing template argument deduction for
    /// parameters and arguments in a top-level template argument
    TDF_TopLevelParameterTypeList = 0x10,

    /// Within template argument deduction from overload resolution per
    /// C++ [over.over] allow matching function types that are compatible in
    /// terms of noreturn and default calling convention adjustments, or
    /// similarly matching a declared template specialization against a
    /// possible template, per C++ [temp.deduct.decl]. In either case, permit
    /// deduction where the parameter is a function type that can be converted
    /// to the argument type.
    TDF_AllowCompatibleFunctionType = 0x20,

    /// Within template argument deduction for a conversion function, we are
    /// matching with an argument type for which the original argument was
    /// a reference.
    TDF_ArgWithReferenceType = 0x40,
  };
}

using namespace clang;
using namespace sema;

/// The kind of PartialOrdering we're performing template argument deduction
/// for (C++11 [temp.deduct.partial]).
enum class PartialOrderingKind { None, NonCall, Call };

static TemplateDeductionResult DeduceTemplateArgumentsByTypeMatch(
    Sema &S, TemplateParameterList *TemplateParams, QualType Param,
    QualType Arg, TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced, unsigned TDF,
    PartialOrderingKind POK, bool DeducedFromArrayBound,
    bool *HasDeducedAnyParam);

/// What directions packs are allowed to match non-packs.
enum class PackFold { ParameterToArgument, ArgumentToParameter, Both };

static TemplateDeductionResult
DeduceTemplateArguments(Sema &S, TemplateParameterList *TemplateParams,
                        ArrayRef<TemplateArgument> Ps,
                        ArrayRef<TemplateArgument> As,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                        bool NumberOfArgumentsMustMatch, bool PartialOrdering,
                        PackFold PackFold, bool *HasDeducedAnyParam);

static void MarkUsedTemplateParameters(ASTContext &Ctx,
                                       const TemplateArgument &TemplateArg,
                                       bool OnlyDeduced, unsigned Depth,
                                       llvm::SmallBitVector &Used);

static void MarkUsedTemplateParameters(ASTContext &Ctx, QualType T,
                                       bool OnlyDeduced, unsigned Level,
                                       llvm::SmallBitVector &Deduced);

static const Expr *unwrapExpressionForDeduction(const Expr *E) {
  // If we are within an alias template, the expression may have undergone
  // any number of parameter substitutions already.
  while (true) {
    if (const auto *IC = dyn_cast<ImplicitCastExpr>(E))
      E = IC->getSubExpr();
    else if (const auto *CE = dyn_cast<ConstantExpr>(E))
      E = CE->getSubExpr();
    else if (const auto *Subst = dyn_cast<SubstNonTypeTemplateParmExpr>(E))
      E = Subst->getReplacement();
    else if (const auto *CCE = dyn_cast<CXXConstructExpr>(E)) {
      // Look through implicit copy construction from an lvalue of the same type.
      if (CCE->getParenOrBraceRange().isValid())
        break;
      // Note, there could be default arguments.
      assert(CCE->getNumArgs() >= 1 && "implicit construct expr should have 1 arg");
      E = CCE->getArg(0);
    } else
      break;
  }
  return E;
}

class NonTypeOrVarTemplateParmDecl {
public:
  NonTypeOrVarTemplateParmDecl(const NamedDecl *Template) : Template(Template) {
    assert(
        !Template || isa<NonTypeTemplateParmDecl>(Template) ||
        (isa<TemplateTemplateParmDecl>(Template) &&
         (cast<TemplateTemplateParmDecl>(Template)->templateParameterKind() ==
              TNK_Var_template ||
          cast<TemplateTemplateParmDecl>(Template)->templateParameterKind() ==
              TNK_Concept_template)));
  }

  QualType getType() const {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Template))
      return NTTP->getType();
    return getTemplate()->templateParameterKind() == TNK_Concept_template
               ? getTemplate()->getASTContext().BoolTy
               : getTemplate()->getASTContext().DependentTy;
  }

  unsigned getDepth() const {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Template))
      return NTTP->getDepth();
    return getTemplate()->getDepth();
  }

  unsigned getIndex() const {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Template))
      return NTTP->getIndex();
    return getTemplate()->getIndex();
  }

  const TemplateTemplateParmDecl *getTemplate() const {
    return cast<TemplateTemplateParmDecl>(Template);
  }

  const NonTypeTemplateParmDecl *getNTTP() const {
    return cast<NonTypeTemplateParmDecl>(Template);
  }

  TemplateParameter asTemplateParam() const {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Template))
      return const_cast<NonTypeTemplateParmDecl *>(NTTP);
    return const_cast<TemplateTemplateParmDecl *>(getTemplate());
  }

  bool isExpandedParameterPack() const {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Template))
      return NTTP->isExpandedParameterPack();
    return getTemplate()->isExpandedParameterPack();
  }

  SourceLocation getLocation() const {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Template))
      return NTTP->getLocation();
    return getTemplate()->getLocation();
  }

  operator bool() const { return Template; }

private:
  const NamedDecl *Template;
};

/// If the given expression is of a form that permits the deduction
/// of a non-type template parameter, return the declaration of that
/// non-type template parameter.
static NonTypeOrVarTemplateParmDecl
getDeducedNTTParameterFromExpr(const Expr *E, unsigned Depth) {
  // If we are within an alias template, the expression may have undergone
  // any number of parameter substitutions already.
  E = unwrapExpressionForDeduction(E);
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
      if (NTTP->getDepth() == Depth)
        return NTTP;

  if (const auto *ULE = dyn_cast<UnresolvedLookupExpr>(E);
      ULE && (ULE->isConceptReference() || ULE->isVarDeclReference())) {
    if (auto *TTP = ULE->getTemplateTemplateDecl()) {

      if (TTP->getDepth() == Depth)
        return TTP;
    }
  }
  return nullptr;
}

static const NonTypeOrVarTemplateParmDecl
getDeducedNTTParameterFromExpr(TemplateDeductionInfo &Info, Expr *E) {
  return getDeducedNTTParameterFromExpr(E, Info.getDeducedDepth());
}

/// Determine whether two declaration pointers refer to the same
/// declaration.
static bool isSameDeclaration(Decl *X, Decl *Y) {
  if (NamedDecl *NX = dyn_cast<NamedDecl>(X))
    X = NX->getUnderlyingDecl();
  if (NamedDecl *NY = dyn_cast<NamedDecl>(Y))
    Y = NY->getUnderlyingDecl();

  return X->getCanonicalDecl() == Y->getCanonicalDecl();
}

/// Verify that the given, deduced template arguments are compatible.
///
/// \returns The deduced template argument, or a NULL template argument if
/// the deduced template arguments were incompatible.
static DeducedTemplateArgument
checkDeducedTemplateArguments(ASTContext &Context,
                              const DeducedTemplateArgument &X,
                              const DeducedTemplateArgument &Y,
                              bool AggregateCandidateDeduction = false) {
  // We have no deduction for one or both of the arguments; they're compatible.
  if (X.isNull())
    return Y;
  if (Y.isNull())
    return X;

  // If we have two non-type template argument values deduced for the same
  // parameter, they must both match the type of the parameter, and thus must
  // match each other's type. As we're only keeping one of them, we must check
  // for that now. The exception is that if either was deduced from an array
  // bound, the type is permitted to differ.
  if (!X.wasDeducedFromArrayBound() && !Y.wasDeducedFromArrayBound()) {
    QualType XType = X.getNonTypeTemplateArgumentType();
    if (!XType.isNull()) {
      QualType YType = Y.getNonTypeTemplateArgumentType();
      if (YType.isNull() || !Context.hasSameType(XType, YType))
        return DeducedTemplateArgument();
    }
  }

  switch (X.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Non-deduced template arguments handled above");

  case TemplateArgument::Type: {
    // If two template type arguments have the same type, they're compatible.
    QualType TX = X.getAsType(), TY = Y.getAsType();
    if (Y.getKind() == TemplateArgument::Type && Context.hasSameType(TX, TY))
      return DeducedTemplateArgument(Context.getCommonSugaredType(TX, TY),
                                     X.wasDeducedFromArrayBound() ||
                                         Y.wasDeducedFromArrayBound());

    // If one of the two arguments was deduced from an array bound, the other
    // supersedes it.
    if (X.wasDeducedFromArrayBound() != Y.wasDeducedFromArrayBound())
      return X.wasDeducedFromArrayBound() ? Y : X;

    // The arguments are not compatible.
    return DeducedTemplateArgument();
  }

  case TemplateArgument::Integral:
    // If we deduced a constant in one case and either a dependent expression or
    // declaration in another case, keep the integral constant.
    // If both are integral constants with the same value, keep that value.
    if (Y.getKind() == TemplateArgument::Expression ||
        Y.getKind() == TemplateArgument::Declaration ||
        (Y.getKind() == TemplateArgument::Integral &&
         llvm::APSInt::isSameValue(X.getAsIntegral(), Y.getAsIntegral())))
      return X.wasDeducedFromArrayBound() ? Y : X;

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::StructuralValue:
    // If we deduced a value and a dependent expression, keep the value.
    if (Y.getKind() == TemplateArgument::Expression ||
        (Y.getKind() == TemplateArgument::StructuralValue &&
         X.structurallyEquals(Y)))
      return X;

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::Template:
    if (Y.getKind() == TemplateArgument::Template &&
        Context.hasSameTemplateName(X.getAsTemplate(), Y.getAsTemplate()))
      return X;

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::TemplateExpansion:
    if (Y.getKind() == TemplateArgument::TemplateExpansion &&
        Context.hasSameTemplateName(X.getAsTemplateOrTemplatePattern(),
                                    Y.getAsTemplateOrTemplatePattern()))
      return X;

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::Expression: {
    if (Y.getKind() != TemplateArgument::Expression)
      return checkDeducedTemplateArguments(Context, Y, X);

    // Compare the expressions for equality
    llvm::FoldingSetNodeID ID1, ID2;
    X.getAsExpr()->Profile(ID1, Context, true);
    Y.getAsExpr()->Profile(ID2, Context, true);
    if (ID1 == ID2)
      return X.wasDeducedFromArrayBound() ? Y : X;

    // Differing dependent expressions are incompatible.
    return DeducedTemplateArgument();
  }

  case TemplateArgument::Declaration:
    assert(!X.wasDeducedFromArrayBound());

    // If we deduced a declaration and a dependent expression, keep the
    // declaration.
    if (Y.getKind() == TemplateArgument::Expression)
      return X;

    // If we deduced a declaration and an integral constant, keep the
    // integral constant and whichever type did not come from an array
    // bound.
    if (Y.getKind() == TemplateArgument::Integral) {
      if (Y.wasDeducedFromArrayBound())
        return TemplateArgument(Context, Y.getAsIntegral(),
                                X.getParamTypeForDecl());
      return Y;
    }

    // If we deduced two declarations, make sure that they refer to the
    // same declaration.
    if (Y.getKind() == TemplateArgument::Declaration &&
        isSameDeclaration(X.getAsDecl(), Y.getAsDecl()))
      return X;

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::NullPtr:
    // If we deduced a null pointer and a dependent expression, keep the
    // null pointer.
    if (Y.getKind() == TemplateArgument::Expression)
      return TemplateArgument(Context.getCommonSugaredType(
                                  X.getNullPtrType(), Y.getAsExpr()->getType()),
                              true);

    // If we deduced a null pointer and an integral constant, keep the
    // integral constant.
    if (Y.getKind() == TemplateArgument::Integral)
      return Y;

    // If we deduced two null pointers, they are the same.
    if (Y.getKind() == TemplateArgument::NullPtr)
      return TemplateArgument(
          Context.getCommonSugaredType(X.getNullPtrType(), Y.getNullPtrType()),
          true);

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::Pack: {
    if (Y.getKind() != TemplateArgument::Pack ||
        (!AggregateCandidateDeduction && X.pack_size() != Y.pack_size()))
      return DeducedTemplateArgument();

    llvm::SmallVector<TemplateArgument, 8> NewPack;
    for (TemplateArgument::pack_iterator
             XA = X.pack_begin(),
             XAEnd = X.pack_end(), YA = Y.pack_begin(), YAEnd = Y.pack_end();
         XA != XAEnd; ++XA) {
      if (YA != YAEnd) {
        TemplateArgument Merged = checkDeducedTemplateArguments(
            Context, DeducedTemplateArgument(*XA, X.wasDeducedFromArrayBound()),
            DeducedTemplateArgument(*YA, Y.wasDeducedFromArrayBound()));
        if (Merged.isNull() && !(XA->isNull() && YA->isNull()))
          return DeducedTemplateArgument();
        NewPack.push_back(Merged);
        ++YA;
      } else {
        NewPack.push_back(*XA);
      }
    }

    return DeducedTemplateArgument(
        TemplateArgument::CreatePackCopy(Context, NewPack),
        X.wasDeducedFromArrayBound() && Y.wasDeducedFromArrayBound());
  }
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

/// Deduce the value of the given non-type template parameter
/// as the given deduced template argument. All non-type template parameter
/// deduction is funneled through here.
static TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S, TemplateParameterList *TemplateParams,
                              const NonTypeOrVarTemplateParmDecl NTTP,
                              const DeducedTemplateArgument &NewDeduced,
                              QualType ValueType, TemplateDeductionInfo &Info,
                              bool PartialOrdering,
                              SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                              bool *HasDeducedAnyParam) {
  assert(NTTP.getDepth() == Info.getDeducedDepth() &&
         "deducing non-type template argument with wrong depth");

  DeducedTemplateArgument Result = checkDeducedTemplateArguments(
      S.Context, Deduced[NTTP.getIndex()], NewDeduced);
  if (Result.isNull()) {
    Info.Param = NTTP.asTemplateParam();
    Info.FirstArg = Deduced[NTTP.getIndex()];
    Info.SecondArg = NewDeduced;
    return TemplateDeductionResult::Inconsistent;
  }
  Deduced[NTTP.getIndex()] = Result;
  if (!S.getLangOpts().CPlusPlus17)
    return TemplateDeductionResult::Success;

  if (NTTP.isExpandedParameterPack())
    // FIXME: We may still need to deduce parts of the type here! But we
    // don't have any way to find which slice of the type to use, and the
    // type stored on the NTTP itself is nonsense. Perhaps the type of an
    // expanded NTTP should be a pack expansion type?
    return TemplateDeductionResult::Success;

  // Get the type of the parameter for deduction. If it's a (dependent) array
  // or function type, we will not have decayed it yet, so do that now.
  QualType ParamType = S.Context.getAdjustedParameterType(NTTP.getType());
  if (auto *Expansion = dyn_cast<PackExpansionType>(ParamType))
    ParamType = Expansion->getPattern();

  // FIXME: It's not clear how deduction of a parameter of reference
  // type from an argument (of non-reference type) should be performed.
  // For now, we just make the argument have same reference type as the
  // parameter.
  if (ParamType->isReferenceType() && !ValueType->isReferenceType()) {
    if (ParamType->isRValueReferenceType())
      ValueType = S.Context.getRValueReferenceType(ValueType);
    else
      ValueType = S.Context.getLValueReferenceType(ValueType);
  }

  return DeduceTemplateArgumentsByTypeMatch(
      S, TemplateParams, ParamType, ValueType, Info, Deduced,
      TDF_SkipNonDependent | TDF_IgnoreQualifiers,
      PartialOrdering ? PartialOrderingKind::NonCall
                      : PartialOrderingKind::None,
      /*ArrayBound=*/NewDeduced.wasDeducedFromArrayBound(), HasDeducedAnyParam);
}

/// Deduce the value of the given non-type template parameter
/// from the given integral constant.
static TemplateDeductionResult DeduceNonTypeTemplateArgument(
    Sema &S, TemplateParameterList *TemplateParams,
    NonTypeOrVarTemplateParmDecl NTTP, const llvm::APSInt &Value,
    QualType ValueType, bool DeducedFromArrayBound, TemplateDeductionInfo &Info,
    bool PartialOrdering, SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    bool *HasDeducedAnyParam) {
  return DeduceNonTypeTemplateArgument(
      S, TemplateParams, NTTP,
      DeducedTemplateArgument(S.Context, Value, ValueType,
                              DeducedFromArrayBound),
      ValueType, Info, PartialOrdering, Deduced, HasDeducedAnyParam);
}

/// Deduce the value of the given non-type template parameter
/// from the given null pointer template argument type.
static TemplateDeductionResult
DeduceNullPtrTemplateArgument(Sema &S, TemplateParameterList *TemplateParams,
                              NonTypeOrVarTemplateParmDecl NTTP,
                              QualType NullPtrType, TemplateDeductionInfo &Info,
                              bool PartialOrdering,
                              SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                              bool *HasDeducedAnyParam) {
  Expr *Value = S.ImpCastExprToType(
                     new (S.Context) CXXNullPtrLiteralExpr(S.Context.NullPtrTy,
                                                           NTTP.getLocation()),
                     NullPtrType,
                     NullPtrType->isMemberPointerType() ? CK_NullToMemberPointer
                                                        : CK_NullToPointer)
                    .get();
  return DeduceNonTypeTemplateArgument(
      S, TemplateParams, NTTP, TemplateArgument(Value, /*IsCanonical=*/false),
      Value->getType(), Info, PartialOrdering, Deduced, HasDeducedAnyParam);
}

/// Deduce the value of the given non-type template parameter
/// from the given type- or value-dependent expression.
///
/// \returns true if deduction succeeded, false otherwise.
static TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S, TemplateParameterList *TemplateParams,
                              NonTypeOrVarTemplateParmDecl NTTP, Expr *Value,
                              TemplateDeductionInfo &Info, bool PartialOrdering,
                              SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                              bool *HasDeducedAnyParam) {
  return DeduceNonTypeTemplateArgument(
      S, TemplateParams, NTTP, TemplateArgument(Value, /*IsCanonical=*/false),
      Value->getType(), Info, PartialOrdering, Deduced, HasDeducedAnyParam);
}

/// Deduce the value of the given non-type template parameter
/// from the given declaration.
///
/// \returns true if deduction succeeded, false otherwise.
static TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S, TemplateParameterList *TemplateParams,
                              NonTypeOrVarTemplateParmDecl NTTP, ValueDecl *D,
                              QualType T, TemplateDeductionInfo &Info,
                              bool PartialOrdering,
                              SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                              bool *HasDeducedAnyParam) {
  TemplateArgument New(D, T);
  return DeduceNonTypeTemplateArgument(
      S, TemplateParams, NTTP, DeducedTemplateArgument(New), T, Info,
      PartialOrdering, Deduced, HasDeducedAnyParam);
}

static TemplateDeductionResult DeduceTemplateArguments(
    Sema &S, TemplateParameterList *TemplateParams, TemplateName Param,
    TemplateName Arg, TemplateDeductionInfo &Info,
    ArrayRef<TemplateArgument> DefaultArguments, bool PartialOrdering,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    bool *HasDeducedAnyParam) {
  TemplateDecl *ParamDecl = Param.getAsTemplateDecl();
  if (!ParamDecl) {
    // The parameter type is dependent and is not a template template parameter,
    // so there is nothing that we can deduce.
    return TemplateDeductionResult::Success;
  }

  if (auto *TempParam = dyn_cast<TemplateTemplateParmDecl>(ParamDecl)) {
    // If we're not deducing at this depth, there's nothing to deduce.
    if (TempParam->getDepth() != Info.getDeducedDepth())
      return TemplateDeductionResult::Success;

    ArrayRef<NamedDecl *> Params =
        ParamDecl->getTemplateParameters()->asArray();
    unsigned StartPos = 0;
    for (unsigned I = 0, E = std::min(Params.size(), DefaultArguments.size());
         I < E; ++I) {
      if (Params[I]->isParameterPack()) {
        StartPos = DefaultArguments.size();
        break;
      }
      StartPos = I + 1;
    }

    // Provisional resolution for CWG2398: If Arg names a template
    // specialization, then we deduce a synthesized template name
    // based on A, but using the TS's extra arguments, relative to P, as
    // defaults.
    DeducedTemplateArgument NewDeduced =
        PartialOrdering
            ? TemplateArgument(S.Context.getDeducedTemplateName(
                  Arg, {StartPos, DefaultArguments.drop_front(StartPos)}))
            : Arg;

    DeducedTemplateArgument Result = checkDeducedTemplateArguments(
        S.Context, Deduced[TempParam->getIndex()], NewDeduced);
    if (Result.isNull()) {
      Info.Param = TempParam;
      Info.FirstArg = Deduced[TempParam->getIndex()];
      Info.SecondArg = NewDeduced;
      return TemplateDeductionResult::Inconsistent;
    }

    Deduced[TempParam->getIndex()] = Result;
    if (HasDeducedAnyParam)
      *HasDeducedAnyParam = true;
    return TemplateDeductionResult::Success;
  }

  // Verify that the two template names are equivalent.
  if (S.Context.hasSameTemplateName(
          Param, Arg, /*IgnoreDeduced=*/DefaultArguments.size() != 0))
    return TemplateDeductionResult::Success;

  // Mismatch of non-dependent template parameter to argument.
  Info.FirstArg = TemplateArgument(Param);
  Info.SecondArg = TemplateArgument(Arg);
  return TemplateDeductionResult::NonDeducedMismatch;
}

/// Deduce the template arguments by comparing the template parameter
/// type (which is a template-id) with the template argument type.
///
/// \param S the Sema
///
/// \param TemplateParams the template parameters that we are deducing
///
/// \param P the parameter type
///
/// \param A the argument type
///
/// \param Info information about the template argument deduction itself
///
/// \param Deduced the deduced template arguments
///
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.

static const TemplateSpecializationType *getLastTemplateSpecType(QualType QT) {
  const TemplateSpecializationType *LastTST = nullptr;
  for (const Type *T = QT.getTypePtr(); /**/; /**/) {
    const TemplateSpecializationType *TST =
        T->getAs<TemplateSpecializationType>();
    if (!TST)
      return LastTST;
    if (!TST->isSugared())
      return TST;
    LastTST = TST;
    T = TST->desugar().getTypePtr();
  }
}

static TemplateDeductionResult
DeduceTemplateSpecArguments(Sema &S, TemplateParameterList *TemplateParams,
                            const QualType P, QualType A,
                            TemplateDeductionInfo &Info, bool PartialOrdering,
                            SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                            bool *HasDeducedAnyParam) {
  TemplateName TNP;
  ArrayRef<TemplateArgument> PResolved;
  if (isa<TemplateSpecializationType>(P.getCanonicalType())) {
    const TemplateSpecializationType *TP = ::getLastTemplateSpecType(P);
    TNP = TP->getTemplateName();
    // FIXME: To preserve sugar, the TST needs to carry sugared resolved
    // arguments.
    PResolved = TP->getCanonicalTypeInternal()
                    ->castAs<TemplateSpecializationType>()
                    ->template_arguments();
  } else {
    const auto *TT = P->castAs<InjectedClassNameType>();
    TNP = TT->getTemplateName(S.Context);
    PResolved = TT->getTemplateArgs(S.Context);
  }

  // If the parameter is an alias template, there is nothing to deduce.
  if (const auto *TD = TNP.getAsTemplateDecl(); TD && TD->isTypeAlias())
    return TemplateDeductionResult::Success;

  // Check whether the template argument is a dependent template-id.
  if (isa<TemplateSpecializationType>(A.getCanonicalType())) {
    const TemplateSpecializationType *SA = ::getLastTemplateSpecType(A);
    TemplateName TNA = SA->getTemplateName();

    // If the argument is an alias template, there is nothing to deduce.
    if (const auto *TD = TNA.getAsTemplateDecl(); TD && TD->isTypeAlias())
      return TemplateDeductionResult::Success;

    // FIXME: To preserve sugar, the TST needs to carry sugared resolved
    // arguments.
    ArrayRef<TemplateArgument> AResolved =
        SA->getCanonicalTypeInternal()
            ->castAs<TemplateSpecializationType>()
            ->template_arguments();

    // Perform template argument deduction for the template name.
    if (auto Result = DeduceTemplateArguments(S, TemplateParams, TNP, TNA, Info,
                                              /*DefaultArguments=*/AResolved,
                                              PartialOrdering, Deduced,
                                              HasDeducedAnyParam);
        Result != TemplateDeductionResult::Success)
      return Result;

    // Perform template argument deduction on each template
    // argument. Ignore any missing/extra arguments, since they could be
    // filled in by default arguments.
    return DeduceTemplateArguments(
        S, TemplateParams, PResolved, AResolved, Info, Deduced,
        /*NumberOfArgumentsMustMatch=*/false, PartialOrdering,
        PackFold::ParameterToArgument, HasDeducedAnyParam);
  }

  // If the argument type is a class template specialization, we
  // perform template argument deduction using its template
  // arguments.
  const auto *TA = A->getAs<TagType>();
  TemplateName TNA;
  if (TA) {
    // FIXME: Can't use the template arguments from this TST, as they are not
    // resolved.
    if (const auto *TST = A->getAsNonAliasTemplateSpecializationType())
      TNA = TST->getTemplateName();
    else
      TNA = TA->getTemplateName(S.Context);
  }
  if (TNA.isNull()) {
    Info.FirstArg = TemplateArgument(P);
    Info.SecondArg = TemplateArgument(A);
    return TemplateDeductionResult::NonDeducedMismatch;
  }

  ArrayRef<TemplateArgument> AResolved = TA->getTemplateArgs(S.Context);
  // Perform template argument deduction for the template name.
  if (auto Result =
          DeduceTemplateArguments(S, TemplateParams, TNP, TNA, Info,
                                  /*DefaultArguments=*/AResolved,
                                  PartialOrdering, Deduced, HasDeducedAnyParam);
      Result != TemplateDeductionResult::Success)
    return Result;

  // Perform template argument deduction for the template arguments.
  return DeduceTemplateArguments(
      S, TemplateParams, PResolved, AResolved, Info, Deduced,
      /*NumberOfArgumentsMustMatch=*/true, PartialOrdering,
      PackFold::ParameterToArgument, HasDeducedAnyParam);
}

static bool IsPossiblyOpaquelyQualifiedTypeInternal(const Type *T) {
  assert(T->isCanonicalUnqualified());

  switch (T->getTypeClass()) {
  case Type::TypeOfExpr:
  case Type::TypeOf:
  case Type::DependentName:
  case Type::Decltype:
  case Type::PackIndexing:
  case Type::UnresolvedUsing:
  case Type::TemplateTypeParm:
  case Type::Auto:
    return true;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
  case Type::DependentSizedArray:
    return IsPossiblyOpaquelyQualifiedTypeInternal(
        cast<ArrayType>(T)->getElementType().getTypePtr());

  default:
    return false;
  }
}

/// Determines whether the given type is an opaque type that
/// might be more qualified when instantiated.
static bool IsPossiblyOpaquelyQualifiedType(QualType T) {
  return IsPossiblyOpaquelyQualifiedTypeInternal(
      T->getCanonicalTypeInternal().getTypePtr());
}

/// Helper function to build a TemplateParameter when we don't
/// know its type statically.
static TemplateParameter makeTemplateParameter(Decl *D) {
  if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(D))
    return TemplateParameter(TTP);
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D))
    return TemplateParameter(NTTP);

  return TemplateParameter(cast<TemplateTemplateParmDecl>(D));
}

/// A pack that we're currently deducing.
struct clang::DeducedPack {
  // The index of the pack.
  unsigned Index;

  // The old value of the pack before we started deducing it.
  DeducedTemplateArgument Saved;

  // A deferred value of this pack from an inner deduction, that couldn't be
  // deduced because this deduction hadn't happened yet.
  DeducedTemplateArgument DeferredDeduction;

  // The new value of the pack.
  SmallVector<DeducedTemplateArgument, 4> New;

  // The outer deduction for this pack, if any.
  DeducedPack *Outer = nullptr;

  DeducedPack(unsigned Index) : Index(Index) {}
};

namespace {

/// A scope in which we're performing pack deduction.
class PackDeductionScope {
public:
  /// Prepare to deduce the packs named within Pattern.
  /// \param FinishingDeduction Don't attempt to deduce the pack. Useful when
  /// just checking a previous deduction of the pack.
  PackDeductionScope(Sema &S, TemplateParameterList *TemplateParams,
                     SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                     TemplateDeductionInfo &Info, TemplateArgument Pattern,
                     bool DeducePackIfNotAlreadyDeduced = false,
                     bool FinishingDeduction = false)
      : S(S), TemplateParams(TemplateParams), Deduced(Deduced), Info(Info),
        DeducePackIfNotAlreadyDeduced(DeducePackIfNotAlreadyDeduced),
        FinishingDeduction(FinishingDeduction) {
    unsigned NumNamedPacks = addPacks(Pattern);
    finishConstruction(NumNamedPacks);
  }

  /// Prepare to directly deduce arguments of the parameter with index \p Index.
  PackDeductionScope(Sema &S, TemplateParameterList *TemplateParams,
                     SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                     TemplateDeductionInfo &Info, unsigned Index)
      : S(S), TemplateParams(TemplateParams), Deduced(Deduced), Info(Info) {
    addPack(Index);
    finishConstruction(1);
  }

private:
  void addPack(unsigned Index) {
    // Save the deduced template argument for the parameter pack expanded
    // by this pack expansion, then clear out the deduction.
    DeducedFromEarlierParameter = !Deduced[Index].isNull();
    DeducedPack Pack(Index);
    if (!FinishingDeduction) {
      Pack.Saved = Deduced[Index];
      Deduced[Index] = TemplateArgument();
    }

    // FIXME: What if we encounter multiple packs with different numbers of
    // pre-expanded expansions? (This should already have been diagnosed
    // during substitution.)
    if (UnsignedOrNone ExpandedPackExpansions =
            getExpandedPackSize(TemplateParams->getParam(Index)))
      FixedNumExpansions = ExpandedPackExpansions;

    Packs.push_back(Pack);
  }

  unsigned addPacks(TemplateArgument Pattern) {
    // Compute the set of template parameter indices that correspond to
    // parameter packs expanded by the pack expansion.
    llvm::SmallBitVector SawIndices(TemplateParams->size());
    llvm::SmallVector<TemplateArgument, 4> ExtraDeductions;

    auto AddPack = [&](unsigned Index) {
      if (SawIndices[Index])
        return;
      SawIndices[Index] = true;
      addPack(Index);

      // Deducing a parameter pack that is a pack expansion also constrains the
      // packs appearing in that parameter to have the same deduced arity. Also,
      // in C++17 onwards, deducing a non-type template parameter deduces its
      // type, so we need to collect the pending deduced values for those packs.
      if (auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(
              TemplateParams->getParam(Index))) {
        if (!NTTP->isExpandedParameterPack())
          // FIXME: CWG2982 suggests a type-constraint forms a non-deduced
          // context, however it is not yet resolved.
          if (auto *Expansion = dyn_cast<PackExpansionType>(
                  S.Context.getUnconstrainedType(NTTP->getType())))
            ExtraDeductions.push_back(Expansion->getPattern());
      }
      // FIXME: Also collect the unexpanded packs in any type and template
      // parameter packs that are pack expansions.
    };

    auto Collect = [&](TemplateArgument Pattern) {
      SmallVector<UnexpandedParameterPack, 2> Unexpanded;
      S.collectUnexpandedParameterPacks(Pattern, Unexpanded);
      for (unsigned I = 0, N = Unexpanded.size(); I != N; ++I) {
        unsigned Depth, Index;
        std::tie(Depth, Index) = getDepthAndIndex(Unexpanded[I]);
        if (Depth == Info.getDeducedDepth())
          AddPack(Index);
      }
    };

    // Look for unexpanded packs in the pattern.
    Collect(Pattern);
    assert(!Packs.empty() && "Pack expansion without unexpanded packs?");

    unsigned NumNamedPacks = Packs.size();

    // Also look for unexpanded packs that are indirectly deduced by deducing
    // the sizes of the packs in this pattern.
    while (!ExtraDeductions.empty())
      Collect(ExtraDeductions.pop_back_val());

    return NumNamedPacks;
  }

  void finishConstruction(unsigned NumNamedPacks) {
    // Dig out the partially-substituted pack, if there is one.
    const TemplateArgument *PartialPackArgs = nullptr;
    unsigned NumPartialPackArgs = 0;
    std::pair<unsigned, unsigned> PartialPackDepthIndex(-1u, -1u);
    if (auto *Scope = S.CurrentInstantiationScope)
      if (auto *Partial = Scope->getPartiallySubstitutedPack(
              &PartialPackArgs, &NumPartialPackArgs))
        PartialPackDepthIndex = getDepthAndIndex(Partial);

    // This pack expansion will have been partially or fully expanded if
    // it only names explicitly-specified parameter packs (including the
    // partially-substituted one, if any).
    bool IsExpanded = true;
    for (unsigned I = 0; I != NumNamedPacks; ++I) {
      if (Packs[I].Index >= Info.getNumExplicitArgs()) {
        IsExpanded = false;
        IsPartiallyExpanded = false;
        break;
      }
      if (PartialPackDepthIndex ==
            std::make_pair(Info.getDeducedDepth(), Packs[I].Index)) {
        IsPartiallyExpanded = true;
      }
    }

    // Skip over the pack elements that were expanded into separate arguments.
    // If we partially expanded, this is the number of partial arguments.
    // FIXME: `&& FixedNumExpansions` is a workaround for UB described in
    // https://github.com/llvm/llvm-project/issues/100095
    if (IsPartiallyExpanded)
      PackElements += NumPartialPackArgs;
    else if (IsExpanded && FixedNumExpansions)
      PackElements += *FixedNumExpansions;

    for (auto &Pack : Packs) {
      if (Info.PendingDeducedPacks.size() > Pack.Index)
        Pack.Outer = Info.PendingDeducedPacks[Pack.Index];
      else
        Info.PendingDeducedPacks.resize(Pack.Index + 1);
      Info.PendingDeducedPacks[Pack.Index] = &Pack;

      if (PartialPackDepthIndex ==
            std::make_pair(Info.getDeducedDepth(), Pack.Index)) {
        Pack.New.append(PartialPackArgs, PartialPackArgs + NumPartialPackArgs);
        // We pre-populate the deduced value of the partially-substituted
        // pack with the specified value. This is not entirely correct: the
        // value is supposed to have been substituted, not deduced, but the
        // cases where this is observable require an exact type match anyway.
        //
        // FIXME: If we could represent a "depth i, index j, pack elem k"
        // parameter, we could substitute the partially-substituted pack
        // everywhere and avoid this.
        if (!FinishingDeduction && !IsPartiallyExpanded)
          Deduced[Pack.Index] = Pack.New[PackElements];
      }
    }
  }

public:
  ~PackDeductionScope() {
    for (auto &Pack : Packs)
      Info.PendingDeducedPacks[Pack.Index] = Pack.Outer;
  }

  // Return the size of the saved packs if all of them has the same size.
  UnsignedOrNone getSavedPackSizeIfAllEqual() const {
    unsigned PackSize = Packs[0].Saved.pack_size();

    if (std::all_of(Packs.begin() + 1, Packs.end(), [&PackSize](const auto &P) {
          return P.Saved.pack_size() == PackSize;
        }))
      return PackSize;
    return std::nullopt;
  }

  /// Determine whether this pack has already been deduced from a previous
  /// argument.
  bool isDeducedFromEarlierParameter() const {
    return DeducedFromEarlierParameter;
  }

  /// Determine whether this pack has already been partially expanded into a
  /// sequence of (prior) function parameters / template arguments.
  bool isPartiallyExpanded() { return IsPartiallyExpanded; }

  /// Determine whether this pack expansion scope has a known, fixed arity.
  /// This happens if it involves a pack from an outer template that has
  /// (notionally) already been expanded.
  bool hasFixedArity() { return static_cast<bool>(FixedNumExpansions); }

  /// Determine whether the next element of the argument is still part of this
  /// pack. This is the case unless the pack is already expanded to a fixed
  /// length.
  bool hasNextElement() {
    return !FixedNumExpansions || *FixedNumExpansions > PackElements;
  }

  /// Move to deducing the next element in each pack that is being deduced.
  void nextPackElement() {
    // Capture the deduced template arguments for each parameter pack expanded
    // by this pack expansion, add them to the list of arguments we've deduced
    // for that pack, then clear out the deduced argument.
    if (!FinishingDeduction) {
      for (auto &Pack : Packs) {
        DeducedTemplateArgument &DeducedArg = Deduced[Pack.Index];
        if (!Pack.New.empty() || !DeducedArg.isNull()) {
          while (Pack.New.size() < PackElements)
            Pack.New.push_back(DeducedTemplateArgument());
          if (Pack.New.size() == PackElements)
            Pack.New.push_back(DeducedArg);
          else
            Pack.New[PackElements] = DeducedArg;
          DeducedArg = Pack.New.size() > PackElements + 1
                           ? Pack.New[PackElements + 1]
                           : DeducedTemplateArgument();
        }
      }
    }
    ++PackElements;
  }

  /// Finish template argument deduction for a set of argument packs,
  /// producing the argument packs and checking for consistency with prior
  /// deductions.
  TemplateDeductionResult finish() {
    if (FinishingDeduction)
      return TemplateDeductionResult::Success;
    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    for (auto &Pack : Packs) {
      // Put back the old value for this pack.
      if (!FinishingDeduction)
        Deduced[Pack.Index] = Pack.Saved;

      // Always make sure the size of this pack is correct, even if we didn't
      // deduce any values for it.
      //
      // FIXME: This isn't required by the normative wording, but substitution
      // and post-substitution checking will always fail if the arity of any
      // pack is not equal to the number of elements we processed. (Either that
      // or something else has gone *very* wrong.) We're permitted to skip any
      // hard errors from those follow-on steps by the intent (but not the
      // wording) of C++ [temp.inst]p8:
      //
      //   If the function selected by overload resolution can be determined
      //   without instantiating a class template definition, it is unspecified
      //   whether that instantiation actually takes place
      Pack.New.resize(PackElements);

      // Build or find a new value for this pack.
      DeducedTemplateArgument NewPack;
      if (Pack.New.empty()) {
        // If we deduced an empty argument pack, create it now.
        NewPack = DeducedTemplateArgument(TemplateArgument::getEmptyPack());
      } else {
        TemplateArgument *ArgumentPack =
            new (S.Context) TemplateArgument[Pack.New.size()];
        std::copy(Pack.New.begin(), Pack.New.end(), ArgumentPack);
        NewPack = DeducedTemplateArgument(
            TemplateArgument(llvm::ArrayRef(ArgumentPack, Pack.New.size())),
            // FIXME: This is wrong, it's possible that some pack elements are
            // deduced from an array bound and others are not:
            //   template<typename ...T, T ...V> void g(const T (&...p)[V]);
            //   g({1, 2, 3}, {{}, {}});
            // ... should deduce T = {int, size_t (from array bound)}.
            Pack.New[0].wasDeducedFromArrayBound());
      }

      // Pick where we're going to put the merged pack.
      DeducedTemplateArgument *Loc;
      if (Pack.Outer) {
        if (Pack.Outer->DeferredDeduction.isNull()) {
          // Defer checking this pack until we have a complete pack to compare
          // it against.
          Pack.Outer->DeferredDeduction = NewPack;
          continue;
        }
        Loc = &Pack.Outer->DeferredDeduction;
      } else {
        Loc = &Deduced[Pack.Index];
      }

      // Check the new pack matches any previous value.
      DeducedTemplateArgument OldPack = *Loc;
      DeducedTemplateArgument Result = checkDeducedTemplateArguments(
          S.Context, OldPack, NewPack, DeducePackIfNotAlreadyDeduced);

      Info.AggregateDeductionCandidateHasMismatchedArity =
          OldPack.getKind() == TemplateArgument::Pack &&
          NewPack.getKind() == TemplateArgument::Pack &&
          OldPack.pack_size() != NewPack.pack_size() && !Result.isNull();

      // If we deferred a deduction of this pack, check that one now too.
      if (!Result.isNull() && !Pack.DeferredDeduction.isNull()) {
        OldPack = Result;
        NewPack = Pack.DeferredDeduction;
        Result = checkDeducedTemplateArguments(S.Context, OldPack, NewPack);
      }

      NamedDecl *Param = TemplateParams->getParam(Pack.Index);
      if (Result.isNull()) {
        Info.Param = makeTemplateParameter(Param);
        Info.FirstArg = OldPack;
        Info.SecondArg = NewPack;
        return TemplateDeductionResult::Inconsistent;
      }

      // If we have a pre-expanded pack and we didn't deduce enough elements
      // for it, fail deduction.
      if (UnsignedOrNone Expansions = getExpandedPackSize(Param)) {
        if (*Expansions != PackElements) {
          Info.Param = makeTemplateParameter(Param);
          Info.FirstArg = Result;
          return TemplateDeductionResult::IncompletePack;
        }
      }

      *Loc = Result;
    }

    return TemplateDeductionResult::Success;
  }

private:
  Sema &S;
  TemplateParameterList *TemplateParams;
  SmallVectorImpl<DeducedTemplateArgument> &Deduced;
  TemplateDeductionInfo &Info;
  unsigned PackElements = 0;
  bool IsPartiallyExpanded = false;
  bool DeducePackIfNotAlreadyDeduced = false;
  bool DeducedFromEarlierParameter = false;
  bool FinishingDeduction = false;
  /// The number of expansions, if we have a fully-expanded pack in this scope.
  UnsignedOrNone FixedNumExpansions = std::nullopt;

  SmallVector<DeducedPack, 2> Packs;
};

} // namespace

template <class T>
static TemplateDeductionResult DeduceForEachType(
    Sema &S, TemplateParameterList *TemplateParams, ArrayRef<QualType> Params,
    ArrayRef<QualType> Args, TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced, PartialOrderingKind POK,
    bool FinishingDeduction, T &&DeductFunc) {
  // C++0x [temp.deduct.type]p10:
  //   Similarly, if P has a form that contains (T), then each parameter type
  //   Pi of the respective parameter-type- list of P is compared with the
  //   corresponding parameter type Ai of the corresponding parameter-type-list
  //   of A. [...]
  unsigned ArgIdx = 0, ParamIdx = 0;
  for (; ParamIdx != Params.size(); ++ParamIdx) {
    // Check argument types.
    const PackExpansionType *Expansion
                                = dyn_cast<PackExpansionType>(Params[ParamIdx]);
    if (!Expansion) {
      // Simple case: compare the parameter and argument types at this point.

      // Make sure we have an argument.
      if (ArgIdx >= Args.size())
        return TemplateDeductionResult::MiscellaneousDeductionFailure;

      if (isa<PackExpansionType>(Args[ArgIdx])) {
        // C++0x [temp.deduct.type]p22:
        //   If the original function parameter associated with A is a function
        //   parameter pack and the function parameter associated with P is not
        //   a function parameter pack, then template argument deduction fails.
        return TemplateDeductionResult::MiscellaneousDeductionFailure;
      }

      if (TemplateDeductionResult Result =
              DeductFunc(S, TemplateParams, ParamIdx, ArgIdx,
                         Params[ParamIdx].getUnqualifiedType(),
                         Args[ArgIdx].getUnqualifiedType(), Info, Deduced, POK);
          Result != TemplateDeductionResult::Success)
        return Result;

      ++ArgIdx;
      continue;
    }

    // C++0x [temp.deduct.type]p10:
    //   If the parameter-declaration corresponding to Pi is a function
    //   parameter pack, then the type of its declarator- id is compared with
    //   each remaining parameter type in the parameter-type-list of A. Each
    //   comparison deduces template arguments for subsequent positions in the
    //   template parameter packs expanded by the function parameter pack.

    QualType Pattern = Expansion->getPattern();
    PackDeductionScope PackScope(S, TemplateParams, Deduced, Info, Pattern,
                                 /*DeducePackIfNotAlreadyDeduced=*/false,
                                 FinishingDeduction);

    // A pack scope with fixed arity is not really a pack any more, so is not
    // a non-deduced context.
    if (ParamIdx + 1 == Params.size() || PackScope.hasFixedArity()) {
      for (; ArgIdx < Args.size() && PackScope.hasNextElement(); ++ArgIdx) {
        // Deduce template arguments from the pattern.
        if (TemplateDeductionResult Result = DeductFunc(
                S, TemplateParams, ParamIdx, ArgIdx,
                Pattern.getUnqualifiedType(), Args[ArgIdx].getUnqualifiedType(),
                Info, Deduced, POK);
            Result != TemplateDeductionResult::Success)
          return Result;
        PackScope.nextPackElement();
      }
    } else {
      // C++0x [temp.deduct.type]p5:
      //   The non-deduced contexts are:
      //     - A function parameter pack that does not occur at the end of the
      //       parameter-declaration-clause.
      //
      // FIXME: There is no wording to say what we should do in this case. We
      // choose to resolve this by applying the same rule that is applied for a
      // function call: that is, deduce all contained packs to their
      // explicitly-specified values (or to <> if there is no such value).
      //
      // This is seemingly-arbitrarily different from the case of a template-id
      // with a non-trailing pack-expansion in its arguments, which renders the
      // entire template-argument-list a non-deduced context.

      // If the parameter type contains an explicitly-specified pack that we
      // could not expand, skip the number of parameters notionally created
      // by the expansion.
      UnsignedOrNone NumExpansions = Expansion->getNumExpansions();
      if (NumExpansions && !PackScope.isPartiallyExpanded()) {
        for (unsigned I = 0; I != *NumExpansions && ArgIdx < Args.size();
             ++I, ++ArgIdx)
          PackScope.nextPackElement();
      }
    }

    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    if (auto Result = PackScope.finish();
        Result != TemplateDeductionResult::Success)
      return Result;
  }

  // DR692, DR1395
  // C++0x [temp.deduct.type]p10:
  // If the parameter-declaration corresponding to P_i ...
  // During partial ordering, if Ai was originally a function parameter pack:
  // - if P does not contain a function parameter type corresponding to Ai then
  //   Ai is ignored;
  if (POK == PartialOrderingKind::Call && ArgIdx + 1 == Args.size() &&
      isa<PackExpansionType>(Args[ArgIdx]))
    return TemplateDeductionResult::Success;

  // Make sure we don't have any extra arguments.
  if (ArgIdx < Args.size())
    return TemplateDeductionResult::MiscellaneousDeductionFailure;

  return TemplateDeductionResult::Success;
}

/// Deduce the template arguments by comparing the list of parameter
/// types to the list of argument types, as in the parameter-type-lists of
/// function types (C++ [temp.deduct.type]p10).
///
/// \param S The semantic analysis object within which we are deducing
///
/// \param TemplateParams The template parameters that we are deducing
///
/// \param Params The list of parameter types
///
/// \param Args The list of argument types
///
/// \param Info information about the template argument deduction itself
///
/// \param Deduced the deduced template arguments
///
/// \param TDF bitwise OR of the TemplateDeductionFlags bits that describe
/// how template argument deduction is performed.
///
/// \param PartialOrdering If true, we are performing template argument
/// deduction for during partial ordering for a call
/// (C++0x [temp.deduct.partial]).
///
/// \param HasDeducedAnyParam If set, the object pointed at will indicate
/// whether any template parameter was deduced.
///
/// \param HasDeducedParam If set, the bit vector will be used to represent
/// which template parameters were deduced, in order.
///
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.
static TemplateDeductionResult DeduceTemplateArguments(
    Sema &S, TemplateParameterList *TemplateParams, ArrayRef<QualType> Params,
    ArrayRef<QualType> Args, TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced, unsigned TDF,
    PartialOrderingKind POK, bool *HasDeducedAnyParam,
    llvm::SmallBitVector *HasDeducedParam) {
  return ::DeduceForEachType(
      S, TemplateParams, Params, Args, Info, Deduced, POK,
      /*FinishingDeduction=*/false,
      [&](Sema &S, TemplateParameterList *TemplateParams, int ParamIdx,
          int ArgIdx, QualType P, QualType A, TemplateDeductionInfo &Info,
          SmallVectorImpl<DeducedTemplateArgument> &Deduced,
          PartialOrderingKind POK) {
        bool HasDeducedAnyParamCopy = false;
        TemplateDeductionResult TDR = DeduceTemplateArgumentsByTypeMatch(
            S, TemplateParams, P, A, Info, Deduced, TDF, POK,
            /*DeducedFromArrayBound=*/false, &HasDeducedAnyParamCopy);
        if (HasDeducedAnyParam && HasDeducedAnyParamCopy)
          *HasDeducedAnyParam = true;
        if (HasDeducedParam && HasDeducedAnyParamCopy)
          (*HasDeducedParam)[ParamIdx] = true;
        return TDR;
      });
}

/// Determine whether the parameter has qualifiers that the argument
/// lacks. Put another way, determine whether there is no way to add
/// a deduced set of qualifiers to the ParamType that would result in
/// its qualifiers matching those of the ArgType.
static bool hasInconsistentOrSupersetQualifiersOf(QualType ParamType,
                                                  QualType ArgType) {
  Qualifiers ParamQs = ParamType.getQualifiers();
  Qualifiers ArgQs = ArgType.getQualifiers();

  if (ParamQs == ArgQs)
    return false;

  // Mismatched (but not missing) Objective-C GC attributes.
  if (ParamQs.getObjCGCAttr() != ArgQs.getObjCGCAttr() &&
      ParamQs.hasObjCGCAttr())
    return true;

  // Mismatched (but not missing) address spaces.
  if (ParamQs.getAddressSpace() != ArgQs.getAddressSpace() &&
      ParamQs.hasAddressSpace())
    return true;

  // Mismatched (but not missing) Objective-C lifetime qualifiers.
  if (ParamQs.getObjCLifetime() != ArgQs.getObjCLifetime() &&
      ParamQs.hasObjCLifetime())
    return true;

  // CVR qualifiers inconsistent or a superset.
  return (ParamQs.getCVRQualifiers() & ~ArgQs.getCVRQualifiers()) != 0;
}

bool Sema::isSameOrCompatibleFunctionType(QualType P, QualType A) {
  const FunctionType *PF = P->getAs<FunctionType>(),
                     *AF = A->getAs<FunctionType>();

  // Just compare if not functions.
  if (!PF || !AF)
    return Context.hasSameType(P, A);

  // Noreturn and noexcept adjustment.
  if (QualType AdjustedParam; TryFunctionConversion(P, A, AdjustedParam))
    P = AdjustedParam;

  // FIXME: Compatible calling conventions.
  return Context.hasSameFunctionTypeIgnoringExceptionSpec(P, A);
}

/// Get the index of the first template parameter that was originally from the
/// innermost template-parameter-list. This is 0 except when we concatenate
/// the template parameter lists of a class template and a constructor template
/// when forming an implicit deduction guide.
static unsigned getFirstInnerIndex(FunctionTemplateDecl *FTD) {
  auto *Guide = dyn_cast<CXXDeductionGuideDecl>(FTD->getTemplatedDecl());
  if (!Guide || !Guide->isImplicit())
    return 0;
  return Guide->getDeducedTemplate()->getTemplateParameters()->size();
}

/// Determine whether a type denotes a forwarding reference.
static bool isForwardingReference(QualType Param, unsigned FirstInnerIndex) {
  // C++1z [temp.deduct.call]p3:
  //   A forwarding reference is an rvalue reference to a cv-unqualified
  //   template parameter that does not represent a template parameter of a
  //   class template.
  if (auto *ParamRef = Param->getAs<RValueReferenceType>()) {
    if (ParamRef->getPointeeType().getQualifiers())
      return false;
    auto *TypeParm = ParamRef->getPointeeType()->getAs<TemplateTypeParmType>();
    return TypeParm && TypeParm->getIndex() >= FirstInnerIndex;
  }
  return false;
}

///  Attempt to deduce the template arguments by checking the base types
///  according to (C++20 [temp.deduct.call] p4b3.
///
/// \param S the semantic analysis object within which we are deducing.
///
/// \param RD the top level record object we are deducing against.
///
/// \param TemplateParams the template parameters that we are deducing.
///
/// \param P the template specialization parameter type.
///
/// \param Info information about the template argument deduction itself.
///
/// \param Deduced the deduced template arguments.
///
/// \returns the result of template argument deduction with the bases. "invalid"
/// means no matches, "success" found a single item, and the
/// "MiscellaneousDeductionFailure" result happens when the match is ambiguous.
static TemplateDeductionResult
DeduceTemplateBases(Sema &S, const CXXRecordDecl *RD,
                    TemplateParameterList *TemplateParams, QualType P,
                    TemplateDeductionInfo &Info, bool PartialOrdering,
                    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                    bool *HasDeducedAnyParam) {
  // C++14 [temp.deduct.call] p4b3:
  //   If P is a class and P has the form simple-template-id, then the
  //   transformed A can be a derived class of the deduced A. Likewise if
  //   P is a pointer to a class of the form simple-template-id, the
  //   transformed A can be a pointer to a derived class pointed to by the
  //   deduced A. However, if there is a class C that is a (direct or
  //   indirect) base class of D and derived (directly or indirectly) from a
  //   class B and that would be a valid deduced A, the deduced A cannot be
  //   B or pointer to B, respectively.
  //
  //   These alternatives are considered only if type deduction would
  //   otherwise fail. If they yield more than one possible deduced A, the
  //   type deduction fails.

  // Use a breadth-first search through the bases to collect the set of
  // successful matches. Visited contains the set of nodes we have already
  // visited, while ToVisit is our stack of records that we still need to
  // visit.  Matches contains a list of matches that have yet to be
  // disqualified.
  llvm::SmallPtrSet<const CXXRecordDecl *, 8> Visited;
  SmallVector<QualType, 8> ToVisit;
  // We iterate over this later, so we have to use MapVector to ensure
  // determinism.
  struct MatchValue {
    SmallVector<DeducedTemplateArgument, 8> Deduced;
    bool HasDeducedAnyParam;
  };
  llvm::MapVector<const CXXRecordDecl *, MatchValue> Matches;

  auto AddBases = [&Visited, &ToVisit](const CXXRecordDecl *RD) {
    for (const auto &Base : RD->bases()) {
      QualType T = Base.getType();
      assert(T->isRecordType() && "Base class that isn't a record?");
      if (Visited.insert(T->getAsCXXRecordDecl()).second)
        ToVisit.push_back(T);
    }
  };

  // Set up the loop by adding all the bases.
  AddBases(RD);

  // Search each path of bases until we either run into a successful match
  // (where all bases of it are invalid), or we run out of bases.
  while (!ToVisit.empty()) {
    QualType NextT = ToVisit.pop_back_val();

    SmallVector<DeducedTemplateArgument, 8> DeducedCopy(Deduced.begin(),
                                                        Deduced.end());
    TemplateDeductionInfo BaseInfo(TemplateDeductionInfo::ForBase, Info);
    bool HasDeducedAnyParamCopy = false;
    TemplateDeductionResult BaseResult = DeduceTemplateSpecArguments(
        S, TemplateParams, P, NextT, BaseInfo, PartialOrdering, DeducedCopy,
        &HasDeducedAnyParamCopy);

    // If this was a successful deduction, add it to the list of matches,
    // otherwise we need to continue searching its bases.
    const CXXRecordDecl *RD = NextT->getAsCXXRecordDecl();
    if (BaseResult == TemplateDeductionResult::Success)
      Matches.insert({RD, {DeducedCopy, HasDeducedAnyParamCopy}});
    else
      AddBases(RD);
  }

  // At this point, 'Matches' contains a list of seemingly valid bases, however
  // in the event that we have more than 1 match, it is possible that the base
  // of one of the matches might be disqualified for being a base of another
  // valid match. We can count on cyclical instantiations being invalid to
  // simplify the disqualifications.  That is, if A & B are both matches, and B
  // inherits from A (disqualifying A), we know that A cannot inherit from B.
  if (Matches.size() > 1) {
    Visited.clear();
    for (const auto &Match : Matches)
      AddBases(Match.first);

    // We can give up once we have a single item (or have run out of things to
    // search) since cyclical inheritance isn't valid.
    while (Matches.size() > 1 && !ToVisit.empty()) {
      const CXXRecordDecl *RD = ToVisit.pop_back_val()->getAsCXXRecordDecl();
      Matches.erase(RD);

      // Always add all bases, since the inheritance tree can contain
      // disqualifications for multiple matches.
      AddBases(RD);
    }
  }

  if (Matches.empty())
    return TemplateDeductionResult::Invalid;
  if (Matches.size() > 1)
    return TemplateDeductionResult::MiscellaneousDeductionFailure;

  std::swap(Matches.front().second.Deduced, Deduced);
  if (bool HasDeducedAnyParamCopy = Matches.front().second.HasDeducedAnyParam;
      HasDeducedAnyParamCopy && HasDeducedAnyParam)
    *HasDeducedAnyParam = HasDeducedAnyParamCopy;
  return TemplateDeductionResult::Success;
}

/// When propagating a partial ordering kind into a NonCall context,
/// this is used to downgrade a 'Call' into a 'NonCall', so that
/// the kind still reflects whether we are in a partial ordering context.
static PartialOrderingKind
degradeCallPartialOrderingKind(PartialOrderingKind POK) {
  return std::min(POK, PartialOrderingKind::NonCall);
}

/// Deduce the template arguments by comparing the parameter type and
/// the argument type (C++ [temp.deduct.type]).
///
/// \param S the semantic analysis object within which we are deducing
///
/// \param TemplateParams the template parameters that we are deducing
///
/// \param P the parameter type
///
/// \param A the argument type
///
/// \param Info information about the template argument deduction itself
///
/// \param Deduced the deduced template arguments
///
/// \param TDF bitwise OR of the TemplateDeductionFlags bits that describe
/// how template argument deduction is performed.
///
/// \param PartialOrdering Whether we're performing template argument deduction
/// in the context of partial ordering (C++0x [temp.deduct.partial]).
///
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.
static TemplateDeductionResult DeduceTemplateArgumentsByTypeMatch(
    Sema &S, TemplateParameterList *TemplateParams, QualType P, QualType A,
    TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced, unsigned TDF,
    PartialOrderingKind POK, bool DeducedFromArrayBound,
    bool *HasDeducedAnyParam) {

  // If the argument type is a pack expansion, look at its pattern.
  // This isn't explicitly called out
  if (const auto *AExp = dyn_cast<PackExpansionType>(A))
    A = AExp->getPattern();
  assert(!isa<PackExpansionType>(A.getCanonicalType()));

  if (POK == PartialOrderingKind::Call) {
    // C++11 [temp.deduct.partial]p5:
    //   Before the partial ordering is done, certain transformations are
    //   performed on the types used for partial ordering:
    //     - If P is a reference type, P is replaced by the type referred to.
    const ReferenceType *PRef = P->getAs<ReferenceType>();
    if (PRef)
      P = PRef->getPointeeType();

    //     - If A is a reference type, A is replaced by the type referred to.
    const ReferenceType *ARef = A->getAs<ReferenceType>();
    if (ARef)
      A = A->getPointeeType();

    if (PRef && ARef && S.Context.hasSameUnqualifiedType(P, A)) {
      // C++11 [temp.deduct.partial]p9:
      //   If, for a given type, deduction succeeds in both directions (i.e.,
      //   the types are identical after the transformations above) and both
      //   P and A were reference types [...]:
      //     - if [one type] was an lvalue reference and [the other type] was
      //       not, [the other type] is not considered to be at least as
      //       specialized as [the first type]
      //     - if [one type] is more cv-qualified than [the other type],
      //       [the other type] is not considered to be at least as specialized
      //       as [the first type]
      // Objective-C ARC adds:
      //     - [one type] has non-trivial lifetime, [the other type] has
      //       __unsafe_unretained lifetime, and the types are otherwise
      //       identical
      //
      // A is "considered to be at least as specialized" as P iff deduction
      // succeeds, so we model this as a deduction failure. Note that
      // [the first type] is P and [the other type] is A here; the standard
      // gets this backwards.
      Qualifiers PQuals = P.getQualifiers(), AQuals = A.getQualifiers();
      if ((PRef->isLValueReferenceType() && !ARef->isLValueReferenceType()) ||
          PQuals.isStrictSupersetOf(AQuals) ||
          (PQuals.hasNonTrivialObjCLifetime() &&
           AQuals.getObjCLifetime() == Qualifiers::OCL_ExplicitNone &&
           PQuals.withoutObjCLifetime() == AQuals.withoutObjCLifetime())) {
        Info.FirstArg = TemplateArgument(P);
        Info.SecondArg = TemplateArgument(A);
        return TemplateDeductionResult::NonDeducedMismatch;
      }
    }
    Qualifiers DiscardedQuals;
    // C++11 [temp.deduct.partial]p7:
    //   Remove any top-level cv-qualifiers:
    //     - If P is a cv-qualified type, P is replaced by the cv-unqualified
    //       version of P.
    P = S.Context.getUnqualifiedArrayType(P, DiscardedQuals);
    //     - If A is a cv-qualified type, A is replaced by the cv-unqualified
    //       version of A.
    A = S.Context.getUnqualifiedArrayType(A, DiscardedQuals);
  } else {
    // C++0x [temp.deduct.call]p4 bullet 1:
    //   - If the original P is a reference type, the deduced A (i.e., the type
    //     referred to by the reference) can be more cv-qualified than the
    //     transformed A.
    if (TDF & TDF_ParamWithReferenceType) {
      Qualifiers Quals;
      QualType UnqualP = S.Context.getUnqualifiedArrayType(P, Quals);
      Quals.setCVRQualifiers(Quals.getCVRQualifiers() & A.getCVRQualifiers());
      P = S.Context.getQualifiedType(UnqualP, Quals);
    }

    if ((TDF & TDF_TopLevelParameterTypeList) && !P->isFunctionType()) {
      // C++0x [temp.deduct.type]p10:
      //   If P and A are function types that originated from deduction when
      //   taking the address of a function template (14.8.2.2) or when deducing
      //   template arguments from a function declaration (14.8.2.6) and Pi and
      //   Ai are parameters of the top-level parameter-type-list of P and A,
      //   respectively, Pi is adjusted if it is a forwarding reference and Ai
      //   is an lvalue reference, in
      //   which case the type of Pi is changed to be the template parameter
      //   type (i.e., T&& is changed to simply T). [ Note: As a result, when
      //   Pi is T&& and Ai is X&, the adjusted Pi will be T, causing T to be
      //   deduced as X&. - end note ]
      TDF &= ~TDF_TopLevelParameterTypeList;
      if (isForwardingReference(P, /*FirstInnerIndex=*/0) &&
          A->isLValueReferenceType())
        P = P->getPointeeType();
    }
  }

  // C++ [temp.deduct.type]p9:
  //   A template type argument T, a template template argument TT or a
  //   template non-type argument i can be deduced if P and A have one of
  //   the following forms:
  //
  //     T
  //     cv-list T
  if (const auto *TTP = P->getAs<TemplateTypeParmType>()) {
    // Just skip any attempts to deduce from a placeholder type or a parameter
    // at a different depth.
    if (A->isPlaceholderType() || Info.getDeducedDepth() != TTP->getDepth())
      return TemplateDeductionResult::Success;

    unsigned Index = TTP->getIndex();

    // If the argument type is an array type, move the qualifiers up to the
    // top level, so they can be matched with the qualifiers on the parameter.
    if (A->isArrayType()) {
      Qualifiers Quals;
      A = S.Context.getUnqualifiedArrayType(A, Quals);
      if (Quals)
        A = S.Context.getQualifiedType(A, Quals);
    }

    // The argument type can not be less qualified than the parameter
    // type.
    if (!(TDF & TDF_IgnoreQualifiers) &&
        hasInconsistentOrSupersetQualifiersOf(P, A)) {
      Info.Param = cast<TemplateTypeParmDecl>(TemplateParams->getParam(Index));
      Info.FirstArg = TemplateArgument(P);
      Info.SecondArg = TemplateArgument(A);
      return TemplateDeductionResult::Underqualified;
    }

    // Do not match a function type with a cv-qualified type.
    // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#1584
    if (A->isFunctionType() && P.hasQualifiers())
      return TemplateDeductionResult::NonDeducedMismatch;

    assert(TTP->getDepth() == Info.getDeducedDepth() &&
           "saw template type parameter with wrong depth");
    assert(A->getCanonicalTypeInternal() != S.Context.OverloadTy &&
           "Unresolved overloaded function");
    QualType DeducedType = A;

    // Remove any qualifiers on the parameter from the deduced type.
    // We checked the qualifiers for consistency above.
    Qualifiers DeducedQs = DeducedType.getQualifiers();
    Qualifiers ParamQs = P.getQualifiers();
    DeducedQs.removeCVRQualifiers(ParamQs.getCVRQualifiers());
    if (ParamQs.hasObjCGCAttr())
      DeducedQs.removeObjCGCAttr();
    if (ParamQs.hasAddressSpace())
      DeducedQs.removeAddressSpace();
    if (ParamQs.hasObjCLifetime())
      DeducedQs.removeObjCLifetime();

    // Objective-C ARC:
    //   If template deduction would produce a lifetime qualifier on a type
    //   that is not a lifetime type, template argument deduction fails.
    if (ParamQs.hasObjCLifetime() && !DeducedType->isObjCLifetimeType() &&
        !DeducedType->isDependentType()) {
      Info.Param = cast<TemplateTypeParmDecl>(TemplateParams->getParam(Index));
      Info.FirstArg = TemplateArgument(P);
      Info.SecondArg = TemplateArgument(A);
      return TemplateDeductionResult::Underqualified;
    }

    // Objective-C ARC:
    //   If template deduction would produce an argument type with lifetime type
    //   but no lifetime qualifier, the __strong lifetime qualifier is inferred.
    if (S.getLangOpts().ObjCAutoRefCount && DeducedType->isObjCLifetimeType() &&
        !DeducedQs.hasObjCLifetime())
      DeducedQs.setObjCLifetime(Qualifiers::OCL_Strong);

    DeducedType =
        S.Context.getQualifiedType(DeducedType.getUnqualifiedType(), DeducedQs);

    DeducedTemplateArgument NewDeduced(DeducedType, DeducedFromArrayBound);
    DeducedTemplateArgument Result =
        checkDeducedTemplateArguments(S.Context, Deduced[Index], NewDeduced);
    if (Result.isNull()) {
      // We can also get inconsistencies when matching NTTP type.
      switch (NamedDecl *Param = TemplateParams->getParam(Index);
              Param->getKind()) {
      case Decl::TemplateTypeParm:
        Info.Param = cast<TemplateTypeParmDecl>(Param);
        break;
      case Decl::NonTypeTemplateParm:
        Info.Param = cast<NonTypeTemplateParmDecl>(Param);
        break;
      case Decl::TemplateTemplateParm:
        Info.Param = cast<TemplateTemplateParmDecl>(Param);
        break;
      default:
        llvm_unreachable("unexpected kind");
      }
      Info.FirstArg = Deduced[Index];
      Info.SecondArg = NewDeduced;
      return TemplateDeductionResult::Inconsistent;
    }

    Deduced[Index] = Result;
    if (HasDeducedAnyParam)
      *HasDeducedAnyParam = true;
    return TemplateDeductionResult::Success;
  }

  // Set up the template argument deduction information for a failure.
  Info.FirstArg = TemplateArgument(P);
  Info.SecondArg = TemplateArgument(A);

  // If the parameter is an already-substituted template parameter
  // pack, do nothing: we don't know which of its arguments to look
  // at, so we have to wait until all of the parameter packs in this
  // expansion have arguments.
  if (P->getAs<SubstTemplateTypeParmPackType>())
    return TemplateDeductionResult::Success;

  // Check the cv-qualifiers on the parameter and argument types.
  if (!(TDF & TDF_IgnoreQualifiers)) {
    if (TDF & TDF_ParamWithReferenceType) {
      if (hasInconsistentOrSupersetQualifiersOf(P, A))
        return TemplateDeductionResult::NonDeducedMismatch;
    } else if (TDF & TDF_ArgWithReferenceType) {
      // C++ [temp.deduct.conv]p4:
      //   If the original A is a reference type, A can be more cv-qualified
      //   than the deduced A
      if (!A.getQualifiers().compatiblyIncludes(P.getQualifiers(),
                                                S.getASTContext()))
        return TemplateDeductionResult::NonDeducedMismatch;

      // Strip out all extra qualifiers from the argument to figure out the
      // type we're converting to, prior to the qualification conversion.
      Qualifiers Quals;
      A = S.Context.getUnqualifiedArrayType(A, Quals);
      A = S.Context.getQualifiedType(A, P.getQualifiers());
    } else if (!IsPossiblyOpaquelyQualifiedType(P)) {
      if (P.getCVRQualifiers() != A.getCVRQualifiers())
        return TemplateDeductionResult::NonDeducedMismatch;
    }
  }

  // If the parameter type is not dependent, there is nothing to deduce.
  if (!P->isDependentType()) {
    if (TDF & TDF_SkipNonDependent)
      return TemplateDeductionResult::Success;
    if ((TDF & TDF_IgnoreQualifiers) ? S.Context.hasSameUnqualifiedType(P, A)
                                     : S.Context.hasSameType(P, A))
      return TemplateDeductionResult::Success;
    if (TDF & TDF_AllowCompatibleFunctionType &&
        S.isSameOrCompatibleFunctionType(P, A))
      return TemplateDeductionResult::Success;
    if (!(TDF & TDF_IgnoreQualifiers))
      return TemplateDeductionResult::NonDeducedMismatch;
    // Otherwise, when ignoring qualifiers, the types not having the same
    // unqualified type does not mean they do not match, so in this case we
    // must keep going and analyze with a non-dependent parameter type.
  }

  switch (P.getCanonicalType()->getTypeClass()) {
    // Non-canonical types cannot appear here.
#define NON_CANONICAL_TYPE(Class, Base) \
  case Type::Class: llvm_unreachable("deducing non-canonical type: " #Class);
#define TYPE(Class, Base)
#include "clang/AST/TypeNodes.inc"

    case Type::TemplateTypeParm:
    case Type::SubstTemplateTypeParmPack:
      llvm_unreachable("Type nodes handled above");

    case Type::Auto:
      // C++23 [temp.deduct.funcaddr]/3:
      //   A placeholder type in the return type of a function template is a
      //   non-deduced context.
      // There's no corresponding wording for [temp.deduct.decl], but we treat
      // it the same to match other compilers.
      if (P->isDependentType())
        return TemplateDeductionResult::Success;
      [[fallthrough]];
    case Type::Builtin:
    case Type::VariableArray:
    case Type::Vector:
    case Type::FunctionNoProto:
    case Type::Record:
    case Type::Enum:
    case Type::ObjCObject:
    case Type::ObjCInterface:
    case Type::ObjCObjectPointer:
    case Type::BitInt:
      return (TDF & TDF_SkipNonDependent) ||
                     ((TDF & TDF_IgnoreQualifiers)
                          ? S.Context.hasSameUnqualifiedType(P, A)
                          : S.Context.hasSameType(P, A))
                 ? TemplateDeductionResult::Success
                 : TemplateDeductionResult::NonDeducedMismatch;

    //     _Complex T   [placeholder extension]
    case Type::Complex: {
      const auto *CP = P->castAs<ComplexType>(), *CA = A->getAs<ComplexType>();
      if (!CA)
        return TemplateDeductionResult::NonDeducedMismatch;
      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, CP->getElementType(), CA->getElementType(), Info,
          Deduced, TDF, degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     _Atomic T   [extension]
    case Type::Atomic: {
      const auto *PA = P->castAs<AtomicType>(), *AA = A->getAs<AtomicType>();
      if (!AA)
        return TemplateDeductionResult::NonDeducedMismatch;
      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, PA->getValueType(), AA->getValueType(), Info,
          Deduced, TDF, degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     T *
    case Type::Pointer: {
      QualType PointeeType;
      if (const auto *PA = A->getAs<PointerType>()) {
        PointeeType = PA->getPointeeType();
      } else if (const auto *PA = A->getAs<ObjCObjectPointerType>()) {
        PointeeType = PA->getPointeeType();
      } else {
        return TemplateDeductionResult::NonDeducedMismatch;
      }
      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, P->castAs<PointerType>()->getPointeeType(),
          PointeeType, Info, Deduced,
          TDF & (TDF_IgnoreQualifiers | TDF_DerivedClass),
          degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     T &
    case Type::LValueReference: {
      const auto *RP = P->castAs<LValueReferenceType>(),
                 *RA = A->getAs<LValueReferenceType>();
      if (!RA)
        return TemplateDeductionResult::NonDeducedMismatch;

      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, RP->getPointeeType(), RA->getPointeeType(), Info,
          Deduced, 0, degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     T && [C++0x]
    case Type::RValueReference: {
      const auto *RP = P->castAs<RValueReferenceType>(),
                 *RA = A->getAs<RValueReferenceType>();
      if (!RA)
        return TemplateDeductionResult::NonDeducedMismatch;

      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, RP->getPointeeType(), RA->getPointeeType(), Info,
          Deduced, 0, degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     T [] (implied, but not stated explicitly)
    case Type::IncompleteArray: {
      const auto *IAA = S.Context.getAsIncompleteArrayType(A);
      if (!IAA)
        return TemplateDeductionResult::NonDeducedMismatch;

      const auto *IAP = S.Context.getAsIncompleteArrayType(P);
      assert(IAP && "Template parameter not of incomplete array type");

      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, IAP->getElementType(), IAA->getElementType(), Info,
          Deduced, TDF & TDF_IgnoreQualifiers,
          degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     T [integer-constant]
    case Type::ConstantArray: {
      const auto *CAA = S.Context.getAsConstantArrayType(A),
                 *CAP = S.Context.getAsConstantArrayType(P);
      assert(CAP);
      if (!CAA || CAA->getSize() != CAP->getSize())
        return TemplateDeductionResult::NonDeducedMismatch;

      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, CAP->getElementType(), CAA->getElementType(), Info,
          Deduced, TDF & TDF_IgnoreQualifiers,
          degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     type [i]
    case Type::DependentSizedArray: {
      const auto *AA = S.Context.getAsArrayType(A);
      if (!AA)
        return TemplateDeductionResult::NonDeducedMismatch;

      // Check the element type of the arrays
      const auto *DAP = S.Context.getAsDependentSizedArrayType(P);
      assert(DAP);
      if (auto Result = DeduceTemplateArgumentsByTypeMatch(
              S, TemplateParams, DAP->getElementType(), AA->getElementType(),
              Info, Deduced, TDF & TDF_IgnoreQualifiers,
              degradeCallPartialOrderingKind(POK),
              /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
          Result != TemplateDeductionResult::Success)
        return Result;

      // Determine the array bound is something we can deduce.
      NonTypeOrVarTemplateParmDecl NTTP =
          getDeducedNTTParameterFromExpr(Info, DAP->getSizeExpr());
      if (!NTTP)
        return TemplateDeductionResult::Success;

      // We can perform template argument deduction for the given non-type
      // template parameter.
      assert(NTTP.getDepth() == Info.getDeducedDepth() &&
             "saw non-type template parameter with wrong depth");
      if (const auto *CAA = dyn_cast<ConstantArrayType>(AA)) {
        llvm::APSInt Size(CAA->getSize());
        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, Size, S.Context.getSizeType(),
            /*ArrayBound=*/true, Info, POK != PartialOrderingKind::None,
            Deduced, HasDeducedAnyParam);
      }
      if (const auto *DAA = dyn_cast<DependentSizedArrayType>(AA))
        if (DAA->getSizeExpr())
          return DeduceNonTypeTemplateArgument(
              S, TemplateParams, NTTP, DAA->getSizeExpr(), Info,
              POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);

      // Incomplete type does not match a dependently-sized array type
      return TemplateDeductionResult::NonDeducedMismatch;
    }

    //     type(*)(T)
    //     T(*)()
    //     T(*)(T)
    case Type::FunctionProto: {
      const auto *FPP = P->castAs<FunctionProtoType>(),
                 *FPA = A->getAs<FunctionProtoType>();
      if (!FPA)
        return TemplateDeductionResult::NonDeducedMismatch;

      if (FPP->getMethodQuals() != FPA->getMethodQuals() ||
          FPP->getRefQualifier() != FPA->getRefQualifier() ||
          FPP->isVariadic() != FPA->isVariadic())
        return TemplateDeductionResult::NonDeducedMismatch;

      // Check return types.
      if (auto Result = DeduceTemplateArgumentsByTypeMatch(
              S, TemplateParams, FPP->getReturnType(), FPA->getReturnType(),
              Info, Deduced, 0, degradeCallPartialOrderingKind(POK),
              /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
          Result != TemplateDeductionResult::Success)
        return Result;

      // Check parameter types.
      if (auto Result = DeduceTemplateArguments(
              S, TemplateParams, FPP->param_types(), FPA->param_types(), Info,
              Deduced, TDF & TDF_TopLevelParameterTypeList, POK,
              HasDeducedAnyParam,
              /*HasDeducedParam=*/nullptr);
          Result != TemplateDeductionResult::Success)
        return Result;

      if (TDF & TDF_AllowCompatibleFunctionType)
        return TemplateDeductionResult::Success;

      // FIXME: Per core-2016/10/1019 (no corresponding core issue yet), permit
      // deducing through the noexcept-specifier if it's part of the canonical
      // type. libstdc++ relies on this.
      Expr *NoexceptExpr = FPP->getNoexceptExpr();
      if (NonTypeOrVarTemplateParmDecl NTTP =
              NoexceptExpr ? getDeducedNTTParameterFromExpr(Info, NoexceptExpr)
                           : nullptr) {
        assert(NTTP.getDepth() == Info.getDeducedDepth() &&
               "saw non-type template parameter with wrong depth");

        llvm::APSInt Noexcept(1);
        switch (FPA->canThrow()) {
        case CT_Cannot:
          Noexcept = 1;
          [[fallthrough]];

        case CT_Can:
          // We give E in noexcept(E) the "deduced from array bound" treatment.
          // FIXME: Should we?
          return DeduceNonTypeTemplateArgument(
              S, TemplateParams, NTTP, Noexcept, S.Context.BoolTy,
              /*DeducedFromArrayBound=*/true, Info,
              POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);

        case CT_Dependent:
          if (Expr *ArgNoexceptExpr = FPA->getNoexceptExpr())
            return DeduceNonTypeTemplateArgument(
                S, TemplateParams, NTTP, ArgNoexceptExpr, Info,
                POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);
          // Can't deduce anything from throw(T...).
          break;
        }
      }
      // FIXME: Detect non-deduced exception specification mismatches?
      //
      // Careful about [temp.deduct.call] and [temp.deduct.conv], which allow
      // top-level differences in noexcept-specifications.

      return TemplateDeductionResult::Success;
    }

    case Type::InjectedClassName:
      // Treat a template's injected-class-name as if the template
      // specialization type had been used.

    //     template-name<T> (where template-name refers to a class template)
    //     template-name<i>
    //     TT<T>
    //     TT<i>
    //     TT<>
    case Type::TemplateSpecialization: {
      // When Arg cannot be a derived class, we can just try to deduce template
      // arguments from the template-id.
      if (!(TDF & TDF_DerivedClass) || !A->isRecordType())
        return DeduceTemplateSpecArguments(S, TemplateParams, P, A, Info,
                                           POK != PartialOrderingKind::None,
                                           Deduced, HasDeducedAnyParam);

      SmallVector<DeducedTemplateArgument, 8> DeducedOrig(Deduced.begin(),
                                                          Deduced.end());

      auto Result = DeduceTemplateSpecArguments(
          S, TemplateParams, P, A, Info, POK != PartialOrderingKind::None,
          Deduced, HasDeducedAnyParam);
      if (Result == TemplateDeductionResult::Success)
        return Result;

      // We cannot inspect base classes as part of deduction when the type
      // is incomplete, so either instantiate any templates necessary to
      // complete the type, or skip over it if it cannot be completed.
      if (!S.isCompleteType(Info.getLocation(), A))
        return Result;

      const CXXRecordDecl *RD = A->getAsCXXRecordDecl();
      if (RD->isInvalidDecl())
        return Result;

      // Reset the incorrectly deduced argument from above.
      Deduced = DeducedOrig;

      // Check bases according to C++14 [temp.deduct.call] p4b3:
      auto BaseResult = DeduceTemplateBases(S, RD, TemplateParams, P, Info,
                                            POK != PartialOrderingKind::None,
                                            Deduced, HasDeducedAnyParam);
      return BaseResult != TemplateDeductionResult::Invalid ? BaseResult
                                                            : Result;
    }

    //     T type::*
    //     T T::*
    //     T (type::*)()
    //     type (T::*)()
    //     type (type::*)(T)
    //     type (T::*)(T)
    //     T (type::*)(T)
    //     T (T::*)()
    //     T (T::*)(T)
    case Type::MemberPointer: {
      const auto *MPP = P->castAs<MemberPointerType>(),
                 *MPA = A->getAs<MemberPointerType>();
      if (!MPA)
        return TemplateDeductionResult::NonDeducedMismatch;

      QualType PPT = MPP->getPointeeType();
      if (PPT->isFunctionType())
        S.adjustMemberFunctionCC(PPT, /*HasThisPointer=*/false,
                                 /*IsCtorOrDtor=*/false, Info.getLocation());
      QualType APT = MPA->getPointeeType();
      if (APT->isFunctionType())
        S.adjustMemberFunctionCC(APT, /*HasThisPointer=*/false,
                                 /*IsCtorOrDtor=*/false, Info.getLocation());

      unsigned SubTDF = TDF & TDF_IgnoreQualifiers;
      if (auto Result = DeduceTemplateArgumentsByTypeMatch(
              S, TemplateParams, PPT, APT, Info, Deduced, SubTDF,
              degradeCallPartialOrderingKind(POK),
              /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
          Result != TemplateDeductionResult::Success)
        return Result;

      QualType TP =
          MPP->isSugared()
              ? S.Context.getCanonicalTagType(MPP->getMostRecentCXXRecordDecl())
              : QualType(MPP->getQualifier().getAsType(), 0);
      assert(!TP.isNull() && "member pointer with non-type class");

      QualType TA =
          MPA->isSugared()
              ? S.Context.getCanonicalTagType(MPA->getMostRecentCXXRecordDecl())
              : QualType(MPA->getQualifier().getAsType(), 0)
                    .getUnqualifiedType();
      assert(!TA.isNull() && "member pointer with non-type class");

      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, TP, TA, Info, Deduced, SubTDF,
          degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     (clang extension)
    //
    //     type(^)(T)
    //     T(^)()
    //     T(^)(T)
    case Type::BlockPointer: {
      const auto *BPP = P->castAs<BlockPointerType>(),
                 *BPA = A->getAs<BlockPointerType>();
      if (!BPA)
        return TemplateDeductionResult::NonDeducedMismatch;
      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, BPP->getPointeeType(), BPA->getPointeeType(), Info,
          Deduced, 0, degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    //     (clang extension)
    //
    //     T __attribute__(((ext_vector_type(<integral constant>))))
    case Type::ExtVector: {
      const auto *VP = P->castAs<ExtVectorType>();
      QualType ElementType;
      if (const auto *VA = A->getAs<ExtVectorType>()) {
        // Make sure that the vectors have the same number of elements.
        if (VP->getNumElements() != VA->getNumElements())
          return TemplateDeductionResult::NonDeducedMismatch;
        ElementType = VA->getElementType();
      } else if (const auto *VA = A->getAs<DependentSizedExtVectorType>()) {
        // We can't check the number of elements, since the argument has a
        // dependent number of elements. This can only occur during partial
        // ordering.
        ElementType = VA->getElementType();
      } else {
        return TemplateDeductionResult::NonDeducedMismatch;
      }
      // Perform deduction on the element types.
      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, VP->getElementType(), ElementType, Info, Deduced,
          TDF, degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    case Type::DependentVector: {
      const auto *VP = P->castAs<DependentVectorType>();

      if (const auto *VA = A->getAs<VectorType>()) {
        // Perform deduction on the element types.
        if (auto Result = DeduceTemplateArgumentsByTypeMatch(
                S, TemplateParams, VP->getElementType(), VA->getElementType(),
                Info, Deduced, TDF, degradeCallPartialOrderingKind(POK),
                /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
            Result != TemplateDeductionResult::Success)
          return Result;

        // Perform deduction on the vector size, if we can.
        NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, VP->getSizeExpr());
        if (!NTTP)
          return TemplateDeductionResult::Success;

        llvm::APSInt ArgSize(S.Context.getTypeSize(S.Context.IntTy), false);
        ArgSize = VA->getNumElements();
        // Note that we use the "array bound" rules here; just like in that
        // case, we don't have any particular type for the vector size, but
        // we can provide one if necessary.
        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, ArgSize, S.Context.UnsignedIntTy, true,
            Info, POK != PartialOrderingKind::None, Deduced,
            HasDeducedAnyParam);
      }

      if (const auto *VA = A->getAs<DependentVectorType>()) {
        // Perform deduction on the element types.
        if (auto Result = DeduceTemplateArgumentsByTypeMatch(
                S, TemplateParams, VP->getElementType(), VA->getElementType(),
                Info, Deduced, TDF, degradeCallPartialOrderingKind(POK),
                /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
            Result != TemplateDeductionResult::Success)
          return Result;

        // Perform deduction on the vector size, if we can.
        NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, VP->getSizeExpr());
        if (!NTTP)
          return TemplateDeductionResult::Success;

        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, VA->getSizeExpr(), Info,
            POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);
      }

      return TemplateDeductionResult::NonDeducedMismatch;
    }

    //     (clang extension)
    //
    //     T __attribute__(((ext_vector_type(N))))
    case Type::DependentSizedExtVector: {
      const auto *VP = P->castAs<DependentSizedExtVectorType>();

      if (const auto *VA = A->getAs<ExtVectorType>()) {
        // Perform deduction on the element types.
        if (auto Result = DeduceTemplateArgumentsByTypeMatch(
                S, TemplateParams, VP->getElementType(), VA->getElementType(),
                Info, Deduced, TDF, degradeCallPartialOrderingKind(POK),
                /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
            Result != TemplateDeductionResult::Success)
          return Result;

        // Perform deduction on the vector size, if we can.
        NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, VP->getSizeExpr());
        if (!NTTP)
          return TemplateDeductionResult::Success;

        llvm::APSInt ArgSize(S.Context.getTypeSize(S.Context.IntTy), false);
        ArgSize = VA->getNumElements();
        // Note that we use the "array bound" rules here; just like in that
        // case, we don't have any particular type for the vector size, but
        // we can provide one if necessary.
        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, ArgSize, S.Context.IntTy, true, Info,
            POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);
      }

      if (const auto *VA = A->getAs<DependentSizedExtVectorType>()) {
        // Perform deduction on the element types.
        if (auto Result = DeduceTemplateArgumentsByTypeMatch(
                S, TemplateParams, VP->getElementType(), VA->getElementType(),
                Info, Deduced, TDF, degradeCallPartialOrderingKind(POK),
                /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
            Result != TemplateDeductionResult::Success)
          return Result;

        // Perform deduction on the vector size, if we can.
        NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, VP->getSizeExpr());
        if (!NTTP)
          return TemplateDeductionResult::Success;

        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, VA->getSizeExpr(), Info,
            POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);
      }

      return TemplateDeductionResult::NonDeducedMismatch;
    }

    //     (clang extension)
    //
    //     T __attribute__((matrix_type(<integral constant>,
    //                                  <integral constant>)))
    case Type::ConstantMatrix: {
      const auto *MP = P->castAs<ConstantMatrixType>(),
                 *MA = A->getAs<ConstantMatrixType>();
      if (!MA)
        return TemplateDeductionResult::NonDeducedMismatch;

      // Check that the dimensions are the same
      if (MP->getNumRows() != MA->getNumRows() ||
          MP->getNumColumns() != MA->getNumColumns()) {
        return TemplateDeductionResult::NonDeducedMismatch;
      }
      // Perform deduction on element types.
      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, MP->getElementType(), MA->getElementType(), Info,
          Deduced, TDF, degradeCallPartialOrderingKind(POK),
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    }

    case Type::DependentSizedMatrix: {
      const auto *MP = P->castAs<DependentSizedMatrixType>();
      const auto *MA = A->getAs<MatrixType>();
      if (!MA)
        return TemplateDeductionResult::NonDeducedMismatch;

      // Check the element type of the matrixes.
      if (auto Result = DeduceTemplateArgumentsByTypeMatch(
              S, TemplateParams, MP->getElementType(), MA->getElementType(),
              Info, Deduced, TDF, degradeCallPartialOrderingKind(POK),
              /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
          Result != TemplateDeductionResult::Success)
        return Result;

      // Try to deduce a matrix dimension.
      auto DeduceMatrixArg =
          [&S, &Info, &Deduced, &TemplateParams, &HasDeducedAnyParam, POK](
              Expr *ParamExpr, const MatrixType *A,
              unsigned (ConstantMatrixType::*GetArgDimension)() const,
              Expr *(DependentSizedMatrixType::*GetArgDimensionExpr)() const) {
            const auto *ACM = dyn_cast<ConstantMatrixType>(A);
            const auto *ADM = dyn_cast<DependentSizedMatrixType>(A);
            if (!ParamExpr->isValueDependent()) {
              std::optional<llvm::APSInt> ParamConst =
                  ParamExpr->getIntegerConstantExpr(S.Context);
              if (!ParamConst)
                return TemplateDeductionResult::NonDeducedMismatch;

              if (ACM) {
                if ((ACM->*GetArgDimension)() == *ParamConst)
                  return TemplateDeductionResult::Success;
                return TemplateDeductionResult::NonDeducedMismatch;
              }

              Expr *ArgExpr = (ADM->*GetArgDimensionExpr)();
              if (std::optional<llvm::APSInt> ArgConst =
                      ArgExpr->getIntegerConstantExpr(S.Context))
                if (*ArgConst == *ParamConst)
                  return TemplateDeductionResult::Success;
              return TemplateDeductionResult::NonDeducedMismatch;
            }

            NonTypeOrVarTemplateParmDecl NTTP =
                getDeducedNTTParameterFromExpr(Info, ParamExpr);
            if (!NTTP)
              return TemplateDeductionResult::Success;

            if (ACM) {
              llvm::APSInt ArgConst(
                  S.Context.getTypeSize(S.Context.getSizeType()));
              ArgConst = (ACM->*GetArgDimension)();
              return DeduceNonTypeTemplateArgument(
                  S, TemplateParams, NTTP, ArgConst, S.Context.getSizeType(),
                  /*ArrayBound=*/true, Info, POK != PartialOrderingKind::None,
                  Deduced, HasDeducedAnyParam);
            }

            return DeduceNonTypeTemplateArgument(
                S, TemplateParams, NTTP, (ADM->*GetArgDimensionExpr)(), Info,
                POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);
          };

      if (auto Result = DeduceMatrixArg(MP->getRowExpr(), MA,
                                        &ConstantMatrixType::getNumRows,
                                        &DependentSizedMatrixType::getRowExpr);
          Result != TemplateDeductionResult::Success)
        return Result;

      return DeduceMatrixArg(MP->getColumnExpr(), MA,
                             &ConstantMatrixType::getNumColumns,
                             &DependentSizedMatrixType::getColumnExpr);
    }

    //     (clang extension)
    //
    //     T __attribute__(((address_space(N))))
    case Type::DependentAddressSpace: {
      const auto *ASP = P->castAs<DependentAddressSpaceType>();

      if (const auto *ASA = A->getAs<DependentAddressSpaceType>()) {
        // Perform deduction on the pointer type.
        if (auto Result = DeduceTemplateArgumentsByTypeMatch(
                S, TemplateParams, ASP->getPointeeType(), ASA->getPointeeType(),
                Info, Deduced, TDF, degradeCallPartialOrderingKind(POK),
                /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
            Result != TemplateDeductionResult::Success)
          return Result;

        // Perform deduction on the address space, if we can.
        NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, ASP->getAddrSpaceExpr());
        if (!NTTP)
          return TemplateDeductionResult::Success;

        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, ASA->getAddrSpaceExpr(), Info,
            POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);
      }

      if (isTargetAddressSpace(A.getAddressSpace())) {
        llvm::APSInt ArgAddressSpace(S.Context.getTypeSize(S.Context.IntTy),
                                     false);
        ArgAddressSpace = toTargetAddressSpace(A.getAddressSpace());

        // Perform deduction on the pointer types.
        if (auto Result = DeduceTemplateArgumentsByTypeMatch(
                S, TemplateParams, ASP->getPointeeType(),
                S.Context.removeAddrSpaceQualType(A), Info, Deduced, TDF,
                degradeCallPartialOrderingKind(POK),
                /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
            Result != TemplateDeductionResult::Success)
          return Result;

        // Perform deduction on the address space, if we can.
        NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, ASP->getAddrSpaceExpr());
        if (!NTTP)
          return TemplateDeductionResult::Success;

        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, ArgAddressSpace, S.Context.IntTy, true,
            Info, POK != PartialOrderingKind::None, Deduced,
            HasDeducedAnyParam);
      }

      return TemplateDeductionResult::NonDeducedMismatch;
    }
    case Type::DependentBitInt: {
      const auto *IP = P->castAs<DependentBitIntType>();

      if (const auto *IA = A->getAs<BitIntType>()) {
        if (IP->isUnsigned() != IA->isUnsigned())
          return TemplateDeductionResult::NonDeducedMismatch;

        NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, IP->getNumBitsExpr());
        if (!NTTP)
          return TemplateDeductionResult::Success;

        llvm::APSInt ArgSize(S.Context.getTypeSize(S.Context.IntTy), false);
        ArgSize = IA->getNumBits();

        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, ArgSize, S.Context.IntTy, true, Info,
            POK != PartialOrderingKind::None, Deduced, HasDeducedAnyParam);
      }

      if (const auto *IA = A->getAs<DependentBitIntType>()) {
        if (IP->isUnsigned() != IA->isUnsigned())
          return TemplateDeductionResult::NonDeducedMismatch;
        return TemplateDeductionResult::Success;
      }

      return TemplateDeductionResult::NonDeducedMismatch;
    }

    case Type::TypeOfExpr:
    case Type::TypeOf:
    case Type::DependentName:
    case Type::UnresolvedUsing:
    case Type::Decltype:
    case Type::UnaryTransform:
    case Type::DeducedTemplateSpecialization:
    case Type::DependentTemplateSpecialization:
    case Type::PackExpansion:
    case Type::Pipe:
    case Type::ArrayParameter:
    case Type::HLSLAttributedResource:
    case Type::HLSLInlineSpirv:
      // No template argument deduction for these types
      return TemplateDeductionResult::Success;

    case Type::PackIndexing: {
      const PackIndexingType *PIT = P->getAs<PackIndexingType>();
      if (PIT->hasSelectedType()) {
        return DeduceTemplateArgumentsByTypeMatch(
            S, TemplateParams, PIT->getSelectedType(), A, Info, Deduced, TDF,
            degradeCallPartialOrderingKind(POK),
            /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
      }
      return TemplateDeductionResult::IncompletePack;
    }
    }

  llvm_unreachable("Invalid Type Class!");
}

static TemplateDeductionResult
DeduceTemplateArguments(Sema &S, TemplateParameterList *TemplateParams,
                        const TemplateArgument &P, TemplateArgument A,
                        TemplateDeductionInfo &Info, bool PartialOrdering,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                        bool *HasDeducedAnyParam) {
  // If the template argument is a pack expansion, perform template argument
  // deduction against the pattern of that expansion. This only occurs during
  // partial ordering.
  if (A.isPackExpansion())
    A = A.getPackExpansionPattern();

  switch (P.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Null template argument in parameter list");

  case TemplateArgument::Type:
    if (A.getKind() == TemplateArgument::Type)
      return DeduceTemplateArgumentsByTypeMatch(
          S, TemplateParams, P.getAsType(), A.getAsType(), Info, Deduced, 0,
          PartialOrdering ? PartialOrderingKind::NonCall
                          : PartialOrderingKind::None,
          /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
    Info.FirstArg = P;
    Info.SecondArg = A;
    return TemplateDeductionResult::NonDeducedMismatch;

  case TemplateArgument::Template:
    // PartialOrdering does not matter here, since template specializations are
    // not being deduced.
    if (A.getKind() == TemplateArgument::Template)
      return DeduceTemplateArguments(
          S, TemplateParams, P.getAsTemplate(), A.getAsTemplate(), Info,
          /*DefaultArguments=*/{}, /*PartialOrdering=*/false, Deduced,
          HasDeducedAnyParam);
    Info.FirstArg = P;
    Info.SecondArg = A;
    return TemplateDeductionResult::NonDeducedMismatch;

  case TemplateArgument::TemplateExpansion:
    llvm_unreachable("caller should handle pack expansions");

  case TemplateArgument::Declaration:
    if (A.getKind() == TemplateArgument::Declaration &&
        isSameDeclaration(P.getAsDecl(), A.getAsDecl()))
      return TemplateDeductionResult::Success;

    Info.FirstArg = P;
    Info.SecondArg = A;
    return TemplateDeductionResult::NonDeducedMismatch;

  case TemplateArgument::NullPtr:
    // 'nullptr' has only one possible value, so it always matches.
    if (A.getKind() == TemplateArgument::NullPtr)
      return TemplateDeductionResult::Success;
    Info.FirstArg = P;
    Info.SecondArg = A;
    return TemplateDeductionResult::NonDeducedMismatch;

  case TemplateArgument::Integral:
    if (A.getKind() == TemplateArgument::Integral) {
      if (llvm::APSInt::isSameValue(P.getAsIntegral(), A.getAsIntegral()))
        return TemplateDeductionResult::Success;
    }
    Info.FirstArg = P;
    Info.SecondArg = A;
    return TemplateDeductionResult::NonDeducedMismatch;

  case TemplateArgument::StructuralValue:
    // FIXME: structural equality will also compare types,
    // but they should match iff they have the same value.
    if (A.getKind() == TemplateArgument::StructuralValue &&
        A.structurallyEquals(P))
      return TemplateDeductionResult::Success;

    Info.FirstArg = P;
    Info.SecondArg = A;
    return TemplateDeductionResult::NonDeducedMismatch;

  case TemplateArgument::Expression:
    if (NonTypeOrVarTemplateParmDecl NTTP =
            getDeducedNTTParameterFromExpr(Info, P.getAsExpr())) {
      switch (A.getKind()) {
      case TemplateArgument::Expression: {
        const Expr *E = A.getAsExpr();
        // When checking NTTP, if either the parameter or the argument is
        // dependent, as there would be otherwise nothing to deduce, we force
        // the argument to the parameter type using this dependent implicit
        // cast, in order to maintain invariants. Now we can deduce the
        // resulting type from the original type, and deduce the original type
        // against the parameter we are checking.
        if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E);
            ICE && ICE->getCastKind() == clang::CK_Dependent) {
          E = ICE->getSubExpr();
          if (auto Result = DeduceTemplateArgumentsByTypeMatch(
                  S, TemplateParams, ICE->getType(), E->getType(), Info,
                  Deduced, TDF_SkipNonDependent,
                  PartialOrdering ? PartialOrderingKind::NonCall
                                  : PartialOrderingKind::None,
                  /*DeducedFromArrayBound=*/false, HasDeducedAnyParam);
              Result != TemplateDeductionResult::Success)
            return Result;
        }
        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, DeducedTemplateArgument(A), E->getType(),
            Info, PartialOrdering, Deduced, HasDeducedAnyParam);
      }
      case TemplateArgument::Integral:
      case TemplateArgument::StructuralValue:
        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, DeducedTemplateArgument(A),
            A.getNonTypeTemplateArgumentType(), Info, PartialOrdering, Deduced,
            HasDeducedAnyParam);

      case TemplateArgument::NullPtr:
        return DeduceNullPtrTemplateArgument(
            S, TemplateParams, NTTP, A.getNullPtrType(), Info, PartialOrdering,
            Deduced, HasDeducedAnyParam);

      case TemplateArgument::Declaration:
        return DeduceNonTypeTemplateArgument(
            S, TemplateParams, NTTP, A.getAsDecl(), A.getParamTypeForDecl(),
            Info, PartialOrdering, Deduced, HasDeducedAnyParam);

      case TemplateArgument::Null:
      case TemplateArgument::Type:
      case TemplateArgument::Template:
      case TemplateArgument::TemplateExpansion:
      case TemplateArgument::Pack:
        Info.FirstArg = P;
        Info.SecondArg = A;
        return TemplateDeductionResult::NonDeducedMismatch;
      }
      llvm_unreachable("Unknown template argument kind");
    }
    // Can't deduce anything, but that's okay.
    return TemplateDeductionResult::Success;
  case TemplateArgument::Pack:
    llvm_unreachable("Argument packs should be expanded by the caller!");
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

/// Determine whether there is a template argument to be used for
/// deduction.
///
/// This routine "expands" argument packs in-place, overriding its input
/// parameters so that \c Args[ArgIdx] will be the available template argument.
///
/// \returns true if there is another template argument (which will be at
/// \c Args[ArgIdx]), false otherwise.
static bool hasTemplateArgumentForDeduction(ArrayRef<TemplateArgument> &Args,
                                            unsigned &ArgIdx) {
  if (ArgIdx == Args.size())
    return false;

  const TemplateArgument &Arg = Args[ArgIdx];
  if (Arg.getKind() != TemplateArgument::Pack)
    return true;

  assert(ArgIdx == Args.size() - 1 && "Pack not at the end of argument list?");
  Args = Arg.pack_elements();
  ArgIdx = 0;
  return ArgIdx < Args.size();
}

/// Determine whether the given set of template arguments has a pack
/// expansion that is not the last template argument.
static bool hasPackExpansionBeforeEnd(ArrayRef<TemplateArgument> Args) {
  bool FoundPackExpansion = false;
  for (const auto &A : Args) {
    if (FoundPackExpansion)
      return true;

    if (A.getKind() == TemplateArgument::Pack)
      return hasPackExpansionBeforeEnd(A.pack_elements());

    // FIXME: If this is a fixed-arity pack expansion from an outer level of
    // templates, it should not be treated as a pack expansion.
    if (A.isPackExpansion())
      FoundPackExpansion = true;
  }

  return false;
}

static TemplateDeductionResult
DeduceTemplateArguments(Sema &S, TemplateParameterList *TemplateParams,
                        ArrayRef<TemplateArgument> Ps,
                        ArrayRef<TemplateArgument> As,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                        bool NumberOfArgumentsMustMatch, bool PartialOrdering,
                        PackFold PackFold, bool *HasDeducedAnyParam) {
  bool FoldPackParameter = PackFold == PackFold::ParameterToArgument ||
                           PackFold == PackFold::Both,
       FoldPackArgument = PackFold == PackFold::ArgumentToParameter ||
                          PackFold == PackFold::Both;

  // C++0x [temp.deduct.type]p9:
  //   If the template argument list of P contains a pack expansion that is not
  //   the last template argument, the entire template argument list is a
  //   non-deduced context.
  if (FoldPackParameter && hasPackExpansionBeforeEnd(Ps))
    return TemplateDeductionResult::Success;

  // C++0x [temp.deduct.type]p9:
  //   If P has a form that contains <T> or <i>, then each argument Pi of the
  //   respective template argument list P is compared with the corresponding
  //   argument Ai of the corresponding template argument list of A.
  for (unsigned ArgIdx = 0, ParamIdx = 0; /**/; /**/) {
    if (!hasTemplateArgumentForDeduction(Ps, ParamIdx))
      return !FoldPackParameter && hasTemplateArgumentForDeduction(As, ArgIdx)
                 ? TemplateDeductionResult::MiscellaneousDeductionFailure
                 : TemplateDeductionResult::Success;

    if (!Ps[ParamIdx].isPackExpansion()) {
      // The simple case: deduce template arguments by matching Pi and Ai.

      // Check whether we have enough arguments.
      if (!hasTemplateArgumentForDeduction(As, ArgIdx))
        return !FoldPackArgument && NumberOfArgumentsMustMatch
                   ? TemplateDeductionResult::MiscellaneousDeductionFailure
                   : TemplateDeductionResult::Success;

      if (As[ArgIdx].isPackExpansion()) {
        // C++1z [temp.deduct.type]p9:
        //   During partial ordering, if Ai was originally a pack expansion
        //   [and] Pi is not a pack expansion, template argument deduction
        //   fails.
        if (!FoldPackArgument)
          return TemplateDeductionResult::MiscellaneousDeductionFailure;

        TemplateArgument Pattern = As[ArgIdx].getPackExpansionPattern();
        for (;;) {
          // Deduce template parameters from the pattern.
          if (auto Result = DeduceTemplateArguments(
                  S, TemplateParams, Ps[ParamIdx], Pattern, Info,
                  PartialOrdering, Deduced, HasDeducedAnyParam);
              Result != TemplateDeductionResult::Success)
            return Result;

          ++ParamIdx;
          if (!hasTemplateArgumentForDeduction(Ps, ParamIdx))
            return TemplateDeductionResult::Success;
          if (Ps[ParamIdx].isPackExpansion())
            break;
        }
      } else {
        // Perform deduction for this Pi/Ai pair.
        if (auto Result = DeduceTemplateArguments(
                S, TemplateParams, Ps[ParamIdx], As[ArgIdx], Info,
                PartialOrdering, Deduced, HasDeducedAnyParam);
            Result != TemplateDeductionResult::Success)
          return Result;

        ++ArgIdx;
        ++ParamIdx;
        continue;
      }
    }

    // The parameter is a pack expansion.

    // C++0x [temp.deduct.type]p9:
    //   If Pi is a pack expansion, then the pattern of Pi is compared with
    //   each remaining argument in the template argument list of A. Each
    //   comparison deduces template arguments for subsequent positions in the
    //   template parameter packs expanded by Pi.
    TemplateArgument Pattern = Ps[ParamIdx].getPackExpansionPattern();

    // Prepare to deduce the packs within the pattern.
    PackDeductionScope PackScope(S, TemplateParams, Deduced, Info, Pattern);

    // Keep track of the deduced template arguments for each parameter pack
    // expanded by this pack expansion (the outer index) and for each
    // template argument (the inner SmallVectors).
    for (; hasTemplateArgumentForDeduction(As, ArgIdx) &&
           PackScope.hasNextElement();
         ++ArgIdx) {
      if (!As[ArgIdx].isPackExpansion()) {
        if (!FoldPackParameter)
          return TemplateDeductionResult::MiscellaneousDeductionFailure;
        if (FoldPackArgument)
          Info.setStrictPackMatch();
      }
      // Deduce template arguments from the pattern.
      if (auto Result = DeduceTemplateArguments(
              S, TemplateParams, Pattern, As[ArgIdx], Info, PartialOrdering,
              Deduced, HasDeducedAnyParam);
          Result != TemplateDeductionResult::Success)
        return Result;

      PackScope.nextPackElement();
    }

    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    return PackScope.finish();
  }
}

TemplateDeductionResult Sema::DeduceTemplateArguments(
    TemplateParameterList *TemplateParams, ArrayRef<TemplateArgument> Ps,
    ArrayRef<TemplateArgument> As, sema::TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    bool NumberOfArgumentsMustMatch) {
  return ::DeduceTemplateArguments(
      *this, TemplateParams, Ps, As, Info, Deduced, NumberOfArgumentsMustMatch,
      /*PartialOrdering=*/false, PackFold::ParameterToArgument,
      /*HasDeducedAnyParam=*/nullptr);
}

TemplateArgumentLoc
Sema::getTrivialTemplateArgumentLoc(const TemplateArgument &Arg,
                                    QualType NTTPType, SourceLocation Loc,
                                    NamedDecl *TemplateParam) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Can't get a NULL template argument here");

  case TemplateArgument::Type:
    return TemplateArgumentLoc(
        Arg, Context.getTrivialTypeSourceInfo(Arg.getAsType(), Loc));

  case TemplateArgument::Declaration: {
    if (NTTPType.isNull())
      NTTPType = Arg.getParamTypeForDecl();
    Expr *E = BuildExpressionFromDeclTemplateArgument(Arg, NTTPType, Loc,
                                                      TemplateParam)
                  .getAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(E, /*IsCanonical=*/false), E);
  }

  case TemplateArgument::NullPtr: {
    if (NTTPType.isNull())
      NTTPType = Arg.getNullPtrType();
    Expr *E = BuildExpressionFromDeclTemplateArgument(Arg, NTTPType, Loc)
                  .getAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(NTTPType, /*isNullPtr*/true),
                               E);
  }

  case TemplateArgument::Integral:
  case TemplateArgument::StructuralValue: {
    Expr *E = BuildExpressionFromNonTypeTemplateArgument(Arg, Loc).get();
    return TemplateArgumentLoc(TemplateArgument(E, /*IsCanonical=*/false), E);
  }

    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion: {
      NestedNameSpecifierLocBuilder Builder;
      TemplateName Template = Arg.getAsTemplateOrTemplatePattern();
      Builder.MakeTrivial(Context, Template.getQualifier(), Loc);
      return TemplateArgumentLoc(
          Context, Arg, Loc, Builder.getWithLocInContext(Context), Loc,
          /*EllipsisLoc=*/Arg.getKind() == TemplateArgument::TemplateExpansion
              ? Loc
              : SourceLocation());
    }

  case TemplateArgument::Expression:
    return TemplateArgumentLoc(Arg, Arg.getAsExpr());

  case TemplateArgument::Pack:
    return TemplateArgumentLoc(Arg, TemplateArgumentLocInfo());
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

TemplateArgumentLoc
Sema::getIdentityTemplateArgumentLoc(NamedDecl *TemplateParm,
                                     SourceLocation Location) {
  return getTrivialTemplateArgumentLoc(
      Context.getInjectedTemplateArg(TemplateParm), QualType(), Location);
}

/// Convert the given deduced template argument and add it to the set of
/// fully-converted template arguments.
static bool
ConvertDeducedTemplateArgument(Sema &S, NamedDecl *Param,
                               DeducedTemplateArgument Arg, NamedDecl *Template,
                               TemplateDeductionInfo &Info, bool IsDeduced,
                               Sema::CheckTemplateArgumentInfo &CTAI) {
  auto ConvertArg = [&](DeducedTemplateArgument Arg,
                        unsigned ArgumentPackIndex) {
    // Convert the deduced template argument into a template
    // argument that we can check, almost as if the user had written
    // the template argument explicitly.
    TemplateArgumentLoc ArgLoc = S.getTrivialTemplateArgumentLoc(
        Arg, QualType(), Info.getLocation(), Param);

    SaveAndRestore _1(CTAI.MatchingTTP, false);
    SaveAndRestore _2(CTAI.StrictPackMatch, false);
    // Check the template argument, converting it as necessary.
    auto Res = S.CheckTemplateArgument(
        Param, ArgLoc, Template, Template->getLocation(),
        Template->getSourceRange().getEnd(), ArgumentPackIndex, CTAI,
        IsDeduced
            ? (Arg.wasDeducedFromArrayBound() ? Sema::CTAK_DeducedFromArrayBound
                                              : Sema::CTAK_Deduced)
            : Sema::CTAK_Specified);
    if (CTAI.StrictPackMatch)
      Info.setStrictPackMatch();
    return Res;
  };

  if (Arg.getKind() == TemplateArgument::Pack) {
    // This is a template argument pack, so check each of its arguments against
    // the template parameter.
    SmallVector<TemplateArgument, 2> SugaredPackedArgsBuilder,
        CanonicalPackedArgsBuilder;
    for (const auto &P : Arg.pack_elements()) {
      // When converting the deduced template argument, append it to the
      // general output list. We need to do this so that the template argument
      // checking logic has all of the prior template arguments available.
      DeducedTemplateArgument InnerArg(P);
      InnerArg.setDeducedFromArrayBound(Arg.wasDeducedFromArrayBound());
      assert(InnerArg.getKind() != TemplateArgument::Pack &&
             "deduced nested pack");
      if (P.isNull()) {
        // We deduced arguments for some elements of this pack, but not for
        // all of them. This happens if we get a conditionally-non-deduced
        // context in a pack expansion (such as an overload set in one of the
        // arguments).
        S.Diag(Param->getLocation(),
               diag::err_template_arg_deduced_incomplete_pack)
          << Arg << Param;
        return true;
      }
      if (ConvertArg(InnerArg, SugaredPackedArgsBuilder.size()))
        return true;

      // Move the converted template argument into our argument pack.
      SugaredPackedArgsBuilder.push_back(CTAI.SugaredConverted.pop_back_val());
      CanonicalPackedArgsBuilder.push_back(
          CTAI.CanonicalConverted.pop_back_val());
    }

    // If the pack is empty, we still need to substitute into the parameter
    // itself, in case that substitution fails.
    if (SugaredPackedArgsBuilder.empty()) {
      LocalInstantiationScope Scope(S);
      MultiLevelTemplateArgumentList Args(Template, CTAI.SugaredConverted,
                                          /*Final=*/true);
      Sema::ArgPackSubstIndexRAII OnlySubstNonPackExpansion(S, std::nullopt);

      if (auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        Sema::InstantiatingTemplate Inst(S, Template->getLocation(), Template,
                                         NTTP, CTAI.SugaredConverted,
                                         Template->getSourceRange());
        if (Inst.isInvalid() ||
            S.SubstType(NTTP->getType(), Args, NTTP->getLocation(),
                        NTTP->getDeclName()).isNull())
          return true;
      } else if (auto *TTP = dyn_cast<TemplateTemplateParmDecl>(Param)) {
        Sema::InstantiatingTemplate Inst(S, Template->getLocation(), Template,
                                         TTP, CTAI.SugaredConverted,
                                         Template->getSourceRange());
        if (Inst.isInvalid() || !S.SubstDecl(TTP, S.CurContext, Args))
          return true;
      }
      // For type parameters, no substitution is ever required.
    }

    // Create the resulting argument pack.
    CTAI.SugaredConverted.push_back(
        TemplateArgument::CreatePackCopy(S.Context, SugaredPackedArgsBuilder));
    CTAI.CanonicalConverted.push_back(TemplateArgument::CreatePackCopy(
        S.Context, CanonicalPackedArgsBuilder));
    return false;
  }

  return ConvertArg(Arg, 0);
}

/// \param IsIncomplete When used, we only consider template parameters that
/// were deduced, disregarding any default arguments. After the function
/// finishes, the object pointed at will contain a value indicating if the
/// conversion was actually incomplete.
static TemplateDeductionResult ConvertDeducedTemplateArguments(
    Sema &S, NamedDecl *Template, TemplateParameterList *TemplateParams,
    bool IsDeduced, SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    TemplateDeductionInfo &Info, Sema::CheckTemplateArgumentInfo &CTAI,
    LocalInstantiationScope *CurrentInstantiationScope,
    unsigned NumAlreadyConverted, bool *IsIncomplete) {
  for (unsigned I = 0, N = TemplateParams->size(); I != N; ++I) {
    NamedDecl *Param = TemplateParams->getParam(I);

    // C++0x [temp.arg.explicit]p3:
    //    A trailing template parameter pack (14.5.3) not otherwise deduced will
    //    be deduced to an empty sequence of template arguments.
    // FIXME: Where did the word "trailing" come from?
    if (Deduced[I].isNull() && Param->isTemplateParameterPack()) {
      if (auto Result =
              PackDeductionScope(S, TemplateParams, Deduced, Info, I).finish();
          Result != TemplateDeductionResult::Success)
        return Result;
    }

    if (!Deduced[I].isNull()) {
      if (I < NumAlreadyConverted) {
        // We may have had explicitly-specified template arguments for a
        // template parameter pack (that may or may not have been extended
        // via additional deduced arguments).
        if (Param->isParameterPack() && CurrentInstantiationScope &&
            CurrentInstantiationScope->getPartiallySubstitutedPack() == Param) {
          // Forget the partially-substituted pack; its substitution is now
          // complete.
          CurrentInstantiationScope->ResetPartiallySubstitutedPack();
          // We still need to check the argument in case it was extended by
          // deduction.
        } else {
          // We have already fully type-checked and converted this
          // argument, because it was explicitly-specified. Just record the
          // presence of this argument.
          CTAI.SugaredConverted.push_back(Deduced[I]);
          CTAI.CanonicalConverted.push_back(
              S.Context.getCanonicalTemplateArgument(Deduced[I]));
          continue;
        }
      }

      // We may have deduced this argument, so it still needs to be
      // checked and converted.
      if (ConvertDeducedTemplateArgument(S, Param, Deduced[I], Template, Info,
                                         IsDeduced, CTAI)) {
        Info.Param = makeTemplateParameter(Param);
        // FIXME: These template arguments are temporary. Free them!
        Info.reset(
            TemplateArgumentList::CreateCopy(S.Context, CTAI.SugaredConverted),
            TemplateArgumentList::CreateCopy(S.Context,
                                             CTAI.CanonicalConverted));
        return TemplateDeductionResult::SubstitutionFailure;
      }

      continue;
    }

    // [C++26][temp.deduct.partial]p12 - When partial ordering, it's ok for
    // template parameters to remain not deduced. As a provisional fix for a
    // core issue that does not exist yet, which may be related to CWG2160, only
    // consider template parameters that were deduced, disregarding any default
    // arguments.
    if (IsIncomplete) {
      *IsIncomplete = true;
      CTAI.SugaredConverted.push_back({});
      CTAI.CanonicalConverted.push_back({});
      continue;
    }

    // Substitute into the default template argument, if available.
    bool HasDefaultArg = false;
    TemplateDecl *TD = dyn_cast<TemplateDecl>(Template);
    if (!TD) {
      assert(isa<ClassTemplatePartialSpecializationDecl>(Template) ||
             isa<VarTemplatePartialSpecializationDecl>(Template));
      return TemplateDeductionResult::Incomplete;
    }

    TemplateArgumentLoc DefArg;
    {
      Qualifiers ThisTypeQuals;
      CXXRecordDecl *ThisContext = nullptr;
      if (auto *Rec = dyn_cast<CXXRecordDecl>(TD->getDeclContext()))
        if (Rec->isLambda())
          if (auto *Method = dyn_cast<CXXMethodDecl>(Rec->getDeclContext())) {
            ThisContext = Method->getParent();
            ThisTypeQuals = Method->getMethodQualifiers();
          }

      Sema::CXXThisScopeRAII ThisScope(S, ThisContext, ThisTypeQuals,
                                       S.getLangOpts().CPlusPlus17);

      DefArg = S.SubstDefaultTemplateArgumentIfAvailable(
          TD, /*TemplateKWLoc=*/SourceLocation(), TD->getLocation(),
          TD->getSourceRange().getEnd(), Param, CTAI.SugaredConverted,
          CTAI.CanonicalConverted, HasDefaultArg);
    }

    // If there was no default argument, deduction is incomplete.
    if (DefArg.getArgument().isNull()) {
      Info.Param = makeTemplateParameter(TemplateParams->getParam(I));
      Info.reset(
          TemplateArgumentList::CreateCopy(S.Context, CTAI.SugaredConverted),
          TemplateArgumentList::CreateCopy(S.Context, CTAI.CanonicalConverted));

      return HasDefaultArg ? TemplateDeductionResult::SubstitutionFailure
                           : TemplateDeductionResult::Incomplete;
    }

    SaveAndRestore _1(CTAI.PartialOrdering, false);
    SaveAndRestore _2(CTAI.MatchingTTP, false);
    SaveAndRestore _3(CTAI.StrictPackMatch, false);
    // Check whether we can actually use the default argument.
    if (S.CheckTemplateArgument(
            Param, DefArg, TD, TD->getLocation(), TD->getSourceRange().getEnd(),
            /*ArgumentPackIndex=*/0, CTAI, Sema::CTAK_Specified)) {
      Info.Param = makeTemplateParameter(TemplateParams->getParam(I));
      // FIXME: These template arguments are temporary. Free them!
      Info.reset(
          TemplateArgumentList::CreateCopy(S.Context, CTAI.SugaredConverted),
          TemplateArgumentList::CreateCopy(S.Context, CTAI.CanonicalConverted));
      return TemplateDeductionResult::SubstitutionFailure;
    }

    // If we get here, we successfully used the default template argument.
  }

  return TemplateDeductionResult::Success;
}

static DeclContext *getAsDeclContextOrEnclosing(Decl *D) {
  if (auto *DC = dyn_cast<DeclContext>(D))
    return DC;
  return D->getDeclContext();
}

template<typename T> struct IsPartialSpecialization {
  static constexpr bool value = false;
};
template<>
struct IsPartialSpecialization<ClassTemplatePartialSpecializationDecl> {
  static constexpr bool value = true;
};
template<>
struct IsPartialSpecialization<VarTemplatePartialSpecializationDecl> {
  static constexpr bool value = true;
};

static TemplateDeductionResult
CheckDeducedArgumentConstraints(Sema &S, NamedDecl *Template,
                                ArrayRef<TemplateArgument> SugaredDeducedArgs,
                                ArrayRef<TemplateArgument> CanonicalDeducedArgs,
                                TemplateDeductionInfo &Info) {
  llvm::SmallVector<AssociatedConstraint, 3> AssociatedConstraints;
  bool DeducedArgsNeedReplacement = false;
  if (auto *TD = dyn_cast<ClassTemplatePartialSpecializationDecl>(Template)) {
    TD->getAssociatedConstraints(AssociatedConstraints);
    DeducedArgsNeedReplacement = !TD->isClassScopeExplicitSpecialization();
  } else if (auto *TD =
                 dyn_cast<VarTemplatePartialSpecializationDecl>(Template)) {
    TD->getAssociatedConstraints(AssociatedConstraints);
    DeducedArgsNeedReplacement = !TD->isClassScopeExplicitSpecialization();
  } else {
    cast<TemplateDecl>(Template)->getAssociatedConstraints(
        AssociatedConstraints);
  }

  std::optional<ArrayRef<TemplateArgument>> Innermost;
  // If we don't need to replace the deduced template arguments,
  // we can add them immediately as the inner-most argument list.
  if (!DeducedArgsNeedReplacement)
    Innermost = CanonicalDeducedArgs;

  MultiLevelTemplateArgumentList MLTAL = S.getTemplateInstantiationArgs(
      Template, Template->getDeclContext(), /*Final=*/false, Innermost,
      /*RelativeToPrimary=*/true, /*Pattern=*/
      nullptr, /*ForConstraintInstantiation=*/true);

  // getTemplateInstantiationArgs picks up the non-deduced version of the
  // template args when this is a variable template partial specialization and
  // not class-scope explicit specialization, so replace with Deduced Args
  // instead of adding to inner-most.
  if (!Innermost)
    MLTAL.replaceInnermostTemplateArguments(Template, CanonicalDeducedArgs);

  if (S.CheckConstraintSatisfaction(Template, AssociatedConstraints, MLTAL,
                                    Info.getLocation(),
                                    Info.AssociatedConstraintsSatisfaction) ||
      !Info.AssociatedConstraintsSatisfaction.IsSatisfied) {
    Info.reset(
        TemplateArgumentList::CreateCopy(S.Context, SugaredDeducedArgs),
        TemplateArgumentList::CreateCopy(S.Context, CanonicalDeducedArgs));
    return TemplateDeductionResult::ConstraintsNotSatisfied;
  }
  return TemplateDeductionResult::Success;
}

/// Complete template argument deduction.
static TemplateDeductionResult FinishTemplateArgumentDeduction(
    Sema &S, NamedDecl *Entity, TemplateParameterList *EntityTPL,
    TemplateDecl *Template, bool PartialOrdering,
    ArrayRef<TemplateArgumentLoc> Ps, ArrayRef<TemplateArgument> As,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    TemplateDeductionInfo &Info, bool CopyDeducedArgs) {
  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(
      S, Sema::ExpressionEvaluationContext::Unevaluated);

  Sema::ContextRAII SavedContext(S, getAsDeclContextOrEnclosing(Entity));

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  Sema::CheckTemplateArgumentInfo CTAI(PartialOrdering);
  if (auto Result = ConvertDeducedTemplateArguments(
          S, Entity, EntityTPL, /*IsDeduced=*/PartialOrdering, Deduced, Info,
          CTAI,
          /*CurrentInstantiationScope=*/nullptr,
          /*NumAlreadyConverted=*/0U, /*IsIncomplete=*/nullptr);
      Result != TemplateDeductionResult::Success)
    return Result;

  if (CopyDeducedArgs) {
    // Form the template argument list from the deduced template arguments.
    TemplateArgumentList *SugaredDeducedArgumentList =
        TemplateArgumentList::CreateCopy(S.Context, CTAI.SugaredConverted);
    TemplateArgumentList *CanonicalDeducedArgumentList =
        TemplateArgumentList::CreateCopy(S.Context, CTAI.CanonicalConverted);
    Info.reset(SugaredDeducedArgumentList, CanonicalDeducedArgumentList);
  }

  TemplateParameterList *TPL = Template->getTemplateParameters();
  TemplateArgumentListInfo InstArgs(TPL->getLAngleLoc(), TPL->getRAngleLoc());
  MultiLevelTemplateArgumentList MLTAL(Entity, CTAI.SugaredConverted,
                                       /*Final=*/true);
  MLTAL.addOuterRetainedLevels(TPL->getDepth());

  if (S.SubstTemplateArguments(Ps, MLTAL, InstArgs)) {
    unsigned ArgIdx = InstArgs.size(), ParamIdx = ArgIdx;
    if (ParamIdx >= TPL->size())
      ParamIdx = TPL->size() - 1;

    Decl *Param = TPL->getParam(ParamIdx);
    Info.Param = makeTemplateParameter(Param);
    Info.FirstArg = Ps[ArgIdx].getArgument();
    return TemplateDeductionResult::SubstitutionFailure;
  }

  bool ConstraintsNotSatisfied;
  Sema::CheckTemplateArgumentInfo InstCTAI;
  if (S.CheckTemplateArgumentList(Template, Template->getLocation(), InstArgs,
                                  /*DefaultArgs=*/{}, false, InstCTAI,
                                  /*UpdateArgsWithConversions=*/true,
                                  &ConstraintsNotSatisfied))
    return ConstraintsNotSatisfied
               ? TemplateDeductionResult::ConstraintsNotSatisfied
               : TemplateDeductionResult::SubstitutionFailure;

  // Check that we produced the correct argument list.
  SmallVector<ArrayRef<TemplateArgument>, 4> PsStack{InstCTAI.SugaredConverted},
      AsStack{As};
  for (;;) {
    auto take = [](SmallVectorImpl<ArrayRef<TemplateArgument>> &Stack)
        -> std::tuple<ArrayRef<TemplateArgument> &, TemplateArgument> {
      while (!Stack.empty()) {
        auto &Xs = Stack.back();
        if (Xs.empty()) {
          Stack.pop_back();
          continue;
        }
        auto &X = Xs.front();
        if (X.getKind() == TemplateArgument::Pack) {
          Stack.emplace_back(X.getPackAsArray());
          Xs = Xs.drop_front();
          continue;
        }
        assert(!X.isNull());
        return {Xs, X};
      }
      static constexpr ArrayRef<TemplateArgument> None;
      return {const_cast<ArrayRef<TemplateArgument> &>(None),
              TemplateArgument()};
    };
    auto [Ps, P] = take(PsStack);
    auto [As, A] = take(AsStack);
    if (P.isNull() && A.isNull())
      break;
    TemplateArgument PP = P.isPackExpansion() ? P.getPackExpansionPattern() : P,
                     PA = A.isPackExpansion() ? A.getPackExpansionPattern() : A;
    if (!S.Context.isSameTemplateArgument(PP, PA)) {
      if (!P.isPackExpansion() && !A.isPackExpansion()) {
        Info.Param = makeTemplateParameter(TPL->getParam(
            (AsStack.empty() ? As.end() : AsStack.back().begin()) -
            As.begin()));
        Info.FirstArg = P;
        Info.SecondArg = A;
        return TemplateDeductionResult::NonDeducedMismatch;
      }
      if (P.isPackExpansion()) {
        Ps = Ps.drop_front();
        continue;
      }
      if (A.isPackExpansion()) {
        As = As.drop_front();
        continue;
      }
    }
    Ps = Ps.drop_front(P.isPackExpansion() ? 0 : 1);
    As = As.drop_front(A.isPackExpansion() && !P.isPackExpansion() ? 0 : 1);
  }
  assert(PsStack.empty());
  assert(AsStack.empty());

  if (!PartialOrdering) {
    if (auto Result = CheckDeducedArgumentConstraints(
            S, Entity, CTAI.SugaredConverted, CTAI.CanonicalConverted, Info);
        Result != TemplateDeductionResult::Success)
      return Result;
  }

  return TemplateDeductionResult::Success;
}
static TemplateDeductionResult FinishTemplateArgumentDeduction(
    Sema &S, NamedDecl *Entity, TemplateParameterList *EntityTPL,
    TemplateDecl *Template, bool PartialOrdering, ArrayRef<TemplateArgument> Ps,
    ArrayRef<TemplateArgument> As,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    TemplateDeductionInfo &Info, bool CopyDeducedArgs) {
  TemplateParameterList *TPL = Template->getTemplateParameters();
  SmallVector<TemplateArgumentLoc, 8> PsLoc(Ps.size());
  for (unsigned I = 0, N = Ps.size(); I != N; ++I)
    PsLoc[I] = S.getTrivialTemplateArgumentLoc(Ps[I], QualType(),
                                               TPL->getParam(I)->getLocation());
  return FinishTemplateArgumentDeduction(S, Entity, EntityTPL, Template,
                                         PartialOrdering, PsLoc, As, Deduced,
                                         Info, CopyDeducedArgs);
}

/// Complete template argument deduction for DeduceTemplateArgumentsFromType.
/// FIXME: this is mostly duplicated with the above two versions. Deduplicate
/// the three implementations.
static TemplateDeductionResult FinishTemplateArgumentDeduction(
    Sema &S, TemplateDecl *TD,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    TemplateDeductionInfo &Info) {
  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(
      S, Sema::ExpressionEvaluationContext::Unevaluated);

  Sema::ContextRAII SavedContext(S, getAsDeclContextOrEnclosing(TD));

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  Sema::CheckTemplateArgumentInfo CTAI;
  if (auto Result = ConvertDeducedTemplateArguments(
          S, TD, TD->getTemplateParameters(), /*IsDeduced=*/false, Deduced,
          Info, CTAI,
          /*CurrentInstantiationScope=*/nullptr, /*NumAlreadyConverted=*/0,
          /*IsIncomplete=*/nullptr);
      Result != TemplateDeductionResult::Success)
    return Result;

  return ::CheckDeducedArgumentConstraints(S, TD, CTAI.SugaredConverted,
                                           CTAI.CanonicalConverted, Info);
}

/// Perform template argument deduction to determine whether the given template
/// arguments match the given class or variable template partial specialization
/// per C++ [temp.class.spec.match].
template <typename T>
static std::enable_if_t<IsPartialSpecialization<T>::value,
                        TemplateDeductionResult>
DeduceTemplateArguments(Sema &S, T *Partial,
                        ArrayRef<TemplateArgument> TemplateArgs,
                        TemplateDeductionInfo &Info) {
  if (Partial->isInvalidDecl())
    return TemplateDeductionResult::Invalid;

  // C++ [temp.class.spec.match]p2:
  //   A partial specialization matches a given actual template
  //   argument list if the template arguments of the partial
  //   specialization can be deduced from the actual template argument
  //   list (14.8.2).

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(
      S, Sema::ExpressionEvaluationContext::Unevaluated);
  Sema::SFINAETrap Trap(S);

  // This deduction has no relation to any outer instantiation we might be
  // performing.
  LocalInstantiationScope InstantiationScope(S);

  SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(Partial->getTemplateParameters()->size());
  if (TemplateDeductionResult Result = ::DeduceTemplateArguments(
          S, Partial->getTemplateParameters(),
          Partial->getTemplateArgs().asArray(), TemplateArgs, Info, Deduced,
          /*NumberOfArgumentsMustMatch=*/false, /*PartialOrdering=*/false,
          PackFold::ParameterToArgument,
          /*HasDeducedAnyParam=*/nullptr);
      Result != TemplateDeductionResult::Success)
    return Result;

  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  Sema::InstantiatingTemplate Inst(S, Info.getLocation(), Partial, DeducedArgs,
                                   Info);
  if (Inst.isInvalid())
    return TemplateDeductionResult::InstantiationDepth;

  TemplateDeductionResult Result;
  S.runWithSufficientStackSpace(Info.getLocation(), [&] {
    Result = ::FinishTemplateArgumentDeduction(
        S, Partial, Partial->getTemplateParameters(),
        Partial->getSpecializedTemplate(),
        /*IsPartialOrdering=*/false,
        Partial->getTemplateArgsAsWritten()->arguments(), TemplateArgs, Deduced,
        Info, /*CopyDeducedArgs=*/true);
  });

  if (Result != TemplateDeductionResult::Success)
    return Result;

  if (Trap.hasErrorOccurred())
    return TemplateDeductionResult::SubstitutionFailure;

  return TemplateDeductionResult::Success;
}

TemplateDeductionResult
Sema::DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                              ArrayRef<TemplateArgument> TemplateArgs,
                              TemplateDeductionInfo &Info) {
  return ::DeduceTemplateArguments(*this, Partial, TemplateArgs, Info);
}
TemplateDeductionResult
Sema::DeduceTemplateArguments(VarTemplatePartialSpecializationDecl *Partial,
                              ArrayRef<TemplateArgument> TemplateArgs,
                              TemplateDeductionInfo &Info) {
  return ::DeduceTemplateArguments(*this, Partial, TemplateArgs, Info);
}

TemplateDeductionResult
Sema::DeduceTemplateArgumentsFromType(TemplateDecl *TD, QualType FromType,
                                      sema::TemplateDeductionInfo &Info) {
  if (TD->isInvalidDecl())
    return TemplateDeductionResult::Invalid;

  QualType PType;
  if (const auto *CTD = dyn_cast<ClassTemplateDecl>(TD)) {
    // Use the InjectedClassNameType.
    PType = Context.getCanonicalTagType(CTD->getTemplatedDecl());
  } else if (const auto *AliasTemplate = dyn_cast<TypeAliasTemplateDecl>(TD)) {
    PType = AliasTemplate->getTemplatedDecl()->getUnderlyingType();
  } else {
    assert(false && "Expected a class or alias template");
  }

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(
      *this, Sema::ExpressionEvaluationContext::Unevaluated);
  SFINAETrap Trap(*this);

  // This deduction has no relation to any outer instantiation we might be
  // performing.
  LocalInstantiationScope InstantiationScope(*this);

  SmallVector<DeducedTemplateArgument> Deduced(
      TD->getTemplateParameters()->size());
  SmallVector<TemplateArgument> PArgs = {TemplateArgument(PType)};
  SmallVector<TemplateArgument> AArgs = {TemplateArgument(FromType)};
  if (auto DeducedResult = DeduceTemplateArguments(
          TD->getTemplateParameters(), PArgs, AArgs, Info, Deduced, false);
      DeducedResult != TemplateDeductionResult::Success) {
    return DeducedResult;
  }

  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  InstantiatingTemplate Inst(*this, Info.getLocation(), TD, DeducedArgs, Info);
  if (Inst.isInvalid())
    return TemplateDeductionResult::InstantiationDepth;

  TemplateDeductionResult Result;
  runWithSufficientStackSpace(Info.getLocation(), [&] {
    Result = ::FinishTemplateArgumentDeduction(*this, TD, Deduced, Info);
  });

  if (Result != TemplateDeductionResult::Success)
    return Result;

  if (Trap.hasErrorOccurred())
    return TemplateDeductionResult::SubstitutionFailure;

  return TemplateDeductionResult::Success;
}

/// Determine whether the given type T is a simple-template-id type.
static bool isSimpleTemplateIdType(QualType T) {
  if (const TemplateSpecializationType *Spec
        = T->getAs<TemplateSpecializationType>())
    return Spec->getTemplateName().getAsTemplateDecl() != nullptr;

  // C++17 [temp.local]p2:
  //   the injected-class-name [...] is equivalent to the template-name followed
  //   by the template-arguments of the class template specialization or partial
  //   specialization enclosed in <>
  // ... which means it's equivalent to a simple-template-id.
  //
  // This only arises during class template argument deduction for a copy
  // deduction candidate, where it permits slicing.
  if (T->getAs<InjectedClassNameType>())
    return true;

  return false;
}

TemplateDeductionResult Sema::SubstituteExplicitTemplateArguments(
    FunctionTemplateDecl *FunctionTemplate,
    TemplateArgumentListInfo &ExplicitTemplateArgs,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    SmallVectorImpl<QualType> &ParamTypes, QualType *FunctionType,
    TemplateDeductionInfo &Info) {
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();

  if (ExplicitTemplateArgs.size() == 0) {
    // No arguments to substitute; just copy over the parameter types and
    // fill in the function type.
    for (auto *P : Function->parameters())
      ParamTypes.push_back(P->getType());

    if (FunctionType)
      *FunctionType = Function->getType();
    return TemplateDeductionResult::Success;
  }

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(
      *this, Sema::ExpressionEvaluationContext::Unevaluated);
  SFINAETrap Trap(*this);

  // C++ [temp.arg.explicit]p3:
  //   Template arguments that are present shall be specified in the
  //   declaration order of their corresponding template-parameters. The
  //   template argument list shall not specify more template-arguments than
  //   there are corresponding template-parameters.

  // Enter a new template instantiation context where we check the
  // explicitly-specified template arguments against this function template,
  // and then substitute them into the function parameter types.
  SmallVector<TemplateArgument, 4> DeducedArgs;
  InstantiatingTemplate Inst(
      *this, Info.getLocation(), FunctionTemplate, DeducedArgs,
      CodeSynthesisContext::ExplicitTemplateArgumentSubstitution, Info);
  if (Inst.isInvalid())
    return TemplateDeductionResult::InstantiationDepth;

  CheckTemplateArgumentInfo CTAI;
  if (CheckTemplateArgumentList(FunctionTemplate, SourceLocation(),
                                ExplicitTemplateArgs, /*DefaultArgs=*/{},
                                /*PartialTemplateArgs=*/true, CTAI,
                                /*UpdateArgsWithConversions=*/false) ||
      Trap.hasErrorOccurred()) {
    unsigned Index = CTAI.SugaredConverted.size();
    if (Index >= TemplateParams->size())
      return TemplateDeductionResult::SubstitutionFailure;
    Info.Param = makeTemplateParameter(TemplateParams->getParam(Index));
    return TemplateDeductionResult::InvalidExplicitArguments;
  }

  // Form the template argument list from the explicitly-specified
  // template arguments.
  TemplateArgumentList *SugaredExplicitArgumentList =
      TemplateArgumentList::CreateCopy(Context, CTAI.SugaredConverted);
  TemplateArgumentList *CanonicalExplicitArgumentList =
      TemplateArgumentList::CreateCopy(Context, CTAI.CanonicalConverted);
  Info.setExplicitArgs(SugaredExplicitArgumentList,
                       CanonicalExplicitArgumentList);

  // Template argument deduction and the final substitution should be
  // done in the context of the templated declaration.  Explicit
  // argument substitution, on the other hand, needs to happen in the
  // calling context.
  ContextRAII SavedContext(*this, FunctionTemplate->getTemplatedDecl());

  // If we deduced template arguments for a template parameter pack,
  // note that the template argument pack is partially substituted and record
  // the explicit template arguments. They'll be used as part of deduction
  // for this template parameter pack.
  unsigned PartiallySubstitutedPackIndex = -1u;
  if (!CTAI.SugaredConverted.empty()) {
    const TemplateArgument &Arg = CTAI.SugaredConverted.back();
    if (Arg.getKind() == TemplateArgument::Pack) {
      auto *Param = TemplateParams->getParam(CTAI.SugaredConverted.size() - 1);
      // If this is a fully-saturated fixed-size pack, it should be
      // fully-substituted, not partially-substituted.
      UnsignedOrNone Expansions = getExpandedPackSize(Param);
      if (!Expansions || Arg.pack_size() < *Expansions) {
        PartiallySubstitutedPackIndex = CTAI.SugaredConverted.size() - 1;
        CurrentInstantiationScope->SetPartiallySubstitutedPack(
            Param, Arg.pack_begin(), Arg.pack_size());
      }
    }
  }

  const FunctionProtoType *Proto
    = Function->getType()->getAs<FunctionProtoType>();
  assert(Proto && "Function template does not have a prototype?");

  // Isolate our substituted parameters from our caller.
  LocalInstantiationScope InstScope(*this, /*MergeWithOuterScope*/true);

  ExtParameterInfoBuilder ExtParamInfos;

  MultiLevelTemplateArgumentList MLTAL(FunctionTemplate,
                                       SugaredExplicitArgumentList->asArray(),
                                       /*Final=*/true);

  // Instantiate the types of each of the function parameters given the
  // explicitly-specified template arguments. If the function has a trailing
  // return type, substitute it after the arguments to ensure we substitute
  // in lexical order.
  if (Proto->hasTrailingReturn()) {
    if (SubstParmTypes(Function->getLocation(), Function->parameters(),
                       Proto->getExtParameterInfosOrNull(), MLTAL, ParamTypes,
                       /*params=*/nullptr, ExtParamInfos))
      return TemplateDeductionResult::SubstitutionFailure;
  }

  // Instantiate the return type.
  QualType ResultType;
  {
    // C++11 [expr.prim.general]p3:
    //   If a declaration declares a member function or member function
    //   template of a class X, the expression this is a prvalue of type
    //   "pointer to cv-qualifier-seq X" between the optional cv-qualifer-seq
    //   and the end of the function-definition, member-declarator, or
    //   declarator.
    Qualifiers ThisTypeQuals;
    CXXRecordDecl *ThisContext = nullptr;
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Function)) {
      ThisContext = Method->getParent();
      ThisTypeQuals = Method->getMethodQualifiers();
    }

    CXXThisScopeRAII ThisScope(*this, ThisContext, ThisTypeQuals,
                               getLangOpts().CPlusPlus11);

    ResultType =
        SubstType(Proto->getReturnType(), MLTAL,
                  Function->getTypeSpecStartLoc(), Function->getDeclName());
    if (ResultType.isNull() || Trap.hasErrorOccurred())
      return TemplateDeductionResult::SubstitutionFailure;
    // CUDA: Kernel function must have 'void' return type.
    if (getLangOpts().CUDA)
      if (Function->hasAttr<CUDAGlobalAttr>() && !ResultType->isVoidType()) {
        Diag(Function->getLocation(), diag::err_kern_type_not_void_return)
            << Function->getType() << Function->getSourceRange();
        return TemplateDeductionResult::SubstitutionFailure;
      }
  }

  // Instantiate the types of each of the function parameters given the
  // explicitly-specified template arguments if we didn't do so earlier.
  if (!Proto->hasTrailingReturn() &&
      SubstParmTypes(Function->getLocation(), Function->parameters(),
                     Proto->getExtParameterInfosOrNull(), MLTAL, ParamTypes,
                     /*params*/ nullptr, ExtParamInfos))
    return TemplateDeductionResult::SubstitutionFailure;

  if (FunctionType) {
    auto EPI = Proto->getExtProtoInfo();
    EPI.ExtParameterInfos = ExtParamInfos.getPointerOrNull(ParamTypes.size());
    *FunctionType = BuildFunctionType(ResultType, ParamTypes,
                                      Function->getLocation(),
                                      Function->getDeclName(),
                                      EPI);
    if (FunctionType->isNull() || Trap.hasErrorOccurred())
      return TemplateDeductionResult::SubstitutionFailure;
  }

  // C++ [temp.arg.explicit]p2:
  //   Trailing template arguments that can be deduced (14.8.2) may be
  //   omitted from the list of explicit template-arguments. If all of the
  //   template arguments can be deduced, they may all be omitted; in this
  //   case, the empty template argument list <> itself may also be omitted.
  //
  // Take all of the explicitly-specified arguments and put them into
  // the set of deduced template arguments. The partially-substituted
  // parameter pack, however, will be set to NULL since the deduction
  // mechanism handles the partially-substituted argument pack directly.
  Deduced.reserve(TemplateParams->size());
  for (unsigned I = 0, N = SugaredExplicitArgumentList->size(); I != N; ++I) {
    const TemplateArgument &Arg = SugaredExplicitArgumentList->get(I);
    if (I == PartiallySubstitutedPackIndex)
      Deduced.push_back(DeducedTemplateArgument());
    else
      Deduced.push_back(Arg);
  }

  return TemplateDeductionResult::Success;
}

/// Check whether the deduced argument type for a call to a function
/// template matches the actual argument type per C++ [temp.deduct.call]p4.
static TemplateDeductionResult
CheckOriginalCallArgDeduction(Sema &S, TemplateDeductionInfo &Info,
                              Sema::OriginalCallArg OriginalArg,
                              QualType DeducedA) {
  ASTContext &Context = S.Context;

  auto Failed = [&]() -> TemplateDeductionResult {
    Info.FirstArg = TemplateArgument(DeducedA);
    Info.SecondArg = TemplateArgument(OriginalArg.OriginalArgType);
    Info.CallArgIndex = OriginalArg.ArgIdx;
    return OriginalArg.DecomposedParam
               ? TemplateDeductionResult::DeducedMismatchNested
               : TemplateDeductionResult::DeducedMismatch;
  };

  QualType A = OriginalArg.OriginalArgType;
  QualType OriginalParamType = OriginalArg.OriginalParamType;

  // Check for type equality (top-level cv-qualifiers are ignored).
  if (Context.hasSameUnqualifiedType(A, DeducedA))
    return TemplateDeductionResult::Success;

  // Strip off references on the argument types; they aren't needed for
  // the following checks.
  if (const ReferenceType *DeducedARef = DeducedA->getAs<ReferenceType>())
    DeducedA = DeducedARef->getPointeeType();
  if (const ReferenceType *ARef = A->getAs<ReferenceType>())
    A = ARef->getPointeeType();

  // C++ [temp.deduct.call]p4:
  //   [...] However, there are three cases that allow a difference:
  //     - If the original P is a reference type, the deduced A (i.e., the
  //       type referred to by the reference) can be more cv-qualified than
  //       the transformed A.
  if (const ReferenceType *OriginalParamRef
      = OriginalParamType->getAs<ReferenceType>()) {
    // We don't want to keep the reference around any more.
    OriginalParamType = OriginalParamRef->getPointeeType();

    // FIXME: Resolve core issue (no number yet): if the original P is a
    // reference type and the transformed A is function type "noexcept F",
    // the deduced A can be F.
    if (A->isFunctionType() && S.IsFunctionConversion(A, DeducedA))
      return TemplateDeductionResult::Success;

    Qualifiers AQuals = A.getQualifiers();
    Qualifiers DeducedAQuals = DeducedA.getQualifiers();

    // Under Objective-C++ ARC, the deduced type may have implicitly
    // been given strong or (when dealing with a const reference)
    // unsafe_unretained lifetime. If so, update the original
    // qualifiers to include this lifetime.
    if (S.getLangOpts().ObjCAutoRefCount &&
        ((DeducedAQuals.getObjCLifetime() == Qualifiers::OCL_Strong &&
          AQuals.getObjCLifetime() == Qualifiers::OCL_None) ||
         (DeducedAQuals.hasConst() &&
          DeducedAQuals.getObjCLifetime() == Qualifiers::OCL_ExplicitNone))) {
      AQuals.setObjCLifetime(DeducedAQuals.getObjCLifetime());
    }

    if (AQuals == DeducedAQuals) {
      // Qualifiers match; there's nothing to do.
    } else if (!DeducedAQuals.compatiblyIncludes(AQuals, S.getASTContext())) {
      return Failed();
    } else {
      // Qualifiers are compatible, so have the argument type adopt the
      // deduced argument type's qualifiers as if we had performed the
      // qualification conversion.
      A = Context.getQualifiedType(A.getUnqualifiedType(), DeducedAQuals);
    }
  }

  //    - The transformed A can be another pointer or pointer to member
  //      type that can be converted to the deduced A via a function pointer
  //      conversion and/or a qualification conversion.
  //
  // Also allow conversions which merely strip __attribute__((noreturn)) from
  // function types (recursively).
  bool ObjCLifetimeConversion = false;
  if ((A->isAnyPointerType() || A->isMemberPointerType()) &&
      (S.IsQualificationConversion(A, DeducedA, false,
                                   ObjCLifetimeConversion) ||
       S.IsFunctionConversion(A, DeducedA)))
    return TemplateDeductionResult::Success;

  //    - If P is a class and P has the form simple-template-id, then the
  //      transformed A can be a derived class of the deduced A. [...]
  //     [...] Likewise, if P is a pointer to a class of the form
  //      simple-template-id, the transformed A can be a pointer to a
  //      derived class pointed to by the deduced A.
  if (const PointerType *OriginalParamPtr
      = OriginalParamType->getAs<PointerType>()) {
    if (const PointerType *DeducedAPtr = DeducedA->getAs<PointerType>()) {
      if (const PointerType *APtr = A->getAs<PointerType>()) {
        if (A->getPointeeType()->isRecordType()) {
          OriginalParamType = OriginalParamPtr->getPointeeType();
          DeducedA = DeducedAPtr->getPointeeType();
          A = APtr->getPointeeType();
        }
      }
    }
  }

  if (Context.hasSameUnqualifiedType(A, DeducedA))
    return TemplateDeductionResult::Success;

  if (A->isRecordType() && isSimpleTemplateIdType(OriginalParamType) &&
      S.IsDerivedFrom(Info.getLocation(), A, DeducedA))
    return TemplateDeductionResult::Success;

  return Failed();
}

/// Find the pack index for a particular parameter index in an instantiation of
/// a function template with specific arguments.
///
/// \return The pack index for whichever pack produced this parameter, or -1
///         if this was not produced by a parameter. Intended to be used as the
///         ArgumentPackSubstitutionIndex for further substitutions.
// FIXME: We should track this in OriginalCallArgs so we don't need to
// reconstruct it here.
static UnsignedOrNone
getPackIndexForParam(Sema &S, FunctionTemplateDecl *FunctionTemplate,
                     const MultiLevelTemplateArgumentList &Args,
                     unsigned ParamIdx) {
  unsigned Idx = 0;
  for (auto *PD : FunctionTemplate->getTemplatedDecl()->parameters()) {
    if (PD->isParameterPack()) {
      UnsignedOrNone NumArgs =
          S.getNumArgumentsInExpansion(PD->getType(), Args);
      unsigned NumExpansions = NumArgs ? *NumArgs : 1;
      if (Idx + NumExpansions > ParamIdx)
        return ParamIdx - Idx;
      Idx += NumExpansions;
    } else {
      if (Idx == ParamIdx)
        return std::nullopt; // Not a pack expansion
      ++Idx;
    }
  }

  llvm_unreachable("parameter index would not be produced from template");
}

// if `Specialization` is a `CXXConstructorDecl` or `CXXConversionDecl`,
// we'll try to instantiate and update its explicit specifier after constraint
// checking.
static TemplateDeductionResult instantiateExplicitSpecifierDeferred(
    Sema &S, FunctionDecl *Specialization,
    const MultiLevelTemplateArgumentList &SubstArgs,
    TemplateDeductionInfo &Info, FunctionTemplateDecl *FunctionTemplate,
    ArrayRef<TemplateArgument> DeducedArgs) {
  auto GetExplicitSpecifier = [](FunctionDecl *D) {
    return isa<CXXConstructorDecl>(D)
               ? cast<CXXConstructorDecl>(D)->getExplicitSpecifier()
               : cast<CXXConversionDecl>(D)->getExplicitSpecifier();
  };
  auto SetExplicitSpecifier = [](FunctionDecl *D, ExplicitSpecifier ES) {
    isa<CXXConstructorDecl>(D)
        ? cast<CXXConstructorDecl>(D)->setExplicitSpecifier(ES)
        : cast<CXXConversionDecl>(D)->setExplicitSpecifier(ES);
  };

  ExplicitSpecifier ES = GetExplicitSpecifier(Specialization);
  Expr *ExplicitExpr = ES.getExpr();
  if (!ExplicitExpr)
    return TemplateDeductionResult::Success;
  if (!ExplicitExpr->isValueDependent())
    return TemplateDeductionResult::Success;

  Sema::InstantiatingTemplate Inst(
      S, Info.getLocation(), FunctionTemplate, DeducedArgs,
      Sema::CodeSynthesisContext::DeducedTemplateArgumentSubstitution, Info);
  if (Inst.isInvalid())
    return TemplateDeductionResult::InstantiationDepth;
  Sema::SFINAETrap Trap(S);
  const ExplicitSpecifier InstantiatedES =
      S.instantiateExplicitSpecifier(SubstArgs, ES);
  if (InstantiatedES.isInvalid() || Trap.hasErrorOccurred()) {
    Specialization->setInvalidDecl(true);
    return TemplateDeductionResult::SubstitutionFailure;
  }
  SetExplicitSpecifier(Specialization, InstantiatedES);
  return TemplateDeductionResult::Success;
}

TemplateDeductionResult Sema::FinishTemplateArgumentDeduction(
    FunctionTemplateDecl *FunctionTemplate,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    unsigned NumExplicitlySpecified, FunctionDecl *&Specialization,
    TemplateDeductionInfo &Info,
    SmallVectorImpl<OriginalCallArg> const *OriginalCallArgs,
    bool PartialOverloading, bool PartialOrdering,
    bool ForOverloadSetAddressResolution,
    llvm::function_ref<bool(bool)> CheckNonDependent) {
  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(
      *this, Sema::ExpressionEvaluationContext::Unevaluated);
  SFINAETrap Trap(*this);

  // Enter a new template instantiation context while we instantiate the
  // actual function declaration.
  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  InstantiatingTemplate Inst(
      *this, Info.getLocation(), FunctionTemplate, DeducedArgs,
      CodeSynthesisContext::DeducedTemplateArgumentSubstitution, Info);
  if (Inst.isInvalid())
    return TemplateDeductionResult::InstantiationDepth;

  ContextRAII SavedContext(*this, FunctionTemplate->getTemplatedDecl());

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  bool IsIncomplete = false;
  CheckTemplateArgumentInfo CTAI(PartialOrdering);
  if (auto Result = ConvertDeducedTemplateArguments(
          *this, FunctionTemplate, FunctionTemplate->getTemplateParameters(),
          /*IsDeduced=*/true, Deduced, Info, CTAI, CurrentInstantiationScope,
          NumExplicitlySpecified, PartialOverloading ? &IsIncomplete : nullptr);
      Result != TemplateDeductionResult::Success)
    return Result;

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *SugaredDeducedArgumentList =
      TemplateArgumentList::CreateCopy(Context, CTAI.SugaredConverted);
  TemplateArgumentList *CanonicalDeducedArgumentList =
      TemplateArgumentList::CreateCopy(Context, CTAI.CanonicalConverted);
  Info.reset(SugaredDeducedArgumentList, CanonicalDeducedArgumentList);

  // Substitute the deduced template arguments into the function template
  // declaration to produce the function template specialization.
  DeclContext *Owner = FunctionTemplate->getDeclContext();
  if (FunctionTemplate->getFriendObjectKind())
    Owner = FunctionTemplate->getLexicalDeclContext();
  FunctionDecl *FD = FunctionTemplate->getTemplatedDecl();

  if (CheckNonDependent(/*OnlyInitializeNonUserDefinedConversions=*/true))
    return TemplateDeductionResult::NonDependentConversionFailure;

  // C++20 [temp.deduct.general]p5: [CWG2369]
  //   If the function template has associated constraints, those constraints
  //   are checked for satisfaction. If the constraints are not satisfied, type
  //   deduction fails.
  //
  // FIXME: We haven't implemented CWG2369 for lambdas yet, because we need
  // to figure out how to instantiate lambda captures to the scope without
  // first instantiating the lambda.
  bool IsLambda = isLambdaCallOperator(FD) || isLambdaConversionOperator(FD);
  if (!IsLambda && !IsIncomplete) {
    if (CheckFunctionTemplateConstraints(
            Info.getLocation(),
            FunctionTemplate->getCanonicalDecl()->getTemplatedDecl(),
            CTAI.CanonicalConverted, Info.AssociatedConstraintsSatisfaction))
      return TemplateDeductionResult::MiscellaneousDeductionFailure;
    if (!Info.AssociatedConstraintsSatisfaction.IsSatisfied) {
      Info.reset(Info.takeSugared(), TemplateArgumentList::CreateCopy(
                                         Context, CTAI.CanonicalConverted));
      return TemplateDeductionResult::ConstraintsNotSatisfied;
    }
  }
  // C++ [temp.deduct.call]p10: [CWG1391]
  //   If deduction succeeds for all parameters that contain
  //   template-parameters that participate in template argument deduction,
  //   and all template arguments are explicitly specified, deduced, or
  //   obtained from default template arguments, remaining parameters are then
  //   compared with the corresponding arguments. For each remaining parameter
  //   P with a type that was non-dependent before substitution of any
  //   explicitly-specified template arguments, if the corresponding argument
  //   A cannot be implicitly converted to P, deduction fails.
  if (CheckNonDependent(/*OnlyInitializeNonUserDefinedConversions=*/false))
    return TemplateDeductionResult::NonDependentConversionFailure;

  MultiLevelTemplateArgumentList SubstArgs(
      FunctionTemplate, CanonicalDeducedArgumentList->asArray(),
      /*Final=*/false);
  Specialization = cast_or_null<FunctionDecl>(
      SubstDecl(FD, Owner, SubstArgs));
  if (!Specialization || Specialization->isInvalidDecl())
    return TemplateDeductionResult::SubstitutionFailure;

  assert(isSameDeclaration(Specialization->getPrimaryTemplate(),
                           FunctionTemplate));

  // If the template argument list is owned by the function template
  // specialization, release it.
  if (Specialization->getTemplateSpecializationArgs() ==
          CanonicalDeducedArgumentList &&
      !Trap.hasErrorOccurred())
    Info.takeCanonical();

  // There may have been an error that did not prevent us from constructing a
  // declaration. Mark the declaration invalid and return with a substitution
  // failure.
  if (Trap.hasErrorOccurred()) {
    Specialization->setInvalidDecl(true);
    return TemplateDeductionResult::SubstitutionFailure;
  }

  // C++2a [temp.deduct]p5
  //   [...] When all template arguments have been deduced [...] all uses of
  //   template parameters [...] are replaced with the corresponding deduced
  //   or default argument values.
  //   [...] If the function template has associated constraints
  //   ([temp.constr.decl]), those constraints are checked for satisfaction
  //   ([temp.constr.constr]). If the constraints are not satisfied, type
  //   deduction fails.
  if (IsLambda && !IsIncomplete) {
    if (CheckFunctionTemplateConstraints(
            Info.getLocation(), Specialization, CTAI.CanonicalConverted,
            Info.AssociatedConstraintsSatisfaction))
      return TemplateDeductionResult::MiscellaneousDeductionFailure;

    if (!Info.AssociatedConstraintsSatisfaction.IsSatisfied) {
      Info.reset(Info.takeSugared(), TemplateArgumentList::CreateCopy(
                                         Context, CTAI.CanonicalConverted));
      return TemplateDeductionResult::ConstraintsNotSatisfied;
    }
  }

  // We skipped the instantiation of the explicit-specifier during the
  // substitution of `FD` before. So, we try to instantiate it back if
  // `Specialization` is either a constructor or a conversion function.
  if (isa<CXXConstructorDecl, CXXConversionDecl>(Specialization)) {
    if (TemplateDeductionResult::Success !=
        instantiateExplicitSpecifierDeferred(*this, Specialization, SubstArgs,
                                             Info, FunctionTemplate,
                                             DeducedArgs)) {
      return TemplateDeductionResult::SubstitutionFailure;
    }
  }

  if (OriginalCallArgs) {
    // C++ [temp.deduct.call]p4:
    //   In general, the deduction process attempts to find template argument
    //   values that will make the deduced A identical to A (after the type A
    //   is transformed as described above). [...]
    llvm::SmallDenseMap<std::pair<unsigned, QualType>, QualType> DeducedATypes;
    for (unsigned I = 0, N = OriginalCallArgs->size(); I != N; ++I) {
      OriginalCallArg OriginalArg = (*OriginalCallArgs)[I];

      auto ParamIdx = OriginalArg.ArgIdx;
      unsigned ExplicitOffset =
          (Specialization->hasCXXExplicitFunctionObjectParameter() &&
           !ForOverloadSetAddressResolution)
              ? 1
              : 0;
      if (ParamIdx >= Specialization->getNumParams() - ExplicitOffset)
        // FIXME: This presumably means a pack ended up smaller than we
        // expected while deducing. Should this not result in deduction
        // failure? Can it even happen?
        continue;

      QualType DeducedA;
      if (!OriginalArg.DecomposedParam) {
        // P is one of the function parameters, just look up its substituted
        // type.
        DeducedA =
            Specialization->getParamDecl(ParamIdx + ExplicitOffset)->getType();
      } else {
        // P is a decomposed element of a parameter corresponding to a
        // braced-init-list argument. Substitute back into P to find the
        // deduced A.
        QualType &CacheEntry =
            DeducedATypes[{ParamIdx, OriginalArg.OriginalParamType}];
        if (CacheEntry.isNull()) {
          ArgPackSubstIndexRAII PackIndex(
              *this, getPackIndexForParam(*this, FunctionTemplate, SubstArgs,
                                          ParamIdx));
          CacheEntry =
              SubstType(OriginalArg.OriginalParamType, SubstArgs,
                        Specialization->getTypeSpecStartLoc(),
                        Specialization->getDeclName());
        }
        DeducedA = CacheEntry;
      }

      if (auto TDK =
              CheckOriginalCallArgDeduction(*this, Info, OriginalArg, DeducedA);
          TDK != TemplateDeductionResult::Success)
        return TDK;
    }
  }

  // If we suppressed any diagnostics while performing template argument
  // deduction, and if we haven't already instantiated this declaration,
  // keep track of these diagnostics. They'll be emitted if this specialization
  // is actually used.
  if (Info.diag_begin() != Info.diag_end()) {
    auto [Pos, Inserted] =
        SuppressedDiagnostics.try_emplace(Specialization->getCanonicalDecl());
    if (Inserted)
      Pos->second.append(Info.diag_begin(), Info.diag_end());
  }

  return TemplateDeductionResult::Success;
}

/// Gets the type of a function for template-argument-deducton
/// purposes when it's considered as part of an overload set.
static QualType GetTypeOfFunction(Sema &S, const OverloadExpr::FindResult &R,
                                  FunctionDecl *Fn) {
  // We may need to deduce the return type of the function now.
  if (S.getLangOpts().CPlusPlus14 && Fn->getReturnType()->isUndeducedType() &&
      S.DeduceReturnType(Fn, R.Expression->getExprLoc(), /*Diagnose*/ false))
    return {};

  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn))
    if (Method->isImplicitObjectMemberFunction()) {
      // An instance method that's referenced in a form that doesn't
      // look like a member pointer is just invalid.
      if (!R.HasFormOfMemberPointer)
        return {};

      return S.Context.getMemberPointerType(
          Fn->getType(), /*Qualifier=*/std::nullopt, Method->getParent());
    }

  if (!R.IsAddressOfOperand) return Fn->getType();
  return S.Context.getPointerType(Fn->getType());
}

/// Apply the deduction rules for overload sets.
///
/// \return the null type if this argument should be treated as an
/// undeduced context
static QualType
ResolveOverloadForDeduction(Sema &S, TemplateParameterList *TemplateParams,
                            Expr *Arg, QualType ParamType,
                            bool ParamWasReference,
                            TemplateSpecCandidateSet *FailedTSC = nullptr) {

  OverloadExpr::FindResult R = OverloadExpr::find(Arg);

  OverloadExpr *Ovl = R.Expression;

  // C++0x [temp.deduct.call]p4
  unsigned TDF = 0;
  if (ParamWasReference)
    TDF |= TDF_ParamWithReferenceType;
  if (R.IsAddressOfOperand)
    TDF |= TDF_IgnoreQualifiers;

  // C++0x [temp.deduct.call]p6:
  //   When P is a function type, pointer to function type, or pointer
  //   to member function type:

  if (!ParamType->isFunctionType() &&
      !ParamType->isFunctionPointerType() &&
      !ParamType->isMemberFunctionPointerType()) {
    if (Ovl->hasExplicitTemplateArgs()) {
      // But we can still look for an explicit specialization.
      if (FunctionDecl *ExplicitSpec =
              S.ResolveSingleFunctionTemplateSpecialization(
                  Ovl, /*Complain=*/false,
                  /*Found=*/nullptr, FailedTSC,
                  /*ForTypeDeduction=*/true))
        return GetTypeOfFunction(S, R, ExplicitSpec);
    }

    DeclAccessPair DAP;
    if (FunctionDecl *Viable =
            S.resolveAddressOfSingleOverloadCandidate(Arg, DAP))
      return GetTypeOfFunction(S, R, Viable);

    return {};
  }

  // Gather the explicit template arguments, if any.
  TemplateArgumentListInfo ExplicitTemplateArgs;
  if (Ovl->hasExplicitTemplateArgs())
    Ovl->copyTemplateArgumentsInto(ExplicitTemplateArgs);
  QualType Match;
  for (UnresolvedSetIterator I = Ovl->decls_begin(),
         E = Ovl->decls_end(); I != E; ++I) {
    NamedDecl *D = (*I)->getUnderlyingDecl();

    if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(D)) {
      //   - If the argument is an overload set containing one or more
      //     function templates, the parameter is treated as a
      //     non-deduced context.
      if (!Ovl->hasExplicitTemplateArgs())
        return {};

      // Otherwise, see if we can resolve a function type
      FunctionDecl *Specialization = nullptr;
      TemplateDeductionInfo Info(Ovl->getNameLoc());
      if (S.DeduceTemplateArguments(FunTmpl, &ExplicitTemplateArgs,
                                    Specialization,
                                    Info) != TemplateDeductionResult::Success)
        continue;

      D = Specialization;
    }

    FunctionDecl *Fn = cast<FunctionDecl>(D);
    QualType ArgType = GetTypeOfFunction(S, R, Fn);
    if (ArgType.isNull()) continue;

    // Function-to-pointer conversion.
    if (!ParamWasReference && ParamType->isPointerType() &&
        ArgType->isFunctionType())
      ArgType = S.Context.getPointerType(ArgType);

    //   - If the argument is an overload set (not containing function
    //     templates), trial argument deduction is attempted using each
    //     of the members of the set. If deduction succeeds for only one
    //     of the overload set members, that member is used as the
    //     argument value for the deduction. If deduction succeeds for
    //     more than one member of the overload set the parameter is
    //     treated as a non-deduced context.

    // We do all of this in a fresh context per C++0x [temp.deduct.type]p2:
    //   Type deduction is done independently for each P/A pair, and
    //   the deduced template argument values are then combined.
    // So we do not reject deductions which were made elsewhere.
    SmallVector<DeducedTemplateArgument, 8>
      Deduced(TemplateParams->size());
    TemplateDeductionInfo Info(Ovl->getNameLoc());
    TemplateDeductionResult Result = DeduceTemplateArgumentsByTypeMatch(
        S, TemplateParams, ParamType, ArgType, Info, Deduced, TDF,
        PartialOrderingKind::None, /*DeducedFromArrayBound=*/false,
        /*HasDeducedAnyParam=*/nullptr);
    if (Result != TemplateDeductionResult::Success)
      continue;
    // C++ [temp.deduct.call]p6:
    //   [...] If all successful deductions yield the same deduced A, that
    //   deduced A is the result of deduction; otherwise, the parameter is
    //   treated as a non-deduced context. [...]
    if (!Match.isNull() && !S.isSameOrCompatibleFunctionType(Match, ArgType))
      return {};
    Match = ArgType;
  }

  return Match;
}

/// Perform the adjustments to the parameter and argument types
/// described in C++ [temp.deduct.call].
///
/// \returns true if the caller should not attempt to perform any template
/// argument deduction based on this P/A pair because the argument is an
/// overloaded function set that could not be resolved.
static bool AdjustFunctionParmAndArgTypesForDeduction(
    Sema &S, TemplateParameterList *TemplateParams, unsigned FirstInnerIndex,
    QualType &ParamType, QualType &ArgType,
    Expr::Classification ArgClassification, Expr *Arg, unsigned &TDF,
    TemplateSpecCandidateSet *FailedTSC = nullptr) {
  // C++0x [temp.deduct.call]p3:
  //   If P is a cv-qualified type, the top level cv-qualifiers of P's type
  //   are ignored for type deduction.
  if (ParamType.hasQualifiers())
    ParamType = ParamType.getUnqualifiedType();

  //   [...] If P is a reference type, the type referred to by P is
  //   used for type deduction.
  const ReferenceType *ParamRefType = ParamType->getAs<ReferenceType>();
  if (ParamRefType)
    ParamType = ParamRefType->getPointeeType();

  // Overload sets usually make this parameter an undeduced context,
  // but there are sometimes special circumstances.  Typically
  // involving a template-id-expr.
  if (ArgType == S.Context.OverloadTy) {
    assert(Arg && "expected a non-null arg expression");
    ArgType = ResolveOverloadForDeduction(S, TemplateParams, Arg, ParamType,
                                          ParamRefType != nullptr, FailedTSC);
    if (ArgType.isNull())
      return true;
  }

  if (ParamRefType) {
    // If the argument has incomplete array type, try to complete its type.
    if (ArgType->isIncompleteArrayType()) {
      assert(Arg && "expected a non-null arg expression");
      ArgType = S.getCompletedType(Arg);
    }

    // C++1z [temp.deduct.call]p3:
    //   If P is a forwarding reference and the argument is an lvalue, the type
    //   "lvalue reference to A" is used in place of A for type deduction.
    if (isForwardingReference(QualType(ParamRefType, 0), FirstInnerIndex) &&
        ArgClassification.isLValue()) {
      if (S.getLangOpts().OpenCL && !ArgType.hasAddressSpace())
        ArgType = S.Context.getAddrSpaceQualType(
            ArgType, S.Context.getDefaultOpenCLPointeeAddrSpace());
      ArgType = S.Context.getLValueReferenceType(ArgType);
    }
  } else {
    // C++ [temp.deduct.call]p2:
    //   If P is not a reference type:
    //   - If A is an array type, the pointer type produced by the
    //     array-to-pointer standard conversion (4.2) is used in place of
    //     A for type deduction; otherwise,
    //   - If A is a function type, the pointer type produced by the
    //     function-to-pointer standard conversion (4.3) is used in place
    //     of A for type deduction; otherwise,
    if (ArgType->canDecayToPointerType())
      ArgType = S.Context.getDecayedType(ArgType);
    else {
      // - If A is a cv-qualified type, the top level cv-qualifiers of A's
      //   type are ignored for type deduction.
      ArgType = ArgType.getUnqualifiedType();
    }
  }

  // C++0x [temp.deduct.call]p4:
  //   In general, the deduction process attempts to find template argument
  //   values that will make the deduced A identical to A (after the type A
  //   is transformed as described above). [...]
  TDF = TDF_SkipNonDependent;

  //     - If the original P is a reference type, the deduced A (i.e., the
  //       type referred to by the reference) can be more cv-qualified than
  //       the transformed A.
  if (ParamRefType)
    TDF |= TDF_ParamWithReferenceType;
  //     - The transformed A can be another pointer or pointer to member
  //       type that can be converted to the deduced A via a qualification
  //       conversion (4.4).
  if (ArgType->isPointerType() || ArgType->isMemberPointerType() ||
      ArgType->isObjCObjectPointerType())
    TDF |= TDF_IgnoreQualifiers;
  //     - If P is a class and P has the form simple-template-id, then the
  //       transformed A can be a derived class of the deduced A. Likewise,
  //       if P is a pointer to a class of the form simple-template-id, the
  //       transformed A can be a pointer to a derived class pointed to by
  //       the deduced A.
  if (isSimpleTemplateIdType(ParamType) ||
      (ParamType->getAs<PointerType>() &&
       isSimpleTemplateIdType(
           ParamType->castAs<PointerType>()->getPointeeType())))
    TDF |= TDF_DerivedClass;

  return false;
}

static bool
hasDeducibleTemplateParameters(Sema &S, FunctionTemplateDecl *FunctionTemplate,
                               QualType T);

static TemplateDeductionResult DeduceTemplateArgumentsFromCallArgument(
    Sema &S, TemplateParameterList *TemplateParams, unsigned FirstInnerIndex,
    QualType ParamType, QualType ArgType,
    Expr::Classification ArgClassification, Expr *Arg,
    TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    SmallVectorImpl<Sema::OriginalCallArg> &OriginalCallArgs,
    bool DecomposedParam, unsigned ArgIdx, unsigned TDF,
    TemplateSpecCandidateSet *FailedTSC = nullptr);

/// Attempt template argument deduction from an initializer list
///        deemed to be an argument in a function call.
static TemplateDeductionResult DeduceFromInitializerList(
    Sema &S, TemplateParameterList *TemplateParams, QualType AdjustedParamType,
    InitListExpr *ILE, TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    SmallVectorImpl<Sema::OriginalCallArg> &OriginalCallArgs, unsigned ArgIdx,
    unsigned TDF) {
  // C++ [temp.deduct.call]p1: (CWG 1591)
  //   If removing references and cv-qualifiers from P gives
  //   std::initializer_list<P0> or P0[N] for some P0 and N and the argument is
  //   a non-empty initializer list, then deduction is performed instead for
  //   each element of the initializer list, taking P0 as a function template
  //   parameter type and the initializer element as its argument
  //
  // We've already removed references and cv-qualifiers here.
  if (!ILE->getNumInits())
    return TemplateDeductionResult::Success;

  QualType ElTy;
  auto *ArrTy = S.Context.getAsArrayType(AdjustedParamType);
  if (ArrTy)
    ElTy = ArrTy->getElementType();
  else if (!S.isStdInitializerList(AdjustedParamType, &ElTy)) {
    //   Otherwise, an initializer list argument causes the parameter to be
    //   considered a non-deduced context
    return TemplateDeductionResult::Success;
  }

  // Resolving a core issue: a braced-init-list containing any designators is
  // a non-deduced context.
  for (Expr *E : ILE->inits())
    if (isa<DesignatedInitExpr>(E))
      return TemplateDeductionResult::Success;

  // Deduction only needs to be done for dependent types.
  if (ElTy->isDependentType()) {
    for (Expr *E : ILE->inits()) {
      if (auto Result = DeduceTemplateArgumentsFromCallArgument(
              S, TemplateParams, 0, ElTy, E->getType(),
              E->Classify(S.getASTContext()), E, Info, Deduced,
              OriginalCallArgs, true, ArgIdx, TDF);
          Result != TemplateDeductionResult::Success)
        return Result;
    }
  }

  //   in the P0[N] case, if N is a non-type template parameter, N is deduced
  //   from the length of the initializer list.
  if (auto *DependentArrTy = dyn_cast_or_null<DependentSizedArrayType>(ArrTy)) {
    // Determine the array bound is something we can deduce.
    if (NonTypeOrVarTemplateParmDecl NTTP = getDeducedNTTParameterFromExpr(
            Info, DependentArrTy->getSizeExpr())) {
      // We can perform template argument deduction for the given non-type
      // template parameter.
      // C++ [temp.deduct.type]p13:
      //   The type of N in the type T[N] is std::size_t.
      QualType T = S.Context.getSizeType();
      llvm::APInt Size(S.Context.getIntWidth(T),
                       ILE->getNumInitsWithEmbedExpanded());
      if (auto Result = DeduceNonTypeTemplateArgument(
              S, TemplateParams, NTTP, llvm::APSInt(Size), T,
              /*ArrayBound=*/true, Info, /*PartialOrdering=*/false, Deduced,
              /*HasDeducedAnyParam=*/nullptr);
          Result != TemplateDeductionResult::Success)
        return Result;
    }
  }

  return TemplateDeductionResult::Success;
}

/// Perform template argument deduction per [temp.deduct.call] for a
///        single parameter / argument pair.
static TemplateDeductionResult DeduceTemplateArgumentsFromCallArgument(
    Sema &S, TemplateParameterList *TemplateParams, unsigned FirstInnerIndex,
    QualType ParamType, QualType ArgType,
    Expr::Classification ArgClassification, Expr *Arg,
    TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    SmallVectorImpl<Sema::OriginalCallArg> &OriginalCallArgs,
    bool DecomposedParam, unsigned ArgIdx, unsigned TDF,
    TemplateSpecCandidateSet *FailedTSC) {

  QualType OrigParamType = ParamType;

  //   If P is a reference type [...]
  //   If P is a cv-qualified type [...]
  if (AdjustFunctionParmAndArgTypesForDeduction(
          S, TemplateParams, FirstInnerIndex, ParamType, ArgType,
          ArgClassification, Arg, TDF, FailedTSC))
    return TemplateDeductionResult::Success;

  //   If [...] the argument is a non-empty initializer list [...]
  if (InitListExpr *ILE = dyn_cast_if_present<InitListExpr>(Arg))
    return DeduceFromInitializerList(S, TemplateParams, ParamType, ILE, Info,
                                     Deduced, OriginalCallArgs, ArgIdx, TDF);

  //   [...] the deduction process attempts to find template argument values
  //   that will make the deduced A identical to A
  //
  // Keep track of the argument type and corresponding parameter index,
  // so we can check for compatibility between the deduced A and A.
  if (Arg)
    OriginalCallArgs.push_back(
        Sema::OriginalCallArg(OrigParamType, DecomposedParam, ArgIdx, ArgType));
  return DeduceTemplateArgumentsByTypeMatch(
      S, TemplateParams, ParamType, ArgType, Info, Deduced, TDF,
      PartialOrderingKind::None, /*DeducedFromArrayBound=*/false,
      /*HasDeducedAnyParam=*/nullptr);
}

TemplateDeductionResult Sema::DeduceTemplateArguments(
    FunctionTemplateDecl *FunctionTemplate,
    TemplateArgumentListInfo *ExplicitTemplateArgs, ArrayRef<Expr *> Args,
    FunctionDecl *&Specialization, TemplateDeductionInfo &Info,
    bool PartialOverloading, bool AggregateDeductionCandidate,
    bool PartialOrdering, QualType ObjectType,
    Expr::Classification ObjectClassification,
    bool ForOverloadSetAddressResolution,
    llvm::function_ref<bool(ArrayRef<QualType>, bool)> CheckNonDependent) {
  if (FunctionTemplate->isInvalidDecl())
    return TemplateDeductionResult::Invalid;

  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  unsigned NumParams = Function->getNumParams();
  bool HasExplicitObject = false;
  int ExplicitObjectOffset = 0;

  // [C++26] [over.call.func]p3
  // If the primary-expression is the address of an overload set,
  // the argument list is the same as the expression-list in the call.
  // Otherwise, the argument list is the expression-list in the call augmented
  // by the addition of an implied object argument as in a qualified function
  // call.
  if (!ForOverloadSetAddressResolution &&
      Function->hasCXXExplicitFunctionObjectParameter()) {
    HasExplicitObject = true;
    ExplicitObjectOffset = 1;
  }

  unsigned FirstInnerIndex = getFirstInnerIndex(FunctionTemplate);

  // C++ [temp.deduct.call]p1:
  //   Template argument deduction is done by comparing each function template
  //   parameter type (call it P) with the type of the corresponding argument
  //   of the call (call it A) as described below.
  if (Args.size() < Function->getMinRequiredExplicitArguments() &&
      !PartialOverloading)
    return TemplateDeductionResult::TooFewArguments;
  else if (TooManyArguments(NumParams, Args.size() + ExplicitObjectOffset,
                            PartialOverloading)) {
    const auto *Proto = Function->getType()->castAs<FunctionProtoType>();
    if (Proto->isTemplateVariadic())
      /* Do nothing */;
    else if (!Proto->isVariadic())
      return TemplateDeductionResult::TooManyArguments;
  }

  // The types of the parameters from which we will perform template argument
  // deduction.
  LocalInstantiationScope InstScope(*this);
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  SmallVector<QualType, 8> ParamTypes;
  unsigned NumExplicitlySpecified = 0;
  if (ExplicitTemplateArgs) {
    TemplateDeductionResult Result;
    runWithSufficientStackSpace(Info.getLocation(), [&] {
      Result = SubstituteExplicitTemplateArguments(
          FunctionTemplate, *ExplicitTemplateArgs, Deduced, ParamTypes, nullptr,
          Info);
    });
    if (Result != TemplateDeductionResult::Success)
      return Result;

    NumExplicitlySpecified = Deduced.size();
  } else {
    // Just fill in the parameter types from the function declaration.
    for (unsigned I = 0; I != NumParams; ++I)
      ParamTypes.push_back(Function->getParamDecl(I)->getType());
  }

  SmallVector<OriginalCallArg, 8> OriginalCallArgs;

  // Deduce an argument of type ParamType from an expression with index ArgIdx.
  auto DeduceCallArgument = [&](QualType ParamType, unsigned ArgIdx,
                                bool ExplicitObjectArgument) {
    // C++ [demp.deduct.call]p1: (DR1391)
    //   Template argument deduction is done by comparing each function template
    //   parameter that contains template-parameters that participate in
    //   template argument deduction ...
    if (!hasDeducibleTemplateParameters(*this, FunctionTemplate, ParamType))
      return TemplateDeductionResult::Success;

    if (ExplicitObjectArgument) {
      //   ... with the type of the corresponding argument
      return DeduceTemplateArgumentsFromCallArgument(
          *this, TemplateParams, FirstInnerIndex, ParamType, ObjectType,
          ObjectClassification,
          /*Arg=*/nullptr, Info, Deduced, OriginalCallArgs,
          /*Decomposed*/ false, ArgIdx, /*TDF*/ 0);
    }

    //   ... with the type of the corresponding argument
    return DeduceTemplateArgumentsFromCallArgument(
        *this, TemplateParams, FirstInnerIndex, ParamType,
        Args[ArgIdx]->getType(), Args[ArgIdx]->Classify(getASTContext()),
        Args[ArgIdx], Info, Deduced, OriginalCallArgs, /*Decomposed*/ false,
        ArgIdx, /*TDF*/ 0);
  };

  // Deduce template arguments from the function parameters.
  Deduced.resize(TemplateParams->size());
  SmallVector<QualType, 8> ParamTypesForArgChecking;
  for (unsigned ParamIdx = 0, NumParamTypes = ParamTypes.size(), ArgIdx = 0;
       ParamIdx != NumParamTypes; ++ParamIdx) {
    QualType ParamType = ParamTypes[ParamIdx];

    const PackExpansionType *ParamExpansion =
        dyn_cast<PackExpansionType>(ParamType);
    if (!ParamExpansion) {
      // Simple case: matching a function parameter to a function argument.
      if (ArgIdx >= Args.size() && !(HasExplicitObject && ParamIdx == 0))
        break;

      ParamTypesForArgChecking.push_back(ParamType);

      if (ParamIdx == 0 && HasExplicitObject) {
        if (ObjectType.isNull())
          return TemplateDeductionResult::InvalidExplicitArguments;

        if (auto Result = DeduceCallArgument(ParamType, 0,
                                             /*ExplicitObjectArgument=*/true);
            Result != TemplateDeductionResult::Success)
          return Result;
        continue;
      }

      if (auto Result = DeduceCallArgument(ParamType, ArgIdx++,
                                           /*ExplicitObjectArgument=*/false);
          Result != TemplateDeductionResult::Success)
        return Result;

      continue;
    }

    bool IsTrailingPack = ParamIdx + 1 == NumParamTypes;

    QualType ParamPattern = ParamExpansion->getPattern();
    PackDeductionScope PackScope(*this, TemplateParams, Deduced, Info,
                                 ParamPattern,
                                 AggregateDeductionCandidate && IsTrailingPack);

    // C++0x [temp.deduct.call]p1:
    //   For a function parameter pack that occurs at the end of the
    //   parameter-declaration-list, the type A of each remaining argument of
    //   the call is compared with the type P of the declarator-id of the
    //   function parameter pack. Each comparison deduces template arguments
    //   for subsequent positions in the template parameter packs expanded by
    //   the function parameter pack. When a function parameter pack appears
    //   in a non-deduced context [not at the end of the list], the type of
    //   that parameter pack is never deduced.
    //
    // FIXME: The above rule allows the size of the parameter pack to change
    // after we skip it (in the non-deduced case). That makes no sense, so
    // we instead notionally deduce the pack against N arguments, where N is
    // the length of the explicitly-specified pack if it's expanded by the
    // parameter pack and 0 otherwise, and we treat each deduction as a
    // non-deduced context.
    if (IsTrailingPack || PackScope.hasFixedArity()) {
      for (; ArgIdx < Args.size() && PackScope.hasNextElement();
           PackScope.nextPackElement(), ++ArgIdx) {
        ParamTypesForArgChecking.push_back(ParamPattern);
        if (auto Result = DeduceCallArgument(ParamPattern, ArgIdx,
                                             /*ExplicitObjectArgument=*/false);
            Result != TemplateDeductionResult::Success)
          return Result;
      }
    } else {
      // If the parameter type contains an explicitly-specified pack that we
      // could not expand, skip the number of parameters notionally created
      // by the expansion.
      UnsignedOrNone NumExpansions = ParamExpansion->getNumExpansions();
      if (NumExpansions && !PackScope.isPartiallyExpanded()) {
        for (unsigned I = 0; I != *NumExpansions && ArgIdx < Args.size();
             ++I, ++ArgIdx) {
          ParamTypesForArgChecking.push_back(ParamPattern);
          // FIXME: Should we add OriginalCallArgs for these? What if the
          // corresponding argument is a list?
          PackScope.nextPackElement();
        }
      } else if (!IsTrailingPack && !PackScope.isPartiallyExpanded() &&
                 PackScope.isDeducedFromEarlierParameter()) {
        // [temp.deduct.general#3]
        // When all template arguments have been deduced
        // or obtained from default template arguments, all uses of template
        // parameters in the template parameter list of the template are
        // replaced with the corresponding deduced or default argument values
        //
        // If we have a trailing parameter pack, that has been deduced
        // previously we substitute the pack here in a similar fashion as
        // above with the trailing parameter packs. The main difference here is
        // that, in this case we are not processing all of the remaining
        // arguments. We are only process as many arguments as we have in
        // the already deduced parameter.
        UnsignedOrNone ArgPosAfterSubstitution =
            PackScope.getSavedPackSizeIfAllEqual();
        if (!ArgPosAfterSubstitution)
          continue;

        unsigned PackArgEnd = ArgIdx + *ArgPosAfterSubstitution;
        for (; ArgIdx < PackArgEnd && ArgIdx < Args.size(); ArgIdx++) {
          ParamTypesForArgChecking.push_back(ParamPattern);
          if (auto Result =
                  DeduceCallArgument(ParamPattern, ArgIdx,
                                     /*ExplicitObjectArgument=*/false);
              Result != TemplateDeductionResult::Success)
            return Result;

          PackScope.nextPackElement();
        }
      }
    }

    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    if (auto Result = PackScope.finish();
        Result != TemplateDeductionResult::Success)
      return Result;
  }

  // Capture the context in which the function call is made. This is the context
  // that is needed when the accessibility of template arguments is checked.
  DeclContext *CallingCtx = CurContext;

  TemplateDeductionResult Result;
  runWithSufficientStackSpace(Info.getLocation(), [&] {
    Result = FinishTemplateArgumentDeduction(
        FunctionTemplate, Deduced, NumExplicitlySpecified, Specialization, Info,
        &OriginalCallArgs, PartialOverloading, PartialOrdering,
        ForOverloadSetAddressResolution,
        [&, CallingCtx](bool OnlyInitializeNonUserDefinedConversions) {
          ContextRAII SavedContext(*this, CallingCtx);
          return CheckNonDependent(ParamTypesForArgChecking,
                                   OnlyInitializeNonUserDefinedConversions);
        });
  });
  return Result;
}

QualType Sema::adjustCCAndNoReturn(QualType ArgFunctionType,
                                   QualType FunctionType,
                                   bool AdjustExceptionSpec) {
  if (ArgFunctionType.isNull())
    return ArgFunctionType;

  const auto *FunctionTypeP = FunctionType->castAs<FunctionProtoType>();
  const auto *ArgFunctionTypeP = ArgFunctionType->castAs<FunctionProtoType>();
  FunctionProtoType::ExtProtoInfo EPI = ArgFunctionTypeP->getExtProtoInfo();
  bool Rebuild = false;

  CallingConv CC = FunctionTypeP->getCallConv();
  if (EPI.ExtInfo.getCC() != CC) {
    EPI.ExtInfo = EPI.ExtInfo.withCallingConv(CC);
    Rebuild = true;
  }

  bool NoReturn = FunctionTypeP->getNoReturnAttr();
  if (EPI.ExtInfo.getNoReturn() != NoReturn) {
    EPI.ExtInfo = EPI.ExtInfo.withNoReturn(NoReturn);
    Rebuild = true;
  }

  if (AdjustExceptionSpec && (FunctionTypeP->hasExceptionSpec() ||
                              ArgFunctionTypeP->hasExceptionSpec())) {
    EPI.ExceptionSpec = FunctionTypeP->getExtProtoInfo().ExceptionSpec;
    Rebuild = true;
  }

  if (!Rebuild)
    return ArgFunctionType;

  return Context.getFunctionType(ArgFunctionTypeP->getReturnType(),
                                 ArgFunctionTypeP->getParamTypes(), EPI);
}

TemplateDeductionResult Sema::DeduceTemplateArguments(
    FunctionTemplateDecl *FunctionTemplate,
    TemplateArgumentListInfo *ExplicitTemplateArgs, QualType ArgFunctionType,
    FunctionDecl *&Specialization, TemplateDeductionInfo &Info,
    bool IsAddressOfFunction) {
  if (FunctionTemplate->isInvalidDecl())
    return TemplateDeductionResult::Invalid;

  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  QualType FunctionType = Function->getType();

  // Substitute any explicit template arguments.
  LocalInstantiationScope InstScope(*this);
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  unsigned NumExplicitlySpecified = 0;
  SmallVector<QualType, 4> ParamTypes;
  if (ExplicitTemplateArgs) {
    TemplateDeductionResult Result;
    runWithSufficientStackSpace(Info.getLocation(), [&] {
      Result = SubstituteExplicitTemplateArguments(
          FunctionTemplate, *ExplicitTemplateArgs, Deduced, ParamTypes,
          &FunctionType, Info);
    });
    if (Result != TemplateDeductionResult::Success)
      return Result;

    NumExplicitlySpecified = Deduced.size();
  }

  // When taking the address of a function, we require convertibility of
  // the resulting function type. Otherwise, we allow arbitrary mismatches
  // of calling convention and noreturn.
  if (!IsAddressOfFunction)
    ArgFunctionType = adjustCCAndNoReturn(ArgFunctionType, FunctionType,
                                          /*AdjustExceptionSpec*/false);

  // Unevaluated SFINAE context.
  std::optional<EnterExpressionEvaluationContext> Unevaluated(
      std::in_place, *this, Sema::ExpressionEvaluationContext::Unevaluated);
  SFINAETrap Trap(*this);

  Deduced.resize(TemplateParams->size());

  // If the function has a deduced return type, substitute it for a dependent
  // type so that we treat it as a non-deduced context in what follows.
  bool HasDeducedReturnType = false;
  if (getLangOpts().CPlusPlus14 &&
      Function->getReturnType()->getContainedAutoType()) {
    FunctionType = SubstAutoTypeDependent(FunctionType);
    HasDeducedReturnType = true;
  }

  if (!ArgFunctionType.isNull() && !FunctionType.isNull()) {
    unsigned TDF =
        TDF_TopLevelParameterTypeList | TDF_AllowCompatibleFunctionType;
    // Deduce template arguments from the function type.
    if (TemplateDeductionResult Result = DeduceTemplateArgumentsByTypeMatch(
            *this, TemplateParams, FunctionType, ArgFunctionType, Info, Deduced,
            TDF, PartialOrderingKind::None, /*DeducedFromArrayBound=*/false,
            /*HasDeducedAnyParam=*/nullptr);
        Result != TemplateDeductionResult::Success)
      return Result;
  }

  TemplateDeductionResult Result;
  runWithSufficientStackSpace(Info.getLocation(), [&] {
    Result = FinishTemplateArgumentDeduction(
        FunctionTemplate, Deduced, NumExplicitlySpecified, Specialization, Info,
        /*OriginalCallArgs=*/nullptr, /*PartialOverloading=*/false,
        /*PartialOrdering=*/true, IsAddressOfFunction);
  });
  if (Result != TemplateDeductionResult::Success)
    return Result;

  // If the function has a deduced return type, deduce it now, so we can check
  // that the deduced function type matches the requested type.
  if (HasDeducedReturnType && IsAddressOfFunction &&
      Specialization->getReturnType()->isUndeducedType() &&
      DeduceReturnType(Specialization, Info.getLocation(), false))
    return TemplateDeductionResult::MiscellaneousDeductionFailure;

  Unevaluated = std::nullopt;
  // [C++26][expr.const]/p17
  // An expression or conversion is immediate-escalating if it is not initially
  // in an immediate function context and it is [...]
  // a potentially-evaluated id-expression that denotes an immediate function.
  if (IsAddressOfFunction && getLangOpts().CPlusPlus20 &&
      Specialization->isImmediateEscalating() &&
      currentEvaluationContext().isPotentiallyEvaluated() &&
      CheckIfFunctionSpecializationIsImmediate(Specialization,
                                               Info.getLocation()))
    return TemplateDeductionResult::MiscellaneousDeductionFailure;

  // Adjust the exception specification of the argument to match the
  // substituted and resolved type we just formed. (Calling convention and
  // noreturn can't be dependent, so we don't actually need this for them
  // right now.)
  QualType SpecializationType = Specialization->getType();
  if (!IsAddressOfFunction) {
    ArgFunctionType = adjustCCAndNoReturn(ArgFunctionType, SpecializationType,
                                          /*AdjustExceptionSpec*/true);

    // Revert placeholder types in the return type back to undeduced types so
    // that the comparison below compares the declared return types.
    if (HasDeducedReturnType) {
      SpecializationType = SubstAutoType(SpecializationType, QualType());
      ArgFunctionType = SubstAutoType(ArgFunctionType, QualType());
    }
  }

  // If the requested function type does not match the actual type of the
  // specialization with respect to arguments of compatible pointer to function
  // types, template argument deduction fails.
  if (!ArgFunctionType.isNull()) {
    if (IsAddressOfFunction ? !isSameOrCompatibleFunctionType(
                                  SpecializationType, ArgFunctionType)
                            : !Context.hasSameFunctionTypeIgnoringExceptionSpec(
                                  SpecializationType, ArgFunctionType)) {
      Info.FirstArg = TemplateArgument(SpecializationType);
      Info.SecondArg = TemplateArgument(ArgFunctionType);
      return TemplateDeductionResult::NonDeducedMismatch;
    }
  }

  return TemplateDeductionResult::Success;
}

TemplateDeductionResult Sema::DeduceTemplateArguments(
    FunctionTemplateDecl *ConversionTemplate, QualType ObjectType,
    Expr::Classification ObjectClassification, QualType A,
    CXXConversionDecl *&Specialization, TemplateDeductionInfo &Info) {
  if (ConversionTemplate->isInvalidDecl())
    return TemplateDeductionResult::Invalid;

  CXXConversionDecl *ConversionGeneric
    = cast<CXXConversionDecl>(ConversionTemplate->getTemplatedDecl());

  QualType P = ConversionGeneric->getConversionType();
  bool IsReferenceP = P->isReferenceType();
  bool IsReferenceA = A->isReferenceType();

  // C++0x [temp.deduct.conv]p2:
  //   If P is a reference type, the type referred to by P is used for
  //   type deduction.
  if (const ReferenceType *PRef = P->getAs<ReferenceType>())
    P = PRef->getPointeeType();

  // C++0x [temp.deduct.conv]p4:
  //   [...] If A is a reference type, the type referred to by A is used
  //   for type deduction.
  if (const ReferenceType *ARef = A->getAs<ReferenceType>()) {
    A = ARef->getPointeeType();
    // We work around a defect in the standard here: cv-qualifiers are also
    // removed from P and A in this case, unless P was a reference type. This
    // seems to mostly match what other compilers are doing.
    if (!IsReferenceP) {
      A = A.getUnqualifiedType();
      P = P.getUnqualifiedType();
    }

  // C++ [temp.deduct.conv]p3:
  //
  //   If A is not a reference type:
  } else {
    assert(!A->isReferenceType() && "Reference types were handled above");

    //   - If P is an array type, the pointer type produced by the
    //     array-to-pointer standard conversion (4.2) is used in place
    //     of P for type deduction; otherwise,
    if (P->isArrayType())
      P = Context.getArrayDecayedType(P);
    //   - If P is a function type, the pointer type produced by the
    //     function-to-pointer standard conversion (4.3) is used in
    //     place of P for type deduction; otherwise,
    else if (P->isFunctionType())
      P = Context.getPointerType(P);
    //   - If P is a cv-qualified type, the top level cv-qualifiers of
    //     P's type are ignored for type deduction.
    else
      P = P.getUnqualifiedType();

    // C++0x [temp.deduct.conv]p4:
    //   If A is a cv-qualified type, the top level cv-qualifiers of A's
    //   type are ignored for type deduction. If A is a reference type, the type
    //   referred to by A is used for type deduction.
    A = A.getUnqualifiedType();
  }

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(
      *this, Sema::ExpressionEvaluationContext::Unevaluated);
  SFINAETrap Trap(*this);

  // C++ [temp.deduct.conv]p1:
  //   Template argument deduction is done by comparing the return
  //   type of the template conversion function (call it P) with the
  //   type that is required as the result of the conversion (call it
  //   A) as described in 14.8.2.4.
  TemplateParameterList *TemplateParams
    = ConversionTemplate->getTemplateParameters();
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(TemplateParams->size());

  // C++0x [temp.deduct.conv]p4:
  //   In general, the deduction process attempts to find template
  //   argument values that will make the deduced A identical to
  //   A. However, there are two cases that allow a difference:
  unsigned TDF = 0;
  //     - If the original A is a reference type, A can be more
  //       cv-qualified than the deduced A (i.e., the type referred to
  //       by the reference)
  if (IsReferenceA)
    TDF |= TDF_ArgWithReferenceType;
  //     - The deduced A can be another pointer or pointer to member
  //       type that can be converted to A via a qualification
  //       conversion.
  //
  // (C++0x [temp.deduct.conv]p6 clarifies that this only happens when
  // both P and A are pointers or member pointers. In this case, we
  // just ignore cv-qualifiers completely).
  if ((P->isPointerType() && A->isPointerType()) ||
      (P->isMemberPointerType() && A->isMemberPointerType()))
    TDF |= TDF_IgnoreQualifiers;

  SmallVector<Sema::OriginalCallArg, 1> OriginalCallArgs;
  if (ConversionGeneric->isExplicitObjectMemberFunction()) {
    QualType ParamType = ConversionGeneric->getParamDecl(0)->getType();
    if (TemplateDeductionResult Result =
            DeduceTemplateArgumentsFromCallArgument(
                *this, TemplateParams, getFirstInnerIndex(ConversionTemplate),
                ParamType, ObjectType, ObjectClassification,
                /*Arg=*/nullptr, Info, Deduced, OriginalCallArgs,
                /*Decomposed*/ false, 0, /*TDF*/ 0);
        Result != TemplateDeductionResult::Success)
      return Result;
  }

  if (TemplateDeductionResult Result = DeduceTemplateArgumentsByTypeMatch(
          *this, TemplateParams, P, A, Info, Deduced, TDF,
          PartialOrderingKind::None, /*DeducedFromArrayBound=*/false,
          /*HasDeducedAnyParam=*/nullptr);
      Result != TemplateDeductionResult::Success)
    return Result;

  // Create an Instantiation Scope for finalizing the operator.
  LocalInstantiationScope InstScope(*this);
  // Finish template argument deduction.
  FunctionDecl *ConversionSpecialized = nullptr;
  TemplateDeductionResult Result;
  runWithSufficientStackSpace(Info.getLocation(), [&] {
    Result = FinishTemplateArgumentDeduction(
        ConversionTemplate, Deduced, 0, ConversionSpecialized, Info,
        &OriginalCallArgs, /*PartialOverloading=*/false,
        /*PartialOrdering=*/false, /*ForOverloadSetAddressResolution*/ false);
  });
  Specialization = cast_or_null<CXXConversionDecl>(ConversionSpecialized);
  return Result;
}

TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                              TemplateArgumentListInfo *ExplicitTemplateArgs,
                              FunctionDecl *&Specialization,
                              TemplateDeductionInfo &Info,
                              bool IsAddressOfFunction) {
  return DeduceTemplateArguments(FunctionTemplate, ExplicitTemplateArgs,
                                 QualType(), Specialization, Info,
                                 IsAddressOfFunction);
}

namespace {
  struct DependentAuto { bool IsPack; };

  /// Substitute the 'auto' specifier or deduced template specialization type
  /// specifier within a type for a given replacement type.
  class SubstituteDeducedTypeTransform :
      public TreeTransform<SubstituteDeducedTypeTransform> {
    QualType Replacement;
    bool ReplacementIsPack;
    bool UseTypeSugar;
    using inherited = TreeTransform<SubstituteDeducedTypeTransform>;

  public:
    SubstituteDeducedTypeTransform(Sema &SemaRef, DependentAuto DA)
        : TreeTransform<SubstituteDeducedTypeTransform>(SemaRef),
          ReplacementIsPack(DA.IsPack), UseTypeSugar(true) {}

    SubstituteDeducedTypeTransform(Sema &SemaRef, QualType Replacement,
                                   bool UseTypeSugar = true)
        : TreeTransform<SubstituteDeducedTypeTransform>(SemaRef),
          Replacement(Replacement), ReplacementIsPack(false),
          UseTypeSugar(UseTypeSugar) {}

    QualType TransformDesugared(TypeLocBuilder &TLB, DeducedTypeLoc TL) {
      assert(isa<TemplateTypeParmType>(Replacement) &&
             "unexpected unsugared replacement kind");
      QualType Result = Replacement;
      TemplateTypeParmTypeLoc NewTL = TLB.push<TemplateTypeParmTypeLoc>(Result);
      NewTL.setNameLoc(TL.getNameLoc());
      return Result;
    }

    QualType TransformAutoType(TypeLocBuilder &TLB, AutoTypeLoc TL) {
      // If we're building the type pattern to deduce against, don't wrap the
      // substituted type in an AutoType. Certain template deduction rules
      // apply only when a template type parameter appears directly (and not if
      // the parameter is found through desugaring). For instance:
      //   auto &&lref = lvalue;
      // must transform into "rvalue reference to T" not "rvalue reference to
      // auto type deduced as T" in order for [temp.deduct.call]p3 to apply.
      //
      // FIXME: Is this still necessary?
      if (!UseTypeSugar)
        return TransformDesugared(TLB, TL);

      QualType Result = SemaRef.Context.getAutoType(
          Replacement, TL.getTypePtr()->getKeyword(), Replacement.isNull(),
          ReplacementIsPack, TL.getTypePtr()->getTypeConstraintConcept(),
          TL.getTypePtr()->getTypeConstraintArguments());
      auto NewTL = TLB.push<AutoTypeLoc>(Result);
      NewTL.copy(TL);
      return Result;
    }

    QualType TransformDeducedTemplateSpecializationType(
        TypeLocBuilder &TLB, DeducedTemplateSpecializationTypeLoc TL) {
      if (!UseTypeSugar)
        return TransformDesugared(TLB, TL);

      QualType Result = SemaRef.Context.getDeducedTemplateSpecializationType(
          TL.getTypePtr()->getKeyword(), TL.getTypePtr()->getTemplateName(),
          Replacement, Replacement.isNull());
      auto NewTL = TLB.push<DeducedTemplateSpecializationTypeLoc>(Result);
      NewTL.setElaboratedKeywordLoc(TL.getElaboratedKeywordLoc());
      NewTL.setNameLoc(TL.getNameLoc());
      NewTL.setQualifierLoc(TL.getQualifierLoc());
      return Result;
    }

    ExprResult TransformLambdaExpr(LambdaExpr *E) {
      // Lambdas never need to be transformed.
      return E;
    }
    bool TransformExceptionSpec(SourceLocation Loc,
                                FunctionProtoType::ExceptionSpecInfo &ESI,
                                SmallVectorImpl<QualType> &Exceptions,
                                bool &Changed) {
      if (ESI.Type == EST_Uninstantiated) {
        ESI.instantiate();
        Changed = true;
      }
      return inherited::TransformExceptionSpec(Loc, ESI, Exceptions, Changed);
    }

    QualType Apply(TypeLoc TL) {
      // Create some scratch storage for the transformed type locations.
      // FIXME: We're just going to throw this information away. Don't build it.
      TypeLocBuilder TLB;
      TLB.reserve(TL.getFullDataSize());
      return TransformType(TLB, TL);
    }
  };

} // namespace

static bool CheckDeducedPlaceholderConstraints(Sema &S, const AutoType &Type,
                                               AutoTypeLoc TypeLoc,
                                               QualType Deduced) {
  ConstraintSatisfaction Satisfaction;
  ConceptDecl *Concept = cast<ConceptDecl>(Type.getTypeConstraintConcept());
  TemplateArgumentListInfo TemplateArgs(TypeLoc.getLAngleLoc(),
                                        TypeLoc.getRAngleLoc());
  TemplateArgs.addArgument(
      TemplateArgumentLoc(TemplateArgument(Deduced),
                          S.Context.getTrivialTypeSourceInfo(
                              Deduced, TypeLoc.getNameLoc())));
  for (unsigned I = 0, C = TypeLoc.getNumArgs(); I != C; ++I)
    TemplateArgs.addArgument(TypeLoc.getArgLoc(I));

  Sema::CheckTemplateArgumentInfo CTAI;
  if (S.CheckTemplateArgumentList(Concept, SourceLocation(), TemplateArgs,
                                  /*DefaultArgs=*/{},
                                  /*PartialTemplateArgs=*/false, CTAI))
    return true;
  MultiLevelTemplateArgumentList MLTAL(Concept, CTAI.CanonicalConverted,
                                       /*Final=*/false);
  // Build up an EvaluationContext with an ImplicitConceptSpecializationDecl so
  // that the template arguments of the constraint can be preserved. For
  // example:
  //
  //  template <class T>
  //  concept C = []<D U = void>() { return true; }();
  //
  // We need the argument for T while evaluating type constraint D in
  // building the CallExpr to the lambda.
  EnterExpressionEvaluationContext EECtx(
      S, Sema::ExpressionEvaluationContext::Unevaluated,
      ImplicitConceptSpecializationDecl::Create(
          S.getASTContext(), Concept->getDeclContext(), Concept->getLocation(),
          CTAI.CanonicalConverted));
  if (S.CheckConstraintSatisfaction(
          Concept, AssociatedConstraint(Concept->getConstraintExpr()), MLTAL,
          TypeLoc.getLocalSourceRange(), Satisfaction))
    return true;
  if (!Satisfaction.IsSatisfied) {
    std::string Buf;
    llvm::raw_string_ostream OS(Buf);
    OS << "'" << Concept->getName();
    if (TypeLoc.hasExplicitTemplateArgs()) {
      printTemplateArgumentList(
          OS, Type.getTypeConstraintArguments(), S.getPrintingPolicy(),
          Type.getTypeConstraintConcept()->getTemplateParameters());
    }
    OS << "'";
    S.Diag(TypeLoc.getConceptNameLoc(),
           diag::err_placeholder_constraints_not_satisfied)
        << Deduced << Buf << TypeLoc.getLocalSourceRange();
    S.DiagnoseUnsatisfiedConstraint(Satisfaction);
    return true;
  }
  return false;
}

TemplateDeductionResult
Sema::DeduceAutoType(TypeLoc Type, Expr *Init, QualType &Result,
                     TemplateDeductionInfo &Info, bool DependentDeduction,
                     bool IgnoreConstraints,
                     TemplateSpecCandidateSet *FailedTSC) {
  assert(DependentDeduction || Info.getDeducedDepth() == 0);
  if (Init->containsErrors())
    return TemplateDeductionResult::AlreadyDiagnosed;

  const AutoType *AT = Type.getType()->getContainedAutoType();
  assert(AT);

  if (Init->getType()->isNonOverloadPlaceholderType() || AT->isDecltypeAuto()) {
    ExprResult NonPlaceholder = CheckPlaceholderExpr(Init);
    if (NonPlaceholder.isInvalid())
      return TemplateDeductionResult::AlreadyDiagnosed;
    Init = NonPlaceholder.get();
  }

  DependentAuto DependentResult = {
      /*.IsPack = */ (bool)Type.getAs<PackExpansionTypeLoc>()};

  if (!DependentDeduction &&
      (Type.getType()->isDependentType() || Init->isTypeDependent() ||
       Init->containsUnexpandedParameterPack())) {
    Result = SubstituteDeducedTypeTransform(*this, DependentResult).Apply(Type);
    assert(!Result.isNull() && "substituting DependentTy can't fail");
    return TemplateDeductionResult::Success;
  }

  // Make sure that we treat 'char[]' equaly as 'char*' in C23 mode.
  auto *String = dyn_cast<StringLiteral>(Init);
  if (getLangOpts().C23 && String && Type.getType()->isArrayType()) {
    Diag(Type.getBeginLoc(), diag::ext_c23_auto_non_plain_identifier);
    TypeLoc TL = TypeLoc(Init->getType(), Type.getOpaqueData());
    Result = SubstituteDeducedTypeTransform(*this, DependentResult).Apply(TL);
    assert(!Result.isNull() && "substituting DependentTy can't fail");
    return TemplateDeductionResult::Success;
  }

  // Emit a warning if 'auto*' is used in pedantic and in C23 mode.
  if (getLangOpts().C23 && Type.getType()->isPointerType()) {
    Diag(Type.getBeginLoc(), diag::ext_c23_auto_non_plain_identifier);
  }

  auto *InitList = dyn_cast<InitListExpr>(Init);
  if (!getLangOpts().CPlusPlus && InitList) {
    Diag(Init->getBeginLoc(), diag::err_auto_init_list_from_c)
        << (int)AT->getKeyword() << getLangOpts().C23;
    return TemplateDeductionResult::AlreadyDiagnosed;
  }

  // Deduce type of TemplParam in Func(Init)
  SmallVector<DeducedTemplateArgument, 1> Deduced;
  Deduced.resize(1);

  // If deduction failed, don't diagnose if the initializer is dependent; it
  // might acquire a matching type in the instantiation.
  auto DeductionFailed = [&](TemplateDeductionResult TDK) {
    if (Init->isTypeDependent()) {
      Result =
          SubstituteDeducedTypeTransform(*this, DependentResult).Apply(Type);
      assert(!Result.isNull() && "substituting DependentTy can't fail");
      return TemplateDeductionResult::Success;
    }
    return TDK;
  };

  SmallVector<OriginalCallArg, 4> OriginalCallArgs;

  QualType DeducedType;
  // If this is a 'decltype(auto)' specifier, do the decltype dance.
  if (AT->isDecltypeAuto()) {
    if (InitList) {
      Diag(Init->getBeginLoc(), diag::err_decltype_auto_initializer_list);
      return TemplateDeductionResult::AlreadyDiagnosed;
    }

    DeducedType = getDecltypeForExpr(Init);
    assert(!DeducedType.isNull());
  } else {
    LocalInstantiationScope InstScope(*this);

    // Build template<class TemplParam> void Func(FuncParam);
    SourceLocation Loc = Init->getExprLoc();
    TemplateTypeParmDecl *TemplParam = TemplateTypeParmDecl::Create(
        Context, nullptr, SourceLocation(), Loc, Info.getDeducedDepth(), 0,
        nullptr, false, false, false);
    QualType TemplArg = QualType(TemplParam->getTypeForDecl(), 0);
    NamedDecl *TemplParamPtr = TemplParam;
    FixedSizeTemplateParameterListStorage<1, false> TemplateParamsSt(
        Context, Loc, Loc, TemplParamPtr, Loc, nullptr);

    if (InitList) {
      // Notionally, we substitute std::initializer_list<T> for 'auto' and
      // deduce against that. Such deduction only succeeds if removing
      // cv-qualifiers and references results in std::initializer_list<T>.
      if (!Type.getType().getNonReferenceType()->getAs<AutoType>())
        return TemplateDeductionResult::Invalid;

      SourceRange DeducedFromInitRange;
      for (Expr *Init : InitList->inits()) {
        // Resolving a core issue: a braced-init-list containing any designators
        // is a non-deduced context.
        if (isa<DesignatedInitExpr>(Init))
          return TemplateDeductionResult::Invalid;
        if (auto TDK = DeduceTemplateArgumentsFromCallArgument(
                *this, TemplateParamsSt.get(), 0, TemplArg, Init->getType(),
                Init->Classify(getASTContext()), Init, Info, Deduced,
                OriginalCallArgs,
                /*Decomposed=*/true,
                /*ArgIdx=*/0, /*TDF=*/0);
            TDK != TemplateDeductionResult::Success) {
          if (TDK == TemplateDeductionResult::Inconsistent) {
            Diag(Info.getLocation(), diag::err_auto_inconsistent_deduction)
                << Info.FirstArg << Info.SecondArg << DeducedFromInitRange
                << Init->getSourceRange();
            return DeductionFailed(TemplateDeductionResult::AlreadyDiagnosed);
          }
          return DeductionFailed(TDK);
        }

        if (DeducedFromInitRange.isInvalid() &&
            Deduced[0].getKind() != TemplateArgument::Null)
          DeducedFromInitRange = Init->getSourceRange();
      }
    } else {
      if (!getLangOpts().CPlusPlus && Init->refersToBitField()) {
        Diag(Loc, diag::err_auto_bitfield);
        return TemplateDeductionResult::AlreadyDiagnosed;
      }
      QualType FuncParam =
          SubstituteDeducedTypeTransform(*this, TemplArg).Apply(Type);
      assert(!FuncParam.isNull() &&
             "substituting template parameter for 'auto' failed");
      if (auto TDK = DeduceTemplateArgumentsFromCallArgument(
              *this, TemplateParamsSt.get(), 0, FuncParam, Init->getType(),
              Init->Classify(getASTContext()), Init, Info, Deduced,
              OriginalCallArgs,
              /*Decomposed=*/false, /*ArgIdx=*/0, /*TDF=*/0, FailedTSC);
          TDK != TemplateDeductionResult::Success)
        return DeductionFailed(TDK);
    }

    // Could be null if somehow 'auto' appears in a non-deduced context.
    if (Deduced[0].getKind() != TemplateArgument::Type)
      return DeductionFailed(TemplateDeductionResult::Incomplete);
    DeducedType = Deduced[0].getAsType();

    if (InitList) {
      DeducedType = BuildStdInitializerList(DeducedType, Loc);
      if (DeducedType.isNull())
        return TemplateDeductionResult::AlreadyDiagnosed;
    }
  }

  if (!Result.isNull()) {
    if (!Context.hasSameType(DeducedType, Result)) {
      Info.FirstArg = Result;
      Info.SecondArg = DeducedType;
      return DeductionFailed(TemplateDeductionResult::Inconsistent);
    }
    DeducedType = Context.getCommonSugaredType(Result, DeducedType);
  }

  if (AT->isConstrained() && !IgnoreConstraints &&
      CheckDeducedPlaceholderConstraints(
          *this, *AT, Type.getContainedAutoTypeLoc(), DeducedType))
    return TemplateDeductionResult::AlreadyDiagnosed;

  Result = SubstituteDeducedTypeTransform(*this, DeducedType).Apply(Type);
  if (Result.isNull())
    return TemplateDeductionResult::AlreadyDiagnosed;

  // Check that the deduced argument type is compatible with the original
  // argument type per C++ [temp.deduct.call]p4.
  QualType DeducedA = InitList ? Deduced[0].getAsType() : Result;
  for (const OriginalCallArg &OriginalArg : OriginalCallArgs) {
    assert((bool)InitList == OriginalArg.DecomposedParam &&
           "decomposed non-init-list in auto deduction?");
    if (auto TDK =
            CheckOriginalCallArgDeduction(*this, Info, OriginalArg, DeducedA);
        TDK != TemplateDeductionResult::Success) {
      Result = QualType();
      return DeductionFailed(TDK);
    }
  }

  return TemplateDeductionResult::Success;
}

QualType Sema::SubstAutoType(QualType TypeWithAuto,
                             QualType TypeToReplaceAuto) {
  assert(TypeToReplaceAuto != Context.DependentTy);
  return SubstituteDeducedTypeTransform(*this, TypeToReplaceAuto)
      .TransformType(TypeWithAuto);
}

TypeSourceInfo *Sema::SubstAutoTypeSourceInfo(TypeSourceInfo *TypeWithAuto,
                                              QualType TypeToReplaceAuto) {
  assert(TypeToReplaceAuto != Context.DependentTy);
  return SubstituteDeducedTypeTransform(*this, TypeToReplaceAuto)
      .TransformType(TypeWithAuto);
}

QualType Sema::SubstAutoTypeDependent(QualType TypeWithAuto) {
  return SubstituteDeducedTypeTransform(*this, DependentAuto{false})
      .TransformType(TypeWithAuto);
}

TypeSourceInfo *
Sema::SubstAutoTypeSourceInfoDependent(TypeSourceInfo *TypeWithAuto) {
  return SubstituteDeducedTypeTransform(*this, DependentAuto{false})
      .TransformType(TypeWithAuto);
}

QualType Sema::ReplaceAutoType(QualType TypeWithAuto,
                               QualType TypeToReplaceAuto) {
  return SubstituteDeducedTypeTransform(*this, TypeToReplaceAuto,
                                        /*UseTypeSugar*/ false)
      .TransformType(TypeWithAuto);
}

TypeSourceInfo *Sema::ReplaceAutoTypeSourceInfo(TypeSourceInfo *TypeWithAuto,
                                                QualType TypeToReplaceAuto) {
  return SubstituteDeducedTypeTransform(*this, TypeToReplaceAuto,
                                        /*UseTypeSugar*/ false)
      .TransformType(TypeWithAuto);
}

void Sema::DiagnoseAutoDeductionFailure(const VarDecl *VDecl,
                                        const Expr *Init) {
  if (isa<InitListExpr>(Init))
    Diag(VDecl->getLocation(),
         VDecl->isInitCapture()
             ? diag::err_init_capture_deduction_failure_from_init_list
             : diag::err_auto_var_deduction_failure_from_init_list)
      << VDecl->getDeclName() << VDecl->getType() << Init->getSourceRange();
  else
    Diag(VDecl->getLocation(),
         VDecl->isInitCapture() ? diag::err_init_capture_deduction_failure
                                : diag::err_auto_var_deduction_failure)
      << VDecl->getDeclName() << VDecl->getType() << Init->getType()
      << Init->getSourceRange();
}

bool Sema::DeduceReturnType(FunctionDecl *FD, SourceLocation Loc,
                            bool Diagnose) {
  assert(FD->getReturnType()->isUndeducedType());

  // For a lambda's conversion operator, deduce any 'auto' or 'decltype(auto)'
  // within the return type from the call operator's type.
  if (isLambdaConversionOperator(FD)) {
    CXXRecordDecl *Lambda = cast<CXXMethodDecl>(FD)->getParent();
    FunctionDecl *CallOp = Lambda->getLambdaCallOperator();

    // For a generic lambda, instantiate the call operator if needed.
    if (auto *Args = FD->getTemplateSpecializationArgs()) {
      CallOp = InstantiateFunctionDeclaration(
          CallOp->getDescribedFunctionTemplate(), Args, Loc);
      if (!CallOp || CallOp->isInvalidDecl())
        return true;

      // We might need to deduce the return type by instantiating the definition
      // of the operator() function.
      if (CallOp->getReturnType()->isUndeducedType()) {
        runWithSufficientStackSpace(Loc, [&] {
          InstantiateFunctionDefinition(Loc, CallOp);
        });
      }
    }

    if (CallOp->isInvalidDecl())
      return true;
    assert(!CallOp->getReturnType()->isUndeducedType() &&
           "failed to deduce lambda return type");

    // Build the new return type from scratch.
    CallingConv RetTyCC = FD->getReturnType()
                              ->getPointeeType()
                              ->castAs<FunctionType>()
                              ->getCallConv();
    QualType RetType = getLambdaConversionFunctionResultType(
        CallOp->getType()->castAs<FunctionProtoType>(), RetTyCC);
    if (FD->getReturnType()->getAs<PointerType>())
      RetType = Context.getPointerType(RetType);
    else {
      assert(FD->getReturnType()->getAs<BlockPointerType>());
      RetType = Context.getBlockPointerType(RetType);
    }
    Context.adjustDeducedFunctionResultType(FD, RetType);
    return false;
  }

  if (FD->getTemplateInstantiationPattern()) {
    runWithSufficientStackSpace(Loc, [&] {
      InstantiateFunctionDefinition(Loc, FD);
    });
  }

  bool StillUndeduced = FD->getReturnType()->isUndeducedType();
  if (StillUndeduced && Diagnose && !FD->isInvalidDecl()) {
    Diag(Loc, diag::err_auto_fn_used_before_defined) << FD;
    Diag(FD->getLocation(), diag::note_callee_decl) << FD;
  }

  return StillUndeduced;
}

bool Sema::CheckIfFunctionSpecializationIsImmediate(FunctionDecl *FD,
                                                    SourceLocation Loc) {
  assert(FD->isImmediateEscalating());

  if (isLambdaConversionOperator(FD)) {
    CXXRecordDecl *Lambda = cast<CXXMethodDecl>(FD)->getParent();
    FunctionDecl *CallOp = Lambda->getLambdaCallOperator();

    // For a generic lambda, instantiate the call operator if needed.
    if (auto *Args = FD->getTemplateSpecializationArgs()) {
      CallOp = InstantiateFunctionDeclaration(
          CallOp->getDescribedFunctionTemplate(), Args, Loc);
      if (!CallOp || CallOp->isInvalidDecl())
        return true;
      runWithSufficientStackSpace(
          Loc, [&] { InstantiateFunctionDefinition(Loc, CallOp); });
    }
    return CallOp->isInvalidDecl();
  }

  if (FD->getTemplateInstantiationPattern()) {
    runWithSufficientStackSpace(
        Loc, [&] { InstantiateFunctionDefinition(Loc, FD); });
  }
  return false;
}

static QualType GetImplicitObjectParameterType(ASTContext &Context,
                                               const CXXMethodDecl *Method,
                                               QualType RawType,
                                               bool IsOtherRvr) {
  // C++20 [temp.func.order]p3.1, p3.2:
  //  - The type X(M) is "rvalue reference to cv A" if the optional
  //    ref-qualifier of M is && or if M has no ref-qualifier and the
  //    positionally-corresponding parameter of the other transformed template
  //    has rvalue reference type; if this determination depends recursively
  //    upon whether X(M) is an rvalue reference type, it is not considered to
  //    have rvalue reference type.
  //
  //  - Otherwise, X(M) is "lvalue reference to cv A".
  assert(Method && !Method->isExplicitObjectMemberFunction() &&
         "expected a member function with no explicit object parameter");

  RawType = Context.getQualifiedType(RawType, Method->getMethodQualifiers());
  if (Method->getRefQualifier() == RQ_RValue ||
      (IsOtherRvr && Method->getRefQualifier() == RQ_None))
    return Context.getRValueReferenceType(RawType);
  return Context.getLValueReferenceType(RawType);
}

static TemplateDeductionResult CheckDeductionConsistency(
    Sema &S, FunctionTemplateDecl *FTD, UnsignedOrNone ArgIdx, QualType P,
    QualType A, ArrayRef<TemplateArgument> DeducedArgs, bool CheckConsistency) {
  MultiLevelTemplateArgumentList MLTAL(FTD, DeducedArgs,
                                       /*Final=*/true);
  Sema::ArgPackSubstIndexRAII PackIndex(
      S,
      ArgIdx ? ::getPackIndexForParam(S, FTD, MLTAL, *ArgIdx) : std::nullopt);
  bool IsIncompleteSubstitution = false;
  // FIXME: A substitution can be incomplete on a non-structural part of the
  // type. Use the canonical type for now, until the TemplateInstantiator can
  // deal with that.

  // Workaround: Implicit deduction guides use InjectedClassNameTypes, whereas
  // the explicit guides don't. The substitution doesn't transform these types,
  // so let it transform their specializations instead.
  bool IsDeductionGuide = isa<CXXDeductionGuideDecl>(FTD->getTemplatedDecl());
  if (IsDeductionGuide) {
    if (auto *Injected = P->getAs<InjectedClassNameType>())
      P = Injected->getOriginalDecl()->getCanonicalTemplateSpecializationType(
          S.Context);
  }
  QualType InstP = S.SubstType(P.getCanonicalType(), MLTAL, FTD->getLocation(),
                               FTD->getDeclName(), &IsIncompleteSubstitution);
  if (InstP.isNull() && !IsIncompleteSubstitution)
    return TemplateDeductionResult::SubstitutionFailure;
  if (!CheckConsistency)
    return TemplateDeductionResult::Success;
  if (IsIncompleteSubstitution)
    return TemplateDeductionResult::Incomplete;

  // [temp.deduct.call]/4 - Check we produced a consistent deduction.
  // This handles just the cases that can appear when partial ordering.
  if (auto *PA = dyn_cast<PackExpansionType>(A);
      PA && !isa<PackExpansionType>(InstP))
    A = PA->getPattern();
  auto T1 = S.Context.getUnqualifiedArrayType(InstP.getNonReferenceType());
  auto T2 = S.Context.getUnqualifiedArrayType(A.getNonReferenceType());
  if (IsDeductionGuide) {
    if (auto *Injected = T1->getAs<InjectedClassNameType>())
      T1 = Injected->getOriginalDecl()->getCanonicalTemplateSpecializationType(
          S.Context);
    if (auto *Injected = T2->getAs<InjectedClassNameType>())
      T2 = Injected->getOriginalDecl()->getCanonicalTemplateSpecializationType(
          S.Context);
  }
  if (!S.Context.hasSameType(T1, T2))
    return TemplateDeductionResult::NonDeducedMismatch;
  return TemplateDeductionResult::Success;
}

template <class T>
static TemplateDeductionResult FinishTemplateArgumentDeduction(
    Sema &S, FunctionTemplateDecl *FTD,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    TemplateDeductionInfo &Info, T &&CheckDeductionConsistency) {
  EnterExpressionEvaluationContext Unevaluated(
      S, Sema::ExpressionEvaluationContext::Unevaluated);
  Sema::SFINAETrap Trap(S);

  Sema::ContextRAII SavedContext(S, getAsDeclContextOrEnclosing(FTD));

  // C++26 [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  bool IsIncomplete = false;
  Sema::CheckTemplateArgumentInfo CTAI(/*PartialOrdering=*/true);
  if (auto Result = ConvertDeducedTemplateArguments(
          S, FTD, FTD->getTemplateParameters(), /*IsDeduced=*/true, Deduced,
          Info, CTAI,
          /*CurrentInstantiationScope=*/nullptr,
          /*NumAlreadyConverted=*/0, &IsIncomplete);
      Result != TemplateDeductionResult::Success)
    return Result;

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *SugaredDeducedArgumentList =
      TemplateArgumentList::CreateCopy(S.Context, CTAI.SugaredConverted);
  TemplateArgumentList *CanonicalDeducedArgumentList =
      TemplateArgumentList::CreateCopy(S.Context, CTAI.CanonicalConverted);

  Info.reset(SugaredDeducedArgumentList, CanonicalDeducedArgumentList);

  // Substitute the deduced template arguments into the argument
  // and verify that the instantiated argument is both valid
  // and equivalent to the parameter.
  LocalInstantiationScope InstScope(S);

  if (auto TDR = CheckDeductionConsistency(S, FTD, CTAI.SugaredConverted);
      TDR != TemplateDeductionResult::Success)
    return TDR;

  return Trap.hasErrorOccurred() ? TemplateDeductionResult::SubstitutionFailure
                                 : TemplateDeductionResult::Success;
}

/// Determine whether the function template \p FT1 is at least as
/// specialized as \p FT2.
static bool isAtLeastAsSpecializedAs(
    Sema &S, SourceLocation Loc, FunctionTemplateDecl *FT1,
    FunctionTemplateDecl *FT2, TemplatePartialOrderingContext TPOC,
    ArrayRef<QualType> Args1, ArrayRef<QualType> Args2, bool Args1Offset) {
  FunctionDecl *FD1 = FT1->getTemplatedDecl();
  FunctionDecl *FD2 = FT2->getTemplatedDecl();
  const FunctionProtoType *Proto1 = FD1->getType()->getAs<FunctionProtoType>();
  const FunctionProtoType *Proto2 = FD2->getType()->getAs<FunctionProtoType>();
  assert(Proto1 && Proto2 && "Function templates must have prototypes");

  // C++26 [temp.deduct.partial]p3:
  //   The types used to determine the ordering depend on the context in which
  //   the partial ordering is done:
  //   - In the context of a function call, the types used are those function
  //     parameter types for which the function call has arguments.
  //   - In the context of a call to a conversion operator, the return types
  //     of the conversion function templates are used.
  //   - In other contexts (14.6.6.2) the function template's function type
  //     is used.

  if (TPOC == TPOC_Other) {
    // We wouldn't be partial ordering these candidates if these didn't match.
    assert(Proto1->getMethodQuals() == Proto2->getMethodQuals() &&
           Proto1->getRefQualifier() == Proto2->getRefQualifier() &&
           Proto1->isVariadic() == Proto2->isVariadic() &&
           "shouldn't partial order functions with different qualifiers in a "
           "context where the function type is used");

    assert(Args1.empty() && Args2.empty() &&
           "Only call context should have arguments");
    Args1 = Proto1->getParamTypes();
    Args2 = Proto2->getParamTypes();
  }

  TemplateParameterList *TemplateParams = FT2->getTemplateParameters();
  SmallVector<DeducedTemplateArgument, 4> Deduced(TemplateParams->size());
  TemplateDeductionInfo Info(Loc);

  bool HasDeducedAnyParamFromReturnType = false;
  if (TPOC != TPOC_Call) {
    if (DeduceTemplateArgumentsByTypeMatch(
            S, TemplateParams, Proto2->getReturnType(), Proto1->getReturnType(),
            Info, Deduced, TDF_None, PartialOrderingKind::Call,
            /*DeducedFromArrayBound=*/false,
            &HasDeducedAnyParamFromReturnType) !=
        TemplateDeductionResult::Success)
      return false;
  }

  llvm::SmallBitVector HasDeducedParam;
  if (TPOC != TPOC_Conversion) {
    HasDeducedParam.resize(Args2.size());
    if (DeduceTemplateArguments(S, TemplateParams, Args2, Args1, Info, Deduced,
                                TDF_None, PartialOrderingKind::Call,
                                /*HasDeducedAnyParam=*/nullptr,
                                &HasDeducedParam) !=
        TemplateDeductionResult::Success)
      return false;
  }

  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  Sema::InstantiatingTemplate Inst(
      S, Info.getLocation(), FT2, DeducedArgs,
      Sema::CodeSynthesisContext::DeducedTemplateArgumentSubstitution, Info);
  if (Inst.isInvalid())
    return false;

  bool AtLeastAsSpecialized;
  S.runWithSufficientStackSpace(Info.getLocation(), [&] {
    AtLeastAsSpecialized =
        ::FinishTemplateArgumentDeduction(
            S, FT2, Deduced, Info,
            [&](Sema &S, FunctionTemplateDecl *FTD,
                ArrayRef<TemplateArgument> DeducedArgs) {
              // As a provisional fix for a core issue that does not
              // exist yet, which may be related to CWG2160, only check the
              // consistency of parameters and return types which participated
              // in deduction. We will still try to substitute them though.
              if (TPOC != TPOC_Call) {
                if (auto TDR = ::CheckDeductionConsistency(
                        S, FTD, /*ArgIdx=*/std::nullopt,
                        Proto2->getReturnType(), Proto1->getReturnType(),
                        DeducedArgs,
                        /*CheckConsistency=*/HasDeducedAnyParamFromReturnType);
                    TDR != TemplateDeductionResult::Success)
                  return TDR;
              }

              if (TPOC == TPOC_Conversion)
                return TemplateDeductionResult::Success;

              return ::DeduceForEachType(
                  S, TemplateParams, Args2, Args1, Info, Deduced,
                  PartialOrderingKind::Call, /*FinishingDeduction=*/true,
                  [&](Sema &S, TemplateParameterList *, int ParamIdx,
                      UnsignedOrNone ArgIdx, QualType P, QualType A,
                      TemplateDeductionInfo &Info,
                      SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                      PartialOrderingKind) {
                    if (ArgIdx && *ArgIdx >= static_cast<unsigned>(Args1Offset))
                      ArgIdx = *ArgIdx - Args1Offset;
                    else
                      ArgIdx = std::nullopt;
                    return ::CheckDeductionConsistency(
                        S, FTD, ArgIdx, P, A, DeducedArgs,
                        /*CheckConsistency=*/HasDeducedParam[ParamIdx]);
                  });
            }) == TemplateDeductionResult::Success;
  });
  if (!AtLeastAsSpecialized)
    return false;

  // C++0x [temp.deduct.partial]p11:
  //   In most cases, all template parameters must have values in order for
  //   deduction to succeed, but for partial ordering purposes a template
  //   parameter may remain without a value provided it is not used in the
  //   types being used for partial ordering. [ Note: a template parameter used
  //   in a non-deduced context is considered used. -end note]
  unsigned ArgIdx = 0, NumArgs = Deduced.size();
  for (; ArgIdx != NumArgs; ++ArgIdx)
    if (Deduced[ArgIdx].isNull())
      break;

  if (ArgIdx == NumArgs) {
    // All template arguments were deduced. FT1 is at least as specialized
    // as FT2.
    return true;
  }

  // Figure out which template parameters were used.
  llvm::SmallBitVector UsedParameters(TemplateParams->size());
  switch (TPOC) {
  case TPOC_Call:
    for (unsigned I = 0, N = Args2.size(); I != N; ++I)
      ::MarkUsedTemplateParameters(S.Context, Args2[I], /*OnlyDeduced=*/false,
                                   TemplateParams->getDepth(), UsedParameters);
    break;

  case TPOC_Conversion:
    ::MarkUsedTemplateParameters(S.Context, Proto2->getReturnType(),
                                 /*OnlyDeduced=*/false,
                                 TemplateParams->getDepth(), UsedParameters);
    break;

  case TPOC_Other:
    // We do not deduce template arguments from the exception specification
    // when determining the primary template of a function template
    // specialization or when taking the address of a function template.
    // Therefore, we do not mark template parameters in the exception
    // specification as used during partial ordering to prevent the following
    // from being ambiguous:
    //
    //   template<typename T, typename U>
    //   void f(U) noexcept(noexcept(T())); // #1
    //
    //   template<typename T>
    //   void f(T*) noexcept; // #2
    //
    //   template<>
    //   void f<int>(int*) noexcept; // explicit specialization of #2
    //
    // Although there is no corresponding wording in the standard, this seems
    // to be the intended behavior given the definition of
    // 'deduction substitution loci' in [temp.deduct].
    ::MarkUsedTemplateParameters(
        S.Context,
        S.Context.getFunctionTypeWithExceptionSpec(FD2->getType(), EST_None),
        /*OnlyDeduced=*/false, TemplateParams->getDepth(), UsedParameters);
    break;
  }

  for (; ArgIdx != NumArgs; ++ArgIdx)
    // If this argument had no value deduced but was used in one of the types
    // used for partial ordering, then deduction fails.
    if (Deduced[ArgIdx].isNull() && UsedParameters[ArgIdx])
      return false;

  return true;
}

enum class MoreSpecializedTrailingPackTieBreakerResult { Equal, Less, More };

// This a speculative fix for CWG1432 (Similar to the fix for CWG1395) that
// there is no wording or even resolution for this issue.
static MoreSpecializedTrailingPackTieBreakerResult
getMoreSpecializedTrailingPackTieBreaker(
    const TemplateSpecializationType *TST1,
    const TemplateSpecializationType *TST2) {
  ArrayRef<TemplateArgument> As1 = TST1->template_arguments(),
                             As2 = TST2->template_arguments();
  const TemplateArgument &TA1 = As1.back(), &TA2 = As2.back();
  bool IsPack = TA1.getKind() == TemplateArgument::Pack;
  assert(IsPack == (TA2.getKind() == TemplateArgument::Pack));
  if (!IsPack)
    return MoreSpecializedTrailingPackTieBreakerResult::Equal;
  assert(As1.size() == As2.size());

  unsigned PackSize1 = TA1.pack_size(), PackSize2 = TA2.pack_size();
  bool IsPackExpansion1 =
      PackSize1 && TA1.pack_elements().back().isPackExpansion();
  bool IsPackExpansion2 =
      PackSize2 && TA2.pack_elements().back().isPackExpansion();
  if (PackSize1 == PackSize2 && IsPackExpansion1 == IsPackExpansion2)
    return MoreSpecializedTrailingPackTieBreakerResult::Equal;
  if (PackSize1 > PackSize2 && IsPackExpansion1)
    return MoreSpecializedTrailingPackTieBreakerResult::More;
  if (PackSize1 < PackSize2 && IsPackExpansion2)
    return MoreSpecializedTrailingPackTieBreakerResult::Less;
  return MoreSpecializedTrailingPackTieBreakerResult::Equal;
}

FunctionTemplateDecl *Sema::getMoreSpecializedTemplate(
    FunctionTemplateDecl *FT1, FunctionTemplateDecl *FT2, SourceLocation Loc,
    TemplatePartialOrderingContext TPOC, unsigned NumCallArguments1,
    QualType RawObj1Ty, QualType RawObj2Ty, bool Reversed,
    bool PartialOverloading) {
  SmallVector<QualType> Args1;
  SmallVector<QualType> Args2;
  const FunctionDecl *FD1 = FT1->getTemplatedDecl();
  const FunctionDecl *FD2 = FT2->getTemplatedDecl();
  bool ShouldConvert1 = false;
  bool ShouldConvert2 = false;
  bool Args1Offset = false;
  bool Args2Offset = false;
  QualType Obj1Ty;
  QualType Obj2Ty;
  if (TPOC == TPOC_Call) {
    const FunctionProtoType *Proto1 =
        FD1->getType()->castAs<FunctionProtoType>();
    const FunctionProtoType *Proto2 =
        FD2->getType()->castAs<FunctionProtoType>();

    //   - In the context of a function call, the function parameter types are
    //     used.
    const CXXMethodDecl *Method1 = dyn_cast<CXXMethodDecl>(FD1);
    const CXXMethodDecl *Method2 = dyn_cast<CXXMethodDecl>(FD2);
    // C++20 [temp.func.order]p3
    //   [...] Each function template M that is a member function is
    //   considered to have a new first parameter of type
    //   X(M), described below, inserted in its function parameter list.
    //
    // Note that we interpret "that is a member function" as
    // "that is a member function with no expicit object argument".
    // Otherwise the ordering rules for methods with expicit objet arguments
    // against anything else make no sense.

    bool NonStaticMethod1 = Method1 && !Method1->isStatic(),
         NonStaticMethod2 = Method2 && !Method2->isStatic();

    auto Params1Begin = Proto1->param_type_begin(),
         Params2Begin = Proto2->param_type_begin();

    size_t NumComparedArguments = NumCallArguments1;

    if (auto OO = FD1->getOverloadedOperator();
        (NonStaticMethod1 && NonStaticMethod2) ||
        (OO != OO_None && OO != OO_Call && OO != OO_Subscript)) {
      ShouldConvert1 =
          NonStaticMethod1 && !Method1->hasCXXExplicitFunctionObjectParameter();
      ShouldConvert2 =
          NonStaticMethod2 && !Method2->hasCXXExplicitFunctionObjectParameter();
      NumComparedArguments += 1;

      if (ShouldConvert1) {
        bool IsRValRef2 =
            ShouldConvert2
                ? Method2->getRefQualifier() == RQ_RValue
                : Proto2->param_type_begin()[0]->isRValueReferenceType();
        // Compare 'this' from Method1 against first parameter from Method2.
        Obj1Ty = GetImplicitObjectParameterType(this->Context, Method1,
                                                RawObj1Ty, IsRValRef2);
        Args1.push_back(Obj1Ty);
        Args1Offset = true;
      }
      if (ShouldConvert2) {
        bool IsRValRef1 =
            ShouldConvert1
                ? Method1->getRefQualifier() == RQ_RValue
                : Proto1->param_type_begin()[0]->isRValueReferenceType();
        // Compare 'this' from Method2 against first parameter from Method1.
        Obj2Ty = GetImplicitObjectParameterType(this->Context, Method2,
                                                RawObj2Ty, IsRValRef1);
        Args2.push_back(Obj2Ty);
        Args2Offset = true;
      }
    } else {
      if (NonStaticMethod1 && Method1->hasCXXExplicitFunctionObjectParameter())
        Params1Begin += 1;
      if (NonStaticMethod2 && Method2->hasCXXExplicitFunctionObjectParameter())
        Params2Begin += 1;
    }
    Args1.insert(Args1.end(), Params1Begin, Proto1->param_type_end());
    Args2.insert(Args2.end(), Params2Begin, Proto2->param_type_end());

    // C++ [temp.func.order]p5:
    //   The presence of unused ellipsis and default arguments has no effect on
    //   the partial ordering of function templates.
    Args1.resize(std::min(Args1.size(), NumComparedArguments));
    Args2.resize(std::min(Args2.size(), NumComparedArguments));

    if (Reversed)
      std::reverse(Args2.begin(), Args2.end());
  } else {
    assert(!Reversed && "Only call context could have reversed arguments");
  }
  bool Better1 = isAtLeastAsSpecializedAs(*this, Loc, FT1, FT2, TPOC, Args1,
                                          Args2, Args2Offset);
  bool Better2 = isAtLeastAsSpecializedAs(*this, Loc, FT2, FT1, TPOC, Args2,
                                          Args1, Args1Offset);
  // C++ [temp.deduct.partial]p10:
  //   F is more specialized than G if F is at least as specialized as G and G
  //   is not at least as specialized as F.
  if (Better1 != Better2) // We have a clear winner
    return Better1 ? FT1 : FT2;

  if (!Better1 && !Better2) // Neither is better than the other
    return nullptr;

  // C++ [temp.deduct.partial]p11:
  //   ... and if G has a trailing function parameter pack for which F does not
  //   have a corresponding parameter, and if F does not have a trailing
  //   function parameter pack, then F is more specialized than G.

  SmallVector<QualType> Param1;
  Param1.reserve(FD1->param_size() + ShouldConvert1);
  if (ShouldConvert1)
    Param1.push_back(Obj1Ty);
  for (const auto &P : FD1->parameters())
    Param1.push_back(P->getType());

  SmallVector<QualType> Param2;
  Param2.reserve(FD2->param_size() + ShouldConvert2);
  if (ShouldConvert2)
    Param2.push_back(Obj2Ty);
  for (const auto &P : FD2->parameters())
    Param2.push_back(P->getType());

  unsigned NumParams1 = Param1.size();
  unsigned NumParams2 = Param2.size();

  bool Variadic1 =
      FD1->param_size() && FD1->parameters().back()->isParameterPack();
  bool Variadic2 =
      FD2->param_size() && FD2->parameters().back()->isParameterPack();
  if (Variadic1 != Variadic2) {
    if (Variadic1 && NumParams1 > NumParams2)
      return FT2;
    if (Variadic2 && NumParams2 > NumParams1)
      return FT1;
  }

  // Skip this tie breaker if we are performing overload resolution with partial
  // arguments, as this breaks some assumptions about how closely related the
  // candidates are.
  for (int i = 0, e = std::min(NumParams1, NumParams2);
       !PartialOverloading && i < e; ++i) {
    QualType T1 = Param1[i].getCanonicalType();
    QualType T2 = Param2[i].getCanonicalType();
    auto *TST1 = dyn_cast<TemplateSpecializationType>(T1);
    auto *TST2 = dyn_cast<TemplateSpecializationType>(T2);
    if (!TST1 || !TST2)
      continue;
    switch (getMoreSpecializedTrailingPackTieBreaker(TST1, TST2)) {
    case MoreSpecializedTrailingPackTieBreakerResult::Less:
      return FT1;
    case MoreSpecializedTrailingPackTieBreakerResult::More:
      return FT2;
    case MoreSpecializedTrailingPackTieBreakerResult::Equal:
      continue;
    }
    llvm_unreachable(
        "unknown MoreSpecializedTrailingPackTieBreakerResult value");
  }

  if (!Context.getLangOpts().CPlusPlus20)
    return nullptr;

  // Match GCC on not implementing [temp.func.order]p6.2.1.

  // C++20 [temp.func.order]p6:
  //   If deduction against the other template succeeds for both transformed
  //   templates, constraints can be considered as follows:

  // C++20 [temp.func.order]p6.1:
  //   If their template-parameter-lists (possibly including template-parameters
  //   invented for an abbreviated function template ([dcl.fct])) or function
  //   parameter lists differ in length, neither template is more specialized
  //   than the other.
  TemplateParameterList *TPL1 = FT1->getTemplateParameters();
  TemplateParameterList *TPL2 = FT2->getTemplateParameters();
  if (TPL1->size() != TPL2->size() || NumParams1 != NumParams2)
    return nullptr;

  // C++20 [temp.func.order]p6.2.2:
  //   Otherwise, if the corresponding template-parameters of the
  //   template-parameter-lists are not equivalent ([temp.over.link]) or if the
  //   function parameters that positionally correspond between the two
  //   templates are not of the same type, neither template is more specialized
  //   than the other.
  if (!TemplateParameterListsAreEqual(TPL1, TPL2, false,
                                      Sema::TPL_TemplateParamsEquivalent))
    return nullptr;

  // [dcl.fct]p5:
  // Any top-level cv-qualifiers modifying a parameter type are deleted when
  // forming the function type.
  for (unsigned i = 0; i < NumParams1; ++i)
    if (!Context.hasSameUnqualifiedType(Param1[i], Param2[i]))
      return nullptr;

  // C++20 [temp.func.order]p6.3:
  //   Otherwise, if the context in which the partial ordering is done is
  //   that of a call to a conversion function and the return types of the
  //   templates are not the same, then neither template is more specialized
  //   than the other.
  if (TPOC == TPOC_Conversion &&
      !Context.hasSameType(FD1->getReturnType(), FD2->getReturnType()))
    return nullptr;

  llvm::SmallVector<AssociatedConstraint, 3> AC1, AC2;
  FT1->getAssociatedConstraints(AC1);
  FT2->getAssociatedConstraints(AC2);
  bool AtLeastAsConstrained1, AtLeastAsConstrained2;
  if (IsAtLeastAsConstrained(FT1, AC1, FT2, AC2, AtLeastAsConstrained1))
    return nullptr;
  if (IsAtLeastAsConstrained(FT2, AC2, FT1, AC1, AtLeastAsConstrained2))
    return nullptr;
  if (AtLeastAsConstrained1 == AtLeastAsConstrained2)
    return nullptr;
  return AtLeastAsConstrained1 ? FT1 : FT2;
}

UnresolvedSetIterator Sema::getMostSpecialized(
    UnresolvedSetIterator SpecBegin, UnresolvedSetIterator SpecEnd,
    TemplateSpecCandidateSet &FailedCandidates,
    SourceLocation Loc, const PartialDiagnostic &NoneDiag,
    const PartialDiagnostic &AmbigDiag, const PartialDiagnostic &CandidateDiag,
    bool Complain, QualType TargetType) {
  if (SpecBegin == SpecEnd) {
    if (Complain) {
      Diag(Loc, NoneDiag);
      FailedCandidates.NoteCandidates(*this, Loc);
    }
    return SpecEnd;
  }

  if (SpecBegin + 1 == SpecEnd)
    return SpecBegin;

  // Find the function template that is better than all of the templates it
  // has been compared to.
  UnresolvedSetIterator Best = SpecBegin;
  FunctionTemplateDecl *BestTemplate
    = cast<FunctionDecl>(*Best)->getPrimaryTemplate();
  assert(BestTemplate && "Not a function template specialization?");
  for (UnresolvedSetIterator I = SpecBegin + 1; I != SpecEnd; ++I) {
    FunctionTemplateDecl *Challenger
      = cast<FunctionDecl>(*I)->getPrimaryTemplate();
    assert(Challenger && "Not a function template specialization?");
    if (declaresSameEntity(getMoreSpecializedTemplate(BestTemplate, Challenger,
                                                      Loc, TPOC_Other, 0),
                           Challenger)) {
      Best = I;
      BestTemplate = Challenger;
    }
  }

  // Make sure that the "best" function template is more specialized than all
  // of the others.
  bool Ambiguous = false;
  for (UnresolvedSetIterator I = SpecBegin; I != SpecEnd; ++I) {
    FunctionTemplateDecl *Challenger
      = cast<FunctionDecl>(*I)->getPrimaryTemplate();
    if (I != Best &&
        !declaresSameEntity(getMoreSpecializedTemplate(BestTemplate, Challenger,
                                                       Loc, TPOC_Other, 0),
                            BestTemplate)) {
      Ambiguous = true;
      break;
    }
  }

  if (!Ambiguous) {
    // We found an answer. Return it.
    return Best;
  }

  // Diagnose the ambiguity.
  if (Complain) {
    Diag(Loc, AmbigDiag);

    // FIXME: Can we order the candidates in some sane way?
    for (UnresolvedSetIterator I = SpecBegin; I != SpecEnd; ++I) {
      PartialDiagnostic PD = CandidateDiag;
      const auto *FD = cast<FunctionDecl>(*I);
      PD << FD << getTemplateArgumentBindingsText(
                      FD->getPrimaryTemplate()->getTemplateParameters(),
                      *FD->getTemplateSpecializationArgs());
      if (!TargetType.isNull())
        HandleFunctionTypeMismatch(PD, FD->getType(), TargetType);
      Diag((*I)->getLocation(), PD);
    }
  }

  return SpecEnd;
}

FunctionDecl *Sema::getMoreConstrainedFunction(FunctionDecl *FD1,
                                               FunctionDecl *FD2) {
  assert(!FD1->getDescribedTemplate() && !FD2->getDescribedTemplate() &&
         "not for function templates");
  assert(!FD1->isFunctionTemplateSpecialization() ||
         (isa<CXXConversionDecl, CXXConstructorDecl>(FD1)));
  assert(!FD2->isFunctionTemplateSpecialization() ||
         (isa<CXXConversionDecl, CXXConstructorDecl>(FD2)));

  FunctionDecl *F1 = FD1;
  if (FunctionDecl *P = FD1->getTemplateInstantiationPattern(false))
    F1 = P;

  FunctionDecl *F2 = FD2;
  if (FunctionDecl *P = FD2->getTemplateInstantiationPattern(false))
    F2 = P;

  llvm::SmallVector<AssociatedConstraint, 1> AC1, AC2;
  F1->getAssociatedConstraints(AC1);
  F2->getAssociatedConstraints(AC2);
  bool AtLeastAsConstrained1, AtLeastAsConstrained2;
  if (IsAtLeastAsConstrained(F1, AC1, F2, AC2, AtLeastAsConstrained1))
    return nullptr;
  if (IsAtLeastAsConstrained(F2, AC2, F1, AC1, AtLeastAsConstrained2))
    return nullptr;
  if (AtLeastAsConstrained1 == AtLeastAsConstrained2)
    return nullptr;
  return AtLeastAsConstrained1 ? FD1 : FD2;
}

/// Determine whether one template specialization, P1, is at least as
/// specialized than another, P2.
///
/// \tparam TemplateLikeDecl The kind of P2, which must be a
/// TemplateDecl or {Class,Var}TemplatePartialSpecializationDecl.
/// \param T1 The injected-class-name of P1 (faked for a variable template).
/// \param T2 The injected-class-name of P2 (faked for a variable template).
/// \param Template The primary template of P2, in case it is a partial
/// specialization, the same as P2 otherwise.
template <typename TemplateLikeDecl>
static bool isAtLeastAsSpecializedAs(Sema &S, QualType T1, QualType T2,
                                     TemplateLikeDecl *P2,
                                     TemplateDecl *Template,
                                     TemplateDeductionInfo &Info) {
  // C++ [temp.class.order]p1:
  //   For two class template partial specializations, the first is at least as
  //   specialized as the second if, given the following rewrite to two
  //   function templates, the first function template is at least as
  //   specialized as the second according to the ordering rules for function
  //   templates (14.6.6.2):
  //     - the first function template has the same template parameters as the
  //       first partial specialization and has a single function parameter
  //       whose type is a class template specialization with the template
  //       arguments of the first partial specialization, and
  //     - the second function template has the same template parameters as the
  //       second partial specialization and has a single function parameter
  //       whose type is a class template specialization with the template
  //       arguments of the second partial specialization.
  //
  // Rather than synthesize function templates, we merely perform the
  // equivalent partial ordering by performing deduction directly on
  // the template arguments of the class template partial
  // specializations. This computation is slightly simpler than the
  // general problem of function template partial ordering, because
  // class template partial specializations are more constrained. We
  // know that every template parameter is deducible from the class
  // template partial specialization's template arguments, for
  // example.
  SmallVector<DeducedTemplateArgument, 4> Deduced;

  // Determine whether P1 is at least as specialized as P2.
  Deduced.resize(P2->getTemplateParameters()->size());
  if (DeduceTemplateArgumentsByTypeMatch(
          S, P2->getTemplateParameters(), T2, T1, Info, Deduced, TDF_None,
          PartialOrderingKind::Call, /*DeducedFromArrayBound=*/false,
          /*HasDeducedAnyParam=*/nullptr) != TemplateDeductionResult::Success)
    return false;

  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(),
                                               Deduced.end());
  Sema::InstantiatingTemplate Inst(S, Info.getLocation(), P2, DeducedArgs,
                                   Info);
  if (Inst.isInvalid())
    return false;

  ArrayRef<TemplateArgument>
      Ps = cast<TemplateSpecializationType>(T2)->template_arguments(),
      As = cast<TemplateSpecializationType>(T1)->template_arguments();

  Sema::SFINAETrap Trap(S);

  TemplateDeductionResult Result;
  S.runWithSufficientStackSpace(Info.getLocation(), [&] {
    Result = ::FinishTemplateArgumentDeduction(
        S, P2, P2->getTemplateParameters(), Template,
        /*IsPartialOrdering=*/true, Ps, As, Deduced, Info,
        /*CopyDeducedArgs=*/false);
  });

  if (Result != TemplateDeductionResult::Success)
    return false;

  if (Trap.hasErrorOccurred())
    return false;

  return true;
}

namespace {
// A dummy class to return nullptr instead of P2 when performing "more
// specialized than primary" check.
struct GetP2 {
  template <typename T1, typename T2,
            std::enable_if_t<std::is_same_v<T1, T2>, bool> = true>
  T2 *operator()(T1 *, T2 *P2) {
    return P2;
  }
  template <typename T1, typename T2,
            std::enable_if_t<!std::is_same_v<T1, T2>, bool> = true>
  T1 *operator()(T1 *, T2 *) {
    return nullptr;
  }
};

// The assumption is that two template argument lists have the same size.
struct TemplateArgumentListAreEqual {
  ASTContext &Ctx;
  TemplateArgumentListAreEqual(ASTContext &Ctx) : Ctx(Ctx) {}

  template <typename T1, typename T2,
            std::enable_if_t<std::is_same_v<T1, T2>, bool> = true>
  bool operator()(T1 *PS1, T2 *PS2) {
    ArrayRef<TemplateArgument> Args1 = PS1->getTemplateArgs().asArray(),
                               Args2 = PS2->getTemplateArgs().asArray();

    for (unsigned I = 0, E = Args1.size(); I < E; ++I) {
      // We use profile, instead of structural comparison of the arguments,
      // because canonicalization can't do the right thing for dependent
      // expressions.
      llvm::FoldingSetNodeID IDA, IDB;
      Args1[I].Profile(IDA, Ctx);
      Args2[I].Profile(IDB, Ctx);
      if (IDA != IDB)
        return false;
    }
    return true;
  }

  template <typename T1, typename T2,
            std::enable_if_t<!std::is_same_v<T1, T2>, bool> = true>
  bool operator()(T1 *Spec, T2 *Primary) {
    ArrayRef<TemplateArgument> Args1 = Spec->getTemplateArgs().asArray(),
                               Args2 = Primary->getInjectedTemplateArgs(Ctx);

    for (unsigned I = 0, E = Args1.size(); I < E; ++I) {
      // We use profile, instead of structural comparison of the arguments,
      // because canonicalization can't do the right thing for dependent
      // expressions.
      llvm::FoldingSetNodeID IDA, IDB;
      Args1[I].Profile(IDA, Ctx);
      // Unlike the specialization arguments, the injected arguments are not
      // always canonical.
      Ctx.getCanonicalTemplateArgument(Args2[I]).Profile(IDB, Ctx);
      if (IDA != IDB)
        return false;
    }
    return true;
  }
};
} // namespace

/// Returns the more specialized template specialization between T1/P1 and
/// T2/P2.
/// - If IsMoreSpecialThanPrimaryCheck is true, T1/P1 is the partial
///   specialization and T2/P2 is the primary template.
/// - otherwise, both T1/P1 and T2/P2 are the partial specialization.
///
/// \param T1 the type of the first template partial specialization
///
/// \param T2 if IsMoreSpecialThanPrimaryCheck is true, the type of the second
///           template partial specialization; otherwise, the type of the
///           primary template.
///
/// \param P1 the first template partial specialization
///
/// \param P2 if IsMoreSpecialThanPrimaryCheck is true, the second template
///           partial specialization; otherwise, the primary template.
///
/// \returns - If IsMoreSpecialThanPrimaryCheck is true, returns P1 if P1 is
///            more specialized, returns nullptr if P1 is not more specialized.
///          - otherwise, returns the more specialized template partial
///            specialization. If neither partial specialization is more
///            specialized, returns NULL.
template <typename TemplateLikeDecl, typename PrimaryDel>
static TemplateLikeDecl *
getMoreSpecialized(Sema &S, QualType T1, QualType T2, TemplateLikeDecl *P1,
                   PrimaryDel *P2, TemplateDeductionInfo &Info) {
  constexpr bool IsMoreSpecialThanPrimaryCheck =
      !std::is_same_v<TemplateLikeDecl, PrimaryDel>;

  TemplateDecl *P2T;
  if constexpr (IsMoreSpecialThanPrimaryCheck)
    P2T = P2;
  else
    P2T = P2->getSpecializedTemplate();

  bool Better1 = isAtLeastAsSpecializedAs(S, T1, T2, P2, P2T, Info);
  if (IsMoreSpecialThanPrimaryCheck && !Better1)
    return nullptr;

  bool Better2 = isAtLeastAsSpecializedAs(S, T2, T1, P1,
                                          P1->getSpecializedTemplate(), Info);
  if (IsMoreSpecialThanPrimaryCheck && !Better2)
    return P1;

  // C++ [temp.deduct.partial]p10:
  //   F is more specialized than G if F is at least as specialized as G and G
  //   is not at least as specialized as F.
  if (Better1 != Better2) // We have a clear winner
    return Better1 ? P1 : GetP2()(P1, P2);

  if (!Better1 && !Better2)
    return nullptr;

  switch (getMoreSpecializedTrailingPackTieBreaker(
      cast<TemplateSpecializationType>(T1),
      cast<TemplateSpecializationType>(T2))) {
  case MoreSpecializedTrailingPackTieBreakerResult::Less:
    return P1;
  case MoreSpecializedTrailingPackTieBreakerResult::More:
    return GetP2()(P1, P2);
  case MoreSpecializedTrailingPackTieBreakerResult::Equal:
    break;
  }

  if (!S.Context.getLangOpts().CPlusPlus20)
    return nullptr;

  // Match GCC on not implementing [temp.func.order]p6.2.1.

  // C++20 [temp.func.order]p6:
  //   If deduction against the other template succeeds for both transformed
  //   templates, constraints can be considered as follows:

  TemplateParameterList *TPL1 = P1->getTemplateParameters();
  TemplateParameterList *TPL2 = P2->getTemplateParameters();
  if (TPL1->size() != TPL2->size())
    return nullptr;

  // C++20 [temp.func.order]p6.2.2:
  // Otherwise, if the corresponding template-parameters of the
  // template-parameter-lists are not equivalent ([temp.over.link]) or if the
  // function parameters that positionally correspond between the two
  // templates are not of the same type, neither template is more specialized
  // than the other.
  if (!S.TemplateParameterListsAreEqual(TPL1, TPL2, false,
                                        Sema::TPL_TemplateParamsEquivalent))
    return nullptr;

  if (!TemplateArgumentListAreEqual(S.getASTContext())(P1, P2))
    return nullptr;

  llvm::SmallVector<AssociatedConstraint, 3> AC1, AC2;
  P1->getAssociatedConstraints(AC1);
  P2->getAssociatedConstraints(AC2);
  bool AtLeastAsConstrained1, AtLeastAsConstrained2;
  if (S.IsAtLeastAsConstrained(P1, AC1, P2, AC2, AtLeastAsConstrained1) ||
      (IsMoreSpecialThanPrimaryCheck && !AtLeastAsConstrained1))
    return nullptr;
  if (S.IsAtLeastAsConstrained(P2, AC2, P1, AC1, AtLeastAsConstrained2))
    return nullptr;
  if (AtLeastAsConstrained1 == AtLeastAsConstrained2)
    return nullptr;
  return AtLeastAsConstrained1 ? P1 : GetP2()(P1, P2);
}

ClassTemplatePartialSpecializationDecl *
Sema::getMoreSpecializedPartialSpecialization(
                                  ClassTemplatePartialSpecializationDecl *PS1,
                                  ClassTemplatePartialSpecializationDecl *PS2,
                                              SourceLocation Loc) {
  QualType PT1 = PS1->getCanonicalInjectedSpecializationType(Context);
  QualType PT2 = PS2->getCanonicalInjectedSpecializationType(Context);

  TemplateDeductionInfo Info(Loc);
  return getMoreSpecialized(*this, PT1, PT2, PS1, PS2, Info);
}

bool Sema::isMoreSpecializedThanPrimary(
    ClassTemplatePartialSpecializationDecl *Spec, TemplateDeductionInfo &Info) {
  ClassTemplateDecl *Primary = Spec->getSpecializedTemplate();
  QualType PrimaryT = Primary->getCanonicalInjectedSpecializationType(Context);
  QualType PartialT = Spec->getCanonicalInjectedSpecializationType(Context);

  ClassTemplatePartialSpecializationDecl *MaybeSpec =
      getMoreSpecialized(*this, PartialT, PrimaryT, Spec, Primary, Info);
  if (MaybeSpec)
    Info.clearSFINAEDiagnostic();
  return MaybeSpec;
}

VarTemplatePartialSpecializationDecl *
Sema::getMoreSpecializedPartialSpecialization(
    VarTemplatePartialSpecializationDecl *PS1,
    VarTemplatePartialSpecializationDecl *PS2, SourceLocation Loc) {
  // Pretend the variable template specializations are class template
  // specializations and form a fake injected class name type for comparison.
  assert(PS1->getSpecializedTemplate() == PS2->getSpecializedTemplate() &&
         "the partial specializations being compared should specialize"
         " the same template.");
  TemplateName Name(PS1->getSpecializedTemplate()->getCanonicalDecl());
  QualType PT1 = Context.getCanonicalTemplateSpecializationType(
      Name, PS1->getTemplateArgs().asArray());
  QualType PT2 = Context.getCanonicalTemplateSpecializationType(
      Name, PS2->getTemplateArgs().asArray());

  TemplateDeductionInfo Info(Loc);
  return getMoreSpecialized(*this, PT1, PT2, PS1, PS2, Info);
}

bool Sema::isMoreSpecializedThanPrimary(
    VarTemplatePartialSpecializationDecl *Spec, TemplateDeductionInfo &Info) {
  VarTemplateDecl *Primary = Spec->getSpecializedTemplate();
  TemplateName Name(Primary->getCanonicalDecl());

  SmallVector<TemplateArgument, 8> PrimaryCanonArgs(
      Primary->getInjectedTemplateArgs(Context));
  Context.canonicalizeTemplateArguments(PrimaryCanonArgs);

  QualType PrimaryT =
      Context.getCanonicalTemplateSpecializationType(Name, PrimaryCanonArgs);
  QualType PartialT = Context.getCanonicalTemplateSpecializationType(
      Name, Spec->getTemplateArgs().asArray());

  VarTemplatePartialSpecializationDecl *MaybeSpec =
      getMoreSpecialized(*this, PartialT, PrimaryT, Spec, Primary, Info);
  if (MaybeSpec)
    Info.clearSFINAEDiagnostic();
  return MaybeSpec;
}

bool Sema::isTemplateTemplateParameterAtLeastAsSpecializedAs(
    TemplateParameterList *P, TemplateDecl *PArg, TemplateDecl *AArg,
    const DefaultArguments &DefaultArgs, SourceLocation ArgLoc,
    bool PartialOrdering, bool *StrictPackMatch) {
  // C++1z [temp.arg.template]p4: (DR 150)
  //   A template template-parameter P is at least as specialized as a
  //   template template-argument A if, given the following rewrite to two
  //   function templates...

  // Rather than synthesize function templates, we merely perform the
  // equivalent partial ordering by performing deduction directly on
  // the template parameter lists of the template template parameters.
  //
  TemplateParameterList *A = AArg->getTemplateParameters();

  Sema::InstantiatingTemplate Inst(
      *this, ArgLoc, Sema::InstantiatingTemplate::PartialOrderingTTP(), PArg,
      SourceRange(P->getTemplateLoc(), P->getRAngleLoc()));
  if (Inst.isInvalid())
    return false;

  //   Given an invented class template X with the template parameter list of
  //   A (including default arguments):
  //    - Each function template has a single function parameter whose type is
  //      a specialization of X with template arguments corresponding to the
  //      template parameters from the respective function template
  SmallVector<TemplateArgument, 8> AArgs(A->getInjectedTemplateArgs(Context));

  // Check P's arguments against A's parameter list. This will fill in default
  // template arguments as needed. AArgs are already correct by construction.
  // We can't just use CheckTemplateIdType because that will expand alias
  // templates.
  SmallVector<TemplateArgument, 4> PArgs(P->getInjectedTemplateArgs(Context));
  {
    TemplateArgumentListInfo PArgList(P->getLAngleLoc(),
                                      P->getRAngleLoc());
    for (unsigned I = 0, N = P->size(); I != N; ++I) {
      // Unwrap packs that getInjectedTemplateArgs wrapped around pack
      // expansions, to form an "as written" argument list.
      TemplateArgument Arg = PArgs[I];
      if (Arg.getKind() == TemplateArgument::Pack) {
        assert(Arg.pack_size() == 1 && Arg.pack_begin()->isPackExpansion());
        Arg = *Arg.pack_begin();
      }
      PArgList.addArgument(getTrivialTemplateArgumentLoc(
          Arg, QualType(), P->getParam(I)->getLocation()));
    }
    PArgs.clear();

    // C++1z [temp.arg.template]p3:
    //   If the rewrite produces an invalid type, then P is not at least as
    //   specialized as A.
    CheckTemplateArgumentInfo CTAI(
        /*PartialOrdering=*/false, /*MatchingTTP=*/true);
    CTAI.SugaredConverted = std::move(PArgs);
    if (CheckTemplateArgumentList(AArg, ArgLoc, PArgList, DefaultArgs,
                                  /*PartialTemplateArgs=*/false, CTAI,
                                  /*UpdateArgsWithConversions=*/true,
                                  /*ConstraintsNotSatisfied=*/nullptr))
      return false;
    PArgs = std::move(CTAI.SugaredConverted);
    if (StrictPackMatch)
      *StrictPackMatch |= CTAI.StrictPackMatch;
  }

  // Determine whether P1 is at least as specialized as P2.
  TemplateDeductionInfo Info(ArgLoc, A->getDepth());
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(A->size());

  //   ... the function template corresponding to P is at least as specialized
  //   as the function template corresponding to A according to the partial
  //   ordering rules for function templates.

  // Provisional resolution for CWG2398: Regarding temp.arg.template]p4, when
  // applying the partial ordering rules for function templates on
  // the rewritten template template parameters:
  //   - In a deduced context, the matching of packs versus fixed-size needs to
  //   be inverted between Ps and As. On non-deduced context, matching needs to
  //   happen both ways, according to [temp.arg.template]p3, but this is
  //   currently implemented as a special case elsewhere.
  switch (::DeduceTemplateArguments(
      *this, A, AArgs, PArgs, Info, Deduced,
      /*NumberOfArgumentsMustMatch=*/false, /*PartialOrdering=*/true,
      PartialOrdering ? PackFold::ArgumentToParameter : PackFold::Both,
      /*HasDeducedAnyParam=*/nullptr)) {
  case clang::TemplateDeductionResult::Success:
    if (StrictPackMatch && Info.hasStrictPackMatch())
      *StrictPackMatch = true;
    break;

  case TemplateDeductionResult::MiscellaneousDeductionFailure:
    Diag(AArg->getLocation(), diag::err_template_param_list_different_arity)
        << (A->size() > P->size()) << /*isTemplateTemplateParameter=*/true
        << SourceRange(A->getTemplateLoc(), P->getRAngleLoc());
    return false;
  case TemplateDeductionResult::NonDeducedMismatch:
    Diag(AArg->getLocation(), diag::err_non_deduced_mismatch)
        << Info.FirstArg << Info.SecondArg;
    return false;
  case TemplateDeductionResult::Inconsistent:
    Diag(getAsNamedDecl(Info.Param)->getLocation(),
         diag::err_inconsistent_deduction)
        << Info.FirstArg << Info.SecondArg;
    return false;
  case TemplateDeductionResult::AlreadyDiagnosed:
    return false;

  // None of these should happen for a plain deduction.
  case TemplateDeductionResult::Invalid:
  case TemplateDeductionResult::InstantiationDepth:
  case TemplateDeductionResult::Incomplete:
  case TemplateDeductionResult::IncompletePack:
  case TemplateDeductionResult::Underqualified:
  case TemplateDeductionResult::SubstitutionFailure:
  case TemplateDeductionResult::DeducedMismatch:
  case TemplateDeductionResult::DeducedMismatchNested:
  case TemplateDeductionResult::TooManyArguments:
  case TemplateDeductionResult::TooFewArguments:
  case TemplateDeductionResult::InvalidExplicitArguments:
  case TemplateDeductionResult::NonDependentConversionFailure:
  case TemplateDeductionResult::ConstraintsNotSatisfied:
  case TemplateDeductionResult::CUDATargetMismatch:
    llvm_unreachable("Unexpected Result");
  }

  TemplateDeductionResult TDK;
  runWithSufficientStackSpace(Info.getLocation(), [&] {
    TDK = ::FinishTemplateArgumentDeduction(
        *this, AArg, AArg->getTemplateParameters(), AArg, PartialOrdering,
        AArgs, PArgs, Deduced, Info, /*CopyDeducedArgs=*/false);
  });
  switch (TDK) {
  case TemplateDeductionResult::Success:
    return true;

  // It doesn't seem possible to get a non-deduced mismatch when partial
  // ordering TTPs, except with an invalid template parameter list which has
  // a parameter after a pack.
  case TemplateDeductionResult::NonDeducedMismatch:
    assert(PArg->isInvalidDecl() && "Unexpected NonDeducedMismatch");
    return false;

  // Substitution failures should have already been diagnosed.
  case TemplateDeductionResult::AlreadyDiagnosed:
  case TemplateDeductionResult::SubstitutionFailure:
  case TemplateDeductionResult::InstantiationDepth:
    return false;

  // None of these should happen when just converting deduced arguments.
  case TemplateDeductionResult::Invalid:
  case TemplateDeductionResult::Incomplete:
  case TemplateDeductionResult::IncompletePack:
  case TemplateDeductionResult::Inconsistent:
  case TemplateDeductionResult::Underqualified:
  case TemplateDeductionResult::DeducedMismatch:
  case TemplateDeductionResult::DeducedMismatchNested:
  case TemplateDeductionResult::TooManyArguments:
  case TemplateDeductionResult::TooFewArguments:
  case TemplateDeductionResult::InvalidExplicitArguments:
  case TemplateDeductionResult::NonDependentConversionFailure:
  case TemplateDeductionResult::ConstraintsNotSatisfied:
  case TemplateDeductionResult::MiscellaneousDeductionFailure:
  case TemplateDeductionResult::CUDATargetMismatch:
    llvm_unreachable("Unexpected Result");
  }
  llvm_unreachable("Unexpected TDK");
}

namespace {
struct MarkUsedTemplateParameterVisitor : DynamicRecursiveASTVisitor {
  llvm::SmallBitVector &Used;
  unsigned Depth;

  MarkUsedTemplateParameterVisitor(llvm::SmallBitVector &Used,
                                   unsigned Depth)
      : Used(Used), Depth(Depth) { }

  bool VisitTemplateTypeParmType(TemplateTypeParmType *T) override {
    if (T->getDepth() == Depth)
      Used[T->getIndex()] = true;
    return true;
  }

  bool TraverseTemplateName(TemplateName Template) override {
    if (auto *TTP = llvm::dyn_cast_or_null<TemplateTemplateParmDecl>(
            Template.getAsTemplateDecl()))
      if (TTP->getDepth() == Depth)
        Used[TTP->getIndex()] = true;
    DynamicRecursiveASTVisitor::TraverseTemplateName(Template);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) override {
    if (auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(E->getDecl()))
      if (NTTP->getDepth() == Depth)
        Used[NTTP->getIndex()] = true;
    return true;
  }

  bool VisitUnresolvedLookupExpr(UnresolvedLookupExpr *ULE) override {
    if (ULE->isConceptReference() || ULE->isVarDeclReference()) {
      if (auto *TTP = ULE->getTemplateTemplateDecl()) {
        if (TTP->getDepth() == Depth)
          Used[TTP->getIndex()] = true;
      }
      for (auto &TLoc : ULE->template_arguments())
        DynamicRecursiveASTVisitor::TraverseTemplateArgumentLoc(TLoc);
    }
    return true;
  }
};
}

/// Mark the template parameters that are used by the given
/// expression.
static void
MarkUsedTemplateParameters(ASTContext &Ctx,
                           const Expr *E,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallBitVector &Used) {
  if (!OnlyDeduced) {
    MarkUsedTemplateParameterVisitor(Used, Depth)
        .TraverseStmt(const_cast<Expr *>(E));
    return;
  }

  // We can deduce from a pack expansion.
  if (const PackExpansionExpr *Expansion = dyn_cast<PackExpansionExpr>(E))
    E = Expansion->getPattern();

  E = unwrapExpressionForDeduction(E);
  if (const auto *ULE = dyn_cast<UnresolvedLookupExpr>(E);
      ULE && (ULE->isConceptReference() || ULE->isVarDeclReference())) {
    if (const auto *TTP = ULE->getTemplateTemplateDecl())
      Used[TTP->getIndex()] = true;
    for (auto &TLoc : ULE->template_arguments())
      MarkUsedTemplateParameters(Ctx, TLoc.getArgument(), OnlyDeduced, Depth,
                                 Used);
    return;
  }

  const NonTypeOrVarTemplateParmDecl NTTP =
      getDeducedNTTParameterFromExpr(E, Depth);
  if (!NTTP)
    return;
  if (NTTP.getDepth() == Depth)
    Used[NTTP.getIndex()] = true;

  // In C++17 mode, additional arguments may be deduced from the type of a
  // non-type argument.
  if (Ctx.getLangOpts().CPlusPlus17)
    MarkUsedTemplateParameters(Ctx, NTTP.getType(), OnlyDeduced, Depth, Used);
}

/// Mark the template parameters that are used by the given
/// nested name specifier.
static void MarkUsedTemplateParameters(ASTContext &Ctx, NestedNameSpecifier NNS,
                                       bool OnlyDeduced, unsigned Depth,
                                       llvm::SmallBitVector &Used) {
  if (NNS.getKind() != NestedNameSpecifier::Kind::Type)
    return;
  MarkUsedTemplateParameters(Ctx, QualType(NNS.getAsType(), 0), OnlyDeduced,
                             Depth, Used);
}

/// Mark the template parameters that are used by the given
/// template name.
static void
MarkUsedTemplateParameters(ASTContext &Ctx,
                           TemplateName Name,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallBitVector &Used) {
  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    if (TemplateTemplateParmDecl *TTP
          = dyn_cast<TemplateTemplateParmDecl>(Template)) {
      if (TTP->getDepth() == Depth)
        Used[TTP->getIndex()] = true;
    }
    return;
  }

  if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName())
    MarkUsedTemplateParameters(Ctx, QTN->getQualifier(), OnlyDeduced,
                               Depth, Used);
  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName())
    MarkUsedTemplateParameters(Ctx, DTN->getQualifier(), OnlyDeduced,
                               Depth, Used);
}

/// Mark the template parameters that are used by the given
/// type.
static void
MarkUsedTemplateParameters(ASTContext &Ctx, QualType T,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallBitVector &Used) {
  if (T.isNull())
    return;

  // Non-dependent types have nothing deducible
  if (!T->isDependentType())
    return;

  T = Ctx.getCanonicalType(T);
  switch (T->getTypeClass()) {
  case Type::Pointer:
    MarkUsedTemplateParameters(Ctx,
                               cast<PointerType>(T)->getPointeeType(),
                               OnlyDeduced,
                               Depth,
                               Used);
    break;

  case Type::BlockPointer:
    MarkUsedTemplateParameters(Ctx,
                               cast<BlockPointerType>(T)->getPointeeType(),
                               OnlyDeduced,
                               Depth,
                               Used);
    break;

  case Type::LValueReference:
  case Type::RValueReference:
    MarkUsedTemplateParameters(Ctx,
                               cast<ReferenceType>(T)->getPointeeType(),
                               OnlyDeduced,
                               Depth,
                               Used);
    break;

  case Type::MemberPointer: {
    const MemberPointerType *MemPtr = cast<MemberPointerType>(T.getTypePtr());
    MarkUsedTemplateParameters(Ctx, MemPtr->getPointeeType(), OnlyDeduced,
                               Depth, Used);
    MarkUsedTemplateParameters(Ctx,
                               QualType(MemPtr->getQualifier().getAsType(), 0),
                               OnlyDeduced, Depth, Used);
    break;
  }

  case Type::DependentSizedArray:
    MarkUsedTemplateParameters(Ctx,
                               cast<DependentSizedArrayType>(T)->getSizeExpr(),
                               OnlyDeduced, Depth, Used);
    // Fall through to check the element type
    [[fallthrough]];

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::ArrayParameter:
    MarkUsedTemplateParameters(Ctx,
                               cast<ArrayType>(T)->getElementType(),
                               OnlyDeduced, Depth, Used);
    break;
  case Type::Vector:
  case Type::ExtVector:
    MarkUsedTemplateParameters(Ctx,
                               cast<VectorType>(T)->getElementType(),
                               OnlyDeduced, Depth, Used);
    break;

  case Type::DependentVector: {
    const auto *VecType = cast<DependentVectorType>(T);
    MarkUsedTemplateParameters(Ctx, VecType->getElementType(), OnlyDeduced,
                               Depth, Used);
    MarkUsedTemplateParameters(Ctx, VecType->getSizeExpr(), OnlyDeduced, Depth,
                               Used);
    break;
  }
  case Type::DependentSizedExtVector: {
    const DependentSizedExtVectorType *VecType
      = cast<DependentSizedExtVectorType>(T);
    MarkUsedTemplateParameters(Ctx, VecType->getElementType(), OnlyDeduced,
                               Depth, Used);
    MarkUsedTemplateParameters(Ctx, VecType->getSizeExpr(), OnlyDeduced,
                               Depth, Used);
    break;
  }

  case Type::DependentAddressSpace: {
    const DependentAddressSpaceType *DependentASType =
        cast<DependentAddressSpaceType>(T);
    MarkUsedTemplateParameters(Ctx, DependentASType->getPointeeType(),
                               OnlyDeduced, Depth, Used);
    MarkUsedTemplateParameters(Ctx,
                               DependentASType->getAddrSpaceExpr(),
                               OnlyDeduced, Depth, Used);
    break;
  }

  case Type::ConstantMatrix: {
    const ConstantMatrixType *MatType = cast<ConstantMatrixType>(T);
    MarkUsedTemplateParameters(Ctx, MatType->getElementType(), OnlyDeduced,
                               Depth, Used);
    break;
  }

  case Type::DependentSizedMatrix: {
    const DependentSizedMatrixType *MatType = cast<DependentSizedMatrixType>(T);
    MarkUsedTemplateParameters(Ctx, MatType->getElementType(), OnlyDeduced,
                               Depth, Used);
    MarkUsedTemplateParameters(Ctx, MatType->getRowExpr(), OnlyDeduced, Depth,
                               Used);
    MarkUsedTemplateParameters(Ctx, MatType->getColumnExpr(), OnlyDeduced,
                               Depth, Used);
    break;
  }

  case Type::FunctionProto: {
    const FunctionProtoType *Proto = cast<FunctionProtoType>(T);
    MarkUsedTemplateParameters(Ctx, Proto->getReturnType(), OnlyDeduced, Depth,
                               Used);
    for (unsigned I = 0, N = Proto->getNumParams(); I != N; ++I) {
      // C++17 [temp.deduct.type]p5:
      //   The non-deduced contexts are: [...]
      //   -- A function parameter pack that does not occur at the end of the
      //      parameter-declaration-list.
      if (!OnlyDeduced || I + 1 == N ||
          !Proto->getParamType(I)->getAs<PackExpansionType>()) {
        MarkUsedTemplateParameters(Ctx, Proto->getParamType(I), OnlyDeduced,
                                   Depth, Used);
      } else {
        // FIXME: C++17 [temp.deduct.call]p1:
        //   When a function parameter pack appears in a non-deduced context,
        //   the type of that pack is never deduced.
        //
        // We should also track a set of "never deduced" parameters, and
        // subtract that from the list of deduced parameters after marking.
      }
    }
    if (auto *E = Proto->getNoexceptExpr())
      MarkUsedTemplateParameters(Ctx, E, OnlyDeduced, Depth, Used);
    break;
  }

  case Type::TemplateTypeParm: {
    const TemplateTypeParmType *TTP = cast<TemplateTypeParmType>(T);
    if (TTP->getDepth() == Depth)
      Used[TTP->getIndex()] = true;
    break;
  }

  case Type::SubstTemplateTypeParmPack: {
    const SubstTemplateTypeParmPackType *Subst
      = cast<SubstTemplateTypeParmPackType>(T);
    if (Subst->getReplacedParameter()->getDepth() == Depth)
      Used[Subst->getIndex()] = true;
    MarkUsedTemplateParameters(Ctx, Subst->getArgumentPack(),
                               OnlyDeduced, Depth, Used);
    break;
  }

  case Type::InjectedClassName:
    T = cast<InjectedClassNameType>(T)
            ->getOriginalDecl()
            ->getCanonicalTemplateSpecializationType(Ctx);
    [[fallthrough]];

  case Type::TemplateSpecialization: {
    const TemplateSpecializationType *Spec
      = cast<TemplateSpecializationType>(T);
    MarkUsedTemplateParameters(Ctx, Spec->getTemplateName(), OnlyDeduced,
                               Depth, Used);

    // C++0x [temp.deduct.type]p9:
    //   If the template argument list of P contains a pack expansion that is
    //   not the last template argument, the entire template argument list is a
    //   non-deduced context.
    if (OnlyDeduced &&
        hasPackExpansionBeforeEnd(Spec->template_arguments()))
      break;

    for (const auto &Arg : Spec->template_arguments())
      MarkUsedTemplateParameters(Ctx, Arg, OnlyDeduced, Depth, Used);
    break;
  }

  case Type::Complex:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx,
                                 cast<ComplexType>(T)->getElementType(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::Atomic:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx,
                                 cast<AtomicType>(T)->getValueType(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::DependentName:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx,
                                 cast<DependentNameType>(T)->getQualifier(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::DependentTemplateSpecialization: {
    // C++14 [temp.deduct.type]p5:
    //   The non-deduced contexts are:
    //     -- The nested-name-specifier of a type that was specified using a
    //        qualified-id
    //
    // C++14 [temp.deduct.type]p6:
    //   When a type name is specified in a way that includes a non-deduced
    //   context, all of the types that comprise that type name are also
    //   non-deduced.
    if (OnlyDeduced)
      break;

    const DependentTemplateSpecializationType *Spec
      = cast<DependentTemplateSpecializationType>(T);

    MarkUsedTemplateParameters(Ctx,
                               Spec->getDependentTemplateName().getQualifier(),
                               OnlyDeduced, Depth, Used);

    for (const auto &Arg : Spec->template_arguments())
      MarkUsedTemplateParameters(Ctx, Arg, OnlyDeduced, Depth, Used);
    break;
  }

  case Type::TypeOf:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx, cast<TypeOfType>(T)->getUnmodifiedType(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::TypeOfExpr:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx,
                                 cast<TypeOfExprType>(T)->getUnderlyingExpr(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::Decltype:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx,
                                 cast<DecltypeType>(T)->getUnderlyingExpr(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::PackIndexing:
    if (!OnlyDeduced) {
      MarkUsedTemplateParameters(Ctx, cast<PackIndexingType>(T)->getPattern(),
                                 OnlyDeduced, Depth, Used);
      MarkUsedTemplateParameters(Ctx, cast<PackIndexingType>(T)->getIndexExpr(),
                                 OnlyDeduced, Depth, Used);
    }
    break;

  case Type::UnaryTransform:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx,
                                 cast<UnaryTransformType>(T)->getUnderlyingType(),
                                 OnlyDeduced, Depth, Used);
    break;

  case Type::PackExpansion:
    MarkUsedTemplateParameters(Ctx,
                               cast<PackExpansionType>(T)->getPattern(),
                               OnlyDeduced, Depth, Used);
    break;

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    MarkUsedTemplateParameters(Ctx,
                               cast<DeducedType>(T)->getDeducedType(),
                               OnlyDeduced, Depth, Used);
    break;
  case Type::DependentBitInt:
    MarkUsedTemplateParameters(Ctx,
                               cast<DependentBitIntType>(T)->getNumBitsExpr(),
                               OnlyDeduced, Depth, Used);
    break;

  case Type::HLSLAttributedResource:
    MarkUsedTemplateParameters(
        Ctx, cast<HLSLAttributedResourceType>(T)->getWrappedType(), OnlyDeduced,
        Depth, Used);
    if (cast<HLSLAttributedResourceType>(T)->hasContainedType())
      MarkUsedTemplateParameters(
          Ctx, cast<HLSLAttributedResourceType>(T)->getContainedType(),
          OnlyDeduced, Depth, Used);
    break;

  // None of these types have any template parameters in them.
  case Type::Builtin:
  case Type::VariableArray:
  case Type::FunctionNoProto:
  case Type::Record:
  case Type::Enum:
  case Type::ObjCInterface:
  case Type::ObjCObject:
  case Type::ObjCObjectPointer:
  case Type::UnresolvedUsing:
  case Type::Pipe:
  case Type::BitInt:
  case Type::HLSLInlineSpirv:
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    break;
  }
}

/// Mark the template parameters that are used by this
/// template argument.
static void
MarkUsedTemplateParameters(ASTContext &Ctx,
                           const TemplateArgument &TemplateArg,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallBitVector &Used) {
  switch (TemplateArg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
  case TemplateArgument::Declaration:
  case TemplateArgument::NullPtr:
  case TemplateArgument::StructuralValue:
    break;

  case TemplateArgument::Type:
    MarkUsedTemplateParameters(Ctx, TemplateArg.getAsType(), OnlyDeduced,
                               Depth, Used);
    break;

  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    MarkUsedTemplateParameters(Ctx,
                               TemplateArg.getAsTemplateOrTemplatePattern(),
                               OnlyDeduced, Depth, Used);
    break;

  case TemplateArgument::Expression:
    MarkUsedTemplateParameters(Ctx, TemplateArg.getAsExpr(), OnlyDeduced,
                               Depth, Used);
    break;

  case TemplateArgument::Pack:
    for (const auto &P : TemplateArg.pack_elements())
      MarkUsedTemplateParameters(Ctx, P, OnlyDeduced, Depth, Used);
    break;
  }
}

void
Sema::MarkUsedTemplateParameters(const Expr *E, bool OnlyDeduced,
                                 unsigned Depth,
                                 llvm::SmallBitVector &Used) {
  ::MarkUsedTemplateParameters(Context, E, OnlyDeduced, Depth, Used);
}

void
Sema::MarkUsedTemplateParameters(const TemplateArgumentList &TemplateArgs,
                                 bool OnlyDeduced, unsigned Depth,
                                 llvm::SmallBitVector &Used) {
  // C++0x [temp.deduct.type]p9:
  //   If the template argument list of P contains a pack expansion that is not
  //   the last template argument, the entire template argument list is a
  //   non-deduced context.
  if (OnlyDeduced &&
      hasPackExpansionBeforeEnd(TemplateArgs.asArray()))
    return;

  for (unsigned I = 0, N = TemplateArgs.size(); I != N; ++I)
    ::MarkUsedTemplateParameters(Context, TemplateArgs[I], OnlyDeduced,
                                 Depth, Used);
}

void Sema::MarkUsedTemplateParameters(ArrayRef<TemplateArgument> TemplateArgs,
                                      unsigned Depth,
                                      llvm::SmallBitVector &Used) {
  for (unsigned I = 0, N = TemplateArgs.size(); I != N; ++I)
    ::MarkUsedTemplateParameters(Context, TemplateArgs[I],
                                 /*OnlyDeduced=*/false, Depth, Used);
}

void Sema::MarkDeducedTemplateParameters(
    ASTContext &Ctx, const FunctionTemplateDecl *FunctionTemplate,
    llvm::SmallBitVector &Deduced) {
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  Deduced.clear();
  Deduced.resize(TemplateParams->size());

  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  for (unsigned I = 0, N = Function->getNumParams(); I != N; ++I)
    ::MarkUsedTemplateParameters(Ctx, Function->getParamDecl(I)->getType(),
                                 true, TemplateParams->getDepth(), Deduced);
}

bool hasDeducibleTemplateParameters(Sema &S,
                                    FunctionTemplateDecl *FunctionTemplate,
                                    QualType T) {
  if (!T->isDependentType())
    return false;

  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  llvm::SmallBitVector Deduced(TemplateParams->size());
  ::MarkUsedTemplateParameters(S.Context, T, true, TemplateParams->getDepth(),
                               Deduced);

  return Deduced.any();
}
