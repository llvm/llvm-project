//===------- SemaTemplateDeduction.cpp - Template Argument Deduction ------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template argument deduction.
//
//===----------------------------------------------------------------------===/

#include "clang/Sema/TemplateDeduction.h"
#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/SmallBitVector.h"
#include <algorithm>

namespace clang {
  using namespace sema;
  /// \brief Various flags that control template argument deduction.
  ///
  /// These flags can be bitwise-OR'd together.
  enum TemplateDeductionFlags {
    /// \brief No template argument deduction flags, which indicates the
    /// strictest results for template argument deduction (as used for, e.g.,
    /// matching class template partial specializations).
    TDF_None = 0,
    /// \brief Within template argument deduction from a function call, we are
    /// matching with a parameter type for which the original parameter was
    /// a reference.
    TDF_ParamWithReferenceType = 0x1,
    /// \brief Within template argument deduction from a function call, we
    /// are matching in a case where we ignore cv-qualifiers.
    TDF_IgnoreQualifiers = 0x02,
    /// \brief Within template argument deduction from a function call,
    /// we are matching in a case where we can perform template argument
    /// deduction from a template-id of a derived class of the argument type.
    TDF_DerivedClass = 0x04,
    /// \brief Allow non-dependent types to differ, e.g., when performing
    /// template argument deduction from a function call where conversions
    /// may apply.
    TDF_SkipNonDependent = 0x08,
    /// \brief Whether we are performing template argument deduction for
    /// parameters and arguments in a top-level template argument
    TDF_TopLevelParameterTypeList = 0x10,
    /// \brief Within template argument deduction from overload resolution per
    /// C++ [over.over] allow matching function types that are compatible in
    /// terms of noreturn and default calling convention adjustments.
    TDF_InOverloadResolution = 0x20
  };
}

using namespace clang;

/// \brief Compare two APSInts, extending and switching the sign as
/// necessary to compare their values regardless of underlying type.
static bool hasSameExtendedValue(llvm::APSInt X, llvm::APSInt Y) {
  if (Y.getBitWidth() > X.getBitWidth())
    X = X.extend(Y.getBitWidth());
  else if (Y.getBitWidth() < X.getBitWidth())
    Y = Y.extend(X.getBitWidth());

  // If there is a signedness mismatch, correct it.
  if (X.isSigned() != Y.isSigned()) {
    // If the signed value is negative, then the values cannot be the same.
    if ((Y.isSigned() && Y.isNegative()) || (X.isSigned() && X.isNegative()))
      return false;

    Y.setIsSigned(true);
    X.setIsSigned(true);
  }

  return X == Y;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument &Param,
                        TemplateArgument Arg,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced);

static Sema::TemplateDeductionResult
DeduceTemplateArgumentsByTypeMatch(Sema &S,
                                   TemplateParameterList *TemplateParams,
                                   QualType Param,
                                   QualType Arg,
                                   TemplateDeductionInfo &Info,
                                   SmallVectorImpl<DeducedTemplateArgument> &
                                                      Deduced,
                                   unsigned TDF,
                                   bool PartialOrdering = false);

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument *Params, unsigned NumParams,
                        const TemplateArgument *Args, unsigned NumArgs,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced);

/// \brief If the given expression is of a form that permits the deduction
/// of a non-type template parameter, return the declaration of that
/// non-type template parameter.
static NonTypeTemplateParmDecl *getDeducedParameterFromExpr(Expr *E) {
  // If we are within an alias template, the expression may have undergone
  // any number of parameter substitutions already.
  while (1) {
    if (ImplicitCastExpr *IC = dyn_cast<ImplicitCastExpr>(E))
      E = IC->getSubExpr();
    else if (SubstNonTypeTemplateParmExpr *Subst =
               dyn_cast<SubstNonTypeTemplateParmExpr>(E))
      E = Subst->getReplacement();
    else
      break;
  }

  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl());

  return nullptr;
}

/// \brief Determine whether two declaration pointers refer to the same
/// declaration.
static bool isSameDeclaration(Decl *X, Decl *Y) {
  if (NamedDecl *NX = dyn_cast<NamedDecl>(X))
    X = NX->getUnderlyingDecl();
  if (NamedDecl *NY = dyn_cast<NamedDecl>(Y))
    Y = NY->getUnderlyingDecl();

  return X->getCanonicalDecl() == Y->getCanonicalDecl();
}

/// \brief Verify that the given, deduced template arguments are compatible.
///
/// \returns The deduced template argument, or a NULL template argument if
/// the deduced template arguments were incompatible.
static DeducedTemplateArgument
checkDeducedTemplateArguments(ASTContext &Context,
                              const DeducedTemplateArgument &X,
                              const DeducedTemplateArgument &Y) {
  // We have no deduction for one or both of the arguments; they're compatible.
  if (X.isNull())
    return Y;
  if (Y.isNull())
    return X;

  switch (X.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Non-deduced template arguments handled above");

  case TemplateArgument::Type:
    // If two template type arguments have the same type, they're compatible.
    if (Y.getKind() == TemplateArgument::Type &&
        Context.hasSameType(X.getAsType(), Y.getAsType()))
      return X;

    return DeducedTemplateArgument();

  case TemplateArgument::Integral:
    // If we deduced a constant in one case and either a dependent expression or
    // declaration in another case, keep the integral constant.
    // If both are integral constants with the same value, keep that value.
    if (Y.getKind() == TemplateArgument::Expression ||
        Y.getKind() == TemplateArgument::Declaration ||
        (Y.getKind() == TemplateArgument::Integral &&
         hasSameExtendedValue(X.getAsIntegral(), Y.getAsIntegral())))
      return DeducedTemplateArgument(X,
                                     X.wasDeducedFromArrayBound() &&
                                     Y.wasDeducedFromArrayBound());

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

  case TemplateArgument::Expression:
    // If we deduced a dependent expression in one case and either an integral
    // constant or a declaration in another case, keep the integral constant
    // or declaration.
    if (Y.getKind() == TemplateArgument::Integral ||
        Y.getKind() == TemplateArgument::Declaration)
      return DeducedTemplateArgument(Y, X.wasDeducedFromArrayBound() &&
                                     Y.wasDeducedFromArrayBound());

    if (Y.getKind() == TemplateArgument::Expression) {
      // Compare the expressions for equality
      llvm::FoldingSetNodeID ID1, ID2;
      X.getAsExpr()->Profile(ID1, Context, true);
      Y.getAsExpr()->Profile(ID2, Context, true);
      if (ID1 == ID2)
        return X;
    }

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::Declaration:
    // If we deduced a declaration and a dependent expression, keep the
    // declaration.
    if (Y.getKind() == TemplateArgument::Expression)
      return X;

    // If we deduced a declaration and an integral constant, keep the
    // integral constant.
    if (Y.getKind() == TemplateArgument::Integral)
      return Y;

    // If we deduced two declarations, make sure they they refer to the
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
      return X;

    // If we deduced a null pointer and an integral constant, keep the
    // integral constant.
    if (Y.getKind() == TemplateArgument::Integral)
      return Y;

    // If we deduced two null pointers, make sure they have the same type.
    if (Y.getKind() == TemplateArgument::NullPtr &&
        Context.hasSameType(X.getNullPtrType(), Y.getNullPtrType()))
      return X;

    // All other combinations are incompatible.
    return DeducedTemplateArgument();

  case TemplateArgument::Pack:
    if (Y.getKind() != TemplateArgument::Pack ||
        X.pack_size() != Y.pack_size())
      return DeducedTemplateArgument();

    for (TemplateArgument::pack_iterator XA = X.pack_begin(),
                                      XAEnd = X.pack_end(),
                                         YA = Y.pack_begin();
         XA != XAEnd; ++XA, ++YA) {
      // FIXME: Do we need to merge the results together here?
      if (checkDeducedTemplateArguments(Context,
                    DeducedTemplateArgument(*XA, X.wasDeducedFromArrayBound()),
                    DeducedTemplateArgument(*YA, Y.wasDeducedFromArrayBound()))
            .isNull())
        return DeducedTemplateArgument();
    }

    return X;
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

/// \brief Deduce the value of the given non-type template parameter
/// from the given constant.
static Sema::TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S,
                              NonTypeTemplateParmDecl *NTTP,
                              llvm::APSInt Value, QualType ValueType,
                              bool DeducedFromArrayBound,
                              TemplateDeductionInfo &Info,
                    SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");

  DeducedTemplateArgument NewDeduced(S.Context, Value, ValueType,
                                     DeducedFromArrayBound);
  DeducedTemplateArgument Result = checkDeducedTemplateArguments(S.Context,
                                                     Deduced[NTTP->getIndex()],
                                                                 NewDeduced);
  if (Result.isNull()) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = NewDeduced;
    return Sema::TDK_Inconsistent;
  }

  Deduced[NTTP->getIndex()] = Result;
  return Sema::TDK_Success;
}

/// \brief Deduce the value of the given non-type template parameter
/// from the given type- or value-dependent expression.
///
/// \returns true if deduction succeeded, false otherwise.
static Sema::TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S,
                              NonTypeTemplateParmDecl *NTTP,
                              Expr *Value,
                              TemplateDeductionInfo &Info,
                    SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");
  assert((Value->isTypeDependent() || Value->isValueDependent()) &&
         "Expression template argument must be type- or value-dependent.");

  DeducedTemplateArgument NewDeduced(Value);
  DeducedTemplateArgument Result = checkDeducedTemplateArguments(S.Context,
                                                     Deduced[NTTP->getIndex()],
                                                                 NewDeduced);

  if (Result.isNull()) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = NewDeduced;
    return Sema::TDK_Inconsistent;
  }

  Deduced[NTTP->getIndex()] = Result;
  return Sema::TDK_Success;
}

/// \brief Deduce the value of the given non-type template parameter
/// from the given declaration.
///
/// \returns true if deduction succeeded, false otherwise.
static Sema::TemplateDeductionResult
DeduceNonTypeTemplateArgument(Sema &S,
                            NonTypeTemplateParmDecl *NTTP,
                            ValueDecl *D,
                            TemplateDeductionInfo &Info,
                            SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(NTTP->getDepth() == 0 &&
         "Cannot deduce non-type template argument with depth > 0");

  D = D ? cast<ValueDecl>(D->getCanonicalDecl()) : nullptr;
  TemplateArgument New(D, NTTP->getType());
  DeducedTemplateArgument NewDeduced(New);
  DeducedTemplateArgument Result = checkDeducedTemplateArguments(S.Context,
                                                     Deduced[NTTP->getIndex()],
                                                                 NewDeduced);
  if (Result.isNull()) {
    Info.Param = NTTP;
    Info.FirstArg = Deduced[NTTP->getIndex()];
    Info.SecondArg = NewDeduced;
    return Sema::TDK_Inconsistent;
  }

  Deduced[NTTP->getIndex()] = Result;
  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        TemplateName Param,
                        TemplateName Arg,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  TemplateDecl *ParamDecl = Param.getAsTemplateDecl();
  if (!ParamDecl) {
    // The parameter type is dependent and is not a template template parameter,
    // so there is nothing that we can deduce.
    return Sema::TDK_Success;
  }

  if (TemplateTemplateParmDecl *TempParam
        = dyn_cast<TemplateTemplateParmDecl>(ParamDecl)) {
    DeducedTemplateArgument NewDeduced(S.Context.getCanonicalTemplateName(Arg));
    DeducedTemplateArgument Result = checkDeducedTemplateArguments(S.Context,
                                                 Deduced[TempParam->getIndex()],
                                                                   NewDeduced);
    if (Result.isNull()) {
      Info.Param = TempParam;
      Info.FirstArg = Deduced[TempParam->getIndex()];
      Info.SecondArg = NewDeduced;
      return Sema::TDK_Inconsistent;
    }

    Deduced[TempParam->getIndex()] = Result;
    return Sema::TDK_Success;
  }

  // Verify that the two template names are equivalent.
  if (S.Context.hasSameTemplateName(Param, Arg))
    return Sema::TDK_Success;

  // Mismatch of non-dependent template parameter to argument.
  Info.FirstArg = TemplateArgument(Param);
  Info.SecondArg = TemplateArgument(Arg);
  return Sema::TDK_NonDeducedMismatch;
}

/// \brief Deduce the template arguments by comparing the template parameter
/// type (which is a template-id) with the template argument type.
///
/// \param S the Sema
///
/// \param TemplateParams the template parameters that we are deducing
///
/// \param Param the parameter type
///
/// \param Arg the argument type
///
/// \param Info information about the template argument deduction itself
///
/// \param Deduced the deduced template arguments
///
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.
static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateSpecializationType *Param,
                        QualType Arg,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  assert(Arg.isCanonical() && "Argument type must be canonical");

  // Check whether the template argument is a dependent template-id.
  if (const TemplateSpecializationType *SpecArg
        = dyn_cast<TemplateSpecializationType>(Arg)) {
    // Perform template argument deduction for the template name.
    if (Sema::TemplateDeductionResult Result
          = DeduceTemplateArguments(S, TemplateParams,
                                    Param->getTemplateName(),
                                    SpecArg->getTemplateName(),
                                    Info, Deduced))
      return Result;


    // Perform template argument deduction on each template
    // argument. Ignore any missing/extra arguments, since they could be
    // filled in by default arguments.
    return DeduceTemplateArguments(S, TemplateParams,
                                   Param->getArgs(), Param->getNumArgs(),
                                   SpecArg->getArgs(), SpecArg->getNumArgs(),
                                   Info, Deduced);
  }

  // If the argument type is a class template specialization, we
  // perform template argument deduction using its template
  // arguments.
  const RecordType *RecordArg = dyn_cast<RecordType>(Arg);
  if (!RecordArg) {
    Info.FirstArg = TemplateArgument(QualType(Param, 0));
    Info.SecondArg = TemplateArgument(Arg);
    return Sema::TDK_NonDeducedMismatch;
  }

  ClassTemplateSpecializationDecl *SpecArg
    = dyn_cast<ClassTemplateSpecializationDecl>(RecordArg->getDecl());
  if (!SpecArg) {
    Info.FirstArg = TemplateArgument(QualType(Param, 0));
    Info.SecondArg = TemplateArgument(Arg);
    return Sema::TDK_NonDeducedMismatch;
  }

  // Perform template argument deduction for the template name.
  if (Sema::TemplateDeductionResult Result
        = DeduceTemplateArguments(S,
                                  TemplateParams,
                                  Param->getTemplateName(),
                               TemplateName(SpecArg->getSpecializedTemplate()),
                                  Info, Deduced))
    return Result;

  // Perform template argument deduction for the template arguments.
  return DeduceTemplateArguments(S, TemplateParams,
                                 Param->getArgs(), Param->getNumArgs(),
                                 SpecArg->getTemplateArgs().data(),
                                 SpecArg->getTemplateArgs().size(),
                                 Info, Deduced);
}

/// \brief Determines whether the given type is an opaque type that
/// might be more qualified when instantiated.
static bool IsPossiblyOpaquelyQualifiedType(QualType T) {
  switch (T->getTypeClass()) {
  case Type::TypeOfExpr:
  case Type::TypeOf:
  case Type::DependentName:
  case Type::Decltype:
  case Type::UnresolvedUsing:
  case Type::TemplateTypeParm:
    return true;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
  case Type::DependentSizedArray:
    return IsPossiblyOpaquelyQualifiedType(
                                      cast<ArrayType>(T)->getElementType());

  default:
    return false;
  }
}

/// \brief Retrieve the depth and index of a template parameter.
static std::pair<unsigned, unsigned>
getDepthAndIndex(NamedDecl *ND) {
  if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(ND))
    return std::make_pair(TTP->getDepth(), TTP->getIndex());

  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(ND))
    return std::make_pair(NTTP->getDepth(), NTTP->getIndex());

  TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(ND);
  return std::make_pair(TTP->getDepth(), TTP->getIndex());
}

/// \brief Retrieve the depth and index of an unexpanded parameter pack.
static std::pair<unsigned, unsigned>
getDepthAndIndex(UnexpandedParameterPack UPP) {
  if (const TemplateTypeParmType *TTP
                          = UPP.first.dyn_cast<const TemplateTypeParmType *>())
    return std::make_pair(TTP->getDepth(), TTP->getIndex());

  return getDepthAndIndex(UPP.first.get<NamedDecl *>());
}

/// \brief Helper function to build a TemplateParameter when we don't
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
  DeducedPack(unsigned Index) : Index(Index), Outer(nullptr) {}

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
  DeducedPack *Outer;
};

namespace {
/// A scope in which we're performing pack deduction.
class PackDeductionScope {
public:
  PackDeductionScope(Sema &S, TemplateParameterList *TemplateParams,
                     SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                     TemplateDeductionInfo &Info, TemplateArgument Pattern)
      : S(S), TemplateParams(TemplateParams), Deduced(Deduced), Info(Info) {
    // Compute the set of template parameter indices that correspond to
    // parameter packs expanded by the pack expansion.
    {
      llvm::SmallBitVector SawIndices(TemplateParams->size());
      SmallVector<UnexpandedParameterPack, 2> Unexpanded;
      S.collectUnexpandedParameterPacks(Pattern, Unexpanded);
      for (unsigned I = 0, N = Unexpanded.size(); I != N; ++I) {
        unsigned Depth, Index;
        std::tie(Depth, Index) = getDepthAndIndex(Unexpanded[I]);
        if (Depth == 0 && !SawIndices[Index]) {
          SawIndices[Index] = true;

          // Save the deduced template argument for the parameter pack expanded
          // by this pack expansion, then clear out the deduction.
          DeducedPack Pack(Index);
          Pack.Saved = Deduced[Index];
          Deduced[Index] = TemplateArgument();

          Packs.push_back(Pack);
        }
      }
    }
    assert(!Packs.empty() && "Pack expansion without unexpanded packs?");

    for (auto &Pack : Packs) {
      if (Info.PendingDeducedPacks.size() > Pack.Index)
        Pack.Outer = Info.PendingDeducedPacks[Pack.Index];
      else
        Info.PendingDeducedPacks.resize(Pack.Index + 1);
      Info.PendingDeducedPacks[Pack.Index] = &Pack;

      if (S.CurrentInstantiationScope) {
        // If the template argument pack was explicitly specified, add that to
        // the set of deduced arguments.
        const TemplateArgument *ExplicitArgs;
        unsigned NumExplicitArgs;
        NamedDecl *PartiallySubstitutedPack =
            S.CurrentInstantiationScope->getPartiallySubstitutedPack(
                &ExplicitArgs, &NumExplicitArgs);
        if (PartiallySubstitutedPack &&
            getDepthAndIndex(PartiallySubstitutedPack).second == Pack.Index)
          Pack.New.append(ExplicitArgs, ExplicitArgs + NumExplicitArgs);
      }
    }
  }

  ~PackDeductionScope() {
    for (auto &Pack : Packs)
      Info.PendingDeducedPacks[Pack.Index] = Pack.Outer;
  }

  /// Move to deducing the next element in each pack that is being deduced.
  void nextPackElement() {
    // Capture the deduced template arguments for each parameter pack expanded
    // by this pack expansion, add them to the list of arguments we've deduced
    // for that pack, then clear out the deduced argument.
    for (auto &Pack : Packs) {
      DeducedTemplateArgument &DeducedArg = Deduced[Pack.Index];
      if (!DeducedArg.isNull()) {
        Pack.New.push_back(DeducedArg);
        DeducedArg = DeducedTemplateArgument();
      }
    }
  }

  /// \brief Finish template argument deduction for a set of argument packs,
  /// producing the argument packs and checking for consistency with prior
  /// deductions.
  Sema::TemplateDeductionResult finish(bool HasAnyArguments) {
    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    for (auto &Pack : Packs) {
      // Put back the old value for this pack.
      Deduced[Pack.Index] = Pack.Saved;

      // Build or find a new value for this pack.
      DeducedTemplateArgument NewPack;
      if (HasAnyArguments && Pack.New.empty()) {
        if (Pack.DeferredDeduction.isNull()) {
          // We were not able to deduce anything for this parameter pack
          // (because it only appeared in non-deduced contexts), so just
          // restore the saved argument pack.
          continue;
        }

        NewPack = Pack.DeferredDeduction;
        Pack.DeferredDeduction = TemplateArgument();
      } else if (Pack.New.empty()) {
        // If we deduced an empty argument pack, create it now.
        NewPack = DeducedTemplateArgument(TemplateArgument::getEmptyPack());
      } else {
        TemplateArgument *ArgumentPack =
            new (S.Context) TemplateArgument[Pack.New.size()];
        std::copy(Pack.New.begin(), Pack.New.end(), ArgumentPack);
        NewPack = DeducedTemplateArgument(
            TemplateArgument(llvm::makeArrayRef(ArgumentPack, Pack.New.size())),
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
      DeducedTemplateArgument Result =
          checkDeducedTemplateArguments(S.Context, OldPack, NewPack);

      // If we deferred a deduction of this pack, check that one now too.
      if (!Result.isNull() && !Pack.DeferredDeduction.isNull()) {
        OldPack = Result;
        NewPack = Pack.DeferredDeduction;
        Result = checkDeducedTemplateArguments(S.Context, OldPack, NewPack);
      }

      if (Result.isNull()) {
        Info.Param =
            makeTemplateParameter(TemplateParams->getParam(Pack.Index));
        Info.FirstArg = OldPack;
        Info.SecondArg = NewPack;
        return Sema::TDK_Inconsistent;
      }

      *Loc = Result;
    }

    return Sema::TDK_Success;
  }

private:
  Sema &S;
  TemplateParameterList *TemplateParams;
  SmallVectorImpl<DeducedTemplateArgument> &Deduced;
  TemplateDeductionInfo &Info;

  SmallVector<DeducedPack, 2> Packs;
};
} // namespace

/// \brief Deduce the template arguments by comparing the list of parameter
/// types to the list of argument types, as in the parameter-type-lists of
/// function types (C++ [temp.deduct.type]p10).
///
/// \param S The semantic analysis object within which we are deducing
///
/// \param TemplateParams The template parameters that we are deducing
///
/// \param Params The list of parameter types
///
/// \param NumParams The number of types in \c Params
///
/// \param Args The list of argument types
///
/// \param NumArgs The number of types in \c Args
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
/// \returns the result of template argument deduction so far. Note that a
/// "success" result means that template argument deduction has not yet failed,
/// but it may still fail, later, for other reasons.
static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const QualType *Params, unsigned NumParams,
                        const QualType *Args, unsigned NumArgs,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                        unsigned TDF,
                        bool PartialOrdering = false) {
  // Fast-path check to see if we have too many/too few arguments.
  if (NumParams != NumArgs &&
      !(NumParams && isa<PackExpansionType>(Params[NumParams - 1])) &&
      !(NumArgs && isa<PackExpansionType>(Args[NumArgs - 1])))
    return Sema::TDK_MiscellaneousDeductionFailure;

  // C++0x [temp.deduct.type]p10:
  //   Similarly, if P has a form that contains (T), then each parameter type
  //   Pi of the respective parameter-type- list of P is compared with the
  //   corresponding parameter type Ai of the corresponding parameter-type-list
  //   of A. [...]
  unsigned ArgIdx = 0, ParamIdx = 0;
  for (; ParamIdx != NumParams; ++ParamIdx) {
    // Check argument types.
    const PackExpansionType *Expansion
                                = dyn_cast<PackExpansionType>(Params[ParamIdx]);
    if (!Expansion) {
      // Simple case: compare the parameter and argument types at this point.

      // Make sure we have an argument.
      if (ArgIdx >= NumArgs)
        return Sema::TDK_MiscellaneousDeductionFailure;

      if (isa<PackExpansionType>(Args[ArgIdx])) {
        // C++0x [temp.deduct.type]p22:
        //   If the original function parameter associated with A is a function
        //   parameter pack and the function parameter associated with P is not
        //   a function parameter pack, then template argument deduction fails.
        return Sema::TDK_MiscellaneousDeductionFailure;
      }

      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                 Params[ParamIdx], Args[ArgIdx],
                                                 Info, Deduced, TDF,
                                                 PartialOrdering))
        return Result;

      ++ArgIdx;
      continue;
    }

    // C++0x [temp.deduct.type]p5:
    //   The non-deduced contexts are:
    //     - A function parameter pack that does not occur at the end of the
    //       parameter-declaration-clause.
    if (ParamIdx + 1 < NumParams)
      return Sema::TDK_Success;

    // C++0x [temp.deduct.type]p10:
    //   If the parameter-declaration corresponding to Pi is a function
    //   parameter pack, then the type of its declarator- id is compared with
    //   each remaining parameter type in the parameter-type-list of A. Each
    //   comparison deduces template arguments for subsequent positions in the
    //   template parameter packs expanded by the function parameter pack.

    QualType Pattern = Expansion->getPattern();
    PackDeductionScope PackScope(S, TemplateParams, Deduced, Info, Pattern);

    bool HasAnyArguments = false;
    for (; ArgIdx < NumArgs; ++ArgIdx) {
      HasAnyArguments = true;

      // Deduce template arguments from the pattern.
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArgumentsByTypeMatch(S, TemplateParams, Pattern,
                                                 Args[ArgIdx], Info, Deduced,
                                                 TDF, PartialOrdering))
        return Result;

      PackScope.nextPackElement();
    }

    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    if (auto Result = PackScope.finish(HasAnyArguments))
      return Result;
  }

  // Make sure we don't have any extra arguments.
  if (ArgIdx < NumArgs)
    return Sema::TDK_MiscellaneousDeductionFailure;

  return Sema::TDK_Success;
}

/// \brief Determine whether the parameter has qualifiers that are either
/// inconsistent with or a superset of the argument's qualifiers.
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
  
  // CVR qualifier superset.
  return (ParamQs.getCVRQualifiers() != ArgQs.getCVRQualifiers()) &&
      ((ParamQs.getCVRQualifiers() | ArgQs.getCVRQualifiers())
                                                == ParamQs.getCVRQualifiers());
}

/// \brief Compare types for equality with respect to possibly compatible
/// function types (noreturn adjustment, implicit calling conventions). If any
/// of parameter and argument is not a function, just perform type comparison.
///
/// \param Param the template parameter type.
///
/// \param Arg the argument type.
bool Sema::isSameOrCompatibleFunctionType(CanQualType Param,
                                          CanQualType Arg) {
  const FunctionType *ParamFunction = Param->getAs<FunctionType>(),
                     *ArgFunction   = Arg->getAs<FunctionType>();

  // Just compare if not functions.
  if (!ParamFunction || !ArgFunction)
    return Param == Arg;

  // Noreturn adjustment.
  QualType AdjustedParam;
  if (IsNoReturnConversion(Param, Arg, AdjustedParam))
    return Arg == Context.getCanonicalType(AdjustedParam);

  // FIXME: Compatible calling conventions.

  return Param == Arg;
}

/// \brief Deduce the template arguments by comparing the parameter type and
/// the argument type (C++ [temp.deduct.type]).
///
/// \param S the semantic analysis object within which we are deducing
///
/// \param TemplateParams the template parameters that we are deducing
///
/// \param ParamIn the parameter type
///
/// \param ArgIn the argument type
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
static Sema::TemplateDeductionResult
DeduceTemplateArgumentsByTypeMatch(Sema &S,
                                   TemplateParameterList *TemplateParams,
                                   QualType ParamIn, QualType ArgIn,
                                   TemplateDeductionInfo &Info,
                            SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                   unsigned TDF,
                                   bool PartialOrdering) {
  // We only want to look at the canonical types, since typedefs and
  // sugar are not part of template argument deduction.
  QualType Param = S.Context.getCanonicalType(ParamIn);
  QualType Arg = S.Context.getCanonicalType(ArgIn);

  // If the argument type is a pack expansion, look at its pattern.
  // This isn't explicitly called out
  if (const PackExpansionType *ArgExpansion
                                            = dyn_cast<PackExpansionType>(Arg))
    Arg = ArgExpansion->getPattern();

  if (PartialOrdering) {
    // C++11 [temp.deduct.partial]p5:
    //   Before the partial ordering is done, certain transformations are
    //   performed on the types used for partial ordering:
    //     - If P is a reference type, P is replaced by the type referred to.
    const ReferenceType *ParamRef = Param->getAs<ReferenceType>();
    if (ParamRef)
      Param = ParamRef->getPointeeType();

    //     - If A is a reference type, A is replaced by the type referred to.
    const ReferenceType *ArgRef = Arg->getAs<ReferenceType>();
    if (ArgRef)
      Arg = ArgRef->getPointeeType();

    if (ParamRef && ArgRef && S.Context.hasSameUnqualifiedType(Param, Arg)) {
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
      Qualifiers ParamQuals = Param.getQualifiers();
      Qualifiers ArgQuals = Arg.getQualifiers();
      if ((ParamRef->isLValueReferenceType() &&
           !ArgRef->isLValueReferenceType()) ||
          ParamQuals.isStrictSupersetOf(ArgQuals) ||
          (ParamQuals.hasNonTrivialObjCLifetime() &&
           ArgQuals.getObjCLifetime() == Qualifiers::OCL_ExplicitNone &&
           ParamQuals.withoutObjCLifetime() ==
               ArgQuals.withoutObjCLifetime())) {
        Info.FirstArg = TemplateArgument(ParamIn);
        Info.SecondArg = TemplateArgument(ArgIn);
        return Sema::TDK_NonDeducedMismatch;
      }
    }

    // C++11 [temp.deduct.partial]p7:
    //   Remove any top-level cv-qualifiers:
    //     - If P is a cv-qualified type, P is replaced by the cv-unqualified
    //       version of P.
    Param = Param.getUnqualifiedType();
    //     - If A is a cv-qualified type, A is replaced by the cv-unqualified
    //       version of A.
    Arg = Arg.getUnqualifiedType();
  } else {
    // C++0x [temp.deduct.call]p4 bullet 1:
    //   - If the original P is a reference type, the deduced A (i.e., the type
    //     referred to by the reference) can be more cv-qualified than the
    //     transformed A.
    if (TDF & TDF_ParamWithReferenceType) {
      Qualifiers Quals;
      QualType UnqualParam = S.Context.getUnqualifiedArrayType(Param, Quals);
      Quals.setCVRQualifiers(Quals.getCVRQualifiers() &
                             Arg.getCVRQualifiers());
      Param = S.Context.getQualifiedType(UnqualParam, Quals);
    }

    if ((TDF & TDF_TopLevelParameterTypeList) && !Param->isFunctionType()) {
      // C++0x [temp.deduct.type]p10:
      //   If P and A are function types that originated from deduction when
      //   taking the address of a function template (14.8.2.2) or when deducing
      //   template arguments from a function declaration (14.8.2.6) and Pi and
      //   Ai are parameters of the top-level parameter-type-list of P and A,
      //   respectively, Pi is adjusted if it is an rvalue reference to a
      //   cv-unqualified template parameter and Ai is an lvalue reference, in
      //   which case the type of Pi is changed to be the template parameter
      //   type (i.e., T&& is changed to simply T). [ Note: As a result, when
      //   Pi is T&& and Ai is X&, the adjusted Pi will be T, causing T to be
      //   deduced as X&. - end note ]
      TDF &= ~TDF_TopLevelParameterTypeList;

      if (const RValueReferenceType *ParamRef
                                        = Param->getAs<RValueReferenceType>()) {
        if (isa<TemplateTypeParmType>(ParamRef->getPointeeType()) &&
            !ParamRef->getPointeeType().getQualifiers())
          if (Arg->isLValueReferenceType())
            Param = ParamRef->getPointeeType();
      }
    }
  }

  // C++ [temp.deduct.type]p9:
  //   A template type argument T, a template template argument TT or a
  //   template non-type argument i can be deduced if P and A have one of
  //   the following forms:
  //
  //     T
  //     cv-list T
  if (const TemplateTypeParmType *TemplateTypeParm
        = Param->getAs<TemplateTypeParmType>()) {
    // Just skip any attempts to deduce from a placeholder type.
    if (Arg->isPlaceholderType())
      return Sema::TDK_Success;
    
    unsigned Index = TemplateTypeParm->getIndex();
    bool RecanonicalizeArg = false;

    // If the argument type is an array type, move the qualifiers up to the
    // top level, so they can be matched with the qualifiers on the parameter.
    if (isa<ArrayType>(Arg)) {
      Qualifiers Quals;
      Arg = S.Context.getUnqualifiedArrayType(Arg, Quals);
      if (Quals) {
        Arg = S.Context.getQualifiedType(Arg, Quals);
        RecanonicalizeArg = true;
      }
    }

    // The argument type can not be less qualified than the parameter
    // type.
    if (!(TDF & TDF_IgnoreQualifiers) &&
        hasInconsistentOrSupersetQualifiersOf(Param, Arg)) {
      Info.Param = cast<TemplateTypeParmDecl>(TemplateParams->getParam(Index));
      Info.FirstArg = TemplateArgument(Param);
      Info.SecondArg = TemplateArgument(Arg);
      return Sema::TDK_Underqualified;
    }

    assert(TemplateTypeParm->getDepth() == 0 && "Can't deduce with depth > 0");
    assert(Arg != S.Context.OverloadTy && "Unresolved overloaded function");
    QualType DeducedType = Arg;

    // Remove any qualifiers on the parameter from the deduced type.
    // We checked the qualifiers for consistency above.
    Qualifiers DeducedQs = DeducedType.getQualifiers();
    Qualifiers ParamQs = Param.getQualifiers();
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
      Info.FirstArg = TemplateArgument(Param);
      Info.SecondArg = TemplateArgument(Arg);
      return Sema::TDK_Underqualified;      
    }
    
    // Objective-C ARC:
    //   If template deduction would produce an argument type with lifetime type
    //   but no lifetime qualifier, the __strong lifetime qualifier is inferred.
    if (S.getLangOpts().ObjCAutoRefCount &&
        DeducedType->isObjCLifetimeType() &&
        !DeducedQs.hasObjCLifetime())
      DeducedQs.setObjCLifetime(Qualifiers::OCL_Strong);
    
    DeducedType = S.Context.getQualifiedType(DeducedType.getUnqualifiedType(),
                                             DeducedQs);
    
    if (RecanonicalizeArg)
      DeducedType = S.Context.getCanonicalType(DeducedType);

    DeducedTemplateArgument NewDeduced(DeducedType);
    DeducedTemplateArgument Result = checkDeducedTemplateArguments(S.Context,
                                                                 Deduced[Index],
                                                                   NewDeduced);
    if (Result.isNull()) {
      Info.Param = cast<TemplateTypeParmDecl>(TemplateParams->getParam(Index));
      Info.FirstArg = Deduced[Index];
      Info.SecondArg = NewDeduced;
      return Sema::TDK_Inconsistent;
    }

    Deduced[Index] = Result;
    return Sema::TDK_Success;
  }

  // Set up the template argument deduction information for a failure.
  Info.FirstArg = TemplateArgument(ParamIn);
  Info.SecondArg = TemplateArgument(ArgIn);

  // If the parameter is an already-substituted template parameter
  // pack, do nothing: we don't know which of its arguments to look
  // at, so we have to wait until all of the parameter packs in this
  // expansion have arguments.
  if (isa<SubstTemplateTypeParmPackType>(Param))
    return Sema::TDK_Success;

  // Check the cv-qualifiers on the parameter and argument types.
  CanQualType CanParam = S.Context.getCanonicalType(Param);
  CanQualType CanArg = S.Context.getCanonicalType(Arg);
  if (!(TDF & TDF_IgnoreQualifiers)) {
    if (TDF & TDF_ParamWithReferenceType) {
      if (hasInconsistentOrSupersetQualifiersOf(Param, Arg))
        return Sema::TDK_NonDeducedMismatch;
    } else if (!IsPossiblyOpaquelyQualifiedType(Param)) {
      if (Param.getCVRQualifiers() != Arg.getCVRQualifiers())
        return Sema::TDK_NonDeducedMismatch;
    }
    
    // If the parameter type is not dependent, there is nothing to deduce.
    if (!Param->isDependentType()) {
      if (!(TDF & TDF_SkipNonDependent)) {
        bool NonDeduced = (TDF & TDF_InOverloadResolution)?
                          !S.isSameOrCompatibleFunctionType(CanParam, CanArg) :
                          Param != Arg;
        if (NonDeduced) {
          return Sema::TDK_NonDeducedMismatch;
        }
      }
      return Sema::TDK_Success;
    }
  } else if (!Param->isDependentType()) {
    CanQualType ParamUnqualType = CanParam.getUnqualifiedType(),
                ArgUnqualType = CanArg.getUnqualifiedType();
    bool Success = (TDF & TDF_InOverloadResolution)?
                   S.isSameOrCompatibleFunctionType(ParamUnqualType,
                                                    ArgUnqualType) :
                   ParamUnqualType == ArgUnqualType;
    if (Success)
      return Sema::TDK_Success;
  }

  switch (Param->getTypeClass()) {
    // Non-canonical types cannot appear here.
#define NON_CANONICAL_TYPE(Class, Base) \
  case Type::Class: llvm_unreachable("deducing non-canonical type: " #Class);
#define TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
      
    case Type::TemplateTypeParm:
    case Type::SubstTemplateTypeParmPack:
      llvm_unreachable("Type nodes handled above");

    // These types cannot be dependent, so simply check whether the types are
    // the same.
    case Type::Builtin:
    case Type::VariableArray:
    case Type::Vector:
    case Type::FunctionNoProto:
    case Type::Record:
    case Type::Enum:
    case Type::ObjCObject:
    case Type::ObjCInterface:
    case Type::ObjCObjectPointer: {
      if (TDF & TDF_SkipNonDependent)
        return Sema::TDK_Success;
      
      if (TDF & TDF_IgnoreQualifiers) {
        Param = Param.getUnqualifiedType();
        Arg = Arg.getUnqualifiedType();
      }
            
      return Param == Arg? Sema::TDK_Success : Sema::TDK_NonDeducedMismatch;
    }
      
    //     _Complex T   [placeholder extension]  
    case Type::Complex:
      if (const ComplexType *ComplexArg = Arg->getAs<ComplexType>())
        return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams, 
                                    cast<ComplexType>(Param)->getElementType(), 
                                    ComplexArg->getElementType(),
                                    Info, Deduced, TDF);

      return Sema::TDK_NonDeducedMismatch;

    //     _Atomic T   [extension]
    case Type::Atomic:
      if (const AtomicType *AtomicArg = Arg->getAs<AtomicType>())
        return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                       cast<AtomicType>(Param)->getValueType(),
                                       AtomicArg->getValueType(),
                                       Info, Deduced, TDF);

      return Sema::TDK_NonDeducedMismatch;

    //     T *
    case Type::Pointer: {
      QualType PointeeType;
      if (const PointerType *PointerArg = Arg->getAs<PointerType>()) {
        PointeeType = PointerArg->getPointeeType();
      } else if (const ObjCObjectPointerType *PointerArg
                   = Arg->getAs<ObjCObjectPointerType>()) {
        PointeeType = PointerArg->getPointeeType();
      } else {
        return Sema::TDK_NonDeducedMismatch;
      }

      unsigned SubTDF = TDF & (TDF_IgnoreQualifiers | TDF_DerivedClass);
      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                     cast<PointerType>(Param)->getPointeeType(),
                                     PointeeType,
                                     Info, Deduced, SubTDF);
    }

    //     T &
    case Type::LValueReference: {
      const LValueReferenceType *ReferenceArg =
          Arg->getAs<LValueReferenceType>();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                           cast<LValueReferenceType>(Param)->getPointeeType(),
                           ReferenceArg->getPointeeType(), Info, Deduced, 0);
    }

    //     T && [C++0x]
    case Type::RValueReference: {
      const RValueReferenceType *ReferenceArg =
          Arg->getAs<RValueReferenceType>();
      if (!ReferenceArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                             cast<RValueReferenceType>(Param)->getPointeeType(),
                             ReferenceArg->getPointeeType(),
                             Info, Deduced, 0);
    }

    //     T [] (implied, but not stated explicitly)
    case Type::IncompleteArray: {
      const IncompleteArrayType *IncompleteArrayArg =
        S.Context.getAsIncompleteArrayType(Arg);
      if (!IncompleteArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      unsigned SubTDF = TDF & TDF_IgnoreQualifiers;
      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                    S.Context.getAsIncompleteArrayType(Param)->getElementType(),
                    IncompleteArrayArg->getElementType(),
                    Info, Deduced, SubTDF);
    }

    //     T [integer-constant]
    case Type::ConstantArray: {
      const ConstantArrayType *ConstantArrayArg =
        S.Context.getAsConstantArrayType(Arg);
      if (!ConstantArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      const ConstantArrayType *ConstantArrayParm =
        S.Context.getAsConstantArrayType(Param);
      if (ConstantArrayArg->getSize() != ConstantArrayParm->getSize())
        return Sema::TDK_NonDeducedMismatch;

      unsigned SubTDF = TDF & TDF_IgnoreQualifiers;
      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                           ConstantArrayParm->getElementType(),
                                           ConstantArrayArg->getElementType(),
                                           Info, Deduced, SubTDF);
    }

    //     type [i]
    case Type::DependentSizedArray: {
      const ArrayType *ArrayArg = S.Context.getAsArrayType(Arg);
      if (!ArrayArg)
        return Sema::TDK_NonDeducedMismatch;

      unsigned SubTDF = TDF & TDF_IgnoreQualifiers;

      // Check the element type of the arrays
      const DependentSizedArrayType *DependentArrayParm
        = S.Context.getAsDependentSizedArrayType(Param);
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                          DependentArrayParm->getElementType(),
                                          ArrayArg->getElementType(),
                                          Info, Deduced, SubTDF))
        return Result;

      // Determine the array bound is something we can deduce.
      NonTypeTemplateParmDecl *NTTP
        = getDeducedParameterFromExpr(DependentArrayParm->getSizeExpr());
      if (!NTTP)
        return Sema::TDK_Success;

      // We can perform template argument deduction for the given non-type
      // template parameter.
      assert(NTTP->getDepth() == 0 &&
             "Cannot deduce non-type template argument at depth > 0");
      if (const ConstantArrayType *ConstantArrayArg
            = dyn_cast<ConstantArrayType>(ArrayArg)) {
        llvm::APSInt Size(ConstantArrayArg->getSize());
        return DeduceNonTypeTemplateArgument(S, NTTP, Size,
                                             S.Context.getSizeType(),
                                             /*ArrayBound=*/true,
                                             Info, Deduced);
      }
      if (const DependentSizedArrayType *DependentArrayArg
            = dyn_cast<DependentSizedArrayType>(ArrayArg))
        if (DependentArrayArg->getSizeExpr())
          return DeduceNonTypeTemplateArgument(S, NTTP,
                                               DependentArrayArg->getSizeExpr(),
                                               Info, Deduced);

      // Incomplete type does not match a dependently-sized array type
      return Sema::TDK_NonDeducedMismatch;
    }

    //     type(*)(T)
    //     T(*)()
    //     T(*)(T)
    case Type::FunctionProto: {
      unsigned SubTDF = TDF & TDF_TopLevelParameterTypeList;
      const FunctionProtoType *FunctionProtoArg =
        dyn_cast<FunctionProtoType>(Arg);
      if (!FunctionProtoArg)
        return Sema::TDK_NonDeducedMismatch;

      const FunctionProtoType *FunctionProtoParam =
        cast<FunctionProtoType>(Param);

      if (FunctionProtoParam->getTypeQuals()
            != FunctionProtoArg->getTypeQuals() ||
          FunctionProtoParam->getRefQualifier()
            != FunctionProtoArg->getRefQualifier() ||
          FunctionProtoParam->isVariadic() != FunctionProtoArg->isVariadic())
        return Sema::TDK_NonDeducedMismatch;

      // Check return types.
      if (Sema::TemplateDeductionResult Result =
              DeduceTemplateArgumentsByTypeMatch(
                  S, TemplateParams, FunctionProtoParam->getReturnType(),
                  FunctionProtoArg->getReturnType(), Info, Deduced, 0))
        return Result;

      return DeduceTemplateArguments(
          S, TemplateParams, FunctionProtoParam->param_type_begin(),
          FunctionProtoParam->getNumParams(),
          FunctionProtoArg->param_type_begin(),
          FunctionProtoArg->getNumParams(), Info, Deduced, SubTDF);
    }

    case Type::InjectedClassName: {
      // Treat a template's injected-class-name as if the template
      // specialization type had been used.
      Param = cast<InjectedClassNameType>(Param)
        ->getInjectedSpecializationType();
      assert(isa<TemplateSpecializationType>(Param) &&
             "injected class name is not a template specialization type");
      // fall through
    }

    //     template-name<T> (where template-name refers to a class template)
    //     template-name<i>
    //     TT<T>
    //     TT<i>
    //     TT<>
    case Type::TemplateSpecialization: {
      const TemplateSpecializationType *SpecParam
        = cast<TemplateSpecializationType>(Param);

      // Try to deduce template arguments from the template-id.
      Sema::TemplateDeductionResult Result
        = DeduceTemplateArguments(S, TemplateParams, SpecParam, Arg,
                                  Info, Deduced);

      if (Result && (TDF & TDF_DerivedClass)) {
        // C++ [temp.deduct.call]p3b3:
        //   If P is a class, and P has the form template-id, then A can be a
        //   derived class of the deduced A. Likewise, if P is a pointer to a
        //   class of the form template-id, A can be a pointer to a derived
        //   class pointed to by the deduced A.
        //
        // More importantly:
        //   These alternatives are considered only if type deduction would
        //   otherwise fail.
        if (const RecordType *RecordT = Arg->getAs<RecordType>()) {
          // We cannot inspect base classes as part of deduction when the type
          // is incomplete, so either instantiate any templates necessary to
          // complete the type, or skip over it if it cannot be completed.
          if (!S.isCompleteType(Info.getLocation(), Arg))
            return Result;

          // Use data recursion to crawl through the list of base classes.
          // Visited contains the set of nodes we have already visited, while
          // ToVisit is our stack of records that we still need to visit.
          llvm::SmallPtrSet<const RecordType *, 8> Visited;
          SmallVector<const RecordType *, 8> ToVisit;
          ToVisit.push_back(RecordT);
          bool Successful = false;
          SmallVector<DeducedTemplateArgument, 8> DeducedOrig(Deduced.begin(),
                                                              Deduced.end());
          while (!ToVisit.empty()) {
            // Retrieve the next class in the inheritance hierarchy.
            const RecordType *NextT = ToVisit.pop_back_val();

            // If we have already seen this type, skip it.
            if (!Visited.insert(NextT).second)
              continue;

            // If this is a base class, try to perform template argument
            // deduction from it.
            if (NextT != RecordT) {
              TemplateDeductionInfo BaseInfo(Info.getLocation());
              Sema::TemplateDeductionResult BaseResult
                = DeduceTemplateArguments(S, TemplateParams, SpecParam,
                                          QualType(NextT, 0), BaseInfo,
                                          Deduced);

              // If template argument deduction for this base was successful,
              // note that we had some success. Otherwise, ignore any deductions
              // from this base class.
              if (BaseResult == Sema::TDK_Success) {
                Successful = true;
                DeducedOrig.clear();
                DeducedOrig.append(Deduced.begin(), Deduced.end());
                Info.Param = BaseInfo.Param;
                Info.FirstArg = BaseInfo.FirstArg;
                Info.SecondArg = BaseInfo.SecondArg;
              }
              else
                Deduced = DeducedOrig;
            }

            // Visit base classes
            CXXRecordDecl *Next = cast<CXXRecordDecl>(NextT->getDecl());
            for (const auto &Base : Next->bases()) {
              assert(Base.getType()->isRecordType() &&
                     "Base class that isn't a record?");
              ToVisit.push_back(Base.getType()->getAs<RecordType>());
            }
          }

          if (Successful)
            return Sema::TDK_Success;
        }

      }

      return Result;
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
      const MemberPointerType *MemPtrParam = cast<MemberPointerType>(Param);
      const MemberPointerType *MemPtrArg = dyn_cast<MemberPointerType>(Arg);
      if (!MemPtrArg)
        return Sema::TDK_NonDeducedMismatch;

      QualType ParamPointeeType = MemPtrParam->getPointeeType();
      if (ParamPointeeType->isFunctionType())
        S.adjustMemberFunctionCC(ParamPointeeType, /*IsStatic=*/true,
                                 /*IsCtorOrDtor=*/false, Info.getLocation());
      QualType ArgPointeeType = MemPtrArg->getPointeeType();
      if (ArgPointeeType->isFunctionType())
        S.adjustMemberFunctionCC(ArgPointeeType, /*IsStatic=*/true,
                                 /*IsCtorOrDtor=*/false, Info.getLocation());

      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                 ParamPointeeType,
                                                 ArgPointeeType,
                                                 Info, Deduced,
                                                 TDF & TDF_IgnoreQualifiers))
        return Result;

      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                           QualType(MemPtrParam->getClass(), 0),
                                           QualType(MemPtrArg->getClass(), 0),
                                           Info, Deduced, 
                                           TDF & TDF_IgnoreQualifiers);
    }

    //     (clang extension)
    //
    //     type(^)(T)
    //     T(^)()
    //     T(^)(T)
    case Type::BlockPointer: {
      const BlockPointerType *BlockPtrParam = cast<BlockPointerType>(Param);
      const BlockPointerType *BlockPtrArg = dyn_cast<BlockPointerType>(Arg);

      if (!BlockPtrArg)
        return Sema::TDK_NonDeducedMismatch;

      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                BlockPtrParam->getPointeeType(),
                                                BlockPtrArg->getPointeeType(),
                                                Info, Deduced, 0);
    }

    //     (clang extension)
    //
    //     T __attribute__(((ext_vector_type(<integral constant>))))
    case Type::ExtVector: {
      const ExtVectorType *VectorParam = cast<ExtVectorType>(Param);
      if (const ExtVectorType *VectorArg = dyn_cast<ExtVectorType>(Arg)) {
        // Make sure that the vectors have the same number of elements.
        if (VectorParam->getNumElements() != VectorArg->getNumElements())
          return Sema::TDK_NonDeducedMismatch;
        
        // Perform deduction on the element types.
        return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                  VectorParam->getElementType(),
                                                  VectorArg->getElementType(),
                                                  Info, Deduced, TDF);
      }
      
      if (const DependentSizedExtVectorType *VectorArg 
                                = dyn_cast<DependentSizedExtVectorType>(Arg)) {
        // We can't check the number of elements, since the argument has a
        // dependent number of elements. This can only occur during partial
        // ordering.

        // Perform deduction on the element types.
        return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                  VectorParam->getElementType(),
                                                  VectorArg->getElementType(),
                                                  Info, Deduced, TDF);
      }
      
      return Sema::TDK_NonDeducedMismatch;
    }
      
    //     (clang extension)
    //
    //     T __attribute__(((ext_vector_type(N))))
    case Type::DependentSizedExtVector: {
      const DependentSizedExtVectorType *VectorParam
        = cast<DependentSizedExtVectorType>(Param);

      if (const ExtVectorType *VectorArg = dyn_cast<ExtVectorType>(Arg)) {
        // Perform deduction on the element types.
        if (Sema::TemplateDeductionResult Result
              = DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                  VectorParam->getElementType(),
                                                   VectorArg->getElementType(),
                                                   Info, Deduced, TDF))
          return Result;
        
        // Perform deduction on the vector size, if we can.
        NonTypeTemplateParmDecl *NTTP
          = getDeducedParameterFromExpr(VectorParam->getSizeExpr());
        if (!NTTP)
          return Sema::TDK_Success;

        llvm::APSInt ArgSize(S.Context.getTypeSize(S.Context.IntTy), false);
        ArgSize = VectorArg->getNumElements();
        return DeduceNonTypeTemplateArgument(S, NTTP, ArgSize, S.Context.IntTy,
                                             false, Info, Deduced);
      }
      
      if (const DependentSizedExtVectorType *VectorArg 
                                = dyn_cast<DependentSizedExtVectorType>(Arg)) {
        // Perform deduction on the element types.
        if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                 VectorParam->getElementType(),
                                                 VectorArg->getElementType(),
                                                 Info, Deduced, TDF))
          return Result;
        
        // Perform deduction on the vector size, if we can.
        NonTypeTemplateParmDecl *NTTP
          = getDeducedParameterFromExpr(VectorParam->getSizeExpr());
        if (!NTTP)
          return Sema::TDK_Success;
        
        return DeduceNonTypeTemplateArgument(S, NTTP, VectorArg->getSizeExpr(),
                                             Info, Deduced);
      }
      
      return Sema::TDK_NonDeducedMismatch;
    }
      
    case Type::TypeOfExpr:
    case Type::TypeOf:
    case Type::DependentName:
    case Type::UnresolvedUsing:
    case Type::Decltype:
    case Type::UnaryTransform:
    case Type::Auto:
    case Type::DependentTemplateSpecialization:
    case Type::PackExpansion:
      // No template argument deduction for these types
      return Sema::TDK_Success;
  }

  llvm_unreachable("Invalid Type Class!");
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument &Param,
                        TemplateArgument Arg,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  // If the template argument is a pack expansion, perform template argument
  // deduction against the pattern of that expansion. This only occurs during
  // partial ordering.
  if (Arg.isPackExpansion())
    Arg = Arg.getPackExpansionPattern();

  switch (Param.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Null template argument in parameter list");

  case TemplateArgument::Type:
    if (Arg.getKind() == TemplateArgument::Type)
      return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                                Param.getAsType(),
                                                Arg.getAsType(),
                                                Info, Deduced, 0);
    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;

  case TemplateArgument::Template:
    if (Arg.getKind() == TemplateArgument::Template)
      return DeduceTemplateArguments(S, TemplateParams,
                                     Param.getAsTemplate(),
                                     Arg.getAsTemplate(), Info, Deduced);
    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;

  case TemplateArgument::TemplateExpansion:
    llvm_unreachable("caller should handle pack expansions");

  case TemplateArgument::Declaration:
    if (Arg.getKind() == TemplateArgument::Declaration &&
        isSameDeclaration(Param.getAsDecl(), Arg.getAsDecl()))
      return Sema::TDK_Success;

    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;

  case TemplateArgument::NullPtr:
    if (Arg.getKind() == TemplateArgument::NullPtr &&
        S.Context.hasSameType(Param.getNullPtrType(), Arg.getNullPtrType()))
      return Sema::TDK_Success;

    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;

  case TemplateArgument::Integral:
    if (Arg.getKind() == TemplateArgument::Integral) {
      if (hasSameExtendedValue(Param.getAsIntegral(), Arg.getAsIntegral()))
        return Sema::TDK_Success;

      Info.FirstArg = Param;
      Info.SecondArg = Arg;
      return Sema::TDK_NonDeducedMismatch;
    }

    if (Arg.getKind() == TemplateArgument::Expression) {
      Info.FirstArg = Param;
      Info.SecondArg = Arg;
      return Sema::TDK_NonDeducedMismatch;
    }

    Info.FirstArg = Param;
    Info.SecondArg = Arg;
    return Sema::TDK_NonDeducedMismatch;

  case TemplateArgument::Expression: {
    if (NonTypeTemplateParmDecl *NTTP
          = getDeducedParameterFromExpr(Param.getAsExpr())) {
      if (Arg.getKind() == TemplateArgument::Integral)
        return DeduceNonTypeTemplateArgument(S, NTTP,
                                             Arg.getAsIntegral(),
                                             Arg.getIntegralType(),
                                             /*ArrayBound=*/false,
                                             Info, Deduced);
      if (Arg.getKind() == TemplateArgument::Expression)
        return DeduceNonTypeTemplateArgument(S, NTTP, Arg.getAsExpr(),
                                             Info, Deduced);
      if (Arg.getKind() == TemplateArgument::Declaration)
        return DeduceNonTypeTemplateArgument(S, NTTP, Arg.getAsDecl(),
                                             Info, Deduced);

      Info.FirstArg = Param;
      Info.SecondArg = Arg;
      return Sema::TDK_NonDeducedMismatch;
    }

    // Can't deduce anything, but that's okay.
    return Sema::TDK_Success;
  }
  case TemplateArgument::Pack:
    llvm_unreachable("Argument packs should be expanded by the caller!");
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

/// \brief Determine whether there is a template argument to be used for
/// deduction.
///
/// This routine "expands" argument packs in-place, overriding its input
/// parameters so that \c Args[ArgIdx] will be the available template argument.
///
/// \returns true if there is another template argument (which will be at
/// \c Args[ArgIdx]), false otherwise.
static bool hasTemplateArgumentForDeduction(const TemplateArgument *&Args,
                                            unsigned &ArgIdx,
                                            unsigned &NumArgs) {
  if (ArgIdx == NumArgs)
    return false;

  const TemplateArgument &Arg = Args[ArgIdx];
  if (Arg.getKind() != TemplateArgument::Pack)
    return true;

  assert(ArgIdx == NumArgs - 1 && "Pack not at the end of argument list?");
  Args = Arg.pack_begin();
  NumArgs = Arg.pack_size();
  ArgIdx = 0;
  return ArgIdx < NumArgs;
}

/// \brief Determine whether the given set of template arguments has a pack
/// expansion that is not the last template argument.
static bool hasPackExpansionBeforeEnd(const TemplateArgument *Args,
                                      unsigned NumArgs) {
  unsigned ArgIdx = 0;
  while (ArgIdx < NumArgs) {
    const TemplateArgument &Arg = Args[ArgIdx];

    // Unwrap argument packs.
    if (Args[ArgIdx].getKind() == TemplateArgument::Pack) {
      Args = Arg.pack_begin();
      NumArgs = Arg.pack_size();
      ArgIdx = 0;
      continue;
    }

    ++ArgIdx;
    if (ArgIdx == NumArgs)
      return false;

    if (Arg.isPackExpansion())
      return true;
  }

  return false;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgument *Params, unsigned NumParams,
                        const TemplateArgument *Args, unsigned NumArgs,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  // C++0x [temp.deduct.type]p9:
  //   If the template argument list of P contains a pack expansion that is not
  //   the last template argument, the entire template argument list is a
  //   non-deduced context.
  if (hasPackExpansionBeforeEnd(Params, NumParams))
    return Sema::TDK_Success;

  // C++0x [temp.deduct.type]p9:
  //   If P has a form that contains <T> or <i>, then each argument Pi of the
  //   respective template argument list P is compared with the corresponding
  //   argument Ai of the corresponding template argument list of A.
  unsigned ArgIdx = 0, ParamIdx = 0;
  for (; hasTemplateArgumentForDeduction(Params, ParamIdx, NumParams);
       ++ParamIdx) {
    if (!Params[ParamIdx].isPackExpansion()) {
      // The simple case: deduce template arguments by matching Pi and Ai.

      // Check whether we have enough arguments.
      if (!hasTemplateArgumentForDeduction(Args, ArgIdx, NumArgs))
        return Sema::TDK_Success;

      if (Args[ArgIdx].isPackExpansion()) {
        // FIXME: We follow the logic of C++0x [temp.deduct.type]p22 here,
        // but applied to pack expansions that are template arguments.
        return Sema::TDK_MiscellaneousDeductionFailure;
      }

      // Perform deduction for this Pi/Ai pair.
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(S, TemplateParams,
                                      Params[ParamIdx], Args[ArgIdx],
                                      Info, Deduced))
        return Result;

      // Move to the next argument.
      ++ArgIdx;
      continue;
    }

    // The parameter is a pack expansion.

    // C++0x [temp.deduct.type]p9:
    //   If Pi is a pack expansion, then the pattern of Pi is compared with
    //   each remaining argument in the template argument list of A. Each
    //   comparison deduces template arguments for subsequent positions in the
    //   template parameter packs expanded by Pi.
    TemplateArgument Pattern = Params[ParamIdx].getPackExpansionPattern();

    // FIXME: If there are no remaining arguments, we can bail out early
    // and set any deduced parameter packs to an empty argument pack.
    // The latter part of this is a (minor) correctness issue.

    // Prepare to deduce the packs within the pattern.
    PackDeductionScope PackScope(S, TemplateParams, Deduced, Info, Pattern);

    // Keep track of the deduced template arguments for each parameter pack
    // expanded by this pack expansion (the outer index) and for each
    // template argument (the inner SmallVectors).
    bool HasAnyArguments = false;
    for (; hasTemplateArgumentForDeduction(Args, ArgIdx, NumArgs); ++ArgIdx) {
      HasAnyArguments = true;

      // Deduce template arguments from the pattern.
      if (Sema::TemplateDeductionResult Result
            = DeduceTemplateArguments(S, TemplateParams, Pattern, Args[ArgIdx],
                                      Info, Deduced))
        return Result;

      PackScope.nextPackElement();
    }

    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    if (auto Result = PackScope.finish(HasAnyArguments))
      return Result;
  }

  return Sema::TDK_Success;
}

static Sema::TemplateDeductionResult
DeduceTemplateArguments(Sema &S,
                        TemplateParameterList *TemplateParams,
                        const TemplateArgumentList &ParamList,
                        const TemplateArgumentList &ArgList,
                        TemplateDeductionInfo &Info,
                        SmallVectorImpl<DeducedTemplateArgument> &Deduced) {
  return DeduceTemplateArguments(S, TemplateParams,
                                 ParamList.data(), ParamList.size(),
                                 ArgList.data(), ArgList.size(),
                                 Info, Deduced);
}

/// \brief Determine whether two template arguments are the same.
static bool isSameTemplateArg(ASTContext &Context,
                              const TemplateArgument &X,
                              const TemplateArgument &Y) {
  if (X.getKind() != Y.getKind())
    return false;

  switch (X.getKind()) {
    case TemplateArgument::Null:
      llvm_unreachable("Comparing NULL template argument");

    case TemplateArgument::Type:
      return Context.getCanonicalType(X.getAsType()) ==
             Context.getCanonicalType(Y.getAsType());

    case TemplateArgument::Declaration:
      return isSameDeclaration(X.getAsDecl(), Y.getAsDecl());

    case TemplateArgument::NullPtr:
      return Context.hasSameType(X.getNullPtrType(), Y.getNullPtrType());

    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion:
      return Context.getCanonicalTemplateName(
                    X.getAsTemplateOrTemplatePattern()).getAsVoidPointer() ==
             Context.getCanonicalTemplateName(
                    Y.getAsTemplateOrTemplatePattern()).getAsVoidPointer();

    case TemplateArgument::Integral:
      return X.getAsIntegral() == Y.getAsIntegral();

    case TemplateArgument::Expression: {
      llvm::FoldingSetNodeID XID, YID;
      X.getAsExpr()->Profile(XID, Context, true);
      Y.getAsExpr()->Profile(YID, Context, true);
      return XID == YID;
    }

    case TemplateArgument::Pack:
      if (X.pack_size() != Y.pack_size())
        return false;

      for (TemplateArgument::pack_iterator XP = X.pack_begin(),
                                        XPEnd = X.pack_end(),
                                           YP = Y.pack_begin();
           XP != XPEnd; ++XP, ++YP)
        if (!isSameTemplateArg(Context, *XP, *YP))
          return false;

      return true;
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}

/// \brief Allocate a TemplateArgumentLoc where all locations have
/// been initialized to the given location.
///
/// \param S The semantic analysis object.
///
/// \param Arg The template argument we are producing template argument
/// location information for.
///
/// \param NTTPType For a declaration template argument, the type of
/// the non-type template parameter that corresponds to this template
/// argument.
///
/// \param Loc The source location to use for the resulting template
/// argument.
static TemplateArgumentLoc
getTrivialTemplateArgumentLoc(Sema &S,
                              const TemplateArgument &Arg,
                              QualType NTTPType,
                              SourceLocation Loc) {
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("Can't get a NULL template argument here");

  case TemplateArgument::Type:
    return TemplateArgumentLoc(Arg,
                     S.Context.getTrivialTypeSourceInfo(Arg.getAsType(), Loc));

  case TemplateArgument::Declaration: {
    Expr *E
      = S.BuildExpressionFromDeclTemplateArgument(Arg, NTTPType, Loc)
          .getAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(E), E);
  }

  case TemplateArgument::NullPtr: {
    Expr *E
      = S.BuildExpressionFromDeclTemplateArgument(Arg, NTTPType, Loc)
          .getAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(NTTPType, /*isNullPtr*/true),
                               E);
  }

  case TemplateArgument::Integral: {
    Expr *E
      = S.BuildExpressionFromIntegralTemplateArgument(Arg, Loc).getAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(E), E);
  }

    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion: {
      NestedNameSpecifierLocBuilder Builder;
      TemplateName Template = Arg.getAsTemplate();
      if (DependentTemplateName *DTN = Template.getAsDependentTemplateName())
        Builder.MakeTrivial(S.Context, DTN->getQualifier(), Loc);
      else if (QualifiedTemplateName *QTN =
                   Template.getAsQualifiedTemplateName())
        Builder.MakeTrivial(S.Context, QTN->getQualifier(), Loc);
      
      if (Arg.getKind() == TemplateArgument::Template)
        return TemplateArgumentLoc(Arg, 
                                   Builder.getWithLocInContext(S.Context),
                                   Loc);
      
      
      return TemplateArgumentLoc(Arg, Builder.getWithLocInContext(S.Context),
                                 Loc, Loc);
    }

  case TemplateArgument::Expression:
    return TemplateArgumentLoc(Arg, Arg.getAsExpr());

  case TemplateArgument::Pack:
    return TemplateArgumentLoc(Arg, TemplateArgumentLocInfo());
  }

  llvm_unreachable("Invalid TemplateArgument Kind!");
}


/// \brief Convert the given deduced template argument and add it to the set of
/// fully-converted template arguments.
static bool
ConvertDeducedTemplateArgument(Sema &S, NamedDecl *Param,
                               DeducedTemplateArgument Arg,
                               NamedDecl *Template,
                               QualType NTTPType,
                               unsigned ArgumentPackIndex,
                               TemplateDeductionInfo &Info,
                               bool InFunctionTemplate,
                               SmallVectorImpl<TemplateArgument> &Output) {
  if (Arg.getKind() == TemplateArgument::Pack) {
    // This is a template argument pack, so check each of its arguments against
    // the template parameter.
    SmallVector<TemplateArgument, 2> PackedArgsBuilder;
    for (const auto &P : Arg.pack_elements()) {
      // When converting the deduced template argument, append it to the
      // general output list. We need to do this so that the template argument
      // checking logic has all of the prior template arguments available.
      DeducedTemplateArgument InnerArg(P);
      InnerArg.setDeducedFromArrayBound(Arg.wasDeducedFromArrayBound());
      if (ConvertDeducedTemplateArgument(S, Param, InnerArg, Template,
                                         NTTPType, PackedArgsBuilder.size(),
                                         Info, InFunctionTemplate, Output))
        return true;

      // Move the converted template argument into our argument pack.
      PackedArgsBuilder.push_back(Output.pop_back_val());
    }

    // Create the resulting argument pack.
    Output.push_back(
        TemplateArgument::CreatePackCopy(S.Context, PackedArgsBuilder));
    return false;
  }

  // Convert the deduced template argument into a template
  // argument that we can check, almost as if the user had written
  // the template argument explicitly.
  TemplateArgumentLoc ArgLoc = getTrivialTemplateArgumentLoc(S, Arg, NTTPType,
                                                             Info.getLocation());

  // Check the template argument, converting it as necessary.
  return S.CheckTemplateArgument(Param, ArgLoc,
                                 Template,
                                 Template->getLocation(),
                                 Template->getSourceRange().getEnd(),
                                 ArgumentPackIndex,
                                 Output,
                                 InFunctionTemplate
                                  ? (Arg.wasDeducedFromArrayBound()
                                       ? Sema::CTAK_DeducedFromArrayBound
                                       : Sema::CTAK_Deduced)
                                 : Sema::CTAK_Specified);
}

/// Complete template argument deduction for a class template partial
/// specialization.
static Sema::TemplateDeductionResult
FinishTemplateArgumentDeduction(Sema &S,
                                ClassTemplatePartialSpecializationDecl *Partial,
                                const TemplateArgumentList &TemplateArgs,
                      SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                TemplateDeductionInfo &Info) {
  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(S, Sema::Unevaluated);
  Sema::SFINAETrap Trap(S);

  Sema::ContextRAII SavedContext(S, Partial);

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  SmallVector<TemplateArgument, 4> Builder;
  TemplateParameterList *PartialParams = Partial->getTemplateParameters();
  for (unsigned I = 0, N = PartialParams->size(); I != N; ++I) {
    NamedDecl *Param = PartialParams->getParam(I);
    if (Deduced[I].isNull()) {
      Info.Param = makeTemplateParameter(Param);
      return Sema::TDK_Incomplete;
    }

    // We have deduced this argument, so it still needs to be
    // checked and converted.

    // First, for a non-type template parameter type that is
    // initialized by a declaration, we need the type of the
    // corresponding non-type template parameter.
    QualType NTTPType;
    if (NonTypeTemplateParmDecl *NTTP
                                  = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      NTTPType = NTTP->getType();
      if (NTTPType->isDependentType()) {
        TemplateArgumentList TemplateArgs(TemplateArgumentList::OnStack,
                                          Builder.data(), Builder.size());
        NTTPType = S.SubstType(NTTPType,
                               MultiLevelTemplateArgumentList(TemplateArgs),
                               NTTP->getLocation(),
                               NTTP->getDeclName());
        if (NTTPType.isNull()) {
          Info.Param = makeTemplateParameter(Param);
          // FIXME: These template arguments are temporary. Free them!
          Info.reset(TemplateArgumentList::CreateCopy(S.Context,
                                                      Builder.data(),
                                                      Builder.size()));
          return Sema::TDK_SubstitutionFailure;
        }
      }
    }

    if (ConvertDeducedTemplateArgument(S, Param, Deduced[I],
                                       Partial, NTTPType, 0, Info, false,
                                       Builder)) {
      Info.Param = makeTemplateParameter(Param);
      // FIXME: These template arguments are temporary. Free them!
      Info.reset(TemplateArgumentList::CreateCopy(S.Context, Builder.data(),
                                                  Builder.size()));
      return Sema::TDK_SubstitutionFailure;
    }
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList
    = TemplateArgumentList::CreateCopy(S.Context, Builder.data(),
                                       Builder.size());

  Info.reset(DeducedArgumentList);

  // Substitute the deduced template arguments into the template
  // arguments of the class template partial specialization, and
  // verify that the instantiated template arguments are both valid
  // and are equivalent to the template arguments originally provided
  // to the class template.
  LocalInstantiationScope InstScope(S);
  ClassTemplateDecl *ClassTemplate = Partial->getSpecializedTemplate();
  const ASTTemplateArgumentListInfo *PartialTemplArgInfo
    = Partial->getTemplateArgsAsWritten();
  const TemplateArgumentLoc *PartialTemplateArgs
    = PartialTemplArgInfo->getTemplateArgs();

  TemplateArgumentListInfo InstArgs(PartialTemplArgInfo->LAngleLoc,
                                    PartialTemplArgInfo->RAngleLoc);

  if (S.Subst(PartialTemplateArgs, PartialTemplArgInfo->NumTemplateArgs,
              InstArgs, MultiLevelTemplateArgumentList(*DeducedArgumentList))) {
    unsigned ArgIdx = InstArgs.size(), ParamIdx = ArgIdx;
    if (ParamIdx >= Partial->getTemplateParameters()->size())
      ParamIdx = Partial->getTemplateParameters()->size() - 1;

    Decl *Param
      = const_cast<NamedDecl *>(
                          Partial->getTemplateParameters()->getParam(ParamIdx));
    Info.Param = makeTemplateParameter(Param);
    Info.FirstArg = PartialTemplateArgs[ArgIdx].getArgument();
    return Sema::TDK_SubstitutionFailure;
  }

  SmallVector<TemplateArgument, 4> ConvertedInstArgs;
  if (S.CheckTemplateArgumentList(ClassTemplate, Partial->getLocation(),
                                  InstArgs, false, ConvertedInstArgs))
    return Sema::TDK_SubstitutionFailure;

  TemplateParameterList *TemplateParams
    = ClassTemplate->getTemplateParameters();
  for (unsigned I = 0, E = TemplateParams->size(); I != E; ++I) {
    TemplateArgument InstArg = ConvertedInstArgs.data()[I];
    if (!isSameTemplateArg(S.Context, TemplateArgs[I], InstArg)) {
      Info.Param = makeTemplateParameter(TemplateParams->getParam(I));
      Info.FirstArg = TemplateArgs[I];
      Info.SecondArg = InstArg;
      return Sema::TDK_NonDeducedMismatch;
    }
  }

  if (Trap.hasErrorOccurred())
    return Sema::TDK_SubstitutionFailure;

  return Sema::TDK_Success;
}

/// \brief Perform template argument deduction to determine whether
/// the given template arguments match the given class template
/// partial specialization per C++ [temp.class.spec.match].
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(ClassTemplatePartialSpecializationDecl *Partial,
                              const TemplateArgumentList &TemplateArgs,
                              TemplateDeductionInfo &Info) {
  if (Partial->isInvalidDecl())
    return TDK_Invalid;

  // C++ [temp.class.spec.match]p2:
  //   A partial specialization matches a given actual template
  //   argument list if the template arguments of the partial
  //   specialization can be deduced from the actual template argument
  //   list (14.8.2).

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);
  SFINAETrap Trap(*this);

  SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(Partial->getTemplateParameters()->size());
  if (TemplateDeductionResult Result
        = ::DeduceTemplateArguments(*this,
                                    Partial->getTemplateParameters(),
                                    Partial->getTemplateArgs(),
                                    TemplateArgs, Info, Deduced))
    return Result;

  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  InstantiatingTemplate Inst(*this, Info.getLocation(), Partial, DeducedArgs,
                             Info);
  if (Inst.isInvalid())
    return TDK_InstantiationDepth;

  if (Trap.hasErrorOccurred())
    return Sema::TDK_SubstitutionFailure;

  return ::FinishTemplateArgumentDeduction(*this, Partial, TemplateArgs,
                                           Deduced, Info);
}

/// Complete template argument deduction for a variable template partial
/// specialization.
/// TODO: Unify with ClassTemplatePartialSpecializationDecl version?
///       May require unifying ClassTemplate(Partial)SpecializationDecl and
///        VarTemplate(Partial)SpecializationDecl with a new data
///        structure Template(Partial)SpecializationDecl, and
///        using Template(Partial)SpecializationDecl as input type.
static Sema::TemplateDeductionResult FinishTemplateArgumentDeduction(
    Sema &S, VarTemplatePartialSpecializationDecl *Partial,
    const TemplateArgumentList &TemplateArgs,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced,
    TemplateDeductionInfo &Info) {
  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(S, Sema::Unevaluated);
  Sema::SFINAETrap Trap(S);

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  SmallVector<TemplateArgument, 4> Builder;
  TemplateParameterList *PartialParams = Partial->getTemplateParameters();
  for (unsigned I = 0, N = PartialParams->size(); I != N; ++I) {
    NamedDecl *Param = PartialParams->getParam(I);
    if (Deduced[I].isNull()) {
      Info.Param = makeTemplateParameter(Param);
      return Sema::TDK_Incomplete;
    }

    // We have deduced this argument, so it still needs to be
    // checked and converted.

    // First, for a non-type template parameter type that is
    // initialized by a declaration, we need the type of the
    // corresponding non-type template parameter.
    QualType NTTPType;
    if (NonTypeTemplateParmDecl *NTTP =
            dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      NTTPType = NTTP->getType();
      if (NTTPType->isDependentType()) {
        TemplateArgumentList TemplateArgs(TemplateArgumentList::OnStack,
                                          Builder.data(), Builder.size());
        NTTPType =
            S.SubstType(NTTPType, MultiLevelTemplateArgumentList(TemplateArgs),
                        NTTP->getLocation(), NTTP->getDeclName());
        if (NTTPType.isNull()) {
          Info.Param = makeTemplateParameter(Param);
          // FIXME: These template arguments are temporary. Free them!
          Info.reset(TemplateArgumentList::CreateCopy(S.Context, Builder.data(),
                                                      Builder.size()));
          return Sema::TDK_SubstitutionFailure;
        }
      }
    }

    if (ConvertDeducedTemplateArgument(S, Param, Deduced[I], Partial, NTTPType,
                                       0, Info, false, Builder)) {
      Info.Param = makeTemplateParameter(Param);
      // FIXME: These template arguments are temporary. Free them!
      Info.reset(TemplateArgumentList::CreateCopy(S.Context, Builder.data(),
                                                  Builder.size()));
      return Sema::TDK_SubstitutionFailure;
    }
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList = TemplateArgumentList::CreateCopy(
      S.Context, Builder.data(), Builder.size());

  Info.reset(DeducedArgumentList);

  // Substitute the deduced template arguments into the template
  // arguments of the class template partial specialization, and
  // verify that the instantiated template arguments are both valid
  // and are equivalent to the template arguments originally provided
  // to the class template.
  LocalInstantiationScope InstScope(S);
  VarTemplateDecl *VarTemplate = Partial->getSpecializedTemplate();
  const ASTTemplateArgumentListInfo *PartialTemplArgInfo
    = Partial->getTemplateArgsAsWritten();
  const TemplateArgumentLoc *PartialTemplateArgs
    = PartialTemplArgInfo->getTemplateArgs();

  TemplateArgumentListInfo InstArgs(PartialTemplArgInfo->LAngleLoc,
                                    PartialTemplArgInfo->RAngleLoc);

  if (S.Subst(PartialTemplateArgs, PartialTemplArgInfo->NumTemplateArgs,
              InstArgs, MultiLevelTemplateArgumentList(*DeducedArgumentList))) {
    unsigned ArgIdx = InstArgs.size(), ParamIdx = ArgIdx;
    if (ParamIdx >= Partial->getTemplateParameters()->size())
      ParamIdx = Partial->getTemplateParameters()->size() - 1;

    Decl *Param = const_cast<NamedDecl *>(
        Partial->getTemplateParameters()->getParam(ParamIdx));
    Info.Param = makeTemplateParameter(Param);
    Info.FirstArg = PartialTemplateArgs[ArgIdx].getArgument();
    return Sema::TDK_SubstitutionFailure;
  }
  SmallVector<TemplateArgument, 4> ConvertedInstArgs;
  if (S.CheckTemplateArgumentList(VarTemplate, Partial->getLocation(), InstArgs,
                                  false, ConvertedInstArgs))
    return Sema::TDK_SubstitutionFailure;

  TemplateParameterList *TemplateParams = VarTemplate->getTemplateParameters();
  for (unsigned I = 0, E = TemplateParams->size(); I != E; ++I) {
    TemplateArgument InstArg = ConvertedInstArgs.data()[I];
    if (!isSameTemplateArg(S.Context, TemplateArgs[I], InstArg)) {
      Info.Param = makeTemplateParameter(TemplateParams->getParam(I));
      Info.FirstArg = TemplateArgs[I];
      Info.SecondArg = InstArg;
      return Sema::TDK_NonDeducedMismatch;
    }
  }

  if (Trap.hasErrorOccurred())
    return Sema::TDK_SubstitutionFailure;

  return Sema::TDK_Success;
}

/// \brief Perform template argument deduction to determine whether
/// the given template arguments match the given variable template
/// partial specialization per C++ [temp.class.spec.match].
/// TODO: Unify with ClassTemplatePartialSpecializationDecl version?
///       May require unifying ClassTemplate(Partial)SpecializationDecl and
///        VarTemplate(Partial)SpecializationDecl with a new data
///        structure Template(Partial)SpecializationDecl, and
///        using Template(Partial)SpecializationDecl as input type.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(VarTemplatePartialSpecializationDecl *Partial,
                              const TemplateArgumentList &TemplateArgs,
                              TemplateDeductionInfo &Info) {
  if (Partial->isInvalidDecl())
    return TDK_Invalid;

  // C++ [temp.class.spec.match]p2:
  //   A partial specialization matches a given actual template
  //   argument list if the template arguments of the partial
  //   specialization can be deduced from the actual template argument
  //   list (14.8.2).

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);
  SFINAETrap Trap(*this);

  SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(Partial->getTemplateParameters()->size());
  if (TemplateDeductionResult Result = ::DeduceTemplateArguments(
          *this, Partial->getTemplateParameters(), Partial->getTemplateArgs(),
          TemplateArgs, Info, Deduced))
    return Result;

  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  InstantiatingTemplate Inst(*this, Info.getLocation(), Partial, DeducedArgs,
                             Info);
  if (Inst.isInvalid())
    return TDK_InstantiationDepth;

  if (Trap.hasErrorOccurred())
    return Sema::TDK_SubstitutionFailure;

  return ::FinishTemplateArgumentDeduction(*this, Partial, TemplateArgs,
                                           Deduced, Info);
}

/// \brief Determine whether the given type T is a simple-template-id type.
static bool isSimpleTemplateIdType(QualType T) {
  if (const TemplateSpecializationType *Spec
        = T->getAs<TemplateSpecializationType>())
    return Spec->getTemplateName().getAsTemplateDecl() != nullptr;

  return false;
}

/// \brief Substitute the explicitly-provided template arguments into the
/// given function template according to C++ [temp.arg.explicit].
///
/// \param FunctionTemplate the function template into which the explicit
/// template arguments will be substituted.
///
/// \param ExplicitTemplateArgs the explicitly-specified template
/// arguments.
///
/// \param Deduced the deduced template arguments, which will be populated
/// with the converted and checked explicit template arguments.
///
/// \param ParamTypes will be populated with the instantiated function
/// parameters.
///
/// \param FunctionType if non-NULL, the result type of the function template
/// will also be instantiated and the pointed-to value will be updated with
/// the instantiated function type.
///
/// \param Info if substitution fails for any reason, this object will be
/// populated with more information about the failure.
///
/// \returns TDK_Success if substitution was successful, or some failure
/// condition.
Sema::TemplateDeductionResult
Sema::SubstituteExplicitTemplateArguments(
                                      FunctionTemplateDecl *FunctionTemplate,
                               TemplateArgumentListInfo &ExplicitTemplateArgs,
                       SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                 SmallVectorImpl<QualType> &ParamTypes,
                                          QualType *FunctionType,
                                          TemplateDeductionInfo &Info) {
  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();

  if (ExplicitTemplateArgs.size() == 0) {
    // No arguments to substitute; just copy over the parameter types and
    // fill in the function type.
    for (auto P : Function->params())
      ParamTypes.push_back(P->getType());

    if (FunctionType)
      *FunctionType = Function->getType();
    return TDK_Success;
  }

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);
  SFINAETrap Trap(*this);

  // C++ [temp.arg.explicit]p3:
  //   Template arguments that are present shall be specified in the
  //   declaration order of their corresponding template-parameters. The
  //   template argument list shall not specify more template-arguments than
  //   there are corresponding template-parameters.
  SmallVector<TemplateArgument, 4> Builder;

  // Enter a new template instantiation context where we check the
  // explicitly-specified template arguments against this function template,
  // and then substitute them into the function parameter types.
  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  InstantiatingTemplate Inst(*this, Info.getLocation(), FunctionTemplate,
                             DeducedArgs,
           ActiveTemplateInstantiation::ExplicitTemplateArgumentSubstitution,
                             Info);
  if (Inst.isInvalid())
    return TDK_InstantiationDepth;

  if (CheckTemplateArgumentList(FunctionTemplate,
                                SourceLocation(),
                                ExplicitTemplateArgs,
                                true,
                                Builder) || Trap.hasErrorOccurred()) {
    unsigned Index = Builder.size();
    if (Index >= TemplateParams->size())
      Index = TemplateParams->size() - 1;
    Info.Param = makeTemplateParameter(TemplateParams->getParam(Index));
    return TDK_InvalidExplicitArguments;
  }

  // Form the template argument list from the explicitly-specified
  // template arguments.
  TemplateArgumentList *ExplicitArgumentList
    = TemplateArgumentList::CreateCopy(Context, Builder.data(), Builder.size());
  Info.reset(ExplicitArgumentList);

  // Template argument deduction and the final substitution should be
  // done in the context of the templated declaration.  Explicit
  // argument substitution, on the other hand, needs to happen in the
  // calling context.
  ContextRAII SavedContext(*this, FunctionTemplate->getTemplatedDecl());

  // If we deduced template arguments for a template parameter pack,
  // note that the template argument pack is partially substituted and record
  // the explicit template arguments. They'll be used as part of deduction
  // for this template parameter pack.
  for (unsigned I = 0, N = Builder.size(); I != N; ++I) {
    const TemplateArgument &Arg = Builder[I];
    if (Arg.getKind() == TemplateArgument::Pack) {
      CurrentInstantiationScope->SetPartiallySubstitutedPack(
                                                 TemplateParams->getParam(I),
                                                             Arg.pack_begin(),
                                                             Arg.pack_size());
      break;
    }
  }

  const FunctionProtoType *Proto
    = Function->getType()->getAs<FunctionProtoType>();
  assert(Proto && "Function template does not have a prototype?");

  // Isolate our substituted parameters from our caller.
  LocalInstantiationScope InstScope(*this, /*MergeWithOuterScope*/true);

  // Instantiate the types of each of the function parameters given the
  // explicitly-specified template arguments. If the function has a trailing
  // return type, substitute it after the arguments to ensure we substitute
  // in lexical order.
  if (Proto->hasTrailingReturn()) {
    if (SubstParmTypes(Function->getLocation(),
                       Function->param_begin(), Function->getNumParams(),
                       MultiLevelTemplateArgumentList(*ExplicitArgumentList),
                       ParamTypes))
      return TDK_SubstitutionFailure;
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
    unsigned ThisTypeQuals = 0;
    CXXRecordDecl *ThisContext = nullptr;
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Function)) {
      ThisContext = Method->getParent();
      ThisTypeQuals = Method->getTypeQualifiers();
    }
      
    CXXThisScopeRAII ThisScope(*this, ThisContext, ThisTypeQuals,
                               getLangOpts().CPlusPlus11);

    ResultType =
        SubstType(Proto->getReturnType(),
                  MultiLevelTemplateArgumentList(*ExplicitArgumentList),
                  Function->getTypeSpecStartLoc(), Function->getDeclName());
    if (ResultType.isNull() || Trap.hasErrorOccurred())
      return TDK_SubstitutionFailure;
  }
  
  // Instantiate the types of each of the function parameters given the
  // explicitly-specified template arguments if we didn't do so earlier.
  if (!Proto->hasTrailingReturn() &&
      SubstParmTypes(Function->getLocation(),
                     Function->param_begin(), Function->getNumParams(),
                     MultiLevelTemplateArgumentList(*ExplicitArgumentList),
                     ParamTypes))
    return TDK_SubstitutionFailure;

  if (FunctionType) {
    *FunctionType = BuildFunctionType(ResultType, ParamTypes,
                                      Function->getLocation(),
                                      Function->getDeclName(),
                                      Proto->getExtProtoInfo());
    if (FunctionType->isNull() || Trap.hasErrorOccurred())
      return TDK_SubstitutionFailure;
  }

  // C++ [temp.arg.explicit]p2:
  //   Trailing template arguments that can be deduced (14.8.2) may be
  //   omitted from the list of explicit template-arguments. If all of the
  //   template arguments can be deduced, they may all be omitted; in this
  //   case, the empty template argument list <> itself may also be omitted.
  //
  // Take all of the explicitly-specified arguments and put them into
  // the set of deduced template arguments. Explicitly-specified
  // parameter packs, however, will be set to NULL since the deduction
  // mechanisms handle explicitly-specified argument packs directly.
  Deduced.reserve(TemplateParams->size());
  for (unsigned I = 0, N = ExplicitArgumentList->size(); I != N; ++I) {
    const TemplateArgument &Arg = ExplicitArgumentList->get(I);
    if (Arg.getKind() == TemplateArgument::Pack)
      Deduced.push_back(DeducedTemplateArgument());
    else
      Deduced.push_back(Arg);
  }

  return TDK_Success;
}

/// \brief Check whether the deduced argument type for a call to a function
/// template matches the actual argument type per C++ [temp.deduct.call]p4.
static bool 
CheckOriginalCallArgDeduction(Sema &S, Sema::OriginalCallArg OriginalArg, 
                              QualType DeducedA) {
  ASTContext &Context = S.Context;
  
  QualType A = OriginalArg.OriginalArgType;
  QualType OriginalParamType = OriginalArg.OriginalParamType;
  
  // Check for type equality (top-level cv-qualifiers are ignored).
  if (Context.hasSameUnqualifiedType(A, DeducedA))
    return false;
  
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
    } else if (!DeducedAQuals.compatiblyIncludes(AQuals)) {
      return true;
    } else {        
      // Qualifiers are compatible, so have the argument type adopt the
      // deduced argument type's qualifiers as if we had performed the
      // qualification conversion.
      A = Context.getQualifiedType(A.getUnqualifiedType(), DeducedAQuals);
    }
  }
  
  //    - The transformed A can be another pointer or pointer to member 
  //      type that can be converted to the deduced A via a qualification 
  //      conversion.
  //
  // Also allow conversions which merely strip [[noreturn]] from function types
  // (recursively) as an extension.
  // FIXME: Currently, this doesn't play nicely with qualification conversions.
  bool ObjCLifetimeConversion = false;
  QualType ResultTy;
  if ((A->isAnyPointerType() || A->isMemberPointerType()) &&
      (S.IsQualificationConversion(A, DeducedA, false,
                                   ObjCLifetimeConversion) ||
       S.IsNoReturnConversion(A, DeducedA, ResultTy)))
    return false;
  
  
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
    return false;
  
  if (A->isRecordType() && isSimpleTemplateIdType(OriginalParamType) &&
      S.IsDerivedFrom(SourceLocation(), A, DeducedA))
    return false;
  
  return true;
}

/// \brief Finish template argument deduction for a function template,
/// checking the deduced template arguments for completeness and forming
/// the function template specialization.
///
/// \param OriginalCallArgs If non-NULL, the original call arguments against
/// which the deduced argument types should be compared.
Sema::TemplateDeductionResult
Sema::FinishTemplateArgumentDeduction(FunctionTemplateDecl *FunctionTemplate,
                       SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                      unsigned NumExplicitlySpecified,
                                      FunctionDecl *&Specialization,
                                      TemplateDeductionInfo &Info,
        SmallVectorImpl<OriginalCallArg> const *OriginalCallArgs,
                                      bool PartialOverloading) {
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);
  SFINAETrap Trap(*this);

  // Enter a new template instantiation context while we instantiate the
  // actual function declaration.
  SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(), Deduced.end());
  InstantiatingTemplate Inst(*this, Info.getLocation(), FunctionTemplate,
                             DeducedArgs,
              ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution,
                             Info);
  if (Inst.isInvalid())
    return TDK_InstantiationDepth;

  ContextRAII SavedContext(*this, FunctionTemplate->getTemplatedDecl());

  // C++ [temp.deduct.type]p2:
  //   [...] or if any template argument remains neither deduced nor
  //   explicitly specified, template argument deduction fails.
  SmallVector<TemplateArgument, 4> Builder;
  for (unsigned I = 0, N = TemplateParams->size(); I != N; ++I) {
    NamedDecl *Param = TemplateParams->getParam(I);

    if (!Deduced[I].isNull()) {
      if (I < NumExplicitlySpecified) {
        // We have already fully type-checked and converted this
        // argument, because it was explicitly-specified. Just record the
        // presence of this argument.
        Builder.push_back(Deduced[I]);
        // We may have had explicitly-specified template arguments for a
        // template parameter pack (that may or may not have been extended
        // via additional deduced arguments).
        if (Param->isParameterPack() && CurrentInstantiationScope) {
          if (CurrentInstantiationScope->getPartiallySubstitutedPack() ==
              Param) {
            // Forget the partially-substituted pack; its substitution is now
            // complete.
            CurrentInstantiationScope->ResetPartiallySubstitutedPack();
          }
        }
        continue;
      }
      // We have deduced this argument, so it still needs to be
      // checked and converted.

      // First, for a non-type template parameter type that is
      // initialized by a declaration, we need the type of the
      // corresponding non-type template parameter.
      QualType NTTPType;
      if (NonTypeTemplateParmDecl *NTTP
                                = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        NTTPType = NTTP->getType();
        if (NTTPType->isDependentType()) {
          TemplateArgumentList TemplateArgs(TemplateArgumentList::OnStack,
                                            Builder.data(), Builder.size());
          NTTPType = SubstType(NTTPType,
                               MultiLevelTemplateArgumentList(TemplateArgs),
                               NTTP->getLocation(),
                               NTTP->getDeclName());
          if (NTTPType.isNull()) {
            Info.Param = makeTemplateParameter(Param);
            // FIXME: These template arguments are temporary. Free them!
            Info.reset(TemplateArgumentList::CreateCopy(Context,
                                                        Builder.data(),
                                                        Builder.size()));
            return TDK_SubstitutionFailure;
          }
        }
      }

      if (ConvertDeducedTemplateArgument(*this, Param, Deduced[I],
                                         FunctionTemplate, NTTPType, 0, Info,
                                         true, Builder)) {
        Info.Param = makeTemplateParameter(Param);
        // FIXME: These template arguments are temporary. Free them!
        Info.reset(TemplateArgumentList::CreateCopy(Context, Builder.data(),
                                                    Builder.size()));
        return TDK_SubstitutionFailure;
      }

      continue;
    }

    // C++0x [temp.arg.explicit]p3:
    //    A trailing template parameter pack (14.5.3) not otherwise deduced will
    //    be deduced to an empty sequence of template arguments.
    // FIXME: Where did the word "trailing" come from?
    if (Param->isTemplateParameterPack()) {
      // We may have had explicitly-specified template arguments for this
      // template parameter pack. If so, our empty deduction extends the
      // explicitly-specified set (C++0x [temp.arg.explicit]p9).
      const TemplateArgument *ExplicitArgs;
      unsigned NumExplicitArgs;
      if (CurrentInstantiationScope &&
          CurrentInstantiationScope->getPartiallySubstitutedPack(&ExplicitArgs,
                                                             &NumExplicitArgs)
            == Param) {
        Builder.push_back(TemplateArgument(
            llvm::makeArrayRef(ExplicitArgs, NumExplicitArgs)));

        // Forget the partially-substituted pack; it's substitution is now
        // complete.
        CurrentInstantiationScope->ResetPartiallySubstitutedPack();
      } else {
        Builder.push_back(TemplateArgument::getEmptyPack());
      }
      continue;
    }

    // Substitute into the default template argument, if available.
    bool HasDefaultArg = false;
    TemplateArgumentLoc DefArg
      = SubstDefaultTemplateArgumentIfAvailable(FunctionTemplate,
                                              FunctionTemplate->getLocation(),
                                  FunctionTemplate->getSourceRange().getEnd(),
                                                Param,
                                                Builder, HasDefaultArg);

    // If there was no default argument, deduction is incomplete.
    if (DefArg.getArgument().isNull()) {
      Info.Param = makeTemplateParameter(
                         const_cast<NamedDecl *>(TemplateParams->getParam(I)));
      Info.reset(TemplateArgumentList::CreateCopy(Context, Builder.data(),
                                                  Builder.size()));
      if (PartialOverloading) break;

      return HasDefaultArg ? TDK_SubstitutionFailure : TDK_Incomplete;
    }

    // Check whether we can actually use the default argument.
    if (CheckTemplateArgument(Param, DefArg,
                              FunctionTemplate,
                              FunctionTemplate->getLocation(),
                              FunctionTemplate->getSourceRange().getEnd(),
                              0, Builder,
                              CTAK_Specified)) {
      Info.Param = makeTemplateParameter(
                         const_cast<NamedDecl *>(TemplateParams->getParam(I)));
      // FIXME: These template arguments are temporary. Free them!
      Info.reset(TemplateArgumentList::CreateCopy(Context, Builder.data(),
                                                  Builder.size()));
      return TDK_SubstitutionFailure;
    }

    // If we get here, we successfully used the default template argument.
  }

  // Form the template argument list from the deduced template arguments.
  TemplateArgumentList *DeducedArgumentList
    = TemplateArgumentList::CreateCopy(Context, Builder.data(), Builder.size());
  Info.reset(DeducedArgumentList);

  // Substitute the deduced template arguments into the function template
  // declaration to produce the function template specialization.
  DeclContext *Owner = FunctionTemplate->getDeclContext();
  if (FunctionTemplate->getFriendObjectKind())
    Owner = FunctionTemplate->getLexicalDeclContext();
  Specialization = cast_or_null<FunctionDecl>(
                      SubstDecl(FunctionTemplate->getTemplatedDecl(), Owner,
                         MultiLevelTemplateArgumentList(*DeducedArgumentList)));
  if (!Specialization || Specialization->isInvalidDecl())
    return TDK_SubstitutionFailure;

  assert(Specialization->getPrimaryTemplate()->getCanonicalDecl() ==
         FunctionTemplate->getCanonicalDecl());

  // If the template argument list is owned by the function template
  // specialization, release it.
  if (Specialization->getTemplateSpecializationArgs() == DeducedArgumentList &&
      !Trap.hasErrorOccurred())
    Info.take();

  // There may have been an error that did not prevent us from constructing a
  // declaration. Mark the declaration invalid and return with a substitution
  // failure.
  if (Trap.hasErrorOccurred()) {
    Specialization->setInvalidDecl(true);
    return TDK_SubstitutionFailure;
  }

  if (OriginalCallArgs) {
    // C++ [temp.deduct.call]p4:
    //   In general, the deduction process attempts to find template argument
    //   values that will make the deduced A identical to A (after the type A 
    //   is transformed as described above). [...]
    for (unsigned I = 0, N = OriginalCallArgs->size(); I != N; ++I) {
      OriginalCallArg OriginalArg = (*OriginalCallArgs)[I];
      unsigned ParamIdx = OriginalArg.ArgIdx;
      
      if (ParamIdx >= Specialization->getNumParams())
        continue;
      
      QualType DeducedA = Specialization->getParamDecl(ParamIdx)->getType();
      if (CheckOriginalCallArgDeduction(*this, OriginalArg, DeducedA))
        return Sema::TDK_SubstitutionFailure;
    }
  }
  
  // If we suppressed any diagnostics while performing template argument
  // deduction, and if we haven't already instantiated this declaration,
  // keep track of these diagnostics. They'll be emitted if this specialization
  // is actually used.
  if (Info.diag_begin() != Info.diag_end()) {
    SuppressedDiagnosticsMap::iterator
      Pos = SuppressedDiagnostics.find(Specialization->getCanonicalDecl());
    if (Pos == SuppressedDiagnostics.end())
        SuppressedDiagnostics[Specialization->getCanonicalDecl()]
          .append(Info.diag_begin(), Info.diag_end());
  }

  return TDK_Success;
}

/// Gets the type of a function for template-argument-deducton
/// purposes when it's considered as part of an overload set.
static QualType GetTypeOfFunction(Sema &S, const OverloadExpr::FindResult &R,
                                  FunctionDecl *Fn) {
  // We may need to deduce the return type of the function now.
  if (S.getLangOpts().CPlusPlus14 && Fn->getReturnType()->isUndeducedType() &&
      S.DeduceReturnType(Fn, R.Expression->getExprLoc(), /*Diagnose*/ false))
    return QualType();

  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Fn))
    if (Method->isInstance()) {
      // An instance method that's referenced in a form that doesn't
      // look like a member pointer is just invalid.
      if (!R.HasFormOfMemberPointer) return QualType();

      return S.Context.getMemberPointerType(Fn->getType(),
               S.Context.getTypeDeclType(Method->getParent()).getTypePtr());
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
                            bool ParamWasReference) {

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
      if (FunctionDecl *ExplicitSpec
            = S.ResolveSingleFunctionTemplateSpecialization(Ovl))
        return GetTypeOfFunction(S, R, ExplicitSpec);
    }

    return QualType();
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
        return QualType();
      
      // Otherwise, see if we can resolve a function type 
      FunctionDecl *Specialization = nullptr;
      TemplateDeductionInfo Info(Ovl->getNameLoc());
      if (S.DeduceTemplateArguments(FunTmpl, &ExplicitTemplateArgs,
                                    Specialization, Info))
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
    Sema::TemplateDeductionResult Result
      = DeduceTemplateArgumentsByTypeMatch(S, TemplateParams, ParamType,
                                           ArgType, Info, Deduced, TDF);
    if (Result) continue;
    if (!Match.isNull()) return QualType();
    Match = ArgType;
  }

  return Match;
}

/// \brief Perform the adjustments to the parameter and argument types
/// described in C++ [temp.deduct.call].
///
/// \returns true if the caller should not attempt to perform any template
/// argument deduction based on this P/A pair because the argument is an
/// overloaded function set that could not be resolved.
static bool AdjustFunctionParmAndArgTypesForDeduction(Sema &S,
                                          TemplateParameterList *TemplateParams,
                                                      QualType &ParamType,
                                                      QualType &ArgType,
                                                      Expr *Arg,
                                                      unsigned &TDF) {
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
    ArgType = ResolveOverloadForDeduction(S, TemplateParams,
                                          Arg, ParamType,
                                          ParamRefType != nullptr);
    if (ArgType.isNull())
      return true;
  }

  if (ParamRefType) {
    // If the argument has incomplete array type, try to complete its type.
    if (ArgType->isIncompleteArrayType()) {
      S.completeExprArrayBound(Arg);
      ArgType = Arg->getType();
    }

    // C++0x [temp.deduct.call]p3:
    //   If P is an rvalue reference to a cv-unqualified template
    //   parameter and the argument is an lvalue, the type "lvalue
    //   reference to A" is used in place of A for type deduction.
    if (ParamRefType->isRValueReferenceType() &&
        !ParamType.getQualifiers() &&
        isa<TemplateTypeParmType>(ParamType) &&
        Arg->isLValue())
      ArgType = S.Context.getLValueReferenceType(ArgType);
  } else {
    // C++ [temp.deduct.call]p2:
    //   If P is not a reference type:
    //   - If A is an array type, the pointer type produced by the
    //     array-to-pointer standard conversion (4.2) is used in place of
    //     A for type deduction; otherwise,
    if (ArgType->isArrayType())
      ArgType = S.Context.getArrayDecayedType(ArgType);
    //   - If A is a function type, the pointer type produced by the
    //     function-to-pointer standard conversion (4.3) is used in place
    //     of A for type deduction; otherwise,
    else if (ArgType->isFunctionType())
      ArgType = S.Context.getPointerType(ArgType);
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
      (isa<PointerType>(ParamType) &&
       isSimpleTemplateIdType(
                              ParamType->getAs<PointerType>()->getPointeeType())))
    TDF |= TDF_DerivedClass;

  return false;
}

static bool
hasDeducibleTemplateParameters(Sema &S, FunctionTemplateDecl *FunctionTemplate,
                               QualType T);

static Sema::TemplateDeductionResult DeduceTemplateArgumentByListElement(
    Sema &S, TemplateParameterList *TemplateParams, QualType ParamType,
    Expr *Arg, TemplateDeductionInfo &Info,
    SmallVectorImpl<DeducedTemplateArgument> &Deduced, unsigned TDF);

/// \brief Attempt template argument deduction from an initializer list
///        deemed to be an argument in a function call.
static bool
DeduceFromInitializerList(Sema &S, TemplateParameterList *TemplateParams,
                          QualType AdjustedParamType, InitListExpr *ILE,
                          TemplateDeductionInfo &Info,
                          SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                          unsigned TDF, Sema::TemplateDeductionResult &Result) {

  // [temp.deduct.call] p1 (post CWG-1591)
  // If removing references and cv-qualifiers from P gives
  // std::initializer_list<P0> or P0[N] for some P0 and N and the argument is a
  // non-empty initializer list (8.5.4), then deduction is performed instead for
  // each element of the initializer list, taking P0 as a function template
  // parameter type and the initializer element as its argument, and in the
  // P0[N] case, if N is a non-type template parameter, N is deduced from the
  // length of the initializer list. Otherwise, an initializer list argument
  // causes the parameter to be considered a non-deduced context

  const bool IsConstSizedArray = AdjustedParamType->isConstantArrayType();

  const bool IsDependentSizedArray =
      !IsConstSizedArray && AdjustedParamType->isDependentSizedArrayType();

  QualType ElTy;  // The element type of the std::initializer_list or the array.

  const bool IsSTDList = !IsConstSizedArray && !IsDependentSizedArray &&
                         S.isStdInitializerList(AdjustedParamType, &ElTy);

  if (!IsConstSizedArray && !IsDependentSizedArray && !IsSTDList)
    return false;

  Result = Sema::TDK_Success;
  // If we are not deducing against the 'T' in a std::initializer_list<T> then
  // deduce against the 'T' in T[N].
  if (ElTy.isNull()) {
    assert(!IsSTDList);
    ElTy = S.Context.getAsArrayType(AdjustedParamType)->getElementType();
  }
  // Deduction only needs to be done for dependent types.
  if (ElTy->isDependentType()) {
    for (Expr *E : ILE->inits()) {
      if ((Result = DeduceTemplateArgumentByListElement(S, TemplateParams, ElTy,
                                                        E, Info, Deduced, TDF)))
        return true;
    }
  }
  if (IsDependentSizedArray) {
    const DependentSizedArrayType *ArrTy =
        S.Context.getAsDependentSizedArrayType(AdjustedParamType);
    // Determine the array bound is something we can deduce.
    if (NonTypeTemplateParmDecl *NTTP =
            getDeducedParameterFromExpr(ArrTy->getSizeExpr())) {
      // We can perform template argument deduction for the given non-type
      // template parameter.
      assert(NTTP->getDepth() == 0 &&
             "Cannot deduce non-type template argument at depth > 0");
      llvm::APInt Size(S.Context.getIntWidth(NTTP->getType()),
                       ILE->getNumInits());

      Result = DeduceNonTypeTemplateArgument(
          S, NTTP, llvm::APSInt(Size), NTTP->getType(),
          /*ArrayBound=*/true, Info, Deduced);
    }
  }
  return true;
}

/// \brief Perform template argument deduction by matching a parameter type
///        against a single expression, where the expression is an element of
///        an initializer list that was originally matched against a parameter
///        of type \c initializer_list\<ParamType\>.
static Sema::TemplateDeductionResult
DeduceTemplateArgumentByListElement(Sema &S,
                                    TemplateParameterList *TemplateParams,
                                    QualType ParamType, Expr *Arg,
                                    TemplateDeductionInfo &Info,
                              SmallVectorImpl<DeducedTemplateArgument> &Deduced,
                                    unsigned TDF) {
  // Handle the case where an init list contains another init list as the
  // element.
  if (InitListExpr *ILE = dyn_cast<InitListExpr>(Arg)) {
    Sema::TemplateDeductionResult Result;
    if (!DeduceFromInitializerList(S, TemplateParams,
                                   ParamType.getNonReferenceType(), ILE, Info,
                                   Deduced, TDF, Result))
      return Sema::TDK_Success; // Just ignore this expression.

    return Result;
  }

  // For all other cases, just match by type.
  QualType ArgType = Arg->getType();
  if (AdjustFunctionParmAndArgTypesForDeduction(S, TemplateParams, ParamType, 
                                                ArgType, Arg, TDF)) {
    Info.Expression = Arg;
    return Sema::TDK_FailedOverloadResolution;
  }
  return DeduceTemplateArgumentsByTypeMatch(S, TemplateParams, ParamType,
                                            ArgType, Info, Deduced, TDF);
}

/// \brief Perform template argument deduction from a function call
/// (C++ [temp.deduct.call]).
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param ExplicitTemplateArgs the explicit template arguments provided
/// for this call.
///
/// \param Args the function call arguments
///
/// \param Specialization if template argument deduction was successful,
/// this will be set to the function template specialization produced by
/// template argument deduction.
///
/// \param Info the argument will be updated to provide additional information
/// about template argument deduction.
///
/// \returns the result of template argument deduction.
Sema::TemplateDeductionResult Sema::DeduceTemplateArguments(
    FunctionTemplateDecl *FunctionTemplate,
    TemplateArgumentListInfo *ExplicitTemplateArgs, ArrayRef<Expr *> Args,
    FunctionDecl *&Specialization, TemplateDeductionInfo &Info,
    bool PartialOverloading) {
  if (FunctionTemplate->isInvalidDecl())
    return TDK_Invalid;

  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  unsigned NumParams = Function->getNumParams();

  // C++ [temp.deduct.call]p1:
  //   Template argument deduction is done by comparing each function template
  //   parameter type (call it P) with the type of the corresponding argument
  //   of the call (call it A) as described below.
  unsigned CheckArgs = Args.size();
  if (Args.size() < Function->getMinRequiredArguments() && !PartialOverloading)
    return TDK_TooFewArguments;
  else if (TooManyArguments(NumParams, Args.size(), PartialOverloading)) {
    const FunctionProtoType *Proto
      = Function->getType()->getAs<FunctionProtoType>();
    if (Proto->isTemplateVariadic())
      /* Do nothing */;
    else if (Proto->isVariadic())
      CheckArgs = NumParams;
    else
      return TDK_TooManyArguments;
  }

  // The types of the parameters from which we will perform template argument
  // deduction.
  LocalInstantiationScope InstScope(*this);
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  SmallVector<QualType, 4> ParamTypes;
  unsigned NumExplicitlySpecified = 0;
  if (ExplicitTemplateArgs) {
    TemplateDeductionResult Result =
      SubstituteExplicitTemplateArguments(FunctionTemplate,
                                          *ExplicitTemplateArgs,
                                          Deduced,
                                          ParamTypes,
                                          nullptr,
                                          Info);
    if (Result)
      return Result;

    NumExplicitlySpecified = Deduced.size();
  } else {
    // Just fill in the parameter types from the function declaration.
    for (unsigned I = 0; I != NumParams; ++I)
      ParamTypes.push_back(Function->getParamDecl(I)->getType());
  }

  // Deduce template arguments from the function parameters.
  Deduced.resize(TemplateParams->size());
  unsigned ArgIdx = 0;
  SmallVector<OriginalCallArg, 4> OriginalCallArgs;
  for (unsigned ParamIdx = 0, NumParamTypes = ParamTypes.size();
       ParamIdx != NumParamTypes; ++ParamIdx) {
    QualType OrigParamType = ParamTypes[ParamIdx];
    QualType ParamType = OrigParamType;
    
    const PackExpansionType *ParamExpansion
      = dyn_cast<PackExpansionType>(ParamType);
    if (!ParamExpansion) {
      // Simple case: matching a function parameter to a function argument.
      if (ArgIdx >= CheckArgs)
        break;

      Expr *Arg = Args[ArgIdx++];
      QualType ArgType = Arg->getType();
      
      unsigned TDF = 0;
      if (AdjustFunctionParmAndArgTypesForDeduction(*this, TemplateParams,
                                                    ParamType, ArgType, Arg,
                                                    TDF))
        continue;

      // If we have nothing to deduce, we're done.
      if (!hasDeducibleTemplateParameters(*this, FunctionTemplate, ParamType))
        continue;

      // If the argument is an initializer list ...
      if (InitListExpr *ILE = dyn_cast<InitListExpr>(Arg)) {
        TemplateDeductionResult Result;
        // Removing references was already done.
        if (!DeduceFromInitializerList(*this, TemplateParams, ParamType, ILE,
                                       Info, Deduced, TDF, Result))
          continue;

        if (Result)
          return Result;
        // Don't track the argument type, since an initializer list has none.
        continue;
      }

      // Keep track of the argument type and corresponding parameter index,
      // so we can check for compatibility between the deduced A and A.
      OriginalCallArgs.push_back(OriginalCallArg(OrigParamType, ArgIdx-1, 
                                                 ArgType));

      if (TemplateDeductionResult Result
            = DeduceTemplateArgumentsByTypeMatch(*this, TemplateParams,
                                                 ParamType, ArgType,
                                                 Info, Deduced, TDF))
        return Result;

      continue;
    }

    // C++0x [temp.deduct.call]p1:
    //   For a function parameter pack that occurs at the end of the
    //   parameter-declaration-list, the type A of each remaining argument of
    //   the call is compared with the type P of the declarator-id of the
    //   function parameter pack. Each comparison deduces template arguments
    //   for subsequent positions in the template parameter packs expanded by
    //   the function parameter pack. For a function parameter pack that does
    //   not occur at the end of the parameter-declaration-list, the type of
    //   the parameter pack is a non-deduced context.
    if (ParamIdx + 1 < NumParamTypes)
      break;

    QualType ParamPattern = ParamExpansion->getPattern();
    PackDeductionScope PackScope(*this, TemplateParams, Deduced, Info,
                                 ParamPattern);

    bool HasAnyArguments = false;
    for (; ArgIdx < Args.size(); ++ArgIdx) {
      HasAnyArguments = true;

      QualType OrigParamType = ParamPattern;
      ParamType = OrigParamType;
      Expr *Arg = Args[ArgIdx];
      QualType ArgType = Arg->getType();

      unsigned TDF = 0;
      if (AdjustFunctionParmAndArgTypesForDeduction(*this, TemplateParams,
                                                    ParamType, ArgType, Arg,
                                                    TDF)) {
        // We can't actually perform any deduction for this argument, so stop
        // deduction at this point.
        ++ArgIdx;
        break;
      }

      // As above, initializer lists need special handling.
      if (InitListExpr *ILE = dyn_cast<InitListExpr>(Arg)) {
        TemplateDeductionResult Result;
        if (!DeduceFromInitializerList(*this, TemplateParams, ParamType, ILE,
                                       Info, Deduced, TDF, Result)) {
          ++ArgIdx;
          break;
        }

        if (Result)
          return Result;
      } else {

        // Keep track of the argument type and corresponding argument index,
        // so we can check for compatibility between the deduced A and A.
        if (hasDeducibleTemplateParameters(*this, FunctionTemplate, ParamType))
          OriginalCallArgs.push_back(OriginalCallArg(OrigParamType, ArgIdx, 
                                                     ArgType));

        if (TemplateDeductionResult Result
            = DeduceTemplateArgumentsByTypeMatch(*this, TemplateParams,
                                                 ParamType, ArgType, Info,
                                                 Deduced, TDF))
          return Result;
      }

      PackScope.nextPackElement();
    }

    // Build argument packs for each of the parameter packs expanded by this
    // pack expansion.
    if (auto Result = PackScope.finish(HasAnyArguments))
      return Result;

    // After we've matching against a parameter pack, we're done.
    break;
  }

  return FinishTemplateArgumentDeduction(FunctionTemplate, Deduced,
                                         NumExplicitlySpecified, Specialization,
                                         Info, &OriginalCallArgs,
                                         PartialOverloading);
}

QualType Sema::adjustCCAndNoReturn(QualType ArgFunctionType,
                                   QualType FunctionType) {
  if (ArgFunctionType.isNull())
    return ArgFunctionType;

  const FunctionProtoType *FunctionTypeP =
      FunctionType->castAs<FunctionProtoType>();
  CallingConv CC = FunctionTypeP->getCallConv();
  bool NoReturn = FunctionTypeP->getNoReturnAttr();
  const FunctionProtoType *ArgFunctionTypeP =
      ArgFunctionType->getAs<FunctionProtoType>();
  if (ArgFunctionTypeP->getCallConv() == CC &&
      ArgFunctionTypeP->getNoReturnAttr() == NoReturn)
    return ArgFunctionType;

  FunctionType::ExtInfo EI = ArgFunctionTypeP->getExtInfo().withCallingConv(CC);
  EI = EI.withNoReturn(NoReturn);
  ArgFunctionTypeP =
      cast<FunctionProtoType>(Context.adjustFunctionType(ArgFunctionTypeP, EI));
  return QualType(ArgFunctionTypeP, 0);
}

/// \brief Deduce template arguments when taking the address of a function
/// template (C++ [temp.deduct.funcaddr]) or matching a specialization to
/// a template.
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param ExplicitTemplateArgs the explicitly-specified template
/// arguments.
///
/// \param ArgFunctionType the function type that will be used as the
/// "argument" type (A) when performing template argument deduction from the
/// function template's function type. This type may be NULL, if there is no
/// argument type to compare against, in C++0x [temp.arg.explicit]p3.
///
/// \param Specialization if template argument deduction was successful,
/// this will be set to the function template specialization produced by
/// template argument deduction.
///
/// \param Info the argument will be updated to provide additional information
/// about template argument deduction.
///
/// \returns the result of template argument deduction.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                              TemplateArgumentListInfo *ExplicitTemplateArgs,
                              QualType ArgFunctionType,
                              FunctionDecl *&Specialization,
                              TemplateDeductionInfo &Info,
                              bool InOverloadResolution) {
  if (FunctionTemplate->isInvalidDecl())
    return TDK_Invalid;

  FunctionDecl *Function = FunctionTemplate->getTemplatedDecl();
  TemplateParameterList *TemplateParams
    = FunctionTemplate->getTemplateParameters();
  QualType FunctionType = Function->getType();
  if (!InOverloadResolution)
    ArgFunctionType = adjustCCAndNoReturn(ArgFunctionType, FunctionType);

  // Substitute any explicit template arguments.
  LocalInstantiationScope InstScope(*this);
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  unsigned NumExplicitlySpecified = 0;
  SmallVector<QualType, 4> ParamTypes;
  if (ExplicitTemplateArgs) {
    if (TemplateDeductionResult Result
          = SubstituteExplicitTemplateArguments(FunctionTemplate,
                                                *ExplicitTemplateArgs,
                                                Deduced, ParamTypes,
                                                &FunctionType, Info))
      return Result;

    NumExplicitlySpecified = Deduced.size();
  }

  // Unevaluated SFINAE context.
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);
  SFINAETrap Trap(*this);

  Deduced.resize(TemplateParams->size());

  // If the function has a deduced return type, substitute it for a dependent
  // type so that we treat it as a non-deduced context in what follows.
  bool HasDeducedReturnType = false;
  if (getLangOpts().CPlusPlus14 && InOverloadResolution &&
      Function->getReturnType()->getContainedAutoType()) {
    FunctionType = SubstAutoType(FunctionType, Context.DependentTy);
    HasDeducedReturnType = true;
  }

  if (!ArgFunctionType.isNull()) {
    unsigned TDF = TDF_TopLevelParameterTypeList;
    if (InOverloadResolution) TDF |= TDF_InOverloadResolution;
    // Deduce template arguments from the function type.
    if (TemplateDeductionResult Result
          = DeduceTemplateArgumentsByTypeMatch(*this, TemplateParams,
                                               FunctionType, ArgFunctionType,
                                               Info, Deduced, TDF))
      return Result;
  }

  if (TemplateDeductionResult Result
        = FinishTemplateArgumentDeduction(FunctionTemplate, Deduced,
                                          NumExplicitlySpecified,
                                          Specialization, Info))
    return Result;

  // If the function has a deduced return type, deduce it now, so we can check
  // that the deduced function type matches the requested type.
  if (HasDeducedReturnType &&
      Specialization->getReturnType()->isUndeducedType() &&
      DeduceReturnType(Specialization, Info.getLocation(), false))
    return TDK_MiscellaneousDeductionFailure;

  // If the requested function type does not match the actual type of the
  // specialization with respect to arguments of compatible pointer to function
  // types, template argument deduction fails.
  if (!ArgFunctionType.isNull()) {
    if (InOverloadResolution && !isSameOrCompatibleFunctionType(
                           Context.getCanonicalType(Specialization->getType()),
                           Context.getCanonicalType(ArgFunctionType)))
      return TDK_MiscellaneousDeductionFailure;
    else if(!InOverloadResolution &&
            !Context.hasSameType(Specialization->getType(), ArgFunctionType))
      return TDK_MiscellaneousDeductionFailure;
  }

  return TDK_Success;
}

/// \brief Given a function declaration (e.g. a generic lambda conversion 
///  function) that contains an 'auto' in its result type, substitute it 
///  with TypeToReplaceAutoWith.  Be careful to pass in the type you want
///  to replace 'auto' with and not the actual result type you want
///  to set the function to.
static inline void 
SubstAutoWithinFunctionReturnType(FunctionDecl *F, 
                                    QualType TypeToReplaceAutoWith, Sema &S) {
  assert(!TypeToReplaceAutoWith->getContainedAutoType());
  QualType AutoResultType = F->getReturnType();
  assert(AutoResultType->getContainedAutoType()); 
  QualType DeducedResultType = S.SubstAutoType(AutoResultType, 
                                               TypeToReplaceAutoWith);
  S.Context.adjustDeducedFunctionResultType(F, DeducedResultType);
}

/// \brief Given a specialized conversion operator of a generic lambda 
/// create the corresponding specializations of the call operator and 
/// the static-invoker. If the return type of the call operator is auto, 
/// deduce its return type and check if that matches the 
/// return type of the destination function ptr.

static inline Sema::TemplateDeductionResult 
SpecializeCorrespondingLambdaCallOperatorAndInvoker(
    CXXConversionDecl *ConversionSpecialized,
    SmallVectorImpl<DeducedTemplateArgument> &DeducedArguments,
    QualType ReturnTypeOfDestFunctionPtr,
    TemplateDeductionInfo &TDInfo,
    Sema &S) {
  
  CXXRecordDecl *LambdaClass = ConversionSpecialized->getParent();
  assert(LambdaClass && LambdaClass->isGenericLambda()); 
  
  CXXMethodDecl *CallOpGeneric = LambdaClass->getLambdaCallOperator();
  QualType CallOpResultType = CallOpGeneric->getReturnType();
  const bool GenericLambdaCallOperatorHasDeducedReturnType = 
      CallOpResultType->getContainedAutoType();
  
  FunctionTemplateDecl *CallOpTemplate = 
      CallOpGeneric->getDescribedFunctionTemplate();

  FunctionDecl *CallOpSpecialized = nullptr;
  // Use the deduced arguments of the conversion function, to specialize our 
  // generic lambda's call operator.
  if (Sema::TemplateDeductionResult Result
      = S.FinishTemplateArgumentDeduction(CallOpTemplate, 
                                          DeducedArguments, 
                                          0, CallOpSpecialized, TDInfo))
    return Result;
 
  // If we need to deduce the return type, do so (instantiates the callop).
  if (GenericLambdaCallOperatorHasDeducedReturnType &&
      CallOpSpecialized->getReturnType()->isUndeducedType())
    S.DeduceReturnType(CallOpSpecialized, 
                       CallOpSpecialized->getPointOfInstantiation(),
                       /*Diagnose*/ true);
    
  // Check to see if the return type of the destination ptr-to-function
  // matches the return type of the call operator.
  if (!S.Context.hasSameType(CallOpSpecialized->getReturnType(),
                             ReturnTypeOfDestFunctionPtr))
    return Sema::TDK_NonDeducedMismatch;
  // Since we have succeeded in matching the source and destination
  // ptr-to-functions (now including return type), and have successfully 
  // specialized our corresponding call operator, we are ready to
  // specialize the static invoker with the deduced arguments of our
  // ptr-to-function.
  FunctionDecl *InvokerSpecialized = nullptr;
  FunctionTemplateDecl *InvokerTemplate = LambdaClass->
                  getLambdaStaticInvoker()->getDescribedFunctionTemplate();

#ifndef NDEBUG
  Sema::TemplateDeductionResult LLVM_ATTRIBUTE_UNUSED Result =
#endif
    S.FinishTemplateArgumentDeduction(InvokerTemplate, DeducedArguments, 0, 
          InvokerSpecialized, TDInfo);
  assert(Result == Sema::TDK_Success && 
    "If the call operator succeeded so should the invoker!");
  // Set the result type to match the corresponding call operator
  // specialization's result type.
  if (GenericLambdaCallOperatorHasDeducedReturnType &&
      InvokerSpecialized->getReturnType()->isUndeducedType()) {
    // Be sure to get the type to replace 'auto' with and not
    // the full result type of the call op specialization 
    // to substitute into the 'auto' of the invoker and conversion
    // function.
    // For e.g.
    //  int* (*fp)(int*) = [](auto* a) -> auto* { return a; };
    // We don't want to subst 'int*' into 'auto' to get int**.

    QualType TypeToReplaceAutoWith = CallOpSpecialized->getReturnType()
                                         ->getContainedAutoType()
                                         ->getDeducedType();
    SubstAutoWithinFunctionReturnType(InvokerSpecialized,
        TypeToReplaceAutoWith, S);
    SubstAutoWithinFunctionReturnType(ConversionSpecialized, 
        TypeToReplaceAutoWith, S);
  }
    
  // Ensure that static invoker doesn't have a const qualifier.
  // FIXME: When creating the InvokerTemplate in SemaLambda.cpp 
  // do not use the CallOperator's TypeSourceInfo which allows
  // the const qualifier to leak through. 
  const FunctionProtoType *InvokerFPT = InvokerSpecialized->
                  getType().getTypePtr()->castAs<FunctionProtoType>();
  FunctionProtoType::ExtProtoInfo EPI = InvokerFPT->getExtProtoInfo();
  EPI.TypeQuals = 0;
  InvokerSpecialized->setType(S.Context.getFunctionType(
      InvokerFPT->getReturnType(), InvokerFPT->getParamTypes(), EPI));
  return Sema::TDK_Success;
}
/// \brief Deduce template arguments for a templated conversion
/// function (C++ [temp.deduct.conv]) and, if successful, produce a
/// conversion function template specialization.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *ConversionTemplate,
                              QualType ToType,
                              CXXConversionDecl *&Specialization,
                              TemplateDeductionInfo &Info) {
  if (ConversionTemplate->isInvalidDecl())
    return TDK_Invalid;

  CXXConversionDecl *ConversionGeneric
    = cast<CXXConversionDecl>(ConversionTemplate->getTemplatedDecl());

  QualType FromType = ConversionGeneric->getConversionType();

  // Canonicalize the types for deduction.
  QualType P = Context.getCanonicalType(FromType);
  QualType A = Context.getCanonicalType(ToType);

  // C++0x [temp.deduct.conv]p2:
  //   If P is a reference type, the type referred to by P is used for
  //   type deduction.
  if (const ReferenceType *PRef = P->getAs<ReferenceType>())
    P = PRef->getPointeeType();

  // C++0x [temp.deduct.conv]p4:
  //   [...] If A is a reference type, the type referred to by A is used
  //   for type deduction.
  if (const ReferenceType *ARef = A->getAs<ReferenceType>())
    A = ARef->getPointeeType().getUnqualifiedType();
  // C++ [temp.deduct.conv]p3:
  //
  //   If A is not a reference type:
  else {
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
  EnterExpressionEvaluationContext Unevaluated(*this, Sema::Unevaluated);
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
  if (ToType->isReferenceType())
    TDF |= TDF_ParamWithReferenceType;
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
  if (TemplateDeductionResult Result
        = DeduceTemplateArgumentsByTypeMatch(*this, TemplateParams,
                                             P, A, Info, Deduced, TDF))
    return Result;

  // Create an Instantiation Scope for finalizing the operator.
  LocalInstantiationScope InstScope(*this);
  // Finish template argument deduction.
  FunctionDecl *ConversionSpecialized = nullptr;
  TemplateDeductionResult Result
      = FinishTemplateArgumentDeduction(ConversionTemplate, Deduced, 0, 
                                        ConversionSpecialized, Info);
  Specialization = cast_or_null<CXXConversionDecl>(ConversionSpecialized);

  // If the conversion operator is being invoked on a lambda closure to convert
  // to a ptr-to-function, use the deduced arguments from the conversion
  // function to specialize the corresponding call operator.
  //   e.g., int (*fp)(int) = [](auto a) { return a; };
  if (Result == TDK_Success && isLambdaConversionOperator(ConversionGeneric)) {
    
    // Get the return type of the destination ptr-to-function we are converting
    // to.  This is necessary for matching the lambda call operator's return 
    // type to that of the destination ptr-to-function's return type.
    assert(A->isPointerType() && 
        "Can only convert from lambda to ptr-to-function");
    const FunctionType *ToFunType = 
        A->getPointeeType().getTypePtr()->getAs<FunctionType>();
    const QualType DestFunctionPtrReturnType = ToFunType->getReturnType();

    // Create the corresponding specializations of the call operator and 
    // the static-invoker; and if the return type is auto, 
    // deduce the return type and check if it matches the 
    // DestFunctionPtrReturnType.
    // For instance:
    //   auto L = [](auto a) { return f(a); };
    //   int (*fp)(int) = L;
    //   char (*fp2)(int) = L; <-- Not OK.

    Result = SpecializeCorrespondingLambdaCallOperatorAndInvoker(
        Specialization, Deduced, DestFunctionPtrReturnType, 
        Info, *this);
  }
  return Result;
}

/// \brief Deduce template arguments for a function template when there is
/// nothing to deduce against (C++0x [temp.arg.explicit]p3).
///
/// \param FunctionTemplate the function template for which we are performing
/// template argument deduction.
///
/// \param ExplicitTemplateArgs the explicitly-specified template
/// arguments.
///
/// \param Specialization if template argument deduction was successful,
/// this will be set to the function template specialization produced by
/// template argument deduction.
///
/// \param Info the argument will be updated to provide additional information
/// about template argument deduction.
///
/// \returns the result of template argument deduction.
Sema::TemplateDeductionResult
Sema::DeduceTemplateArguments(FunctionTemplateDecl *FunctionTemplate,
                              TemplateArgumentListInfo *ExplicitTemplateArgs,
                              FunctionDecl *&Specialization,
                              TemplateDeductionInfo &Info,
                              bool InOverloadResolution) {
  return DeduceTemplateArguments(FunctionTemplate, ExplicitTemplateArgs,
                                 QualType(), Specialization, Info,
                                 InOverloadResolution);
}

namespace {
  /// Substitute the 'auto' type specifier within a type for a given replacement
  /// type.
  class SubstituteAutoTransform :
    public TreeTransform<SubstituteAutoTransform> {
    QualType Replacement;
  public:
    SubstituteAutoTransform(Sema &SemaRef, QualType Replacement)
        : TreeTransform<SubstituteAutoTransform>(SemaRef),
          Replacement(Replacement) {}

    QualType TransformAutoType(TypeLocBuilder &TLB, AutoTypeLoc TL) {
      // If we're building the type pattern to deduce against, don't wrap the
      // substituted type in an AutoType. Certain template deduction rules
      // apply only when a template type parameter appears directly (and not if
      // the parameter is found through desugaring). For instance:
      //   auto &&lref = lvalue;
      // must transform into "rvalue reference to T" not "rvalue reference to
      // auto type deduced as T" in order for [temp.deduct.call]p3 to apply.
      if (!Replacement.isNull() && isa<TemplateTypeParmType>(Replacement)) {
        QualType Result = Replacement;
        TemplateTypeParmTypeLoc NewTL =
          TLB.push<TemplateTypeParmTypeLoc>(Result);
        NewTL.setNameLoc(TL.getNameLoc());
        return Result;
      } else {
        bool Dependent =
          !Replacement.isNull() && Replacement->isDependentType();
        QualType Result =
          SemaRef.Context.getAutoType(Dependent ? QualType() : Replacement,
                                      TL.getTypePtr()->getKeyword(),
                                      Dependent);
        AutoTypeLoc NewTL = TLB.push<AutoTypeLoc>(Result);
        NewTL.setNameLoc(TL.getNameLoc());
        return Result;
      }
    }

    ExprResult TransformLambdaExpr(LambdaExpr *E) {
      // Lambdas never need to be transformed.
      return E;
    }

    QualType Apply(TypeLoc TL) {
      // Create some scratch storage for the transformed type locations.
      // FIXME: We're just going to throw this information away. Don't build it.
      TypeLocBuilder TLB;
      TLB.reserve(TL.getFullDataSize());
      return TransformType(TLB, TL);
    }
  };
}

Sema::DeduceAutoResult
Sema::DeduceAutoType(TypeSourceInfo *Type, Expr *&Init, QualType &Result) {
  return DeduceAutoType(Type->getTypeLoc(), Init, Result);
}

/// \brief Deduce the type for an auto type-specifier (C++11 [dcl.spec.auto]p6)
///
/// \param Type the type pattern using the auto type-specifier.
/// \param Init the initializer for the variable whose type is to be deduced.
/// \param Result if type deduction was successful, this will be set to the
///        deduced type.
Sema::DeduceAutoResult
Sema::DeduceAutoType(TypeLoc Type, Expr *&Init, QualType &Result) {
  if (Init->getType()->isNonOverloadPlaceholderType()) {
    ExprResult NonPlaceholder = CheckPlaceholderExpr(Init);
    if (NonPlaceholder.isInvalid())
      return DAR_FailedAlreadyDiagnosed;
    Init = NonPlaceholder.get();
  }

  if (Init->isTypeDependent() || Type.getType()->isDependentType()) {
    Result = SubstituteAutoTransform(*this, Context.DependentTy).Apply(Type);
    assert(!Result.isNull() && "substituting DependentTy can't fail");
    return DAR_Succeeded;
  }

  // If this is a 'decltype(auto)' specifier, do the decltype dance.
  // Since 'decltype(auto)' can only occur at the top of the type, we
  // don't need to go digging for it.
  if (const AutoType *AT = Type.getType()->getAs<AutoType>()) {
    if (AT->isDecltypeAuto()) {
      if (isa<InitListExpr>(Init)) {
        Diag(Init->getLocStart(), diag::err_decltype_auto_initializer_list);
        return DAR_FailedAlreadyDiagnosed;
      }

      QualType Deduced = BuildDecltypeType(Init, Init->getLocStart(), false);
      if (Deduced.isNull())
        return DAR_FailedAlreadyDiagnosed;
      // FIXME: Support a non-canonical deduced type for 'auto'.
      Deduced = Context.getCanonicalType(Deduced);
      Result = SubstituteAutoTransform(*this, Deduced).Apply(Type);
      if (Result.isNull())
        return DAR_FailedAlreadyDiagnosed;
      return DAR_Succeeded;
    } else if (!getLangOpts().CPlusPlus) {
      if (isa<InitListExpr>(Init)) {
        Diag(Init->getLocStart(), diag::err_auto_init_list_from_c);
        return DAR_FailedAlreadyDiagnosed;
      }
    }
  }

  SourceLocation Loc = Init->getExprLoc();

  LocalInstantiationScope InstScope(*this);

  // Build template<class TemplParam> void Func(FuncParam);
  TemplateTypeParmDecl *TemplParam =
    TemplateTypeParmDecl::Create(Context, nullptr, SourceLocation(), Loc, 0, 0,
                                 nullptr, false, false);
  QualType TemplArg = QualType(TemplParam->getTypeForDecl(), 0);
  NamedDecl *TemplParamPtr = TemplParam;
  FixedSizeTemplateParameterListStorage<1> TemplateParamsSt(
      Loc, Loc, TemplParamPtr, Loc);

  QualType FuncParam = SubstituteAutoTransform(*this, TemplArg).Apply(Type);
  assert(!FuncParam.isNull() &&
         "substituting template parameter for 'auto' failed");

  // Deduce type of TemplParam in Func(Init)
  SmallVector<DeducedTemplateArgument, 1> Deduced;
  Deduced.resize(1);
  QualType InitType = Init->getType();
  unsigned TDF = 0;

  TemplateDeductionInfo Info(Loc);

  InitListExpr *InitList = dyn_cast<InitListExpr>(Init);
  if (InitList) {
    for (unsigned i = 0, e = InitList->getNumInits(); i < e; ++i) {
      if (DeduceTemplateArgumentByListElement(*this, TemplateParamsSt.get(),
                                              TemplArg, InitList->getInit(i),
                                              Info, Deduced, TDF))
        return DAR_Failed;
    }
  } else {
    if (!getLangOpts().CPlusPlus && Init->refersToBitField()) {
      Diag(Loc, diag::err_auto_bitfield);
      return DAR_FailedAlreadyDiagnosed;
    }

    if (AdjustFunctionParmAndArgTypesForDeduction(
            *this, TemplateParamsSt.get(), FuncParam, InitType, Init, TDF))
      return DAR_Failed;

    if (DeduceTemplateArgumentsByTypeMatch(*this, TemplateParamsSt.get(),
                                           FuncParam, InitType, Info, Deduced,
                                           TDF))
      return DAR_Failed;
  }

  if (Deduced[0].getKind() != TemplateArgument::Type)
    return DAR_Failed;

  QualType DeducedType = Deduced[0].getAsType();

  if (InitList) {
    DeducedType = BuildStdInitializerList(DeducedType, Loc);
    if (DeducedType.isNull())
      return DAR_FailedAlreadyDiagnosed;
  }

  Result = SubstituteAutoTransform(*this, DeducedType).Apply(Type);
  if (Result.isNull())
   return DAR_FailedAlreadyDiagnosed;

  // Check that the deduced argument type is compatible with the original
  // argument type per C++ [temp.deduct.call]p4.
  if (!InitList && !Result.isNull() &&
      CheckOriginalCallArgDeduction(*this,
                                    Sema::OriginalCallArg(FuncParam,0,InitType),
                                    Result)) {
    Result = QualType();
    return DAR_Failed;
  }

  return DAR_Succeeded;
}

QualType Sema::SubstAutoType(QualType TypeWithAuto, 
                             QualType TypeToReplaceAuto) {
  return SubstituteAutoTransform(*this, TypeToReplaceAuto).
               TransformType(TypeWithAuto);
}

TypeSourceInfo* Sema::SubstAutoTypeSourceInfo(TypeSourceInfo *TypeWithAuto, 
                             QualType TypeToReplaceAuto) {
    return SubstituteAutoTransform(*this, TypeToReplaceAuto).
               TransformType(TypeWithAuto);
}

void Sema::DiagnoseAutoDeductionFailure(VarDecl *VDecl, Expr *Init) {
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

  if (FD->getTemplateInstantiationPattern())
    InstantiateFunctionDefinition(Loc, FD);

  bool StillUndeduced = FD->getReturnType()->isUndeducedType();
  if (StillUndeduced && Diagnose && !FD->isInvalidDecl()) {
    Diag(Loc, diag::err_auto_fn_used_before_defined) << FD;
    Diag(FD->getLocation(), diag::note_callee_decl) << FD;
  }

  return StillUndeduced;
}

static void
MarkUsedTemplateParameters(ASTContext &Ctx, QualType T,
                           bool OnlyDeduced,
                           unsigned Level,
                           llvm::SmallBitVector &Deduced);

/// \brief If this is a non-static member function,
static void
AddImplicitObjectParameterType(ASTContext &Context,
                               CXXMethodDecl *Method,
                               SmallVectorImpl<QualType> &ArgTypes) {
  // C++11 [temp.func.order]p3:
  //   [...] The new parameter is of type "reference to cv A," where cv are
  //   the cv-qualifiers of the function template (if any) and A is
  //   the class of which the function template is a member.
  //
  // The standard doesn't say explicitly, but we pick the appropriate kind of
  // reference type based on [over.match.funcs]p4.
  QualType ArgTy = Context.getTypeDeclType(Method->getParent());
  ArgTy = Context.getQualifiedType(ArgTy,
                        Qualifiers::fromCVRMask(Method->getTypeQualifiers()));
  if (Method->getRefQualifier() == RQ_RValue)
    ArgTy = Context.getRValueReferenceType(ArgTy);
  else
    ArgTy = Context.getLValueReferenceType(ArgTy);
  ArgTypes.push_back(ArgTy);
}

/// \brief Determine whether the function template \p FT1 is at least as
/// specialized as \p FT2.
static bool isAtLeastAsSpecializedAs(Sema &S,
                                     SourceLocation Loc,
                                     FunctionTemplateDecl *FT1,
                                     FunctionTemplateDecl *FT2,
                                     TemplatePartialOrderingContext TPOC,
                                     unsigned NumCallArguments1) {
  FunctionDecl *FD1 = FT1->getTemplatedDecl();
  FunctionDecl *FD2 = FT2->getTemplatedDecl();
  const FunctionProtoType *Proto1 = FD1->getType()->getAs<FunctionProtoType>();
  const FunctionProtoType *Proto2 = FD2->getType()->getAs<FunctionProtoType>();

  assert(Proto1 && Proto2 && "Function templates must have prototypes");
  TemplateParameterList *TemplateParams = FT2->getTemplateParameters();
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  Deduced.resize(TemplateParams->size());

  // C++0x [temp.deduct.partial]p3:
  //   The types used to determine the ordering depend on the context in which
  //   the partial ordering is done:
  TemplateDeductionInfo Info(Loc);
  SmallVector<QualType, 4> Args2;
  switch (TPOC) {
  case TPOC_Call: {
    //   - In the context of a function call, the function parameter types are
    //     used.
    CXXMethodDecl *Method1 = dyn_cast<CXXMethodDecl>(FD1);
    CXXMethodDecl *Method2 = dyn_cast<CXXMethodDecl>(FD2);

    // C++11 [temp.func.order]p3:
    //   [...] If only one of the function templates is a non-static
    //   member, that function template is considered to have a new
    //   first parameter inserted in its function parameter list. The
    //   new parameter is of type "reference to cv A," where cv are
    //   the cv-qualifiers of the function template (if any) and A is
    //   the class of which the function template is a member.
    //
    // Note that we interpret this to mean "if one of the function
    // templates is a non-static member and the other is a non-member";
    // otherwise, the ordering rules for static functions against non-static
    // functions don't make any sense.
    //
    // C++98/03 doesn't have this provision but we've extended DR532 to cover
    // it as wording was broken prior to it.
    SmallVector<QualType, 4> Args1;

    unsigned NumComparedArguments = NumCallArguments1;

    if (!Method2 && Method1 && !Method1->isStatic()) {
      // Compare 'this' from Method1 against first parameter from Method2.
      AddImplicitObjectParameterType(S.Context, Method1, Args1);
      ++NumComparedArguments;
    } else if (!Method1 && Method2 && !Method2->isStatic()) {
      // Compare 'this' from Method2 against first parameter from Method1.
      AddImplicitObjectParameterType(S.Context, Method2, Args2);
    }

    Args1.insert(Args1.end(), Proto1->param_type_begin(),
                 Proto1->param_type_end());
    Args2.insert(Args2.end(), Proto2->param_type_begin(),
                 Proto2->param_type_end());

    // C++ [temp.func.order]p5:
    //   The presence of unused ellipsis and default arguments has no effect on
    //   the partial ordering of function templates.
    if (Args1.size() > NumComparedArguments)
      Args1.resize(NumComparedArguments);
    if (Args2.size() > NumComparedArguments)
      Args2.resize(NumComparedArguments);
    if (DeduceTemplateArguments(S, TemplateParams, Args2.data(), Args2.size(),
                                Args1.data(), Args1.size(), Info, Deduced,
                                TDF_None, /*PartialOrdering=*/true))
      return false;

    break;
  }

  case TPOC_Conversion:
    //   - In the context of a call to a conversion operator, the return types
    //     of the conversion function templates are used.
    if (DeduceTemplateArgumentsByTypeMatch(
            S, TemplateParams, Proto2->getReturnType(), Proto1->getReturnType(),
            Info, Deduced, TDF_None,
            /*PartialOrdering=*/true))
      return false;
    break;

  case TPOC_Other:
    //   - In other contexts (14.6.6.2) the function template's function type
    //     is used.
    if (DeduceTemplateArgumentsByTypeMatch(S, TemplateParams,
                                           FD2->getType(), FD1->getType(),
                                           Info, Deduced, TDF_None,
                                           /*PartialOrdering=*/true))
      return false;
    break;
  }

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
      ::MarkUsedTemplateParameters(S.Context, Args2[I], false,
                                   TemplateParams->getDepth(),
                                   UsedParameters);
    break;

  case TPOC_Conversion:
    ::MarkUsedTemplateParameters(S.Context, Proto2->getReturnType(), false,
                                 TemplateParams->getDepth(), UsedParameters);
    break;

  case TPOC_Other:
    ::MarkUsedTemplateParameters(S.Context, FD2->getType(), false,
                                 TemplateParams->getDepth(),
                                 UsedParameters);
    break;
  }

  for (; ArgIdx != NumArgs; ++ArgIdx)
    // If this argument had no value deduced but was used in one of the types
    // used for partial ordering, then deduction fails.
    if (Deduced[ArgIdx].isNull() && UsedParameters[ArgIdx])
      return false;

  return true;
}

/// \brief Determine whether this a function template whose parameter-type-list
/// ends with a function parameter pack.
static bool isVariadicFunctionTemplate(FunctionTemplateDecl *FunTmpl) {
  FunctionDecl *Function = FunTmpl->getTemplatedDecl();
  unsigned NumParams = Function->getNumParams();
  if (NumParams == 0)
    return false;

  ParmVarDecl *Last = Function->getParamDecl(NumParams - 1);
  if (!Last->isParameterPack())
    return false;

  // Make sure that no previous parameter is a parameter pack.
  while (--NumParams > 0) {
    if (Function->getParamDecl(NumParams - 1)->isParameterPack())
      return false;
  }

  return true;
}

/// \brief Returns the more specialized function template according
/// to the rules of function template partial ordering (C++ [temp.func.order]).
///
/// \param FT1 the first function template
///
/// \param FT2 the second function template
///
/// \param TPOC the context in which we are performing partial ordering of
/// function templates.
///
/// \param NumCallArguments1 The number of arguments in the call to FT1, used
/// only when \c TPOC is \c TPOC_Call.
///
/// \param NumCallArguments2 The number of arguments in the call to FT2, used
/// only when \c TPOC is \c TPOC_Call.
///
/// \returns the more specialized function template. If neither
/// template is more specialized, returns NULL.
FunctionTemplateDecl *
Sema::getMoreSpecializedTemplate(FunctionTemplateDecl *FT1,
                                 FunctionTemplateDecl *FT2,
                                 SourceLocation Loc,
                                 TemplatePartialOrderingContext TPOC,
                                 unsigned NumCallArguments1,
                                 unsigned NumCallArguments2) {
  bool Better1 = isAtLeastAsSpecializedAs(*this, Loc, FT1, FT2, TPOC,
                                          NumCallArguments1);
  bool Better2 = isAtLeastAsSpecializedAs(*this, Loc, FT2, FT1, TPOC,
                                          NumCallArguments2);

  if (Better1 != Better2) // We have a clear winner
    return Better1 ? FT1 : FT2;

  if (!Better1 && !Better2) // Neither is better than the other
    return nullptr;

  // FIXME: This mimics what GCC implements, but doesn't match up with the
  // proposed resolution for core issue 692. This area needs to be sorted out,
  // but for now we attempt to maintain compatibility.
  bool Variadic1 = isVariadicFunctionTemplate(FT1);
  bool Variadic2 = isVariadicFunctionTemplate(FT2);
  if (Variadic1 != Variadic2)
    return Variadic1? FT2 : FT1;

  return nullptr;
}

/// \brief Determine if the two templates are equivalent.
static bool isSameTemplate(TemplateDecl *T1, TemplateDecl *T2) {
  if (T1 == T2)
    return true;

  if (!T1 || !T2)
    return false;

  return T1->getCanonicalDecl() == T2->getCanonicalDecl();
}

/// \brief Retrieve the most specialized of the given function template
/// specializations.
///
/// \param SpecBegin the start iterator of the function template
/// specializations that we will be comparing.
///
/// \param SpecEnd the end iterator of the function template
/// specializations, paired with \p SpecBegin.
///
/// \param Loc the location where the ambiguity or no-specializations
/// diagnostic should occur.
///
/// \param NoneDiag partial diagnostic used to diagnose cases where there are
/// no matching candidates.
///
/// \param AmbigDiag partial diagnostic used to diagnose an ambiguity, if one
/// occurs.
///
/// \param CandidateDiag partial diagnostic used for each function template
/// specialization that is a candidate in the ambiguous ordering. One parameter
/// in this diagnostic should be unbound, which will correspond to the string
/// describing the template arguments for the function template specialization.
///
/// \returns the most specialized function template specialization, if
/// found. Otherwise, returns SpecEnd.
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
    if (isSameTemplate(getMoreSpecializedTemplate(BestTemplate, Challenger,
                                                  Loc, TPOC_Other, 0, 0),
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
        !isSameTemplate(getMoreSpecializedTemplate(BestTemplate, Challenger,
                                                   Loc, TPOC_Other, 0, 0),
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
      PD << getTemplateArgumentBindingsText(
          cast<FunctionDecl>(*I)->getPrimaryTemplate()->getTemplateParameters(),
                    *cast<FunctionDecl>(*I)->getTemplateSpecializationArgs());
      if (!TargetType.isNull())
        HandleFunctionTypeMismatch(PD, cast<FunctionDecl>(*I)->getType(),
                                   TargetType);
      Diag((*I)->getLocation(), PD);
    }
  }

  return SpecEnd;
}

/// \brief Returns the more specialized class template partial specialization
/// according to the rules of partial ordering of class template partial
/// specializations (C++ [temp.class.order]).
///
/// \param PS1 the first class template partial specialization
///
/// \param PS2 the second class template partial specialization
///
/// \returns the more specialized class template partial specialization. If
/// neither partial specialization is more specialized, returns NULL.
ClassTemplatePartialSpecializationDecl *
Sema::getMoreSpecializedPartialSpecialization(
                                  ClassTemplatePartialSpecializationDecl *PS1,
                                  ClassTemplatePartialSpecializationDecl *PS2,
                                              SourceLocation Loc) {
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
  TemplateDeductionInfo Info(Loc);

  QualType PT1 = PS1->getInjectedSpecializationType();
  QualType PT2 = PS2->getInjectedSpecializationType();

  // Determine whether PS1 is at least as specialized as PS2
  Deduced.resize(PS2->getTemplateParameters()->size());
  bool Better1 = !DeduceTemplateArgumentsByTypeMatch(*this,
                                            PS2->getTemplateParameters(),
                                            PT2, PT1, Info, Deduced, TDF_None,
                                            /*PartialOrdering=*/true);
  if (Better1) {
    SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(),Deduced.end());
    InstantiatingTemplate Inst(*this, Loc, PS2, DeducedArgs, Info);
    Better1 = !::FinishTemplateArgumentDeduction(
        *this, PS2, PS1->getTemplateArgs(), Deduced, Info);
  }

  // Determine whether PS2 is at least as specialized as PS1
  Deduced.clear();
  Deduced.resize(PS1->getTemplateParameters()->size());
  bool Better2 = !DeduceTemplateArgumentsByTypeMatch(
      *this, PS1->getTemplateParameters(), PT1, PT2, Info, Deduced, TDF_None,
      /*PartialOrdering=*/true);
  if (Better2) {
    SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(),
                                                 Deduced.end());
    InstantiatingTemplate Inst(*this, Loc, PS1, DeducedArgs, Info);
    Better2 = !::FinishTemplateArgumentDeduction(
        *this, PS1, PS2->getTemplateArgs(), Deduced, Info);
  }

  if (Better1 == Better2)
    return nullptr;

  return Better1 ? PS1 : PS2;
}

/// TODO: Unify with ClassTemplatePartialSpecializationDecl version?
///       May require unifying ClassTemplate(Partial)SpecializationDecl and
///        VarTemplate(Partial)SpecializationDecl with a new data
///        structure Template(Partial)SpecializationDecl, and
///        using Template(Partial)SpecializationDecl as input type.
VarTemplatePartialSpecializationDecl *
Sema::getMoreSpecializedPartialSpecialization(
    VarTemplatePartialSpecializationDecl *PS1,
    VarTemplatePartialSpecializationDecl *PS2, SourceLocation Loc) {
  SmallVector<DeducedTemplateArgument, 4> Deduced;
  TemplateDeductionInfo Info(Loc);

  assert(PS1->getSpecializedTemplate() == PS2->getSpecializedTemplate() &&
         "the partial specializations being compared should specialize"
         " the same template.");
  TemplateName Name(PS1->getSpecializedTemplate());
  TemplateName CanonTemplate = Context.getCanonicalTemplateName(Name);
  QualType PT1 = Context.getTemplateSpecializationType(
      CanonTemplate, PS1->getTemplateArgs().data(),
      PS1->getTemplateArgs().size());
  QualType PT2 = Context.getTemplateSpecializationType(
      CanonTemplate, PS2->getTemplateArgs().data(),
      PS2->getTemplateArgs().size());

  // Determine whether PS1 is at least as specialized as PS2
  Deduced.resize(PS2->getTemplateParameters()->size());
  bool Better1 = !DeduceTemplateArgumentsByTypeMatch(
      *this, PS2->getTemplateParameters(), PT2, PT1, Info, Deduced, TDF_None,
      /*PartialOrdering=*/true);
  if (Better1) {
    SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(),
                                                 Deduced.end());
    InstantiatingTemplate Inst(*this, Loc, PS2, DeducedArgs, Info);
    Better1 = !::FinishTemplateArgumentDeduction(*this, PS2,
                                                 PS1->getTemplateArgs(),
                                                 Deduced, Info);
  }

  // Determine whether PS2 is at least as specialized as PS1
  Deduced.clear();
  Deduced.resize(PS1->getTemplateParameters()->size());
  bool Better2 = !DeduceTemplateArgumentsByTypeMatch(*this,
                                            PS1->getTemplateParameters(),
                                            PT1, PT2, Info, Deduced, TDF_None,
                                            /*PartialOrdering=*/true);
  if (Better2) {
    SmallVector<TemplateArgument, 4> DeducedArgs(Deduced.begin(),Deduced.end());
    InstantiatingTemplate Inst(*this, Loc, PS1, DeducedArgs, Info);
    Better2 = !::FinishTemplateArgumentDeduction(*this, PS1,
                                                 PS2->getTemplateArgs(),
                                                 Deduced, Info);
  }

  if (Better1 == Better2)
    return nullptr;

  return Better1? PS1 : PS2;
}

static void
MarkUsedTemplateParameters(ASTContext &Ctx,
                           const TemplateArgument &TemplateArg,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallBitVector &Used);

/// \brief Mark the template parameters that are used by the given
/// expression.
static void
MarkUsedTemplateParameters(ASTContext &Ctx,
                           const Expr *E,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallBitVector &Used) {
  // We can deduce from a pack expansion.
  if (const PackExpansionExpr *Expansion = dyn_cast<PackExpansionExpr>(E))
    E = Expansion->getPattern();

  // Skip through any implicit casts we added while type-checking, and any
  // substitutions performed by template alias expansion.
  while (1) {
    if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
      E = ICE->getSubExpr();
    else if (const SubstNonTypeTemplateParmExpr *Subst =
               dyn_cast<SubstNonTypeTemplateParmExpr>(E))
      E = Subst->getReplacement();
    else
      break;
  }

  // FIXME: if !OnlyDeduced, we have to walk the whole subexpression to
  // find other occurrences of template parameters.
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E);
  if (!DRE)
    return;

  const NonTypeTemplateParmDecl *NTTP
    = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl());
  if (!NTTP)
    return;

  if (NTTP->getDepth() == Depth)
    Used[NTTP->getIndex()] = true;
}

/// \brief Mark the template parameters that are used by the given
/// nested name specifier.
static void
MarkUsedTemplateParameters(ASTContext &Ctx,
                           NestedNameSpecifier *NNS,
                           bool OnlyDeduced,
                           unsigned Depth,
                           llvm::SmallBitVector &Used) {
  if (!NNS)
    return;

  MarkUsedTemplateParameters(Ctx, NNS->getPrefix(), OnlyDeduced, Depth,
                             Used);
  MarkUsedTemplateParameters(Ctx, QualType(NNS->getAsType(), 0),
                             OnlyDeduced, Depth, Used);
}

/// \brief Mark the template parameters that are used by the given
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

/// \brief Mark the template parameters that are used by the given
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
    MarkUsedTemplateParameters(Ctx, QualType(MemPtr->getClass(), 0),
                               OnlyDeduced, Depth, Used);
    break;
  }

  case Type::DependentSizedArray:
    MarkUsedTemplateParameters(Ctx,
                               cast<DependentSizedArrayType>(T)->getSizeExpr(),
                               OnlyDeduced, Depth, Used);
    // Fall through to check the element type

  case Type::ConstantArray:
  case Type::IncompleteArray:
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

  case Type::DependentSizedExtVector: {
    const DependentSizedExtVectorType *VecType
      = cast<DependentSizedExtVectorType>(T);
    MarkUsedTemplateParameters(Ctx, VecType->getElementType(), OnlyDeduced,
                               Depth, Used);
    MarkUsedTemplateParameters(Ctx, VecType->getSizeExpr(), OnlyDeduced,
                               Depth, Used);
    break;
  }

  case Type::FunctionProto: {
    const FunctionProtoType *Proto = cast<FunctionProtoType>(T);
    MarkUsedTemplateParameters(Ctx, Proto->getReturnType(), OnlyDeduced, Depth,
                               Used);
    for (unsigned I = 0, N = Proto->getNumParams(); I != N; ++I)
      MarkUsedTemplateParameters(Ctx, Proto->getParamType(I), OnlyDeduced,
                                 Depth, Used);
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
    MarkUsedTemplateParameters(Ctx,
                               QualType(Subst->getReplacedParameter(), 0),
                               OnlyDeduced, Depth, Used);
    MarkUsedTemplateParameters(Ctx, Subst->getArgumentPack(),
                               OnlyDeduced, Depth, Used);
    break;
  }

  case Type::InjectedClassName:
    T = cast<InjectedClassNameType>(T)->getInjectedSpecializationType();
    // fall through

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
        hasPackExpansionBeforeEnd(Spec->getArgs(), Spec->getNumArgs()))
      break;

    for (unsigned I = 0, N = Spec->getNumArgs(); I != N; ++I)
      MarkUsedTemplateParameters(Ctx, Spec->getArg(I), OnlyDeduced, Depth,
                                 Used);
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
    const DependentTemplateSpecializationType *Spec
      = cast<DependentTemplateSpecializationType>(T);
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx, Spec->getQualifier(),
                                 OnlyDeduced, Depth, Used);

    // C++0x [temp.deduct.type]p9:
    //   If the template argument list of P contains a pack expansion that is not
    //   the last template argument, the entire template argument list is a
    //   non-deduced context.
    if (OnlyDeduced &&
        hasPackExpansionBeforeEnd(Spec->getArgs(), Spec->getNumArgs()))
      break;

    for (unsigned I = 0, N = Spec->getNumArgs(); I != N; ++I)
      MarkUsedTemplateParameters(Ctx, Spec->getArg(I), OnlyDeduced, Depth,
                                 Used);
    break;
  }

  case Type::TypeOf:
    if (!OnlyDeduced)
      MarkUsedTemplateParameters(Ctx,
                                 cast<TypeOfType>(T)->getUnderlyingType(),
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
    MarkUsedTemplateParameters(Ctx,
                               cast<AutoType>(T)->getDeducedType(),
                               OnlyDeduced, Depth, Used);

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
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    break;
  }
}

/// \brief Mark the template parameters that are used by this
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
    break;

  case TemplateArgument::NullPtr:
    MarkUsedTemplateParameters(Ctx, TemplateArg.getNullPtrType(), OnlyDeduced,
                               Depth, Used);
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

/// \brief Mark which template parameters can be deduced from a given
/// template argument list.
///
/// \param TemplateArgs the template argument list from which template
/// parameters will be deduced.
///
/// \param Used a bit vector whose elements will be set to \c true
/// to indicate when the corresponding template parameter will be
/// deduced.
void
Sema::MarkUsedTemplateParameters(const TemplateArgumentList &TemplateArgs,
                                 bool OnlyDeduced, unsigned Depth,
                                 llvm::SmallBitVector &Used) {
  // C++0x [temp.deduct.type]p9:
  //   If the template argument list of P contains a pack expansion that is not
  //   the last template argument, the entire template argument list is a
  //   non-deduced context.
  if (OnlyDeduced &&
      hasPackExpansionBeforeEnd(TemplateArgs.data(), TemplateArgs.size()))
    return;

  for (unsigned I = 0, N = TemplateArgs.size(); I != N; ++I)
    ::MarkUsedTemplateParameters(Context, TemplateArgs[I], OnlyDeduced,
                                 Depth, Used);
}

/// \brief Marks all of the template parameters that will be deduced by a
/// call to the given function template.
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
