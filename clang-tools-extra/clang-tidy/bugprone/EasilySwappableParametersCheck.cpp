//===--- EasilySwappableParametersCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EasilySwappableParametersCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "EasilySwappableParametersCheck"
#include "llvm/Support/Debug.h"
#include <optional>

namespace optutils = clang::tidy::utils::options;

/// The default value for the MinimumLength check option.
static constexpr std::size_t DefaultMinimumLength = 2;

/// The default value for ignored parameter names.
static constexpr llvm::StringLiteral DefaultIgnoredParameterNames = "\"\";"
                                                                    "iterator;"
                                                                    "Iterator;"
                                                                    "begin;"
                                                                    "Begin;"
                                                                    "end;"
                                                                    "End;"
                                                                    "first;"
                                                                    "First;"
                                                                    "last;"
                                                                    "Last;"
                                                                    "lhs;"
                                                                    "LHS;"
                                                                    "rhs;"
                                                                    "RHS";

/// The default value for ignored parameter type suffixes.
static constexpr llvm::StringLiteral DefaultIgnoredParameterTypeSuffixes =
    "bool;"
    "Bool;"
    "_Bool;"
    "it;"
    "It;"
    "iterator;"
    "Iterator;"
    "inputit;"
    "InputIt;"
    "forwardit;"
    "ForwardIt;"
    "bidirit;"
    "BidirIt;"
    "constiterator;"
    "const_iterator;"
    "Const_Iterator;"
    "Constiterator;"
    "ConstIterator;"
    "RandomIt;"
    "randomit;"
    "random_iterator;"
    "ReverseIt;"
    "reverse_iterator;"
    "reverse_const_iterator;"
    "ConstReverseIterator;"
    "Const_Reverse_Iterator;"
    "const_reverse_iterator;"
    "Constreverseiterator;"
    "constreverseiterator";

/// The default value for the QualifiersMix check option.
static constexpr bool DefaultQualifiersMix = false;

/// The default value for the ModelImplicitConversions check option.
static constexpr bool DefaultModelImplicitConversions = true;

/// The default value for suppressing diagnostics about parameters that are
/// used together.
static constexpr bool DefaultSuppressParametersUsedTogether = true;

/// The default value for the NamePrefixSuffixSilenceDissimilarityThreshold
/// check option.
static constexpr std::size_t
    DefaultNamePrefixSuffixSilenceDissimilarityTreshold = 1;

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

using TheCheck = EasilySwappableParametersCheck;

namespace filter {
class SimilarlyUsedParameterPairSuppressor;

static bool isIgnoredParameter(const TheCheck &Check, const ParmVarDecl *Node);
static inline bool
isSimilarlyUsedParameter(const SimilarlyUsedParameterPairSuppressor &Suppressor,
                         const ParmVarDecl *Param1, const ParmVarDecl *Param2);
static bool prefixSuffixCoverUnderThreshold(std::size_t Threshold,
                                            StringRef Str1, StringRef Str2);
} // namespace filter

namespace model {

/// The language features involved in allowing the mix between two parameters.
enum class MixFlags : unsigned char {
  Invalid = 0, ///< Sentinel bit pattern. DO NOT USE!

  /// Certain constructs (such as pointers to noexcept/non-noexcept functions)
  /// have the same CanonicalType, which would result in false positives.
  /// During the recursive modelling call, this flag is set if a later diagnosed
  /// canonical type equivalence should be thrown away.
  WorkaroundDisableCanonicalEquivalence = 1,

  None = 2,           ///< Mix between the two parameters is not possible.
  Trivial = 4,        ///< The two mix trivially, and are the exact same type.
  Canonical = 8,      ///< The two mix because the types refer to the same
                      /// CanonicalType, but we do not elaborate as to how.
  TypeAlias = 16,     ///< The path from one type to the other involves
                      /// desugaring type aliases.
  ReferenceBind = 32, ///< The mix involves the binding power of "const &".
  Qualifiers = 64,    ///< The mix involves change in the qualifiers.
  ImplicitConversion = 128, ///< The mixing of the parameters is possible
                            /// through implicit conversions between the types.

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue =*/ImplicitConversion)
};
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Returns whether the SearchedFlag is turned on in the Data.
static inline bool hasFlag(MixFlags Data, MixFlags SearchedFlag) {
  assert(SearchedFlag != MixFlags::Invalid &&
         "can't be used to detect lack of all bits!");

  // "Data & SearchedFlag" would need static_cast<bool>() in conditions.
  return (Data & SearchedFlag) == SearchedFlag;
}

#ifndef NDEBUG

// The modelling logic of this check is more complex than usual, and
// potentially hard to understand without the ability to see into the
// representation during the recursive descent. This debug code is only
// compiled in 'Debug' mode, or if LLVM_ENABLE_ASSERTIONS config is turned on.

/// Formats the MixFlags enum into a useful, user-readable representation.
static inline std::string formatMixFlags(MixFlags F) {
  if (F == MixFlags::Invalid)
    return "#Inv!";

  SmallString<8> Str{"-------"};

  if (hasFlag(F, MixFlags::None))
    // Shows the None bit explicitly, as it can be applied in the recursion
    // even if other bits are set.
    Str[0] = '!';
  if (hasFlag(F, MixFlags::Trivial))
    Str[1] = 'T';
  if (hasFlag(F, MixFlags::Canonical))
    Str[2] = 'C';
  if (hasFlag(F, MixFlags::TypeAlias))
    Str[3] = 't';
  if (hasFlag(F, MixFlags::ReferenceBind))
    Str[4] = '&';
  if (hasFlag(F, MixFlags::Qualifiers))
    Str[5] = 'Q';
  if (hasFlag(F, MixFlags::ImplicitConversion))
    Str[6] = 'i';

  if (hasFlag(F, MixFlags::WorkaroundDisableCanonicalEquivalence))
    Str.append("(~C)");

  return Str.str().str();
}

#endif // NDEBUG

/// The results of the steps of an Implicit Conversion Sequence is saved in
/// an instance of this record.
///
/// A ConversionSequence maps the steps of the conversion with a member for
/// each type involved in the conversion. Imagine going from a hypothetical
/// Complex class to projecting it to the real part as a const double.
///
/// I.e., given:
///
///    struct Complex {
///      operator double() const;
///    };
///
///    void functionBeingAnalysed(Complex C, const double R);
///
/// we will get the following sequence:
///
/// (Begin=) Complex
///
///     The first standard conversion is a qualification adjustment.
/// (AfterFirstStandard=) const Complex
///
///     Then the user-defined conversion is executed.
/// (UDConvOp.ConversionOperatorResultType=) double
///
///     Then this 'double' is qualifier-adjusted to 'const double'.
/// (AfterSecondStandard=) double
///
/// The conversion's result has now been calculated, so it ends here.
/// (End=) double.
///
/// Explicit storing of Begin and End in this record is needed, because
/// getting to what Begin and End here are needs further resolution of types,
/// e.g. in the case of typedefs:
///
///     using Comp = Complex;
///     using CD = const double;
///     void functionBeingAnalysed2(Comp C, CD R);
///
/// In this case, the user will be diagnosed with a potential conversion
/// between the two typedefs as written in the code, but to elaborate the
/// reasoning behind this conversion, we also need to show what the typedefs
/// mean. See FormattedConversionSequence towards the bottom of this file!
struct ConversionSequence {
  enum UserDefinedConversionKind { UDCK_None, UDCK_Ctor, UDCK_Oper };

  struct UserDefinedConvertingConstructor {
    const CXXConstructorDecl *Fun;
    QualType ConstructorParameterType;
    QualType UserDefinedType;
  };

  struct UserDefinedConversionOperator {
    const CXXConversionDecl *Fun;
    QualType UserDefinedType;
    QualType ConversionOperatorResultType;
  };

  /// The type the conversion stared from.
  QualType Begin;

  /// The intermediate type after the first Standard Conversion Sequence.
  QualType AfterFirstStandard;

  /// The details of the user-defined conversion involved, as a tagged union.
  union {
    char None;
    UserDefinedConvertingConstructor UDConvCtor;
    UserDefinedConversionOperator UDConvOp;
  };
  UserDefinedConversionKind UDConvKind;

  /// The intermediate type after performing the second Standard Conversion
  /// Sequence.
  QualType AfterSecondStandard;

  /// The result type the conversion targeted.
  QualType End;

  ConversionSequence() : None(0), UDConvKind(UDCK_None) {}
  ConversionSequence(QualType From, QualType To)
      : Begin(From), None(0), UDConvKind(UDCK_None), End(To) {}

  explicit operator bool() const {
    return !AfterFirstStandard.isNull() || UDConvKind != UDCK_None ||
           !AfterSecondStandard.isNull();
  }

  /// Returns all the "steps" (non-unique and non-similar) types involved in
  /// the conversion sequence. This method does **NOT** return Begin and End.
  SmallVector<QualType, 4> getInvolvedTypesInSequence() const {
    SmallVector<QualType, 4> Ret;
    auto EmplaceIfDifferent = [&Ret](QualType QT) {
      if (QT.isNull())
        return;
      if (Ret.empty())
        Ret.emplace_back(QT);
      else if (Ret.back() != QT)
        Ret.emplace_back(QT);
    };

    EmplaceIfDifferent(AfterFirstStandard);
    switch (UDConvKind) {
    case UDCK_Ctor:
      EmplaceIfDifferent(UDConvCtor.ConstructorParameterType);
      EmplaceIfDifferent(UDConvCtor.UserDefinedType);
      break;
    case UDCK_Oper:
      EmplaceIfDifferent(UDConvOp.UserDefinedType);
      EmplaceIfDifferent(UDConvOp.ConversionOperatorResultType);
      break;
    case UDCK_None:
      break;
    }
    EmplaceIfDifferent(AfterSecondStandard);

    return Ret;
  }

  /// Updates the steps of the conversion sequence with the steps from the
  /// other instance.
  ///
  /// \note This method does not check if the resulting conversion sequence is
  /// sensible!
  ConversionSequence &update(const ConversionSequence &RHS) {
    if (!RHS.AfterFirstStandard.isNull())
      AfterFirstStandard = RHS.AfterFirstStandard;
    switch (RHS.UDConvKind) {
    case UDCK_Ctor:
      UDConvKind = UDCK_Ctor;
      UDConvCtor = RHS.UDConvCtor;
      break;
    case UDCK_Oper:
      UDConvKind = UDCK_Oper;
      UDConvOp = RHS.UDConvOp;
      break;
    case UDCK_None:
      break;
    }
    if (!RHS.AfterSecondStandard.isNull())
      AfterSecondStandard = RHS.AfterSecondStandard;

    return *this;
  }

  /// Sets the user-defined conversion to the given constructor.
  void setConversion(const UserDefinedConvertingConstructor &UDCC) {
    UDConvKind = UDCK_Ctor;
    UDConvCtor = UDCC;
  }

  /// Sets the user-defined conversion to the given operator.
  void setConversion(const UserDefinedConversionOperator &UDCO) {
    UDConvKind = UDCK_Oper;
    UDConvOp = UDCO;
  }

  /// Returns the type in the conversion that's formally "in our hands" once
  /// the user-defined conversion is executed.
  QualType getTypeAfterUserDefinedConversion() const {
    switch (UDConvKind) {
    case UDCK_Ctor:
      return UDConvCtor.UserDefinedType;
    case UDCK_Oper:
      return UDConvOp.ConversionOperatorResultType;
    case UDCK_None:
      return {};
    }
    llvm_unreachable("Invalid UDConv kind.");
  }

  const CXXMethodDecl *getUserDefinedConversionFunction() const {
    switch (UDConvKind) {
    case UDCK_Ctor:
      return UDConvCtor.Fun;
    case UDCK_Oper:
      return UDConvOp.Fun;
    case UDCK_None:
      return {};
    }
    llvm_unreachable("Invalid UDConv kind.");
  }

  /// Returns the SourceRange in the text that corresponds to the interesting
  /// part of the user-defined conversion. This is either the parameter type
  /// in a converting constructor, or the conversion result type in a conversion
  /// operator.
  SourceRange getUserDefinedConversionHighlight() const {
    switch (UDConvKind) {
    case UDCK_Ctor:
      return UDConvCtor.Fun->getParamDecl(0)->getSourceRange();
    case UDCK_Oper:
      // getReturnTypeSourceRange() does not work for CXXConversionDecls as the
      // returned type is physically behind the declaration's name ("operator").
      if (const FunctionTypeLoc FTL = UDConvOp.Fun->getFunctionTypeLoc())
        if (const TypeLoc RetLoc = FTL.getReturnLoc())
          return RetLoc.getSourceRange();
      return {};
    case UDCK_None:
      return {};
    }
    llvm_unreachable("Invalid UDConv kind.");
  }
};

/// Contains the metadata for the mixability result between two types,
/// independently of which parameters they were calculated from.
struct MixData {
  /// The flag bits of the mix indicating what language features allow for it.
  MixFlags Flags = MixFlags::Invalid;

  /// A potentially calculated common underlying type after desugaring, that
  /// both sides of the mix can originate from.
  QualType CommonType;

  /// The steps an implicit conversion performs to get from one type to the
  /// other.
  ConversionSequence Conversion, ConversionRTL;

  /// True if the MixData was specifically created with only a one-way
  /// conversion modelled.
  bool CreatedFromOneWayConversion = false;

  MixData(MixFlags Flags) : Flags(Flags) {}
  MixData(MixFlags Flags, QualType CommonType)
      : Flags(Flags), CommonType(CommonType) {}
  MixData(MixFlags Flags, ConversionSequence Conv)
      : Flags(Flags), Conversion(Conv), CreatedFromOneWayConversion(true) {}
  MixData(MixFlags Flags, ConversionSequence LTR, ConversionSequence RTL)
      : Flags(Flags), Conversion(LTR), ConversionRTL(RTL) {}
  MixData(MixFlags Flags, QualType CommonType, ConversionSequence LTR,
          ConversionSequence RTL)
      : Flags(Flags), CommonType(CommonType), Conversion(LTR),
        ConversionRTL(RTL) {}

  void sanitize() {
    assert(Flags != MixFlags::Invalid && "sanitize() called on invalid bitvec");

    MixFlags CanonicalAndWorkaround =
        MixFlags::Canonical | MixFlags::WorkaroundDisableCanonicalEquivalence;
    if ((Flags & CanonicalAndWorkaround) == CanonicalAndWorkaround) {
      // A workaround for too eagerly equivalent canonical types was requested,
      // and a canonical equivalence was proven. Fulfill the request and throw
      // this result away.
      Flags = MixFlags::None;
      return;
    }

    if (hasFlag(Flags, MixFlags::None)) {
      // If anywhere down the recursion a potential mix "path" is deemed
      // impossible, throw away all the other bits because the mix is not
      // possible.
      Flags = MixFlags::None;
      return;
    }

    if (Flags == MixFlags::Trivial)
      return;

    if (static_cast<bool>(Flags ^ MixFlags::Trivial))
      // If the mix involves somewhere trivial equivalence but down the
      // recursion other bit(s) were set, remove the trivial bit, as it is not
      // trivial.
      Flags &= ~MixFlags::Trivial;

    bool ShouldHaveImplicitConvFlag = false;
    if (CreatedFromOneWayConversion && Conversion)
      ShouldHaveImplicitConvFlag = true;
    else if (!CreatedFromOneWayConversion && Conversion && ConversionRTL)
      // Only say that we have implicit conversion mix possibility if it is
      // bidirectional. Otherwise, the compiler would report an *actual* swap
      // at a call site...
      ShouldHaveImplicitConvFlag = true;

    if (ShouldHaveImplicitConvFlag)
      Flags |= MixFlags::ImplicitConversion;
    else
      Flags &= ~MixFlags::ImplicitConversion;
  }

  bool isValid() const { return Flags >= MixFlags::None; }

  bool indicatesMixability() const { return Flags > MixFlags::None; }

  /// Add the specified flag bits to the flags.
  MixData operator|(MixFlags EnableFlags) const {
    if (CreatedFromOneWayConversion) {
      MixData M{Flags | EnableFlags, Conversion};
      M.CommonType = CommonType;
      return M;
    }
    return {Flags | EnableFlags, CommonType, Conversion, ConversionRTL};
  }

  /// Add the specified flag bits to the flags.
  MixData &operator|=(MixFlags EnableFlags) {
    Flags |= EnableFlags;
    return *this;
  }

  template <typename F> MixData withCommonTypeTransformed(const F &Func) const {
    if (CommonType.isNull())
      return *this;

    QualType NewCommonType = Func(CommonType);

    if (CreatedFromOneWayConversion) {
      MixData M{Flags, Conversion};
      M.CommonType = NewCommonType;
      return M;
    }

    return {Flags, NewCommonType, Conversion, ConversionRTL};
  }
};

/// A named tuple that contains the information for a mix between two concrete
/// parameters.
struct Mix {
  const ParmVarDecl *First, *Second;
  MixData Data;

  Mix(const ParmVarDecl *F, const ParmVarDecl *S, MixData Data)
      : First(F), Second(S), Data(std::move(Data)) {}

  void sanitize() { Data.sanitize(); }
  MixFlags flags() const { return Data.Flags; }
  bool flagsValid() const { return Data.isValid(); }
  bool mixable() const { return Data.indicatesMixability(); }
  QualType commonUnderlyingType() const { return Data.CommonType; }
  const ConversionSequence &leftToRightConversionSequence() const {
    return Data.Conversion;
  }
  const ConversionSequence &rightToLeftConversionSequence() const {
    return Data.ConversionRTL;
  }
};

// NOLINTNEXTLINE(misc-redundant-expression): Seems to be a bogus warning.
static_assert(std::is_trivially_copyable_v<Mix> &&
                  std::is_trivially_move_constructible_v<Mix> &&
                  std::is_trivially_move_assignable_v<Mix>,
              "Keep frequently used data simple!");

struct MixableParameterRange {
  /// A container for Mixes.
  using MixVector = SmallVector<Mix, 8>;

  /// The number of parameters iterated to build the instance.
  std::size_t NumParamsChecked = 0;

  /// The individual flags and supporting information for the mixes.
  MixVector Mixes;

  /// Gets the leftmost parameter of the range.
  const ParmVarDecl *getFirstParam() const {
    // The first element is the LHS of the very first mix in the range.
    assert(!Mixes.empty());
    return Mixes.front().First;
  }

  /// Gets the rightmost parameter of the range.
  const ParmVarDecl *getLastParam() const {
    // The builder function breaks building an instance of this type if it
    // finds something that can not be mixed with the rest, by going *forward*
    // in the list of parameters. So at any moment of break, the RHS of the last
    // element of the mix vector is also the last element of the mixing range.
    assert(!Mixes.empty());
    return Mixes.back().Second;
  }
};

/// Helper enum for the recursive calls in the modelling that toggle what kinds
/// of implicit conversions are to be modelled.
enum class ImplicitConversionModellingMode : unsigned char {
  ///< No implicit conversions are modelled.
  None,

  ///< The full implicit conversion sequence is modelled.
  All,

  ///< Only model a unidirectional implicit conversion and within it only one
  /// standard conversion sequence.
  OneWaySingleStandardOnly
};

static MixData
isLRefEquallyBindingToType(const TheCheck &Check,
                           const LValueReferenceType *LRef, QualType Ty,
                           const ASTContext &Ctx, bool IsRefRHS,
                           ImplicitConversionModellingMode ImplicitMode);

static MixData
approximateImplicitConversion(const TheCheck &Check, QualType LType,
                              QualType RType, const ASTContext &Ctx,
                              ImplicitConversionModellingMode ImplicitMode);

static inline bool isUselessSugar(const Type *T) {
  return isa<AttributedType, DecayedType, ParenType>(T);
}

namespace {

struct NonCVRQualifiersResult {
  /// True if the types are qualified in a way that even after equating or
  /// removing local CVR qualification, even if the unqualified types
  /// themselves would mix, the qualified ones don't, because there are some
  /// other local qualifiers that are not equal.
  bool HasMixabilityBreakingQualifiers;

  /// The set of equal qualifiers between the two types.
  Qualifiers CommonQualifiers;
};

} // namespace

/// Returns if the two types are qualified in a way that ever after equating or
/// removing local CVR qualification, even if the unqualified types would mix,
/// the qualified ones don't, because there are some other local qualifiers
/// that aren't equal.
static NonCVRQualifiersResult
getNonCVRQualifiers(const ASTContext &Ctx, QualType LType, QualType RType) {
  LLVM_DEBUG(llvm::dbgs() << ">>> getNonCVRQualifiers for LType:\n";
             LType.dump(llvm::dbgs(), Ctx); llvm::dbgs() << "\nand RType:\n";
             RType.dump(llvm::dbgs(), Ctx); llvm::dbgs() << '\n';);
  Qualifiers LQual = LType.getLocalQualifiers(),
             RQual = RType.getLocalQualifiers();

  // Strip potential CVR. That is handled by the check option QualifiersMix.
  LQual.removeCVRQualifiers();
  RQual.removeCVRQualifiers();

  NonCVRQualifiersResult Ret;
  Ret.CommonQualifiers = Qualifiers::removeCommonQualifiers(LQual, RQual);

  LLVM_DEBUG(llvm::dbgs() << "--- hasNonCVRMixabilityBreakingQualifiers. "
                             "Removed common qualifiers: ";
             Ret.CommonQualifiers.print(llvm::dbgs(), Ctx.getPrintingPolicy());
             llvm::dbgs() << "\n\tremaining on LType: ";
             LQual.print(llvm::dbgs(), Ctx.getPrintingPolicy());
             llvm::dbgs() << "\n\tremaining on RType: ";
             RQual.print(llvm::dbgs(), Ctx.getPrintingPolicy());
             llvm::dbgs() << '\n';);

  // If there are no other non-cvr non-common qualifiers left, we can deduce
  // that mixability isn't broken.
  Ret.HasMixabilityBreakingQualifiers =
      LQual.hasQualifiers() || RQual.hasQualifiers();

  return Ret;
}

/// Approximate the way how LType and RType might refer to "essentially the
/// same" type, in a sense that at a particular call site, an expression of
/// type LType and RType might be successfully passed to a variable (in our
/// specific case, a parameter) of type RType and LType, respectively.
/// Note the swapped order!
///
/// The returned data structure is not guaranteed to be properly set, as this
/// function is potentially recursive. It is the caller's responsibility to
/// call sanitize() on the result once the recursion is over.
static MixData
calculateMixability(const TheCheck &Check, QualType LType, QualType RType,
                    const ASTContext &Ctx,
                    ImplicitConversionModellingMode ImplicitMode) {
  LLVM_DEBUG(llvm::dbgs() << ">>> calculateMixability for LType:\n";
             LType.dump(llvm::dbgs(), Ctx); llvm::dbgs() << "\nand RType:\n";
             RType.dump(llvm::dbgs(), Ctx); llvm::dbgs() << '\n';);
  if (LType == RType) {
    LLVM_DEBUG(llvm::dbgs() << "<<< calculateMixability. Trivial equality.\n");
    return {MixFlags::Trivial, LType};
  }

  // Dissolve certain type sugars that do not affect the mixability of one type
  // with the other, and also do not require any sort of elaboration for the
  // user to understand.
  if (isUselessSugar(LType.getTypePtr())) {
    LLVM_DEBUG(llvm::dbgs()
               << "--- calculateMixability. LHS is useless sugar.\n");
    return calculateMixability(Check, LType.getSingleStepDesugaredType(Ctx),
                               RType, Ctx, ImplicitMode);
  }
  if (isUselessSugar(RType.getTypePtr())) {
    LLVM_DEBUG(llvm::dbgs()
               << "--- calculateMixability. RHS is useless sugar.\n");
    return calculateMixability(
        Check, LType, RType.getSingleStepDesugaredType(Ctx), Ctx, ImplicitMode);
  }

  const auto *LLRef = LType->getAs<LValueReferenceType>();
  const auto *RLRef = RType->getAs<LValueReferenceType>();
  if (LLRef && RLRef) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. LHS and RHS are &.\n");

    return calculateMixability(Check, LLRef->getPointeeType(),
                               RLRef->getPointeeType(), Ctx, ImplicitMode)
        .withCommonTypeTransformed(
            [&Ctx](QualType QT) { return Ctx.getLValueReferenceType(QT); });
  }
  // At a particular call site, what could be passed to a 'T' or 'const T' might
  // also be passed to a 'const T &' without the call site putting a direct
  // side effect on the passed expressions.
  if (LLRef) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. LHS is &.\n");
    return isLRefEquallyBindingToType(Check, LLRef, RType, Ctx, false,
                                      ImplicitMode) |
           MixFlags::ReferenceBind;
  }
  if (RLRef) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. RHS is &.\n");
    return isLRefEquallyBindingToType(Check, RLRef, LType, Ctx, true,
                                      ImplicitMode) |
           MixFlags::ReferenceBind;
  }

  if (LType->getAs<TypedefType>()) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. LHS is typedef.\n");
    return calculateMixability(Check, LType.getSingleStepDesugaredType(Ctx),
                               RType, Ctx, ImplicitMode) |
           MixFlags::TypeAlias;
  }
  if (RType->getAs<TypedefType>()) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. RHS is typedef.\n");
    return calculateMixability(Check, LType,
                               RType.getSingleStepDesugaredType(Ctx), Ctx,
                               ImplicitMode) |
           MixFlags::TypeAlias;
  }

  // A parameter of type 'cvr1 T' and another of potentially differently
  // qualified 'cvr2 T' may bind with the same power, if the user so requested.
  //
  // Whether to do this check for the inner unqualified types.
  bool CompareUnqualifiedTypes = false;
  if (LType.getLocalCVRQualifiers() != RType.getLocalCVRQualifiers()) {
    LLVM_DEBUG(if (LType.getLocalCVRQualifiers()) {
      llvm::dbgs() << "--- calculateMixability. LHS has CVR-Qualifiers: ";
      Qualifiers::fromCVRMask(LType.getLocalCVRQualifiers())
          .print(llvm::dbgs(), Ctx.getPrintingPolicy());
      llvm::dbgs() << '\n';
    });
    LLVM_DEBUG(if (RType.getLocalCVRQualifiers()) {
      llvm::dbgs() << "--- calculateMixability. RHS has CVR-Qualifiers: ";
      Qualifiers::fromCVRMask(RType.getLocalCVRQualifiers())
          .print(llvm::dbgs(), Ctx.getPrintingPolicy());
      llvm::dbgs() << '\n';
    });

    if (!Check.QualifiersMix) {
      LLVM_DEBUG(llvm::dbgs()
                 << "<<< calculateMixability. QualifiersMix turned off - not "
                    "mixable.\n");
      return {MixFlags::None};
    }

    CompareUnqualifiedTypes = true;
  }
  // Whether the two types had the same CVR qualifiers.
  bool OriginallySameQualifiers = false;
  if (LType.getLocalCVRQualifiers() == RType.getLocalCVRQualifiers() &&
      LType.getLocalCVRQualifiers() != 0) {
    LLVM_DEBUG(if (LType.getLocalCVRQualifiers()) {
      llvm::dbgs()
          << "--- calculateMixability. LHS and RHS have same CVR-Qualifiers: ";
      Qualifiers::fromCVRMask(LType.getLocalCVRQualifiers())
          .print(llvm::dbgs(), Ctx.getPrintingPolicy());
      llvm::dbgs() << '\n';
    });

    CompareUnqualifiedTypes = true;
    OriginallySameQualifiers = true;
  }

  if (CompareUnqualifiedTypes) {
    NonCVRQualifiersResult AdditionalQuals =
        getNonCVRQualifiers(Ctx, LType, RType);
    if (AdditionalQuals.HasMixabilityBreakingQualifiers) {
      LLVM_DEBUG(llvm::dbgs() << "<<< calculateMixability. Additional "
                                 "non-equal incompatible qualifiers.\n");
      return {MixFlags::None};
    }

    MixData UnqualifiedMixability =
        calculateMixability(Check, LType.getLocalUnqualifiedType(),
                            RType.getLocalUnqualifiedType(), Ctx, ImplicitMode)
            .withCommonTypeTransformed([&AdditionalQuals, &Ctx](QualType QT) {
              // Once the mixability was deduced, apply the qualifiers common
              // to the two type back onto the diagnostic printout.
              return Ctx.getQualifiedType(QT, AdditionalQuals.CommonQualifiers);
            });

    if (!OriginallySameQualifiers)
      // User-enabled qualifier change modelled for the mix.
      return UnqualifiedMixability | MixFlags::Qualifiers;

    // Apply the same qualifier back into the found common type if they were
    // the same.
    return UnqualifiedMixability.withCommonTypeTransformed(
        [&Ctx, LType](QualType QT) {
          return Ctx.getQualifiedType(QT, LType.getLocalQualifiers());
        });
  }

  // Certain constructs match on the last catch-all getCanonicalType() equality,
  // which is perhaps something not what we want. If this variable is true,
  // the canonical type equality will be ignored.
  bool RecursiveReturnDiscardingCanonicalType = false;

  if (LType->isPointerType() && RType->isPointerType()) {
    // If both types are pointers, and pointed to the exact same type,
    // LType == RType took care of that. Try to see if the pointee type has
    // some other match. However, this must not consider implicit conversions.
    LLVM_DEBUG(llvm::dbgs()
               << "--- calculateMixability. LHS and RHS are Ptrs.\n");
    MixData MixOfPointee =
        calculateMixability(Check, LType->getPointeeType(),
                            RType->getPointeeType(), Ctx,
                            ImplicitConversionModellingMode::None)
            .withCommonTypeTransformed(
                [&Ctx](QualType QT) { return Ctx.getPointerType(QT); });
    if (hasFlag(MixOfPointee.Flags,
                MixFlags::WorkaroundDisableCanonicalEquivalence))
      RecursiveReturnDiscardingCanonicalType = true;

    MixOfPointee.sanitize();
    if (MixOfPointee.indicatesMixability()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "<<< calculateMixability. Pointees are mixable.\n");
      return MixOfPointee;
    }
  }

  if (ImplicitMode > ImplicitConversionModellingMode::None) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. Start implicit...\n");
    MixData MixLTR =
        approximateImplicitConversion(Check, LType, RType, Ctx, ImplicitMode);
    LLVM_DEBUG(
        if (hasFlag(MixLTR.Flags, MixFlags::ImplicitConversion)) llvm::dbgs()
            << "--- calculateMixability. Implicit Left -> Right found.\n";);

    if (ImplicitMode ==
            ImplicitConversionModellingMode::OneWaySingleStandardOnly &&
        MixLTR.Conversion && !MixLTR.Conversion.AfterFirstStandard.isNull() &&
        MixLTR.Conversion.UDConvKind == ConversionSequence::UDCK_None &&
        MixLTR.Conversion.AfterSecondStandard.isNull()) {
      // The invoker of the method requested only modelling a single standard
      // conversion, in only the forward direction, and they got just that.
      LLVM_DEBUG(llvm::dbgs() << "<<< calculateMixability. Implicit "
                                 "conversion, one-way, standard-only.\n");
      return {MixFlags::ImplicitConversion, MixLTR.Conversion};
    }

    // Otherwise if the invoker requested a full modelling, do the other
    // direction as well.
    MixData MixRTL =
        approximateImplicitConversion(Check, RType, LType, Ctx, ImplicitMode);
    LLVM_DEBUG(
        if (hasFlag(MixRTL.Flags, MixFlags::ImplicitConversion)) llvm::dbgs()
            << "--- calculateMixability. Implicit Right -> Left found.\n";);

    if (MixLTR.Conversion && MixRTL.Conversion) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "<<< calculateMixability. Implicit conversion, bidirectional.\n");
      return {MixFlags::ImplicitConversion, MixLTR.Conversion,
              MixRTL.Conversion};
    }
  }

  if (RecursiveReturnDiscardingCanonicalType)
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. Before CanonicalType, "
                               "Discard was enabled.\n");

  // Certain kinds unfortunately need to be side-stepped for canonical type
  // matching.
  if (LType->getAs<FunctionProtoType>() || RType->getAs<FunctionProtoType>()) {
    // Unfortunately, the canonical type of a function pointer becomes the
    // same even if exactly one is "noexcept" and the other isn't, making us
    // give a false positive report irrespective of implicit conversions.
    LLVM_DEBUG(llvm::dbgs()
               << "--- calculateMixability. Discarding potential canonical "
                  "equivalence on FunctionProtoTypes.\n");
    RecursiveReturnDiscardingCanonicalType = true;
  }

  MixData MixToReturn{MixFlags::None};

  // If none of the previous logic found a match, try if Clang otherwise
  // believes the types to be the same.
  QualType LCanonical = LType.getCanonicalType();
  if (LCanonical == RType.getCanonicalType()) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< calculateMixability. Same CanonicalType.\n");
    MixToReturn = {MixFlags::Canonical, LCanonical};
  }

  if (RecursiveReturnDiscardingCanonicalType)
    MixToReturn |= MixFlags::WorkaroundDisableCanonicalEquivalence;

  LLVM_DEBUG(if (MixToReturn.Flags == MixFlags::None) llvm::dbgs()
             << "<<< calculateMixability. No match found.\n");
  return MixToReturn;
}

/// Calculates if the reference binds an expression of the given type. This is
/// true iff 'LRef' is some 'const T &' type, and the 'Ty' is 'T' or 'const T'.
///
/// \param ImplicitMode is forwarded in the possible recursive call to
/// calculateMixability.
static MixData
isLRefEquallyBindingToType(const TheCheck &Check,
                           const LValueReferenceType *LRef, QualType Ty,
                           const ASTContext &Ctx, bool IsRefRHS,
                           ImplicitConversionModellingMode ImplicitMode) {
  LLVM_DEBUG(llvm::dbgs() << ">>> isLRefEquallyBindingToType for LRef:\n";
             LRef->dump(llvm::dbgs(), Ctx); llvm::dbgs() << "\nand Type:\n";
             Ty.dump(llvm::dbgs(), Ctx); llvm::dbgs() << '\n';);

  QualType ReferredType = LRef->getPointeeType();
  if (!ReferredType.isLocalConstQualified() &&
      ReferredType->getAs<TypedefType>()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "--- isLRefEquallyBindingToType. Non-const LRef to Typedef.\n");
    ReferredType = ReferredType.getDesugaredType(Ctx);
    if (!ReferredType.isLocalConstQualified()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "<<< isLRefEquallyBindingToType. Typedef is not const.\n");
      return {MixFlags::None};
    }

    LLVM_DEBUG(llvm::dbgs() << "--- isLRefEquallyBindingToType. Typedef is "
                               "const, considering as const LRef.\n");
  } else if (!ReferredType.isLocalConstQualified()) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< isLRefEquallyBindingToType. Not const LRef.\n");
    return {MixFlags::None};
  };

  assert(ReferredType.isLocalConstQualified() &&
         "Reaching this point means we are sure LRef is effectively a const&.");

  if (ReferredType == Ty) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "<<< isLRefEquallyBindingToType. Type of referred matches.\n");
    return {MixFlags::Trivial, ReferredType};
  }

  QualType NonConstReferredType = ReferredType;
  NonConstReferredType.removeLocalConst();
  if (NonConstReferredType == Ty) {
    LLVM_DEBUG(llvm::dbgs() << "<<< isLRefEquallyBindingToType. Type of "
                               "referred matches to non-const qualified.\n");
    return {MixFlags::Trivial, NonConstReferredType};
  }

  LLVM_DEBUG(
      llvm::dbgs()
      << "--- isLRefEquallyBindingToType. Checking mix for underlying type.\n");
  return IsRefRHS ? calculateMixability(Check, Ty, NonConstReferredType, Ctx,
                                        ImplicitMode)
                  : calculateMixability(Check, NonConstReferredType, Ty, Ctx,
                                        ImplicitMode);
}

static inline bool isDerivedToBase(const CXXRecordDecl *Derived,
                                   const CXXRecordDecl *Base) {
  return Derived && Base && Derived->isCompleteDefinition() &&
         Base->isCompleteDefinition() && Derived->isDerivedFrom(Base);
}

static std::optional<QualType>
approximateStandardConversionSequence(const TheCheck &Check, QualType From,
                                      QualType To, const ASTContext &Ctx) {
  LLVM_DEBUG(llvm::dbgs() << ">>> approximateStdConv for LType:\n";
             From.dump(llvm::dbgs(), Ctx); llvm::dbgs() << "\nand RType:\n";
             To.dump(llvm::dbgs(), Ctx); llvm::dbgs() << '\n';);

  // A standard conversion sequence consists of the following, in order:
  //  * Maybe either LValue->RValue conv., Array->Ptr conv., Function->Ptr conv.
  //  * Maybe Numeric promotion or conversion.
  //  * Maybe function pointer conversion.
  //  * Maybe qualifier adjustments.
  QualType WorkType = From;
  // Get out the qualifiers of the original type. This will always be
  // re-applied to the WorkType to ensure it is the same qualification as the
  // original From was.
  auto FastQualifiersToApply = static_cast<unsigned>(
      From.split().Quals.getAsOpaqueValue() & Qualifiers::FastMask);

  // LValue->RValue is irrelevant for the check, because it is a thing to be
  // done at a call site, and will be performed if need be performed.

  // Array->Pointer decay is handled by the main method in desugaring
  // the parameter's DecayedType as "useless sugar".

  // Function->Pointer conversions are also irrelevant, because a
  // "FunctionType" cannot be the type of a parameter variable, so this
  // conversion is only meaningful at call sites.

  // Numeric promotions and conversions.
  const auto *FromBuiltin = WorkType->getAs<BuiltinType>();
  const auto *ToBuiltin = To->getAs<BuiltinType>();
  bool FromNumeric = FromBuiltin && (FromBuiltin->isIntegerType() ||
                                     FromBuiltin->isFloatingType());
  bool ToNumeric =
      ToBuiltin && (ToBuiltin->isIntegerType() || ToBuiltin->isFloatingType());
  if (FromNumeric && ToNumeric) {
    // If both are integral types, the numeric conversion is performed.
    // Reapply the qualifiers of the original type, however, so
    // "const int -> double" in this case moves over to
    // "const double -> double".
    LLVM_DEBUG(llvm::dbgs()
               << "--- approximateStdConv. Conversion between numerics.\n");
    WorkType = QualType{ToBuiltin, FastQualifiersToApply};
  }

  const auto *FromEnum = WorkType->getAsCanonical<EnumType>();
  const auto *ToEnum = To->getAs<EnumType>();
  if (FromEnum && ToNumeric && FromEnum->isUnscopedEnumerationType()) {
    // Unscoped enumerations (or enumerations in C) convert to numerics.
    LLVM_DEBUG(llvm::dbgs()
               << "--- approximateStdConv. Unscoped enum to numeric.\n");
    WorkType = QualType{ToBuiltin, FastQualifiersToApply};
  } else if (FromNumeric && ToEnum && ToEnum->isUnscopedEnumerationType()) {
    // Numeric types convert to enumerations only in C.
    if (Ctx.getLangOpts().CPlusPlus) {
      LLVM_DEBUG(llvm::dbgs() << "<<< approximateStdConv. Numeric to unscoped "
                                 "enum, not possible in C++!\n");
      return {};
    }

    LLVM_DEBUG(llvm::dbgs()
               << "--- approximateStdConv. Numeric to unscoped enum.\n");
    WorkType = QualType{ToEnum, FastQualifiersToApply};
  }

  // Check for pointer conversions.
  const auto *FromPtr = WorkType->getAs<PointerType>();
  const auto *ToPtr = To->getAs<PointerType>();
  if (FromPtr && ToPtr) {
    if (ToPtr->isVoidPointerType()) {
      LLVM_DEBUG(llvm::dbgs() << "--- approximateStdConv. To void pointer.\n");
      WorkType = QualType{ToPtr, FastQualifiersToApply};
    }

    const auto *FromRecordPtr = FromPtr->getPointeeCXXRecordDecl();
    const auto *ToRecordPtr = ToPtr->getPointeeCXXRecordDecl();
    if (isDerivedToBase(FromRecordPtr, ToRecordPtr)) {
      LLVM_DEBUG(llvm::dbgs() << "--- approximateStdConv. Derived* to Base*\n");
      WorkType = QualType{ToPtr, FastQualifiersToApply};
    }
  }

  // Model the slicing Derived-to-Base too, as "BaseT temporary = derived;"
  // can also be compiled.
  const auto *FromRecord = WorkType->getAsCXXRecordDecl();
  const auto *ToRecord = To->getAsCXXRecordDecl();
  if (isDerivedToBase(FromRecord, ToRecord)) {
    LLVM_DEBUG(llvm::dbgs() << "--- approximateStdConv. Derived To Base.\n");
    WorkType = QualType{
        ToRecord->getASTContext().getCanonicalTagType(ToRecord)->getTypePtr(),
        FastQualifiersToApply};
  }

  if (Ctx.getLangOpts().CPlusPlus17 && FromPtr && ToPtr) {
    // Function pointer conversion: A noexcept function pointer can be passed
    // to a non-noexcept one.
    const auto *FromFunctionPtr =
        FromPtr->getPointeeType()->getAs<FunctionProtoType>();
    const auto *ToFunctionPtr =
        ToPtr->getPointeeType()->getAs<FunctionProtoType>();
    if (FromFunctionPtr && ToFunctionPtr &&
        FromFunctionPtr->hasNoexceptExceptionSpec() &&
        !ToFunctionPtr->hasNoexceptExceptionSpec()) {
      LLVM_DEBUG(llvm::dbgs() << "--- approximateStdConv. noexcept function "
                                 "pointer to non-noexcept.\n");
      WorkType = QualType{ToPtr, FastQualifiersToApply};
    }
  }

  // Qualifier adjustments are modelled according to the user's request in
  // the QualifiersMix check config.
  LLVM_DEBUG(llvm::dbgs()
             << "--- approximateStdConv. Trying qualifier adjustment...\n");
  MixData QualConv = calculateMixability(Check, WorkType, To, Ctx,
                                         ImplicitConversionModellingMode::None);
  QualConv.sanitize();
  if (hasFlag(QualConv.Flags, MixFlags::Qualifiers)) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< approximateStdConv. Qualifiers adjusted.\n");
    WorkType = To;
  }

  if (Ctx.hasSameType(WorkType, To)) {
    LLVM_DEBUG(llvm::dbgs() << "<<< approximateStdConv. Reached 'To' type.\n");
    return {Ctx.getCommonSugaredType(WorkType, To)};
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< approximateStdConv. Did not reach 'To'.\n");
  return {};
}

namespace {

/// Helper class for storing possible user-defined conversion calls that
/// *could* take place in an implicit conversion, and selecting the one that
/// most likely *does*, if any.
class UserDefinedConversionSelector {
public:
  /// The conversion associated with a conversion function, together with the
  /// mixability flags of the conversion function's parameter or return type
  /// to the rest of the sequence the selector is used in, and the sequence
  /// that applied through the conversion itself.
  struct PreparedConversion {
    const CXXMethodDecl *ConversionFun;
    MixFlags Flags;
    ConversionSequence Seq;

    PreparedConversion(const CXXMethodDecl *CMD, MixFlags F,
                       ConversionSequence S)
        : ConversionFun(CMD), Flags(F), Seq(S) {}
  };

  UserDefinedConversionSelector(const TheCheck &Check) : Check(Check) {}

  /// Adds the conversion between the two types for the given function into
  /// the possible implicit conversion set. FromType and ToType is either:
  ///   * the result of a standard sequence and a converting ctor parameter
  ///   * the return type of a conversion operator and the expected target of
  ///     an implicit conversion.
  void addConversion(const CXXMethodDecl *ConvFun, QualType FromType,
                     QualType ToType) {
    // Try to go from the FromType to the ToType with only a single implicit
    // conversion, to see if the conversion function is applicable.
    MixData Mix = calculateMixability(
        Check, FromType, ToType, ConvFun->getASTContext(),
        ImplicitConversionModellingMode::OneWaySingleStandardOnly);
    Mix.sanitize();
    if (!Mix.indicatesMixability())
      return;

    LLVM_DEBUG(llvm::dbgs() << "--- tryConversion. Found viable with flags: "
                            << formatMixFlags(Mix.Flags) << '\n');
    FlaggedConversions.emplace_back(ConvFun, Mix.Flags, Mix.Conversion);
  }

  /// Selects the best conversion function that is applicable from the
  /// prepared set of potential conversion functions taken.
  std::optional<PreparedConversion> operator()() const {
    if (FlaggedConversions.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "--- selectUserDefinedConv. Empty.\n");
      return {};
    }
    if (FlaggedConversions.size() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "--- selectUserDefinedConv. Single.\n");
      return FlaggedConversions.front();
    }

    std::optional<PreparedConversion> BestConversion;
    unsigned short HowManyGoodConversions = 0;
    for (const auto &Prepared : FlaggedConversions) {
      LLVM_DEBUG(llvm::dbgs() << "--- selectUserDefinedConv. Candidate flags: "
                              << formatMixFlags(Prepared.Flags) << '\n');
      if (!BestConversion) {
        BestConversion = Prepared;
        ++HowManyGoodConversions;
        continue;
      }

      bool BestConversionHasImplicit =
          hasFlag(BestConversion->Flags, MixFlags::ImplicitConversion);
      bool ThisConversionHasImplicit =
          hasFlag(Prepared.Flags, MixFlags::ImplicitConversion);
      if (!BestConversionHasImplicit && ThisConversionHasImplicit)
        // This is a worse conversion, because a better one was found earlier.
        continue;

      if (BestConversionHasImplicit && !ThisConversionHasImplicit) {
        // If the so far best selected conversion needs a previous implicit
        // conversion to match the user-defined converting function, but this
        // conversion does not, this is a better conversion, and we can throw
        // away the previously selected conversion(s).
        BestConversion = Prepared;
        HowManyGoodConversions = 1;
        continue;
      }

      if (BestConversionHasImplicit == ThisConversionHasImplicit)
        // The current conversion is the same in term of goodness than the
        // already selected one.
        ++HowManyGoodConversions;
    }

    if (HowManyGoodConversions == 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "--- selectUserDefinedConv. Unique result. Flags: "
                 << formatMixFlags(BestConversion->Flags) << '\n');
      return BestConversion;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "--- selectUserDefinedConv. No, or ambiguous.\n");
    return {};
  }

private:
  llvm::SmallVector<PreparedConversion, 2> FlaggedConversions;
  const TheCheck &Check;
};

} // namespace

static std::optional<ConversionSequence>
tryConversionOperators(const TheCheck &Check, const CXXRecordDecl *RD,
                       QualType ToType) {
  if (!RD || !RD->isCompleteDefinition())
    return {};
  RD = RD->getDefinition();

  LLVM_DEBUG(llvm::dbgs() << ">>> tryConversionOperators: " << RD->getName()
                          << " to:\n";
             ToType.dump(llvm::dbgs(), RD->getASTContext());
             llvm::dbgs() << '\n';);

  UserDefinedConversionSelector ConversionSet{Check};

  for (const NamedDecl *Method : RD->getVisibleConversionFunctions()) {
    const auto *Con = dyn_cast<CXXConversionDecl>(Method);
    if (!Con || Con->isExplicit())
      continue;
    LLVM_DEBUG(llvm::dbgs() << "--- tryConversionOperators. Trying:\n";
               Con->dump(llvm::dbgs()); llvm::dbgs() << '\n';);

    // Try to go from the result of conversion operator to the expected type,
    // without calculating another user-defined conversion.
    ConversionSet.addConversion(Con, Con->getConversionType(), ToType);
  }

  if (std::optional<UserDefinedConversionSelector::PreparedConversion>
          SelectedConversion = ConversionSet()) {
    CanQualType RecordType = RD->getASTContext().getCanonicalTagType(RD);

    ConversionSequence Result{RecordType, ToType};
    // The conversion from the operator call's return type to ToType was
    // modelled as a "pre-conversion" in the operator call, but it is the
    // "post-conversion" from the point of view of the original conversion
    // we are modelling.
    Result.AfterSecondStandard = SelectedConversion->Seq.AfterFirstStandard;

    ConversionSequence::UserDefinedConversionOperator ConvOp;
    ConvOp.Fun = cast<CXXConversionDecl>(SelectedConversion->ConversionFun);
    ConvOp.UserDefinedType = RecordType;
    ConvOp.ConversionOperatorResultType = ConvOp.Fun->getConversionType();
    Result.setConversion(ConvOp);

    LLVM_DEBUG(llvm::dbgs() << "<<< tryConversionOperators. Found result.\n");
    return Result;
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< tryConversionOperators. No conversion.\n");
  return {};
}

static std::optional<ConversionSequence>
tryConvertingConstructors(const TheCheck &Check, QualType FromType,
                          const CXXRecordDecl *RD) {
  if (!RD || !RD->isCompleteDefinition())
    return {};
  RD = RD->getDefinition();

  LLVM_DEBUG(llvm::dbgs() << ">>> tryConveringConstructors: " << RD->getName()
                          << " from:\n";
             FromType.dump(llvm::dbgs(), RD->getASTContext());
             llvm::dbgs() << '\n';);

  UserDefinedConversionSelector ConversionSet{Check};

  for (const CXXConstructorDecl *Con : RD->ctors()) {
    if (Con->isCopyOrMoveConstructor() ||
        !Con->isConvertingConstructor(/* AllowExplicit =*/false))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "--- tryConvertingConstructors. Trying:\n";
               Con->dump(llvm::dbgs()); llvm::dbgs() << '\n';);

    // Try to go from the original FromType to the converting constructor's
    // parameter type without another user-defined conversion.
    ConversionSet.addConversion(Con, FromType, Con->getParamDecl(0)->getType());
  }

  if (std::optional<UserDefinedConversionSelector::PreparedConversion>
          SelectedConversion = ConversionSet()) {
    CanQualType RecordType = RD->getASTContext().getCanonicalTagType(RD);

    ConversionSequence Result{FromType, RecordType};
    Result.AfterFirstStandard = SelectedConversion->Seq.AfterFirstStandard;

    ConversionSequence::UserDefinedConvertingConstructor Ctor;
    Ctor.Fun = cast<CXXConstructorDecl>(SelectedConversion->ConversionFun);
    Ctor.ConstructorParameterType = Ctor.Fun->getParamDecl(0)->getType();
    Ctor.UserDefinedType = RecordType;
    Result.setConversion(Ctor);

    LLVM_DEBUG(llvm::dbgs()
               << "<<< tryConvertingConstructors. Found result.\n");
    return Result;
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< tryConvertingConstructors. No conversion.\n");
  return {};
}

/// Returns whether an expression of LType can be used in an RType context, as
/// per the implicit conversion rules.
///
/// Note: the result of this operation, unlike that of calculateMixability, is
/// **NOT** symmetric.
static MixData
approximateImplicitConversion(const TheCheck &Check, QualType LType,
                              QualType RType, const ASTContext &Ctx,
                              ImplicitConversionModellingMode ImplicitMode) {
  LLVM_DEBUG(llvm::dbgs() << ">>> approximateImplicitConversion for LType:\n";
             LType.dump(llvm::dbgs(), Ctx); llvm::dbgs() << "\nand RType:\n";
             RType.dump(llvm::dbgs(), Ctx);
             llvm::dbgs() << "\nimplicit mode: "; switch (ImplicitMode) {
               case ImplicitConversionModellingMode::None:
                 llvm::dbgs() << "None";
                 break;
               case ImplicitConversionModellingMode::All:
                 llvm::dbgs() << "All";
                 break;
               case ImplicitConversionModellingMode::OneWaySingleStandardOnly:
                 llvm::dbgs() << "OneWay, Single, STD Only";
                 break;
             } llvm::dbgs() << '\n';);
  if (LType == RType)
    return {MixFlags::Trivial, LType};

  // An implicit conversion sequence consists of the following, in order:
  //  * Maybe standard conversion sequence.
  //  * Maybe user-defined conversion.
  //  * Maybe standard conversion sequence.
  ConversionSequence ImplicitSeq{LType, RType};
  QualType WorkType = LType;

  std::optional<QualType> AfterFirstStdConv =
      approximateStandardConversionSequence(Check, LType, RType, Ctx);
  if (AfterFirstStdConv) {
    LLVM_DEBUG(llvm::dbgs() << "--- approximateImplicitConversion. Standard "
                               "Pre-Conversion found!\n");
    ImplicitSeq.AfterFirstStandard = *AfterFirstStdConv;
    WorkType = ImplicitSeq.AfterFirstStandard;
  }

  if (ImplicitMode == ImplicitConversionModellingMode::OneWaySingleStandardOnly)
    // If the caller only requested modelling of a standard conversion, bail.
    return {ImplicitSeq.AfterFirstStandard.isNull()
                ? MixFlags::None
                : MixFlags::ImplicitConversion,
            ImplicitSeq};

  if (Ctx.getLangOpts().CPlusPlus) {
    bool FoundConversionOperator = false, FoundConvertingCtor = false;

    if (const auto *LRD = WorkType->getAsCXXRecordDecl()) {
      std::optional<ConversionSequence> ConversionOperatorResult =
          tryConversionOperators(Check, LRD, RType);
      if (ConversionOperatorResult) {
        LLVM_DEBUG(llvm::dbgs() << "--- approximateImplicitConversion. Found "
                                   "conversion operator.\n");
        ImplicitSeq.update(*ConversionOperatorResult);
        WorkType = ImplicitSeq.getTypeAfterUserDefinedConversion();
        FoundConversionOperator = true;
      }
    }

    if (const auto *RRD = RType->getAsCXXRecordDecl()) {
      // Use the original "LType" here, and not WorkType, because the
      // conversion to the converting constructors' parameters will be
      // modelled in the recursive call.
      std::optional<ConversionSequence> ConvCtorResult =
          tryConvertingConstructors(Check, LType, RRD);
      if (ConvCtorResult) {
        LLVM_DEBUG(llvm::dbgs() << "--- approximateImplicitConversion. Found "
                                   "converting constructor.\n");
        ImplicitSeq.update(*ConvCtorResult);
        WorkType = ImplicitSeq.getTypeAfterUserDefinedConversion();
        FoundConvertingCtor = true;
      }
    }

    if (FoundConversionOperator && FoundConvertingCtor) {
      // If both an operator and a ctor matches, the sequence is ambiguous.
      LLVM_DEBUG(llvm::dbgs()
                 << "<<< approximateImplicitConversion. Found both "
                    "user-defined conversion kinds in the same sequence!\n");
      return {MixFlags::None};
    }
  }

  // After the potential user-defined conversion, another standard conversion
  // sequence might exist.
  LLVM_DEBUG(
      llvm::dbgs()
      << "--- approximateImplicitConversion. Try to find post-conversion.\n");
  MixData SecondStdConv = approximateImplicitConversion(
      Check, WorkType, RType, Ctx,
      ImplicitConversionModellingMode::OneWaySingleStandardOnly);
  if (SecondStdConv.indicatesMixability()) {
    LLVM_DEBUG(llvm::dbgs() << "--- approximateImplicitConversion. Standard "
                               "Post-Conversion found!\n");

    // The single-step modelling puts the modelled conversion into the "PreStd"
    // variable in the recursive call, but from the PoV of this function, it is
    // the post-conversion.
    ImplicitSeq.AfterSecondStandard =
        SecondStdConv.Conversion.AfterFirstStandard;
    WorkType = ImplicitSeq.AfterSecondStandard;
  }

  if (ImplicitSeq) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< approximateImplicitConversion. Found a conversion.\n");
    return {MixFlags::ImplicitConversion, ImplicitSeq};
  }

  LLVM_DEBUG(
      llvm::dbgs() << "<<< approximateImplicitConversion. No match found.\n");
  return {MixFlags::None};
}

static MixableParameterRange modelMixingRange(
    const TheCheck &Check, const FunctionDecl *FD, std::size_t StartIndex,
    const filter::SimilarlyUsedParameterPairSuppressor &UsageBasedSuppressor) {
  std::size_t NumParams = FD->getNumParams();
  assert(StartIndex < NumParams && "out of bounds for start");
  const ASTContext &Ctx = FD->getASTContext();

  MixableParameterRange Ret;
  // A parameter at index 'StartIndex' had been trivially "checked".
  Ret.NumParamsChecked = 1;

  for (std::size_t I = StartIndex + 1; I < NumParams; ++I) {
    const ParmVarDecl *Ith = FD->getParamDecl(I);
    StringRef ParamName = Ith->getName();
    LLVM_DEBUG(llvm::dbgs()
               << "Check param #" << I << " '" << ParamName << "'...\n");
    if (filter::isIgnoredParameter(Check, Ith)) {
      LLVM_DEBUG(llvm::dbgs() << "Param #" << I << " is ignored. Break!\n");
      break;
    }

    StringRef PrevParamName = FD->getParamDecl(I - 1)->getName();
    if (!ParamName.empty() && !PrevParamName.empty() &&
        filter::prefixSuffixCoverUnderThreshold(
            Check.NamePrefixSuffixSilenceDissimilarityThreshold, PrevParamName,
            ParamName)) {
      LLVM_DEBUG(llvm::dbgs() << "Parameter '" << ParamName
                              << "' follows a pattern with previous parameter '"
                              << PrevParamName << "'. Break!\n");
      break;
    }

    // Now try to go forward and build the range of [Start, ..., I, I + 1, ...]
    // parameters that can be messed up at a call site.
    MixableParameterRange::MixVector MixesOfIth;
    for (std::size_t J = StartIndex; J < I; ++J) {
      const ParmVarDecl *Jth = FD->getParamDecl(J);
      LLVM_DEBUG(llvm::dbgs()
                 << "Check mix of #" << J << " against #" << I << "...\n");

      if (isSimilarlyUsedParameter(UsageBasedSuppressor, Ith, Jth)) {
        // Consider the two similarly used parameters to not be possible in a
        // mix-up at the user's request, if they enabled this heuristic.
        LLVM_DEBUG(llvm::dbgs() << "Parameters #" << I << " and #" << J
                                << " deemed related, ignoring...\n");

        // If the parameter #I and #J mixes, then I is mixable with something
        // in the current range, so the range has to be broken and I not
        // included.
        MixesOfIth.clear();
        break;
      }

      Mix M{Jth, Ith,
            calculateMixability(Check, Jth->getType(), Ith->getType(), Ctx,
                                Check.ModelImplicitConversions
                                    ? ImplicitConversionModellingMode::All
                                    : ImplicitConversionModellingMode::None)};
      LLVM_DEBUG(llvm::dbgs() << "Mix flags (raw)           : "
                              << formatMixFlags(M.flags()) << '\n');
      M.sanitize();
      LLVM_DEBUG(llvm::dbgs() << "Mix flags (after sanitize): "
                              << formatMixFlags(M.flags()) << '\n');

      assert(M.flagsValid() && "All flags decayed!");

      if (M.mixable())
        MixesOfIth.emplace_back(std::move(M));
    }

    if (MixesOfIth.empty()) {
      // If there weren't any new mixes stored for Ith, the range is
      // [Start, ..., I].
      LLVM_DEBUG(llvm::dbgs()
                 << "Param #" << I
                 << " does not mix with any in the current range. Break!\n");
      break;
    }

    Ret.Mixes.insert(Ret.Mixes.end(), MixesOfIth.begin(), MixesOfIth.end());
    ++Ret.NumParamsChecked; // Otherwise a new param was iterated.
  }

  return Ret;
}

} // namespace model

namespace {
/// Matches DeclRefExprs and their ignorable wrappers to ParmVarDecls.
AST_MATCHER_FUNCTION(ast_matchers::internal::Matcher<Stmt>, paramRefExpr) {
  return expr(ignoringParenImpCasts(ignoringElidableConstructorCall(
      declRefExpr(to(parmVarDecl().bind("param"))))));
}
} // namespace

namespace filter {

/// Returns whether the parameter's name or the parameter's type's name is
/// configured by the user to be ignored from analysis and diagnostic.
static bool isIgnoredParameter(const TheCheck &Check, const ParmVarDecl *Node) {
  LLVM_DEBUG(llvm::dbgs() << "Checking if '" << Node->getName()
                          << "' is ignored.\n");

  if (!Node->getIdentifier())
    return llvm::is_contained(Check.IgnoredParameterNames, "\"\"");

  StringRef NodeName = Node->getName();
  if (llvm::is_contained(Check.IgnoredParameterNames, NodeName)) {
    LLVM_DEBUG(llvm::dbgs() << "\tName ignored.\n");
    return true;
  }

  StringRef NodeTypeName = [Node] {
    const ASTContext &Ctx = Node->getASTContext();
    const SourceManager &SM = Ctx.getSourceManager();
    SourceLocation B = Node->getTypeSpecStartLoc();
    SourceLocation E = Node->getTypeSpecEndLoc();
    LangOptions LO;

    LLVM_DEBUG(llvm::dbgs() << "\tType name code is '"
                            << Lexer::getSourceText(
                                   CharSourceRange::getTokenRange(B, E), SM, LO)
                            << "'...\n");
    if (B.isMacroID()) {
      LLVM_DEBUG(llvm::dbgs() << "\t\tBeginning is macro.\n");
      B = SM.getTopMacroCallerLoc(B);
    }
    if (E.isMacroID()) {
      LLVM_DEBUG(llvm::dbgs() << "\t\tEnding is macro.\n");
      E = Lexer::getLocForEndOfToken(SM.getTopMacroCallerLoc(E), 0, SM, LO);
    }
    LLVM_DEBUG(llvm::dbgs() << "\tType name code is '"
                            << Lexer::getSourceText(
                                   CharSourceRange::getTokenRange(B, E), SM, LO)
                            << "'...\n");

    return Lexer::getSourceText(CharSourceRange::getTokenRange(B, E), SM, LO);
  }();

  LLVM_DEBUG(llvm::dbgs() << "\tType name is '" << NodeTypeName << "'\n");
  if (!NodeTypeName.empty()) {
    if (llvm::any_of(Check.IgnoredParameterTypeSuffixes,
                     [NodeTypeName](StringRef E) {
                       return !E.empty() && NodeTypeName.ends_with(E);
                     })) {
      LLVM_DEBUG(llvm::dbgs() << "\tType suffix ignored.\n");
      return true;
    }
  }

  return false;
}

/// This namespace contains the implementations for the suppression of
/// diagnostics from similarly-used ("related") parameters.
namespace relatedness_heuristic {

static constexpr std::size_t SmallDataStructureSize = 4;

template <typename T, std::size_t N = SmallDataStructureSize>
using ParamToSmallSetMap =
    llvm::DenseMap<const ParmVarDecl *, llvm::SmallSet<T, N>>;

template <typename T, std::size_t N = SmallDataStructureSize>
using ParamToSmallPtrSetMap =
    llvm::DenseMap<const ParmVarDecl *, llvm::SmallPtrSet<T, N>>;

/// Returns whether the sets mapped to the two elements in the map have at
/// least one element in common.
template <typename MapTy, typename ElemTy>
static bool lazyMapOfSetsIntersectionExists(const MapTy &Map, const ElemTy &E1,
                                            const ElemTy &E2) {
  auto E1Iterator = Map.find(E1);
  auto E2Iterator = Map.find(E2);
  if (E1Iterator == Map.end() || E2Iterator == Map.end())
    return false;

  for (const auto &E1SetElem : E1Iterator->second)
    if (E2Iterator->second.contains(E1SetElem))
      return true;

  return false;
}

/// Implements the heuristic that marks two parameters related if there is
/// a usage for both in the same strict expression subtree. A strict
/// expression subtree is a tree which only includes Expr nodes, i.e. no
/// Stmts and no Decls.
class AppearsInSameExpr : public RecursiveASTVisitor<AppearsInSameExpr> {
  using Base = RecursiveASTVisitor<AppearsInSameExpr>;

  const FunctionDecl *FD;
  const Expr *CurrentExprOnlyTreeRoot = nullptr;
  llvm::DenseMap<const ParmVarDecl *,
                 llvm::SmallPtrSet<const Expr *, SmallDataStructureSize>>
      ParentExprsForParamRefs;

public:
  void setup(const FunctionDecl *FD) {
    this->FD = FD;
    TraverseFunctionDecl(const_cast<FunctionDecl *>(FD));
  }

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return lazyMapOfSetsIntersectionExists(ParentExprsForParamRefs, Param1,
                                           Param2);
  }

  bool TraverseDecl(Decl *D) {
    CurrentExprOnlyTreeRoot = nullptr;
    return Base::TraverseDecl(D);
  }

  bool TraverseStmt(Stmt *S, DataRecursionQueue *Queue = nullptr) {
    if (auto *E = dyn_cast_or_null<Expr>(S)) {
      bool RootSetInCurrentStackFrame = false;
      if (!CurrentExprOnlyTreeRoot) {
        CurrentExprOnlyTreeRoot = E;
        RootSetInCurrentStackFrame = true;
      }

      bool Ret = Base::TraverseStmt(S);

      if (RootSetInCurrentStackFrame)
        CurrentExprOnlyTreeRoot = nullptr;

      return Ret;
    }

    // A Stmt breaks the strictly Expr subtree.
    CurrentExprOnlyTreeRoot = nullptr;
    return Base::TraverseStmt(S);
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (!CurrentExprOnlyTreeRoot)
      return true;

    if (auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl()))
      if (llvm::find(FD->parameters(), PVD))
        ParentExprsForParamRefs[PVD].insert(CurrentExprOnlyTreeRoot);

    return true;
  }
};

/// Implements the heuristic that marks two parameters related if there are
/// two separate calls to the same function (overload) and the parameters are
/// passed to the same index in both calls, i.e f(a, b) and f(a, c) passes
/// b and c to the same index (2) of f(), marking them related.
class PassedToSameFunction {
  ParamToSmallSetMap<std::pair<const FunctionDecl *, unsigned>> TargetParams;

public:
  void setup(const FunctionDecl *FD) {
    auto ParamsAsArgsInFnCalls =
        match(functionDecl(forEachDescendant(
                  callExpr(forEachArgumentWithParam(
                               paramRefExpr(), parmVarDecl().bind("passed-to")))
                      .bind("call-expr"))),
              *FD, FD->getASTContext());
    for (const auto &Match : ParamsAsArgsInFnCalls) {
      const auto *PassedParamOfThisFn = Match.getNodeAs<ParmVarDecl>("param");
      const auto *CE = Match.getNodeAs<CallExpr>("call-expr");
      const auto *PassedToParam = Match.getNodeAs<ParmVarDecl>("passed-to");
      assert(PassedParamOfThisFn && CE && PassedToParam);

      const FunctionDecl *CalledFn = CE->getDirectCallee();
      if (!CalledFn)
        continue;

      std::optional<unsigned> TargetIdx;
      unsigned NumFnParams = CalledFn->getNumParams();
      for (unsigned Idx = 0; Idx < NumFnParams; ++Idx)
        if (CalledFn->getParamDecl(Idx) == PassedToParam)
          TargetIdx.emplace(Idx);

      assert(TargetIdx && "Matched, but didn't find index?");
      TargetParams[PassedParamOfThisFn].insert(
          {CalledFn->getCanonicalDecl(), *TargetIdx});
    }
  }

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return lazyMapOfSetsIntersectionExists(TargetParams, Param1, Param2);
  }
};

/// Implements the heuristic that marks two parameters related if the same
/// member is accessed (referred to) inside the current function's body.
class AccessedSameMemberOf {
  ParamToSmallPtrSetMap<const Decl *> AccessedMembers;

public:
  void setup(const FunctionDecl *FD) {
    auto MembersCalledOnParams = match(
        functionDecl(forEachDescendant(
            memberExpr(hasObjectExpression(paramRefExpr())).bind("mem-expr"))),
        *FD, FD->getASTContext());

    for (const auto &Match : MembersCalledOnParams) {
      const auto *AccessedParam = Match.getNodeAs<ParmVarDecl>("param");
      const auto *ME = Match.getNodeAs<MemberExpr>("mem-expr");
      assert(AccessedParam && ME);
      AccessedMembers[AccessedParam].insert(
          ME->getMemberDecl()->getCanonicalDecl());
    }
  }

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return lazyMapOfSetsIntersectionExists(AccessedMembers, Param1, Param2);
  }
};

/// Implements the heuristic that marks two parameters related if different
/// ReturnStmts return them from the function.
class Returned {
  llvm::SmallVector<const ParmVarDecl *, SmallDataStructureSize> ReturnedParams;

public:
  void setup(const FunctionDecl *FD) {
    // TODO: Handle co_return.
    auto ParamReturns = match(functionDecl(forEachDescendant(
                                  returnStmt(hasReturnValue(paramRefExpr())))),
                              *FD, FD->getASTContext());
    for (const auto &Match : ParamReturns) {
      const auto *ReturnedParam = Match.getNodeAs<ParmVarDecl>("param");
      assert(ReturnedParam);

      if (find(FD->parameters(), ReturnedParam) == FD->param_end())
        // Inside the subtree of a FunctionDecl there might be ReturnStmts of
        // a parameter that isn't the parameter of the function, e.g. in the
        // case of lambdas.
        continue;

      ReturnedParams.emplace_back(ReturnedParam);
    }
  }

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return llvm::is_contained(ReturnedParams, Param1) &&
           llvm::is_contained(ReturnedParams, Param2);
  }
};

} // namespace relatedness_heuristic

/// Helper class that is used to detect if two parameters of the same function
/// are used in a similar fashion, to suppress the result.
class SimilarlyUsedParameterPairSuppressor {
  const bool Enabled;
  relatedness_heuristic::AppearsInSameExpr SameExpr;
  relatedness_heuristic::PassedToSameFunction PassToFun;
  relatedness_heuristic::AccessedSameMemberOf SameMember;
  relatedness_heuristic::Returned Returns;

public:
  SimilarlyUsedParameterPairSuppressor(const FunctionDecl *FD, bool Enable)
      : Enabled(Enable) {
    if (!Enable)
      return;

    SameExpr.setup(FD);
    PassToFun.setup(FD);
    SameMember.setup(FD);
    Returns.setup(FD);
  }

  /// Returns whether the specified two parameters are deemed similarly used
  /// or related by the heuristics.
  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    if (!Enabled)
      return false;

    LLVM_DEBUG(llvm::dbgs()
               << "::: Matching similar usage / relatedness heuristic...\n");

    if (SameExpr(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs() << "::: Used in the same expression.\n");
      return true;
    }

    if (PassToFun(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "::: Passed to same function in different calls.\n");
      return true;
    }

    if (SameMember(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "::: Same member field access or method called.\n");
      return true;
    }

    if (Returns(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs() << "::: Both parameter returned.\n");
      return true;
    }

    LLVM_DEBUG(llvm::dbgs() << "::: None.\n");
    return false;
  }
};

// (This function hoists the call to operator() of the wrapper, so we do not
// need to define the previous class at the top of the file.)
static inline bool
isSimilarlyUsedParameter(const SimilarlyUsedParameterPairSuppressor &Suppressor,
                         const ParmVarDecl *Param1, const ParmVarDecl *Param2) {
  return Suppressor(Param1, Param2);
}

static void padStringAtEnd(SmallVectorImpl<char> &Str, std::size_t ToLen) {
  while (Str.size() < ToLen)
    Str.emplace_back('\0');
}

static void padStringAtBegin(SmallVectorImpl<char> &Str, std::size_t ToLen) {
  while (Str.size() < ToLen)
    Str.insert(Str.begin(), '\0');
}

static bool isCommonPrefixWithoutSomeCharacters(std::size_t N, StringRef S1,
                                                StringRef S2) {
  assert(S1.size() >= N && S2.size() >= N);
  StringRef S1Prefix = S1.take_front(S1.size() - N),
            S2Prefix = S2.take_front(S2.size() - N);
  return S1Prefix == S2Prefix && !S1Prefix.empty();
}

static bool isCommonSuffixWithoutSomeCharacters(std::size_t N, StringRef S1,
                                                StringRef S2) {
  assert(S1.size() >= N && S2.size() >= N);
  StringRef S1Suffix = S1.take_back(S1.size() - N),
            S2Suffix = S2.take_back(S2.size() - N);
  return S1Suffix == S2Suffix && !S1Suffix.empty();
}

/// Returns whether the two strings are prefixes or suffixes of each other with
/// at most Threshold characters differing on the non-common end.
static bool prefixSuffixCoverUnderThreshold(std::size_t Threshold,
                                            StringRef Str1, StringRef Str2) {
  if (Threshold == 0)
    return false;

  // Pad the two strings to the longer length.
  std::size_t BiggerLength = std::max(Str1.size(), Str2.size());

  if (BiggerLength <= Threshold)
    // If the length of the strings is still smaller than the threshold, they
    // would be covered by an empty prefix/suffix with the rest differing.
    // (E.g. "A" and "X" with Threshold = 1 would mean we think they are
    // similar and do not warn about them, which is a too eager assumption.)
    return false;

  SmallString<32> S1PadE{Str1}, S2PadE{Str2};
  padStringAtEnd(S1PadE, BiggerLength);
  padStringAtEnd(S2PadE, BiggerLength);

  if (isCommonPrefixWithoutSomeCharacters(
          Threshold, StringRef{S1PadE.begin(), BiggerLength},
          StringRef{S2PadE.begin(), BiggerLength}))
    return true;

  SmallString<32> S1PadB{Str1}, S2PadB{Str2};
  padStringAtBegin(S1PadB, BiggerLength);
  padStringAtBegin(S2PadB, BiggerLength);

  if (isCommonSuffixWithoutSomeCharacters(
          Threshold, StringRef{S1PadB.begin(), BiggerLength},
          StringRef{S2PadB.begin(), BiggerLength}))
    return true;

  return false;
}

} // namespace filter

namespace {

/// Matches functions that have at least the specified amount of parameters.
AST_MATCHER_P(FunctionDecl, parameterCountGE, unsigned, N) {
  return Node.getNumParams() >= N;
}

/// Matches *any* overloaded unary and binary operators.
AST_MATCHER(FunctionDecl, isOverloadedUnaryOrBinaryOperator) {
  switch (Node.getOverloadedOperator()) {
  case OO_None:
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Conditional:
  case OO_Coawait:
    return false;

  default:
    return Node.getNumParams() <= 2;
  }
}

} // namespace

/// Returns the DefaultMinimumLength if the Value of requested minimum length
/// is less than 2. Minimum lengths of 0 or 1 are not accepted.
static inline unsigned clampMinimumLength(const unsigned Value) {
  return Value < 2 ? DefaultMinimumLength : Value;
}

// FIXME: Maybe unneeded, getNameForDiagnostic() is expected to change to return
// a crafted location when the node itself is unnamed. (See D84658, D85033.)
/// Returns the diagnostic-friendly name of the node, or empty string.
static SmallString<64> getName(const NamedDecl *ND) {
  SmallString<64> Name;
  llvm::raw_svector_ostream OS{Name};
  ND->getNameForDiagnostic(OS, ND->getASTContext().getPrintingPolicy(), false);
  return Name;
}

/// Returns the diagnostic-friendly name of the node, or a constant value.
static SmallString<64> getNameOrUnnamed(const NamedDecl *ND) {
  auto Name = getName(ND);
  if (Name.empty())
    Name = "<unnamed>";
  return Name;
}

/// Returns whether a particular Mix between two parameters should have the
/// types involved diagnosed to the user. This is only a flag check.
static inline bool needsToPrintTypeInDiagnostic(const model::Mix &M) {
  using namespace model;
  return static_cast<bool>(
      M.flags() &
      (MixFlags::TypeAlias | MixFlags::ReferenceBind | MixFlags::Qualifiers));
}

/// Returns whether a particular Mix between the two parameters should have
/// implicit conversions elaborated.
static inline bool needsToElaborateImplicitConversion(const model::Mix &M) {
  return hasFlag(M.flags(), model::MixFlags::ImplicitConversion);
}

namespace {

/// This class formats a conversion sequence into a "Ty1 -> Ty2 -> Ty3" line
/// that can be used in diagnostics.
struct FormattedConversionSequence {
  std::string DiagnosticText;

  /// The formatted sequence is trivial if it is "Ty1 -> Ty2", but Ty1 and
  /// Ty2 are the types that are shown in the code. A trivial diagnostic
  /// does not need to be printed.
  bool Trivial = true;

  FormattedConversionSequence(const PrintingPolicy &PP,
                              StringRef StartTypeAsDiagnosed,
                              const model::ConversionSequence &Conv,
                              StringRef DestinationTypeAsDiagnosed) {
    llvm::raw_string_ostream OS{DiagnosticText};

    // Print the type name as it is printed in other places in the diagnostic.
    OS << '\'' << StartTypeAsDiagnosed << '\'';
    std::string LastAddedType = StartTypeAsDiagnosed.str();
    std::size_t NumElementsAdded = 1;

    // However, the parameter's defined type might not be what the implicit
    // conversion started with, e.g. if a typedef is found to convert.
    std::string SeqBeginTypeStr = Conv.Begin.getAsString(PP);
    std::string SeqEndTypeStr = Conv.End.getAsString(PP);
    if (StartTypeAsDiagnosed != SeqBeginTypeStr) {
      OS << " (as '" << SeqBeginTypeStr << "')";
      LastAddedType = SeqBeginTypeStr;
      Trivial = false;
    }

    auto AddType = [&](StringRef ToAdd) {
      if (LastAddedType != ToAdd && ToAdd != SeqEndTypeStr) {
        OS << " -> '" << ToAdd << "'";
        LastAddedType = ToAdd.str();
        ++NumElementsAdded;
      }
    };
    for (QualType InvolvedType : Conv.getInvolvedTypesInSequence())
      // Print every type that's unique in the sequence into the diagnosis.
      AddType(InvolvedType.getAsString(PP));

    if (LastAddedType != DestinationTypeAsDiagnosed) {
      OS << " -> '" << DestinationTypeAsDiagnosed << "'";
      LastAddedType = DestinationTypeAsDiagnosed.str();
      ++NumElementsAdded;
    }

    // Same reasoning as with the Begin, e.g. if the converted-to type is a
    // typedef, it will not be the same inside the conversion sequence (where
    // the model already tore off typedefs) as in the code.
    if (DestinationTypeAsDiagnosed != SeqEndTypeStr) {
      OS << " (as '" << SeqEndTypeStr << "')";
      LastAddedType = SeqEndTypeStr;
      Trivial = false;
    }

    if (Trivial && NumElementsAdded > 2)
      // If the thing is still marked trivial but we have more than the
      // from and to types added, it should not be trivial, and elaborated
      // when printing the diagnostic.
      Trivial = false;
  }
};

/// Retains the elements called with and returns whether the call is done with
/// a new element.
template <typename E, std::size_t N> class InsertOnce {
  llvm::SmallSet<E, N> CalledWith;

public:
  bool operator()(E El) { return CalledWith.insert(std::move(El)).second; }

  bool calledWith(const E &El) const { return CalledWith.contains(El); }
};

struct SwappedEqualQualTypePair {
  QualType LHSType, RHSType;

  bool operator==(const SwappedEqualQualTypePair &Other) const {
    return (LHSType == Other.LHSType && RHSType == Other.RHSType) ||
           (LHSType == Other.RHSType && RHSType == Other.LHSType);
  }

  bool operator<(const SwappedEqualQualTypePair &Other) const {
    return LHSType < Other.LHSType && RHSType < Other.RHSType;
  }
};

struct TypeAliasDiagnosticTuple {
  QualType LHSType, RHSType, CommonType;

  bool operator==(const TypeAliasDiagnosticTuple &Other) const {
    return CommonType == Other.CommonType &&
           ((LHSType == Other.LHSType && RHSType == Other.RHSType) ||
            (LHSType == Other.RHSType && RHSType == Other.LHSType));
  }

  bool operator<(const TypeAliasDiagnosticTuple &Other) const {
    return CommonType < Other.CommonType && LHSType < Other.LHSType &&
           RHSType < Other.RHSType;
  }
};

/// Helper class to only emit a diagnostic related to MixFlags::TypeAlias once.
class UniqueTypeAliasDiagnosticHelper
    : public InsertOnce<TypeAliasDiagnosticTuple, 8> {
  using Base = InsertOnce<TypeAliasDiagnosticTuple, 8>;

public:
  /// Returns whether the diagnostic for LHSType and RHSType which are both
  /// referring to CommonType being the same has not been emitted already.
  bool operator()(QualType LHSType, QualType RHSType, QualType CommonType) {
    if (CommonType.isNull() || CommonType == LHSType || CommonType == RHSType)
      return Base::operator()({LHSType, RHSType, {}});

    TypeAliasDiagnosticTuple ThreeTuple{LHSType, RHSType, CommonType};
    if (!Base::operator()(ThreeTuple))
      return false;

    bool AlreadySaidLHSAndCommonIsSame = calledWith({LHSType, CommonType, {}});
    bool AlreadySaidRHSAndCommonIsSame = calledWith({RHSType, CommonType, {}});
    if (AlreadySaidLHSAndCommonIsSame && AlreadySaidRHSAndCommonIsSame) {
      // "SomeInt == int" && "SomeOtherInt == int" => "Common(SomeInt,
      // SomeOtherInt) == int", no need to diagnose it. Save the 3-tuple only
      // for shortcut if it ever appears again.
      return false;
    }

    return true;
  }
};

} // namespace

EasilySwappableParametersCheck::EasilySwappableParametersCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MinimumLength(clampMinimumLength(
          Options.get("MinimumLength", DefaultMinimumLength))),
      IgnoredParameterNames(optutils::parseStringList(
          Options.get("IgnoredParameterNames", DefaultIgnoredParameterNames))),
      IgnoredParameterTypeSuffixes(optutils::parseStringList(
          Options.get("IgnoredParameterTypeSuffixes",
                      DefaultIgnoredParameterTypeSuffixes))),
      QualifiersMix(Options.get("QualifiersMix", DefaultQualifiersMix)),
      ModelImplicitConversions(Options.get("ModelImplicitConversions",
                                           DefaultModelImplicitConversions)),
      SuppressParametersUsedTogether(
          Options.get("SuppressParametersUsedTogether",
                      DefaultSuppressParametersUsedTogether)),
      NamePrefixSuffixSilenceDissimilarityThreshold(
          Options.get("NamePrefixSuffixSilenceDissimilarityThreshold",
                      DefaultNamePrefixSuffixSilenceDissimilarityTreshold)) {}

void EasilySwappableParametersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MinimumLength", MinimumLength);
  Options.store(Opts, "IgnoredParameterNames",
                optutils::serializeStringList(IgnoredParameterNames));
  Options.store(Opts, "IgnoredParameterTypeSuffixes",
                optutils::serializeStringList(IgnoredParameterTypeSuffixes));
  Options.store(Opts, "QualifiersMix", QualifiersMix);
  Options.store(Opts, "ModelImplicitConversions", ModelImplicitConversions);
  Options.store(Opts, "SuppressParametersUsedTogether",
                SuppressParametersUsedTogether);
  Options.store(Opts, "NamePrefixSuffixSilenceDissimilarityThreshold",
                NamePrefixSuffixSilenceDissimilarityThreshold);
}

void EasilySwappableParametersCheck::registerMatchers(MatchFinder *Finder) {
  const auto BaseConstraints = functionDecl(
      // Only report for definition nodes, as fixing the issues reported
      // requires the user to be able to change code.
      isDefinition(), parameterCountGE(MinimumLength),
      unless(isOverloadedUnaryOrBinaryOperator()));

  Finder->addMatcher(
      functionDecl(BaseConstraints,
                   unless(ast_matchers::isTemplateInstantiation()))
          .bind("func"),
      this);
  Finder->addMatcher(
      functionDecl(BaseConstraints, isExplicitTemplateSpecialization())
          .bind("func"),
      this);
}

void EasilySwappableParametersCheck::check(
    const MatchFinder::MatchResult &Result) {
  using namespace model;
  using namespace filter;

  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("func");
  assert(FD);

  const PrintingPolicy &PP = FD->getASTContext().getPrintingPolicy();
  std::size_t NumParams = FD->getNumParams();
  std::size_t MixableRangeStartIndex = 0;

  // Spawn one suppressor and if the user requested, gather information from
  // the AST for the parameters' usages.
  filter::SimilarlyUsedParameterPairSuppressor UsageBasedSuppressor{
      FD, SuppressParametersUsedTogether};

  LLVM_DEBUG(llvm::dbgs() << "Begin analysis of " << getName(FD) << " with "
                          << NumParams << " parameters...\n");
  while (MixableRangeStartIndex < NumParams) {
    if (isIgnoredParameter(*this, FD->getParamDecl(MixableRangeStartIndex))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Parameter #" << MixableRangeStartIndex << " ignored.\n");
      ++MixableRangeStartIndex;
      continue;
    }

    MixableParameterRange R = modelMixingRange(
        *this, FD, MixableRangeStartIndex, UsageBasedSuppressor);
    assert(R.NumParamsChecked > 0 && "Ensure forward progress!");
    MixableRangeStartIndex += R.NumParamsChecked;
    if (R.NumParamsChecked < MinimumLength) {
      LLVM_DEBUG(llvm::dbgs() << "Ignoring range of " << R.NumParamsChecked
                              << " lower than limit.\n");
      continue;
    }

    bool NeedsAnyTypeNote = llvm::any_of(R.Mixes, needsToPrintTypeInDiagnostic);
    bool HasAnyImplicits =
        llvm::any_of(R.Mixes, needsToElaborateImplicitConversion);
    const ParmVarDecl *First = R.getFirstParam(), *Last = R.getLastParam();
    std::string FirstParamTypeAsWritten = First->getType().getAsString(PP);
    {
      StringRef DiagText;

      if (HasAnyImplicits)
        DiagText = "%0 adjacent parameters of %1 of convertible types are "
                   "easily swapped by mistake";
      else if (NeedsAnyTypeNote)
        DiagText = "%0 adjacent parameters of %1 of similar type are easily "
                   "swapped by mistake";
      else
        DiagText = "%0 adjacent parameters of %1 of similar type ('%2') are "
                   "easily swapped by mistake";

      auto Diag = diag(First->getOuterLocStart(), DiagText)
                  << static_cast<unsigned>(R.NumParamsChecked) << FD;
      if (!NeedsAnyTypeNote)
        Diag << FirstParamTypeAsWritten;

      CharSourceRange HighlightRange = CharSourceRange::getTokenRange(
          First->getBeginLoc(), Last->getEndLoc());
      Diag << HighlightRange;
    }

    // There is a chance that the previous highlight did not succeed, e.g. when
    // the two parameters are on different lines. For clarity, show the user
    // the involved variable explicitly.
    diag(First->getLocation(), "the first parameter in the range is '%0'",
         DiagnosticIDs::Note)
        << getNameOrUnnamed(First)
        << CharSourceRange::getTokenRange(First->getLocation(),
                                          First->getLocation());
    diag(Last->getLocation(), "the last parameter in the range is '%0'",
         DiagnosticIDs::Note)
        << getNameOrUnnamed(Last)
        << CharSourceRange::getTokenRange(Last->getLocation(),
                                          Last->getLocation());

    // Helper classes to silence elaborative diagnostic notes that would be
    // too verbose.
    UniqueTypeAliasDiagnosticHelper UniqueTypeAlias;
    InsertOnce<SwappedEqualQualTypePair, 8> UniqueBindPower;
    InsertOnce<SwappedEqualQualTypePair, 8> UniqueImplicitConversion;

    for (const model::Mix &M : R.Mixes) {
      assert(M.mixable() && "Sentinel or false mix in result.");
      if (!needsToPrintTypeInDiagnostic(M) &&
          !needsToElaborateImplicitConversion(M))
        continue;

      // Typedefs might result in the type of the variable needing to be
      // emitted to a note diagnostic, so prepare it.
      const ParmVarDecl *LVar = M.First;
      const ParmVarDecl *RVar = M.Second;
      QualType LType = LVar->getType();
      QualType RType = RVar->getType();
      QualType CommonType = M.commonUnderlyingType();
      std::string LTypeStr = LType.getAsString(PP);
      std::string RTypeStr = RType.getAsString(PP);
      std::string CommonTypeStr = CommonType.getAsString(PP);

      if (hasFlag(M.flags(), MixFlags::TypeAlias) &&
          UniqueTypeAlias(LType, RType, CommonType)) {
        StringRef DiagText;
        bool ExplicitlyPrintCommonType = false;
        if (LTypeStr == CommonTypeStr || RTypeStr == CommonTypeStr) {
          if (hasFlag(M.flags(), MixFlags::Qualifiers))
            DiagText = "after resolving type aliases, '%0' and '%1' share a "
                       "common type";
          else
            DiagText =
                "after resolving type aliases, '%0' and '%1' are the same";
        } else if (!CommonType.isNull()) {
          DiagText = "after resolving type aliases, the common type of '%0' "
                     "and '%1' is '%2'";
          ExplicitlyPrintCommonType = true;
        }

        auto Diag =
            diag(LVar->getOuterLocStart(), DiagText, DiagnosticIDs::Note)
            << LTypeStr << RTypeStr;
        if (ExplicitlyPrintCommonType)
          Diag << CommonTypeStr;
      }

      if ((hasFlag(M.flags(), MixFlags::ReferenceBind) ||
           hasFlag(M.flags(), MixFlags::Qualifiers)) &&
          UniqueBindPower({LType, RType})) {
        StringRef DiagText = "'%0' and '%1' parameters accept and bind the "
                             "same kind of values";
        diag(RVar->getOuterLocStart(), DiagText, DiagnosticIDs::Note)
            << LTypeStr << RTypeStr;
      }

      if (needsToElaborateImplicitConversion(M) &&
          UniqueImplicitConversion({LType, RType})) {
        const model::ConversionSequence &LTR =
            M.leftToRightConversionSequence();
        const model::ConversionSequence &RTL =
            M.rightToLeftConversionSequence();
        FormattedConversionSequence LTRFmt{PP, LTypeStr, LTR, RTypeStr};
        FormattedConversionSequence RTLFmt{PP, RTypeStr, RTL, LTypeStr};

        StringRef DiagText = "'%0' and '%1' may be implicitly converted";
        if (!LTRFmt.Trivial || !RTLFmt.Trivial)
          DiagText = "'%0' and '%1' may be implicitly converted: %2, %3";

        {
          auto Diag =
              diag(RVar->getOuterLocStart(), DiagText, DiagnosticIDs::Note)
              << LTypeStr << RTypeStr;

          if (!LTRFmt.Trivial || !RTLFmt.Trivial)
            Diag << LTRFmt.DiagnosticText << RTLFmt.DiagnosticText;
        }

        StringRef ConversionFunctionDiagText =
            "the implicit conversion involves the "
            "%select{|converting constructor|conversion operator}0 "
            "declared here";
        if (const FunctionDecl *LFD = LTR.getUserDefinedConversionFunction())
          diag(LFD->getLocation(), ConversionFunctionDiagText,
               DiagnosticIDs::Note)
              << static_cast<unsigned>(LTR.UDConvKind)
              << LTR.getUserDefinedConversionHighlight();
        if (const FunctionDecl *RFD = RTL.getUserDefinedConversionFunction())
          diag(RFD->getLocation(), ConversionFunctionDiagText,
               DiagnosticIDs::Note)
              << static_cast<unsigned>(RTL.UDConvKind)
              << RTL.getUserDefinedConversionHighlight();
      }
    }
  }
}

} // namespace clang::tidy::bugprone
