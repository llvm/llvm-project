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
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "EasilySwappableParametersCheck"
#include "llvm/Support/Debug.h"

namespace optutils = clang::tidy::utils::options;

/// The default value for the MinimumLength check option.
static constexpr std::size_t DefaultMinimumLength = 2;

/// The default value for ignored parameter names.
static const std::string DefaultIgnoredParameterNames =
    optutils::serializeStringList({"\"\"", "iterator", "Iterator", "begin",
                                   "Begin", "end", "End", "first", "First",
                                   "last", "Last", "lhs", "LHS", "rhs", "RHS"});

/// The default value for ignored parameter type suffixes.
static const std::string DefaultIgnoredParameterTypeSuffixes =
    optutils::serializeStringList({"bool",
                                   "Bool",
                                   "_Bool",
                                   "it",
                                   "It",
                                   "iterator",
                                   "Iterator",
                                   "inputit",
                                   "InputIt",
                                   "forwardit",
                                   "FowardIt",
                                   "bidirit",
                                   "BidirIt",
                                   "constiterator",
                                   "const_iterator",
                                   "Const_Iterator",
                                   "Constiterator",
                                   "ConstIterator",
                                   "RandomIt",
                                   "randomit",
                                   "random_iterator",
                                   "ReverseIt",
                                   "reverse_iterator",
                                   "reverse_const_iterator",
                                   "ConstReverseIterator",
                                   "Const_Reverse_Iterator",
                                   "const_reverse_iterator"
                                   "Constreverseiterator",
                                   "constreverseiterator"});

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

using TheCheck = EasilySwappableParametersCheck;

namespace filter {
static bool isIgnoredParameter(const TheCheck &Check, const ParmVarDecl *Node);
} // namespace filter

namespace model {

/// The language features involved in allowing the mix between two parameters.
enum class MixFlags : unsigned char {
  Invalid = 0, //< Sentinel bit pattern. DO NOT USE!

  None = 1,           //< Mix between the two parameters is not possible.
  Trivial = 2,        //< The two mix trivially, and are the exact same type.
  Canonical = 4,      //< The two mix because the types refer to the same
                      // CanonicalType, but we do not elaborate as to how.
  TypeAlias = 8,      //< The path from one type to the other involves
                      // desugaring type aliases.
  ReferenceBind = 16, //< The mix involves the binding power of "const &".

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue =*/ReferenceBind)
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

  SmallString<8> Str{"-----"};

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

  return Str.str().str();
}

#else

static inline std::string formatMixFlags(MixFlags F);

#endif // NDEBUG

/// Contains the metadata for the mixability result between two types,
/// independently of which parameters they were calculated from.
struct MixData {
  /// The flag bits of the mix indicating what language features allow for it.
  MixFlags Flags;

  /// A potentially calculated common underlying type after desugaring, that
  /// both sides of the mix can originate from.
  QualType CommonType;

  MixData(MixFlags Flags) : Flags(Flags) {}
  MixData(MixFlags Flags, QualType CommonType)
      : Flags(Flags), CommonType(CommonType) {}

  void sanitize() {
    assert(Flags != MixFlags::Invalid && "sanitize() called on invalid bitvec");

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
  }

  MixData operator|(MixFlags EnableFlags) const {
    return {Flags | EnableFlags, CommonType};
  }
  MixData &operator|=(MixFlags EnableFlags) {
    Flags |= EnableFlags;
    return *this;
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
  QualType commonUnderlyingType() const { return Data.CommonType; }
};

// NOLINTNEXTLINE(misc-redundant-expression): Seems to be a bogus warning.
static_assert(std::is_trivially_copyable<Mix>::value &&
                  std::is_trivially_move_constructible<Mix>::value &&
                  std::is_trivially_move_assignable<Mix>::value,
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

static MixData isLRefEquallyBindingToType(const TheCheck &Check,
                                          const LValueReferenceType *LRef,
                                          QualType Ty, const ASTContext &Ctx,
                                          bool IsRefRHS);

/// Approximate the way how LType and RType might refer to "essentially the
/// same" type, in a sense that at a particular call site, an expression of
/// type LType and RType might be successfully passed to a variable (in our
/// specific case, a parameter) of type RType and LType, respectively.
/// Note the swapped order!
///
/// The returned data structure is not guaranteed to be properly set, as this
/// function is potentially recursive. It is the caller's responsibility to
/// call sanitize() on the result once the recursion is over.
static MixData calculateMixability(const TheCheck &Check, const QualType LType,
                                   const QualType RType,
                                   const ASTContext &Ctx) {
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
  if (isa<ParenType>(LType.getTypePtr())) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. LHS is ParenType.\n");
    return calculateMixability(Check, LType.getSingleStepDesugaredType(Ctx),
                               RType, Ctx);
  }
  if (isa<ParenType>(RType.getTypePtr())) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. RHS is ParenType.\n");
    return calculateMixability(Check, LType,
                               RType.getSingleStepDesugaredType(Ctx), Ctx);
  }

  // Dissolve typedefs.
  if (const auto *LTypedef = LType->getAs<TypedefType>()) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. LHS is typedef.\n");
    return calculateMixability(Check, LTypedef->desugar(), RType, Ctx) |
           MixFlags::TypeAlias;
  }
  if (const auto *RTypedef = RType->getAs<TypedefType>()) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. RHS is typedef.\n");
    return calculateMixability(Check, LType, RTypedef->desugar(), Ctx) |
           MixFlags::TypeAlias;
  }

  // At a particular call site, what could be passed to a 'T' or 'const T' might
  // also be passed to a 'const T &' without the call site putting a direct
  // side effect on the passed expressions.
  if (const auto *LRef = LType->getAs<LValueReferenceType>()) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. LHS is &.\n");
    return isLRefEquallyBindingToType(Check, LRef, RType, Ctx, false) |
           MixFlags::ReferenceBind;
  }
  if (const auto *RRef = RType->getAs<LValueReferenceType>()) {
    LLVM_DEBUG(llvm::dbgs() << "--- calculateMixability. RHS is &.\n");
    return isLRefEquallyBindingToType(Check, RRef, LType, Ctx, true) |
           MixFlags::ReferenceBind;
  }

  // If none of the previous logic found a match, try if Clang otherwise
  // believes the types to be the same.
  if (LType.getCanonicalType() == RType.getCanonicalType()) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< calculateMixability. Same CanonicalType.\n");
    return {MixFlags::Canonical, LType.getCanonicalType()};
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< calculateMixability. No match found.\n");
  return {MixFlags::None};
}

/// Calculates if the reference binds an expression of the given type. This is
/// true iff 'LRef' is some 'const T &' type, and the 'Ty' is 'T' or 'const T'.
static MixData isLRefEquallyBindingToType(const TheCheck &Check,
                                          const LValueReferenceType *LRef,
                                          QualType Ty, const ASTContext &Ctx,
                                          bool IsRefRHS) {
  LLVM_DEBUG(llvm::dbgs() << ">>> isLRefEquallyBindingToType for LRef:\n";
             LRef->dump(llvm::dbgs(), Ctx); llvm::dbgs() << "\nand Type:\n";
             Ty.dump(llvm::dbgs(), Ctx); llvm::dbgs() << '\n';);

  QualType ReferredType = LRef->getPointeeType();
  if (!ReferredType.isLocalConstQualified()) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< isLRefEquallyBindingToType. Not const ref.\n");
    return {MixFlags::None};
  };

  QualType NonConstReferredType = ReferredType;
  NonConstReferredType.removeLocalConst();
  if (ReferredType == Ty || NonConstReferredType == Ty) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "<<< isLRefEquallyBindingToType. Type of referred matches.\n");
    return {MixFlags::Trivial, ReferredType};
  }

  LLVM_DEBUG(
      llvm::dbgs()
      << "--- isLRefEquallyBindingToType. Checking mix for underlying type.\n");
  return IsRefRHS ? calculateMixability(Check, Ty, NonConstReferredType, Ctx)
                  : calculateMixability(Check, NonConstReferredType, Ty, Ctx);
}

static MixableParameterRange modelMixingRange(const TheCheck &Check,
                                              const FunctionDecl *FD,
                                              std::size_t StartIndex) {
  std::size_t NumParams = FD->getNumParams();
  assert(StartIndex < NumParams && "out of bounds for start");
  const ASTContext &Ctx = FD->getASTContext();

  MixableParameterRange Ret;
  // A parameter at index 'StartIndex' had been trivially "checked".
  Ret.NumParamsChecked = 1;

  for (std::size_t I = StartIndex + 1; I < NumParams; ++I) {
    const ParmVarDecl *Ith = FD->getParamDecl(I);
    LLVM_DEBUG(llvm::dbgs() << "Check param #" << I << "...\n");

    if (filter::isIgnoredParameter(Check, Ith)) {
      LLVM_DEBUG(llvm::dbgs() << "Param #" << I << " is ignored. Break!\n");
      break;
    }

    // Now try to go forward and build the range of [Start, ..., I, I + 1, ...]
    // parameters that can be messed up at a call site.
    MixableParameterRange::MixVector MixesOfIth;
    for (std::size_t J = StartIndex; J < I; ++J) {
      const ParmVarDecl *Jth = FD->getParamDecl(J);
      LLVM_DEBUG(llvm::dbgs()
                 << "Check mix of #" << J << " against #" << I << "...\n");

      Mix M{Jth, Ith,
            calculateMixability(Check, Jth->getType(), Ith->getType(), Ctx)};
      LLVM_DEBUG(llvm::dbgs() << "Mix flags (raw)           : "
                              << formatMixFlags(M.flags()) << '\n');
      M.sanitize();
      LLVM_DEBUG(llvm::dbgs() << "Mix flags (after sanitize): "
                              << formatMixFlags(M.flags()) << '\n');

      assert(M.flags() != MixFlags::Invalid && "All flags decayed!");

      if (M.flags() != MixFlags::None)
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

namespace filter {

/// Returns whether the parameter's name or the parameter's type's name is
/// configured by the user to be ignored from analysis and diagnostic.
static bool isIgnoredParameter(const TheCheck &Check, const ParmVarDecl *Node) {
  LLVM_DEBUG(llvm::dbgs() << "Checking if '" << Node->getName()
                          << "' is ignored.\n");

  if (!Node->getIdentifier())
    return llvm::find(Check.IgnoredParameterNames, "\"\"") !=
           Check.IgnoredParameterNames.end();

  StringRef NodeName = Node->getName();
  if (llvm::find(Check.IgnoredParameterNames, NodeName) !=
      Check.IgnoredParameterNames.end()) {
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
                     [NodeTypeName](const std::string &E) {
                       return !E.empty() && NodeTypeName.endswith(E);
                     })) {
      LLVM_DEBUG(llvm::dbgs() << "\tType suffix ignored.\n");
      return true;
    }
  }

  return false;
}

} // namespace filter

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
  return static_cast<bool>(M.flags() & (model::MixFlags::TypeAlias |
                                        model::MixFlags::ReferenceBind));
}

namespace {

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
                      DefaultIgnoredParameterTypeSuffixes))) {}

void EasilySwappableParametersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MinimumLength", MinimumLength);
  Options.store(Opts, "IgnoredParameterNames",
                optutils::serializeStringList(IgnoredParameterNames));
  Options.store(Opts, "IgnoredParameterTypeSuffixes",
                optutils::serializeStringList(IgnoredParameterTypeSuffixes));
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

  LLVM_DEBUG(llvm::dbgs() << "Begin analysis of " << getName(FD) << " with "
                          << NumParams << " parameters...\n");
  while (MixableRangeStartIndex < NumParams) {
    if (isIgnoredParameter(*this, FD->getParamDecl(MixableRangeStartIndex))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Parameter #" << MixableRangeStartIndex << " ignored.\n");
      ++MixableRangeStartIndex;
      continue;
    }

    MixableParameterRange R =
        modelMixingRange(*this, FD, MixableRangeStartIndex);
    assert(R.NumParamsChecked > 0 && "Ensure forward progress!");
    MixableRangeStartIndex += R.NumParamsChecked;
    if (R.NumParamsChecked < MinimumLength) {
      LLVM_DEBUG(llvm::dbgs() << "Ignoring range of " << R.NumParamsChecked
                              << " lower than limit.\n");
      continue;
    }

    bool NeedsAnyTypeNote = llvm::any_of(R.Mixes, needsToPrintTypeInDiagnostic);
    const ParmVarDecl *First = R.getFirstParam(), *Last = R.getLastParam();
    std::string FirstParamTypeAsWritten = First->getType().getAsString(PP);
    {
      StringRef DiagText;

      if (NeedsAnyTypeNote)
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

    for (const Mix &M : R.Mixes) {
      assert(M.flags() >= MixFlags::Trivial &&
             "Sentinel or false mix in result.");

      if (needsToPrintTypeInDiagnostic(M)) {
        // Typedefs might result in the type of the variable needing to be
        // emitted to a note diagnostic, so prepare it.
        const ParmVarDecl *LVar = M.First;
        const ParmVarDecl *RVar = M.Second;
        QualType LType = LVar->getType();
        QualType RType = RVar->getType();
        QualType CommonType = M.commonUnderlyingType();
        std::string LTypeAsWritten = LType.getAsString(PP);
        std::string RTypeAsWritten = RType.getAsString(PP);
        std::string CommonTypeStr = CommonType.getAsString(PP);

        if (hasFlag(M.flags(), MixFlags::TypeAlias) &&
            UniqueTypeAlias(LType, RType, CommonType)) {
          StringRef DiagText;
          bool ExplicitlyPrintCommonType = false;
          if (LTypeAsWritten == CommonTypeStr ||
              RTypeAsWritten == CommonTypeStr)
            DiagText =
                "after resolving type aliases, '%0' and '%1' are the same";
          else {
            DiagText = "after resolving type aliases, the common type of '%0' "
                       "and '%1' is '%2'";
            ExplicitlyPrintCommonType = true;
          }

          auto Diag =
              diag(LVar->getOuterLocStart(), DiagText, DiagnosticIDs::Note)
              << LTypeAsWritten << RTypeAsWritten;
          if (ExplicitlyPrintCommonType)
            Diag << CommonTypeStr;
        }

        if (hasFlag(M.flags(), MixFlags::ReferenceBind) &&
            UniqueBindPower({LType, RType})) {
          StringRef DiagText = "'%0' and '%1' parameters accept and bind the "
                               "same kind of values";
          diag(RVar->getOuterLocStart(), DiagText, DiagnosticIDs::Note)
              << LTypeAsWritten << RTypeAsWritten;
        }
      }
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
