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

  None = 1,      //< Mix between the two parameters is not possible.
  Trivial = 2,   //< The two mix trivially, and are the exact same type.
  Canonical = 4, //< The two mix because the types refer to the same
                 // CanonicalType, but we do not elaborate as to how.

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue =*/Canonical)
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

  SmallString<4> Str{"---"};

  if (hasFlag(F, MixFlags::None))
    // Shows the None bit explicitly, as it can be applied in the recursion
    // even if other bits are set.
    Str[0] = '!';
  if (hasFlag(F, MixFlags::Trivial))
    Str[1] = 'T';
  if (hasFlag(F, MixFlags::Canonical))
    Str[2] = 'C';

  return Str.str().str();
}

#else

static inline std::string formatMixFlags(MixFlags F);

#endif // NDEBUG

/// Contains the metadata for the mixability result between two types,
/// independently of which parameters they were calculated from.
struct MixData {
  MixFlags Flags;

  MixData(MixFlags Flags) : Flags(Flags) {}

  void sanitize() {
    assert(Flags != MixFlags::Invalid && "sanitize() called on invalid bitvec");
    // TODO: There will be statements here in further extensions of the check.
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
    return {MixFlags::Trivial};
  }

  // TODO: Implement more elaborate logic, such as typedef, implicit
  // conversions, etc.

  // If none of the previous logic found a match, try if Clang otherwise
  // believes the types to be the same.
  if (LType.getCanonicalType() == RType.getCanonicalType()) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< calculateMixability. Same CanonicalType.\n");
    return {MixFlags::Canonical};
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< calculateMixability. No match found.\n");
  return {MixFlags::None};
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
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("func");
  assert(FD);

  const PrintingPolicy &PP = FD->getASTContext().getPrintingPolicy();
  std::size_t NumParams = FD->getNumParams();
  std::size_t MixableRangeStartIndex = 0;

  LLVM_DEBUG(llvm::dbgs() << "Begin analysis of " << getName(FD) << " with "
                          << NumParams << " parameters...\n");
  while (MixableRangeStartIndex < NumParams) {
    if (filter::isIgnoredParameter(*this,
                                   FD->getParamDecl(MixableRangeStartIndex))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Parameter #" << MixableRangeStartIndex << " ignored.\n");
      ++MixableRangeStartIndex;
      continue;
    }

    model::MixableParameterRange R =
        model::modelMixingRange(*this, FD, MixableRangeStartIndex);
    assert(R.NumParamsChecked > 0 && "Ensure forward progress!");
    MixableRangeStartIndex += R.NumParamsChecked;
    if (R.NumParamsChecked < MinimumLength) {
      LLVM_DEBUG(llvm::dbgs() << "Ignoring range of " << R.NumParamsChecked
                              << " lower than limit.\n");
      continue;
    }

    const ParmVarDecl *First = R.getFirstParam(), *Last = R.getLastParam();
    std::string FirstParamTypeAsWritten = First->getType().getAsString(PP);
    {
      StringRef DiagText = "%0 adjacent parameters of %1 of similar type "
                           "('%2') are easily swapped by mistake";
      // TODO: This logic will get extended here with future flags.

      auto Diag = diag(First->getOuterLocStart(), DiagText)
                  << static_cast<unsigned>(R.NumParamsChecked) << FD
                  << FirstParamTypeAsWritten;

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
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
