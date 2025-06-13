//===--- SuspiciousCallArgumentCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousCallArgumentCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <optional>

using namespace clang::ast_matchers;
namespace optutils = clang::tidy::utils::options;

namespace clang::tidy::readability {

namespace {
struct DefaultHeuristicConfiguration {
  /// Whether the heuristic is to be enabled by default.
  const bool Enabled;

  /// The upper bound of % of similarity the two strings might have to be
  /// considered dissimilar.
  /// (For purposes of configuration, -1 if the heuristic is not configurable
  /// with bounds.)
  const int8_t DissimilarBelow;

  /// The lower bound of % of similarity the two string must have to be
  /// considered similar.
  /// (For purposes of configuration, -1 if the heuristic is not configurable
  /// with bounds.)
  const int8_t SimilarAbove;

  /// Can the heuristic be configured with bounds?
  bool hasBounds() const { return DissimilarBelow > -1 && SimilarAbove > -1; }
};
} // namespace

static constexpr std::size_t DefaultMinimumIdentifierNameLength = 3;

static constexpr StringRef HeuristicToString[] = {
    "Equality",  "Abbreviation", "Prefix",      "Suffix",
    "Substring", "Levenshtein",  "JaroWinkler", "Dice"};

static constexpr DefaultHeuristicConfiguration Defaults[] = {
    {true, -1, -1}, // Equality.
    {true, -1, -1}, // Abbreviation.
    {true, 25, 30}, // Prefix.
    {true, 25, 30}, // Suffix.
    {true, 40, 50}, // Substring.
    {true, 50, 66}, // Levenshtein.
    {true, 75, 85}, // Jaro-Winkler.
    {true, 60, 70}, // Dice.
};

static_assert(
    sizeof(HeuristicToString) / sizeof(HeuristicToString[0]) ==
        SuspiciousCallArgumentCheck::HeuristicCount,
    "Ensure that every heuristic has a corresponding stringified name");
static_assert(sizeof(Defaults) / sizeof(Defaults[0]) ==
                  SuspiciousCallArgumentCheck::HeuristicCount,
              "Ensure that every heuristic has a default configuration.");

namespace {
template <std::size_t I> struct HasWellConfiguredBounds {
  static constexpr bool Value =
      !((Defaults[I].DissimilarBelow == -1) ^ (Defaults[I].SimilarAbove == -1));
  static_assert(Value, "A heuristic must either have a dissimilarity and "
                       "similarity bound, or neither!");
};

template <std::size_t I> struct HasWellConfiguredBoundsFold {
  static constexpr bool Value = HasWellConfiguredBounds<I>::Value &&
                                HasWellConfiguredBoundsFold<I - 1>::Value;
};

template <> struct HasWellConfiguredBoundsFold<0> {
  static constexpr bool Value = HasWellConfiguredBounds<0>::Value;
};

struct AllHeuristicsBoundsWellConfigured {
  static constexpr bool Value =
      HasWellConfiguredBoundsFold<SuspiciousCallArgumentCheck::HeuristicCount -
                                  1>::Value;
};

static_assert(AllHeuristicsBoundsWellConfigured::Value);
} // namespace

static constexpr llvm::StringLiteral DefaultAbbreviations = "addr=address;"
                                                            "arr=array;"
                                                            "attr=attribute;"
                                                            "buf=buffer;"
                                                            "cl=client;"
                                                            "cnt=count;"
                                                            "col=column;"
                                                            "cpy=copy;"
                                                            "dest=destination;"
                                                            "dist=distance"
                                                            "dst=distance;"
                                                            "elem=element;"
                                                            "hght=height;"
                                                            "i=index;"
                                                            "idx=index;"
                                                            "len=length;"
                                                            "ln=line;"
                                                            "lst=list;"
                                                            "nr=number;"
                                                            "num=number;"
                                                            "pos=position;"
                                                            "ptr=pointer;"
                                                            "ref=reference;"
                                                            "src=source;"
                                                            "srv=server;"
                                                            "stmt=statement;"
                                                            "str=string;"
                                                            "val=value;"
                                                            "var=variable;"
                                                            "vec=vector;"
                                                            "wdth=width";

static constexpr std::size_t SmallVectorSize =
    SuspiciousCallArgumentCheck::SmallVectorSize;

/// Returns how many % X is of Y.
static inline double percentage(double X, double Y) { return X / Y * 100.0; }

static bool applyEqualityHeuristic(StringRef Arg, StringRef Param) {
  return Arg.equals_insensitive(Param);
}

static bool applyAbbreviationHeuristic(
    const llvm::StringMap<std::string> &AbbreviationDictionary, StringRef Arg,
    StringRef Param) {
  if (AbbreviationDictionary.contains(Arg) &&
      Param == AbbreviationDictionary.lookup(Arg))
    return true;

  if (AbbreviationDictionary.contains(Param) &&
      Arg == AbbreviationDictionary.lookup(Param))
    return true;

  return false;
}

/// Check whether the shorter String is a prefix of the longer String.
static bool applyPrefixHeuristic(StringRef Arg, StringRef Param,
                                 int8_t Threshold) {
  StringRef Shorter = Arg.size() < Param.size() ? Arg : Param;
  StringRef Longer = Arg.size() >= Param.size() ? Arg : Param;

  if (Longer.starts_with_insensitive(Shorter))
    return percentage(Shorter.size(), Longer.size()) > Threshold;

  return false;
}

/// Check whether the shorter String is a suffix of the longer String.
static bool applySuffixHeuristic(StringRef Arg, StringRef Param,
                                 int8_t Threshold) {
  StringRef Shorter = Arg.size() < Param.size() ? Arg : Param;
  StringRef Longer = Arg.size() >= Param.size() ? Arg : Param;

  if (Longer.ends_with_insensitive(Shorter))
    return percentage(Shorter.size(), Longer.size()) > Threshold;

  return false;
}

static bool applySubstringHeuristic(StringRef Arg, StringRef Param,
                                    int8_t Threshold) {

  std::size_t MaxLength = 0;
  SmallVector<std::size_t, SmallVectorSize> Current(Param.size());
  SmallVector<std::size_t, SmallVectorSize> Previous(Param.size());
  std::string ArgLower = Arg.lower();
  std::string ParamLower = Param.lower();

  for (std::size_t I = 0; I < Arg.size(); ++I) {
    for (std::size_t J = 0; J < Param.size(); ++J) {
      if (ArgLower[I] == ParamLower[J]) {
        if (I == 0 || J == 0)
          Current[J] = 1;
        else
          Current[J] = 1 + Previous[J - 1];

        MaxLength = std::max(MaxLength, Current[J]);
      } else
        Current[J] = 0;
    }

    Current.swap(Previous);
  }

  size_t LongerLength = std::max(Arg.size(), Param.size());
  return percentage(MaxLength, LongerLength) > Threshold;
}

static bool applyLevenshteinHeuristic(StringRef Arg, StringRef Param,
                                      int8_t Threshold) {
  std::size_t LongerLength = std::max(Arg.size(), Param.size());
  double Dist = Arg.edit_distance(Param);
  Dist = (1.0 - Dist / LongerLength) * 100.0;
  return Dist > Threshold;
}

// Based on http://en.wikipedia.org/wiki/Jaro–Winkler_distance.
static bool applyJaroWinklerHeuristic(StringRef Arg, StringRef Param,
                                      int8_t Threshold) {
  std::size_t Match = 0, Transpos = 0;
  std::ptrdiff_t ArgLen = Arg.size();
  std::ptrdiff_t ParamLen = Param.size();
  SmallVector<int, SmallVectorSize> ArgFlags(ArgLen);
  SmallVector<int, SmallVectorSize> ParamFlags(ParamLen);
  std::ptrdiff_t Range =
      std::max(std::ptrdiff_t{0}, std::max(ArgLen, ParamLen) / 2 - 1);

  // Calculate matching characters.
  for (std::ptrdiff_t I = 0; I < ParamLen; ++I)
    for (std::ptrdiff_t J = std::max(I - Range, std::ptrdiff_t{0}),
                        L = std::min(I + Range + 1, ArgLen);
         J < L; ++J)
      if (tolower(Param[I]) == tolower(Arg[J]) && !ArgFlags[J]) {
        ArgFlags[J] = 1;
        ParamFlags[I] = 1;
        ++Match;
        break;
      }

  if (!Match)
    return false;

  // Calculate character transpositions.
  std::ptrdiff_t L = 0;
  for (std::ptrdiff_t I = 0; I < ParamLen; ++I) {
    if (ParamFlags[I] == 1) {
      std::ptrdiff_t J = 0;
      for (J = L; J < ArgLen; ++J)
        if (ArgFlags[J] == 1) {
          L = J + 1;
          break;
        }

      if (tolower(Param[I]) != tolower(Arg[J]))
        ++Transpos;
    }
  }
  Transpos /= 2;

  // Jaro distance.
  double MatchD = Match;
  double Dist = ((MatchD / ArgLen) + (MatchD / ParamLen) +
                 ((MatchD - Transpos) / Match)) /
                3.0;

  // Calculate common string prefix up to 4 chars.
  L = 0;
  for (std::ptrdiff_t I = 0;
       I < std::min(std::min(ArgLen, ParamLen), std::ptrdiff_t{4}); ++I)
    if (tolower(Arg[I]) == tolower(Param[I]))
      ++L;

  // Jaro-Winkler distance.
  Dist = (Dist + (L * 0.1 * (1.0 - Dist))) * 100.0;
  return Dist > Threshold;
}

// Based on http://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
static bool applyDiceHeuristic(StringRef Arg, StringRef Param,
                               int8_t Threshold) {
  llvm::StringSet<> ArgBigrams;
  llvm::StringSet<> ParamBigrams;

  // Extract character bigrams from Arg.
  for (std::ptrdiff_t I = 0; I < static_cast<std::ptrdiff_t>(Arg.size()) - 1;
       ++I)
    ArgBigrams.insert(Arg.substr(I, 2).lower());

  // Extract character bigrams from Param.
  for (std::ptrdiff_t I = 0; I < static_cast<std::ptrdiff_t>(Param.size()) - 1;
       ++I)
    ParamBigrams.insert(Param.substr(I, 2).lower());

  std::size_t Intersection = 0;

  // Find the intersection between the two sets.
  for (auto IT = ParamBigrams.begin(); IT != ParamBigrams.end(); ++IT)
    Intersection += ArgBigrams.count((IT->getKey()));

  // Calculate Dice coefficient.
  return percentage(Intersection * 2.0,
                    ArgBigrams.size() + ParamBigrams.size()) > Threshold;
}

/// Checks if ArgType binds to ParamType regarding reference-ness and
/// cv-qualifiers.
static bool areRefAndQualCompatible(QualType ArgType, QualType ParamType,
                                    const ASTContext &Ctx) {
  return !ParamType->isReferenceType() ||
         ParamType.getNonReferenceType().isAtLeastAsQualifiedAs(
             ArgType.getNonReferenceType(), Ctx);
}

static bool isPointerOrArray(QualType TypeToCheck) {
  return TypeToCheck->isPointerType() || TypeToCheck->isArrayType();
}

/// Checks whether ArgType is an array type identical to ParamType's array type.
/// Enforces array elements' qualifier compatibility as well.
static bool isCompatibleWithArrayReference(QualType ArgType, QualType ParamType,
                                           const ASTContext &Ctx) {
  if (!ArgType->isArrayType())
    return false;
  // Here, qualifiers belong to the elements of the arrays.
  if (!ParamType.isAtLeastAsQualifiedAs(ArgType, Ctx))
    return false;

  return ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType();
}

static QualType convertToPointeeOrArrayElementQualType(QualType TypeToConvert) {
  unsigned CVRqualifiers = 0;
  // Save array element qualifiers, since getElementType() removes qualifiers
  // from array elements.
  if (TypeToConvert->isArrayType())
    CVRqualifiers = TypeToConvert.getLocalQualifiers().getCVRQualifiers();
  TypeToConvert = TypeToConvert->isPointerType()
                      ? TypeToConvert->getPointeeType()
                      : TypeToConvert->getAsArrayTypeUnsafe()->getElementType();
  TypeToConvert = TypeToConvert.withCVRQualifiers(CVRqualifiers);
  return TypeToConvert;
}

/// Checks if multilevel pointers' qualifiers compatibility continues on the
/// current pointer level. For multilevel pointers, C++ permits conversion, if
/// every cv-qualifier in ArgType also appears in the corresponding position in
/// ParamType, and if PramType has a cv-qualifier that's not in ArgType, then
/// every * in ParamType to the right of that cv-qualifier, except the last
/// one, must also be const-qualified.
static bool arePointersStillQualCompatible(QualType ArgType, QualType ParamType,
                                           bool &IsParamContinuouslyConst,
                                           const ASTContext &Ctx) {
  // The types are compatible, if the parameter is at least as qualified as the
  // argument, and if it is more qualified, it has to be const on upper pointer
  // levels.
  bool AreTypesQualCompatible =
      ParamType.isAtLeastAsQualifiedAs(ArgType, Ctx) &&
      (!ParamType.hasQualifiers() || IsParamContinuouslyConst);
  // Check whether the parameter's constness continues at the current pointer
  // level.
  IsParamContinuouslyConst &= ParamType.isConstQualified();

  return AreTypesQualCompatible;
}

/// Checks whether multilevel pointers are compatible in terms of levels,
/// qualifiers and pointee type.
static bool arePointerTypesCompatible(QualType ArgType, QualType ParamType,
                                      bool IsParamContinuouslyConst,
                                      const ASTContext &Ctx) {
  if (!arePointersStillQualCompatible(ArgType, ParamType,
                                      IsParamContinuouslyConst, Ctx))
    return false;

  do {
    // Step down one pointer level.
    ArgType = convertToPointeeOrArrayElementQualType(ArgType);
    ParamType = convertToPointeeOrArrayElementQualType(ParamType);

    // Check whether cv-qualifiers permit compatibility on
    // current level.
    if (!arePointersStillQualCompatible(ArgType, ParamType,
                                        IsParamContinuouslyConst, Ctx))
      return false;

    if (ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType())
      return true;

  } while (ParamType->isPointerType() && ArgType->isPointerType());
  // The final type does not match, or pointer levels differ.
  return false;
}

/// Checks whether ArgType converts implicitly to ParamType.
static bool areTypesCompatible(QualType ArgType, QualType ParamType,
                               const ASTContext &Ctx) {
  if (ArgType.isNull() || ParamType.isNull())
    return false;

  ArgType = ArgType.getCanonicalType();
  ParamType = ParamType.getCanonicalType();

  if (ArgType == ParamType)
    return true;

  // Check for constness and reference compatibility.
  if (!areRefAndQualCompatible(ArgType, ParamType, Ctx))
    return false;

  bool IsParamReference = ParamType->isReferenceType();

  // Reference-ness has already been checked and should be removed
  // before further checking.
  ArgType = ArgType.getNonReferenceType();
  ParamType = ParamType.getNonReferenceType();

  if (ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType())
    return true;

  // Arithmetic types are interconvertible, except scoped enums.
  if (ParamType->isArithmeticType() && ArgType->isArithmeticType()) {
    if ((ParamType->isEnumeralType() &&
         ParamType->castAs<EnumType>()->getDecl()->isScoped()) ||
        (ArgType->isEnumeralType() &&
         ArgType->castAs<EnumType>()->getDecl()->isScoped()))
      return false;

    return true;
  }

  // Check if the argument and the param are both function types (the parameter
  // decayed to a function pointer).
  if (ArgType->isFunctionType() && ParamType->isFunctionPointerType()) {
    ParamType = ParamType->getPointeeType();
    return ArgType == ParamType;
  }

  // Arrays or pointer arguments convert to array or pointer parameters.
  if (!(isPointerOrArray(ArgType) && isPointerOrArray(ParamType)))
    return false;

  // When ParamType is an array reference, ArgType has to be of the same-sized
  // array-type with cv-compatible element type.
  if (IsParamReference && ParamType->isArrayType())
    return isCompatibleWithArrayReference(ArgType, ParamType, Ctx);

  bool IsParamContinuouslyConst =
      !IsParamReference || ParamType.getNonReferenceType().isConstQualified();

  // Remove the first level of indirection.
  ArgType = convertToPointeeOrArrayElementQualType(ArgType);
  ParamType = convertToPointeeOrArrayElementQualType(ParamType);

  // Check qualifier compatibility on the next level.
  if (!ParamType.isAtLeastAsQualifiedAs(ArgType, Ctx))
    return false;

  if (ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType())
    return true;

  // At this point, all possible C language implicit conversion were checked.
  if (!Ctx.getLangOpts().CPlusPlus)
    return false;

  // Check whether ParamType and ArgType were both pointers to a class or a
  // struct, and check for inheritance.
  if (ParamType->isStructureOrClassType() &&
      ArgType->isStructureOrClassType()) {
    const auto *ArgDecl = ArgType->getAsCXXRecordDecl();
    const auto *ParamDecl = ParamType->getAsCXXRecordDecl();
    if (!ArgDecl || !ArgDecl->hasDefinition() || !ParamDecl ||
        !ParamDecl->hasDefinition())
      return false;

    return ArgDecl->isDerivedFrom(ParamDecl);
  }

  // Unless argument and param are both multilevel pointers, the types are not
  // convertible.
  if (!(ParamType->isAnyPointerType() && ArgType->isAnyPointerType()))
    return false;

  return arePointerTypesCompatible(ArgType, ParamType, IsParamContinuouslyConst,
                                   Ctx);
}

static bool isOverloadedUnaryOrBinarySymbolOperator(const FunctionDecl *FD) {
  switch (FD->getOverloadedOperator()) {
  case OO_None:
  case OO_Call:
  case OO_Subscript:
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Conditional:
  case OO_Coawait:
    return false;

  default:
    return FD->getNumParams() <= 2;
  }
}

SuspiciousCallArgumentCheck::SuspiciousCallArgumentCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MinimumIdentifierNameLength(Options.get(
          "MinimumIdentifierNameLength", DefaultMinimumIdentifierNameLength)) {
  auto GetToggleOpt = [this](Heuristic H) -> bool {
    auto Idx = static_cast<std::size_t>(H);
    assert(Idx < HeuristicCount);
    return Options.get(HeuristicToString[Idx], Defaults[Idx].Enabled);
  };
  auto GetBoundOpt = [this](Heuristic H, BoundKind BK) -> int8_t {
    auto Idx = static_cast<std::size_t>(H);
    assert(Idx < HeuristicCount);

    SmallString<32> Key = HeuristicToString[Idx];
    Key.append(BK == BoundKind::DissimilarBelow ? "DissimilarBelow"
                                                : "SimilarAbove");
    int8_t Default = BK == BoundKind::DissimilarBelow
                         ? Defaults[Idx].DissimilarBelow
                         : Defaults[Idx].SimilarAbove;
    return Options.get(Key, Default);
  };
  for (std::size_t Idx = 0; Idx < HeuristicCount; ++Idx) {
    auto H = static_cast<Heuristic>(Idx);
    if (GetToggleOpt(H))
      AppliedHeuristics.emplace_back(H);
    ConfiguredBounds.emplace_back(
        std::make_pair(GetBoundOpt(H, BoundKind::DissimilarBelow),
                       GetBoundOpt(H, BoundKind::SimilarAbove)));
  }

  for (StringRef Abbreviation : optutils::parseStringList(
           Options.get("Abbreviations", DefaultAbbreviations))) {
    auto KeyAndValue = Abbreviation.split("=");
    assert(!KeyAndValue.first.empty() && !KeyAndValue.second.empty());
    AbbreviationDictionary.insert(
        std::make_pair(KeyAndValue.first, KeyAndValue.second.str()));
  }
}

void SuspiciousCallArgumentCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MinimumIdentifierNameLength",
                MinimumIdentifierNameLength);
  const auto &SetToggleOpt = [this, &Opts](Heuristic H) -> void {
    auto Idx = static_cast<std::size_t>(H);
    Options.store(Opts, HeuristicToString[Idx], isHeuristicEnabled(H));
  };
  const auto &SetBoundOpt = [this, &Opts](Heuristic H, BoundKind BK) -> void {
    auto Idx = static_cast<std::size_t>(H);
    assert(Idx < HeuristicCount);
    if (!Defaults[Idx].hasBounds())
      return;

    SmallString<32> Key = HeuristicToString[Idx];
    Key.append(BK == BoundKind::DissimilarBelow ? "DissimilarBelow"
                                                : "SimilarAbove");
    Options.store(Opts, Key, *getBound(H, BK));
  };

  for (std::size_t Idx = 0; Idx < HeuristicCount; ++Idx) {
    auto H = static_cast<Heuristic>(Idx);
    SetToggleOpt(H);
    SetBoundOpt(H, BoundKind::DissimilarBelow);
    SetBoundOpt(H, BoundKind::SimilarAbove);
  }

  SmallVector<std::string, 32> Abbreviations;
  for (const auto &Abbreviation : AbbreviationDictionary) {
    SmallString<32> EqualSignJoined;
    EqualSignJoined.append(Abbreviation.first());
    EqualSignJoined.append("=");
    EqualSignJoined.append(Abbreviation.second);

    if (!Abbreviation.second.empty())
      Abbreviations.emplace_back(EqualSignJoined.str());
  }
  Options.store(Opts, "Abbreviations",
                optutils::serializeStringList(std::vector<StringRef>(
                    Abbreviations.begin(), Abbreviations.end())));
}

bool SuspiciousCallArgumentCheck::isHeuristicEnabled(Heuristic H) const {
  return llvm::is_contained(AppliedHeuristics, H);
}

std::optional<int8_t>
SuspiciousCallArgumentCheck::getBound(Heuristic H, BoundKind BK) const {
  auto Idx = static_cast<std::size_t>(H);
  assert(Idx < HeuristicCount);

  if (!Defaults[Idx].hasBounds())
    return std::nullopt;

  switch (BK) {
  case BoundKind::DissimilarBelow:
    return ConfiguredBounds[Idx].first;
  case BoundKind::SimilarAbove:
    return ConfiguredBounds[Idx].second;
  }
  llvm_unreachable("Unhandled Bound kind.");
}

void SuspiciousCallArgumentCheck::registerMatchers(MatchFinder *Finder) {
  // Only match calls with at least 2 arguments.
  Finder->addMatcher(
      functionDecl(forEachDescendant(callExpr(unless(anyOf(argumentCountIs(0),
                                                           argumentCountIs(1))))
                                         .bind("functionCall")))
          .bind("callingFunc"),
      this);
}

void SuspiciousCallArgumentCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedCallExpr =
      Result.Nodes.getNodeAs<CallExpr>("functionCall");
  const auto *Caller = Result.Nodes.getNodeAs<FunctionDecl>("callingFunc");
  assert(MatchedCallExpr && Caller);

  const Decl *CalleeDecl = MatchedCallExpr->getCalleeDecl();
  if (!CalleeDecl)
    return;

  const FunctionDecl *CalleeFuncDecl = CalleeDecl->getAsFunction();
  if (!CalleeFuncDecl)
    return;
  if (CalleeFuncDecl == Caller)
    // Ignore recursive calls.
    return;
  if (isOverloadedUnaryOrBinarySymbolOperator(CalleeFuncDecl))
    return;

  // Get param attributes.
  setParamNamesAndTypes(CalleeFuncDecl);

  if (ParamNames.empty())
    return;

  // Get Arg attributes.
  std::size_t InitialArgIndex = 0;

  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(CalleeFuncDecl)) {
    if (MethodDecl->getParent()->isLambda())
      // Lambda functions' first Arg are the lambda object.
      InitialArgIndex = 1;
    else if (MethodDecl->getOverloadedOperator() == OO_Call)
      // For custom operator()s, the first Arg is the called object.
      InitialArgIndex = 1;
  }

  setArgNamesAndTypes(MatchedCallExpr, InitialArgIndex);

  if (ArgNames.empty())
    return;

  std::size_t ParamCount = ParamNames.size();

  // Check similarity.
  for (std::size_t I = 0; I < ParamCount; ++I) {
    for (std::size_t J = I + 1; J < ParamCount; ++J) {
      // Do not check if param or arg names are short, or not convertible.
      if (!areParamAndArgComparable(I, J, *Result.Context))
        continue;
      if (!areArgsSwapped(I, J))
        continue;

      // Warning at the call itself.
      diag(MatchedCallExpr->getExprLoc(),
           "%ordinal0 argument '%1' (passed to '%2') looks like it might be "
           "swapped with the %ordinal3, '%4' (passed to '%5')")
          << static_cast<unsigned>(I + 1) << ArgNames[I] << ParamNames[I]
          << static_cast<unsigned>(J + 1) << ArgNames[J] << ParamNames[J]
          << MatchedCallExpr->getArg(I)->getSourceRange()
          << MatchedCallExpr->getArg(J)->getSourceRange();

      // Note at the functions declaration.
      SourceLocation IParNameLoc =
          CalleeFuncDecl->getParamDecl(I)->getLocation();
      SourceLocation JParNameLoc =
          CalleeFuncDecl->getParamDecl(J)->getLocation();

      diag(CalleeFuncDecl->getLocation(), "in the call to %0, declared here",
           DiagnosticIDs::Note)
          << CalleeFuncDecl
          << CharSourceRange::getTokenRange(IParNameLoc, IParNameLoc)
          << CharSourceRange::getTokenRange(JParNameLoc, JParNameLoc);
    }
  }
}

void SuspiciousCallArgumentCheck::setParamNamesAndTypes(
    const FunctionDecl *CalleeFuncDecl) {
  // Reset vectors, and fill them with the currently checked function's
  // parameters' data.
  ParamNames.clear();
  ParamTypes.clear();

  for (const ParmVarDecl *Param : CalleeFuncDecl->parameters()) {
    ParamTypes.push_back(Param->getType());

    if (IdentifierInfo *II = Param->getIdentifier())
      ParamNames.push_back(II->getName());
    else
      ParamNames.push_back(StringRef());
  }
}

void SuspiciousCallArgumentCheck::setArgNamesAndTypes(
    const CallExpr *MatchedCallExpr, std::size_t InitialArgIndex) {
  // Reset vectors, and fill them with the currently checked function's
  // arguments' data.
  ArgNames.clear();
  ArgTypes.clear();

  for (std::size_t I = InitialArgIndex, J = MatchedCallExpr->getNumArgs();
       I < J; ++I) {
    assert(ArgTypes.size() == I - InitialArgIndex &&
           ArgNames.size() == ArgTypes.size() &&
           "Every iteration must put an element into the vectors!");

    if (const auto *ArgExpr = dyn_cast<DeclRefExpr>(
            MatchedCallExpr->getArg(I)->IgnoreUnlessSpelledInSource())) {
      if (const auto *Var = dyn_cast<VarDecl>(ArgExpr->getDecl())) {
        ArgTypes.push_back(Var->getType());
        ArgNames.push_back(Var->getName());
        continue;
      }
      if (const auto *FCall = dyn_cast<FunctionDecl>(ArgExpr->getDecl())) {
        if (FCall->getNameInfo().getName().isIdentifier()) {
          ArgTypes.push_back(FCall->getType());
          ArgNames.push_back(FCall->getName());
          continue;
        }
      }
    }

    ArgTypes.push_back(QualType());
    ArgNames.push_back(StringRef());
  }
}

bool SuspiciousCallArgumentCheck::areParamAndArgComparable(
    std::size_t Position1, std::size_t Position2, const ASTContext &Ctx) const {
  if (Position1 >= ArgNames.size() || Position2 >= ArgNames.size())
    return false;

  // Do not report for too short strings.
  if (ArgNames[Position1].size() < MinimumIdentifierNameLength ||
      ArgNames[Position2].size() < MinimumIdentifierNameLength ||
      ParamNames[Position1].size() < MinimumIdentifierNameLength ||
      ParamNames[Position2].size() < MinimumIdentifierNameLength)
    return false;

  if (!areTypesCompatible(ArgTypes[Position1], ParamTypes[Position2], Ctx) ||
      !areTypesCompatible(ArgTypes[Position2], ParamTypes[Position1], Ctx))
    return false;

  return true;
}

bool SuspiciousCallArgumentCheck::areArgsSwapped(std::size_t Position1,
                                                 std::size_t Position2) const {
  for (Heuristic H : AppliedHeuristics) {
    bool A1ToP2Similar = areNamesSimilar(
        ArgNames[Position2], ParamNames[Position1], H, BoundKind::SimilarAbove);
    bool A2ToP1Similar = areNamesSimilar(
        ArgNames[Position1], ParamNames[Position2], H, BoundKind::SimilarAbove);

    bool A1ToP1Dissimilar =
        !areNamesSimilar(ArgNames[Position1], ParamNames[Position1], H,
                         BoundKind::DissimilarBelow);
    bool A2ToP2Dissimilar =
        !areNamesSimilar(ArgNames[Position2], ParamNames[Position2], H,
                         BoundKind::DissimilarBelow);

    if ((A1ToP2Similar || A2ToP1Similar) && A1ToP1Dissimilar &&
        A2ToP2Dissimilar)
      return true;
  }
  return false;
}

bool SuspiciousCallArgumentCheck::areNamesSimilar(StringRef Arg,
                                                  StringRef Param, Heuristic H,
                                                  BoundKind BK) const {
  int8_t Threshold = -1;
  if (std::optional<int8_t> GotBound = getBound(H, BK))
    Threshold = *GotBound;

  switch (H) {
  case Heuristic::Equality:
    return applyEqualityHeuristic(Arg, Param);
  case Heuristic::Abbreviation:
    return applyAbbreviationHeuristic(AbbreviationDictionary, Arg, Param);
  case Heuristic::Prefix:
    return applyPrefixHeuristic(Arg, Param, Threshold);
  case Heuristic::Suffix:
    return applySuffixHeuristic(Arg, Param, Threshold);
  case Heuristic::Substring:
    return applySubstringHeuristic(Arg, Param, Threshold);
  case Heuristic::Levenshtein:
    return applyLevenshteinHeuristic(Arg, Param, Threshold);
  case Heuristic::JaroWinkler:
    return applyJaroWinklerHeuristic(Arg, Param, Threshold);
  case Heuristic::Dice:
    return applyDiceHeuristic(Arg, Param, Threshold);
  }
  llvm_unreachable("Unhandled heuristic kind");
}

} // namespace clang::tidy::readability
