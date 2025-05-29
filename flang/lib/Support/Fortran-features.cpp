//===-- lib/Support/Fortran-features.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/Fortran-features.h"
#include "flang/Common/idioms.h"
#include "flang/Support/Fortran.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::common {

LanguageFeatureControl::LanguageFeatureControl() {
  // These features must be explicitly enabled by command line options.
  disable_.set(LanguageFeature::OldDebugLines);
  disable_.set(LanguageFeature::OpenACC);
  disable_.set(LanguageFeature::OpenMP);
  disable_.set(LanguageFeature::CUDA); // !@cuf
  disable_.set(LanguageFeature::CudaManaged);
  disable_.set(LanguageFeature::CudaUnified);
  disable_.set(LanguageFeature::ImplicitNoneTypeNever);
  disable_.set(LanguageFeature::ImplicitNoneTypeAlways);
  disable_.set(LanguageFeature::ImplicitNoneExternal);
  disable_.set(LanguageFeature::DefaultSave);
  disable_.set(LanguageFeature::SaveMainProgram);
  // These features, if enabled, conflict with valid standard usage,
  // so there are disabled here by default.
  disable_.set(LanguageFeature::BackslashEscapes);
  disable_.set(LanguageFeature::LogicalAbbreviations);
  disable_.set(LanguageFeature::XOROperator);
  disable_.set(LanguageFeature::OldStyleParameter);
  // Possibly an accidental "feature" of nvfortran.
  disable_.set(LanguageFeature::AssumedRankPassedToNonAssumedRank);
  // These warnings are enabled by default, but only because they used
  // to be unconditional.  TODO: prune this list
  warnLanguage_.set(LanguageFeature::ExponentMatchingKindParam);
  warnLanguage_.set(LanguageFeature::RedundantAttribute);
  warnLanguage_.set(LanguageFeature::SubroutineAndFunctionSpecifics);
  warnLanguage_.set(LanguageFeature::EmptySequenceType);
  warnLanguage_.set(LanguageFeature::NonSequenceCrayPointee);
  warnLanguage_.set(LanguageFeature::BranchIntoConstruct);
  warnLanguage_.set(LanguageFeature::BadBranchTarget);
  warnLanguage_.set(LanguageFeature::HollerithPolymorphic);
  warnLanguage_.set(LanguageFeature::ListDirectedSize);
  warnLanguage_.set(LanguageFeature::IgnoreIrrelevantAttributes);
  warnLanguage_.set(LanguageFeature::AmbiguousStructureConstructor);
  warnUsage_.set(UsageWarning::ShortArrayActual);
  warnUsage_.set(UsageWarning::FoldingException);
  warnUsage_.set(UsageWarning::FoldingAvoidsRuntimeCrash);
  warnUsage_.set(UsageWarning::FoldingValueChecks);
  warnUsage_.set(UsageWarning::FoldingFailure);
  warnUsage_.set(UsageWarning::FoldingLimit);
  warnUsage_.set(UsageWarning::Interoperability);
  // CharacterInteroperability warnings about length are off by default
  warnUsage_.set(UsageWarning::Bounds);
  warnUsage_.set(UsageWarning::Preprocessing);
  warnUsage_.set(UsageWarning::Scanning);
  warnUsage_.set(UsageWarning::OpenAccUsage);
  warnUsage_.set(UsageWarning::ProcPointerCompatibility);
  warnUsage_.set(UsageWarning::VoidMold);
  warnUsage_.set(UsageWarning::KnownBadImplicitInterface);
  warnUsage_.set(UsageWarning::EmptyCase);
  warnUsage_.set(UsageWarning::CaseOverflow);
  warnUsage_.set(UsageWarning::CUDAUsage);
  warnUsage_.set(UsageWarning::IgnoreTKRUsage);
  warnUsage_.set(UsageWarning::ExternalInterfaceMismatch);
  warnUsage_.set(UsageWarning::DefinedOperatorArgs);
  warnUsage_.set(UsageWarning::Final);
  warnUsage_.set(UsageWarning::ZeroDoStep);
  warnUsage_.set(UsageWarning::UnusedForallIndex);
  warnUsage_.set(UsageWarning::OpenMPUsage);
  warnUsage_.set(UsageWarning::DataLength);
  warnUsage_.set(UsageWarning::IgnoredDirective);
  warnUsage_.set(UsageWarning::HomonymousSpecific);
  warnUsage_.set(UsageWarning::HomonymousResult);
  warnUsage_.set(UsageWarning::IgnoredIntrinsicFunctionType);
  warnUsage_.set(UsageWarning::PreviousScalarUse);
  warnUsage_.set(UsageWarning::RedeclaredInaccessibleComponent);
  warnUsage_.set(UsageWarning::ImplicitShared);
  warnUsage_.set(UsageWarning::IndexVarRedefinition);
  warnUsage_.set(UsageWarning::IncompatibleImplicitInterfaces);
  warnUsage_.set(UsageWarning::VectorSubscriptFinalization);
  warnUsage_.set(UsageWarning::UndefinedFunctionResult);
  warnUsage_.set(UsageWarning::UselessIomsg);
  warnUsage_.set(UsageWarning::UnsignedLiteralTruncation);
  warnUsage_.set(UsageWarning::NullActualForDefaultIntentAllocatable);
  warnUsage_.set(UsageWarning::UseAssociationIntoSameNameSubprogram);
  warnUsage_.set(UsageWarning::HostAssociatedIntentOutInSpecExpr);
  warnUsage_.set(UsageWarning::NonVolatilePointerToVolatile);
  // New warnings, on by default
  warnLanguage_.set(LanguageFeature::SavedLocalInSpecExpr);
  warnLanguage_.set(LanguageFeature::NullActualForAllocatable);
}

// Split a string with camel case into the individual words.
// Note, the small vector is just an array of a few pointers and lengths
// into the original input string. So all this allocation should be pretty
// cheap.
llvm::SmallVector<llvm::StringRef> splitCamelCase(llvm::StringRef input) {
  using namespace llvm;
  if (input.empty()) {
    return {};
  }
  SmallVector<StringRef> parts{};
  parts.reserve(input.size());
  auto check = [&input](size_t j, function_ref<bool(char)> predicate) {
    return j < input.size() && predicate(input[j]);
  };
  size_t i{0};
  size_t startWord = i;
  for (; i < input.size(); i++) {
    if ((check(i, isUpper) && check(i + 1, isUpper) && check(i + 2, isLower)) ||
        ((check(i, isLower) || check(i, isDigit)) && check(i + 1, isUpper))) {
      parts.push_back(StringRef(input.data() + startWord, i - startWord + 1));
      startWord = i + 1;
    }
  }
  parts.push_back(llvm::StringRef(input.data() + startWord, i - startWord));
  return parts;
}

// Split a string whith hyphens into the individual words.
llvm::SmallVector<llvm::StringRef> splitHyphenated(llvm::StringRef input) {
  auto parts = llvm::SmallVector<llvm::StringRef>{};
  llvm::SplitString(input, parts, "-");
  return parts;
}

// Check if two strings are equal while normalizing case for the
// right word which is assumed to be a single word in camel case.
bool equalLowerCaseWithCamelCaseWord(llvm::StringRef l, llvm::StringRef r) {
  size_t ls = l.size();
  if (ls != r.size())
    return false;
  size_t j{0};
  // Process the upper case characters.
  for (; j < ls; j++) {
    char rc = r[j];
    char rc2l = llvm::toLower(rc);
    if (rc == rc2l) {
      // Past run of Uppers Case;
      break;
    }
    if (l[j] != rc2l)
      return false;
  }
  // Process the lower case characters.
  for (; j < ls; j++) {
    if (l[j] != r[j]) {
      return false;
    }
  }
  return true;
}

// Parse a CLI enum option return the enum index and whether it should be
// enabled (true) or disabled (false).
std::optional<std::pair<bool, int>> parseCLIEnumIndex(
    llvm::StringRef input, std::function<std::optional<int>(Predicate)> find) {
  auto parts = splitHyphenated(input);
  bool negated = false;
  if (parts.size() >= 1 && !parts[0].compare(llvm::StringRef("no", 2))) {
    negated = true;
    // Remove the "no" part
    parts = llvm::SmallVector<llvm::StringRef>(parts.begin() + 1, parts.end());
  }
  size_t chars = 0;
  for (auto p : parts) {
    chars += p.size();
  }
  auto pred = [&](auto s) {
    if (chars != s.size()) {
      return false;
    }
    auto ccParts = splitCamelCase(s);
    auto num_ccParts = ccParts.size();
    if (parts.size() != num_ccParts) {
      return false;
    }
    for (size_t i{0}; i < num_ccParts; i++) {
      if (!equalLowerCaseWithCamelCaseWord(parts[i], ccParts[i])) {
        return false;
      }
    }
    return true;
  };
  auto cast = [negated](int x) { return std::pair{!negated, x}; };
  return fmap<int, std::pair<bool, int>>(find(pred), cast);
}

std::optional<std::pair<bool, LanguageFeature>> parseCLILanguageFeature(
    llvm::StringRef input) {
  return parseCLIEnum<LanguageFeature>(input, FindLanguageFeatureIndex);
}

std::optional<std::pair<bool, UsageWarning>> parseCLIUsageWarning(
    llvm::StringRef input) {
  return parseCLIEnum<UsageWarning>(input, FindUsageWarningIndex);
}

// Take a string from the CLI and apply it to the LanguageFeatureControl.
// Return true if the option was applied recognized.
bool LanguageFeatureControl::applyCLIOption(llvm::StringRef input) {
  if (auto result = parseCLILanguageFeature(input)) {
    EnableWarning(result->second, result->first);
    return true;
  } else if (auto result = parseCLIUsageWarning(input)) {
    EnableWarning(result->second, result->first);
    return true;
  }
  return false;
}

std::vector<const char *> LanguageFeatureControl::GetNames(
    LogicalOperator opr) const {
  std::vector<const char *> result;
  result.push_back(AsFortran(opr));
  if (opr == LogicalOperator::Neqv && IsEnabled(LanguageFeature::XOROperator)) {
    result.push_back(".xor.");
  }
  if (IsEnabled(LanguageFeature::LogicalAbbreviations)) {
    switch (opr) {
      SWITCH_COVERS_ALL_CASES
    case LogicalOperator::And:
      result.push_back(".a.");
      break;
    case LogicalOperator::Or:
      result.push_back(".o.");
      break;
    case LogicalOperator::Not:
      result.push_back(".n.");
      break;
    case LogicalOperator::Neqv:
      if (IsEnabled(LanguageFeature::XOROperator)) {
        result.push_back(".x.");
      }
      break;
    case LogicalOperator::Eqv:
      break;
    }
  }
  return result;
}

std::vector<const char *> LanguageFeatureControl::GetNames(
    RelationalOperator opr) const {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case RelationalOperator::LT:
    return {".lt.", "<"};
  case RelationalOperator::LE:
    return {".le.", "<="};
  case RelationalOperator::EQ:
    return {".eq.", "=="};
  case RelationalOperator::GE:
    return {".ge.", ">="};
  case RelationalOperator::GT:
    return {".gt.", ">"};
  case RelationalOperator::NE:
    if (IsEnabled(LanguageFeature::AlternativeNE)) {
      return {".ne.", "/=", "<>"};
    } else {
      return {".ne.", "/="};
    }
  }
}

template <typename ENUM, std::size_t N>
void ForEachEnum(std::function<void(ENUM)> f) {
  for (size_t j{0}; j < N; ++j) {
    f(static_cast<ENUM>(j));
  }
}

void LanguageFeatureControl::WarnOnAllNonstandard(bool yes) {
  warnAllLanguage_ = yes;
  disableAllWarnings_ = yes ? false : disableAllWarnings_;
  // should be equivalent to: reset().flip() set ...
  ForEachEnum<LanguageFeature, LanguageFeature_enumSize>(
      [&](LanguageFeature f) { warnLanguage_.set(f, yes); });
  if (yes) {
    // These three features do not need to be warned about,
    // but we do want their feature flags.
    warnLanguage_.set(LanguageFeature::OpenMP, false);
    warnLanguage_.set(LanguageFeature::OpenACC, false);
    warnLanguage_.set(LanguageFeature::CUDA, false);
  }
}

void LanguageFeatureControl::WarnOnAllUsage(bool yes) {
  warnAllUsage_ = yes;
  disableAllWarnings_ = yes ? false : disableAllWarnings_;
  ForEachEnum<UsageWarning, UsageWarning_enumSize>(
      [&](UsageWarning w) { warnUsage_.set(w, yes); });
}
} // namespace Fortran::common
