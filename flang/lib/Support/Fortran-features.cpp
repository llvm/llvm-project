//===-- lib/Support/Fortran-features.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/Fortran-features.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/characters.h"
#include "flang/Support/Fortran.h"
#include <string>
#include <string_view>

namespace Fortran::common {

static std::vector<std::string_view> SplitCamelCase(std::string_view x) {
  std::vector<std::string_view> result;
  // NB, we start at 1 because the first character is never a word boundary.
  size_t xSize{x.size()}, wordStart{0}, wordEnd{1};
  for (; wordEnd < xSize; ++wordEnd) {
    // Identify when wordEnd is at the start of a new word.
    if ((!parser::IsUpperCaseLetter(x[wordEnd - 1]) &&
            parser::IsUpperCaseLetter(x[wordEnd])) ||
        // ACCUsage => ACC-Usage, CComment => C-Comment, etc.
        (parser::IsUpperCaseLetter(x[wordEnd]) && wordEnd + 1 < xSize &&
            parser::IsLowerCaseLetter(x[wordEnd + 1]))) {
      result.push_back(x.substr(wordStart, wordEnd - wordStart));
      wordStart = wordEnd;
    }
  }
  // We went one past the end of the last word.
  result.push_back(x.substr(wordStart, wordEnd - wordStart));
  return result;
}

// Namespace for helper functions for parsing Cli options used instead of static
// so that there can be unit tests for this function.
namespace details {
std::string CamelCaseToLowerCaseHyphenated(std::string_view x) {
  std::vector<std::string_view> words{SplitCamelCase(x)};
  std::string result{};
  result.reserve(x.size() + words.size() + 1);
  for (size_t i{0}; i < words.size(); ++i) {
    std::string word{parser::ToLowerCaseLetters(words[i])};
    result += i == 0 ? "" : "-";
    result += word;
  }
  return result;
}
} // namespace details

LanguageFeatureControl::LanguageFeatureControl() {
  // Initialize the bidirectional maps with the default spellings.
  cliOptions_.reserve(LanguageFeature_enumSize + UsageWarning_enumSize);
  ForEachLanguageFeature([&](auto feature) {
    std::string_view name{Fortran::common::EnumToString(feature)};
    std::string cliOption{details::CamelCaseToLowerCaseHyphenated(name)};
    cliOptions_.insert({cliOption, {feature}});
    languageFeatureCliCanonicalSpelling_[EnumToInt(feature)] =
        std::move(cliOption);
  });

  ForEachUsageWarning([&](auto warning) {
    std::string_view name{Fortran::common::EnumToString(warning)};
    std::string cliOption{details::CamelCaseToLowerCaseHyphenated(name)};
    cliOptions_.insert({cliOption, {warning}});
    usageWarningCliCanonicalSpelling_[EnumToInt(warning)] =
        std::move(cliOption);
  });

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
  warnLanguage_.set(LanguageFeature::TransferBOZ);
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
  warnUsage_.set(UsageWarning::RealConstantWidening);
  // New warnings, on by default
  warnLanguage_.set(LanguageFeature::SavedLocalInSpecExpr);
  warnLanguage_.set(LanguageFeature::NullActualForAllocatable);
}

std::optional<LanguageControlFlag> LanguageFeatureControl::FindWarning(
    std::string_view input) {
  bool negated{false};
  if (input.size() > 3 && input.substr(0, 3) == "no-") {
    negated = true;
    input = input.substr(3);
  }
  if (auto it{cliOptions_.find(std::string{input})}; it != cliOptions_.end()) {
    return std::make_pair(it->second, !negated);
  }
  return std::nullopt;
}

bool LanguageFeatureControl::EnableWarning(std::string_view input) {
  if (auto warningAndEnabled{FindWarning(input)}) {
    EnableWarning(warningAndEnabled->first, warningAndEnabled->second);
    return true;
  }
  return false;
}

void LanguageFeatureControl::ReplaceCliCanonicalSpelling(
    LanguageFeature f, std::string input) {
  cliOptions_.erase(languageFeatureCliCanonicalSpelling_[EnumToInt(f)]);
  cliOptions_.insert({input, {f}});
  languageFeatureCliCanonicalSpelling_[EnumToInt(f)] = std::move(input);
}

void LanguageFeatureControl::ReplaceCliCanonicalSpelling(
    UsageWarning w, std::string input) {
  cliOptions_.erase(usageWarningCliCanonicalSpelling_[EnumToInt(w)]);
  cliOptions_.insert({input, {w}});
  usageWarningCliCanonicalSpelling_[EnumToInt(w)] = std::move(input);
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

void LanguageFeatureControl::WarnOnAllNonstandard(bool yes) {
  warnAllLanguage_ = yes;
  warnLanguage_.reset();
  if (yes) {
    disableAllWarnings_ = false;
    warnLanguage_.flip();
    // These three features do not need to be warned about,
    // but we do want their feature flags.
    warnLanguage_.set(LanguageFeature::OpenMP, false);
    warnLanguage_.set(LanguageFeature::OpenACC, false);
    warnLanguage_.set(LanguageFeature::CUDA, false);
  }
}

void LanguageFeatureControl::WarnOnAllUsage(bool yes) {
  warnAllUsage_ = yes;
  warnUsage_.reset();
  if (yes) {
    disableAllWarnings_ = false;
    warnUsage_.flip();
  }
}
} // namespace Fortran::common
