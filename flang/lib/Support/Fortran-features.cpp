//===-- lib/Support/Fortran-features.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/Fortran-features.h"
#include "flang/Common/idioms.h"
#include "flang/Common/optional.h"
#include "flang/Support/Fortran.h"
#include "llvm/ADT/StringExtras.h"

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

// Namespace for helper functions for parsing CLI options
// used instead of static so that there can be unit tests for these
// functions.
namespace FortranFeaturesHelpers {

// Ignore case and any inserted punctuation (like '-'/'_')
static std::optional<char> GetWarningChar(char ch) {
  if (ch >= 'a' && ch <= 'z') {
    return ch;
  } else if (ch >= 'A' && ch <= 'Z') {
    return ch - 'A' + 'a';
  } else if (ch >= '0' && ch <= '9') {
    return ch;
  } else {
    return std::nullopt;
  }
}

// Check for case and punctuation insensitive string equality.
// NB, b is probably not null terminated, so don't treat is like a C string.
static bool InsensitiveWarningNameMatch(
    std::string_view a, std::string_view b) {
  size_t j{0}, aSize{a.size()}, k{0}, bSize{b.size()};
  while (true) {
    optional<char> ach{nullopt};
    while (!ach && j < aSize) {
      ach = GetWarningChar(a[j++]);
    }
    optional<char> bch{};
    while (!bch && k < bSize) {
      bch = GetWarningChar(b[k++]);
    }
    if (!ach && !bch) {
      return true;
    } else if (!ach || !bch || *ach != *bch) {
      return false;
    }
    ach = bch = nullopt;
  }
}

// Check if lower case hyphenated words are equal to camel case words.
// Because of out use case we know that 'r' the camel case string is
// well formed in the sense that it is a sequence [a-zA-Z]+[a-zA-Z0-9]*.
// This is checked in the enum-class.h file.
static bool SensitiveWarningNameMatch(llvm::StringRef l, llvm::StringRef r) {
  size_t ls{l.size()}, rs{r.size()};
  if (ls < rs) {
    return false;
  }
  bool atStartOfWord{true};
  size_t wordCount{0}, j{0}; // j is the number of word characters checked in r.
  for (; j < rs; j++) {
    if (wordCount + j >= ls) {
      // `l` was shorter once the hiphens were removed.
      // If r is null terminated, then we are good.
      return r[j] == '\0';
    }
    if (atStartOfWord) {
      if (llvm::isUpper(r[j])) {
        // Upper Case Run
        if (l[wordCount + j] != llvm::toLower(r[j])) {
          return false;
        }
      } else {
        atStartOfWord = false;
        if (l[wordCount + j] != r[j]) {
          return false;
        }
      }
    } else {
      if (llvm::isUpper(r[j])) {
        atStartOfWord = true;
        if (l[wordCount + j] != '-') {
          return false;
        }
        ++wordCount;
        if (l[wordCount + j] != llvm::toLower(r[j])) {
          return false;
        }
      } else if (l[wordCount + j] != r[j]) {
        return false;
      }
    }
  }
  // If there are more characters in l after processing all the characters in r.
  // then fail unless the string is null terminated.
  if (ls > wordCount + j) {
    return l[wordCount + j] == '\0';
  }
  return true;
}

// Parse a CLI enum option return the enum index and whether it should be
// enabled (true) or disabled (false).
template <typename T>
optional<std::pair<bool, T>> ParseCLIEnum(llvm::StringRef input,
    EnumClass::FindIndexType findIndex, bool insensitive) {
  bool negated{false};
  EnumClass::Predicate predicate;
  if (insensitive) {
    if (input.starts_with_insensitive("no")) {
      negated = true;
      input = input.drop_front(2);
    }
    predicate = [input](std::string_view r) {
      return InsensitiveWarningNameMatch(input, r);
    };
  } else {
    if (input.starts_with("no-")) {
      negated = true;
      input = input.drop_front(3);
    }
    predicate = [input](std::string_view r) {
      return SensitiveWarningNameMatch(input, r);
    };
  }
  optional<T> x = EnumClass::Find<T>(predicate, findIndex);
  return MapOption<T, std::pair<bool, T>>(
      x, [negated](T x) { return std::pair{!negated, x}; });
}

optional<std::pair<bool, UsageWarning>> parseCLIUsageWarning(
    llvm::StringRef input, bool insensitive) {
  return ParseCLIEnum<UsageWarning>(input, FindUsageWarningIndex, insensitive);
}

optional<std::pair<bool, LanguageFeature>> parseCLILanguageFeature(
    llvm::StringRef input, bool insensitive) {
  return ParseCLIEnum<LanguageFeature>(
      input, FindLanguageFeatureIndex, insensitive);
}

} // namespace FortranFeaturesHelpers

// Take a string from the CLI and apply it to the LanguageFeatureControl.
// Return true if the option was applied recognized.
bool LanguageFeatureControl::applyCLIOption(
    std::string_view input, bool insensitive) {
  llvm::StringRef inputRef{input};
  if (auto result = FortranFeaturesHelpers::parseCLILanguageFeature(
          inputRef, insensitive)) {
    EnableWarning(result->second, result->first);
    return true;
  } else if (auto result = FortranFeaturesHelpers::parseCLIUsageWarning(
                 inputRef, insensitive)) {
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
