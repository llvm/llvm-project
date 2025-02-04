//===-- lib/Common/Fortran-features.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran-features.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"

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
  // New warnings, on by default
  warnLanguage_.set(LanguageFeature::SavedLocalInSpecExpr);
}

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

static bool WarningNameMatch(const char *a, const char *b) {
  while (true) {
    auto ach{GetWarningChar(*a)};
    while (!ach && *a) {
      ach = GetWarningChar(*++a);
    }
    auto bch{GetWarningChar(*b)};
    while (!bch && *b) {
      bch = GetWarningChar(*++b);
    }
    if (!ach && !bch) {
      return true;
    } else if (!ach || !bch || *ach != *bch) {
      return false;
    }
    ++a, ++b;
  }
}

template <typename ENUM, std::size_t N>
std::optional<ENUM> ScanEnum(const char *name) {
  if (name) {
    for (std::size_t j{0}; j < N; ++j) {
      auto feature{static_cast<ENUM>(j)};
      if (WarningNameMatch(name, EnumToString(feature).data())) {
        return feature;
      }
    }
  }
  return std::nullopt;
}

std::optional<LanguageFeature> FindLanguageFeature(const char *name) {
  return ScanEnum<LanguageFeature, LanguageFeature_enumSize>(name);
}

std::optional<UsageWarning> FindUsageWarning(const char *name) {
  return ScanEnum<UsageWarning, UsageWarning_enumSize>(name);
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

} // namespace Fortran::common
