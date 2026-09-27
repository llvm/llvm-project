//===--- FlangTidyCheck.cpp - flang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FlangTidyCheck.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/YAMLParser.h"
#include <optional>

namespace Fortran::tidy {

FlangTidyCheck::FlangTidyCheck(llvm::StringRef CheckName,
                               FlangTidyContext *Context)
    : Options(CheckName, Context->getOptions().CheckOptions, Context),
      name_(CheckName), context_(Context),
      warningsAsErrors_(Context->isWarningsAsErrorsEnabled(CheckName)) {}

FlangTidyCheck::OptionsView::OptionsView(
    llvm::StringRef CheckName, const FlangTidyOptions::OptionMap &CheckOptions,
    FlangTidyContext *Context)
    : NamePrefix((CheckName + ".").str()), CheckOptions(CheckOptions),
      Context(Context) {}

std::optional<llvm::StringRef>
FlangTidyCheck::OptionsView::get(llvm::StringRef LocalName) const {
  const auto &Iter = CheckOptions.find((NamePrefix + LocalName).str());
  if (Iter != CheckOptions.end())
    return llvm::StringRef(Iter->getValue().Value);
  return std::nullopt;
}

static const llvm::StringSet<> DeprecatedGlobalOptions{
    "StrictMode",
    "IgnoreMacros",
};

static FlangTidyOptions::OptionMap::const_iterator
findPriorityOption(const FlangTidyOptions::OptionMap &Options,
                   llvm::StringRef NamePrefix, llvm::StringRef LocalName,
                   FlangTidyContext *Context) {
  auto IterLocal = Options.find((NamePrefix + LocalName).str());
  auto IterGlobal = Options.find(LocalName);

  // FIXME: temporary solution for deprecation warnings, should be removed
  // after migration. Warn configuration deps on deprecation global options.
  if (IterLocal == Options.end() && IterGlobal != Options.end() &&
      DeprecatedGlobalOptions.contains(LocalName)) {
    // Could add deprecation warning here if needed
  }

  if (IterLocal == Options.end())
    return IterGlobal;
  if (IterGlobal == Options.end())
    return IterLocal;
  if (IterLocal->getValue().Priority >= IterGlobal->getValue().Priority)
    return IterLocal;
  return IterGlobal;
}

std::optional<llvm::StringRef>
FlangTidyCheck::OptionsView::getLocalOrGlobal(llvm::StringRef LocalName) const {
  auto Iter = findPriorityOption(CheckOptions, NamePrefix, LocalName, Context);
  if (Iter != CheckOptions.end())
    return llvm::StringRef(Iter->getValue().Value);
  return std::nullopt;
}

static std::optional<bool> getAsBool(llvm::StringRef Value,
                                     const llvm::Twine &LookupName) {
  if (std::optional<bool> Parsed = llvm::yaml::parseBool(Value))
    return Parsed;
  // To maintain backwards compatability, we support parsing numbers as
  // booleans, even though its not supported in YAML.
  long long Number = 0;
  if (!Value.getAsInteger(10, Number))
    return Number != 0;
  return std::nullopt;
}

template <>
std::optional<bool>
FlangTidyCheck::OptionsView::get<bool>(llvm::StringRef LocalName) const {
  if (std::optional<llvm::StringRef> ValueOr = get(LocalName)) {
    if (auto Result = getAsBool(*ValueOr, NamePrefix + LocalName))
      return Result;
    diagnoseBadBooleanOption(NamePrefix + LocalName, *ValueOr);
  }
  return std::nullopt;
}

template <>
std::optional<bool> FlangTidyCheck::OptionsView::getLocalOrGlobal<bool>(
    llvm::StringRef LocalName) const {
  auto Iter = findPriorityOption(CheckOptions, NamePrefix, LocalName, Context);
  if (Iter != CheckOptions.end()) {
    if (auto Result = getAsBool(Iter->getValue().Value, Iter->getKey()))
      return Result;
    diagnoseBadBooleanOption(Iter->getKey(), Iter->getValue().Value);
  }
  return std::nullopt;
}

void FlangTidyCheck::OptionsView::store(FlangTidyOptions::OptionMap &Options,
                                        llvm::StringRef LocalName,
                                        llvm::StringRef Value) const {
  Options[(NamePrefix + LocalName).str()] =
      FlangTidyOptions::FlangTidyValue(Value);
}

void FlangTidyCheck::OptionsView::storeInt(FlangTidyOptions::OptionMap &Options,
                                           llvm::StringRef LocalName,
                                           int64_t Value) const {
  store(Options, LocalName, llvm::itostr(Value));
}

void FlangTidyCheck::OptionsView::storeUnsigned(
    FlangTidyOptions::OptionMap &Options, llvm::StringRef LocalName,
    uint64_t Value) const {
  store(Options, LocalName, llvm::utostr(Value));
}

template <>
void FlangTidyCheck::OptionsView::store<bool>(
    FlangTidyOptions::OptionMap &Options, llvm::StringRef LocalName,
    bool Value) const {
  store(Options, LocalName,
        Value ? llvm::StringRef("true") : llvm::StringRef("false"));
}

std::optional<int64_t>
FlangTidyCheck::OptionsView::getEnumInt(llvm::StringRef LocalName,
                                        llvm::ArrayRef<NameAndValue> Mapping,
                                        bool CheckGlobal) const {
  auto Iter = CheckGlobal ? findPriorityOption(CheckOptions, NamePrefix,
                                               LocalName, Context)
                          : CheckOptions.find((NamePrefix + LocalName).str());
  if (Iter == CheckOptions.end())
    return std::nullopt;

  llvm::StringRef Value = Iter->getValue().Value;
  llvm::StringRef Closest;
  unsigned EditDistance = 3;
  for (const auto &NameAndEnum : Mapping) {
    if (Value == NameAndEnum.second) {
      return NameAndEnum.first;
    }
    if (Value.equals_insensitive(NameAndEnum.second)) {
      Closest = NameAndEnum.second;
      EditDistance = 0;
      continue;
    }
    unsigned Distance =
        Value.edit_distance(NameAndEnum.second, true, EditDistance);
    if (Distance < EditDistance) {
      EditDistance = Distance;
      Closest = NameAndEnum.second;
    }
  }
  if (EditDistance < 3)
    diagnoseBadEnumOption(Iter->getKey(), Iter->getValue().Value, Closest);
  else
    diagnoseBadEnumOption(Iter->getKey(), Iter->getValue().Value);
  return std::nullopt;
}

/*
static constexpr llvm::StringLiteral ConfigWarning(
    "invalid configuration value '%0' for option '%1'%select{|; expected a "
    "bool|; expected an integer|; did you mean '%3'?}2");
 */
void FlangTidyCheck::OptionsView::diagnoseBadBooleanOption(
    const llvm::Twine &Lookup, llvm::StringRef Unparsed) const {
  llvm::SmallString<64> Buffer;
  // For now, just use a simple error message since we don't have the diagnostic
  // infrastructure
  llvm::errs() << "Error: invalid boolean value '" << Unparsed
               << "' for option '" << Lookup.toStringRef(Buffer) << "'\n";
}

void FlangTidyCheck::OptionsView::diagnoseBadIntegerOption(
    const llvm::Twine &Lookup, llvm::StringRef Unparsed) const {
  llvm::SmallString<64> Buffer;
  llvm::errs() << "Error: invalid integer value '" << Unparsed
               << "' for option '" << Lookup.toStringRef(Buffer) << "'\n";
}

void FlangTidyCheck::OptionsView::diagnoseBadEnumOption(
    const llvm::Twine &Lookup, llvm::StringRef Unparsed,
    llvm::StringRef Suggestion) const {
  llvm::SmallString<64> Buffer;
  llvm::errs() << "Error: invalid enum value '" << Unparsed << "' for option '"
               << Lookup.toStringRef(Buffer) << "'";
  if (!Suggestion.empty())
    llvm::errs() << "; did you mean '" << Suggestion << "'?";
  llvm::errs() << "\n";
}

llvm::StringRef
FlangTidyCheck::OptionsView::get(llvm::StringRef LocalName,
                                 llvm::StringRef Default) const {
  return get(LocalName).value_or(Default);
}

llvm::StringRef
FlangTidyCheck::OptionsView::getLocalOrGlobal(llvm::StringRef LocalName,
                                              llvm::StringRef Default) const {
  return getLocalOrGlobal(LocalName).value_or(Default);
}

} // namespace Fortran::tidy
