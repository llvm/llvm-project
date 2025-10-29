// EntryPointStats.h - Tracking statistics per entry point ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_ENTRYPOINTSTATS_H
#define CLANG_INCLUDE_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_ENTRYPOINTSTATS_H

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace clang {
class Decl;

namespace ento {

class EntryPointStat {
public:
  llvm::StringLiteral name() const { return Name; }

  static void lockRegistry(llvm::StringRef CPPFileName);

  static void takeSnapshot(const Decl *EntryPoint);
  static void dumpStatsAsCSV(llvm::raw_ostream &OS);
  static void dumpStatsAsCSV(llvm::StringRef FileName);

protected:
  explicit EntryPointStat(llvm::StringLiteral Name) : Name{Name} {}
  EntryPointStat(const EntryPointStat &) = delete;
  EntryPointStat(EntryPointStat &&) = delete;
  EntryPointStat &operator=(EntryPointStat &) = delete;
  EntryPointStat &operator=(EntryPointStat &&) = delete;

private:
  llvm::StringLiteral Name;
};

class BoolEPStat : public EntryPointStat {
  std::optional<bool> Value = {};

public:
  explicit BoolEPStat(llvm::StringLiteral Name);
  unsigned value() const { return Value && *Value; }
  void set(bool V) {
    assert(!Value.has_value());
    Value = V;
  }
  void reset() { Value = {}; }
};

// used by CounterEntryPointTranslationUnitStat
class CounterEPStat : public EntryPointStat {
  using EntryPointStat::EntryPointStat;
  unsigned Value = {};

public:
  explicit CounterEPStat(llvm::StringLiteral Name);
  unsigned value() const { return Value; }
  void reset() { Value = {}; }
  CounterEPStat &operator++() {
    ++Value;
    return *this;
  }

  CounterEPStat &operator++(int) {
    // No difference as you can't extract the value
    return ++(*this);
  }

  CounterEPStat &operator+=(unsigned Inc) {
    Value += Inc;
    return *this;
  }
};

// used by UnsignedMaxEtryPointTranslationUnitStatistic
class UnsignedMaxEPStat : public EntryPointStat {
  using EntryPointStat::EntryPointStat;
  unsigned Value = {};

public:
  explicit UnsignedMaxEPStat(llvm::StringLiteral Name);
  unsigned value() const { return Value; }
  void reset() { Value = {}; }
  void updateMax(unsigned X) { Value = std::max(Value, X); }
};

class UnsignedEPStat : public EntryPointStat {
  using EntryPointStat::EntryPointStat;
  std::optional<unsigned> Value = {};

public:
  explicit UnsignedEPStat(llvm::StringLiteral Name);
  unsigned value() const { return Value.value_or(0); }
  void reset() { Value.reset(); }
  void set(unsigned V) {
    assert(!Value.has_value());
    Value = V;
  }
};

class CounterEntryPointTranslationUnitStat {
  CounterEPStat M;
  llvm::TrackingStatistic S;

public:
  CounterEntryPointTranslationUnitStat(const char *DebugType,
                                       llvm::StringLiteral Name,
                                       llvm::StringLiteral Desc)
      : M(Name), S(DebugType, Name.data(), Desc.data()) {}
  CounterEntryPointTranslationUnitStat &operator++() {
    ++M;
    ++S;
    return *this;
  }

  CounterEntryPointTranslationUnitStat &operator++(int) {
    // No difference with prefix as the value is not observable.
    return ++(*this);
  }

  CounterEntryPointTranslationUnitStat &operator+=(unsigned Inc) {
    M += Inc;
    S += Inc;
    return *this;
  }
};

class UnsignedMaxEntryPointTranslationUnitStatistic {
  UnsignedMaxEPStat M;
  llvm::TrackingStatistic S;

public:
  UnsignedMaxEntryPointTranslationUnitStatistic(const char *DebugType,
                                                llvm::StringLiteral Name,
                                                llvm::StringLiteral Desc)
      : M(Name), S(DebugType, Name.data(), Desc.data()) {}
  void updateMax(uint64_t Value) {
    M.updateMax(static_cast<unsigned>(Value));
    S.updateMax(Value);
  }
};

#define STAT_COUNTER(VARNAME, DESC)                                            \
  static clang::ento::CounterEntryPointTranslationUnitStat VARNAME = {         \
      DEBUG_TYPE, #VARNAME, DESC}

#define STAT_MAX(VARNAME, DESC)                                                \
  static clang::ento::UnsignedMaxEntryPointTranslationUnitStatistic VARNAME =  \
      {DEBUG_TYPE, #VARNAME, DESC}

} // namespace ento
} // namespace clang

#endif // CLANG_INCLUDE_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_ENTRYPOINTSTATS_H
