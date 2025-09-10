//===- LifetimeSafety.h - C++ Lifetime Safety Analysis -*----------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the entry point for a dataflow-based static analysis
// that checks for C++ lifetime violations.
//
// The analysis is based on the concepts of "origins" and "loans" to track
// pointer lifetimes and detect issues like use-after-free and dangling
// pointers. See the RFC for more details:
// https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/StringMap.h"
#include <memory>

namespace clang::lifetimes {

/// Enum to track the confidence level of a potential error.
enum class Confidence {
  None,
  Maybe,   // Reported as a potential error (-Wlifetime-safety-strict)
  Definite // Reported as a definite error (-Wlifetime-safety-permissive)
};

class LifetimeSafetyReporter {
public:
  LifetimeSafetyReporter() = default;
  virtual ~LifetimeSafetyReporter() = default;

  virtual void reportUseAfterFree(const Expr *IssueExpr, const Expr *UseExpr,
                                  SourceLocation FreeLoc,
                                  Confidence Confidence) {}
};

/// The main entry point for the analysis.
void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetyReporter *Reporter);

namespace internal {
// Forward declarations of internal types.
class Fact;
class FactManager;
class LoanPropagationAnalysis;
class ExpiredLoansAnalysis;
struct LifetimeFactory;

/// A generic, type-safe wrapper for an ID, distinguished by its `Tag` type.
/// Used for giving ID to loans and origins.
template <typename Tag> struct ID {
  uint32_t Value = 0;

  bool operator==(const ID<Tag> &Other) const { return Value == Other.Value; }
  bool operator!=(const ID<Tag> &Other) const { return !(*this == Other); }
  bool operator<(const ID<Tag> &Other) const { return Value < Other.Value; }
  ID<Tag> operator++(int) {
    ID<Tag> Tmp = *this;
    ++Value;
    return Tmp;
  }
  void Profile(llvm::FoldingSetNodeID &IDBuilder) const {
    IDBuilder.AddInteger(Value);
  }
};

template <typename Tag>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, ID<Tag> ID) {
  return OS << ID.Value;
}

using LoanID = ID<struct LoanTag>;
using OriginID = ID<struct OriginTag>;

// Using LLVM's immutable collections is efficient for dataflow analysis
// as it avoids deep copies during state transitions.
// TODO(opt): Consider using a bitset to represent the set of loans.
using LoanSet = llvm::ImmutableSet<LoanID>;
using OriginSet = llvm::ImmutableSet<OriginID>;

/// A `ProgramPoint` identifies a location in the CFG by pointing to a specific
/// `Fact`. identified by a lifetime-related event (`Fact`).
///
/// A `ProgramPoint` has "after" semantics: it represents the location
/// immediately after its corresponding `Fact`.
using ProgramPoint = const Fact *;

/// Running the lifetime safety analysis and querying its results. It
/// encapsulates the various dataflow analyses.
class LifetimeSafetyAnalysis {
public:
  LifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                         LifetimeSafetyReporter *Reporter);
  ~LifetimeSafetyAnalysis();

  void run();

  /// Returns the set of loans an origin holds at a specific program point.
  LoanSet getLoansAtPoint(OriginID OID, ProgramPoint PP) const;

  /// Returns the set of loans that have expired at a specific program point.
  std::vector<LoanID> getExpiredLoansAtPoint(ProgramPoint PP) const;

  /// Finds the OriginID for a given declaration.
  /// Returns a null optional if not found.
  std::optional<OriginID> getOriginIDForDecl(const ValueDecl *D) const;

  /// Finds the LoanID's for the loan created with the specific variable as
  /// their Path.
  std::vector<LoanID> getLoanIDForVar(const VarDecl *VD) const;

  /// Retrieves program points that were specially marked in the source code
  /// for testing.
  ///
  /// The analysis recognizes special function calls of the form
  /// `void("__lifetime_test_point_<name>")` as test points. This method returns
  /// a map from the annotation string (<name>) to the corresponding
  /// `ProgramPoint`. This allows test harnesses to query the analysis state at
  /// user-defined locations in the code.
  /// \note This is intended for testing only.
  llvm::StringMap<ProgramPoint> getTestPoints() const;

private:
  AnalysisDeclContext &AC;
  LifetimeSafetyReporter *Reporter;
  std::unique_ptr<LifetimeFactory> Factory;
  std::unique_ptr<FactManager> FactMgr;
  std::unique_ptr<LoanPropagationAnalysis> LoanPropagation;
  std::unique_ptr<ExpiredLoansAnalysis> ExpiredLoans;
};
} // namespace internal
} // namespace clang::lifetimes

namespace llvm {
template <typename Tag>
struct DenseMapInfo<clang::lifetimes::internal::ID<Tag>> {
  using ID = clang::lifetimes::internal::ID<Tag>;

  static inline ID getEmptyKey() {
    return {DenseMapInfo<uint32_t>::getEmptyKey()};
  }

  static inline ID getTombstoneKey() {
    return {DenseMapInfo<uint32_t>::getTombstoneKey()};
  }

  static unsigned getHashValue(const ID &Val) {
    return DenseMapInfo<uint32_t>::getHashValue(Val.Value);
  }

  static bool isEqual(const ID &LHS, const ID &RHS) { return LHS == RHS; }
};
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
