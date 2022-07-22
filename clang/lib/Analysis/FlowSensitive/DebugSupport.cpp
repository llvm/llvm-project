//===- DebugSupport.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions which generate more readable forms of data
//  structures used in the dataflow analyses, for debugging purposes.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "clang/Analysis/FlowSensitive/DebugSupport.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatCommon.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace dataflow {

using llvm::AlignStyle;
using llvm::fmt_pad;
using llvm::formatv;

namespace {

class DebugStringGenerator {
public:
  explicit DebugStringGenerator(
      llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNamesArg)
      : Counter(0), AtomNames(std::move(AtomNamesArg)) {
#ifndef NDEBUG
    llvm::StringSet<> Names;
    for (auto &N : AtomNames) {
      assert(Names.insert(N.second).second &&
             "The same name must not assigned to different atoms");
    }
#endif
  }

  /// Returns a string representation of a boolean value `B`.
  std::string debugString(const BoolValue &B, size_t Depth = 0) {
    std::string S;
    switch (B.getKind()) {
    case Value::Kind::AtomicBool: {
      S = getAtomName(&cast<AtomicBoolValue>(B));
      break;
    }
    case Value::Kind::Conjunction: {
      auto &C = cast<ConjunctionValue>(B);
      auto L = debugString(C.getLeftSubValue(), Depth + 1);
      auto R = debugString(C.getRightSubValue(), Depth + 1);
      S = formatv("(and\n{0}\n{1})", L, R);
      break;
    }
    case Value::Kind::Disjunction: {
      auto &D = cast<DisjunctionValue>(B);
      auto L = debugString(D.getLeftSubValue(), Depth + 1);
      auto R = debugString(D.getRightSubValue(), Depth + 1);
      S = formatv("(or\n{0}\n{1})", L, R);
      break;
    }
    case Value::Kind::Negation: {
      auto &N = cast<NegationValue>(B);
      S = formatv("(not\n{0})", debugString(N.getSubVal(), Depth + 1));
      break;
    }
    default:
      llvm_unreachable("Unhandled value kind");
    }
    auto Indent = Depth * 4;
    return formatv("{0}", fmt_pad(S, Indent, 0));
  }

  /// Returns a string representation of a set of boolean `Constraints` and the
  /// `Result` of satisfiability checking on the `Constraints`.
  std::string debugString(ArrayRef<BoolValue *> &Constraints,
                          const Solver::Result &Result) {
    auto Template = R"(
Constraints
------------
{0:$[

]}
------------
{1}.
{2}
)";

    std::vector<std::string> ConstraintsStrings;
    ConstraintsStrings.reserve(Constraints.size());
    for (auto &Constraint : Constraints) {
      ConstraintsStrings.push_back(debugString(*Constraint));
    }

    auto StatusString = debugString(Result.getStatus());
    auto Solution = Result.getSolution();
    auto SolutionString = Solution ? "\n" + debugString(Solution.value()) : "";

    return formatv(
        Template,
        llvm::make_range(ConstraintsStrings.begin(), ConstraintsStrings.end()),
        StatusString, SolutionString);
  }

private:
  /// Returns a string representation of a truth assignment to atom booleans.
  std::string debugString(
      const llvm::DenseMap<AtomicBoolValue *, Solver::Result::Assignment>
          &AtomAssignments) {
    size_t MaxNameLength = 0;
    for (auto &AtomName : AtomNames) {
      MaxNameLength = std::max(MaxNameLength, AtomName.second.size());
    }

    std::vector<std::string> Lines;
    for (auto &AtomAssignment : AtomAssignments) {
      auto Line = formatv("{0} = {1}",
                          fmt_align(getAtomName(AtomAssignment.first),
                                    AlignStyle::Left, MaxNameLength),
                          debugString(AtomAssignment.second));
      Lines.push_back(Line);
    }
    llvm::sort(Lines.begin(), Lines.end());

    return formatv("{0:$[\n]}", llvm::make_range(Lines.begin(), Lines.end()));
  }

  /// Returns a string representation of a boolean assignment to true or false.
  std::string debugString(Solver::Result::Assignment Assignment) {
    switch (Assignment) {
    case Solver::Result::Assignment::AssignedFalse:
      return "False";
    case Solver::Result::Assignment::AssignedTrue:
      return "True";
    }
    llvm_unreachable("Booleans can only be assigned true/false");
  }

  /// Returns a string representation of the result status of a SAT check.
  std::string debugString(Solver::Result::Status Status) {
    switch (Status) {
    case Solver::Result::Status::Satisfiable:
      return "Satisfiable";
    case Solver::Result::Status::Unsatisfiable:
      return "Unsatisfiable";
    case Solver::Result::Status::TimedOut:
      return "TimedOut";
    }
    llvm_unreachable("Unhandled SAT check result status");
  }

  /// Returns the name assigned to `Atom`, either user-specified or created by
  /// default rules (B0, B1, ...).
  std::string getAtomName(const AtomicBoolValue *Atom) {
    auto Entry = AtomNames.try_emplace(Atom, formatv("B{0}", Counter));
    if (Entry.second) {
      Counter++;
    }
    return Entry.first->second;
  }

  // Keep track of number of atoms without a user-specified name, used to assign
  // non-repeating default names to such atoms.
  size_t Counter;

  // Keep track of names assigned to atoms.
  llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNames;
};

} // namespace

std::string
debugString(const BoolValue &B,
            llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNames) {
  return DebugStringGenerator(std::move(AtomNames)).debugString(B);
}

std::string
debugString(ArrayRef<BoolValue *> Constraints, const Solver::Result &Result,
            llvm::DenseMap<const AtomicBoolValue *, std::string> AtomNames) {
  return DebugStringGenerator(std::move(AtomNames))
      .debugString(Constraints, Result);
}

} // namespace dataflow
} // namespace clang
