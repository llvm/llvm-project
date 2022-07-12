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

#include "clang/Analysis/FlowSensitive/DebugSupport.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace dataflow {

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

private:
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

} // namespace dataflow
} // namespace clang
