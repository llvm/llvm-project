//===-- lib/Semantics/resolve-names-utils.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_RESOLVE_NAMES_UTILS_H_
#define FORTRAN_SEMANTICS_RESOLVE_NAMES_UTILS_H_

// Utility functions and class for use in resolve-names.cpp.

#include "flang/Parser/message.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/type.h"
#include <forward_list>

namespace Fortran::parser {
class CharBlock;
struct ArraySpec;
struct CoarraySpec;
struct ComponentArraySpec;
struct DataRef;
struct DefinedOpName;
struct Designator;
struct Expr;
struct GenericSpec;
struct Name;
}

namespace Fortran::semantics {

using SourceName = parser::CharBlock;
class SemanticsContext;

// Record that a Name has been resolved to a Symbol
Symbol &Resolve(const parser::Name &, Symbol &);
Symbol *Resolve(const parser::Name &, Symbol *);

// Create a copy of msg with a new isFatal value.
parser::MessageFixedText WithIsFatal(
    const parser::MessageFixedText &msg, bool isFatal);

// Is this the name of a defined operator, e.g. ".foo."
bool IsDefinedOperator(const SourceName &);
bool IsIntrinsicOperator(const SemanticsContext &, const SourceName &);
bool IsLogicalConstant(const SemanticsContext &, const SourceName &);

// Analyze a generic-spec and generate a symbol name and GenericKind for it.
class GenericSpecInfo {
public:
  GenericSpecInfo(const parser::DefinedOpName &x) { Analyze(x); }
  GenericSpecInfo(const parser::GenericSpec &x) { Analyze(x); }

  GenericKind kind() const { return kind_; }
  const SourceName &symbolName() const { return symbolName_.value(); }
  // Some intrinsic operators have more than one name (e.g. `operator(.eq.)` and
  // `operator(==)`). GetAllNames() returns them all, including symbolName.
  std::forward_list<std::string> GetAllNames(SemanticsContext &) const;
  // Set the GenericKind in this symbol and resolve the corresponding
  // name if there is one
  void Resolve(Symbol *) const;
  Symbol *FindInScope(SemanticsContext &, const Scope &) const;

private:
  GenericKind kind_;
  const parser::Name *parseName_{nullptr};
  std::optional<SourceName> symbolName_;

  void Analyze(const parser::DefinedOpName &);
  void Analyze(const parser::GenericSpec &);
};

// Analyze a parser::ArraySpec or parser::CoarraySpec
ArraySpec AnalyzeArraySpec(SemanticsContext &, const parser::ArraySpec &);
ArraySpec AnalyzeArraySpec(
    SemanticsContext &, const parser::ComponentArraySpec &);
ArraySpec AnalyzeCoarraySpec(
    SemanticsContext &context, const parser::CoarraySpec &);

// Perform consistency checks on equivalence sets
class EquivalenceSets {
public:
  EquivalenceSets(SemanticsContext &context) : context_{context} {}
  std::vector<EquivalenceSet> &sets() { return sets_; };
  // Resolve this designator and add to the current equivalence set
  void AddToSet(const parser::Designator &);
  // Finish the current equivalence set: determine if it overlaps
  // with any of the others and perform necessary merges if it does.
  void FinishSet(const parser::CharBlock &);

private:
  bool CheckCanEquivalence(
      const parser::CharBlock &, const Symbol &, const Symbol &);
  void MergeInto(const parser::CharBlock &, EquivalenceSet &, std::size_t);
  const EquivalenceObject *Find(const EquivalenceSet &, const Symbol &);
  bool CheckDesignator(const parser::Designator &);
  bool CheckDataRef(const parser::CharBlock &, const parser::DataRef &);
  bool CheckObject(const parser::Name &);
  bool CheckArrayBound(const parser::Expr &);
  bool CheckSubstringBound(const parser::Expr &, bool);
  bool IsCharacterSequenceType(const DeclTypeSpec *);
  bool IsDefaultKindNumericType(const IntrinsicTypeSpec &);
  bool IsNumericSequenceType(const DeclTypeSpec *);
  bool IsSequenceType(
      const DeclTypeSpec *, std::function<bool(const IntrinsicTypeSpec &)>);

  SemanticsContext &context_;
  std::vector<EquivalenceSet> sets_;  // all equivalence sets in this scope
  // Map object to index of set it is in
  std::map<EquivalenceObject, std::size_t> objectToSet_;
  EquivalenceSet currSet_;  // equivalence set currently being constructed
  struct {
    Symbol *symbol{nullptr};
    std::vector<ConstantSubscript> subscripts;
    std::optional<ConstantSubscript> substringStart;
  } currObject_;  // equivalence object currently being constructed
};

}
#endif  // FORTRAN_SEMANTICS_RESOLVE_NAMES_H_
