//===-- include/flang/Semantics/semantics.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_SEMANTICS_H_
#define FORTRAN_SEMANTICS_SEMANTICS_H_

#include "scope.h"
#include "symbol.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Parser/message.h"
#include <iosfwd>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::common {
class IntrinsicTypeDefaultKinds;
}

namespace Fortran::parser {
struct Name;
struct Program;
class AllCookedSources;
struct AssociateConstruct;
struct BlockConstruct;
struct CaseConstruct;
struct DoConstruct;
struct ChangeTeamConstruct;
struct CriticalConstruct;
struct ForallConstruct;
struct IfConstruct;
struct SelectRankConstruct;
struct SelectTypeConstruct;
struct Variable;
struct WhereConstruct;
} // namespace Fortran::parser

namespace Fortran::semantics {

class Symbol;

using ConstructNode = std::variant<const parser::AssociateConstruct *,
    const parser::BlockConstruct *, const parser::CaseConstruct *,
    const parser::ChangeTeamConstruct *, const parser::CriticalConstruct *,
    const parser::DoConstruct *, const parser::ForallConstruct *,
    const parser::IfConstruct *, const parser::SelectRankConstruct *,
    const parser::SelectTypeConstruct *, const parser::WhereConstruct *>;
using ConstructStack = std::vector<ConstructNode>;

class SemanticsContext {
public:
  SemanticsContext(const common::IntrinsicTypeDefaultKinds &,
      const common::LanguageFeatureControl &, parser::AllCookedSources &);
  ~SemanticsContext();

  const common::IntrinsicTypeDefaultKinds &defaultKinds() const {
    return defaultKinds_;
  }
  const common::LanguageFeatureControl &languageFeatures() const {
    return languageFeatures_;
  };
  int GetDefaultKind(TypeCategory) const;
  int doublePrecisionKind() const {
    return defaultKinds_.doublePrecisionKind();
  }
  int quadPrecisionKind() const { return defaultKinds_.quadPrecisionKind(); }
  bool IsEnabled(common::LanguageFeature) const;
  bool ShouldWarn(common::LanguageFeature) const;
  const std::optional<parser::CharBlock> &location() const { return location_; }
  const std::vector<std::string> &searchDirectories() const {
    return searchDirectories_;
  }
  const std::string &moduleDirectory() const { return moduleDirectory_; }
  const std::string &moduleFileSuffix() const { return moduleFileSuffix_; }
  bool warnOnNonstandardUsage() const { return warnOnNonstandardUsage_; }
  bool warningsAreErrors() const { return warningsAreErrors_; }
  bool debugModuleWriter() const { return debugModuleWriter_; }
  const evaluate::IntrinsicProcTable &intrinsics() const { return intrinsics_; }
  Scope &globalScope() { return globalScope_; }
  parser::Messages &messages() { return messages_; }
  evaluate::FoldingContext &foldingContext() { return foldingContext_; }
  parser::AllCookedSources &allCookedSources() { return allCookedSources_; }

  SemanticsContext &set_location(
      const std::optional<parser::CharBlock> &location) {
    location_ = location;
    return *this;
  }
  SemanticsContext &set_searchDirectories(const std::vector<std::string> &x) {
    searchDirectories_ = x;
    return *this;
  }
  SemanticsContext &set_moduleDirectory(const std::string &x) {
    moduleDirectory_ = x;
    return *this;
  }
  SemanticsContext &set_moduleFileSuffix(const std::string &x) {
    moduleFileSuffix_ = x;
    return *this;
  }
  SemanticsContext &set_warnOnNonstandardUsage(bool x) {
    warnOnNonstandardUsage_ = x;
    return *this;
  }
  SemanticsContext &set_warningsAreErrors(bool x) {
    warningsAreErrors_ = x;
    return *this;
  }

  SemanticsContext &set_debugModuleWriter(bool x) {
    debugModuleWriter_ = x;
    return *this;
  }

  const DeclTypeSpec &MakeNumericType(TypeCategory, int kind = 0);
  const DeclTypeSpec &MakeLogicalType(int kind = 0);

  bool AnyFatalError() const;

  // Test or set the Error flag on a Symbol
  bool HasError(const Symbol &);
  bool HasError(const Symbol *);
  bool HasError(const parser::Name &);
  void SetError(const Symbol &, bool = true);

  template <typename... A> parser::Message &Say(A &&...args) {
    CHECK(location_);
    return messages_.Say(*location_, std::forward<A>(args)...);
  }
  template <typename... A>
  parser::Message &Say(parser::CharBlock at, A &&...args) {
    return messages_.Say(at, std::forward<A>(args)...);
  }
  parser::Message &Say(parser::Message &&msg) {
    return messages_.Say(std::move(msg));
  }
  template <typename... A>
  void SayWithDecl(const Symbol &symbol, const parser::CharBlock &at,
      parser::MessageFixedText &&msg, A &&...args) {
    auto &message{Say(at, std::move(msg), args...)};
    evaluate::AttachDeclaration(&message, symbol);
  }

  const Scope &FindScope(parser::CharBlock) const;
  Scope &FindScope(parser::CharBlock);

  const ConstructStack &constructStack() const { return constructStack_; }
  template <typename N> void PushConstruct(const N &node) {
    constructStack_.emplace_back(&node);
  }
  void PopConstruct();

  ENUM_CLASS(IndexVarKind, DO, FORALL)
  // Check to see if a variable being redefined is a DO or FORALL index.
  // If so, emit a message.
  void WarnIndexVarRedefine(const parser::CharBlock &, const Symbol &);
  void CheckIndexVarRedefine(const parser::CharBlock &, const Symbol &);
  void CheckIndexVarRedefine(const parser::Variable &);
  void CheckIndexVarRedefine(const parser::Name &);
  void ActivateIndexVar(const parser::Name &, IndexVarKind);
  void DeactivateIndexVar(const parser::Name &);
  SymbolVector GetIndexVars(IndexVarKind);
  SourceName SaveTempName(std::string &&);
  SourceName GetTempName(const Scope &);

private:
  void CheckIndexVarRedefine(
      const parser::CharBlock &, const Symbol &, parser::MessageFixedText &&);
  void CheckError(const Symbol &);

  const common::IntrinsicTypeDefaultKinds &defaultKinds_;
  const common::LanguageFeatureControl languageFeatures_;
  parser::AllCookedSources &allCookedSources_;
  std::optional<parser::CharBlock> location_;
  std::vector<std::string> searchDirectories_;
  std::string moduleDirectory_{"."s};
  std::string moduleFileSuffix_{".mod"};
  bool warnOnNonstandardUsage_{false};
  bool warningsAreErrors_{false};
  bool debugModuleWriter_{false};
  const evaluate::IntrinsicProcTable intrinsics_;
  Scope globalScope_;
  parser::Messages messages_;
  evaluate::FoldingContext foldingContext_;
  ConstructStack constructStack_;
  struct IndexVarInfo {
    parser::CharBlock location;
    IndexVarKind kind;
  };
  std::map<SymbolRef, const IndexVarInfo, SymbolAddressCompare>
      activeIndexVars_;
  UnorderedSymbolSet errorSymbols_;
  std::set<std::string> tempNames_;
};

class Semantics {
public:
  explicit Semantics(SemanticsContext &context, parser::Program &program,
      parser::CharBlock charBlock, bool debugModuleWriter = false)
      : context_{context}, program_{program} {
    context.set_debugModuleWriter(debugModuleWriter);
    context.globalScope().AddSourceRange(charBlock);
  }

  SemanticsContext &context() const { return context_; }
  bool Perform();
  const Scope &FindScope(const parser::CharBlock &where) const {
    return context_.FindScope(where);
  }
  bool AnyFatalError() const { return context_.AnyFatalError(); }
  void EmitMessages(llvm::raw_ostream &) const;
  void DumpSymbols(llvm::raw_ostream &);
  void DumpSymbolsSources(llvm::raw_ostream &) const;

private:
  SemanticsContext &context_;
  parser::Program &program_;
};

// Base class for semantics checkers.
struct BaseChecker {
  template <typename N> void Enter(const N &) {}
  template <typename N> void Leave(const N &) {}
};
} // namespace Fortran::semantics
#endif
