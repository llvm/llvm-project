//===-- lib/semantics/semantics.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/semantics/semantics.h"
#include "assignment.h"
#include "canonicalize-do.h"
#include "canonicalize-omp.h"
#include "check-allocate.h"
#include "check-arithmeticif.h"
#include "check-coarray.h"
#include "check-deallocate.h"
#include "check-declarations.h"
#include "check-do-forall.h"
#include "check-if-stmt.h"
#include "check-io.h"
#include "check-nullify.h"
#include "check-omp-structure.h"
#include "check-purity.h"
#include "check-return.h"
#include "check-stop.h"
#include "mod-file.h"
#include "resolve-labels.h"
#include "resolve-names.h"
#include "rewrite-parse-tree.h"
#include "flang/common/default-kinds.h"
#include "flang/parser/parse-tree-visitor.h"
#include "flang/parser/tools.h"
#include "flang/semantics/expression.h"
#include "flang/semantics/scope.h"
#include "flang/semantics/symbol.h"

namespace Fortran::semantics {

using NameToSymbolMap = std::map<const char *, SymbolRef>;
static void DoDumpSymbols(std::ostream &, const Scope &, int indent = 0);
static void PutIndent(std::ostream &, int indent);

static void GetSymbolNames(const Scope &scope, NameToSymbolMap &symbols) {
  // Finds all symbol names in the scope without collecting duplicates.
  for (const auto &pair : scope) {
    symbols.emplace(pair.second->name().begin(), *pair.second);
  }
  for (const auto &pair : scope.commonBlocks()) {
    symbols.emplace(pair.second->name().begin(), *pair.second);
  }
  for (const auto &child : scope.children()) {
    GetSymbolNames(child, symbols);
  }
}

// A parse tree visitor that calls Enter/Leave functions from each checker
// class C supplied as template parameters. Enter is called before the node's
// children are visited, Leave is called after. No two checkers may have the
// same Enter or Leave function. Each checker must be constructible from
// SemanticsContext and have BaseChecker as a virtual base class.
template<typename... C> class SemanticsVisitor : public virtual C... {
public:
  using C::Enter...;
  using C::Leave...;
  using BaseChecker::Enter;
  using BaseChecker::Leave;
  SemanticsVisitor(SemanticsContext &context)
    : C{context}..., context_{context} {}

  template<typename N> bool Pre(const N &node) {
    if constexpr (common::HasMember<const N *, ConstructNode>) {
      context_.PushConstruct(node);
    }
    Enter(node);
    return true;
  }
  template<typename N> void Post(const N &node) {
    Leave(node);
    if constexpr (common::HasMember<const N *, ConstructNode>) {
      context_.PopConstruct();
    }
  }

  template<typename T> bool Pre(const parser::Statement<T> &node) {
    context_.set_location(node.source);
    Enter(node);
    return true;
  }
  template<typename T> bool Pre(const parser::UnlabeledStatement<T> &node) {
    context_.set_location(node.source);
    Enter(node);
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &node) {
    Leave(node);
    context_.set_location(std::nullopt);
  }
  template<typename T> void Post(const parser::UnlabeledStatement<T> &node) {
    Leave(node);
    context_.set_location(std::nullopt);
  }

  bool Walk(const parser::Program &program) {
    parser::Walk(program, *this);
    return !context_.AnyFatalError();
  }

private:
  SemanticsContext &context_;
};

using StatementSemanticsPass1 = ExprChecker;
using StatementSemanticsPass2 = SemanticsVisitor<  //
    AllocateChecker, ArithmeticIfStmtChecker, AssignmentChecker, CoarrayChecker,
    DeallocateChecker, DoForallChecker, IfStmtChecker, IoChecker,
    NullifyChecker, OmpStructureChecker, PurityChecker, ReturnStmtChecker,
    StopChecker>;

static bool PerformStatementSemantics(
    SemanticsContext &context, parser::Program &program) {
  ResolveNames(context, program);
  RewriteParseTree(context, program);
  CheckDeclarations(context);
  StatementSemanticsPass1{context}.Walk(program);
  return StatementSemanticsPass2{context}.Walk(program);
}

SemanticsContext::SemanticsContext(
    const common::IntrinsicTypeDefaultKinds &defaultKinds,
    const common::LanguageFeatureControl &languageFeatures,
    parser::AllSources &allSources)
  : defaultKinds_{defaultKinds}, languageFeatures_{languageFeatures},
    allSources_{allSources},
    intrinsics_{evaluate::IntrinsicProcTable::Configure(defaultKinds_)},
    foldingContext_{
        parser::ContextualMessages{&messages_}, defaultKinds_, intrinsics_} {}

SemanticsContext::~SemanticsContext() {}

int SemanticsContext::GetDefaultKind(TypeCategory category) const {
  return defaultKinds_.GetDefaultKind(category);
}

bool SemanticsContext::IsEnabled(common::LanguageFeature feature) const {
  return languageFeatures_.IsEnabled(feature);
}

bool SemanticsContext::ShouldWarn(common::LanguageFeature feature) const {
  return languageFeatures_.ShouldWarn(feature);
}

const DeclTypeSpec &SemanticsContext::MakeNumericType(
    TypeCategory category, int kind) {
  if (kind == 0) {
    kind = GetDefaultKind(category);
  }
  return globalScope_.MakeNumericType(category, KindExpr{kind});
}
const DeclTypeSpec &SemanticsContext::MakeLogicalType(int kind) {
  if (kind == 0) {
    kind = GetDefaultKind(TypeCategory::Logical);
  }
  return globalScope_.MakeLogicalType(KindExpr{kind});
}

bool SemanticsContext::AnyFatalError() const {
  return !messages_.empty() &&
      (warningsAreErrors_ || messages_.AnyFatalError());
}
bool SemanticsContext::HasError(const Symbol &symbol) {
  return CheckError(symbol.test(Symbol::Flag::Error));
}
bool SemanticsContext::HasError(const Symbol *symbol) {
  return CheckError(!symbol || HasError(*symbol));
}
bool SemanticsContext::HasError(const parser::Name &name) {
  return HasError(name.symbol);
}
void SemanticsContext::SetError(Symbol &symbol, bool value) {
  if (value) {
    CHECK(AnyFatalError());
    symbol.set(Symbol::Flag::Error);
  }
}
bool SemanticsContext::CheckError(bool error) {
  CHECK(!error || AnyFatalError());
  return error;
}

const Scope &SemanticsContext::FindScope(parser::CharBlock source) const {
  return const_cast<SemanticsContext *>(this)->FindScope(source);
}

Scope &SemanticsContext::FindScope(parser::CharBlock source) {
  if (auto *scope{globalScope_.FindScope(source)}) {
    return *scope;
  } else {
    common::die("SemanticsContext::FindScope(): invalid source location");
  }
}

void SemanticsContext::PopConstruct() {
  CHECK(!constructStack_.empty());
  constructStack_.pop_back();
}

void SemanticsContext::CheckIndexVarRedefine(const parser::CharBlock &location,
    const Symbol &variable, parser::MessageFixedText &&message) {
  if (const Symbol * root{GetAssociationRoot(variable)}) {
    auto it{activeIndexVars_.find(*root)};
    if (it != activeIndexVars_.end()) {
      std::string kind{EnumToString(it->second.kind)};
      Say(location, std::move(message), kind, root->name())
          .Attach(it->second.location, "Enclosing %s construct"_en_US, kind);
    }
  }
}

void SemanticsContext::WarnIndexVarRedefine(
    const parser::CharBlock &location, const Symbol &variable) {
  CheckIndexVarRedefine(
      location, variable, "Possible redefinition of %s variable '%s'"_en_US);
}

void SemanticsContext::CheckIndexVarRedefine(
    const parser::CharBlock &location, const Symbol &variable) {
  CheckIndexVarRedefine(
      location, variable, "Cannot redefine %s variable '%s'"_err_en_US);
}

void SemanticsContext::CheckIndexVarRedefine(const parser::Variable &variable) {
  if (const Symbol * entity{GetLastName(variable).symbol}) {
    CheckIndexVarRedefine(variable.GetSource(), *entity);
  }
}

void SemanticsContext::CheckIndexVarRedefine(const parser::Name &name) {
  if (const Symbol * entity{name.symbol}) {
    CheckIndexVarRedefine(name.source, *entity);
  }
}

void SemanticsContext::ActivateIndexVar(
    const parser::Name &name, IndexVarKind kind) {
  CheckIndexVarRedefine(name);
  if (const Symbol * indexVar{name.symbol}) {
    if (const Symbol * root{GetAssociationRoot(*indexVar)}) {
      activeIndexVars_.emplace(*root, IndexVarInfo{name.source, kind});
    }
  }
}

void SemanticsContext::DeactivateIndexVar(const parser::Name &name) {
  if (Symbol * indexVar{name.symbol}) {
    if (const Symbol * root{GetAssociationRoot(*indexVar)}) {
      auto it{activeIndexVars_.find(*root)};
      if (it != activeIndexVars_.end() && it->second.location == name.source) {
        activeIndexVars_.erase(it);
      }
    }
  }
}

bool Semantics::Perform() {
  return ValidateLabels(context_, program_) &&
      parser::CanonicalizeDo(program_) &&  // force line break
      CanonicalizeOmp(context_.messages(), program_) &&
      PerformStatementSemantics(context_, program_) &&
      ModFileWriter{context_}.WriteAll();
}

void Semantics::EmitMessages(std::ostream &os) const {
  context_.messages().Emit(os, cooked_);
}

void Semantics::DumpSymbols(std::ostream &os) {
  DoDumpSymbols(os, context_.globalScope());
}

void Semantics::DumpSymbolsSources(std::ostream &os) const {
  NameToSymbolMap symbols;
  GetSymbolNames(context_.globalScope(), symbols);
  for (const auto &pair : symbols) {
    const Symbol &symbol{pair.second};
    if (auto sourceInfo{cooked_.GetSourcePositionRange(symbol.name())}) {
      os << symbol.name().ToString() << ": " << sourceInfo->first.file.path()
         << ", " << sourceInfo->first.line << ", " << sourceInfo->first.column
         << "-" << sourceInfo->second.column << "\n";
    } else if (symbol.has<semantics::UseDetails>()) {
      os << symbol.name().ToString() << ": "
         << symbol.GetUltimate().owner().symbol()->name().ToString() << "\n";
    }
  }
}

void DoDumpSymbols(std::ostream &os, const Scope &scope, int indent) {
  PutIndent(os, indent);
  os << Scope::EnumToString(scope.kind()) << " scope:";
  if (const auto *symbol{scope.symbol()}) {
    os << ' ' << symbol->name();
  }
  os << '\n';
  ++indent;
  for (const auto &pair : scope) {
    const auto &symbol{*pair.second};
    PutIndent(os, indent);
    os << symbol << '\n';
    if (const auto *details{symbol.detailsIf<GenericDetails>()}) {
      if (const auto &type{details->derivedType()}) {
        PutIndent(os, indent);
        os << *type << '\n';
      }
    }
  }
  if (!scope.equivalenceSets().empty()) {
    PutIndent(os, indent);
    os << "Equivalence Sets:";
    for (const auto &set : scope.equivalenceSets()) {
      os << ' ';
      char sep = '(';
      for (const auto &object : set) {
        os << sep << object.AsFortran();
        sep = ',';
      }
      os << ')';
    }
    os << '\n';
  }
  if (!scope.crayPointers().empty()) {
    PutIndent(os, indent);
    os << "Cray Pointers:";
    for (const auto &[pointee, pointer] : scope.crayPointers()) {
      os << " (" << pointer->name() << ',' << pointee << ')';
    }
  }
  for (const auto &pair : scope.commonBlocks()) {
    const auto &symbol{*pair.second};
    PutIndent(os, indent);
    os << symbol << '\n';
  }
  for (const auto &child : scope.children()) {
    DoDumpSymbols(os, child, indent);
  }
  --indent;
}

static void PutIndent(std::ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";
  }
}
}
