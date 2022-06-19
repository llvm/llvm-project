//===-- lib/Semantics/semantics.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/semantics.h"
#include "assignment.h"
#include "canonicalize-acc.h"
#include "canonicalize-do.h"
#include "canonicalize-omp.h"
#include "check-acc-structure.h"
#include "check-allocate.h"
#include "check-arithmeticif.h"
#include "check-case.h"
#include "check-coarray.h"
#include "check-data.h"
#include "check-deallocate.h"
#include "check-declarations.h"
#include "check-do-forall.h"
#include "check-if-stmt.h"
#include "check-io.h"
#include "check-namelist.h"
#include "check-nullify.h"
#include "check-omp-structure.h"
#include "check-purity.h"
#include "check-return.h"
#include "check-select-rank.h"
#include "check-select-type.h"
#include "check-stop.h"
#include "compute-offsets.h"
#include "mod-file.h"
#include "resolve-labels.h"
#include "resolve-names.h"
#include "rewrite-parse-tree.h"
#include "flang/Common/default-kinds.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::semantics {

using NameToSymbolMap = std::multimap<parser::CharBlock, SymbolRef>;
static void DoDumpSymbols(llvm::raw_ostream &, const Scope &, int indent = 0);
static void PutIndent(llvm::raw_ostream &, int indent);

static void GetSymbolNames(const Scope &scope, NameToSymbolMap &symbols) {
  // Finds all symbol names in the scope without collecting duplicates.
  for (const auto &pair : scope) {
    symbols.emplace(pair.second->name(), *pair.second);
  }
  for (const auto &pair : scope.commonBlocks()) {
    symbols.emplace(pair.second->name(), *pair.second);
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
template <typename... C> class SemanticsVisitor : public virtual C... {
public:
  using C::Enter...;
  using C::Leave...;
  using BaseChecker::Enter;
  using BaseChecker::Leave;
  SemanticsVisitor(SemanticsContext &context)
      : C{context}..., context_{context} {}

  template <typename N> bool Pre(const N &node) {
    if constexpr (common::HasMember<const N *, ConstructNode>) {
      context_.PushConstruct(node);
    }
    Enter(node);
    return true;
  }
  template <typename N> void Post(const N &node) {
    Leave(node);
    if constexpr (common::HasMember<const N *, ConstructNode>) {
      context_.PopConstruct();
    }
  }

  template <typename T> bool Pre(const parser::Statement<T> &node) {
    context_.set_location(node.source);
    Enter(node);
    return true;
  }
  template <typename T> bool Pre(const parser::UnlabeledStatement<T> &node) {
    context_.set_location(node.source);
    Enter(node);
    return true;
  }
  template <typename T> void Post(const parser::Statement<T> &node) {
    Leave(node);
    context_.set_location(std::nullopt);
  }
  template <typename T> void Post(const parser::UnlabeledStatement<T> &node) {
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

class MiscChecker : public virtual BaseChecker {
public:
  explicit MiscChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::EntryStmt &) {
    if (!context_.constructStack().empty()) { // C1571
      context_.Say("ENTRY may not appear in an executable construct"_err_en_US);
    }
  }
  void Leave(const parser::AssignStmt &stmt) {
    CheckAssignGotoName(std::get<parser::Name>(stmt.t));
  }
  void Leave(const parser::AssignedGotoStmt &stmt) {
    CheckAssignGotoName(std::get<parser::Name>(stmt.t));
  }

private:
  void CheckAssignGotoName(const parser::Name &name) {
    if (context_.HasError(name.symbol)) {
      return;
    }
    const Symbol &symbol{DEREF(name.symbol)};
    auto type{evaluate::DynamicType::From(symbol)};
    if (!IsVariableName(symbol) || symbol.Rank() != 0 || !type ||
        type->category() != TypeCategory::Integer ||
        type->kind() !=
            context_.defaultKinds().GetDefaultKind(TypeCategory::Integer)) {
      context_
          .Say(name.source,
              "'%s' must be a default integer scalar variable"_err_en_US,
              name.source)
          .Attach(symbol.name(), "Declaration of '%s'"_en_US, symbol.name());
    }
  }

  SemanticsContext &context_;
};

using StatementSemanticsPass1 = ExprChecker;
using StatementSemanticsPass2 = SemanticsVisitor<AccStructureChecker,
    AllocateChecker, ArithmeticIfStmtChecker, AssignmentChecker, CaseChecker,
    CoarrayChecker, DataChecker, DeallocateChecker, DoForallChecker,
    IfStmtChecker, IoChecker, MiscChecker, NamelistChecker, NullifyChecker,
    OmpStructureChecker, PurityChecker, ReturnStmtChecker,
    SelectRankConstructChecker, SelectTypeChecker, StopChecker>;

static bool PerformStatementSemantics(
    SemanticsContext &context, parser::Program &program) {
  ResolveNames(context, program, context.globalScope());
  RewriteParseTree(context, program);
  ComputeOffsets(context, context.globalScope());
  CheckDeclarations(context);
  StatementSemanticsPass1{context}.Walk(program);
  StatementSemanticsPass2 pass2{context};
  pass2.Walk(program);
  if (!context.AnyFatalError()) {
    pass2.CompileDataInitializationsIntoInitializers();
  }
  return !context.AnyFatalError();
}

/// This class keeps track of the common block appearances with the biggest size
/// and with an initial value (if any) in a program. This allows reporting
/// conflicting initialization and warning about appearances of a same
/// named common block with different sizes. The biggest common block size and
/// initialization (if any) can later be provided so that lowering can generate
/// the correct symbol size and initial values, even when named common blocks
/// appears with different sizes and are initialized outside of block data.
class CommonBlockMap {
private:
  struct CommonBlockInfo {
    // Common block symbol for the appearance with the biggest size.
    SymbolRef biggestSize;
    // Common block symbol for the appearance with the initialized members (if
    // any).
    std::optional<SymbolRef> initialization;
  };

public:
  void MapCommonBlockAndCheckConflicts(
      SemanticsContext &context, const Symbol &common) {
    const Symbol *isInitialized{CommonBlockIsInitialized(common)};
    auto [it, firstAppearance] = commonBlocks_.insert({common.name(),
        isInitialized ? CommonBlockInfo{common, common}
                      : CommonBlockInfo{common, std::nullopt}});
    if (!firstAppearance) {
      CommonBlockInfo &info{it->second};
      if (isInitialized) {
        if (info.initialization.has_value() &&
            &**info.initialization != &common) {
          // Use the location of the initialization in the error message because
          // common block symbols may have no location if they are blank
          // commons.
          const Symbol &previousInit{
              DEREF(CommonBlockIsInitialized(**info.initialization))};
          context
              .Say(isInitialized->name(),
                  "Multiple initialization of COMMON block /%s/"_err_en_US,
                  common.name())
              .Attach(previousInit.name(),
                  "Previous initialization of COMMON block /%s/"_en_US,
                  common.name());
        } else {
          info.initialization = common;
        }
      }
      if (common.size() != info.biggestSize->size() && !common.name().empty()) {
        context
            .Say(common.name(),
                "A named COMMON block should have the same size everywhere it appears (%zd bytes here)"_port_en_US,
                common.size())
            .Attach(info.biggestSize->name(),
                "Previously defined with a size of %zd bytes"_en_US,
                info.biggestSize->size());
      }
      if (common.size() > info.biggestSize->size()) {
        info.biggestSize = common;
      }
    }
  }

  CommonBlockList GetCommonBlocks() const {
    CommonBlockList result;
    for (const auto &[_, blockInfo] : commonBlocks_) {
      result.emplace_back(
          std::make_pair(blockInfo.initialization ? *blockInfo.initialization
                                                  : blockInfo.biggestSize,
              blockInfo.biggestSize->size()));
    }
    return result;
  }

private:
  /// Return the symbol of an initialized member if a COMMON block
  /// is initalized. Otherwise, return nullptr.
  static Symbol *CommonBlockIsInitialized(const Symbol &common) {
    const auto &commonDetails =
        common.get<Fortran::semantics::CommonBlockDetails>();

    for (const auto &member : commonDetails.objects()) {
      if (IsInitialized(*member)) {
        return &*member;
      }
    }

    // Common block may be initialized via initialized variables that are in an
    // equivalence with the common block members.
    for (const Fortran::semantics::EquivalenceSet &set :
        common.owner().equivalenceSets()) {
      for (const Fortran::semantics::EquivalenceObject &obj : set) {
        if (!obj.symbol.test(
                Fortran::semantics::Symbol::Flag::CompilerCreated)) {
          if (FindCommonBlockContaining(obj.symbol) == &common &&
              IsInitialized(obj.symbol)) {
            return &obj.symbol;
          }
        }
      }
    }
    return nullptr;
  }
  std::map<SourceName, CommonBlockInfo> commonBlocks_;
};

SemanticsContext::SemanticsContext(
    const common::IntrinsicTypeDefaultKinds &defaultKinds,
    const common::LanguageFeatureControl &languageFeatures,
    parser::AllCookedSources &allCookedSources)
    : defaultKinds_{defaultKinds}, languageFeatures_{languageFeatures},
      allCookedSources_{allCookedSources},
      intrinsics_{evaluate::IntrinsicProcTable::Configure(defaultKinds_)},
      globalScope_{*this}, intrinsicModulesScope_{globalScope_.MakeScope(
                               Scope::Kind::IntrinsicModules, nullptr)},
      foldingContext_{
          parser::ContextualMessages{&messages_}, defaultKinds_, intrinsics_} {}

SemanticsContext::~SemanticsContext() {}

int SemanticsContext::GetDefaultKind(TypeCategory category) const {
  return defaultKinds_.GetDefaultKind(category);
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
  return errorSymbols_.count(symbol) > 0;
}
bool SemanticsContext::HasError(const Symbol *symbol) {
  return !symbol || HasError(*symbol);
}
bool SemanticsContext::HasError(const parser::Name &name) {
  return HasError(name.symbol);
}
void SemanticsContext::SetError(const Symbol &symbol, bool value) {
  if (value) {
    CheckError(symbol);
    errorSymbols_.emplace(symbol);
  }
}
void SemanticsContext::CheckError(const Symbol &symbol) {
  if (!AnyFatalError()) {
    std::string buf;
    llvm::raw_string_ostream ss{buf};
    ss << symbol;
    common::die(
        "No error was reported but setting error on: %s", ss.str().c_str());
  }
}

const Scope &SemanticsContext::FindScope(parser::CharBlock source) const {
  return const_cast<SemanticsContext *>(this)->FindScope(source);
}

Scope &SemanticsContext::FindScope(parser::CharBlock source) {
  if (auto *scope{globalScope_.FindScope(source)}) {
    return *scope;
  } else {
    common::die(
        "SemanticsContext::FindScope(): invalid source location for '%s'",
        source.ToString().c_str());
  }
}

bool SemanticsContext::IsInModuleFile(parser::CharBlock source) const {
  for (const Scope *scope{&FindScope(source)}; !scope->IsGlobal();
       scope = &scope->parent()) {
    if (scope->IsModuleFile()) {
      return true;
    }
  }
  return false;
}

void SemanticsContext::PopConstruct() {
  CHECK(!constructStack_.empty());
  constructStack_.pop_back();
}

void SemanticsContext::CheckIndexVarRedefine(const parser::CharBlock &location,
    const Symbol &variable, parser::MessageFixedText &&message) {
  const Symbol &symbol{ResolveAssociations(variable)};
  auto it{activeIndexVars_.find(symbol)};
  if (it != activeIndexVars_.end()) {
    std::string kind{EnumToString(it->second.kind)};
    Say(location, std::move(message), kind, symbol.name())
        .Attach(it->second.location, "Enclosing %s construct"_en_US, kind);
  }
}

void SemanticsContext::WarnIndexVarRedefine(
    const parser::CharBlock &location, const Symbol &variable) {
  CheckIndexVarRedefine(location, variable,
      "Possible redefinition of %s variable '%s'"_warn_en_US);
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
    activeIndexVars_.emplace(
        ResolveAssociations(*indexVar), IndexVarInfo{name.source, kind});
  }
}

void SemanticsContext::DeactivateIndexVar(const parser::Name &name) {
  if (Symbol * indexVar{name.symbol}) {
    auto it{activeIndexVars_.find(ResolveAssociations(*indexVar))};
    if (it != activeIndexVars_.end() && it->second.location == name.source) {
      activeIndexVars_.erase(it);
    }
  }
}

SymbolVector SemanticsContext::GetIndexVars(IndexVarKind kind) {
  SymbolVector result;
  for (const auto &[symbol, info] : activeIndexVars_) {
    if (info.kind == kind) {
      result.push_back(symbol);
    }
  }
  return result;
}

SourceName SemanticsContext::SaveTempName(std::string &&name) {
  return {*tempNames_.emplace(std::move(name)).first};
}

SourceName SemanticsContext::GetTempName(const Scope &scope) {
  for (const auto &str : tempNames_) {
    if (IsTempName(str)) {
      SourceName name{str};
      if (scope.find(name) == scope.end()) {
        return name;
      }
    }
  }
  return SaveTempName(".F18."s + std::to_string(tempNames_.size()));
}

bool SemanticsContext::IsTempName(const std::string &name) {
  return name.size() > 5 && name.substr(0, 5) == ".F18.";
}

Scope *SemanticsContext::GetBuiltinModule(const char *name) {
  return ModFileReader{*this}.Read(SourceName{name, std::strlen(name)},
      true /*intrinsic*/, nullptr, true /*silence errors*/);
}

void SemanticsContext::UseFortranBuiltinsModule() {
  if (builtinsScope_ == nullptr) {
    builtinsScope_ = GetBuiltinModule("__fortran_builtins");
    if (builtinsScope_) {
      intrinsics_.SupplyBuiltins(*builtinsScope_);
    }
  }
}

parser::Program &SemanticsContext::SaveParseTree(parser::Program &&tree) {
  return modFileParseTrees_.emplace_back(std::move(tree));
}

bool Semantics::Perform() {
  // Implicitly USE the __Fortran_builtins module so that special types
  // (e.g., __builtin_team_type) are available to semantics, esp. for
  // intrinsic checking.
  if (!program_.v.empty()) {
    const auto *frontModule{std::get_if<common::Indirection<parser::Module>>(
        &program_.v.front().u)};
    if (frontModule &&
        std::get<parser::Statement<parser::ModuleStmt>>(frontModule->value().t)
                .statement.v.source == "__fortran_builtins") {
      // Don't try to read the builtins module when we're actually building it.
    } else {
      context_.UseFortranBuiltinsModule();
    }
  }
  return ValidateLabels(context_, program_) &&
      parser::CanonicalizeDo(program_) && // force line break
      CanonicalizeAcc(context_.messages(), program_) &&
      CanonicalizeOmp(context_.messages(), program_) &&
      PerformStatementSemantics(context_, program_) &&
      ModFileWriter{context_}.WriteAll();
}

void Semantics::EmitMessages(llvm::raw_ostream &os) const {
  context_.messages().Emit(os, context_.allCookedSources());
}

void Semantics::DumpSymbols(llvm::raw_ostream &os) {
  DoDumpSymbols(os, context_.globalScope());
}

void Semantics::DumpSymbolsSources(llvm::raw_ostream &os) const {
  NameToSymbolMap symbols;
  GetSymbolNames(context_.globalScope(), symbols);
  const parser::AllCookedSources &allCooked{context_.allCookedSources()};
  for (const auto &pair : symbols) {
    const Symbol &symbol{pair.second};
    if (auto sourceInfo{allCooked.GetSourcePositionRange(symbol.name())}) {
      os << symbol.name().ToString() << ": " << sourceInfo->first.file.path()
         << ", " << sourceInfo->first.line << ", " << sourceInfo->first.column
         << "-" << sourceInfo->second.column << "\n";
    } else if (symbol.has<semantics::UseDetails>()) {
      os << symbol.name().ToString() << ": "
         << symbol.GetUltimate().owner().symbol()->name().ToString() << "\n";
    }
  }
}

void DoDumpSymbols(llvm::raw_ostream &os, const Scope &scope, int indent) {
  PutIndent(os, indent);
  os << Scope::EnumToString(scope.kind()) << " scope:";
  if (const auto *symbol{scope.symbol()}) {
    os << ' ' << symbol->name();
  }
  if (scope.alignment().has_value()) {
    os << " size=" << scope.size() << " alignment=" << *scope.alignment();
  }
  if (scope.derivedTypeSpec()) {
    os << " instantiation of " << *scope.derivedTypeSpec();
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

static void PutIndent(llvm::raw_ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";
  }
}

void SemanticsContext::MapCommonBlockAndCheckConflicts(const Symbol &common) {
  if (!commonBlockMap_) {
    commonBlockMap_ = std::make_unique<CommonBlockMap>();
  }
  commonBlockMap_->MapCommonBlockAndCheckConflicts(*this, common);
}

CommonBlockList SemanticsContext::GetCommonBlocks() const {
  if (commonBlockMap_) {
    return commonBlockMap_->GetCommonBlocks();
  }
  return {};
}

} // namespace Fortran::semantics
