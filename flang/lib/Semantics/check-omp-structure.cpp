//===-- lib/Semantics/check-omp-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"
#include "definable.h"
#include "resolve-names-utils.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <variant>

namespace Fortran::semantics {

template <typename T, typename U>
static bool operator!=(const evaluate::Expr<T> &e, const evaluate::Expr<U> &f) {
  return !(e == f);
}

// Use when clause falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &) { \
    CheckAllowedClause(llvm::omp::Clause::Y); \
  }

#define CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowedClause(llvm::omp::Clause::Y); \
    RequiresConstantPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

#define CHECK_REQ_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowedClause(llvm::omp::Clause::Y); \
    RequiresPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

// Use when clause don't falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_PARSER_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::X &) { \
    CheckAllowedClause(llvm::omp::Y); \
  }

std::string ThisVersion(unsigned version) {
  std::string tv{
      std::to_string(version / 10) + "." + std::to_string(version % 10)};
  return "OpenMP v" + tv;
}

std::string TryVersion(unsigned version) {
  return "try -fopenmp-version=" + std::to_string(version);
}

static const parser::Designator *GetDesignatorFromObj(
    const parser::OmpObject &object) {
  return std::get_if<parser::Designator>(&object.u);
}

static const parser::DataRef *GetDataRefFromObj(
    const parser::OmpObject &object) {
  if (auto *desg{GetDesignatorFromObj(object)}) {
    return std::get_if<parser::DataRef>(&desg->u);
  }
  return nullptr;
}

static const parser::ArrayElement *GetArrayElementFromObj(
    const parser::OmpObject &object) {
  if (auto *dataRef{GetDataRefFromObj(object)}) {
    using ElementIndirection = common::Indirection<parser::ArrayElement>;
    if (auto *ind{std::get_if<ElementIndirection>(&dataRef->u)}) {
      return &ind->value();
    }
  }
  return nullptr;
}

static bool IsVarOrFunctionRef(const MaybeExpr &expr) {
  if (expr) {
    return evaluate::UnwrapProcedureRef(*expr) != nullptr ||
        evaluate::IsVariable(*expr);
  } else {
    return false;
  }
}

static std::optional<SomeExpr> GetEvaluateExpr(const parser::Expr &parserExpr) {
  const parser::TypedExpr &typedExpr{parserExpr.typedExpr};
  // ForwardOwningPointer           typedExpr
  // `- GenericExprWrapper          ^.get()
  //    `- std::optional<Expr>      ^->v
  return typedExpr.get()->v;
}

static std::optional<evaluate::DynamicType> GetDynamicType(
    const parser::Expr &parserExpr) {
  if (auto maybeExpr{GetEvaluateExpr(parserExpr)}) {
    return maybeExpr->GetType();
  } else {
    return std::nullopt;
  }
}

// 'OmpWorkshareBlockChecker' is used to check the validity of the assignment
// statements and the expressions enclosed in an OpenMP Workshare construct
class OmpWorkshareBlockChecker {
public:
  OmpWorkshareBlockChecker(SemanticsContext &context, parser::CharBlock source)
      : context_{context}, source_{source} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::AssignmentStmt &assignment) {
    const auto &var{std::get<parser::Variable>(assignment.t)};
    const auto &expr{std::get<parser::Expr>(assignment.t)};
    const auto *lhs{GetExpr(context_, var)};
    const auto *rhs{GetExpr(context_, expr)};
    if (lhs && rhs) {
      Tristate isDefined{semantics::IsDefinedAssignment(
          lhs->GetType(), lhs->Rank(), rhs->GetType(), rhs->Rank())};
      if (isDefined == Tristate::Yes) {
        context_.Say(expr.source,
            "Defined assignment statement is not "
            "allowed in a WORKSHARE construct"_err_en_US);
      }
    }
    return true;
  }

  bool Pre(const parser::Expr &expr) {
    if (const auto *e{GetExpr(context_, expr)}) {
      for (const Symbol &symbol : evaluate::CollectSymbols(*e)) {
        const Symbol &root{GetAssociationRoot(symbol)};
        if (IsFunction(root)) {
          std::string attrs{""};
          if (!IsElementalProcedure(root)) {
            attrs = " non-ELEMENTAL";
          }
          if (root.attrs().test(Attr::IMPURE)) {
            if (attrs != "") {
              attrs = "," + attrs;
            }
            attrs = " IMPURE" + attrs;
          }
          if (attrs != "") {
            context_.Say(expr.source,
                "User defined%s function '%s' is not allowed in a "
                "WORKSHARE construct"_err_en_US,
                attrs, root.name());
          }
        }
      }
    }
    return false;
  }

private:
  SemanticsContext &context_;
  parser::CharBlock source_;
};

class AssociatedLoopChecker {
public:
  AssociatedLoopChecker(SemanticsContext &context, std::int64_t level)
      : context_{context}, level_{level} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::DoConstruct &dc) {
    level_--;
    const auto &doStmt{
        std::get<parser::Statement<parser::NonLabelDoStmt>>(dc.t)};
    const auto &constructName{
        std::get<std::optional<parser::Name>>(doStmt.statement.t)};
    if (constructName) {
      constructNamesAndLevels_.emplace(
          constructName.value().ToString(), level_);
    }
    if (level_ >= 0) {
      if (dc.IsDoWhile()) {
        context_.Say(doStmt.source,
            "The associated loop of a loop-associated directive cannot be a DO WHILE."_err_en_US);
      }
      if (!dc.GetLoopControl()) {
        context_.Say(doStmt.source,
            "The associated loop of a loop-associated directive cannot be a DO without control."_err_en_US);
      }
    }
    return true;
  }

  void Post(const parser::DoConstruct &dc) { level_++; }

  bool Pre(const parser::CycleStmt &cyclestmt) {
    std::map<std::string, std::int64_t>::iterator it;
    bool err{false};
    if (cyclestmt.v) {
      it = constructNamesAndLevels_.find(cyclestmt.v->source.ToString());
      err = (it != constructNamesAndLevels_.end() && it->second > 0);
    } else { // If there is no label then use the level of the last enclosing DO
      err = level_ > 0;
    }
    if (err) {
      context_.Say(*source_,
          "CYCLE statement to non-innermost associated loop of an OpenMP DO "
          "construct"_err_en_US);
    }
    return true;
  }

  bool Pre(const parser::ExitStmt &exitStmt) {
    std::map<std::string, std::int64_t>::iterator it;
    bool err{false};
    if (exitStmt.v) {
      it = constructNamesAndLevels_.find(exitStmt.v->source.ToString());
      err = (it != constructNamesAndLevels_.end() && it->second >= 0);
    } else { // If there is no label then use the level of the last enclosing DO
      err = level_ >= 0;
    }
    if (err) {
      context_.Say(*source_,
          "EXIT statement terminates associated loop of an OpenMP DO "
          "construct"_err_en_US);
    }
    return true;
  }

  bool Pre(const parser::Statement<parser::ActionStmt> &actionstmt) {
    source_ = &actionstmt.source;
    return true;
  }

private:
  SemanticsContext &context_;
  const parser::CharBlock *source_;
  std::int64_t level_;
  std::map<std::string, std::int64_t> constructNamesAndLevels_;
};

// `OmpUnitedTaskDesignatorChecker` is used to check if the designator
// can appear within the TASK construct
class OmpUnitedTaskDesignatorChecker {
public:
  OmpUnitedTaskDesignatorChecker(SemanticsContext &context)
      : context_{context} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::Name &name) {
    if (name.symbol->test(Symbol::Flag::OmpThreadprivate)) {
      // OpenMP 5.2: 5.2 threadprivate directive restriction
      context_.Say(name.source,
          "A THREADPRIVATE variable `%s` cannot appear in an UNTIED TASK region"_err_en_US,
          name.source);
    }
    return true;
  }

private:
  SemanticsContext &context_;
};

bool OmpStructureChecker::CheckAllowedClause(llvmOmpClause clause) {
  // Do not do clause checks while processing METADIRECTIVE.
  // Context selectors can contain clauses that are not given as a part
  // of a construct, but as trait properties. Testing whether they are
  // valid or not is deferred to the checks of the context selectors.
  // As it stands now, these clauses would appear as if they were present
  // on METADIRECTIVE, leading to incorrect diagnostics.
  if (GetDirectiveNest(ContextSelectorNest) > 0) {
    return true;
  }

  unsigned version{context_.langOptions().OpenMPVersion};
  DirectiveContext &dirCtx = GetContext();
  llvm::omp::Directive dir{dirCtx.directive};

  if (!llvm::omp::isAllowedClauseForDirective(dir, clause, version)) {
    unsigned allowedInVersion{[&] {
      for (unsigned v : llvm::omp::getOpenMPVersions()) {
        if (v <= version) {
          continue;
        }
        if (llvm::omp::isAllowedClauseForDirective(dir, clause, v)) {
          return v;
        }
      }
      return 0u;
    }()};

    // Only report it if there is a later version that allows it.
    // If it's not allowed at all, it will be reported by CheckAllowed.
    if (allowedInVersion != 0) {
      auto clauseName{parser::ToUpperCaseLetters(getClauseName(clause).str())};
      auto dirName{parser::ToUpperCaseLetters(getDirectiveName(dir).str())};

      context_.Say(dirCtx.clauseSource,
          "%s clause is not allowed on directive %s in %s, %s"_err_en_US,
          clauseName, dirName, ThisVersion(version),
          TryVersion(allowedInVersion));
    }
  }
  return CheckAllowed(clause);
}

bool OmpStructureChecker::IsCommonBlock(const Symbol &sym) {
  return sym.detailsIf<CommonBlockDetails>() != nullptr;
}

bool OmpStructureChecker::IsVariableListItem(const Symbol &sym) {
  return evaluate::IsVariable(sym) || sym.attrs().test(Attr::POINTER);
}

bool OmpStructureChecker::IsExtendedListItem(const Symbol &sym) {
  return IsVariableListItem(sym) || sym.IsSubprogram();
}

bool OmpStructureChecker::IsCloselyNestedRegion(const OmpDirectiveSet &set) {
  // Definition of close nesting:
  //
  // `A region nested inside another region with no parallel region nested
  // between them`
  //
  // Examples:
  //   non-parallel construct 1
  //    non-parallel construct 2
  //      parallel construct
  //        construct 3
  // In the above example, construct 3 is NOT closely nested inside construct 1
  // or 2
  //
  //   non-parallel construct 1
  //    non-parallel construct 2
  //        construct 3
  // In the above example, construct 3 is closely nested inside BOTH construct 1
  // and 2
  //
  // Algorithm:
  // Starting from the parent context, Check in a bottom-up fashion, each level
  // of the context stack. If we have a match for one of the (supplied)
  // violating directives, `close nesting` is satisfied. If no match is there in
  // the entire stack, `close nesting` is not satisfied. If at any level, a
  // `parallel` region is found, `close nesting` is not satisfied.

  if (CurrentDirectiveIsNested()) {
    int index = dirContext_.size() - 2;
    while (index != -1) {
      if (set.test(dirContext_[index].directive)) {
        return true;
      } else if (llvm::omp::allParallelSet.test(dirContext_[index].directive)) {
        return false;
      }
      index--;
    }
  }
  return false;
}

namespace {
struct ContiguousHelper {
  ContiguousHelper(SemanticsContext &context)
      : fctx_(context.foldingContext()) {}

  template <typename Contained>
  std::optional<bool> Visit(const common::Indirection<Contained> &x) {
    return Visit(x.value());
  }
  template <typename Contained>
  std::optional<bool> Visit(const common::Reference<Contained> &x) {
    return Visit(x.get());
  }
  template <typename T> std::optional<bool> Visit(const evaluate::Expr<T> &x) {
    return common::visit([&](auto &&s) { return Visit(s); }, x.u);
  }
  template <typename T>
  std::optional<bool> Visit(const evaluate::Designator<T> &x) {
    return common::visit(
        [this](auto &&s) { return evaluate::IsContiguous(s, fctx_); }, x.u);
  }
  template <typename T> std::optional<bool> Visit(const T &) {
    // Everything else.
    return std::nullopt;
  }

private:
  evaluate::FoldingContext &fctx_;
};
} // namespace

// Return values:
// - std::optional<bool>{true} if the object is known to be contiguous
// - std::optional<bool>{false} if the object is known not to be contiguous
// - std::nullopt if the object contiguity cannot be determined
std::optional<bool> OmpStructureChecker::IsContiguous(
    const parser::OmpObject &object) {
  return common::visit( //
      common::visitors{
          [&](const parser::Name &x) {
            // Any member of a common block must be contiguous.
            return std::optional<bool>{true};
          },
          [&](const parser::Designator &x) {
            evaluate::ExpressionAnalyzer ea{context_};
            if (MaybeExpr maybeExpr{ea.Analyze(x)}) {
              return ContiguousHelper{context_}.Visit(*maybeExpr);
            }
            return std::optional<bool>{};
          },
      },
      object.u);
}

void OmpStructureChecker::CheckVariableListItem(
    const SymbolSourceMap &symbols) {
  for (auto &[symbol, source] : symbols) {
    if (!IsVariableListItem(*symbol)) {
      context_.SayWithDecl(
          *symbol, source, "'%s' must be a variable"_err_en_US, symbol->name());
    }
  }
}

void OmpStructureChecker::CheckMultipleOccurrence(
    semantics::UnorderedSymbolSet &listVars,
    const std::list<parser::Name> &nameList, const parser::CharBlock &item,
    const std::string &clauseName) {
  for (auto const &var : nameList) {
    if (llvm::is_contained(listVars, *(var.symbol))) {
      context_.Say(item,
          "List item '%s' present at multiple %s clauses"_err_en_US,
          var.ToString(), clauseName);
    }
    listVars.insert(*(var.symbol));
  }
}

void OmpStructureChecker::CheckMultListItems() {
  semantics::UnorderedSymbolSet listVars;

  // Aligned clause
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_aligned)) {
    const auto &alignedClause{std::get<parser::OmpClause::Aligned>(clause->u)};
    const auto &alignedList{std::get<0>(alignedClause.v.t)};
    std::list<parser::Name> alignedNameList;
    for (const auto &ompObject : alignedList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          if (FindCommonBlockContaining(*(name->symbol))) {
            context_.Say(clause->source,
                "'%s' is a common block name and can not appear in an "
                "ALIGNED clause"_err_en_US,
                name->ToString());
          } else if (!(IsBuiltinCPtr(*(name->symbol)) ||
                         IsAllocatableOrObjectPointer(
                             &name->symbol->GetUltimate()))) {
            context_.Say(clause->source,
                "'%s' in ALIGNED clause must be of type C_PTR, POINTER or "
                "ALLOCATABLE"_err_en_US,
                name->ToString());
          } else {
            alignedNameList.push_back(*name);
          }
        } else {
          // The symbol is null, return early
          return;
        }
      }
    }
    CheckMultipleOccurrence(
        listVars, alignedNameList, clause->source, "ALIGNED");
  }

  // Nontemporal clause
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_nontemporal)) {
    const auto &nontempClause{
        std::get<parser::OmpClause::Nontemporal>(clause->u)};
    const auto &nontempNameList{nontempClause.v};
    CheckMultipleOccurrence(
        listVars, nontempNameList, clause->source, "NONTEMPORAL");
  }

  // Linear clause
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_linear)) {
    auto &linearClause{std::get<parser::OmpClause::Linear>(clause->u)};
    std::list<parser::Name> nameList;
    SymbolSourceMap symbols;
    GetSymbolsInObjectList(
        std::get<parser::OmpObjectList>(linearClause.v.t), symbols);
    llvm::transform(symbols, std::back_inserter(nameList), [&](auto &&pair) {
      return parser::Name{pair.second, const_cast<Symbol *>(pair.first)};
    });
    CheckMultipleOccurrence(listVars, nameList, clause->source, "LINEAR");
  }
}

bool OmpStructureChecker::HasInvalidWorksharingNesting(
    const parser::CharBlock &source, const OmpDirectiveSet &set) {
  // set contains all the invalid closely nested directives
  // for the given directive (`source` here)
  if (IsCloselyNestedRegion(set)) {
    context_.Say(source,
        "A worksharing region may not be closely nested inside a "
        "worksharing, explicit task, taskloop, critical, ordered, atomic, or "
        "master region"_err_en_US);
    return true;
  }
  return false;
}

void OmpStructureChecker::HasInvalidDistributeNesting(
    const parser::OpenMPLoopConstruct &x) {
  bool violation{false};
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
  if (llvm::omp::topDistributeSet.test(beginDir.v)) {
    // `distribute` region has to be nested
    if (!CurrentDirectiveIsNested()) {
      violation = true;
    } else {
      // `distribute` region has to be strictly nested inside `teams`
      if (!llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
        violation = true;
      }
    }
  }
  if (violation) {
    context_.Say(beginDir.source,
        "`DISTRIBUTE` region has to be strictly nested inside `TEAMS` "
        "region."_err_en_US);
  }
}
void OmpStructureChecker::HasInvalidLoopBinding(
    const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};

  auto teamsBindingChecker = [&](parser::MessageFixedText msg) {
    const auto &clauseList{std::get<parser::OmpClauseList>(beginLoopDir.t)};
    for (const auto &clause : clauseList.v) {
      if (const auto *bindClause{
              std::get_if<parser::OmpClause::Bind>(&clause.u)}) {
        if (bindClause->v.v != parser::OmpBindClause::Binding::Teams) {
          context_.Say(beginDir.source, msg);
        }
      }
    }
  };

  if (llvm::omp::Directive::OMPD_loop == beginDir.v &&
      CurrentDirectiveIsNested() &&
      llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
    teamsBindingChecker(
        "`BIND(TEAMS)` must be specified since the `LOOP` region is "
        "strictly nested inside a `TEAMS` region."_err_en_US);
  }

  if (OmpDirectiveSet{
          llvm::omp::OMPD_teams_loop, llvm::omp::OMPD_target_teams_loop}
          .test(beginDir.v)) {
    teamsBindingChecker(
        "`BIND(TEAMS)` must be specified since the `LOOP` directive is "
        "combined with a `TEAMS` construct."_err_en_US);
  }
}

void OmpStructureChecker::HasInvalidTeamsNesting(
    const llvm::omp::Directive &dir, const parser::CharBlock &source) {
  if (!llvm::omp::nestedTeamsAllowedSet.test(dir)) {
    context_.Say(source,
        "Only `DISTRIBUTE`, `PARALLEL`, or `LOOP` regions are allowed to be "
        "strictly nested inside `TEAMS` region."_err_en_US);
  }
}

void OmpStructureChecker::CheckPredefinedAllocatorRestriction(
    const parser::CharBlock &source, const parser::Name &name) {
  if (const auto *symbol{name.symbol}) {
    const auto *commonBlock{FindCommonBlockContaining(*symbol)};
    const auto &scope{context_.FindScope(symbol->name())};
    const Scope &containingScope{GetProgramUnitContaining(scope)};
    if (!isPredefinedAllocator &&
        (IsSaved(*symbol) || commonBlock ||
            containingScope.kind() == Scope::Kind::Module)) {
      context_.Say(source,
          "If list items within the %s directive have the "
          "SAVE attribute, are a common block name, or are "
          "declared in the scope of a module, then only "
          "predefined memory allocator parameters can be used "
          "in the allocator clause"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}

void OmpStructureChecker::CheckPredefinedAllocatorRestriction(
    const parser::CharBlock &source,
    const parser::OmpObjectList &ompObjectList) {
  for (const auto &ompObject : ompObjectList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              if (const auto *dataRef{
                      std::get_if<parser::DataRef>(&designator.u)}) {
                if (const auto *name{std::get_if<parser::Name>(&dataRef->u)}) {
                  CheckPredefinedAllocatorRestriction(source, *name);
                }
              }
            },
            [&](const parser::Name &name) {
              CheckPredefinedAllocatorRestriction(source, name);
            },
        },
        ompObject.u);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Hint &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_hint);
  auto &dirCtx{GetContext()};

  if (std::optional<int64_t> maybeVal{GetIntValue(x.v.v)}) {
    int64_t val{*maybeVal};
    if (val >= 0) {
      // Check contradictory values.
      if ((val & 0xC) == 0xC || // omp_sync_hint_speculative and nonspeculative
          (val & 0x3) == 0x3) { // omp_sync_hint_contended and uncontended
        context_.Say(dirCtx.clauseSource,
            "The synchronization hint is not valid"_err_en_US);
      }
    } else {
      context_.Say(dirCtx.clauseSource,
          "Synchronization hint must be non-negative"_err_en_US);
    }
  } else {
    context_.Say(dirCtx.clauseSource,
        "Synchronization hint must be a constant integer value"_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OmpDirectiveSpecification &x) {
  // OmpDirectiveSpecification exists on its own only in METADIRECTIVE.
  // In other cases it's a part of other constructs that handle directive
  // context stack by themselves.
  if (GetDirectiveNest(MetadirectiveNest)) {
    PushContextAndClauseSets(
        std::get<parser::OmpDirectiveName>(x.t).source, x.DirId());
  }
}

void OmpStructureChecker::Leave(const parser::OmpDirectiveSpecification &) {
  if (GetDirectiveNest(MetadirectiveNest)) {
    dirContext_.pop_back();
  }
}

void OmpStructureChecker::Enter(const parser::OmpMetadirectiveDirective &x) {
  EnterDirectiveNest(MetadirectiveNest);
  PushContextAndClauseSets(x.source, llvm::omp::Directive::OMPD_metadirective);
}

void OmpStructureChecker::Leave(const parser::OmpMetadirectiveDirective &) {
  ExitDirectiveNest(MetadirectiveNest);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPConstruct &x) {
  // Simd Construct with Ordered Construct Nesting check
  // We cannot use CurrentDirectiveIsNested() here because
  // PushContextAndClauseSets() has not been called yet, it is
  // called individually for each construct.  Therefore a
  // dirContext_ size `1` means the current construct is nested
  if (dirContext_.size() >= 1) {
    if (GetDirectiveNest(SIMDNest) > 0) {
      CheckSIMDNest(x);
    }
    if (GetDirectiveNest(TargetNest) > 0) {
      CheckTargetNest(x);
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPConstruct &) {
  for (const auto &[sym, source] : deferredNonVariables_) {
    context_.SayWithDecl(
        *sym, source, "'%s' must be a variable"_err_en_US, sym->name());
  }
  deferredNonVariables_.clear();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeConstruct &x) {
  EnterDirectiveNest(DeclarativeNest);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeConstruct &x) {
  ExitDirectiveNest(DeclarativeNest);
}

void OmpStructureChecker::AddEndDirectiveClauses(
    const parser::OmpClauseList &clauses) {
  for (const parser::OmpClause &clause : clauses.v) {
    GetContext().endDirectiveClauses.push_back(clause.Id());
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  loopStack_.push_back(&x);
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};

  PushContextAndClauseSets(beginDir.source, beginDir.v);

  // check matching, End directive is optional
  if (const auto &endLoopDir{
          std::get<std::optional<parser::OmpEndLoopDirective>>(x.t)}) {
    const auto &endDir{
        std::get<parser::OmpLoopDirective>(endLoopDir.value().t)};

    CheckMatching<parser::OmpLoopDirective>(beginDir, endDir);

    AddEndDirectiveClauses(std::get<parser::OmpClauseList>(endLoopDir->t));
  }

  if (llvm::omp::allSimdSet.test(GetContext().directive)) {
    EnterDirectiveNest(SIMDNest);
  }

  // Combined target loop constructs are target device constructs. Keep track of
  // whether any such construct has been visited to later check that REQUIRES
  // directives for target-related options don't appear after them.
  if (llvm::omp::allTargetSet.test(beginDir.v)) {
    deviceConstructFound_ = true;
  }

  if (beginDir.v == llvm::omp::Directive::OMPD_do) {
    // 2.7.1 do-clause -> private-clause |
    //                    firstprivate-clause |
    //                    lastprivate-clause |
    //                    linear-clause |
    //                    reduction-clause |
    //                    schedule-clause |
    //                    collapse-clause |
    //                    ordered-clause

    // nesting check
    HasInvalidWorksharingNesting(
        beginDir.source, llvm::omp::nestedWorkshareErrSet);
  }
  SetLoopInfo(x);

  if (const auto &doConstruct{
          std::get<std::optional<parser::DoConstruct>>(x.t)}) {
    const auto &doBlock{std::get<parser::Block>(doConstruct->t)};
    CheckNoBranching(doBlock, beginDir.v, beginDir.source);
  }
  CheckLoopItrVariableIsInt(x);
  CheckAssociatedLoopConstraints(x);
  HasInvalidDistributeNesting(x);
  HasInvalidLoopBinding(x);
  if (CurrentDirectiveIsNested() &&
      llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
    HasInvalidTeamsNesting(beginDir.v, beginDir.source);
  }
  if ((beginDir.v == llvm::omp::Directive::OMPD_distribute_parallel_do_simd) ||
      (beginDir.v == llvm::omp::Directive::OMPD_distribute_simd)) {
    CheckDistLinear(x);
  }
}
const parser::Name OmpStructureChecker::GetLoopIndex(
    const parser::DoConstruct *x) {
  using Bounds = parser::LoopControl::Bounds;
  return std::get<Bounds>(x->GetLoopControl()->u).name.thing;
}
void OmpStructureChecker::SetLoopInfo(const parser::OpenMPLoopConstruct &x) {
  if (const auto &loopConstruct{
          std::get<std::optional<parser::DoConstruct>>(x.t)}) {
    const parser::DoConstruct *loop{&*loopConstruct};
    if (loop && loop->IsDoNormal()) {
      const parser::Name &itrVal{GetLoopIndex(loop)};
      SetLoopIv(itrVal.symbol);
    }
  }
}

void OmpStructureChecker::CheckIteratorRange(
    const parser::OmpIteratorSpecifier &x) {
  // Check:
  // 1. Whether begin/end are present.
  // 2. Whether the step value is non-zero.
  // 3. If the step has a known sign, whether the lower/upper bounds form
  // a proper interval.
  const auto &[begin, end, step]{std::get<parser::SubscriptTriplet>(x.t).t};
  if (!begin || !end) {
    context_.Say(x.source,
        "The begin and end expressions in iterator range-specification are "
        "mandatory"_err_en_US);
  }
  // [5.2:67:19] In a range-specification, if the step is not specified its
  // value is implicitly defined to be 1.
  if (auto stepv{step ? GetIntValue(*step) : std::optional<int64_t>{1}}) {
    if (*stepv == 0) {
      context_.Say(
          x.source, "The step value in the iterator range is 0"_warn_en_US);
    } else if (begin && end) {
      std::optional<int64_t> beginv{GetIntValue(*begin)};
      std::optional<int64_t> endv{GetIntValue(*end)};
      if (beginv && endv) {
        if (*stepv > 0 && *beginv > *endv) {
          context_.Say(x.source,
              "The begin value is greater than the end value in iterator "
              "range-specification with a positive step"_warn_en_US);
        } else if (*stepv < 0 && *beginv < *endv) {
          context_.Say(x.source,
              "The begin value is less than the end value in iterator "
              "range-specification with a negative step"_warn_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckIteratorModifier(const parser::OmpIterator &x) {
  // Check if all iterator variables have integer type.
  for (auto &&iterSpec : x.v) {
    bool isInteger{true};
    auto &typeDecl{std::get<parser::TypeDeclarationStmt>(iterSpec.t)};
    auto &typeSpec{std::get<parser::DeclarationTypeSpec>(typeDecl.t)};
    if (!std::holds_alternative<parser::IntrinsicTypeSpec>(typeSpec.u)) {
      isInteger = false;
    } else {
      auto &intrinType{std::get<parser::IntrinsicTypeSpec>(typeSpec.u)};
      if (!std::holds_alternative<parser::IntegerTypeSpec>(intrinType.u)) {
        isInteger = false;
      }
    }
    if (!isInteger) {
      context_.Say(iterSpec.source,
          "The iterator variable must be of integer type"_err_en_US);
    }
    CheckIteratorRange(iterSpec);
  }
}

void OmpStructureChecker::CheckLoopItrVariableIsInt(
    const parser::OpenMPLoopConstruct &x) {
  if (const auto &loopConstruct{
          std::get<std::optional<parser::DoConstruct>>(x.t)}) {

    for (const parser::DoConstruct *loop{&*loopConstruct}; loop;) {
      if (loop->IsDoNormal()) {
        const parser::Name &itrVal{GetLoopIndex(loop)};
        if (itrVal.symbol) {
          const auto *type{itrVal.symbol->GetType()};
          if (!type->IsNumeric(TypeCategory::Integer)) {
            context_.Say(itrVal.source,
                "The DO loop iteration"
                " variable must be of the type integer."_err_en_US,
                itrVal.ToString());
          }
        }
      }
      // Get the next DoConstruct if block is not empty.
      const auto &block{std::get<parser::Block>(loop->t)};
      const auto it{block.begin()};
      loop = it != block.end() ? parser::Unwrap<parser::DoConstruct>(*it)
                               : nullptr;
    }
  }
}

void OmpStructureChecker::CheckSIMDNest(const parser::OpenMPConstruct &c) {
  // Check the following:
  //  The only OpenMP constructs that can be encountered during execution of
  // a simd region are the `atomic` construct, the `loop` construct, the `simd`
  // construct and the `ordered` construct with the `simd` clause.

  // Check if the parent context has the SIMD clause
  // Please note that we use GetContext() instead of GetContextParent()
  // because PushContextAndClauseSets() has not been called on the
  // current context yet.
  // TODO: Check for declare simd regions.
  bool eligibleSIMD{false};
  common::visit(
      common::visitors{
          // Allow `!$OMP ORDERED SIMD`
          [&](const parser::OpenMPBlockConstruct &c) {
            const auto &beginBlockDir{
                std::get<parser::OmpBeginBlockDirective>(c.t)};
            const auto &beginDir{
                std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
            if (beginDir.v == llvm::omp::Directive::OMPD_ordered) {
              const auto &clauses{
                  std::get<parser::OmpClauseList>(beginBlockDir.t)};
              for (const auto &clause : clauses.v) {
                if (std::get_if<parser::OmpClause::Simd>(&clause.u)) {
                  eligibleSIMD = true;
                  break;
                }
              }
            }
          },
          [&](const parser::OpenMPStandaloneConstruct &c) {
            if (auto *ssc{std::get_if<parser::OpenMPSimpleStandaloneConstruct>(
                    &c.u)}) {
              llvm::omp::Directive dirId{ssc->v.DirId()};
              if (dirId == llvm::omp::Directive::OMPD_ordered) {
                for (const parser::OmpClause &x : ssc->v.Clauses().v) {
                  if (x.Id() == llvm::omp::Clause::OMPC_simd) {
                    eligibleSIMD = true;
                    break;
                  }
                }
              } else if (dirId == llvm::omp::Directive::OMPD_scan) {
                eligibleSIMD = true;
              }
            }
          },
          // Allowing SIMD and loop construct
          [&](const parser::OpenMPLoopConstruct &c) {
            const auto &beginLoopDir{
                std::get<parser::OmpBeginLoopDirective>(c.t)};
            const auto &beginDir{
                std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
            if ((beginDir.v == llvm::omp::Directive::OMPD_simd) ||
                (beginDir.v == llvm::omp::Directive::OMPD_do_simd) ||
                (beginDir.v == llvm::omp::Directive::OMPD_loop)) {
              eligibleSIMD = true;
            }
          },
          [&](const parser::OpenMPAtomicConstruct &c) {
            // Allow `!$OMP ATOMIC`
            eligibleSIMD = true;
          },
          [&](const auto &c) {},
      },
      c.u);
  if (!eligibleSIMD) {
    context_.Say(parser::FindSourceLocation(c),
        "The only OpenMP constructs that can be encountered during execution "
        "of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, "
        "the `SIMD` construct, the `SCAN` construct and the `ORDERED` "
        "construct with the `SIMD` clause."_err_en_US);
  }
}

void OmpStructureChecker::CheckTargetNest(const parser::OpenMPConstruct &c) {
  // 2.12.5 Target Construct Restriction
  bool eligibleTarget{true};
  llvm::omp::Directive ineligibleTargetDir;
  common::visit(
      common::visitors{
          [&](const parser::OpenMPBlockConstruct &c) {
            const auto &beginBlockDir{
                std::get<parser::OmpBeginBlockDirective>(c.t)};
            const auto &beginDir{
                std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
            if (beginDir.v == llvm::omp::Directive::OMPD_target_data) {
              eligibleTarget = false;
              ineligibleTargetDir = beginDir.v;
            }
          },
          [&](const parser::OpenMPStandaloneConstruct &c) {
            common::visit(
                common::visitors{
                    [&](const parser::OpenMPSimpleStandaloneConstruct &c) {
                      switch (llvm::omp::Directive dirId{c.v.DirId()}) {
                      case llvm::omp::Directive::OMPD_target_update:
                      case llvm::omp::Directive::OMPD_target_enter_data:
                      case llvm::omp::Directive::OMPD_target_exit_data:
                        eligibleTarget = false;
                        ineligibleTargetDir = dirId;
                        break;
                      default:
                        break;
                      }
                    },
                    [&](const auto &c) {},
                },
                c.u);
          },
          [&](const parser::OpenMPLoopConstruct &c) {
            const auto &beginLoopDir{
                std::get<parser::OmpBeginLoopDirective>(c.t)};
            const auto &beginDir{
                std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
            if (llvm::omp::allTargetSet.test(beginDir.v)) {
              eligibleTarget = false;
              ineligibleTargetDir = beginDir.v;
            }
          },
          [&](const auto &c) {},
      },
      c.u);
  if (!eligibleTarget) {
    context_.Warn(common::UsageWarning::OpenMPUsage,
        parser::FindSourceLocation(c),
        "If %s directive is nested inside TARGET region, the behaviour is unspecified"_port_en_US,
        parser::ToUpperCaseLetters(
            getDirectiveName(ineligibleTargetDir).str()));
  }
}

std::int64_t OmpStructureChecker::GetOrdCollapseLevel(
    const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &clauseList{std::get<parser::OmpClauseList>(beginLoopDir.t)};
  std::int64_t orderedCollapseLevel{1};
  std::int64_t orderedLevel{1};
  std::int64_t collapseLevel{1};

  for (const auto &clause : clauseList.v) {
    if (const auto *collapseClause{
            std::get_if<parser::OmpClause::Collapse>(&clause.u)}) {
      if (const auto v{GetIntValue(collapseClause->v)}) {
        collapseLevel = *v;
      }
    }
    if (const auto *orderedClause{
            std::get_if<parser::OmpClause::Ordered>(&clause.u)}) {
      if (const auto v{GetIntValue(orderedClause->v)}) {
        orderedLevel = *v;
      }
    }
  }
  if (orderedLevel >= collapseLevel) {
    orderedCollapseLevel = orderedLevel;
  } else {
    orderedCollapseLevel = collapseLevel;
  }
  return orderedCollapseLevel;
}

void OmpStructureChecker::CheckAssociatedLoopConstraints(
    const parser::OpenMPLoopConstruct &x) {
  std::int64_t ordCollapseLevel{GetOrdCollapseLevel(x)};
  AssociatedLoopChecker checker{context_, ordCollapseLevel};
  parser::Walk(x, checker);
}

void OmpStructureChecker::CheckDistLinear(
    const parser::OpenMPLoopConstruct &x) {

  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &clauses{std::get<parser::OmpClauseList>(beginLoopDir.t)};

  SymbolSourceMap indexVars;

  // Collect symbols of all the variables from linear clauses
  for (auto &clause : clauses.v) {
    if (auto *linearClause{std::get_if<parser::OmpClause::Linear>(&clause.u)}) {
      auto &objects{std::get<parser::OmpObjectList>(linearClause->v.t)};
      GetSymbolsInObjectList(objects, indexVars);
    }
  }

  if (!indexVars.empty()) {
    // Get collapse level, if given, to find which loops are "associated."
    std::int64_t collapseVal{GetOrdCollapseLevel(x)};
    // Include the top loop if no collapse is specified
    if (collapseVal == 0) {
      collapseVal = 1;
    }

    // Match the loop index variables with the collected symbols from linear
    // clauses.
    if (const auto &loopConstruct{
            std::get<std::optional<parser::DoConstruct>>(x.t)}) {
      for (const parser::DoConstruct *loop{&*loopConstruct}; loop;) {
        if (loop->IsDoNormal()) {
          const parser::Name &itrVal{GetLoopIndex(loop)};
          if (itrVal.symbol) {
            // Remove the symbol from the collected set
            indexVars.erase(&itrVal.symbol->GetUltimate());
          }
          collapseVal--;
          if (collapseVal == 0) {
            break;
          }
        }
        // Get the next DoConstruct if block is not empty.
        const auto &block{std::get<parser::Block>(loop->t)};
        const auto it{block.begin()};
        loop = it != block.end() ? parser::Unwrap<parser::DoConstruct>(*it)
                                 : nullptr;
      }
    }

    // Show error for the remaining variables
    for (auto &[symbol, source] : indexVars) {
      const Symbol &root{GetAssociationRoot(*symbol)};
      context_.Say(source,
          "Variable '%s' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE"_err_en_US,
          root.name());
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &clauseList{std::get<parser::OmpClauseList>(beginLoopDir.t)};

  // A few semantic checks for InScan reduction are performed below as SCAN
  // constructs inside LOOP may add the relevant information. Scan reduction is
  // supported only in loop constructs, so same checks are not applicable to
  // other directives.
  using ReductionModifier = parser::OmpReductionModifier;
  for (const auto &clause : clauseList.v) {
    if (const auto *reductionClause{
            std::get_if<parser::OmpClause::Reduction>(&clause.u)}) {
      auto &modifiers{OmpGetModifiers(reductionClause->v)};
      auto *maybeModifier{OmpGetUniqueModifier<ReductionModifier>(modifiers)};
      if (maybeModifier &&
          maybeModifier->v == ReductionModifier::Value::Inscan) {
        const auto &objectList{
            std::get<parser::OmpObjectList>(reductionClause->v.t)};
        auto checkReductionSymbolInScan = [&](const parser::Name *name) {
          if (auto &symbol = name->symbol) {
            if (!symbol->test(Symbol::Flag::OmpInclusiveScan) &&
                !symbol->test(Symbol::Flag::OmpExclusiveScan)) {
              context_.Say(name->source,
                  "List item %s must appear in EXCLUSIVE or "
                  "INCLUSIVE clause of an "
                  "enclosed SCAN directive"_err_en_US,
                  name->ToString());
            }
          }
        };
        for (const auto &ompObj : objectList.v) {
          common::visit(
              common::visitors{
                  [&](const parser::Designator &designator) {
                    if (const auto *name{semantics::getDesignatorNameIfDataRef(
                            designator)}) {
                      checkReductionSymbolInScan(name);
                    }
                  },
                  [&](const auto &name) { checkReductionSymbolInScan(&name); },
              },
              ompObj.u);
        }
      }
    }
  }
  if (llvm::omp::allSimdSet.test(GetContext().directive)) {
    ExitDirectiveNest(SIMDNest);
  }
  dirContext_.pop_back();

  assert(!loopStack_.empty() && "Expecting non-empty loop stack");
#ifndef NDEBUG
  const LoopConstruct &top{loopStack_.back()};
  auto *loopc{std::get_if<const parser::OpenMPLoopConstruct *>(&top)};
  assert(loopc != nullptr && *loopc == &x && "Mismatched loop constructs");
#endif
  loopStack_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndLoopDirective &x) {
  const auto &dir{std::get<parser::OmpLoopDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  // 2.7.1 end-do -> END DO [nowait-clause]
  // 2.8.3 end-do-simd -> END DO SIMD [nowait-clause]
  case llvm::omp::Directive::OMPD_do:
    PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_end_do);
    break;
  case llvm::omp::Directive::OMPD_do_simd:
    PushContextAndClauseSets(
        dir.source, llvm::omp::Directive::OMPD_end_do_simd);
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OmpEndLoopDirective &x) {
  if ((GetContext().directive == llvm::omp::Directive::OMPD_end_do) ||
      (GetContext().directive == llvm::omp::Directive::OMPD_end_do_simd)) {
    dirContext_.pop_back();
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OmpBeginBlockDirective>(x.t)};
  const auto &endBlockDir{std::get<parser::OmpEndBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
  const auto &endDir{std::get<parser::OmpBlockDirective>(endBlockDir.t)};
  const parser::Block &block{std::get<parser::Block>(x.t)};

  CheckMatching<parser::OmpBlockDirective>(beginDir, endDir);

  PushContextAndClauseSets(beginDir.source, beginDir.v);
  if (llvm::omp::allTargetSet.test(GetContext().directive)) {
    EnterDirectiveNest(TargetNest);
  }

  if (CurrentDirectiveIsNested()) {
    if (llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
      HasInvalidTeamsNesting(beginDir.v, beginDir.source);
    }
    if (GetContext().directive == llvm::omp::Directive::OMPD_master) {
      CheckMasterNesting(x);
    }
    // A teams region can only be strictly nested within the implicit parallel
    // region or a target region.
    if (GetContext().directive == llvm::omp::Directive::OMPD_teams &&
        GetContextParent().directive != llvm::omp::Directive::OMPD_target) {
      context_.Say(parser::FindSourceLocation(x),
          "%s region can only be strictly nested within the implicit parallel "
          "region or TARGET region"_err_en_US,
          ContextDirectiveAsFortran());
    }
    // If a teams construct is nested within a target construct, that target
    // construct must contain no statements, declarations or directives outside
    // of the teams construct.
    if (GetContext().directive == llvm::omp::Directive::OMPD_teams &&
        GetContextParent().directive == llvm::omp::Directive::OMPD_target &&
        !GetDirectiveNest(TargetBlockOnlyTeams)) {
      context_.Say(GetContextParent().directiveSource,
          "TARGET construct with nested TEAMS region contains statements or "
          "directives outside of the TEAMS construct"_err_en_US);
    }
  }

  CheckNoBranching(block, beginDir.v, beginDir.source);

  // Target block constructs are target device constructs. Keep track of
  // whether any such construct has been visited to later check that REQUIRES
  // directives for target-related options don't appear after them.
  if (llvm::omp::allTargetSet.test(beginDir.v)) {
    deviceConstructFound_ = true;
  }

  if (GetContext().directive == llvm::omp::Directive::OMPD_single) {
    std::set<Symbol *> singleCopyprivateSyms;
    std::set<Symbol *> endSingleCopyprivateSyms;
    bool foundNowait{false};
    parser::CharBlock NowaitSource;

    auto catchCopyPrivateNowaitClauses = [&](const auto &dir, bool endDir) {
      for (auto &clause : std::get<parser::OmpClauseList>(dir.t).v) {
        if (clause.Id() == llvm::omp::Clause::OMPC_copyprivate) {
          for (const auto &ompObject : GetOmpObjectList(clause)->v) {
            const auto *name{parser::Unwrap<parser::Name>(ompObject)};
            if (Symbol * symbol{name->symbol}) {
              if (singleCopyprivateSyms.count(symbol)) {
                if (endDir) {
                  context_.Warn(common::UsageWarning::OpenMPUsage, name->source,
                      "The COPYPRIVATE clause with '%s' is already used on the SINGLE directive"_warn_en_US,
                      name->ToString());
                } else {
                  context_.Say(name->source,
                      "'%s' appears in more than one COPYPRIVATE clause on the SINGLE directive"_err_en_US,
                      name->ToString());
                }
              } else if (endSingleCopyprivateSyms.count(symbol)) {
                context_.Say(name->source,
                    "'%s' appears in more than one COPYPRIVATE clause on the END SINGLE directive"_err_en_US,
                    name->ToString());
              } else {
                if (endDir) {
                  endSingleCopyprivateSyms.insert(symbol);
                } else {
                  singleCopyprivateSyms.insert(symbol);
                }
              }
            }
          }
        } else if (clause.Id() == llvm::omp::Clause::OMPC_nowait) {
          if (foundNowait) {
            context_.Say(clause.source,
                "At most one NOWAIT clause can appear on the SINGLE directive"_err_en_US);
          } else {
            foundNowait = !endDir;
          }
          if (!NowaitSource.ToString().size()) {
            NowaitSource = clause.source;
          }
        }
      }
    };
    catchCopyPrivateNowaitClauses(beginBlockDir, false);
    catchCopyPrivateNowaitClauses(endBlockDir, true);
    unsigned version{context_.langOptions().OpenMPVersion};
    if (version <= 52 && NowaitSource.ToString().size() &&
        (singleCopyprivateSyms.size() || endSingleCopyprivateSyms.size())) {
      context_.Say(NowaitSource,
          "NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive"_err_en_US);
    }
  }

  switch (beginDir.v) {
  case llvm::omp::Directive::OMPD_target:
    if (CheckTargetBlockOnlyTeams(block)) {
      EnterDirectiveNest(TargetBlockOnlyTeams);
    }
    break;
  case llvm::omp::OMPD_workshare:
  case llvm::omp::OMPD_parallel_workshare:
    CheckWorkshareBlockStmts(block, beginDir.source);
    HasInvalidWorksharingNesting(
        beginDir.source, llvm::omp::nestedWorkshareErrSet);
    break;
  case llvm::omp::Directive::OMPD_scope:
  case llvm::omp::Directive::OMPD_single:
    // TODO: This check needs to be extended while implementing nesting of
    // regions checks.
    HasInvalidWorksharingNesting(
        beginDir.source, llvm::omp::nestedWorkshareErrSet);
    break;
  case llvm::omp::Directive::OMPD_task: {
    const auto &clauses{std::get<parser::OmpClauseList>(beginBlockDir.t)};
    for (const auto &clause : clauses.v) {
      if (std::get_if<parser::OmpClause::Untied>(&clause.u)) {
        OmpUnitedTaskDesignatorChecker check{context_};
        parser::Walk(block, check);
      }
    }
    break;
  }
  default:
    break;
  }
}

void OmpStructureChecker::CheckMasterNesting(
    const parser::OpenMPBlockConstruct &x) {
  // A MASTER region may not be `closely nested` inside a worksharing, loop,
  // task, taskloop, or atomic region.
  // TODO:  Expand the check to include `LOOP` construct as well when it is
  // supported.
  if (IsCloselyNestedRegion(llvm::omp::nestedMasterErrSet)) {
    context_.Say(parser::FindSourceLocation(x),
        "`MASTER` region may not be closely nested inside of `WORKSHARING`, "
        "`LOOP`, `TASK`, `TASKLOOP`,"
        " or `ATOMIC` region."_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPAssumeConstruct &x) {
  PushContextAndClauseSets(x.source, llvm::omp::Directive::OMPD_assume);
}

void OmpStructureChecker::Leave(const parser::OpenMPAssumeConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeAssumes &x) {
  PushContextAndClauseSets(x.source, llvm::omp::Directive::OMPD_assumes);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeAssumes &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Leave(const parser::OpenMPBlockConstruct &) {
  if (GetDirectiveNest(TargetBlockOnlyTeams)) {
    ExitDirectiveNest(TargetBlockOnlyTeams);
  }
  if (llvm::omp::allTargetSet.test(GetContext().directive)) {
    ExitDirectiveNest(TargetNest);
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::ChecksOnOrderedAsBlock() {
  if (FindClause(llvm::omp::Clause::OMPC_depend)) {
    context_.Say(GetContext().clauseSource,
        "DEPEND clauses are not allowed when ORDERED construct is a block construct with an ORDERED region"_err_en_US);
    return;
  }

  bool isNestedInDo{false};
  bool isNestedInDoSIMD{false};
  bool isNestedInSIMD{false};
  bool noOrderedClause{false};
  bool isOrderedClauseWithPara{false};
  bool isCloselyNestedRegion{true};
  if (CurrentDirectiveIsNested()) {
    for (int i = (int)dirContext_.size() - 2; i >= 0; i--) {
      if (llvm::omp::nestedOrderedErrSet.test(dirContext_[i].directive)) {
        context_.Say(GetContext().directiveSource,
            "`ORDERED` region may not be closely nested inside of `CRITICAL`, "
            "`ORDERED`, explicit `TASK` or `TASKLOOP` region."_err_en_US);
        break;
      } else if (llvm::omp::allDoSet.test(dirContext_[i].directive)) {
        isNestedInDo = true;
        isNestedInDoSIMD =
            llvm::omp::allDoSimdSet.test(dirContext_[i].directive);
        if (const auto *clause{
                FindClause(dirContext_[i], llvm::omp::Clause::OMPC_ordered)}) {
          const auto &orderedClause{
              std::get<parser::OmpClause::Ordered>(clause->u)};
          const auto orderedValue{GetIntValue(orderedClause.v)};
          isOrderedClauseWithPara = orderedValue > 0;
        } else {
          noOrderedClause = true;
        }
        break;
      } else if (llvm::omp::allSimdSet.test(dirContext_[i].directive)) {
        isNestedInSIMD = true;
        break;
      } else if (llvm::omp::nestedOrderedParallelErrSet.test(
                     dirContext_[i].directive)) {
        isCloselyNestedRegion = false;
        break;
      }
    }
  }

  if (!isCloselyNestedRegion) {
    context_.Say(GetContext().directiveSource,
        "An ORDERED directive without the DEPEND clause must be closely nested "
        "in a SIMD, worksharing-loop, or worksharing-loop SIMD "
        "region"_err_en_US);
  } else {
    if (CurrentDirectiveIsNested() &&
        FindClause(llvm::omp::Clause::OMPC_simd) &&
        (!isNestedInDoSIMD && !isNestedInSIMD)) {
      context_.Say(GetContext().directiveSource,
          "An ORDERED directive with SIMD clause must be closely nested in a "
          "SIMD or worksharing-loop SIMD region"_err_en_US);
    }
    if (isNestedInDo && (noOrderedClause || isOrderedClauseWithPara)) {
      context_.Say(GetContext().directiveSource,
          "An ORDERED directive without the DEPEND clause must be closely "
          "nested in a worksharing-loop (or worksharing-loop SIMD) region with "
          "ORDERED clause without the parameter"_err_en_US);
    }
  }
}

void OmpStructureChecker::Leave(const parser::OmpBeginBlockDirective &) {
  switch (GetContext().directive) {
  case llvm::omp::Directive::OMPD_ordered:
    // [5.1] 2.19.9 Ordered Construct Restriction
    ChecksOnOrderedAsBlock();
    break;
  default:
    break;
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPSectionsConstruct &x) {
  const auto &beginSectionsDir{
      std::get<parser::OmpBeginSectionsDirective>(x.t)};
  const auto &endSectionsDir{std::get<parser::OmpEndSectionsDirective>(x.t)};
  const auto &beginDir{
      std::get<parser::OmpSectionsDirective>(beginSectionsDir.t)};
  const auto &endDir{std::get<parser::OmpSectionsDirective>(endSectionsDir.t)};
  CheckMatching<parser::OmpSectionsDirective>(beginDir, endDir);

  PushContextAndClauseSets(beginDir.source, beginDir.v);
  AddEndDirectiveClauses(std::get<parser::OmpClauseList>(endSectionsDir.t));

  const auto &sectionBlocks{std::get<parser::OmpSectionBlocks>(x.t)};
  for (const parser::OpenMPConstruct &block : sectionBlocks.v) {
    CheckNoBranching(std::get<parser::OpenMPSectionConstruct>(block.u).v,
        beginDir.v, beginDir.source);
  }
  HasInvalidWorksharingNesting(
      beginDir.source, llvm::omp::nestedWorkshareErrSet);
}

void OmpStructureChecker::Leave(const parser::OpenMPSectionsConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndSectionsDirective &x) {
  const auto &dir{std::get<parser::OmpSectionsDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
    // 2.7.2 end-sections -> END SECTIONS [nowait-clause]
  case llvm::omp::Directive::OMPD_sections:
    PushContextAndClauseSets(
        dir.source, llvm::omp::Directive::OMPD_end_sections);
    break;
  default:
    // no clauses are allowed
    break;
  }
}

// TODO: Verify the popping of dirContext requirement after nowait
// implementation, as there is an implicit barrier at the end of the worksharing
// constructs unless a nowait clause is specified. Only OMPD_end_sections is
// popped becuase it is pushed while entering the EndSectionsDirective.
void OmpStructureChecker::Leave(const parser::OmpEndSectionsDirective &x) {
  if (GetContext().directive == llvm::omp::Directive::OMPD_end_sections) {
    dirContext_.pop_back();
  }
}

void OmpStructureChecker::CheckThreadprivateOrDeclareTargetVar(
    const parser::OmpObjectList &objList) {
  for (const auto &ompObject : objList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &) {
              if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
                // The symbol is null, return early, CheckSymbolNames
                // should have already reported the missing symbol as a
                // diagnostic error
                if (!name->symbol) {
                  return;
                }

                if (name->symbol->GetUltimate().IsSubprogram()) {
                  if (GetContext().directive ==
                      llvm::omp::Directive::OMPD_threadprivate)
                    context_.Say(name->source,
                        "The procedure name cannot be in a %s "
                        "directive"_err_en_US,
                        ContextDirectiveAsFortran());
                  // TODO: Check for procedure name in declare target directive.
                } else if (name->symbol->attrs().test(Attr::PARAMETER)) {
                  if (GetContext().directive ==
                      llvm::omp::Directive::OMPD_threadprivate)
                    context_.Say(name->source,
                        "The entity with PARAMETER attribute cannot be in a %s "
                        "directive"_err_en_US,
                        ContextDirectiveAsFortran());
                  else if (GetContext().directive ==
                      llvm::omp::Directive::OMPD_declare_target)
                    context_.Warn(common::UsageWarning::OpenMPUsage,
                        name->source,
                        "The entity with PARAMETER attribute is used in a %s directive"_warn_en_US,
                        ContextDirectiveAsFortran());
                } else if (FindCommonBlockContaining(*name->symbol)) {
                  context_.Say(name->source,
                      "A variable in a %s directive cannot be an element of a "
                      "common block"_err_en_US,
                      ContextDirectiveAsFortran());
                } else if (FindEquivalenceSet(*name->symbol)) {
                  context_.Say(name->source,
                      "A variable in a %s directive cannot appear in an "
                      "EQUIVALENCE statement"_err_en_US,
                      ContextDirectiveAsFortran());
                } else if (name->symbol->test(Symbol::Flag::OmpThreadprivate) &&
                    GetContext().directive ==
                        llvm::omp::Directive::OMPD_declare_target) {
                  context_.Say(name->source,
                      "A THREADPRIVATE variable cannot appear in a %s "
                      "directive"_err_en_US,
                      ContextDirectiveAsFortran());
                } else {
                  const semantics::Scope &useScope{
                      context_.FindScope(GetContext().directiveSource)};
                  const semantics::Scope &curScope =
                      name->symbol->GetUltimate().owner();
                  if (!curScope.IsTopLevel()) {
                    const semantics::Scope &declScope =
                        GetProgramUnitOrBlockConstructContaining(curScope);
                    const semantics::Symbol *sym{
                        declScope.parent().FindSymbol(name->symbol->name())};
                    if (sym &&
                        (sym->has<MainProgramDetails>() ||
                            sym->has<ModuleDetails>())) {
                      context_.Say(name->source,
                          "The module name or main program name cannot be in a "
                          "%s "
                          "directive"_err_en_US,
                          ContextDirectiveAsFortran());
                    } else if (!IsSaved(*name->symbol) &&
                        declScope.kind() != Scope::Kind::MainProgram &&
                        declScope.kind() != Scope::Kind::Module) {
                      context_.Say(name->source,
                          "A variable that appears in a %s directive must be "
                          "declared in the scope of a module or have the SAVE "
                          "attribute, either explicitly or "
                          "implicitly"_err_en_US,
                          ContextDirectiveAsFortran());
                    } else if (useScope != declScope) {
                      context_.Say(name->source,
                          "The %s directive and the common block or variable "
                          "in it must appear in the same declaration section "
                          "of a scoping unit"_err_en_US,
                          ContextDirectiveAsFortran());
                    }
                  }
                }
              }
            },
            [&](const parser::Name &name) {
              if (name.symbol) {
                if (auto *cb{name.symbol->detailsIf<CommonBlockDetails>()}) {
                  for (const auto &obj : cb->objects()) {
                    if (FindEquivalenceSet(*obj)) {
                      context_.Say(name.source,
                          "A variable in a %s directive cannot appear in an EQUIVALENCE statement (variable '%s' from common block '/%s/')"_err_en_US,
                          ContextDirectiveAsFortran(), obj->name(),
                          name.symbol->name());
                    }
                  }
                }
              }
            },
        },
        ompObject.u);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPThreadprivate &c) {
  const auto &dir{std::get<parser::Verbatim>(c.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_threadprivate);
}

void OmpStructureChecker::Leave(const parser::OpenMPThreadprivate &c) {
  const auto &dir{std::get<parser::Verbatim>(c.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(c.t)};
  CheckSymbolNames(dir.source, objectList);
  CheckVarIsNotPartOfAnotherVar(dir.source, objectList);
  CheckThreadprivateOrDeclareTargetVar(objectList);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareSimdConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_declare_simd);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareSimdConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpDeclareVariantDirective &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_declare_variant);
}

void OmpStructureChecker::Leave(const parser::OmpDeclareVariantDirective &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDepobjConstruct &x) {
  const auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  PushContextAndClauseSets(dirName.source, llvm::omp::Directive::OMPD_depobj);
  unsigned version{context_.langOptions().OpenMPVersion};

  const parser::OmpArgumentList &arguments{x.v.Arguments()};
  const parser::OmpClauseList &clauses{x.v.Clauses()};

  // Ref: [6.0:505-506]

  if (version < 60) {
    if (arguments.v.size() != 1) {
      parser::CharBlock source(
          arguments.v.empty() ? dirName.source : arguments.source);
      context_.Say(
          source, "The DEPOBJ directive requires a single argument"_err_en_US);
    }
  }
  if (clauses.v.size() != 1) {
    context_.Say(
        x.source, "The DEPOBJ construct requires a single clause"_err_en_US);
    return;
  }

  auto &clause{clauses.v.front()};

  if (version >= 60 && arguments.v.empty()) {
    context_.Say(x.source,
        "DEPOBJ syntax with no argument is not handled yet"_err_en_US);
    return;
  }

  // [5.2:73:27-28]
  // If the destroy clause appears on a depobj construct, destroy-var must
  // refer to the same depend object as the depobj argument of the construct.
  if (clause.Id() == llvm::omp::Clause::OMPC_destroy) {
    auto getObjSymbol{[&](const parser::OmpObject &obj) {
      return common::visit(
          [&](auto &&s) { return GetLastName(s).symbol; }, obj.u);
    }};
    auto getArgSymbol{[&](const parser::OmpArgument &arg) {
      if (auto *locator{std::get_if<parser::OmpLocator>(&arg.u)}) {
        if (auto *object{std::get_if<parser::OmpObject>(&locator->u)}) {
          return getObjSymbol(*object);
        }
      }
      return static_cast<Symbol *>(nullptr);
    }};

    auto &wrapper{std::get<parser::OmpClause::Destroy>(clause.u)};
    if (const std::optional<parser::OmpDestroyClause> &destroy{wrapper.v}) {
      const Symbol *constrSym{getArgSymbol(arguments.v.front())};
      const Symbol *clauseSym{getObjSymbol(destroy->v)};
      assert(constrSym && "Unresolved depobj construct symbol");
      assert(clauseSym && "Unresolved destroy symbol on depobj construct");
      if (constrSym != clauseSym) {
        context_.Say(x.source,
            "The DESTROY clause must refer to the same object as the "
            "DEPOBJ construct"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDepobjConstruct &x) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPRequiresConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_requires);
}

void OmpStructureChecker::Leave(const parser::OpenMPRequiresConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::CheckAlignValue(const parser::OmpClause &clause) {
  if (auto *align{std::get_if<parser::OmpClause::Align>(&clause.u)}) {
    if (const auto &v{GetIntValue(align->v)}; !v || *v <= 0) {
      context_.Say(clause.source,
          "The alignment value should be a constant positive integer"_err_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeAllocate &x) {
  isPredefinedAllocator = true;
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(objectList, currSymbols);
  for (auto &[symbol, source] : currSymbols) {
    if (IsPointer(*symbol)) {
      context_.Say(source,
          "List item '%s' in ALLOCATE directive must not have POINTER "
          "attribute"_err_en_US,
          source.ToString());
    }
    if (IsDummy(*symbol)) {
      context_.Say(source,
          "List item '%s' in ALLOCATE directive must not be a dummy "
          "argument"_err_en_US,
          source.ToString());
    }
    if (symbol->GetUltimate().has<AssocEntityDetails>()) {
      context_.Say(source,
          "List item '%s' in ALLOCATE directive must not be an associate "
          "name"_err_en_US,
          source.ToString());
    }
  }
  for (const auto &clause : clauseList.v) {
    CheckAlignValue(clause);
  }
  CheckVarIsNotPartOfAnotherVar(dir.source, objectList);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeAllocate &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(x.t)};
  CheckPredefinedAllocatorRestriction(dir.source, objectList);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpClause::Allocator &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_allocator);
  // Note: Predefined allocators are stored in ScalarExpr as numbers
  //   whereas custom allocators are stored as strings, so if the ScalarExpr
  //   actually has an int value, then it must be a predefined allocator
  isPredefinedAllocator = GetIntValue(x.v).has_value();
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_allocator, x.v);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Allocate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_allocate);
  if (OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_allocate, GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    if (auto *align{
            OmpGetUniqueModifier<parser::OmpAlignModifier>(modifiers)}) {
      if (const auto &v{GetIntValue(align->v)}; !v || *v <= 0) {
        context_.Say(OmpGetModifierSource(modifiers, align),
            "The alignment value should be a constant positive integer"_err_en_US);
      }
    }
    // The simple and complex modifiers have the same structure. They only
    // differ in their syntax.
    if (auto *alloc{OmpGetUniqueModifier<parser::OmpAllocatorComplexModifier>(
            modifiers)}) {
      isPredefinedAllocator = GetIntValue(alloc->v).has_value();
    }
    if (auto *alloc{OmpGetUniqueModifier<parser::OmpAllocatorSimpleModifier>(
            modifiers)}) {
      isPredefinedAllocator = GetIntValue(alloc->v).has_value();
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpDeclareTargetWithClause &x) {
  SetClauseSets(llvm::omp::Directive::OMPD_declare_target);
}

void OmpStructureChecker::Leave(const parser::OmpDeclareTargetWithClause &x) {
  if (x.v.v.size() > 0) {
    const parser::OmpClause *enterClause =
        FindClause(llvm::omp::Clause::OMPC_enter);
    const parser::OmpClause *toClause = FindClause(llvm::omp::Clause::OMPC_to);
    const parser::OmpClause *linkClause =
        FindClause(llvm::omp::Clause::OMPC_link);
    if (!enterClause && !toClause && !linkClause) {
      context_.Say(x.source,
          "If the DECLARE TARGET directive has a clause, it must contain at least one ENTER clause or LINK clause"_err_en_US);
    }
    unsigned version{context_.langOptions().OpenMPVersion};
    if (toClause && version >= 52) {
      context_.Warn(common::UsageWarning::OpenMPUsage, toClause->source,
          "The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead."_warn_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareMapperConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_declare_mapper);
  const auto &spec{std::get<parser::OmpMapperSpecifier>(x.t)};
  const auto &type = std::get<parser::TypeSpec>(spec.t);
  if (!std::get_if<parser::DerivedTypeSpec>(&type.u)) {
    context_.Say(dir.source, "Type is not a derived type"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareMapperConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPDeclareReductionConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_declare_reduction);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPDeclareReductionConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_declare_target);
}

void OmpStructureChecker::Enter(const parser::OmpDeclareTargetWithList &x) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(x.v, symbols);
  for (auto &[symbol, source] : symbols) {
    const GenericDetails *genericDetails = symbol->detailsIf<GenericDetails>();
    if (genericDetails) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be a generic name."_err_en_US,
          symbol->name());
      genericDetails->specific();
    }
    if (IsProcedurePointer(*symbol)) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be a procedure pointer."_err_en_US,
          symbol->name());
    }
    const SubprogramDetails *entryDetails =
        symbol->detailsIf<SubprogramDetails>();
    if (entryDetails && entryDetails->entryScope()) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be an entry name."_err_en_US,
          symbol->name());
    }
    if (IsStmtFunction(*symbol)) {
      context_.Say(source,
          "The procedure '%s' in DECLARE TARGET construct cannot be a statement function."_err_en_US,
          symbol->name());
    }
  }
}

void OmpStructureChecker::CheckSymbolNames(
    const parser::CharBlock &source, const parser::OmpObjectList &objList) {
  for (const auto &ompObject : objList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
                if (!name->symbol) {
                  context_.Say(source,
                      "The given %s directive clause has an invalid argument"_err_en_US,
                      ContextDirectiveAsFortran());
                }
              }
            },
            [&](const parser::Name &name) {
              if (!name.symbol) {
                context_.Say(source,
                    "The given %s directive clause has an invalid argument"_err_en_US,
                    ContextDirectiveAsFortran());
              }
            },
        },
        ompObject.u);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
  // Handle both forms of DECLARE TARGET.
  // - Extended list: It behaves as if there was an ENTER/TO clause with the
  //   list of objects as argument. It accepts no explicit clauses.
  // - With clauses.
  if (const auto *objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
    deviceConstructFound_ = true;
    CheckSymbolNames(dir.source, *objectList);
    CheckVarIsNotPartOfAnotherVar(dir.source, *objectList);
    CheckThreadprivateOrDeclareTargetVar(*objectList);
  } else if (const auto *clauseList{
                 parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
    bool toClauseFound{false}, deviceTypeClauseFound{false},
        enterClauseFound{false};
    for (const auto &clause : clauseList->v) {
      common::visit(
          common::visitors{
              [&](const parser::OmpClause::To &toClause) {
                toClauseFound = true;
                auto &objList{std::get<parser::OmpObjectList>(toClause.v.t)};
                CheckSymbolNames(dir.source, objList);
                CheckVarIsNotPartOfAnotherVar(dir.source, objList);
                CheckThreadprivateOrDeclareTargetVar(objList);
              },
              [&](const parser::OmpClause::Link &linkClause) {
                CheckSymbolNames(dir.source, linkClause.v);
                CheckVarIsNotPartOfAnotherVar(dir.source, linkClause.v);
                CheckThreadprivateOrDeclareTargetVar(linkClause.v);
              },
              [&](const parser::OmpClause::Enter &enterClause) {
                enterClauseFound = true;
                CheckSymbolNames(dir.source, enterClause.v);
                CheckVarIsNotPartOfAnotherVar(dir.source, enterClause.v);
                CheckThreadprivateOrDeclareTargetVar(enterClause.v);
              },
              [&](const parser::OmpClause::DeviceType &deviceTypeClause) {
                deviceTypeClauseFound = true;
                if (deviceTypeClause.v.v !=
                    parser::OmpDeviceTypeClause::DeviceTypeDescription::Host) {
                  // Function / subroutine explicitly marked as runnable by the
                  // target device.
                  deviceConstructFound_ = true;
                }
              },
              [&](const auto &) {},
          },
          clause.u);

      if ((toClauseFound || enterClauseFound) && !deviceTypeClauseFound) {
        deviceConstructFound_ = true;
      }
    }
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpErrorDirective &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_error);
}

void OmpStructureChecker::Enter(const parser::OpenMPDispatchConstruct &x) {
  PushContextAndClauseSets(x.source, llvm::omp::Directive::OMPD_dispatch);
  const auto &block{std::get<parser::Block>(x.t)};
  if (block.empty() || block.size() > 1) {
    context_.Say(x.source,
        "The DISPATCH construct is empty or contains more than one statement"_err_en_US);
    return;
  }

  auto it{block.begin()};
  bool passChecks{false};
  if (const parser::AssignmentStmt *
      assignStmt{parser::Unwrap<parser::AssignmentStmt>(*it)}) {
    if (parser::Unwrap<parser::FunctionReference>(assignStmt->t)) {
      passChecks = true;
    }
  } else if (parser::Unwrap<parser::CallStmt>(*it)) {
    passChecks = true;
  }

  if (!passChecks) {
    context_.Say(x.source,
        "The DISPATCH construct does not contain a SUBROUTINE or FUNCTION"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDispatchConstruct &x) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Leave(const parser::OmpErrorDirective &x) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpClause::At &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_at);
  if (GetDirectiveNest(DeclarativeNest) > 0) {
    if (x.v.v == parser::OmpAtClause::ActionTime::Execution) {
      context_.Say(GetContext().clauseSource,
          "The ERROR directive with AT(EXECUTION) cannot appear in the specification part"_err_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPExecutableAllocate &x) {
  isPredefinedAllocator = true;
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<std::optional<parser::OmpObjectList>>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  for (const auto &clause : clauseList.v) {
    CheckAlignValue(clause);
  }
  if (objectList) {
    CheckVarIsNotPartOfAnotherVar(dir.source, *objectList);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPExecutableAllocate &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<std::optional<parser::OmpObjectList>>(x.t)};
  if (objectList)
    CheckPredefinedAllocatorRestriction(dir.source, *objectList);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPAllocatorsConstruct &x) {
  isPredefinedAllocator = true;
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocators);
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  for (const auto &clause : clauseList.v) {
    if (const auto *allocClause{
            parser::Unwrap<parser::OmpClause::Allocate>(clause)}) {
      CheckVarIsNotPartOfAnotherVar(
          dir.source, std::get<parser::OmpObjectList>(allocClause->v.t));
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPAllocatorsConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &clauseList{std::get<parser::OmpClauseList>(x.t)};
  for (const auto &clause : clauseList.v) {
    if (const auto *allocClause{
            std::get_if<parser::OmpClause::Allocate>(&clause.u)}) {
      CheckPredefinedAllocatorRestriction(
          dir.source, std::get<parser::OmpObjectList>(allocClause->v.t));
    }
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::CheckScan(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  if (x.v.Clauses().v.size() != 1) {
    context_.Say(x.source,
        "Exactly one of EXCLUSIVE or INCLUSIVE clause is expected"_err_en_US);
  }
  if (!CurrentDirectiveIsNested() ||
      !llvm::omp::scanParentAllowedSet.test(GetContextParent().directive)) {
    context_.Say(x.source,
        "Orphaned SCAN directives are prohibited; perhaps you forgot "
        "to enclose the directive in to a WORKSHARING LOOP, a WORKSHARING "
        "LOOP SIMD or a SIMD directive."_err_en_US);
  }
}

void OmpStructureChecker::CheckBarrierNesting(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  // A barrier region may not be `closely nested` inside a worksharing, loop,
  // task, taskloop, critical, ordered, atomic, or master region.
  // TODO:  Expand the check to include `LOOP` construct as well when it is
  // supported.
  if (IsCloselyNestedRegion(llvm::omp::nestedBarrierErrSet)) {
    context_.Say(parser::FindSourceLocation(x),
        "`BARRIER` region may not be closely nested inside of `WORKSHARING`, "
        "`LOOP`, `TASK`, `TASKLOOP`,"
        "`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region."_err_en_US);
  }
}

void OmpStructureChecker::ChecksOnOrderedAsStandalone() {
  if (FindClause(llvm::omp::Clause::OMPC_threads) ||
      FindClause(llvm::omp::Clause::OMPC_simd)) {
    context_.Say(GetContext().clauseSource,
        "THREADS and SIMD clauses are not allowed when ORDERED construct is a standalone construct with no ORDERED region"_err_en_US);
  }

  int dependSinkCount{0}, dependSourceCount{0};
  bool exclusiveShown{false}, duplicateSourceShown{false};

  auto visitDoacross{[&](const parser::OmpDoacross &doa,
                         const parser::CharBlock &src) {
    common::visit(
        common::visitors{
            [&](const parser::OmpDoacross::Source &) { dependSourceCount++; },
            [&](const parser::OmpDoacross::Sink &) { dependSinkCount++; }},
        doa.u);
    if (!exclusiveShown && dependSinkCount > 0 && dependSourceCount > 0) {
      exclusiveShown = true;
      context_.Say(src,
          "The SINK and SOURCE dependence types are mutually exclusive"_err_en_US);
    }
    if (!duplicateSourceShown && dependSourceCount > 1) {
      duplicateSourceShown = true;
      context_.Say(src,
          "At most one SOURCE dependence type can appear on the ORDERED directive"_err_en_US);
    }
  }};

  // Visit the DEPEND and DOACROSS clauses.
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_depend)) {
    const auto &dependClause{std::get<parser::OmpClause::Depend>(clause->u)};
    if (auto *doAcross{std::get_if<parser::OmpDoacross>(&dependClause.v.u)}) {
      visitDoacross(*doAcross, clause->source);
    } else {
      context_.Say(clause->source,
          "Only SINK or SOURCE dependence types are allowed when ORDERED construct is a standalone construct with no ORDERED region"_err_en_US);
    }
  }
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_doacross)) {
    auto &doaClause{std::get<parser::OmpClause::Doacross>(clause->u)};
    visitDoacross(doaClause.v.v, clause->source);
  }

  bool isNestedInDoOrderedWithPara{false};
  if (CurrentDirectiveIsNested() &&
      llvm::omp::nestedOrderedDoAllowedSet.test(GetContextParent().directive)) {
    if (const auto *clause{
            FindClause(GetContextParent(), llvm::omp::Clause::OMPC_ordered)}) {
      const auto &orderedClause{
          std::get<parser::OmpClause::Ordered>(clause->u)};
      const auto orderedValue{GetIntValue(orderedClause.v)};
      if (orderedValue > 0) {
        isNestedInDoOrderedWithPara = true;
        CheckOrderedDependClause(orderedValue);
      }
    }
  }

  if (FindClause(llvm::omp::Clause::OMPC_depend) &&
      !isNestedInDoOrderedWithPara) {
    context_.Say(GetContext().clauseSource,
        "An ORDERED construct with the DEPEND clause must be closely nested "
        "in a worksharing-loop (or parallel worksharing-loop) construct with "
        "ORDERED clause with a parameter"_err_en_US);
  }
}

void OmpStructureChecker::CheckOrderedDependClause(
    std::optional<int64_t> orderedValue) {
  auto visitDoacross{[&](const parser::OmpDoacross &doa,
                         const parser::CharBlock &src) {
    if (auto *sinkVector{std::get_if<parser::OmpDoacross::Sink>(&doa.u)}) {
      int64_t numVar = sinkVector->v.v.size();
      if (orderedValue != numVar) {
        context_.Say(src,
            "The number of variables in the SINK iteration vector does not match the parameter specified in ORDERED clause"_err_en_US);
      }
    }
  }};
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_depend)) {
    auto &dependClause{std::get<parser::OmpClause::Depend>(clause->u)};
    if (auto *doAcross{std::get_if<parser::OmpDoacross>(&dependClause.v.u)}) {
      visitDoacross(*doAcross, clause->source);
    }
  }
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_doacross)) {
    auto &doaClause{std::get<parser::OmpClause::Doacross>(clause->u)};
    visitDoacross(doaClause.v.v, clause->source);
  }
}

void OmpStructureChecker::CheckTargetUpdate() {
  const parser::OmpClause *toWrapper{FindClause(llvm::omp::Clause::OMPC_to)};
  const parser::OmpClause *fromWrapper{
      FindClause(llvm::omp::Clause::OMPC_from)};
  if (!toWrapper && !fromWrapper) {
    context_.Say(GetContext().directiveSource,
        "At least one motion-clause (TO/FROM) must be specified on "
        "TARGET UPDATE construct."_err_en_US);
  }
  if (toWrapper && fromWrapper) {
    SymbolSourceMap toSymbols, fromSymbols;
    auto &fromClause{std::get<parser::OmpClause::From>(fromWrapper->u).v};
    auto &toClause{std::get<parser::OmpClause::To>(toWrapper->u).v};
    GetSymbolsInObjectList(
        std::get<parser::OmpObjectList>(fromClause.t), fromSymbols);
    GetSymbolsInObjectList(
        std::get<parser::OmpObjectList>(toClause.t), toSymbols);

    for (auto &[symbol, source] : toSymbols) {
      auto fromSymbol{fromSymbols.find(symbol)};
      if (fromSymbol != fromSymbols.end()) {
        context_.Say(source,
            "A list item ('%s') can only appear in a TO or FROM clause, but not in both."_err_en_US,
            symbol->name());
        context_.Say(source, "'%s' appears in the TO clause."_because_en_US,
            symbol->name());
        context_.Say(fromSymbol->second,
            "'%s' appears in the FROM clause."_because_en_US,
            fromSymbol->first->name());
      }
    }
  }
}

void OmpStructureChecker::CheckTaskDependenceType(
    const parser::OmpTaskDependenceType::Value &x) {
  // Common checks for task-dependence-type (DEPEND and UPDATE clauses).
  unsigned version{context_.langOptions().OpenMPVersion};
  unsigned since{0};

  switch (x) {
  case parser::OmpTaskDependenceType::Value::In:
  case parser::OmpTaskDependenceType::Value::Out:
  case parser::OmpTaskDependenceType::Value::Inout:
    break;
  case parser::OmpTaskDependenceType::Value::Mutexinoutset:
  case parser::OmpTaskDependenceType::Value::Depobj:
    since = 50;
    break;
  case parser::OmpTaskDependenceType::Value::Inoutset:
    since = 52;
    break;
  }

  if (version < since) {
    context_.Say(GetContext().clauseSource,
        "%s task dependence type is not supported in %s, %s"_warn_en_US,
        parser::ToUpperCaseLetters(
            parser::OmpTaskDependenceType::EnumToString(x)),
        ThisVersion(version), TryVersion(since));
  }
}

void OmpStructureChecker::CheckDependenceType(
    const parser::OmpDependenceType::Value &x) {
  // Common checks for dependence-type (DEPEND and UPDATE clauses).
  unsigned version{context_.langOptions().OpenMPVersion};
  unsigned deprecatedIn{~0u};

  switch (x) {
  case parser::OmpDependenceType::Value::Source:
  case parser::OmpDependenceType::Value::Sink:
    deprecatedIn = 52;
    break;
  }

  if (version >= deprecatedIn) {
    context_.Say(GetContext().clauseSource,
        "%s dependence type is deprecated in %s"_warn_en_US,
        parser::ToUpperCaseLetters(parser::OmpDependenceType::EnumToString(x)),
        ThisVersion(deprecatedIn));
  }
}

void OmpStructureChecker::Enter(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &dir{std::get<parser::OmpDirectiveName>(x.v.t)};
  PushContextAndClauseSets(dir.source, dir.v);
  switch (dir.v) {
  case llvm::omp::Directive::OMPD_barrier:
    CheckBarrierNesting(x);
    break;
  case llvm::omp::Directive::OMPD_scan:
    CheckScan(x);
    break;
  default:
    break;
  }
}

void OmpStructureChecker::Leave(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  switch (GetContext().directive) {
  case llvm::omp::Directive::OMPD_ordered:
    // [5.1] 2.19.9 Ordered Construct Restriction
    ChecksOnOrderedAsStandalone();
    break;
  case llvm::omp::Directive::OMPD_target_update:
    CheckTargetUpdate();
    break;
  default:
    break;
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPFlushConstruct &x) {
  const auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  PushContextAndClauseSets(dirName.source, llvm::omp::Directive::OMPD_flush);
}

void OmpStructureChecker::Leave(const parser::OpenMPFlushConstruct &x) {
  auto &flushList{std::get<std::optional<parser::OmpArgumentList>>(x.v.t)};

  auto isVariableListItemOrCommonBlock{[this](const Symbol &sym) {
    return IsVariableListItem(sym) ||
        sym.detailsIf<semantics::CommonBlockDetails>();
  }};

  if (flushList) {
    for (const parser::OmpArgument &arg : flushList->v) {
      if (auto *sym{GetArgumentSymbol(arg)};
          sym && !isVariableListItemOrCommonBlock(*sym)) {
        context_.Say(arg.source,
            "FLUSH argument must be a variable list item"_err_en_US);
      }
    }

    if (FindClause(llvm::omp::Clause::OMPC_acquire) ||
        FindClause(llvm::omp::Clause::OMPC_release) ||
        FindClause(llvm::omp::Clause::OMPC_acq_rel)) {
      context_.Say(flushList->source,
          "If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive"_err_en_US);
    }
  }

  unsigned version{context_.langOptions().OpenMPVersion};
  if (version >= 52) {
    using Flags = parser::OmpDirectiveSpecification::Flags;
    if (std::get<Flags>(x.v.t) == Flags::DeprecatedSyntax) {
      context_.Say(x.source,
          "The syntax \"FLUSH clause (object, ...)\" has been deprecated, use \"FLUSH(object, ...) clause\" instead"_warn_en_US);
    }
  }

  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCancelConstruct &x) {
  auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  auto &maybeClauses{std::get<std::optional<parser::OmpClauseList>>(x.v.t)};
  PushContextAndClauseSets(dirName.source, llvm::omp::Directive::OMPD_cancel);

  if (auto maybeConstruct{GetCancelType(
          llvm::omp::Directive::OMPD_cancel, x.source, maybeClauses)}) {
    CheckCancellationNest(dirName.source, *maybeConstruct);

    if (CurrentDirectiveIsNested()) {
      // nowait can be put on the end directive rather than the start directive
      // so we need to check both
      auto getParentClauses{[&]() {
        const DirectiveContext &parent{GetContextParent()};
        return llvm::concat<const llvm::omp::Clause>(
            parent.actualClauses, parent.endDirectiveClauses);
      }};

      if (llvm::omp::nestedCancelDoAllowedSet.test(*maybeConstruct)) {
        for (llvm::omp::Clause clause : getParentClauses()) {
          if (clause == llvm::omp::Clause::OMPC_nowait) {
            context_.Say(dirName.source,
                "The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause"_err_en_US);
          }
          if (clause == llvm::omp::Clause::OMPC_ordered) {
            context_.Say(dirName.source,
                "The CANCEL construct cannot be nested inside of a worksharing construct with the ORDERED clause"_err_en_US);
          }
        }
      } else if (llvm::omp::nestedCancelSectionsAllowedSet.test(
                     *maybeConstruct)) {
        for (llvm::omp::Clause clause : getParentClauses()) {
          if (clause == llvm::omp::Clause::OMPC_nowait) {
            context_.Say(dirName.source,
                "The CANCEL construct cannot be nested inside of a worksharing construct with the NOWAIT clause"_err_en_US);
          }
        }
      }
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPCancelConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCriticalConstruct &x) {
  const auto &dir{std::get<parser::OmpCriticalDirective>(x.t)};
  const auto &dirSource{std::get<parser::Verbatim>(dir.t).source};
  const auto &endDir{std::get<parser::OmpEndCriticalDirective>(x.t)};
  PushContextAndClauseSets(dirSource, llvm::omp::Directive::OMPD_critical);
  const auto &block{std::get<parser::Block>(x.t)};
  CheckNoBranching(block, llvm::omp::Directive::OMPD_critical, dir.source);
  const auto &dirName{std::get<std::optional<parser::Name>>(dir.t)};
  const auto &endDirName{std::get<std::optional<parser::Name>>(endDir.t)};
  const auto &ompClause{std::get<parser::OmpClauseList>(dir.t)};
  if (dirName && endDirName &&
      dirName->ToString().compare(endDirName->ToString())) {
    context_
        .Say(endDirName->source,
            parser::MessageFormattedText{
                "CRITICAL directive names do not match"_err_en_US})
        .Attach(dirName->source, "should be "_en_US);
  } else if (dirName && !endDirName) {
    context_
        .Say(dirName->source,
            parser::MessageFormattedText{
                "CRITICAL directive names do not match"_err_en_US})
        .Attach(dirName->source, "should be NULL"_en_US);
  } else if (!dirName && endDirName) {
    context_
        .Say(endDirName->source,
            parser::MessageFormattedText{
                "CRITICAL directive names do not match"_err_en_US})
        .Attach(endDirName->source, "should be NULL"_en_US);
  }
  if (!dirName && !ompClause.source.empty() &&
      ompClause.source.NULTerminatedToString() != "hint(omp_sync_hint_none)") {
    context_.Say(dir.source,
        parser::MessageFormattedText{
            "Hint clause other than omp_sync_hint_none cannot be specified for "
            "an unnamed CRITICAL directive"_err_en_US});
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPCriticalConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OmpClause::CancellationConstructType &x) {
  llvm::omp::Directive dir{GetContext().directive};
  auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};

  if (dir != llvm::omp::Directive::OMPD_cancel &&
      dir != llvm::omp::Directive::OMPD_cancellation_point) {
    // Do not call CheckAllowed/CheckAllowedClause, because in case of an error
    // it will print "CANCELLATION_CONSTRUCT_TYPE" as the clause name instead
    // of the contained construct name.
    context_.Say(dirName.source, "%s cannot follow %s"_err_en_US,
        parser::ToUpperCaseLetters(getDirectiveName(dirName.v)),
        parser::ToUpperCaseLetters(getDirectiveName(dir)));
  } else {
    switch (dirName.v) {
    case llvm::omp::Directive::OMPD_do:
    case llvm::omp::Directive::OMPD_parallel:
    case llvm::omp::Directive::OMPD_sections:
    case llvm::omp::Directive::OMPD_taskgroup:
      break;
    default:
      context_.Say(dirName.source,
          "%s is not a cancellable construct"_err_en_US,
          parser::ToUpperCaseLetters(getDirectiveName(dirName.v)));
      break;
    }
  }
}

void OmpStructureChecker::Enter(
    const parser::OpenMPCancellationPointConstruct &x) {
  auto &dirName{std::get<parser::OmpDirectiveName>(x.v.t)};
  auto &maybeClauses{std::get<std::optional<parser::OmpClauseList>>(x.v.t)};
  PushContextAndClauseSets(
      dirName.source, llvm::omp::Directive::OMPD_cancellation_point);

  if (auto maybeConstruct{
          GetCancelType(llvm::omp::Directive::OMPD_cancellation_point, x.source,
              maybeClauses)}) {
    CheckCancellationNest(dirName.source, *maybeConstruct);
  }
}

void OmpStructureChecker::Leave(
    const parser::OpenMPCancellationPointConstruct &) {
  dirContext_.pop_back();
}

std::optional<llvm::omp::Directive> OmpStructureChecker::GetCancelType(
    llvm::omp::Directive cancelDir, const parser::CharBlock &cancelSource,
    const std::optional<parser::OmpClauseList> &maybeClauses) {
  if (!maybeClauses) {
    return std::nullopt;
  }
  // Given clauses from CANCEL or CANCELLATION_POINT, identify the construct
  // to which the cancellation applies.
  std::optional<llvm::omp::Directive> cancelee;
  llvm::StringRef cancelName{getDirectiveName(cancelDir)};

  for (const parser::OmpClause &clause : maybeClauses->v) {
    using CancellationConstructType =
        parser::OmpClause::CancellationConstructType;
    if (auto *cctype{std::get_if<CancellationConstructType>(&clause.u)}) {
      if (cancelee) {
        context_.Say(cancelSource,
            "Multiple cancel-directive-name clauses are not allowed on the %s construct"_err_en_US,
            parser::ToUpperCaseLetters(cancelName.str()));
        return std::nullopt;
      }
      cancelee = std::get<parser::OmpDirectiveName>(cctype->v.t).v;
    }
  }

  if (!cancelee) {
    context_.Say(cancelSource,
        "Missing cancel-directive-name clause on the %s construct"_err_en_US,
        parser::ToUpperCaseLetters(cancelName.str()));
    return std::nullopt;
  }

  return cancelee;
}

void OmpStructureChecker::CheckCancellationNest(
    const parser::CharBlock &source, llvm::omp::Directive type) {
  llvm::StringRef typeName{getDirectiveName(type)};

  if (CurrentDirectiveIsNested()) {
    // If construct-type-clause is taskgroup, the cancellation construct must be
    // closely nested inside a task or a taskloop construct and the cancellation
    // region must be closely nested inside a taskgroup region. If
    // construct-type-clause is sections, the cancellation construct must be
    // closely nested inside a sections or section construct. Otherwise, the
    // cancellation construct must be closely nested inside an OpenMP construct
    // that matches the type specified in construct-type-clause of the
    // cancellation construct.
    bool eligibleCancellation{false};

    switch (type) {
    case llvm::omp::Directive::OMPD_taskgroup:
      if (llvm::omp::nestedCancelTaskgroupAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
        if (dirContext_.size() >= 3) {
          // Check if the cancellation region is closely nested inside a
          // taskgroup region when there are more than two levels of directives
          // in the directive context stack.
          if (GetContextParent().directive == llvm::omp::Directive::OMPD_task ||
              FindClauseParent(llvm::omp::Clause::OMPC_nogroup)) {
            for (int i = dirContext_.size() - 3; i >= 0; i--) {
              if (dirContext_[i].directive ==
                  llvm::omp::Directive::OMPD_taskgroup) {
                break;
              }
              if (llvm::omp::nestedCancelParallelAllowedSet.test(
                      dirContext_[i].directive)) {
                eligibleCancellation = false;
                break;
              }
            }
          }
        }
      }
      if (!eligibleCancellation) {
        context_.Say(source,
            "With %s clause, %s construct must be closely nested inside TASK or TASKLOOP construct and %s region must be closely nested inside TASKGROUP region"_err_en_US,
            parser::ToUpperCaseLetters(typeName.str()),
            ContextDirectiveAsFortran(), ContextDirectiveAsFortran());
      }
      return;
    case llvm::omp::Directive::OMPD_sections:
      if (llvm::omp::nestedCancelSectionsAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case llvm::omp::Directive::OMPD_do:
      if (llvm::omp::nestedCancelDoAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case llvm::omp::Directive::OMPD_parallel:
      if (llvm::omp::nestedCancelParallelAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    default:
      // This is diagnosed later.
      return;
    }
    if (!eligibleCancellation) {
      context_.Say(source,
          "With %s clause, %s construct cannot be closely nested inside %s construct"_err_en_US,
          parser::ToUpperCaseLetters(typeName.str()),
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(
              getDirectiveName(GetContextParent().directive).str()));
    }
  } else {
    // The cancellation directive cannot be orphaned.
    switch (type) {
    case llvm::omp::Directive::OMPD_taskgroup:
      context_.Say(source,
          "%s %s directive is not closely nested inside TASK or TASKLOOP"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    case llvm::omp::Directive::OMPD_sections:
      context_.Say(source,
          "%s %s directive is not closely nested inside SECTION or SECTIONS"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    case llvm::omp::Directive::OMPD_do:
      context_.Say(source,
          "%s %s directive is not closely nested inside the construct that matches the DO clause type"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    case llvm::omp::Directive::OMPD_parallel:
      context_.Say(source,
          "%s %s directive is not closely nested inside the construct that matches the PARALLEL clause type"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(typeName.str()));
      break;
    default:
      // This is diagnosed later.
      return;
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpEndBlockDirective &x) {
  const auto &dir{std::get<parser::OmpBlockDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  case llvm::omp::Directive::OMPD_scope:
    PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_end_scope);
    break;
  // 2.7.3 end-single-clause -> copyprivate-clause |
  //                            nowait-clause
  case llvm::omp::Directive::OMPD_single:
    PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_end_single);
    break;
  // 2.7.4 end-workshare -> END WORKSHARE [nowait-clause]
  case llvm::omp::Directive::OMPD_workshare:
    PushContextAndClauseSets(
        dir.source, llvm::omp::Directive::OMPD_end_workshare);
    break;
  default:
    // no clauses are allowed
    break;
  }
}

// TODO: Verify the popping of dirContext requirement after nowait
// implementation, as there is an implicit barrier at the end of the worksharing
// constructs unless a nowait clause is specified. Only OMPD_end_single and
// end_workshareare popped as they are pushed while entering the
// EndBlockDirective.
void OmpStructureChecker::Leave(const parser::OmpEndBlockDirective &x) {
  if ((GetContext().directive == llvm::omp::Directive::OMPD_end_scope) ||
      (GetContext().directive == llvm::omp::Directive::OMPD_end_single) ||
      (GetContext().directive == llvm::omp::Directive::OMPD_end_workshare)) {
    dirContext_.pop_back();
  }
}

/// parser::Block is a list of executable constructs, parser::BlockConstruct
/// is Fortran's BLOCK/ENDBLOCK construct.
/// Strip the outermost BlockConstructs, return the reference to the Block
/// in the executable part of the innermost of the stripped constructs.
/// Specifically, if the given `block` has a single entry (it's a list), and
/// the entry is a BlockConstruct, get the Block contained within. Repeat
/// this step as many times as possible.
static const parser::Block &GetInnermostExecPart(const parser::Block &block) {
  const parser::Block *iter{&block};
  while (iter->size() == 1) {
    const parser::ExecutionPartConstruct &ep{iter->front()};
    if (auto *exec{std::get_if<parser::ExecutableConstruct>(&ep.u)}) {
      using BlockConstruct = common::Indirection<parser::BlockConstruct>;
      if (auto *bc{std::get_if<BlockConstruct>(&exec->u)}) {
        iter = &std::get<parser::Block>(bc->value().t);
        continue;
      }
    }
    break;
  }
  return *iter;
}

// There is no consistent way to get the source of a given ActionStmt, so
// extract the source information from Statement<ActionStmt> when we can,
// and keep it around for error reporting in further analyses.
struct SourcedActionStmt {
  const parser::ActionStmt *stmt{nullptr};
  parser::CharBlock source;

  operator bool() const { return stmt != nullptr; }
};

struct AnalyzedCondStmt {
  SomeExpr cond{evaluate::NullPointer{}}; // Default ctor is deleted
  parser::CharBlock source;
  SourcedActionStmt ift, iff;
};

static SourcedActionStmt GetActionStmt(
    const parser::ExecutionPartConstruct *x) {
  if (x == nullptr) {
    return SourcedActionStmt{};
  }
  if (auto *exec{std::get_if<parser::ExecutableConstruct>(&x->u)}) {
    using ActionStmt = parser::Statement<parser::ActionStmt>;
    if (auto *stmt{std::get_if<ActionStmt>(&exec->u)}) {
      return SourcedActionStmt{&stmt->statement, stmt->source};
    }
  }
  return SourcedActionStmt{};
}

static SourcedActionStmt GetActionStmt(const parser::Block &block) {
  if (block.size() == 1) {
    return GetActionStmt(&block.front());
  }
  return SourcedActionStmt{};
}

// Compute the `evaluate::Assignment` from parser::ActionStmt. The assumption
// is that the ActionStmt will be either an assignment or a pointer-assignment,
// otherwise return std::nullopt.
// Note: This function can return std::nullopt on [Pointer]AssignmentStmt where
// the "typedAssignment" is unset. This can happen if there are semantic errors
// in the purported assignment.
static std::optional<evaluate::Assignment> GetEvaluateAssignment(
    const parser::ActionStmt *x) {
  if (x == nullptr) {
    return std::nullopt;
  }

  using AssignmentStmt = common::Indirection<parser::AssignmentStmt>;
  using PointerAssignmentStmt =
      common::Indirection<parser::PointerAssignmentStmt>;
  using TypedAssignment = parser::AssignmentStmt::TypedAssignment;

  return common::visit(
      [](auto &&s) -> std::optional<evaluate::Assignment> {
        using BareS = llvm::remove_cvref_t<decltype(s)>;
        if constexpr (std::is_same_v<BareS, AssignmentStmt> ||
            std::is_same_v<BareS, PointerAssignmentStmt>) {
          const TypedAssignment &typed{s.value().typedAssignment};
          // ForwardOwningPointer                 typedAssignment
          // `- GenericAssignmentWrapper          ^.get()
          //    `- std::optional<Assignment>      ^->v
          return typed.get()->v;
        } else {
          return std::nullopt;
        }
      },
      x->u);
}

// Check if the ActionStmt is actually a [Pointer]AssignmentStmt. This is
// to separate cases where the source has something that looks like an
// assignment, but is semantically wrong (diagnosed by general semantic
// checks), and where the source has some other statement (which we want
// to report as "should be an assignment").
static bool IsAssignment(const parser::ActionStmt *x) {
  if (x == nullptr) {
    return false;
  }

  using AssignmentStmt = common::Indirection<parser::AssignmentStmt>;
  using PointerAssignmentStmt =
      common::Indirection<parser::PointerAssignmentStmt>;

  return common::visit(
      [](auto &&s) -> bool {
        using BareS = llvm::remove_cvref_t<decltype(s)>;
        return std::is_same_v<BareS, AssignmentStmt> ||
            std::is_same_v<BareS, PointerAssignmentStmt>;
      },
      x->u);
}

static std::optional<AnalyzedCondStmt> AnalyzeConditionalStmt(
    const parser::ExecutionPartConstruct *x) {
  if (x == nullptr) {
    return std::nullopt;
  }

  // Extract the evaluate::Expr from ScalarLogicalExpr.
  auto getFromLogical{[](const parser::ScalarLogicalExpr &logical) {
    // ScalarLogicalExpr is Scalar<Logical<common::Indirection<Expr>>>
    const parser::Expr &expr{logical.thing.thing.value()};
    return GetEvaluateExpr(expr);
  }};

  // Recognize either
  // ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> IfStmt, or
  // ExecutionPartConstruct -> ExecutableConstruct -> IfConstruct.

  if (auto &&action{GetActionStmt(x)}) {
    if (auto *ifs{std::get_if<common::Indirection<parser::IfStmt>>(
            &action.stmt->u)}) {
      const parser::IfStmt &s{ifs->value()};
      auto &&maybeCond{
          getFromLogical(std::get<parser::ScalarLogicalExpr>(s.t))};
      auto &thenStmt{
          std::get<parser::UnlabeledStatement<parser::ActionStmt>>(s.t)};
      if (maybeCond) {
        return AnalyzedCondStmt{std::move(*maybeCond), action.source,
            SourcedActionStmt{&thenStmt.statement, thenStmt.source},
            SourcedActionStmt{}};
      }
    }
    return std::nullopt;
  }

  if (auto *exec{std::get_if<parser::ExecutableConstruct>(&x->u)}) {
    if (auto *ifc{
            std::get_if<common::Indirection<parser::IfConstruct>>(&exec->u)}) {
      using ElseBlock = parser::IfConstruct::ElseBlock;
      using ElseIfBlock = parser::IfConstruct::ElseIfBlock;
      const parser::IfConstruct &s{ifc->value()};

      if (!std::get<std::list<ElseIfBlock>>(s.t).empty()) {
        // Not expecting any else-if statements.
        return std::nullopt;
      }
      auto &stmt{std::get<parser::Statement<parser::IfThenStmt>>(s.t)};
      auto &&maybeCond{getFromLogical(
          std::get<parser::ScalarLogicalExpr>(stmt.statement.t))};
      if (!maybeCond) {
        return std::nullopt;
      }

      if (auto &maybeElse{std::get<std::optional<ElseBlock>>(s.t)}) {
        AnalyzedCondStmt result{std::move(*maybeCond), stmt.source,
            GetActionStmt(std::get<parser::Block>(s.t)),
            GetActionStmt(std::get<parser::Block>(maybeElse->t))};
        if (result.ift.stmt && result.iff.stmt) {
          return result;
        }
      } else {
        AnalyzedCondStmt result{std::move(*maybeCond), stmt.source,
            GetActionStmt(std::get<parser::Block>(s.t)), SourcedActionStmt{}};
        if (result.ift.stmt) {
          return result;
        }
      }
    }
    return std::nullopt;
  }

  return std::nullopt;
}

static std::pair<parser::CharBlock, parser::CharBlock> SplitAssignmentSource(
    parser::CharBlock source) {
  // Find => in the range, if not found, find = that is not a part of
  // <=, >=, ==, or /=.
  auto trim{[](std::string_view v) {
    const char *begin{v.data()};
    const char *end{begin + v.size()};
    while (*begin == ' ' && begin != end) {
      ++begin;
    }
    while (begin != end && end[-1] == ' ') {
      --end;
    }
    assert(begin != end && "Source should not be empty");
    return parser::CharBlock(begin, end - begin);
  }};

  std::string_view sv(source.begin(), source.size());

  if (auto where{sv.find("=>")}; where != sv.npos) {
    std::string_view lhs(sv.data(), where);
    std::string_view rhs(sv.data() + where + 2, sv.size() - where - 2);
    return std::make_pair(trim(lhs), trim(rhs));
  }

  // Go backwards, since all the exclusions above end with a '='.
  for (size_t next{source.size()}; next > 1; --next) {
    if (sv[next - 1] == '=' && !llvm::is_contained("<>=/", sv[next - 2])) {
      std::string_view lhs(sv.data(), next - 1);
      std::string_view rhs(sv.data() + next, sv.size() - next);
      return std::make_pair(trim(lhs), trim(rhs));
    }
  }
  llvm_unreachable("Could not find assignment operator");
}

namespace atomic {

struct DesignatorCollector : public evaluate::Traverse<DesignatorCollector,
                                 std::vector<SomeExpr>, false> {
  using Result = std::vector<SomeExpr>;
  using Base = evaluate::Traverse<DesignatorCollector, Result, false>;
  DesignatorCollector() : Base(*this) {}

  Result Default() const { return {}; }

  using Base::operator();

  template <typename T> //
  Result operator()(const evaluate::Designator<T> &x) const {
    // Once in a designator, don't traverse it any further (i.e. only
    // collect top-level designators).
    auto copy{x};
    return Result{AsGenericExpr(std::move(copy))};
  }

  template <typename... Rs> //
  Result Combine(Result &&result, Rs &&...results) const {
    Result v(std::move(result));
    auto moveAppend{[](auto &accum, auto &&other) {
      for (auto &&s : other) {
        accum.push_back(std::move(s));
      }
    }};
    (moveAppend(v, std::move(results)), ...);
    return v;
  }
};

struct VariableFinder : public evaluate::AnyTraverse<VariableFinder> {
  using Base = evaluate::AnyTraverse<VariableFinder>;
  VariableFinder(const SomeExpr &v) : Base(*this), var(v) {}

  using Base::operator();

  template <typename T>
  bool operator()(const evaluate::Designator<T> &x) const {
    auto copy{x};
    return evaluate::AsGenericExpr(std::move(copy)) == var;
  }

  template <typename T>
  bool operator()(const evaluate::FunctionRef<T> &x) const {
    auto copy{x};
    return evaluate::AsGenericExpr(std::move(copy)) == var;
  }

private:
  const SomeExpr &var;
};
} // namespace atomic

static bool IsPointerAssignment(const evaluate::Assignment &x) {
  return std::holds_alternative<evaluate::Assignment::BoundsSpec>(x.u) ||
      std::holds_alternative<evaluate::Assignment::BoundsRemapping>(x.u);
}

static bool IsCheckForAssociated(const SomeExpr &cond) {
  return GetTopLevelOperation(cond).first == operation::Operator::Associated;
}

static bool HasCommonDesignatorSymbols(
    const evaluate::SymbolVector &baseSyms, const SomeExpr &other) {
  // Compare the designators used in "other" with the designators whose
  // symbols are given in baseSyms.
  // This is a part of the check if these two expressions can access the same
  // storage: if the designators used in them are different enough, then they
  // will be assumed not to access the same memory.
  //
  // Consider an (array element) expression x%y(w%z), the corresponding symbol
  // vector will be {x, y, w, z} (i.e. the symbols for these names).
  // Check whether this exact sequence appears anywhere in any the symbol
  // vector for "other". This will be true for x(y) and x(y+1), so this is
  // not a sufficient condition, but can be used to eliminate candidates
  // before doing more exhaustive checks.
  //
  // If any of the symbols in this sequence are function names, assume that
  // there is no storage overlap, mostly because it would be impossible in
  // general to determine what storage the function will access.
  // Note: if f is pure, then two calls to f will access the same storage
  // when called with the same arguments. This check is not done yet.

  if (llvm::any_of(
          baseSyms, [](const SymbolRef &s) { return s->IsSubprogram(); })) {
    // If there is a function symbol in the chain then we can't infer much
    // about the accessed storage.
    return false;
  }

  auto isSubsequence{// Is u a subsequence of v.
      [](const evaluate::SymbolVector &u, const evaluate::SymbolVector &v) {
        size_t us{u.size()}, vs{v.size()};
        if (us > vs) {
          return false;
        }
        for (size_t off{0}; off != vs - us + 1; ++off) {
          bool same{true};
          for (size_t i{0}; i != us; ++i) {
            if (u[i] != v[off + i]) {
              same = false;
              break;
            }
          }
          if (same) {
            return true;
          }
        }
        return false;
      }};

  evaluate::SymbolVector otherSyms{evaluate::GetSymbolVector(other)};
  return isSubsequence(baseSyms, otherSyms);
}

static bool HasCommonTopLevelDesignators(
    const std::vector<SomeExpr> &baseDsgs, const SomeExpr &other) {
  // Compare designators directly as expressions. This will ensure
  // that x(y) and x(y+1) are not flagged as overlapping, whereas
  // the symbol vectors for both of these would be identical.
  std::vector<SomeExpr> otherDsgs{atomic::DesignatorCollector{}(other)};

  for (auto &s : baseDsgs) {
    if (llvm::any_of(otherDsgs, [&](auto &&t) { return s == t; })) {
      return true;
    }
  }
  return false;
}

static const SomeExpr *HasStorageOverlap(
    const SomeExpr &base, llvm::ArrayRef<SomeExpr> exprs) {
  evaluate::SymbolVector baseSyms{evaluate::GetSymbolVector(base)};
  std::vector<SomeExpr> baseDsgs{atomic::DesignatorCollector{}(base)};

  for (const SomeExpr &expr : exprs) {
    if (!HasCommonDesignatorSymbols(baseSyms, expr)) {
      continue;
    }
    if (HasCommonTopLevelDesignators(baseDsgs, expr)) {
      return &expr;
    }
  }
  return nullptr;
}

static bool IsMaybeAtomicWrite(const evaluate::Assignment &assign) {
  // This ignores function calls, so it will accept "f(x) = f(x) + 1"
  // for example.
  return HasStorageOverlap(assign.lhs, assign.rhs) == nullptr;
}

static bool IsSubexpressionOf(const SomeExpr &sub, const SomeExpr &super) {
  return atomic::VariableFinder{sub}(super);
}

static void SetExpr(parser::TypedExpr &expr, MaybeExpr value) {
  if (value) {
    expr.Reset(new evaluate::GenericExprWrapper(std::move(value)),
        evaluate::GenericExprWrapper::Deleter);
  }
}

static void SetAssignment(parser::AssignmentStmt::TypedAssignment &assign,
    std::optional<evaluate::Assignment> value) {
  if (value) {
    assign.Reset(new evaluate::GenericAssignmentWrapper(std::move(value)),
        evaluate::GenericAssignmentWrapper::Deleter);
  }
}

static parser::OpenMPAtomicConstruct::Analysis::Op MakeAtomicAnalysisOp(
    int what,
    const std::optional<evaluate::Assignment> &maybeAssign = std::nullopt) {
  parser::OpenMPAtomicConstruct::Analysis::Op operation;
  operation.what = what;
  SetAssignment(operation.assign, maybeAssign);
  return operation;
}

static parser::OpenMPAtomicConstruct::Analysis MakeAtomicAnalysis(
    const SomeExpr &atom, const MaybeExpr &cond,
    parser::OpenMPAtomicConstruct::Analysis::Op &&op0,
    parser::OpenMPAtomicConstruct::Analysis::Op &&op1) {
  // Defined in flang/include/flang/Parser/parse-tree.h
  //
  // struct Analysis {
  //   struct Kind {
  //     static constexpr int None = 0;
  //     static constexpr int Read = 1;
  //     static constexpr int Write = 2;
  //     static constexpr int Update = Read | Write;
  //     static constexpr int Action = 3; // Bits containing N, R, W, U
  //     static constexpr int IfTrue = 4;
  //     static constexpr int IfFalse = 8;
  //     static constexpr int Condition = 12; // Bits containing IfTrue, IfFalse
  //   };
  //   struct Op {
  //     int what;
  //     TypedAssignment assign;
  //   };
  //   TypedExpr atom, cond;
  //   Op op0, op1;
  // };

  parser::OpenMPAtomicConstruct::Analysis an;
  SetExpr(an.atom, atom);
  SetExpr(an.cond, cond);
  an.op0 = std::move(op0);
  an.op1 = std::move(op1);
  return an;
}

void OmpStructureChecker::CheckStorageOverlap(const SomeExpr &base,
    llvm::ArrayRef<evaluate::Expr<evaluate::SomeType>> exprs,
    parser::CharBlock source) {
  if (auto *expr{HasStorageOverlap(base, exprs)}) {
    context_.Say(source,
        "Within atomic operation %s and %s access the same storage"_warn_en_US,
        base.AsFortran(), expr->AsFortran());
  }
}

void OmpStructureChecker::ErrorShouldBeVariable(
    const MaybeExpr &expr, parser::CharBlock source) {
  if (expr) {
    context_.Say(source, "Atomic expression %s should be a variable"_err_en_US,
        expr->AsFortran());
  } else {
    context_.Say(source, "Atomic expression should be a variable"_err_en_US);
  }
}

/// Check if `expr` satisfies the following conditions for x and v:
///
/// [6.0:189:10-12]
/// - x and v (as applicable) are either scalar variables or
///   function references with scalar data pointer result of non-character
///   intrinsic type or variables that are non-polymorphic scalar pointers
///   and any length type parameter must be constant.
void OmpStructureChecker::CheckAtomicType(
    SymbolRef sym, parser::CharBlock source, std::string_view name) {
  const DeclTypeSpec *typeSpec{sym->GetType()};
  if (!typeSpec) {
    return;
  }

  if (!IsPointer(sym)) {
    using Category = DeclTypeSpec::Category;
    Category cat{typeSpec->category()};
    if (cat == Category::Character) {
      context_.Say(source,
          "Atomic variable %s cannot have CHARACTER type"_err_en_US, name);
    } else if (cat != Category::Numeric && cat != Category::Logical) {
      context_.Say(source,
          "Atomic variable %s should have an intrinsic type"_err_en_US, name);
    }
    return;
  }

  // Variable is a pointer.
  if (typeSpec->IsPolymorphic()) {
    context_.Say(source,
        "Atomic variable %s cannot be a pointer to a polymorphic type"_err_en_US,
        name);
    return;
  }

  // Go over all length parameters, if any, and check if they are
  // explicit.
  if (const DerivedTypeSpec *derived{typeSpec->AsDerived()}) {
    if (llvm::any_of(derived->parameters(), [](auto &&entry) {
          // "entry" is a map entry
          return entry.second.isLen() && !entry.second.isExplicit();
        })) {
      context_.Say(source,
          "Atomic variable %s is a pointer to a type with non-constant length parameter"_err_en_US,
          name);
    }
  }
}

void OmpStructureChecker::CheckAtomicVariable(
    const SomeExpr &atom, parser::CharBlock source) {
  if (atom.Rank() != 0) {
    context_.Say(source, "Atomic variable %s should be a scalar"_err_en_US,
        atom.AsFortran());
  }

  std::vector<SomeExpr> dsgs{atomic::DesignatorCollector{}(atom)};
  assert(dsgs.size() == 1 && "Should have a single top-level designator");
  evaluate::SymbolVector syms{evaluate::GetSymbolVector(dsgs.front())};

  CheckAtomicType(syms.back(), source, atom.AsFortran());

  if (IsAllocatable(syms.back()) && !IsArrayElement(atom)) {
    context_.Say(source, "Atomic variable %s cannot be ALLOCATABLE"_err_en_US,
        atom.AsFortran());
  }
}

std::pair<const parser::ExecutionPartConstruct *,
    const parser::ExecutionPartConstruct *>
OmpStructureChecker::CheckUpdateCapture(
    const parser::ExecutionPartConstruct *ec1,
    const parser::ExecutionPartConstruct *ec2, parser::CharBlock source) {
  // Decide which statement is the atomic update and which is the capture.
  //
  // The two allowed cases are:
  //   x = ...      atomic-var = ...
  //   ... = x      capture-var = atomic-var (with optional converts)
  // or
  //   ... = x      capture-var = atomic-var (with optional converts)
  //   x = ...      atomic-var = ...
  //
  // The case of 'a = b; b = a' is ambiguous, so pick the first one as capture
  // (which makes more sense, as it captures the original value of the atomic
  // variable).
  //
  // If the two statements don't fit these criteria, return a pair of default-
  // constructed values.
  using ReturnTy = std::pair<const parser::ExecutionPartConstruct *,
      const parser::ExecutionPartConstruct *>;

  SourcedActionStmt act1{GetActionStmt(ec1)};
  SourcedActionStmt act2{GetActionStmt(ec2)};
  auto maybeAssign1{GetEvaluateAssignment(act1.stmt)};
  auto maybeAssign2{GetEvaluateAssignment(act2.stmt)};
  if (!maybeAssign1 || !maybeAssign2) {
    if (!IsAssignment(act1.stmt) || !IsAssignment(act2.stmt)) {
      context_.Say(source,
          "ATOMIC UPDATE operation with CAPTURE should contain two assignments"_err_en_US);
    }
    return std::make_pair(nullptr, nullptr);
  }

  auto as1{*maybeAssign1}, as2{*maybeAssign2};

  auto isUpdateCapture{
      [](const evaluate::Assignment &u, const evaluate::Assignment &c) {
        return IsSameOrConvertOf(c.rhs, u.lhs);
      }};

  // Do some checks that narrow down the possible choices for the update
  // and the capture statements. This will help to emit better diagnostics.
  // 1. An assignment could be an update (cbu) if the left-hand side is a
  //    subexpression of the right-hand side.
  // 2. An assignment could be a capture (cbc) if the right-hand side is
  //    a variable (or a function ref), with potential type conversions.
  bool cbu1{IsSubexpressionOf(as1.lhs, as1.rhs)}; // Can as1 be an update?
  bool cbu2{IsSubexpressionOf(as2.lhs, as2.rhs)}; // Can as2 be an update?
  bool cbc1{IsVarOrFunctionRef(GetConvertInput(as1.rhs))}; // Can 1 be capture?
  bool cbc2{IsVarOrFunctionRef(GetConvertInput(as2.rhs))}; // Can 2 be capture?

  // We want to diagnose cases where both assignments cannot be an update,
  // or both cannot be a capture, as well as cases where either assignment
  // cannot be any of these two.
  //
  // If we organize these boolean values into a matrix
  //   |cbu1 cbu2|
  //   |cbc1 cbc2|
  // then we want to diagnose cases where the matrix has a zero (i.e. "false")
  // row or column, including the case where everything is zero. All these
  // cases correspond to the determinant of the matrix being 0, which suggests
  // that checking the det may be a convenient diagnostic check. There is only
  // one additional case where the det is 0, which is when the matrix is all 1
  // ("true"). The "all true" case represents the situation where both
  // assignments could be an update as well as a capture. On the other hand,
  // whenever det != 0, the roles of the update and the capture can be
  // unambiguously assigned to as1 and as2 [1].
  //
  // [1] This can be easily verified by hand: there are 10 2x2 matrices with
  // det = 0, leaving 6 cases where det != 0:
  //   0 1   0 1   1 0   1 0   1 1   1 1
  //   1 0   1 1   0 1   1 1   0 1   1 0
  // In each case the classification is unambiguous.

  //     |cbu1 cbu2|
  // det |cbc1 cbc2| = cbu1*cbc2 - cbu2*cbc1
  int det{int(cbu1) * int(cbc2) - int(cbu2) * int(cbc1)};

  auto errorCaptureShouldRead{[&](const parser::CharBlock &source,
                                  const std::string &expr) {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE the right-hand side of the capture assignment should read %s"_err_en_US,
        expr);
  }};

  auto errorNeitherWorks{[&]() {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE neither statement could be the update or the capture"_err_en_US);
  }};

  auto makeSelectionFromDet{[&](int det) -> ReturnTy {
    // If det != 0, then the checks unambiguously suggest a specific
    // categorization.
    // If det == 0, then this function should be called only if the
    // checks haven't ruled out any possibility, i.e. when both assigments
    // could still be either updates or captures.
    if (det > 0) {
      // as1 is update, as2 is capture
      if (isUpdateCapture(as1, as2)) {
        return std::make_pair(/*Update=*/ec1, /*Capture=*/ec2);
      } else {
        errorCaptureShouldRead(act2.source, as1.lhs.AsFortran());
        return std::make_pair(nullptr, nullptr);
      }
    } else if (det < 0) {
      // as2 is update, as1 is capture
      if (isUpdateCapture(as2, as1)) {
        return std::make_pair(/*Update=*/ec2, /*Capture=*/ec1);
      } else {
        errorCaptureShouldRead(act1.source, as2.lhs.AsFortran());
        return std::make_pair(nullptr, nullptr);
      }
    } else {
      bool updateFirst{isUpdateCapture(as1, as2)};
      bool captureFirst{isUpdateCapture(as2, as1)};
      if (updateFirst && captureFirst) {
        // If both assignment could be the update and both could be the
        // capture, emit a warning about the ambiguity.
        context_.Say(act1.source,
            "In ATOMIC UPDATE operation with CAPTURE either statement could be the update and the capture, assuming the first one is the capture statement"_warn_en_US);
        return std::make_pair(/*Update=*/ec2, /*Capture=*/ec1);
      }
      if (updateFirst != captureFirst) {
        const parser::ExecutionPartConstruct *upd{updateFirst ? ec1 : ec2};
        const parser::ExecutionPartConstruct *cap{captureFirst ? ec1 : ec2};
        return std::make_pair(upd, cap);
      }
      assert(!updateFirst && !captureFirst);
      errorNeitherWorks();
      return std::make_pair(nullptr, nullptr);
    }
  }};

  if (det != 0 || (cbu1 && cbu2 && cbc1 && cbc2)) {
    return makeSelectionFromDet(det);
  }
  assert(det == 0 && "Prior checks should have covered det != 0");

  // If neither of the statements is an RMW update, it could still be a
  // "write" update. Pretty much any assignment can be a write update, so
  // recompute det with cbu1 = cbu2 = true.
  if (int writeDet{int(cbc2) - int(cbc1)}; writeDet || (cbc1 && cbc2)) {
    return makeSelectionFromDet(writeDet);
  }

  // It's only errors from here on.

  if (!cbu1 && !cbu2 && !cbc1 && !cbc2) {
    errorNeitherWorks();
    return std::make_pair(nullptr, nullptr);
  }

  // The remaining cases are that
  // - no candidate for update, or for capture,
  // - one of the assigments cannot be anything.

  if (!cbu1 && !cbu2) {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE neither statement could be the update"_err_en_US);
    return std::make_pair(nullptr, nullptr);
  } else if (!cbc1 && !cbc2) {
    context_.Say(source,
        "In ATOMIC UPDATE operation with CAPTURE neither statement could be the capture"_err_en_US);
    return std::make_pair(nullptr, nullptr);
  }

  if ((!cbu1 && !cbc1) || (!cbu2 && !cbc2)) {
    auto &src = (!cbu1 && !cbc1) ? act1.source : act2.source;
    context_.Say(src,
        "In ATOMIC UPDATE operation with CAPTURE the statement could be neither the update nor the capture"_err_en_US);
    return std::make_pair(nullptr, nullptr);
  }

  // All cases should have been covered.
  llvm_unreachable("Unchecked condition");
}

void OmpStructureChecker::CheckAtomicCaptureAssignment(
    const evaluate::Assignment &capture, const SomeExpr &atom,
    parser::CharBlock source) {
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};
  const SomeExpr &cap{capture.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, rsrc);
  } else {
    CheckAtomicVariable(atom, rsrc);
    // This part should have been checked prior to calling this function.
    assert(*GetConvertInput(capture.rhs) == atom &&
        "This cannot be a capture assignment");
    CheckStorageOverlap(atom, {cap}, source);
  }
}

void OmpStructureChecker::CheckAtomicReadAssignment(
    const evaluate::Assignment &read, parser::CharBlock source) {
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};

  if (auto maybe{GetConvertInput(read.rhs)}) {
    const SomeExpr &atom{*maybe};

    if (!IsVarOrFunctionRef(atom)) {
      ErrorShouldBeVariable(atom, rsrc);
    } else {
      CheckAtomicVariable(atom, rsrc);
      CheckStorageOverlap(atom, {read.lhs}, source);
    }
  } else {
    ErrorShouldBeVariable(read.rhs, rsrc);
  }
}

void OmpStructureChecker::CheckAtomicWriteAssignment(
    const evaluate::Assignment &write, parser::CharBlock source) {
  // [6.0:190:13-15]
  // A write structured block is write-statement, a write statement that has
  // one of the following forms:
  //   x = expr
  //   x => expr
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};
  const SomeExpr &atom{write.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, rsrc);
  } else {
    CheckAtomicVariable(atom, lsrc);
    CheckStorageOverlap(atom, {write.rhs}, source);
  }
}

void OmpStructureChecker::CheckAtomicUpdateAssignment(
    const evaluate::Assignment &update, parser::CharBlock source) {
  // [6.0:191:1-7]
  // An update structured block is update-statement, an update statement
  // that has one of the following forms:
  //   x = x operator expr
  //   x = expr operator x
  //   x = intrinsic-procedure-name (x)
  //   x = intrinsic-procedure-name (x, expr-list)
  //   x = intrinsic-procedure-name (expr-list, x)
  auto [lsrc, rsrc]{SplitAssignmentSource(source)};
  const SomeExpr &atom{update.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, rsrc);
    // Skip other checks.
    return;
  }

  CheckAtomicVariable(atom, lsrc);

  std::pair<operation::Operator, std::vector<SomeExpr>> top{
      operation::Operator::Unknown, {}};
  if (auto &&maybeInput{GetConvertInput(update.rhs)}) {
    top = GetTopLevelOperation(*maybeInput);
  }
  switch (top.first) {
  case operation::Operator::Add:
  case operation::Operator::Sub:
  case operation::Operator::Mul:
  case operation::Operator::Div:
  case operation::Operator::And:
  case operation::Operator::Or:
  case operation::Operator::Eqv:
  case operation::Operator::Neqv:
  case operation::Operator::Min:
  case operation::Operator::Max:
  case operation::Operator::Identity:
    break;
  case operation::Operator::Call:
    context_.Say(source,
        "A call to this function is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  case operation::Operator::Convert:
    context_.Say(source,
        "An implicit or explicit type conversion is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  case operation::Operator::Intrinsic:
    context_.Say(source,
        "This intrinsic function is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  case operation::Operator::Constant:
  case operation::Operator::Unknown:
    context_.Say(
        source, "This is not a valid ATOMIC UPDATE operation"_err_en_US);
    return;
  default:
    assert(
        top.first != operation::Operator::Identity && "Handle this separately");
    context_.Say(source,
        "The %s operator is not a valid ATOMIC UPDATE operation"_err_en_US,
        operation::ToString(top.first));
    return;
  }
  // Check if `atom` occurs exactly once in the argument list.
  std::vector<SomeExpr> nonAtom;
  auto unique{[&]() { // -> iterator
    auto found{top.second.end()};
    for (auto i{top.second.begin()}, e{top.second.end()}; i != e; ++i) {
      if (IsSameOrConvertOf(*i, atom)) {
        if (found != top.second.end()) {
          return top.second.end();
        }
        found = i;
      } else {
        nonAtom.push_back(*i);
      }
    }
    return found;
  }()};

  if (unique == top.second.end()) {
    if (top.first == operation::Operator::Identity) {
      // This is "x = y".
      context_.Say(rsrc,
          "The atomic variable %s should appear as an argument in the update operation"_err_en_US,
          atom.AsFortran());
    } else {
      assert(top.first != operation::Operator::Identity &&
          "Handle this separately");
      context_.Say(rsrc,
          "The atomic variable %s should occur exactly once among the arguments of the top-level %s operator"_err_en_US,
          atom.AsFortran(), operation::ToString(top.first));
    }
  } else {
    CheckStorageOverlap(atom, nonAtom, source);
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdateAssignment(
    const SomeExpr &cond, parser::CharBlock condSource,
    const evaluate::Assignment &assign, parser::CharBlock assignSource) {
  auto [alsrc, arsrc]{SplitAssignmentSource(assignSource)};
  const SomeExpr &atom{assign.lhs};

  if (!IsVarOrFunctionRef(atom)) {
    ErrorShouldBeVariable(atom, arsrc);
    // Skip other checks.
    return;
  }

  CheckAtomicVariable(atom, alsrc);

  auto top{GetTopLevelOperation(cond)};
  // Missing arguments to operations would have been diagnosed by now.

  switch (top.first) {
  case operation::Operator::Associated:
    if (atom != top.second.front()) {
      context_.Say(assignSource,
          "The pointer argument to ASSOCIATED must be same as the target of the assignment"_err_en_US);
    }
    break;
  // x equalop e | e equalop x  (allowing "e equalop x" is an extension)
  case operation::Operator::Eq:
  case operation::Operator::Eqv:
  // x ordop expr | expr ordop x
  case operation::Operator::Lt:
  case operation::Operator::Gt: {
    const SomeExpr &arg0{top.second[0]};
    const SomeExpr &arg1{top.second[1]};
    if (IsSameOrConvertOf(arg0, atom)) {
      CheckStorageOverlap(atom, {arg1}, condSource);
    } else if (IsSameOrConvertOf(arg1, atom)) {
      CheckStorageOverlap(atom, {arg0}, condSource);
    } else {
      assert(top.first != operation::Operator::Identity &&
          "Handle this separately");
      context_.Say(assignSource,
          "An argument of the %s operator should be the target of the assignment"_err_en_US,
          operation::ToString(top.first));
    }
    break;
  }
  case operation::Operator::Identity:
  case operation::Operator::True:
  case operation::Operator::False:
    break;
  default:
    assert(
        top.first != operation::Operator::Identity && "Handle this separately");
    context_.Say(condSource,
        "The %s operator is not a valid condition for ATOMIC operation"_err_en_US,
        operation::ToString(top.first));
    break;
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdateStmt(
    const AnalyzedCondStmt &update, parser::CharBlock source) {
  // The condition/statements must be:
  // - cond: x equalop e      ift: x =  d     iff: -
  // - cond: x ordop expr     ift: x =  expr  iff: -  (+ commute ordop)
  // - cond: associated(x)    ift: x => expr  iff: -
  // - cond: associated(x, e) ift: x => expr  iff: -

  // The if-true statement must be present, and must be an assignment.
  auto maybeAssign{GetEvaluateAssignment(update.ift.stmt)};
  if (!maybeAssign) {
    if (update.ift.stmt && !IsAssignment(update.ift.stmt)) {
      context_.Say(update.ift.source,
          "In ATOMIC UPDATE COMPARE the update statement should be an assignment"_err_en_US);
    } else {
      context_.Say(
          source, "Invalid body of ATOMIC UPDATE COMPARE operation"_err_en_US);
    }
    return;
  }
  const evaluate::Assignment assign{*maybeAssign};
  const SomeExpr &atom{assign.lhs};

  CheckAtomicConditionalUpdateAssignment(
      update.cond, update.source, assign, update.ift.source);

  CheckStorageOverlap(atom, {assign.rhs}, update.ift.source);

  if (update.iff) {
    context_.Say(update.iff.source,
        "In ATOMIC UPDATE COMPARE the update statement should not have an ELSE branch"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicUpdateOnly(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  if (body.size() == 1) {
    SourcedActionStmt action{GetActionStmt(&body.front())};
    if (auto maybeUpdate{GetEvaluateAssignment(action.stmt)}) {
      const SomeExpr &atom{maybeUpdate->lhs};
      CheckAtomicUpdateAssignment(*maybeUpdate, action.source);

      using Analysis = parser::OpenMPAtomicConstruct::Analysis;
      x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
          MakeAtomicAnalysisOp(Analysis::Update, maybeUpdate),
          MakeAtomicAnalysisOp(Analysis::None));
    } else if (!IsAssignment(action.stmt)) {
      context_.Say(
          source, "ATOMIC UPDATE operation should be an assignment"_err_en_US);
    }
  } else {
    context_.Say(x.source,
        "ATOMIC UPDATE operation should have a single statement"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdate(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  // Allowable forms are (single-statement):
  // - if ...
  // - x = (... ? ... : x)
  // and two-statement:
  // - r = cond ; if (r) ...

  const parser::ExecutionPartConstruct *ust{nullptr}; // update
  const parser::ExecutionPartConstruct *cst{nullptr}; // condition

  if (body.size() == 1) {
    ust = &body.front();
  } else if (body.size() == 2) {
    cst = &body.front();
    ust = &body.back();
  } else {
    context_.Say(source,
        "ATOMIC UPDATE COMPARE operation should contain one or two statements"_err_en_US);
    return;
  }

  // Flang doesn't support conditional-expr yet, so all update statements
  // are if-statements.

  // IfStmt:        if (...) ...
  // IfConstruct:   if (...) then ... endif
  auto maybeUpdate{AnalyzeConditionalStmt(ust)};
  if (!maybeUpdate) {
    context_.Say(source,
        "In ATOMIC UPDATE COMPARE the update statement should be a conditional statement"_err_en_US);
    return;
  }

  AnalyzedCondStmt &update{*maybeUpdate};

  if (SourcedActionStmt action{GetActionStmt(cst)}) {
    // The "condition" statement must be `r = cond`.
    if (auto maybeCond{GetEvaluateAssignment(action.stmt)}) {
      if (maybeCond->lhs != update.cond) {
        context_.Say(update.source,
            "In ATOMIC UPDATE COMPARE the conditional statement must use %s as the condition"_err_en_US,
            maybeCond->lhs.AsFortran());
      } else {
        // If it's "r = ...; if (r) ..." then put the original condition
        // in `update`.
        update.cond = maybeCond->rhs;
      }
    } else {
      context_.Say(action.source,
          "In ATOMIC UPDATE COMPARE with two statements the first statement should compute the condition"_err_en_US);
    }
  }

  evaluate::Assignment assign{*GetEvaluateAssignment(update.ift.stmt)};

  CheckAtomicConditionalUpdateStmt(update, source);
  if (IsCheckForAssociated(update.cond)) {
    if (!IsPointerAssignment(assign)) {
      context_.Say(source,
          "The assignment should be a pointer-assignment when the condition is ASSOCIATED"_err_en_US);
    }
  } else {
    if (IsPointerAssignment(assign)) {
      context_.Say(source,
          "The assignment cannot be a pointer-assignment except when the condition is ASSOCIATED"_err_en_US);
    }
  }

  using Analysis = parser::OpenMPAtomicConstruct::Analysis;
  x.analysis = MakeAtomicAnalysis(assign.lhs, update.cond,
      MakeAtomicAnalysisOp(Analysis::Update | Analysis::IfTrue, assign),
      MakeAtomicAnalysisOp(Analysis::None));
}

void OmpStructureChecker::CheckAtomicUpdateCapture(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  if (body.size() != 2) {
    context_.Say(source,
        "ATOMIC UPDATE operation with CAPTURE should contain two statements"_err_en_US);
    return;
  }

  auto [uec, cec]{CheckUpdateCapture(&body.front(), &body.back(), source)};
  if (!uec || !cec) {
    // Diagnostics already emitted.
    return;
  }
  SourcedActionStmt uact{GetActionStmt(uec)};
  SourcedActionStmt cact{GetActionStmt(cec)};
  // The "dereferences" of std::optional are guaranteed to be valid after
  // CheckUpdateCapture.
  evaluate::Assignment update{*GetEvaluateAssignment(uact.stmt)};
  evaluate::Assignment capture{*GetEvaluateAssignment(cact.stmt)};

  const SomeExpr &atom{update.lhs};

  using Analysis = parser::OpenMPAtomicConstruct::Analysis;
  int action;

  if (IsMaybeAtomicWrite(update)) {
    action = Analysis::Write;
    CheckAtomicWriteAssignment(update, uact.source);
  } else {
    action = Analysis::Update;
    CheckAtomicUpdateAssignment(update, uact.source);
  }
  CheckAtomicCaptureAssignment(capture, atom, cact.source);

  if (IsPointerAssignment(update) != IsPointerAssignment(capture)) {
    context_.Say(cact.source,
        "The update and capture assignments should both be pointer-assignments or both be non-pointer-assignments"_err_en_US);
    return;
  }

  if (GetActionStmt(&body.front()).stmt == uact.stmt) {
    x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
        MakeAtomicAnalysisOp(action, update),
        MakeAtomicAnalysisOp(Analysis::Read, capture));
  } else {
    x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
        MakeAtomicAnalysisOp(Analysis::Read, capture),
        MakeAtomicAnalysisOp(action, update));
  }
}

void OmpStructureChecker::CheckAtomicConditionalUpdateCapture(
    const parser::OpenMPAtomicConstruct &x, const parser::Block &body,
    parser::CharBlock source) {
  // There are two different variants of this:
  // (1) conditional-update and capture separately:
  //     This form only allows single-statement updates, i.e. the update
  //     form "r = cond; if (r) ..." is not allowed.
  // (2) conditional-update combined with capture in a single statement:
  //     This form does allow the condition to be calculated separately,
  //     i.e. "r = cond; if (r) ...".
  // Regardless of what form it is, the actual update assignment is a
  // proper write, i.e. "x = d", where d does not depend on x.

  AnalyzedCondStmt update;
  SourcedActionStmt capture;
  bool captureAlways{true}, captureFirst{true};

  auto extractCapture{[&]() {
    capture = update.iff;
    captureAlways = false;
    update.iff = SourcedActionStmt{};
  }};

  auto classifyNonUpdate{[&](const SourcedActionStmt &action) {
    // The non-update statement is either "r = cond" or the capture.
    if (auto maybeAssign{GetEvaluateAssignment(action.stmt)}) {
      if (update.cond == maybeAssign->lhs) {
        // If this is "r = cond; if (r) ...", then update the condition.
        update.cond = maybeAssign->rhs;
        update.source = action.source;
        // In this form, the update and the capture are combined into
        // an IF-THEN-ELSE statement.
        extractCapture();
      } else {
        // Assume this is the capture-statement.
        capture = action;
      }
    }
  }};

  if (body.size() == 2) {
    // This could be
    // - capture; conditional-update (in any order), or
    // - r = cond; if (r) capture-update
    const parser::ExecutionPartConstruct *st1{&body.front()};
    const parser::ExecutionPartConstruct *st2{&body.back()};
    // In either case, the conditional statement can be analyzed by
    // AnalyzeConditionalStmt, whereas the other statement cannot.
    if (auto maybeUpdate1{AnalyzeConditionalStmt(st1)}) {
      update = *maybeUpdate1;
      classifyNonUpdate(GetActionStmt(st2));
      captureFirst = false;
    } else if (auto maybeUpdate2{AnalyzeConditionalStmt(st2)}) {
      update = *maybeUpdate2;
      classifyNonUpdate(GetActionStmt(st1));
    } else {
      // None of the statements are conditional, this rules out the
      // "r = cond; if (r) ..." and the "capture + conditional-update"
      // variants. This could still be capture + write (which is classified
      // as conditional-update-capture in the spec).
      auto [uec, cec]{CheckUpdateCapture(st1, st2, source)};
      if (!uec || !cec) {
        // Diagnostics already emitted.
        return;
      }
      SourcedActionStmt uact{GetActionStmt(uec)};
      SourcedActionStmt cact{GetActionStmt(cec)};
      update.ift = uact;
      capture = cact;
      if (uec == st1) {
        captureFirst = false;
      }
    }
  } else if (body.size() == 1) {
    if (auto maybeUpdate{AnalyzeConditionalStmt(&body.front())}) {
      update = *maybeUpdate;
      // This is the form with update and capture combined into an IF-THEN-ELSE
      // statement. The capture-statement is always the ELSE branch.
      extractCapture();
    } else {
      goto invalid;
    }
  } else {
    context_.Say(source,
        "ATOMIC UPDATE COMPARE CAPTURE operation should contain one or two statements"_err_en_US);
    return;
  invalid:
    context_.Say(source,
        "Invalid body of ATOMIC UPDATE COMPARE CAPTURE operation"_err_en_US);
    return;
  }

  // The update must have a form `x = d` or `x => d`.
  if (auto maybeWrite{GetEvaluateAssignment(update.ift.stmt)}) {
    const SomeExpr &atom{maybeWrite->lhs};
    CheckAtomicWriteAssignment(*maybeWrite, update.ift.source);
    if (auto maybeCapture{GetEvaluateAssignment(capture.stmt)}) {
      CheckAtomicCaptureAssignment(*maybeCapture, atom, capture.source);

      if (IsPointerAssignment(*maybeWrite) !=
          IsPointerAssignment(*maybeCapture)) {
        context_.Say(capture.source,
            "The update and capture assignments should both be pointer-assignments or both be non-pointer-assignments"_err_en_US);
        return;
      }
    } else {
      if (!IsAssignment(capture.stmt)) {
        context_.Say(capture.source,
            "In ATOMIC UPDATE COMPARE CAPTURE the capture statement should be an assignment"_err_en_US);
      }
      return;
    }
  } else {
    if (!IsAssignment(update.ift.stmt)) {
      context_.Say(update.ift.source,
          "In ATOMIC UPDATE COMPARE CAPTURE the update statement should be an assignment"_err_en_US);
    }
    return;
  }

  // update.iff should be empty here, the capture statement should be
  // stored in "capture".

  // Fill out the analysis in the AST node.
  using Analysis = parser::OpenMPAtomicConstruct::Analysis;
  bool condUnused{std::visit(
      [](auto &&s) {
        using BareS = llvm::remove_cvref_t<decltype(s)>;
        if constexpr (std::is_same_v<BareS, evaluate::NullPointer>) {
          return true;
        } else {
          return false;
        }
      },
      update.cond.u)};

  int updateWhen{!condUnused ? Analysis::IfTrue : 0};
  int captureWhen{!captureAlways ? Analysis::IfFalse : 0};

  evaluate::Assignment updAssign{*GetEvaluateAssignment(update.ift.stmt)};
  evaluate::Assignment capAssign{*GetEvaluateAssignment(capture.stmt)};

  if (captureFirst) {
    x.analysis = MakeAtomicAnalysis(updAssign.lhs, update.cond,
        MakeAtomicAnalysisOp(Analysis::Read | captureWhen, capAssign),
        MakeAtomicAnalysisOp(Analysis::Write | updateWhen, updAssign));
  } else {
    x.analysis = MakeAtomicAnalysis(updAssign.lhs, update.cond,
        MakeAtomicAnalysisOp(Analysis::Write | updateWhen, updAssign),
        MakeAtomicAnalysisOp(Analysis::Read | captureWhen, capAssign));
  }
}

void OmpStructureChecker::CheckAtomicRead(
    const parser::OpenMPAtomicConstruct &x) {
  // [6.0:190:5-7]
  // A read structured block is read-statement, a read statement that has one
  // of the following forms:
  //   v = x
  //   v => x
  auto &dirSpec{std::get<parser::OmpDirectiveSpecification>(x.t)};
  auto &block{std::get<parser::Block>(x.t)};

  // Read cannot be conditional or have a capture statement.
  if (x.IsCompare() || x.IsCapture()) {
    context_.Say(dirSpec.source,
        "ATOMIC READ cannot have COMPARE or CAPTURE clauses"_err_en_US);
    return;
  }

  const parser::Block &body{GetInnermostExecPart(block)};

  if (body.size() == 1) {
    SourcedActionStmt action{GetActionStmt(&body.front())};
    if (auto maybeRead{GetEvaluateAssignment(action.stmt)}) {
      CheckAtomicReadAssignment(*maybeRead, action.source);

      if (auto maybe{GetConvertInput(maybeRead->rhs)}) {
        const SomeExpr &atom{*maybe};
        using Analysis = parser::OpenMPAtomicConstruct::Analysis;
        x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
            MakeAtomicAnalysisOp(Analysis::Read, maybeRead),
            MakeAtomicAnalysisOp(Analysis::None));
      }
    } else if (!IsAssignment(action.stmt)) {
      context_.Say(
          x.source, "ATOMIC READ operation should be an assignment"_err_en_US);
    }
  } else {
    context_.Say(x.source,
        "ATOMIC READ operation should have a single statement"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicWrite(
    const parser::OpenMPAtomicConstruct &x) {
  auto &dirSpec{std::get<parser::OmpDirectiveSpecification>(x.t)};
  auto &block{std::get<parser::Block>(x.t)};

  // Write cannot be conditional or have a capture statement.
  if (x.IsCompare() || x.IsCapture()) {
    context_.Say(dirSpec.source,
        "ATOMIC WRITE cannot have COMPARE or CAPTURE clauses"_err_en_US);
    return;
  }

  const parser::Block &body{GetInnermostExecPart(block)};

  if (body.size() == 1) {
    SourcedActionStmt action{GetActionStmt(&body.front())};
    if (auto maybeWrite{GetEvaluateAssignment(action.stmt)}) {
      const SomeExpr &atom{maybeWrite->lhs};
      CheckAtomicWriteAssignment(*maybeWrite, action.source);

      using Analysis = parser::OpenMPAtomicConstruct::Analysis;
      x.analysis = MakeAtomicAnalysis(atom, std::nullopt,
          MakeAtomicAnalysisOp(Analysis::Write, maybeWrite),
          MakeAtomicAnalysisOp(Analysis::None));
    } else if (!IsAssignment(action.stmt)) {
      context_.Say(
          x.source, "ATOMIC WRITE operation should be an assignment"_err_en_US);
    }
  } else {
    context_.Say(x.source,
        "ATOMIC WRITE operation should have a single statement"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicUpdate(
    const parser::OpenMPAtomicConstruct &x) {
  auto &block{std::get<parser::Block>(x.t)};

  bool isConditional{x.IsCompare()};
  bool isCapture{x.IsCapture()};
  const parser::Block &body{GetInnermostExecPart(block)};

  if (isConditional && isCapture) {
    CheckAtomicConditionalUpdateCapture(x, body, x.source);
  } else if (isConditional) {
    CheckAtomicConditionalUpdate(x, body, x.source);
  } else if (isCapture) {
    CheckAtomicUpdateCapture(x, body, x.source);
  } else { // update-only
    CheckAtomicUpdateOnly(x, body, x.source);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPAtomicConstruct &x) {
  // All of the following groups have the "exclusive" property, i.e. at
  // most one clause from each group is allowed.
  // The exclusivity-checking code should eventually be unified for all
  // clauses, with clause groups defined in OMP.td.
  std::array atomic{llvm::omp::Clause::OMPC_read,
      llvm::omp::Clause::OMPC_update, llvm::omp::Clause::OMPC_write};
  std::array memoryOrder{llvm::omp::Clause::OMPC_acq_rel,
      llvm::omp::Clause::OMPC_acquire, llvm::omp::Clause::OMPC_relaxed,
      llvm::omp::Clause::OMPC_release, llvm::omp::Clause::OMPC_seq_cst};

  auto checkExclusive{[&](llvm::ArrayRef<llvm::omp::Clause> group,
                          std::string_view name,
                          const parser::OmpClauseList &clauses) {
    const parser::OmpClause *present{nullptr};
    for (const parser::OmpClause &clause : clauses.v) {
      llvm::omp::Clause id{clause.Id()};
      if (!llvm::is_contained(group, id)) {
        continue;
      }
      if (present == nullptr) {
        present = &clause;
        continue;
      } else if (id == present->Id()) {
        // Ignore repetitions of the same clause, those will be diagnosed
        // separately.
        continue;
      }
      parser::MessageFormattedText txt(
          "At most one clause from the '%s' group is allowed on ATOMIC construct"_err_en_US,
          name.data());
      parser::Message message(clause.source, txt);
      message.Attach(present->source,
          "Previous clause from this group provided here"_en_US);
      context_.Say(std::move(message));
      return;
    }
  }};

  auto &dirSpec{std::get<parser::OmpDirectiveSpecification>(x.t)};
  auto &dir{std::get<parser::OmpDirectiveName>(dirSpec.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_atomic);
  llvm::omp::Clause kind{x.GetKind()};

  checkExclusive(atomic, "atomic", dirSpec.Clauses());
  checkExclusive(memoryOrder, "memory-order", dirSpec.Clauses());

  switch (kind) {
  case llvm::omp::Clause::OMPC_read:
    CheckAtomicRead(x);
    break;
  case llvm::omp::Clause::OMPC_write:
    CheckAtomicWrite(x);
    break;
  case llvm::omp::Clause::OMPC_update:
    CheckAtomicUpdate(x);
    break;
  default:
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPAtomicConstruct &) {
  dirContext_.pop_back();
}

// Clauses
// Mainly categorized as
// 1. Checks on 'OmpClauseList' from 'parse-tree.h'.
// 2. Checks on clauses which fall under 'struct OmpClause' from parse-tree.h.
// 3. Checks on clauses which are not in 'struct OmpClause' from parse-tree.h.

void OmpStructureChecker::Leave(const parser::OmpClauseList &) {
  // 2.7.1 Loop Construct Restriction
  if (llvm::omp::allDoSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_schedule)}) {
      // only one schedule clause is allowed
      const auto &schedClause{std::get<parser::OmpClause::Schedule>(clause->u)};
      auto &modifiers{OmpGetModifiers(schedClause.v)};
      auto *ordering{
          OmpGetUniqueModifier<parser::OmpOrderingModifier>(modifiers)};
      if (ordering &&
          ordering->v == parser::OmpOrderingModifier::Value::Nonmonotonic) {
        if (FindClause(llvm::omp::Clause::OMPC_ordered)) {
          context_.Say(clause->source,
              "The NONMONOTONIC modifier cannot be specified "
              "if an ORDERED clause is specified"_err_en_US);
        }
      }
    }

    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_ordered)}) {
      // only one ordered clause is allowed
      const auto &orderedClause{
          std::get<parser::OmpClause::Ordered>(clause->u)};

      if (orderedClause.v) {
        CheckNotAllowedIfClause(
            llvm::omp::Clause::OMPC_ordered, {llvm::omp::Clause::OMPC_linear});

        if (auto *clause2{FindClause(llvm::omp::Clause::OMPC_collapse)}) {
          const auto &collapseClause{
              std::get<parser::OmpClause::Collapse>(clause2->u)};
          // ordered and collapse both have parameters
          if (const auto orderedValue{GetIntValue(orderedClause.v)}) {
            if (const auto collapseValue{GetIntValue(collapseClause.v)}) {
              if (*orderedValue > 0 && *orderedValue < *collapseValue) {
                context_.Say(clause->source,
                    "The parameter of the ORDERED clause must be "
                    "greater than or equal to "
                    "the parameter of the COLLAPSE clause"_err_en_US);
              }
            }
          }
        }
      }

      // TODO: ordered region binding check (requires nesting implementation)
    }
  } // doSet

  // 2.8.1 Simd Construct Restriction
  if (llvm::omp::allSimdSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_simdlen)}) {
      if (auto *clause2{FindClause(llvm::omp::Clause::OMPC_safelen)}) {
        const auto &simdlenClause{
            std::get<parser::OmpClause::Simdlen>(clause->u)};
        const auto &safelenClause{
            std::get<parser::OmpClause::Safelen>(clause2->u)};
        // simdlen and safelen both have parameters
        if (const auto simdlenValue{GetIntValue(simdlenClause.v)}) {
          if (const auto safelenValue{GetIntValue(safelenClause.v)}) {
            if (*safelenValue > 0 && *simdlenValue > *safelenValue) {
              context_.Say(clause->source,
                  "The parameter of the SIMDLEN clause must be less than or "
                  "equal to the parameter of the SAFELEN clause"_err_en_US);
            }
          }
        }
      }
    }

    // 2.11.5 Simd construct restriction (OpenMP 5.1)
    if (auto *sl_clause{FindClause(llvm::omp::Clause::OMPC_safelen)}) {
      if (auto *o_clause{FindClause(llvm::omp::Clause::OMPC_order)}) {
        const auto &orderClause{
            std::get<parser::OmpClause::Order>(o_clause->u)};
        if (std::get<parser::OmpOrderClause::Ordering>(orderClause.v.t) ==
            parser::OmpOrderClause::Ordering::Concurrent) {
          context_.Say(sl_clause->source,
              "The `SAFELEN` clause cannot appear in the `SIMD` directive "
              "with `ORDER(CONCURRENT)` clause"_err_en_US);
        }
      }
    }
  } // SIMD

  // Semantic checks related to presence of multiple list items within the same
  // clause
  CheckMultListItems();

  if (GetContext().directive == llvm::omp::Directive::OMPD_task) {
    if (auto *detachClause{FindClause(llvm::omp::Clause::OMPC_detach)}) {
      unsigned version{context_.langOptions().OpenMPVersion};
      if (version == 50 || version == 51) {
        // OpenMP 5.0: 2.10.1 Task construct restrictions
        CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_detach,
            {llvm::omp::Clause::OMPC_mergeable});
      } else if (version >= 52) {
        // OpenMP 5.2: 12.5.2 Detach construct restrictions
        if (FindClause(llvm::omp::Clause::OMPC_final)) {
          context_.Say(GetContext().clauseSource,
              "If a DETACH clause appears on a directive, then the encountering task must not be a FINAL task"_err_en_US);
        }

        const auto &detach{
            std::get<parser::OmpClause::Detach>(detachClause->u)};
        if (const auto *name{parser::Unwrap<parser::Name>(detach.v.v)}) {
          Symbol *eventHandleSym{name->symbol};
          auto checkVarAppearsInDataEnvClause = [&](const parser::OmpObjectList
                                                        &objs,
                                                    std::string clause) {
            for (const auto &obj : objs.v) {
              if (const parser::Name *
                  objName{parser::Unwrap<parser::Name>(obj)}) {
                if (&objName->symbol->GetUltimate() == eventHandleSym) {
                  context_.Say(GetContext().clauseSource,
                      "A variable: `%s` that appears in a DETACH clause cannot appear on %s clause on the same construct"_err_en_US,
                      objName->source, clause);
                }
              }
            }
          };
          if (auto *dataEnvClause{
                  FindClause(llvm::omp::Clause::OMPC_private)}) {
            const auto &pClause{
                std::get<parser::OmpClause::Private>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(pClause.v, "PRIVATE");
          } else if (auto *dataEnvClause{
                         FindClause(llvm::omp::Clause::OMPC_shared)}) {
            const auto &sClause{
                std::get<parser::OmpClause::Shared>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(sClause.v, "SHARED");
          } else if (auto *dataEnvClause{
                         FindClause(llvm::omp::Clause::OMPC_firstprivate)}) {
            const auto &fpClause{
                std::get<parser::OmpClause::Firstprivate>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(fpClause.v, "FIRSTPRIVATE");
          } else if (auto *dataEnvClause{
                         FindClause(llvm::omp::Clause::OMPC_in_reduction)}) {
            const auto &irClause{
                std::get<parser::OmpClause::InReduction>(dataEnvClause->u)};
            checkVarAppearsInDataEnvClause(
                std::get<parser::OmpObjectList>(irClause.v.t), "IN_REDUCTION");
          }
        }
      }
    }
  }

  auto testThreadprivateVarErr = [&](Symbol sym, parser::Name name,
                                     llvmOmpClause clauseTy) {
    if (sym.test(Symbol::Flag::OmpThreadprivate))
      context_.Say(name.source,
          "A THREADPRIVATE variable cannot be in %s clause"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseTy).str()));
  };

  // [5.1] 2.21.2 Threadprivate Directive Restriction
  OmpClauseSet threadprivateAllowedSet{llvm::omp::Clause::OMPC_copyin,
      llvm::omp::Clause::OMPC_copyprivate, llvm::omp::Clause::OMPC_schedule,
      llvm::omp::Clause::OMPC_num_threads, llvm::omp::Clause::OMPC_thread_limit,
      llvm::omp::Clause::OMPC_if};
  for (auto it : GetContext().clauseInfo) {
    llvmOmpClause type = it.first;
    const auto *clause = it.second;
    if (!threadprivateAllowedSet.test(type)) {
      if (const auto *objList{GetOmpObjectList(*clause)}) {
        for (const auto &ompObject : objList->v) {
          common::visit(
              common::visitors{
                  [&](const parser::Designator &) {
                    if (const auto *name{
                            parser::Unwrap<parser::Name>(ompObject)}) {
                      if (name->symbol) {
                        testThreadprivateVarErr(
                            name->symbol->GetUltimate(), *name, type);
                      }
                    }
                  },
                  [&](const parser::Name &name) {
                    if (name.symbol) {
                      for (const auto &mem :
                          name.symbol->get<CommonBlockDetails>().objects()) {
                        testThreadprivateVarErr(mem->GetUltimate(), name, type);
                        break;
                      }
                    }
                  },
              },
              ompObject.u);
        }
      }
    }
  }

  CheckRequireAtLeastOneOf();
}

void OmpStructureChecker::Enter(const parser::OmpClause &x) {
  SetContextClause(x);

  // The visitors for these clauses do their own checks.
  switch (x.Id()) {
  case llvm::omp::Clause::OMPC_copyprivate:
  case llvm::omp::Clause::OMPC_enter:
  case llvm::omp::Clause::OMPC_lastprivate:
  case llvm::omp::Clause::OMPC_reduction:
  case llvm::omp::Clause::OMPC_to:
    return;
  default:
    break;
  }

  if (const parser::OmpObjectList *objList{GetOmpObjectList(x)}) {
    SymbolSourceMap symbols;
    GetSymbolsInObjectList(*objList, symbols);
    for (const auto &[symbol, source] : symbols) {
      if (!IsVariableListItem(*symbol)) {
        deferredNonVariables_.insert({symbol, source});
      }
    }
  }
}

// Following clauses do not have a separate node in parse-tree.h.
CHECK_SIMPLE_CLAUSE(Absent, OMPC_absent)
CHECK_SIMPLE_CLAUSE(Affinity, OMPC_affinity)
CHECK_SIMPLE_CLAUSE(Capture, OMPC_capture)
CHECK_SIMPLE_CLAUSE(Contains, OMPC_contains)
CHECK_SIMPLE_CLAUSE(Default, OMPC_default)
CHECK_SIMPLE_CLAUSE(Depobj, OMPC_depobj)
CHECK_SIMPLE_CLAUSE(DeviceType, OMPC_device_type)
CHECK_SIMPLE_CLAUSE(DistSchedule, OMPC_dist_schedule)
CHECK_SIMPLE_CLAUSE(Exclusive, OMPC_exclusive)
CHECK_SIMPLE_CLAUSE(Final, OMPC_final)
CHECK_SIMPLE_CLAUSE(Flush, OMPC_flush)
CHECK_SIMPLE_CLAUSE(Full, OMPC_full)
CHECK_SIMPLE_CLAUSE(Grainsize, OMPC_grainsize)
CHECK_SIMPLE_CLAUSE(Holds, OMPC_holds)
CHECK_SIMPLE_CLAUSE(Inclusive, OMPC_inclusive)
CHECK_SIMPLE_CLAUSE(Initializer, OMPC_initializer)
CHECK_SIMPLE_CLAUSE(Match, OMPC_match)
CHECK_SIMPLE_CLAUSE(Nontemporal, OMPC_nontemporal)
CHECK_SIMPLE_CLAUSE(NumTasks, OMPC_num_tasks)
CHECK_SIMPLE_CLAUSE(Order, OMPC_order)
CHECK_SIMPLE_CLAUSE(Read, OMPC_read)
CHECK_SIMPLE_CLAUSE(Threadprivate, OMPC_threadprivate)
CHECK_SIMPLE_CLAUSE(Threads, OMPC_threads)
CHECK_SIMPLE_CLAUSE(Inbranch, OMPC_inbranch)
CHECK_SIMPLE_CLAUSE(Link, OMPC_link)
CHECK_SIMPLE_CLAUSE(Indirect, OMPC_indirect)
CHECK_SIMPLE_CLAUSE(Mergeable, OMPC_mergeable)
CHECK_SIMPLE_CLAUSE(NoOpenmp, OMPC_no_openmp)
CHECK_SIMPLE_CLAUSE(NoOpenmpRoutines, OMPC_no_openmp_routines)
CHECK_SIMPLE_CLAUSE(NoOpenmpConstructs, OMPC_no_openmp_constructs)
CHECK_SIMPLE_CLAUSE(NoParallelism, OMPC_no_parallelism)
CHECK_SIMPLE_CLAUSE(Nogroup, OMPC_nogroup)
CHECK_SIMPLE_CLAUSE(Notinbranch, OMPC_notinbranch)
CHECK_SIMPLE_CLAUSE(Partial, OMPC_partial)
CHECK_SIMPLE_CLAUSE(ProcBind, OMPC_proc_bind)
CHECK_SIMPLE_CLAUSE(Simd, OMPC_simd)
CHECK_SIMPLE_CLAUSE(Sizes, OMPC_sizes)
CHECK_SIMPLE_CLAUSE(Permutation, OMPC_permutation)
CHECK_SIMPLE_CLAUSE(Uniform, OMPC_uniform)
CHECK_SIMPLE_CLAUSE(Unknown, OMPC_unknown)
CHECK_SIMPLE_CLAUSE(Untied, OMPC_untied)
CHECK_SIMPLE_CLAUSE(UsesAllocators, OMPC_uses_allocators)
CHECK_SIMPLE_CLAUSE(Write, OMPC_write)
CHECK_SIMPLE_CLAUSE(Init, OMPC_init)
CHECK_SIMPLE_CLAUSE(Use, OMPC_use)
CHECK_SIMPLE_CLAUSE(Novariants, OMPC_novariants)
CHECK_SIMPLE_CLAUSE(Nocontext, OMPC_nocontext)
CHECK_SIMPLE_CLAUSE(Severity, OMPC_severity)
CHECK_SIMPLE_CLAUSE(Message, OMPC_message)
CHECK_SIMPLE_CLAUSE(Filter, OMPC_filter)
CHECK_SIMPLE_CLAUSE(Otherwise, OMPC_otherwise)
CHECK_SIMPLE_CLAUSE(AdjustArgs, OMPC_adjust_args)
CHECK_SIMPLE_CLAUSE(AppendArgs, OMPC_append_args)
CHECK_SIMPLE_CLAUSE(MemoryOrder, OMPC_memory_order)
CHECK_SIMPLE_CLAUSE(Bind, OMPC_bind)
CHECK_SIMPLE_CLAUSE(Align, OMPC_align)
CHECK_SIMPLE_CLAUSE(Compare, OMPC_compare)
CHECK_SIMPLE_CLAUSE(OmpxAttribute, OMPC_ompx_attribute)
CHECK_SIMPLE_CLAUSE(Weak, OMPC_weak)
CHECK_SIMPLE_CLAUSE(AcqRel, OMPC_acq_rel)
CHECK_SIMPLE_CLAUSE(Acquire, OMPC_acquire)
CHECK_SIMPLE_CLAUSE(Relaxed, OMPC_relaxed)
CHECK_SIMPLE_CLAUSE(Release, OMPC_release)
CHECK_SIMPLE_CLAUSE(SeqCst, OMPC_seq_cst)
CHECK_SIMPLE_CLAUSE(Fail, OMPC_fail)

CHECK_REQ_SCALAR_INT_CLAUSE(NumTeams, OMPC_num_teams)
CHECK_REQ_SCALAR_INT_CLAUSE(NumThreads, OMPC_num_threads)
CHECK_REQ_SCALAR_INT_CLAUSE(OmpxDynCgroupMem, OMPC_ompx_dyn_cgroup_mem)
CHECK_REQ_SCALAR_INT_CLAUSE(Priority, OMPC_priority)
CHECK_REQ_SCALAR_INT_CLAUSE(ThreadLimit, OMPC_thread_limit)

CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Collapse, OMPC_collapse)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Safelen, OMPC_safelen)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Simdlen, OMPC_simdlen)

// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.

void OmpStructureChecker::Enter(const parser::OmpClause::Destroy &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_destroy);

  llvm::omp::Directive dir{GetContext().directive};
  unsigned version{context_.langOptions().OpenMPVersion};
  if (dir == llvm::omp::Directive::OMPD_depobj) {
    unsigned argSince{52}, noargDeprecatedIn{52};
    if (x.v) {
      if (version < argSince) {
        context_.Say(GetContext().clauseSource,
            "The object parameter in DESTROY clause on DEPOPJ construct is not allowed in %s, %s"_warn_en_US,
            ThisVersion(version), TryVersion(argSince));
      }
    } else {
      if (version >= noargDeprecatedIn) {
        context_.Say(GetContext().clauseSource,
            "The DESTROY clause without argument on DEPOBJ construct is deprecated in %s"_warn_en_US,
            ThisVersion(noargDeprecatedIn));
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Reduction &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_reduction);
  auto &objects{std::get<parser::OmpObjectList>(x.v.t)};

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_reduction,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    const auto *ident{
        OmpGetUniqueModifier<parser::OmpReductionIdentifier>(modifiers)};
    assert(ident && "reduction-identifier is a required modifier");
    if (CheckReductionOperator(*ident, OmpGetModifierSource(modifiers, ident),
            llvm::omp::OMPC_reduction)) {
      CheckReductionObjectTypes(objects, *ident);
    }
    using ReductionModifier = parser::OmpReductionModifier;
    if (auto *modifier{OmpGetUniqueModifier<ReductionModifier>(modifiers)}) {
      CheckReductionModifier(*modifier);
    }
  }
  CheckReductionObjects(objects, llvm::omp::Clause::OMPC_reduction);

  // If this is a worksharing construct then ensure the reduction variable
  // is not private in the parallel region that it binds to.
  if (llvm::omp::nestedReduceWorkshareAllowedSet.test(GetContext().directive)) {
    CheckSharedBindingInOuterContext(objects);
  }

  if (GetContext().directive == llvm::omp::Directive::OMPD_loop) {
    for (auto clause : GetContext().clauseInfo) {
      if (const auto *bindClause{
              std::get_if<parser::OmpClause::Bind>(&clause.second->u)}) {
        if (bindClause->v.v == parser::OmpBindClause::Binding::Teams) {
          context_.Say(GetContext().clauseSource,
              "'REDUCTION' clause not allowed with '!$OMP LOOP BIND(TEAMS)'."_err_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::InReduction &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_in_reduction);
  auto &objects{std::get<parser::OmpObjectList>(x.v.t)};

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_in_reduction,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    const auto *ident{
        OmpGetUniqueModifier<parser::OmpReductionIdentifier>(modifiers)};
    assert(ident && "reduction-identifier is a required modifier");
    if (CheckReductionOperator(*ident, OmpGetModifierSource(modifiers, ident),
            llvm::omp::OMPC_in_reduction)) {
      CheckReductionObjectTypes(objects, *ident);
    }
  }
  CheckReductionObjects(objects, llvm::omp::Clause::OMPC_in_reduction);
}

void OmpStructureChecker::Enter(const parser::OmpClause::TaskReduction &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_task_reduction);
  auto &objects{std::get<parser::OmpObjectList>(x.v.t)};

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_task_reduction,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    const auto *ident{
        OmpGetUniqueModifier<parser::OmpReductionIdentifier>(modifiers)};
    assert(ident && "reduction-identifier is a required modifier");
    if (CheckReductionOperator(*ident, OmpGetModifierSource(modifiers, ident),
            llvm::omp::OMPC_task_reduction)) {
      CheckReductionObjectTypes(objects, *ident);
    }
  }
  CheckReductionObjects(objects, llvm::omp::Clause::OMPC_task_reduction);
}

bool OmpStructureChecker::CheckReductionOperator(
    const parser::OmpReductionIdentifier &ident, parser::CharBlock source,
    llvm::omp::Clause clauseId) {
  auto visitOperator{[&](const parser::DefinedOperator &dOpr) {
    if (const auto *intrinsicOp{
            std::get_if<parser::DefinedOperator::IntrinsicOperator>(&dOpr.u)}) {
      switch (*intrinsicOp) {
      case parser::DefinedOperator::IntrinsicOperator::Add:
      case parser::DefinedOperator::IntrinsicOperator::Multiply:
      case parser::DefinedOperator::IntrinsicOperator::AND:
      case parser::DefinedOperator::IntrinsicOperator::OR:
      case parser::DefinedOperator::IntrinsicOperator::EQV:
      case parser::DefinedOperator::IntrinsicOperator::NEQV:
        return true;
      case parser::DefinedOperator::IntrinsicOperator::Subtract:
        context_.Say(GetContext().clauseSource,
            "The minus reduction operator is deprecated since OpenMP 5.2 and is not supported in the REDUCTION clause."_err_en_US,
            ContextDirectiveAsFortran());
        return false;
      default:
        break;
      }
    }
    // User-defined operators are OK if there has been a declared reduction
    // for that. We mangle those names to store the user details.
    if (const auto *definedOp{std::get_if<parser::DefinedOpName>(&dOpr.u)}) {
      std::string mangled{MangleDefinedOperator(definedOp->v.symbol->name())};
      const Scope &scope{definedOp->v.symbol->owner()};
      if (const Symbol *symbol{scope.FindSymbol(mangled)}) {
        if (symbol->detailsIf<UserReductionDetails>()) {
          return true;
        }
      }
    }
    context_.Say(source, "Invalid reduction operator in %s clause."_err_en_US,
        parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    return false;
  }};

  auto visitDesignator{[&](const parser::ProcedureDesignator &procD) {
    const parser::Name *name{std::get_if<parser::Name>(&procD.u)};
    bool valid{false};
    if (name && name->symbol) {
      const SourceName &realName{name->symbol->GetUltimate().name()};
      valid =
          llvm::is_contained({"max", "min", "iand", "ior", "ieor"}, realName);
      if (!valid) {
        valid = name->symbol->detailsIf<UserReductionDetails>();
      }
    }
    if (!valid) {
      context_.Say(source,
          "Invalid reduction identifier in %s clause."_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    }
    return valid;
  }};

  return common::visit(
      common::visitors{visitOperator, visitDesignator}, ident.u);
}

/// Check restrictions on objects that are common to all reduction clauses.
void OmpStructureChecker::CheckReductionObjects(
    const parser::OmpObjectList &objects, llvm::omp::Clause clauseId) {
  unsigned version{context_.langOptions().OpenMPVersion};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objects, symbols);

  // Array sections must be a contiguous storage, have non-zero length.
  for (const parser::OmpObject &object : objects.v) {
    CheckIfContiguous(object);
  }
  CheckReductionArraySection(objects, clauseId);
  // An object must be definable.
  CheckDefinableObjects(symbols, clauseId);
  // Procedure pointers are not allowed.
  CheckProcedurePointer(symbols, clauseId);
  // Pointers must not have INTENT(IN).
  CheckIntentInPointer(symbols, clauseId);

  // Disallow common blocks.
  // Iterate on objects because `GetSymbolsInObjectList` expands common block
  // names into the lists of their members.
  for (const parser::OmpObject &object : objects.v) {
    auto *symbol{GetObjectSymbol(object)};
    if (symbol && IsCommonBlock(*symbol)) {
      auto source{GetObjectSource(object)};
      context_.Say(source ? *source : GetContext().clauseSource,
          "Common block names are not allowed in %s clause"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    }
  }

  // Denied in all current versions of the standard because structure components
  // are not definable (i.e. they are expressions not variables).
  // Object cannot be a part of another object (except array elements).
  CheckStructureComponent(objects, clauseId);

  if (version >= 50) {
    // If object is an array section or element, the base expression must be
    // a language identifier.
    for (const parser::OmpObject &object : objects.v) {
      if (auto *elem{GetArrayElementFromObj(object)}) {
        const parser::DataRef &base = elem->base;
        if (!std::holds_alternative<parser::Name>(base.u)) {
          auto source{GetObjectSource(object)};
          context_.Say(source ? *source : GetContext().clauseSource,
              "The base expression of an array element or section in %s clause must be an identifier"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
        }
      }
    }
    // Type parameter inquiries are not allowed.
    for (const parser::OmpObject &object : objects.v) {
      if (auto *dataRef{GetDataRefFromObj(object)}) {
        if (IsDataRefTypeParamInquiry(dataRef)) {
          auto source{GetObjectSource(object)};
          context_.Say(source ? *source : GetContext().clauseSource,
              "Type parameter inquiry is not permitted in %s clause"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
        }
      }
    }
  }
}

static bool CheckSymbolSupportsType(const Scope &scope,
    const parser::CharBlock &name, const DeclTypeSpec &type) {
  if (const auto *symbol{scope.FindSymbol(name)}) {
    if (const auto *reductionDetails{
            symbol->detailsIf<UserReductionDetails>()}) {
      return reductionDetails->SupportsType(type);
    }
  }
  return false;
}

static bool IsReductionAllowedForType(
    const parser::OmpReductionIdentifier &ident, const DeclTypeSpec &type,
    const Scope &scope, SemanticsContext &context) {
  auto isLogical{[](const DeclTypeSpec &type) -> bool {
    return type.category() == DeclTypeSpec::Logical;
  }};
  auto isCharacter{[](const DeclTypeSpec &type) -> bool {
    return type.category() == DeclTypeSpec::Character;
  }};

  auto checkOperator{[&](const parser::DefinedOperator &dOpr) {
    if (const auto *intrinsicOp{
            std::get_if<parser::DefinedOperator::IntrinsicOperator>(&dOpr.u)}) {
      // OMP5.2: The type [...] of a list item that appears in a
      // reduction clause must be valid for the combiner expression
      // See F2023: Table 10.2
      // .LT., .LE., .GT., .GE. are handled as procedure designators
      // below.
      switch (*intrinsicOp) {
      case parser::DefinedOperator::IntrinsicOperator::Multiply:
      case parser::DefinedOperator::IntrinsicOperator::Add:
      case parser::DefinedOperator::IntrinsicOperator::Subtract:
        if (type.IsNumeric(TypeCategory::Integer) ||
            type.IsNumeric(TypeCategory::Real) ||
            type.IsNumeric(TypeCategory::Complex))
          return true;
        break;

      case parser::DefinedOperator::IntrinsicOperator::AND:
      case parser::DefinedOperator::IntrinsicOperator::OR:
      case parser::DefinedOperator::IntrinsicOperator::EQV:
      case parser::DefinedOperator::IntrinsicOperator::NEQV:
        if (isLogical(type)) {
          return true;
        }
        break;

      // Reduction identifier is not in OMP5.2 Table 5.2
      default:
        DIE("This should have been caught in CheckIntrinsicOperator");
        return false;
      }
      parser::CharBlock name{MakeNameFromOperator(*intrinsicOp, context)};
      return CheckSymbolSupportsType(scope, name, type);
    } else if (const auto *definedOp{
                   std::get_if<parser::DefinedOpName>(&dOpr.u)}) {
      return CheckSymbolSupportsType(
          scope, MangleDefinedOperator(definedOp->v.symbol->name()), type);
    }
    llvm_unreachable(
        "A DefinedOperator is either a DefinedOpName or an IntrinsicOperator");
  }};

  auto checkDesignator{[&](const parser::ProcedureDesignator &procD) {
    const parser::Name *name{std::get_if<parser::Name>(&procD.u)};
    CHECK(name && name->symbol);
    if (name && name->symbol) {
      const SourceName &realName{name->symbol->GetUltimate().name()};
      // OMP5.2: The type [...] of a list item that appears in a
      // reduction clause must be valid for the combiner expression
      if (realName == "iand" || realName == "ior" || realName == "ieor") {
        // IAND: arguments must be integers: F2023 16.9.100
        // IEOR: arguments must be integers: F2023 16.9.106
        // IOR: arguments must be integers: F2023 16.9.111
        if (type.IsNumeric(TypeCategory::Integer)) {
          return true;
        }
      } else if (realName == "max" || realName == "min") {
        // MAX: arguments must be integer, real, or character:
        // F2023 16.9.135
        // MIN: arguments must be integer, real, or character:
        // F2023 16.9.141
        if (type.IsNumeric(TypeCategory::Integer) ||
            type.IsNumeric(TypeCategory::Real) || isCharacter(type)) {
          return true;
        }
      }

      // If we get here, it may be a user declared reduction, so check
      // if the symbol has UserReductionDetails, and if so, the type is
      // supported.
      if (const auto *reductionDetails{
              name->symbol->detailsIf<UserReductionDetails>()}) {
        return reductionDetails->SupportsType(type);
      }

      // We also need to check for mangled names (max, min, iand, ieor and ior)
      // and then check if the type is there.
      parser::CharBlock mangledName{MangleSpecialFunctions(name->source)};
      return CheckSymbolSupportsType(scope, mangledName, type);
    }
    // Everything else is "not matching type".
    return false;
  }};

  return common::visit(
      common::visitors{checkOperator, checkDesignator}, ident.u);
}

void OmpStructureChecker::CheckReductionObjectTypes(
    const parser::OmpObjectList &objects,
    const parser::OmpReductionIdentifier &ident) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objects, symbols);

  for (auto &[symbol, source] : symbols) {
    if (auto *type{symbol->GetType()}) {
      const auto &scope{context_.FindScope(symbol->name())};
      if (!IsReductionAllowedForType(ident, *type, scope, context_)) {
        context_.Say(source,
            "The type of '%s' is incompatible with the reduction operator."_err_en_US,
            symbol->name());
      }
    } else {
      assert(IsProcedurePointer(*symbol) && "Unexpected symbol properties");
    }
  }
}

void OmpStructureChecker::CheckReductionModifier(
    const parser::OmpReductionModifier &modifier) {
  using ReductionModifier = parser::OmpReductionModifier;
  if (modifier.v == ReductionModifier::Value::Default) {
    // The default one is always ok.
    return;
  }
  const DirectiveContext &dirCtx{GetContext()};
  if (dirCtx.directive == llvm::omp::Directive::OMPD_loop ||
      dirCtx.directive == llvm::omp::Directive::OMPD_taskloop) {
    // [5.2:257:33-34]
    // If a reduction-modifier is specified in a reduction clause that
    // appears on the directive, then the reduction modifier must be
    // default.
    // [5.2:268:16]
    // The reduction-modifier must be default.
    context_.Say(GetContext().clauseSource,
        "REDUCTION modifier on %s directive must be DEFAULT"_err_en_US,
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    return;
  }
  if (modifier.v == ReductionModifier::Value::Task) {
    // "Task" is only allowed on worksharing or "parallel" directive.
    static llvm::omp::Directive worksharing[]{
        llvm::omp::Directive::OMPD_do, llvm::omp::Directive::OMPD_scope,
        llvm::omp::Directive::OMPD_sections,
        // There are more worksharing directives, but they do not apply:
        // "for" is C++ only,
        // "single" and "workshare" don't allow reduction clause,
        // "loop" has different restrictions (checked above).
    };
    if (dirCtx.directive != llvm::omp::Directive::OMPD_parallel &&
        !llvm::is_contained(worksharing, dirCtx.directive)) {
      context_.Say(GetContext().clauseSource,
          "Modifier 'TASK' on REDUCTION clause is only allowed with "
          "PARALLEL or worksharing directive"_err_en_US);
    }
  } else if (modifier.v == ReductionModifier::Value::Inscan) {
    // "Inscan" is only allowed on worksharing-loop, worksharing-loop simd,
    // or "simd" directive.
    // The worksharing-loop directives are OMPD_do and OMPD_for. Only the
    // former is allowed in Fortran.
    if (!llvm::omp::scanParentAllowedSet.test(dirCtx.directive)) {
      context_.Say(GetContext().clauseSource,
          "Modifier 'INSCAN' on REDUCTION clause is only allowed with "
          "WORKSHARING LOOP, WORKSHARING LOOP SIMD, "
          "or SIMD directive"_err_en_US);
    }
  } else {
    // Catch-all for potential future modifiers to make sure that this
    // function is up-to-date.
    context_.Say(GetContext().clauseSource,
        "Unexpected modifier on REDUCTION clause"_err_en_US);
  }
}

void OmpStructureChecker::CheckReductionArraySection(
    const parser::OmpObjectList &ompObjectList, llvm::omp::Clause clauseId) {
  for (const auto &ompObject : ompObjectList.v) {
    if (const auto *dataRef{parser::Unwrap<parser::DataRef>(ompObject)}) {
      if (const auto *arrayElement{
              parser::Unwrap<parser::ArrayElement>(ompObject)}) {
        CheckArraySection(*arrayElement, GetLastName(*dataRef), clauseId);
      }
    }
  }
}

void OmpStructureChecker::CheckSharedBindingInOuterContext(
    const parser::OmpObjectList &redObjectList) {
  //  TODO: Verify the assumption here that the immediately enclosing region is
  //  the parallel region to which the worksharing construct having reduction
  //  binds to.
  if (auto *enclosingContext{GetEnclosingDirContext()}) {
    for (auto it : enclosingContext->clauseInfo) {
      llvmOmpClause type = it.first;
      const auto *clause = it.second;
      if (llvm::omp::privateReductionSet.test(type)) {
        if (const auto *objList{GetOmpObjectList(*clause)}) {
          for (const auto &ompObject : objList->v) {
            if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
              if (const auto *symbol{name->symbol}) {
                for (const auto &redOmpObject : redObjectList.v) {
                  if (const auto *rname{
                          parser::Unwrap<parser::Name>(redOmpObject)}) {
                    if (const auto *rsymbol{rname->symbol}) {
                      if (rsymbol->name() == symbol->name()) {
                        context_.Say(GetContext().clauseSource,
                            "%s variable '%s' is %s in outer context must"
                            " be shared in the parallel regions to which any"
                            " of the worksharing regions arising from the "
                            "worksharing construct bind."_err_en_US,
                            parser::ToUpperCaseLetters(
                                getClauseName(llvm::omp::Clause::OMPC_reduction)
                                    .str()),
                            symbol->name(),
                            parser::ToUpperCaseLetters(
                                getClauseName(type).str()));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Ordered &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_ordered);
  // the parameter of ordered clause is optional
  if (const auto &expr{x.v}) {
    RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_ordered, *expr);
    // 2.8.3 Loop SIMD Construct Restriction
    if (llvm::omp::allDoSimdSet.test(GetContext().directive)) {
      context_.Say(GetContext().clauseSource,
          "No ORDERED clause with a parameter can be specified "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Shared &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_shared);
  CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v, "SHARED");
  CheckCrayPointee(x.v, "SHARED");
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(x.v, symbols);
  CheckAllowedClause(llvm::omp::Clause::OMPC_private);
  CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v, "PRIVATE");
  CheckIntentInPointer(symbols, llvm::omp::Clause::OMPC_private);
  CheckCrayPointee(x.v, "PRIVATE");
}

void OmpStructureChecker::Enter(const parser::OmpClause::Nowait &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_nowait);
}

bool OmpStructureChecker::IsDataRefTypeParamInquiry(
    const parser::DataRef *dataRef) {
  bool dataRefIsTypeParamInquiry{false};
  if (const auto *structComp{
          parser::Unwrap<parser::StructureComponent>(dataRef)}) {
    if (const auto *compSymbol{structComp->component.symbol}) {
      if (const auto *compSymbolMiscDetails{
              std::get_if<MiscDetails>(&compSymbol->details())}) {
        const auto detailsKind = compSymbolMiscDetails->kind();
        dataRefIsTypeParamInquiry =
            (detailsKind == MiscDetails::Kind::KindParamInquiry ||
                detailsKind == MiscDetails::Kind::LenParamInquiry);
      } else if (compSymbol->has<TypeParamDetails>()) {
        dataRefIsTypeParamInquiry = true;
      }
    }
  }
  return dataRefIsTypeParamInquiry;
}

void OmpStructureChecker::CheckVarIsNotPartOfAnotherVar(
    const parser::CharBlock &source, const parser::OmpObjectList &objList,
    llvm::StringRef clause) {
  for (const auto &ompObject : objList.v) {
    CheckVarIsNotPartOfAnotherVar(source, ompObject, clause);
  }
}

void OmpStructureChecker::CheckVarIsNotPartOfAnotherVar(
    const parser::CharBlock &source, const parser::OmpObject &ompObject,
    llvm::StringRef clause) {
  common::visit(
      common::visitors{
          [&](const parser::Designator &designator) {
            if (const auto *dataRef{
                    std::get_if<parser::DataRef>(&designator.u)}) {
              if (IsDataRefTypeParamInquiry(dataRef)) {
                context_.Say(source,
                    "A type parameter inquiry cannot appear on the %s directive"_err_en_US,
                    ContextDirectiveAsFortran());
              } else if (parser::Unwrap<parser::StructureComponent>(
                             ompObject) ||
                  parser::Unwrap<parser::ArrayElement>(ompObject)) {
                if (llvm::omp::nonPartialVarSet.test(GetContext().directive)) {
                  context_.Say(source,
                      "A variable that is part of another variable (as an array or structure element) cannot appear on the %s directive"_err_en_US,
                      ContextDirectiveAsFortran());
                } else {
                  context_.Say(source,
                      "A variable that is part of another variable (as an array or structure element) cannot appear in a %s clause"_err_en_US,
                      clause.data());
                }
              }
            }
          },
          [&](const parser::Name &name) {},
      },
      ompObject.u);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Firstprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_firstprivate);

  CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v, "FIRSTPRIVATE");
  CheckCrayPointee(x.v, "FIRSTPRIVATE");
  CheckIsLoopIvPartOfClause(llvmOmpClause::OMPC_firstprivate, x.v);

  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  CheckCopyingPolymorphicAllocatable(
      currSymbols, llvm::omp::Clause::OMPC_firstprivate);

  DirectivesClauseTriple dirClauseTriple;
  // Check firstprivate variables in worksharing constructs
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_do,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_sections,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_single,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  // Check firstprivate variables in distribute construct
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_distribute,
      std::make_pair(
          llvm::omp::Directive::OMPD_teams, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_distribute,
      std::make_pair(llvm::omp::Directive::OMPD_target_teams,
          llvm::omp::privateReductionSet));
  // Check firstprivate variables in task and taskloop constructs
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_task,
      std::make_pair(llvm::omp::Directive::OMPD_parallel,
          OmpClauseSet{llvm::omp::Clause::OMPC_reduction}));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_taskloop,
      std::make_pair(llvm::omp::Directive::OMPD_parallel,
          OmpClauseSet{llvm::omp::Clause::OMPC_reduction}));

  CheckPrivateSymbolsInOuterCxt(
      currSymbols, dirClauseTriple, llvm::omp::Clause::OMPC_firstprivate);
}

void OmpStructureChecker::CheckIsLoopIvPartOfClause(
    llvmOmpClause clause, const parser::OmpObjectList &ompObjectList) {
  for (const auto &ompObject : ompObjectList.v) {
    if (const parser::Name *name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (name->symbol == GetContext().loopIV) {
        context_.Say(name->source,
            "DO iteration variable %s is not allowed in %s clause."_err_en_US,
            name->ToString(),
            parser::ToUpperCaseLetters(getClauseName(clause).str()));
      }
    }
  }
}

// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.
void OmpStructureChecker::Enter(const parser::OmpClause::Aligned &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_aligned);
  if (OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_aligned, GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    if (auto *align{OmpGetUniqueModifier<parser::OmpAlignment>(modifiers)}) {
      if (const auto &v{GetIntValue(align->v)}; !v || *v <= 0) {
        context_.Say(OmpGetModifierSource(modifiers, align),
            "The alignment value should be a constant positive integer"_err_en_US);
      }
    }
  }
  // 2.8.1 TODO: list-item attribute check
}

void OmpStructureChecker::Enter(const parser::OmpClause::Defaultmap &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_defaultmap);
  unsigned version{context_.langOptions().OpenMPVersion};
  using ImplicitBehavior = parser::OmpDefaultmapClause::ImplicitBehavior;
  auto behavior{std::get<ImplicitBehavior>(x.v.t)};
  if (version <= 45) {
    if (behavior != ImplicitBehavior::Tofrom) {
      context_.Say(GetContext().clauseSource,
          "%s is not allowed in %s, %s"_warn_en_US,
          parser::ToUpperCaseLetters(
              parser::OmpDefaultmapClause::EnumToString(behavior)),
          ThisVersion(version), TryVersion(50));
    }
  }
  if (!OmpVerifyModifiers(x.v, llvm::omp::OMPC_defaultmap,
          GetContext().clauseSource, context_)) {
    // If modifier verification fails, return early.
    return;
  }
  auto &modifiers{OmpGetModifiers(x.v)};
  auto *maybeCategory{
      OmpGetUniqueModifier<parser::OmpVariableCategory>(modifiers)};
  if (maybeCategory) {
    using VariableCategory = parser::OmpVariableCategory;
    VariableCategory::Value category{maybeCategory->v};
    unsigned tryVersion{0};
    if (version <= 45 && category != VariableCategory::Value::Scalar) {
      tryVersion = 50;
    }
    if (version < 52 && category == VariableCategory::Value::All) {
      tryVersion = 52;
    }
    if (tryVersion) {
      context_.Say(GetContext().clauseSource,
          "%s is not allowed in %s, %s"_warn_en_US,
          parser::ToUpperCaseLetters(VariableCategory::EnumToString(category)),
          ThisVersion(version), TryVersion(tryVersion));
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::If &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_if);
  unsigned version{context_.langOptions().OpenMPVersion};
  llvm::omp::Directive dir{GetContext().directive};

  auto isConstituent{[](llvm::omp::Directive dir, llvm::omp::Directive part) {
    using namespace llvm::omp;
    llvm::ArrayRef<Directive> dirLeafs{getLeafConstructsOrSelf(dir)};
    llvm::ArrayRef<Directive> partLeafs{getLeafConstructsOrSelf(part)};
    // Maybe it's sufficient to check if every leaf of `part` is also a leaf
    // of `dir`, but to be safe check if `partLeafs` is a sub-sequence of
    // `dirLeafs`.
    size_t dirSize{dirLeafs.size()}, partSize{partLeafs.size()};
    // Find the first leaf from `part` in `dir`.
    if (auto first = llvm::find(dirLeafs, partLeafs.front());
        first != dirLeafs.end()) {
      // A leaf can only appear once in a compound directive, so if `part`
      // is a subsequence of `dir`, it must start here.
      size_t firstPos{
          static_cast<size_t>(std::distance(dirLeafs.begin(), first))};
      llvm::ArrayRef<Directive> subSeq{
          first, std::min<size_t>(dirSize - firstPos, partSize)};
      return subSeq == partLeafs;
    }
    return false;
  }};

  if (OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_if, GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    if (auto *dnm{OmpGetUniqueModifier<parser::OmpDirectiveNameModifier>(
            modifiers)}) {
      llvm::omp::Directive sub{dnm->v};
      std::string subName{
          parser::ToUpperCaseLetters(getDirectiveName(sub).str())};
      std::string dirName{
          parser::ToUpperCaseLetters(getDirectiveName(dir).str())};

      parser::CharBlock modifierSource{OmpGetModifierSource(modifiers, dnm)};
      auto desc{OmpGetDescriptor<parser::OmpDirectiveNameModifier>()};
      std::string modName{desc.name.str()};

      if (!isConstituent(dir, sub)) {
        context_
            .Say(modifierSource,
                "%s is not a constituent of the %s directive"_err_en_US,
                subName, dirName)
            .Attach(GetContext().directiveSource,
                "Cannot apply to directive"_en_US);
      } else {
        static llvm::omp::Directive valid45[]{
            llvm::omp::OMPD_cancel, //
            llvm::omp::OMPD_parallel, //
            /* OMP 5.0+ also allows OMPD_simd */
            llvm::omp::OMPD_target, //
            llvm::omp::OMPD_target_data, //
            llvm::omp::OMPD_target_enter_data, //
            llvm::omp::OMPD_target_exit_data, //
            llvm::omp::OMPD_target_update, //
            llvm::omp::OMPD_task, //
            llvm::omp::OMPD_taskloop, //
            /* OMP 5.2+ also allows OMPD_teams */
        };
        if (version < 50 && sub == llvm::omp::OMPD_simd) {
          context_.Say(modifierSource,
              "%s is not allowed as '%s' in %s, %s"_warn_en_US, subName,
              modName, ThisVersion(version), TryVersion(50));
        } else if (version < 52 && sub == llvm::omp::OMPD_teams) {
          context_.Say(modifierSource,
              "%s is not allowed as '%s' in %s, %s"_warn_en_US, subName,
              modName, ThisVersion(version), TryVersion(52));
        } else if (!llvm::is_contained(valid45, sub) &&
            sub != llvm::omp::OMPD_simd && sub != llvm::omp::OMPD_teams) {
          context_.Say(modifierSource,
              "%s is not allowed as '%s' in %s"_err_en_US, subName, modName,
              ThisVersion(version));
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Linear &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_linear);
  unsigned version{context_.langOptions().OpenMPVersion};
  llvm::omp::Directive dir{GetContext().directive};
  parser::CharBlock clauseSource{GetContext().clauseSource};
  const parser::OmpLinearModifier *linearMod{nullptr};

  SymbolSourceMap symbols;
  auto &objects{std::get<parser::OmpObjectList>(x.v.t)};
  CheckCrayPointee(objects, "LINEAR", false);
  GetSymbolsInObjectList(objects, symbols);

  auto CheckIntegerNoRef{[&](const Symbol *symbol, parser::CharBlock source) {
    if (!symbol->GetType()->IsNumeric(TypeCategory::Integer)) {
      auto &desc{OmpGetDescriptor<parser::OmpLinearModifier>()};
      context_.Say(source,
          "The list item '%s' specified without the REF '%s' must be of INTEGER type"_err_en_US,
          symbol->name(), desc.name.str());
    }
  }};

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_linear, clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    linearMod = OmpGetUniqueModifier<parser::OmpLinearModifier>(modifiers);
    if (linearMod) {
      // 2.7 Loop Construct Restriction
      if ((llvm::omp::allDoSet | llvm::omp::allSimdSet).test(dir)) {
        context_.Say(clauseSource,
            "A modifier may not be specified in a LINEAR clause on the %s directive"_err_en_US,
            ContextDirectiveAsFortran());
        return;
      }

      auto &desc{OmpGetDescriptor<parser::OmpLinearModifier>()};
      for (auto &[symbol, source] : symbols) {
        if (linearMod->v != parser::OmpLinearModifier::Value::Ref) {
          CheckIntegerNoRef(symbol, source);
        } else {
          if (!IsAllocatable(*symbol) && !IsAssumedShape(*symbol) &&
              !IsPolymorphic(*symbol)) {
            context_.Say(source,
                "The list item `%s` specified with the REF '%s' must be polymorphic variable, assumed-shape array, or a variable with the `ALLOCATABLE` attribute"_err_en_US,
                symbol->name(), desc.name.str());
          }
        }
        if (linearMod->v == parser::OmpLinearModifier::Value::Ref ||
            linearMod->v == parser::OmpLinearModifier::Value::Uval) {
          if (!IsDummy(*symbol) || IsValue(*symbol)) {
            context_.Say(source,
                "If the `%s` is REF or UVAL, the list item '%s' must be a dummy argument without the VALUE attribute"_err_en_US,
                desc.name.str(), symbol->name());
          }
        }
      } // for (symbol, source)

      if (version >= 52 && !std::get</*PostModified=*/bool>(x.v.t)) {
        context_.Say(OmpGetModifierSource(modifiers, linearMod),
            "The 'modifier(<list>)' syntax is deprecated in %s, use '<list> : modifier' instead"_warn_en_US,
            ThisVersion(version));
      }
    }
  }

  // OpenMP 5.2: Ordered clause restriction
  if (const auto *clause{
          FindClause(GetContext(), llvm::omp::Clause::OMPC_ordered)}) {
    const auto &orderedClause{std::get<parser::OmpClause::Ordered>(clause->u)};
    if (orderedClause.v) {
      return;
    }
  }

  // OpenMP 5.2: Linear clause Restrictions
  for (auto &[symbol, source] : symbols) {
    if (!linearMod) {
      // Already checked this with the modifier present.
      CheckIntegerNoRef(symbol, source);
    }
    if (dir == llvm::omp::Directive::OMPD_declare_simd && !IsDummy(*symbol)) {
      context_.Say(source,
          "The list item `%s` must be a dummy argument"_err_en_US,
          symbol->name());
    }
    if (IsPointer(*symbol) || symbol->test(Symbol::Flag::CrayPointer)) {
      context_.Say(source,
          "The list item `%s` in a LINEAR clause must not be Cray Pointer or a variable with POINTER attribute"_err_en_US,
          symbol->name());
    }
    if (FindCommonBlockContaining(*symbol)) {
      context_.Say(source,
          "'%s' is a common block name and must not appear in an LINEAR clause"_err_en_US,
          symbol->name());
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Detach &x) {
  unsigned version{context_.langOptions().OpenMPVersion};
  if (version >= 52) {
    SetContextClauseInfo(llvm::omp::Clause::OMPC_detach);
  } else {
    // OpenMP 5.0: 2.10.1 Task construct restrictions
    CheckAllowedClause(llvm::omp::Clause::OMPC_detach);
  }
  // OpenMP 5.2: 12.5.2 Detach clause restrictions
  if (version >= 52) {
    CheckVarIsNotPartOfAnotherVar(GetContext().clauseSource, x.v.v, "DETACH");
  }

  if (const auto *name{parser::Unwrap<parser::Name>(x.v.v)}) {
    if (version >= 52 && IsPointer(*name->symbol)) {
      context_.Say(GetContext().clauseSource,
          "The event-handle: `%s` must not have the POINTER attribute"_err_en_US,
          name->ToString());
    }
    if (!name->symbol->GetType()->IsNumeric(TypeCategory::Integer)) {
      context_.Say(GetContext().clauseSource,
          "The event-handle: `%s` must be of type integer(kind=omp_event_handle_kind)"_err_en_US,
          name->ToString());
    }
  }
}

void OmpStructureChecker::CheckAllowedMapTypes(
    const parser::OmpMapType::Value &type,
    const std::list<parser::OmpMapType::Value> &allowedMapTypeList) {
  if (!llvm::is_contained(allowedMapTypeList, type)) {
    std::string commaSeparatedMapTypes;
    llvm::interleave(
        allowedMapTypeList.begin(), allowedMapTypeList.end(),
        [&](const parser::OmpMapType::Value &mapType) {
          commaSeparatedMapTypes.append(parser::ToUpperCaseLetters(
              parser::OmpMapType::EnumToString(mapType)));
        },
        [&] { commaSeparatedMapTypes.append(", "); });
    context_.Say(GetContext().clauseSource,
        "Only the %s map types are permitted "
        "for MAP clauses on the %s directive"_err_en_US,
        commaSeparatedMapTypes, ContextDirectiveAsFortran());
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Map &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_map);
  if (!OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_map, GetContext().clauseSource, context_)) {
    return;
  }

  auto &modifiers{OmpGetModifiers(x.v)};
  unsigned version{context_.langOptions().OpenMPVersion};
  if (auto commas{std::get<bool>(x.v.t)}; !commas && version >= 52) {
    context_.Say(GetContext().clauseSource,
        "The specification of modifiers without comma separators for the "
        "'MAP' clause has been deprecated in OpenMP 5.2"_port_en_US);
  }
  if (auto *iter{OmpGetUniqueModifier<parser::OmpIterator>(modifiers)}) {
    CheckIteratorModifier(*iter);
  }
  if (auto *type{OmpGetUniqueModifier<parser::OmpMapType>(modifiers)}) {
    using Value = parser::OmpMapType::Value;
    switch (GetContext().directive) {
    case llvm::omp::Directive::OMPD_target:
    case llvm::omp::Directive::OMPD_target_teams:
    case llvm::omp::Directive::OMPD_target_teams_distribute:
    case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
    case llvm::omp::Directive::OMPD_target_data:
      CheckAllowedMapTypes(
          type->v, {Value::To, Value::From, Value::Tofrom, Value::Alloc});
      break;
    case llvm::omp::Directive::OMPD_target_enter_data:
      CheckAllowedMapTypes(type->v, {Value::To, Value::Alloc});
      break;
    case llvm::omp::Directive::OMPD_target_exit_data:
      CheckAllowedMapTypes(
          type->v, {Value::From, Value::Release, Value::Delete});
      break;
    default:
      break;
    }
  }

  auto &&typeMods{
      OmpGetRepeatableModifier<parser::OmpMapTypeModifier>(modifiers)};
  struct Less {
    using Iterator = decltype(typeMods.begin());
    bool operator()(Iterator a, Iterator b) const {
      const parser::OmpMapTypeModifier *pa = *a;
      const parser::OmpMapTypeModifier *pb = *b;
      return pa->v < pb->v;
    }
  };
  if (auto maybeIter{FindDuplicate<Less>(typeMods)}) {
    context_.Say(GetContext().clauseSource,
        "Duplicate map-type-modifier entry '%s' will be ignored"_warn_en_US,
        parser::ToUpperCaseLetters(
            parser::OmpMapTypeModifier::EnumToString((**maybeIter)->v)));
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Schedule &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_schedule);
  const parser::OmpScheduleClause &scheduleClause = x.v;
  if (!OmpVerifyModifiers(scheduleClause, llvm::omp::OMPC_schedule,
          GetContext().clauseSource, context_)) {
    return;
  }

  // 2.7 Loop Construct Restriction
  if (llvm::omp::allDoSet.test(GetContext().directive)) {
    auto &modifiers{OmpGetModifiers(scheduleClause)};
    auto kind{std::get<parser::OmpScheduleClause::Kind>(scheduleClause.t)};
    auto &chunk{
        std::get<std::optional<parser::ScalarIntExpr>>(scheduleClause.t)};
    if (chunk) {
      if (kind == parser::OmpScheduleClause::Kind::Runtime ||
          kind == parser::OmpScheduleClause::Kind::Auto) {
        context_.Say(GetContext().clauseSource,
            "When SCHEDULE clause has %s specified, "
            "it must not have chunk size specified"_err_en_US,
            parser::ToUpperCaseLetters(
                parser::OmpScheduleClause::EnumToString(kind)));
      }
      if (const auto &chunkExpr{std::get<std::optional<parser::ScalarIntExpr>>(
              scheduleClause.t)}) {
        RequiresPositiveParameter(
            llvm::omp::Clause::OMPC_schedule, *chunkExpr, "chunk size");
      }
    }

    auto *ordering{
        OmpGetUniqueModifier<parser::OmpOrderingModifier>(modifiers)};
    if (ordering &&
        ordering->v == parser::OmpOrderingModifier::Value::Nonmonotonic) {
      if (kind != parser::OmpScheduleClause::Kind::Dynamic &&
          kind != parser::OmpScheduleClause::Kind::Guided) {
        context_.Say(GetContext().clauseSource,
            "The NONMONOTONIC modifier can only be specified with "
            "SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Device &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_device);
  const parser::OmpDeviceClause &deviceClause{x.v};
  const auto &device{std::get<parser::ScalarIntExpr>(deviceClause.t)};
  RequiresPositiveParameter(
      llvm::omp::Clause::OMPC_device, device, "device expression");
  llvm::omp::Directive dir{GetContext().directive};

  if (OmpVerifyModifiers(deviceClause, llvm::omp::OMPC_device,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(deviceClause)};

    if (auto *deviceMod{
            OmpGetUniqueModifier<parser::OmpDeviceModifier>(modifiers)}) {
      using Value = parser::OmpDeviceModifier::Value;
      if (dir != llvm::omp::OMPD_target && deviceMod->v == Value::Ancestor) {
        auto name{OmpGetDescriptor<parser::OmpDeviceModifier>().name};
        context_.Say(OmpGetModifierSource(modifiers, deviceMod),
            "The ANCESTOR %s must not appear on the DEVICE clause on any directive other than the TARGET construct. Found on %s construct."_err_en_US,
            name.str(), parser::ToUpperCaseLetters(getDirectiveName(dir)));
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Depend &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_depend);
  llvm::omp::Directive dir{GetContext().directive};
  unsigned version{context_.langOptions().OpenMPVersion};

  auto *doaDep{std::get_if<parser::OmpDoacross>(&x.v.u)};
  auto *taskDep{std::get_if<parser::OmpDependClause::TaskDep>(&x.v.u)};
  assert(((doaDep == nullptr) != (taskDep == nullptr)) &&
      "Unexpected alternative in update clause");

  if (doaDep) {
    CheckDoacross(*doaDep);
    CheckDependenceType(doaDep->GetDepType());
  } else {
    using Modifier = parser::OmpDependClause::TaskDep::Modifier;
    auto &modifiers{std::get<std::optional<std::list<Modifier>>>(taskDep->t)};
    if (!modifiers) {
      context_.Say(GetContext().clauseSource,
          "A DEPEND clause on a TASK construct must have a valid task dependence type"_err_en_US);
      return;
    }
    CheckTaskDependenceType(taskDep->GetTaskDepType());
  }

  if (dir == llvm::omp::OMPD_depobj) {
    // [5.0:255:11], [5.1:288:3]
    // A depend clause on a depobj construct must not have source, sink [or
    // depobj](5.0) as dependence-type.
    if (version >= 50) {
      bool invalidDep{false};
      if (taskDep) {
        if (version == 50) {
          invalidDep = taskDep->GetTaskDepType() ==
              parser::OmpTaskDependenceType::Value::Depobj;
        }
      } else {
        invalidDep = true;
      }
      if (invalidDep) {
        context_.Say(GetContext().clauseSource,
            "A DEPEND clause on a DEPOBJ construct must not have %s as dependence type"_err_en_US,
            version == 50 ? "SINK, SOURCE or DEPOBJ" : "SINK or SOURCE");
      }
    }
  } else if (dir != llvm::omp::OMPD_ordered) {
    if (doaDep) {
      context_.Say(GetContext().clauseSource,
          "The SINK and SOURCE dependence types can only be used with the ORDERED directive, used here in the %s construct"_err_en_US,
          parser::ToUpperCaseLetters(getDirectiveName(dir)));
    }
  }
  if (taskDep) {
    auto &objList{std::get<parser::OmpObjectList>(taskDep->t)};
    if (dir == llvm::omp::OMPD_depobj) {
      // [5.0:255:13], [5.1:288:6], [5.2:322:26]
      // A depend clause on a depobj construct must only specify one locator.
      if (objList.v.size() != 1) {
        context_.Say(GetContext().clauseSource,
            "A DEPEND clause on a DEPOBJ construct must only specify "
            "one locator"_err_en_US);
      }
    }
    for (const auto &object : objList.v) {
      if (const auto *name{std::get_if<parser::Name>(&object.u)}) {
        context_.Say(GetContext().clauseSource,
            "Common block name ('%s') cannot appear in a DEPEND "
            "clause"_err_en_US,
            name->ToString());
      } else if (auto *designator{std::get_if<parser::Designator>(&object.u)}) {
        if (auto *dataRef{std::get_if<parser::DataRef>(&designator->u)}) {
          CheckDependList(*dataRef);
          if (const auto *arr{
                  std::get_if<common::Indirection<parser::ArrayElement>>(
                      &dataRef->u)}) {
            CheckArraySection(arr->value(), GetLastName(*dataRef),
                llvm::omp::Clause::OMPC_depend);
          }
        }
      }
    }
    if (OmpVerifyModifiers(*taskDep, llvm::omp::OMPC_depend,
            GetContext().clauseSource, context_)) {
      auto &modifiers{OmpGetModifiers(*taskDep)};
      if (OmpGetUniqueModifier<parser::OmpIterator>(modifiers)) {
        if (dir == llvm::omp::OMPD_depobj) {
          context_.Say(GetContext().clauseSource,
              "An iterator-modifier may specify multiple locators, a DEPEND clause on a DEPOBJ construct must only specify one locator"_warn_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Doacross &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_doacross);
  CheckDoacross(x.v.v);
}

void OmpStructureChecker::CheckDoacross(const parser::OmpDoacross &doa) {
  if (std::holds_alternative<parser::OmpDoacross::Source>(doa.u)) {
    // Nothing to check here.
    return;
  }

  // Process SINK dependence type. SINK may only appear in an ORDER construct,
  // which references a prior ORDERED(n) clause on a DO or SIMD construct
  // that marks the top of the loop nest.

  auto &sink{std::get<parser::OmpDoacross::Sink>(doa.u)};
  const std::list<parser::OmpIteration> &vec{sink.v.v};

  // Check if the variables in the iteration vector are unique.
  struct Less {
    using Iterator = std::list<parser::OmpIteration>::const_iterator;
    bool operator()(Iterator a, Iterator b) const {
      auto namea{std::get<parser::Name>(a->t)};
      auto nameb{std::get<parser::Name>(b->t)};
      assert(namea.symbol && nameb.symbol && "Unresolved symbols");
      // The non-determinism of the "<" doesn't matter, we only care about
      // equality, i.e.  a == b  <=>  !(a < b) && !(b < a)
      return reinterpret_cast<uintptr_t>(namea.symbol) <
          reinterpret_cast<uintptr_t>(nameb.symbol);
    }
  };
  if (auto maybeIter{FindDuplicate<Less>(vec)}) {
    auto name{std::get<parser::Name>((*maybeIter)->t)};
    context_.Say(name.source,
        "Duplicate variable '%s' in the iteration vector"_err_en_US,
        name.ToString());
  }

  // Check if the variables in the iteration vector are induction variables.
  // Ignore any mismatch between the size of the iteration vector and the
  // number of DO constructs on the stack. This is checked elsewhere.

  auto GetLoopDirective{[](const parser::OpenMPLoopConstruct &x) {
    auto &begin{std::get<parser::OmpBeginLoopDirective>(x.t)};
    return std::get<parser::OmpLoopDirective>(begin.t).v;
  }};
  auto GetLoopClauses{[](const parser::OpenMPLoopConstruct &x)
                          -> const std::list<parser::OmpClause> & {
    auto &begin{std::get<parser::OmpBeginLoopDirective>(x.t)};
    return std::get<parser::OmpClauseList>(begin.t).v;
  }};

  std::set<const Symbol *> inductionVars;
  for (const LoopConstruct &loop : llvm::reverse(loopStack_)) {
    if (auto *doc{std::get_if<const parser::DoConstruct *>(&loop)}) {
      // Do-construct, collect the induction variable.
      if (auto &control{(*doc)->GetLoopControl()}) {
        if (auto *b{std::get_if<parser::LoopControl::Bounds>(&control->u)}) {
          inductionVars.insert(b->name.thing.symbol);
        }
      }
    } else {
      // Omp-loop-construct, check if it's do/simd with an ORDERED clause.
      auto *loopc{std::get_if<const parser::OpenMPLoopConstruct *>(&loop)};
      assert(loopc && "Expecting OpenMPLoopConstruct");
      llvm::omp::Directive loopDir{GetLoopDirective(**loopc)};
      if (loopDir == llvm::omp::OMPD_do || loopDir == llvm::omp::OMPD_simd) {
        auto IsOrdered{[](const parser::OmpClause &c) {
          return c.Id() == llvm::omp::OMPC_ordered;
        }};
        // If it has ORDERED clause, stop the traversal.
        if (llvm::any_of(GetLoopClauses(**loopc), IsOrdered)) {
          break;
        }
      }
    }
  }
  for (const parser::OmpIteration &iter : vec) {
    auto &name{std::get<parser::Name>(iter.t)};
    if (!inductionVars.count(name.symbol)) {
      context_.Say(name.source,
          "The iteration vector element '%s' is not an induction variable within the ORDERED loop nest"_err_en_US,
          name.ToString());
    }
  }
}

void OmpStructureChecker::CheckCopyingPolymorphicAllocatable(
    SymbolSourceMap &symbols, const llvm::omp::Clause clause) {
  if (context_.ShouldWarn(common::UsageWarning::Portability)) {
    for (auto &[symbol, source] : symbols) {
      if (IsPolymorphicAllocatable(*symbol)) {
        context_.Warn(common::UsageWarning::Portability, source,
            "If a polymorphic variable with allocatable attribute '%s' is in %s clause, the behavior is unspecified"_port_en_US,
            symbol->name(),
            parser::ToUpperCaseLetters(getClauseName(clause).str()));
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Copyprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_copyprivate);
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(x.v, symbols);
  CheckVariableListItem(symbols);
  CheckIntentInPointer(symbols, llvm::omp::Clause::OMPC_copyprivate);
  CheckCopyingPolymorphicAllocatable(
      symbols, llvm::omp::Clause::OMPC_copyprivate);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Lastprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_lastprivate);

  const auto &objectList{std::get<parser::OmpObjectList>(x.v.t)};
  CheckVarIsNotPartOfAnotherVar(
      GetContext().clauseSource, objectList, "LASTPRIVATE");
  CheckCrayPointee(objectList, "LASTPRIVATE");

  DirectivesClauseTriple dirClauseTriple;
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(objectList, currSymbols);
  CheckDefinableObjects(currSymbols, llvm::omp::Clause::OMPC_lastprivate);
  CheckCopyingPolymorphicAllocatable(
      currSymbols, llvm::omp::Clause::OMPC_lastprivate);

  // Check lastprivate variables in worksharing constructs
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_do,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_sections,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));

  CheckPrivateSymbolsInOuterCxt(
      currSymbols, dirClauseTriple, llvm::omp::Clause::OMPC_lastprivate);

  if (OmpVerifyModifiers(x.v, llvm::omp::OMPC_lastprivate,
          GetContext().clauseSource, context_)) {
    auto &modifiers{OmpGetModifiers(x.v)};
    using LastprivateModifier = parser::OmpLastprivateModifier;
    if (auto *modifier{OmpGetUniqueModifier<LastprivateModifier>(modifiers)}) {
      CheckLastprivateModifier(*modifier);
    }
  }
}

// Add any restrictions related to Modifiers/Directives with
// Lastprivate clause here:
void OmpStructureChecker::CheckLastprivateModifier(
    const parser::OmpLastprivateModifier &modifier) {
  using LastprivateModifier = parser::OmpLastprivateModifier;
  const DirectiveContext &dirCtx{GetContext()};
  if (modifier.v == LastprivateModifier::Value::Conditional &&
      dirCtx.directive == llvm::omp::Directive::OMPD_taskloop) {
    // [5.2:268:17]
    // The conditional lastprivate-modifier must not be specified.
    context_.Say(GetContext().clauseSource,
        "'CONDITIONAL' modifier on lastprivate clause with TASKLOOP "
        "directive is not allowed"_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Copyin &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_copyin);

  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  CheckCopyingPolymorphicAllocatable(
      currSymbols, llvm::omp::Clause::OMPC_copyin);
}

void OmpStructureChecker::CheckStructureComponent(
    const parser::OmpObjectList &objects, llvm::omp::Clause clauseId) {
  auto CheckComponent{[&](const parser::Designator &designator) {
    if (auto *dataRef{std::get_if<parser::DataRef>(&designator.u)}) {
      if (!IsDataRefTypeParamInquiry(dataRef)) {
        if (auto *comp{parser::Unwrap<parser::StructureComponent>(*dataRef)}) {
          context_.Say(comp->component.source,
              "A variable that is part of another variable cannot appear on the %s clause"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
        }
      }
    }
  }};

  for (const auto &object : objects.v) {
    common::visit(
        common::visitors{
            CheckComponent,
            [&](const parser::Name &name) {},
        },
        object.u);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Update &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_update);
  llvm::omp::Directive dir{GetContext().directive};
  unsigned version{context_.langOptions().OpenMPVersion};

  const parser::OmpDependenceType *depType{nullptr};
  const parser::OmpTaskDependenceType *taskType{nullptr};
  if (auto &maybeUpdate{x.v}) {
    depType = std::get_if<parser::OmpDependenceType>(&maybeUpdate->u);
    taskType = std::get_if<parser::OmpTaskDependenceType>(&maybeUpdate->u);
  }

  if (!depType && !taskType) {
    assert(dir == llvm::omp::Directive::OMPD_atomic &&
        "Unexpected alternative in update clause");
    return;
  }

  if (depType) {
    CheckDependenceType(depType->v);
  } else if (taskType) {
    CheckTaskDependenceType(taskType->v);
  }

  // [5.1:288:4-5]
  // An update clause on a depobj construct must not have source, sink or depobj
  // as dependence-type.
  // [5.2:322:3]
  // task-dependence-type must not be depobj.
  if (dir == llvm::omp::OMPD_depobj) {
    if (version >= 51) {
      bool invalidDep{false};
      if (taskType) {
        invalidDep =
            taskType->v == parser::OmpTaskDependenceType::Value::Depobj;
      } else {
        invalidDep = true;
      }
      if (invalidDep) {
        context_.Say(GetContext().clauseSource,
            "An UPDATE clause on a DEPOBJ construct must not have SINK, SOURCE or DEPOBJ as dependence type"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::UseDevicePtr &x) {
  CheckStructureComponent(x.v, llvm::omp::Clause::OMPC_use_device_ptr);
  CheckAllowedClause(llvm::omp::Clause::OMPC_use_device_ptr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_use_device_ptr)) {
    const auto &useDevicePtrClause{
        std::get<parser::OmpClause::UseDevicePtr>(clause->u)};
    const auto &useDevicePtrList{useDevicePtrClause.v};
    std::list<parser::Name> useDevicePtrNameList;
    for (const auto &ompObject : useDevicePtrList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          if (!(IsBuiltinCPtr(*(name->symbol)))) {
            context_.Warn(common::UsageWarning::OpenMPUsage, clause->source,
                "Use of non-C_PTR type '%s' in USE_DEVICE_PTR is deprecated, use USE_DEVICE_ADDR instead"_warn_en_US,
                name->ToString());
          } else {
            useDevicePtrNameList.push_back(*name);
          }
        }
      }
    }
    CheckMultipleOccurrence(
        listVars, useDevicePtrNameList, clause->source, "USE_DEVICE_PTR");
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::UseDeviceAddr &x) {
  CheckStructureComponent(x.v, llvm::omp::Clause::OMPC_use_device_addr);
  CheckAllowedClause(llvm::omp::Clause::OMPC_use_device_addr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] :
      FindClauses(llvm::omp::Clause::OMPC_use_device_addr)) {
    const auto &useDeviceAddrClause{
        std::get<parser::OmpClause::UseDeviceAddr>(clause->u)};
    const auto &useDeviceAddrList{useDeviceAddrClause.v};
    std::list<parser::Name> useDeviceAddrNameList;
    for (const auto &ompObject : useDeviceAddrList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          useDeviceAddrNameList.push_back(*name);
        }
      }
    }
    CheckMultipleOccurrence(
        listVars, useDeviceAddrNameList, clause->source, "USE_DEVICE_ADDR");
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::IsDevicePtr &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_is_device_ptr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] : FindClauses(llvm::omp::Clause::OMPC_is_device_ptr)) {
    const auto &isDevicePtrClause{
        std::get<parser::OmpClause::IsDevicePtr>(clause->u)};
    const auto &isDevicePtrList{isDevicePtrClause.v};
    SymbolSourceMap currSymbols;
    GetSymbolsInObjectList(isDevicePtrList, currSymbols);
    for (auto &[symbol, source] : currSymbols) {
      if (!(IsBuiltinCPtr(*symbol))) {
        context_.Say(clause->source,
            "Variable '%s' in IS_DEVICE_PTR clause must be of type C_PTR"_err_en_US,
            source.ToString());
      } else if (!(IsDummy(*symbol))) {
        context_.Warn(common::UsageWarning::OpenMPUsage, clause->source,
            "Variable '%s' in IS_DEVICE_PTR clause must be a dummy argument. "
            "This semantic check is deprecated from OpenMP 5.2 and later."_warn_en_US,
            source.ToString());
      } else if (IsAllocatableOrPointer(*symbol) || IsValue(*symbol)) {
        context_.Warn(common::UsageWarning::OpenMPUsage, clause->source,
            "Variable '%s' in IS_DEVICE_PTR clause must be a dummy argument "
            "that does not have the ALLOCATABLE, POINTER or VALUE attribute. "
            "This semantic check is deprecated from OpenMP 5.2 and later."_warn_en_US,
            source.ToString());
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::HasDeviceAddr &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_has_device_addr);
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  semantics::UnorderedSymbolSet listVars;
  for (auto [_, clause] :
      FindClauses(llvm::omp::Clause::OMPC_has_device_addr)) {
    const auto &hasDeviceAddrClause{
        std::get<parser::OmpClause::HasDeviceAddr>(clause->u)};
    const auto &hasDeviceAddrList{hasDeviceAddrClause.v};
    std::list<parser::Name> hasDeviceAddrNameList;
    for (const auto &ompObject : hasDeviceAddrList.v) {
      if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
        if (name->symbol) {
          hasDeviceAddrNameList.push_back(*name);
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Enter &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_enter);
  const parser::OmpObjectList &objList{x.v};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objList, symbols);
  for (const auto &[symbol, source] : symbols) {
    if (!IsExtendedListItem(*symbol)) {
      context_.SayWithDecl(*symbol, source,
          "'%s' must be a variable or a procedure"_err_en_US, symbol->name());
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::From &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_from);
  if (!OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_from, GetContext().clauseSource, context_)) {
    return;
  }

  auto &modifiers{OmpGetModifiers(x.v)};
  unsigned version{context_.langOptions().OpenMPVersion};

  if (auto *iter{OmpGetUniqueModifier<parser::OmpIterator>(modifiers)}) {
    CheckIteratorModifier(*iter);
  }

  const auto &objList{std::get<parser::OmpObjectList>(x.v.t)};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objList, symbols);
  CheckVariableListItem(symbols);

  // Ref: [4.5:109:19]
  // If a list item is an array section it must specify contiguous storage.
  if (version <= 45) {
    for (const parser::OmpObject &object : objList.v) {
      CheckIfContiguous(object);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::To &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_to);
  if (!OmpVerifyModifiers(
          x.v, llvm::omp::OMPC_to, GetContext().clauseSource, context_)) {
    return;
  }

  auto &modifiers{OmpGetModifiers(x.v)};
  unsigned version{context_.langOptions().OpenMPVersion};

  // The "to" clause is only allowed on "declare target" (pre-5.1), and
  // "target update". In the former case it can take an extended list item,
  // in the latter a variable (a locator).

  // The "declare target" construct (and the "to" clause on it) are already
  // handled (in the declare-target checkers), so just look at "to" in "target
  // update".
  if (GetContext().directive == llvm::omp::OMPD_declare_target) {
    return;
  }

  assert(GetContext().directive == llvm::omp::OMPD_target_update);
  if (auto *iter{OmpGetUniqueModifier<parser::OmpIterator>(modifiers)}) {
    CheckIteratorModifier(*iter);
  }

  const auto &objList{std::get<parser::OmpObjectList>(x.v.t)};
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objList, symbols);
  CheckVariableListItem(symbols);

  // Ref: [4.5:109:19]
  // If a list item is an array section it must specify contiguous storage.
  if (version <= 45) {
    for (const parser::OmpObject &object : objList.v) {
      CheckIfContiguous(object);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::OmpxBare &x) {
  // Don't call CheckAllowedClause, because it allows "ompx_bare" on
  // a non-combined "target" directive (for reasons of splitting combined
  // directives). In source code it's only allowed on "target teams".
  if (GetContext().directive != llvm::omp::Directive::OMPD_target_teams) {
    context_.Say(GetContext().clauseSource,
        "%s clause is only allowed on combined TARGET TEAMS"_err_en_US,
        parser::ToUpperCaseLetters(getClauseName(llvm::omp::OMPC_ompx_bare)));
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::When &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_when);
  OmpVerifyModifiers(
      x.v, llvm::omp::OMPC_when, GetContext().clauseSource, context_);
}

void OmpStructureChecker::Enter(const parser::OmpContextSelector &ctx) {
  EnterDirectiveNest(ContextSelectorNest);

  using SetName = parser::OmpTraitSetSelectorName;
  std::map<SetName::Value, const SetName *> visited;

  for (const parser::OmpTraitSetSelector &traitSet : ctx.v) {
    auto &name{std::get<SetName>(traitSet.t)};
    auto [prev, unique]{visited.insert(std::make_pair(name.v, &name))};
    if (!unique) {
      std::string showName{parser::ToUpperCaseLetters(name.ToString())};
      parser::MessageFormattedText txt(
          "Repeated trait set name %s in a context specifier"_err_en_US,
          showName);
      parser::Message message(name.source, txt);
      message.Attach(prev->second->source,
          "Previous trait set %s provided here"_en_US, showName);
      context_.Say(std::move(message));
    }
    CheckTraitSetSelector(traitSet);
  }
}

void OmpStructureChecker::Leave(const parser::OmpContextSelector &) {
  ExitDirectiveNest(ContextSelectorNest);
}

const std::list<parser::OmpTraitProperty> &
OmpStructureChecker::GetTraitPropertyList(
    const parser::OmpTraitSelector &trait) {
  static const std::list<parser::OmpTraitProperty> empty{};
  auto &[_, maybeProps]{trait.t};
  if (maybeProps) {
    using PropertyList = std::list<parser::OmpTraitProperty>;
    return std::get<PropertyList>(maybeProps->t);
  } else {
    return empty;
  }
}

std::optional<llvm::omp::Clause> OmpStructureChecker::GetClauseFromProperty(
    const parser::OmpTraitProperty &property) {
  using MaybeClause = std::optional<llvm::omp::Clause>;

  // The parser for OmpClause will only succeed if the clause was
  // given with all required arguments.
  // If this is a string or complex extension with a clause name,
  // treat it as a clause and let the trait checker deal with it.

  auto getClauseFromString{[&](const std::string &s) -> MaybeClause {
    auto id{llvm::omp::getOpenMPClauseKind(parser::ToLowerCaseLetters(s))};
    if (id != llvm::omp::Clause::OMPC_unknown) {
      return id;
    } else {
      return std::nullopt;
    }
  }};

  return common::visit( //
      common::visitors{
          [&](const parser::OmpTraitPropertyName &x) -> MaybeClause {
            return getClauseFromString(x.v);
          },
          [&](const common::Indirection<parser::OmpClause> &x) -> MaybeClause {
            return x.value().Id();
          },
          [&](const parser::ScalarExpr &x) -> MaybeClause {
            return std::nullopt;
          },
          [&](const parser::OmpTraitPropertyExtension &x) -> MaybeClause {
            using ExtProperty = parser::OmpTraitPropertyExtension;
            if (auto *name{std::get_if<parser::OmpTraitPropertyName>(&x.u)}) {
              return getClauseFromString(name->v);
            } else if (auto *cpx{std::get_if<ExtProperty::Complex>(&x.u)}) {
              return getClauseFromString(
                  std::get<parser::OmpTraitPropertyName>(cpx->t).v);
            }
            return std::nullopt;
          },
      },
      property.u);
}

void OmpStructureChecker::CheckTraitSelectorList(
    const std::list<parser::OmpTraitSelector> &traits) {
  // [6.0:322:20]
  // Each trait-selector-name may only be specified once in a trait selector
  // set.

  // Cannot store OmpTraitSelectorName directly, because it's not copyable.
  using TraitName = parser::OmpTraitSelectorName;
  using BareName = decltype(TraitName::u);
  std::map<BareName, const TraitName *> visited;

  for (const parser::OmpTraitSelector &trait : traits) {
    auto &name{std::get<TraitName>(trait.t)};

    auto [prev, unique]{visited.insert(std::make_pair(name.u, &name))};
    if (!unique) {
      std::string showName{parser::ToUpperCaseLetters(name.ToString())};
      parser::MessageFormattedText txt(
          "Repeated trait name %s in a trait set"_err_en_US, showName);
      parser::Message message(name.source, txt);
      message.Attach(prev->second->source,
          "Previous trait %s provided here"_en_US, showName);
      context_.Say(std::move(message));
    }
  }
}

void OmpStructureChecker::CheckTraitSetSelector(
    const parser::OmpTraitSetSelector &traitSet) {

  // Trait Set      |           Allowed traits | D-traits | X-traits | Score |
  //
  // Construct      |     Simd, directive-name |      Yes |       No |    No |
  // Device         |          Arch, Isa, Kind |       No |      Yes |    No |
  // Implementation | Atomic_Default_Mem_Order |       No |      Yes |   Yes |
  //                |      Extension, Requires |          |          |       |
  //                |                   Vendor |          |          |       |
  // Target_Device  |    Arch, Device_Num, Isa |       No |      Yes |    No |
  //                |                Kind, Uid |          |          |       |
  // User           |                Condition |       No |       No |   Yes |

  struct TraitSetConfig {
    std::set<parser::OmpTraitSelectorName::Value> allowed;
    bool allowsDirectiveTraits;
    bool allowsExtensionTraits;
    bool allowsScore;
  };

  using SName = parser::OmpTraitSetSelectorName::Value;
  using TName = parser::OmpTraitSelectorName::Value;

  static const std::map<SName, TraitSetConfig> configs{
      {SName::Construct, //
          {{TName::Simd}, true, false, false}},
      {SName::Device, //
          {{TName::Arch, TName::Isa, TName::Kind}, false, true, false}},
      {SName::Implementation, //
          {{TName::Atomic_Default_Mem_Order, TName::Extension, TName::Requires,
               TName::Vendor},
              false, true, true}},
      {SName::Target_Device, //
          {{TName::Arch, TName::Device_Num, TName::Isa, TName::Kind,
               TName::Uid},
              false, true, false}},
      {SName::User, //
          {{TName::Condition}, false, false, true}},
  };

  auto checkTraitSet{[&](const TraitSetConfig &config) {
    auto &[setName, traits]{traitSet.t};
    auto usn{parser::ToUpperCaseLetters(setName.ToString())};

    // Check if there are any duplicate traits.
    CheckTraitSelectorList(traits);

    for (const parser::OmpTraitSelector &trait : traits) {
      // Don't use structured bindings here, because they cannot be captured
      // before C++20.
      auto &traitName = std::get<parser::OmpTraitSelectorName>(trait.t);
      auto &maybeProps =
          std::get<std::optional<parser::OmpTraitSelector::Properties>>(
              trait.t);

      // Check allowed traits
      common::visit( //
          common::visitors{
              [&](parser::OmpTraitSelectorName::Value v) {
                if (!config.allowed.count(v)) {
                  context_.Say(traitName.source,
                      "%s is not a valid trait for %s trait set"_err_en_US,
                      parser::ToUpperCaseLetters(traitName.ToString()), usn);
                }
              },
              [&](llvm::omp::Directive) {
                if (!config.allowsDirectiveTraits) {
                  context_.Say(traitName.source,
                      "Directive name is not a valid trait for %s trait set"_err_en_US,
                      usn);
                }
              },
              [&](const std::string &) {
                if (!config.allowsExtensionTraits) {
                  context_.Say(traitName.source,
                      "Extension traits are not valid for %s trait set"_err_en_US,
                      usn);
                }
              },
          },
          traitName.u);

      // Check score
      if (maybeProps) {
        auto &[maybeScore, _]{maybeProps->t};
        if (maybeScore) {
          CheckTraitScore(*maybeScore);
        }
      }

      // Check the properties of the individual traits
      CheckTraitSelector(traitSet, trait);
    }
  }};

  checkTraitSet(
      configs.at(std::get<parser::OmpTraitSetSelectorName>(traitSet.t).v));
}

void OmpStructureChecker::CheckTraitScore(const parser::OmpTraitScore &score) {
  // [6.0:322:23]
  // A score-expression must be a non-negative constant integer expression.
  if (auto value{GetIntValue(score)}; !value || value < 0) {
    context_.Say(score.source,
        "SCORE expression must be a non-negative constant integer expression"_err_en_US);
  }
}

bool OmpStructureChecker::VerifyTraitPropertyLists(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  using TraitName = parser::OmpTraitSelectorName;
  using PropertyList = std::list<parser::OmpTraitProperty>;
  auto &[traitName, maybeProps]{trait.t};

  auto checkPropertyList{[&](const PropertyList &properties, auto isValid,
                             const std::string &message) {
    bool foundInvalid{false};
    for (const parser::OmpTraitProperty &prop : properties) {
      if (!isValid(prop)) {
        if (foundInvalid) {
          context_.Say(
              prop.source, "More invalid properties are present"_err_en_US);
          break;
        }
        context_.Say(prop.source, "%s"_err_en_US, message);
        foundInvalid = true;
      }
    }
    return !foundInvalid;
  }};

  bool invalid{false};

  if (std::holds_alternative<llvm::omp::Directive>(traitName.u)) {
    // Directive-name traits don't have properties.
    if (maybeProps) {
      context_.Say(trait.source,
          "Directive-name traits cannot have properties"_err_en_US);
      invalid = true;
    }
  }
  // Ignore properties on extension traits.

  // See `TraitSelectorParser` in openmp-parser.cpp
  if (auto *v{std::get_if<TraitName::Value>(&traitName.u)}) {
    switch (*v) {
    // name-list properties
    case parser::OmpTraitSelectorName::Value::Arch:
    case parser::OmpTraitSelectorName::Value::Extension:
    case parser::OmpTraitSelectorName::Value::Isa:
    case parser::OmpTraitSelectorName::Value::Kind:
    case parser::OmpTraitSelectorName::Value::Uid:
    case parser::OmpTraitSelectorName::Value::Vendor:
      if (maybeProps) {
        auto isName{[](const parser::OmpTraitProperty &prop) {
          return std::holds_alternative<parser::OmpTraitPropertyName>(prop.u);
        }};
        invalid = !checkPropertyList(std::get<PropertyList>(maybeProps->t),
            isName, "Trait property should be a name");
      }
      break;
    // clause-list
    case parser::OmpTraitSelectorName::Value::Atomic_Default_Mem_Order:
    case parser::OmpTraitSelectorName::Value::Requires:
    case parser::OmpTraitSelectorName::Value::Simd:
      if (maybeProps) {
        auto isClause{[&](const parser::OmpTraitProperty &prop) {
          return GetClauseFromProperty(prop).has_value();
        }};
        invalid = !checkPropertyList(std::get<PropertyList>(maybeProps->t),
            isClause, "Trait property should be a clause");
      }
      break;
    // expr-list
    case parser::OmpTraitSelectorName::Value::Condition:
    case parser::OmpTraitSelectorName::Value::Device_Num:
      if (maybeProps) {
        auto isExpr{[](const parser::OmpTraitProperty &prop) {
          return std::holds_alternative<parser::ScalarExpr>(prop.u);
        }};
        invalid = !checkPropertyList(std::get<PropertyList>(maybeProps->t),
            isExpr, "Trait property should be a scalar expression");
      }
      break;
    } // switch
  }

  return !invalid;
}

void OmpStructureChecker::CheckTraitSelector(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  using TraitName = parser::OmpTraitSelectorName;
  auto &[traitName, maybeProps]{trait.t};

  // Only do the detailed checks if the property lists are valid.
  if (VerifyTraitPropertyLists(traitSet, trait)) {
    if (std::holds_alternative<llvm::omp::Directive>(traitName.u) ||
        std::holds_alternative<std::string>(traitName.u)) {
      // No properties here: directives don't have properties, and
      // we don't implement any extension traits now.
      return;
    }

    // Specific traits we want to check.
    // Limitations:
    // (1) The properties for these traits are defined in "Additional
    // Definitions for the OpenMP API Specification". It's not clear how
    // to define them in a portable way, and how to verify their validity,
    // especially if they get replaced by their integer values (in case
    // they are defined as enums).
    // (2) These are entirely implementation-defined, and at the moment
    // there is no known schema to validate these values.
    auto v{std::get<TraitName::Value>(traitName.u)};
    switch (v) {
    case TraitName::Value::Arch:
      // Unchecked, TBD(1)
      break;
    case TraitName::Value::Atomic_Default_Mem_Order:
      CheckTraitADMO(traitSet, trait);
      break;
    case TraitName::Value::Condition:
      CheckTraitCondition(traitSet, trait);
      break;
    case TraitName::Value::Device_Num:
      CheckTraitDeviceNum(traitSet, trait);
      break;
    case TraitName::Value::Extension:
      // Ignore
      break;
    case TraitName::Value::Isa:
      // Unchecked, TBD(1)
      break;
    case TraitName::Value::Kind:
      // Unchecked, TBD(1)
      break;
    case TraitName::Value::Requires:
      CheckTraitRequires(traitSet, trait);
      break;
    case TraitName::Value::Simd:
      CheckTraitSimd(traitSet, trait);
      break;
    case TraitName::Value::Uid:
      // Unchecked, TBD(2)
      break;
    case TraitName::Value::Vendor:
      // Unchecked, TBD(1)
      break;
    }
  }
}

void OmpStructureChecker::CheckTraitADMO(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  if (properties.size() != 1) {
    context_.Say(trait.source,
        "%s trait requires a single clause property"_err_en_US,
        parser::ToUpperCaseLetters(traitName.ToString()));
  } else {
    const parser::OmpTraitProperty &property{properties.front()};
    auto clauseId{*GetClauseFromProperty(property)};
    // Check that the clause belongs to the memory-order clause-set.
    // Clause sets will hopefully be autogenerated at some point.
    switch (clauseId) {
    case llvm::omp::Clause::OMPC_acq_rel:
    case llvm::omp::Clause::OMPC_acquire:
    case llvm::omp::Clause::OMPC_relaxed:
    case llvm::omp::Clause::OMPC_release:
    case llvm::omp::Clause::OMPC_seq_cst:
      break;
    default:
      context_.Say(property.source,
          "%s trait requires a clause from the memory-order clause set"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()));
    }

    using ClauseProperty = common::Indirection<parser::OmpClause>;
    if (!std::holds_alternative<ClauseProperty>(property.u)) {
      context_.Say(property.source,
          "Invalid clause specification for %s"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId)));
    }
  }
}

void OmpStructureChecker::CheckTraitCondition(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  if (properties.size() != 1) {
    context_.Say(trait.source,
        "%s trait requires a single expression property"_err_en_US,
        parser::ToUpperCaseLetters(traitName.ToString()));
  } else {
    const parser::OmpTraitProperty &property{properties.front()};
    auto &scalarExpr{std::get<parser::ScalarExpr>(property.u)};

    auto maybeType{GetDynamicType(scalarExpr.thing.value())};
    if (!maybeType || maybeType->category() != TypeCategory::Logical) {
      context_.Say(property.source,
          "%s trait requires a single LOGICAL expression"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()));
    }
  }
}

void OmpStructureChecker::CheckTraitDeviceNum(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  if (properties.size() != 1) {
    context_.Say(trait.source,
        "%s trait requires a single expression property"_err_en_US,
        parser::ToUpperCaseLetters(traitName.ToString()));
  }
  // No other checks at the moment.
}

void OmpStructureChecker::CheckTraitRequires(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  unsigned version{context_.langOptions().OpenMPVersion};
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  for (const parser::OmpTraitProperty &property : properties) {
    auto clauseId{*GetClauseFromProperty(property)};
    if (!llvm::omp::isAllowedClauseForDirective(
            llvm::omp::OMPD_requires, clauseId, version)) {
      context_.Say(property.source,
          "%s trait requires a clause from the requirement clause set"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()));
    }

    using ClauseProperty = common::Indirection<parser::OmpClause>;
    if (!std::holds_alternative<ClauseProperty>(property.u)) {
      context_.Say(property.source,
          "Invalid clause specification for %s"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId)));
    }
  }
}

void OmpStructureChecker::CheckTraitSimd(
    const parser::OmpTraitSetSelector &traitSet,
    const parser::OmpTraitSelector &trait) {
  unsigned version{context_.langOptions().OpenMPVersion};
  auto &traitName{std::get<parser::OmpTraitSelectorName>(trait.t)};
  auto &properties{GetTraitPropertyList(trait)};

  for (const parser::OmpTraitProperty &property : properties) {
    auto clauseId{*GetClauseFromProperty(property)};
    if (!llvm::omp::isAllowedClauseForDirective(
            llvm::omp::OMPD_declare_simd, clauseId, version)) {
      context_.Say(property.source,
          "%s trait requires a clause that is allowed on the %s directive"_err_en_US,
          parser::ToUpperCaseLetters(traitName.ToString()),
          parser::ToUpperCaseLetters(
              getDirectiveName(llvm::omp::OMPD_declare_simd)));
    }

    using ClauseProperty = common::Indirection<parser::OmpClause>;
    if (!std::holds_alternative<ClauseProperty>(property.u)) {
      context_.Say(property.source,
          "Invalid clause specification for %s"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId)));
    }
  }
}

llvm::StringRef OmpStructureChecker::getClauseName(llvm::omp::Clause clause) {
  return llvm::omp::getOpenMPClauseName(clause);
}

llvm::StringRef OmpStructureChecker::getDirectiveName(
    llvm::omp::Directive directive) {
  unsigned version{context_.langOptions().OpenMPVersion};
  return llvm::omp::getOpenMPDirectiveName(directive, version);
}

const Symbol *OmpStructureChecker::GetObjectSymbol(
    const parser::OmpObject &object) {
  // Some symbols may be missing if the resolution failed, e.g. when an
  // undeclared name is used with implicit none.
  if (auto *name{std::get_if<parser::Name>(&object.u)}) {
    return name->symbol ? &name->symbol->GetUltimate() : nullptr;
  } else if (auto *desg{std::get_if<parser::Designator>(&object.u)}) {
    auto &last{GetLastName(*desg)};
    return last.symbol ? &GetLastName(*desg).symbol->GetUltimate() : nullptr;
  }
  return nullptr;
}

const Symbol *OmpStructureChecker::GetArgumentSymbol(
    const parser::OmpArgument &argument) {
  if (auto *locator{std::get_if<parser::OmpLocator>(&argument.u)}) {
    if (auto *object{std::get_if<parser::OmpObject>(&locator->u)}) {
      return GetObjectSymbol(*object);
    }
  }
  return nullptr;
}

std::optional<parser::CharBlock> OmpStructureChecker::GetObjectSource(
    const parser::OmpObject &object) {
  if (auto *name{std::get_if<parser::Name>(&object.u)}) {
    return name->source;
  } else if (auto *desg{std::get_if<parser::Designator>(&object.u)}) {
    return GetLastName(*desg).source;
  }
  return std::nullopt;
}

void OmpStructureChecker::CheckDependList(const parser::DataRef &d) {
  common::visit(
      common::visitors{
          [&](const common::Indirection<parser::ArrayElement> &elem) {
            // Check if the base element is valid on Depend Clause
            CheckDependList(elem.value().base);
          },
          [&](const common::Indirection<parser::StructureComponent> &comp) {
            CheckDependList(comp.value().base);
          },
          [&](const common::Indirection<parser::CoindexedNamedObject> &) {
            context_.Say(GetContext().clauseSource,
                "Coarrays are not supported in DEPEND clause"_err_en_US);
          },
          [&](const parser::Name &) {},
      },
      d.u);
}

// Called from both Reduction and Depend clause.
void OmpStructureChecker::CheckArraySection(
    const parser::ArrayElement &arrayElement, const parser::Name &name,
    const llvm::omp::Clause clause) {
  // Sometimes substring operations are incorrectly parsed as array accesses.
  // Detect this by looking for array accesses on character variables which are
  // not arrays.
  bool isSubstring{false};
  evaluate::ExpressionAnalyzer ea{context_};
  if (MaybeExpr expr = ea.Analyze(arrayElement.base)) {
    std::optional<evaluate::Shape> shape = evaluate::GetShape(expr);
    // Not an array: rank 0
    if (shape && shape->size() == 0) {
      if (std::optional<evaluate::DynamicType> type = expr->GetType()) {
        if (type->category() == evaluate::TypeCategory::Character) {
          // Substrings are explicitly denied by the standard [6.0:163:9-11].
          // This is supported as an extension. This restriction was added in
          // OpenMP 5.2.
          isSubstring = true;
          context_.Say(GetContext().clauseSource,
              "The use of substrings in OpenMP argument lists has been disallowed since OpenMP 5.2."_port_en_US);
        } else {
          llvm_unreachable("Array indexing on a variable that isn't an array");
        }
      }
    }
  }
  if (!arrayElement.subscripts.empty()) {
    for (const auto &subscript : arrayElement.subscripts) {
      if (const auto *triplet{
              std::get_if<parser::SubscriptTriplet>(&subscript.u)}) {
        if (std::get<0>(triplet->t) && std::get<1>(triplet->t)) {
          std::optional<int64_t> strideVal{std::nullopt};
          if (const auto &strideExpr = std::get<2>(triplet->t)) {
            // OpenMP 6.0 Section 5.2.5: Array Sections
            // Restrictions: if a stride expression is specified it must be
            // positive. A stride of 0 doesn't make sense.
            strideVal = GetIntValue(strideExpr);
            if (strideVal && *strideVal < 1) {
              context_.Say(GetContext().clauseSource,
                  "'%s' in %s clause must have a positive stride"_err_en_US,
                  name.ToString(),
                  parser::ToUpperCaseLetters(getClauseName(clause).str()));
            }
            if (isSubstring) {
              context_.Say(GetContext().clauseSource,
                  "Cannot specify a step for a substring"_err_en_US);
            }
          }
          const auto &lower{std::get<0>(triplet->t)};
          const auto &upper{std::get<1>(triplet->t)};
          if (lower && upper) {
            const auto lval{GetIntValue(lower)};
            const auto uval{GetIntValue(upper)};
            if (lval && uval) {
              int64_t sectionLen = *uval - *lval;
              if (strideVal) {
                sectionLen = sectionLen / *strideVal;
              }

              if (sectionLen < 1) {
                context_.Say(GetContext().clauseSource,
                    "'%s' in %s clause"
                    " is a zero size array section"_err_en_US,
                    name.ToString(),
                    parser::ToUpperCaseLetters(getClauseName(clause).str()));
                break;
              }
            }
          }
        }
      } else if (std::get_if<parser::IntExpr>(&subscript.u)) {
        // base(n) is valid as an array index but not as a substring operation
        if (isSubstring) {
          context_.Say(GetContext().clauseSource,
              "Substrings must be in the form parent-string(lb:ub)"_err_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckIntentInPointer(
    SymbolSourceMap &symbols, llvm::omp::Clause clauseId) {
  for (auto &[symbol, source] : symbols) {
    if (IsPointer(*symbol) && IsIntentIn(*symbol)) {
      context_.Say(source,
          "Pointer '%s' with the INTENT(IN) attribute may not appear in a %s clause"_err_en_US,
          symbol->name(),
          parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    }
  }
}

void OmpStructureChecker::CheckProcedurePointer(
    SymbolSourceMap &symbols, llvm::omp::Clause clause) {
  for (const auto &[symbol, source] : symbols) {
    if (IsProcedurePointer(*symbol)) {
      context_.Say(source,
          "Procedure pointer '%s' may not appear in a %s clause"_err_en_US,
          symbol->name(),
          parser::ToUpperCaseLetters(getClauseName(clause).str()));
    }
  }
}

void OmpStructureChecker::CheckCrayPointee(
    const parser::OmpObjectList &objectList, llvm::StringRef clause,
    bool suggestToUseCrayPointer) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objectList, symbols);
  for (auto it{symbols.begin()}; it != symbols.end(); ++it) {
    const auto *symbol{it->first};
    const auto source{it->second};
    if (symbol->test(Symbol::Flag::CrayPointee)) {
      std::string suggestionMsg = "";
      if (suggestToUseCrayPointer)
        suggestionMsg = ", use Cray Pointer '" +
            semantics::GetCrayPointer(*symbol).name().ToString() + "' instead";
      context_.Say(source,
          "Cray Pointee '%s' may not appear in %s clause%s"_err_en_US,
          symbol->name(), clause.str(), suggestionMsg);
    }
  }
}

void OmpStructureChecker::GetSymbolsInObjectList(
    const parser::OmpObjectList &objectList, SymbolSourceMap &symbols) {
  for (const auto &ompObject : objectList.v) {
    if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (const auto *symbol{name->symbol}) {
        if (const auto *commonBlockDetails{
                symbol->detailsIf<CommonBlockDetails>()}) {
          for (const auto &object : commonBlockDetails->objects()) {
            symbols.emplace(&object->GetUltimate(), name->source);
          }
        } else {
          symbols.emplace(&symbol->GetUltimate(), name->source);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckDefinableObjects(
    SymbolSourceMap &symbols, const llvm::omp::Clause clause) {
  for (auto &[symbol, source] : symbols) {
    if (auto msg{WhyNotDefinable(source, context_.FindScope(source),
            DefinabilityFlags{}, *symbol)}) {
      context_
          .Say(source,
              "Variable '%s' on the %s clause is not definable"_err_en_US,
              symbol->name(),
              parser::ToUpperCaseLetters(getClauseName(clause).str()))
          .Attach(std::move(msg->set_severity(parser::Severity::Because)));
    }
  }
}

void OmpStructureChecker::CheckPrivateSymbolsInOuterCxt(
    SymbolSourceMap &currSymbols, DirectivesClauseTriple &dirClauseTriple,
    const llvm::omp::Clause currClause) {
  SymbolSourceMap enclosingSymbols;
  auto range{dirClauseTriple.equal_range(GetContext().directive)};
  for (auto dirIter{range.first}; dirIter != range.second; ++dirIter) {
    auto enclosingDir{dirIter->second.first};
    auto enclosingClauseSet{dirIter->second.second};
    if (auto *enclosingContext{GetEnclosingContextWithDir(enclosingDir)}) {
      for (auto it{enclosingContext->clauseInfo.begin()};
          it != enclosingContext->clauseInfo.end(); ++it) {
        if (enclosingClauseSet.test(it->first)) {
          if (const auto *ompObjectList{GetOmpObjectList(*it->second)}) {
            GetSymbolsInObjectList(*ompObjectList, enclosingSymbols);
          }
        }
      }

      // Check if the symbols in current context are private in outer context
      for (auto &[symbol, source] : currSymbols) {
        if (enclosingSymbols.find(symbol) != enclosingSymbols.end()) {
          context_.Say(source,
              "%s variable '%s' is PRIVATE in outer context"_err_en_US,
              parser::ToUpperCaseLetters(getClauseName(currClause).str()),
              symbol->name());
        }
      }
    }
  }
}

bool OmpStructureChecker::CheckTargetBlockOnlyTeams(
    const parser::Block &block) {
  bool nestedTeams{false};

  if (!block.empty()) {
    auto it{block.begin()};
    if (const auto *ompConstruct{
            parser::Unwrap<parser::OpenMPConstruct>(*it)}) {
      if (const auto *ompBlockConstruct{
              std::get_if<parser::OpenMPBlockConstruct>(&ompConstruct->u)}) {
        const auto &beginBlockDir{
            std::get<parser::OmpBeginBlockDirective>(ompBlockConstruct->t)};
        const auto &beginDir{
            std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
        if (beginDir.v == llvm::omp::Directive::OMPD_teams) {
          nestedTeams = true;
        }
      }
    }

    if (nestedTeams && ++it == block.end()) {
      return true;
    }
  }

  return false;
}

void OmpStructureChecker::CheckWorkshareBlockStmts(
    const parser::Block &block, parser::CharBlock source) {
  OmpWorkshareBlockChecker ompWorkshareBlockChecker{context_, source};

  for (auto it{block.begin()}; it != block.end(); ++it) {
    if (parser::Unwrap<parser::AssignmentStmt>(*it) ||
        parser::Unwrap<parser::ForallStmt>(*it) ||
        parser::Unwrap<parser::ForallConstruct>(*it) ||
        parser::Unwrap<parser::WhereStmt>(*it) ||
        parser::Unwrap<parser::WhereConstruct>(*it)) {
      parser::Walk(*it, ompWorkshareBlockChecker);
    } else if (const auto *ompConstruct{
                   parser::Unwrap<parser::OpenMPConstruct>(*it)}) {
      if (const auto *ompAtomicConstruct{
              std::get_if<parser::OpenMPAtomicConstruct>(&ompConstruct->u)}) {
        // Check if assignment statements in the enclosing OpenMP Atomic
        // construct are allowed in the Workshare construct
        parser::Walk(*ompAtomicConstruct, ompWorkshareBlockChecker);
      } else if (const auto *ompCriticalConstruct{
                     std::get_if<parser::OpenMPCriticalConstruct>(
                         &ompConstruct->u)}) {
        // All the restrictions on the Workshare construct apply to the
        // statements in the enclosing critical constructs
        const auto &criticalBlock{
            std::get<parser::Block>(ompCriticalConstruct->t)};
        CheckWorkshareBlockStmts(criticalBlock, source);
      } else {
        // Check if OpenMP constructs enclosed in the Workshare construct are
        // 'Parallel' constructs
        auto currentDir{llvm::omp::Directive::OMPD_unknown};
        if (const auto *ompBlockConstruct{
                std::get_if<parser::OpenMPBlockConstruct>(&ompConstruct->u)}) {
          const auto &beginBlockDir{
              std::get<parser::OmpBeginBlockDirective>(ompBlockConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
          currentDir = beginDir.v;
        } else if (const auto *ompLoopConstruct{
                       std::get_if<parser::OpenMPLoopConstruct>(
                           &ompConstruct->u)}) {
          const auto &beginLoopDir{
              std::get<parser::OmpBeginLoopDirective>(ompLoopConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
          currentDir = beginDir.v;
        } else if (const auto *ompSectionsConstruct{
                       std::get_if<parser::OpenMPSectionsConstruct>(
                           &ompConstruct->u)}) {
          const auto &beginSectionsDir{
              std::get<parser::OmpBeginSectionsDirective>(
                  ompSectionsConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpSectionsDirective>(beginSectionsDir.t)};
          currentDir = beginDir.v;
        }

        if (!llvm::omp::topParallelSet.test(currentDir)) {
          context_.Say(source,
              "OpenMP constructs enclosed in WORKSHARE construct may consist "
              "of ATOMIC, CRITICAL or PARALLEL constructs only"_err_en_US);
        }
      }
    } else {
      context_.Say(source,
          "The structured block in a WORKSHARE construct may consist of only "
          "SCALAR or ARRAY assignments, FORALL or WHERE statements, "
          "FORALL, WHERE, ATOMIC, CRITICAL or PARALLEL constructs"_err_en_US);
    }
  }
}

void OmpStructureChecker::CheckIfContiguous(const parser::OmpObject &object) {
  if (auto contig{IsContiguous(object)}; contig && !*contig) {
    const parser::Name *name{GetObjectName(object)};
    assert(name && "Expecting name component");
    context_.Say(name->source,
        "Reference to '%s' must be a contiguous object"_err_en_US,
        name->ToString());
  }
}

namespace {
struct NameHelper {
  template <typename T>
  static const parser::Name *Visit(const common::Indirection<T> &x) {
    return Visit(x.value());
  }
  static const parser::Name *Visit(const parser::Substring &x) {
    return Visit(std::get<parser::DataRef>(x.t));
  }
  static const parser::Name *Visit(const parser::ArrayElement &x) {
    return Visit(x.base);
  }
  static const parser::Name *Visit(const parser::Designator &x) {
    return common::visit([](auto &&s) { return Visit(s); }, x.u);
  }
  static const parser::Name *Visit(const parser::DataRef &x) {
    return common::visit([](auto &&s) { return Visit(s); }, x.u);
  }
  static const parser::Name *Visit(const parser::OmpObject &x) {
    return common::visit([](auto &&s) { return Visit(s); }, x.u);
  }
  template <typename T> static const parser::Name *Visit(T &&) {
    return nullptr;
  }
  static const parser::Name *Visit(const parser::Name &x) { return &x; }
};
} // namespace

const parser::Name *OmpStructureChecker::GetObjectName(
    const parser::OmpObject &object) {
  return NameHelper::Visit(object);
}

const parser::OmpObjectList *OmpStructureChecker::GetOmpObjectList(
    const parser::OmpClause &clause) {

  // Clauses with OmpObjectList as its data member
  using MemberObjectListClauses = std::tuple<parser::OmpClause::Copyprivate,
      parser::OmpClause::Copyin, parser::OmpClause::Enter,
      parser::OmpClause::Firstprivate, parser::OmpClause::Link,
      parser::OmpClause::Private, parser::OmpClause::Shared,
      parser::OmpClause::UseDevicePtr, parser::OmpClause::UseDeviceAddr>;

  // Clauses with OmpObjectList in the tuple
  using TupleObjectListClauses = std::tuple<parser::OmpClause::Aligned,
      parser::OmpClause::Allocate, parser::OmpClause::From,
      parser::OmpClause::Lastprivate, parser::OmpClause::Map,
      parser::OmpClause::Reduction, parser::OmpClause::To>;

  // TODO:: Generate the tuples using TableGen.
  // Handle other constructs with OmpObjectList such as OpenMPThreadprivate.
  return common::visit(
      common::visitors{
          [&](const auto &x) -> const parser::OmpObjectList * {
            using Ty = std::decay_t<decltype(x)>;
            if constexpr (common::HasMember<Ty, MemberObjectListClauses>) {
              return &x.v;
            } else if constexpr (common::HasMember<Ty,
                                     TupleObjectListClauses>) {
              return &(std::get<parser::OmpObjectList>(x.v.t));
            } else {
              return nullptr;
            }
          },
      },
      clause.u);
}

void OmpStructureChecker::Enter(
    const parser::OmpClause::AtomicDefaultMemOrder &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_atomic_default_mem_order);
}

void OmpStructureChecker::Enter(const parser::OmpClause::DynamicAllocators &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_dynamic_allocators);
}

void OmpStructureChecker::Enter(const parser::OmpClause::ReverseOffload &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_reverse_offload);
}

void OmpStructureChecker::Enter(const parser::OmpClause::UnifiedAddress &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_unified_address);
}

void OmpStructureChecker::Enter(
    const parser::OmpClause::UnifiedSharedMemory &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_unified_shared_memory);
}

void OmpStructureChecker::Enter(const parser::OmpClause::SelfMaps &x) {
  CheckAllowedRequiresClause(llvm::omp::Clause::OMPC_self_maps);
}

void OmpStructureChecker::Enter(const parser::DoConstruct &x) {
  Base::Enter(x);
  loopStack_.push_back(&x);
}

void OmpStructureChecker::Leave(const parser::DoConstruct &x) {
  assert(!loopStack_.empty() && "Expecting non-empty loop stack");
#ifndef NDEBUG
  const LoopConstruct &top = loopStack_.back();
  auto *doc{std::get_if<const parser::DoConstruct *>(&top)};
  assert(doc != nullptr && *doc == &x && "Mismatched loop constructs");
#endif
  loopStack_.pop_back();
  Base::Leave(x);
}

void OmpStructureChecker::Enter(const parser::OpenMPInteropConstruct &x) {
  bool isDependClauseOccured{false};
  int targetCount{0}, targetSyncCount{0};
  const auto &dir{std::get<parser::OmpDirectiveName>(x.v.t)};
  std::set<const Symbol *> objectSymbolList;
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_interop);
  const auto &clauseList{std::get<std::optional<parser::OmpClauseList>>(x.v.t)};
  for (const auto &clause : clauseList->v) {
    common::visit(
        common::visitors{
            [&](const parser::OmpClause::Init &initClause) {
              if (OmpVerifyModifiers(initClause.v, llvm::omp::OMPC_init,
                      GetContext().directiveSource, context_)) {

                auto &modifiers{OmpGetModifiers(initClause.v)};
                auto &&interopTypeModifier{
                    OmpGetRepeatableModifier<parser::OmpInteropType>(
                        modifiers)};
                for (const auto &it : interopTypeModifier) {
                  if (it->v == parser::OmpInteropType::Value::TargetSync) {
                    ++targetSyncCount;
                  } else {
                    ++targetCount;
                  }
                }
              }
              const auto &interopVar{parser::Unwrap<parser::OmpObject>(
                  std::get<parser::OmpObject>(initClause.v.t))};
              const auto *name{parser::Unwrap<parser::Name>(interopVar)};
              const auto *objectSymbol{name->symbol};
              if (llvm::is_contained(objectSymbolList, objectSymbol)) {
                context_.Say(GetContext().directiveSource,
                    "Each interop-var may be specified for at most one action-clause of each INTEROP construct."_err_en_US);
              } else {
                objectSymbolList.insert(objectSymbol);
              }
            },
            [&](const parser::OmpClause::Depend &dependClause) {
              isDependClauseOccured = true;
            },
            [&](const parser::OmpClause::Destroy &destroyClause) {
              const auto &interopVar{
                  parser::Unwrap<parser::OmpObject>(destroyClause.v)};
              const auto *name{parser::Unwrap<parser::Name>(interopVar)};
              const auto *objectSymbol{name->symbol};
              if (llvm::is_contained(objectSymbolList, objectSymbol)) {
                context_.Say(GetContext().directiveSource,
                    "Each interop-var may be specified for at most one action-clause of each INTEROP construct."_err_en_US);
              } else {
                objectSymbolList.insert(objectSymbol);
              }
            },
            [&](const parser::OmpClause::Use &useClause) {
              const auto &interopVar{
                  parser::Unwrap<parser::OmpObject>(useClause.v)};
              const auto *name{parser::Unwrap<parser::Name>(interopVar)};
              const auto *objectSymbol{name->symbol};
              if (llvm::is_contained(objectSymbolList, objectSymbol)) {
                context_.Say(GetContext().directiveSource,
                    "Each interop-var may be specified for at most one action-clause of each INTEROP construct."_err_en_US);
              } else {
                objectSymbolList.insert(objectSymbol);
              }
            },
            [&](const auto &) {},
        },
        clause.u);
  }
  if (targetCount > 1 || targetSyncCount > 1) {
    context_.Say(GetContext().directiveSource,
        "Each interop-type may be specified at most once."_err_en_US);
  }
  if (isDependClauseOccured && !targetSyncCount) {
    context_.Say(GetContext().directiveSource,
        "A DEPEND clause can only appear on the directive if the interop-type includes TARGETSYNC"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPInteropConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::CheckAllowedRequiresClause(llvmOmpClause clause) {
  CheckAllowedClause(clause);

  if (clause != llvm::omp::Clause::OMPC_atomic_default_mem_order) {
    // Check that it does not appear after a device construct
    if (deviceConstructFound_) {
      context_.Say(GetContext().clauseSource,
          "REQUIRES directive with '%s' clause found lexically after device "
          "construct"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clause).str()));
    }
  }
}

} // namespace Fortran::semantics
