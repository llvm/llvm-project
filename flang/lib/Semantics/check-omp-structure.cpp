//===-- lib/Semantics/check-omp-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"
#include "definable.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/tools.h"
#include <variant>

namespace Fortran::semantics {

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

bool OmpStructureChecker::CheckAllowedClause(llvmOmpClause clause) {
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
      if (!OmpDirectiveSet{llvm::omp::OMPD_teams, llvm::omp::OMPD_target_teams}
              .test(GetContextParent().directive)) {
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
      OmpDirectiveSet{llvm::omp::OMPD_teams, llvm::omp::OMPD_target_teams}.test(
          GetContextParent().directive)) {
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

template <class D>
void OmpStructureChecker::CheckHintClause(
    D *leftOmpClauseList, D *rightOmpClauseList) {
  auto checkForValidHintClause = [&](const D *clauseList) {
    for (const auto &clause : clauseList->v) {
      const parser::OmpClause *ompClause = nullptr;
      if constexpr (std::is_same_v<D, const parser::OmpAtomicClauseList>) {
        ompClause = std::get_if<parser::OmpClause>(&clause.u);
        if (!ompClause)
          continue;
      } else if constexpr (std::is_same_v<D, const parser::OmpClauseList>) {
        ompClause = &clause;
      }
      if (const parser::OmpClause::Hint *hintClause{
              std::get_if<parser::OmpClause::Hint>(&ompClause->u)}) {
        std::optional<std::int64_t> hintValue = GetIntValue(hintClause->v);
        if (hintValue && *hintValue >= 0) {
          /*`omp_sync_hint_nonspeculative` and `omp_lock_hint_speculative`*/
          if ((*hintValue & 0xC) == 0xC
              /*`omp_sync_hint_uncontended` and omp_sync_hint_contended*/
              || (*hintValue & 0x3) == 0x3)
            context_.Say(clause.source,
                "Hint clause value "
                "is not a valid OpenMP synchronization value"_err_en_US);
        } else {
          context_.Say(clause.source,
              "Hint clause must have non-negative constant "
              "integer expression"_err_en_US);
        }
      }
    }
  };

  if (leftOmpClauseList) {
    checkForValidHintClause(leftOmpClauseList);
  }
  if (rightOmpClauseList) {
    checkForValidHintClause(rightOmpClauseList);
  }
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

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  loopStack_.push_back(&x);
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};

  // check matching, End directive is optional
  if (const auto &endLoopDir{
          std::get<std::optional<parser::OmpEndLoopDirective>>(x.t)}) {
    const auto &endDir{
        std::get<parser::OmpLoopDirective>(endLoopDir.value().t)};

    CheckMatching<parser::OmpLoopDirective>(beginDir, endDir);
  }

  PushContextAndClauseSets(beginDir.source, beginDir.v);
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
      llvm::omp::topTeamsSet.test(GetContextParent().directive)) {
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
  // TODO:  Expand the check to include `LOOP` construct as well when it is
  // supported.

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
            if (const auto &simpleConstruct =
                    std::get_if<parser::OpenMPSimpleStandaloneConstruct>(
                        &c.u)) {
              const auto &dir{std::get<parser::OmpSimpleStandaloneDirective>(
                  simpleConstruct->t)};
              if (dir.v == llvm::omp::Directive::OMPD_ordered) {
                const auto &clauses{
                    std::get<parser::OmpClauseList>(simpleConstruct->t)};
                for (const auto &clause : clauses.v) {
                  if (std::get_if<parser::OmpClause::Simd>(&clause.u)) {
                    eligibleSIMD = true;
                    break;
                  }
                }
              } else if (dir.v == llvm::omp::Directive::OMPD_scan) {
                eligibleSIMD = true;
              }
            }
          },
          // Allowing SIMD construct
          [&](const parser::OpenMPLoopConstruct &c) {
            const auto &beginLoopDir{
                std::get<parser::OmpBeginLoopDirective>(c.t)};
            const auto &beginDir{
                std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
            if ((beginDir.v == llvm::omp::Directive::OMPD_simd) ||
                (beginDir.v == llvm::omp::Directive::OMPD_do_simd)) {
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
                      const auto &dir{
                          std::get<parser::OmpSimpleStandaloneDirective>(c.t)};
                      if (dir.v == llvm::omp::Directive::OMPD_target_update ||
                          dir.v ==
                              llvm::omp::Directive::OMPD_target_enter_data ||
                          dir.v ==
                              llvm::omp::Directive::OMPD_target_exit_data) {
                        eligibleTarget = false;
                        ineligibleTargetDir = dir.v;
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
    if (llvm::omp::topTeamsSet.test(GetContextParent().directive)) {
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
            [&](const parser::Name &) {}, // common block
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
  CheckIsVarPartOfAnotherVar(dir.source, objectList);
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

void OmpStructureChecker::Enter(const parser::OpenMPDepobjConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_depobj);

  // [5.2:73:27-28]
  // If the destroy clause appears on a depobj construct, destroy-var must
  // refer to the same depend object as the depobj argument of the construct.
  auto &clause{std::get<parser::OmpClause>(x.t)};
  if (clause.Id() == llvm::omp::Clause::OMPC_destroy) {
    auto getSymbol{[&](const parser::OmpObject &obj) {
      return common::visit(
          [&](auto &&s) { return GetLastName(s).symbol; }, obj.u);
    }};

    auto &wrapper{std::get<parser::OmpClause::Destroy>(clause.u)};
    if (const std::optional<parser::OmpDestroyClause> &destroy{wrapper.v}) {
      const Symbol *constrSym{getSymbol(std::get<parser::OmpObject>(x.t))};
      const Symbol *clauseSym{getSymbol(destroy->v)};
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
  for (const auto &clause : clauseList.v) {
    CheckAlignValue(clause);
  }
  CheckIsVarPartOfAnotherVar(dir.source, objectList);
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
  const auto &spec{std::get<parser::OmpDeclareMapperSpecifier>(x.t)};
  const auto &type = std::get<parser::TypeSpec>(spec.t);
  if (!std::get_if<parser::DerivedTypeSpec>(&type.u)) {
    context_.Say(dir.source, "Type is not a derived type"_err_en_US);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareMapperConstruct &) {
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
    CheckIsVarPartOfAnotherVar(dir.source, *objectList);
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
                CheckIsVarPartOfAnotherVar(dir.source, objList);
                CheckThreadprivateOrDeclareTargetVar(objList);
              },
              [&](const parser::OmpClause::Link &linkClause) {
                CheckSymbolNames(dir.source, linkClause.v);
                CheckIsVarPartOfAnotherVar(dir.source, linkClause.v);
                CheckThreadprivateOrDeclareTargetVar(linkClause.v);
              },
              [&](const parser::OmpClause::Enter &enterClause) {
                enterClauseFound = true;
                CheckSymbolNames(dir.source, enterClause.v);
                CheckIsVarPartOfAnotherVar(dir.source, enterClause.v);
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
    CheckIsVarPartOfAnotherVar(dir.source, *objectList);
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
      CheckIsVarPartOfAnotherVar(
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
  if (std::get<parser::OmpClauseList>(x.t).v.size() != 1) {
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
  const auto &dir{std::get<parser::OmpSimpleStandaloneDirective>(x.t)};
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
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_flush);
}

void OmpStructureChecker::Leave(const parser::OpenMPFlushConstruct &x) {
  if (FindClause(llvm::omp::Clause::OMPC_acquire) ||
      FindClause(llvm::omp::Clause::OMPC_release) ||
      FindClause(llvm::omp::Clause::OMPC_acq_rel)) {
    if (const auto &flushList{
            std::get<std::optional<parser::OmpObjectList>>(x.t)}) {
      context_.Say(parser::FindSourceLocation(flushList),
          "If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items "
          "must not be specified on the FLUSH directive"_err_en_US);
    }
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCancelConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &type{std::get<parser::OmpCancelType>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_cancel);
  CheckCancellationNest(dir.source, type.v);
}

void OmpStructureChecker::Leave(const parser::OpenMPCancelConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCriticalConstruct &x) {
  const auto &dir{std::get<parser::OmpCriticalDirective>(x.t)};
  const auto &endDir{std::get<parser::OmpEndCriticalDirective>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_critical);
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
  CheckHintClause<const parser::OmpClauseList>(&ompClause, nullptr);
}

void OmpStructureChecker::Leave(const parser::OpenMPCriticalConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPCancellationPointConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &type{std::get<parser::OmpCancelType>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_cancellation_point);
  CheckCancellationNest(dir.source, type.v);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPCancellationPointConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::CheckCancellationNest(
    const parser::CharBlock &source, const parser::OmpCancelType::Type &type) {
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
    case parser::OmpCancelType::Type::Taskgroup:
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
            "With %s clause, %s construct must be closely nested inside TASK "
            "or TASKLOOP construct and %s region must be closely nested inside "
            "TASKGROUP region"_err_en_US,
            parser::ToUpperCaseLetters(
                parser::OmpCancelType::EnumToString(type)),
            ContextDirectiveAsFortran(), ContextDirectiveAsFortran());
      }
      return;
    case parser::OmpCancelType::Type::Sections:
      if (llvm::omp::nestedCancelSectionsAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case parser::OmpCancelType::Type::Do:
      if (llvm::omp::nestedCancelDoAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case parser::OmpCancelType::Type::Parallel:
      if (llvm::omp::nestedCancelParallelAllowedSet.test(
              GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    }
    if (!eligibleCancellation) {
      context_.Say(source,
          "With %s clause, %s construct cannot be closely nested inside %s "
          "construct"_err_en_US,
          parser::ToUpperCaseLetters(parser::OmpCancelType::EnumToString(type)),
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(
              getDirectiveName(GetContextParent().directive).str()));
    }
  } else {
    // The cancellation directive cannot be orphaned.
    switch (type) {
    case parser::OmpCancelType::Type::Taskgroup:
      context_.Say(source,
          "%s %s directive is not closely nested inside "
          "TASK or TASKLOOP"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(
              parser::OmpCancelType::EnumToString(type)));
      break;
    case parser::OmpCancelType::Type::Sections:
      context_.Say(source,
          "%s %s directive is not closely nested inside "
          "SECTION or SECTIONS"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(
              parser::OmpCancelType::EnumToString(type)));
      break;
    case parser::OmpCancelType::Type::Do:
      context_.Say(source,
          "%s %s directive is not closely nested inside "
          "the construct that matches the DO clause type"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(
              parser::OmpCancelType::EnumToString(type)));
      break;
    case parser::OmpCancelType::Type::Parallel:
      context_.Say(source,
          "%s %s directive is not closely nested inside "
          "the construct that matches the PARALLEL clause type"_err_en_US,
          ContextDirectiveAsFortran(),
          parser::ToUpperCaseLetters(
              parser::OmpCancelType::EnumToString(type)));
      break;
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

inline void OmpStructureChecker::ErrIfAllocatableVariable(
    const parser::Variable &var) {
  // Err out if the given symbol has
  // ALLOCATABLE attribute
  if (const auto *e{GetExpr(context_, var)})
    for (const Symbol &symbol : evaluate::CollectSymbols(*e))
      if (IsAllocatable(symbol)) {
        const auto &designator =
            std::get<common::Indirection<parser::Designator>>(var.u);
        const auto *dataRef =
            std::get_if<parser::DataRef>(&designator.value().u);
        const parser::Name *name =
            dataRef ? std::get_if<parser::Name>(&dataRef->u) : nullptr;
        if (name)
          context_.Say(name->source,
              "%s must not have ALLOCATABLE "
              "attribute"_err_en_US,
              name->ToString());
      }
}

inline void OmpStructureChecker::ErrIfLHSAndRHSSymbolsMatch(
    const parser::Variable &var, const parser::Expr &expr) {
  // Err out if the symbol on the LHS is also used on the RHS of the assignment
  // statement
  const auto *e{GetExpr(context_, expr)};
  const auto *v{GetExpr(context_, var)};
  if (e && v) {
    auto vSyms{evaluate::GetSymbolVector(*v)};
    const Symbol &varSymbol = vSyms.front();
    for (const Symbol &symbol : evaluate::GetSymbolVector(*e)) {
      if (varSymbol == symbol) {
        const common::Indirection<parser::Designator> *designator =
            std::get_if<common::Indirection<parser::Designator>>(&expr.u);
        if (designator) {
          auto *z{var.typedExpr.get()};
          auto *c{expr.typedExpr.get()};
          if (z->v == c->v) {
            context_.Say(expr.source,
                "RHS expression on atomic assignment statement cannot access '%s'"_err_en_US,
                var.GetSource());
          }
        } else {
          context_.Say(expr.source,
              "RHS expression on atomic assignment statement cannot access '%s'"_err_en_US,
              var.GetSource());
        }
      }
    }
  }
}

inline void OmpStructureChecker::ErrIfNonScalarAssignmentStmt(
    const parser::Variable &var, const parser::Expr &expr) {
  // Err out if either the variable on the LHS or the expression on the RHS of
  // the assignment statement are non-scalar (i.e. have rank > 0 or is of
  // CHARACTER type)
  const auto *e{GetExpr(context_, expr)};
  const auto *v{GetExpr(context_, var)};
  if (e && v) {
    if (e->Rank() != 0 ||
        (e->GetType().has_value() &&
            e->GetType().value().category() == common::TypeCategory::Character))
      context_.Say(expr.source,
          "Expected scalar expression "
          "on the RHS of atomic assignment "
          "statement"_err_en_US);
    if (v->Rank() != 0 ||
        (v->GetType().has_value() &&
            v->GetType()->category() == common::TypeCategory::Character))
      context_.Say(var.GetSource(),
          "Expected scalar variable "
          "on the LHS of atomic assignment "
          "statement"_err_en_US);
  }
}

template <typename T, typename D>
bool OmpStructureChecker::IsOperatorValid(const T &node, const D &variable) {
  using AllowedBinaryOperators =
      std::variant<parser::Expr::Add, parser::Expr::Multiply,
          parser::Expr::Subtract, parser::Expr::Divide, parser::Expr::AND,
          parser::Expr::OR, parser::Expr::EQV, parser::Expr::NEQV>;
  using BinaryOperators = std::variant<parser::Expr::Add,
      parser::Expr::Multiply, parser::Expr::Subtract, parser::Expr::Divide,
      parser::Expr::AND, parser::Expr::OR, parser::Expr::EQV,
      parser::Expr::NEQV, parser::Expr::Power, parser::Expr::Concat,
      parser::Expr::LT, parser::Expr::LE, parser::Expr::EQ, parser::Expr::NE,
      parser::Expr::GE, parser::Expr::GT>;

  if constexpr (common::HasMember<T, BinaryOperators>) {
    const auto &variableName{variable.GetSource().ToString()};
    const auto &exprLeft{std::get<0>(node.t)};
    const auto &exprRight{std::get<1>(node.t)};
    if ((exprLeft.value().source.ToString() != variableName) &&
        (exprRight.value().source.ToString() != variableName)) {
      context_.Say(variable.GetSource(),
          "Atomic update statement should be of form "
          "`%s = %s operator expr` OR `%s = expr operator %s`"_err_en_US,
          variableName, variableName, variableName, variableName);
    }
    return common::HasMember<T, AllowedBinaryOperators>;
  }
  return false;
}

void OmpStructureChecker::CheckAtomicCaptureStmt(
    const parser::AssignmentStmt &assignmentStmt) {
  const auto &var{std::get<parser::Variable>(assignmentStmt.t)};
  const auto &expr{std::get<parser::Expr>(assignmentStmt.t)};
  common::visit(
      common::visitors{
          [&](const common::Indirection<parser::Designator> &designator) {
            const auto *dataRef =
                std::get_if<parser::DataRef>(&designator.value().u);
            const auto *name =
                dataRef ? std::get_if<parser::Name>(&dataRef->u) : nullptr;
            if (name && IsAllocatable(*name->symbol))
              context_.Say(name->source,
                  "%s must not have ALLOCATABLE "
                  "attribute"_err_en_US,
                  name->ToString());
          },
          [&](const auto &) {
            // Anything other than a `parser::Designator` is not allowed
            context_.Say(expr.source,
                "Expected scalar variable "
                "of intrinsic type on RHS of atomic "
                "assignment statement"_err_en_US);
          }},
      expr.u);
  ErrIfLHSAndRHSSymbolsMatch(var, expr);
  ErrIfNonScalarAssignmentStmt(var, expr);
}

void OmpStructureChecker::CheckAtomicWriteStmt(
    const parser::AssignmentStmt &assignmentStmt) {
  const auto &var{std::get<parser::Variable>(assignmentStmt.t)};
  const auto &expr{std::get<parser::Expr>(assignmentStmt.t)};
  ErrIfAllocatableVariable(var);
  ErrIfLHSAndRHSSymbolsMatch(var, expr);
  ErrIfNonScalarAssignmentStmt(var, expr);
}

void OmpStructureChecker::CheckAtomicUpdateStmt(
    const parser::AssignmentStmt &assignment) {
  const auto &expr{std::get<parser::Expr>(assignment.t)};
  const auto &var{std::get<parser::Variable>(assignment.t)};
  bool isIntrinsicProcedure{false};
  bool isValidOperator{false};
  common::visit(
      common::visitors{
          [&](const common::Indirection<parser::FunctionReference> &x) {
            isIntrinsicProcedure = true;
            const auto &procedureDesignator{
                std::get<parser::ProcedureDesignator>(x.value().v.t)};
            const parser::Name *name{
                std::get_if<parser::Name>(&procedureDesignator.u)};
            if (name &&
                !(name->source == "max" || name->source == "min" ||
                    name->source == "iand" || name->source == "ior" ||
                    name->source == "ieor")) {
              context_.Say(expr.source,
                  "Invalid intrinsic procedure name in "
                  "OpenMP ATOMIC (UPDATE) statement"_err_en_US);
            }
          },
          [&](const auto &x) {
            if (!IsOperatorValid(x, var)) {
              context_.Say(expr.source,
                  "Invalid or missing operator in atomic update "
                  "statement"_err_en_US);
            } else
              isValidOperator = true;
          },
      },
      expr.u);
  if (const auto *e{GetExpr(context_, expr)}) {
    const auto *v{GetExpr(context_, var)};
    if (e->Rank() != 0 ||
        (e->GetType().has_value() &&
            e->GetType().value().category() == common::TypeCategory::Character))
      context_.Say(expr.source,
          "Expected scalar expression "
          "on the RHS of atomic update assignment "
          "statement"_err_en_US);
    if (v->Rank() != 0 ||
        (v->GetType().has_value() &&
            v->GetType()->category() == common::TypeCategory::Character))
      context_.Say(var.GetSource(),
          "Expected scalar variable "
          "on the LHS of atomic update assignment "
          "statement"_err_en_US);
    auto vSyms{evaluate::GetSymbolVector(*v)};
    const Symbol &varSymbol = vSyms.front();
    int numOfSymbolMatches{0};
    SymbolVector exprSymbols{evaluate::GetSymbolVector(*e)};
    for (const Symbol &symbol : exprSymbols) {
      if (varSymbol == symbol) {
        numOfSymbolMatches++;
      }
    }
    if (isIntrinsicProcedure) {
      std::string varName = var.GetSource().ToString();
      if (numOfSymbolMatches != 1)
        context_.Say(expr.source,
            "Intrinsic procedure"
            " arguments in atomic update statement"
            " must have exactly one occurence of '%s'"_err_en_US,
            varName);
      else if (varSymbol != exprSymbols.front() &&
          varSymbol != exprSymbols.back())
        context_.Say(expr.source,
            "Atomic update statement "
            "should be of the form `%s = intrinsic_procedure(%s, expr_list)` "
            "OR `%s = intrinsic_procedure(expr_list, %s)`"_err_en_US,
            varName, varName, varName, varName);
    } else if (isValidOperator) {
      if (numOfSymbolMatches != 1)
        context_.Say(expr.source,
            "Exactly one occurence of '%s' "
            "expected on the RHS of atomic update assignment statement"_err_en_US,
            var.GetSource().ToString());
    }
  }

  ErrIfAllocatableVariable(var);
}

void OmpStructureChecker::CheckAtomicCompareConstruct(
    const parser::OmpAtomicCompare &atomicCompareConstruct) {

  // TODO: Check that the if-stmt is `if (var == expr) var = new`
  //       [with or without then/end-do]

  unsigned version{context_.langOptions().OpenMPVersion};
  if (version < 51) {
    context_.Say(atomicCompareConstruct.source,
        "%s construct not allowed in %s, %s"_err_en_US,
        atomicCompareConstruct.source, ThisVersion(version), TryVersion(51));
  }

  // TODO: More work needed here. Some of the Update restrictions need to
  // be added, but Update isn't the same either.
}

// TODO: Allow cond-update-stmt once compare clause is supported.
void OmpStructureChecker::CheckAtomicCaptureConstruct(
    const parser::OmpAtomicCapture &atomicCaptureConstruct) {
  const parser::AssignmentStmt &stmt1 =
      std::get<parser::OmpAtomicCapture::Stmt1>(atomicCaptureConstruct.t)
          .v.statement;
  const auto &stmt1Var{std::get<parser::Variable>(stmt1.t)};
  const auto &stmt1Expr{std::get<parser::Expr>(stmt1.t)};

  const parser::AssignmentStmt &stmt2 =
      std::get<parser::OmpAtomicCapture::Stmt2>(atomicCaptureConstruct.t)
          .v.statement;
  const auto &stmt2Var{std::get<parser::Variable>(stmt2.t)};
  const auto &stmt2Expr{std::get<parser::Expr>(stmt2.t)};

  if (semantics::checkForSingleVariableOnRHS(stmt1)) {
    CheckAtomicCaptureStmt(stmt1);
    if (semantics::checkForSymbolMatch(stmt2)) {
      // ATOMIC CAPTURE construct is of the form [capture-stmt, update-stmt]
      CheckAtomicUpdateStmt(stmt2);
    } else {
      // ATOMIC CAPTURE construct is of the form [capture-stmt, write-stmt]
      CheckAtomicWriteStmt(stmt2);
    }
    auto *v{stmt2Var.typedExpr.get()};
    auto *e{stmt1Expr.typedExpr.get()};
    if (v && e && !(v->v == e->v)) {
      context_.Say(stmt1Expr.source,
          "Captured variable/array element/derived-type component %s expected to be assigned in the second statement of ATOMIC CAPTURE construct"_err_en_US,
          stmt1Expr.source);
    }
  } else if (semantics::checkForSymbolMatch(stmt1) &&
      semantics::checkForSingleVariableOnRHS(stmt2)) {
    // ATOMIC CAPTURE construct is of the form [update-stmt, capture-stmt]
    CheckAtomicUpdateStmt(stmt1);
    CheckAtomicCaptureStmt(stmt2);
    // Variable updated in stmt1 should be captured in stmt2
    auto *v{stmt1Var.typedExpr.get()};
    auto *e{stmt2Expr.typedExpr.get()};
    if (v && e && !(v->v == e->v)) {
      context_.Say(stmt1Var.GetSource(),
          "Updated variable/array element/derived-type component %s expected to be captured in the second statement of ATOMIC CAPTURE construct"_err_en_US,
          stmt1Var.GetSource());
    }
  } else {
    context_.Say(stmt1Expr.source,
        "Invalid ATOMIC CAPTURE construct statements. Expected one of [update-stmt, capture-stmt], [capture-stmt, update-stmt], or [capture-stmt, write-stmt]"_err_en_US);
  }
}

void OmpStructureChecker::CheckAtomicMemoryOrderClause(
    const parser::OmpAtomicClauseList *leftHandClauseList,
    const parser::OmpAtomicClauseList *rightHandClauseList) {
  int numMemoryOrderClause{0};
  int numFailClause{0};
  auto checkForValidMemoryOrderClause = [&](const parser::OmpAtomicClauseList
                                                *clauseList) {
    for (const auto &clause : clauseList->v) {
      if (std::get_if<parser::OmpFailClause>(&clause.u)) {
        numFailClause++;
        if (numFailClause > 1) {
          context_.Say(clause.source,
              "More than one FAIL clause not allowed on OpenMP ATOMIC construct"_err_en_US);
          return;
        }
      } else {
        if (std::get_if<parser::OmpMemoryOrderClause>(&clause.u)) {
          numMemoryOrderClause++;
          if (numMemoryOrderClause > 1) {
            context_.Say(clause.source,
                "More than one memory order clause not allowed on OpenMP ATOMIC construct"_err_en_US);
            return;
          }
        }
      }
    }
  };
  if (leftHandClauseList) {
    checkForValidMemoryOrderClause(leftHandClauseList);
  }
  if (rightHandClauseList) {
    checkForValidMemoryOrderClause(rightHandClauseList);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPAtomicConstruct &x) {
  common::visit(
      common::visitors{
          [&](const parser::OmpAtomic &atomicConstruct) {
            const auto &dir{std::get<parser::Verbatim>(atomicConstruct.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicUpdateStmt(
                std::get<parser::Statement<parser::AssignmentStmt>>(
                    atomicConstruct.t)
                    .statement);
            CheckAtomicMemoryOrderClause(
                &std::get<parser::OmpAtomicClauseList>(atomicConstruct.t),
                nullptr);
            CheckHintClause<const parser::OmpAtomicClauseList>(
                &std::get<parser::OmpAtomicClauseList>(atomicConstruct.t),
                nullptr);
          },
          [&](const parser::OmpAtomicUpdate &atomicUpdate) {
            const auto &dir{std::get<parser::Verbatim>(atomicUpdate.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicUpdateStmt(
                std::get<parser::Statement<parser::AssignmentStmt>>(
                    atomicUpdate.t)
                    .statement);
            CheckAtomicMemoryOrderClause(
                &std::get<0>(atomicUpdate.t), &std::get<2>(atomicUpdate.t));
            CheckHintClause<const parser::OmpAtomicClauseList>(
                &std::get<0>(atomicUpdate.t), &std::get<2>(atomicUpdate.t));
          },
          [&](const parser::OmpAtomicRead &atomicRead) {
            const auto &dir{std::get<parser::Verbatim>(atomicRead.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicMemoryOrderClause(
                &std::get<0>(atomicRead.t), &std::get<2>(atomicRead.t));
            CheckHintClause<const parser::OmpAtomicClauseList>(
                &std::get<0>(atomicRead.t), &std::get<2>(atomicRead.t));
            CheckAtomicCaptureStmt(
                std::get<parser::Statement<parser::AssignmentStmt>>(
                    atomicRead.t)
                    .statement);
          },
          [&](const parser::OmpAtomicWrite &atomicWrite) {
            const auto &dir{std::get<parser::Verbatim>(atomicWrite.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicMemoryOrderClause(
                &std::get<0>(atomicWrite.t), &std::get<2>(atomicWrite.t));
            CheckHintClause<const parser::OmpAtomicClauseList>(
                &std::get<0>(atomicWrite.t), &std::get<2>(atomicWrite.t));
            CheckAtomicWriteStmt(
                std::get<parser::Statement<parser::AssignmentStmt>>(
                    atomicWrite.t)
                    .statement);
          },
          [&](const parser::OmpAtomicCapture &atomicCapture) {
            const auto &dir{std::get<parser::Verbatim>(atomicCapture.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicMemoryOrderClause(
                &std::get<0>(atomicCapture.t), &std::get<2>(atomicCapture.t));
            CheckHintClause<const parser::OmpAtomicClauseList>(
                &std::get<0>(atomicCapture.t), &std::get<2>(atomicCapture.t));
            CheckAtomicCaptureConstruct(atomicCapture);
          },
          [&](const parser::OmpAtomicCompare &atomicCompare) {
            const auto &dir{std::get<parser::Verbatim>(atomicCompare.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicMemoryOrderClause(
                &std::get<0>(atomicCompare.t), &std::get<2>(atomicCompare.t));
            CheckHintClause<const parser::OmpAtomicClauseList>(
                &std::get<0>(atomicCompare.t), &std::get<2>(atomicCompare.t));
            CheckAtomicCompareConstruct(atomicCompare);
          },
      },
      x.u);
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

  // 2.7.3 Single Construct Restriction
  if (GetContext().directive == llvm::omp::Directive::OMPD_end_single) {
    CheckNotAllowedIfClause(
        llvm::omp::Clause::OMPC_copyprivate, {llvm::omp::Clause::OMPC_nowait});
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
CHECK_SIMPLE_CLAUSE(Detach, OMPC_detach)
CHECK_SIMPLE_CLAUSE(DeviceType, OMPC_device_type)
CHECK_SIMPLE_CLAUSE(DistSchedule, OMPC_dist_schedule)
CHECK_SIMPLE_CLAUSE(Exclusive, OMPC_exclusive)
CHECK_SIMPLE_CLAUSE(Final, OMPC_final)
CHECK_SIMPLE_CLAUSE(Flush, OMPC_flush)
CHECK_SIMPLE_CLAUSE(Full, OMPC_full)
CHECK_SIMPLE_CLAUSE(Grainsize, OMPC_grainsize)
CHECK_SIMPLE_CLAUSE(Hint, OMPC_hint)
CHECK_SIMPLE_CLAUSE(Holds, OMPC_holds)
CHECK_SIMPLE_CLAUSE(Inclusive, OMPC_inclusive)
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
CHECK_SIMPLE_CLAUSE(When, OMPC_when)
CHECK_SIMPLE_CLAUSE(AdjustArgs, OMPC_adjust_args)
CHECK_SIMPLE_CLAUSE(AppendArgs, OMPC_append_args)
CHECK_SIMPLE_CLAUSE(MemoryOrder, OMPC_memory_order)
CHECK_SIMPLE_CLAUSE(Bind, OMPC_bind)
CHECK_SIMPLE_CLAUSE(Align, OMPC_align)
CHECK_SIMPLE_CLAUSE(Compare, OMPC_compare)
CHECK_SIMPLE_CLAUSE(CancellationConstructType, OMPC_cancellation_construct_type)
CHECK_SIMPLE_CLAUSE(OmpxAttribute, OMPC_ompx_attribute)
CHECK_SIMPLE_CLAUSE(Weak, OMPC_weak)

CHECK_REQ_SCALAR_INT_CLAUSE(NumTeams, OMPC_num_teams)
CHECK_REQ_SCALAR_INT_CLAUSE(NumThreads, OMPC_num_threads)
CHECK_REQ_SCALAR_INT_CLAUSE(OmpxDynCgroupMem, OMPC_ompx_dyn_cgroup_mem)
CHECK_REQ_SCALAR_INT_CLAUSE(Priority, OMPC_priority)
CHECK_REQ_SCALAR_INT_CLAUSE(ThreadLimit, OMPC_thread_limit)

CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Collapse, OMPC_collapse)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Safelen, OMPC_safelen)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Simdlen, OMPC_simdlen)

void OmpStructureChecker::Enter(const parser::OmpClause::AcqRel &) {
  if (!isFailClause)
    CheckAllowedClause(llvm::omp::Clause::OMPC_acq_rel);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Acquire &) {
  if (!isFailClause)
    CheckAllowedClause(llvm::omp::Clause::OMPC_acquire);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Release &) {
  if (!isFailClause)
    CheckAllowedClause(llvm::omp::Clause::OMPC_release);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Relaxed &) {
  if (!isFailClause)
    CheckAllowedClause(llvm::omp::Clause::OMPC_relaxed);
}

void OmpStructureChecker::Enter(const parser::OmpClause::SeqCst &) {
  if (!isFailClause)
    CheckAllowedClause(llvm::omp::Clause::OMPC_seq_cst);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Fail &) {
  assert(!isFailClause && "Unexpected FAIL clause inside a FAIL clause?");
  isFailClause = true;
  CheckAllowedClause(llvm::omp::Clause::OMPC_fail);
}

void OmpStructureChecker::Leave(const parser::OmpClause::Fail &) {
  assert(isFailClause && "Expected to be inside a FAIL clause here");
  isFailClause = false;
}

void OmpStructureChecker::Enter(const parser::OmpFailClause &) {
  assert(!isFailClause && "Unexpected FAIL clause inside a FAIL clause?");
  isFailClause = true;
  CheckAllowedClause(llvm::omp::Clause::OMPC_fail);
}

void OmpStructureChecker::Leave(const parser::OmpFailClause &) {
  assert(isFailClause && "Expected to be inside a FAIL clause here");
  isFailClause = false;
}

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
    assert(symbol && "Expecting a symbol for object");
    if (IsCommonBlock(*symbol)) {
      auto source{GetObjectSource(object)};
      context_.Say(source ? *source : GetContext().clauseSource,
          "Common block names are not allowed in %s clause"_err_en_US,
          parser::ToUpperCaseLetters(getClauseName(clauseId).str()));
    }
  }

  if (version >= 50) {
    // Object cannot be a part of another object (except array elements)
    CheckStructureComponent(objects, clauseId);
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

static bool IsReductionAllowedForType(
    const parser::OmpReductionIdentifier &ident, const DeclTypeSpec &type) {
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
        return type.IsNumeric(TypeCategory::Integer) ||
            type.IsNumeric(TypeCategory::Real) ||
            type.IsNumeric(TypeCategory::Complex);

      case parser::DefinedOperator::IntrinsicOperator::AND:
      case parser::DefinedOperator::IntrinsicOperator::OR:
      case parser::DefinedOperator::IntrinsicOperator::EQV:
      case parser::DefinedOperator::IntrinsicOperator::NEQV:
        return isLogical(type);

      // Reduction identifier is not in OMP5.2 Table 5.2
      default:
        DIE("This should have been caught in CheckIntrinsicOperator");
        return false;
      }
    }
    return true;
  }};

  auto checkDesignator{[&](const parser::ProcedureDesignator &procD) {
    const parser::Name *name{std::get_if<parser::Name>(&procD.u)};
    if (name && name->symbol) {
      const SourceName &realName{name->symbol->GetUltimate().name()};
      // OMP5.2: The type [...] of a list item that appears in a
      // reduction clause must be valid for the combiner expression
      if (realName == "iand" || realName == "ior" || realName == "ieor") {
        // IAND: arguments must be integers: F2023 16.9.100
        // IEOR: arguments must be integers: F2023 16.9.106
        // IOR: arguments must be integers: F2023 16.9.111
        return type.IsNumeric(TypeCategory::Integer);
      } else if (realName == "max" || realName == "min") {
        // MAX: arguments must be integer, real, or character:
        // F2023 16.9.135
        // MIN: arguments must be integer, real, or character:
        // F2023 16.9.141
        return type.IsNumeric(TypeCategory::Integer) ||
            type.IsNumeric(TypeCategory::Real) || isCharacter(type);
      }
    }
    // TODO: user defined reduction operators. Just allow everything for now.
    return true;
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
      if (!IsReductionAllowedForType(ident, *type)) {
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
  if (dirCtx.directive == llvm::omp::Directive::OMPD_loop) {
    // [5.2:257:33-34]
    // If a reduction-modifier is specified in a reduction clause that
    // appears on the directive, then the reduction modifier must be
    // default.
    context_.Say(GetContext().clauseSource,
        "REDUCTION modifier on LOOP directive must be DEFAULT"_err_en_US);
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
  CheckIsVarPartOfAnotherVar(GetContext().clauseSource, x.v, "SHARED");
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(x.v, symbols);
  CheckAllowedClause(llvm::omp::Clause::OMPC_private);
  CheckIsVarPartOfAnotherVar(GetContext().clauseSource, x.v, "PRIVATE");
  CheckIntentInPointer(symbols, llvm::omp::Clause::OMPC_private);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Nowait &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_nowait);
  if (llvm::omp::noWaitClauseNotAllowedSet.test(GetContext().directive)) {
    context_.Say(GetContext().clauseSource,
        "%s clause is not allowed on the OMP %s directive,"
        " use it on OMP END %s directive "_err_en_US,
        parser::ToUpperCaseLetters(
            getClauseName(llvm::omp::Clause::OMPC_nowait).str()),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
  }
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

void OmpStructureChecker::CheckIsVarPartOfAnotherVar(
    const parser::CharBlock &source, const parser::OmpObjectList &objList,
    llvm::StringRef clause) {
  for (const auto &ompObject : objList.v) {
    common::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              if (const auto *dataRef{
                      std::get_if<parser::DataRef>(&designator.u)}) {
                if (IsDataRefTypeParamInquiry(dataRef)) {
                  context_.Say(source,
                      "A type parameter inquiry cannot appear on the %s "
                      "directive"_err_en_US,
                      ContextDirectiveAsFortran());
                } else if (parser::Unwrap<parser::StructureComponent>(
                               ompObject) ||
                    parser::Unwrap<parser::ArrayElement>(ompObject)) {
                  if (llvm::omp::nonPartialVarSet.test(
                          GetContext().directive)) {
                    context_.Say(source,
                        "A variable that is part of another variable (as an "
                        "array or structure element) cannot appear on the %s "
                        "directive"_err_en_US,
                        ContextDirectiveAsFortran());
                  } else {
                    context_.Say(source,
                        "A variable that is part of another variable (as an "
                        "array or structure element) cannot appear in a "
                        "%s clause"_err_en_US,
                        clause.data());
                  }
                }
              }
            },
            [&](const parser::Name &name) {},
        },
        ompObject.u);
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Firstprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_firstprivate);

  CheckIsVarPartOfAnotherVar(GetContext().clauseSource, x.v, "FIRSTPRIVATE");
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
// Following clauses have a separate node in parse-tree.h.
// Atomic-clause
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicRead, OMPC_read)
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicWrite, OMPC_write)
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicUpdate, OMPC_update)
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicCapture, OMPC_capture)

void OmpStructureChecker::Leave(const parser::OmpAtomicRead &) {
  CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_read,
      {llvm::omp::Clause::OMPC_release, llvm::omp::Clause::OMPC_acq_rel});
}

void OmpStructureChecker::Leave(const parser::OmpAtomicWrite &) {
  CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_write,
      {llvm::omp::Clause::OMPC_acquire, llvm::omp::Clause::OMPC_acq_rel});
}

void OmpStructureChecker::Leave(const parser::OmpAtomicUpdate &) {
  CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_update,
      {llvm::omp::Clause::OMPC_acquire, llvm::omp::Clause::OMPC_acq_rel});
}

// OmpAtomic node represents atomic directive without atomic-clause.
// atomic-clause - READ,WRITE,UPDATE,CAPTURE.
void OmpStructureChecker::Leave(const parser::OmpAtomic &) {
  if (const auto *clause{FindClause(llvm::omp::Clause::OMPC_acquire)}) {
    context_.Say(clause->source,
        "Clause ACQUIRE is not allowed on the ATOMIC directive"_err_en_US);
  }
  if (const auto *clause{FindClause(llvm::omp::Clause::OMPC_acq_rel)}) {
    context_.Say(clause->source,
        "Clause ACQ_REL is not allowed on the ATOMIC directive"_err_en_US);
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
      std::string subName{parser::ToUpperCaseLetters(
          llvm::omp::getOpenMPDirectiveName(sub).str())};
      std::string dirName{parser::ToUpperCaseLetters(
          llvm::omp::getOpenMPDirectiveName(dir).str())};

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
  CheckIntentInPointer(symbols, llvm::omp::Clause::OMPC_copyprivate);
  CheckCopyingPolymorphicAllocatable(
      symbols, llvm::omp::Clause::OMPC_copyprivate);
  if (GetContext().directive == llvm::omp::Directive::OMPD_single) {
    context_.Say(GetContext().clauseSource,
        "%s clause is not allowed on the OMP %s directive,"
        " use it on OMP END %s directive "_err_en_US,
        parser::ToUpperCaseLetters(
            getClauseName(llvm::omp::Clause::OMPC_copyprivate).str()),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Lastprivate &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_lastprivate);

  const auto &objectList{std::get<parser::OmpObjectList>(x.v.t)};
  CheckIsVarPartOfAnotherVar(
      GetContext().clauseSource, objectList, "LASTPRIVATE");

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

  OmpVerifyModifiers(
      x.v, llvm::omp::OMPC_lastprivate, GetContext().clauseSource, context_);
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

  auto *depType{std::get_if<parser::OmpDependenceType>(&x.v.u)};
  auto *taskType{std::get_if<parser::OmpTaskDependenceType>(&x.v.u)};
  assert(((depType == nullptr) != (taskType == nullptr)) &&
      "Unexpected alternative in update clause");

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
  for (const auto &[symbol, source] : symbols) {
    if (!IsVariableListItem(*symbol)) {
      context_.SayWithDecl(
          *symbol, source, "'%s' must be a variable"_err_en_US, symbol->name());
    }
  }

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
  for (const auto &[symbol, source] : symbols) {
    if (!IsVariableListItem(*symbol)) {
      context_.SayWithDecl(
          *symbol, source, "'%s' must be a variable"_err_en_US, symbol->name());
    }
  }

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

llvm::StringRef OmpStructureChecker::getClauseName(llvm::omp::Clause clause) {
  return llvm::omp::getOpenMPClauseName(clause);
}

llvm::StringRef OmpStructureChecker::getDirectiveName(
    llvm::omp::Directive directive) {
  return llvm::omp::getOpenMPDirectiveName(directive);
}

const Symbol *OmpStructureChecker::GetObjectSymbol(
    const parser::OmpObject &object) {
  if (auto *name{std::get_if<parser::Name>(&object.u)}) {
    return &name->symbol->GetUltimate();
  } else if (auto *desg{std::get_if<parser::Designator>(&object.u)}) {
    return &GetLastName(*desg).symbol->GetUltimate();
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
          [&](const common::Indirection<parser::StructureComponent> &) {
            context_.Say(GetContext().clauseSource,
                "A variable that is part of another variable "
                "(such as an element of a structure) but is not an array "
                "element or an array section cannot appear in a DEPEND "
                "clause"_err_en_US);
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
  if (!arrayElement.subscripts.empty()) {
    for (const auto &subscript : arrayElement.subscripts) {
      if (const auto *triplet{
              std::get_if<parser::SubscriptTriplet>(&subscript.u)}) {
        if (std::get<0>(triplet->t) && std::get<1>(triplet->t)) {
          const auto &lower{std::get<0>(triplet->t)};
          const auto &upper{std::get<1>(triplet->t)};
          if (lower && upper) {
            const auto lval{GetIntValue(lower)};
            const auto uval{GetIntValue(upper)};
            if (lval && uval && *uval < *lval) {
              context_.Say(GetContext().clauseSource,
                  "'%s' in %s clause"
                  " is a zero size array section"_err_en_US,
                  name.ToString(),
                  parser::ToUpperCaseLetters(getClauseName(clause).str()));
              break;
            } else if (std::get<2>(triplet->t)) {
              const auto &strideExpr{std::get<2>(triplet->t)};
              if (strideExpr) {
                if (clause == llvm::omp::Clause::OMPC_depend) {
                  context_.Say(GetContext().clauseSource,
                      "Stride should not be specified for array section in "
                      "DEPEND "
                      "clause"_err_en_US);
                }
              }
            }
          }
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
