//===-- lib/Semantics/check-omp-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include <algorithm>

namespace Fortran::semantics {

// Use when clause falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &) { \
    CheckAllowed(llvm::omp::Clause::Y); \
  }

#define CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowed(llvm::omp::Clause::Y); \
    RequiresConstantPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

#define CHECK_REQ_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowed(llvm::omp::Clause::Y); \
    RequiresPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

// Use when clause don't falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_PARSER_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::X &) { \
    CheckAllowed(llvm::omp::Y); \
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
    const auto *lhs{GetExpr(var)};
    const auto *rhs{GetExpr(expr)};
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
    if (const auto *e{GetExpr(expr)}) {
      for (const Symbol &symbol : evaluate::CollectSymbols(*e)) {
        const Symbol &root{GetAssociationRoot(symbol)};
        if (IsFunction(root) &&
            !(root.attrs().test(Attr::ELEMENTAL) ||
                root.attrs().test(Attr::INTRINSIC))) {
          context_.Say(expr.source,
              "User defined non-ELEMENTAL function "
              "'%s' is not allowed in a WORKSHARE construct"_err_en_US,
              root.name());
        }
      }
    }
    return false;
  }

private:
  SemanticsContext &context_;
  parser::CharBlock source_;
};

class OmpCycleChecker {
public:
  OmpCycleChecker(SemanticsContext &context, std::int64_t cycleLevel)
      : context_{context}, cycleLevel_{cycleLevel} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::DoConstruct &dc) {
    cycleLevel_--;
    const auto &labelName{std::get<0>(std::get<0>(dc.t).statement.t)};
    if (labelName) {
      labelNamesandLevels_.emplace(labelName.value().ToString(), cycleLevel_);
    }
    return true;
  }

  bool Pre(const parser::CycleStmt &cyclestmt) {
    std::map<std::string, std::int64_t>::iterator it;
    bool err{false};
    if (cyclestmt.v) {
      it = labelNamesandLevels_.find(cyclestmt.v->source.ToString());
      err = (it != labelNamesandLevels_.end() && it->second > 0);
    }
    if (cycleLevel_ > 0 || err) {
      context_.Say(*cycleSource_,
          "CYCLE statement to non-innermost associated loop of an OpenMP DO construct"_err_en_US);
    }
    return true;
  }

  bool Pre(const parser::Statement<parser::ActionStmt> &actionstmt) {
    cycleSource_ = &actionstmt.source;
    return true;
  }

private:
  SemanticsContext &context_;
  const parser::CharBlock *cycleSource_;
  std::int64_t cycleLevel_;
  std::map<std::string, std::int64_t> labelNamesandLevels_;
};

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
      } else if (llvm::omp::parallelSet.test(dirContext_[index].directive)) {
        return false;
      }
      index--;
    }
  }
  return false;
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

  OmpDirectiveSet distributeSet{llvm::omp::Directive::OMPD_distribute,
      llvm::omp::Directive::OMPD_distribute_parallel_do,
      llvm::omp::Directive::OMPD_distribute_parallel_do_simd,
      llvm::omp::Directive::OMPD_distribute_parallel_for,
      llvm::omp::Directive::OMPD_distribute_parallel_for_simd,
      llvm::omp::Directive::OMPD_distribute_simd};

  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
  if (distributeSet.test(beginDir.v)) {
    // `distribute` region has to be nested
    if (!CurrentDirectiveIsNested()) {
      violation = true;
    } else {
      // `distribute` region has to be strictly nested inside `teams`
      if (!llvm::omp::teamSet.test(GetContextParent().directive)) {
        violation = true;
      }
    }
  }
  if (violation) {
    context_.Say(beginDir.source,
        "`DISTRIBUTE` region has to be strictly nested inside `TEAMS` region."_err_en_US);
  }
}

void OmpStructureChecker::HasInvalidTeamsNesting(
    const llvm::omp::Directive &dir, const parser::CharBlock &source) {
  OmpDirectiveSet allowedSet{llvm::omp::Directive::OMPD_parallel,
      llvm::omp::Directive::OMPD_parallel_do,
      llvm::omp::Directive::OMPD_parallel_do_simd,
      llvm::omp::Directive::OMPD_parallel_for,
      llvm::omp::Directive::OMPD_parallel_for_simd,
      llvm::omp::Directive::OMPD_parallel_master,
      llvm::omp::Directive::OMPD_parallel_master_taskloop,
      llvm::omp::Directive::OMPD_parallel_master_taskloop_simd,
      llvm::omp::Directive::OMPD_parallel_sections,
      llvm::omp::Directive::OMPD_parallel_workshare,
      llvm::omp::Directive::OMPD_distribute,
      llvm::omp::Directive::OMPD_distribute_parallel_do,
      llvm::omp::Directive::OMPD_distribute_parallel_do_simd,
      llvm::omp::Directive::OMPD_distribute_parallel_for,
      llvm::omp::Directive::OMPD_distribute_parallel_for_simd,
      llvm::omp::Directive::OMPD_distribute_simd};

  if (!allowedSet.test(dir)) {
    context_.Say(source,
        "Only `DISTRIBUTE` or `PARALLEL` regions are allowed to be strictly nested inside `TEAMS` region."_err_en_US);
  }
}

void OmpStructureChecker::CheckPredefinedAllocatorRestriction(
    const parser::CharBlock &source, const parser::Name &name) {
  if (const auto *symbol{name.symbol}) {
    const auto *commonBlock{FindCommonBlockContaining(*symbol)};
    const auto &scope{context_.FindScope(symbol->name())};
    const Scope &containingScope{GetProgramUnitContaining(scope)};
    if (!isPredefinedAllocator &&
        (IsSave(*symbol) || commonBlock ||
            containingScope.kind() == Scope::Kind::Module)) {
      context_.Say(source,
          "If list items within the ALLOCATE directive have the "
          "SAVE attribute, are a common block name, or are "
          "declared in the scope of a module, then only "
          "predefined memory allocator parameters can be used "
          "in the allocator clause"_err_en_US);
    }
  }
}

void OmpStructureChecker::CheckPredefinedAllocatorRestriction(
    const parser::CharBlock &source,
    const parser::OmpObjectList &ompObjectList) {
  for (const auto &ompObject : ompObjectList.v) {
    std::visit(
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

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
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
  if (llvm::omp::simdSet.test(GetContext().directive)) {
    EnterDirectiveNest(SIMDNest);
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
  CheckDoWhile(x);
  CheckLoopItrVariableIsInt(x);
  CheckCycleConstraints(x);
  HasInvalidDistributeNesting(x);
  if (CurrentDirectiveIsNested() &&
      llvm::omp::teamSet.test(GetContextParent().directive)) {
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
void OmpStructureChecker::CheckDoWhile(const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
  if (beginDir.v == llvm::omp::Directive::OMPD_do) {
    if (const auto &doConstruct{
            std::get<std::optional<parser::DoConstruct>>(x.t)}) {
      if (doConstruct.value().IsDoWhile()) {
        const auto &doStmt{std::get<parser::Statement<parser::NonLabelDoStmt>>(
            doConstruct.value().t)};
        context_.Say(doStmt.source,
            "The DO loop cannot be a DO WHILE with DO directive."_err_en_US);
      }
    }
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
  std::visit(Fortran::common::visitors{
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
                 [&](const parser::OpenMPSimpleStandaloneConstruct &c) {
                   const auto &dir{
                       std::get<parser::OmpSimpleStandaloneDirective>(c.t)};
                   if (dir.v == llvm::omp::Directive::OMPD_ordered) {
                     const auto &clauses{std::get<parser::OmpClauseList>(c.t)};
                     for (const auto &clause : clauses.v) {
                       if (std::get_if<parser::OmpClause::Simd>(&clause.u)) {
                         eligibleSIMD = true;
                         break;
                       }
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
        "of a 'SIMD'"
        " region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD`"
        " construct and the `ORDERED` construct with the `SIMD` clause."_err_en_US);
  }
}

void OmpStructureChecker::CheckTargetNest(const parser::OpenMPConstruct &c) {
  // 2.12.5 Target Construct Restriction
  bool eligibleTarget{true};
  llvm::omp::Directive ineligibleTargetDir;
  std::visit(
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
            std::visit(
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
          [&](const auto &c) {},
      },
      c.u);
  if (!eligibleTarget) {
    context_.Say(parser::FindSourceLocation(c),
        "If %s directive is nested inside TARGET region, the behaviour "
        "is unspecified"_port_en_US,
        parser::ToUpperCaseLetters(
            getDirectiveName(ineligibleTargetDir).str()));
  }
}

std::int64_t OmpStructureChecker::GetOrdCollapseLevel(
    const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &clauseList{std::get<parser::OmpClauseList>(beginLoopDir.t)};
  std::int64_t orderedCollapseLevel{1};
  std::int64_t orderedLevel{0};
  std::int64_t collapseLevel{0};

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

void OmpStructureChecker::CheckCycleConstraints(
    const parser::OpenMPLoopConstruct &x) {
  std::int64_t ordCollapseLevel{GetOrdCollapseLevel(x)};
  OmpCycleChecker ompCycleChecker{context_, ordCollapseLevel};
  parser::Walk(x, ompCycleChecker);
}

void OmpStructureChecker::CheckDistLinear(
    const parser::OpenMPLoopConstruct &x) {

  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &clauses{std::get<parser::OmpClauseList>(beginLoopDir.t)};

  semantics::UnorderedSymbolSet indexVars;

  // Collect symbols of all the variables from linear clauses
  for (const auto &clause : clauses.v) {
    if (const auto *linearClause{
            std::get_if<parser::OmpClause::Linear>(&clause.u)}) {

      std::list<parser::Name> values;
      // Get the variant type
      if (std::holds_alternative<parser::OmpLinearClause::WithModifier>(
              linearClause->v.u)) {
        const auto &withM{
            std::get<parser::OmpLinearClause::WithModifier>(linearClause->v.u)};
        values = withM.names;
      } else {
        const auto &withOutM{std::get<parser::OmpLinearClause::WithoutModifier>(
            linearClause->v.u)};
        values = withOutM.names;
      }
      for (auto const &v : values) {
        indexVars.insert(*(v.symbol));
      }
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
            // Remove the symbol from the collcted set
            indexVars.erase(*(itrVal.symbol));
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
    for (auto var : indexVars) {
      const Symbol &root{GetAssociationRoot(var)};
      context_.Say(parser::FindSourceLocation(x),
          "Variable '%s' not allowed in `LINEAR` clause, only loop iterator can be specified in `LINEAR` clause of a construct combined with `DISTRIBUTE`"_err_en_US,
          root.name());
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPLoopConstruct &) {
  if (llvm::omp::simdSet.test(GetContext().directive)) {
    ExitDirectiveNest(SIMDNest);
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndLoopDirective &x) {
  const auto &dir{std::get<parser::OmpLoopDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  // 2.7.1 end-do -> END DO [nowait-clause]
  // 2.8.3 end-do-simd -> END DO SIMD [nowait-clause]
  case llvm::omp::Directive::OMPD_do:
  case llvm::omp::Directive::OMPD_do_simd:
    SetClauseSets(dir.v);
    break;
  default:
    // no clauses are allowed
    break;
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
  if (GetContext().directive == llvm::omp::Directive::OMPD_target) {
    EnterDirectiveNest(TargetNest);
  }

  if (CurrentDirectiveIsNested()) {
    if (llvm::omp::teamSet.test(GetContextParent().directive)) {
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
  if (GetContext().directive == llvm::omp::Directive::OMPD_target) {
    ExitDirectiveNest(TargetNest);
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::ChecksOnOrderedAsBlock() {
  if (FindClause(llvm::omp::Clause::OMPC_depend)) {
    context_.Say(GetContext().clauseSource,
        "DEPEND(*) clauses are not allowed when ORDERED construct is a block"
        " construct with an ORDERED region"_err_en_US);
    return;
  }

  OmpDirectiveSet notAllowedParallelSet{llvm::omp::Directive::OMPD_parallel,
      llvm::omp::Directive::OMPD_target_parallel,
      llvm::omp::Directive::OMPD_parallel_sections,
      llvm::omp::Directive::OMPD_parallel_workshare};
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
      } else if (llvm::omp::doSet.test(dirContext_[i].directive)) {
        isNestedInDo = true;
        isNestedInDoSIMD = llvm::omp::doSimdSet.test(dirContext_[i].directive);
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
      } else if (llvm::omp::simdSet.test(dirContext_[i].directive)) {
        isNestedInSIMD = true;
        break;
      } else if (notAllowedParallelSet.test(dirContext_[i].directive)) {
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
    std::visit(
        common::visitors{
            [&](const parser::Designator &) {
              if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
                const auto &declScope{
                    GetProgramUnitContaining(name->symbol->GetUltimate())};
                const auto *sym =
                    declScope.parent().FindSymbol(name->symbol->name());
                if (sym &&
                    (sym->has<MainProgramDetails>() ||
                        sym->has<ModuleDetails>())) {
                  context_.Say(name->source,
                      "The module name or main program name cannot be in a %s "
                      "directive"_err_en_US,
                      ContextDirectiveAsFortran());
                } else if (name->symbol->GetUltimate().IsSubprogram()) {
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
                    context_.Say(name->source,
                        "The entity with PARAMETER attribute is used in a %s "
                        "directive"_warn_en_US,
                        ContextDirectiveAsFortran());
                } else if (FindCommonBlockContaining(*name->symbol)) {
                  context_.Say(name->source,
                      "A variable in a %s directive cannot be an element of a "
                      "common block"_err_en_US,
                      ContextDirectiveAsFortran());
                } else if (!IsSave(*name->symbol) &&
                    declScope.kind() != Scope::Kind::MainProgram &&
                    declScope.kind() != Scope::Kind::Module) {
                  context_.Say(name->source,
                      "A variable that appears in a %s directive must be "
                      "declared in the scope of a module or have the SAVE "
                      "attribute, either explicitly or implicitly"_err_en_US,
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

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeAllocate &x) {
  isPredefinedAllocator = true;
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
  CheckIsVarPartOfAnotherVar(dir.source, objectList);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeAllocate &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<parser::OmpObjectList>(x.t)};
  CheckPredefinedAllocatorRestriction(dir.source, objectList);
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpClause::Allocator &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_allocator);
  // Note: Predefined allocators are stored in ScalarExpr as numbers
  //   whereas custom allocators are stored as strings, so if the ScalarExpr
  //   actually has an int value, then it must be a predefined allocator
  isPredefinedAllocator = GetIntValue(x.v).has_value();
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_allocator, x.v);
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_declare_target);
  const auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
  if (std::holds_alternative<parser::OmpDeclareTargetWithClause>(spec.u)) {
    SetClauseSets(llvm::omp::Directive::OMPD_declare_target);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
  if (const auto *objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
    CheckIsVarPartOfAnotherVar(dir.source, *objectList);
    CheckThreadprivateOrDeclareTargetVar(*objectList);
  } else if (const auto *clauseList{
                 parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
    for (const auto &clause : clauseList->v) {
      if (const auto *toClause{std::get_if<parser::OmpClause::To>(&clause.u)}) {
        CheckIsVarPartOfAnotherVar(dir.source, toClause->v);
        CheckThreadprivateOrDeclareTargetVar(toClause->v);
      } else if (const auto *linkClause{
                     std::get_if<parser::OmpClause::Link>(&clause.u)}) {
        CheckIsVarPartOfAnotherVar(dir.source, linkClause->v);
        CheckThreadprivateOrDeclareTargetVar(linkClause->v);
      }
    }
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPExecutableAllocate &x) {
  isPredefinedAllocator = true;
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  const auto &objectList{std::get<std::optional<parser::OmpObjectList>>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
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

void OmpStructureChecker::CheckBarrierNesting(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  // A barrier region may not be `closely nested` inside a worksharing, loop,
  // task, taskloop, critical, ordered, atomic, or master region.
  // TODO:  Expand the check to include `LOOP` construct as well when it is
  // supported.
  if (GetContext().directive == llvm::omp::Directive::OMPD_barrier) {
    if (IsCloselyNestedRegion(llvm::omp::nestedBarrierErrSet)) {
      context_.Say(parser::FindSourceLocation(x),
          "`BARRIER` region may not be closely nested inside of `WORKSHARING`, "
          "`LOOP`, `TASK`, `TASKLOOP`,"
          "`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region."_err_en_US);
    }
  }
}

void OmpStructureChecker::ChecksOnOrderedAsStandalone() {
  if (FindClause(llvm::omp::Clause::OMPC_threads) ||
      FindClause(llvm::omp::Clause::OMPC_simd)) {
    context_.Say(GetContext().clauseSource,
        "THREADS, SIMD clauses are not allowed when ORDERED construct is a "
        "standalone construct with no ORDERED region"_err_en_US);
  }

  bool isSinkPresent{false};
  int dependSourceCount{0};
  auto clauseAll = FindClauses(llvm::omp::Clause::OMPC_depend);
  for (auto itr = clauseAll.first; itr != clauseAll.second; ++itr) {
    const auto &dependClause{
        std::get<parser::OmpClause::Depend>(itr->second->u)};
    if (std::get_if<parser::OmpDependClause::Source>(&dependClause.v.u)) {
      dependSourceCount++;
      if (isSinkPresent) {
        context_.Say(itr->second->source,
            "DEPEND(SOURCE) is not allowed when DEPEND(SINK: vec) is present "
            "on ORDERED directive"_err_en_US);
      }
      if (dependSourceCount > 1) {
        context_.Say(itr->second->source,
            "At most one DEPEND(SOURCE) clause can appear on the ORDERED "
            "directive"_err_en_US);
      }
    } else if (std::get_if<parser::OmpDependClause::Sink>(&dependClause.v.u)) {
      isSinkPresent = true;
      if (dependSourceCount > 0) {
        context_.Say(itr->second->source,
            "DEPEND(SINK: vec) is not allowed when DEPEND(SOURCE) is present "
            "on ORDERED directive"_err_en_US);
      }
    } else {
      context_.Say(itr->second->source,
          "Only DEPEND(SOURCE) or DEPEND(SINK: vec) are allowed when ORDERED "
          "construct is a standalone construct with no ORDERED "
          "region"_err_en_US);
    }
  }

  OmpDirectiveSet allowedDoSet{llvm::omp::Directive::OMPD_do,
      llvm::omp::Directive::OMPD_parallel_do,
      llvm::omp::Directive::OMPD_target_parallel_do};
  bool isNestedInDoOrderedWithPara{false};
  if (CurrentDirectiveIsNested() &&
      allowedDoSet.test(GetContextParent().directive)) {
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
    std::optional<std::int64_t> orderedValue) {
  auto clauseAll{FindClauses(llvm::omp::Clause::OMPC_depend)};
  for (auto itr = clauseAll.first; itr != clauseAll.second; ++itr) {
    const auto &dependClause{
        std::get<parser::OmpClause::Depend>(itr->second->u)};
    if (const auto *sinkVectors{
            std::get_if<parser::OmpDependClause::Sink>(&dependClause.v.u)}) {
      std::int64_t numVar = sinkVectors->v.size();
      if (orderedValue != numVar) {
        context_.Say(itr->second->source,
            "The number of variables in DEPEND(SINK: vec) clause does not "
            "match the parameter specified in ORDERED clause"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &dir{std::get<parser::OmpSimpleStandaloneDirective>(x.t)};
  PushContextAndClauseSets(dir.source, dir.v);
  CheckBarrierNesting(x);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPSimpleStandaloneConstruct &) {
  switch (GetContext().directive) {
  case llvm::omp::Directive::OMPD_ordered:
    // [5.1] 2.19.9 Ordered Construct Restriction
    ChecksOnOrderedAsStandalone();
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
            "Hint clause other than omp_sync_hint_none cannot be specified for an unnamed CRITICAL directive"_err_en_US});
  }
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

    OmpDirectiveSet allowedTaskgroupSet{
        llvm::omp::Directive::OMPD_task, llvm::omp::Directive::OMPD_taskloop};
    OmpDirectiveSet allowedSectionsSet{llvm::omp::Directive::OMPD_sections,
        llvm::omp::Directive::OMPD_parallel_sections};
    OmpDirectiveSet allowedDoSet{llvm::omp::Directive::OMPD_do,
        llvm::omp::Directive::OMPD_distribute_parallel_do,
        llvm::omp::Directive::OMPD_parallel_do,
        llvm::omp::Directive::OMPD_target_parallel_do,
        llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do,
        llvm::omp::Directive::OMPD_teams_distribute_parallel_do};
    OmpDirectiveSet allowedParallelSet{llvm::omp::Directive::OMPD_parallel,
        llvm::omp::Directive::OMPD_target_parallel};

    bool eligibleCancellation{false};
    switch (type) {
    case parser::OmpCancelType::Type::Taskgroup:
      if (allowedTaskgroupSet.test(GetContextParent().directive)) {
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
              if (allowedParallelSet.test(dirContext_[i].directive)) {
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
      if (allowedSectionsSet.test(GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case Fortran::parser::OmpCancelType::Type::Do:
      if (allowedDoSet.test(GetContextParent().directive)) {
        eligibleCancellation = true;
      }
      break;
    case parser::OmpCancelType::Type::Parallel:
      if (allowedParallelSet.test(GetContextParent().directive)) {
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
    case Fortran::parser::OmpCancelType::Type::Do:
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
  if ((GetContext().directive == llvm::omp::Directive::OMPD_end_single) ||
      (GetContext().directive == llvm::omp::Directive::OMPD_end_workshare)) {
    dirContext_.pop_back();
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
          "Atomic update variable '%s' not found in the RHS of the "
          "assignment statement in an ATOMIC (UPDATE) construct"_err_en_US,
          variableName);
    }
    return common::HasMember<T, AllowedBinaryOperators>;
  }
  return true;
}

void OmpStructureChecker::CheckAtomicUpdateAssignmentStmt(
    const parser::AssignmentStmt &assignment) {
  const auto &expr{std::get<parser::Expr>(assignment.t)};
  const auto &var{std::get<parser::Variable>(assignment.t)};
  std::visit(
      common::visitors{
          [&](const common::Indirection<parser::FunctionReference> &x) {
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
            } else if (name) {
              bool foundMatch{false};
              if (auto varDesignatorIndirection =
                      std::get_if<Fortran::common::Indirection<
                          Fortran::parser::Designator>>(&var.u)) {
                const auto &varDesignator = varDesignatorIndirection->value();
                if (const auto *dataRef = std::get_if<Fortran::parser::DataRef>(
                        &varDesignator.u)) {
                  if (const auto *name =
                          std::get_if<Fortran::parser::Name>(&dataRef->u)) {
                    const auto &varSymbol = *name->symbol;
                    if (const auto *e{GetExpr(expr)}) {
                      for (const Symbol &symbol :
                          evaluate::CollectSymbols(*e)) {
                        if (symbol == varSymbol) {
                          foundMatch = true;
                          break;
                        }
                      }
                    }
                  }
                }
              }
              if (!foundMatch) {
                context_.Say(expr.source,
                    "Atomic update variable '%s' not found in the "
                    "argument list of intrinsic procedure"_err_en_US,
                    var.GetSource().ToString());
              }
            }
          },
          [&](const auto &x) {
            if (!IsOperatorValid(x, var)) {
              context_.Say(expr.source,
                  "Invalid operator in OpenMP ATOMIC (UPDATE) statement"_err_en_US);
            }
          },
      },
      expr.u);
}

void OmpStructureChecker::CheckAtomicMemoryOrderClause(
    const parser::OmpAtomicClauseList &clauseList) {
  int numMemoryOrderClause = 0;
  for (const auto &clause : clauseList.v) {
    if (std::get_if<Fortran::parser::OmpMemoryOrderClause>(&clause.u)) {
      numMemoryOrderClause++;
      if (numMemoryOrderClause > 1) {
        context_.Say(clause.source,
            "More than one memory order clause not allowed on OpenMP "
            "Atomic construct"_err_en_US);
        return;
      }
    }
  }
}

void OmpStructureChecker::CheckAtomicMemoryOrderClause(
    const parser::OmpAtomicClauseList &leftHandClauseList,
    const parser::OmpAtomicClauseList &rightHandClauseList) {
  int numMemoryOrderClause = 0;
  for (const auto &clause : leftHandClauseList.v) {
    if (std::get_if<Fortran::parser::OmpMemoryOrderClause>(&clause.u)) {
      numMemoryOrderClause++;
      if (numMemoryOrderClause > 1) {
        context_.Say(clause.source,
            "More than one memory order clause not allowed on "
            "OpenMP Atomic construct"_err_en_US);
        return;
      }
    }
  }
  for (const auto &clause : rightHandClauseList.v) {
    if (std::get_if<Fortran::parser::OmpMemoryOrderClause>(&clause.u)) {
      numMemoryOrderClause++;
      if (numMemoryOrderClause > 1) {
        context_.Say(clause.source,
            "More than one memory order clause not "
            "allowed on OpenMP Atomic construct"_err_en_US);
        return;
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPAtomicConstruct &x) {
  std::visit(
      common::visitors{
          [&](const parser::OmpAtomic &atomicConstruct) {
            const auto &dir{std::get<parser::Verbatim>(atomicConstruct.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicUpdateAssignmentStmt(
                std::get<parser::Statement<parser::AssignmentStmt>>(
                    atomicConstruct.t)
                    .statement);
            CheckAtomicMemoryOrderClause(
                std::get<parser::OmpAtomicClauseList>(atomicConstruct.t));
          },
          [&](const parser::OmpAtomicUpdate &atomicConstruct) {
            const auto &dir{std::get<parser::Verbatim>(atomicConstruct.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicUpdateAssignmentStmt(
                std::get<parser::Statement<parser::AssignmentStmt>>(
                    atomicConstruct.t)
                    .statement);
            CheckAtomicMemoryOrderClause(
                std::get<0>(atomicConstruct.t), std::get<2>(atomicConstruct.t));
          },
          [&](const auto &atomicConstruct) {
            const auto &dir{std::get<parser::Verbatim>(atomicConstruct.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
            CheckAtomicMemoryOrderClause(
                std::get<0>(atomicConstruct.t), std::get<2>(atomicConstruct.t));
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
  if (llvm::omp::doSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_schedule)}) {
      // only one schedule clause is allowed
      const auto &schedClause{std::get<parser::OmpClause::Schedule>(clause->u)};
      if (ScheduleModifierHasType(schedClause.v,
              parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
        if (FindClause(llvm::omp::Clause::OMPC_ordered)) {
          context_.Say(clause->source,
              "The NONMONOTONIC modifier cannot be specified "
              "if an ORDERED clause is specified"_err_en_US);
        }
        if (ScheduleModifierHasType(schedClause.v,
                parser::OmpScheduleModifierType::ModType::Monotonic)) {
          context_.Say(clause->source,
              "The MONOTONIC and NONMONOTONIC modifiers "
              "cannot be both specified"_err_en_US);
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
  if (llvm::omp::simdSet.test(GetContext().directive)) {
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
    // A list-item cannot appear in more than one aligned clause
    semantics::UnorderedSymbolSet alignedVars;
    auto clauseAll = FindClauses(llvm::omp::Clause::OMPC_aligned);
    for (auto itr = clauseAll.first; itr != clauseAll.second; ++itr) {
      const auto &alignedClause{
          std::get<parser::OmpClause::Aligned>(itr->second->u)};
      const auto &alignedNameList{
          std::get<std::list<parser::Name>>(alignedClause.v.t)};
      for (auto const &var : alignedNameList) {
        if (alignedVars.count(*(var.symbol)) == 1) {
          context_.Say(itr->second->source,
              "List item '%s' present at multiple ALIGNED clauses"_err_en_US,
              var.ToString());
          break;
        }
        alignedVars.insert(*(var.symbol));
      }
    }
  } // SIMD

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
          std::visit(
              common::visitors{
                  [&](const parser::Designator &) {
                    if (const auto *name{
                            parser::Unwrap<parser::Name>(ompObject)})
                      testThreadprivateVarErr(
                          name->symbol->GetUltimate(), *name, type);
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
}

// Following clauses do not have a separate node in parse-tree.h.
CHECK_SIMPLE_CLAUSE(AcqRel, OMPC_acq_rel)
CHECK_SIMPLE_CLAUSE(Acquire, OMPC_acquire)
CHECK_SIMPLE_CLAUSE(AtomicDefaultMemOrder, OMPC_atomic_default_mem_order)
CHECK_SIMPLE_CLAUSE(Affinity, OMPC_affinity)
CHECK_SIMPLE_CLAUSE(Allocate, OMPC_allocate)
CHECK_SIMPLE_CLAUSE(Capture, OMPC_capture)
CHECK_SIMPLE_CLAUSE(Copyin, OMPC_copyin)
CHECK_SIMPLE_CLAUSE(Default, OMPC_default)
CHECK_SIMPLE_CLAUSE(Depobj, OMPC_depobj)
CHECK_SIMPLE_CLAUSE(Destroy, OMPC_destroy)
CHECK_SIMPLE_CLAUSE(Detach, OMPC_detach)
CHECK_SIMPLE_CLAUSE(DeviceType, OMPC_device_type)
CHECK_SIMPLE_CLAUSE(DistSchedule, OMPC_dist_schedule)
CHECK_SIMPLE_CLAUSE(DynamicAllocators, OMPC_dynamic_allocators)
CHECK_SIMPLE_CLAUSE(Exclusive, OMPC_exclusive)
CHECK_SIMPLE_CLAUSE(Final, OMPC_final)
CHECK_SIMPLE_CLAUSE(Flush, OMPC_flush)
CHECK_SIMPLE_CLAUSE(From, OMPC_from)
CHECK_SIMPLE_CLAUSE(Full, OMPC_full)
CHECK_SIMPLE_CLAUSE(Hint, OMPC_hint)
CHECK_SIMPLE_CLAUSE(InReduction, OMPC_in_reduction)
CHECK_SIMPLE_CLAUSE(Inclusive, OMPC_inclusive)
CHECK_SIMPLE_CLAUSE(Match, OMPC_match)
CHECK_SIMPLE_CLAUSE(Nontemporal, OMPC_nontemporal)
CHECK_SIMPLE_CLAUSE(Order, OMPC_order)
CHECK_SIMPLE_CLAUSE(Read, OMPC_read)
CHECK_SIMPLE_CLAUSE(ReverseOffload, OMPC_reverse_offload)
CHECK_SIMPLE_CLAUSE(Threadprivate, OMPC_threadprivate)
CHECK_SIMPLE_CLAUSE(Threads, OMPC_threads)
CHECK_SIMPLE_CLAUSE(Inbranch, OMPC_inbranch)
CHECK_SIMPLE_CLAUSE(IsDevicePtr, OMPC_is_device_ptr)
CHECK_SIMPLE_CLAUSE(HasDeviceAddr, OMPC_has_device_addr)
CHECK_SIMPLE_CLAUSE(Link, OMPC_link)
CHECK_SIMPLE_CLAUSE(Indirect, OMPC_indirect)
CHECK_SIMPLE_CLAUSE(Mergeable, OMPC_mergeable)
CHECK_SIMPLE_CLAUSE(Nogroup, OMPC_nogroup)
CHECK_SIMPLE_CLAUSE(Notinbranch, OMPC_notinbranch)
CHECK_SIMPLE_CLAUSE(Nowait, OMPC_nowait)
CHECK_SIMPLE_CLAUSE(Partial, OMPC_partial)
CHECK_SIMPLE_CLAUSE(ProcBind, OMPC_proc_bind)
CHECK_SIMPLE_CLAUSE(Release, OMPC_release)
CHECK_SIMPLE_CLAUSE(Relaxed, OMPC_relaxed)
CHECK_SIMPLE_CLAUSE(SeqCst, OMPC_seq_cst)
CHECK_SIMPLE_CLAUSE(Simd, OMPC_simd)
CHECK_SIMPLE_CLAUSE(Sizes, OMPC_sizes)
CHECK_SIMPLE_CLAUSE(TaskReduction, OMPC_task_reduction)
CHECK_SIMPLE_CLAUSE(To, OMPC_to)
CHECK_SIMPLE_CLAUSE(UnifiedAddress, OMPC_unified_address)
CHECK_SIMPLE_CLAUSE(UnifiedSharedMemory, OMPC_unified_shared_memory)
CHECK_SIMPLE_CLAUSE(Uniform, OMPC_uniform)
CHECK_SIMPLE_CLAUSE(Unknown, OMPC_unknown)
CHECK_SIMPLE_CLAUSE(Untied, OMPC_untied)
CHECK_SIMPLE_CLAUSE(UseDevicePtr, OMPC_use_device_ptr)
CHECK_SIMPLE_CLAUSE(UsesAllocators, OMPC_uses_allocators)
CHECK_SIMPLE_CLAUSE(Update, OMPC_update)
CHECK_SIMPLE_CLAUSE(UseDeviceAddr, OMPC_use_device_addr)
CHECK_SIMPLE_CLAUSE(Write, OMPC_write)
CHECK_SIMPLE_CLAUSE(Init, OMPC_init)
CHECK_SIMPLE_CLAUSE(Use, OMPC_use)
CHECK_SIMPLE_CLAUSE(Novariants, OMPC_novariants)
CHECK_SIMPLE_CLAUSE(Nocontext, OMPC_nocontext)
CHECK_SIMPLE_CLAUSE(Filter, OMPC_filter)
CHECK_SIMPLE_CLAUSE(When, OMPC_when)
CHECK_SIMPLE_CLAUSE(AdjustArgs, OMPC_adjust_args)
CHECK_SIMPLE_CLAUSE(AppendArgs, OMPC_append_args)
CHECK_SIMPLE_CLAUSE(MemoryOrder, OMPC_memory_order)
CHECK_SIMPLE_CLAUSE(Bind, OMPC_bind)
CHECK_SIMPLE_CLAUSE(Align, OMPC_align)
CHECK_SIMPLE_CLAUSE(Compare, OMPC_compare)

CHECK_REQ_SCALAR_INT_CLAUSE(Grainsize, OMPC_grainsize)
CHECK_REQ_SCALAR_INT_CLAUSE(NumTasks, OMPC_num_tasks)
CHECK_REQ_SCALAR_INT_CLAUSE(NumTeams, OMPC_num_teams)
CHECK_REQ_SCALAR_INT_CLAUSE(NumThreads, OMPC_num_threads)
CHECK_REQ_SCALAR_INT_CLAUSE(Priority, OMPC_priority)
CHECK_REQ_SCALAR_INT_CLAUSE(ThreadLimit, OMPC_thread_limit)
CHECK_REQ_SCALAR_INT_CLAUSE(Device, OMPC_device)

CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Collapse, OMPC_collapse)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Safelen, OMPC_safelen)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Simdlen, OMPC_simdlen)

// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.
void OmpStructureChecker::Enter(const parser::OmpClause::Reduction &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_reduction);
  if (CheckReductionOperators(x)) {
    CheckReductionTypeList(x);
  }
}
bool OmpStructureChecker::CheckReductionOperators(
    const parser::OmpClause::Reduction &x) {

  const auto &definedOp{std::get<0>(x.v.t)};
  bool ok = false;
  std::visit(
      common::visitors{
          [&](const parser::DefinedOperator &dOpr) {
            const auto &intrinsicOp{
                std::get<parser::DefinedOperator::IntrinsicOperator>(dOpr.u)};
            ok = CheckIntrinsicOperator(intrinsicOp);
          },
          [&](const parser::ProcedureDesignator &procD) {
            const parser::Name *name{std::get_if<parser::Name>(&procD.u)};
            if (name) {
              if (name->source == "max" || name->source == "min" ||
                  name->source == "iand" || name->source == "ior" ||
                  name->source == "ieor") {
                ok = true;
              } else {
                context_.Say(GetContext().clauseSource,
                    "Invalid reduction identifier in REDUCTION clause."_err_en_US,
                    ContextDirectiveAsFortran());
              }
            }
          },
      },
      definedOp.u);

  return ok;
}
bool OmpStructureChecker::CheckIntrinsicOperator(
    const parser::DefinedOperator::IntrinsicOperator &op) {

  switch (op) {
  case parser::DefinedOperator::IntrinsicOperator::Add:
  case parser::DefinedOperator::IntrinsicOperator::Subtract:
  case parser::DefinedOperator::IntrinsicOperator::Multiply:
  case parser::DefinedOperator::IntrinsicOperator::AND:
  case parser::DefinedOperator::IntrinsicOperator::OR:
  case parser::DefinedOperator::IntrinsicOperator::EQV:
  case parser::DefinedOperator::IntrinsicOperator::NEQV:
    return true;
  default:
    context_.Say(GetContext().clauseSource,
        "Invalid reduction operator in REDUCTION clause."_err_en_US,
        ContextDirectiveAsFortran());
  }
  return false;
}

void OmpStructureChecker::CheckReductionTypeList(
    const parser::OmpClause::Reduction &x) {
  const auto &ompObjectList{std::get<parser::OmpObjectList>(x.v.t)};
  CheckIntentInPointerAndDefinable(
      ompObjectList, llvm::omp::Clause::OMPC_reduction);
  CheckReductionArraySection(ompObjectList);
  CheckMultipleAppearanceAcrossContext(ompObjectList);
}

void OmpStructureChecker::CheckIntentInPointerAndDefinable(
    const parser::OmpObjectList &objectList, const llvm::omp::Clause clause) {
  for (const auto &ompObject : objectList.v) {
    if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (const auto *symbol{name->symbol}) {
        if (IsPointer(symbol->GetUltimate()) &&
            IsIntentIn(symbol->GetUltimate())) {
          context_.Say(GetContext().clauseSource,
              "Pointer '%s' with the INTENT(IN) attribute may not appear "
              "in a %s clause"_err_en_US,
              symbol->name(),
              parser::ToUpperCaseLetters(getClauseName(clause).str()));
        }
        if (auto msg{
                WhyNotModifiable(*symbol, context_.FindScope(name->source))}) {
          context_
              .Say(GetContext().clauseSource,
                  "Variable '%s' on the %s clause is not definable"_err_en_US,
                  symbol->name(),
                  parser::ToUpperCaseLetters(getClauseName(clause).str()))
              .Attach(std::move(*msg));
        }
      }
    }
  }
}

void OmpStructureChecker::CheckReductionArraySection(
    const parser::OmpObjectList &ompObjectList) {
  for (const auto &ompObject : ompObjectList.v) {
    if (const auto *dataRef{parser::Unwrap<parser::DataRef>(ompObject)}) {
      if (const auto *arrayElement{
              parser::Unwrap<parser::ArrayElement>(ompObject)}) {
        if (arrayElement) {
          CheckArraySection(*arrayElement, GetLastName(*dataRef),
              llvm::omp::Clause::OMPC_reduction);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckMultipleAppearanceAcrossContext(
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
                            "worksharing"
                            " construct bind."_err_en_US,
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
  CheckAllowed(llvm::omp::Clause::OMPC_ordered);
  // the parameter of ordered clause is optional
  if (const auto &expr{x.v}) {
    RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_ordered, *expr);
    // 2.8.3 Loop SIMD Construct Restriction
    if (llvm::omp::doSimdSet.test(GetContext().directive)) {
      context_.Say(GetContext().clauseSource,
          "No ORDERED clause with a parameter can be specified "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Shared &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_shared);
  CheckIsVarPartOfAnotherVar(GetContext().clauseSource, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_private);
  CheckIsVarPartOfAnotherVar(GetContext().clauseSource, x.v);
  CheckIntentInPointer(x.v, llvm::omp::Clause::OMPC_private);
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
    const parser::CharBlock &source, const parser::OmpObjectList &objList) {
  OmpDirectiveSet nonPartialVarSet{llvm::omp::Directive::OMPD_allocate,
      llvm::omp::Directive::OMPD_threadprivate,
      llvm::omp::Directive::OMPD_declare_target};
  for (const auto &ompObject : objList.v) {
    std::visit(
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
                  if (nonPartialVarSet.test(GetContext().directive)) {
                    context_.Say(source,
                        "A variable that is part of another variable (as an "
                        "array or structure element) cannot appear on the %s "
                        "directive"_err_en_US,
                        ContextDirectiveAsFortran());
                  } else {
                    context_.Say(source,
                        "A variable that is part of another variable (as an "
                        "array or structure element) cannot appear in a "
                        "PRIVATE or SHARED clause"_err_en_US);
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
  CheckAllowed(llvm::omp::Clause::OMPC_firstprivate);
  CheckIsLoopIvPartOfClause(llvmOmpClause::OMPC_firstprivate, x.v);

  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);

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
    if (const parser::Name * name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (name->symbol == GetContext().loopIV) {
        context_.Say(name->source,
            "DO iteration variable %s is not allowed in %s clause."_err_en_US,
            name->ToString(),
            parser::ToUpperCaseLetters(getClauseName(clause).str()));
      }
    }
  }
}
// Following clauses have a seperate node in parse-tree.h.
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
  CheckAllowed(llvm::omp::Clause::OMPC_aligned);

  if (const auto &expr{
          std::get<std::optional<parser::ScalarIntConstantExpr>>(x.v.t)}) {
    RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_aligned, *expr);
  }
  // 2.8.1 TODO: list-item attribute check
}
void OmpStructureChecker::Enter(const parser::OmpClause::Defaultmap &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_defaultmap);
  using VariableCategory = parser::OmpDefaultmapClause::VariableCategory;
  if (!std::get<std::optional<VariableCategory>>(x.v.t)) {
    context_.Say(GetContext().clauseSource,
        "The argument TOFROM:SCALAR must be specified on the DEFAULTMAP "
        "clause"_err_en_US);
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::If &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_if);
  using dirNameModifier = parser::OmpIfClause::DirectiveNameModifier;
  static std::unordered_map<dirNameModifier, OmpDirectiveSet>
      dirNameModifierMap{{dirNameModifier::Parallel, llvm::omp::parallelSet},
          {dirNameModifier::Target, llvm::omp::targetSet},
          {dirNameModifier::TargetEnterData,
              {llvm::omp::Directive::OMPD_target_enter_data}},
          {dirNameModifier::TargetExitData,
              {llvm::omp::Directive::OMPD_target_exit_data}},
          {dirNameModifier::TargetData,
              {llvm::omp::Directive::OMPD_target_data}},
          {dirNameModifier::TargetUpdate,
              {llvm::omp::Directive::OMPD_target_update}},
          {dirNameModifier::Task, {llvm::omp::Directive::OMPD_task}},
          {dirNameModifier::Taskloop, llvm::omp::taskloopSet}};
  if (const auto &directiveName{
          std::get<std::optional<dirNameModifier>>(x.v.t)}) {
    auto search{dirNameModifierMap.find(*directiveName)};
    if (search == dirNameModifierMap.end() ||
        !search->second.test(GetContext().directive)) {
      context_
          .Say(GetContext().clauseSource,
              "Unmatched directive name modifier %s on the IF clause"_err_en_US,
              parser::ToUpperCaseLetters(
                  parser::OmpIfClause::EnumToString(*directiveName)))
          .Attach(
              GetContext().directiveSource, "Cannot apply to directive"_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Linear &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_linear);

  // 2.7 Loop Construct Restriction
  if ((llvm::omp::doSet | llvm::omp::simdSet).test(GetContext().directive)) {
    if (std::holds_alternative<parser::OmpLinearClause::WithModifier>(x.v.u)) {
      context_.Say(GetContext().clauseSource,
          "A modifier may not be specified in a LINEAR clause "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}

void OmpStructureChecker::CheckAllowedMapTypes(
    const parser::OmpMapType::Type &type,
    const std::list<parser::OmpMapType::Type> &allowedMapTypeList) {
  const auto found{std::find(
      std::begin(allowedMapTypeList), std::end(allowedMapTypeList), type)};
  if (found == std::end(allowedMapTypeList)) {
    std::string commaSeperatedMapTypes;
    llvm::interleave(
        allowedMapTypeList.begin(), allowedMapTypeList.end(),
        [&](const parser::OmpMapType::Type &mapType) {
          commaSeperatedMapTypes.append(parser::ToUpperCaseLetters(
              parser::OmpMapType::EnumToString(mapType)));
        },
        [&] { commaSeperatedMapTypes.append(", "); });
    context_.Say(GetContext().clauseSource,
        "Only the %s map types are permitted "
        "for MAP clauses on the %s directive"_err_en_US,
        commaSeperatedMapTypes, ContextDirectiveAsFortran());
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Map &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_map);

  if (const auto &maptype{std::get<std::optional<parser::OmpMapType>>(x.v.t)}) {
    using Type = parser::OmpMapType::Type;
    const Type &type{std::get<Type>(maptype->t)};
    switch (GetContext().directive) {
    case llvm::omp::Directive::OMPD_target:
    case llvm::omp::Directive::OMPD_target_teams:
    case llvm::omp::Directive::OMPD_target_teams_distribute:
    case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
    case llvm::omp::Directive::OMPD_target_data:
      CheckAllowedMapTypes(
          type, {Type::To, Type::From, Type::Tofrom, Type::Alloc});
      break;
    case llvm::omp::Directive::OMPD_target_enter_data:
      CheckAllowedMapTypes(type, {Type::To, Type::Alloc});
      break;
    case llvm::omp::Directive::OMPD_target_exit_data:
      CheckAllowedMapTypes(type, {Type::From, Type::Release, Type::Delete});
      break;
    default:
      break;
    }
  }
}

bool OmpStructureChecker::ScheduleModifierHasType(
    const parser::OmpScheduleClause &x,
    const parser::OmpScheduleModifierType::ModType &type) {
  const auto &modifier{
      std::get<std::optional<parser::OmpScheduleModifier>>(x.t)};
  if (modifier) {
    const auto &modType1{
        std::get<parser::OmpScheduleModifier::Modifier1>(modifier->t)};
    const auto &modType2{
        std::get<std::optional<parser::OmpScheduleModifier::Modifier2>>(
            modifier->t)};
    if (modType1.v.v == type || (modType2 && modType2->v.v == type)) {
      return true;
    }
  }
  return false;
}
void OmpStructureChecker::Enter(const parser::OmpClause::Schedule &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_schedule);
  const parser::OmpScheduleClause &scheduleClause = x.v;

  // 2.7 Loop Construct Restriction
  if (llvm::omp::doSet.test(GetContext().directive)) {
    const auto &kind{std::get<1>(scheduleClause.t)};
    const auto &chunk{std::get<2>(scheduleClause.t)};
    if (chunk) {
      if (kind == parser::OmpScheduleClause::ScheduleType::Runtime ||
          kind == parser::OmpScheduleClause::ScheduleType::Auto) {
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

    if (ScheduleModifierHasType(scheduleClause,
            parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
      if (kind != parser::OmpScheduleClause::ScheduleType::Dynamic &&
          kind != parser::OmpScheduleClause::ScheduleType::Guided) {
        context_.Say(GetContext().clauseSource,
            "The NONMONOTONIC modifier can only be specified with "
            "SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Depend &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_depend);
  if (const auto *inOut{std::get_if<parser::OmpDependClause::InOut>(&x.v.u)}) {
    const auto &designators{std::get<std::list<parser::Designator>>(inOut->t)};
    for (const auto &ele : designators) {
      if (const auto *dataRef{std::get_if<parser::DataRef>(&ele.u)}) {
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
}

void OmpStructureChecker::Enter(const parser::OmpClause::Copyprivate &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_copyprivate);
  CheckIntentInPointer(x.v, llvm::omp::Clause::OMPC_copyprivate);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Lastprivate &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_lastprivate);

  DirectivesClauseTriple dirClauseTriple;
  SymbolSourceMap currSymbols;
  GetSymbolsInObjectList(x.v, currSymbols);
  CheckDefinableObjects(currSymbols, GetClauseKindForParserClass(x));

  // Check lastprivate variables in worksharing constructs
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_do,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));
  dirClauseTriple.emplace(llvm::omp::Directive::OMPD_sections,
      std::make_pair(
          llvm::omp::Directive::OMPD_parallel, llvm::omp::privateReductionSet));

  CheckPrivateSymbolsInOuterCxt(
      currSymbols, dirClauseTriple, GetClauseKindForParserClass(x));
}

llvm::StringRef OmpStructureChecker::getClauseName(llvm::omp::Clause clause) {
  return llvm::omp::getOpenMPClauseName(clause);
}

llvm::StringRef OmpStructureChecker::getDirectiveName(
    llvm::omp::Directive directive) {
  return llvm::omp::getOpenMPDirectiveName(directive);
}

void OmpStructureChecker::CheckDependList(const parser::DataRef &d) {
  std::visit(
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
          [&](const parser::Name &) { return; },
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
                const auto stride{GetIntValue(strideExpr)};
                if ((stride && stride != 1)) {
                  context_.Say(GetContext().clauseSource,
                      "A list item that appears in a REDUCTION clause"
                      " should have a contiguous storage array section."_err_en_US,
                      ContextDirectiveAsFortran());
                  break;
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
    const parser::OmpObjectList &objectList, const llvm::omp::Clause clause) {
  SymbolSourceMap symbols;
  GetSymbolsInObjectList(objectList, symbols);
  for (auto it{symbols.begin()}; it != symbols.end(); ++it) {
    const auto *symbol{it->first};
    const auto source{it->second};
    if (IsPointer(*symbol) && IsIntentIn(*symbol)) {
      context_.Say(source,
          "Pointer '%s' with the INTENT(IN) attribute may not appear "
          "in a %s clause"_err_en_US,
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
  for (auto it{symbols.begin()}; it != symbols.end(); ++it) {
    const auto *symbol{it->first};
    const auto source{it->second};
    if (auto msg{WhyNotModifiable(*symbol, context_.FindScope(source))}) {
      context_
          .Say(source,
              "Variable '%s' on the %s clause is not definable"_err_en_US,
              symbol->name(),
              parser::ToUpperCaseLetters(getClauseName(clause).str()))
          .Attach(std::move(*msg));
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
      for (auto iter{currSymbols.begin()}; iter != currSymbols.end(); ++iter) {
        const auto *symbol{iter->first};
        const auto source{iter->second};
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
  auto it{block.begin()};

  if (const auto *ompConstruct{parser::Unwrap<parser::OpenMPConstruct>(*it)}) {
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
        const OmpDirectiveSet parallelDirSet{
            llvm::omp::Directive::OMPD_parallel,
            llvm::omp::Directive::OMPD_parallel_do,
            llvm::omp::Directive::OMPD_parallel_sections,
            llvm::omp::Directive::OMPD_parallel_workshare,
            llvm::omp::Directive::OMPD_parallel_do_simd};

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

        if (!parallelDirSet.test(currentDir)) {
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

const parser::OmpObjectList *OmpStructureChecker::GetOmpObjectList(
    const parser::OmpClause &clause) {

  // Clauses with OmpObjectList as its data member
  using MemberObjectListClauses = std::tuple<parser::OmpClause::Copyprivate,
      parser::OmpClause::Copyin, parser::OmpClause::Firstprivate,
      parser::OmpClause::From, parser::OmpClause::Lastprivate,
      parser::OmpClause::Link, parser::OmpClause::Private,
      parser::OmpClause::Shared, parser::OmpClause::To>;

  // Clauses with OmpObjectList in the tuple
  using TupleObjectListClauses = std::tuple<parser::OmpClause::Allocate,
      parser::OmpClause::Map, parser::OmpClause::Reduction>;

  // TODO:: Generate the tuples using TableGen.
  // Handle other constructs with OmpObjectList such as OpenMPThreadprivate.
  return std::visit(
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

} // namespace Fortran::semantics
