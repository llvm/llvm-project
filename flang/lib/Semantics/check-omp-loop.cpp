//===-- lib/Semantics/check-omp-loop.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic checks for constructs and clauses related to loops.
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"

#include "check-directive-structure.h"

#include "flang/Common/idioms.h"
#include "flang/Common/visit.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/openmp-utils.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/openmp-modifiers.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

#include "llvm/Frontend/OpenMP/OMP.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <variant>

namespace {
using namespace Fortran;

class AssociatedLoopChecker {
public:
  AssociatedLoopChecker(
      semantics::SemanticsContext &context, std::int64_t level)
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
  semantics::SemanticsContext &context_;
  const parser::CharBlock *source_;
  std::int64_t level_;
  std::map<std::string, std::int64_t> constructNamesAndLevels_;
};
} // namespace

namespace Fortran::semantics {

using namespace Fortran::semantics::omp;

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
            const parser::OmpDirectiveSpecification &beginSpec{c.BeginDir()};
            if (beginSpec.DirId() == llvm::omp::Directive::OMPD_ordered) {
              for (const auto &clause : beginSpec.Clauses().v) {
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
    context_.Say(parser::omp::GetOmpDirectiveName(c).source,
        "The only OpenMP constructs that can be encountered during execution "
        "of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, "
        "the `SIMD` construct, the `SCAN` construct and the `ORDERED` "
        "construct with the `SIMD` clause."_err_en_US);
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

  auto &optLoopCons = std::get<std::optional<parser::NestedConstruct>>(x.t);
  if (optLoopCons.has_value()) {
    if (const auto &doConstruct{
            std::get_if<parser::DoConstruct>(&*optLoopCons)}) {
      const auto &doBlock{std::get<parser::Block>(doConstruct->t)};
      CheckNoBranching(doBlock, beginDir.v, beginDir.source);
    }
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
  auto &optLoopCons = std::get<std::optional<parser::NestedConstruct>>(x.t);
  if (optLoopCons.has_value()) {
    if (const auto &loopConstruct{
            std::get_if<parser::DoConstruct>(&*optLoopCons)}) {
      const parser::DoConstruct *loop{&*loopConstruct};
      if (loop && loop->IsDoNormal()) {
        const parser::Name &itrVal{GetLoopIndex(loop)};
        SetLoopIv(itrVal.symbol);
      }
    }
  }
}

void OmpStructureChecker::CheckLoopItrVariableIsInt(
    const parser::OpenMPLoopConstruct &x) {
  auto &optLoopCons = std::get<std::optional<parser::NestedConstruct>>(x.t);
  if (optLoopCons.has_value()) {
    if (const auto &loopConstruct{
            std::get_if<parser::DoConstruct>(&*optLoopCons)}) {

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
    auto &optLoopCons = std::get<std::optional<parser::NestedConstruct>>(x.t);
    if (optLoopCons.has_value()) {
      if (const auto &loopConstruct{
              std::get_if<parser::DoConstruct>(&*optLoopCons)}) {
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

} // namespace Fortran::semantics
