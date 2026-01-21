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

namespace Fortran::semantics {
static bool IsLoopTransforming(llvm::omp::Directive dir);
static bool IsFullUnroll(const parser::OpenMPLoopConstruct &x);
static std::optional<size_t> CountGeneratedNests(
    const parser::ExecutionPartConstruct &epc);
static std::optional<size_t> CountGeneratedNests(const parser::Block &block);
} // namespace Fortran::semantics

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
  const parser::OmpDirectiveName &beginName{x.BeginDir().DirName()};
  if (llvm::omp::topDistributeSet.test(beginName.v)) {
    // `distribute` region has to be nested
    if (CurrentDirectiveIsNested()) {
      // `distribute` region has to be strictly nested inside `teams`
      if (!llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
        context_.Say(beginName.source,
            "`DISTRIBUTE` region has to be strictly nested inside `TEAMS` "
            "region."_err_en_US);
      }
    } else {
      // If not lexically nested (orphaned), issue a warning.
      context_.Say(beginName.source,
          "`DISTRIBUTE` must be dynamically enclosed in a `TEAMS` "
          "region."_warn_en_US);
    }
  }
}
void OmpStructureChecker::HasInvalidLoopBinding(
    const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};
  const parser::OmpDirectiveName &beginName{beginSpec.DirName()};

  auto teamsBindingChecker = [&](parser::MessageFixedText msg) {
    for (const auto &clause : beginSpec.Clauses().v) {
      if (const auto *bindClause{
              std::get_if<parser::OmpClause::Bind>(&clause.u)}) {
        if (bindClause->v.v != parser::OmpBindClause::Binding::Teams) {
          context_.Say(beginName.source, msg);
        }
      }
    }
  };

  if (llvm::omp::Directive::OMPD_loop == beginName.v &&
      CurrentDirectiveIsNested() &&
      llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
    teamsBindingChecker(
        "`BIND(TEAMS)` must be specified since the `LOOP` region is "
        "strictly nested inside a `TEAMS` region."_err_en_US);
  }

  if (OmpDirectiveSet{
          llvm::omp::OMPD_teams_loop, llvm::omp::OMPD_target_teams_loop}
          .test(beginName.v)) {
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
          [&](const parser::OmpBlockConstruct &c) {
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
            const auto &beginName{c.BeginDir().DirName()};
            if (beginName.v == llvm::omp::Directive::OMPD_simd ||
                beginName.v == llvm::omp::Directive::OMPD_do_simd ||
                beginName.v == llvm::omp::Directive::OMPD_loop) {
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

static bool IsLoopTransforming(llvm::omp::Directive dir) {
  switch (dir) {
  // TODO case llvm::omp::Directive::OMPD_flatten:
  case llvm::omp::Directive::OMPD_fuse:
  case llvm::omp::Directive::OMPD_interchange:
  case llvm::omp::Directive::OMPD_nothing:
  case llvm::omp::Directive::OMPD_reverse:
  // TODO case llvm::omp::Directive::OMPD_split:
  case llvm::omp::Directive::OMPD_stripe:
  case llvm::omp::Directive::OMPD_tile:
  case llvm::omp::Directive::OMPD_unroll:
    return true;
  default:
    return false;
  }
}

void OmpStructureChecker::CheckNestedBlock(
    const parser::OpenMPLoopConstruct &x, const parser::Block &body) {
  for (auto &stmt : body) {
    if (auto *dir{parser::Unwrap<parser::CompilerDirective>(stmt)}) {
      context_.Say(dir->source,
          "Compiler directives are not allowed inside OpenMP loop constructs"_warn_en_US);
    } else if (auto *omp{parser::Unwrap<parser::OpenMPLoopConstruct>(stmt)}) {
      if (!IsLoopTransforming(omp->BeginDir().DirId())) {
        context_.Say(omp->source,
            "Only loop-transforming OpenMP constructs are allowed inside OpenMP loop constructs"_err_en_US);
      }
    } else if (auto *block{parser::Unwrap<parser::BlockConstruct>(stmt)}) {
      CheckNestedBlock(x, std::get<parser::Block>(block->t));
    } else if (!parser::Unwrap<parser::DoConstruct>(stmt)) {
      parser::CharBlock source{parser::GetSource(stmt).value_or(x.source)};
      context_.Say(source,
          "OpenMP loop construct can only contain DO loops or loop-nest-generating OpenMP constructs"_err_en_US);
    }
  }
}

static bool IsFullUnroll(const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};

  if (beginSpec.DirName().v == llvm::omp::Directive::OMPD_unroll) {
    return llvm::none_of(beginSpec.Clauses().v, [](const parser::OmpClause &c) {
      return c.Id() == llvm::omp::Clause::OMPC_partial;
    });
  }
  return false;
}

static std::optional<size_t> CountGeneratedNests(
    const parser::ExecutionPartConstruct &epc) {
  if (parser::Unwrap<parser::DoConstruct>(epc)) {
    return 1;
  }

  auto &omp{DEREF(parser::Unwrap<parser::OpenMPLoopConstruct>(epc))};
  const parser::OmpDirectiveSpecification &beginSpec{omp.BeginDir()};
  llvm::omp::Directive dir{beginSpec.DirName().v};

  // TODO: Handle split, apply.
  if (IsFullUnroll(omp)) {
    return std::nullopt;
  }
  if (dir == llvm::omp::Directive::OMPD_fuse) {
    auto rangeAt{
        llvm::find_if(beginSpec.Clauses().v, [](const parser::OmpClause &c) {
          return c.Id() == llvm::omp::Clause::OMPC_looprange;
        })};
    if (rangeAt == beginSpec.Clauses().v.end()) {
      return std::nullopt;
    }

    auto *loopRange{parser::Unwrap<parser::OmpLooprangeClause>(*rangeAt)};
    std::optional<int64_t> count{GetIntValue(std::get<1>(loopRange->t))};
    if (!count || *count <= 0) {
      return std::nullopt;
    }
    if (auto nestedCount{CountGeneratedNests(std::get<parser::Block>(omp.t))}) {
      if (static_cast<size_t>(*count) <= *nestedCount)
        return 1 + *nestedCount - static_cast<size_t>(*count);
    }
    return std::nullopt;
  }

  // For every other loop construct return 1.
  return 1;
}

static std::optional<size_t> CountGeneratedNests(const parser::Block &block) {
  // Count the number of loops in the associated block. If there are any
  // malformed construct in there, getting the number may be meaningless.
  // These issues will be diagnosed elsewhere, and we should not emit any
  // messages about a potentially incorrect loop count.
  // In such cases reset the count to nullopt. Once it becomes nullopt,
  // keep it that way.
  std::optional<size_t> numLoops{0};
  for (auto &epc : parser::omp::LoopRange(block)) {
    if (auto genCount{CountGeneratedNests(epc)}) {
      *numLoops += *genCount;
    } else {
      numLoops = std::nullopt;
      break;
    }
  }
  return numLoops;
}

void OmpStructureChecker::CheckNestedConstruct(
    const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};

  // End-directive is not allowed in such cases:
  //   do 100 i = ...
  //     !$omp do
  //     do 100 j = ...
  //   100 continue
  //   !$omp end do    ! error
  auto &flags{std::get<parser::OmpDirectiveSpecification::Flags>(beginSpec.t)};
  if (flags.test(parser::OmpDirectiveSpecification::Flag::CrossesLabelDo)) {
    if (auto &endSpec{x.EndDir()}) {
      parser::CharBlock beginSource{beginSpec.DirName().source};
      context_
          .Say(endSpec->DirName().source,
              "END %s directive is not allowed when the construct does not contain all loops that share a loop-terminating statement"_err_en_US,
              parser::ToUpperCaseLetters(beginSource.ToString()))
          .Attach(beginSource, "The construct starts here"_en_US);
    }
  }

  auto &body{std::get<parser::Block>(x.t)};

  CheckNestedBlock(x, body);

  // Check if a loop-nest-associated construct has only one top-level loop
  // in it.
  if (std::optional<size_t> numLoops{CountGeneratedNests(body)}) {
    if (*numLoops == 0) {
      context_.Say(beginSpec.DirName().source,
          "This construct should contain a DO-loop or a loop-nest-generating OpenMP construct"_err_en_US);
    } else {
      auto assoc{llvm::omp::getDirectiveAssociation(beginSpec.DirName().v)};
      if (*numLoops > 1 && assoc == llvm::omp::Association::LoopNest) {
        context_.Say(beginSpec.DirName().source,
            "This construct applies to a loop nest, but has a loop sequence of length %zu"_err_en_US,
            *numLoops);
      }
    }
  }
}

void OmpStructureChecker::CheckFullUnroll(
    const parser::OpenMPLoopConstruct &x) {
  // If the nested construct is a full unroll, then this construct is invalid
  // since it won't contain a loop.
  if (const parser::OpenMPLoopConstruct *nested{x.GetNestedConstruct()}) {
    if (IsFullUnroll(*nested)) {
      context_.Say(x.source,
          "OpenMP loop construct cannot apply to a fully unrolled loop"_err_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  loopStack_.push_back(&x);

  const parser::OmpDirectiveName &beginName{x.BeginDir().DirName()};
  PushContextAndClauseSets(beginName.source, beginName.v);

  // Check matching, end directive is optional
  if (auto &endSpec{x.EndDir()}) {
    CheckMatching<parser::OmpDirectiveName>(beginName, endSpec->DirName());

    AddEndDirectiveClauses(endSpec->Clauses());
  }

  if (llvm::omp::allSimdSet.test(GetContext().directive)) {
    EnterDirectiveNest(SIMDNest);
  }

  if (CurrentDirectiveIsNested() &&
      llvm::omp::topTeamsSet.test(GetContext().directive) &&
      GetContextParent().directive == llvm::omp::Directive::OMPD_target &&
      !GetDirectiveNest(TargetBlockOnlyTeams)) {
    context_.Say(GetContextParent().directiveSource,
        "TARGET construct with nested TEAMS region contains statements or "
        "directives outside of the TEAMS construct"_err_en_US);
  }

  // Combined target loop constructs are target device constructs. Keep track of
  // whether any such construct has been visited to later check that REQUIRES
  // directives for target-related options don't appear after them.
  if (llvm::omp::allTargetSet.test(beginName.v)) {
    deviceConstructFound_ = true;
  }

  if (beginName.v == llvm::omp::Directive::OMPD_do) {
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
        beginName.source, llvm::omp::nestedWorkshareErrSet);
  }
  SetLoopInfo(x);

  for (auto &construct : std::get<parser::Block>(x.t)) {
    if (const auto *doConstruct{parser::omp::GetDoConstruct(construct)}) {
      const auto &doBlock{std::get<parser::Block>(doConstruct->t)};
      CheckNoBranching(doBlock, beginName.v, beginName.source);
    }
  }
  CheckLoopItrVariableIsInt(x);
  CheckNestedConstruct(x);
  CheckFullUnroll(x);
  CheckAssociatedLoopConstraints(x);
  HasInvalidDistributeNesting(x);
  HasInvalidLoopBinding(x);
  if (CurrentDirectiveIsNested() &&
      llvm::omp::bottomTeamsSet.test(GetContextParent().directive)) {
    HasInvalidTeamsNesting(beginName.v, beginName.source);
  }
  if (beginName.v == llvm::omp::Directive::OMPD_distribute_parallel_do_simd ||
      beginName.v == llvm::omp::Directive::OMPD_distribute_simd) {
    CheckDistLinear(x);
  }
}

const parser::Name OmpStructureChecker::GetLoopIndex(
    const parser::DoConstruct *x) {
  using Bounds = parser::LoopControl::Bounds;
  return std::get<Bounds>(x->GetLoopControl()->u).Name().thing;
}

void OmpStructureChecker::SetLoopInfo(const parser::OpenMPLoopConstruct &x) {
  if (const auto *loop{x.GetNestedLoop()}) {
    if (loop->IsDoNormal()) {
      const parser::Name &itrVal{GetLoopIndex(loop)};
      SetLoopIv(itrVal.symbol);
    }
  }
}

void OmpStructureChecker::CheckLoopItrVariableIsInt(
    const parser::OpenMPLoopConstruct &x) {
  for (auto &construct : std::get<parser::Block>(x.t)) {
    for (const parser::DoConstruct *loop{
             parser::omp::GetDoConstruct(construct)};
        loop;) {
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

std::int64_t OmpStructureChecker::GetOrdCollapseLevel(
    const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};
  std::int64_t orderedCollapseLevel{1};
  std::int64_t orderedLevel{1};
  std::int64_t collapseLevel{1};

  for (const auto &clause : beginSpec.Clauses().v) {
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
  const parser::OmpClauseList &clauses{x.BeginDir().Clauses()};

  SymbolSourceMap indexVars;

  // Collect symbols of all the variables from linear clauses
  for (auto &clause : clauses.v) {
    if (std::get_if<parser::OmpClause::Linear>(&clause.u)) {
      GetSymbolsInObjectList(*parser::omp::GetOmpObjectList(clause), indexVars);
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
    for (auto &construct : std::get<parser::Block>(x.t)) {
      std::int64_t curCollapseVal{collapseVal};
      for (const parser::DoConstruct *loop{
               parser::omp::GetDoConstruct(construct)};
          loop;) {
        if (loop->IsDoNormal()) {
          const parser::Name &itrVal{GetLoopIndex(loop)};
          if (itrVal.symbol) {
            // Remove the symbol from the collected set
            indexVars.erase(&itrVal.symbol->GetUltimate());
          }
          curCollapseVal--;
          if (curCollapseVal == 0) {
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

void OmpStructureChecker::CheckLooprangeBounds(
    const parser::OpenMPLoopConstruct &x) {
  for (const parser::OmpClause &clause : x.BeginDir().Clauses().v) {
    if (auto *lrClause{parser::Unwrap<parser::OmpLooprangeClause>(clause)}) {
      auto first{GetIntValue(std::get<0>(lrClause->t))};
      auto count{GetIntValue(std::get<1>(lrClause->t))};
      if (!first || !count || *first <= 0 || *count <= 0) {
        return;
      }
      auto requiredCount{static_cast<size_t>(*first + *count - 1)};
      if (auto loopCount{CountGeneratedNests(std::get<parser::Block>(x.t))}) {
        if (*loopCount < requiredCount) {
          context_.Say(clause.source,
              "The specified loop range requires %zu loops, but the loop sequence has a length of %zu"_err_en_US,
              requiredCount, *loopCount);
        }
      }
      return;
    }
  }
}

void OmpStructureChecker::CheckScanModifier(
    const parser::OmpClause::Reduction &x) {
  using ReductionModifier = parser::OmpReductionModifier;

  auto checkReductionSymbolInScan{[&](const parser::Name &name) {
    if (auto *symbol{name.symbol}) {
      if (!symbol->test(Symbol::Flag::OmpInclusiveScan) &&
          !symbol->test(Symbol::Flag::OmpExclusiveScan)) {
        context_.Say(name.source,
            "List item %s must appear in EXCLUSIVE or INCLUSIVE clause of an enclosed SCAN directive"_err_en_US,
            name.ToString());
      }
    }
  }};

  auto &modifiers{OmpGetModifiers(x.v)};
  auto *maybeModifier{OmpGetUniqueModifier<ReductionModifier>(modifiers)};
  if (maybeModifier && maybeModifier->v == ReductionModifier::Value::Inscan) {
    for (const auto &ompObj : parser::omp::GetOmpObjectList(x)->v) {
      common::visit(
          common::visitors{
              [&](const parser::Designator &desg) {
                if (auto *name{parser::GetDesignatorNameIfDataRef(desg)}) {
                  checkReductionSymbolInScan(*name);
                }
              },
              [&](const parser::Name &name) {
                checkReductionSymbolInScan(name);
              },
              [&](const parser::OmpObject::Invalid &invalid) {},
          },
          ompObj.u);
    }
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};

  // A few semantic checks for InScan reduction are performed below as SCAN
  // constructs inside LOOP may add the relevant information. Scan reduction is
  // supported only in loop constructs, so same checks are not applicable to
  // other directives.
  for (const auto &clause : beginSpec.Clauses().v) {
    if (auto *reduction{std::get_if<parser::OmpClause::Reduction>(&clause.u)}) {
      CheckScanModifier(*reduction);
    }
  }
  if (beginSpec.DirName().v == llvm::omp::Directive::OMPD_fuse) {
    CheckLooprangeBounds(x);
  }
  if (llvm::omp::allSimdSet.test(beginSpec.DirName().v)) {
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
  const parser::OmpDirectiveName &dir{x.DirName()};
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
        // Don't return early - continue to check other restrictions
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
    // Check that the list item is a scalar variable (rank 0)
    // For declare simd with REF modifier, arrays are allowed
    bool isArrayAllowed{dir == llvm::omp::Directive::OMPD_declare_simd &&
        linearMod && linearMod->v == parser::OmpLinearModifier::Value::Ref};
    if (symbol->Rank() != 0 && !isArrayAllowed) {
      context_.Say(source,
          "List item '%s' in LINEAR clause must be a scalar variable"_err_en_US,
          symbol->name());
    }
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

void OmpStructureChecker::Enter(const parser::OmpClause::Sizes &c) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_sizes);
  for (const parser::Cosubscript &v : c.v)
    RequiresPositiveParameter(llvm::omp::Clause::OMPC_sizes, v,
        /*paramName=*/"parameter", /*allowZero=*/false);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Looprange &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_looprange);
  auto &[first, count]{x.v.t};
  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_looprange, first);
  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_looprange, count);
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
