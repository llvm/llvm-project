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

#include "llvm/ADT/BitVector.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <cinttypes>
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
    if (auto *clause{
            parser::omp::FindClause(beginSpec, llvm::omp::Clause::OMPC_bind)}) {
      auto &bind{std::get<parser::OmpClause::Bind>(clause->u).v};
      if (bind.v != parser::OmpBindClause::Binding::Teams) {
        context_.Say(beginName.source, msg);
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
              if (parser::omp::FindClause(
                      beginSpec, llvm::omp::Clause::OMPC_simd)) {
                eligibleSIMD = true;
              }
            }
          },
          [&](const parser::OpenMPStandaloneConstruct &c) {
            if (auto *ssc{std::get_if<parser::OpenMPSimpleStandaloneConstruct>(
                    &c.u)}) {
              llvm::omp::Directive dirId{ssc->v.DirId()};
              if (dirId == llvm::omp::Directive::OMPD_ordered) {
                if (parser::omp::FindClause(
                        ssc->v, llvm::omp::Clause::OMPC_simd)) {
                  eligibleSIMD = true;
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

void OmpStructureChecker::CheckRectangularNest(
    const parser::OmpDirectiveSpecification &spec, const LoopSequence &nest) {
  unsigned version{context_.langOptions().OpenMPVersion};
  auto depth{GetRectangularNestDepthWithReason(spec, version)};
  if (!depth || *depth.value == 0) {
    return;
  }

  int64_t height{0};
  std::vector<const LoopSequence *> outer;
  for (const LoopSequence *n{&nest}; n;) {
    if (n->owner()) {
      WithReason<bool> rect{n->isRectangular(outer)};
      if (!rect.value.value_or(true)) {
        auto &msg{context_.Say(spec.DirName().source,
            "This construct requires a rectangular loop nest, but the associated nest is not"_err_en_US)};
        depth.reason.AttachTo(msg);
        rect.reason.AttachTo(msg);
      }
      outer.push_back(n);
    }
    height += n->height().value.value_or(1);
    if (height >= *depth.value) {
      break;
    }
    n = n->children().empty() ? nullptr : &n->children().front();
  }
}

void OmpStructureChecker::CheckNestedConstruct(
    const parser::OpenMPLoopConstruct &x) {
  const parser::OmpDirectiveSpecification &beginSpec{x.BeginDir()};
  llvm::omp::Directive dir{beginSpec.DirId()};
  unsigned version{context_.langOptions().OpenMPVersion};
  parser::CharBlock beginSource{beginSpec.DirName().source};

  // End-directive is not allowed in such cases:
  //   do 100 i = ...
  //     !$omp do
  //     do 100 j = ...
  //   100 continue
  //   !$omp end do    ! error
  auto &flags{std::get<parser::OmpDirectiveSpecification::Flags>(beginSpec.t)};
  if (flags.test(parser::OmpDirectiveSpecification::Flag::CrossesLabelDo)) {
    if (auto &endSpec{x.EndDir()}) {
      context_
          .Say(endSpec->DirName().source,
              "END %s directive is not allowed when the construct does not contain all loops that share a loop-terminating statement"_err_en_US,
              parser::ToUpperCaseLetters(beginSource.ToString()))
          .Attach(beginSource, "The construct starts here"_en_US);
    }
  }

  // Check constructs contained in the body of the loop construct.
  auto &body{std::get<parser::Block>(x.t)};

  for (auto &stmt : BlockRange(body, BlockRange::Step::Over)) {
    if (auto *d{parser::Unwrap<parser::CompilerDirective>(stmt)}) {
      context_.Say(d->source,
          "Compiler directives are not allowed inside OpenMP loop constructs"_warn_en_US);
    }
  }

  // The loop sequence will correspond to the nest associated with the
  // loop-associated construct being visited.
  LoopSequence sequence(body, version, true);
  auto assoc{llvm::omp::getDirectiveAssociation(dir)};
  auto needRange{GetAffectedLoopRangeWithReason(beginSpec, version)};
  auto haveLength{sequence.length()};

  const auto MsgShouldContainDoOr{
      "This construct should contain a DO-loop or a loop-%s-generating construct"_err_en_US};
  const auto MsgRequiresCanonical{
      "This construct requires a canonical loop %s"_err_en_US};

  if (assoc == llvm::omp::Association::LoopNest) {
    if (sequence.children().size() == 0) {
      context_.Say(beginSource, MsgShouldContainDoOr, "nest");
    } else if (haveLength.value > 1) {
      auto &msg{context_.Say(beginSource,
          "This construct applies to a loop nest, but has a loop sequence of "
          "length %" PRId64 ""_err_en_US,
          *haveLength.value)};
      haveLength.reason.AttachTo(msg);
    }
    auto [isWellFormed, whyNot]{sequence.isWellFormedNest()};
    if (isWellFormed && !*isWellFormed) {
      auto &msg{context_.Say(beginSource, MsgRequiresCanonical, "nest")};
      whyNot.AttachTo(msg);
    }

    // Check requirements on nest depth.
    auto [needDepth, needPerfect]{
        GetAffectedNestDepthWithReason(beginSpec, version)};
    auto &[haveSema, havePerf]{sequence.depth()};

    auto haveDepth{needPerfect ? havePerf : haveSema};
    std::string_view perfectTxt{needPerfect ? " perfect" : ""};

    if (needDepth.value > 1 && IsDoConcurrentLegal(version)) {
      if (auto *conc{sequence.getNestedDoConcurrent()}) {
        auto &msg{context_.Say(*parser::GetSource(*conc->owner()),
            "DO CONCURRENT must be the only affected loop in a loop nest"_err_en_US)};
        needDepth.reason.AttachTo(msg);
      }
    }

    // If the present depth is 0, it's likely that the construct doesn't
    // have any loops in it, which would be diagnosed above.
    if (needDepth && haveDepth.value > 0) {
      if (*needDepth.value > *haveDepth.value) {
        auto &msg{context_.Say(beginSource,
            "This construct requires a%s nest of depth %" PRId64
            ", but the associated nest is a%s nest of depth %" PRId64
            ""_err_en_US,
            perfectTxt, *needDepth.value, perfectTxt, *haveDepth.value)};
        haveDepth.reason.AttachTo(msg);
        needDepth.reason.AttachTo(msg);
      } else {
        CheckRectangularNest(beginSpec, sequence);
      }
    }

  } else if (assoc == llvm::omp::Association::LoopSeq) {
    if (haveLength.value == 0) {
      context_.Say(beginSource, MsgShouldContainDoOr, "sequence");
    } else {
      auto [isWellFormed, whyNot]{sequence.isWellFormedSequence()};
      if (isWellFormed && !*isWellFormed) {
        auto &msg{context_.Say(beginSource, MsgRequiresCanonical, "sequence")};
        whyNot.AttachTo(msg);
      }
      if (auto requiredCount{GetMinimumSequenceCount(needRange.value)}) {
        if (*requiredCount > 0 && haveLength.value < *requiredCount) {
          auto &msg{context_.Say(beginSource,
              "This construct requires a sequence of at least %" PRId64
              " loops, but the loop sequence has a length of %" PRId64
              ""_err_en_US,
              *requiredCount, *haveLength.value)};
          haveLength.reason.AttachTo(msg);
          needRange.reason.AttachTo(msg);
        }
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
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
    // nesting check
    HasInvalidWorksharingNesting(beginName, llvm::omp::nestedWorkshareErrSet);
  }

  for (auto &construct : std::get<parser::Block>(x.t)) {
    if (const auto *doConstruct{parser::omp::GetDoConstruct(construct)}) {
      const auto &doBlock{std::get<parser::Block>(doConstruct->t)};
      CheckNoBranching(doBlock, beginName.v, beginName.source);
    }
  }
  CheckIterationVariables(x);
  CheckNestedConstruct(x);
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

void OmpStructureChecker::CheckIterationVariables(
    const parser::OpenMPLoopConstruct &x) {
  unsigned version{context_.langOptions().OpenMPVersion};
  auto doLoops{CollectAffectedDoLoops(x, version, &context_)};
  if (!doLoops) {
    return;
  }
  const parser::OmpDirectiveSpecification &spec{x.BeginDir()};
  llvm::omp::Directive dirId{spec.DirId()};

  // Collect symbols from DSA clauses on the construct. These symbols
  // are the "host" versions of symbols inside the construct. The flags
  // of interest are on the associated symbols.
  struct ClauseAppearance {
    llvm::omp::Clause clauseId;
    parser::CharBlock source;
  };
  std::multimap<const Symbol *, ClauseAppearance> dsa;
  for (const parser::OmpClause &clause : spec.Clauses().v) {
    llvm::omp::Clause clauseId{clause.Id()};
    if (llvm::omp::isDataSharingAttributeClause(clauseId, version)) {
      for (const parser::OmpObject &object :
          parser::omp::GetOmpObjectList(clause)->v) {
        if (const Symbol *symbol{GetObjectSymbol(object)}) {
          auto maybeSource{parser::omp::GetObjectSource(object)};
          assert(maybeSource && "Expecting object source");
          dsa.insert(
              std::make_pair(symbol, ClauseAppearance{clauseId, *maybeSource}));
        }
      }
    }
  }

  auto [depth, _]{GetAffectedNestDepthWithReason(spec, version)};
  bool isLinearAllowed{false};
  if (!depth || depth.value == 1) {
    auto leafs{llvm::omp::getLeafConstructsOrSelf(dirId)};
    isLinearAllowed = leafs.back() == llvm::omp::Directive::OMPD_simd;
  }

  std::vector<parser::Name> ivs;
  for (const parser::DoConstruct *loop : *doLoops) {
    for (auto &control : GetLoopControls(*loop)) {
      if (control.iv.symbol) {
        ivs.push_back(control.iv);
      }
    }
  }

  for (const parser::Name &iv : ivs) {
    const auto *type{iv.symbol->GetType()};
    if (!type->IsNumeric(TypeCategory::Integer)) {
      context_.Say(iv.source,
          "The DO loop iteration variable must be of integer type"_err_en_US,
          iv.ToString());
    }
    const Symbol *host{GetHostSymbol(*iv.symbol)};
    if (!host) {
      continue;
    }
    if (host->test(Symbol::Flag::OmpThreadprivate)) {
      context_.Say(iv.source,
          "Loop iteration variable of an affected loop cannot be THREADPRIVATE"_err_en_US,
          iv.ToString());
    }
    // Check conflict between a predetermined DSA and explicit DSA.
    assert(iv.symbol->test(Symbol::Flag::OmpPreDetermined) &&
        "Expecting affected iteration variable to have predetermined DSA");
    if (iv.symbol->test(Symbol::Flag::OmpExplicit)) {
      auto range{dsa.equal_range(host)};
      for (auto found{range.first}; found != range.second; ++found) {
        llvm::omp::Clause id{found->second.clauseId};
        if (!llvm::omp::isAllowedClauseForDirective(dirId, id, version)) {
          continue;
        }
        if (id == llvm::omp::Clause::OMPC_private ||
            id == llvm::omp::Clause::OMPC_lastprivate) {
          continue;
        }
        if (id == llvm::omp::Clause::OMPC_linear && isLinearAllowed) {
          continue;
        }
        context_
            .Say(found->second.source,
                "Loop iteration variable with a predetermined data sharing attribute cannot appear in a %s clause"_err_en_US,
                parser::omp::GetUpperName(id, version))
            .Attach(iv.source,
                "'%s' is an iteration variable of an affected loop"_because_en_US,
                iv.ToString());
      }
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
  if (llvm::omp::allSimdSet.test(beginSpec.DirName().v)) {
    ExitDirectiveNest(SIMDNest);
  }
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpClause::Depth &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_depth);

  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_depth, x.v);
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
  CheckAssumedSizeArray(symbols, llvm::omp::Clause::OMPC_linear);

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
      auto &desc{OmpGetDescriptor<parser::OmpLinearModifier>()};
      parser::CharBlock modSource{OmpGetModifierSource(modifiers, linearMod)};
      bool valid{true};

      if (version < 52) {
        // Modifiers on LINEAR are only allowed on DECLARE SIMD
        if (dir != llvm::omp::Directive::OMPD_declare_simd) {
          context_.Say(modSource,
              "A modifier may not be specified in a LINEAR clause on the %s directive"_err_en_US,
              parser::omp::GetUpperName(dir, version));
          valid = false;
        }
      } else {
        if (linearMod->v == parser::OmpLinearModifier::Value::Ref ||
            linearMod->v == parser::OmpLinearModifier::Value::Uval) {
          if (dir != llvm::omp::Directive::OMPD_declare_simd) {
            context_.Say(modSource,
                "A REF or UVAL '%s' may not be specified in a LINEAR clause on the %s directive"_err_en_US,
                desc.name.str(), parser::omp::GetUpperName(dir, version));
            valid = false;
          }
        }
        if (!std::get</*PostModified=*/bool>(x.v.t)) {
          context_.Say(modSource,
              "The 'modifier(<list>)' syntax is deprecated in %s, use '<list> : modifier' instead"_warn_en_US,
              ThisVersion(version));
        }
      }

      if (valid) {
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
      }
    }
  }

  // Linear clause restrictions.
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

void OmpStructureChecker::Enter(const parser::OmpClause::Permutation &c) {
  unsigned version{context_.langOptions().OpenMPVersion};
  llvm::omp::Clause clause = llvm::omp::Clause::OMPC_permutation;
  CheckAllowedClause(clause);
  if (c.v.size() < 2)
    context_.Say(GetContext().clauseSource,
        "The %s clause must have a length of at least two"_err_en_US,
        parser::omp::GetUpperName(clause, version));

  llvm::BitVector found(c.v.size(), false);
  bool cont = true;
  for (const auto &val : c.v) {
    if (const auto v{GetIntValue(val)}) {
      if (*v <= 0) {
        cont = false;
        context_.Say(GetContext().clauseSource,
            "The parameter of the %s clause must be a constant positive integer expression"_err_en_US,
            parser::omp::GetUpperName(clause, version));
      } else if ((unsigned)*v - 1 < c.v.size()) {
        found.set(*v - 1);
      }
    } else
      cont = false;
  }

  if (!cont)
    return;
  if (!found.all()) {
    context_.Say(GetContext().clauseSource,
        "Every integer from 1 must appear in the %s clause"_err_en_US,
        parser::omp::GetUpperName(clause, version));
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Looprange &x) {
  CheckAllowedClause(llvm::omp::Clause::OMPC_looprange);
  auto &[first, count]{x.v.t};
  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_looprange, first);
  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_looprange, count);
}

void OmpStructureChecker::Enter(const parser::DoConstruct &x) {
  Base::Enter(x);
  constructStack_.push_back(&x);
}

void OmpStructureChecker::Leave(const parser::DoConstruct &x) {
  assert(!constructStack_.empty() && "Expecting non-empty construct stack");
#ifndef NDEBUG
  const LoopOrConstruct &top = constructStack_.back();
  auto *doc{std::get_if<const parser::DoConstruct *>(&top)};
  assert(doc != nullptr && *doc == &x && "Mismatched constructs");
#endif
  constructStack_.pop_back();
  Base::Leave(x);
}

} // namespace Fortran::semantics
