//===-- lib/Semantics/canonicalize-do.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonicalize-do.h"
#include "flang/Parser/parse-tree-visitor.h"

namespace Fortran::parser {

class CanonicalizationOfDoLoops {
  struct LabelInfo {
    Block::iterator iter;
    Label label;
  };

public:
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}
  void Post(Block &block) {
    std::vector<LabelInfo> stack;
    for (auto i{block.begin()}, end{block.end()}; i != end; ++i) {
      if (auto *executableConstruct{std::get_if<ExecutableConstruct>(&i->u)}) {
        common::visit(
            common::visitors{
                [](auto &) {},
                // Labels on end-stmt of constructs are accepted by f18 as an
                // extension.
                [&](common::Indirection<AssociateConstruct> &associate) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndAssociateStmt>>(
                          associate.value().t));
                },
                [&](common::Indirection<BlockConstruct> &blockConstruct) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndBlockStmt>>(
                          blockConstruct.value().t));
                },
                [&](common::Indirection<ChangeTeamConstruct> &changeTeam) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndChangeTeamStmt>>(
                          changeTeam.value().t));
                },
                [&](common::Indirection<CriticalConstruct> &critical) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndCriticalStmt>>(critical.value().t));
                },
                [&](common::Indirection<DoConstruct> &doConstruct) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndDoStmt>>(doConstruct.value().t));
                },
                [&](common::Indirection<IfConstruct> &ifConstruct) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndIfStmt>>(ifConstruct.value().t));
                },
                [&](common::Indirection<CaseConstruct> &caseConstruct) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndSelectStmt>>(
                          caseConstruct.value().t));
                },
                [&](common::Indirection<SelectRankConstruct> &selectRank) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndSelectStmt>>(selectRank.value().t));
                },
                [&](common::Indirection<SelectTypeConstruct> &selectType) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndSelectStmt>>(selectType.value().t));
                },
                [&](common::Indirection<ForallConstruct> &forall) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndForallStmt>>(forall.value().t));
                },
                [&](common::Indirection<WhereConstruct> &where) {
                  CanonicalizeIfMatch(block, stack, i,
                      std::get<Statement<EndWhereStmt>>(where.value().t));
                },
                [&](Statement<common::Indirection<LabelDoStmt>> &labelDoStmt) {
                  auto &label{std::get<Label>(labelDoStmt.statement.value().t)};
                  stack.push_back(LabelInfo{i, label});
                },
                [&](Statement<common::Indirection<EndDoStmt>> &endDoStmt) {
                  CanonicalizeIfMatch(block, stack, i, endDoStmt);
                },
                [&](Statement<ActionStmt> &actionStmt) {
                  CanonicalizeIfMatch(block, stack, i, actionStmt);
                },
            },
            executableConstruct->u);
      }
    }
  }

private:
  template <typename T>
  void CanonicalizeIfMatch(Block &originalBlock, std::vector<LabelInfo> &stack,
      Block::iterator &i, Statement<T> &statement) {
    if (!stack.empty() && statement.label &&
        stack.back().label == *statement.label) {
      auto currentLabel{stack.back().label};
      if constexpr (std::is_same_v<T, common::Indirection<EndDoStmt>>) {
        std::get<ExecutableConstruct>(i->u).u = Statement<ActionStmt>{
            std::optional<Label>{currentLabel}, ContinueStmt{}};
      }
      auto next{++i};
      do {
        Block block;
        auto doLoop{stack.back().iter};
        auto originalSource{
            std::get<Statement<common::Indirection<LabelDoStmt>>>(
                std::get<ExecutableConstruct>(doLoop->u).u)
                .source};
        block.splice(block.begin(), originalBlock, ++stack.back().iter, next);
        auto &labelDo{std::get<Statement<common::Indirection<LabelDoStmt>>>(
            std::get<ExecutableConstruct>(doLoop->u).u)};
        auto &loopControl{
            std::get<std::optional<LoopControl>>(labelDo.statement.value().t)};
        auto &name{std::get<std::optional<Name>>(labelDo.statement.value().t)};
        Statement<NonLabelDoStmt> nonLabelDoStmt{std::move(labelDo.label),
            NonLabelDoStmt{
                std::make_tuple(common::Clone(name), std::move(loopControl))}};
        nonLabelDoStmt.source = originalSource;
        std::get<ExecutableConstruct>(doLoop->u).u =
            common::Indirection<DoConstruct>{
                std::make_tuple(std::move(nonLabelDoStmt), std::move(block),
                    Statement<EndDoStmt>{
                        std::optional<Label>{}, EndDoStmt{std::move(name)}})};
        stack.pop_back();
      } while (!stack.empty() && stack.back().label == currentLabel);
      i = --next;
    }
  }
};

bool CanonicalizeDo(Program &program) {
  CanonicalizationOfDoLoops canonicalizationOfDoLoops;
  Walk(program, canonicalizationOfDoLoops);
  return true;
}

} // namespace Fortran::parser
