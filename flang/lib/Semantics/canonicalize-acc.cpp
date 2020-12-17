//===-- lib/Semantics/canonicalize-acc.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonicalize-acc.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/tools.h"

// After Loop Canonicalization, rewrite OpenACC parse tree to make OpenACC
// Constructs more structured which provide explicit scopes for later
// structural checks and semantic analysis.
//   1. move structured DoConstruct into
//      OpenACCLoopConstruct. Compilation will not proceed in case of errors
//      after this pass.
//   2. move structured DoConstruct into OpenACCCombinedConstruct. Move
//      AccEndCombinedConstruct into OpenACCCombinedConstruct if present.
//      Compilation will not proceed in case of errors after this pass.
namespace Fortran::semantics {

using namespace parser::literals;

class CanonicalizationOfAcc {
public:
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}
  CanonicalizationOfAcc(parser::Messages &messages) : messages_{messages} {}

  void Post(parser::Block &block) {
    for (auto it{block.begin()}; it != block.end(); ++it) {
      if (auto *accLoop{parser::Unwrap<parser::OpenACCLoopConstruct>(*it)}) {
        RewriteOpenACCLoopConstruct(*accLoop, block, it);
      } else if (auto *accCombined{
                     parser::Unwrap<parser::OpenACCCombinedConstruct>(*it)}) {
        RewriteOpenACCCombinedConstruct(*accCombined, block, it);
      } else if (auto *endDir{
                     parser::Unwrap<parser::AccEndCombinedDirective>(*it)}) {
        // Unmatched AccEndCombinedDirective
        messages_.Say(endDir->v.source,
            "The %s directive must follow the DO loop associated with the "
            "loop construct"_err_en_US,
            parser::ToUpperCaseLetters(endDir->v.source.ToString()));
      }
    } // Block list
  }

private:
  // Check constraint in 2.9.7
  // If there are n tile sizes in the list, the loop construct must be
  // immediately followed by n tightly-nested loops.
  template <typename C, typename D>
  void CheckTileClauseRestriction(const C &x) {
    const auto &beginLoopDirective = std::get<D>(x.t);
    const auto &accClauseList =
        std::get<parser::AccClauseList>(beginLoopDirective.t);
    for (const auto &clause : accClauseList.v) {
      if (const auto *tileClause =
              std::get_if<parser::AccClause::Tile>(&clause.u)) {
        const parser::AccTileExprList &tileExprList = tileClause->v;
        const std::list<parser::AccTileExpr> &listTileExpr = tileExprList.v;
        std::size_t tileArgNb = listTileExpr.size();

        const auto &outer{std::get<std::optional<parser::DoConstruct>>(x.t)};
        if (outer->IsDoConcurrent())
          return; // Tile is not allowed on DO CONURRENT
        for (const parser::DoConstruct *loop{&*outer}; loop && tileArgNb > 0;
             --tileArgNb) {
          const auto &block{std::get<parser::Block>(loop->t)};
          const auto it{block.begin()};
          loop = it != block.end() ? parser::Unwrap<parser::DoConstruct>(*it)
                                   : nullptr;
        }

        if (tileArgNb > 0) {
          messages_.Say(beginLoopDirective.source,
              "The loop construct with the TILE clause must be followed by %d "
              "tightly-nested loops"_err_en_US,
              listTileExpr.size());
        }
      }
    }
  }

  // Check constraint on line 1835 in Section 2.9
  // A tile and collapse clause may not appear on loop that is associated with
  // do concurrent.
  template <typename C, typename D>
  void CheckDoConcurrentClauseRestriction(const C &x) {
    const auto &doCons{std::get<std::optional<parser::DoConstruct>>(x.t)};
    if (!doCons->IsDoConcurrent())
      return;
    const auto &beginLoopDirective = std::get<D>(x.t);
    const auto &accClauseList =
        std::get<parser::AccClauseList>(beginLoopDirective.t);
    for (const auto &clause : accClauseList.v) {
      if (std::holds_alternative<parser::AccClause::Collapse>(clause.u) ||
          std::holds_alternative<parser::AccClause::Tile>(clause.u)) {
        messages_.Say(beginLoopDirective.source,
            "TILE and COLLAPSE clause may not appear on loop construct "
            "associated with DO CONCURRENT"_err_en_US);
      }
    }
  }

  void RewriteOpenACCLoopConstruct(parser::OpenACCLoopConstruct &x,
      parser::Block &block, parser::Block::iterator it) {
    // Check the sequence of DoConstruct in the same iteration
    //
    // Original:
    //   ExecutableConstruct -> OpenACCConstruct -> OpenACCLoopConstruct
    //     ACCBeginLoopDirective
    //   ExecutableConstruct -> DoConstruct
    //
    // After rewriting:
    //   ExecutableConstruct -> OpenACCConstruct -> OpenACCLoopConstruct
    //     AccBeginLoopDirective
    //     DoConstruct
    parser::Block::iterator nextIt;
    auto &beginDir{std::get<parser::AccBeginLoopDirective>(x.t)};
    auto &dir{std::get<parser::AccLoopDirective>(beginDir.t)};

    nextIt = it;
    if (++nextIt != block.end()) {
      if (auto *doCons{parser::Unwrap<parser::DoConstruct>(*nextIt)}) {
        if (doCons->GetLoopControl()) {
          // move DoConstruct
          std::get<std::optional<parser::DoConstruct>>(x.t) =
              std::move(*doCons);
          nextIt = block.erase(nextIt);
        } else {
          messages_.Say(dir.source,
              "DO loop after the %s directive must have loop control"_err_en_US,
              parser::ToUpperCaseLetters(dir.source.ToString()));
        }

        CheckDoConcurrentClauseRestriction<parser::OpenACCLoopConstruct,
            parser::AccBeginLoopDirective>(x);
        CheckTileClauseRestriction<parser::OpenACCLoopConstruct,
            parser::AccBeginLoopDirective>(x);

        return; // found do-loop
      }
    }
    messages_.Say(dir.source,
        "A DO loop must follow the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(dir.source.ToString()));
  }

  void RewriteOpenACCCombinedConstruct(parser::OpenACCCombinedConstruct &x,
      parser::Block &block, parser::Block::iterator it) {
    // Check the sequence of DoConstruct in the same iteration
    //
    // Original:
    //   ExecutableConstruct -> OpenACCConstruct -> OpenACCCombinedConstruct
    //     ACCBeginCombinedDirective
    //   ExecutableConstruct -> DoConstruct
    //   ExecutableConstruct -> AccEndCombinedDirective (if available)
    //
    // After rewriting:
    //   ExecutableConstruct -> OpenACCConstruct -> OpenACCCombinedConstruct
    //     ACCBeginCombinedDirective
    //     DoConstruct
    //     AccEndCombinedDirective (if available)
    parser::Block::iterator nextIt;
    auto &beginDir{std::get<parser::AccBeginCombinedDirective>(x.t)};
    auto &dir{std::get<parser::AccCombinedDirective>(beginDir.t)};

    nextIt = it;
    if (++nextIt != block.end()) {
      if (auto *doCons{parser::Unwrap<parser::DoConstruct>(*nextIt)}) {
        if (doCons->GetLoopControl()) {
          // move DoConstruct
          std::get<std::optional<parser::DoConstruct>>(x.t) =
              std::move(*doCons);
          nextIt = block.erase(nextIt);
          // try to match AccEndCombinedDirective
          if (nextIt != block.end()) {
            if (auto *endDir{
                    parser::Unwrap<parser::AccEndCombinedDirective>(*nextIt)}) {
              std::get<std::optional<parser::AccEndCombinedDirective>>(x.t) =
                  std::move(*endDir);
              block.erase(nextIt);
            }
          }
        } else {
          messages_.Say(dir.source,
              "DO loop after the %s directive must have loop control"_err_en_US,
              parser::ToUpperCaseLetters(dir.source.ToString()));
        }

        CheckDoConcurrentClauseRestriction<parser::OpenACCCombinedConstruct,
            parser::AccBeginCombinedDirective>(x);
        CheckTileClauseRestriction<parser::OpenACCCombinedConstruct,
            parser::AccBeginCombinedDirective>(x);

        return; // found do-loop
      }
    }
    messages_.Say(dir.source,
        "A DO loop must follow the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(dir.source.ToString()));
  }

  parser::Messages &messages_;
};

bool CanonicalizeAcc(parser::Messages &messages, parser::Program &program) {
  CanonicalizationOfAcc acc{messages};
  Walk(program, acc);
  return !messages.AnyFatalError();
}
} // namespace Fortran::semantics
