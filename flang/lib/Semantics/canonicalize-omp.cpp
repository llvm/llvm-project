//===-- lib/Semantics/canonicalize-omp.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonicalize-omp.h"
#include "flang/Parser/parse-tree-visitor.h"

// After Loop Canonicalization, rewrite OpenMP parse tree to make OpenMP
// Constructs more structured which provide explicit scopes for later
// structural checks and semantic analysis.
//   1. move structured DoConstruct and OmpEndLoopDirective into
//      OpenMPLoopConstruct. Compilation will not proceed in case of errors
//      after this pass.
//   2. Associate declarative OMP allocation directives with their
//      respective executable allocation directive
//   3. TBD
namespace Fortran::semantics {

using namespace parser::literals;

class CanonicalizationOfOmp {
public:
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}
  CanonicalizationOfOmp(parser::Messages &messages) : messages_{messages} {}

  void Post(parser::Block &block) {
    for (auto it{block.begin()}; it != block.end(); ++it) {
      if (auto *ompCons{GetConstructIf<parser::OpenMPConstruct>(*it)}) {
        // OpenMPLoopConstruct
        if (auto *ompLoop{
                std::get_if<parser::OpenMPLoopConstruct>(&ompCons->u)}) {
          RewriteOpenMPLoopConstruct(*ompLoop, block, it);
        }
      } else if (auto *endDir{
                     GetConstructIf<parser::OmpEndLoopDirective>(*it)}) {
        // Unmatched OmpEndLoopDirective
        auto &dir{std::get<parser::OmpLoopDirective>(endDir->t)};
        messages_.Say(dir.source,
            "The %s directive must follow the DO loop associated with the "
            "loop construct"_err_en_US,
            parser::ToUpperCaseLetters(dir.source.ToString()));
      }
    } // Block list
  }

  void Post(parser::ExecutionPart &body) { RewriteOmpAllocations(body); }

private:
  template <typename T> T *GetConstructIf(parser::ExecutionPartConstruct &x) {
    if (auto *y{std::get_if<parser::ExecutableConstruct>(&x.u)}) {
      if (auto *z{std::get_if<common::Indirection<T>>(&y->u)}) {
        return &z->value();
      }
    }
    return nullptr;
  }

  template <typename T> T *GetOmpIf(parser::ExecutionPartConstruct &x) {
    if (auto *construct{GetConstructIf<parser::OpenMPConstruct>(x)}) {
      if (auto *omp{std::get_if<T>(&construct->u)}) {
        return omp;
      }
    }
    return nullptr;
  }

  void RewriteOpenMPLoopConstruct(parser::OpenMPLoopConstruct &x,
      parser::Block &block, parser::Block::iterator it) {
    // Check the sequence of DoConstruct and OmpEndLoopDirective
    // in the same iteration
    //
    // Original:
    //   ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
    //     OmpBeginLoopDirective
    //   ExecutableConstruct -> DoConstruct
    //   ExecutableConstruct -> OmpEndLoopDirective (if available)
    //
    // After rewriting:
    //   ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
    //     OmpBeginLoopDirective
    //     DoConstruct
    //     OmpEndLoopDirective (if available)
    parser::Block::iterator nextIt;
    auto &beginDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
    auto &dir{std::get<parser::OmpLoopDirective>(beginDir.t)};

    nextIt = it;
    while (++nextIt != block.end()) {
      // Ignore compiler directives.
      if (GetConstructIf<parser::CompilerDirective>(*nextIt))
        continue;

      if (auto *doCons{GetConstructIf<parser::DoConstruct>(*nextIt)}) {
        if (doCons->GetLoopControl()) {
          // move DoConstruct
          std::get<std::optional<parser::DoConstruct>>(x.t) =
              std::move(*doCons);
          nextIt = block.erase(nextIt);
          // try to match OmpEndLoopDirective
          if (nextIt != block.end()) {
            if (auto *endDir{
                    GetConstructIf<parser::OmpEndLoopDirective>(*nextIt)}) {
              std::get<std::optional<parser::OmpEndLoopDirective>>(x.t) =
                  std::move(*endDir);
              block.erase(nextIt);
            }
          }
        } else {
          messages_.Say(dir.source,
              "DO loop after the %s directive must have loop control"_err_en_US,
              parser::ToUpperCaseLetters(dir.source.ToString()));
        }
      } else {
        messages_.Say(dir.source,
            "A DO loop must follow the %s directive"_err_en_US,
            parser::ToUpperCaseLetters(dir.source.ToString()));
      }
      // If we get here, we either found a loop, or issued an error message.
      return;
    }
  }

  void RewriteOmpAllocations(parser::ExecutionPart &body) {
    // Rewrite leading declarative allocations so they are nested
    // within their respective executable allocate directive
    //
    // Original:
    //   ExecutionPartConstruct -> OpenMPDeclarativeAllocate
    //   ExecutionPartConstruct -> OpenMPDeclarativeAllocate
    //   ExecutionPartConstruct -> OpenMPExecutableAllocate
    //
    // After rewriting:
    //   ExecutionPartConstruct -> OpenMPExecutableAllocate
    //     ExecutionPartConstruct -> OpenMPDeclarativeAllocate
    //     ExecutionPartConstruct -> OpenMPDeclarativeAllocate
    for (auto it = body.v.rbegin(); it != body.v.rend();) {
      if (auto *exec = GetOmpIf<parser::OpenMPExecutableAllocate>(*(it++))) {
        parser::OpenMPDeclarativeAllocate *decl;
        std::list<parser::OpenMPDeclarativeAllocate> subAllocates;
        while (it != body.v.rend() &&
            (decl = GetOmpIf<parser::OpenMPDeclarativeAllocate>(*it))) {
          subAllocates.push_front(std::move(*decl));
          it = decltype(it)(body.v.erase(std::next(it).base()));
        }
        if (!subAllocates.empty()) {
          std::get<std::optional<std::list<parser::OpenMPDeclarativeAllocate>>>(
              exec->t) = {std::move(subAllocates)};
        }
      }
    }
  }

  parser::Messages &messages_;
};

bool CanonicalizeOmp(parser::Messages &messages, parser::Program &program) {
  CanonicalizationOfOmp omp{messages};
  Walk(program, omp);
  return !messages.AnyFatalError();
}
} // namespace Fortran::semantics
