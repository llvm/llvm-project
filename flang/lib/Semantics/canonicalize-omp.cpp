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

  // Pre-visit all constructs that have both a specification part and
  // an execution part, and store the connection between the two.
  bool Pre(parser::BlockConstruct &x) {
    auto *spec = &std::get<parser::BlockSpecificationPart>(x.t).v;
    auto *block = &std::get<parser::Block>(x.t);
    blockForSpec_.insert(std::make_pair(spec, block));
    return true;
  }
  bool Pre(parser::MainProgram &x) {
    auto *spec = &std::get<parser::SpecificationPart>(x.t);
    auto *block = &std::get<parser::ExecutionPart>(x.t).v;
    blockForSpec_.insert(std::make_pair(spec, block));
    return true;
  }
  bool Pre(parser::FunctionSubprogram &x) {
    auto *spec = &std::get<parser::SpecificationPart>(x.t);
    auto *block = &std::get<parser::ExecutionPart>(x.t).v;
    blockForSpec_.insert(std::make_pair(spec, block));
    return true;
  }
  bool Pre(parser::SubroutineSubprogram &x) {
    auto *spec = &std::get<parser::SpecificationPart>(x.t);
    auto *block = &std::get<parser::ExecutionPart>(x.t).v;
    blockForSpec_.insert(std::make_pair(spec, block));
    return true;
  }
  bool Pre(parser::SeparateModuleSubprogram &x) {
    auto *spec = &std::get<parser::SpecificationPart>(x.t);
    auto *block = &std::get<parser::ExecutionPart>(x.t).v;
    blockForSpec_.insert(std::make_pair(spec, block));
    return true;
  }

  void Post(parser::SpecificationPart &spec) {
    CanonicalizeUtilityConstructs(spec);
  }

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

  // Canonicalization of utility constructs.
  //
  // This addresses the issue of utility constructs that appear at the
  // boundary between the specification and the execution parts, e.g.
  //   subroutine foo
  //     integer :: x     ! Specification
  //     !$omp nothing
  //     x = 1            ! Execution
  //     ...
  //   end
  //
  // Utility constructs (error and nothing) can appear in both the
  // specification part and the execution part, except "error at(execution)",
  // which cannot be present in the specification part (whereas any utility
  // construct can be in the execution part).
  // When a utility construct is at the boundary, it should preferably be
  // parsed as an element of the execution part, but since the specification
  // part is parsed first, the utility construct ends up belonging to the
  // specification part.
  //
  // To allow the likes of the following code to compile, move all utility
  // construct that are at the end of the specification part to the beginning
  // of the execution part.
  //
  // subroutine foo
  //   !$omp error at(execution)  ! Initially parsed as declarative construct.
  //                              ! Move it to the execution part.
  // end

  void CanonicalizeUtilityConstructs(parser::SpecificationPart &spec) {
    auto found = blockForSpec_.find(&spec);
    if (found == blockForSpec_.end()) {
      // There is no corresponding execution part, so there is nothing to do.
      return;
    }
    parser::Block &block = *found->second;

    // There are two places where an OpenMP declarative construct can
    // show up in the tuple in specification part:
    // (1) in std::list<OpenMPDeclarativeConstruct>, or
    // (2) in std::list<DeclarationConstruct>.
    // The case (1) is only possible is the list (2) is empty.

    auto &omps =
        std::get<std::list<parser::OpenMPDeclarativeConstruct>>(spec.t);
    auto &decls = std::get<std::list<parser::DeclarationConstruct>>(spec.t);

    if (!decls.empty()) {
      MoveUtilityConstructsFromDecls(decls, block);
    } else {
      MoveUtilityConstructsFromOmps(omps, block);
    }
  }

  void MoveUtilityConstructsFromDecls(
      std::list<parser::DeclarationConstruct> &decls, parser::Block &block) {
    // Find the trailing range of DeclarationConstructs that are OpenMP
    // utility construct, that are to be moved to the execution part.
    std::list<parser::DeclarationConstruct>::reverse_iterator rlast = [&]() {
      for (auto rit = decls.rbegin(), rend = decls.rend(); rit != rend; ++rit) {
        parser::DeclarationConstruct &dc = *rit;
        if (!std::holds_alternative<parser::SpecificationConstruct>(dc.u)) {
          return rit;
        }
        auto &sc = std::get<parser::SpecificationConstruct>(dc.u);
        using OpenMPDeclarativeConstruct =
            common::Indirection<parser::OpenMPDeclarativeConstruct>;
        if (!std::holds_alternative<OpenMPDeclarativeConstruct>(sc.u)) {
          return rit;
        }
        // Got OpenMPDeclarativeConstruct. If it's not a utility construct
        // then stop.
        auto &odc = std::get<OpenMPDeclarativeConstruct>(sc.u).value();
        if (!std::holds_alternative<parser::OpenMPUtilityConstruct>(odc.u)) {
          return rit;
        }
      }
      return decls.rend();
    }();

    std::transform(decls.rbegin(), rlast, std::front_inserter(block),
        [](parser::DeclarationConstruct &dc) {
          auto &sc = std::get<parser::SpecificationConstruct>(dc.u);
          using OpenMPDeclarativeConstruct =
              common::Indirection<parser::OpenMPDeclarativeConstruct>;
          auto &oc = std::get<OpenMPDeclarativeConstruct>(sc.u).value();
          auto &ut = std::get<parser::OpenMPUtilityConstruct>(oc.u);

          return parser::ExecutionPartConstruct(parser::ExecutableConstruct(
              common::Indirection(parser::OpenMPConstruct(std::move(ut)))));
        });

    decls.erase(rlast.base(), decls.end());
  }

  void MoveUtilityConstructsFromOmps(
      std::list<parser::OpenMPDeclarativeConstruct> &omps,
      parser::Block &block) {
    using OpenMPDeclarativeConstruct = parser::OpenMPDeclarativeConstruct;
    // Find the trailing range of OpenMPDeclarativeConstruct that are OpenMP
    // utility construct, that are to be moved to the execution part.
    std::list<OpenMPDeclarativeConstruct>::reverse_iterator rlast = [&]() {
      for (auto rit = omps.rbegin(), rend = omps.rend(); rit != rend; ++rit) {
        OpenMPDeclarativeConstruct &dc = *rit;
        if (!std::holds_alternative<parser::OpenMPUtilityConstruct>(dc.u)) {
          return rit;
        }
      }
      return omps.rend();
    }();

    std::transform(omps.rbegin(), rlast, std::front_inserter(block),
        [](parser::OpenMPDeclarativeConstruct &dc) {
          auto &ut = std::get<parser::OpenMPUtilityConstruct>(dc.u);
          return parser::ExecutionPartConstruct(parser::ExecutableConstruct(
              common::Indirection(parser::OpenMPConstruct(std::move(ut)))));
        });

    omps.erase(rlast.base(), omps.end());
  }

  // Mapping from the specification parts to the blocks that follow in the
  // same construct. This is for converting utility constructs to executable
  // constructs.
  std::map<parser::SpecificationPart *, parser::Block *> blockForSpec_;
  parser::Messages &messages_;
};

bool CanonicalizeOmp(parser::Messages &messages, parser::Program &program) {
  CanonicalizationOfOmp omp{messages};
  Walk(program, omp);
  return !messages.AnyFatalError();
}
} // namespace Fortran::semantics
