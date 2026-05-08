//===-- lib/Semantics/canonicalize-omp.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonicalize-omp.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/semantics.h"

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
  CanonicalizationOfOmp(SemanticsContext &context)
      : context_{context}, messages_{context.messages()} {}

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
    CanonicalizeAllocateDirectives(spec);
  }

  void Post(parser::OmpMapClause &map) { CanonicalizeMapModifiers(map); }

private:
  // Canonicalization of allocate directives
  //
  // In OpenMP 5.0 and 5.1 the allocate directive could either be a declarative
  // one or an executable one. As usual in such cases, this poses a problem
  // when the directive appears at the boundary between the specification part
  // and the execution part.
  // The executable form can actually consist of several adjacent directives,
  // whereas the declarative form is always standalone. Additionally, the
  // executable form must be associated with an allocate statement.
  //
  // The parser tries to parse declarative statements first, so in the
  // following case, the two directives will be declarative, even though
  // they should be treated as a single executable form:
  //   integer, allocatable :: x, y   ! Specification
  //   !$omp allocate(x)
  //   !$omp allocate(y)
  //   allocate(x, y)                 ! Execution
  //
  void CanonicalizeAllocateDirectives(parser::SpecificationPart &spec) {
    auto found = blockForSpec_.find(&spec);
    if (found == blockForSpec_.end()) {
      // There is no corresponding execution part, so there is nothing to do.
      return;
    }
    parser::Block &block = *found->second;

    auto isAllocateStmt = [](const parser::ExecutionPartConstruct &epc) {
      if (auto *ec = std::get_if<parser::ExecutableConstruct>(&epc.u)) {
        if (auto *as =
                std::get_if<parser::Statement<parser::ActionStmt>>(&ec->u)) {
          return std::holds_alternative<
              common::Indirection<parser::AllocateStmt>>(as->statement.u);
        }
      }
      return false;
    };

    if (!block.empty() && isAllocateStmt(block.front())) {
      // There are two places where an OpenMP declarative construct can
      // show up in the tuple in specification part:
      // (1) in std::list<OpenMPDeclarativeConstruct>, or
      // (2) in std::list<DeclarationConstruct>.
      // The case (1) is only possible if the list (2) is empty.

      auto &omps =
          std::get<std::list<parser::OpenMPDeclarativeConstruct>>(spec.t);
      auto &decls = std::get<std::list<parser::DeclarationConstruct>>(spec.t);

      if (!decls.empty()) {
        MakeExecutableAllocateFromDecls(decls, block);
      } else {
        MakeExecutableAllocateFromOmps(omps, block);
      }
    }
  }

  parser::ExecutionPartConstruct EmbedInExec(
      parser::OmpAllocateDirective *alo, parser::ExecutionPartConstruct &&epc) {
    // Nest current epc inside the allocate directive.
    std::get<parser::Block>(alo->t).push_front(std::move(epc));
    // Set the new epc to be the ExecutionPartConstruct made from
    // the allocate directive.
    parser::OpenMPConstruct opc(std::move(*alo));
    common::Indirection<parser::OpenMPConstruct> ind(std::move(opc));
    parser::ExecutableConstruct ec(std::move(ind));
    return parser::ExecutionPartConstruct(std::move(ec));
  }

  void MakeExecutableAllocateFromDecls(
      std::list<parser::DeclarationConstruct> &decls, parser::Block &body) {
    using OpenMPDeclarativeConstruct =
        common::Indirection<parser::OpenMPDeclarativeConstruct>;

    auto getAllocate = [](parser::DeclarationConstruct *dc) {
      if (auto *sc = std::get_if<parser::SpecificationConstruct>(&dc->u)) {
        if (auto *odc = std::get_if<OpenMPDeclarativeConstruct>(&sc->u)) {
          if (auto *alo =
                  std::get_if<parser::OmpAllocateDirective>(&odc->value().u)) {
            return alo;
          }
        }
      }
      return static_cast<parser::OmpAllocateDirective *>(nullptr);
    };

    std::list<parser::DeclarationConstruct>::reverse_iterator rlast = [&]() {
      for (auto rit = decls.rbegin(), rend = decls.rend(); rit != rend; ++rit) {
        if (getAllocate(&*rit) == nullptr) {
          return rit;
        }
      }
      return decls.rend();
    }();

    if (rlast != decls.rbegin()) {
      // We have already checked that the first statement in body is
      // ALLOCATE.
      parser::ExecutionPartConstruct epc(std::move(body.front()));
      for (auto rit = decls.rbegin(); rit != rlast; ++rit) {
        epc = EmbedInExec(getAllocate(&*rit), std::move(epc));
      }

      body.pop_front();
      body.push_front(std::move(epc));
      decls.erase(rlast.base(), decls.end());
    }
  }

  void MakeExecutableAllocateFromOmps(
      std::list<parser::OpenMPDeclarativeConstruct> &omps,
      parser::Block &body) {
    using OpenMPDeclarativeConstruct = parser::OpenMPDeclarativeConstruct;

    std::list<OpenMPDeclarativeConstruct>::reverse_iterator rlast = [&]() {
      for (auto rit = omps.rbegin(), rend = omps.rend(); rit != rend; ++rit) {
        if (!std::holds_alternative<parser::OmpAllocateDirective>(rit->u)) {
          return rit;
        }
      }
      return omps.rend();
    }();

    if (rlast != omps.rbegin()) {
      parser::ExecutionPartConstruct epc(std::move(body.front()));
      for (auto rit = omps.rbegin(); rit != rlast; ++rit) {
        epc = EmbedInExec(
            &std::get<parser::OmpAllocateDirective>(rit->u), std::move(epc));
      }

      body.pop_front();
      body.push_front(std::move(epc));
      omps.erase(rlast.base(), omps.end());
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

  // Map clause modifiers are parsed as per OpenMP 6.0 spec. That spec has
  // changed properties of some of the modifiers, for example it has expanded
  // map-type-modifier into 3 individual modifiers (one for each of the
  // possible values of the original modifier), and the "map-type" modifier
  // is no longer ultimate.
  // To utilize the modifier validation framework for semantic checks,
  // if the specified OpenMP version is less than 6.0, rewrite the affected
  // modifiers back into the pre-6.0 forms.
  void CanonicalizeMapModifiers(parser::OmpMapClause &map) {
    unsigned version{context_.langOptions().OpenMPVersion};
    if (version >= 60) {
      return;
    }

    // Omp{Always, Close, Present, xHold}Modifier -> OmpMapTypeModifier
    // OmpDeleteModifier -> OmpMapType
    using Modifier = parser::OmpMapClause::Modifier;
    using Modifiers = std::optional<std::list<Modifier>>;
    auto &modifiers{std::get<Modifiers>(map.t)};
    if (!modifiers) {
      return;
    }

    using MapTypeModifier = parser::OmpMapTypeModifier;
    using MapType = parser::OmpMapType;

    for (auto &mod : *modifiers) {
      if (std::holds_alternative<parser::OmpAlwaysModifier>(mod.u)) {
        mod.u = MapTypeModifier(MapTypeModifier::Value::Always);
      } else if (std::holds_alternative<parser::OmpCloseModifier>(mod.u)) {
        mod.u = MapTypeModifier(MapTypeModifier::Value::Close);
      } else if (std::holds_alternative<parser::OmpPresentModifier>(mod.u)) {
        mod.u = MapTypeModifier(MapTypeModifier::Value::Present);
      } else if (std::holds_alternative<parser::OmpxHoldModifier>(mod.u)) {
        mod.u = MapTypeModifier(MapTypeModifier::Value::Ompx_Hold);
      } else if (std::holds_alternative<parser::OmpDeleteModifier>(mod.u)) {
        mod.u = MapType(MapType::Value::Delete);
      }
    }
  }

  // Mapping from the specification parts to the blocks that follow in the
  // same construct. This is for converting utility constructs to executable
  // constructs.
  std::map<parser::SpecificationPart *, parser::Block *> blockForSpec_;
  SemanticsContext &context_;
  parser::Messages &messages_;
};

bool CanonicalizeOmp(SemanticsContext &context, parser::Program &program) {
  CanonicalizationOfOmp omp{context};
  Walk(program, omp);
  return !context.messages().AnyFatalError();
}
} // namespace Fortran::semantics
