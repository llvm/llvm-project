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
#include "flang/Semantics/semantics.h"

// After Loop Canonicalization, rewrite OpenMP parse tree to make OpenMP
// Constructs more structured which provide explicit scopes for later
// structural checks and semantic analysis.
//   1. Associate declarative OMP allocation directives with their
//      respective executable allocation directive
//   2. TBD
namespace Fortran::semantics {
using namespace parser::literals;

class CanonicalizationOfOmp {
public:
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}
  CanonicalizationOfOmp(SemanticsContext &context)
      : messages_{context.messages()} {}

  bool Pre(parser::BlockConstruct &x);
  bool Pre(parser::MainProgram &x);
  bool Pre(parser::FunctionSubprogram &x);
  bool Pre(parser::SubroutineSubprogram &x);
  bool Pre(parser::SeparateModuleSubprogram &x);

  void Post(parser::SpecificationPart &spec);

private:
  void CanonicalizeAllocateDirectives(parser::SpecificationPart &spec);

  parser::ExecutionPartConstruct EmbedInExec(
      parser::OmpAllocateDirective *alo, parser::ExecutionPartConstruct &&epc);

  void MakeExecutableAllocateFromDecls(
      std::list<parser::DeclarationConstruct> &decls, parser::Block &body);

  void MakeExecutableAllocateFromOmps(
      std::list<parser::OpenMPDeclarativeConstruct> &omps, parser::Block &body);

  void CanonicalizeUtilityDirectives(parser::SpecificationPart &spec);

  void MoveUtilityDirectivesFromDecls(
      std::list<parser::DeclarationConstruct> &decls, parser::Block &block);

  void MoveUtilityDirectivesFromOmps(
      std::list<parser::OpenMPDeclarativeConstruct> &omps,
      parser::Block &block);

  // Mapping from the specification parts to the blocks that follow in the
  // same program unit/construct. This is for converting utility directives
  // to executable constructs.
  std::map<parser::SpecificationPart *, parser::Block *> blockForSpec_;
  parser::Messages &messages_;
};

// Pre-visit all constructs that have both a specification part and
// an execution part, and store the connection between the two.
bool CanonicalizationOfOmp::Pre(parser::BlockConstruct &x) {
  auto *spec = &std::get<parser::BlockSpecificationPart>(x.t).v;
  auto *block = &std::get<parser::Block>(x.t);
  blockForSpec_.insert(std::make_pair(spec, block));
  return true;
}

bool CanonicalizationOfOmp::Pre(parser::MainProgram &x) {
  auto *spec = &std::get<parser::SpecificationPart>(x.t);
  auto *block = &std::get<parser::ExecutionPart>(x.t).v;
  blockForSpec_.insert(std::make_pair(spec, block));
  return true;
}

bool CanonicalizationOfOmp::Pre(parser::FunctionSubprogram &x) {
  auto *spec = &std::get<parser::SpecificationPart>(x.t);
  auto *block = &std::get<parser::ExecutionPart>(x.t).v;
  blockForSpec_.insert(std::make_pair(spec, block));
  return true;
}

bool CanonicalizationOfOmp::Pre(parser::SubroutineSubprogram &x) {
  auto *spec = &std::get<parser::SpecificationPart>(x.t);
  auto *block = &std::get<parser::ExecutionPart>(x.t).v;
  blockForSpec_.insert(std::make_pair(spec, block));
  return true;
}

bool CanonicalizationOfOmp::Pre(parser::SeparateModuleSubprogram &x) {
  auto *spec = &std::get<parser::SpecificationPart>(x.t);
  auto *block = &std::get<parser::ExecutionPart>(x.t).v;
  blockForSpec_.insert(std::make_pair(spec, block));
  return true;
}

void CanonicalizationOfOmp::Post(parser::SpecificationPart &spec) {
  CanonicalizeUtilityDirectives(spec);
  CanonicalizeAllocateDirectives(spec);
}

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
void CanonicalizationOfOmp::CanonicalizeAllocateDirectives(
    parser::SpecificationPart &spec) {
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
    // There are two places where an OpenMP declarative directive can
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

template <typename T> parser::ExecutionPartConstruct MakeExec(T &&x) {
  return parser::ExecutionPartConstruct(parser::ExecutableConstruct(
      common::Indirection(parser::OpenMPConstruct(std::move(x)))));
}

parser::ExecutionPartConstruct CanonicalizationOfOmp::EmbedInExec(
    parser::OmpAllocateDirective *alo, parser::ExecutionPartConstruct &&epc) {
  // Nest current epc inside the allocate directive.
  std::get<parser::Block>(alo->t).push_front(std::move(epc));
  // Set the new epc to be the ExecutionPartConstruct made from
  // the allocate directive.
  return MakeExec(std::move(*alo));
}

void CanonicalizationOfOmp::MakeExecutableAllocateFromDecls(
    std::list<parser::DeclarationConstruct> &decls, parser::Block &body) {
  std::list<parser::DeclarationConstruct>::reverse_iterator rlast = [&]() {
    for (auto rit = decls.rbegin(), rend = decls.rend(); rit != rend; ++rit) {
      if (!parser::Unwrap<parser::OmpAllocateDirective>(*rit)) {
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
      epc = EmbedInExec(
          parser::Unwrap<parser::OmpAllocateDirective>(*rit), std::move(epc));
    }

    body.pop_front();
    body.push_front(std::move(epc));
    decls.erase(rlast.base(), decls.end());
  }
}

void CanonicalizationOfOmp::MakeExecutableAllocateFromOmps(
    std::list<parser::OpenMPDeclarativeConstruct> &omps, parser::Block &body) {
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

// Canonicalization of utility and metadirectives.
//
// This addresses the issue of directives that appear at the boundary
// between the specification and the execution parts, e.g.
//   subroutine foo
//     integer :: x     ! Specification
//     !$omp nothing
//     x = 1            ! Execution
//     ...
//   end
//
// Utility directives (error and nothing) can appear in both the
// specification part and the execution part, except "error at(execution)",
// which cannot be present in the specification part (whereas any utility
// directive can be in the execution part).
// When a utility directive is at the boundary, it should preferably be
// parsed as an element of the execution part, but since the specification
// part is parsed first, the utility diretives ends up belonging to the
// specification part.
//
// To allow the likes of the following code to compile, move all utility
// directives that are at the end of the specification part to the beginning
// of the execution part.
//
// subroutine foo
//   !$omp error at(execution)  ! Initially parsed as declarative directive.
//                              ! Move it to the execution part.
// end

void CanonicalizationOfOmp::CanonicalizeUtilityDirectives(
    parser::SpecificationPart &spec) {
  auto found = blockForSpec_.find(&spec);
  if (found == blockForSpec_.end()) {
    // There is no corresponding execution part, so there is nothing to do.
    return;
  }
  parser::Block &block = *found->second;

  // There are two places where an OpenMP declarative directive can
  // show up in the tuple in specification part:
  // (1) in std::list<OpenMPDeclarativeConstruct>, or
  // (2) in std::list<DeclarationConstruct>.
  // The case (1) is only possible is the list (2) is empty.

  auto &omps = std::get<std::list<parser::OpenMPDeclarativeConstruct>>(spec.t);
  auto &decls = std::get<std::list<parser::DeclarationConstruct>>(spec.t);

  if (!decls.empty()) {
    MoveUtilityDirectivesFromDecls(decls, block);
  } else {
    MoveUtilityDirectivesFromOmps(omps, block);
  }
}

static const parser::OmpDirectiveSpecification *GetODS(
    const parser::OmpClause &clause) {
  if (auto *other = parser::Unwrap<parser::OmpOtherwiseClause>(clause)) {
    return other->v.has_value() ? &other->v->value() : nullptr;
  }
  if (auto *when = parser::Unwrap<parser::OmpWhenClause>(clause)) {
    using OmpDirectiveSpecification =
        std::optional<common::Indirection<parser::OmpDirectiveSpecification>>;
    if (auto &ods = std::get<OmpDirectiveSpecification>(when->t)) {
      return &ods->value();
    }
    return nullptr;
  }
  if (auto *dv = parser::Unwrap<parser::OmpDefaultVariantClause>(clause)) {
    return &dv->v.value();
  }
  return nullptr;
}

static std::optional<bool> IsConsideredExecutable(
    const parser::OmpDirectiveSpecification &x) {
  switch (llvm::omp::getDirectiveCategory(x.DirId())) {
  case llvm::omp::Category::Executable:
    return true;
  case llvm::omp::Category::Utility:
    if (x.DirId() == llvm::omp::Directive::OMPD_nothing) {
      return std::nullopt;
    }
    if (x.DirId() == llvm::omp::Directive::OMPD_error) {
      for (const parser::OmpClause &clause : x.Clauses().v) {
        if (auto *at = parser::Unwrap<parser::OmpAtClause>(clause)) {
          switch (at->v) {
            SWITCH_COVERS_ALL_CASES
          case parser::OmpAtClause::ActionTime::Compilation:
            return std::nullopt;
          case parser::OmpAtClause::ActionTime::Execution:
            return true;
          }
        }
      }
      return std::nullopt;
    }
    return true;
  default:
    return false;
  }
}

static bool IsConsideredExecutable(
    const parser::OmpMetadirectiveDirective &md) {
  // NOTHING as a variant by itself should not decide whether the METADIRECTIVE
  // as a whole is executable or not. But when all variants are NOTHING then
  // the METADIRECTIVE should be treated as executable.
  // The same applies to ERROR AT(COMPILATION).
  bool nonExec = false;
  for (const parser::OmpClause &clause : md.v.Clauses().v) {
    if (auto *ods = GetODS(clause)) {
      if (auto exe = IsConsideredExecutable(*ods)) {
        if (*exe) {
          // If it's known to be executable, return true.
          return true;
        }
        // Known to be non-executable.
        nonExec = true;
      }
    }
    // If the directive specification is missing, it's NOTHING.
  }
  return !nonExec;
}

static bool IsConsideredExecutable(
    const parser::OpenMPDeclarativeConstruct &x) {
  if (std::holds_alternative<parser::OmpUtilityDirective>(x.u)) {
    return true;
  }
  if (auto *md = parser::Unwrap<parser::OmpMetadirectiveDirective>(x)) {
    if (IsConsideredExecutable(*md)) {
      return true;
    }
  }
  return false;
}

static parser::ExecutionPartConstruct MakeExec(
    parser::OpenMPDeclarativeConstruct &x) {
  if (auto *s = std::get_if<parser::OmpMetadirectiveDirective>(&x.u)) {
    return MakeExec(parser::OpenMPStandaloneConstruct(std::move(*s)));
  }
  if (auto *s = std::get_if<parser::OmpUtilityDirective>(&x.u)) {
    return MakeExec(std::move(*s));
  }
  llvm_unreachable("Unexpected OpenMPDeclarativeConstruct");
}

void CanonicalizationOfOmp::MoveUtilityDirectivesFromDecls(
    std::list<parser::DeclarationConstruct> &decls, parser::Block &block) {
  // Find the trailing range of DeclarationConstructs that are OpenMP
  // utility directive, that are to be moved to the execution part.
  std::list<parser::DeclarationConstruct>::reverse_iterator rlast = [&]() {
    for (auto rit = decls.rbegin(), rend = decls.rend(); rit != rend; ++rit) {
      parser::DeclarationConstruct &dc = *rit;
      auto *odc = parser::Unwrap<parser::OpenMPDeclarativeConstruct>(dc);
      if (!odc || !IsConsideredExecutable(*odc)) {
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
        return MakeExec(oc);
      });

  decls.erase(rlast.base(), decls.end());
}

void CanonicalizationOfOmp::MoveUtilityDirectivesFromOmps(
    std::list<parser::OpenMPDeclarativeConstruct> &omps, parser::Block &block) {
  using OpenMPDeclarativeConstruct = parser::OpenMPDeclarativeConstruct;
  // Find the trailing range of OpenMPDeclarativeConstruct that are OpenMP
  // utility directive, that are to be moved to the execution part.
  std::list<OpenMPDeclarativeConstruct>::reverse_iterator rlast = [&]() {
    for (auto rit = omps.rbegin(), rend = omps.rend(); rit != rend; ++rit) {
      if (!IsConsideredExecutable(*rit)) {
        return rit;
      }
    }
    return omps.rend();
  }();

  std::transform(omps.rbegin(), rlast, std::front_inserter(block),
      [](parser::OpenMPDeclarativeConstruct &dc) { return MakeExec(dc); });

  omps.erase(rlast.base(), omps.end());
}

bool CanonicalizeOmp(SemanticsContext &context, parser::Program &program) {
  CanonicalizationOfOmp omp{context};
  Walk(program, omp);
  return !context.messages().AnyFatalError();
}
} // namespace Fortran::semantics
