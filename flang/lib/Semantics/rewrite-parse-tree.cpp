//===-- lib/Semantics/rewrite-parse-tree.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rewrite-parse-tree.h"

#include "flang/Common/indirection.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <list>

namespace Fortran::semantics {

using namespace parser::literals;

/// Convert misidentified statement functions to array element assignments
/// or pointer-valued function result assignments.
/// Convert misidentified format expressions to namelist group names.
/// Convert misidentified character variables in I/O units to integer
/// unit number expressions.
/// Convert misidentified named constants in data statement values to
/// initial data targets
class RewriteMutator {
public:
  RewriteMutator(SemanticsContext &context)
      : context_{context}, errorOnUnresolvedName_{!context.AnyFatalError()},
        messages_{context.messages()} {}

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  void Post(parser::Name &);
  bool Pre(parser::MainProgram &);
  bool Pre(parser::Module &);
  bool Pre(parser::FunctionSubprogram &);
  bool Pre(parser::SubroutineSubprogram &);
  bool Pre(parser::SeparateModuleSubprogram &);
  bool Pre(parser::BlockConstruct &);
  bool Pre(parser::Block &);
  bool Pre(parser::DoConstruct &);
  bool Pre(parser::IfConstruct &);
  bool Pre(parser::ActionStmt &);
  void Post(parser::MainProgram &);
  void Post(parser::FunctionSubprogram &);
  void Post(parser::SubroutineSubprogram &);
  void Post(parser::SeparateModuleSubprogram &);
  void Post(parser::BlockConstruct &);
  void Post(parser::Block &);
  void Post(parser::DoConstruct &);
  void Post(parser::IfConstruct &);
  void Post(parser::ReadStmt &);
  void Post(parser::WriteStmt &);

  // Name resolution yet implemented:
  // TODO: Can some/all of these now be enabled?
  bool Pre(parser::EquivalenceStmt &) { return false; }
  bool Pre(parser::Keyword &) { return false; }
  bool Pre(parser::EntryStmt &) { return false; }
  bool Pre(parser::CompilerDirective &) { return false; }

  // Don't bother resolving names in end statements.
  bool Pre(parser::EndBlockDataStmt &) { return false; }
  bool Pre(parser::EndFunctionStmt &) { return false; }
  bool Pre(parser::EndInterfaceStmt &) { return false; }
  bool Pre(parser::EndModuleStmt &) { return false; }
  bool Pre(parser::EndMpSubprogramStmt &) { return false; }
  bool Pre(parser::EndProgramStmt &) { return false; }
  bool Pre(parser::EndSubmoduleStmt &) { return false; }
  bool Pre(parser::EndSubroutineStmt &) { return false; }
  bool Pre(parser::EndTypeStmt &) { return false; }

  bool Pre(parser::OmpBlockConstruct &);
  bool Pre(parser::OpenMPLoopConstruct &);
  void Post(parser::OmpBlockConstruct &);
  void Post(parser::OpenMPLoopConstruct &);

private:
  void FixMisparsedStmtFuncs(parser::SpecificationPart &, parser::Block &);
  void OpenMPSimdOnly(parser::Block &, bool);
  void OpenMPSimdOnly(parser::SpecificationPart &);

  SemanticsContext &context_;
  bool errorOnUnresolvedName_{true};
  parser::Messages &messages_;
};

// Check that name has been resolved to a symbol
void RewriteMutator::Post(parser::Name &name) {
  if (!name.symbol && errorOnUnresolvedName_) {
    messages_.Say(name.source, "Internal: no symbol found for '%s'"_err_en_US,
        name.source);
  }
}

static bool ReturnsDataPointer(const Symbol &symbol) {
  if (const Symbol * funcRes{FindFunctionResult(symbol)}) {
    return IsPointer(*funcRes) && !IsProcedure(*funcRes);
  } else if (const auto *generic{symbol.detailsIf<GenericDetails>()}) {
    for (auto ref : generic->specificProcs()) {
      if (ReturnsDataPointer(*ref)) {
        return true;
      }
    }
  }
  return false;
}

static bool LoopConstructIsSIMD(parser::OpenMPLoopConstruct *ompLoop) {
  auto &begin = std::get<parser::OmpBeginLoopDirective>(ompLoop->t);
  auto directive = std::get<parser::OmpLoopDirective>(begin.t).v;
  return llvm::omp::allSimdSet.test(directive);
}

// Remove non-SIMD OpenMPConstructs once they are parsed.
// This massively simplifies the logic inside the SimdOnlyPass for
// -fopenmp-simd.
void RewriteMutator::OpenMPSimdOnly(parser::SpecificationPart &specPart) {
  auto &list{std::get<std::list<parser::DeclarationConstruct>>(specPart.t)};
  for (auto it{list.begin()}; it != list.end();) {
    if (auto *specConstr{std::get_if<parser::SpecificationConstruct>(&it->u)}) {
      if (auto *ompDecl{std::get_if<
              common::Indirection<parser::OpenMPDeclarativeConstruct>>(
              &specConstr->u)}) {
        if (std::holds_alternative<parser::OpenMPThreadprivate>(
                ompDecl->value().u) ||
            std::holds_alternative<parser::OpenMPDeclareMapperConstruct>(
                ompDecl->value().u)) {
          it = list.erase(it);
          continue;
        }
      }
    }
    ++it;
  }
}

// Remove non-SIMD OpenMPConstructs once they are parsed.
// This massively simplifies the logic inside the SimdOnlyPass for
// -fopenmp-simd. `isNonSimdLoopBody` should be set to true if `block` is the
// body of a non-simd OpenMP loop. This is to indicate that scan constructs
// should be removed from the body, where they would be kept if it were a simd
// loop.
void RewriteMutator::OpenMPSimdOnly(
    parser::Block &block, bool isNonSimdLoopBody = false) {
  auto replaceInlineBlock =
      [&](std::list<parser::ExecutionPartConstruct> &innerBlock,
          auto it) -> auto {
    auto insertPos = std::next(it);
    block.splice(insertPos, innerBlock);
    block.erase(it);
    return insertPos;
  };

  for (auto it{block.begin()}; it != block.end();) {
    if (auto *stmt{std::get_if<parser::ExecutableConstruct>(&it->u)}) {
      if (auto *omp{std::get_if<common::Indirection<parser::OpenMPConstruct>>(
              &stmt->u)}) {
        if (auto *ompStandalone{std::get_if<parser::OpenMPStandaloneConstruct>(
                &omp->value().u)}) {
          if (std::holds_alternative<parser::OpenMPCancelConstruct>(
                  ompStandalone->u) ||
              std::holds_alternative<parser::OpenMPFlushConstruct>(
                  ompStandalone->u) ||
              std::holds_alternative<parser::OpenMPCancellationPointConstruct>(
                  ompStandalone->u)) {
            it = block.erase(it);
            continue;
          }
          if (auto *constr{std::get_if<parser::OpenMPSimpleStandaloneConstruct>(
                  &ompStandalone->u)}) {
            auto directive = constr->v.DirId();
            // Scan should only be removed from non-simd loops
            if (llvm::omp::simpleStandaloneNonSimdOnlySet.test(directive) ||
                (isNonSimdLoopBody && directive == llvm::omp::OMPD_scan)) {
              it = block.erase(it);
              continue;
            }
          }
        } else if (auto *ompBlock{std::get_if<parser::OmpBlockConstruct>(
                       &omp->value().u)}) {
          it = replaceInlineBlock(std::get<parser::Block>(ompBlock->t), it);
          continue;
        } else if (auto *ompLoop{std::get_if<parser::OpenMPLoopConstruct>(
                       &omp->value().u)}) {
          if (LoopConstructIsSIMD(ompLoop)) {
            ++it;
            continue;
          }
          auto &nest =
              std::get<std::optional<parser::NestedConstruct>>(ompLoop->t);

          if (auto *doConstruct =
                  std::get_if<parser::DoConstruct>(&nest.value())) {
            auto &loopBody = std::get<parser::Block>(doConstruct->t);
            // We can only remove some constructs from a loop when it's _not_ a
            // OpenMP simd loop
            OpenMPSimdOnly(loopBody, /*isNonSimdLoopBody=*/true);
            auto newDoConstruct = std::move(*doConstruct);
            auto newLoop = parser::ExecutionPartConstruct{
                parser::ExecutableConstruct{std::move(newDoConstruct)}};
            it = block.erase(it);
            block.insert(it, std::move(newLoop));
            continue;
          }
        } else if (auto *ompCon{std::get_if<parser::OpenMPSectionsConstruct>(
                       &omp->value().u)}) {
          auto &sections =
              std::get<std::list<parser::OpenMPConstruct>>(ompCon->t);
          auto insertPos = std::next(it);
          for (auto &sectionCon : sections) {
            auto &section =
                std::get<parser::OpenMPSectionConstruct>(sectionCon.u);
            auto &innerBlock = std::get<parser::Block>(section.t);
            block.splice(insertPos, innerBlock);
          }
          block.erase(it);
          it = insertPos;
          continue;
        } else if (auto *atomic{std::get_if<parser::OpenMPAtomicConstruct>(
                       &omp->value().u)}) {
          it = replaceInlineBlock(std::get<parser::Block>(atomic->t), it);
          continue;
        } else if (auto *critical{std::get_if<parser::OpenMPCriticalConstruct>(
                       &omp->value().u)}) {
          it = replaceInlineBlock(std::get<parser::Block>(critical->t), it);
          continue;
        }
      }
    }
    ++it;
  }
}

// Finds misparsed statement functions in a specification part, rewrites
// them into array element assignment statements, and moves them into the
// beginning of the corresponding (execution part's) block.
void RewriteMutator::FixMisparsedStmtFuncs(
    parser::SpecificationPart &specPart, parser::Block &block) {
  auto &list{std::get<std::list<parser::DeclarationConstruct>>(specPart.t)};
  auto origFirst{block.begin()}; // insert each elem before origFirst
  for (auto it{list.begin()}; it != list.end();) {
    bool convert{false};
    if (auto *stmt{std::get_if<
            parser::Statement<common::Indirection<parser::StmtFunctionStmt>>>(
            &it->u)}) {
      if (const Symbol *
          symbol{std::get<parser::Name>(stmt->statement.value().t).symbol}) {
        const Symbol &ultimate{symbol->GetUltimate()};
        convert =
            ultimate.has<ObjectEntityDetails>() || ReturnsDataPointer(ultimate);
        if (convert) {
          auto newStmt{stmt->statement.value().ConvertToAssignment()};
          newStmt.source = stmt->source;
          block.insert(origFirst,
              parser::ExecutionPartConstruct{
                  parser::ExecutableConstruct{std::move(newStmt)}});
        }
      }
    }
    if (convert) {
      it = list.erase(it);
    } else {
      ++it;
    }
  }
}

bool RewriteMutator::Pre(parser::MainProgram &program) {
  FixMisparsedStmtFuncs(std::get<parser::SpecificationPart>(program.t),
      std::get<parser::ExecutionPart>(program.t).v);
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(program.t).v);
    OpenMPSimdOnly(std::get<parser::SpecificationPart>(program.t));
  }
  return true;
}

void RewriteMutator::Post(parser::MainProgram &program) {
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(program.t).v);
  }
}

bool RewriteMutator::Pre(parser::Module &module) {
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::SpecificationPart>(module.t));
  }
  return true;
}

bool RewriteMutator::Pre(parser::FunctionSubprogram &func) {
  FixMisparsedStmtFuncs(std::get<parser::SpecificationPart>(func.t),
      std::get<parser::ExecutionPart>(func.t).v);
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(func.t).v);
  }
  return true;
}

void RewriteMutator::Post(parser::FunctionSubprogram &func) {
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(func.t).v);
  }
}

bool RewriteMutator::Pre(parser::SubroutineSubprogram &subr) {
  FixMisparsedStmtFuncs(std::get<parser::SpecificationPart>(subr.t),
      std::get<parser::ExecutionPart>(subr.t).v);
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(subr.t).v);
  }
  return true;
}

void RewriteMutator::Post(parser::SubroutineSubprogram &subr) {
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(subr.t).v);
  }
}

bool RewriteMutator::Pre(parser::SeparateModuleSubprogram &subp) {
  FixMisparsedStmtFuncs(std::get<parser::SpecificationPart>(subp.t),
      std::get<parser::ExecutionPart>(subp.t).v);
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(subp.t).v);
  }
  return true;
}

void RewriteMutator::Post(parser::SeparateModuleSubprogram &subp) {
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::ExecutionPart>(subp.t).v);
  }
}

bool RewriteMutator::Pre(parser::BlockConstruct &block) {
  FixMisparsedStmtFuncs(std::get<parser::BlockSpecificationPart>(block.t).v,
      std::get<parser::Block>(block.t));
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::Block>(block.t));
  }
  return true;
}

void RewriteMutator::Post(parser::BlockConstruct &block) {
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(std::get<parser::Block>(block.t));
  }
}

bool RewriteMutator::Pre(parser::Block &block) {
  if (context_.langOptions().OpenMPSimd) {
    OpenMPSimdOnly(block);
  }
  return true;
}

void RewriteMutator::Post(parser::Block &block) { this->Pre(block); }

bool RewriteMutator::Pre(parser::OmpBlockConstruct &block) {
  if (context_.langOptions().OpenMPSimd) {
    auto &innerBlock = std::get<parser::Block>(block.t);
    OpenMPSimdOnly(innerBlock);
  }
  return true;
}

void RewriteMutator::Post(parser::OmpBlockConstruct &block) {
  this->Pre(block);
}

bool RewriteMutator::Pre(parser::OpenMPLoopConstruct &ompLoop) {
  if (context_.langOptions().OpenMPSimd) {
    if (LoopConstructIsSIMD(&ompLoop)) {
      return true;
    }
    // If we're looking at a non-simd OpenMP loop, we need to explicitly
    // call OpenMPSimdOnly on the nested loop block while indicating where
    // the block comes from.
    auto &nest = std::get<std::optional<parser::NestedConstruct>>(ompLoop.t);
    if (!nest.has_value()) {
      return true;
    }
    if (auto *doConstruct = std::get_if<parser::DoConstruct>(&*nest)) {
      auto &innerBlock = std::get<parser::Block>(doConstruct->t);
      OpenMPSimdOnly(innerBlock, /*isNonSimdLoopBody=*/true);
    }
  }
  return true;
}

void RewriteMutator::Post(parser::OpenMPLoopConstruct &ompLoop) {
  this->Pre(ompLoop);
}

bool RewriteMutator::Pre(parser::DoConstruct &doConstruct) {
  if (context_.langOptions().OpenMPSimd) {
    auto &innerBlock = std::get<parser::Block>(doConstruct.t);
    OpenMPSimdOnly(innerBlock);
  }
  return true;
}

void RewriteMutator::Post(parser::DoConstruct &doConstruct) {
  this->Pre(doConstruct);
}

bool RewriteMutator::Pre(parser::IfConstruct &ifConstruct) {
  if (context_.langOptions().OpenMPSimd) {
    auto &innerBlock = std::get<parser::Block>(ifConstruct.t);
    OpenMPSimdOnly(innerBlock);
  }
  return true;
}

void RewriteMutator::Post(parser::IfConstruct &ifConstruct) {
  this->Pre(ifConstruct);
}

// Rewrite PRINT NML -> WRITE(*,NML=NML)
bool RewriteMutator::Pre(parser::ActionStmt &x) {
  if (auto *print{std::get_if<common::Indirection<parser::PrintStmt>>(&x.u)};
      print &&
      std::get<std::list<parser::OutputItem>>(print->value().t).empty()) {
    auto &format{std::get<parser::Format>(print->value().t)};
    if (std::holds_alternative<parser::Expr>(format.u)) {
      if (auto *name{parser::Unwrap<parser::Name>(format)}; name &&
          name->symbol && name->symbol->GetUltimate().has<NamelistDetails>() &&
          context_.IsEnabled(common::LanguageFeature::PrintNamelist)) {
        context_.Warn(common::LanguageFeature::PrintNamelist, name->source,
            "nonstandard: namelist in PRINT statement"_port_en_US);
        std::list<parser::IoControlSpec> controls;
        controls.emplace_back(std::move(*name));
        x.u = common::Indirection<parser::WriteStmt>::Make(
            parser::IoUnit{parser::Star{}}, std::optional<parser::Format>{},
            std::move(controls), std::list<parser::OutputItem>{});
      }
    }
  }
  return true;
}

// When a namelist group name appears (without NML=) in a READ or WRITE
// statement in such a way that it can be misparsed as a format expression,
// rewrite the I/O statement's parse tree node as if the namelist group
// name had appeared with NML=.
template <typename READ_OR_WRITE>
void FixMisparsedUntaggedNamelistName(READ_OR_WRITE &x) {
  if (x.iounit && x.format &&
      std::holds_alternative<parser::Expr>(x.format->u)) {
    if (const parser::Name * name{parser::Unwrap<parser::Name>(x.format)}) {
      if (name->symbol && name->symbol->GetUltimate().has<NamelistDetails>()) {
        x.controls.emplace_front(parser::IoControlSpec{std::move(*name)});
        x.format.reset();
      }
    }
  }
}

// READ(CVAR) [, ...] will be misparsed as UNIT=CVAR; correct
// it to READ CVAR [,...] with CVAR as a format rather than as
// an internal I/O unit for unformatted I/O, which Fortran does
// not support.
void RewriteMutator::Post(parser::ReadStmt &x) {
  if (x.iounit && !x.format && x.controls.empty()) {
    if (auto *var{std::get_if<parser::Variable>(&x.iounit->u)}) {
      const parser::Name &last{parser::GetLastName(*var)};
      DeclTypeSpec *type{last.symbol ? last.symbol->GetType() : nullptr};
      if (type && type->category() == DeclTypeSpec::Character) {
        x.format = common::visit(
            [](auto &&indirection) {
              return parser::Expr{std::move(indirection)};
            },
            std::move(var->u));
        x.iounit.reset();
      }
    }
  }
  FixMisparsedUntaggedNamelistName(x);
}

void RewriteMutator::Post(parser::WriteStmt &x) {
  FixMisparsedUntaggedNamelistName(x);
}

bool RewriteParseTree(SemanticsContext &context, parser::Program &program) {
  RewriteMutator mutator{context};
  parser::Walk(program, mutator);
  return !context.AnyFatalError();
}

} // namespace Fortran::semantics
