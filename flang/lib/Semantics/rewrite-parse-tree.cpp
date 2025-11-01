//===-- lib/Semantics/rewrite-parse-tree.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rewrite-parse-tree.h"

#include "flang/Common/indirection.h"
#include "flang/Parser/openmp-utils.h"
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

class ReplacementTemp {
public:
  ReplacementTemp() {}

  void createTempSymbol(
      SourceName &source, Scope &scope, SemanticsContext &context);
  void setOriginalSubscriptInt(
      std::list<parser::SectionSubscript> &sectionSubscript);
  Symbol *getTempSymbol() { return replacementTempSymbol_; }
  Symbol *getPrivateReductionSymbol() { return privateReductionSymbol_; }
  parser::CharBlock getOriginalSource() { return originalSource_; }
  parser::Name getOriginalName() { return originalName_; }
  parser::CharBlock getOriginalSubscript() {
    return originalSubscriptCharBlock_;
  }
  Scope *getTempScope() { return tempScope_; }
  bool isArrayElementReassigned() { return arrayElementReassigned_; }
  bool isSectionTriplet() { return isSectionTriplet_; }
  void arrayElementReassigned() { arrayElementReassigned_ = true; }
  void setOriginalName(parser::Name &name) {
    originalName_ = common::Clone(name);
  }
  void setOriginalSource(parser::CharBlock &source) {
    originalSource_ = source;
  }
  void setOriginalSubscriptInt(parser::CharBlock &subscript) {
    originalSubscriptCharBlock_ = subscript;
  }
  void setTempScope(Scope &scope) { tempScope_ = &scope; };
  void setTempSymbol(Symbol *symbol) { replacementTempSymbol_ = symbol; }

private:
  Symbol *replacementTempSymbol_{nullptr};
  Symbol *privateReductionSymbol_{nullptr};
  Scope *tempScope_{nullptr};
  parser::CharBlock originalSource_;
  parser::Name originalName_;
  parser::CharBlock originalSubscriptCharBlock_;
  bool arrayElementReassigned_{false};
  bool isSectionTriplet_{false};
};

class RewriteOmpReductionArrayElements {
public:
  explicit RewriteOmpReductionArrayElements(SemanticsContext &context)
      : context_(context) {}
  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  void Post(parser::Block &block);
  void Post(parser::Variable &var);
  void Post(parser::Expr &expr);
  void Post(parser::AssignmentStmt &assignmentStmt);
  void Post(parser::PointerAssignmentStmt &ptrAssignmentStmt);
  void rewriteReductionArrayElementToTemp(parser::Block &block);
  bool isArrayElementRewritten() { return arrayElementReassigned_; }

private:
  bool isMatchingArrayElement(parser::Designator &existingDesignator);
  template <typename T>
  void processFunctionReference(
      T &node, parser::CharBlock source, parser::FunctionReference &funcRef);
  parser::Designator makeTempDesignator(parser::CharBlock source);
  bool rewriteArrayElementToTemp(parser::Block::iterator &it,
      parser::Designator &designator, parser::Block &block,
      ReplacementTemp &temp);
  bool identifyArrayElementReduced(
      parser::Designator &designator, ReplacementTemp &temp);
  void reassignTempValueToArrayElement(parser::ArrayElement &arrayElement);
  void setCurrentTemp(ReplacementTemp *temp) { currentTemp_ = temp; }
  void resetCurrentTemp() { currentTemp_ = nullptr; }

  SemanticsContext &context_;
  bool arrayElementReassigned_{false};
  parser::Block::iterator reassignmentInsertionPoint_;
  parser::Block *block_{nullptr};
  ReplacementTemp *currentTemp_{nullptr};
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
  return llvm::omp::allSimdSet.test(ompLoop->BeginDir().DirName().v);
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

void ReplacementTemp::createTempSymbol(
    SourceName &source, Scope &scope, SemanticsContext &context) {
  replacementTempSymbol_ =
      const_cast<Scope &>(originalName_.symbol->owner()).FindSymbol(source);
  replacementTempSymbol_->set_scope(
      &const_cast<Scope &>(originalName_.symbol->owner()));
  DeclTypeSpec *tempType = originalName_.symbol->GetUltimate().GetType();
  replacementTempSymbol_->get<ObjectEntityDetails>().set_type(*tempType);
  replacementTempSymbol_->flags().set(Symbol::Flag::CompilerCreated);
}

void ReplacementTemp::setOriginalSubscriptInt(
    std::list<parser::SectionSubscript> &sectionSubscript) {
  bool setSubscript{false};

  auto visitDataRef = [&](parser::DataRef &dataRef) {
    std::visit(
        llvm::makeVisitor(
            [&](parser::Name &name) { setOriginalSubscriptInt(name.source); },
            [&](auto &) {}),
        dataRef.u);
  };
  auto visitDesignator = [&](parser::Designator &designator) {
    std::visit(llvm::makeVisitor(
                   [&](parser::DataRef &dataRef) { visitDataRef(dataRef); },
                   [&](auto &) {}),
        designator.u);
  };
  auto visitLiteralConstant = [&](parser::LiteralConstant &literalConstant) {
    std::visit(llvm::makeVisitor(
                   [&](parser::IntLiteralConstant &intLiteralConstant) {
                     originalSubscriptCharBlock_ =
                         std::get<parser::CharBlock>(intLiteralConstant.t);
                     setSubscript = true;
                   },
                   [&](auto &) {}),
        literalConstant.u);
  };
  auto visitIntExpr = [&](parser::IntExpr &intExpr) {
    parser::Expr &expr = intExpr.thing.value();
    std::visit(
        llvm::makeVisitor(
            [&](parser::LiteralConstant &literalConstant) -> void {
              visitLiteralConstant(literalConstant);
            },
            [&](common::Indirection<parser::Designator> &designator) -> void {
              visitDesignator(designator.value());
            },
            [&](auto &) {}),
        expr.u);
  };
  for (parser::SectionSubscript &subscript : sectionSubscript) {
    std::visit(llvm::makeVisitor(
                   [&](parser::IntExpr &intExpr) { visitIntExpr(intExpr); },
                   [&](parser::SubscriptTriplet &triplet) {
                     isSectionTriplet_ = true;
                     setSubscript = true;
                   },
                   [&](auto &) {}),
        subscript.u);
    if (setSubscript) {
      break;
    }
  }
}

void RewriteOmpReductionArrayElements::rewriteReductionArrayElementToTemp(
    parser::Block &block) {
  if (block.empty()) {
    return;
  }

  auto processReductionClause = [&](const parser::OmpObjectList *objectList,
                                    parser::OpenMPLoopConstruct &ompLoop,
                                    parser::Block::iterator &it) -> bool {
    bool rewritten{false};
    for (const parser::OmpObject &object : objectList->v) {
      ReplacementTemp temp;
      rewritten |= std::visit(
          llvm::makeVisitor(
              [&](const parser::Designator &designator) -> bool {
                return rewriteArrayElementToTemp(it,
                    const_cast<parser::Designator &>(designator), block, temp);
              },
              [&](const auto &) -> bool { return false; }),
          object.u);

      std::optional<parser::NestedConstruct> &NestedConstruct =
          std::get<std::optional<parser::NestedConstruct>>(ompLoop.t);
      if (!NestedConstruct.has_value()) {
        return false;
      }
      if (parser::DoConstruct *
          doConst{std::get_if<parser::DoConstruct>(&NestedConstruct.value())}) {
        block_ = &block;
        parser::Block &doBlock{std::get<parser::Block>(doConst->t)};
        parser::Walk(doBlock, *this);
        // Reset the current temp value so future
        // iterations use their own version.
        resetCurrentTemp();
      }
    };
    return rewritten;
  };
  auto processOmpClause = [&](const parser::OmpClause &clause,
                              parser::OpenMPLoopConstruct &ompLoop,
                              parser::Block::iterator &it) {
    return std::visit(
        llvm::makeVisitor(
            [&](const parser::OmpClause::Reduction &reductionClause) -> bool {
              const parser::OmpObjectList *objectList =
                  parser::omp::GetOmpObjectList(clause);
              return processReductionClause(objectList, ompLoop, it);
            },
            [&](auto &) -> bool { return false; }),
        clause.u);
  };
  auto visitOpenMPLoopConstruct = [&](parser::OpenMPLoopConstruct &ompLoop,
                                      parser::Block::iterator &it) {
    const parser::OmpClauseList &clauseList{ompLoop.BeginDir().Clauses()};

    for (const parser::OmpClause &clause : clauseList.v) {
      if (!processOmpClause(clause, ompLoop, it)) {
        return;
      }
    }
  };
  auto visitOpenMPConstruct = [&](parser::OpenMPConstruct &ompConstruct,
                                  parser::Block::iterator &it) {
    std::visit(llvm::makeVisitor(
                   [&](parser::OpenMPLoopConstruct &ompLoop) {
                     visitOpenMPLoopConstruct(ompLoop, it);
                   },
                   [&](auto &) {}),
        ompConstruct.u);
  };
  auto visitExecutableConstruct = [&](parser::ExecutableConstruct
                                          &execConstruct,
                                      parser::Block::iterator &it) {
    std::visit(
        llvm::makeVisitor(
            [&](common::Indirection<parser::OpenMPConstruct> &ompConstruct) {
              visitOpenMPConstruct(ompConstruct.value(), it);
            },
            [&](auto &) {}),
        execConstruct.u);
  };
  for (auto it{block.begin()}; it != block.end(); ++it) {
    std::visit(llvm::makeVisitor(
                   [&](parser::ExecutableConstruct &execConstruct) {
                     visitExecutableConstruct(execConstruct, it);
                   },
                   [&](auto &) {}),
        it->u);
  }
}

bool RewriteOmpReductionArrayElements::isMatchingArrayElement(
    parser::Designator &existingDesignator) {
  bool matchesArrayElement{false};
  std::list<parser::SectionSubscript> *subscripts{nullptr};

  auto visitName = [&](parser::Name &name) {
    if (name.symbol->GetUltimate() ==
        currentTemp_->getOriginalName().symbol->GetUltimate()) {
      matchesArrayElement = true;
    }
  };
  auto visitArrayElement = [&](parser::ArrayElement &arrayElement) {
    subscripts = &arrayElement.subscripts;
    std::visit(llvm::makeVisitor(
                   [&](parser::Name &name) {
                     visitName(name);

                     if (!currentTemp_->isArrayElementReassigned()) {
                       reassignTempValueToArrayElement(arrayElement);
                     }
                   },
                   [](auto &) {}),
        arrayElement.base.u);
  };
  auto visitDataRef = [&](parser::DataRef &dataRef) {
    std::visit(
        llvm::makeVisitor(
            [&](common::Indirection<parser::ArrayElement> &arrayElement) {
              visitArrayElement(arrayElement.value());
            },
            [&](parser::Name &name) {
              if (name.symbol->GetUltimate() ==
                  currentTemp_->getOriginalName().symbol->GetUltimate()) {
                matchesArrayElement = true;
              }
            },
            [](auto &) {}),
        dataRef.u);
  };
  std::visit(llvm::makeVisitor(
                 [&](parser::DataRef &dataRef) { visitDataRef(dataRef); },
                 [&](auto &) {}),
      existingDesignator.u);

  if (subscripts && matchesArrayElement) {
    bool foundSubscript{false};

    auto visitDataRef = [&](parser::DataRef &dataRef) -> bool {
      return std::visit(llvm::makeVisitor(
                            [&](parser::Name &name) -> bool {
                              foundSubscript = true;
                              return name.source ==
                                  currentTemp_->getOriginalSubscript();
                            },
                            [&](auto &) -> bool { return false; }),
          dataRef.u);
    };
    auto visitLiteralConstant =
        [&](parser::LiteralConstant &literalConstant) -> bool {
      return std::visit(
          llvm::makeVisitor(
              [&](parser::IntLiteralConstant &intLiteralConstant) -> bool {
                foundSubscript = true;
                return std::get<parser::CharBlock>(intLiteralConstant.t) ==
                    currentTemp_->getOriginalSubscript();
              },
              [](auto &) -> bool { return false; }),
          literalConstant.u);
    };
    auto visitIntExpr = [&](parser::IntExpr &intExpr) -> bool {
      parser::Expr &expr = intExpr.thing.value();
      return std::visit(
          llvm::makeVisitor(
              [&](parser::LiteralConstant &literalConstant) -> bool {
                return visitLiteralConstant(literalConstant);
              },
              [&](common::Indirection<parser::Designator> &designator) -> bool {
                return std::visit(llvm::makeVisitor(
                                      [&](parser::DataRef &dataRef) -> bool {
                                        return visitDataRef(dataRef);
                                      },
                                      [&](auto &) -> bool { return false; }),
                    designator.value().u);
              },
              [](auto &) -> bool { return false; }),
          expr.u);
    };
    for (parser::SectionSubscript &subscript : *subscripts) {
      assert(currentTemp_ != nullptr &&
          "Value for ReplacementTemp should have "
          "been found");
      matchesArrayElement =
          std::visit(llvm::makeVisitor(
                         [&](parser::IntExpr &intExpr) -> bool {
                           return visitIntExpr(intExpr);
                         },
                         [](auto &) -> bool { return false; }),
              subscript.u);
      if (foundSubscript) {
        break;
      }
    }
  }
  return matchesArrayElement;
}

template <typename T>
void RewriteOmpReductionArrayElements::processFunctionReference(
    T &node, parser::CharBlock source, parser::FunctionReference &funcRef) {
  auto visitFunctionReferenceName = [&](parser::Name &functionReferenceName)
      -> std::optional<parser::Designator> {
    if (currentTemp_->getOriginalName().symbol ==
        functionReferenceName.symbol) {
      return funcRef.ConvertToArrayElementRef();
    }
    return std::nullopt;
  };

  auto &[procedureDesignator, ArgSpecList] = funcRef.v.t;
  std::optional<parser::Designator> arrayElementDesignator =
      std::visit(llvm::makeVisitor(
                     [&](parser::Name &functionReferenceName)
                         -> std::optional<parser::Designator> {
                       return visitFunctionReferenceName(functionReferenceName);
                     },
                     [&](auto &) -> std::optional<parser::Designator> {
                       return std::nullopt;
                     }),
          procedureDesignator.u);

  if (arrayElementDesignator.has_value()) {
    if (this->isMatchingArrayElement(arrayElementDesignator.value())) {
      node = T{
          common::Indirection<parser::Designator>{makeTempDesignator(source)}};
    }
  }
}

parser::Designator RewriteOmpReductionArrayElements::makeTempDesignator(
    parser::CharBlock source) {
  parser::Name tempVariableName{currentTemp_->getTempSymbol()->name()};
  tempVariableName.symbol = currentTemp_->getTempSymbol();
  parser::Designator tempDesignator{
      parser::DataRef{std::move(tempVariableName)}};
  tempDesignator.source = source;
  return tempDesignator;
}

bool RewriteOmpReductionArrayElements::rewriteArrayElementToTemp(
    parser::Block::iterator &it, parser::Designator &designator,
    parser::Block &block, ReplacementTemp &temp) {

  if (!identifyArrayElementReduced(designator, temp)) {
    return false;
  }
  if (temp.isSectionTriplet()) {
    return false;
  }

  reassignmentInsertionPoint_ = it;
  std::string tempSourceString = "reduction_temp_" +
      temp.getOriginalSource().ToString() + "(" +
      temp.getOriginalSubscript().ToString() + ")";
  SourceName source = context_.SaveTempName(std::move(tempSourceString));
  Scope &scope = const_cast<Scope &>(temp.getOriginalName().symbol->owner());
  if (Symbol * symbol{scope.FindSymbol(source)}) {
    temp.setTempSymbol(symbol);
  } else {
    if (scope.try_emplace(source, Attrs{}, ObjectEntityDetails{}).second) {
      temp.createTempSymbol(source, scope, context_);
    } else {
      common::die(
          "Failed to create temp symbol for %s", source.ToString().c_str());
    }
  }
  setCurrentTemp(&temp);
  temp.setTempScope(scope);

  // Assign the value of the array element to the
  // temporary variable
  parser::Variable newVariable{makeTempDesignator(temp.getOriginalSource())};
  parser::Expr newExpr{
      common::Indirection<parser::Designator>{std::move(designator)}};
  newExpr.source = temp.getOriginalSource();
  std::tuple<parser::Variable, parser::Expr> newT{
      std::move(newVariable), std::move(newExpr)};
  parser::AssignmentStmt assignment{std::move(newT)};
  parser::ExecutionPartConstruct tempVariablePartConstruct{
      parser::ExecutionPartConstruct{
          parser::ExecutableConstruct{parser::Statement<parser::ActionStmt>{
              std::optional<parser::Label>{}, std::move(assignment)}}}};
  block.insert(it, std::move(tempVariablePartConstruct));
  arrayElementReassigned_ = true;

  designator = makeTempDesignator(temp.getOriginalSource());
  return true;
}

bool RewriteOmpReductionArrayElements::identifyArrayElementReduced(
    parser::Designator &designator, ReplacementTemp &temp) {
  auto visitArrayElement = [&](parser::ArrayElement &arrayElement) -> bool {
    std::visit(llvm::makeVisitor(
                   [&](parser::Name &name) -> void {
                     temp.setOriginalName(name);
                     temp.setOriginalSource(name.source);
                   },
                   [&](auto &) -> void {}),
        arrayElement.base.u);
    temp.setOriginalSubscriptInt(arrayElement.subscripts);
    return !temp.isSectionTriplet();
  };
  auto visitDataRef = [&](parser::DataRef &dataRef) -> bool {
    return std::visit(
        llvm::makeVisitor(
            [&](common::Indirection<parser::ArrayElement> &arrayElement)
                -> bool { return visitArrayElement(arrayElement.value()); },
            [&](auto &) -> bool { return false; }),
        dataRef.u);
  };
  return std::visit(llvm::makeVisitor(
                        [&](parser::DataRef &dataRef) -> bool {
                          return visitDataRef(dataRef);
                        },
                        [&](auto &) -> bool { return false; }),
      designator.u);
}

void RewriteOmpReductionArrayElements::reassignTempValueToArrayElement(
    parser::ArrayElement &arrayElement) {
  assert(block_ && "Need iterator to reassign");
  parser::CharBlock originalSource = currentTemp_->getOriginalSource();
  parser::DataRef reassignmentDataRef{std::move(arrayElement)};
  common::Indirection<parser::Designator> arrayElementDesignator{
      std::move(reassignmentDataRef)};
  arrayElementDesignator.value().source = originalSource;
  parser::Variable exisitingVar{std::move(arrayElementDesignator)};
  std::get<common::Indirection<parser::Designator>>(exisitingVar.u)
      .value()
      .source = originalSource;
  parser::Expr reassignmentExpr{makeTempDesignator(originalSource)};
  SourceName source{"reductionTemp"};
  reassignmentExpr.source = source;
  std::tuple<parser::Variable, parser::Expr> reassignment{
      std::move(exisitingVar), std::move(reassignmentExpr)};
  parser::AssignmentStmt reassignStmt{std::move(reassignment)};
  parser::ExecutionPartConstruct tempVariableReassignment{
      parser::ExecutionPartConstruct{
          parser::ExecutableConstruct{parser::Statement<parser::ActionStmt>{
              std::optional<parser::Label>{}, std::move(reassignStmt)}}}};
  block_->insert(std::next(reassignmentInsertionPoint_),
      std::move(tempVariableReassignment));
  currentTemp_->arrayElementReassigned();
}

void RewriteOmpReductionArrayElements::Post(
    parser::AssignmentStmt &assignmentStmt) {
  if (arrayElementReassigned_) {
    // The typed expression needs to be reset where we are reassigning array
    // elements so the semantics can regenerate the expressions correctly.
    assignmentStmt.typedAssignment.Reset();
  }
}
void RewriteOmpReductionArrayElements::Post(
    parser::PointerAssignmentStmt &ptrAssignmentStmt) {
  if (arrayElementReassigned_) {
    // The typed expression needs to be reset where we are reassigning array
    // elements so the semantics can regenerate the expressions correctly.
    ptrAssignmentStmt.typedAssignment.Reset();
  }
}
void RewriteOmpReductionArrayElements::Post(parser::Variable &var) {
  if (currentTemp_) {
    std::visit(
        llvm::makeVisitor(
            [&](common::Indirection<parser::FunctionReference> &funcRef)
                -> void {
              this->processFunctionReference<parser::Variable>(
                  var, var.GetSource(), funcRef.value());
            },
            [&](common::Indirection<parser::Designator> &designator) -> void {
              if (isMatchingArrayElement(designator.value())) {
                designator = makeTempDesignator(designator.value().source);
                var = parser::Variable{std::move(designator)};
              }
            },
            [&](auto &) -> void {}),
        var.u);
  }
  if (arrayElementReassigned_) {
    // The typed expression needs to be reset where we are reassigning array
    // elements so the semantics can regenerate the expressions correctly.
    var.typedExpr.Reset();
  }
}
void RewriteOmpReductionArrayElements::Post(parser::Expr &expr) {
  if (currentTemp_) {
    std::visit(
        llvm::makeVisitor(
            [&](common::Indirection<parser::FunctionReference> &funcRef)
                -> void {
              this->processFunctionReference<parser::Expr>(
                  expr, expr.source, funcRef.value());
            },
            [&](common::Indirection<parser::Designator> &designator) -> void {
              if (isMatchingArrayElement(designator.value())) {
                designator = makeTempDesignator(designator.value().source);
                expr = parser::Expr{std::move(designator)};
              }
            },
            [&](auto &) {}),
        expr.u);
  }
  if (arrayElementReassigned_) {
    // The typed expression needs to be reset where we are reassigning array
    // elements so the semantics can regenerate the expressions correctly.
    expr.typedExpr.Reset();
  }
}

void RewriteOmpReductionArrayElements::Post(parser::Block &block) {
  rewriteReductionArrayElementToTemp(block);
}

bool RewriteParseTree(SemanticsContext &context, parser::Program &program) {
  RewriteMutator mutator{context};
  parser::Walk(program, mutator);
  return !context.AnyFatalError();
}

bool RewriteReductionArrayElements(
    SemanticsContext &context, parser::Program &program) {
  RewriteOmpReductionArrayElements mutator{context};
  parser::Walk(program, mutator);
  return mutator.isArrayElementRewritten();
}

} // namespace Fortran::semantics
