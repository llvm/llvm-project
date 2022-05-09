//===-- lib/Semantics/program-tree.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "program-tree.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/char-block.h"
#include "flang/Semantics/scope.h"

namespace Fortran::semantics {

static void GetEntryStmts(
    ProgramTree &node, const parser::SpecificationPart &spec) {
  const auto &implicitPart{std::get<parser::ImplicitPart>(spec.t)};
  for (const parser::ImplicitPartStmt &stmt : implicitPart.v) {
    if (const auto *entryStmt{std::get_if<
            parser::Statement<common::Indirection<parser::EntryStmt>>>(
            &stmt.u)}) {
      node.AddEntry(entryStmt->statement.value());
    }
  }
  for (const auto &decl :
      std::get<std::list<parser::DeclarationConstruct>>(spec.t)) {
    if (const auto *entryStmt{std::get_if<
            parser::Statement<common::Indirection<parser::EntryStmt>>>(
            &decl.u)}) {
      node.AddEntry(entryStmt->statement.value());
    }
  }
}

static void GetEntryStmts(
    ProgramTree &node, const parser::ExecutionPart &exec) {
  for (const auto &epConstruct : exec.v) {
    if (const auto *entryStmt{std::get_if<
            parser::Statement<common::Indirection<parser::EntryStmt>>>(
            &epConstruct.u)}) {
      node.AddEntry(entryStmt->statement.value());
    }
  }
}

// Collects generics that define simple names that could include
// identically-named subprograms as specific procedures.
static void GetGenerics(
    ProgramTree &node, const parser::SpecificationPart &spec) {
  for (const auto &decl :
      std::get<std::list<parser::DeclarationConstruct>>(spec.t)) {
    if (const auto *spec{
            std::get_if<parser::SpecificationConstruct>(&decl.u)}) {
      if (const auto *generic{std::get_if<
              parser::Statement<common::Indirection<parser::GenericStmt>>>(
              &spec->u)}) {
        const parser::GenericStmt &genericStmt{generic->statement.value()};
        const auto &genericSpec{std::get<parser::GenericSpec>(genericStmt.t)};
        node.AddGeneric(genericSpec);
      } else if (const auto *interface{
                     std::get_if<common::Indirection<parser::InterfaceBlock>>(
                         &spec->u)}) {
        const parser::InterfaceBlock &interfaceBlock{interface->value()};
        const parser::InterfaceStmt &interfaceStmt{
            std::get<parser::Statement<parser::InterfaceStmt>>(interfaceBlock.t)
                .statement};
        const auto *genericSpec{
            std::get_if<std::optional<parser::GenericSpec>>(&interfaceStmt.u)};
        if (genericSpec && genericSpec->has_value()) {
          node.AddGeneric(**genericSpec);
        }
      }
    }
  }
}

template <typename T>
static ProgramTree BuildSubprogramTree(const parser::Name &name, const T &x) {
  const auto &spec{std::get<parser::SpecificationPart>(x.t)};
  const auto &exec{std::get<parser::ExecutionPart>(x.t)};
  const auto &subps{
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  ProgramTree node{name, spec, &exec};
  GetEntryStmts(node, spec);
  GetEntryStmts(node, exec);
  GetGenerics(node, spec);
  if (subps) {
    for (const auto &subp :
        std::get<std::list<parser::InternalSubprogram>>(subps->t)) {
      common::visit(
          [&](const auto &y) { node.AddChild(ProgramTree::Build(y.value())); },
          subp.u);
    }
  }
  return node;
}

static ProgramTree BuildSubprogramTree(
    const parser::Name &name, const parser::BlockData &x) {
  const auto &spec{std::get<parser::SpecificationPart>(x.t)};
  return ProgramTree{name, spec};
}

template <typename T>
static ProgramTree BuildModuleTree(const parser::Name &name, const T &x) {
  const auto &spec{std::get<parser::SpecificationPart>(x.t)};
  const auto &subps{std::get<std::optional<parser::ModuleSubprogramPart>>(x.t)};
  ProgramTree node{name, spec};
  GetGenerics(node, spec);
  if (subps) {
    for (const auto &subp :
        std::get<std::list<parser::ModuleSubprogram>>(subps->t)) {
      common::visit(
          [&](const auto &y) { node.AddChild(ProgramTree::Build(y.value())); },
          subp.u);
    }
  }
  return node;
}

ProgramTree ProgramTree::Build(const parser::ProgramUnit &x) {
  return common::visit([](const auto &y) { return Build(y.value()); }, x.u);
}

ProgramTree ProgramTree::Build(const parser::MainProgram &x) {
  const auto &stmt{
      std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndProgramStmt>>(x.t)};
  static parser::Name emptyName;
  auto result{stmt ? BuildSubprogramTree(stmt->statement.v, x).set_stmt(*stmt)
                   : BuildSubprogramTree(emptyName, x)};
  return result.set_endStmt(end);
}

ProgramTree ProgramTree::Build(const parser::FunctionSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::FunctionStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndFunctionStmt>>(x.t)};
  const auto &name{std::get<parser::Name>(stmt.statement.t)};
  const parser::LanguageBindingSpec *bindingSpec{};
  if (const auto &suffix{
          std::get<std::optional<parser::Suffix>>(stmt.statement.t)}) {
    if (suffix->binding) {
      bindingSpec = &*suffix->binding;
    }
  }
  return BuildSubprogramTree(name, x)
      .set_stmt(stmt)
      .set_endStmt(end)
      .set_bindingSpec(bindingSpec);
}

ProgramTree ProgramTree::Build(const parser::SubroutineSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::SubroutineStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndSubroutineStmt>>(x.t)};
  const auto &name{std::get<parser::Name>(stmt.statement.t)};
  const parser::LanguageBindingSpec *bindingSpec{};
  if (const auto &binding{std::get<std::optional<parser::LanguageBindingSpec>>(
          stmt.statement.t)}) {
    bindingSpec = &*binding;
  }
  return BuildSubprogramTree(name, x)
      .set_stmt(stmt)
      .set_endStmt(end)
      .set_bindingSpec(bindingSpec);
}

ProgramTree ProgramTree::Build(const parser::SeparateModuleSubprogram &x) {
  const auto &stmt{std::get<parser::Statement<parser::MpSubprogramStmt>>(x.t)};
  const auto &end{
      std::get<parser::Statement<parser::EndMpSubprogramStmt>>(x.t)};
  const auto &name{stmt.statement.v};
  return BuildSubprogramTree(name, x).set_stmt(stmt).set_endStmt(end);
}

ProgramTree ProgramTree::Build(const parser::Module &x) {
  const auto &stmt{std::get<parser::Statement<parser::ModuleStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndModuleStmt>>(x.t)};
  const auto &name{stmt.statement.v};
  return BuildModuleTree(name, x).set_stmt(stmt).set_endStmt(end);
}

ProgramTree ProgramTree::Build(const parser::Submodule &x) {
  const auto &stmt{std::get<parser::Statement<parser::SubmoduleStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndSubmoduleStmt>>(x.t)};
  const auto &name{std::get<parser::Name>(stmt.statement.t)};
  return BuildModuleTree(name, x).set_stmt(stmt).set_endStmt(end);
}

ProgramTree ProgramTree::Build(const parser::BlockData &x) {
  const auto &stmt{std::get<parser::Statement<parser::BlockDataStmt>>(x.t)};
  const auto &end{std::get<parser::Statement<parser::EndBlockDataStmt>>(x.t)};
  static parser::Name emptyName;
  auto result{stmt.statement.v ? BuildSubprogramTree(*stmt.statement.v, x)
                               : BuildSubprogramTree(emptyName, x)};
  return result.set_stmt(stmt).set_endStmt(end);
}

ProgramTree ProgramTree::Build(const parser::CompilerDirective &) {
  DIE("ProgramTree::Build() called for CompilerDirective");
}

const parser::ParentIdentifier &ProgramTree::GetParentId() const {
  const auto *stmt{
      std::get<const parser::Statement<parser::SubmoduleStmt> *>(stmt_)};
  return std::get<parser::ParentIdentifier>(stmt->statement.t);
}

bool ProgramTree::IsModule() const {
  auto kind{GetKind()};
  return kind == Kind::Module || kind == Kind::Submodule;
}

Symbol::Flag ProgramTree::GetSubpFlag() const {
  return GetKind() == Kind::Function ? Symbol::Flag::Function
                                     : Symbol::Flag::Subroutine;
}

bool ProgramTree::HasModulePrefix() const {
  using ListType = std::list<parser::PrefixSpec>;
  const auto *prefixes{common::visit(
      common::visitors{
          [](const parser::Statement<parser::FunctionStmt> *x) {
            return &std::get<ListType>(x->statement.t);
          },
          [](const parser::Statement<parser::SubroutineStmt> *x) {
            return &std::get<ListType>(x->statement.t);
          },
          [](const auto *) -> const ListType * { return nullptr; },
      },
      stmt_)};
  if (prefixes) {
    for (const auto &prefix : *prefixes) {
      if (std::holds_alternative<parser::PrefixSpec::Module>(prefix.u)) {
        return true;
      }
    }
  }
  return false;
}

ProgramTree::Kind ProgramTree::GetKind() const {
  return common::visit(
      common::visitors{
          [](const parser::Statement<parser::ProgramStmt> *) {
            return Kind::Program;
          },
          [](const parser::Statement<parser::FunctionStmt> *) {
            return Kind::Function;
          },
          [](const parser::Statement<parser::SubroutineStmt> *) {
            return Kind::Subroutine;
          },
          [](const parser::Statement<parser::MpSubprogramStmt> *) {
            return Kind::MpSubprogram;
          },
          [](const parser::Statement<parser::ModuleStmt> *) {
            return Kind::Module;
          },
          [](const parser::Statement<parser::SubmoduleStmt> *) {
            return Kind::Submodule;
          },
          [](const parser::Statement<parser::BlockDataStmt> *) {
            return Kind::BlockData;
          },
      },
      stmt_);
}

void ProgramTree::set_scope(Scope &scope) {
  scope_ = &scope;
  CHECK(endStmt_);
  scope.AddSourceRange(*endStmt_);
}

void ProgramTree::AddChild(ProgramTree &&child) {
  children_.emplace_back(std::move(child));
}

void ProgramTree::AddEntry(const parser::EntryStmt &entryStmt) {
  entryStmts_.emplace_back(entryStmt);
}

void ProgramTree::AddGeneric(const parser::GenericSpec &generic) {
  genericSpecs_.emplace_back(generic);
}

} // namespace Fortran::semantics
