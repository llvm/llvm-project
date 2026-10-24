//===-- lib/Semantics/canonicalize-directives.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "canonicalize-directives.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/tools.h"
#include "flang/Support/PluginDirectives.h"

namespace Fortran::semantics {

using namespace parser::literals;

// Check that directives are associated with the correct constructs.
// Directives that need to be associated with other constructs in the execution
// part are moved to the execution part so they can be checked there.
class CanonicalizationOfDirectives {
public:
  CanonicalizationOfDirectives(parser::Messages &messages)
      : messages_{messages} {}

  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  // Move directives that must appear in the Execution part out of the
  // Specification part.
  void Post(parser::SpecificationPart &spec);
  bool Pre(parser::ExecutionPart &x);

  // A directive defined by a plugin for the loop that follows it, after the
  // declarations, is a declaration construct: move it to the start of the
  // execution part, where it is checked like the others.
  bool Pre(parser::MainProgram &x) {
    MovePluginLoopDirectives(std::get<parser::SpecificationPart>(x.t),
        std::get<parser::ExecutionPart>(x.t).v);
    return true;
  }
  bool Pre(parser::FunctionSubprogram &x) {
    MovePluginLoopDirectives(std::get<parser::SpecificationPart>(x.t),
        std::get<parser::ExecutionPart>(x.t).v);
    return true;
  }
  bool Pre(parser::SubroutineSubprogram &x) {
    MovePluginLoopDirectives(std::get<parser::SpecificationPart>(x.t),
        std::get<parser::ExecutionPart>(x.t).v);
    return true;
  }
  bool Pre(parser::SeparateModuleSubprogram &x) {
    MovePluginLoopDirectives(std::get<parser::SpecificationPart>(x.t),
        std::get<parser::ExecutionPart>(x.t).v);
    return true;
  }
  bool Pre(parser::BlockConstruct &x) {
    MovePluginLoopDirectives(std::get<parser::BlockSpecificationPart>(x.t).v,
        std::get<parser::Block>(x.t));
    return true;
  }

  // Ensure that directives associated with constructs appear accompanying the
  // construct.
  void Post(parser::Block &block);

private:
  // Ensure that loop directives appear immediately before a loop.
  void CheckLoopDirective(parser::CompilerDirective &dir, parser::Block &block,
      std::list<parser::ExecutionPartConstruct>::iterator it);
  // Ensure that a directive a plugin defines for a loop appears immediately
  // before a DO or DO WHILE loop.
  void CheckPluginLoopDirective(parser::CompilerDirective &dir,
      parser::Block &block,
      std::list<parser::ExecutionPartConstruct>::iterator it);
  void MovePluginLoopDirectives(
      parser::SpecificationPart &spec, parser::Block &block);

  parser::Messages &messages_;

  // Directives to be moved to the Execution part from the Specification part.
  std::list<common::Indirection<parser::CompilerDirective>>
      directivesToConvert_;
};

bool CanonicalizeDirectives(
    parser::Messages &messages, parser::Program &program) {
  CanonicalizationOfDirectives dirs{messages};
  Walk(program, dirs);
  return !messages.AnyFatalError();
}

// A directive defined by a plugin whose subject is the loop that follows it.
static bool IsPluginLoopDirective(const parser::CompilerDirective &dir) {
  if (const auto *plugin{
          std::get_if<parser::CompilerDirective::Plugin>(&dir.u)}) {
    const auto &[prefix, keyword, args]{plugin->t};
    const common::PluginDirectiveSpec *spec{
        common::lookupPluginDirective(prefix.ToString(), keyword.ToString())};
    return spec && spec->subject == common::PluginDirectiveSubject::Loop;
  }
  return false;
}

static bool IsExecutionDirective(const parser::CompilerDirective &dir) {
  return IsPluginLoopDirective(dir) ||
      std::holds_alternative<parser::CompilerDirective::VectorAlways>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::VectorLength>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::Unroll>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::UnrollAndJam>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::NoVector>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::NoUnroll>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::NoUnrollAndJam>(
          dir.u) ||
      std::holds_alternative<parser::CompilerDirective::ForceInline>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::Inline>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::NoInline>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::IVDep>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::InlineAlways>(dir.u) ||
      std::holds_alternative<parser::CompilerDirective::Simd>(dir.u);
}

// The directive of a declaration construct, if it is one.
static common::Indirection<parser::CompilerDirective> *GetCompilerDirective(
    parser::DeclarationConstruct &x) {
  if (auto *spec{std::get_if<parser::SpecificationConstruct>(&x.u)}) {
    return std::get_if<common::Indirection<parser::CompilerDirective>>(
        &spec->u);
  }
  return nullptr;
}

void CanonicalizationOfDirectives::MovePluginLoopDirectives(
    parser::SpecificationPart &spec, parser::Block &block) {
  auto &decls{std::get<std::list<parser::DeclarationConstruct>>(spec.t)};
  // The directives after the last declaration.
  auto it{decls.end()};
  while (it != decls.begin() && GetCompilerDirective(*std::prev(it))) {
    --it;
  }
  auto first{block.begin()};
  while (it != decls.end()) {
    common::Indirection<parser::CompilerDirective> *dir{
        GetCompilerDirective(*it)};
    if (IsPluginLoopDirective(dir->value())) {
      block.insert(first,
          parser::ExecutionPartConstruct{
              parser::ExecutableConstruct{std::move(*dir)}});
      it = decls.erase(it);
    } else {
      ++it;
    }
  }
}

void CanonicalizationOfDirectives::Post(parser::SpecificationPart &spec) {
  auto &list{
      std::get<std::list<common::Indirection<parser::CompilerDirective>>>(
          spec.t)};
  for (auto it{list.begin()}; it != list.end();) {
    if (IsExecutionDirective(it->value())) {
      directivesToConvert_.emplace_back(std::move(*it));
      it = list.erase(it);
    } else {
      ++it;
    }
  }
  // A directive for a loop that is still among the declarations has no loop
  // to follow: other declarations do, or the scope has no execution part.
  for (parser::DeclarationConstruct &decl :
      std::get<std::list<parser::DeclarationConstruct>>(spec.t)) {
    if (auto *dir{GetCompilerDirective(decl)};
        dir && IsPluginLoopDirective(dir->value())) {
      const auto &[prefix, keyword, args]{
          std::get<parser::CompilerDirective::Plugin>(dir->value().u).t};
      messages_.Say(dir->value().source,
          "A DO or DO WHILE loop must follow the '%s %s' directive"_err_en_US,
          parser::ToUpperCaseLetters(prefix.ToString()),
          parser::ToUpperCaseLetters(keyword.ToString()));
    }
  }
}

bool CanonicalizationOfDirectives::Pre(parser::ExecutionPart &x) {
  auto origFirst{x.v.begin()};
  for (auto &dir : directivesToConvert_) {
    x.v.insert(origFirst,
        parser::ExecutionPartConstruct{
            parser::ExecutableConstruct{std::move(dir)}});
  }

  directivesToConvert_.clear();
  return true;
}

void CanonicalizationOfDirectives::CheckLoopDirective(
    parser::CompilerDirective &dir, parser::Block &block,
    std::list<parser::ExecutionPartConstruct>::iterator it) {

  // Skip over this and other compiler directives
  while (it != block.end() && parser::Unwrap<parser::CompilerDirective>(*it)) {
    ++it;
  }

  if (it == block.end() ||
      (!parser::Unwrap<parser::DoConstruct>(*it) &&
          !parser::Unwrap<parser::OpenACCLoopConstruct>(*it) &&
          !parser::Unwrap<parser::OpenACCCombinedConstruct>(*it))) {
    std::string s{parser::ToUpperCaseLetters(dir.source.ToString())};
    s.pop_back(); // Remove trailing newline from source string
    messages_.Say(
        dir.source, "A DO loop must follow the %s directive"_warn_en_US, s);
  }
}

void CanonicalizationOfDirectives::CheckPluginLoopDirective(
    parser::CompilerDirective &dir, parser::Block &block,
    std::list<parser::ExecutionPartConstruct>::iterator it) {
  // Skip over this and other compiler directives
  while (it != block.end() && parser::Unwrap<parser::CompilerDirective>(*it)) {
    ++it;
  }
  const parser::DoConstruct *loop{
      it == block.end() ? nullptr : parser::Unwrap<parser::DoConstruct>(*it)};
  if (!loop || loop->IsDoConcurrent()) {
    const auto &[prefix, keyword, args]{
        std::get<parser::CompilerDirective::Plugin>(dir.u).t};
    messages_.Say(dir.source,
        "A DO or DO WHILE loop must follow the '%s %s' directive"_err_en_US,
        parser::ToUpperCaseLetters(prefix.ToString()),
        parser::ToUpperCaseLetters(keyword.ToString()));
  }
}

void CanonicalizationOfDirectives::Post(parser::Block &block) {
  for (auto it{block.begin()}; it != block.end(); ++it) {
    if (auto *dir{parser::Unwrap<parser::CompilerDirective>(*it)}) {
      std::visit(
          common::visitors{[&](parser::CompilerDirective::VectorAlways &) {
                             CheckLoopDirective(*dir, block, it);
                           },
              [&](parser::CompilerDirective::VectorLength &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::Unroll &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::UnrollAndJam &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::NoVector &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::NoUnroll &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::NoUnrollAndJam &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::IVDep &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::Simd &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](parser::CompilerDirective::Plugin &) {
                if (IsPluginLoopDirective(*dir)) {
                  CheckPluginLoopDirective(*dir, block, it);
                }
              },
              [&](auto &) {}},
          dir->u);
    }
  }
}

} // namespace Fortran::semantics
