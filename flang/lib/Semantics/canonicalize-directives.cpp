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

  // Ensure that directives associated with constructs appear accompanying the
  // construct.
  void Post(parser::Block &block);

private:
  // Ensure that loop directives appear immediately before a loop.
  void CheckLoopDirective(parser::CompilerDirective &dir, parser::Block &block,
      std::list<parser::ExecutionPartConstruct>::iterator it);

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

static bool IsExecutionDirective(const parser::CompilerDirective &dir) {
  return std::holds_alternative<parser::CompilerDirective::VectorAlways>(
             dir.u) ||
      std::holds_alternative<parser::CompilerDirective::Unroll>(dir.u);
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

void CanonicalizationOfDirectives::Post(parser::Block &block) {
  for (auto it{block.begin()}; it != block.end(); ++it) {
    if (auto *dir{parser::Unwrap<parser::CompilerDirective>(*it)}) {
      std::visit(
          common::visitors{[&](parser::CompilerDirective::VectorAlways &) {
                             CheckLoopDirective(*dir, block, it);
                           },
              [&](parser::CompilerDirective::Unroll &) {
                CheckLoopDirective(*dir, block, it);
              },
              [&](auto &) {}},
          dir->u);
    }
  }
}

} // namespace Fortran::semantics
