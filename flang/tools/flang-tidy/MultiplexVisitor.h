//===--- MultiplexVisitor.h - flang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_MULTIPLEXVISITOR_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_MULTIPLEXVISITOR_H

#include "FlangTidyCheck.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/semantics.h"
#include <memory>
#include <vector>

namespace Fortran::tidy {

class MultiplexVisitor {
public:
  MultiplexVisitor(semantics::SemanticsContext &context) : context_{context} {}

  void AddChecker(std::unique_ptr<FlangTidyCheck> checker) {
    checkers_.emplace_back(std::move(checker));
  }

  template <typename N>
  bool Pre(const N &node) {
    if constexpr (common::HasMember<const N *, semantics::ConstructNode>) {
      context_.PushConstruct(node);
    }
    for (auto &checker : checkers_) {
      checker->Enter(node);
    }
    return true;
  }

  template <typename N>
  void Post(const N &node) {
    for (auto &checker : checkers_) {
      checker->Leave(node);
    }
    if constexpr (common::HasMember<const N *, semantics::ConstructNode>) {
      context_.PopConstruct();
    }
  }

  template <typename T>
  bool Pre(const parser::Statement<T> &node) {
    if (context_.IsInModuleFile(node.source))
      return true;
    context_.set_location(node.source);
    for (auto &checker : checkers_) {
      checker->Enter(node);
    }
    return true;
  }

  template <typename T>
  bool Pre(const parser::UnlabeledStatement<T> &node) {
    if (context_.IsInModuleFile(node.source))
      return true;
    context_.set_location(node.source);
    for (auto &checker : checkers_) {
      checker->Enter(node);
    }
    return true;
  }

  template <typename T>
  void Post(const parser::Statement<T> &node) {
    if (context_.IsInModuleFile(node.source))
      return;
    for (auto &checker : checkers_) {
      checker->Leave(node);
    }
    context_.set_location(std::nullopt);
  }

  template <typename T>
  void Post(const parser::UnlabeledStatement<T> &node) {
    if (context_.IsInModuleFile(node.source))
      return;
    for (auto &checker : checkers_) {
      checker->Leave(node);
    }
    context_.set_location(std::nullopt);
  }

  bool Walk(const parser::Program &program) {
    parser::Walk(program, *this);
    return !context_.AnyFatalError();
  }

public:
  semantics::SemanticsContext &context_;
  std::vector<std::unique_ptr<FlangTidyCheck>> checkers_;
};

} // namespace Fortran::tidy

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_MULTIPLEXVISITOR_H
