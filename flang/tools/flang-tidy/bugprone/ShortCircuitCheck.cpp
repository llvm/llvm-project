//===--- ShortCircuitCheck.cpp - flang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ShortCircuitCheck.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include <set>
#include <string>

namespace Fortran::tidy::bugprone {

using namespace parser::literals;

using SymbolSet = std::set<const semantics::Symbol *>;

struct OptionalSymbolCollector
    : public evaluate::SetTraverse<OptionalSymbolCollector, SymbolSet> {
  using Base = evaluate::SetTraverse<OptionalSymbolCollector, SymbolSet>;
  OptionalSymbolCollector() : Base{*this} {}
  using Base::operator();

  SymbolSet operator()(const semantics::Symbol &symbol) const {
    if (semantics::IsOptional(symbol)) {
      return SymbolSet{&symbol};
    }
    return SymbolSet{};
  }

  SymbolSet operator()(const evaluate::ProcedureRef &procRef) const {
    const auto &proc = procRef.proc();

    if (auto *intrinsic = proc.GetSpecificIntrinsic()) {
      if (intrinsic->name == "present") {
        return SymbolSet{};
      }
    }

    return Base::operator()(procRef);
  }
};

struct PresentCallCollector
    : public evaluate::SetTraverse<PresentCallCollector, SymbolSet> {
  using Base = evaluate::SetTraverse<PresentCallCollector, SymbolSet>;
  PresentCallCollector() : Base{*this} {}
  using Base::operator();

  SymbolSet operator()(const evaluate::ProcedureRef &procRef) const {
    const auto &proc = procRef.proc();
    if (auto *intrinsic = proc.GetSpecificIntrinsic()) {
      if (intrinsic->name == "present" && procRef.arguments().size() == 1) {
        const auto &arg = procRef.arguments()[0];
        if (arg && arg->UnwrapExpr()) {
          OptionalSymbolCollector argCollector;
          auto argSymbols = argCollector(*arg->UnwrapExpr());
          return argSymbols;
        }
      }
    }
    return SymbolSet{};
  }
};

void ShortCircuitCheck::Enter(const parser::IfConstruct &ifConstruct) {
  const auto &ifThenStmt{
      std::get<parser::Statement<parser::IfThenStmt>>(ifConstruct.t)};
  const auto &ex{std::get<parser::ScalarLogicalExpr>(ifThenStmt.statement.t)};
  const auto *expr{semantics::GetExpr(context()->getSemanticsContext(), ex)};
  if (!expr) {
    return;
  }

  OptionalSymbolCollector optionalCollector;
  auto optionalSymbols = optionalCollector(*expr);

  PresentCallCollector presentCollector;
  auto presentCallSymbols = presentCollector(*expr);

  for (const auto *optionalSym : optionalSymbols) {
    if (presentCallSymbols.count(optionalSym)) {
      std::string symbolName = optionalSym->name().ToString();
      Say(ifThenStmt.source,
          "optional argument '%s' used in logical expression alongside "
          "present()"_warn_en_US,
          symbolName);
    }
  }
}

} // namespace Fortran::tidy::bugprone
