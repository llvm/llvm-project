//===--- ImpliedSaveCheck.cpp - flang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImpliedSaveCheck.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

namespace Fortran::tidy::bugprone {

using namespace parser::literals;
void ImpliedSaveCheck::Enter(const parser::EntityDecl &entityDecl) {
  const auto &objectName = std::get<parser::ObjectName>(entityDecl.t);
  const auto *symbol = objectName.symbol;
  const auto &scope{symbol->owner()};
  if (symbol && semantics::IsSaved(*symbol) &&
      scope.kind() != semantics::Scope::Kind::Module &&
      !symbol->attrs().test(semantics::Attr::SAVE)) {

    if (scope.hasSAVE()) {
      const auto name = scope.GetName();
      if (name) {
        Say(symbol->name(),
            "Symbol '%s' is implicitly saved in scope '%s', but does not have "
            "the SAVE attribute"_warn_en_US,
            symbol->name(), name->ToString());
      } else {
        Say(symbol->name(),
            "Symbol '%s' is implicitly saved in enclosing scope, but does not "
            "have the SAVE attribute"_warn_en_US,
            symbol->name());
      }
    } else {
      Say(symbol->name(), "Implicit SAVE on symbol '%s'"_warn_en_US,
          symbol->name());
    }
  }
}

} // namespace Fortran::tidy::bugprone
