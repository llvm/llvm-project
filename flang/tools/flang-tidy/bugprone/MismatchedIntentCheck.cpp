//===--- MismatchedIntentCheck.cpp - flang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MismatchedIntentCheck.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include <unordered_map>

namespace Fortran::tidy::bugprone {

using namespace parser::literals;

static std::string IntentToString(common::Intent intent) {
  switch (intent) {
  case common::Intent::In:
    return "in";
  case common::Intent::Out:
    return "out";
  case common::Intent::InOut:
    return "inout";
  default:
    return "unknown";
  }
}

static bool IsMoreRestrictive(common::Intent first, common::Intent second) {
  if (first == common::Intent::In &&
      (second == common::Intent::InOut || second == common::Intent::Out)) {
    return true;
  }
  return false;
}

static bool AreConflictingIntents(common::Intent first, common::Intent second) {
  if ((first == common::Intent::In && second == common::Intent::Out) ||
      (first == common::Intent::Out && second == common::Intent::In)) {
    return true;
  }
  return false;
}

void MismatchedIntentCheck::Enter(const parser::CallStmt &callStmt) {
  const auto *procedureRef = callStmt.typedCall.get();
  if (!procedureRef) {
    return;
  }

  std::unordered_map<const semantics::Symbol *, common::Intent> argIntents;

  for (const auto &arg : procedureRef->arguments()) {
    if (!arg)
      continue;

    (void)IntentToString;
    common::Intent intent{arg->dummyIntent()};

    if (const auto *expr{arg->UnwrapExpr()}) {
      if (const auto *symbol{evaluate::UnwrapWholeSymbolDataRef(*expr)}) {
        if (argIntents.find(symbol) != argIntents.end()) {
          common::Intent existingIntent = argIntents[symbol];

          if (existingIntent != intent &&
              (IsMoreRestrictive(existingIntent, intent) ||
               AreConflictingIntents(existingIntent, intent))) {
            Say(callStmt.source,
                "argument '%s' has mismatched intent"_warn_en_US,
                symbol->name());
          }
        } else {
          argIntents[symbol] = intent;
        }
      } else if (const auto component{evaluate::ExtractDataRef(*expr, true)}) {
        const semantics::Symbol &baseSymbol = component->GetFirstSymbol();

        if (argIntents.find(&baseSymbol) != argIntents.end()) {
          common::Intent baseIntent = argIntents[&baseSymbol];

          if (IsMoreRestrictive(baseIntent, intent)) {
            Say(callStmt.source,
                "mismatched intent between class '%s' and its member '%s'"_warn_en_US,
                baseSymbol.name(), component->GetLastSymbol().name());
          }
        }
      }
    }
  }
}

} // namespace Fortran::tidy::bugprone
