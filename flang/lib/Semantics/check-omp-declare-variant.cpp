//===-- lib/Semantics/check-omp-declare-variant.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structure checks for DECLARE VARIANT.
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"

#include "flang/Common/idioms.h"
#include "flang/Common/visit.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

#include "llvm/Frontend/OpenMP/OMP.h"

namespace Fortran::semantics {

using namespace Fortran::semantics::omp;

static const parser::traits::OmpContextSelectorSpecification *
getMatchClauseContextSelector(const parser::OmpDirectiveSpecification &spec) {
  for (const parser::OmpClause &clause : spec.Clauses().v) {
    if (clause.Id() == llvm::omp::Clause::OMPC_match)
      return &std::get<parser::OmpClause::Match>(clause.u).v.v;
  }
  return nullptr;
}

void OmpStructureChecker::CheckDeclareVariantUserConditions(
    const parser::OmpContextSelector &ctx) {
  using SetName = parser::OmpTraitSetSelectorName;
  using TraitName = parser::OmpTraitSelectorName;

  for (const parser::OmpTraitSetSelector &traitSet : ctx.v) {
    if (std::get<SetName>(traitSet.t).v != SetName::Value::User) {
      continue;
    }
    for (const parser::OmpTraitSelector &trait :
        std::get<std::list<parser::OmpTraitSelector>>(traitSet.t)) {
      const auto &traitName{std::get<TraitName>(trait.t)};
      if (!std::holds_alternative<TraitName::Value>(traitName.u) ||
          std::get<TraitName::Value>(traitName.u) !=
              TraitName::Value::Condition) {
        continue;
      }
      const auto &maybeProps{
          std::get<std::optional<parser::OmpTraitSelector::Properties>>(
              trait.t)};
      if (!maybeProps) {
        continue;
      }
      const auto &properties{
          std::get<std::list<parser::OmpTraitProperty>>(maybeProps->t)};
      if (properties.size() != 1) {
        continue;
      }
      const parser::OmpTraitProperty &property{properties.front()};
      const parser::ScalarExpr &scalarExpr{
          std::get<parser::ScalarExpr>(property.u)};
      auto maybeType{GetDynamicType(scalarExpr.thing.value())};
      if (!maybeType || maybeType->category() != TypeCategory::Logical) {
        continue;
      }
      if (const auto *expr{GetExpr(scalarExpr)}) {
        if (!IsConstantExpr(*expr, &context_.foldingContext())) {
          context_.Say(property.source,
              "USER condition in the MATCH clause must be a constant expression"_err_en_US);
        }
      }
    }
  }
}

void OmpStructureChecker::CheckOmpDeclareVariantDirective(
    const parser::OmpDeclareVariantDirective &x) {
  const parser::OmpDirectiveSpecification &spec{x.v};
  const parser::OmpArgumentList &args{spec.Arguments()};

  if (args.v.size() != 1) {
    context_.Say(args.source,
        "DECLARE_VARIANT directive should have a single argument"_err_en_US);
    return;
  }

  auto InvalidArgument{[&](parser::CharBlock source) {
    context_.Say(source,
        "The argument to the DECLARE_VARIANT directive should be [base-name:]variant-name"_err_en_US);
  }};

  auto CheckProcedureSymbol{[&](const Symbol *sym, parser::CharBlock source) {
    if (sym) {
      if (!IsProcedure(*sym) && !IsFunction(*sym)) {
        auto &msg{context_.Say(source,
            "The name '%s' should refer to a procedure"_err_en_US,
            sym->name())};
        if (sym->test(Symbol::Flag::Implicit)) {
          msg.Attach(source, "The name '%s' has been implicitly declared"_en_US,
              sym->name());
        }
      }
    } else {
      InvalidArgument(source);
    }
  }};

  const Symbol *base{nullptr};
  const Symbol *variant{nullptr};
  const parser::OmpArgument &arg{args.v.front()};
  common::visit( //
      common::visitors{
          [&](const parser::OmpBaseVariantNames &y) {
            base = GetObjectSymbol(std::get<0>(y.t));
            variant = GetObjectSymbol(std::get<1>(y.t));
            CheckProcedureSymbol(base, arg.source);
            CheckProcedureSymbol(variant, arg.source);
          },
          [&](const parser::OmpLocator &y) {
            variant = GetArgumentSymbol(arg);
            CheckProcedureSymbol(variant, arg.source);
            // OpenMP 5.1 [2.3.5, declare variant directive, Restrictions]:
            // "If base-proc-name is omitted then the declare variant directive
            // must appear in an interface block or the specification part of a
            // procedure." The same section requires the directive to appear in
            // the specification part of the subprogram or interface body to
            // which it applies. Infer the base procedure from that program
            // unit.
            const Scope &containingScope{context_.FindScope(x.source)};
            if (const Symbol *host{
                    GetProgramUnitContaining(containingScope).symbol()}) {
              base = host;
            }
          },
          [&](auto &&y) { InvalidArgument(arg.source); },
      },
      arg.u);

  if (base && variant) {
    base = &base->GetUltimate();
    variant = &variant->GetUltimate();
    if (base == variant) {
      context_.Say(arg.source,
          "The variant procedure must differ from the base procedure"_err_en_US);
    } else if (!declareVariantPairs_.emplace(base, variant).second) {
      context_.Say(arg.source,
          "Variant '%s' was already specified for '%s' in another DECLARE VARIANT directive"_err_en_US,
          variant->name(), base->name());
    }
  }

  const parser::traits::OmpContextSelectorSpecification *matchSelector{
      getMatchClauseContextSelector(spec)};
  if (!matchSelector) {
    context_.Say(x.source,
        "DECLARE_VARIANT directive requires a MATCH clause"_err_en_US);
    return;
  }

  EnterDirectiveNest(ContextSelectorNest);
  CheckContextSelectorSpecification(*matchSelector);
  CheckDeclareVariantUserConditions(*matchSelector);
  ExitDirectiveNest(ContextSelectorNest);
}

void OmpStructureChecker::Enter(const parser::OmpDeclareVariantDirective &x) {
  const parser::OmpDirectiveName &dirName{x.v.DirName()};
  PushContextAndClauseSets(dirName.source, dirName.v);
  CheckOmpDeclareVariantDirective(x);
}

void OmpStructureChecker::Leave(const parser::OmpDeclareVariantDirective &) {
  dirContext_.pop_back();
}

} // namespace Fortran::semantics
