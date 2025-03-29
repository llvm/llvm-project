//===-- lib/Semantics/check-select-type.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-select-type.h"
#include "flang/Common/idioms.h"
#include "flang/Common/reference.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include <optional>

namespace Fortran::semantics {

class TypeCaseValues {
public:
  TypeCaseValues(SemanticsContext &c, const evaluate::DynamicType &t)
      : context_{c}, selectorType_{t} {}
  void Check(const std::list<parser::SelectTypeConstruct::TypeCase> &cases) {
    for (const auto &c : cases) {
      AddTypeCase(c);
    }
    if (!hasErrors_) {
      ReportConflictingTypeCases();
    }
  }

private:
  void AddTypeCase(const parser::SelectTypeConstruct::TypeCase &c) {
    const auto &stmt{std::get<parser::Statement<parser::TypeGuardStmt>>(c.t)};
    const parser::TypeGuardStmt &typeGuardStmt{stmt.statement};
    const auto &guard{std::get<parser::TypeGuardStmt::Guard>(typeGuardStmt.t)};
    if (std::holds_alternative<parser::Default>(guard.u)) {
      typeCases_.emplace_back(stmt, std::nullopt);
    } else if (std::optional<evaluate::DynamicType> type{GetGuardType(guard)}) {
      if (PassesChecksOnGuard(stmt, *type)) {
        typeCases_.emplace_back(stmt, *type);
      } else {
        hasErrors_ = true;
      }
    } else {
      hasErrors_ = true;
    }
  }

  std::optional<evaluate::DynamicType> GetGuardType(
      const parser::TypeGuardStmt::Guard &guard) {
    return common::visit(
        common::visitors{
            [](const parser::Default &)
                -> std::optional<evaluate::DynamicType> {
              return std::nullopt;
            },
            [](const parser::TypeSpec &typeSpec) {
              return evaluate::DynamicType::From(typeSpec.declTypeSpec);
            },
            [](const parser::DerivedTypeSpec &spec)
                -> std::optional<evaluate::DynamicType> {
              if (const auto *derivedTypeSpec{spec.derivedTypeSpec}) {
                return evaluate::DynamicType(*derivedTypeSpec);
              }
              return std::nullopt;
            },
        },
        guard.u);
  }

  bool PassesChecksOnGuard(const parser::Statement<parser::TypeGuardStmt> &stmt,
      const evaluate::DynamicType &guardDynamicType) {
    const parser::TypeGuardStmt &typeGuardStmt{stmt.statement};
    const auto &guard{std::get<parser::TypeGuardStmt::Guard>(typeGuardStmt.t)};
    return common::visit(
        common::visitors{
            [](const parser::Default &) { return true; },
            [&](const parser::TypeSpec &typeSpec) {
              const DeclTypeSpec *spec{typeSpec.declTypeSpec};
              CHECK(spec);
              CHECK(spec->AsIntrinsic() || spec->AsDerived());
              bool typeSpecRetVal{false};
              if (spec->AsIntrinsic()) {
                typeSpecRetVal = true;
                if (!selectorType_.IsUnlimitedPolymorphic()) { // C1162
                  context_.Say(stmt.source,
                      "If selector is not unlimited polymorphic, "
                      "an intrinsic type specification must not be specified "
                      "in the type guard statement"_err_en_US);
                  typeSpecRetVal = false;
                }
                if (spec->category() == DeclTypeSpec::Character &&
                    !guardDynamicType.IsAssumedLengthCharacter()) { // C1160
                  auto location{parser::FindSourceLocation(typeSpec)};
                  context_.Say(location.empty() ? stmt.source : location,
                      "The type specification statement must have "
                      "LEN type parameter as assumed"_err_en_US);
                  typeSpecRetVal = false;
                }
              } else {
                const DerivedTypeSpec *derived{spec->AsDerived()};
                typeSpecRetVal = PassesDerivedTypeChecks(
                    *derived, parser::FindSourceLocation(typeSpec));
              }
              return typeSpecRetVal;
            },
            [&](const parser::DerivedTypeSpec &x) {
              CHECK(x.derivedTypeSpec);
              const semantics::DerivedTypeSpec *derived{x.derivedTypeSpec};
              return PassesDerivedTypeChecks(
                  *derived, parser::FindSourceLocation(x));
            },
        },
        guard.u);
  }

  bool PassesDerivedTypeChecks(const semantics::DerivedTypeSpec &derived,
      parser::CharBlock sourceLoc) const {
    for (const auto &pair : derived.parameters()) {
      if (pair.second.isLen() && !pair.second.isAssumed()) { // F'2023 C1165
        context_.Say(sourceLoc,
            "The type specification statement must have LEN type parameter as assumed"_err_en_US);
        return false;
      }
    }
    if (!IsExtensibleType(&derived)) { // F'2023 C1166
      context_.Say(sourceLoc,
          "The type specification statement must not specify a type with a SEQUENCE attribute or a BIND attribute"_err_en_US);
      return false;
    }
    if (!selectorType_.IsUnlimitedPolymorphic()) { // F'2023 C1167
      if (const auto *selDerivedTypeSpec{
              evaluate::GetDerivedTypeSpec(selectorType_)}) {
        if (!derived.MatchesOrExtends(*selDerivedTypeSpec)) {
          context_.Say(sourceLoc,
              "Type specification '%s' must be an extension of TYPE '%s'"_err_en_US,
              derived.AsFortran(), selDerivedTypeSpec->AsFortran());
          return false;
        }
      }
    }
    return true;
  }

  struct TypeCase {
    explicit TypeCase(const parser::Statement<parser::TypeGuardStmt> &s,
        std::optional<evaluate::DynamicType> guardTypeDynamic)
        : stmt{s} {
      SetGuardType(guardTypeDynamic);
    }

    void SetGuardType(std::optional<evaluate::DynamicType> guardTypeDynamic) {
      const auto &guard{GetGuardFromStmt(stmt)};
      common::visit(common::visitors{
                        [&](const parser::Default &) {},
                        [&](const auto &) { guardType_ = *guardTypeDynamic; },
                    },
          guard.u);
    }

    bool IsDefault() const {
      const auto &guard{GetGuardFromStmt(stmt)};
      return std::holds_alternative<parser::Default>(guard.u);
    }

    bool IsTypeSpec() const {
      const auto &guard{GetGuardFromStmt(stmt)};
      return std::holds_alternative<parser::TypeSpec>(guard.u);
    }

    bool IsDerivedTypeSpec() const {
      const auto &guard{GetGuardFromStmt(stmt)};
      return std::holds_alternative<parser::DerivedTypeSpec>(guard.u);
    }

    const parser::TypeGuardStmt::Guard &GetGuardFromStmt(
        const parser::Statement<parser::TypeGuardStmt> &stmt) const {
      const parser::TypeGuardStmt &typeGuardStmt{stmt.statement};
      return std::get<parser::TypeGuardStmt::Guard>(typeGuardStmt.t);
    }

    std::optional<evaluate::DynamicType> guardType() const {
      return guardType_;
    }

    std::string AsFortran() const {
      std::string result;
      if (this->guardType()) {
        auto type{*this->guardType()};
        result += type.AsFortran();
      } else {
        result += "DEFAULT";
      }
      return result;
    }
    const parser::Statement<parser::TypeGuardStmt> &stmt;
    std::optional<evaluate::DynamicType> guardType_; // is this POD?
  };

  // Returns true if and only if the values are different
  // Does apple to apple comparision, in case of TypeSpec or DerivedTypeSpec
  // checks for kinds as well.
  static bool TypesAreDifferent(const TypeCase &x, const TypeCase &y) {
    if (x.IsDefault()) { // C1164
      return !y.IsDefault();
    } else if (x.IsTypeSpec() && y.IsTypeSpec()) { // C1163
      return !AreTypeKindCompatible(x, y);
    } else if (x.IsDerivedTypeSpec() && y.IsDerivedTypeSpec()) { // C1163
      return !AreTypeKindCompatible(x, y);
    }
    return true;
  }

  static bool AreTypeKindCompatible(const TypeCase &x, const TypeCase &y) {
    return (*x.guardType()).IsTkCompatibleWith((*y.guardType()));
  }

  void ReportConflictingTypeCases() {
    for (auto iter{typeCases_.begin()}; iter != typeCases_.end(); ++iter) {
      parser::Message *msg{nullptr};
      for (auto p{typeCases_.begin()}; p != typeCases_.end(); ++p) {
        if (p->stmt.source.begin() < iter->stmt.source.begin() &&
            !TypesAreDifferent(*p, *iter)) {
          if (!msg) {
            msg = &context_.Say(iter->stmt.source,
                "Type specification '%s' conflicts with "
                "previous type specification"_err_en_US,
                iter->AsFortran());
          }
          msg->Attach(p->stmt.source,
              "Conflicting type specification '%s'"_en_US, p->AsFortran());
        }
      }
    }
  }

  SemanticsContext &context_;
  const evaluate::DynamicType &selectorType_;
  std::list<TypeCase> typeCases_;
  bool hasErrors_{false};
};

void SelectTypeChecker::Enter(const parser::SelectTypeConstruct &construct) {
  const auto &selectTypeStmt{
      std::get<parser::Statement<parser::SelectTypeStmt>>(construct.t)};
  const auto &selectType{selectTypeStmt.statement};
  const auto &unResolvedSel{std::get<parser::Selector>(selectType.t)};
  if (const auto *selector{GetExprFromSelector(unResolvedSel)}) {
    if (IsProcedure(*selector)) {
      context_.Say(
          selectTypeStmt.source, "Selector may not be a procedure"_err_en_US);
    } else if (evaluate::IsAssumedRank(*selector)) {
      context_.Say(selectTypeStmt.source,
          "Assumed-rank variable may only be used as actual argument"_err_en_US);
    } else if (auto exprType{selector->GetType()}) {
      const auto &typeCaseList{
          std::get<std::list<parser::SelectTypeConstruct::TypeCase>>(
              construct.t)};
      TypeCaseValues{context_, *exprType}.Check(typeCaseList);
    }
  }
}

const SomeExpr *SelectTypeChecker::GetExprFromSelector(
    const parser::Selector &selector) {
  return common::visit([](const auto &x) { return GetExpr(x); }, selector.u);
}
} // namespace Fortran::semantics
