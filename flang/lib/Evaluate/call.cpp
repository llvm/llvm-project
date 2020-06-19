//===-- lib/Evaluate/call.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/call.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/symbol.h"

namespace Fortran::evaluate {

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ActualArgument)
ActualArgument::ActualArgument(Expr<SomeType> &&x) : u_{std::move(x)} {}
ActualArgument::ActualArgument(common::CopyableIndirection<Expr<SomeType>> &&v)
    : u_{std::move(v)} {}
ActualArgument::ActualArgument(AssumedType x) : u_{x} {}
ActualArgument::~ActualArgument() {}

ActualArgument::AssumedType::AssumedType(const Symbol &symbol)
    : symbol_{symbol} {
  const semantics::DeclTypeSpec *type{symbol.GetType()};
  CHECK(type && type->category() == semantics::DeclTypeSpec::TypeStar);
}

int ActualArgument::AssumedType::Rank() const { return symbol_->Rank(); }

ActualArgument &ActualArgument::operator=(Expr<SomeType> &&expr) {
  u_ = std::move(expr);
  return *this;
}

std::optional<DynamicType> ActualArgument::GetType() const {
  if (const Expr<SomeType> *expr{UnwrapExpr()}) {
    return expr->GetType();
  } else if (std::holds_alternative<AssumedType>(u_)) {
    return DynamicType::AssumedType();
  } else {
    return std::nullopt;
  }
}

int ActualArgument::Rank() const {
  if (const Expr<SomeType> *expr{UnwrapExpr()}) {
    return expr->Rank();
  } else {
    return std::get<AssumedType>(u_).Rank();
  }
}

bool ActualArgument::operator==(const ActualArgument &that) const {
  return keyword_ == that.keyword_ &&
      isAlternateReturn_ == that.isAlternateReturn_ &&
      isPassedObject_ == that.isPassedObject_ && u_ == that.u_;
}

void ActualArgument::Parenthesize() {
  u_ = evaluate::Parenthesize(std::move(DEREF(UnwrapExpr())));
}

SpecificIntrinsic::SpecificIntrinsic(
    IntrinsicProcedure n, characteristics::Procedure &&chars)
    : name{n}, characteristics{
                   new characteristics::Procedure{std::move(chars)}} {}

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(SpecificIntrinsic)

SpecificIntrinsic::~SpecificIntrinsic() {}

bool SpecificIntrinsic::operator==(const SpecificIntrinsic &that) const {
  return name == that.name && characteristics == that.characteristics;
}

ProcedureDesignator::ProcedureDesignator(Component &&c)
    : u{common::CopyableIndirection<Component>::Make(std::move(c))} {}

bool ProcedureDesignator::operator==(const ProcedureDesignator &that) const {
  return u == that.u;
}

std::optional<DynamicType> ProcedureDesignator::GetType() const {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    if (const auto &result{intrinsic->characteristics.value().functionResult}) {
      if (const auto *typeAndShape{result->GetTypeAndShape()}) {
        return typeAndShape->type();
      }
    }
  } else {
    return DynamicType::From(GetSymbol());
  }
  return std::nullopt;
}

int ProcedureDesignator::Rank() const {
  if (const Symbol * symbol{GetSymbol()}) {
    // Subtle: will be zero for functions returning procedure pointers
    return symbol->Rank();
  }
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    if (const auto &result{intrinsic->characteristics.value().functionResult}) {
      if (const auto *typeAndShape{result->GetTypeAndShape()}) {
        CHECK(!typeAndShape->attrs().test(
            characteristics::TypeAndShape::Attr::AssumedRank));
        return typeAndShape->Rank();
      }
      // Otherwise, intrinsic returns a procedure pointer (e.g. NULL(MOLD=pptr))
    }
  }
  return 0;
}

const Symbol *ProcedureDesignator::GetInterfaceSymbol() const {
  if (const Symbol * symbol{GetSymbol()}) {
    if (const auto *details{
            symbol->detailsIf<semantics::ProcEntityDetails>()}) {
      return details->interface().symbol();
    }
  }
  return nullptr;
}

bool ProcedureDesignator::IsElemental() const {
  if (const Symbol * interface{GetInterfaceSymbol()}) {
    return interface->attrs().test(semantics::Attr::ELEMENTAL);
  } else if (const Symbol * symbol{GetSymbol()}) {
    return symbol->attrs().test(semantics::Attr::ELEMENTAL);
  } else if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    return intrinsic->characteristics.value().attrs.test(
        characteristics::Procedure::Attr::Elemental);
  } else {
    DIE("ProcedureDesignator::IsElemental(): no case");
  }
  return false;
}

const SpecificIntrinsic *ProcedureDesignator::GetSpecificIntrinsic() const {
  return std::get_if<SpecificIntrinsic>(&u);
}

const Component *ProcedureDesignator::GetComponent() const {
  if (auto *c{std::get_if<common::CopyableIndirection<Component>>(&u)}) {
    return &c->value();
  } else {
    return nullptr;
  }
}

const Symbol *ProcedureDesignator::GetSymbol() const {
  return std::visit(common::visitors{
                        [](SymbolRef symbol) { return &*symbol; },
                        [](const common::CopyableIndirection<Component> &c) {
                          return &c.value().GetLastSymbol();
                        },
                        [](const auto &) -> const Symbol * { return nullptr; },
                    },
      u);
}

std::string ProcedureDesignator::GetName() const {
  return std::visit(
      common::visitors{
          [](const SpecificIntrinsic &i) { return i.name; },
          [](const Symbol &symbol) { return symbol.name().ToString(); },
          [](const common::CopyableIndirection<Component> &c) {
            return c.value().GetLastSymbol().name().ToString();
          },
      },
      u);
}

std::optional<Expr<SubscriptInteger>> ProcedureRef::LEN() const {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&proc_.u)}) {
    if (intrinsic->name == "repeat") {
      // LEN(REPEAT(ch,n)) == LEN(ch) * n
      CHECK(arguments_.size() == 2);
      const auto *stringArg{
          UnwrapExpr<Expr<SomeCharacter>>(arguments_[0].value())};
      const auto *nCopiesArg{
          UnwrapExpr<Expr<SomeInteger>>(arguments_[1].value())};
      CHECK(stringArg && nCopiesArg);
      if (auto stringLen{stringArg->LEN()}) {
        auto converted{ConvertTo(*stringLen, common::Clone(*nCopiesArg))};
        return *std::move(stringLen) * std::move(converted);
      }
    }
    // Some other cases (e.g., LEN(CHAR(...))) are handled in
    // ProcedureDesignator::LEN() because they're independent of the
    // lengths of the actual arguments.
  }
  return proc_.LEN();
}

int ProcedureRef::Rank() const {
  if (IsElemental()) {
    for (const auto &arg : arguments_) {
      if (arg) {
        if (int rank{arg->Rank()}; rank > 0) {
          return rank;
        }
      }
    }
    return 0;
  } else {
    return proc_.Rank();
  }
}

ProcedureRef::~ProcedureRef() {}

FOR_EACH_SPECIFIC_TYPE(template class FunctionRef, )
} // namespace Fortran::evaluate
DEFINE_DELETER(Fortran::evaluate::ProcedureRef)
