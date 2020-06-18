//===-- lib/Evaluate/variable.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/variable.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/symbol.h"
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// Constructors, accessors, mutators

Triplet::Triplet() : stride_{Expr<SubscriptInteger>{1}} {}

Triplet::Triplet(std::optional<Expr<SubscriptInteger>> &&l,
    std::optional<Expr<SubscriptInteger>> &&u,
    std::optional<Expr<SubscriptInteger>> &&s)
    : stride_{s ? std::move(*s) : Expr<SubscriptInteger>{1}} {
  if (l) {
    lower_.emplace(std::move(*l));
  }
  if (u) {
    upper_.emplace(std::move(*u));
  }
}

std::optional<Expr<SubscriptInteger>> Triplet::lower() const {
  if (lower_) {
    return {lower_.value().value()};
  }
  return std::nullopt;
}

Triplet &Triplet::set_lower(Expr<SubscriptInteger> &&expr) {
  lower_.emplace(std::move(expr));
  return *this;
}

std::optional<Expr<SubscriptInteger>> Triplet::upper() const {
  if (upper_) {
    return {upper_.value().value()};
  }
  return std::nullopt;
}

Triplet &Triplet::set_upper(Expr<SubscriptInteger> &&expr) {
  upper_.emplace(std::move(expr));
  return *this;
}

Expr<SubscriptInteger> Triplet::stride() const { return stride_.value(); }

Triplet &Triplet::set_stride(Expr<SubscriptInteger> &&expr) {
  stride_.value() = std::move(expr);
  return *this;
}

bool Triplet::IsStrideOne() const {
  if (auto stride{ToInt64(stride_.value())}) {
    return stride == 1;
  } else {
    return false;
  }
}

CoarrayRef::CoarrayRef(SymbolVector &&base, std::vector<Subscript> &&ss,
    std::vector<Expr<SubscriptInteger>> &&css)
    : base_{std::move(base)}, subscript_(std::move(ss)),
      cosubscript_(std::move(css)) {
  CHECK(!base_.empty());
  CHECK(!cosubscript_.empty());
}

std::optional<Expr<SomeInteger>> CoarrayRef::stat() const {
  if (stat_) {
    return stat_.value().value();
  } else {
    return std::nullopt;
  }
}

std::optional<Expr<SomeInteger>> CoarrayRef::team() const {
  if (team_) {
    return team_.value().value();
  } else {
    return std::nullopt;
  }
}

CoarrayRef &CoarrayRef::set_stat(Expr<SomeInteger> &&v) {
  CHECK(IsVariable(v));
  stat_.emplace(std::move(v));
  return *this;
}

CoarrayRef &CoarrayRef::set_team(Expr<SomeInteger> &&v, bool isTeamNumber) {
  CHECK(IsVariable(v));
  team_.emplace(std::move(v));
  teamIsTeamNumber_ = isTeamNumber;
  return *this;
}

const Symbol &CoarrayRef::GetFirstSymbol() const { return base_.front(); }

const Symbol &CoarrayRef::GetLastSymbol() const { return base_.back(); }

void Substring::SetBounds(std::optional<Expr<SubscriptInteger>> &lower,
    std::optional<Expr<SubscriptInteger>> &upper) {
  if (lower) {
    set_lower(std::move(lower.value()));
  }
  if (upper) {
    set_upper(std::move(upper.value()));
  }
}

Expr<SubscriptInteger> Substring::lower() const {
  if (lower_) {
    return lower_.value().value();
  } else {
    return AsExpr(Constant<SubscriptInteger>{1});
  }
}

Substring &Substring::set_lower(Expr<SubscriptInteger> &&expr) {
  lower_.emplace(std::move(expr));
  return *this;
}

std::optional<Expr<SubscriptInteger>> Substring::upper() const {
  if (upper_) {
    return upper_.value().value();
  } else {
    return std::visit(
        common::visitors{
            [](const DataRef &dataRef) { return dataRef.LEN(); },
            [](const StaticDataObject::Pointer &object)
                -> std::optional<Expr<SubscriptInteger>> {
              return AsExpr(Constant<SubscriptInteger>{object->data().size()});
            },
        },
        parent_);
  }
}

Substring &Substring::set_upper(Expr<SubscriptInteger> &&expr) {
  upper_.emplace(std::move(expr));
  return *this;
}

std::optional<Expr<SomeCharacter>> Substring::Fold(FoldingContext &context) {
  if (!lower_) {
    lower_ = AsExpr(Constant<SubscriptInteger>{1});
  }
  lower_.value() = evaluate::Fold(context, std::move(lower_.value().value()));
  std::optional<ConstantSubscript> lbi{ToInt64(lower_.value().value())};
  if (lbi && *lbi < 1) {
    context.messages().Say(
        "Lower bound (%jd) on substring is less than one"_en_US, *lbi);
    *lbi = 1;
    lower_ = AsExpr(Constant<SubscriptInteger>{1});
  }
  if (!upper_) {
    upper_ = upper();
    if (!upper_) {
      return std::nullopt;
    }
  }
  upper_.value() = evaluate::Fold(context, std::move(upper_.value().value()));
  if (std::optional<ConstantSubscript> ubi{ToInt64(upper_.value().value())}) {
    auto *literal{std::get_if<StaticDataObject::Pointer>(&parent_)};
    std::optional<ConstantSubscript> length;
    if (literal) {
      length = (*literal)->data().size();
    } else if (const Symbol * symbol{GetLastSymbol()}) {
      if (const semantics::DeclTypeSpec * type{symbol->GetType()}) {
        if (type->category() == semantics::DeclTypeSpec::Character) {
          length = ToInt64(type->characterTypeSpec().length().GetExplicit());
        }
      }
    }
    if (*ubi < 1 || (lbi && *ubi < *lbi)) {
      // Zero-length string: canonicalize
      *lbi = 1, *ubi = 0;
      lower_ = AsExpr(Constant<SubscriptInteger>{*lbi});
      upper_ = AsExpr(Constant<SubscriptInteger>{*ubi});
    } else if (length && *ubi > *length) {
      context.messages().Say("Upper bound (%jd) on substring is greater "
                             "than character length (%jd)"_en_US,
          *ubi, *length);
      *ubi = *length;
    }
    if (lbi && literal) {
      CHECK(*ubi >= *lbi);
      auto newStaticData{StaticDataObject::Create()};
      auto items{*ubi - *lbi + 1};
      auto width{(*literal)->itemBytes()};
      auto bytes{items * width};
      auto startByte{(*lbi - 1) * width};
      const auto *from{&(*literal)->data()[0] + startByte};
      for (auto j{0}; j < bytes; ++j) {
        newStaticData->data().push_back(from[j]);
      }
      parent_ = newStaticData;
      lower_ = AsExpr(Constant<SubscriptInteger>{1});
      ConstantSubscript length = newStaticData->data().size();
      upper_ = AsExpr(Constant<SubscriptInteger>{length});
      switch (width) {
      case 1:
        return {
            AsCategoryExpr(AsExpr(Constant<Type<TypeCategory::Character, 1>>{
                *newStaticData->AsString()}))};
      case 2:
        return {AsCategoryExpr(Constant<Type<TypeCategory::Character, 2>>{
            *newStaticData->AsU16String()})};
      case 4:
        return {AsCategoryExpr(Constant<Type<TypeCategory::Character, 4>>{
            *newStaticData->AsU32String()})};
      default:
        CRASH_NO_CASE;
      }
    }
  }
  return std::nullopt;
}

DescriptorInquiry::DescriptorInquiry(
    const NamedEntity &base, Field field, int dim)
    : base_{base}, field_{field}, dimension_{dim} {
  const Symbol &last{base_.GetLastSymbol()};
  CHECK(IsDescriptor(last));
  CHECK((field == Field::Len && dim == 0) ||
      (field != Field::Len && dim >= 0 && dim < last.Rank()));
}

DescriptorInquiry::DescriptorInquiry(NamedEntity &&base, Field field, int dim)
    : base_{std::move(base)}, field_{field}, dimension_{dim} {
  const Symbol &last{base_.GetLastSymbol()};
  CHECK(IsDescriptor(last));
  CHECK((field == Field::Len && dim == 0) ||
      (field != Field::Len && dim >= 0 && dim < last.Rank()));
}

// LEN()
static std::optional<Expr<SubscriptInteger>> SymbolLEN(const Symbol &sym) {
  if (auto dyType{DynamicType::From(sym)}) {
    if (const semantics::ParamValue * len{dyType->charLength()}) {
      if (len->isExplicit()) {
        if (auto intExpr{len->GetExplicit()}) {
          return ConvertToType<SubscriptInteger>(*std::move(intExpr));
        } else {
          // There was an error constructing this symbol's type.  It should
          // have a length expression, but we couldn't retrieve it
          return std::nullopt;
        }
      } else {
        return Expr<SubscriptInteger>{
            DescriptorInquiry{NamedEntity{sym}, DescriptorInquiry::Field::Len}};
      }
    }
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> BaseObject::LEN() const {
  return std::visit(
      common::visitors{
          [](const Symbol &symbol) { return SymbolLEN(symbol); },
          [](const StaticDataObject::Pointer &object)
              -> std::optional<Expr<SubscriptInteger>> {
            return AsExpr(Constant<SubscriptInteger>{object->data().size()});
          },
      },
      u);
}

std::optional<Expr<SubscriptInteger>> Component::LEN() const {
  return SymbolLEN(GetLastSymbol());
}

std::optional<Expr<SubscriptInteger>> NamedEntity::LEN() const {
  return SymbolLEN(GetLastSymbol());
}

std::optional<Expr<SubscriptInteger>> ArrayRef::LEN() const {
  return base_.LEN();
}

std::optional<Expr<SubscriptInteger>> CoarrayRef::LEN() const {
  return SymbolLEN(GetLastSymbol());
}

std::optional<Expr<SubscriptInteger>> DataRef::LEN() const {
  return std::visit(common::visitors{
                        [](SymbolRef symbol) { return SymbolLEN(symbol); },
                        [](const auto &x) { return x.LEN(); },
                    },
      u);
}

std::optional<Expr<SubscriptInteger>> Substring::LEN() const {
  if (auto top{upper()}) {
    return AsExpr(Extremum<SubscriptInteger>{Ordering::Greater,
        AsExpr(Constant<SubscriptInteger>{0}),
        *std::move(top) - lower() + AsExpr(Constant<SubscriptInteger>{1})});
  } else {
    return std::nullopt;
  }
}

template <typename T>
std::optional<Expr<SubscriptInteger>> Designator<T>::LEN() const {
  if constexpr (T::category == TypeCategory::Character) {
    return std::visit(common::visitors{
                          [](SymbolRef symbol) { return SymbolLEN(symbol); },
                          [](const auto &x) { return x.LEN(); },
                      },
        u);
  } else {
    common::die("Designator<non-char>::LEN() called");
    return std::nullopt;
  }
}

std::optional<Expr<SubscriptInteger>> ProcedureDesignator::LEN() const {
  using T = std::optional<Expr<SubscriptInteger>>;
  return std::visit(
      common::visitors{
          [](SymbolRef symbol) -> T { return SymbolLEN(symbol); },
          [](const common::CopyableIndirection<Component> &c) -> T {
            return c.value().LEN();
          },
          [](const SpecificIntrinsic &i) -> T {
            if (i.name == "char") {
              return Expr<SubscriptInteger>{1};
            }
            // Some other cases whose results' lengths can be determined
            // from the lengths of their arguments are handled in
            // ProcedureRef::LEN().
            return std::nullopt;
          },
      },
      u);
}

// Rank()
int BaseObject::Rank() const {
  return std::visit(common::visitors{
                        [](SymbolRef symbol) { return symbol->Rank(); },
                        [](const StaticDataObject::Pointer &) { return 0; },
                    },
      u);
}

int Component::Rank() const {
  if (int rank{symbol_->Rank()}; rank > 0) {
    return rank;
  }
  return base().Rank();
}

int NamedEntity::Rank() const {
  return std::visit(common::visitors{
                        [](const SymbolRef s) { return s->Rank(); },
                        [](const Component &c) { return c.Rank(); },
                    },
      u_);
}

int Subscript::Rank() const {
  return std::visit(common::visitors{
                        [](const IndirectSubscriptIntegerExpr &x) {
                          return x.value().Rank();
                        },
                        [](const Triplet &) { return 1; },
                    },
      u);
}

int ArrayRef::Rank() const {
  int rank{0};
  for (const auto &expr : subscript_) {
    rank += expr.Rank();
  }
  if (rank > 0) {
    return rank;
  } else if (const Component * component{base_.UnwrapComponent()}) {
    return component->base().Rank();
  } else {
    return 0;
  }
}

int CoarrayRef::Rank() const {
  if (!subscript_.empty()) {
    int rank{0};
    for (const auto &expr : subscript_) {
      rank += expr.Rank();
    }
    return rank;
  } else {
    return base_.back()->Rank();
  }
}

int DataRef::Rank() const {
  return std::visit(common::visitors{
                        [](SymbolRef symbol) { return symbol->Rank(); },
                        [](const auto &x) { return x.Rank(); },
                    },
      u);
}

int Substring::Rank() const {
  return std::visit(common::visitors{
                        [](const DataRef &dataRef) { return dataRef.Rank(); },
                        [](const StaticDataObject::Pointer &) { return 0; },
                    },
      parent_);
}

int ComplexPart::Rank() const { return complex_.Rank(); }

template <typename T> int Designator<T>::Rank() const {
  return std::visit(common::visitors{
                        [](SymbolRef symbol) { return symbol->Rank(); },
                        [](const auto &x) { return x.Rank(); },
                    },
      u);
}

// GetBaseObject(), GetFirstSymbol(), GetLastSymbol(), &c.
const Symbol &Component::GetFirstSymbol() const {
  return base_.value().GetFirstSymbol();
}

const Symbol &NamedEntity::GetFirstSymbol() const {
  return std::visit(common::visitors{
                        [](SymbolRef s) -> const Symbol & { return s; },
                        [](const Component &c) -> const Symbol & {
                          return c.GetFirstSymbol();
                        },
                    },
      u_);
}

const Symbol &NamedEntity::GetLastSymbol() const {
  return std::visit(common::visitors{
                        [](SymbolRef s) -> const Symbol & { return s; },
                        [](const Component &c) -> const Symbol & {
                          return c.GetLastSymbol();
                        },
                    },
      u_);
}

const Component *NamedEntity::UnwrapComponent() const {
  return std::visit(common::visitors{
                        [](SymbolRef) -> const Component * { return nullptr; },
                        [](const Component &c) { return &c; },
                    },
      u_);
}

Component *NamedEntity::UnwrapComponent() {
  return std::visit(common::visitors{
                        [](SymbolRef &) -> Component * { return nullptr; },
                        [](Component &c) { return &c; },
                    },
      u_);
}

const Symbol &ArrayRef::GetFirstSymbol() const {
  return base_.GetFirstSymbol();
}

const Symbol &ArrayRef::GetLastSymbol() const { return base_.GetLastSymbol(); }

const Symbol &DataRef::GetFirstSymbol() const {
  return *std::visit(common::visitors{
                         [](SymbolRef symbol) { return &*symbol; },
                         [](const auto &x) { return &x.GetFirstSymbol(); },
                     },
      u);
}

const Symbol &DataRef::GetLastSymbol() const {
  return *std::visit(common::visitors{
                         [](SymbolRef symbol) { return &*symbol; },
                         [](const auto &x) { return &x.GetLastSymbol(); },
                     },
      u);
}

BaseObject Substring::GetBaseObject() const {
  return std::visit(common::visitors{
                        [](const DataRef &dataRef) {
                          return BaseObject{dataRef.GetFirstSymbol()};
                        },
                        [](StaticDataObject::Pointer pointer) {
                          return BaseObject{std::move(pointer)};
                        },
                    },
      parent_);
}

const Symbol *Substring::GetLastSymbol() const {
  return std::visit(
      common::visitors{
          [](const DataRef &dataRef) { return &dataRef.GetLastSymbol(); },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      parent_);
}

template <typename T> BaseObject Designator<T>::GetBaseObject() const {
  return std::visit(
      common::visitors{
          [](SymbolRef symbol) { return BaseObject{symbol}; },
          [](const Substring &sstring) { return sstring.GetBaseObject(); },
          [](const auto &x) {
#if !__clang__ && __GNUC__ == 7 && __GNUC_MINOR__ == 2
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                              Substring>) {
              return x.GetBaseObject();
            } else
#endif
              return BaseObject{x.GetFirstSymbol()};
          },
      },
      u);
}

template <typename T> const Symbol *Designator<T>::GetLastSymbol() const {
  return std::visit(
      common::visitors{
          [](SymbolRef symbol) { return &*symbol; },
          [](const Substring &sstring) { return sstring.GetLastSymbol(); },
          [](const auto &x) {
#if !__clang__ && __GNUC__ == 7 && __GNUC_MINOR__ == 2
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                              Substring>) {
              return x.GetLastSymbol();
            } else
#endif
              return &x.GetLastSymbol();
          },
      },
      u);
}

template <typename T>
std::optional<DynamicType> Designator<T>::GetType() const {
  if constexpr (IsLengthlessIntrinsicType<Result>) {
    return {Result::GetType()};
  } else {
    return DynamicType::From(GetLastSymbol());
  }
}

static NamedEntity AsNamedEntity(const SymbolVector &x) {
  CHECK(!x.empty());
  NamedEntity result{x.front()};
  int j{0};
  for (const Symbol &symbol : x) {
    if (j++ != 0) {
      DataRef base{result.IsSymbol() ? DataRef{result.GetLastSymbol()}
                                     : DataRef{result.GetComponent()}};
      result = NamedEntity{Component{std::move(base), symbol}};
    }
  }
  return result;
}

NamedEntity CoarrayRef::GetBase() const { return AsNamedEntity(base_); }

// Equality testing

// For the purposes of comparing type parameter expressions while
// testing the compatibility of procedure characteristics, two
// object dummy arguments with the same name are considered equal.
static bool AreSameSymbol(const Symbol &x, const Symbol &y) {
  if (&x == &y) {
    return true;
  }
  if (x.name() == y.name()) {
    if (const auto *xObject{x.detailsIf<semantics::ObjectEntityDetails>()}) {
      if (const auto *yObject{y.detailsIf<semantics::ObjectEntityDetails>()}) {
        return xObject->isDummy() && yObject->isDummy();
      }
    }
  }
  return false;
}

// Implements operator==() for a union type, using special case handling
// for Symbol references.
template <typename A> static bool TestVariableEquality(const A &x, const A &y) {
  const SymbolRef *xSymbol{std::get_if<SymbolRef>(&x.u)};
  if (const SymbolRef * ySymbol{std::get_if<SymbolRef>(&y.u)}) {
    return xSymbol && AreSameSymbol(*xSymbol, *ySymbol);
  } else {
    return x.u == y.u;
  }
}

bool BaseObject::operator==(const BaseObject &that) const {
  return TestVariableEquality(*this, that);
}
bool Component::operator==(const Component &that) const {
  return base_ == that.base_ && &*symbol_ == &*that.symbol_;
}
bool NamedEntity::operator==(const NamedEntity &that) const {
  if (IsSymbol()) {
    return that.IsSymbol() &&
        AreSameSymbol(GetFirstSymbol(), that.GetFirstSymbol());
  } else {
    return !that.IsSymbol() && GetComponent() == that.GetComponent();
  }
}
template <int KIND>
bool TypeParamInquiry<KIND>::operator==(
    const TypeParamInquiry<KIND> &that) const {
  return &*parameter_ == &*that.parameter_ && base_ == that.base_;
}
bool Triplet::operator==(const Triplet &that) const {
  return lower_ == that.lower_ && upper_ == that.upper_ &&
      stride_ == that.stride_;
}
bool Subscript::operator==(const Subscript &that) const { return u == that.u; }
bool ArrayRef::operator==(const ArrayRef &that) const {
  return base_ == that.base_ && subscript_ == that.subscript_;
}
bool CoarrayRef::operator==(const CoarrayRef &that) const {
  return base_ == that.base_ && subscript_ == that.subscript_ &&
      cosubscript_ == that.cosubscript_ && stat_ == that.stat_ &&
      team_ == that.team_ && teamIsTeamNumber_ == that.teamIsTeamNumber_;
}
bool DataRef::operator==(const DataRef &that) const {
  return TestVariableEquality(*this, that);
}
bool Substring::operator==(const Substring &that) const {
  return parent_ == that.parent_ && lower_ == that.lower_ &&
      upper_ == that.upper_;
}
bool ComplexPart::operator==(const ComplexPart &that) const {
  return part_ == that.part_ && complex_ == that.complex_;
}
bool ProcedureRef::operator==(const ProcedureRef &that) const {
  return proc_ == that.proc_ && arguments_ == that.arguments_;
}
template <typename T>
bool Designator<T>::operator==(const Designator<T> &that) const {
  return TestVariableEquality(*this, that);
}
bool DescriptorInquiry::operator==(const DescriptorInquiry &that) const {
  return field_ == that.field_ && base_ == that.base_ &&
      dimension_ == that.dimension_;
}

INSTANTIATE_VARIABLE_TEMPLATES
} // namespace Fortran::evaluate

template class Fortran::common::Indirection<Fortran::evaluate::Component, true>;
