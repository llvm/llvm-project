//===-- lib/Evaluate/initial-image.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/initial-image.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/tools.h"
#include <cstring>

namespace Fortran::evaluate {

auto InitialImage::Add(ConstantSubscript offset, std::size_t bytes,
    const Constant<SomeDerived> &x) -> Result {
  if (offset < 0 || offset + bytes > data_.size()) {
    return OutOfRange;
  } else {
    auto elements{TotalElementCount(x.shape())};
    auto elementBytes{bytes > 0 ? bytes / elements : 0};
    if (elements * elementBytes != bytes) {
      return SizeMismatch;
    } else {
      auto at{x.lbounds()};
      for (auto elements{TotalElementCount(x.shape())}; elements-- > 0;
           x.IncrementSubscripts(at)) {
        auto scalar{x.At(at)};
        // TODO: length type parameter values?
        for (const auto &[symbolRef, indExpr] : scalar) {
          const Symbol &component{*symbolRef};
          if (component.offset() + component.size() > elementBytes) {
            return SizeMismatch;
          } else if (IsPointer(component)) {
            AddPointer(offset + component.offset(), indExpr.value());
          } else {
            Result added{Add(offset + component.offset(), component.size(),
                indExpr.value())};
            if (added != Ok) {
              return Ok;
            }
          }
        }
        offset += elementBytes;
      }
    }
    return Ok;
  }
}

void InitialImage::AddPointer(
    ConstantSubscript offset, const Expr<SomeType> &pointer) {
  pointers_.emplace(offset, pointer);
}

void InitialImage::Incorporate(
    ConstantSubscript offset, const InitialImage &that) {
  CHECK(that.pointers_.empty()); // pointers are not allowed in EQUIVALENCE
  CHECK(offset + that.size() <= size());
  std::memcpy(&data_[offset], &that.data_[0], that.size());
}

// Classes used with common::SearchTypes() to (re)construct Constant<> values
// of the right type to initialize each symbol from the values that have
// been placed into its initialization image by DATA statements.
class AsConstantHelper {
public:
  using Result = std::optional<Expr<SomeType>>;
  using Types = AllTypes;
  AsConstantHelper(FoldingContext &context, const DynamicType &type,
      const ConstantSubscripts &extents, const InitialImage &image,
      ConstantSubscript offset = 0)
      : context_{context}, type_{type}, image_{image}, extents_{extents},
        offset_{offset} {
    CHECK(!type.IsPolymorphic());
  }
  template <typename T> Result Test() {
    if (T::category != type_.category()) {
      return std::nullopt;
    }
    if constexpr (T::category != TypeCategory::Derived) {
      if (T::kind != type_.kind()) {
        return std::nullopt;
      }
    }
    using Const = Constant<T>;
    using Scalar = typename Const::Element;
    std::size_t elements{TotalElementCount(extents_)};
    std::vector<Scalar> typedValue(elements);
    auto stride{type_.MeasureSizeInBytes()};
    CHECK(stride > 0);
    CHECK(offset_ + elements * *stride <= image_.data_.size());
    if constexpr (T::category == TypeCategory::Derived) {
      const semantics::DerivedTypeSpec &derived{type_.GetDerivedTypeSpec()};
      for (auto iter : DEREF(derived.scope())) {
        const Symbol &component{*iter.second};
        bool isPointer{IsPointer(component)};
        if (component.has<semantics::ObjectEntityDetails>() ||
            component.has<semantics::ProcEntityDetails>()) {
          auto componentType{DynamicType::From(component)};
          CHECK(componentType);
          auto at{offset_ + component.offset()};
          if (isPointer) {
            for (std::size_t j{0}; j < elements; ++j, at += *stride) {
              Result value{image_.AsConstantDataPointer(*componentType, at)};
              CHECK(value);
              typedValue[j].emplace(component, std::move(*value));
            }
          } else {
            auto componentExtents{GetConstantExtents(context_, component)};
            CHECK(componentExtents);
            for (std::size_t j{0}; j < elements; ++j, at += *stride) {
              Result value{image_.AsConstant(
                  context_, *componentType, *componentExtents, at)};
              CHECK(value);
              typedValue[j].emplace(component, std::move(*value));
            }
          }
        }
      }
      return AsGenericExpr(
          Const{derived, std::move(typedValue), std::move(extents_)});
    } else if constexpr (T::category == TypeCategory::Character) {
      auto length{static_cast<ConstantSubscript>(*stride) / T::kind};
      for (std::size_t j{0}; j < elements; ++j) {
        using Char = typename Scalar::value_type;
        const Char *data{reinterpret_cast<const Char *>(
            &image_.data_[offset_ + j * *stride])};
        typedValue[j].assign(data, length);
      }
      return AsGenericExpr(
          Const{length, std::move(typedValue), std::move(extents_)});
    } else {
      // Lengthless intrinsic type
      CHECK(sizeof(Scalar) <= *stride);
      for (std::size_t j{0}; j < elements; ++j) {
        std::memcpy(&typedValue[j], &image_.data_[offset_ + j * *stride],
            sizeof(Scalar));
      }
      return AsGenericExpr(Const{std::move(typedValue), std::move(extents_)});
    }
  }

private:
  FoldingContext &context_;
  const DynamicType &type_;
  const InitialImage &image_;
  ConstantSubscripts extents_; // a copy
  ConstantSubscript offset_;
};

std::optional<Expr<SomeType>> InitialImage::AsConstant(FoldingContext &context,
    const DynamicType &type, const ConstantSubscripts &extents,
    ConstantSubscript offset) const {
  return common::SearchTypes(
      AsConstantHelper{context, type, extents, *this, offset});
}

class AsConstantDataPointerHelper {
public:
  using Result = std::optional<Expr<SomeType>>;
  using Types = AllTypes;
  AsConstantDataPointerHelper(const DynamicType &type,
      const InitialImage &image, ConstantSubscript offset = 0)
      : type_{type}, image_{image}, offset_{offset} {}
  template <typename T> Result Test() {
    if (T::category != type_.category()) {
      return std::nullopt;
    }
    if constexpr (T::category != TypeCategory::Derived) {
      if (T::kind != type_.kind()) {
        return std::nullopt;
      }
    }
    auto iter{image_.pointers_.find(offset_)};
    if (iter == image_.pointers_.end()) {
      return AsGenericExpr(NullPointer{});
    }
    return iter->second;
  }

private:
  const DynamicType &type_;
  const InitialImage &image_;
  ConstantSubscript offset_;
};

std::optional<Expr<SomeType>> InitialImage::AsConstantDataPointer(
    const DynamicType &type, ConstantSubscript offset) const {
  return common::SearchTypes(AsConstantDataPointerHelper{type, *this, offset});
}

const ProcedureDesignator &InitialImage::AsConstantProcPointer(
    ConstantSubscript offset) const {
  auto iter{pointers_.find(0)};
  CHECK(iter != pointers_.end());
  return DEREF(std::get_if<ProcedureDesignator>(&iter->second.u));
}

} // namespace Fortran::evaluate
