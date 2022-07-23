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
    const Constant<SomeDerived> &x, FoldingContext &context) -> Result {
  if (offset < 0 || offset + bytes > data_.size()) {
    return OutOfRange;
  } else {
    auto elements{TotalElementCount(x.shape())};
    auto elementBytes{bytes > 0 ? bytes / elements : 0};
    if (elements * elementBytes != bytes) {
      return SizeMismatch;
    } else {
      auto at{x.lbounds()};
      for (; elements-- > 0; x.IncrementSubscripts(at)) {
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
                indExpr.value(), context)};
            if (added != Ok) {
              return added;
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

void InitialImage::Incorporate(ConstantSubscript toOffset,
    const InitialImage &from, ConstantSubscript fromOffset,
    ConstantSubscript bytes) {
  CHECK(from.pointers_.empty()); // pointers are not allowed in EQUIVALENCE
  CHECK(fromOffset >= 0 && bytes >= 0 &&
      static_cast<std::size_t>(fromOffset + bytes) <= from.size());
  CHECK(static_cast<std::size_t>(toOffset + bytes) <= size());
  std::memcpy(&data_[toOffset], &from.data_[fromOffset], bytes);
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
      bool padWithZero = false, ConstantSubscript offset = 0)
      : context_{context}, type_{type}, image_{image}, extents_{extents},
        padWithZero_{padWithZero}, offset_{offset} {
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
    auto elemBytes{
        ToInt64(type_.MeasureSizeInBytes(context_, GetRank(extents_) > 0))};
    CHECK(elemBytes && *elemBytes >= 0);
    std::size_t stride{static_cast<std::size_t>(*elemBytes)};
    CHECK(offset_ + elements * stride <= image_.data_.size() || padWithZero_);
    if constexpr (T::category == TypeCategory::Derived) {
      const semantics::DerivedTypeSpec &derived{type_.GetDerivedTypeSpec()};
      for (auto iter : DEREF(derived.scope())) {
        const Symbol &component{*iter.second};
        bool isProcPtr{IsProcedurePointer(component)};
        if (isProcPtr || component.has<semantics::ObjectEntityDetails>()) {
          auto at{offset_ + component.offset()};
          if (isProcPtr) {
            for (std::size_t j{0}; j < elements; ++j, at += stride) {
              if (Result value{image_.AsConstantPointer(at)}) {
                typedValue[j].emplace(component, std::move(*value));
              }
            }
          } else if (IsPointer(component)) {
            for (std::size_t j{0}; j < elements; ++j, at += stride) {
              if (Result value{image_.AsConstantPointer(at)}) {
                typedValue[j].emplace(component, std::move(*value));
              }
            }
          } else if (!IsAllocatable(component)) {
            auto componentType{DynamicType::From(component)};
            CHECK(componentType.has_value());
            auto componentExtents{GetConstantExtents(context_, component)};
            CHECK(componentExtents.has_value());
            for (std::size_t j{0}; j < elements; ++j, at += stride) {
              if (Result value{image_.AsConstant(context_, *componentType,
                      *componentExtents, padWithZero_, at)}) {
                typedValue[j].emplace(component, std::move(*value));
              }
            }
          }
        }
      }
      return AsGenericExpr(
          Const{derived, std::move(typedValue), std::move(extents_)});
    } else if constexpr (T::category == TypeCategory::Character) {
      auto length{static_cast<ConstantSubscript>(stride) / T::kind};
      for (std::size_t j{0}; j < elements; ++j) {
        using Char = typename Scalar::value_type;
        auto at{static_cast<std::size_t>(offset_ + j * stride)};
        auto chunk{length};
        if (at + chunk > image_.data_.size()) {
          CHECK(padWithZero_);
          if (at >= image_.data_.size()) {
            chunk = 0;
          } else {
            chunk = image_.data_.size() - at;
          }
        }
        if (chunk > 0) {
          const Char *data{reinterpret_cast<const Char *>(&image_.data_[at])};
          typedValue[j].assign(data, chunk);
        }
        if (chunk < length && padWithZero_) {
          typedValue[j].append(length - chunk, Char{});
        }
      }
      return AsGenericExpr(
          Const{length, std::move(typedValue), std::move(extents_)});
    } else {
      // Lengthless intrinsic type
      CHECK(sizeof(Scalar) <= stride);
      for (std::size_t j{0}; j < elements; ++j) {
        auto at{static_cast<std::size_t>(offset_ + j * stride)};
        std::size_t chunk{sizeof(Scalar)};
        if (at + chunk > image_.data_.size()) {
          CHECK(padWithZero_);
          if (at >= image_.data_.size()) {
            chunk = 0;
          } else {
            chunk = image_.data_.size() - at;
          }
        }
        // TODO endianness
        if (chunk > 0) {
          std::memcpy(&typedValue[j], &image_.data_[at], chunk);
        }
      }
      return AsGenericExpr(Const{std::move(typedValue), std::move(extents_)});
    }
  }

private:
  FoldingContext &context_;
  const DynamicType &type_;
  const InitialImage &image_;
  ConstantSubscripts extents_; // a copy
  bool padWithZero_;
  ConstantSubscript offset_;
};

std::optional<Expr<SomeType>> InitialImage::AsConstant(FoldingContext &context,
    const DynamicType &type, const ConstantSubscripts &extents,
    bool padWithZero, ConstantSubscript offset) const {
  return common::SearchTypes(
      AsConstantHelper{context, type, extents, *this, padWithZero, offset});
}

std::optional<Expr<SomeType>> InitialImage::AsConstantPointer(
    ConstantSubscript offset) const {
  auto iter{pointers_.find(offset)};
  return iter == pointers_.end() ? std::optional<Expr<SomeType>>{}
                                 : iter->second;
}

} // namespace Fortran::evaluate
