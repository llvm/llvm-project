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
    auto optElements{TotalElementCount(x.shape())};
    if (!optElements) {
      return TooManyElems;
    }
    auto elements{*optElements};
    auto elementBytes{bytes > 0 ? bytes / elements : 0};
    if (elements * elementBytes != bytes) {
      return SizeMismatch;
    } else {
      auto at{x.lbounds()};
      Result result{OkNoChange};
      for (; elements-- > 0; x.IncrementSubscripts(at)) {
        auto scalar{x.At(at)};
        // TODO: length type parameter values?
        for (const auto &[symbolRef, indExpr] : scalar) {
          const Symbol &component{*symbolRef};
          Result status{OkNoChange};
          if (component.offset() + component.size() > elementBytes) {
            return SizeMismatch;
          } else if (IsPointer(component)) {
            status = AddPointer(offset + component.offset(), indExpr.value());
          } else if (IsAllocatable(component) || IsAutomatic(component)) {
            return NotAConstant;
          } else {
            status = Add(offset + component.offset(), component.size(),
                indExpr.value(), context);
          }
          if (status == Ok) {
            result = Ok;
          } else if (status != OkNoChange) {
            return status;
          }
        }
        offset += elementBytes;
      }
      return result;
    }
  }
}

auto InitialImage::AddPointer(
    ConstantSubscript offset, const Expr<SomeType> &pointer) -> Result {
  auto [iter, isNew]{pointers_.emplace(offset, pointer)};
  return !isNew && iter->second == pointer ? OkNoChange : Ok;
}

bool InitialImage::Incorporate(ConstantSubscript toOffset,
    const InitialImage &from, ConstantSubscript fromOffset,
    ConstantSubscript bytes) {
  CHECK(from.pointers_.empty()); // pointers are not allowed in EQUIVALENCE
  CHECK(fromOffset >= 0 && bytes >= 0 &&
      static_cast<std::size_t>(fromOffset + bytes) <= from.size());
  CHECK(static_cast<std::size_t>(toOffset + bytes) <= size());
  auto *dest{&data_[toOffset]};
  const auto *source{&from.data_[fromOffset]};
  if (std::memcmp(dest, source, bytes) != 0) {
    std::memcpy(dest, source, bytes);
    return true;
  } else {
    return false; // no change
  }
}

// Classes used with common::SearchTypes() to (re)construct Constant<> values
// of the right type to initialize each symbol from the values that have
// been placed into its initialization image by DATA statements.
class AsConstantHelper {
public:
  using Result = std::optional<Expr<SomeType>>;
  using Types = AllTypes;
  AsConstantHelper(FoldingContext &context, const DynamicType &type,
      std::optional<std::int64_t> charLength, const ConstantSubscripts &extents,
      const InitialImage &image, bool padWithZero = false,
      ConstantSubscript offset = 0)
      : context_{context}, type_{type}, charLength_{charLength}, image_{image},
        extents_{extents}, padWithZero_{padWithZero}, offset_{offset} {
    CHECK(!type.IsPolymorphic());
  }
  template <typename T> Result Test(KindsEnum kind) {
    if (T::category != type_.category()) {
      return std::nullopt;
    }
    if constexpr (T::category != TypeCategory::Derived) {
      if (kind != type_.kind()) {
        return std::nullopt;
      }
    }
    CHECK_KIND(kind, T);
    using Const = Constant<T>;
    using Scalar = typename Const::Element;
    std::optional<uint64_t> optElements{TotalElementCount(extents_)};
    CHECK(optElements);
    uint64_t elements{*optElements};
    std::vector<Scalar> typedValue(elements);
    auto elemBytes{ToInt64(type_.MeasureSizeInBytes(
        context_, GetRank(extents_) > 0, charLength_))};
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
              } else {
                typedValue[j].emplace(component, Expr<SomeType>{NullPointer{}});
              }
            }
          } else if (IsAllocatable(component)) {
            // Lowering needs an explicit NULL() for allocatables
            for (std::size_t j{0}; j < elements; ++j, at += stride) {
              typedValue[j].emplace(component, Expr<SomeType>{NullPointer{}});
            }
          } else {
            auto componentType{DynamicType::From(component)};
            CHECK(componentType.has_value());
            auto componentExtents{GetConstantExtents(context_, component)};
            CHECK(componentExtents.has_value());
            for (std::size_t j{0}; j < elements; ++j, at += stride) {
              if (Result value{image_.AsConstant(context_, *componentType,
                      std::nullopt, *componentExtents, padWithZero_, at)}) {
                typedValue[j].emplace(component, std::move(*value));
              }
            }
          }
        }
      }
      return AsGenericExpr(
          Const{derived, std::move(typedValue), std::move(extents_)});
    } else if constexpr (T::category == TypeCategory::Character) {
      auto length{
          static_cast<ConstantSubscript>(stride) / static_cast<int>(kind)};
      llvm::SmallVector<char, 256> buffer;
      const char *data{GetTailPaddedData(offset_, elements * stride, buffer)};
      for (std::size_t j{0}; j < elements; ++j) {
        typedValue[j] = value::CharacterValue::FromRawBytes(
            kind, data + j * stride, length * static_cast<int>(kind));
      }
      return AsGenericExpr(
          Const{kind, length, std::move(typedValue), std::move(extents_)});
    } else {
      // Lengthless intrinsic type
      llvm::SmallVector<char, 256> buffer;
      const char *data{GetTailPaddedData(offset_,
          elements == 0 ? 0
                        : (elements - 1) * stride +
                  evaluate::Scalar<T>::bytesStored(kind),
          buffer)};
      // TODO endianness
      LoadSerialValues(kind, data,
          llvm::MutableArrayRef<evaluate::Scalar<T>>(typedValue), stride);
      return AsGenericExpr(
          Const{kind, std::move(typedValue), std::move(extents_)});
    }
  }

private:
  /// Returns the image's bytes, extended with zero bytes when a value is being
  /// built whose representation reaches past the end of the image.  That
  /// happens when TRANSFER() is folded with a MOLD= whose representation is
  /// longer than SOURCE=, and when deserializing a scalar accesses more bytes
  /// than its element size because its host representation is padded (e.g.,
  /// REAL(10)).  F2023 16.9.212 leaves the bytes beyond SOURCE= processor
  /// dependent; flang zero-fills them, as the runtime does.
  const char *GetTailPaddedData(std::size_t offset, std::size_t bytes,
      llvm::SmallVectorImpl<char> &buffer) const {
    if (bytes + offset <= image_.data_.size()) {
      // If no padding is needed, use original data without copy
      return image_.data_.data() + offset;
    }
    CHECK(padWithZero_);
    buffer.assign(bytes, 0);
    if (offset < image_.data_.size()) {
      std::memcpy(buffer.data(), image_.data_.data() + offset,
          image_.data_.size() - offset);
    }
    return buffer.data();
  }

  FoldingContext &context_;
  const DynamicType &type_;
  std::optional<std::int64_t> charLength_;
  const InitialImage &image_;
  ConstantSubscripts extents_; // a copy
  bool padWithZero_;
  ConstantSubscript offset_;
};

std::optional<Expr<SomeType>> InitialImage::AsConstant(FoldingContext &context,
    const DynamicType &type, std::optional<std::int64_t> charLength,
    const ConstantSubscripts &extents, bool padWithZero,
    ConstantSubscript offset) const {
  return SearchTypes(AsConstantHelper{
      context, type, charLength, extents, *this, padWithZero, offset});
}

std::optional<Expr<SomeType>> InitialImage::AsConstantPointer(
    ConstantSubscript offset) const {
  auto iter{pointers_.find(offset)};
  return iter == pointers_.end() ? std::optional<Expr<SomeType>>{}
                                 : iter->second;
}

} // namespace Fortran::evaluate
