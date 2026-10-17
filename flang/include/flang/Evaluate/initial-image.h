//===-------include/flang/Evaluate/initial-image.h ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INITIAL_IMAGE_H_
#define FORTRAN_EVALUATE_INITIAL_IMAGE_H_

// Represents the initialized storage of an object during DATA statement
// processing, including the conversion of that image to a constant
// initializer for a symbol.

#include "expression.h"
#include "flang/Evaluate/char.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <optional>
#include <vector>

namespace Fortran::evaluate {

/// Reverses the bytes of each \p unit sized piece of \p p[0..bytes).
inline void ReverseByteUnits(char *p, std::size_t bytes, std::size_t unit) {
  if (unit > 1) {
    for (std::size_t j{0}; j + unit <= bytes; j += unit) {
      std::reverse(p + j, p + j + unit);
    }
  }
}

/// Serializes \p values to \p dst. If \p swapUnit is greater than one, the
/// bytes of each \p swapUnit sized piece are reversed (host to target byte
/// order).
template <typename SCALAR>
inline void StoreSerialValues(char *dst, llvm::ArrayRef<SCALAR> values,
    size_t elementSize, bool *changed = nullptr, size_t swapUnit = 0) {
  for (auto [i, v] : llvm::enumerate(values)) {
    char *to{dst + i * elementSize};
    if (swapUnit > 1) {
      llvm::SmallVector<char, 32> buffer(elementSize);
      v.StoreRawBytes(buffer.data(), elementSize);
      ReverseByteUnits(buffer.data(), elementSize, swapUnit);
      if (changed) {
        if (std::memcmp(to, buffer.data(), elementSize) == 0) {
          continue;
        }
        *changed = true;
      }
      std::memcpy(to, buffer.data(), elementSize);
    } else {
      v.StoreRawBytes(to, elementSize, changed);
    }
  }
}

/// De-serializes \p values from \p src; \p swapUnit as for
/// StoreSerialValues (target to host byte order).
template <typename SCALAR>
inline void LoadSerialValues(const char *src,
    llvm::MutableArrayRef<SCALAR> values, size_t stride, size_t swapUnit = 0) {
  for (auto it : llvm::enumerate(values)) {
    const char *from{src + stride * it.index()};
    if (swapUnit > 1) {
      llvm::SmallVector<char, 32> buffer(from, from + SCALAR::bytesStored());
      ReverseByteUnits(buffer.data(), buffer.size(), swapUnit);
      it.value() = SCALAR::FromRawBytes(buffer.data(), SCALAR::bytesStored());
    } else {
      it.value() = SCALAR::FromRawBytes(from, SCALAR::bytesStored());
    }
  }
}

class InitialImage {
public:
  enum Result {
    Ok,
    OkNoChange,
    NotAConstant,
    OutOfRange,
    SizeMismatch,
    LengthMismatch,
    TooManyElems,
  };

  explicit InitialImage(std::size_t bytes) : data_(bytes) {}
  InitialImage(InitialImage &&that) = default;

  std::size_t size() const { return data_.size(); }

  /// The image holds the bytes of the values in the byte order of the target.
  /// Returns the size of the pieces whose bytes have to be reversed when the
  /// byte orders of the target and the host differ, or 0.
  template <typename T>
  static std::size_t ByteSwapUnit(
      const FoldingContext &context, std::size_t elementBytes) {
    if (context.targetCharacteristics().isBigEndian() != isHostLittleEndian) {
      return 0; // same byte order
    } else if constexpr (T::category == TypeCategory::Character) {
      return T::kind;
    } else if constexpr (T::category == TypeCategory::Complex) {
      return elementBytes / 2; // real and imaginary parts
    } else {
      return elementBytes;
    }
  }

  template <typename A>
  Result Add(ConstantSubscript, std::size_t, const A &, FoldingContext &) {
    return NotAConstant;
  }
  template <typename T>
  Result Add(ConstantSubscript offset, std::size_t bytes, const Constant<T> &x,
      FoldingContext &context) {
    if (offset < 0 || offset + bytes > data_.size()) {
      return OutOfRange;
    } else {
      auto elementBytes{ToInt64(x.GetType().MeasureSizeInBytes(context, true))};
      if (!elementBytes ||
          bytes !=
              x.values().size() * static_cast<std::size_t>(*elementBytes)) {
        return SizeMismatch;
      } else if (bytes == 0) {
        return OkNoChange;
      } else {
        bool changed{false};
        StoreSerialValues<Scalar<T>>(&data_.at(offset),
            llvm::ArrayRef<Scalar<T>>(x.values()), *elementBytes, &changed,
            ByteSwapUnit<T>(context, *elementBytes));
        return changed ? Ok : OkNoChange;
      }
    }
  }
  template <int KIND>
  Result Add(ConstantSubscript offset, std::size_t bytes,
      const Constant<Type<TypeCategory::Character, KIND>> &x,
      FoldingContext &context) {
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
      } else if (bytes == 0) {
        return OkNoChange;
      } else {
        Result result{OkNoChange};
        for (auto at{x.lbounds()}; elements-- > 0; x.IncrementSubscripts(at)) {
          typename value::Character<KIND> scalar{x.At(at)};
          auto scalarBytes{scalar.size() * KIND};
          if (scalarBytes != elementBytes) {
            result = LengthMismatch;
          }
          auto *to{&data_.at(offset)};
          bool changed{false};
          if (std::size_t unit{
                  ByteSwapUnit<Type<TypeCategory::Character, KIND>>(
                      context, elementBytes)};
              unit > 1) {
            llvm::SmallVector<char, 64> buffer(elementBytes);
            scalar.StoreRawBytes(buffer.data(), elementBytes);
            ReverseByteUnits(buffer.data(), elementBytes, unit);
            if (std::memcmp(to, buffer.data(), elementBytes) != 0) {
              std::memcpy(to, buffer.data(), elementBytes);
              changed = true;
            }
          } else {
            scalar.StoreRawBytes(to, elementBytes, &changed);
          }
          if (changed && result == OkNoChange) {
            result = Ok;
          }
          offset += elementBytes;
        }
        return result;
      }
    }
  }
  Result Add(ConstantSubscript, std::size_t, const Constant<SomeDerived> &,
      FoldingContext &);
  template <typename T>
  Result Add(ConstantSubscript offset, std::size_t bytes, const Expr<T> &x,
      FoldingContext &c) {
    return common::visit(
        [&](const auto &y) { return Add(offset, bytes, y, c); }, x.u);
  }

  Result AddPointer(ConstantSubscript, const Expr<SomeType> &);

  // Returns true if anything changes
  bool Incorporate(ConstantSubscript toOffset, const InitialImage &from,
      ConstantSubscript fromOffset, ConstantSubscript bytes);

  // Conversions to constant initializers
  std::optional<Expr<SomeType>> AsConstant(FoldingContext &,
      const DynamicType &, std::optional<std::int64_t> charLength,
      const ConstantSubscripts &, bool padWithZero = false,
      ConstantSubscript offset = 0) const;
  std::optional<Expr<SomeType>> AsConstantPointer(
      ConstantSubscript offset = 0) const;

  friend class AsConstantHelper;

private:
  std::vector<char> data_;
  std::map<ConstantSubscript, Expr<SomeType>> pointers_;
};

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_INITIAL_IMAGE_H_
