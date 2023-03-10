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
#include <map>
#include <optional>
#include <vector>

namespace Fortran::evaluate {

class InitialImage {
public:
  enum Result {
    Ok,
    NotAConstant,
    OutOfRange,
    SizeMismatch,
  };

  explicit InitialImage(std::size_t bytes) : data_(bytes) {}
  InitialImage(InitialImage &&that) = default;

  std::size_t size() const { return data_.size(); }

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
        return Ok;
      } else {
        // TODO endianness
        std::memcpy(&data_.at(offset), &x.values().at(0), bytes);
        return Ok;
      }
    }
  }
  template <int KIND>
  Result Add(ConstantSubscript offset, std::size_t bytes,
      const Constant<Type<TypeCategory::Character, KIND>> &x,
      FoldingContext &) {
    if (offset < 0 || offset + bytes > data_.size()) {
      return OutOfRange;
    } else {
      auto elements{TotalElementCount(x.shape())};
      auto elementBytes{bytes > 0 ? bytes / elements : 0};
      if (elements * elementBytes != bytes) {
        return SizeMismatch;
      } else if (bytes == 0) {
        return Ok;
      } else {
        Result result{Ok};
        for (auto at{x.lbounds()}; elements-- > 0; x.IncrementSubscripts(at)) {
          auto scalar{x.At(at)}; // this is a std string; size() in chars
          auto scalarBytes{scalar.size() * KIND};
          if (scalarBytes != elementBytes) {
            result = SizeMismatch;
          }
          // Blank padding when short
          for (; scalarBytes < elementBytes; scalarBytes += KIND) {
            scalar += ' ';
          }
          // TODO endianness
          std::memcpy(&data_.at(offset), scalar.data(), elementBytes);
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

  void AddPointer(ConstantSubscript, const Expr<SomeType> &);

  void Incorporate(ConstantSubscript toOffset, const InitialImage &from,
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
