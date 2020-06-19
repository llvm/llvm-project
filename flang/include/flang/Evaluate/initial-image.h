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
  explicit InitialImage(std::size_t bytes) : data_(bytes) {}

  std::size_t size() const { return data_.size(); }

  template <typename A> bool Add(ConstantSubscript, std::size_t, const A &) {
    return false;
  }
  template <typename T>
  bool Add(ConstantSubscript offset, std::size_t bytes, const Constant<T> &x) {
    CHECK(offset >= 0 && offset + bytes <= data_.size());
    auto elementBytes{x.GetType().MeasureSizeInBytes()};
    CHECK(elementBytes && bytes == x.values().size() * *elementBytes);
    std::memcpy(&data_.at(offset), &x.values().at(0), bytes);
    return true;
  }
  template <int KIND>
  bool Add(ConstantSubscript offset, std::size_t bytes,
      const Constant<Type<TypeCategory::Character, KIND>> &x) {
    CHECK(offset >= 0 && offset + bytes <= data_.size());
    auto elements{TotalElementCount(x.shape())};
    auto elementBytes{bytes > 0 ? bytes / elements : 0};
    CHECK(elements * elementBytes == bytes);
    for (auto at{x.lbounds()}; elements-- > 0; x.IncrementSubscripts(at)) {
      auto scalar{x.At(at)}; // this is a std string; size() in chars
      // Subtle: an initializer for a substring may have been
      // expanded to the length of the entire string.
      CHECK(scalar.size() * KIND == elementBytes ||
          (elements == 0 && scalar.size() * KIND > elementBytes));
      std::memcpy(&data_[offset], scalar.data(), elementBytes);
      offset += elementBytes;
    }
    return true;
  }
  bool Add(ConstantSubscript, std::size_t, const Constant<SomeDerived> &);
  template <typename T>
  bool Add(ConstantSubscript offset, std::size_t bytes, const Expr<T> &x) {
    return std::visit(
        [&](const auto &y) { return Add(offset, bytes, y); }, x.u);
  }

  void AddPointer(ConstantSubscript, const Expr<SomeType> &);

  // Conversions to constant initializers
  std::optional<Expr<SomeType>> AsConstant(FoldingContext &,
      const DynamicType &, const ConstantSubscripts &,
      ConstantSubscript offset = 0) const;
  std::optional<Expr<SomeType>> AsConstantDataPointer(
      const DynamicType &, ConstantSubscript offset = 0) const;
  const ProcedureDesignator &AsConstantProcPointer(
      ConstantSubscript offset = 0) const;

  friend class AsConstantHelper;
  friend class AsConstantDataPointerHelper;

private:
  std::vector<char> data_;
  std::map<ConstantSubscript, Expr<SomeType>> pointers_;
};

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_INITIAL_IMAGE_H_
