//===-- lib/Evaluate/constant.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/constant.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/type.h"
#include <string>

namespace Fortran::evaluate {

ConstantBounds::ConstantBounds(const ConstantSubscripts &shape)
    : shape_(shape), lbounds_(shape_.size(), 1) {}

ConstantBounds::ConstantBounds(ConstantSubscripts &&shape)
    : shape_(std::move(shape)), lbounds_(shape_.size(), 1) {}

ConstantBounds::~ConstantBounds() = default;

void ConstantBounds::set_lbounds(ConstantSubscripts &&lb) {
  CHECK(lb.size() == shape_.size());
  lbounds_ = std::move(lb);
  for (std::size_t j{0}; j < shape_.size(); ++j) {
    if (shape_[j] == 0) {
      lbounds_[j] = 1;
    }
  }
}

void ConstantBounds::SetLowerBoundsToOne() {
  for (auto &n : lbounds_) {
    n = 1;
  }
}

Constant<SubscriptInteger> ConstantBounds::SHAPE() const {
  return AsConstantShape(shape_);
}

ConstantSubscript ConstantBounds::SubscriptsToOffset(
    const ConstantSubscripts &index) const {
  CHECK(GetRank(index) == GetRank(shape_));
  ConstantSubscript stride{1}, offset{0};
  int dim{0};
  for (auto j : index) {
    auto lb{lbounds_[dim]};
    auto extent{shape_[dim++]};
    CHECK(j >= lb && j < lb + extent);
    offset += stride * (j - lb);
    stride *= extent;
  }
  return offset;
}

std::size_t TotalElementCount(const ConstantSubscripts &shape) {
  return static_cast<std::size_t>(GetSize(shape));
}

bool ConstantBounds::IncrementSubscripts(
    ConstantSubscripts &indices, const std::vector<int> *dimOrder) const {
  int rank{GetRank(shape_)};
  CHECK(GetRank(indices) == rank);
  CHECK(!dimOrder || static_cast<int>(dimOrder->size()) == rank);
  for (int j{0}; j < rank; ++j) {
    ConstantSubscript k{dimOrder ? (*dimOrder)[j] : j};
    auto lb{lbounds_[k]};
    CHECK(indices[k] >= lb);
    if (++indices[k] < lb + shape_[k]) {
      return true;
    } else {
      CHECK(indices[k] == lb + std::max<ConstantSubscript>(shape_[k], 1));
      indices[k] = lb;
    }
  }
  return false; // all done
}

std::optional<std::vector<int>> ValidateDimensionOrder(
    int rank, const std::vector<int> &order) {
  std::vector<int> dimOrder(rank);
  if (static_cast<int>(order.size()) == rank) {
    std::bitset<common::maxRank> seenDimensions;
    for (int j{0}; j < rank; ++j) {
      int dim{order[j]};
      if (dim < 1 || dim > rank || seenDimensions.test(dim - 1)) {
        return std::nullopt;
      }
      dimOrder[j] = dim - 1;
      seenDimensions.set(dim - 1);
    }
    return dimOrder;
  } else {
    return std::nullopt;
  }
}

bool HasNegativeExtent(const ConstantSubscripts &shape) {
  for (ConstantSubscript extent : shape) {
    if (extent < 0) {
      return true;
    }
  }
  return false;
}

template <typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::ConstantBase(
    std::vector<Element> &&x, ConstantSubscripts &&sh, Result res)
    : ConstantBounds(std::move(sh)), result_{res}, values_(std::move(x)) {
  CHECK(size() == TotalElementCount(shape()));
}

template <typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::~ConstantBase() {}

template <typename RESULT, typename ELEMENT>
bool ConstantBase<RESULT, ELEMENT>::operator==(const ConstantBase &that) const {
  return shape() == that.shape() && values_ == that.values_;
}

template <typename RESULT, typename ELEMENT>
auto ConstantBase<RESULT, ELEMENT>::Reshape(
    const ConstantSubscripts &dims) const -> std::vector<Element> {
  std::size_t n{TotalElementCount(dims)};
  CHECK(!empty() || n == 0);
  std::vector<Element> elements;
  auto iter{values().cbegin()};
  while (n-- > 0) {
    elements.push_back(*iter);
    if (++iter == values().cend()) {
      iter = values().cbegin();
    }
  }
  return elements;
}

template <typename RESULT, typename ELEMENT>
std::size_t ConstantBase<RESULT, ELEMENT>::CopyFrom(
    const ConstantBase<RESULT, ELEMENT> &source, std::size_t count,
    ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder) {
  std::size_t copied{0};
  ConstantSubscripts sourceSubscripts{source.lbounds()};
  while (copied < count) {
    values_.at(SubscriptsToOffset(resultSubscripts)) =
        source.values_.at(source.SubscriptsToOffset(sourceSubscripts));
    copied++;
    source.IncrementSubscripts(sourceSubscripts);
    IncrementSubscripts(resultSubscripts, dimOrder);
  }
  return copied;
}

template <typename T>
auto Constant<T>::At(const ConstantSubscripts &index) const -> Element {
  return Base::values_.at(Base::SubscriptsToOffset(index));
}

template <typename T>
auto Constant<T>::Reshape(ConstantSubscripts &&dims) const -> Constant {
  return {Base::Reshape(dims), std::move(dims)};
}

template <typename T>
std::size_t Constant<T>::CopyFrom(const Constant<T> &source, std::size_t count,
    ConstantSubscripts &resultSubscripts, const std::vector<int> *dimOrder) {
  return Base::CopyFrom(source, count, resultSubscripts, dimOrder);
}

// Constant<Type<TypeCategory::Character, KIND> specializations
template <int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(
    const Scalar<Result> &str)
    : values_{str}, length_{static_cast<ConstantSubscript>(values_.size())} {}

template <int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(Scalar<Result> &&str)
    : values_{std::move(str)}, length_{static_cast<ConstantSubscript>(
                                   values_.size())} {}

template <int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(ConstantSubscript len,
    std::vector<Scalar<Result>> &&strings, ConstantSubscripts &&sh)
    : ConstantBounds(std::move(sh)), length_{len} {
  CHECK(strings.size() == TotalElementCount(shape()));
  values_.assign(strings.size() * length_,
      static_cast<typename Scalar<Result>::value_type>(' '));
  ConstantSubscript at{0};
  for (const auto &str : strings) {
    auto strLen{static_cast<ConstantSubscript>(str.size())};
    if (strLen > length_) {
      values_.replace(at, length_, str.substr(0, length_));
    } else {
      values_.replace(at, strLen, str);
    }
    at += length_;
  }
  CHECK(at == static_cast<ConstantSubscript>(values_.size()));
}

template <int KIND>
Constant<Type<TypeCategory::Character, KIND>>::~Constant() {}

template <int KIND>
bool Constant<Type<TypeCategory::Character, KIND>>::empty() const {
  return size() == 0;
}

template <int KIND>
std::size_t Constant<Type<TypeCategory::Character, KIND>>::size() const {
  if (length_ == 0) {
    return TotalElementCount(shape());
  } else {
    return static_cast<ConstantSubscript>(values_.size()) / length_;
  }
}

template <int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::At(
    const ConstantSubscripts &index) const -> Scalar<Result> {
  auto offset{SubscriptsToOffset(index)};
  return values_.substr(offset * length_, length_);
}

template <int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::Substring(
    ConstantSubscript lo, ConstantSubscript hi) const
    -> std::optional<Constant> {
  std::vector<Element> elements;
  ConstantSubscript n{GetSize(shape())};
  ConstantSubscript newLength{0};
  if (lo > hi) { // zero-length results
    while (n-- > 0) {
      elements.emplace_back(); // ""
    }
  } else if (lo < 1 || hi > length_) {
    return std::nullopt;
  } else {
    newLength = hi - lo + 1;
    for (ConstantSubscripts at{lbounds()}; n-- > 0; IncrementSubscripts(at)) {
      elements.emplace_back(At(at).substr(lo - 1, newLength));
    }
  }
  return Constant{newLength, std::move(elements), ConstantSubscripts{shape()}};
}

template <int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::Reshape(
    ConstantSubscripts &&dims) const -> Constant<Result> {
  std::size_t n{TotalElementCount(dims)};
  CHECK(!empty() || n == 0);
  std::vector<Element> elements;
  ConstantSubscript at{0},
      limit{static_cast<ConstantSubscript>(values_.size())};
  while (n-- > 0) {
    elements.push_back(values_.substr(at, length_));
    at += length_;
    if (at == limit) { // subtle: at > limit somehow? substr() will catch it
      at = 0;
    }
  }
  return {length_, std::move(elements), std::move(dims)};
}

template <int KIND>
std::size_t Constant<Type<TypeCategory::Character, KIND>>::CopyFrom(
    const Constant<Type<TypeCategory::Character, KIND>> &source,
    std::size_t count, ConstantSubscripts &resultSubscripts,
    const std::vector<int> *dimOrder) {
  CHECK(length_ == source.length_);
  if (length_ == 0) {
    // It's possible that the array of strings consists of all empty strings.
    // If so, constant folding will result in a string that's completely empty
    // and the length_ will be zero, and there's nothing to do.
    return count;
  } else {
    std::size_t copied{0};
    std::size_t elementBytes{length_ * sizeof(decltype(values_[0]))};
    ConstantSubscripts sourceSubscripts{source.lbounds()};
    while (copied < count) {
      auto *dest{&values_.at(SubscriptsToOffset(resultSubscripts) * length_)};
      const auto *src{&source.values_.at(
          source.SubscriptsToOffset(sourceSubscripts) * length_)};
      std::memcpy(dest, src, elementBytes);
      copied++;
      source.IncrementSubscripts(sourceSubscripts);
      IncrementSubscripts(resultSubscripts, dimOrder);
    }
    return copied;
  }
}

// Constant<SomeDerived> specialization
Constant<SomeDerived>::Constant(const StructureConstructor &x)
    : Base{x.values(), Result{x.derivedTypeSpec()}} {}

Constant<SomeDerived>::Constant(StructureConstructor &&x)
    : Base{std::move(x.values()), Result{x.derivedTypeSpec()}} {}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructorValues> &&x, ConstantSubscripts &&s)
    : Base{std::move(x), std::move(s), Result{spec}} {}

static std::vector<StructureConstructorValues> AcquireValues(
    std::vector<StructureConstructor> &&x) {
  std::vector<StructureConstructorValues> result;
  for (auto &&structure : std::move(x)) {
    result.emplace_back(std::move(structure.values()));
  }
  return result;
}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructor> &&x, ConstantSubscripts &&shape)
    : Base{AcquireValues(std::move(x)), std::move(shape), Result{spec}} {}

std::optional<StructureConstructor>
Constant<SomeDerived>::GetScalarValue() const {
  if (Rank() == 0) {
    return StructureConstructor{result().derivedTypeSpec(), values_.at(0)};
  } else {
    return std::nullopt;
  }
}

StructureConstructor Constant<SomeDerived>::At(
    const ConstantSubscripts &index) const {
  return {result().derivedTypeSpec(), values_.at(SubscriptsToOffset(index))};
}

auto Constant<SomeDerived>::Reshape(ConstantSubscripts &&dims) const
    -> Constant {
  return {result().derivedTypeSpec(), Base::Reshape(dims), std::move(dims)};
}

std::size_t Constant<SomeDerived>::CopyFrom(const Constant<SomeDerived> &source,
    std::size_t count, ConstantSubscripts &resultSubscripts,
    const std::vector<int> *dimOrder) {
  return Base::CopyFrom(source, count, resultSubscripts, dimOrder);
}

bool ComponentCompare::operator()(SymbolRef x, SymbolRef y) const {
  return semantics::SymbolSourcePositionCompare{}(x, y);
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
INSTANTIATE_CONSTANT_TEMPLATES
} // namespace Fortran::evaluate
