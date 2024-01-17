//===- AttrIterator.h - Classes for attribute iteration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Attr vector and specific_attr_iterator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ATTRITERATOR_H
#define LLVM_CLANG_AST_ATTRITERATOR_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstddef>
#include <iterator>

namespace clang {

class Attr;

/// AttrVec - A vector of Attr, which is how they are stored on the AST.
using AttrVec = SmallVector<Attr *, 4>;

/// Iterates over a subrange of container, only providing attributes that are of
/// a specific type/s.
template <typename Container, typename... SpecificAttrs>
class specific_attr_iterator_impl {
  using Iterator = typename Container::const_iterator;

  /// Helper class to get either the singular 'specific-attr', or Attr,
  /// depending on how many are specified.
  template <typename... Ts> struct type_helper {
    using type = Attr;
  };
  template <typename T> struct type_helper<T> {
    using type = T;
  };

  /// The pointee type of the value_type, used for internal implementation.
  using base_type = typename type_helper<SpecificAttrs...>::type;

  /// Current - The current, underlying iterator.
  /// In order to ensure we don't dereference an invalid iterator unless
  /// specifically requested, we don't necessarily advance this all the
  /// way. Instead, we advance it when an operation is requested; if the
  /// operation is acting on what should be a past-the-end iterator,
  /// then we offer no guarantees, but this way we do not dereference a
  /// past-the-end iterator when we move to a past-the-end position.
  mutable Iterator Current;

  void AdvanceToNext() const {
    while (!isa<SpecificAttrs...>(*Current))
      ++Current;
  }

  void AdvanceToNext(Iterator I) const {
    while (Current != I && !isa<SpecificAttrs...>(*Current))
      ++Current;
  }

public:
  using value_type = base_type *;
  using reference = value_type;
  using pointer = value_type;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;

  specific_attr_iterator_impl() = default;
  explicit specific_attr_iterator_impl(Iterator i) : Current(i) {}

  reference operator*() const {
    AdvanceToNext();
    return cast<base_type>(*Current);
  }
  pointer operator->() const {
    AdvanceToNext();
    return cast<base_type>(*Current);
  }

  specific_attr_iterator_impl &operator++() {
    ++Current;
    return *this;
  }
  specific_attr_iterator_impl operator++(int) {
    specific_attr_iterator_impl Tmp(*this);
    ++(*this);
    return Tmp;
  }

  friend bool operator==(specific_attr_iterator_impl Left,
                         specific_attr_iterator_impl Right) {
    assert((Left.Current == nullptr) == (Right.Current == nullptr));
    if (Left.Current < Right.Current)
      Left.AdvanceToNext(Right.Current);
    else
      Right.AdvanceToNext(Left.Current);
    return Left.Current == Right.Current;
  }
  friend bool operator!=(specific_attr_iterator_impl Left,
                         specific_attr_iterator_impl Right) {
    return !(Left == Right);
  }
};

/// Iterates over a subrange of a collection, only providing attributes that are
/// of a specific type/s.
template <typename Container, typename... SpecificAttrs>
class specific_attr_iterator;

template <typename SpecificAttr>
class specific_attr_iterator<SpecificAttr>
    : public specific_attr_iterator_impl<AttrVec, SpecificAttr> {
  using specific_attr_iterator_impl<AttrVec,
                                    SpecificAttr>::specific_attr_iterator_impl;
};

template <typename Container, typename... SpecificAttrs>
class specific_attr_iterator
    : public specific_attr_iterator_impl<Container, SpecificAttrs...> {
  using specific_attr_iterator_impl<
      Container, SpecificAttrs...>::specific_attr_iterator_impl;
};

template <typename... SpecificAttrs, typename Container>
inline specific_attr_iterator<Container, SpecificAttrs...>
specific_attr_begin(const Container &container) {
  return specific_attr_iterator<Container, SpecificAttrs...>(container.begin());
}

template <typename... SpecificAttrs, typename Container>
inline specific_attr_iterator<Container, SpecificAttrs...>
specific_attr_end(const Container &container) {
  return specific_attr_iterator<Container, SpecificAttrs...>(container.end());
}

template <typename... SpecificAttrs, typename Container>
inline bool hasSpecificAttr(const Container &container) {
  return specific_attr_begin<SpecificAttrs...>(container) !=
         specific_attr_end<SpecificAttrs...>(container);
}

template <typename... SpecificAttrs, typename Container>
inline typename specific_attr_iterator_impl<Container,
                                            SpecificAttrs...>::value_type
getSpecificAttr(const Container &container) {
  auto i = specific_attr_begin<SpecificAttrs...>(container);
  if (i != specific_attr_end<SpecificAttrs...>(container))
    return *i;
  else
    return nullptr;
}

} // namespace clang

#endif // LLVM_CLANG_AST_ATTRITERATOR_H
