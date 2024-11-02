//===-- include/flang/Common/enum-set.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_ENUM_SET_H_
#define FORTRAN_COMMON_ENUM_SET_H_

// Implements a set of enums as a std::bitset<>.  APIs from bitset<> and set<>
// can be used on these sets, whichever might be more clear to the user.
// This class template facilitates the use of the more type-safe C++ "enum
// class" feature without loss of convenience.

#include "constexpr-bitset.h"
#include "idioms.h"
#include <bitset>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>

namespace Fortran::common {

template <typename ENUM, std::size_t BITS> class EnumSet {
  static_assert(BITS > 0);

public:
  // When the bitset fits in a word, use a custom local bitset class that is
  // more amenable to constexpr evaluation than the current std::bitset<>.
  using bitsetType =
      std::conditional_t<(BITS <= 64), common::BitSet<BITS>, std::bitset<BITS>>;
  using enumerationType = ENUM;

  constexpr EnumSet() {}
  constexpr EnumSet(const std::initializer_list<enumerationType> &enums) {
    for (auto it{enums.begin()}; it != enums.end(); ++it) {
      set(*it);
    }
  }
  constexpr EnumSet(const EnumSet &) = default;
  constexpr EnumSet(EnumSet &&) = default;

  constexpr EnumSet &operator=(const EnumSet &) = default;
  constexpr EnumSet &operator=(EnumSet &&) = default;

  const bitsetType &bitset() const { return bitset_; }

  constexpr EnumSet &operator&=(const EnumSet &that) {
    bitset_ &= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator&=(EnumSet &&that) {
    bitset_ &= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator|=(const EnumSet &that) {
    bitset_ |= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator|=(EnumSet &&that) {
    bitset_ |= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator^=(const EnumSet &that) {
    bitset_ ^= that.bitset_;
    return *this;
  }
  constexpr EnumSet &operator^=(EnumSet &&that) {
    bitset_ ^= that.bitset_;
    return *this;
  }

  constexpr EnumSet operator~() const {
    EnumSet result;
    result.bitset_ = ~bitset_;
    return result;
  }
  constexpr EnumSet operator&(const EnumSet &that) const {
    EnumSet result{*this};
    result.bitset_ &= that.bitset_;
    return result;
  }
  constexpr EnumSet operator&(EnumSet &&that) const {
    EnumSet result{*this};
    result.bitset_ &= that.bitset_;
    return result;
  }
  constexpr EnumSet operator|(const EnumSet &that) const {
    EnumSet result{*this};
    result.bitset_ |= that.bitset_;
    return result;
  }
  constexpr EnumSet operator|(EnumSet &&that) const {
    EnumSet result{*this};
    result.bitset_ |= that.bitset_;
    return result;
  }
  constexpr EnumSet operator^(const EnumSet &that) const {
    EnumSet result{*this};
    result.bitset_ ^= that.bitset_;
    return result;
  }
  constexpr EnumSet operator^(EnumSet &&that) const {
    EnumSet result{*this};
    result.bitset_ ^= that.bitset_;
    return result;
  }

  constexpr EnumSet operator+(enumerationType v) const {
    return {*this | EnumSet{v}};
  }
  constexpr EnumSet operator-(enumerationType v) const {
    return {*this & ~EnumSet{v}};
  }

  constexpr bool operator==(const EnumSet &that) const {
    return bitset_ == that.bitset_;
  }
  constexpr bool operator==(EnumSet &&that) const {
    return bitset_ == that.bitset_;
  }
  constexpr bool operator!=(const EnumSet &that) const {
    return bitset_ != that.bitset_;
  }
  constexpr bool operator!=(EnumSet &&that) const {
    return bitset_ != that.bitset_;
  }

  // N.B. std::bitset<> has size() for max_size(), but that's not the same
  // thing as std::set<>::size(), which is an element count.
  static constexpr std::size_t max_size() { return BITS; }
  constexpr bool test(enumerationType x) const {
    return bitset_.test(static_cast<std::size_t>(x));
  }
  constexpr bool all() const { return bitset_.all(); }
  constexpr bool any() const { return bitset_.any(); }
  constexpr bool none() const { return bitset_.none(); }

  // N.B. std::bitset<> has count() as an element count, while
  // std::set<>::count(x) returns 0 or 1 to indicate presence.
  constexpr std::size_t count() const { return bitset_.count(); }
  constexpr std::size_t count(enumerationType x) const {
    return test(x) ? 1 : 0;
  }

  constexpr EnumSet &set() {
    bitset_.set();
    return *this;
  }
  constexpr EnumSet &set(enumerationType x, bool value = true) {
    bitset_.set(static_cast<std::size_t>(x), value);
    return *this;
  }
  constexpr EnumSet &reset() {
    bitset_.reset();
    return *this;
  }
  constexpr EnumSet &reset(enumerationType x) {
    bitset_.reset(static_cast<std::size_t>(x));
    return *this;
  }
  constexpr EnumSet &flip() {
    bitset_.flip();
    return *this;
  }
  constexpr EnumSet &flip(enumerationType x) {
    bitset_.flip(static_cast<std::size_t>(x));
    return *this;
  }

  constexpr bool empty() const { return none(); }
  void clear() { reset(); }
  void insert(enumerationType x) { set(x); }
  void insert(enumerationType &&x) { set(x); }
  void emplace(enumerationType &&x) { set(x); }
  void erase(enumerationType x) { reset(x); }
  void erase(enumerationType &&x) { reset(x); }

  constexpr std::optional<enumerationType> LeastElement() const {
    if (empty()) {
      return std::nullopt;
    } else if constexpr (std::is_same_v<bitsetType, common::BitSet<BITS>>) {
      return {static_cast<enumerationType>(bitset_.LeastElement().value())};
    } else {
      // std::bitset: just iterate
      for (std::size_t j{0}; j < BITS; ++j) {
        auto enumerator{static_cast<enumerationType>(j)};
        if (bitset_.test(j)) {
          return {enumerator};
        }
      }
      die("EnumSet::LeastElement(): no bit found in non-empty std::bitset");
    }
  }

  template <typename FUNC> void IterateOverMembers(const FUNC &f) const {
    EnumSet copy{*this};
    while (auto least{copy.LeastElement()}) {
      f(*least);
      copy.erase(*least);
    }
  }

  template <typename STREAM>
  STREAM &Dump(
      STREAM &o, const std::string &EnumToString(enumerationType)) const {
    char sep{'{'};
    IterateOverMembers([&](auto e) {
      o << sep << EnumToString(e);
      sep = ',';
    });
    return o << (sep == '{' ? "{}" : "}");
  }

private:
  bitsetType bitset_{};
};
} // namespace Fortran::common

template <typename ENUM, std::size_t values>
struct std::hash<Fortran::common::EnumSet<ENUM, values>> {
  std::size_t operator()(
      const Fortran::common::EnumSet<ENUM, values> &x) const {
    return std::hash(x.bitset());
  }
};
#endif // FORTRAN_COMMON_ENUM_SET_H_
