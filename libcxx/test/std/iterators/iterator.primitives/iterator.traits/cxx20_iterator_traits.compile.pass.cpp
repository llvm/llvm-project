//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// struct iterator_traits;

#include <iterator>

#include <array>
#include <concepts>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef _LIBCPP_HAS_NO_LOCALIZATION
# include <regex>
# include <ostream>
# include <istream>
#endif

#ifndef _LIBCPP_HAS_NO_FILESYSTEM_LIBRARY
# include <filesystem>
#endif

#include "test_macros.h"
#include "test_iterators.h"
#include "iterator_traits_cpp17_iterators.h"

template <class Traits>
constexpr bool has_iterator_concept_v = requires {
  typename Traits::iterator_concept;
};

template <class Iter, class Category>
constexpr bool testIOIterator() {
  using Traits = std::iterator_traits<Iter>;
  static_assert(std::same_as<typename Traits::iterator_category, Category>);
  static_assert(std::same_as<typename Traits::value_type, void>);
  static_assert(std::same_as<typename Traits::difference_type, std::ptrdiff_t>);
  static_assert(std::same_as<typename Traits::reference, void>);
  static_assert(std::same_as<typename Traits::pointer, void>);
  static_assert(!has_iterator_concept_v<Traits>);

  return true;
}

template <class Iter, class ValueType, class Category>
constexpr bool testConstWithoutConcept() {
  using Traits = std::iterator_traits<Iter>;
  static_assert(std::same_as<typename Traits::iterator_category, Category>);
  static_assert(std::same_as<typename Traits::value_type, ValueType>);
  static_assert(std::same_as<typename Traits::difference_type, std::ptrdiff_t>);
  static_assert(std::same_as<typename Traits::reference, const ValueType&>);
  static_assert(std::same_as<typename Traits::pointer, const ValueType*>);
  static_assert(!has_iterator_concept_v<Traits>);

  return true;
}

template <class Iter, class ValueType, class Category, class IterConcept>
constexpr bool testConstWithConcept() {
  using Traits = std::iterator_traits<Iter>;
  static_assert(std::same_as<typename Traits::iterator_category, Category>);
  static_assert(std::same_as<typename Traits::value_type, ValueType>);
  static_assert(std::same_as<typename Traits::difference_type, std::ptrdiff_t>);
  static_assert(std::same_as<typename Traits::reference, const ValueType&>);
  static_assert(std::same_as<typename Traits::pointer, const ValueType*>);
  static_assert(std::same_as<typename Traits::iterator_concept, IterConcept>);

  return true;
}

template <class Iter, class ValueType, class Category>
constexpr bool testWithoutConcept() {
  using Traits = std::iterator_traits<Iter>;
  static_assert(std::same_as<typename Traits::iterator_category, Category>);
  static_assert(std::same_as<typename Traits::value_type, ValueType>);
  static_assert(std::same_as<typename Traits::difference_type, std::ptrdiff_t>);
  static_assert(std::same_as<typename Traits::reference, ValueType&>);
  static_assert(std::same_as<typename Traits::pointer, ValueType*>);
  static_assert(!has_iterator_concept_v<Traits>);

  return true;
}

template <class Iter, class ValueType, class Category, class IterConcept>
constexpr bool testWithConcept() {
  using Traits = std::iterator_traits<Iter>;
  static_assert(std::same_as<typename Traits::iterator_category, Category>);
  static_assert(std::same_as<typename Traits::value_type, ValueType>);
  static_assert(std::same_as<typename Traits::difference_type, std::ptrdiff_t>);
  static_assert(std::same_as<typename Traits::reference, ValueType&>);
  static_assert(std::same_as<typename Traits::pointer, ValueType*>);
  static_assert(std::same_as<typename Traits::iterator_concept, IterConcept>);

  return true;
}

template <class Iter, class ValueType, class DiffType, class RefType, class PtrType, class Category>
constexpr bool testWithoutConcept() {
  using Traits = std::iterator_traits<Iter>;
  static_assert(std::same_as<typename Traits::iterator_category, Category>);
  static_assert(std::same_as<typename Traits::value_type, ValueType>);
  static_assert(std::same_as<typename Traits::difference_type, DiffType>);
  static_assert(std::same_as<typename Traits::reference, RefType>);
  static_assert(std::same_as<typename Traits::pointer, PtrType>);
  static_assert(!has_iterator_concept_v<Traits>);

  return true;
}

template <class Iter, class ValueType, class DiffType, class RefType, class PtrType, class Category, class IterConcept>
constexpr bool testWithConcept() {
  using Traits = std::iterator_traits<Iter>;
  static_assert(std::same_as<typename Traits::iterator_category, Category>);
  static_assert(std::same_as<typename Traits::value_type, ValueType>);
  static_assert(std::same_as<typename Traits::difference_type, DiffType>);
  static_assert(std::same_as<typename Traits::reference, RefType>);
  static_assert(std::same_as<typename Traits::pointer, PtrType>);
  static_assert(std::same_as<typename Traits::iterator_concept, IterConcept>);

  return true;
}

// Standard types.

// These tests depend on implementation details of libc++,
// e.g. that std::array::iterator is a raw pointer type but std::string::iterator is not.
// The Standard does not specify whether iterator_traits<It>::iterator_concept exists for any particular non-pointer type.
//
static_assert(testWithConcept<std::array<int, 10>::iterator, int, std::random_access_iterator_tag, std::contiguous_iterator_tag>());
static_assert(testConstWithConcept<std::array<int, 10>::const_iterator, int, std::random_access_iterator_tag, std::contiguous_iterator_tag>());
static_assert(testWithoutConcept<std::string::iterator, char, std::random_access_iterator_tag>());
static_assert(testConstWithoutConcept<std::string::const_iterator, char, std::random_access_iterator_tag>());
static_assert(testConstWithConcept<std::string_view::iterator, char, std::random_access_iterator_tag, std::contiguous_iterator_tag>());
static_assert(testConstWithConcept<std::string_view::const_iterator, char, std::random_access_iterator_tag, std::contiguous_iterator_tag>());
static_assert(testWithoutConcept<std::vector<int>::iterator, int, std::random_access_iterator_tag>());
static_assert(testConstWithoutConcept<std::vector<int>::const_iterator, int, std::random_access_iterator_tag>());

static_assert(testWithoutConcept<std::deque<int>::iterator, int, std::random_access_iterator_tag>());
static_assert(testConstWithoutConcept<std::deque<int>::const_iterator, int, std::random_access_iterator_tag>());
static_assert(testWithoutConcept<std::forward_list<int>::iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::forward_list<int>::const_iterator, int, std::forward_iterator_tag>());
static_assert(testWithoutConcept<std::list<int>::iterator, int, std::bidirectional_iterator_tag>());
static_assert(testConstWithoutConcept<std::list<int>::const_iterator, int, std::bidirectional_iterator_tag>());

static_assert(testWithoutConcept<std::map<int, int>::iterator, std::pair<const int, int>, std::bidirectional_iterator_tag>());
static_assert(testConstWithoutConcept<std::map<int, int>::const_iterator, std::pair<const int, int>, std::bidirectional_iterator_tag>());
static_assert(testWithoutConcept<std::multimap<int, int>::iterator, std::pair<const int, int>, std::bidirectional_iterator_tag>());
static_assert(testConstWithoutConcept<std::multimap<int, int>::const_iterator, std::pair<const int, int>, std::bidirectional_iterator_tag>());

static_assert(testConstWithoutConcept<std::set<int>::iterator, int, std::bidirectional_iterator_tag>());
static_assert(testConstWithoutConcept<std::set<int>::const_iterator, int, std::bidirectional_iterator_tag>());
static_assert(testConstWithoutConcept<std::multiset<int>::iterator, int, std::bidirectional_iterator_tag>());
static_assert(testConstWithoutConcept<std::multiset<int>::const_iterator, int, std::bidirectional_iterator_tag>());

static_assert(testWithoutConcept<std::unordered_map<int, int>::iterator, std::pair<const int, int>, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_map<int, int>::const_iterator, std::pair<const int, int>, std::forward_iterator_tag>());
static_assert(testWithoutConcept<std::unordered_map<int, int>::local_iterator, std::pair<const int, int>, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_map<int, int>::const_local_iterator, std::pair<const int, int>, std::forward_iterator_tag>());
static_assert(testWithoutConcept<std::unordered_multimap<int, int>::iterator, std::pair<const int, int>, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_multimap<int, int>::const_iterator, std::pair<const int, int>, std::forward_iterator_tag>());
static_assert(testWithoutConcept<std::unordered_multimap<int, int>::local_iterator, std::pair<const int, int>, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_multimap<int, int>::const_local_iterator, std::pair<const int, int>, std::forward_iterator_tag>());

static_assert(testConstWithoutConcept<std::unordered_set<int>::iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_set<int>::const_iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_set<int>::local_iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_set<int>::const_local_iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_multiset<int>::iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_multiset<int>::const_iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_multiset<int>::local_iterator, int, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::unordered_multiset<int>::const_local_iterator, int, std::forward_iterator_tag>());

static_assert(testWithoutConcept<std::reverse_iterator<int*>, int, std::random_access_iterator_tag>());
static_assert(testIOIterator<std::back_insert_iterator<std::vector<int>>, std::output_iterator_tag>());
static_assert(testIOIterator<std::front_insert_iterator<std::vector<int>>, std::output_iterator_tag>());
static_assert(testIOIterator<std::insert_iterator<std::vector<int>>, std::output_iterator_tag>());
static_assert(testConstWithoutConcept<std::istream_iterator<int, char>, int, std::input_iterator_tag>());

#if !defined(_LIBCPP_HAS_NO_LOCALIZATION)
static_assert(testWithoutConcept<std::istreambuf_iterator<char>, char, long long, char, char*, std::input_iterator_tag>());
static_assert(testWithoutConcept<std::move_iterator<int*>, int, std::ptrdiff_t, int&&, int*, std::random_access_iterator_tag>());
static_assert(testIOIterator<std::ostream_iterator<int, char>, std::output_iterator_tag>());
static_assert(testIOIterator<std::ostreambuf_iterator<int, char>, std::output_iterator_tag>());
static_assert(testConstWithoutConcept<std::cregex_iterator, std::cmatch, std::forward_iterator_tag>());
static_assert(testConstWithoutConcept<std::cregex_token_iterator, std::csub_match, std::forward_iterator_tag>());
#endif // !_LIBCPP_HAS_NO_LOCALIZATION

#ifndef _LIBCPP_HAS_NO_FILESYSTEM_LIBRARY
static_assert(testWithoutConcept<std::filesystem::directory_iterator, std::filesystem::directory_entry, std::ptrdiff_t,
                                 const std::filesystem::directory_entry&, const std::filesystem::directory_entry*,
                                 std::input_iterator_tag>());
static_assert(testWithoutConcept<std::filesystem::recursive_directory_iterator, std::filesystem::directory_entry,
                                 std::ptrdiff_t, const std::filesystem::directory_entry&,
                                 const std::filesystem::directory_entry*, std::input_iterator_tag>());
#endif

// Local test iterators.

struct AllMembers {
  struct iterator_category {};
  struct value_type {};
  struct difference_type {};
  struct reference {};
  struct pointer {};
};
using AllMembersTraits = std::iterator_traits<AllMembers>;
static_assert(std::same_as<AllMembersTraits::iterator_category, AllMembers::iterator_category>);
static_assert(std::same_as<AllMembersTraits::value_type, AllMembers::value_type>);
static_assert(std::same_as<AllMembersTraits::difference_type, AllMembers::difference_type>);
static_assert(std::same_as<AllMembersTraits::reference, AllMembers::reference>);
static_assert(std::same_as<AllMembersTraits::pointer, AllMembers::pointer>);
static_assert(!has_iterator_concept_v<AllMembersTraits>);

struct NoPointerMember {
  struct iterator_category {};
  struct value_type {};
  struct difference_type {};
  struct reference {};
  // ignored, because NoPointerMember is not a LegacyInputIterator:
  value_type* operator->() const;
};
using NoPointerMemberTraits = std::iterator_traits<NoPointerMember>;
static_assert(std::same_as<NoPointerMemberTraits::iterator_category, NoPointerMember::iterator_category>);
static_assert(std::same_as<NoPointerMemberTraits::value_type, NoPointerMember::value_type>);
static_assert(std::same_as<NoPointerMemberTraits::difference_type, NoPointerMember::difference_type>);
static_assert(std::same_as<NoPointerMemberTraits::reference, NoPointerMember::reference>);
static_assert(std::same_as<NoPointerMemberTraits::pointer, void>);
static_assert(!has_iterator_concept_v<NoPointerMemberTraits>);

struct IterConcept {
  struct iterator_category {};
  struct value_type {};
  struct difference_type {};
  struct reference {};
  struct pointer {};
  // iterator_traits does NOT pass through the iterator_concept of the type itself.
  struct iterator_concept {};
};
using IterConceptTraits = std::iterator_traits<IterConcept>;
static_assert(std::same_as<IterConceptTraits::iterator_category, IterConcept::iterator_category>);
static_assert(std::same_as<IterConceptTraits::value_type, IterConcept::value_type>);
static_assert(std::same_as<IterConceptTraits::difference_type, IterConcept::difference_type>);
static_assert(std::same_as<IterConceptTraits::reference, IterConcept::reference>);
static_assert(std::same_as<IterConceptTraits::pointer, IterConcept::pointer>);
static_assert(!has_iterator_concept_v<IterConceptTraits>);

struct LegacyInput {
  struct iterator_category {};
  struct value_type {};
  struct reference { operator value_type() const; };

  friend bool operator==(LegacyInput, LegacyInput);
  reference operator*() const;
  LegacyInput& operator++();
  LegacyInput operator++(int);
};
template <>
struct std::incrementable_traits<LegacyInput> {
  using difference_type = short;
};
using LegacyInputTraits = std::iterator_traits<LegacyInput>;
static_assert(std::same_as<LegacyInputTraits::iterator_category, LegacyInput::iterator_category>);
static_assert(std::same_as<LegacyInputTraits::value_type, LegacyInput::value_type>);
static_assert(std::same_as<LegacyInputTraits::difference_type, short>);
static_assert(std::same_as<LegacyInputTraits::reference, LegacyInput::reference>);
static_assert(std::same_as<LegacyInputTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyInputTraits>);

struct LegacyInputNoValueType {
  struct not_value_type {};
  using difference_type = int; // or any signed integral type
  struct reference { operator not_value_type&() const; };

  friend bool operator==(LegacyInputNoValueType, LegacyInputNoValueType);
  reference operator*() const;
  LegacyInputNoValueType& operator++();
  LegacyInputNoValueType operator++(int);
};
template <>
struct std::indirectly_readable_traits<LegacyInputNoValueType> {
  using value_type = LegacyInputNoValueType::not_value_type;
};
using LegacyInputNoValueTypeTraits = std::iterator_traits<LegacyInputNoValueType>;
static_assert(std::same_as<LegacyInputNoValueTypeTraits::iterator_category, std::input_iterator_tag>);
static_assert(std::same_as<LegacyInputNoValueTypeTraits::value_type, LegacyInputNoValueType::not_value_type>);
static_assert(std::same_as<LegacyInputNoValueTypeTraits::difference_type, int>);
static_assert(std::same_as<LegacyInputNoValueTypeTraits::reference, LegacyInputNoValueType::reference>);
static_assert(std::same_as<LegacyInputNoValueTypeTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyInputNoValueTypeTraits>);

struct LegacyForward {
  struct not_value_type {};

  friend bool operator==(LegacyForward, LegacyForward);
  const not_value_type& operator*() const;
  LegacyForward& operator++();
  LegacyForward operator++(int);
};
template <>
struct std::indirectly_readable_traits<LegacyForward> {
  using value_type = LegacyForward::not_value_type;
};
template <>
struct std::incrementable_traits<LegacyForward> {
  using difference_type = short; // or any signed integral type
};
using LegacyForwardTraits = std::iterator_traits<LegacyForward>;
static_assert(std::same_as<LegacyForwardTraits::iterator_category, std::forward_iterator_tag>);
static_assert(std::same_as<LegacyForwardTraits::value_type, LegacyForward::not_value_type>);
static_assert(std::same_as<LegacyForwardTraits::difference_type, short>);
static_assert(std::same_as<LegacyForwardTraits::reference, const LegacyForward::not_value_type&>);
static_assert(std::same_as<LegacyForwardTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyForwardTraits>);

struct LegacyBidirectional {
  struct value_type {};

  friend bool operator==(LegacyBidirectional, LegacyBidirectional);
  const value_type& operator*() const;
  LegacyBidirectional& operator++();
  LegacyBidirectional operator++(int);
  LegacyBidirectional& operator--();
  LegacyBidirectional operator--(int);
  friend short operator-(LegacyBidirectional, LegacyBidirectional);
};
using LegacyBidirectionalTraits = std::iterator_traits<LegacyBidirectional>;
static_assert(std::same_as<LegacyBidirectionalTraits::iterator_category, std::bidirectional_iterator_tag>);
static_assert(std::same_as<LegacyBidirectionalTraits::value_type, LegacyBidirectional::value_type>);
static_assert(std::same_as<LegacyBidirectionalTraits::difference_type, short>);
static_assert(std::same_as<LegacyBidirectionalTraits::reference, const LegacyBidirectional::value_type&>);
static_assert(std::same_as<LegacyBidirectionalTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyBidirectionalTraits>);

// Almost a random access iterator except it is missing operator-(It, It).
struct MinusNotDeclaredIter {
  struct value_type {};

  friend auto operator<=>(MinusNotDeclaredIter, MinusNotDeclaredIter) = default;
  const value_type& operator*() const;
  const value_type& operator[](long) const;
  MinusNotDeclaredIter& operator++();
  MinusNotDeclaredIter operator++(int);
  MinusNotDeclaredIter& operator--();
  MinusNotDeclaredIter operator--(int);
  MinusNotDeclaredIter& operator+=(long);
  MinusNotDeclaredIter& operator-=(long);

  // Providing difference_type does not fully compensate for missing operator-(It, It).
  friend MinusNotDeclaredIter operator-(MinusNotDeclaredIter, int);
  friend MinusNotDeclaredIter operator+(MinusNotDeclaredIter, int);
  friend MinusNotDeclaredIter operator+(int, MinusNotDeclaredIter);
};
template <>
struct std::incrementable_traits<MinusNotDeclaredIter> {
  using difference_type = short;
};
using MinusNotDeclaredIterTraits = std::iterator_traits<MinusNotDeclaredIter>;
static_assert(std::same_as<MinusNotDeclaredIterTraits::iterator_category, std::bidirectional_iterator_tag>);
static_assert(std::same_as<MinusNotDeclaredIterTraits::value_type, MinusNotDeclaredIter::value_type>);
static_assert(std::same_as<MinusNotDeclaredIterTraits::difference_type, short>);
static_assert(std::same_as<MinusNotDeclaredIterTraits::reference, const MinusNotDeclaredIter::value_type&>);
static_assert(std::same_as<MinusNotDeclaredIterTraits::pointer, void>);
static_assert(!has_iterator_concept_v<MinusNotDeclaredIterTraits>);

struct WrongSubscriptReturnType {
  struct value_type {};

  friend auto operator<=>(WrongSubscriptReturnType, WrongSubscriptReturnType) = default;

  // The type of it[n] is not convertible to the type of *it; therefore, this is not random-access.
  value_type& operator*() const;
  const value_type& operator[](long) const;
  WrongSubscriptReturnType& operator++();
  WrongSubscriptReturnType operator++(int);
  WrongSubscriptReturnType& operator--();
  WrongSubscriptReturnType operator--(int);
  WrongSubscriptReturnType& operator+=(long);
  WrongSubscriptReturnType& operator-=(long);
  friend short operator-(WrongSubscriptReturnType, WrongSubscriptReturnType);
  friend WrongSubscriptReturnType operator-(WrongSubscriptReturnType, int);
  friend WrongSubscriptReturnType operator+(WrongSubscriptReturnType, int);
  friend WrongSubscriptReturnType operator+(int, WrongSubscriptReturnType);
};
using WrongSubscriptReturnTypeTraits = std::iterator_traits<WrongSubscriptReturnType>;
static_assert(std::same_as<WrongSubscriptReturnTypeTraits::iterator_category, std::bidirectional_iterator_tag>);
static_assert(std::same_as<WrongSubscriptReturnTypeTraits::value_type, WrongSubscriptReturnType::value_type>);
static_assert(std::same_as<WrongSubscriptReturnTypeTraits::difference_type, short>);
static_assert(std::same_as<WrongSubscriptReturnTypeTraits::reference, WrongSubscriptReturnType::value_type&>);
static_assert(std::same_as<WrongSubscriptReturnTypeTraits::pointer, void>);
static_assert(!has_iterator_concept_v<WrongSubscriptReturnTypeTraits>);

struct LegacyRandomAccess {
  struct value_type {};

  friend bool operator==(LegacyRandomAccess, LegacyRandomAccess);
  friend bool operator<(LegacyRandomAccess, LegacyRandomAccess);
  friend bool operator<=(LegacyRandomAccess, LegacyRandomAccess);
  friend bool operator>(LegacyRandomAccess, LegacyRandomAccess);
  friend bool operator>=(LegacyRandomAccess, LegacyRandomAccess);
  const value_type& operator*() const;
  const value_type& operator[](long) const;
  LegacyRandomAccess& operator++();
  LegacyRandomAccess operator++(int);
  LegacyRandomAccess& operator--();
  LegacyRandomAccess operator--(int);
  LegacyRandomAccess& operator+=(long);
  LegacyRandomAccess& operator-=(long);
  friend short operator-(LegacyRandomAccess, LegacyRandomAccess);
  friend LegacyRandomAccess operator-(LegacyRandomAccess, int);
  friend LegacyRandomAccess operator+(LegacyRandomAccess, int);
  friend LegacyRandomAccess operator+(int, LegacyRandomAccess);
};
using LegacyRandomAccessTraits = std::iterator_traits<LegacyRandomAccess>;
static_assert(std::same_as<LegacyRandomAccessTraits::iterator_category, std::random_access_iterator_tag>);
static_assert(std::same_as<LegacyRandomAccessTraits::value_type, LegacyRandomAccess::value_type>);
static_assert(std::same_as<LegacyRandomAccessTraits::difference_type, short>);
static_assert(std::same_as<LegacyRandomAccessTraits::reference, const LegacyRandomAccess::value_type&>);
static_assert(std::same_as<LegacyRandomAccessTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyRandomAccessTraits>);

struct LegacyRandomAccessSpaceship {
  struct not_value_type {};
  struct ReferenceConvertible { operator not_value_type&() const; };

  friend auto operator<=>(LegacyRandomAccessSpaceship, LegacyRandomAccessSpaceship) = default;
  not_value_type& operator*() const;
  ReferenceConvertible operator[](long) const;
  LegacyRandomAccessSpaceship& operator++();
  LegacyRandomAccessSpaceship operator++(int);
  LegacyRandomAccessSpaceship& operator--();
  LegacyRandomAccessSpaceship operator--(int);
  LegacyRandomAccessSpaceship& operator+=(long);
  LegacyRandomAccessSpaceship& operator-=(long);
  friend short operator-(LegacyRandomAccessSpaceship, LegacyRandomAccessSpaceship);
  friend LegacyRandomAccessSpaceship operator-(LegacyRandomAccessSpaceship, int);
  friend LegacyRandomAccessSpaceship operator+(LegacyRandomAccessSpaceship, int);
  friend LegacyRandomAccessSpaceship operator+(int, LegacyRandomAccessSpaceship);
};
template <>
struct std::indirectly_readable_traits<LegacyRandomAccessSpaceship> {
  using value_type = LegacyRandomAccessSpaceship::not_value_type;
};
template <>
struct std::incrementable_traits<LegacyRandomAccessSpaceship> {
  using difference_type = short; // or any signed integral type
};
using LegacyRandomAccessSpaceshipTraits = std::iterator_traits<LegacyRandomAccessSpaceship>;
static_assert(std::same_as<LegacyRandomAccessSpaceshipTraits::iterator_category, std::random_access_iterator_tag>);
static_assert(std::same_as<LegacyRandomAccessSpaceshipTraits::value_type, LegacyRandomAccessSpaceship::not_value_type>);
static_assert(std::same_as<LegacyRandomAccessSpaceshipTraits::difference_type, short>);
static_assert(std::same_as<LegacyRandomAccessSpaceshipTraits::reference, LegacyRandomAccessSpaceship::not_value_type&>);
static_assert(std::same_as<LegacyRandomAccessSpaceshipTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyRandomAccessSpaceshipTraits>);

// For output iterators, value_type, difference_type, and reference may be void.
struct BareLegacyOutput {
  struct Empty {};
  Empty operator*() const;
  BareLegacyOutput& operator++();
  BareLegacyOutput operator++(int);
};
using BareLegacyOutputTraits = std::iterator_traits<BareLegacyOutput>;
static_assert(std::same_as<BareLegacyOutputTraits::iterator_category, std::output_iterator_tag>);
static_assert(std::same_as<BareLegacyOutputTraits::value_type, void>);
static_assert(std::same_as<BareLegacyOutputTraits::difference_type, void>);
static_assert(std::same_as<BareLegacyOutputTraits::reference, void>);
static_assert(std::same_as<BareLegacyOutputTraits::pointer, void>);
static_assert(!has_iterator_concept_v<BareLegacyOutputTraits>);

// The operator- means we get difference_type.
struct LegacyOutputWithMinus {
  struct Empty {};
  Empty operator*() const;
  LegacyOutputWithMinus& operator++();
  LegacyOutputWithMinus operator++(int);
  friend short operator-(LegacyOutputWithMinus, LegacyOutputWithMinus);
  // Lacking operator==, this is a LegacyIterator but not a LegacyInputIterator.
};
using LegacyOutputWithMinusTraits = std::iterator_traits<LegacyOutputWithMinus>;
static_assert(std::same_as<LegacyOutputWithMinusTraits::iterator_category, std::output_iterator_tag>);
static_assert(std::same_as<LegacyOutputWithMinusTraits::value_type, void>);
static_assert(std::same_as<LegacyOutputWithMinusTraits::difference_type, short>);
static_assert(std::same_as<LegacyOutputWithMinusTraits::reference, void>);
static_assert(std::same_as<LegacyOutputWithMinusTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyOutputWithMinusTraits>);

struct LegacyOutputWithMemberTypes {
  struct value_type {}; // ignored
  struct reference {};  // ignored
  using difference_type = long;

  friend bool operator==(LegacyOutputWithMemberTypes, LegacyOutputWithMemberTypes);
  reference operator*() const;
  LegacyOutputWithMemberTypes& operator++();
  LegacyOutputWithMemberTypes operator++(int);
  friend short operator-(LegacyOutputWithMemberTypes, LegacyOutputWithMemberTypes); // ignored
  // Since (*it) is not convertible to value_type, this is not a LegacyInputIterator.
};
using LegacyOutputWithMemberTypesTraits = std::iterator_traits<LegacyOutputWithMemberTypes>;
static_assert(std::same_as<LegacyOutputWithMemberTypesTraits::iterator_category, std::output_iterator_tag>);
static_assert(std::same_as<LegacyOutputWithMemberTypesTraits::value_type, void>);
static_assert(std::same_as<LegacyOutputWithMemberTypesTraits::difference_type, long>);
static_assert(std::same_as<LegacyOutputWithMemberTypesTraits::reference, void>);
static_assert(std::same_as<LegacyOutputWithMemberTypesTraits::pointer, void>);
static_assert(!has_iterator_concept_v<LegacyOutputWithMemberTypesTraits>);

struct LegacyRandomAccessSpecialized {
  struct not_value_type {};

  friend auto operator<=>(LegacyRandomAccessSpecialized, LegacyRandomAccessSpecialized) = default;
  not_value_type& operator*() const;
  not_value_type& operator[](long) const;
  LegacyRandomAccessSpecialized& operator++();
  LegacyRandomAccessSpecialized operator++(int);
  LegacyRandomAccessSpecialized& operator--();
  LegacyRandomAccessSpecialized operator--(int);
  LegacyRandomAccessSpecialized& operator+=(long);
  LegacyRandomAccessSpecialized& operator-=(long);
  friend long operator-(LegacyRandomAccessSpecialized, LegacyRandomAccessSpecialized);
  friend LegacyRandomAccessSpecialized operator-(LegacyRandomAccessSpecialized, int);
  friend LegacyRandomAccessSpecialized operator+(LegacyRandomAccessSpecialized, int);
  friend LegacyRandomAccessSpecialized operator+(int, LegacyRandomAccessSpecialized);
};
template <class I>
  requires std::same_as<I, LegacyRandomAccessSpecialized>
struct std::iterator_traits<I>
{
  using iterator_category = std::output_iterator_tag;
  using value_type = short;
  using difference_type = short;
  using reference = short&;
  using pointer = short*;
};
using LegacyRandomAccessSpecializedTraits = std::iterator_traits<LegacyRandomAccessSpecialized>;
static_assert(std::same_as<LegacyRandomAccessSpecializedTraits::iterator_category, std::output_iterator_tag>);
static_assert(std::same_as<LegacyRandomAccessSpecializedTraits::value_type, short>);
static_assert(std::same_as<LegacyRandomAccessSpecializedTraits::difference_type, short>);
static_assert(std::same_as<LegacyRandomAccessSpecializedTraits::reference, short&>);
static_assert(std::same_as<LegacyRandomAccessSpecializedTraits::pointer, short*>);
static_assert(!has_iterator_concept_v<LegacyRandomAccessSpecializedTraits>);

// Other test iterators.

using InputTestIteratorTraits = std::iterator_traits<cpp17_input_iterator<int*>>;
static_assert(std::same_as<InputTestIteratorTraits::iterator_category, std::input_iterator_tag>);
static_assert(std::same_as<InputTestIteratorTraits::value_type, int>);
static_assert(std::same_as<InputTestIteratorTraits::difference_type, std::ptrdiff_t>);
static_assert(std::same_as<InputTestIteratorTraits::reference, int&>);
static_assert(std::same_as<InputTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<InputTestIteratorTraits>);

using OutputTestIteratorTraits = std::iterator_traits<output_iterator<int*>>;
static_assert(std::same_as<OutputTestIteratorTraits::iterator_category, std::output_iterator_tag>);
static_assert(std::same_as<OutputTestIteratorTraits::value_type, void>);
static_assert(std::same_as<OutputTestIteratorTraits::difference_type, std::ptrdiff_t>);
static_assert(std::same_as<OutputTestIteratorTraits::reference, int&>);
static_assert(std::same_as<OutputTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<OutputTestIteratorTraits>);

using ForwardTestIteratorTraits = std::iterator_traits<forward_iterator<int*>>;
static_assert(std::same_as<ForwardTestIteratorTraits::iterator_category, std::forward_iterator_tag>);
static_assert(std::same_as<ForwardTestIteratorTraits::value_type, int>);
static_assert(std::same_as<ForwardTestIteratorTraits::difference_type, std::ptrdiff_t>);
static_assert(std::same_as<ForwardTestIteratorTraits::reference, int&>);
static_assert(std::same_as<ForwardTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<ForwardTestIteratorTraits>);

using BidirectionalTestIteratorTraits = std::iterator_traits<bidirectional_iterator<int*>>;
static_assert(std::same_as<BidirectionalTestIteratorTraits::iterator_category, std::bidirectional_iterator_tag>);
static_assert(std::same_as<BidirectionalTestIteratorTraits::value_type, int>);
static_assert(std::same_as<BidirectionalTestIteratorTraits::difference_type, std::ptrdiff_t>);
static_assert(std::same_as<BidirectionalTestIteratorTraits::reference, int&>);
static_assert(std::same_as<BidirectionalTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<BidirectionalTestIteratorTraits>);

using RandomAccessTestIteratorTraits = std::iterator_traits<random_access_iterator<int*>>;
static_assert(std::same_as<RandomAccessTestIteratorTraits::iterator_category, std::random_access_iterator_tag>);
static_assert(std::same_as<RandomAccessTestIteratorTraits::value_type, int>);
static_assert(std::same_as<RandomAccessTestIteratorTraits::difference_type, std::ptrdiff_t>);
static_assert(std::same_as<RandomAccessTestIteratorTraits::reference, int&>);
static_assert(std::same_as<RandomAccessTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<RandomAccessTestIteratorTraits>);

using ContiguousTestIteratorTraits = std::iterator_traits<contiguous_iterator<int*>>;
static_assert(std::same_as<ContiguousTestIteratorTraits::iterator_category, std::contiguous_iterator_tag>);
static_assert(std::same_as<ContiguousTestIteratorTraits::value_type, int>);
static_assert(std::same_as<ContiguousTestIteratorTraits::difference_type, std::ptrdiff_t>);
static_assert(std::same_as<ContiguousTestIteratorTraits::reference, int&>);
static_assert(std::same_as<ContiguousTestIteratorTraits::pointer, int*>);
static_assert(!has_iterator_concept_v<ContiguousTestIteratorTraits>);

using Cpp17BasicIteratorTraits = std::iterator_traits<iterator_traits_cpp17_iterator>;
static_assert(std::same_as<Cpp17BasicIteratorTraits::iterator_category, std::output_iterator_tag>);
static_assert(std::same_as<Cpp17BasicIteratorTraits::value_type, void>);
static_assert(std::same_as<Cpp17BasicIteratorTraits::difference_type, void>);
static_assert(std::same_as<Cpp17BasicIteratorTraits::reference, void>);
static_assert(std::same_as<Cpp17BasicIteratorTraits::pointer, void>);
static_assert(!has_iterator_concept_v<Cpp17BasicIteratorTraits>);

using Cpp17InputIteratorTraits = std::iterator_traits<iterator_traits_cpp17_input_iterator>;
static_assert(std::same_as<Cpp17InputIteratorTraits::iterator_category, std::input_iterator_tag>);
static_assert(std::same_as<Cpp17InputIteratorTraits::value_type, long>);
static_assert(std::same_as<Cpp17InputIteratorTraits::difference_type, int>);
static_assert(std::same_as<Cpp17InputIteratorTraits::reference, int&>);
static_assert(std::same_as<Cpp17InputIteratorTraits::pointer, void>);
static_assert(!has_iterator_concept_v<Cpp17InputIteratorTraits>);

using Cpp17ForwardIteratorTraits = std::iterator_traits<iterator_traits_cpp17_forward_iterator>;
static_assert(std::same_as<Cpp17ForwardIteratorTraits::iterator_category, std::forward_iterator_tag>);
static_assert(std::same_as<Cpp17ForwardIteratorTraits::value_type, int>);
static_assert(std::same_as<Cpp17ForwardIteratorTraits::difference_type, int>);
static_assert(std::same_as<Cpp17ForwardIteratorTraits::reference, int&>);
static_assert(std::same_as<Cpp17ForwardIteratorTraits::pointer, void>);
static_assert(!has_iterator_concept_v<Cpp17ForwardIteratorTraits>);
