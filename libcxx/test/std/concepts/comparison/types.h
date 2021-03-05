//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef TEST_STD_CONCEPTS_COMPARISON_EQUALITYCOMPARABLE_H
#define TEST_STD_CONCEPTS_COMPARISON_EQUALITYCOMPARABLE_H

#include <compare>
#include <concepts>
#include <type_traits>

// `noexcept` specifiers deliberately imperfect since not all programmers bother to put the
// specifiers on their overloads.

struct equality_comparable_with_ec1;
struct no_neq;

struct cxx20_member_eq {
  bool operator==(cxx20_member_eq const&) const = default;
};

struct cxx20_friend_eq {
  friend bool operator==(cxx20_friend_eq const&,
                         cxx20_friend_eq const&) = default;
};

struct member_three_way_comparable {
  auto operator<=>(member_three_way_comparable const&) const = default;
};

struct friend_three_way_comparable {
  friend auto operator<=>(friend_three_way_comparable const&,
                          friend_three_way_comparable const&) = default;
};

struct explicit_operators {
  friend bool operator==(explicit_operators, explicit_operators) noexcept;
  friend bool operator!=(explicit_operators, explicit_operators) noexcept;

  friend bool operator==(explicit_operators const&,
                         equality_comparable_with_ec1 const&) noexcept;
  friend bool operator==(equality_comparable_with_ec1 const&,
                         explicit_operators const&) noexcept;
  friend bool operator!=(explicit_operators const&,
                         equality_comparable_with_ec1 const&) noexcept;
  friend bool operator!=(equality_comparable_with_ec1 const&,
                         explicit_operators const&) noexcept;
};

struct eq_neq_different_return_types {
  int operator==(eq_neq_different_return_types) const noexcept;
  friend long operator!=(eq_neq_different_return_types,
                         eq_neq_different_return_types) noexcept;

  friend int operator==(explicit_operators, eq_neq_different_return_types);
  friend int operator==(eq_neq_different_return_types, explicit_operators);
  friend long operator!=(explicit_operators, eq_neq_different_return_types);
  friend long operator!=(eq_neq_different_return_types, explicit_operators);

  operator explicit_operators() const;
};

struct boolean {
  operator bool() const noexcept;
};

struct one_member_one_friend {
  friend boolean operator==(one_member_one_friend,
                            one_member_one_friend) noexcept;
  boolean operator!=(one_member_one_friend) const noexcept;

  operator explicit_operators() const noexcept;
  operator eq_neq_different_return_types() const noexcept;
};

struct equality_comparable_with_ec1 {
  bool operator==(equality_comparable_with_ec1) const noexcept;
  bool operator!=(equality_comparable_with_ec1) const noexcept;
  operator explicit_operators() const noexcept;
};

struct no_eq {
  friend bool operator!=(no_eq, no_eq) noexcept;
};

struct no_neq {
  friend bool operator==(no_neq, no_neq) noexcept;
  friend bool operator!=(no_neq, no_neq) = delete;
};

struct wrong_return_type_eq {
  void operator==(wrong_return_type_eq) const noexcept;
  bool operator!=(wrong_return_type_eq) const noexcept;
};

struct wrong_return_type_ne {
  bool operator==(wrong_return_type_ne) const noexcept;
  void operator!=(wrong_return_type_ne) const noexcept;
};

struct wrong_return_type {
  void operator==(wrong_return_type) const noexcept;
  void operator!=(wrong_return_type) const noexcept;
};

struct cxx20_member_eq_operator_with_deleted_ne {
  bool
  operator==(cxx20_member_eq_operator_with_deleted_ne const&) const = default;
  bool
  operator!=(cxx20_member_eq_operator_with_deleted_ne const&) const = delete;
};

struct cxx20_friend_eq_operator_with_deleted_ne {
  friend bool
  operator==(cxx20_friend_eq_operator_with_deleted_ne const&,
             cxx20_friend_eq_operator_with_deleted_ne const&) = default;
  friend bool
  operator!=(cxx20_friend_eq_operator_with_deleted_ne const&,
             cxx20_friend_eq_operator_with_deleted_ne const&) = delete;
};

struct member_three_way_comparable_with_deleted_eq {
  auto operator<=>(member_three_way_comparable_with_deleted_eq const&) const =
      default;
  bool
  operator==(member_three_way_comparable_with_deleted_eq const&) const = delete;
};

struct member_three_way_comparable_with_deleted_ne {
  auto operator<=>(member_three_way_comparable_with_deleted_ne const&) const =
      default;
  bool
  operator!=(member_three_way_comparable_with_deleted_ne const&) const = delete;
};

struct friend_three_way_comparable_with_deleted_eq {
  friend auto
  operator<=>(friend_three_way_comparable_with_deleted_eq const&,
              friend_three_way_comparable_with_deleted_eq const&) = default;
  friend bool
  operator==(friend_three_way_comparable_with_deleted_eq const&,
             friend_three_way_comparable_with_deleted_eq const&) = delete;
};

struct friend_three_way_comparable_with_deleted_ne {
  friend auto
  operator<=>(friend_three_way_comparable_with_deleted_ne const&,
              friend_three_way_comparable_with_deleted_ne const&) = default;
  friend bool
  operator!=(friend_three_way_comparable_with_deleted_ne const&,
             friend_three_way_comparable_with_deleted_ne const&) = delete;
};

struct one_way_eq {
  bool operator==(one_way_eq const&) const = default;
  friend bool operator==(one_way_eq, explicit_operators);
  friend bool operator==(explicit_operators, one_way_eq) = delete;

  operator explicit_operators() const;
};

struct one_way_ne {
  bool operator==(one_way_ne const&) const = default;
  friend bool operator==(one_way_ne, explicit_operators);
  friend bool operator!=(one_way_ne, explicit_operators) = delete;

  operator explicit_operators() const;
};
static_assert(requires(explicit_operators const x, one_way_ne const y) {
  x != y;
});

struct explicit_bool {
  explicit operator bool() const noexcept;
};

struct returns_explicit_bool {
  friend explicit_bool operator==(returns_explicit_bool, returns_explicit_bool);
  friend explicit_bool operator!=(returns_explicit_bool, returns_explicit_bool);
};

struct returns_true_type {
  friend std::true_type operator==(returns_true_type, returns_true_type);
  friend std::true_type operator!=(returns_true_type, returns_true_type);
};

struct returns_false_type {
  friend std::false_type operator==(returns_false_type, returns_false_type);
  friend std::false_type operator!=(returns_false_type, returns_false_type);
};

struct returns_int_ptr {
  friend int* operator==(returns_int_ptr, returns_int_ptr);
  friend int* operator!=(returns_int_ptr, returns_int_ptr);
};

#endif // TEST_STD_CONCEPTS_COMPARISON_EQUALITYCOMPARABLE_H
