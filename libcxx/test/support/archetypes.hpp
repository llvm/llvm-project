#ifndef TEST_SUPPORT_ARCHETYPES_HPP
#define TEST_SUPPORT_ARCHETYPES_HPP

#include "test_macros.h"

#if TEST_STD_VER >= 11

struct NoDefault {
    NoDefault() = delete;
};

// Implicit copy/move types

struct AllCtors {
  AllCtors() = default;
  AllCtors(AllCtors const&) = default;
  AllCtors(AllCtors &&) = default;
  AllCtors& operator=(AllCtors const&) = default;
  AllCtors& operator=(AllCtors &&) = default;
};

struct Copyable {
  Copyable() = default;
  Copyable(Copyable const &) = default;
  Copyable &operator=(Copyable const &) = default;
};

struct CopyOnly {
  CopyOnly() = default;
  CopyOnly(CopyOnly const &) = default;
  CopyOnly &operator=(CopyOnly const &) = default;
  CopyOnly(CopyOnly &&) = delete;
  CopyOnly &operator=(CopyOnly &&) = delete;
};

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(NonCopyable const &) = delete;
  NonCopyable &operator=(NonCopyable const &) = delete;
};

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(MoveOnly &&) = default;
  MoveOnly &operator=(MoveOnly &&) = default;
};

struct ConvertingType {
  ConvertingType() = default;
  ConvertingType(ConvertingType const&) = default;
  ConvertingType(ConvertingType &&) = default;
  ConvertingType& operator=(ConvertingType const&) = default;
  ConvertingType& operator=(ConvertingType &&) = default;
  template <class ...Args>
  ConvertingType(Args&&...) {}
  template <class Arg>
  ConvertingType& operator=(Arg&&) { return *this; }
};

struct ExplicitConvertingType {
  ExplicitConvertingType() = default;
  explicit ExplicitConvertingType(ExplicitConvertingType const&) = default;
  explicit ExplicitConvertingType(ExplicitConvertingType &&) = default;
  ExplicitConvertingType& operator=(ExplicitConvertingType const&) = default;
  ExplicitConvertingType& operator=(ExplicitConvertingType &&) = default;
  template <class ...Args>
  explicit ExplicitConvertingType(Args&&...) {}
  template <class Arg>
  ExplicitConvertingType& operator=(Arg&&) { return *this; }
};

// Explicit copy/move types

struct ExplicitAllCtors {
  explicit ExplicitAllCtors() = default;
  explicit ExplicitAllCtors(ExplicitAllCtors const&) = default;
  explicit ExplicitAllCtors(ExplicitAllCtors &&) = default;
  ExplicitAllCtors& operator=(ExplicitAllCtors const&) = default;
  ExplicitAllCtors& operator=(ExplicitAllCtors &&) = default;
};

struct ExplicitCopyable {
  explicit ExplicitCopyable() = default;
  explicit ExplicitCopyable(ExplicitCopyable const &) = default;
  ExplicitCopyable &operator=(ExplicitCopyable const &) = default;
};

struct ExplicitCopyOnly {
  explicit ExplicitCopyOnly() = default;
  explicit ExplicitCopyOnly(ExplicitCopyOnly const &) = default;
  ExplicitCopyOnly &operator=(ExplicitCopyOnly const &) = default;
  explicit ExplicitCopyOnly(ExplicitCopyOnly &&) = delete;
  ExplicitCopyOnly &operator=(ExplicitCopyOnly &&) = delete;
};

struct ExplicitNonCopyable {
  explicit ExplicitNonCopyable() = default;
  explicit ExplicitNonCopyable(ExplicitNonCopyable const &) = delete;
  ExplicitNonCopyable &operator=(ExplicitNonCopyable const &) = delete;
};

struct ExplicitMoveOnly {
  explicit ExplicitMoveOnly() = default;
  explicit ExplicitMoveOnly(ExplicitMoveOnly &&) = default;
  ExplicitMoveOnly &operator=(ExplicitMoveOnly &&) = default;
};

#endif // TEST_STD_VER >= 11

#endif // TEST_SUPPORT_ARCHETYPES_HPP
