//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUTOUTPUT_SPANSTREAMS_TYPES_H
#define TEST_STD_INPUTOUTPUT_SPANSTREAMS_TYPES_H

#include <concepts>
#include <cstddef>
#include <span>
#include <ranges>

#include "test_macros.h"

#include <print>

template <typename CharT, std::size_t N = 0>
class ReadOnlySpan {
public:
  explicit ReadOnlySpan(CharT (&arr)[N]) : arr_{arr} {}

  operator std::span<CharT>() = delete;

  operator std::span<const CharT>() {
    std::println(stderr, "----> ROspan");
    return std::span<const CharT, N>{arr_};
  }

  const CharT* begin() {
    std::println(stderr, "----> ROspan begin");
    return arr_;
  }
  const CharT* end() {
    std::println(stderr, "----> ROspan end");
    return arr_ + N;
  }

private:
  CharT* arr_;
};

template <typename CharT, std::size_t N>
inline constexpr bool std::ranges::enable_borrowed_range<ReadOnlySpan<CharT, N>> = true;

static_assert(std::ranges::borrowed_range<ReadOnlySpan<char>>);

static_assert(!std::constructible_from<std::span<char>, ReadOnlySpan<char>>);
static_assert(!std::convertible_to<ReadOnlySpan<char>, std::span<char>>);

static_assert(!std::constructible_from<std::span<char>, const ReadOnlySpan<char>>);
static_assert(!std::convertible_to<const ReadOnlySpan<char>, std::span<char>>);

static_assert(std::constructible_from<std::span<const char>, ReadOnlySpan<char>>);
static_assert(std::convertible_to<ReadOnlySpan<char>, std::span<const char>>);

static_assert(!std::constructible_from<std::span<const char>, const ReadOnlySpan<char>>);
static_assert(!std::convertible_to<const ReadOnlySpan<char>, std::span<const char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS

static_assert(std::ranges::borrowed_range<ReadOnlySpan<wchar_t>>);

static_assert(!std::constructible_from<std::span<wchar_t>, ReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<ReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(!std::constructible_from<std::span<wchar_t>, const ReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<const ReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(std::constructible_from<std::span<const wchar_t>, ReadOnlySpan<wchar_t>>);
static_assert(std::convertible_to<ReadOnlySpan<wchar_t>, std::span<const wchar_t>>);

static_assert(!std::constructible_from<std::span<const wchar_t>, const ReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<const ReadOnlySpan<wchar_t>, std::span<const wchar_t>>);
#endif

template <typename CharT, std::size_t N = 0>
class NonReadOnlySpan {
public:
  explicit NonReadOnlySpan(CharT (&arr)[N]) : arr_{arr} {}

  operator std::span<CharT>() { return std::span<CharT, N>{arr_}; }

  operator std::span<const CharT>() = delete;

  CharT* begin() { return arr_; }
  CharT* end() { return arr_ + N; }

private:
  CharT* arr_;
};

template <typename CharT, std::size_t N>
inline constexpr bool std::ranges::enable_borrowed_range<NonReadOnlySpan<CharT, N>> = true;

static_assert(std::ranges::borrowed_range<NonReadOnlySpan<char>>);

static_assert(std::constructible_from<std::span<char>, NonReadOnlySpan<char>>);
static_assert(std::convertible_to<NonReadOnlySpan<char>, std::span<char>>);

static_assert(!std::constructible_from<std::span<char>, const NonReadOnlySpan<char>>);
static_assert(!std::convertible_to<const NonReadOnlySpan<char>, std::span<char>>);

static_assert(std::constructible_from<std::span<const char>, NonReadOnlySpan<char>>);
static_assert(!std::convertible_to<NonReadOnlySpan<char>, std::span<const char>>);

static_assert(!std::constructible_from<std::span<const char>, const NonReadOnlySpan<char>>);
static_assert(!std::convertible_to<const NonReadOnlySpan<char>, std::span<const char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::ranges::borrowed_range<NonReadOnlySpan<wchar_t>>);

static_assert(std::constructible_from<std::span<wchar_t>, NonReadOnlySpan<wchar_t>>);
static_assert(std::convertible_to<NonReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(!std::constructible_from<std::span<wchar_t>, const NonReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<const NonReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(std::constructible_from<std::span<const wchar_t>, NonReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<NonReadOnlySpan<wchar_t>, std::span<const wchar_t>>);

static_assert(!std::constructible_from<std::span<const wchar_t>, const NonReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<const NonReadOnlySpan<wchar_t>, std::span<const wchar_t>>);
#endif

template <typename CharT, std::size_t N = 0>
class ConstConvertibleReadOnlySpan {
  explicit ConstConvertibleReadOnlySpan(CharT (&arr)[N]) : arr_{arr} {}

public:
  operator std::span<CharT>()       = delete;
  operator std::span<CharT>() const = delete;

  operator std::span<const CharT>() = delete;
  operator std::span<const CharT>() const { return std::span<const CharT, N>{arr_}; }

  const CharT* begin() { return arr_; }
  const CharT* end() { return arr_ + N; }

private:
  CharT* arr_;
};

template <typename CharT, std::size_t N>
inline constexpr bool std::ranges::enable_borrowed_range<ConstConvertibleReadOnlySpan<CharT, N>> = true;

static_assert(std::ranges::borrowed_range<ConstConvertibleReadOnlySpan<char>>);

static_assert(!std::constructible_from<std::span<char>, ConstConvertibleReadOnlySpan<char>>);
static_assert(!std::convertible_to<ConstConvertibleReadOnlySpan<char>, std::span<char>>);

static_assert(!std::constructible_from<std::span<char>, const ConstConvertibleReadOnlySpan<char>>);
static_assert(!std::convertible_to<const ConstConvertibleReadOnlySpan<char>, std::span<char>>);

static_assert(std::constructible_from<std::span<const char>, ConstConvertibleReadOnlySpan<char>>);
static_assert(!std::convertible_to<ConstConvertibleReadOnlySpan<char>, std::span<const char>>);

static_assert(std::constructible_from<std::span<const char>, const ConstConvertibleReadOnlySpan<char>>);
static_assert(std::convertible_to<const ConstConvertibleReadOnlySpan<char>, std::span<const char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS

static_assert(std::ranges::borrowed_range<ConstConvertibleReadOnlySpan<wchar_t>>);

static_assert(!std::constructible_from<std::span<wchar_t>, ConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<ConstConvertibleReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(!std::constructible_from<std::span<wchar_t>, const ConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<const ConstConvertibleReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(std::constructible_from<std::span<const wchar_t>, ConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<ConstConvertibleReadOnlySpan<wchar_t>, std::span<const wchar_t>>);

static_assert(std::constructible_from<std::span<const wchar_t>, const ConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(std::convertible_to<const ConstConvertibleReadOnlySpan<wchar_t>, std::span<const wchar_t>>);
#endif

template <typename CharT, std::size_t N = 0>
class NonConstConvertibleReadOnlySpan {
  explicit NonConstConvertibleReadOnlySpan(CharT (&arr)[N]) : arr_{arr} {}

public:
  operator std::span<CharT>()       = delete;
  operator std::span<CharT>() const = delete;

  operator std::span<const CharT>() { return std::span<const CharT, N>{arr_}; }
  operator std::span<const CharT>() const = delete;

  const CharT* begin() { return arr_; }
  const CharT* end() { return arr_ + N; }

private:
  CharT* arr_;
};

template <typename CharT, std::size_t N>
inline constexpr bool std::ranges::enable_borrowed_range<NonConstConvertibleReadOnlySpan<CharT, N>> = true;

static_assert(std::ranges::borrowed_range<NonConstConvertibleReadOnlySpan<char>>);

static_assert(!std::constructible_from<std::span<char>, NonConstConvertibleReadOnlySpan<char>>);
static_assert(!std::convertible_to<NonConstConvertibleReadOnlySpan<char>, std::span<char>>);

static_assert(!std::constructible_from<std::span<char>, const NonConstConvertibleReadOnlySpan<char>>);
static_assert(!std::convertible_to<const NonConstConvertibleReadOnlySpan<char>, std::span<char>>);

static_assert(std::constructible_from<std::span<const char>, NonConstConvertibleReadOnlySpan<char>>);
static_assert(std::convertible_to<NonConstConvertibleReadOnlySpan<char>, std::span<const char>>);

static_assert(!std::constructible_from<std::span<const char>, const NonConstConvertibleReadOnlySpan<char>>);
static_assert(!std::convertible_to<const NonConstConvertibleReadOnlySpan<char>, std::span<const char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::ranges::borrowed_range<NonConstConvertibleReadOnlySpan<wchar_t>>);

static_assert(!std::constructible_from<std::span<wchar_t>, NonConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<NonConstConvertibleReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(!std::constructible_from<std::span<wchar_t>, const NonConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<const NonConstConvertibleReadOnlySpan<wchar_t>, std::span<wchar_t>>);

static_assert(std::constructible_from<std::span<const wchar_t>, NonConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(std::convertible_to<NonConstConvertibleReadOnlySpan<wchar_t>, std::span<const wchar_t>>);

static_assert(!std::constructible_from<std::span<const wchar_t>, const NonConstConvertibleReadOnlySpan<wchar_t>>);
static_assert(!std::convertible_to<const NonConstConvertibleReadOnlySpan<wchar_t>, std::span<const wchar_t>>);
#endif

struct SomeObject {};

struct NonMode {};

#endif // TEST_STD_INPUTOUTPUT_SPANSTREAMS_TYPES_H
