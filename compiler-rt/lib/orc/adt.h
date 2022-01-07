//===----------------------- adt.h - Handy ADTs -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_ADT_H
#define ORC_RT_ADT_H

#include <cstring>
#include <limits>
#include <string>

namespace __orc_rt {

constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

/// A substitute for std::span (and llvm::ArrayRef).
/// FIXME: Remove in favor of std::span once we can use c++20.
template <typename T, std::size_t Extent = dynamic_extent> class span {
public:
  typedef T element_type;
  typedef std::remove_cv<T> value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;

  typedef pointer iterator;

  static constexpr std::size_t extent = Extent;

  constexpr span() noexcept = default;
  constexpr span(T *first, size_type count) noexcept
      : Data(first), Size(count) {}

  template <std::size_t N>
  constexpr span(T (&arr)[N]) noexcept : Data(&arr[0]), Size(N) {}

  constexpr iterator begin() const noexcept { return Data; }
  constexpr iterator end() const noexcept { return Data + Size; }
  constexpr pointer data() const noexcept { return Data; }
  constexpr reference operator[](size_type idx) const { return Data[idx]; }
  constexpr size_type size() const noexcept { return Size; }
  constexpr bool empty() const noexcept { return Size == 0; }

private:
  T *Data = nullptr;
  size_type Size = 0;
};

/// A substitue for std::string_view (and llvm::StringRef).
/// FIXME: Remove in favor of std::string_view once we have c++17.
class string_view {
public:
  typedef char value_type;
  typedef char *pointer;
  typedef const char *const_pointer;
  typedef char &reference;
  typedef const char &const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef const_pointer const_iterator;
  typedef const_iterator iterator;

  constexpr string_view() noexcept = default;
  constexpr string_view(const char *S, size_type Count)
      : Data(S), Size(Count) {}
  string_view(const char *S) : Data(S), Size(strlen(S)) {}

  constexpr const_iterator begin() const noexcept { return Data; }
  constexpr const_iterator end() const noexcept { return Data + Size; }
  constexpr const_pointer data() const noexcept { return Data; }
  constexpr const_reference operator[](size_type idx) { return Data[idx]; }
  constexpr size_type size() const noexcept { return Size; }
  constexpr bool empty() const noexcept { return Size == 0; }

  friend bool operator==(const string_view &LHS, const string_view &RHS) {
    if (LHS.Size != RHS.Size)
      return false;
    if (LHS.Data == RHS.Data)
      return true;
    for (size_t I = 0; I != LHS.Size; ++I)
      if (LHS.Data[I] != RHS.Data[I])
        return false;
    return true;
  }

  friend bool operator!=(const string_view &LHS, const string_view &RHS) {
    return !(LHS == RHS);
  }

private:
  const char *Data = nullptr;
  size_type Size = 0;
};

inline std::string to_string(string_view SV) {
  return std::string(SV.data(), SV.size());
}

} // end namespace __orc_rt

#endif // ORC_RT_ADT_H
