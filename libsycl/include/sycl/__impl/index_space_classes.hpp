//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 ranges and index space
/// identifiers (4.9.1.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INDEX_SPACE_CLASSES_HPP
#define _LIBSYCL___IMPL_INDEX_SPACE_CLASSES_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstddef>
#include <type_traits>
#include <variant>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class Builder;

/// Helper class for dimensions data management.
template <int Dimensions = 1> class RawArray {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "RawArray can only be 1, 2, or 3 Dimensional.");

public:
  /// Constructs a one-dimensional instance and assigns the corresponding data
  /// to Dim0 value. Available only if Dimensions = 1.
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  RawArray(size_t Dim0 = 0) : MArray{Dim0} {}

  /// Constructs a two-dimensional instance and assigns the corresponding data.
  /// Available only if Dimensions = 2.
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  RawArray(size_t Dim0, size_t Dim1) : MArray{Dim0, Dim1} {}

  /// Constructs a two-dimensional instance with the zero-initialized
  /// corresponding data. Available only if Dimensions = 2.
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  RawArray() : RawArray(0, 0) {}

  /// Constructs a three-dimensional instance and assigns the corresponding
  /// data. Available only if Dimensions = 3.
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  RawArray(size_t Dim0, size_t Dim1, size_t Dim2) : MArray{Dim0, Dim1, Dim2} {}

  /// Constructs a three-dimensional instance with the zero-initialized
  /// corresponding data. Available only if Dimensions = 3.
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  RawArray() : RawArray(0, 0, 0) {}

  /// Returns the value for the specified dimension.
  /// Results in undefined behavior if dimension is not in the range [0,
  /// Dimensions).
  /// \param Dimension the dimension to return the value for.
  /// \return the value matching the requested dimension.
  std::size_t get(int Dimension) const noexcept { return MArray[Dimension]; }

  /// Returns the value for the specified dimension.
  /// Results in undefined behavior if dimension is not in the range [0,
  /// Dimensions).
  /// \param Dimension the dimension to return the value for.
  /// \return the value matching the requested dimension.
  std::size_t &operator[](int Dimension) noexcept { return MArray[Dimension]; }

  /// Returns the value for the specified dimension.
  /// Results in undefined behavior if dimension is not in the range [0,
  /// Dimensions).
  /// \param Dimension the dimension to return the value for.
  /// \return the value matching the requested dimension.
  std::size_t operator[](int Dimension) const noexcept {
    return MArray[Dimension];
  }

  RawArray(const RawArray<Dimensions> &rhs) = default;
  RawArray(RawArray<Dimensions> &&rhs) = default;
  RawArray<Dimensions> &operator=(const RawArray<Dimensions> &rhs) = default;
  RawArray<Dimensions> &operator=(RawArray<Dimensions> &&rhs) = default;
  ~RawArray() = default;

  friend bool operator==(const RawArray<Dimensions> &lhs,
                         const RawArray<Dimensions> &rhs) {
    for (int i = 0; i < Dimensions; ++i) {
      if (lhs.MArray[i] != rhs.MArray[i]) {
        return false;
      }
    }
    return true;
  }

  friend bool operator!=(const RawArray<Dimensions> &lhs,
                         const RawArray<Dimensions> &rhs) {
    return !(lhs == rhs);
  }

protected:
  size_t MArray[Dimensions];
};
} // namespace detail

/// SYCL 2020 4.9.1.1. range class.
/// range<int Dimensions> is a 1D, 2D or 3D vector that defines the iteration
/// domain of either a single work-group in a parallel dispatch, or the overall
/// Dimensions of the dispatch.
template <int Dimensions = 1>
class range : public detail::RawArray<Dimensions> {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "range can only be 1-, 2-, or 3-dimensional.");
  using Base = detail::RawArray<Dimensions>;

public:
  static constexpr int dimensions = Dimensions;
  range() noexcept = default;
  range(const range<Dimensions> &rhs) = default;
  range(range<Dimensions> &&rhs) = default;
  range<Dimensions> &operator=(const range<Dimensions> &rhs) = default;
  range<Dimensions> &operator=(range<Dimensions> &&rhs) = default;

  /// Constructs a 1D range with value dim0.
  ///  Only valid when the template parameter Dimensions is equal to 1.
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  range(std::size_t dim0) noexcept : Base(dim0) {}

  /// Constructs a 2D range with values dim0 and dim1.
  /// Only valid when the template parameter Dimensions is equal to 2.
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  range(std::size_t dim0, std::size_t dim1) noexcept : Base(dim0, dim1) {}

  /// Constructs a 3D range with values dim0, dim1 and dim2.
  /// Only valid when the template parameter Dimensions is equal to 3.
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  range(std::size_t dim0, std::size_t dim1, std::size_t dim2) noexcept
      : Base(dim0, dim1, dim2) {}

  /*
  Declared and implemented in detail::RawArray:
      std::size_t get(int dimension) const noexcept;
      std::size_t& operator[](int dimension) noexcept;
      std::size_t operator[](int dimension) const noexcept;
  */

  /// \return the size of the range computed as dimension0*…​*dimensionN.
  std::size_t size() const noexcept {
    std::size_t size = 1;
    for (int i = 0; i < Dimensions; ++i) {
      size *= Base::MArray[i];
    }
    return size;
  }

  // TODO: operators to be added
};

/// c++ deduction guides.
#ifdef __cpp_deduction_guides
range(std::size_t) -> range<1>;
range(std::size_t, std::size_t) -> range<2>;
range(std::size_t, std::size_t, std::size_t) -> range<3>;
#endif

template <int Dimensions = 1, bool WithOffset = true> class item;

/// SYCL 2020 4.9.1.3. id class.
/// id<int Dimensions> is a vector of Dimensions that is used to represent an id
/// into a global or local range. It can be used as an index in an accessor of
/// the same rank.
template <int Dimensions = 1> class id : public detail::RawArray<Dimensions> {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "id can only be 1-, 2-, or 3-dimensional.");
  using Base = detail::RawArray<Dimensions>;

  // Helper class for conversion operator. Void type is not suitable. User
  // cannot even try to get address of the operator PrivateTag(). User
  // may try to get an address of operator void() and will get the
  // compile-time error
  class PrivateTag;
  template <bool Condition, typename T>
  using EnableIfT = std::conditional_t<Condition, T, PrivateTag>;

public:
  static constexpr int dimensions = Dimensions;

  id() noexcept = default;
  id(const id<Dimensions> &rhs) = default;
  id(id<Dimensions> &&rhs) = default;
  id<Dimensions> &operator=(const id<Dimensions> &rhs) = default;
  id<Dimensions> &operator=(id<Dimensions> &&rhs) = default;

  /// Constructs a 1D id with value dim0.
  /// Only valid when the template parameter Dimensions is equal to 1.
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  id(std::size_t dim0) noexcept : Base(dim0) {}

  /// Constructs a 2D id with values dim0, dim1.
  /// Only valid when the template parameter Dimensions is equal to 2.
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  id(std::size_t dim0, std::size_t dim1) noexcept : Base(dim0, dim1) {}

  /// Constructs a 3D id with values dim0, dim1, dim2.
  /// Only valid when the template parameter Dimensions is equal to 3.
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  id(std::size_t dim0, std::size_t dim1, std::size_t dim2) noexcept
      : Base(dim0, dim1, dim2) {}

  /// Constructs an id from the dimensions of range.
  /// Only valid when the template parameter Dimensions is equal to 1.
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  id(const range<Dimensions> &range) noexcept : Base(range.get(0)) {}

  /// Constructs an id from the dimensions of range.
  /// Only valid when the template parameter Dimensions is equal to 2.
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  id(const range<Dimensions> &range) noexcept
      : Base(range.get(0), range.get(1)) {}

  /// Constructs an id from the dimensions of range.
  /// Only valid when the template parameter Dimensions is equal to 3.
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  id(const range<Dimensions> &range) noexcept
      : Base(range.get(0), range.get(1), range.get(2)) {}

  /// Constructs an id from item.get_id().
  /// Only valid when the template parameter Dimensions is equal to 1.
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  id(const item<Dimensions> &item) noexcept : Base(item.get_id(0)) {}

  /// Constructs an id from item.get_id().
  /// Only valid when the template parameter Dimensions is equal to 2.
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  id(const item<Dimensions> &item) noexcept
      : Base(item.get_id(0), item.get_id(1)) {}

  /// Constructs an id from item.get_id().
  /// Only valid when the template parameter Dimensions is equal to 3.
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  id(const item<Dimensions> &item) noexcept
      : Base(item.get_id(0), item.get_id(1), item.get_id(2)) {}

  /*
    Declared and implemented in detail::RawArray:
        std::size_t get(int dimension) const noexcept;
        std::size_t& operator[](int dimension) noexcept;
        std::size_t operator[](int dimension) const noexcept;
    */

  // Template operator is not allowed because it disables further type
  //   conversion. For example, the next code will not work in case of template
  //   conversion:
  //   int a = id<1>(value);
  /// Returns the same value as get(0).
  ///  Available only when: Dimensions == 1.
  operator EnableIfT<(Dimensions == 1), std::size_t>() const noexcept {
    return Base::get(0);
  }

  // TODO: operators to be added
};

/// c++ deduction guides.
#ifdef __cpp_deduction_guides
id(std::size_t) -> id<1>;
id(std::size_t, std::size_t) -> id<2>;
id(std::size_t, std::size_t, std::size_t) -> id<3>;
#endif

/// SYCL 2020 4.9.1.4. item class.
/// item identifies an instance of the function object executing at each point
/// in a range.
template <int Dimensions /* = 1*/, bool WithOffset /* = true*/> class item {
  /* Helper class for conversion operator. Void type is not suitable. User
   * cannot even try to get address of the operator PrivateTag(). User
   * may try to get an address of operator void() and will get the
   * compile-time error */
  class PrivateTag;
  template <bool Condition, typename T>
  using EnableIfT = std::conditional_t<Condition, T, PrivateTag>;

public:
  static constexpr int dimensions = Dimensions;

  item() = delete;

  item(const item &rhs) = default;

  item(item<Dimensions, WithOffset> &&rhs) = default;

  item &operator=(const item &rhs) = default;

  item &operator=(item &&rhs) = default;

  friend bool operator==(const item<Dimensions, WithOffset> &lhs,
                         const item<Dimensions, WithOffset> &rhs) {
    if constexpr (WithOffset)
      return (lhs.MId == rhs.MId) && (lhs.MRange == rhs.MRange) &&
             (lhs.MOffset == rhs.MOffset);
    else
      return (lhs.MId == rhs.MId) && (lhs.MRange == rhs.MRange);
  }

  friend bool operator!=(const item<Dimensions, WithOffset> &lhs,
                         const item<Dimensions, WithOffset> &rhs) {
    return !(lhs == rhs);
  }

  /// \return the constituent id representing the work-item’s position in the
  /// iteration space.
  id<Dimensions> get_id() const noexcept { return MId; }

  /// Equivalent to return get_id()[dimension].
  std::size_t get_id(int dimension) const noexcept {
    return MId.get(dimension);
  }

  /// Equivalent to return get_id(dimension).
  std::size_t operator[](int dimension) const noexcept {
    return MId[dimension];
  }

  /// \return a range representing the dimensions of the range of possible
  /// values of the item.
  range<Dimensions> get_range() const noexcept { return MRange; }

  /// Equivalent to return get_range().get(dimension).
  std::size_t get_range(int dimension) const noexcept {
    return MRange[dimension];
  }

  /// Deprecated in SYCL 2020.
  /// For an item converted from an item with no offset, this will always return
  /// an id of all 0 values. This member function is only available if
  /// WithOffset is true.
  /// \return an id representing the n-dimensional offset provided to the
  /// parallel_for and added by the runtime to the global-ID of each
  /// work-item, if this item represents a global range.
  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == true, bool> = true>
  id<Dimensions> get_offset() const noexcept {
    return MOffset;
  }

  /// Deprecated in SYCL 2020.
  /// This conversion allows users to seamlessly write code that assumes an
  /// offset and still provides an offset-less item. Available only when:
  /// WithOffset == false.
  /// \return an item representing the same information as the object holds but
  /// also includes the offset set to 0.
  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == false, bool> = true>
  operator item<Dimensions, true>() const noexcept {
    return item<Dimensions, true>(MRange, MId, id<Dimensions>{});
  }

  /// Equivalent to get_id(0).
  /// Available only when: Dimensions == 1.
  operator EnableIfT<(Dimensions == 1), std::size_t>() const noexcept {
    return get_id(0);
  }

  /// \return Return the id as a linear index value.
  std::size_t get_linear_id() const noexcept {
    if constexpr (WithOffset) {
      if constexpr (1 == Dimensions) {
        return MId[0] - MOffset[0];
      }
      if constexpr (2 == Dimensions) {
        return (MId[0] - MOffset[0]) * MRange[1] + MId[1] - MOffset[1];
      }
      return (MId[0] - MOffset[0]) * MRange[1] * MRange[2] +
             (MId[1] - MOffset[1]) * MRange[2] + MId[2] - MOffset[2];
    } else {
      if constexpr (1 == Dimensions) {
        return MId[0];
      }
      if constexpr (2 == Dimensions) {
        return MId[0] * MRange[1] + MId[1];
      }
      return MId[0] * MRange[1] * MRange[2] + MId[1] * MRange[2] + MId[2];
    }
  }

protected:
  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == true, bool> = true>
  item(const sycl::range<Dimensions> &range, const sycl::id<Dimensions> &id,
       const sycl::id<Dimensions> &offset)
      : MRange(range), MId(id), MOffset(offset) {}

  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == false, bool> = true>
  item(const range<Dimensions> &range, const id<Dimensions> &id)
      : MRange(range), MId(id), MOffset() {}

private:
  range<Dimensions> MRange;
  id<Dimensions> MId;
  std::conditional_t<WithOffset, id<Dimensions>, std::monostate> MOffset;

  friend class detail::Builder;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INDEX_SPACE_CLASSES_HPP
