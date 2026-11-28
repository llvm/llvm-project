//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the shared implementation of the SYCL local accessors:
/// sycl::local_accessor and the deprecated specialization of sycl::accessor
/// for access::target::local.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_LOCAL_ACCESSOR_BASE_HPP
#define _LIBSYCL___IMPL_DETAIL_LOCAL_ACCESSOR_BASE_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/linearization.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// \return a process-wide unique non-zero identifier for a newly constructed
/// local accessor.
_LIBSYCL_EXPORT std::uint64_t createLocalAccessorId() noexcept;

/// \internal
/// Helper for operator[] of a multi-dimensional accessor. Each subscript
/// consumes one dimension, the last one yields the element reference.
template <typename DataT, int Dimensions, int CurDim> class AccessorSubscript {
  static_assert(Dimensions > CurDim,
                "Current dimension must be less than total dimensions");
  static constexpr bool IsLastDim = CurDim + 1 == Dimensions;

public:
  AccessorSubscript(DataT *Ptr, const range<Dimensions> &Range,
                    std::size_t LinearIndex)
      : MPtr(Ptr), MRange(Range), MLinearIndex(LinearIndex) {}

  decltype(auto) operator[](std::size_t Index) const {
    const std::size_t LinearIndex = MLinearIndex * MRange[CurDim] + Index;
    if constexpr (IsLastDim)
      return MPtr[LinearIndex];
    else
      return AccessorSubscript<DataT, Dimensions, CurDim + 1>(MPtr, MRange,
                                                              LinearIndex);
  }

private:
  DataT *MPtr;
  range<Dimensions> MRange;
  std::size_t MLinearIndex;
};

template <typename DataT, int Dimensions> class LocalAccessorBase;

/// \internal
/// Lays out the local accessors of one kernel invocation in the single block
/// of work-group local memory allocated for that invocation.
///
/// The layout is computed on host only. Each accessor in the kernel object
/// records the offset of its region before the kernel is submitted, and at the
/// start of the kernel the device turns the offset into a pointer to the
/// region.
///
/// Copies of one accessor in the kernel object share a single region, as SYCL
/// 2020 4.5.2 requires a copy to behave as if it were the original instance.
///
/// The functions take LocalAccessorBase so that template argument deduction
/// accepts both local accessor classes deriving from it.
class LocalAccessorReifier {
public:
  /// Places the regions of the local accessors of a kernel object and records
  /// the offset of its region in each of them.
  ///
  /// \param Accs the local accessors of the kernel object, in the order the
  /// compiler passes them.
  /// \return the size in bytes of the block that holds the regions of \p Accs,
  /// to request at launch.
  template <typename... AccTs>
  static std::size_t placeRegions(const AccTs &...Accs) {
    BlockLayout<sizeof...(AccTs)> Layout;
    (setOffset(Accs, placeRegion(Accs, Layout)), ...);
    return Layout.BlockSize;
  }

  /// Points each of \p Accs to its region of the block, at the start of the
  /// kernel.
  template <typename... AccTs>
  static void bind([[maybe_unused]] const AccTs &...Accs) noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    // The work-group local memory block allocated for the kernel invocation,
    // its size is ol_kernel_launch_size_args_t::DynSharedMemory.
    unsigned char _LIBSYCL_LOCAL_AS *Block =
        __builtin_sycl_dynamic_local_memory();
    (bindOne(Accs, Block), ...);
#endif
  }

private:
  /// The regions placed so far in one walk over the accessors of a kernel.
  template <std::size_t NumAccessors> struct BlockLayout {
    std::size_t BlockSize = 0;
    std::size_t NumRegions = 0;
    std::array<std::uint64_t, NumAccessors> Ids{};
    std::array<std::size_t, NumAccessors> Offsets{};
  };

  /// \return the offset of the region of \p Acc within the block, the region
  /// of an equal accessor placed earlier or a new one appended to \p Layout.
  template <typename DataT, int Dimensions, std::size_t NumAccessors>
  static std::size_t
  placeRegion(const LocalAccessorBase<DataT, Dimensions> &Acc,
              BlockLayout<NumAccessors> &Layout) {
    for (std::size_t I = 0; I < Layout.NumRegions; ++I)
      if (Layout.Ids[I] == Acc.MId)
        return Layout.Offsets[I];

    const std::size_t Offset = appendRegion(Acc, Layout.BlockSize);
    Layout.Ids[Layout.NumRegions] = Acc.MId;
    Layout.Offsets[Layout.NumRegions] = Offset;
    ++Layout.NumRegions;
    return Offset;
  }

  /// Appends the region of an accessor to the end of the block.
  ///
  /// \param Acc the accessor the region is placed for.
  /// \param BlockSize the size in bytes of the regions appended so far, it is
  /// increased by the region of \p Acc.
  /// \return the offset of the region of \p Acc within the block.
  template <typename DataT, int Dimensions>
  static std::size_t
  appendRegion(const LocalAccessorBase<DataT, Dimensions> &Acc,
               std::size_t &BlockSize) {
    // TODO: the alignment of the block start is not guaranteed, it depends on
    // where the driver places the local memory argument. A DataT aligned
    // stricter than that needs an over-allocated block.
    constexpr std::size_t Align = alignof(DataT);
    const std::size_t Offset = (BlockSize + Align - 1) / Align * Align;
    BlockSize = Offset + Acc.byte_size();
    return Offset;
  }

  template <typename DataT, int Dimensions>
  static void setOffset(const LocalAccessorBase<DataT, Dimensions> &Acc,
                        std::size_t Offset) noexcept {
    Acc.MOffset = Offset;
  }

  template <typename DataT, int Dimensions>
  static void bindOne(const LocalAccessorBase<DataT, Dimensions> &Acc,
                      unsigned char _LIBSYCL_LOCAL_AS *Block) noexcept {
    using LocalPtrT = typename LocalAccessorBase<DataT, Dimensions>::LocalPtrT;
    Acc.MData = reinterpret_cast<LocalPtrT>(Block + Acc.MOffset);
  }
};

/// \internal
/// Implements the members shared by sycl::local_accessor and the deprecated
/// sycl::accessor specialization for access::target::local.
///
/// The class has a single definition and a single layout for host and device.
/// On host element access is not available and the object records the
/// requested allocation size and the identity of the accessor. Before the
/// kernel is submitted, the host records in the accessor the offset of its
/// region of the kernel's local memory block, and at the start of the kernel
/// the device points the accessor to its region, see
/// LocalAccessorReifier.
///
/// The base is parameterized on DataT and Dimensions only, so that both
/// sycl::local_accessor and the deprecated sycl::accessor specialization for
/// access::target::local share a single instantiation.
template <typename DataT, int Dimensions> class LocalAccessorBase {
  static_assert(Dimensions >= 0 && Dimensions <= 3,
                "A local accessor can only be 0-, 1-, 2-, or 3-dimensional.");

public:
  using value_type = DataT;
  using reference = value_type &;
  using const_reference = const DataT &;
  // TODO: add accessor_ptr, get_pointer and get_multi_ptr once multi_ptr is
  // supported.

  // TODO: SYCL 2020 Table 51 requires the pointer underlying the iterator of a
  // local accessor to be address space qualified. A plain pointer is a
  // conforming choice for the unspecified iterator type only because the
  // qualifier is dropped on conversion to the generic address space.
  using iterator = value_type *;
  using const_iterator = const value_type *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  /// \return the size in bytes of the accessor's local memory allocation, per
  /// work-group.
  size_type byte_size() const noexcept { return size() * sizeof(DataT); }

  /// \return the number of elements in the accessor's local memory
  /// allocation, per work-group.
  size_type size() const noexcept { return MRange.size(); }

  /// \return the maximum number of elements any accessor of this type would
  /// be able to access.
  __SYCL2020_DEPRECATED("max_size() is deprecated by SYCL 2020")
  size_type max_size() const noexcept {
    return empty() ? 0 : (std::numeric_limits<difference_type>::max)();
  }

  /// \return true if (size() == 0).
  bool empty() const noexcept { return size() == 0; }

  /// Queries the number of elements of DataT per dimension that this accessor
  /// may access.
  ///
  /// Available only when (Dimensions > 0).
  ///
  /// \return the range that was specified when the accessor was constructed.
  template <int Dims = Dimensions, std::enable_if_t<(Dims > 0), bool> = true>
  range<Dimensions> get_range() const {
    return MRange;
  }

  /// \return false, as a local accessor has no properties.
  template <typename Property> bool has_property() const noexcept {
    return false;
  }

  /// Reports that a local accessor has no properties.
  ///
  /// \throw sycl::exception with sycl::errc::invalid for any Property.
  template <typename Property> Property get_property() const {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "a local accessor doesn't have the requested property");
  }

  // The following functions may only be called from within a command.

  /// Accesses the single element of a zero-dimensional accessor.
  ///
  /// Available only when (Dimensions == 0).
  ///
  /// \return a reference to the single element that is accessed by this
  /// accessor.
  template <int Dims = Dimensions, std::enable_if_t<Dims == 0, bool> = true>
  operator reference() const {
    return *getPtr();
  }

  /// Accesses the element at the location specified by \p index.
  ///
  /// Available only when (Dimensions > 0).
  ///
  /// \param index is the location of the element in the allocation.
  /// \return a reference to the element.
  template <int Dims = Dimensions, std::enable_if_t<(Dims > 0), bool> = true>
  reference operator[](id<Dimensions> index) const {
    return getPtr()[linearizeId(index, MRange)];
  }

  /// \overload
  ///
  /// Available only when (Dimensions == 1).
  ///
  /// \param index is the location of the element in the allocation.
  template <int Dims = Dimensions, std::enable_if_t<Dims == 1, bool> = true>
  reference operator[](std::size_t index) const {
    return getPtr()[index];
  }

  /// Selects \p index in the first dimension of this accessor.
  ///
  /// Available only when (Dimensions > 1).
  ///
  /// \param index is the index in the first dimension.
  /// \return an intermediate object with the dimensionality Dimensions-1,
  /// whose subscript operators select the remaining dimensions.
  template <int Dims = Dimensions, std::enable_if_t<(Dims > 1), bool> = true>
  auto operator[](std::size_t index) const {
    return AccessorSubscript<DataT, Dimensions, 1>(getPtr(), MRange, index);
  }

  /// \return an iterator to the first element of the memory this accessor may
  /// access.
  iterator begin() const noexcept { return getPtr(); }
  /// \return an iterator to one element past the last element of the memory
  /// this accessor may access.
  iterator end() const noexcept { return begin() + size(); }

  /// \return a const iterator to the first element of the memory this
  /// accessor may access.
  const_iterator cbegin() const noexcept { return const_iterator(begin()); }
  /// \return a const iterator to one element past the last element of the
  /// memory this accessor may access.
  const_iterator cend() const noexcept { return const_iterator(end()); }

  /// \return an iterator adaptor to the last element of the memory this
  /// accessor may access.
  reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
  /// \return an iterator adaptor to one element before the first element of
  /// the memory this accessor may access.
  reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

  /// \return a const iterator adaptor to the last element of the memory this
  /// accessor may access.
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  /// \return a const iterator adaptor to one element before the first element
  /// of the memory this accessor may access.
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

protected:
  static constexpr int AdjustedDims = Dimensions == 0 ? 1 : Dimensions;

  using LocalPtrT = DataT _LIBSYCL_LOCAL_AS *;

  // createLocalAccessorId() is not available on device.
  LocalAccessorBase() noexcept
#ifndef __SYCL_DEVICE_ONLY__
      : MId(createLocalAccessorId())
#endif
  {
  }

  LocalAccessorBase(range<AdjustedDims> AllocationSize,
                    const property_list &PropList)
      : MId(createLocalAccessorId()), MRange(AllocationSize) {
    checkPropertyList<LocalAccessorBase>(PropList);
  }

  template <typename OtherDataT>
  LocalAccessorBase(
      const LocalAccessorBase<OtherDataT, Dimensions> &Other) noexcept
      : MOffset(Other.MOffset), MId(Other.MId), MRange(Other.MRange),
        MData(Other.MData) {}

  template <typename OtherDataT>
  bool
  isEqual(const LocalAccessorBase<OtherDataT, Dimensions> &Rhs) const noexcept {
    // TODO: on device, assert that equal accessors have equal MData.
    return MId == Rhs.MId;
  }

  std::uint64_t getId() const noexcept { return MId; }

  /// \return a generic pointer to the first element of this accessor's region,
  /// or nullptr on host.
  DataT *getPtr() const noexcept { return MData; }

private:
  // The offset and the pointer below are mutable, so that the region can be
  // recorded in a const accessor, see LocalAccessorReifier.

  // Offset of this accessor's region in the kernel's local memory block. Set
  // on host in the kernel object that is submitted.
  mutable std::size_t MOffset = 0;

  // Shared by the copies of an accessor, see createLocalAccessorId().
  std::uint64_t MId = 0;

  range<AdjustedDims> MRange;

  // Set on device at the start of the kernel. Declared last, so that a local
  // pointer narrower than a host pointer does not move the other members.
  // TODO: give the pointer a fixed size of 8 bytes, so that the layout does not
  // depend on the device target.
  mutable LocalPtrT MData = nullptr;

  template <typename, int> friend class LocalAccessorBase;
  friend class LocalAccessorReifier;
};

template <typename DataT, int Dimensions>
struct ObjectKindOf<LocalAccessorBase<DataT, Dimensions>> {
  static constexpr ObjectKind value = ObjectKind::LocalAccessor;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_LOCAL_ACCESSOR_BASE_HPP
