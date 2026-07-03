//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helpers for kernel invocation.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_KERNEL_ARG_HELPERS
#define _LIBSYCL___IMPL_DETAIL_KERNEL_ARG_HELPERS

#include <sycl/__impl/index_space_classes.hpp>

#include <sycl/__impl/detail/config.hpp>

#include <sycl/__spirv/spirv_vars.hpp>

#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// \name  Helpers for unnamed lambdas.
/// @{
/// This class is the default kernel name template parameter type for kernel
/// invocation APIs such as single_task.
class AutoName {};

/// Helper struct to get a kernel name type based on given Name and Type
/// types: if Name is undefined (is AutoName) then Type becomes
/// the Name.
template <typename Name, typename Type> struct get_kernel_name_t {
  using name = Name;
};

/// Specialization for the case when Name is undefined.
/// This is only legal with our compiler with the unnamed lambda support or if
/// the kernel is a functor.
template <typename Type> struct get_kernel_name_t<detail::AutoName, Type> {
  using name = Type;
};
/// @}

/// \name  Helpers to verify kernel lambda types.
/// \brief Checks that the function is callable with operator().
/// @{
template <typename, typename T> struct CheckFunctionSignature {
  static_assert(std::integral_constant<T, false>::value,
                "Second template parameter is required to be of function type");
};

template <typename F, typename RetT, typename... Args>
struct CheckFunctionSignature<F, RetT(Args...)> {
private:
  template <typename T>
  static constexpr auto check(T *) -> typename std::is_same<
      decltype(std::declval<T>().operator()(std::declval<Args>()...)),
      RetT>::type;

  template <typename> static constexpr std::false_type check(...);

  using type = decltype(check<F>(0));

public:
  static constexpr bool value = type::value;
};
/// @}

/// \name  Helpers to extract types of lambda arguments.
/// @{
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg) const);

// Non-const version of the above template to match functors whose
// 'operator()' is declared w/o the 'const' qualifier.
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg));

template <typename F, typename SuggestedArgType>
decltype(member_ptr_helper(&F::operator())) argument_helper(int);

template <typename F, typename SuggestedArgType>
SuggestedArgType argument_helper(...);

template <typename F, typename SuggestedArgType>
using lambda_arg_type = decltype(argument_helper<F, SuggestedArgType>(0));

#if __has_builtin(__type_pack_element)
template <int N, typename... Ts>
using nth_type_t = __type_pack_element<N, Ts...>;
#else
template <int N, typename T, typename... Ts> struct nth_type {
  using type = typename nth_type<N - 1, Ts...>::type;
};

template <typename T, typename... Ts> struct nth_type<0, T, Ts...> {
  using type = T;
};

template <int N, typename... Ts>
using nth_type_t = typename nth_type<N, Ts...>::type;
#endif
/// @}

template <typename T> T *declptr() { return static_cast<T *>(nullptr); }

template <int N>
static inline constexpr bool isValidDimensions = (N > 0) && (N < 4);

/// Class provides helper functions for iteration space coordinates in kernel
/// invocation on device.
class Builder {
public:
  Builder() = delete;

  /// \return the global index of the work item currently being operated on by
  /// the device.
  template <int Dims> static const id<Dims> getElement(id<Dims> *) {
    static_assert(isValidDimensions<Dims>, "invalid dimensions");
    return __spirv::initBuiltInGlobalInvocationId<Dims, id<Dims>>();
  }

  /// Constructs item with the given data.
  /// \param Extent a range representing the dimensions of the range of possible
  /// values of the item.
  /// \param Index a constituent id representing the work-item’s position in the
  /// iteration space.
  /// \param Offset an id representing the n-dimensional offset that should be
  /// added to the global-ID of each work-item, if this item represents a global
  /// range. Deprecated in SYCL 2020.
  template <int Dims, bool WithOffset>
  static std::enable_if_t<WithOffset, item<Dims, WithOffset>>
  createItem(const range<Dims> &Extent, const id<Dims> &Index,
             const id<Dims> &Offset) {
    return item<Dims, WithOffset>(Extent, Index, Offset);
  }

  /// Constructs item with the given data.
  /// \param Extent a range representing the dimensions of the range of possible
  /// values of the item.
  /// \param Index a constituent id representing the work-item’s position in the
  /// iteration space.
  template <int Dims, bool WithOffset>
  static std::enable_if_t<!WithOffset, item<Dims, WithOffset>>
  createItem(const range<Dims> &Extent, const id<Dims> &Index) {
    return item<Dims, WithOffset>(Extent, Index);
  }

  /// Creates a sycl::item instance for the work item that is currently being
  /// operated on.
  template <int Dims, bool WithOffset>
  static std::enable_if_t<WithOffset, const item<Dims, WithOffset>> getItem() {
    static_assert(isValidDimensions<Dims>, "invalid dimensions");
    id<Dims> GlobalId{__spirv::initBuiltInGlobalInvocationId<Dims, id<Dims>>()};
    range<Dims> GlobalSize{__spirv::initBuiltInGlobalSize<Dims, range<Dims>>()};
    id<Dims> GlobalOffset{__spirv::initBuiltInGlobalOffset<Dims, id<Dims>>()};
    return createItem<Dims, true>(GlobalSize, GlobalId, GlobalOffset);
  }

  /// Creates a sycl::item instance for the work item that is currently being
  /// operated on.
  template <int Dims, bool WithOffset>
  static std::enable_if_t<!WithOffset, const item<Dims, WithOffset>> getItem() {
    static_assert(isValidDimensions<Dims>, "invalid dimensions");
    id<Dims> GlobalId{__spirv::initBuiltInGlobalInvocationId<Dims, id<Dims>>()};
    range<Dims> GlobalSize{__spirv::initBuiltInGlobalSize<Dims, range<Dims>>()};
    return createItem<Dims, false>(GlobalSize, GlobalId);
  }

  /// \return the work item currently being operated on by the device.
  template <int Dims, bool WithOffset>
  static auto getElement(item<Dims, WithOffset> *)
      -> decltype(getItem<Dims, WithOffset>()) {
    return getItem<Dims, WithOffset>();
  }
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_KERNEL_ARG_HELPERS
