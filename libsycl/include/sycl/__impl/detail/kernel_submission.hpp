//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helpers for kernel submission entry points.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_KERNEL_SUBMISSION_HPP
#define _LIBSYCL___IMPL_DETAIL_KERNEL_SUBMISSION_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/detail/kernel_arg_helpers.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/nd_item.hpp>
#include <sycl/__impl/nd_range.hpp>

#include <tuple>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

template <int Dims>
void checkNDRangeAndThrow(const sycl::nd_range<Dims> executionRange) {
  if (executionRange.get_global_range() != range<Dims>{} &&
      (executionRange.get_local_range().size() == 0 ||
       executionRange.get_global_range() % executionRange.get_local_range() !=
           range<Dims>{}))
    throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                          "Invalid nd_range submission: global size must be "
                          "evenly divisible by local size.");
}

template <typename DerivedT> class KernelSubmissionBase {
protected:
  template <typename KernelName, typename KernelType>
#ifdef SYCL_LANGUAGE_VERSION
  [[clang::sycl_kernel_entry_point(KernelName)]]
#endif
  void submitSingleTask(const KernelType &KernelFunc) {
    KernelFunc();
  }

  template <typename KernelName, typename ElementType, typename KernelType>
#ifdef SYCL_LANGUAGE_VERSION
  [[clang::sycl_kernel_entry_point(KernelName)]]
#endif
  void submitParallelFor(const KernelType &KernelFunc) {
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
  }

  template <typename KN, typename... Args>
  void sycl_kernel_launch(const char *KernelName, Args &&...args) {
    static_assert(
        sizeof...(args) == 1,
        "sycl_kernel_launch expects only 2 arguments now: name of kernel and "
        "callable object passed to kernel invocation by the user.");

    auto FirstArg = std::get<0>(std::tie(args...));
    static_cast<DerivedT *>(this)->submitKernelImpl(
        detail::getDeviceKernelInfo<KN>(KernelName), &FirstArg,
        sizeof(FirstArg));
  }

  template <typename KernelName, int Dims, template <int> class Range,
            typename... Rest>
  void parallelForImpl(Range<Dims> numWorkItems, Rest &&...rest) {
    if constexpr (sizeof...(Rest) != 1)
      throw sycl::exception(errc::feature_not_supported,
                            "Reductions are not supported");

    using KernelType =
        std::decay_t<detail::nth_type_t<sizeof...(Rest) - 1, Rest...>>;
    constexpr bool IsNdRangeSubmission =
        std::is_same_v<Range<Dims>, nd_range<Dims>>;
    using SuggestedArgType =
        std::conditional_t<IsNdRangeSubmission, nd_item<Dims>, item<Dims>>;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, SuggestedArgType>;

    if constexpr (IsNdRangeSubmission) {
      static_assert(
          std::is_convertible_v<sycl::nd_item<Dims>, LambdaArgType>,
          "Kernel argument of a sycl::parallel_for with sycl::nd_range "
          "must be sycl::nd_item or be convertible from sycl::nd_item");
    } else {
      static_assert(
          std::is_convertible_v<sycl::item<Dims>, LambdaArgType> ||
              std::is_convertible_v<sycl::item<Dims, false>, LambdaArgType>,
          "Kernel argument of a sycl::parallel_for with sycl::range "
          "must be sycl::item or be convertible from sycl::item");
    }

    using TransformedLambdaArgType = std::conditional_t<
        IsNdRangeSubmission, nd_item<Dims>,
        std::conditional_t<
            std::is_convertible_v<sycl::item<Dims>, LambdaArgType>, item<Dims>,
            std::conditional_t<
                std::is_convertible_v<sycl::item<Dims, false>, LambdaArgType>,
                item<Dims, false>, LambdaArgType>>>;

    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    return submitParallelFor<NameT, TransformedLambdaArgType, KernelType>(
        std::forward<Rest>(rest)...);
  }
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_KERNEL_SUBMISSION_HPP
