//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of sycl::kernel_id and its implementation
/// counterpart, which represent a kernel identificator.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_KERNEL_ID
#define _LIBSYCL_KERNEL_ID

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

#include <memory>
#include <string>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
/// The class is the implementation counterpart for sycl::kernel_id, which
/// represents a kernel identificator.
class KernelIdImpl {
public:
  KernelIdImpl(std::string_view Name) : MName(Name) {}
  KernelIdImpl() {}
  /// \return a null-terminated string representing the name of the kernel this
  /// id stands for.
  const char *get_name() { return MName.data(); }

private:
  std::string MName;
};
} // namespace detail

// TODO: It is not exported now, but is a part of SYCL spec.
/// Kernel identifier.
class kernel_id {
public:
  kernel_id() = delete;

  kernel_id(const kernel_id &rhs) = default;

  kernel_id(kernel_id &&rhs) = default;

  kernel_id &operator=(const kernel_id &rhs) = default;

  kernel_id &operator=(kernel_id &&rhs) = default;

  friend bool operator==(const kernel_id &lhs, const kernel_id &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const kernel_id &lhs, const kernel_id &rhs) {
    return !(lhs == rhs);
  }

  /// \returns a null-terminated string that contains the kernel name.
  const char *get_name() const noexcept { return impl->get_name(); }

private:
  kernel_id(const char *Name);

  kernel_id(const std::shared_ptr<detail::KernelIdImpl> &Impl)
      : impl(std::move(Impl)) {}

  std::shared_ptr<detail::KernelIdImpl> impl;
  friend sycl::detail::ImplUtils;
};

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::kernel_id>
    : public sycl::detail::HashBase<sycl::kernel_id> {};

#endif // _LIBSYCL_KERNEL_ID
