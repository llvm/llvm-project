//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/exception.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class SYCLCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "sycl"; }
  std::string message(int) const override { return "SYCL Error"; }
};
} // namespace detail

// Free functions
const std::error_category &sycl_category() noexcept {
  static const detail::SYCLCategory SYCLCategoryObj;
  return SYCLCategoryObj;
}

std::error_code make_error_code(sycl::errc Err) noexcept {
  return std::error_code(static_cast<int>(Err), sycl_category());
}

// Exception methods implementation
exception::exception(std::error_code EC, const char *Msg)
    : MMessage(std::make_shared<std::string>(Msg)), MErrC(EC) {}

exception::~exception() {}

const std::error_code &exception::code() const noexcept { return MErrC; }

const std::error_category &exception::category() const noexcept {
  return code().category();
}

const char *exception::what() const noexcept { return MMessage->c_str(); }

bool exception::has_context() const noexcept { return false; }

_LIBSYCL_END_NAMESPACE_SYCL
