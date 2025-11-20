//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/platform.hpp>

#include <detail/platform_impl.hpp>

#include <stdexcept>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

backend platform::get_backend() const noexcept { return impl.getBackend(); }

std::vector<platform> platform::get_platforms() {
  auto PlatformsView = detail::platform_impl::getPlatforms();
  std::vector<platform> Platforms;
  Platforms.reserve(PlatformsView.size());
  for (size_t i = 0; i < PlatformsView.size(); i++) {
    platform Platform =
        detail::createSyclObjFromImpl<platform>(PlatformsView[i]);
    Platforms.push_back(std::move(Platform));
  }
  return Platforms;
}

template <typename Param>
typename detail::is_platform_info_desc<Param>::return_type
platform::get_info() const {
  return impl.get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template _LIBSYCL_EXPORT ReturnT platform::get_info<info::platform::Desc>()  \
      const;
#include <sycl/__impl/info/platform.def>
#undef __SYCL_PARAM_TRAITS_SPEC

_LIBSYCL_END_NAMESPACE_SYCL
