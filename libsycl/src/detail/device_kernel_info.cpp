//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/device_kernel_info.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

ol_symbol_handle_t
DeviceKernelInfo::tryGetCachedKernel(ContextImpl *Context,
                                     ol_device_handle_t Device) {
  CacheKeyT Key = {Context, Device};
  std::lock_guard<std::mutex> Guard(MCacheMutex);
  if (auto Result = MCache.find(Key); Result != MCache.end())
    return Result->second;
  return nullptr;
}

void DeviceKernelInfo::addCachedKernel(ContextImpl *Context,
                                       ol_device_handle_t Device,
                                       ol_symbol_handle_t Kernel) {
  CacheKeyT Key = {Context, Device};
  {
    std::lock_guard<std::mutex> Guard(MCacheMutex);
    MCache.try_emplace(Key, Kernel);
  }
  Context->trackKernelInfoCache(this);
}

void DeviceKernelInfo::removeCachedKernelsFor(ContextImpl *Context) {
  std::lock_guard<std::mutex> Guard(MCacheMutex);
  for (auto It = MCache.begin(); It != MCache.end();) {
    CacheKeyT Key = It->first;
    if (Key.first == Context) {
      It = MCache.erase(It);
    } else {
      ++It;
    }
  }
}
} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL
