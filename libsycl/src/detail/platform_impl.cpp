//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/property_list.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/global_objects.hpp>
#include <detail/platform_impl.hpp>

#include <algorithm>
#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

bool PlatformImpl::rediscoverIfEmpty = false;

const std::vector<PlatformImplUPtr> &PlatformImpl::getPlatforms() {
  static auto InitPlatforms = []() {
    discoverOffloadDevices();

    registerStaticVarShutdownHandler();

    auto &PlatformCache = getPlatformCache();
    for (const auto &Topo : getOffloadTopologies()) {
      if (Topo.getBackend() == OL_PLATFORM_BACKEND_HOST)
        continue;
      for (const auto &PlatformGroup : Topo.getPlatformGroups()) {
        PlatformCache.emplace_back(std::make_unique<PlatformImpl>(
            PlatformGroup.Platform, PlatformGroup.Devices, PrivateTag{}));
      }
    }
  };

  [[maybe_unused]] static auto InitPlatformsOnce = []() {
    callAndThrow(olInit, nullptr);
    InitPlatforms();
    return true;
  }();
  auto &PlatformCache = getPlatformCache();
  if (rediscoverIfEmpty && PlatformCache.empty())
    InitPlatforms();

  return PlatformCache;
}

PlatformImpl::PlatformImpl(ol_platform_handle_t Platform,
                           const std::vector<ol_device_handle_t> &Devices,
                           PrivateTag)
    : MOffloadPlatform(Platform) {
  assert(!Devices.empty() && "Platform must contain at least one device");

  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  callAndThrow(olGetPlatformInfo, MOffloadPlatform, OL_PLATFORM_INFO_BACKEND,
               sizeof(Backend), &Backend);
  MBackend = convertBackend(Backend);

  MRootDevices.reserve(Devices.size());
  for (const ol_device_handle_t &Device : Devices) {
    MRootDevices.emplace_back(
        std::make_unique<DeviceImpl>(Device, *this, DeviceImpl::PrivateTag{}));
  }

  std::vector<DeviceImpl *> DeviceImpls;
  DeviceImpls.reserve(MRootDevices.size());
  for (const auto &Device : MRootDevices)
    DeviceImpls.push_back(Device.get());

  MDefaultContext = ContextImpl::create(std::move(DeviceImpls),
                                        defaultAsyncHandler, property_list{});
}

const std::vector<DeviceImplUPtr> &PlatformImpl::getRootDevices() const {
  return MRootDevices;
}

bool PlatformImpl::has(aspect Aspect) const {
  const auto &Devices = getRootDevices();
  return std::all_of(
      Devices.begin(), Devices.end(),
      [&Aspect](const DeviceImplUPtr &Device) { return Device->has(Aspect); });
}

void PlatformImpl::iterateDevices(
    info::device_type DeviceType,
    std::function<void(DeviceImpl *)> callback) const {
  // Early exit if host/custom/accelerator device is requested:
  // - host device is deprecated and not required by the SYCL 2020
  // specification.
  // - accelerator and custom devices are unsupported by liboffload.
  if ((DeviceType == info::device_type::host) ||
      (DeviceType == info::device_type::custom) ||
      (DeviceType == info::device_type::accelerator))
    return;

  const auto &DeviceImpls = getRootDevices();
  assert(!DeviceImpls.empty() &&
         "Platform can't exist without at least one device.");

  // TODO: Need a way to get default device from liboffload.
  // As a temporal solution just return the first device for DeviceType ==
  // automatic.
  if (DeviceType == info::device_type::automatic) {
    callback(DeviceImpls[0].get());
    return;
  }

  bool KeepAll = DeviceType == info::device_type::all;
  for (auto &Impl : DeviceImpls) {
    if (KeepAll || DeviceType == Impl->getDeviceType())
      callback(Impl.get());
  }
}

ContextImpl &PlatformImpl::getDefaultContext() {
  assert(MDefaultContext && "Default context must be created in platform ctor");
  return *MDefaultContext;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
