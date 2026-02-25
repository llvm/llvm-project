//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

#include <detail/device_impl.hpp>
#include <detail/global_objects.hpp>
#include <detail/platform_impl.hpp>

#include <algorithm>
#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

PlatformImpl &PlatformImpl::getPlatformImpl(ol_platform_handle_t Platform) {
  auto &PlatformCache = getPlatformCache();
  for (auto &PlatImpl : PlatformCache) {
    assert(PlatImpl && "Platform impl can not be nullptr");
    if (PlatImpl->getHandleRef() == Platform)
      return *PlatImpl;
  }

  throw sycl::exception(
      sycl::make_error_code(sycl::errc::runtime),
      "Platform for requested handle can't be created. This handle is not in "
      "the list of platforms discovered by liboffload");
}

const std::vector<PlatformImplUPtr> &PlatformImpl::getPlatforms() {
  [[maybe_unused]] static auto InitPlatformsOnce = []() {
    discoverOffloadDevices();

    registerStaticVarShutdownHandler();

    auto &PlatformCache = getPlatformCache();
    for (const auto &Topo : getOffloadTopologies()) {
      size_t PlatformIndex = 0;
      for (const auto &OffloadPlatform : Topo.getPlatforms()) {
        PlatformCache.emplace_back(std::make_unique<PlatformImpl>(
            OffloadPlatform, PlatformIndex++, PrivateTag{}));
      }
    }
    return true;
  }();
  return getPlatformCache();
}

PlatformImpl::PlatformImpl(ol_platform_handle_t Platform, size_t PlatformIndex,
                           PrivateTag)
    : MOffloadPlatform(Platform), MOffloadPlatformIndex(PlatformIndex) {
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  callAndThrow(olGetPlatformInfo, MOffloadPlatform, OL_PLATFORM_INFO_BACKEND,
               sizeof(Backend), &Backend);
  MBackend = convertBackend(Backend);
  MOffloadBackend = Backend;

  const auto &Topologies = getOffloadTopologies();
  auto RootTopologyIt = std::find_if(
      Topologies.begin(), Topologies.end(), [&](const OffloadTopology &Topo) {
        return Topo.getBackend() == MOffloadBackend;
      });

  assert(RootTopologyIt != Topologies.end() &&
         "Root topology for platform must always exist");
  auto DevRange = RootTopologyIt->getDevices(MOffloadPlatformIndex);
  MRootDevices.reserve(DevRange.size());
  std::for_each(DevRange.begin(), DevRange.end(),
                [&](const ol_device_handle_t &Device) {
                  MRootDevices.emplace_back(std::make_unique<DeviceImpl>(
                      Device, *this, DeviceImpl::PrivateTag{}));
                });
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

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
