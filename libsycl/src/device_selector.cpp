//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/device_selector.hpp>

#include <detail/device_impl.hpp>
#include <detail/program_manager.hpp>

#include <algorithm>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

static constexpr int MatchedTypeDefaultScore = 1000;
static constexpr int GPUDeviceDefaultScore = 500;
static constexpr int CPUDeviceDefaultScore = 300;
static constexpr int AccDeviceDefaultScore = 75;
static constexpr int RejectDeviceScore = -1;

static constexpr int CompatibleImageBonus = 1000;
static constexpr int LevelZeroBonus = 50;

static int getDevicePreference(const device &Device) {
  int Score = 0;
  detail::DeviceImpl *Impl = detail::getSyclObjImpl(Device);

  auto &ProgramManager = detail::ProgramAndKernelManager::getInstance();
  if (ProgramManager.hasCompatibleImage(*Impl))
    Score += CompatibleImageBonus;

  if (Impl->getBackend() == backend::level_zero)
    Score += LevelZeroBonus;

  return Score;
}

_LIBSYCL_EXPORT int default_selector_v(const device &dev) {
  int Score = getDevicePreference(dev);

  if (dev.is_gpu())
    Score += GPUDeviceDefaultScore;
  else if (dev.is_cpu())
    Score += CPUDeviceDefaultScore;
  else if (dev.is_accelerator())
    Score += AccDeviceDefaultScore;

  return Score;
}

_LIBSYCL_EXPORT int gpu_selector_v(const device &dev) {
  return dev.is_gpu() ? MatchedTypeDefaultScore + getDevicePreference(dev)
                      : RejectDeviceScore;
}

_LIBSYCL_EXPORT int cpu_selector_v(const device &dev) {
  return dev.is_cpu() ? MatchedTypeDefaultScore + getDevicePreference(dev)
                      : RejectDeviceScore;
}

_LIBSYCL_EXPORT int accelerator_selector_v(const device &dev) {
  return dev.is_accelerator()
             ? MatchedTypeDefaultScore + getDevicePreference(dev)
             : RejectDeviceScore;
}

_LIBSYCL_EXPORT detail::DeviceSelectorInvocableType
aspect_selector(const std::vector<aspect> &aspectList,
                const std::vector<aspect> &denyList) {
  return [=](const sycl::device &Dev) {
    // 4.6.1.1. Device selector:
    // If no aspects are passed in, the generated selector behaves like
    // default_selector_v.
    if (aspectList.empty() && denyList.empty())
      return default_selector_v(Dev);

    auto HasAspect = [&Dev](const aspect &Aspect) -> bool {
      return Dev.has(Aspect);
    };
    if (!std::all_of(aspectList.begin(), aspectList.end(), HasAspect))
      return RejectDeviceScore;

    if (std::any_of(denyList.begin(), denyList.end(), HasAspect))
      return RejectDeviceScore;

    return MatchedTypeDefaultScore + getDevicePreference(Dev);
  };
}

namespace detail {

_LIBSYCL_EXPORT device
SelectDevice(const DeviceSelectorInvocableType &DeviceSelector) {
  int ChosenDeviceScore = RejectDeviceScore;
  const device *ChosenDevice = nullptr;

  std::vector<device> Devices = device::get_devices();
  for (const auto &Device : Devices) {
    int CurrentDevScore = DeviceSelector(Device);
    if (CurrentDevScore < 0)
      continue;

    if (!ChosenDevice || (ChosenDeviceScore < CurrentDevScore) ||
        ((ChosenDeviceScore == CurrentDevScore) &&
         (getDevicePreference(*ChosenDevice) < getDevicePreference(Device)))) {
      ChosenDevice = &Device;
      ChosenDeviceScore = CurrentDevScore;
    }
  }

  if (ChosenDevice != nullptr)
    return *ChosenDevice;

  throw exception(make_error_code(errc::runtime),
                  "No device of requested type is available");
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL
