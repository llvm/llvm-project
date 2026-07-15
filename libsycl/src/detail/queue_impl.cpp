//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>

#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_objects.hpp>
#include <detail/program_manager.hpp>

#include <algorithm>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

static void setKernelLaunchArgs(const detail::UnifiedRangeView &Range,
                                ol_kernel_launch_size_args_t &ArgsToSet) {
  assert(Range.MDims < 4 && "Invalid dimensions.");
  uint32_t GlobalSize[3] = {1, 1, 1};
  if (Range.MGlobalSize) {
    for (size_t I = 0; I < Range.MDims; ++I) {
      assert(Range.MGlobalSize[I] <= std::numeric_limits<uint32_t>::max());
      GlobalSize[I] = static_cast<uint32_t>(Range.MGlobalSize[I]);
    }
  }

  uint32_t GroupSize[3] = {1, 1, 1};
  if (Range.MLocalSize) {
    for (size_t I = 0; I < Range.MDims; ++I) {
      assert(Range.MLocalSize[I] <= std::numeric_limits<uint32_t>::max() &&
             Range.MLocalSize[I] != 0);
      GroupSize[I] = static_cast<uint32_t>(Range.MLocalSize[I]);
    }
  }

  ArgsToSet.Dimensions = Range.MDims;
  ArgsToSet.NumGroups.x = GlobalSize[0] / GroupSize[0];
  ArgsToSet.NumGroups.y = GlobalSize[1] / GroupSize[1];
  ArgsToSet.NumGroups.z = GlobalSize[2] / GroupSize[2];
  ArgsToSet.GroupSize.x = GroupSize[0];
  ArgsToSet.GroupSize.y = GroupSize[1];
  ArgsToSet.GroupSize.z = GroupSize[2];
  ArgsToSet.DynSharedMemory = 0;
}

QueueImpl::QueueImpl(DeviceImpl &deviceImpl, const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag)
    : MIsInorder(false), MAsyncHandler(asyncHandler), MPropList(propList),
      MDevice(deviceImpl),
      MContext(MDevice.getPlatformImpl().getDefaultContext()) {
  callAndThrow(olCreateQueue, MDevice.getOLHandle(), &MOffloadQueue);
}

QueueImpl::~QueueImpl() {
  // TODO: consider where to report errors
  if (MOffloadQueue)
    std::ignore = olDestroyQueue(MOffloadQueue);
}

backend QueueImpl::getBackend() const noexcept { return MDevice.getBackend(); }

static ol_device_handle_t getHostOLDevice() {
  static ol_device_handle_t HostDevice =
      *(getOffloadTopologies()[OL_PLATFORM_BACKEND_HOST].getDevices(0).begin());
  return HostDevice;
}

void QueueImpl::wait() { callAndThrow(olSyncQueue, MOffloadQueue); }

void QueueImpl::waitAndThrow() {
  wait();
  throwAsynchronous();
}

void QueueImpl::throwAsynchronous() { flushAsyncExceptions(); }

static void checkEventsPlatformMatch(const std::vector<EventImplPtr> &Events,
                                     const PlatformImpl &QueuePlatform) {
  // liboffload limitation to olWaitEvents. We can't do any extra handling for
  // cross context/platform events without host task support now.
  //   "The input events can be from any queue on any device provided by the
  //   same platform as `Queue`."
  if (!std::all_of(Events.cbegin(), Events.cend(),
                   [&QueuePlatform](const EventImplPtr &Event) {
                     return &Event->getPlatformImpl() == &QueuePlatform;
                   })) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "libsycl doesn't support cross-context/platform event dependencies "
        "yet.");
  }
}

void QueueImpl::setKernelParameters(std::vector<EventImplPtr> &&Events,
                                    const detail::UnifiedRangeView &Range) {
  checkEventsPlatformMatch(Events, MDevice.getPlatformImpl());

  // It is done at the beginning of a new submission to ensure that we can still
  // submit a kernel properly if the previous submission throws.
  MCurrentSubmitInfo.DepEvents.clear();
  MCurrentSubmitInfo.Range = {};

  MCurrentSubmitInfo.DepEvents = std::move(Events);
  setKernelLaunchArgs(Range, MCurrentSubmitInfo.Range);
}

void QueueImpl::submitKernelImpl(DeviceKernelInfo &KernelInfo, void *ArgData,
                                 size_t ArgSize) {
  ol_symbol_handle_t Kernel =
      detail::ProgramAndKernelManager::getInstance().getOrCreateKernel(
          KernelInfo, MDevice);
  assert(Kernel);

  handleEventDependencies(MCurrentSubmitInfo.DepEvents);

  assert(ArgData && "At least one argument must exist");
  assert(ArgSize && "Arguments size must be greater than 0");

  void *ArgPtrs[] = {ArgData};
  size_t ArgSizes[] = {ArgSize};
  auto Result =
      olLaunchKernel(MOffloadQueue, MDevice.getOLHandle(), Kernel,
                     &MCurrentSubmitInfo.Range, NULL, 1, ArgPtrs, ArgSizes);
  if (isFailed(Result))
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          std::string("Kernel submission (") +
                              KernelInfo.getName().data() + ") failed with " +
                              formatCodeString(Result));

  MCurrentSubmitInfo.LastEvent =
      createEvent(std::move(MCurrentSubmitInfo.DepEvents));
}

// Returns the {DeviceHandle, IsHostDevice} pair associated with the ptr.
static std::pair<ol_device_handle_t, bool> getAllocDevice(const void *ptr) {
  // TODO: consider caching this information to avoid querying it every time.
  ol_device_handle_t Device{};
  [[maybe_unused]] ol_result_t Result =
      callNoCheck(olGetMemInfo, ptr, OL_MEM_INFO_DEVICE,
                  sizeof(ol_device_handle_t), &Device);
  if (detail::isFailed(Result)) {
    // If liboffload could not find the allocation, assume it is a host one.
    if (Result->Code == OL_ERRC_NOT_FOUND) {
      return {getHostOLDevice(), true};
    }
    checkAndThrow(Result);
  }

  assert(Device);
  return {Device, false};
}

std::shared_ptr<EventImpl>
QueueImpl::memcpy(void *Dest, const void *Src, std::size_t NumBytes,
                  const std::vector<EventImplPtr> &DepEvents) {
  checkEventsPlatformMatch(DepEvents, MDevice.getPlatformImpl());
  if (NumBytes == 0) {
    handleEventDependencies(DepEvents);
    return createEvent();
  }

  if (!Dest || !Src) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Nullptr argument in memcpy operation");
  }

  auto [DestOLDevice, IsDestOLDeviceHost] = getAllocDevice(Dest);
  auto [SrcOLDevice, IsSrcOLDeviceHost] = getAllocDevice(Src);

  // TODO: Currently, liboffload does not let us specify a queue for
  // host-to-host cases, which means that the operation would be synchronous and
  // any implicit dependencies wouldn't be respected for an in-order queue.
  if (IsDestOLDeviceHost && IsSrcOLDeviceHost)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Host-to-host copy is not implemented yet");

  handleEventDependencies(DepEvents);
  callAndThrow(olMemcpy, MOffloadQueue, Dest, DestOLDevice, Src, SrcOLDevice,
               NumBytes);
  return createEvent();
}

void QueueImpl::handleEventDependencies(const std::vector<EventImplPtr> &Deps) {
  // TODO: liboffload supports only in-order queues and no cross context waiting
  // is available now that means that this code is excessive but correct. I
  // don't want to skip it and rely on default liboffload behaviour that is
  // applicable for in-order queue only. Once OOO queues are added this waiting
  // must be disabled for in-order queues. Once host tasks are added - cross
  // context dependencies should be enabled and checked as well.
  if (!Deps.empty()) {
    auto EventHandles = getSyclObjHandles(Deps);
    callAndThrow(olWaitEvents, MOffloadQueue, EventHandles.data(),
                 EventHandles.size());
  }
}

EventImplPtr QueueImpl::createEvent(std::vector<EventImplPtr> &&Deps) {
  ol_event_handle_t NewEvent{};
  ol_event_flags_t Flags{};
  callAndThrow(olCreateEvent, MOffloadQueue, Flags, &NewEvent);
  return EventImpl::createEventWithHandle(NewEvent, MDevice.getPlatformImpl(),
                                          std::move(Deps));
}
} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
