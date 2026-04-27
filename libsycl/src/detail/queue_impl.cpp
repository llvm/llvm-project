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
      GroupSize[I] = static_cast<uint32_t>(Range.MLocalSize[I]);
      assert(GroupSize[I] && "Local range contains zero size. Causes division "
                             "by zero for group range counting.");
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

void QueueImpl::wait() { callAndThrow(olSyncQueue, MOffloadQueue); }

static bool checkEventsPlatformMatch(std::vector<EventImplPtr> &Events,
                                     const PlatformImpl &QueuePlatform) {
  // liboffload limitation to olWaitEvents. We can't do any extra handling for
  // cross context/platform events without host task support now.
  //   "The input events can be from any queue on any device provided by the
  //   same platform as `Queue`."
  return std::all_of(Events.cbegin(), Events.cend(),
                     [&QueuePlatform](const EventImplPtr &Event) {
                       return &Event->getPlatformImpl() == &QueuePlatform;
                     });
}

void QueueImpl::setKernelParameters(std::vector<EventImplPtr> &&Events,
                                    const detail::UnifiedRangeView &Range) {
  if (!checkEventsPlatformMatch(Events, MDevice.getPlatformImpl()))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "libsycl doesn't support cross-context/platform event dependencies "
        "now.");

  // TODO: this conversion and storing of only offload events is possible only
  // while we don't have host tasks (or features based on host tasks, like
  // streams). With them - it is very likely we should copy EventImplPtr
  // (shared_ptr) and keep it here. Although it may differ if host tasks will be
  // implemented on offload level (no data now).
  assert(MCurrentSubmitInfo.DepEvents.empty() &&
         "Kernel submission must clean up dependencies.");
  MCurrentSubmitInfo.DepEvents.reserve(Events.size());
  for (auto &Event : Events) {
    assert(Event && "Event impl object can't be nullptr");
    MCurrentSubmitInfo.DepEvents.push_back(Event->getHandle());
  }
  setKernelLaunchArgs(Range, MCurrentSubmitInfo.Range);
}

void QueueImpl::submitKernelImpl(DeviceKernelInfo &KernelInfo, void *ArgData,
                                 size_t ArgSize) {
  ol_symbol_handle_t Kernel =
      detail::ProgramAndKernelManager::getInstance().getOrCreateKernel(
          KernelInfo, MDevice);
  assert(Kernel);

  // TODO: liboffload supports only in-order queues and no cross context waiting
  // is available now that means that this code is excessive but correct. I
  // don't want to skip it and rely on default liboffload behaviour that is
  // applicable for in-order queue only. Once OOO queues are added this waiting
  // must be disabled for in-order queues. Once host tasks are added - cross
  // context dependencies should be enabled and checked as well.
  if (!MCurrentSubmitInfo.DepEvents.empty()) {
    callAndThrow(olWaitEvents, MOffloadQueue,
                 MCurrentSubmitInfo.DepEvents.data(),
                 MCurrentSubmitInfo.DepEvents.size());
  }

  assert(ArgData && "At least one argument must exist");
  assert(ArgSize && "Arguments size must be greater than 0");

  // ol_kernel_launch_prop_t Props[2];
  // Props[0].type = OL_KERNEL_LAUNCH_PROP_TYPE_SIZE;
  // Props[0].data = &ArgSize;
  // Props[1] = OL_KERNEL_LAUNCH_PROP_END;
  auto Result =
      olLaunchKernel(MOffloadQueue, MDevice.getOLHandle(), Kernel, &ArgData,
                     ArgSize, &MCurrentSubmitInfo.Range /*, Props*/);
  // Clean up current kernel submit data to prepare structures for next
  // submission.
  MCurrentSubmitInfo.DepEvents.clear();
  MCurrentSubmitInfo.Range = {};
  if (isFailed(Result))
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          std::string("Kernel submission (") +
                              KernelInfo.getName().data() + ") failed with " +
                              formatCodeString(Result));

  ol_event_handle_t NewEvent{};
  callAndThrow(olCreateEvent, MOffloadQueue, &NewEvent);

  MCurrentSubmitInfo.LastEvent =
      EventImpl::createEventWithHandle(NewEvent, MDevice.getPlatformImpl());
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
