//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/scoped_binary_registration.hpp>
#include <mock/helpers.hpp>

#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/platform.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace sycl;
using namespace ::testing;

using EventImplPtr = std::shared_ptr<detail::EventImpl>;

static EventImplPtr
createEventImplWithHandle(detail::PlatformImpl &PlatformImpl,
                          std::vector<EventImplPtr> WaitList = {}) {
  ol_event_handle_t Handle = mock::createDummyHandle<ol_event_handle_t>();
  return detail::EventImpl::createEventWithHandle(Handle, PlatformImpl,
                                                  std::move(WaitList));
}

// MockQueue to expose protected methods for testing
class sycl::detail::MockQueue : public sycl::queue {
public:
  using sycl::queue::getLastEvent;
  using sycl::queue::setKernelParameters;
  using sycl::queue::sycl_kernel_launch;
};

TEST(Event, CopyMoveAssign) {
  mock::MockWrapper Mock;

  event Event;
  const size_t Hash = std::hash<event>{}(Event);
  event MovedConstructed(std::move(Event));
  EXPECT_EQ(Hash, std::hash<event>{}(MovedConstructed));

  event AssignedSource;
  const size_t AssignedHash = std::hash<event>{}(AssignedSource);
  event MoveAssigned;
  MoveAssigned = std::move(AssignedSource);
  EXPECT_EQ(AssignedHash, std::hash<event>{}(MoveAssigned));

  event CopiedSource;
  const size_t CopiedHash = std::hash<event>{}(CopiedSource);
  event CopyConstructed(CopiedSource);
  EXPECT_EQ(CopiedHash, std::hash<event>{}(CopiedSource));
  EXPECT_EQ(CopiedHash, std::hash<event>{}(CopyConstructed));
  EXPECT_EQ(CopiedSource, CopyConstructed);

  event CopyAssigned;
  CopyAssigned = CopiedSource;
  EXPECT_EQ(CopiedHash, std::hash<event>{}(CopyAssigned));
  EXPECT_EQ(CopiedSource, CopyAssigned);
}

TEST(Event, WaitAPIsForDefaultConstructedEvent) {
  mock::MockWrapper Mock;

  event Event;
  EXPECT_CALL(Mock.get(), olSyncEvent(_)).Times(0);

  EXPECT_NO_THROW(Event.wait());
  EXPECT_NO_THROW(Event.wait_and_throw());
  EXPECT_TRUE(Event.get_wait_list().empty());

  std::vector<event> EventList = {Event};
  EXPECT_NO_THROW(event::wait(EventList));
  EXPECT_NO_THROW(event::wait_and_throw(EventList));
}

TEST(Event, GetWaitListWithSetKernelParametersAndLaunch) {
  static constexpr char TestKernelWithDeps[] = "TestKernelWithDeps";
  mock::MockWrapper Mock;
  sycl::unittests::ScopedKernelRegistration Registration{TestKernelWithDeps};

  platform P = device(default_selector_v).get_platform();
  auto &PlatformImpl = *detail::getSyclObjImpl(P);

  detail::MockQueue Q;

  EventImplPtr Dep1Impl = createEventImplWithHandle(PlatformImpl);
  EventImplPtr Dep2Impl = createEventImplWithHandle(PlatformImpl);
  event Dep1 = detail::createSyclObjFromImpl<event>(Dep1Impl);
  event Dep2 = detail::createSyclObjFromImpl<event>(Dep2Impl);

  struct KernelData {
    int Value = 42;
  } Data;

  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, 1, _, _))
      .WillOnce([](ol_queue_handle_t Queue, ol_device_handle_t Device,
                   ol_symbol_handle_t Kernel,
                   const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                   const ol_kernel_launch_prop_t *Properties, size_t NumArgs,
                   void **ArgPtrs, const size_t *ArgSizes) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(Kernel, nullptr);
        return OL_SUCCESS;
      });

  std::vector<event> DepEvents = {Dep1, Dep2};

  Q.setKernelParameters(DepEvents);
  Q.sycl_kernel_launch<class TestKernelWithDeps>(TestKernelWithDeps, Data);
  event KernelEvent = Q.getLastEvent();

  std::vector<event> WaitList = KernelEvent.get_wait_list();
  ASSERT_THAT(WaitList, SizeIs(2));
  EXPECT_EQ(WaitList[0], Dep1);
  EXPECT_EQ(WaitList[1], Dep2);

  auto KernelEventImpl = detail::getSyclObjImpl(KernelEvent);
  EXPECT_CALL(Mock.get(), olSyncEvent(KernelEventImpl->getHandle())).Times(1);
  KernelEvent.wait();

  EXPECT_CALL(Mock.get(), olSyncEvent(Dep1Impl->getHandle())).Times(1);
  EXPECT_CALL(Mock.get(), olSyncEvent(Dep2Impl->getHandle())).Times(1);
  event::wait(WaitList);
}

template <typename Param>
void expectGetProfilingInfoThrows(const event &Event) {
  try {
    (void)Event.template get_profiling_info<Param>();
    FAIL() << "Expected sycl::exception";
  } catch (const exception &Ex) {
    EXPECT_EQ(Ex.code(), make_error_code(errc::feature_not_supported));
    EXPECT_THAT(std::string(Ex.what()),
                HasSubstr("Profiling features are not supported."));
  }
}

TEST(Event, GetProfilingInfoThrows) {
  mock::MockWrapper Mock;
  event Event;

  expectGetProfilingInfoThrows<info::event_profiling::command_submit>(Event);
  expectGetProfilingInfoThrows<info::event_profiling::command_start>(Event);
  expectGetProfilingInfoThrows<info::event_profiling::command_end>(Event);
}
