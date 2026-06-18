//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <detail/event_impl.hpp>

#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/platform.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
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

TEST(Event, WaitAndWaitListForRuntimeEvents) {
  mock::MockWrapper Mock;

  platform P = device(default_selector_v).get_platform();
  auto &PlatformImpl = *detail::getSyclObjImpl(P);

  EventImplPtr Dep1Impl = createEventImplWithHandle(PlatformImpl);
  EventImplPtr Dep2Impl = createEventImplWithHandle(PlatformImpl);
  std::vector<EventImplPtr> WaitList = {Dep1Impl, Dep2Impl};
  EventImplPtr RootImpl =
      createEventImplWithHandle(PlatformImpl, std::move(WaitList));

  event Dep1 = detail::createSyclObjFromImpl<event>(Dep1Impl);
  event Dep2 = detail::createSyclObjFromImpl<event>(Dep2Impl);
  event Root = detail::createSyclObjFromImpl<event>(RootImpl);

  std::vector<event> RetrievedWaitList = Root.get_wait_list();
  ASSERT_THAT(RetrievedWaitList, SizeIs(2));
  EXPECT_EQ(RetrievedWaitList[0], Dep1);
  EXPECT_EQ(RetrievedWaitList[1], Dep2);

  EXPECT_CALL(Mock.get(), olSyncEvent(RootImpl->getHandle())).Times(2);
  Root.wait();
  Root.wait_and_throw();

  EXPECT_CALL(Mock.get(), olSyncEvent(Dep1Impl->getHandle())).Times(2);
  EXPECT_CALL(Mock.get(), olSyncEvent(Dep2Impl->getHandle())).Times(2);
  std::vector<event> DepEvents = {Dep1, Dep2};
  event::wait(DepEvents);
  event::wait_and_throw(DepEvents);
}

// TODO: test it all with queue::single_task with deps
