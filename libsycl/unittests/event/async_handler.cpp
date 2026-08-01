//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <detail/event_impl.hpp>
#include <detail/global_objects.hpp>

#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/platform.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <exception>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace sycl;
using namespace ::testing;

namespace {

using EventImplPtr = std::shared_ptr<detail::EventImpl>;

EventImplPtr createEventImplWithHandle(detail::PlatformImpl &PlatformImpl) {
  ol_event_handle_t Handle = mock::createDummyHandle<ol_event_handle_t>();
  return detail::EventImpl::createEventWithHandle(Handle, PlatformImpl, {});
}

template <typename FlushAction>
void runAsyncExceptionFlushTest(mock::MockLiboffload &Mock,
                                const char *ErrorMsg, FlushAction Flush) {
  size_t NumExceptionsHandled = 0;
  std::vector<std::error_code> ReportedCodes;
  std::vector<std::string> ReportedMessages;
  auto expectSingleRuntimeErrorContaining = [&](const char *ExpectedMessage) {
    EXPECT_EQ(NumExceptionsHandled, 1u);
    ASSERT_THAT(ReportedCodes, SizeIs(1));
    ASSERT_THAT(ReportedMessages, SizeIs(1));
    EXPECT_EQ(ReportedCodes.back(), make_error_code(errc::runtime));
    EXPECT_THAT(ReportedMessages.back(), HasSubstr(ExpectedMessage));
  };

  async_handler Handler = [&](exception_list Exceptions) {
    for (const auto &ExceptionPtr : Exceptions) {
      ++NumExceptionsHandled;
      try {
        std::rethrow_exception(ExceptionPtr);
      } catch (const sycl::exception &E) {
        ReportedCodes.push_back(E.code());
        ReportedMessages.emplace_back(E.what());
      } catch (...) {
        ADD_FAILURE() << "Unexpected exception type in async_handler";
      }
    }
  };

  auto Device = device(default_selector_v);
  queue Q(Device, Handler);
  auto QImpl = detail::getSyclObjImpl(Q);

  platform P = Device.get_platform();
  EventImplPtr EImpl = createEventImplWithHandle(*detail::getSyclObjImpl(P));
  event E = detail::createSyclObjFromImpl<event>(EImpl);

  detail::recordAsyncException(
      QImpl, std::make_exception_ptr(
                 exception(make_error_code(errc::runtime), ErrorMsg)));

  Flush(Mock, Q, QImpl, E, EImpl);
  expectSingleRuntimeErrorContaining(ErrorMsg);
}

} // namespace

TEST(EventAsyncHandler, QueueWaitAndThrow) {
  mock::MockWrapper Mock;
  constexpr const char *QueueWaitAndThrowErrorMsg =
      "queue wait_and_throw error";

  runAsyncExceptionFlushTest(Mock.get(), QueueWaitAndThrowErrorMsg,
                             [](mock::MockLiboffload &Mock, queue &Q,
                                const auto &, const auto &, const auto &) {
                               EXPECT_CALL(Mock, olSyncQueue(_)).Times(1);
                               Q.wait_and_throw();
                             });
}

TEST(EventAsyncHandler, EventWaitAndThrow) {
  mock::MockWrapper Mock;
  constexpr const char *EventWaitAndThrowErrorMsg =
      "event wait_and_throw error";

  runAsyncExceptionFlushTest(
      Mock.get(), EventWaitAndThrowErrorMsg,
      [](mock::MockLiboffload &Mock, queue &, const auto &, event &E,
         EventImplPtr &EImpl) {
        EXPECT_CALL(Mock, olSyncQueue(_)).Times(0);
        EXPECT_CALL(Mock, olSyncEvent(EImpl->getHandle())).Times(1);
        E.wait_and_throw();
      });
}

TEST(EventAsyncHandler, QueueThrowAsynchronous) {
  mock::MockWrapper Mock;
  constexpr const char *QueueThrowAsynchronousErrorMsg =
      "queue throw_asynchronous error";

  runAsyncExceptionFlushTest(Mock.get(), QueueThrowAsynchronousErrorMsg,
                             [](mock::MockLiboffload &Mock, queue &Q,
                                const auto &, const auto &, const auto &) {
                               EXPECT_CALL(Mock, olSyncQueue(_)).Times(0);
                               Q.throw_asynchronous();
                             });
}

TEST(EventAsyncHandler, DeadQueueFallsBackToDefaultAsyncHandler) {
// EXPECT_DEATH is not supported on Windows.
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(
      {
        mock::MockWrapper Mock;
        async_handler Handler = [](exception_list) {
          ADD_FAILURE() << "Queue async_handler should not be called";
        };
        {
          queue Q(device(default_selector_v), Handler);
          auto QImpl = detail::getSyclObjImpl(Q);
          detail::recordAsyncException(
              QImpl,
              std::make_exception_ptr(exception(make_error_code(errc::runtime),
                                                "fallback async error")));
        }

        detail::flushAsyncExceptions();
      },
      "Default async_handler caught exceptions:\n\tfallback async error");
#else
  GTEST_SKIP() << "Death tests are not supported on this platform";
#endif
}
