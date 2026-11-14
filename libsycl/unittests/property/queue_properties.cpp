//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/properties/queue_properties.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(QueueProperties, QueueWithoutPropertiesHoldsNone) {
  mock::MockWrapper Mock;

  const queue Q;
  EXPECT_FALSE(Q.is_in_order());
  EXPECT_FALSE(Q.has_property<property::queue::in_order>());
  EXPECT_FALSE(Q.has_property<property::queue::enable_profiling>());
}

TEST(QueueProperties, InOrderPropertyIsReported) {
  mock::MockWrapper Mock;

  const queue Q{property::queue::in_order()};
  EXPECT_TRUE(Q.is_in_order());
  EXPECT_TRUE(Q.has_property<property::queue::in_order>());
  EXPECT_FALSE(Q.has_property<property::queue::enable_profiling>());
}

TEST(QueueProperties, PropertiesSurviveEveryConstructorOverload) {
  mock::MockWrapper Mock;

  const device Device;
  const context Context = Device.get_platform().khr_get_default_context();
  const auto Selector = [](const device &) { return 1; };
  const property_list Props{property::queue::in_order()};

  EXPECT_TRUE(queue(Props).is_in_order());
  EXPECT_TRUE(queue(Selector, Props).is_in_order());
  EXPECT_TRUE(queue(Device, Props).is_in_order());
  EXPECT_TRUE(queue(Context, Device, Props).is_in_order());
}

// SYCL 2020 4.6.5.5. requires enable_profiling to throw
// errc::feature_not_supported when the device lacks aspect::queue_profiling.
// DeviceImpl::has() does not implement that aspect yet, so no device reports
// it. The queue rejects the property for the second reason too, because
// profiling itself is not implemented; see the TODO in QueueImpl.
TEST(QueueProperties, EnableProfilingWithoutAspectThrows) {
  mock::MockWrapper Mock;

  const device Device;
  ASSERT_FALSE(Device.has(aspect::queue_profiling));

  try {
    queue Q{property::queue::enable_profiling()};
    FAIL() << "enable_profiling must be rejected without queue_profiling";
  } catch (const exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::feature_not_supported));
  }
}

TEST(QueueProperties, GetPropertyThrowsWhenNotRequested) {
  mock::MockWrapper Mock;

  const queue Q;
  try {
    Q.get_property<property::queue::in_order>();
    FAIL() << "get_property() must throw when the queue lacks the property";
  } catch (const exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }
}

// A queue must reject a property that belongs to some other SYCL class. There
// is no such property implemented yet, so stand one up here: the point is that
// the check is driven by the property's own declaration.
namespace {
// Stand-in kind: the PropKind registry has no enumerator for a property of a
// SYCL class libsycl does not implement yet. It still has to fit the mask held
// by property_list, so the last kind the mask can hold is used.
class not_for_queue
    : public detail::property_of<
          not_for_queue, static_cast<detail::PropKind>(63), sycl::context> {};
} // namespace

TEST(QueueProperties, ForeignPropertyThrowsInvalid) {
  mock::MockWrapper Mock;

  static_assert(is_property_v<not_for_queue>);
  static_assert(!is_property_of_v<not_for_queue, queue>);

  // Constructing the list is fine; only passing it to a queue is an error.
  const property_list Props{not_for_queue()};
  EXPECT_TRUE(Props.has_property<not_for_queue>());

  try {
    queue Q{Props};
    FAIL() << "queue must reject a property it does not support";
  } catch (const exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }
}

// SYCL 2020 Table 12 constrains has_property() only to properties, so a
// property of another SYCL class must answer false rather than fail to
// compile or throw.
TEST(QueueProperties, ForeignPropertyQueryIsFalse) {
  mock::MockWrapper Mock;

  const queue Q;
  EXPECT_FALSE(Q.has_property<not_for_queue>());
  EXPECT_THROW(Q.get_property<not_for_queue>(), exception);
}

TEST(QueueProperties, ForeignPropertyIsRejectedAlongsideSupportedOnes) {
  mock::MockWrapper Mock;

  const property_list Props{property::queue::in_order(), not_for_queue()};
  EXPECT_THROW(queue{Props}, exception);
}
