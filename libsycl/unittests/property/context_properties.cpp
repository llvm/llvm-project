//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/properties/queue_properties.hpp>
#include <sycl/__impl/property_list.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

// SYCL 2020 defines no context properties, so a context accepts an empty list
// only.
TEST(ContextProperties, EmptyListIsAccepted) {
  mock::MockWrapper Mock;

  const device Dev;
  context Ctx{Dev, property_list{}};
}

TEST(ContextProperties, QueuePropertyThrowsInvalid) {
  mock::MockWrapper Mock;

  const device Dev;
  const property_list Props{property::queue::in_order()};

  try {
    context Ctx{Dev, Props};
    FAIL() << "context must reject a property it does not support";
  } catch (const exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }
}
