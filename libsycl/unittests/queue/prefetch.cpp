#include <mock/helpers.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/queue.hpp>

#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(Queue, Prefetch) {}