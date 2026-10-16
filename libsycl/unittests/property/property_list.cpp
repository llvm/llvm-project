//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/properties/queue_properties.hpp>
#include <sycl/__impl/property_list.hpp>
#include <sycl/__impl/property_traits.hpp>
#include <sycl/__impl/queue.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using namespace sycl;

// Stands in for a property of a SYCL class other than queue that carries data.
// No such property exists yet: every data-carrying property in SYCL 2020
// belongs to buffer or image, and libsycl has neither class. One is defined
// here so that the storage path for properties with data is still covered.
namespace {
// Stand-in kinds: the PropKind registry has no enumerator for a property of a
// SYCL class libsycl does not implement yet.
constexpr auto BoundToKind = static_cast<detail::PropKind>(1000);

class bound_to
    : public detail::property_of<bound_to, BoundToKind, sycl::context> {
public:
  explicit bound_to(int Id) : MId(Id) {}

  int get_id() const { return MId; }

private:
  int MId;
};
} // namespace

static_assert(!detail::propertyNeedsStorage<property::queue::in_order>);
static_assert(detail::propertyNeedsStorage<bound_to>);

static_assert(is_property_v<bound_to>);
static_assert(is_property_of_v<bound_to, context>);
static_assert(!is_property_of_v<bound_to, queue>);

// A non-property must not merely be diagnosed: SYCL 2020 Table 13 phrases the
// constructor as available only when every argument is a property.
static_assert(!std::is_constructible_v<property_list, int>);
static_assert(std::is_constructible_v<property_list, bound_to>);
static_assert(
    std::is_constructible_v<property_list, property::queue::in_order>);

// A property's identity is its kind, a value both libsycl and its users read
// from the same header, so it survives a shared library boundary.
static_assert(detail::getPropKind<property::queue::in_order>() ==
              detail::PropKind::QueueInOrder);
static_assert(detail::getPropKind<property::queue::enable_profiling>() ==
              detail::PropKind::QueueEnableProfiling);
// Every implemented property needs its own entry in the registry; a copy-paste
// in PropKind would silently alias two properties.
static_assert(detail::getPropKind<property::queue::in_order>() !=
              detail::getPropKind<property::queue::enable_profiling>());

static_assert(is_property_v<property::queue::in_order>);
static_assert(is_property_v<property::queue::enable_profiling>);
static_assert(!is_property_v<int>);
static_assert(!is_property_v<queue>);

static_assert(is_property_of_v<property::queue::in_order, queue>);
static_assert(is_property_of_v<property::queue::enable_profiling, queue>);
static_assert(!is_property_of_v<property::queue::in_order, context>);
static_assert(!is_property_of_v<int, queue>);

TEST(PropertyList, EmptyListHoldsNothing) {
  const property_list Props;

  EXPECT_FALSE(Props.has_property<property::queue::in_order>());
  EXPECT_FALSE(Props.has_property<property::queue::enable_profiling>());
}

TEST(PropertyList, DefaultArgumentIsEmpty) {
  // The spelling SYCL classes use for their defaulted property_list argument.
  const property_list Props = {};

  EXPECT_FALSE(Props.has_property<property::queue::in_order>());
}

TEST(PropertyList, HasOnlyTheGivenProperty) {
  const property_list Props{property::queue::in_order()};

  EXPECT_TRUE(Props.has_property<property::queue::in_order>());
  EXPECT_FALSE(Props.has_property<property::queue::enable_profiling>());
}

TEST(PropertyList, HoldsSeveralProperties) {
  const property_list Props{property::queue::in_order(),
                            property::queue::enable_profiling()};

  EXPECT_TRUE(Props.has_property<property::queue::in_order>());
  EXPECT_TRUE(Props.has_property<property::queue::enable_profiling>());
}

TEST(PropertyList, GetPropertyReturnsHeldProperty) {
  const property_list Props{property::queue::in_order()};

  // A dataless property has no observable state; reaching this point without
  // an exception is what get_property() promises.
  const auto Prop = Props.get_property<property::queue::in_order>();
  static_assert(
      std::is_same_v<std::decay_t<decltype(Prop)>, property::queue::in_order>);
  (void)Prop;
}

TEST(PropertyList, GetMissingPropertyThrowsInvalid) {
  const property_list Props{property::queue::in_order()};

  try {
    Props.get_property<property::queue::enable_profiling>();
    FAIL() << "get_property() must throw for a property that is not held";
  } catch (const exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }
}

TEST(PropertyList, HoldsPropertyWithData) {
  const property_list Props{bound_to(42)};

  ASSERT_TRUE(Props.has_property<bound_to>());
  // The value has to survive being erased into the list and recovered, which
  // is what distinguishes a property with data from a dataless one.
  EXPECT_EQ(Props.get_property<bound_to>().get_id(), 42);
}

TEST(PropertyList, KeepsDataAndDatalessPropertiesApart) {
  const property_list Props{property::queue::in_order(), bound_to(7)};

  EXPECT_TRUE(Props.has_property<property::queue::in_order>());
  EXPECT_EQ(Props.get_property<bound_to>().get_id(), 7);
}

TEST(PropertyList, CopyPreservesPropertyData) {
  const property_list Props{bound_to(5)};
  const property_list Copy(Props);

  EXPECT_EQ(Copy.get_property<bound_to>().get_id(), 5);
}

// Nothing bounds the number of properties a list may hold.
TEST(PropertyList, HoldsManyProperties) {
  const property_list Props{
      bound_to(0),  bound_to(1),  bound_to(2),  bound_to(3),  bound_to(4),
      bound_to(5),  bound_to(6),  bound_to(7),  bound_to(8),  bound_to(9),
      bound_to(10), bound_to(11), bound_to(12), bound_to(13), bound_to(14)};

  ASSERT_TRUE(Props.has_property<bound_to>());
  EXPECT_EQ(Props.get_property<bound_to>().get_id(), 0);
}

TEST(PropertyList, CopyPreservesProperties) {
  const property_list Props{property::queue::in_order()};

  // Must select the copy constructor rather than the variadic one, which
  // would otherwise treat the list itself as a property and fail to compile.
  const property_list Copy(Props);
  EXPECT_TRUE(Copy.has_property<property::queue::in_order>());

  property_list Assigned;
  Assigned = Props;
  EXPECT_TRUE(Assigned.has_property<property::queue::in_order>());
}
