//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -fsyntax-only -Xclang -verify \
// RUN: -Xclang -verify-ignore-unexpected=note

#include <sycl/sycl.hpp>

#include <type_traits>

// Stands in for a property of some SYCL class other than queue; no such
// property is implemented yet. The kind has to fit the mask held by
// property_list, so the last kind the mask can hold is used.
class not_for_queue
    : public sycl::detail::property_of<not_for_queue,
                                       static_cast<sycl::detail::PropKind>(63),
                                       sycl::context> {};

// A property that names no SYCL class could never be passed to anything.
// expected-error@*:* {{A property must name at least one SYCL class that accepts it.}}
class for_nothing : public sycl::detail::property_of<
                        for_nothing, static_cast<sycl::detail::PropKind>(62)> {
};

// A property without data is identified by a bit of its kind, so a kind the
// mask cannot hold is rejected rather than silently dropped.
class out_of_mask
    : public sycl::detail::property_of<
          out_of_mask, static_cast<sycl::detail::PropKind>(64), sycl::queue> {};

// expected-error@*:* {{The property kind does not fit the mask held by property_list.}}
void testPropertyKindOutOfMask() {
  sycl::property_list Props{out_of_mask()};
  (void)Props;
}

// SYCL 2020 Table 13 makes the property_list constructor available only when
// every argument is a property, so a non-property removes it from the overload
// set instead of being silently dropped.
void testNonProperty() {
  sycl::property_list Props{42}; // expected-error {{no matching constructor}}
  (void)Props;
}

static_assert(!std::is_constructible_v<sycl::property_list, int>,
              "the constructor must be constrained, not merely diagnosed");
static_assert(std::is_constructible_v<sycl::property_list,
                                      sycl::property::queue::in_order>);

// Querying a queue for a property of another SYCL class is well-formed and
// returns false, per SYCL 2020 Table 12; only non-properties are rejected.
void testForeignPropertyQuery(const sycl::queue &Q) {
  (void)Q.has_property<not_for_queue>();
}

// has_property() and get_property() are constrained to properties, so a
// non-property is a substitution failure rather than a hard error.
void testNonPropertyQuery(const sycl::property_list &Props) {
  // expected-error@+1 {{no matching member function for call to 'has_property'}}
  (void)Props.has_property<float>();
}

void testNonPropertyQueryOnQueue(const sycl::queue &Q) {
  // expected-error@+1 {{no matching member function for call to 'get_property'}}
  (void)Q.get_property<float>();
}
