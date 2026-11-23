//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions
// UNSUPPORTED: c++03, libcpp-hardening-mode=none, libcpp-hardening-mode=fast

#include <deque>
#include <forward_list>
#include <set>
#include <vector>

#include "check_assertion.h"
#include "test_allocator.h"

template <class Container>
void test_swap(const char* message) {
  using Alloc = typename Container::allocator_type;
  Container c1(Alloc(1));
  Container c2(Alloc(2));
  (void)message; // The quick-enforce semantic checks the trap without matching a message.
  TEST_LIBCPP_ASSERT_FAILURE(c1.swap(c2), message);
}

int main(int, char**) {
  test_swap<std::deque<int, test_allocator<int> > >(
      "deque::swap: Either propagate_on_container_swap must be true or the allocators must compare equal");
  test_swap<std::forward_list<int, test_allocator<int> > >(
      "forward_list::swap: Either propagate_on_container_swap must be true or the allocators must compare equal");
  test_swap<std::set<int, std::less<int>, test_allocator<int> > >(
      "ordered container::swap: Either propagate_on_container_swap must be true or the allocators must compare equal");
  test_swap<std::vector<bool, test_allocator<bool> > >(
      "vector<bool>::swap: Either propagate_on_container_swap must be true or the allocators must compare equal");

  return 0;
}
