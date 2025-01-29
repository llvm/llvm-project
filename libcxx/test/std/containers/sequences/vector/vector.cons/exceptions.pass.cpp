//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// (bug report: https://llvm.org/PR58392)
// Check that vector constructors don't leak memory when an operation inside the constructor throws an exception

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "count_new.h"
#include "test_allocator.h"
#include "test_iterators.h"

int main(int, char**) {
  using AllocVec = std::vector<int, throwing_allocator<int> >;
  try { // vector()
    AllocVec vec;
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(size_type) from type
    std::vector<throwing_t> get_alloc(1);
  } catch (int) {
  }
  check_new_delete_called();

#if TEST_STD_VER >= 14
  try { // Throw in vector(size_type, value_type) from type
    int throw_after = 1;
    throwing_t v(throw_after);
    std::vector<throwing_t> get_alloc(1, v);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(size_type, const allocator_type&) from allocator
    throwing_allocator<int> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec get_alloc(0, alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(size_type, const allocator_type&) from the type
    std::vector<throwing_t> vec(1, std::allocator<throwing_t>());
  } catch (int) {
  }
  check_new_delete_called();
#endif // TEST_STD_VER >= 14

  try { // Throw in vector(size_type, value_type, const allocator_type&) from the type
    int throw_after = 1;
    throwing_t v(throw_after);
    std::vector<throwing_t> vec(1, v, std::allocator<throwing_t>());
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from input iterator
    std::vector<int> vec(
        (throwing_iterator<int, std::input_iterator_tag>()), throwing_iterator<int, std::input_iterator_tag>(2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from forward iterator
    std::vector<int> vec(
        (throwing_iterator<int, std::forward_iterator_tag>()), throwing_iterator<int, std::forward_iterator_tag>(2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from allocator
    int a[] = {1, 2};
    AllocVec vec(cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from input iterator
    std::allocator<int> alloc;
    std::vector<int> vec(
        throwing_iterator<int, std::input_iterator_tag>(), throwing_iterator<int, std::input_iterator_tag>(2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from forward iterator
    std::allocator<int> alloc;
    std::vector<int> vec(throwing_iterator<int, std::forward_iterator_tag>(),
                         throwing_iterator<int, std::forward_iterator_tag>(2),
                         alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from allocator
    int a[] = {1, 2};
    throwing_allocator<int> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec vec(cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from allocator
    int a[] = {1, 2};
    throwing_allocator<int> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec vec(forward_iterator<int*>(a), forward_iterator<int*>(a + 2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(const vector&) from type
    std::vector<throwing_t> vec;
    int throw_after = 1;
    vec.emplace_back(throw_after);
    auto vec2 = vec;
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(const vector&, const allocator_type&) from type
    std::vector<throwing_t> vec;
    int throw_after = 1;
    vec.emplace_back(throw_after);
    std::vector<throwing_t> vec2(vec, std::allocator<int>());
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(vector&&, const allocator_type&) from type during element-wise move
    std::vector<throwing_t, test_allocator<throwing_t> > vec(test_allocator<throwing_t>(1));
    int throw_after = 10;
    throwing_t v(throw_after);
    vec.insert(vec.end(), 6, v);
    std::vector<throwing_t, test_allocator<throwing_t> > vec2(std::move(vec), test_allocator<throwing_t>(2));
  } catch (int) {
  }
  check_new_delete_called();

#if TEST_STD_VER >= 11
  try { // Throw in vector(initializer_list<value_type>) from type
    int throw_after = 1;
    std::vector<throwing_t> vec({throwing_t(throw_after)});
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(initializer_list<value_type>, const allocator_type&) constructor from type
    int throw_after = 1;
    std::vector<throwing_t> vec({throwing_t(throw_after)}, std::allocator<throwing_t>());
  } catch (int) {
  }
  check_new_delete_called();
#endif // TEST_STD_VER >= 11

  return 0;
}
