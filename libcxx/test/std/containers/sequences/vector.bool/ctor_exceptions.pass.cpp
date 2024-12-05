//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// (bug report: https://llvm.org/PR58392)
// Check that vector<bool> constructors don't leak memory when an operation inside the constructor throws an exception

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include "../vector/common.h"
#include "count_new.h"
#include "test_iterators.h"

int main(int, char**) {
  using AllocVec = std::vector<bool, throwing_allocator<bool> >;

  try { // Throw in vector() from allocator
    AllocVec vec;
  } catch (int) {
  }
  check_new_delete_called();

#if TEST_STD_VER >= 14
  try { // Throw in vector(size_type, const allocator_type&) from allocator
    throwing_allocator<bool> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec get_alloc(0, alloc);
  } catch (int) {
  }
  check_new_delete_called();
#endif // TEST_STD_VER >= 14

  try { // Throw in vector(size_type, const value_type&, const allocator_type&) from allocator
    throwing_allocator<bool> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec get_alloc(0, true, alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from input iterator
    std::vector<bool> vec(
        throwing_iterator<bool, std::input_iterator_tag>(), throwing_iterator<bool, std::input_iterator_tag>(2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from forward iterator
    std::vector<bool> vec(
        throwing_iterator<bool, std::forward_iterator_tag>(), throwing_iterator<bool, std::forward_iterator_tag>(2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator) from allocator
    bool a[] = {true, true};
    AllocVec vec(cpp17_input_iterator<bool*>(a), cpp17_input_iterator<bool*>(a + 2));
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from input iterator
    std::allocator<bool> alloc;
    std::vector<bool> vec(
        throwing_iterator<bool, std::input_iterator_tag>(), throwing_iterator<bool, std::input_iterator_tag>(2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from forward iterator
    std::allocator<bool> alloc;
    std::vector<bool> vec(throwing_iterator<bool, std::forward_iterator_tag>(),
                          throwing_iterator<bool, std::forward_iterator_tag>(2),
                          alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from allocator
    bool a[] = {true, false};
    throwing_allocator<bool> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec vec(cpp17_input_iterator<bool*>(a), cpp17_input_iterator<bool*>(a + 2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(InputIterator, InputIterator, const allocator_type&) from allocator
    bool a[] = {true, false};
    throwing_allocator<bool> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec vec(forward_iterator<bool*>(a), forward_iterator<bool*>(a + 2), alloc);
  } catch (int) {
  }
  check_new_delete_called();

#if TEST_STD_VER >= 11
  try { // Throw in vector(const vector&, const allocator_type&) from allocator
    throwing_allocator<bool> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ false);
    AllocVec vec(alloc);
    vec.push_back(true);
    alloc.throw_on_copy_ = true;
    AllocVec vec2(vec, alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(vector&&, const allocator_type&) from allocator
    throwing_allocator<bool> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ false);
    AllocVec vec(alloc);
    vec.push_back(true);
    alloc.throw_on_copy_ = true;
    AllocVec vec2(std::move(vec), alloc);
  } catch (int) {
  }
  check_new_delete_called();

  try { // Throw in vector(initializer_list<value_type>, const allocator_type&) constructor from allocator
    throwing_allocator<bool> alloc(/*throw_on_ctor = */ false, /*throw_on_copy = */ true);
    AllocVec vec({true, true}, alloc);
  } catch (int) {
  }
  check_new_delete_called();
#endif // TEST_STD_VER >= 11

  return 0;
}
