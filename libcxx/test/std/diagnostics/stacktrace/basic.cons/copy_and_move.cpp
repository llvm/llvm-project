//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  basic_stacktrace(const basic_stacktrace& other);
  basic_stacktrace(basic_stacktrace&& other) noexcept;                                    
  basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);           
  basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);                
  basic_stacktrace& operator=(const basic_stacktrace& other);                             
  basic_stacktrace& operator=(basic_stacktrace&& other)
    noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<Allocator>::is_always_equal::value);                               
*/

#include <cassert>
#include <stacktrace>

// clang-format off
uint32_t test1_line;
uint32_t test2_line;

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE
std::basic_stacktrace<A> test1(A& alloc) {
  test1_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret = std::basic_stacktrace<A>::current(alloc);
  return ret;
}

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE
std::basic_stacktrace<A> test2(A& alloc) {
  test2_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret = test1(alloc);
  return ret;
}

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_copy_move_ctors() {
  using A = std::allocator<std::stacktrace_entry>;
  A alloc;
  auto st = std::basic_stacktrace<A>::current(alloc);

  auto copy_constr = std::basic_stacktrace<A>(st);
  assert(st == copy_constr);

  std::basic_stacktrace<A> copy_assign;
  copy_assign = std::basic_stacktrace<A>(st);
  assert(st == copy_assign);

  auto st2 = test2(alloc);
  assert(st2.size());
  std::basic_stacktrace<A> move_constr(std::move(st2));
  assert(move_constr.size());
  assert(!st2.size());

  auto st3 = test2(alloc);
  assert(st3.size());
  std::basic_stacktrace<A> move_assign;
  move_assign = std::move(st3);
  assert(move_assign.size());
  assert(!st3.size());

  // TODO(stacktrace23): should we add test cases with `select_on_container_copy_construction`?
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  test_copy_move_ctors();
  return 0;
}
