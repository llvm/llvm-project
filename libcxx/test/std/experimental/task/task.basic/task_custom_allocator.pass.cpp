// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

#include <experimental/task>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <memory>
#include <experimental/memory_resource>

#include "../sync_wait.hpp"

namespace coro = std::experimental::coroutines_v1;

namespace
{
  static size_t allocator_instance_count = 0;

  // A custom allocator that tracks the number of allocator instances that
  // have been constructed/destructed as well as the number of bytes that
  // have been allocated/deallocated using the allocator.
  template<typename T>
  class my_allocator {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using is_always_equal = std::false_type;

    explicit my_allocator(
      std::shared_ptr<size_type> totalAllocated) noexcept
      : totalAllocated_(std::move(totalAllocated))
    {
      ++allocator_instance_count;
      assert(totalAllocated_);
    }

    my_allocator(const my_allocator& other)
    : totalAllocated_(other.totalAllocated_)
    {
      ++allocator_instance_count;
    }

    my_allocator(my_allocator&& other)
    : totalAllocated_(std::move(other.totalAllocated_))
    {
      ++allocator_instance_count;
    }

    template<typename U>
    my_allocator(const my_allocator<U>& other)
    : totalAllocated_(other.totalAllocated_)
    {
      ++allocator_instance_count;
    }

    template<typename U>
    my_allocator(my_allocator<U>&& other)
    : totalAllocated_(std::move(other.totalAllocated_))
    {
      ++allocator_instance_count;
    }

    ~my_allocator()
    {
      --allocator_instance_count;
    }

    char* allocate(size_t n) {
      const auto byteCount = n * sizeof(T);
      void* p = std::malloc(byteCount);
      if (!p) {
        throw std::bad_alloc{};
      }
      *totalAllocated_ += byteCount;
      return static_cast<char*>(p);
    }

    void deallocate(char* p, size_t n) {
      const auto byteCount = n * sizeof(T);
      *totalAllocated_ -= byteCount;
      std::free(p);
    }
  private:
    template<typename U>
    friend class my_allocator;

    std::shared_ptr<size_type> totalAllocated_;
  };
}

template<typename Allocator>
coro::task<void> f(std::allocator_arg_t, [[maybe_unused]] Allocator alloc)
{
  co_return;
}

void test_custom_allocator_is_destructed()
{
  auto totalAllocated = std::make_shared<size_t>(0);

  assert(allocator_instance_count == 0);

  {
    std::vector<coro::task<>> tasks;
    tasks.push_back(
      f(std::allocator_arg, my_allocator<char>{ totalAllocated }));
    tasks.push_back(
      f(std::allocator_arg, my_allocator<char>{ totalAllocated }));

    assert(allocator_instance_count == 4);
    assert(*totalAllocated > 0);
  }

  assert(allocator_instance_count == 0);
  assert(*totalAllocated == 0);
}

void test_custom_allocator_type_rebinding()
{
  auto totalAllocated = std::make_shared<size_t>(0);
  {
    std::vector<coro::task<>> tasks;
    tasks.emplace_back(
      f(std::allocator_arg, my_allocator<int>{ totalAllocated }));
    coro::sync_wait(tasks[0]);
  }
  assert(*totalAllocated == 0);
  assert(allocator_instance_count == 0);
}

void test_mixed_custom_allocator_type_erasure()
{
  assert(allocator_instance_count == 0);

  // Show that different allocators can be used within a vector of tasks
  // of the same type. ie. that the allocator is type-erased inside the
  // coroutine.
  std::vector<coro::task<>> tasks;
  tasks.push_back(f(
    std::allocator_arg, std::allocator<char>{}));
  tasks.push_back(f(
    std::allocator_arg,
    std::experimental::pmr::polymorphic_allocator<char>{
       std::experimental::pmr::new_delete_resource() }));
  tasks.push_back(f(
    std::allocator_arg,
    my_allocator<char>{ std::make_shared<size_t>(0) }));

  assert(allocator_instance_count > 0);

  for (auto& t : tasks)
  {
    coro::sync_wait(t);
  }

  tasks.clear();

  assert(allocator_instance_count == 0);
}

template<typename Allocator>
coro::task<int> add_async(std::allocator_arg_t, [[maybe_unused]] Allocator alloc, int a, int b)
{
  co_return a + b;
}

void test_task_custom_allocator_with_extra_args()
{
  std::vector<coro::task<int>> tasks;

  for (int i = 0; i < 5; ++i) {
    tasks.push_back(add_async(
      std::allocator_arg,
      std::allocator<char>{},
      i, 2 * i));
  }

  for (int i = 0; i < 5; ++i)
  {
    assert(sync_wait(std::move(tasks[i])) == 3 * i);
  }
}

struct some_type {
  template<typename Allocator>
  coro::task<int> get_async(std::allocator_arg_t, [[maybe_unused]] Allocator alloc) {
    co_return 42;
  }

  template<typename Allocator>
  coro::task<int> add_async(std::allocator_arg_t, [[maybe_unused]] Allocator alloc, int a, int b) {
    co_return a + b;
  }
};

void test_task_custom_allocator_on_member_function()
{
  assert(allocator_instance_count == 0);

  auto totalAllocated = std::make_shared<size_t>(0);
  some_type obj;
  assert(sync_wait(obj.get_async(std::allocator_arg, std::allocator<char>{})) == 42);
  assert(sync_wait(obj.get_async(std::allocator_arg, my_allocator<char>{totalAllocated})) == 42);
  assert(sync_wait(obj.add_async(std::allocator_arg, std::allocator<char>{}, 2, 3)) == 5);
  assert(sync_wait(obj.add_async(std::allocator_arg, my_allocator<char>{totalAllocated}, 2, 3)) == 5);

  assert(allocator_instance_count == 0);
  assert(*totalAllocated == 0);
}

int main()
{
  test_custom_allocator_is_destructed();
  test_custom_allocator_type_rebinding();
  test_mixed_custom_allocator_type_erasure();
  test_task_custom_allocator_with_extra_args();
  test_task_custom_allocator_on_member_function();

  return 0;
}
