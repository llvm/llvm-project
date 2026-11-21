//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <cassert>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

struct Hooks {
  static int calls;
};
int Hooks::calls = 0;

template <class T>
struct MutableHookAllocator {
  using value_type                = T;
  using is_always_equal           = std::true_type;
  MutableHookAllocator() noexcept = default;
  template <class U>
  MutableHookAllocator(const MutableHookAllocator<U>&) noexcept {}
  T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>().deallocate(p, n); }
  void construct(T* p, T& value) noexcept {
    ++Hooks::calls;
    ::new (static_cast<void*>(p)) T(value);
  }
};
template <class T, class U>
bool operator==(const MutableHookAllocator<T>&, const MutableHookAllocator<U>&) noexcept {
  return true;
}
template <class T, class U>
bool operator!=(const MutableHookAllocator<T>&, const MutableHookAllocator<U>&) noexcept {
  return false;
}

int main(int, char**) {
  int source[] = {7, 11, 19};
  using Alloc  = MutableHookAllocator<int>;
  Alloc alloc;
  int* p = alloc.allocate(1);
  std::allocator_traits<Alloc>::construct(alloc, p, source[0]);
  assert(Hooks::calls == 1 && *p == 7);
  std::allocator_traits<Alloc>::destroy(alloc, p);
  alloc.deallocate(p, 1);

  Hooks::calls = 0;
  std::vector<int, Alloc> result(source, source + 3);
  assert(Hooks::calls == 3);
  assert(source[0] == 7 && source[1] == 11 && source[2] == 19);
  assert(result.size() == 3 && result[0] == 7 && result[1] == 11 && result[2] == 19);

  Hooks::calls     = 0;
  const int* first = source;
  std::vector<int, Alloc> const_result(first, first + 3);
  assert(Hooks::calls == 0);
  assert(const_result == result);
  return 0;
}
